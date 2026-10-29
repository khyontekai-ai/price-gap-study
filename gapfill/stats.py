"""Statistical comparisons for the gap-filling study.

- Paired Wilcoxon signed-rank tests on absolute log errors of the same hidden cells, our method vs
  each baseline, Holm-corrected across baselines.
- Cluster bootstrap confidence intervals for the difference in MAPE, resampling whole series
  (commodity x market), because errors within a series are correlated.
- Friedman test over commodities (each commodity a "dataset") with average ranks and the Nemenyi
  critical difference (Demsar 2006, JMLR).
- Interval calibration at 50/80/95% from each method's own standard deviation.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["commodity", "protocol", "run", "online", "series", "week"]


def abs_log_error(ev: pd.DataFrame) -> pd.Series:
    return np.abs(np.log(ev["filled"]) - np.log(ev["actual"]))


def paired(ev: pd.DataFrame, ours: str, others: Sequence[str]) -> pd.DataFrame:
    """Wilcoxon signed-rank (ours vs each baseline) on the cells both methods filled."""
    d = ev.assign(ale=abs_log_error(ev), ape=(ev["filled"] - ev["actual"]).abs() / ev["actual"])
    base = d[d["method"] == ours].set_index(KEY)
    rows = []
    for o in others:
        other = d[d["method"] == o].set_index(KEY)
        j = base[["ale", "ape"]].join(other[["ale", "ape"]], how="inner", rsuffix="_o")
        if len(j) < 10:
            continue
        diff = j["ale"] - j["ale_o"]
        w = stats.wilcoxon(diff, alternative="less", zero_method="zsplit")
        rows.append({"baseline": o, "cells": len(j), "mape_ours": 100 * j["ape"].mean(),
                     "mape_baseline": 100 * j["ape_o"].mean(), "ours_better_share": 100 * (diff < 0).mean(),
                     "p_value": w.pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"].to_numpy())
    return out


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * p[k])
        adj[k] = min(1.0, running)
    return adj


def bootstrap_diff(ev: pd.DataFrame, ours: str, other: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI for MAPE(other) - MAPE(ours), resampling series clusters with replacement."""
    d = ev.assign(ape=(ev["filled"] - ev["actual"]).abs() / ev["actual"])
    a = d[d["method"] == ours].set_index(KEY)["ape"]
    b = d[d["method"] == other].set_index(KEY)["ape"]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").reset_index()
    j["cluster"] = j["commodity"] + "|" + j["series"].astype(str)
    g = j.groupby("cluster")[["a", "b"]].agg(["sum", "count"])
    sa, sb, n = g[("a", "sum")].to_numpy(), g[("b", "sum")].to_numpy(), g[("a", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(n), len(n))
        diffs[i] = (sb[k].sum() - sa[k].sum()) / n[k].sum()
    point = (sb.sum() - sa.sum()) / n.sum()
    return {"baseline": other, "diff_pct_points": 100 * point, "ci_low": 100 * np.quantile(diffs, 0.025),
            "ci_high": 100 * np.quantile(diffs, 0.975), "clusters": len(n)}


def friedman_nemenyi(ev: pd.DataFrame, methods: Iterable[str], alpha: float = 0.05) -> dict:
    """Average rank of each method across commodities (by MAPE), Friedman p-value, Nemenyi CD."""
    methods = list(methods)
    d = ev[ev["method"].isin(methods)].assign(ape=(ev["filled"] - ev["actual"]).abs() / ev["actual"])
    tab = d.groupby(["commodity", "method"])["ape"].mean().unstack()[methods].dropna()
    ranks = tab.rank(axis=1).mean().sort_values()
    k, n = len(methods), len(tab)
    chi = stats.friedmanchisquare(*[tab[m].to_numpy() for m in methods]) if n >= 3 and k >= 3 else None
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n)) if n else np.nan
    return {"ranks": ranks, "datasets": n, "friedman_p": chi.pvalue if chi else np.nan, "cd": cd}


def calibration(ev: pd.DataFrame, levels=(0.5, 0.8, 0.95)) -> pd.DataFrame:
    d = ev.dropna(subset=["sd"])
    err = abs_log_error(d)
    rows = []
    for m, g in d.assign(err=err).groupby("method"):
        row = {"method": m}
        for lv in levels:
            z = stats.norm.ppf(0.5 + lv / 2)
            row[f"cover{int(lv * 100)}"] = 100 * float((g["err"] <= z * g["sd"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)
