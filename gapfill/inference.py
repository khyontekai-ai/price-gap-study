"""Dependence-aware inference, requested in review.

Cell-wise tests treat every hidden week of every market as independent. Here the unit of replication is
the commodity panel (state x commodity: all its markets, weeks and runs together), and common shocks in
time are handled by resampling blocks of 13 panel weeks as well.

1. Commodity-level signed-rank test: mean APE difference per state x commodity unit, Wilcoxon across
   units, Holm over the pre-declared family (bridge_auto against each of the ten baselines).
2. Commodity-cluster bootstrap (commodities resampled within each state) and two-way bootstrap
   (commodities x 13-week blocks, product weights; Owen's pigeonhole bootstrap) for the MAPE difference.

Run:  .venv\\Scripts\\python.exe -m gapfill.inference
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import holm
from .calendar import attach_calendar, calendar_blocks
from .study import BASELINES, NORTH_EAST, OUT

STATES = ("Assam", "Meghalaya", "Manipur", "Nagaland", "Tripura", "Bihar", "Goa")
KEY = ["state", "commodity", "protocol", "run", "series", "week", "calendar_week"]
BLOCK = 13
INF = OUT / "inference"


def _pairs(ev: pd.DataFrame, ours: str, other: str) -> pd.DataFrame:
    """Cells both methods filled, with their absolute percentage errors."""
    d = ev[ev["method"].isin([ours, other])].assign(ape=lambda x: (x["filled"] - x["actual"]).abs() / x["actual"])
    w = d.pivot_table(index=KEY, columns="method", values="ape", aggfunc="first").dropna()
    w = w.rename(columns={ours: "a", other: "b"}).reset_index()
    w["unit"] = w["state"] + ":" + w["commodity"]
    w["block"] = calendar_blocks(pd.DatetimeIndex(w["calendar_week"]), BLOCK)
    return w


def _boot(w: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Point MAPE difference (other minus ours, points) with commodity-cluster and two-way 95% intervals."""
    units = sorted(w["unit"].unique())
    blocks = sorted(w["block"].unique())
    ui = {u: i for i, u in enumerate(units)}
    bi = {b: i for i, b in enumerate(blocks)}
    S = np.zeros((len(units), len(blocks)))
    N = np.zeros_like(S)
    g = w.groupby(["unit", "block"]).agg(s=("b", "sum"), sa=("a", "sum"), n=("a", "size")).reset_index()
    for r in g.itertuples():
        S[ui[r.unit], bi[r.block]] = r.s - r.sa
        N[ui[r.unit], bi[r.block]] = r.n
    state_of = np.array([u.split(":")[0] for u in units])
    strata = [np.where(state_of == s)[0] for s in sorted(set(state_of))]
    rng = np.random.default_rng(seed)
    one, two = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        wc = np.zeros(len(units))
        for st in strata:                       # commodities resampled within each state
            np.add.at(wc, rng.choice(st, len(st)), 1)
        wt = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)).astype(float)
        one[i] = (wc @ S.sum(axis=1)) / (wc @ N.sum(axis=1))
        two[i] = (wc @ S @ wt) / max(wc @ N @ wt, 1e-12)
    point = S.sum() / N.sum()
    q = lambda x: (100 * np.quantile(x, 0.025), 100 * np.quantile(x, 0.975))
    return {"diff": 100 * point, "cl_lo": q(one)[0], "cl_hi": q(one)[1], "tw_lo": q(two)[0], "tw_hi": q(two)[1],
            "units": len(units), "blocks": len(blocks), "cells": len(w)}


def _unit_test(w: pd.DataFrame) -> dict:
    u = w.groupby("unit")[["a", "b"]].mean()
    diff = u["b"] - u["a"]
    p = sps.wilcoxon(diff, zero_method="zsplit").pvalue if len(diff) >= 6 else np.nan
    return {"units_better": int((diff > 0).sum()), "units": len(diff), "p_unit": p}


def _offline() -> pd.DataFrame:
    frames = []
    for s in STATES:
        d = pd.read_csv(OUT / s / "evaluation.csv.gz", low_memory=False,
                        usecols=["method", "series", "week", "actual", "filled", "commodity", "protocol", "run", "online"])
        d = d[~d["online"].astype(bool) & d["method"].isin(BASELINES + ["bridge_auto"])]
        frames.append(d.assign(state=s))
    return attach_calendar(pd.concat(frames, ignore_index=True))


def _controls(prefix: str) -> pd.DataFrame:
    frames = [pd.read_csv(f).assign(state=f.name[len(prefix) + 1:-7]) for f in sorted((OUT / "controls").glob(f"{prefix}_*.csv.gz"))]
    return attach_calendar(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def analyse(ev: pd.DataFrame, others, scope: dict, label: str) -> pd.DataFrame:
    rows = []
    for sname, states in scope.items():
        e = ev[ev["state"].isin(states)]
        for o in others:
            if o not in set(e["method"]):
                continue
            w = _pairs(e, "bridge_auto", o)
            if len(w) < 20:
                continue
            rows.append({"setting": label, "scope": sname, "baseline": o,
                         "mape_ours": 100 * w["a"].mean(), "mape_base": 100 * w["b"].mean(),
                         **_boot(w), **_unit_test(w)})
    r = pd.DataFrame(rows)
    for sname in r["scope"].unique():      # Holm within each scope over the declared baselines
        k = (r["scope"] == sname) & r["p_unit"].notna()
        r.loc[k, "p_unit_holm"] = holm(r.loc[k, "p_unit"].to_numpy())
    return r


def main():
    INF.mkdir(parents=True, exist_ok=True)
    scope = {"North-East pooled": NORTH_EAST, "All seven": STATES, **{s: (s,) for s in STATES}}
    off = analyse(_offline(), BASELINES, scope, "offline")
    out = [off]
    for prefix, label in (("strict", "origin-restricted"), ("causal", "origin-restricted, causal preprocessing")):
        c = _controls(prefix)
        if len(c):
            out.append(analyse(c, ["locf", "kalman", "bridge"], scope, label))
    res = pd.concat(out, ignore_index=True)
    res.assign(time_block_basis="calendar", anchor="1970-01-05", block_weeks=BLOCK).to_csv(
        INF / "inference_calendar.csv", index=False)
    show = res[res["scope"].isin(["North-East pooled", "All seven"]) | (res["setting"] != "offline")]
    cols = ["setting", "scope", "baseline", "mape_ours", "mape_base", "diff", "cl_lo", "cl_hi", "tw_lo", "tw_hi",
            "units_better", "units", "p_unit", "p_unit_holm"]
    print(show[cols].round(4).to_string(index=False))
    return res


NAMES = {"locf": "carry-forward", "linear": "linear", "pchip": "PCHIP", "kalman": "Kalman", "softimpute": "SoftImpute",
         "knn": "kNN", "chained_ml": "chained ML", "dfm_em": "DFM-EM", "brits": "BRITS", "saits": "SAITS",
         "bridge": "fixed bridge"}


def _r(x, nd=1):
    from decimal import Decimal, ROUND_HALF_UP
    v = str(Decimal(str(float(x))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP))
    return f"${v}$" if v.startswith("-") else v


LABELS = {"offline": "offline", "origin-restricted": "origin-restr. (B)",
          "origin-restricted, causal preprocessing": "+ causal prep. (C)"}


def _p(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else _r(p, 3)


def latex(res: pd.DataFrame, path) -> None:
    """Paper table: North-East pooled differences with dependence-aware intervals and unit-level tests,
    then each state against its closest baseline (offline, all runs where the baseline ran)."""
    lines = [r"\begin{table*}[t]",
             r"\caption{Dependence-aware inference. Difference = MAPE of the comparison method minus MAPE of \auto{} (points; positive favours \auto{}) on the cells both filled. "
             r"Commodity: 95\% bootstrap interval resampling state$\times$commodity panels within states. Two-way: resampling panels and shared 13-week calendar blocks (common Monday anchor 5 January 1970). "
             r"Panels: commodity panels where \auto{} has the lower mean error, of all panels. $p$: Wilcoxon signed-rank test across panels, Holm-adjusted within each block of rows. "
             r"Offline rows use every run on which the comparison method ran, so the cells, and \auto{}'s MAPE, differ by row.}",
             r"\label{tab:inference}", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{lll rrr cc rr}", r"\toprule",
             r"Setting & Scope & Comparison & \auto{} & Other & Diff. & Commodity CI & Two-way CI & Panels & $p$ \\", r"\midrule"]
    blocks = [("offline", "North-East pooled", None)]
    for label in ("origin-restricted", "origin-restricted, causal preprocessing"):
        blocks.append((label, "North-East pooled", None))
    for setting, scope, _ in blocks:
        d = res[(res["setting"] == setting) & (res["scope"] == scope)]
        if d.empty:
            continue
        for i, r in enumerate(d.sort_values("diff").itertuples()):
            lines.append(" & ".join([LABELS[setting] if i == 0 else "", "NE pooled" if i == 0 else "", NAMES[r.baseline],
                                     _r(r.mape_ours), _r(r.mape_base), _r(r.diff),
                                     f"{_r(r.cl_lo)} to {_r(r.cl_hi)}", f"{_r(r.tw_lo)} to {_r(r.tw_hi)}",
                                     f"{r.units_better}/{r.units}", _p(r.p_unit_holm)]) + r" \\")
        lines.append(r"\midrule")
    off = res[(res["setting"] == "offline") & res["scope"].isin(STATES)]
    best = off.loc[off.groupby("scope")["diff"].idxmin()].set_index("scope").reindex(list(STATES))
    for i, (s, r) in enumerate(best.iterrows()):
        lines.append(" & ".join(["offline, closest" if i == 0 else "", s, NAMES[r.baseline],
                                 _r(r.mape_ours), _r(r.mape_base), _r(r["diff"]),
                                 f"{_r(r.cl_lo)} to {_r(r.cl_hi)}", f"{_r(r.tw_lo)} to {_r(r.tw_hi)}",
                                 f"{int(r.units_better)}/{int(r.units)}", _p(r.p_unit_holm)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    latex(main(), OUT.parent / "paper" / "tab_inference_revised.tex")
