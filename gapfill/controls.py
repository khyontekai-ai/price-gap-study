"""Leakage controls requested in review.

A. own-average exclusion (offline): the study's own hidden cells (same seeds, same scenario runs on
   which every baseline ran), every method refitted on panels WITHOUT the target state's own monthly
   average, which can contain the very daily reports an evaluation hides.
B. strict origin-wise test: for each week t with hidden cells, every method is refitted on weeks <= t
   only (parameters, stacking weights and calibration included) and fills week t. Panels have no
   own-state average and helper monthly averages are released in the week after their month ends.
   Cells without any earlier report of the same market are left out for every method.
C. as B, with causal preprocessing: hidden reports are removed before cleaning, cleaning uses only past
   data (causal_clean), market and monthly-series eligibility is decided on visible data up to each origin,
   and bridge_auto uses calibration="robust".

Run:  .venv\\Scripts\\python.exe -m gapfill.controls [--part A|B|C|report] [--jobs 3]
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from . import impute, stats
from .sources import data, monthly
from .study import (BASELINES, HELPER_STATES, MAX_COMMODITIES, NORTH_EAST, OUT, bridge_kw, make_masks, panel)

warnings.filterwarnings("ignore")
CTRL = OUT / "controls"
STATES = ("Assam", "Meghalaya", "Manipur", "Nagaland", "Tripura", "Bihar", "Goa")
PROTOCOLS = ["transplant"] * 3 + ["block_4", "block_8", "block_13", "backcast", "outage_4", "outage_8"]
HEAVY_RUN_IDS = (0, 4, 6, 8)        # transplant seed 0, block_8, backcast, outage_8: where every baseline ran
STRICT_RUN_IDS = (0, 8)             # transplant seed 0 and outage_8
STRICT_COMMODITIES = 10
KEY = ["commodity", "protocol", "run", "series", "week"]


def _setup(target):
    days, _ = data.clean_outliers(data.load_days(states=[target, *HELPER_STATES[target]]))
    sm = monthly.load_statewise()
    counts = days[days["state"] == target]["commodity"].value_counts()
    commodities = list(counts[counts >= 150].index)[:MAX_COMMODITIES]
    cfg = json.loads((OUT / "Assam" / "chosen_config.json").read_text(encoding="utf-8"))
    cfg["lags"] = tuple(cfg["lags"])
    return days, sm, commodities, cfg


def _panels(days, sm, c, target, **kw):
    """Full panel (as in the study) and the control panel; the target columns must be identical."""
    full = panel(days, sm, c, target=target)
    ctrl = panel(days, sm, c, target=target, **kw)
    if full is None or ctrl is None:
        return None
    Yf, cf, _, n = full
    Yc, cc, idx, n2 = ctrl
    assert n == n2 and cf[:n] == cc[:n] and np.array_equal(np.isnan(Yf[:, :n]), np.isnan(Yc[:, :n]))
    return Yc, cc, idx, n


# ------------------------------------------------------------------------------------------------ A

def _part_a_commodity(days, sm, c, salt, cfg, target):
    warnings.filterwarnings("ignore")
    built = _panels(days, sm, c, target, own_avg=False)
    if built is None:
        return []
    Y, cols, idx, n = built
    obs = ~np.isnan(Y[:, :n])
    targets = [m for m in range(n) if obs[:, m].sum() >= 20]
    frames = []
    for run_id in HEAVY_RUN_IDS:
        protocol = PROTOCOLS[run_id]
        masks = make_masks(protocol, obs, targets, np.random.default_rng(1000 * run_id + salt))
        if not masks:
            continue
        for name in BASELINES + ["bridge", "bridge_auto"]:
            try:
                e = impute.evaluate(Y, masks, methods=[name], online=False,
                                    bridge_kw=bridge_kw(cfg, n) if name == "bridge" else {"n_primary": n})
            except Exception as err:   # recorded, as in the study
                print("failed", target, c, protocol, name, repr(err)[:120], flush=True)
                continue
            frames.append(e.assign(commodity=c, protocol=protocol, run=run_id))
    return frames


def part_a(jobs: int):
    from joblib import Parallel, delayed
    CTRL.mkdir(parents=True, exist_ok=True)
    for target in STATES:
        f = CTRL / f"noavg_{target}.csv.gz"
        if f.exists():
            continue
        days, sm, commodities, cfg = _setup(target)
        res = Parallel(n_jobs=jobs)(delayed(_part_a_commodity)(days, sm, c, i, cfg, target)
                                    for i, c in enumerate(commodities))
        pd.concat([x for r in res for x in r], ignore_index=True).to_csv(f, index=False)
        print("A done:", target, flush=True)


# ------------------------------------------------------------------------------------------------ B

def _strict_commodity(days, sm, c, salt, cfg, target):
    warnings.filterwarnings("ignore")
    built = _panels(days, sm, c, target, own_avg=False, release="after")
    if built is None:
        return []
    Y, cols, idx, n = built
    obs = ~np.isnan(Y[:, :n])
    targets = [m for m in range(n) if obs[:, m].sum() >= 20]
    rows = []
    for run_id in STRICT_RUN_IDS:
        protocol = PROTOCOLS[run_id]
        masks = make_masks(protocol, obs, targets, np.random.default_rng(1000 * run_id + salt))
        if not masks:
            continue
        Yh = Y.copy()
        hidden = np.zeros_like(Y, dtype=bool)
        for m, h in masks.items():
            Yh[h, m] = np.nan
            hidden[h, m] = True
        for t in np.where(hidden.any(axis=1))[0]:
            pre = Yh[:t + 1]
            cells = [m for m in np.where(hidden[t])[0] if np.isfinite(pre[:t, m]).any()]   # needs an earlier report
            if not cells:
                continue
            last = {m: pre[:t, m][np.isfinite(pre[:t, m])][-1] for m in cells}
            fills = {"locf": {m: last[m] for m in cells}}
            sds = {}
            try:
                f, v = impute.bridge_auto(pre, online=True, return_var=True, n_primary=n)
                fills["bridge_auto"] = {m: f[-1, m] for m in cells}
                sds["bridge_auto"] = {m: float(np.sqrt(max(v[-1, m], 1e-12))) for m in cells}
            except Exception as err:
                print("failed bridge_auto", target, c, t, repr(err)[:100], flush=True)
            try:
                f = impute.bridge(pre, online=True, **bridge_kw(cfg, n))
                fills["bridge"] = {m: f[-1, m] for m in cells}
            except Exception as err:
                print("failed bridge", target, c, t, repr(err)[:100], flush=True)
            k = impute.kalman(pre[:, :n], online=True)
            fills["kalman"] = {m: k[-1, m] for m in cells}
            for name, fm in fills.items():
                for m in cells:
                    v = fm[m]
                    bad = (not np.isfinite(v)) or abs(v - last[m]) > np.log(10)   # the study's sanity rule
                    v = last[m] if bad else v
                    sd = sds.get(name, {}).get(m, np.nan)
                    rows.append((name, c, protocol, run_id, m, t, float(np.exp(Y[t, m])), float(np.exp(v)), sd,
                                 (abs(v - Y[t, m]) <= 1.2816 * sd) if np.isfinite(sd) and not bad else None, bad))
    return rows


def part_b(jobs: int):
    from joblib import Parallel, delayed
    CTRL.mkdir(parents=True, exist_ok=True)
    for target in STATES:
        f = CTRL / f"strict_{target}.csv.gz"
        if f.exists():
            continue
        days, sm, commodities, cfg = _setup(target)
        res = Parallel(n_jobs=jobs)(delayed(_strict_commodity)(days, sm, c, i, cfg, target)
                                    for i, c in enumerate(commodities[:STRICT_COMMODITIES]))
        cols = ["method", "commodity", "protocol", "run", "series", "week", "actual", "filled", "sd", "in80", "fallback"]
        pd.DataFrame([r for x in res for r in x], columns=cols).to_csv(f, index=False)
        print("B done:", target, flush=True)


# ------------------------------------------------------------------------------------------------ C

CAUSAL_MIN_TARGET, CAUSAL_MIN_HELPER, CAUSAL_MIN_MONTHS = 10, 20, 3   # the study's thresholds, applied to the past


def causal_clean(days: pd.DataFrame, ratio: float = 3.0, window_days: int = 21, far: float = 8.0) -> pd.DataFrame:
    """data.clean_outliers with only past information: a report on day d is judged against the median of
    the same market in the trailing 2*window_days+1 days (d included), the median of the state's markets in
    the same week (known when the week closes, which is when a week is filled) and the state's median up to
    d. A decision therefore never changes when later data arrive, so one pass serves every origin."""
    d = days.sort_values("date").copy()
    lr = np.log(d["price"])
    own = np.empty(len(d))
    for _, idx in d.groupby(["state", "market", "commodity"]).groups.items():
        g = d.loc[idx]
        s = pd.Series(np.log(g["price"].to_numpy()), index=g["date"].to_numpy())
        own[d.index.get_indexer(idx)] = s.rolling(f"{2 * window_days + 1}D", min_periods=1).median().to_numpy()
    week = d["date"].dt.to_period("W-SUN")
    state_med = lr.groupby([d["state"], d["commodity"], week]).transform("median")
    level = lr.groupby([d["state"], d["commodity"]]).transform(lambda x: x.expanding().median())
    k = np.log(ratio)
    bad = (((lr - own).abs() > k) & ((lr - state_med).abs() > k)) | ((lr - level).abs() > np.log(far))
    return d[~bad.to_numpy()].reset_index(drop=True)


def _causal_commodity(raw, sm, c, salt, cfg, target):
    """Origin-restricted fitting with causal preprocessing. Per evaluation run: the hidden reports are
    removed from the raw daily data, the rest is cleaned with causal_clean, and the panel is rebuilt. At
    each origin t only markets and monthly series that meet the study's thresholds on visible data up to t
    enter the fit. Truth and masks are those of part B, so cells can be matched."""
    warnings.filterwarnings("ignore")
    days_full, _ = data.clean_outliers(raw)
    built = _panels(days_full, sm, c, target, own_avg=False, release="after")
    if built is None:
        return []
    Y, cols, idx, n = built
    obs = ~np.isnan(Y[:, :n])
    targets = [m for m in range(n) if obs[:, m].sum() >= 20]
    rc = raw[raw["commodity"] == c].copy()
    rc["col"] = rc["state"] + ":" + rc["market"]
    rc["week"] = rc["date"].dt.to_period("W-SUN").dt.start_time
    rows = []
    for run_id in STRICT_RUN_IDS:
        protocol = PROTOCOLS[run_id]
        masks = make_masks(protocol, obs, targets, np.random.default_rng(1000 * run_id + salt))
        if not masks:
            continue
        hidden = np.zeros_like(Y, dtype=bool)
        for m, h in masks.items():
            hidden[h, m] = True
        gone = {(cols[m], idx[t]) for t, m in zip(*np.where(hidden))}
        keep = [(cl, w) not in gone for cl, w in zip(rc["col"], rc["week"])]
        dc = causal_clean(rc[keep])
        p = panel(dc, sm, c, min_assam=1, min_other=1, target=target, own_avg=False, release="after")
        if p is None:
            continue
        Yc_all, ccols, cidx, _ = p
        Yc = pd.DataFrame(Yc_all, index=cidx, columns=ccols).reindex(index=idx)
        helper_cols = [x for x in ccols if not x.startswith(target + ":")]
        T_ = Yc.reindex(columns=cols[:n]).to_numpy()          # target markets in part B's order
        H_ = Yc[helper_cols].to_numpy()
        for t in np.where(hidden.any(axis=1))[0]:
            seen_T = np.isfinite(T_[:t + 1]).sum(axis=0)
            seen_H = np.isfinite(H_[:t + 1]).sum(axis=0)
            tsel = [m for m in range(n) if seen_T[m] >= CAUSAL_MIN_TARGET]
            hsel = [j for j, name in enumerate(helper_cols)
                    if seen_H[j] >= (CAUSAL_MIN_MONTHS if name.startswith("state:") else CAUSAL_MIN_HELPER)]
            cells = [m for m in np.where(hidden[t])[0] if m in tsel and np.isfinite(T_[:t, m]).any()]
            if not cells:
                continue
            pre = np.column_stack([T_[:t + 1, tsel], H_[:t + 1, hsel]]) if hsel else T_[:t + 1, tsel]
            nn = len(tsel)
            pos = {m: tsel.index(m) for m in cells}
            last = {m: T_[:t, m][np.isfinite(T_[:t, m])][-1] for m in cells}
            fills = {"locf": {m: last[m] for m in cells}}
            sds = {}
            try:
                f, v = impute.bridge_auto(pre, online=True, return_var=True, n_primary=nn, calibration="robust")
                fills["bridge_auto"] = {m: f[-1, pos[m]] for m in cells}
                sds["bridge_auto"] = {m: float(np.sqrt(max(v[-1, pos[m]], 1e-12))) for m in cells}
            except Exception as err:
                print("failed bridge_auto", target, c, t, repr(err)[:100], flush=True)
            try:
                f = impute.bridge(pre, online=True, **bridge_kw(cfg, nn))
                fills["bridge"] = {m: f[-1, pos[m]] for m in cells}
            except Exception as err:
                print("failed bridge", target, c, t, repr(err)[:100], flush=True)
            k = impute.kalman(pre[:, :nn], online=True)
            fills["kalman"] = {m: k[-1, pos[m]] for m in cells}
            for name, fm in fills.items():
                for m in cells:
                    v = fm[m]
                    bad = (not np.isfinite(v)) or abs(v - last[m]) > np.log(10)
                    v = last[m] if bad else v
                    sd = sds.get(name, {}).get(m, np.nan)
                    rows.append((name, c, protocol, run_id, m, t, float(np.exp(Y[t, m])), float(np.exp(v)), sd,
                                 (abs(v - Y[t, m]) <= 1.2816 * sd) if np.isfinite(sd) and not bad else None, bad))
    return rows


def part_c(jobs: int):
    from joblib import Parallel, delayed
    CTRL.mkdir(parents=True, exist_ok=True)
    for target in STATES:
        f = CTRL / f"causal_{target}.csv.gz"
        if f.exists():
            continue
        days, sm, commodities, cfg = _setup(target)
        raw = data.load_days(states=[target, *HELPER_STATES[target]])
        res = Parallel(n_jobs=jobs)(delayed(_causal_commodity)(raw[raw["commodity"] == c], sm, c, i, cfg, target)
                                    for i, c in enumerate(commodities[:STRICT_COMMODITIES]))
        cols = ["method", "commodity", "protocol", "run", "series", "week", "actual", "filled", "sd", "in80", "fallback"]
        pd.DataFrame([r for x in res for r in x], columns=cols).to_csv(f, index=False)
        print("C done:", target, flush=True)


# ------------------------------------------------------------------------------------------------ report

def _mape(d):
    return 100 * float(((d["filled"] - d["actual"]).abs() / d["actual"]).mean())


def _md(t: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cols = [t.index.name or ""] + [str(c) for c in t.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in t.iterrows():
        out.append("| " + " | ".join([str(i)] + ["" if pd.isna(v) else f"{v:g}" for v in r]) + " |")
    return "\n".join(out)


def report() -> str:
    lines = ["# Leakage controls", ""]
    # ---- A
    rows, pooled_new, pooled_old = [], [], []
    for target in STATES:
        f = CTRL / f"noavg_{target}.csv.gz"
        if not f.exists():
            continue
        new = pd.read_csv(f)
        old = pd.read_csv(OUT / target / "evaluation.csv.gz", low_memory=False)
        old = old[(~old["online"].astype(bool)) & old["run"].isin(HEAVY_RUN_IDS) & old["method"].isin(set(new["method"]))]
        # identical cells: those every method filled in both runs
        keys = None
        for d in (new, old):
            for m, g in d.groupby("method"):
                k = set(map(tuple, g[KEY].to_numpy()))
                keys = k if keys is None else keys & k
        sel = lambda d: d[[tuple(r) in keys for r in d[KEY].to_numpy()]]
        new_s, old_s = sel(new), sel(old)
        for m in sorted(set(new_s["method"])):
            rows.append((target, m, _mape(old_s[old_s["method"] == m]), _mape(new_s[new_s["method"] == m]),
                         int((new_s["method"] == m).sum())))
        if target in NORTH_EAST:
            pooled_new.append(new_s.assign(state=target))
            pooled_old.append(old_s.assign(state=target))
    if rows:
        a = pd.DataFrame(rows, columns=["state", "method", "with_own_avg", "without_own_avg", "cells"])
        pn, po = pd.concat(pooled_new), pd.concat(pooled_old)
        for m in sorted(set(pn["method"])):
            a.loc[len(a)] = ("North-East pooled", m, _mape(po[po["method"] == m]), _mape(pn[pn["method"] == m]),
                             int((pn["method"] == m).sum()))
        a.to_csv(CTRL / "noavg_summary.csv", index=False)
        piv = a.pivot_table(index="method", columns="state", values="without_own_avg")
        piv = piv[[s for s in list(STATES) + ["North-East pooled"] if s in piv.columns]].sort_values("North-East pooled")
        lines += ["## A. Without the target state's own monthly average (offline, identical cells)", "",
                  _md(piv.round(1)), "", "Change from the saved run (without minus with), points:", "",
                  (a.pivot_table(index="method", columns="state", values="without_own_avg")
                   - a.pivot_table(index="method", columns="state", values="with_own_avg")).round(2)
                  .reindex(piv.index)[piv.columns].pipe(_md), ""]
        pn = pn.assign(online=False, commodity=pn["state"] + ":" + pn["commodity"])   # unique series clusters
        for base in ("saits", "chained_ml", "linear"):
            b = stats.bootstrap_diff(pn[pn["method"].isin(["bridge_auto", base])], "bridge_auto", base)
            lines.append(f"- North-East pooled, {base} minus bridge_auto: {b['diff_pct_points']:.2f} points "
                         f"(95% series-bootstrap {b['ci_low']:.2f} to {b['ci_high']:.2f})")
        lines.append("")
    # ---- B
    rows, pooled = [], []
    for target in STATES:
        f = CTRL / f"strict_{target}.csv.gz"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        complete = d.groupby(KEY)["method"].transform("nunique") == d["method"].nunique()
        d = d[complete]
        for m, g in d.groupby("method"):
            cov = g["in80"].dropna()
            rows.append((target, m, _mape(g), len(g), 100 * cov.astype(str).str.lower().eq("true").mean() if len(cov) else np.nan))
        if target in NORTH_EAST:
            pooled.append(d.assign(commodity=target + ":" + d["commodity"]))
    if rows:
        b = pd.DataFrame(rows, columns=["state", "method", "mape", "cells", "cover80"])
        p = pd.concat(pooled)
        for m, g in p.groupby("method"):
            cov = g["in80"].dropna()
            b.loc[len(b)] = ("North-East pooled", m, _mape(g), len(g),
                             100 * cov.astype(str).str.lower().eq("true").mean() if len(cov) else np.nan)
        b.to_csv(CTRL / "strict_summary.csv", index=False)
        piv = b.pivot_table(index="method", columns="state", values="mape")
        piv = piv[[s for s in list(STATES) + ["North-East pooled"] if s in piv.columns]].sort_values("North-East pooled")
        lines += ["## B. Strict origin-wise test (refit at every week on data up to that week)", "",
                  _md(piv.round(1)), "",
                  "Cells: " + ", ".join(f"{r.state} {int(r.cells)}" for r in b[b.method == "bridge_auto"].itertuples()), "",
                  "bridge_auto nominal-80% coverage: " +
                  ", ".join(f"{r.state} {r.cover80:.1f}" for r in b[b.method == "bridge_auto"].itertuples()), ""]
        p = p.assign(online=True)
        for base in ("locf", "kalman"):
            bb = stats.bootstrap_diff(p[p["method"].isin(["bridge_auto", base])], "bridge_auto", base)
            lines.append(f"- North-East pooled, {base} minus bridge_auto: {bb['diff_pct_points']:.2f} points "
                         f"(95% series-bootstrap {bb['ci_low']:.2f} to {bb['ci_high']:.2f})")
    # ---- C
    rows = []
    for target in STATES:
        f = CTRL / f"causal_{target}.csv.gz"
        if not f.exists():
            continue
        cdf = pd.read_csv(f)
        bdf = pd.read_csv(CTRL / f"strict_{target}.csv.gz")
        for name, d in (("C", cdf), ("B", bdf)):
            complete = d.groupby(KEY)["method"].transform("nunique") == d["method"].nunique()
            d = d[complete]
            cells_c = set(map(tuple, cdf[KEY].drop_duplicates().to_numpy()))
            matched = d[[tuple(r) in cells_c for r in d[KEY].to_numpy()]]
            for scope, dd in (("all", d), ("matched", matched)):
                for m, g in dd.groupby("method"):
                    ok = g[~g["fallback"].astype(bool) & g["sd"].notna()]
                    z = (np.log(ok["filled"]) - np.log(ok["actual"])).abs() / ok["sd"]
                    cov = {lv: 100 * float((z <= q).mean()) if len(z) else np.nan
                           for lv, q in ((50, 0.6745), (80, 1.2816), (95, 1.96))}
                    rows.append((target, name, scope, m, _mape(g), len(g), cov[50], cov[80], cov[95]))
    if rows:
        c = pd.DataFrame(rows, columns=["state", "part", "cells_set", "method", "mape", "cells", "cover50", "cover80", "cover95"])
        ne = c[c.state.isin(NORTH_EAST)]
        for keys, g in ne.groupby(["part", "cells_set", "method"]):
            w = g["cells"]
            c.loc[len(c)] = ("North-East pooled", *keys, float(np.average(g["mape"], weights=w)), int(w.sum()),
                             *(float(np.average(g[k].fillna(0), weights=w)) if g[k].notna().any() else np.nan
                               for k in ("cover50", "cover80", "cover95")))
        c.to_csv(CTRL / "causal_summary.csv", index=False)
        sel = c[(c.part == "C") & (c.cells_set == "all")]
        piv = sel.pivot_table(index="method", columns="state", values="mape")
        piv = piv[[s for s in list(STATES) + ["North-East pooled"] if s in piv.columns]]
        lines += ["", "## C. Origin-restricted fitting with causal preprocessing and robust calibration", "", _md(piv.round(1)), "",
                  "Cells: " + ", ".join(f"{r.state} {int(r.cells)}" for r in sel[sel.method == "bridge_auto"].itertuples()), ""]
        cmp_ = c[(c.cells_set == "matched") & (c.method == "bridge_auto")].pivot_table(
            index="state", columns="part", values=["mape", "cover80", "cover50", "cover95"])
        cmp_.columns = [f"{a} {b}" for a, b in cmp_.columns]
        cmp_.index.name = "state"
        lines += ["bridge_auto on the cells of C, B (legacy calibration, full-sample cleaning) against C:", "",
                  _md(cmp_.round(1).reindex([s for s in list(STATES) + ["North-East pooled"] if s in cmp_.index])), ""]
    text = "\n".join(lines) + "\n"
    (CTRL / "summary.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["A", "B", "C", "report", "all"])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if a.part in ("A", "all"):
        part_a(a.jobs)
    if a.part in ("B", "all"):
        part_b(a.jobs)
    if a.part in ("C", "all"):
        part_c(a.jobs)
    print(report())
