"""Gap-filling study on real data: Assam markets, with neighbouring-state markets and monthly
state averages as helpers.

Protocols (truth = real Assam reports that we hide):
  transplant   hide reports where another Assam series had none (real gap shapes), several seeds
  block_k      hide k consecutive weeks (k = 4, 8, 13)
  backcast     hide a market's first 26 weeks: can its early history be rebuilt?
  sweep_r      hide a random share r of reports (missing-rate sensitivity)
Settings: offline (fill the past, both sides of a gap known) and online (only the past: nowcasting).

Comparisons: literature baselines (impute.METHODS), data ablations of bridge (Assam only, plus
neighbouring markets, plus monthly state averages) and model ablations (no factor, no AR bridge,
no lead-lag, fixed loadings, no interval calibration). Statistics in stats.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import impute, stats
from .sources import data, monthly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
# Target state -> helper states (daily markets where downloaded, plus monthly state averages).
# North-East India first; Jharkhand, Goa and Bihar are an out-of-region check.
HELPER_STATES = {
    "Assam": ("West Bengal", "Meghalaya", "Bihar", "Tripura", "Nagaland", "Manipur"),
    "Meghalaya": ("Assam", "West Bengal"),
    "Manipur": ("Assam", "Nagaland"),
    "Nagaland": ("Assam", "Manipur"),
    "Tripura": ("Assam", "West Bengal"),
    "Jharkhand": ("Bihar", "West Bengal", "Odisha"),
    "Goa": ("Maharashtra", "Karnataka"),
    "Bihar": ("West Bengal", "Jharkhand", "Uttar Pradesh"),
}
NORTH_EAST = ("Assam", "Meghalaya", "Manipur", "Nagaland", "Tripura")
NEIGHBOURS = HELPER_STATES["Assam"]
warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty-slice means on all-missing weeks
warnings.filterwarnings("ignore", category=UserWarning)

BASELINES = ["locf", "linear", "pchip", "kalman", "softimpute", "knn", "chained_ml", "dfm_em", "brits", "saits"]
HEAVY = {"chained_ml", "dfm_em", "brits", "saits"}
# The slow baselines (BRITS takes ~50 s per fill) run on one run of each scenario type; the fast
# methods and our method run on all of them. Paired tests only compare cells both methods filled.
HEAVY_RUNS = {("transplant", 0), ("block_8", None), ("backcast", None), ("outage_8", None)}
MAX_COMMODITIES = 20   # per state, the most-reported ones
OURS = "bridge_auto"   # headline: bridge with the way of borrowing strength chosen per panel
FIXED = "bridge"       # bridge with the Assam development-run settings; base of the ablations
DATA_ABLATIONS = {"bridge[assam_only]": dict(neighbours=False, state_avg=False),
                  "bridge[+neighbours]": dict(neighbours=True, state_avg=False)}
# Settings of our method chosen on a development run (its own transplant masks, offline), never on
# the test runs that are reported.
HELPERS = {"none": dict(helper_weight=0.0, basis=False), "pooled": dict(helper_weight=1.0, basis=False),
           "basis": dict(helper_weight=1.0, basis=True)}
GRID = [dict(free_loadings=fl, lags=lg, helpers=h)
        for fl in (True, False) for lg in ((0,), (0, 1, 2)) for h in HELPERS]


def bridge_kw(cfg: dict, n_primary: int, **extra) -> dict:
    """Keyword arguments for impute.bridge from a configuration."""
    kw = {k: v for k, v in cfg.items() if k != "helpers"}
    return dict(kw, **HELPERS[cfg["helpers"]], n_primary=n_primary, **extra)
DEV_SALT = 99991


def model_ablations(cfg: dict) -> dict:
    """Each ablation switches one part of the chosen configuration off (or to its alternative)."""
    return {"bridge-no_factor": dict(cfg, use_factor=False),
            "bridge-no_ar": dict(cfg, use_ar=False),
            "bridge-alt_leadlag": dict(cfg, lags=(0,) if len(cfg["lags"]) > 1 else (0, 1, 2)),
            "bridge-alt_loadings": dict(cfg, free_loadings=not cfg["free_loadings"]),
            "bridge-no_helpers": dict(cfg, helpers="none"),
            "bridge-pooled_helpers": dict(cfg, helpers="pooled"),
            "bridge-basis_helpers": dict(cfg, helpers="basis"),
            "bridge-uncalibrated": dict(cfg, calibrate=False)}


def _cfg_name(cfg: dict) -> str:
    return (f"loadings={'free' if cfg['free_loadings'] else 'fixed'}, lags={'0-2' if len(cfg['lags']) > 1 else '0'}, "
            f"neighbours={cfg['helpers']}")


def panel(days: pd.DataFrame, states_monthly, commodity: str, min_assam: int = 10, min_other: int = 20,
          neighbours: bool = True, state_avg: bool = True, target: str = "Assam", own_avg: bool = True,
          release: str = "mid"):
    """Weekly log-price matrix for one commodity: the target state's markets first, then helper series
    (helper states' markets and monthly state averages). `min_assam` applies to target markets.

    Leakage controls: own_avg=False leaves out the target state's own monthly average (it may contain
    reports that an evaluation hides); release="after" places each monthly average in the Monday-start
    week that contains the seventh day after its month ends (a week starting 1-7 days after month end),
    instead of mid-month; with fills made at week close this assumes publication within seven days."""
    helpers = HELPER_STATES.get(target, ())
    d = days[(days["commodity"] == commodity) & days["state"].isin((target,) + helpers)]
    if d.empty:
        return None
    wide = data.weekly_prices(d.assign(market=d["state"] + ":" + d["market"]))
    if wide.empty:
        return None
    wide.columns = [c[1] for c in wide.columns]
    assam = [c for c in wide.columns if c.startswith(target + ":") and wide[c].notna().sum() >= min_assam]
    if not assam:
        return None
    others = [c for c in wide.columns if not c.startswith(target + ":") and wide[c].notna().sum() >= min_other] if neighbours else []
    W = wide[assam + others].copy()
    if state_avg and states_monthly is not None:
        sm = states_monthly[states_monthly["commodity"] == commodity]
        for st, g in sm.groupby("state"):
            if st not in (target,) + helpers or (st == target and not own_avg):
                continue
            s = pd.Series(np.nan, index=W.index)
            for _, r in g.iterrows():
                if release == "after":
                    mid = r["month"].to_timestamp(how="end").normalize() + pd.Timedelta(days=7)
                else:
                    mid = r["month"].to_timestamp() + pd.Timedelta(days=14)
                wk = mid - pd.Timedelta(days=mid.weekday())
                if wk in s.index:
                    s[wk] = r["price"]
            if s.notna().sum() >= 3:
                W[f"state:{st}"] = s
    return np.log(W.to_numpy(dtype=float)), list(W.columns), W.index, len(assam)


def make_masks(protocol: str, obs: np.ndarray, targets, rng) -> dict:
    T = obs.shape[0]
    if protocol == "transplant":
        return impute.transplant_masks(obs, targets, rng)
    if protocol == "backcast":
        masks = {}
        for m in targets:
            t_obs = np.where(obs[:, m])[0]
            k = min(len(t_obs) - 8, int(np.searchsorted(t_obs, t_obs[0] + 26)))
            hide = np.zeros(T, dtype=bool)
            hide[t_obs[:max(k, 0)]] = True
            if hide.any():
                masks[m] = hide
        return masks
    if protocol.startswith("block_"):
        return impute.block_masks(obs, targets, int(protocol.split("_")[1]), rng)
    if protocol.startswith("outage_"):
        # Every Assam market silent for k weeks (as when Assam stopped reporting in 2023-24):
        # only other states and state averages can say what happened.
        k = int(protocol.split("_")[1])
        start = int(rng.integers(8, max(9, T - k - 8)))
        hide = np.zeros(T, dtype=bool)
        hide[start:start + k] = True
        return {m: hide & obs[:, m] for m in targets if (hide & obs[:, m]).any()}
    if protocol.startswith("sweep_"):
        r = float(protocol.split("_")[1])
        masks = {}
        for m in targets:
            hide = obs[:, m] & (rng.random(T) < r)
            if hide.any() and (obs[:, m] & ~hide).sum() >= 6:
                masks[m] = hide
        return masks
    raise ValueError(protocol)


def run(out: Path = None, seeds: int = 3, heavy: bool = True, jobs: int = 6, target: str = "Assam",
        select: bool = None, only=None, progress=print) -> str:
    """Full study for one target state. Settings are chosen on a development run for Assam only
    (select=True); other states reuse Assam's choice unchanged, which makes them a transfer test."""
    from joblib import Parallel, delayed

    out = Path(out) if out else OUT / target.replace(" ", "_")
    if only and select is None:   # recompute only some methods: reuse the saved settings, no new selection
        select = not (OUT / "Assam" / "chosen_config.json").exists() and target == "Assam"
    select = (target == "Assam") if select is None else select
    out.mkdir(parents=True, exist_ok=True)
    days, dropped = data.clean_outliers(data.load_days(states=[target, *HELPER_STATES[target]]))
    dropped.to_csv(out / "dropped_reports.csv", index=False)   # entry errors removed before the study
    progress(f"Removed {len(dropped)} likely entry errors ({100 * len(dropped) / (len(days) + len(dropped)):.2f}% of reports)")
    try:
        sm = monthly.load_statewise()
    except FileNotFoundError:
        sm = None
    counts = days[days["state"] == target]["commodity"].value_counts()
    commodities = list(counts[counts >= 150].index)[:MAX_COMMODITIES]
    if select:
        dev = Parallel(n_jobs=jobs, verbose=2)(delayed(_dev_commodity)(days, sm, c, target) for c in commodities)
        dev = pd.concat([d for d in dev if d is not None], ignore_index=True)
        dev_tab = dev.groupby(["config", "protocol"])["ape"].mean().groupby("config").mean().sort_values()  # equal weight per scenario
        cfg = GRID[int(dev_tab.index[0])]
        pd.DataFrame({"config": [_cfg_name(GRID[i]) for i in dev_tab.index], "dev_mape": 100 * dev_tab.to_numpy()})             .to_csv(out / "dev_selection.csv", index=False)
        progress(f"Chosen on the development run: {_cfg_name(cfg)}")
        (out / "chosen_config.json").write_text(json.dumps(dict(cfg, lags=list(cfg["lags"]))), encoding="utf-8")
    else:
        cfg = json.loads((OUT / "Assam" / "chosen_config.json").read_text(encoding="utf-8"))
        cfg["lags"] = tuple(cfg["lags"])
        progress(f"Transfer test: Assam's settings unchanged ({_cfg_name(cfg)})")
    results = Parallel(n_jobs=jobs, verbose=5)(delayed(_one_commodity)(days, sm, c, seeds, heavy, i, cfg, target,
                                                                      set(only) if only else None)
                                               for i, c in enumerate(commodities))
    evs = [f for r in results if r for f in r[0]]
    timing = [t for r in results if r for t in r[1]]
    failures = pd.DataFrame([f for r in results if r for f in r[3]],
                            columns=["commodity", "protocol", "run", "online", "method", "error"])
    for r in results:
        if r:
            progress(r[2])
    ev = pd.concat(evs, ignore_index=True)
    timing = pd.DataFrame(timing, columns=["method", "commodity", "online", "seconds"])
    if only:
        ev, timing, failures = _merge_saved(out, ev, timing, failures)
    failures.to_csv(out / "failures.csv", index=False)
    ev.to_csv(out / "evaluation.csv.gz", index=False)
    timing.to_csv(out / "timing.csv", index=False)
    return report(ev, out)


KEY = ["commodity", "protocol", "run", "online", "series", "week"]


def _merge_saved(out: Path, new: pd.DataFrame, timing: pd.DataFrame, failures: pd.DataFrame):
    """Replace the recomputed methods in the saved results. The hidden cells are regenerated from the
    same seeds, so carry-forward (never changed) must come out identical: if not, refuse to merge."""
    old = pd.read_csv(out / "evaluation.csv.gz")
    redone = set(new["method"])
    if "locf" in redone:
        a = old[old["method"] == "locf"].set_index(KEY)["filled"].rename("old")
        b = new[new["method"] == "locf"].set_index(KEY)["filled"].rename("new")
        j = pd.concat([a, b], axis=1, join="outer")
        if j.isna().any().any() or not np.allclose(j["old"], j["new"], rtol=1e-9):
            raise RuntimeError("The hidden cells differ from the saved run: refusing to merge")
    merged = pd.concat([old[~old["method"].isin(redone)], new], ignore_index=True)
    t_old = pd.read_csv(out / "timing.csv") if (out / "timing.csv").exists() else timing.iloc[:0]
    f_old = pd.read_csv(out / "failures.csv") if (out / "failures.csv").exists() else failures.iloc[:0]
    return (merged, pd.concat([t_old[~t_old["method"].isin(redone)], timing], ignore_index=True),
            pd.concat([f_old[~f_old["method"].isin(redone)], failures], ignore_index=True))


def _dev_commodity(days, sm, c: str, target: str = "Assam"):
    """Development run: one transplant mask set (own seed), offline, every GRID configuration."""
    warnings.filterwarnings("ignore")
    full = panel(days, sm, c, target=target)
    if full is None:
        return None
    Y, _, _, n_assam = full
    obs = ~np.isnan(Y[:, :n_assam])
    targets = [m for m in range(n_assam) if obs[:, m].sum() >= 20]
    if not targets:
        return None
    frames = []
    for k, protocol in enumerate(("transplant", "transplant", "transplant", "block_8", "outage_4", "outage_8")):
        masks = make_masks(protocol, obs, targets, np.random.default_rng(DEV_SALT + k))
        if not masks:
            continue
        for i, cfg in enumerate(GRID):
            e = impute.evaluate(Y, masks, methods=[FIXED], bridge_kw=bridge_kw(cfg, n_assam, calibrate=False))
            frames.append(e.assign(config=i, protocol=protocol.split("_")[0],
                                   ape=(e["filled"] - e["actual"]).abs() / e["actual"]))
    return pd.concat(frames).assign(commodity=c) if frames else None


def _one_commodity(days, sm, c: str, seeds: int, heavy: bool, salt: int, cfg: dict, target: str = "Assam",
                   only=None):
    warnings.filterwarnings("ignore")
    protocols = (["transplant"] * seeds + ["block_4", "block_8", "block_13", "backcast", "outage_4", "outage_8"] +
                 ["sweep_0.2", "sweep_0.4", "sweep_0.6", "sweep_0.8"])
    evs, timing, failures = [], [], []
    full = panel(days, sm, c, target=target)
    if full is None:
        return None
    Y, cols, idx, n_assam = full
    obs = ~np.isnan(Y[:, :n_assam])
    targets = [m for m in range(n_assam) if obs[:, m].sum() >= 20]
    if not targets:
        return None
    ours = bridge_kw(cfg, n_assam)
    reduced = {name: panel(days, sm, c, target=target, **kw)[0] for name, kw in DATA_ABLATIONS.items()}
    note = f"{target} {c}: {n_assam} market series ({len(targets)} testable), {len(cols) - n_assam} helper series"
    for run_id, protocol in enumerate(protocols):
        masks = make_masks(protocol, obs, targets, np.random.default_rng(1000 * run_id + salt))
        if not masks:
            continue
        sweep = protocol.startswith("sweep_")
        for online in (False, True):
            frames = []
            heavy_here = heavy and ((protocol, run_id) in HEAVY_RUNS or (protocol, None) in HEAVY_RUNS)
            methods = [m for m in BASELINES if heavy_here or m not in HEAVY]
            if sweep:
                methods = [m for m in methods if m not in ("brits", "saits")]
            for name in methods + [FIXED, OURS]:
                if only is not None and name not in only:
                    continue
                if online and name in impute.TWO_SIDED:
                    continue
                t0 = time.perf_counter()
                try:
                    frames.append(impute.evaluate(Y, masks, methods=[name], online=online,
                                                  bridge_kw=ours if name == FIXED else
                                                  ({"n_primary": n_assam} if name == OURS else None)))
                except Exception as err:   # a method failing on one panel is recorded, not fatal
                    failures.append((c, protocol, run_id, online, name, repr(err)[:200]))
                    continue
                timing.append((name, c, online, time.perf_counter() - t0))
            if not sweep and (only is None or "ablations" in only):
                for name, Yr in reduced.items():
                    frames.append(impute.evaluate(Yr, masks, methods=[FIXED], online=online, label=name,
                                                  bridge_kw=bridge_kw(cfg, n_assam)))
                for name, kw in model_ablations(cfg).items():
                    frames.append(impute.evaluate(Y, masks, methods=[FIXED], online=online,
                                                  bridge_kw=bridge_kw(kw, n_assam), label=name))
            if frames:
                evs.append(pd.concat(frames).assign(commodity=c, protocol=protocol, run=run_id, online=online))
    return evs, timing, note, failures


# ------------------------------------------------------------------------------------------------
# Report

def _gap_bin(g: int) -> str:
    return "1" if g <= 1 else "2-3" if g <= 3 else "4-8" if g <= 8 else "9+"


def _table(df: pd.DataFrame, fmt="{:.1f}") -> list:
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
             "|---" * (len(df.columns) + 1) + "|"]
    for i, r in df.iterrows():
        cells = ["" if (isinstance(v, float) and np.isnan(v)) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                 for v in r]
        lines.append(f"| {i} | " + " | ".join(cells) + " |")
    return lines


def common_cells(e: pd.DataFrame) -> pd.DataFrame:
    """Offline results restricted to the test runs on which every baseline ran (the slow baselines ran
    on one run per scenario type), so that all methods are compared on identical hidden cells."""
    if "saits" not in set(e["method"]):
        return e
    k = e[e["method"] == "saits"][["commodity", "protocol", "run", "online"]].drop_duplicates()
    return e.merge(k, on=["commodity", "protocol", "run", "online"], how="inner")


def report(ev: pd.DataFrame, out: Path) -> str:
    ev = ev.assign(group=np.where(ev["protocol"].str.startswith("block"), "block",
                                  np.where(ev["protocol"].str.startswith("outage"), "outage",
                                  np.where(ev["protocol"].str.startswith("sweep"), "sweep", ev["protocol"]))),
                   gap_bin=ev["gap"].map(_gap_bin),
                   ape=(ev["filled"] - ev["actual"]).abs() / ev["actual"])
    main = ev[ev["group"] != "sweep"]
    main = pd.concat([common_cells(main[~main["online"]]), main[main["online"]]], ignore_index=True)
    baselines = [m for m in BASELINES if m in set(ev["method"])]
    state = out.name.replace("_", " ")
    lines = [f"# Filling gaps in {state} market prices: results", "",
             f"Truth: real weekly {state} market prices (Agmarknet) that we hide and then fill. "
             "Error = mean absolute % error (MAPE) on hidden weeks. Offline = filling history; "
             "online = only data up to that week (nowcasting).", ""]
    if (out / "dev_selection.csv").exists():
        dv = pd.read_csv(out / "dev_selection.csv").set_index("config")
        dv.index.name = "Configuration (development run, offline)"
        lines += ["## 0. Settings of our method, chosen on a separate development run", ""] + _table(dv, "{:.2f}") + [""]
    for online in (False, True):
        e = main[main["online"] == online]
        title = "Online (nowcasting)" if online else "Offline (filling history)"
        tab = e.groupby(["method", "group"])["ape"].mean().unstack() * 100
        tab["all"] = e.groupby("method")["ape"].mean() * 100
        tab["median APE"] = e.groupby("method")["ape"].median() * 100
        tab["mean abs log error x100"] = e.assign(ale=np.abs(np.log(e["filled"] / e["actual"]))).groupby("method")["ale"].mean() * 100
        tab["cells"] = e.groupby("method").size()
        tab = tab.sort_values("all")
        tab.index.name = "Method"
        lines += [f"## 1{'b' if online else 'a'}. {title}: all methods and ablations", ""] + _table(tab) + [""]
        gb = e.groupby(["method", "gap_bin"])["ape"].mean().unstack() * 100
        gb = gb.loc[[m for m in tab.index if m in gb.index]]
        gb.index.name = "Weeks to nearest report"
        lines += ["By distance to the nearest report:", ""] + _table(gb) + [""]

        others = [m for m in baselines if m in set(e["method"])]
        pt = stats.paired(e, OURS, others)
        if len(pt):
            pt = pt.set_index("baseline")
            pt.index.name = "Baseline"
            lines += [f"Paired Wilcoxon signed-rank tests, {OURS} vs each baseline on the same hidden cells "
                      "(one-sided, Holm-corrected):", ""] + _table(pt, "{:.3g}") + [""]
        bs = pd.DataFrame([stats.bootstrap_diff(e, OURS, o) for o in others]).set_index("baseline")
        bs.index.name = "Baseline"
        lines += ["MAPE(baseline) - MAPE(ours), percentage points, 95% cluster-bootstrap CI (resampling series):", ""] + \
            _table(bs, "{:.2f}") + [""]
        fr = stats.friedman_nemenyi(e, others + [OURS])
        rk = fr["ranks"].to_frame("average rank")
        rk.index.name = "Method"
        lines += [f"Average rank across {fr['datasets']} commodities (1 = best). Friedman p = {fr['friedman_p']:.2g}; "
                  f"Nemenyi critical difference = {fr['cd']:.2f}.", ""] + _table(rk, "{:.2f}") + [""]
        cal = stats.calibration(e).set_index("method")
        cal.index.name = "Method"
        lines += ["Interval coverage (target 50 / 80 / 95):", ""] + _table(cal) + [""]
        _figures(e, tab, gb, fr, out, "online" if online else "offline")

    bc = main[~main["online"]].groupby(["commodity", "method"])["ape"].mean().unstack() * 100
    keep = [m for m in baselines + [OURS] if m in bc.columns]
    bc = bc[keep]
    bc.index.name = "Commodity"
    lines += ["## 2. By commodity (offline, all protocols)", ""] + _table(bc) + [""]

    sw = ev[ev["group"] == "sweep"]
    if len(sw):
        for online in (False, True):
            s = sw[sw["online"] == online]
            t = s.groupby(["method", "protocol"])["ape"].mean().unstack() * 100
            t.columns = [c.replace("sweep_", "hide ") for c in t.columns]
            t.index.name = "Method"
            lines += [f"## 3{'b' if online else 'a'}. Missing-rate sensitivity ({'online' if online else 'offline'}): "
                      "share of reports hidden at random", ""] + _table(t.sort_values(t.columns[-1])) + [""]
        _sweep_figure(sw, out)
    if (out / "failures.csv").exists():
        fl = pd.read_csv(out / "failures.csv")
        if len(fl):
            ft = fl.groupby("method").size().to_frame("panels where the method failed")
            ft.index.name = "Method"
            lines += ["## Method failures (numerical errors; those panels are left out for that method)", ""] +                 _table(ft, "{:.0f}") + [""]
    tm = pd.read_csv(out / "timing.csv") if (out / "timing.csv").exists() else None
    if tm is not None:
        t = tm.groupby("method")["seconds"].mean().sort_values().to_frame("seconds per fill")
        t.index.name = "Method"
        lines += ["## 4. Runtime (one commodity panel, CPU)", ""] + _table(t, "{:.2f}") + [""]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    paper_fig = ROOT / "paper" / "figures"
    paper_fig.mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        shutil.copy2(f, paper_fig / f.name)
    return text


def _figures(e, tab, gb, fr, out: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    main = [m for m in tab.index if not m.startswith("bridge-") and not m.startswith("bridge[")]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m in main:
        if m in gb.index:
            ax.plot(range(gb.shape[1]), gb.loc[m].to_numpy(), "o-", lw=2.5 if m == OURS else 1,
                    color="#1b6e4c" if m == OURS else None, label=m)
    ax.set_xticks(range(gb.shape[1]), gb.columns)
    ax.set_xlabel("Weeks to the nearest real report")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Error by gap length ({tag})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"gap_length_{tag}.png", dpi=200)
    plt.close(fig)

    ranks = fr["ranks"]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(ranks) + 1))
    y = np.arange(len(ranks))
    ax.hlines(y, 1, ranks.to_numpy(), color="#cccccc")
    ax.plot(ranks.to_numpy(), y, "o", color="#1b6e4c")
    best = ranks.iloc[0]
    ax.axvspan(best, best + fr["cd"], color="#1b6e4c", alpha=0.12, label=f"Nemenyi CD = {fr['cd']:.2f}")
    ax.set_yticks(y, ranks.index)
    ax.invert_yaxis()
    ax.set_xlabel(f"Average rank over {fr['datasets']} commodities (lower is better)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"ranks_{tag}.png", dpi=200)
    plt.close(fig)


def _sweep_figure(sw: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = sw[~sw["online"]]
    t = s.groupby(["method", "protocol"])["ape"].mean().unstack() * 100
    rates = [float(c.split("_")[1]) for c in t.columns]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m, r in t.iterrows():
        ax.plot(rates, r.to_numpy(), "o-", lw=2.5 if m == OURS else 1, color="#1b6e4c" if m == OURS else None, label=m)
    ax.set_xlabel("Share of real reports hidden")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Missing-rate sensitivity (offline)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "missing_rate.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="")
    p.add_argument("--state", default="Assam", help="Target state (" + ", ".join(HELPER_STATES) + ")")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--no-heavy", action="store_true", help="Skip chained ML, DFM-EM, BRITS and SAITS (quick run)")
    p.add_argument("--report-only", action="store_true", help="Rebuild the report from saved evaluation.csv.gz")
    p.add_argument("--only", default="", help="Comma-separated methods to recompute and merge into the saved "
                                              "results ('ablations' for the ablation variants)")
    a = p.parse_args()
    if a.report_only:
        o = Path(a.out) if a.out else OUT / a.state.replace(" ", "_")
        print(report(pd.read_csv(o / "evaluation.csv.gz"), o))
    else:
        print(run(Path(a.out) if a.out else None, a.seeds, heavy=not a.no_heavy, jobs=a.jobs, target=a.state,
                  only=[m for m in a.only.split(",") if m] or None))
