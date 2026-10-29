"""North-East India summary across target states (run after `study --state X` for each state).

- Method x state error tables (offline and online) and our gain over the best baseline in each state.
- Friedman test and Nemenyi critical difference with every (state, commodity) as a dataset.
- Paired Wilcoxon (Holm) and cluster-bootstrap intervals pooled over all states.
- "Before and after" coverage: share of market-commodity-weeks with a real report, and with an
  estimate whose calibrated 80% range is within +/-15% of the price.
Settings were chosen on Assam's development run only; other states are a transfer test.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import impute, stats
from .sources import data, monthly
from .study import BASELINES, FIXED, HELPER_STATES, NORTH_EAST, OURS, OUT, ROOT, _table, bridge_kw, common_cells, panel

warnings.filterwarnings("ignore")


def load(states) -> pd.DataFrame:
    frames = []
    for st in states:
        f = OUT / st.replace(" ", "_") / "evaluation.csv.gz"
        if f.exists():
            e = pd.read_csv(f)
            e = e[~e["protocol"].str.startswith("sweep")]
            e = pd.concat([common_cells(e[~e["online"]]), e[e["online"]]], ignore_index=True)   # identical cells
            frames.append(e.assign(state=st, commodity=st + " | " + e["commodity"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage(state: str, width: float = 0.15) -> pd.DataFrame:
    """Per commodity: share of target-market weeks reported, and share with a usable estimate after filling."""
    days, _ = data.clean_outliers(data.load_days(states=[state, *HELPER_STATES[state]]))
    sm = monthly.load_statewise()
    rows = []
    for c, n in days[days["state"] == state]["commodity"].value_counts().items():
        if n < 150:
            continue
        built = panel(days, sm, c, target=state, min_assam=4)
        if built is None:
            continue
        Y, cols, idx, n_t = built
        filled, var = impute.bridge_auto(Y, return_var=True, n_primary=n_t)
        obs = ~np.isnan(Y[:, :n_t])
        half = 1.2816 * np.sqrt(var[:, :n_t])
        usable = obs | (np.expm1(half) <= width)
        rows.append({"state": state, "commodity": c, "market_weeks": obs.size,
                     "reported_pct": 100 * obs.mean(), "usable_pct": 100 * usable.mean()})
    return pd.DataFrame(rows)


def run(states=NORTH_EAST + ("Goa", "Bihar")) -> str:
    out = OUT / "north_east"
    out.mkdir(parents=True, exist_ok=True)
    ev = load(states)
    if ev.empty:
        return "No state results yet."
    ev["ape"] = (ev["filled"] - ev["actual"]).abs() / ev["actual"]
    present = [s for s in states if s in set(ev["state"])]
    lines = ["# Gap filling across North-East India (and two out-of-region states)", "",
             "Our method's settings were chosen on Assam's development run only and applied unchanged to "
             "every other state (a transfer test). Error = MAPE on real reports hidden and then filled.", ""]
    baselines = [m for m in BASELINES if m in set(ev["method"])]
    methods = baselines + [OURS, FIXED]
    for online in (False, True):
        e = ev[(ev["online"] == online) & ev["method"].isin(methods)]
        tab = e.groupby(["method", "state"])["ape"].mean().unstack()[present] * 100
        ne = [s for s in NORTH_EAST if s in tab.columns]
        tab["North-East pooled"] = e[e["state"].isin(ne)].groupby("method")["ape"].mean() * 100
        tab = tab.sort_values("North-East pooled")
        tab.index.name = "Method"
        lines += [f"## {'Online (nowcasting)' if online else 'Offline (filling history)'}: MAPE by state", ""] + _table(tab) + [""]
        best_base = tab.drop(index=[OURS, FIXED], errors="ignore")
        gain = pd.DataFrame({"best baseline": best_base.idxmin(), "baseline MAPE": best_base.min(),
                             "ours": tab.loc[OURS] if OURS in tab.index else np.nan,
                             "ours (fixed settings)": tab.loc[FIXED] if FIXED in tab.index else np.nan})
        gain["error cut (%)"] = 100 * (1 - gain["ours"] / gain["baseline MAPE"])
        gain.index.name = "State"
        lines += ["Our method against the best baseline in each state:", ""] + _table(gain) + [""]
        others = [m for m in baselines if m in set(e["method"])]
        nee = e[e["state"].isin(ne)]
        pt = stats.paired(nee, OURS, others)
        if len(pt):
            pt = pt.set_index("baseline")
            pt.index.name = "Baseline (North-East pooled)"
            lines += ["Paired one-sided Wilcoxon tests, Holm-corrected:", ""] + _table(pt, "{:.3g}") + [""]
        bs = pd.DataFrame([stats.bootstrap_diff(nee, OURS, o, n_boot=1000) for o in others]).set_index("baseline")
        bs.index.name = "Baseline (North-East pooled)"
        lines += ["MAPE(baseline) - MAPE(ours), points, 95% cluster-bootstrap CI:", ""] + _table(bs, "{:.2f}") + [""]
        fr = stats.friedman_nemenyi(nee, others + [OURS])
        rk = fr["ranks"].to_frame("average rank")
        rk.index.name = "Method"
        lines += [f"Average rank over {fr['datasets']} (state, commodity) datasets in the North-East; Friedman p = "
                  f"{fr['friedman_p']:.2g}, Nemenyi CD = {fr['cd']:.2f}:", ""] + _table(rk, "{:.2f}") + [""]

    import json
    cov = pd.concat([coverage(s) for s in present], ignore_index=True)
    cov.to_csv(out / "coverage.csv", index=False)
    w = cov.assign(rep=cov["reported_pct"] * cov["market_weeks"], use=cov["usable_pct"] * cov["market_weeks"])
    g = w.groupby("state")[["rep", "use", "market_weeks"]].sum()
    ct = pd.DataFrame({"market-commodity-weeks": g["market_weeks"], "with a real report (%)": g["rep"] / g["market_weeks"],
                       "with a usable estimate after filling (%)": g["use"] / g["market_weeks"]}).loc[present]
    ct.index.name = "State"
    lines += ["## Price coverage before and after filling", "",
              "Usable estimate = a real report, or a fill whose calibrated 80% range is within +/-15% of the price.", ""] + \
        _table(ct) + [""]
    _coverage_figure(ct, out)
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    fig = ROOT / "paper" / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        import shutil
        shutil.copy2(f, fig / f.name)
    return text


def _coverage_figure(ct: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(ct))
    ax.barh(y + 0.2, ct.iloc[:, 1], height=0.38, color="#bbbbbb", label="real reports")
    ax.barh(y - 0.2, ct.iloc[:, 2], height=0.38, color="#1b6e4c", label="usable after filling (80% range within ±15%)")
    ax.set_yticks(y, ct.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of market-commodity-weeks (%)")
    ax.set_title("Weekly market price coverage")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ne_coverage.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(run())
