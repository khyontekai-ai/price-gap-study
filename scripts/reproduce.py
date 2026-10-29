"""Recompute selected manuscript results from saved predictions, without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from gapfill import inference
from gapfill.revision_controls import matched_predictions
from gapfill.study import BASELINES, NORTH_EAST, common_cells
from scripts.release_safety import ROOT, sha256, verify_manifest

STATES = inference.STATES
METHODS = BASELINES + ["bridge_auto", "bridge"]
CELL = ["commodity", "protocol", "run", "online", "series", "week"]
CALENDAR_KEY = ["state", "commodity", "series", "week"]


def load_evaluation(path):
    ev = pd.read_csv(path, low_memory=False)
    required = {"method", "commodity", "protocol", "run", "series", "week", "actual", "filled"}
    if not required <= set(ev):
        raise ValueError(f"Missing evaluation columns in {path.name}: {sorted(required - set(ev))}")
    if not np.isfinite(ev.actual).all() or (ev.actual <= 0).any():
        raise ValueError("Scoring truth must be positive and finite")
    if np.isinf(ev.filled).any():
        raise ValueError("Infinite prediction")
    keys = [c for c in CELL if c in ev] + ["method"]
    if ev.duplicated(keys).any():
        raise ValueError("Duplicate method/cell evaluation rows")
    if "online" in ev:
        if not ev.online.isin([True, False]).all():
            raise ValueError("online must contain booleans, not truthy strings")
        ev["online"] = ev.online.astype(bool)
    return ev


def with_calendar(ev, lookup):
    if lookup.duplicated(CALENDAR_KEY).any():
        raise ValueError("Duplicate calendar mapping")
    out = ev.merge(lookup, on=CALENDAR_KEY, how="left", validate="many_to_one", suffixes=("", "_reference"))
    if out.calendar_week.isna().any():
        raise ValueError("Missing calendar mapping; do not guess panel start dates")
    if not np.allclose(out.actual, out.actual_reference, rtol=1e-9, atol=1e-8):
        raise ValueError("Saved truth differs from validated calendar mapping")
    out["calendar_week"] = pd.to_datetime(out.calendar_week)
    inference.calendar_blocks(pd.DatetimeIndex(out.calendar_week))
    return out.drop(columns="actual_reference")


def main_table(frames):
    selected = []
    for state, ev in frames.items():
        ev = ev[~ev.online & ~ev.protocol.str.startswith("sweep") & ev.method.isin(METHODS)]
        ev = common_cells(ev)
        selected.append(ev.assign(state=state))
    cells = pd.concat(selected, ignore_index=True)
    truth = cells.groupby(["state", *CELL]).actual.agg(["min", "max"])
    if not np.allclose(truth["min"], truth["max"], rtol=1e-10, atol=1e-8):
        raise ValueError("Methods have different scoring truth")
    counts = cells.groupby(["state", *CELL]).method.nunique()
    if not (counts == len(METHODS)).all():
        raise ValueError("The archived main benchmark is no longer a twelve-method common cohort")
    cells["ape"] = 100 * (cells.filled - cells.actual).abs() / cells.actual
    table = cells.groupby(["method", "state"]).ape.mean().unstack().reindex(METHODS)
    table["North-East pooled"] = cells[cells.state.isin(NORTH_EAST)].groupby("method").ape.mean()
    table = table[[*STATES, "North-East pooled"]]
    return table, cells


def compare_csv(actual, expected_path, keys, columns, atol=1e-8):
    expected = pd.read_csv(expected_path)
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    if left.index.has_duplicates or right.index.has_duplicates or not left.index.equals(right.index):
        raise ValueError(f"Result cohort differs: {expected_path.name}")
    for column in columns:
        if not np.allclose(left[column], right[column], equal_nan=True, rtol=1e-9, atol=atol):
            raise ValueError(f"Numerical mismatch: {expected_path.name}/{column}")
    return len(expected) * len(columns)


def control_comparisons(controls):
    rows, pairs = [], []
    for state in STATES:
        for method in ("bridge", "bridge_auto"):
            joined = matched_predictions(controls["strict"][state], controls["causal"][state], method)
            pairs.append(joined.assign(state=state, method=method))
    joined = pd.concat(pairs, ignore_index=True)
    for scope in (*STATES, "North-East pooled"):
        selected = joined[joined.state.isin(NORTH_EAST)] if scope == "North-East pooled" else joined[joined.state == scope]
        for method, group in selected.groupby("method"):
            b = 100 * (group.filled_b - group.actual_b).abs() / group.actual_b
            c = 100 * (group.filled_c - group.actual_c).abs() / group.actual_c
            rows.append(dict(state=scope, method=method, cells=len(group), mape_b=b.mean(), mape_c=c.mean(), change=c.mean()-b.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path, help="Separate local/safely released artifact folder")
    parser.add_argument("--out", type=Path, default=ROOT / "_reproduced")
    parser.add_argument("--skip-inference", action="store_true", help="Quick partial check; not full fast-path verification")
    args = parser.parse_args()
    start = time.perf_counter()
    artifacts = args.artifacts.resolve()
    verify_manifest(artifacts, "artifact-manifest.json", "saved-predictions")
    out = args.out.resolve()
    if out == artifacts or out.is_relative_to(artifacts) or artifacts.is_relative_to(out):
        parser.error("Output must be separate from immutable input artifacts")
    out.mkdir(parents=True, exist_ok=True)
    inputs = artifacts / "outputs"
    frames = {state: load_evaluation(inputs / state / "evaluation.csv.gz") for state in STATES}
    table, cells = main_table(frames)
    table.to_csv(out / "main_mape.csv", float_format="%.12g")
    expected = pd.read_csv(ROOT / "expected/main_mape_display.csv").set_index("method")
    # Match the paper's one-decimal display, without treating rounded figures as exact statistics.
    if not np.allclose(table.loc[expected.index, expected.columns].round(1), expected):
        raise ValueError("Main table no longer matches the paper's 96 displayed values")
    print("Verified all 96 displayed main-table values.", flush=True)
    controls = {prefix: {state: load_evaluation(inputs / "controls" / f"{prefix}_{state}.csv.gz")
                         for state in STATES} for prefix in ("strict", "causal")}
    matched = control_comparisons(controls)
    n_matched = compare_csv(matched, ROOT / "expected/matched_preprocessing.csv", ["state", "method"],
                            ["cells", "mape_b", "mape_c", "change"])
    matched.to_csv(out / "matched_preprocessing.csv", index=False)
    print("Verified matched B/C fixed-bridge and adaptive contrasts.", flush=True)
    # Sensitivity summaries are recomputed, not copied from the saved summary markdown.
    cells.groupby(["state", "method", "protocol"]).ape.agg(["mean", "size"]).to_csv(out / "scenario_mape.csv")
    adaptive = cells[cells.method == "bridge_auto"].copy()
    for field in ("gap", "sd"):
        if field not in adaptive:
            raise ValueError(f"Missing interval/gap field: {field}")
    gap = pd.cut(adaptive.gap, bins=[-1, 1, 3, 8, np.inf], labels=["1", "2-3", "4-8", "9+"])
    adaptive["gap_bin"] = gap
    coverage = []
    for nominal, quantile in ((80, 1.2815515655446004), (95, 1.959963984540054)):
        valid = adaptive.sd.notna() & (adaptive.sd >= 0) & (adaptive.filled > 0)
        subset = adaptive[valid].copy()
        subset["covered"] = (np.log(subset.actual / subset.filled).abs() <= quantile * subset.sd)
        summary = subset.groupby(["state", "gap_bin"], observed=True).covered.agg(["mean", "size"]).reset_index()
        coverage.append(summary.assign(nominal_pct=nominal, coverage_pct=100 * summary["mean"]))
    pd.concat(coverage, ignore_index=True).to_csv(out / "interval_coverage.csv", index=False)
    inference_cells = 0
    if not args.skip_inference:
        lookup = pd.read_csv(artifacts / "calendar_cells.csv.gz")
        offline = pd.concat([ev[~ev.online & ev.method.isin(BASELINES + ["bridge_auto"])].assign(state=state)
                             for state, ev in frames.items()], ignore_index=True)
        scopes = {"North-East pooled": NORTH_EAST, "All seven": STATES, **{state: (state,) for state in STATES}}
        pieces = [inference.analyse(with_calendar(offline, lookup), BASELINES, scopes, "offline")]
        for prefix, label in (("strict", "origin-restricted"), ("causal", "origin-restricted, causal preprocessing")):
            # Preserve original sorted filename order: this also preserves floating-point summation order.
            ev = pd.concat([controls[prefix][state].assign(state=state) for state in sorted(STATES)], ignore_index=True)
            pieces.append(inference.analyse(with_calendar(ev, lookup), ["locf", "kalman", "bridge"], scopes, label))
        result = pd.concat(pieces, ignore_index=True)
        columns = ["mape_ours", "mape_base", "diff", "cl_lo", "cl_hi", "tw_lo", "tw_hi", "units", "blocks", "cells", "units_better", "p_unit", "p_unit_holm"]
        inference_cells = compare_csv(result, ROOT / "expected/inference_calendar.csv", ["setting", "scope", "baseline"], columns)
        result.to_csv(out / "inference_calendar.csv", index=False)
        print("Verified corrected calendar-block inference (2,000 draws; seed 0).", flush=True)
    report = dict(status="passed" if not args.skip_inference else "partial", model_refits=False,
                  main_display_values_checked=96, matched_values_checked=n_matched,
                  inference_values_checked=inference_cells,
                  artifact_manifest_sha256=sha256(artifacts / "artifact-manifest.json"),
                  elapsed_seconds=round(time.perf_counter()-start, 2),
                  scope="main table, matched B/C, calendar inference; interval/scenario summaries also regenerated",
                  exclusions="raw-data refitting, downstream model/decision reruns, original runtime measurements")
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
