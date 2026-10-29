"""Matched B/C comparisons using saved predictions only; no model refitting."""
import numpy as np
import pandas as pd

from .controls import CTRL, KEY, STATES
from .study import NORTH_EAST, OUT


def matched_predictions(b, c, method):
    b = b[b.method == method]
    c = c[c.method == method]
    joined = c.merge(b, on=KEY, suffixes=("_c", "_b"), validate="one_to_one")
    if len(joined) != len(c):
        raise ValueError(f"B does not cover every C cell for {method}")
    if not np.allclose(joined.actual_c, joined.actual_b, rtol=1e-10, atol=1e-8):
        raise ValueError("Scoring truth differs between controls")
    return joined


def main():
    rows, pooled = [], []
    for state in STATES:
        b = pd.read_csv(CTRL / f"strict_{state}.csv.gz")
        c = pd.read_csv(CTRL / f"causal_{state}.csv.gz")
        for method in ("bridge", "bridge_auto"):
            j = matched_predictions(b, c, method)
            j["state"] = state
            j["method"] = method
            pooled.append(j)
    all_rows = pd.concat(pooled, ignore_index=True)
    for state in (*STATES, "North-East pooled"):
        selected = all_rows[all_rows.state.isin(NORTH_EAST)] if state == "North-East pooled" else all_rows[all_rows.state == state]
        for method, g in selected.groupby("method"):
            ape_b = 100 * (g.filled_b - g.actual_b).abs() / g.actual_b
            ape_c = 100 * (g.filled_c - g.actual_c).abs() / g.actual_c
            rows.append(dict(state=state, method=method, cells=len(g), mape_b=ape_b.mean(),
                             mape_c=ape_c.mean(), change=ape_c.mean()-ape_b.mean(),
                             interpretation="input/preprocessing change" if method == "bridge" else "combined input and ensemble change"))
    result = pd.DataFrame(rows)
    result.to_csv(CTRL / "matched_preprocessing_revised.csv", index=False)
    text = "# Matched preprocessing comparison (saved predictions only)\n\n"
    text += "B and C are paired on C's cells, with identical scoring truth. Fixed bridge settings are unchanged; "
    text += "its contrast isolates the change in inputs/cleaning/eligibility for that model. "
    text += "The adaptive ensemble contrast combines this intervention with robust inner-validation changes; it cannot isolate either effect.\n\n"
    text += result.to_csv(index=False)
    (CTRL / "matched_preprocessing_revised.md").write_text(text, encoding="utf-8")
    lines = [r"\begin{table*}[t]", r"\caption{Matched B--C MAPE (\%) on C's cells, using saved predictions. Fixed bridge uses identical model settings in both arms, isolating the input/preprocessing intervention for that model. Adaptive ensemble C also changes inner validation and stacking weights, so its contrast is a combined intervention. Change is C minus B, in percentage points.}",
             r"\label{tab:matchedprep}", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Fixed bridge: input change} & \multicolumn{3}{c}{Adaptive ensemble: combined change} & \\",
             r"State & B & C & Change & B & C & Change & Cells \\", r"\midrule"]
    for state in (*STATES, "North-East pooled"):
        g = result[result.state == state].set_index("method")
        a, b = g.loc["bridge"], g.loc["bridge_auto"]
        lines.append(" & ".join([state, *[f"{r[k]:.2f}" for r in (a,b) for k in ("mape_b","mape_c","change")], str(int(a.cells))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT.parent / "paper" / "tab_matchedprep_revised.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
