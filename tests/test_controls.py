import numpy as np
import pandas as pd

from gapfill import impute
from gapfill.controls import causal_clean
from tests.test_impute import factor_panel


def _days(seed=0, n_days=200):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for mk in ("A", "B", "C"):
        for d in dates[rng.random(n_days) < 0.6]:
            rows.append({"date": d, "state": "Assam", "market": mk, "commodity": "Potato",
                         "price": float(np.exp(7 + 0.002 * (d - dates[0]).days + rng.normal(0, 0.05)))})
    return pd.DataFrame(rows)


def test_causal_clean_decisions_ignore_later_reports():
    days = _days()
    cut = pd.Timestamp("2025-04-30")      # a Wednesday: the week of the cut is complete in both copies
    week_end = cut + pd.Timedelta(days=4)
    early = days[days["date"] <= week_end]
    later = days[days["date"] > week_end].assign(price=lambda d: d["price"] * 50)   # wild future data
    a = causal_clean(early)
    b = causal_clean(pd.concat([early, later]))
    key = ["date", "market", "price"]
    np.testing.assert_array_equal(a.sort_values(key)[key].to_numpy(),
                                  b[b["date"] <= week_end].sort_values(key)[key].to_numpy())


def test_causal_clean_drops_a_lone_unit_error():
    days = _days(seed=1)
    i = days.index[len(days) // 2]
    days.loc[i, "price"] *= 100
    out = causal_clean(days)
    assert len(out) == len(days) - 1


def test_back_gap_counts_weeks_since_previous_report():
    Y = np.array([[1.0], [np.nan], [np.nan], [2.0], [np.nan]])
    np.testing.assert_array_equal(impute._back_gap(Y)[:, 0], [5, 1, 2, 3, 1])


def test_robust_calibration_widens_short_panel_intervals():
    Y, obs = factor_panel(T=40, N=6, seed=4, dense=2)
    Yo = np.where(obs, Y, np.nan)
    Yo[-1, 0] = np.nan
    _, v_legacy = impute.bridge_auto(Yo, online=True, return_var=True, n_primary=4)
    _, v_robust = impute.bridge_auto(Yo, online=True, return_var=True, n_primary=4, calibration="robust")
    assert np.isfinite(v_robust[-1, 0]) and v_robust[-1, 0] > 0
    assert not np.array_equal(v_legacy, v_robust)
