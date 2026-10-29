import numpy as np
import pandas as pd
import pytest

from gapfill import impute, stats
from tests.test_impute import factor_panel


def test_holm_matches_hand_calculation():
    np.testing.assert_allclose(stats.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def _ev():
    Y, obs = factor_panel(seed=7)
    masks = impute.transplant_masks(obs, range(4), np.random.default_rng(0))
    ev = impute.evaluate(np.where(obs, Y, np.nan), masks, methods=["locf", "linear", "bridge"])
    return ev.assign(commodity="x", protocol="transplant", run=0, online=False)


def test_paired_and_bootstrap_agree_on_direction():
    ev = _ev()
    pt = stats.paired(ev, "bridge", ["locf"]).iloc[0]
    bs = stats.bootstrap_diff(ev, "bridge", "locf", n_boot=300)
    assert pt["mape_ours"] < pt["mape_baseline"] and pt["p_value"] < 0.05
    assert bs["ci_low"] <= bs["diff_pct_points"] <= bs["ci_high"] and bs["diff_pct_points"] > 0


def test_friedman_ranks_cover_every_method():
    ev = pd.concat([_ev().assign(commodity=c) for c in "abc"])
    fr = stats.friedman_nemenyi(ev, ["locf", "linear", "bridge"])
    assert set(fr["ranks"].index) == {"locf", "linear", "bridge"} and fr["cd"] > 0


@pytest.mark.parametrize("kw", [dict(use_factor=False), dict(use_ar=False), dict(lags=(0,)),
                                dict(free_loadings=False)])
def test_model_ablations_still_fill_every_gap(kw):
    Y, obs = factor_panel(seed=8)
    out = impute.bridge(np.where(obs, Y, np.nan), **kw)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[obs], Y[obs])


def test_calibration_widens_too_narrow_intervals():
    ev = _ev()
    cal = stats.calibration(ev).set_index("method")
    assert 60 <= cal.loc["bridge", "cover80"] <= 99   # toy panel: few cells per gap bin, so loose
