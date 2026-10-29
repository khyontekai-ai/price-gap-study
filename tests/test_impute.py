import numpy as np
import pytest

from gapfill import impute


def factor_panel(T=100, N=12, seed=0, dense=4):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 0.05, T))
    Y = np.empty((T, N))
    for m in range(N):
        e = np.zeros(T)
        for t in range(1, T):
            e[t] = 0.7 * e[t - 1] + rng.normal(0, 0.03)
        Y[:, m] = 7 + 0.1 * m / N + (0.8 + 0.4 * rng.random()) * f + e
    p = np.r_[np.full(dense, 0.95), np.full(N - dense, 0.4)]
    obs = rng.random((T, N)) < p[None, :]
    return Y, obs


def test_methods_keep_reported_values_and_fill_everything():
    Y, obs = factor_panel()
    Yo = np.where(obs, Y, np.nan)
    for name, fn in {k: v for k, v in impute.METHODS.items() if k not in ("brits", "saits", "dfm_em", "chained_ml")}.items():
        out = fn(Yo)
        assert not np.isnan(out).any(), name
        np.testing.assert_allclose(out[obs], Y[obs], atol=1e-9, err_msg=name)


def test_bridge_beats_carry_forward_when_markets_share_a_factor():
    Y, obs = factor_panel(seed=2)
    rng = np.random.default_rng(0)
    masks = impute.transplant_masks(obs, range(4), rng)
    s = impute.summarise(impute.evaluate(np.where(obs, Y, np.nan), masks, methods=["locf", "bridge"]))
    assert s.loc["bridge", "mape"] < s.loc["locf", "mape"]


def test_retrospective_filter_has_limited_future_parameter_sensitivity():
    Y, obs = factor_panel(seed=3)
    Yo = np.where(obs, Y, np.nan)
    Yo[60:, 0] = np.nan                       # series 0 goes silent after week 60
    a = impute.bridge(Yo, online=True)
    Yb = Yo.copy()
    Yb[70:, 1:] += 1.0                        # change other series only after week 70
    b = impute.bridge(Yb, online=True)
    # Fills at weeks before 70 may move only through re-estimated parameters, not through f[70:].
    assert np.max(np.abs(a[60:70, 0] - b[60:70, 0])) < 0.2


@pytest.mark.parametrize("method", ["bridge", "bridge_auto"])
def test_origin_fit_is_exactly_invariant_to_future_reports(method):
    Y, obs = factor_panel(T=65, N=5, seed=3, dense=2)
    panel = np.where(obs, Y, np.nan)
    panel[40, 0] = np.nan
    changed = panel.copy()
    changed[41:] = 100.0
    a = impute.fill_at_origin(panel, 40, method=method, return_var=True, n_primary=3)
    b = impute.fill_at_origin(changed, 40, method=method, return_var=True, n_primary=3)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, rtol=0, atol=0, equal_nan=True)


def test_origin_fit_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="origin"):
        impute.fill_at_origin(np.ones((5, 2)), 5)


def test_ar_bridge_returns_to_reports_at_both_ends():
    Y, obs = factor_panel(seed=4)
    Yo = np.where(obs, Y, np.nan)
    out, var = impute.bridge(Yo, return_var=True)
    assert (var[obs] == 0).all() and (var[~obs] > 0).all()


def test_transplant_masks_hide_only_reported_weeks():
    _, obs = factor_panel(seed=5)
    masks = impute.transplant_masks(obs, range(4), np.random.default_rng(1))
    for m, hide in masks.items():
        assert (obs[hide, m]).all()
        assert (obs[:, m] & ~hide).sum() >= 6


def test_basis_mode_follows_neighbours_through_an_assam_outage():
    rng = np.random.default_rng(11)
    T = 90
    fN = np.cumsum(rng.normal(0, 0.06, T))
    d = 0.2 + np.cumsum(rng.normal(0, 0.005, T))          # slowly drifting Assam premium
    Y = np.empty((T, 10))
    for m in range(5):                                      # Assam markets
        Y[:, m] = 7 + fN + d + rng.normal(0, 0.03, T)
    for m in range(5, 10):                                  # neighbouring-state markets
        Y[:, m] = 6.9 + fN + rng.normal(0, 0.03, T)
    Yh = Y.copy()
    Yh[40:52, :5] = np.nan                                  # every Assam market silent for 12 weeks
    err = {}
    for name, kw in {"none": dict(helper_weight=0.0), "basis": dict(basis=True)}.items():
        out = impute.bridge(Yh, n_primary=5, **kw)
        err[name] = np.mean(np.abs(out[40:52, :5] - Y[40:52, :5]))
    assert err["basis"] < 0.5 * err["none"]


def test_random_walk_bridge_is_linear_interpolation_when_there_is_no_factor():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 0.05, 60))
    Yo = y[:, None].copy()
    Yo[20:30, 0] = np.nan
    out, var = impute._bridge_fill(Yo, idio="rw", use_factor=False)
    lin = impute.linear(Yo)
    np.testing.assert_allclose(out[20:30, 0], lin[20:30, 0], atol=1e-9)
    assert var[25, 0] > var[20, 0] > 0          # widest in the middle of the gap, none outside it


def test_bridge_auto_weights_sum_to_one_and_keep_reports():
    Y, obs = factor_panel(seed=12)
    Yo = np.where(obs, Y, np.nan)
    out, var, w = impute.bridge_auto(Yo, n_primary=6, return_var=True, return_weights=True)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    np.testing.assert_allclose(out[obs], Y[obs])
    assert (var[~obs] > 0).all()
