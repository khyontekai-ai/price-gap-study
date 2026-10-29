"""Filling gaps in thin-market price series by borrowing from comparable markets.

Input per commodity: a weekly matrix of log prices, rows = weeks, columns = series (Assam markets,
markets in neighbouring states, and state-level monthly averages placed in mid-month weeks), with
NaN where nothing was reported.

Methods
  locf            last observation carried forward (what a dashboard shows)
  linear          linear interpolation in time, per series
  kalman          local-level model per series, Kalman smoother (univariate state-space)
  softimpute      low-rank matrix completion (as used for Agmarknet in PECAD, arXiv 2304.09761)
  knn             nearest comparable markets, offset by their typical price gap
                  (collaborative filtering across mandis, as in arXiv 1812.05173)
  bridge          ours: cross-market dynamic factor with lead-lag loadings and an AR(1) bridge
                  for each market's own deviation (see `bridge_fit`)

`bridge` in short. For one commodity, every series m follows
    y[m, t] = a[m] + b[m] * f[t - lag[m]] + e[m, t],    e[m, t] = rho[m] * e[m, t-1] + noise
f is a common price factor (local level, random walk) estimated by a Kalman smoother from all
markets at once, including dense neighbouring-state markets and monthly state averages, so a week
with no Assam report still has an estimate of where the regional price was. lag lets destination
markets follow source markets with a delay. A market's own deviation e is known on its reported
weeks and is bridged across gaps by the exact AR(1) interpolation, so the fill returns to the
market's own level near its reports and to the regional signal in long gaps. Each fill comes with
a variance, giving prediction intervals. `online=True` filters the state forward but estimates
parameters on the supplied panel; it is retrospective unless the panel is truncated at the origin.
Use `fill_at_origin` for a fit restricted to a supplied information cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------
# Baselines

def locf(Y: np.ndarray) -> np.ndarray:
    return pd.DataFrame(Y).ffill().bfill().to_numpy()


def fill_at_origin(Y: np.ndarray, origin: int, method: str = "bridge_auto", **kw):
    """Fit on Y[:origin+1] and return only the origin row (log-price scale).

    Callers must also restrict cleaning and helper publication dates to the information cutoff.
    No future row, including one used for parameter or weight estimation, enters this fit.
    """
    if not isinstance(origin, (int, np.integer)) or not 0 <= origin < len(Y):
        raise ValueError("origin must be an integer row index inside Y")
    if method not in {"bridge", "bridge_auto"}:
        raise ValueError("method must be bridge or bridge_auto")
    prefix = np.asarray(Y[:origin + 1], dtype=float).copy()
    result = {"bridge": bridge, "bridge_auto": bridge_auto}[method](prefix, online=True, **kw)
    if isinstance(result, tuple):
        return tuple(value[-1].copy() if np.shape(value) == prefix.shape else value for value in result)
    return result[-1].copy()


def linear(Y: np.ndarray) -> np.ndarray:
    return pd.DataFrame(Y).interpolate(limit_direction="both").to_numpy()


def _local_level(y: np.ndarray, q: float, r: float, smooth: bool = True):
    """Scalar local-level Kalman filter (+ RTS smoother). Returns mean, variance, log-likelihood."""
    n = y.size
    obs = ~np.isnan(y)
    if not obs.any():
        return np.full(n, np.nan), np.full(n, np.inf), -np.inf
    x, p = y[obs][0], 1e2 * (r + q)
    xf, pf, xp, pp = (np.empty(n) for _ in range(4))
    ll = 0.0
    for t in range(n):
        if t:
            x, p = x, p + q
        xp[t], pp[t] = x, p
        if obs[t]:
            s = p + r
            k = p / s
            v = y[t] - x
            ll += -0.5 * (np.log(2 * np.pi * s) + v * v / s)
            x, p = x + k * v, (1 - k) * p
        xf[t], pf[t] = x, p
    if not smooth:
        return xf, pf, ll
    xs, ps = xf.copy(), pf.copy()
    for t in range(n - 2, -1, -1):
        c = pf[t] / pp[t + 1]
        xs[t] = xf[t] + c * (xs[t + 1] - xp[t + 1])
        ps[t] = pf[t] + c * c * (ps[t + 1] - pp[t + 1])
    return xs, ps, ll


def kalman(Y: np.ndarray, online: bool = False) -> np.ndarray:
    out = np.empty_like(Y)
    for j in range(Y.shape[1]):
        y = Y[:, j]
        d = np.diff(y[~np.isnan(y)])
        v = max(float(np.var(d)) if d.size > 2 else 1e-2, 1e-6)   # constant series: avoid a zero variance
        best = None
        for ratio in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            r = v / (2 + ratio)
            q = ratio * r
            res = _local_level(y, q, r, smooth=not online)
            if best is None or res[2] > best[2]:
                best = res
        out[:, j] = best[0]
    out = np.where(np.isnan(Y), out, Y)   # keep reports as they are; fill only the gaps
    return np.where(np.isnan(out), locf(Y), out)


def softimpute(Y: np.ndarray, shrink: float = 0.15, iters: int = 100, tol: float = 1e-5) -> np.ndarray:
    obs = ~np.isnan(Y)
    mu = np.nanmean(Y, axis=0)
    mu = np.where(np.isnan(mu), np.nanmean(Y), mu)
    X = np.where(obs, Y, mu) - mu
    lam = None
    Z = np.zeros_like(X)
    for _ in range(iters):
        filled = np.where(obs, Y - mu, Z)
        U, s, Vt = np.linalg.svd(filled, full_matrices=False)
        if lam is None:
            lam = shrink * s[0]
        s = np.maximum(s - lam, 0)
        Znew = (U * s) @ Vt
        if np.linalg.norm(Znew - Z) <= tol * max(np.linalg.norm(Z), 1e-9):
            Z = Znew
            break
        Z = Znew
    return np.where(obs, Y, Z + mu)


def knn(Y: np.ndarray, k: int = 5, min_common: int = 8) -> np.ndarray:
    T, N = Y.shape
    obs = ~np.isnan(Y)
    out = Y.copy()
    dY = np.diff(Y, axis=0)
    for m in range(N):
        cands = []
        for j in range(N):
            if j == m:
                continue
            common = obs[:, m] & obs[:, j]
            if common.sum() < min_common:
                continue
            gap = float(np.median(Y[common, m] - Y[common, j]))
            both = ~np.isnan(dY[:, m]) & ~np.isnan(dY[:, j])
            c = np.corrcoef(dY[both, m], dY[both, j])[0, 1] if both.sum() >= min_common else 0.0
            cands.append((0.0 if np.isnan(c) else c, j, gap))
        cands.sort(reverse=True)
        cands = [c for c in cands if c[0] > 0][:k] or cands[:k]
        for t in np.where(~obs[:, m])[0]:
            vals = [(max(c, 1e-3), Y[t, j] + gap) for c, j, gap in cands if obs[t, j]]
            if vals:
                w = np.array([v[0] for v in vals])
                out[t, m] = float(np.dot(w, [v[1] for v in vals]) / w.sum())
    return np.where(np.isnan(out), locf(Y), out)


# --------------------------------------------------------------------------------------------
# Bridge: cross-market dynamic factor + AR(1) bridge

@dataclass
class BridgeFit:
    a: np.ndarray
    b: np.ndarray
    lag: np.ndarray
    rho: np.ndarray
    sigma2: np.ndarray       # stationary variance of each series' own deviation e
    f: np.ndarray            # common factor (smoothed, or filtered when online)
    fvar: np.ndarray
    q: float
    F: Optional[np.ndarray] = None    # factor each series loads on (T x N); differs by series in basis mode
    FV: Optional[np.ndarray] = None

    def factor(self, m: int):
        if self.F is None:
            return self.f, self.fvar
        return self.F[:, m], self.FV[:, m]


def _factor(Y: np.ndarray, a, b, lag, noise, q: float, smooth: bool):
    """Kalman (filter/smoother) for a random-walk factor observed by many series at shifted times."""
    T, N = Y.shape
    obs_at = [[] for _ in range(T)]   # (value, variance) observations of f[s]
    for m in range(N):
        if abs(b[m]) < 1e-3:
            continue
        for t in np.where(~np.isnan(Y[:, m]))[0]:
            s = t - lag[m]
            if 0 <= s < T:
                obs_at[s if smooth else t].append(((Y[t, m] - a[m]) / b[m], noise[m] / b[m] ** 2))
    x, p = 0.0, 1e3
    xf, pf, xp, pp = (np.empty(T) for _ in range(4))
    for t in range(T):
        if t:
            p = p + q
        xp[t], pp[t] = x, p
        for v, r in obs_at[t]:
            k = p / (p + r)
            x, p = x + k * (v - x), (1 - k) * p
        xf[t], pf[t] = x, p
    if not smooth:
        return xf, pf
    xs, ps = xf.copy(), pf.copy()
    for t in range(T - 2, -1, -1):
        c = pf[t] / pp[t + 1]
        xs[t] = xf[t] + c * (xs[t + 1] - xp[t + 1])
        ps[t] = pf[t] + c * c * (ps[t + 1] - pp[t + 1])
    return xs, ps


def _basis_factors(Y: np.ndarray, a, b, lag, noise, n_primary: int, qn: float, qd: float, smooth: bool):
    """Two linked random walks: the neighbours' factor fN and Assam's fA = fN + d, where the basis d
    (Assam's price gap to its neighbours) drifts slowly. Assam series observe fA, helpers fN.

    Returns (fN, d, var_fN, var_fA)."""
    T, N = Y.shape
    obs_at = [[] for _ in range(T)]
    for m in range(N):
        if abs(b[m]) < 1e-3 or not np.isfinite(noise[m]):
            continue
        h = np.array([1.0, 1.0]) if m < n_primary else np.array([1.0, 0.0])
        for t in np.where(~np.isnan(Y[:, m]))[0]:
            s = t - lag[m]
            if 0 <= s < T:
                obs_at[s if smooth else t].append(((Y[t, m] - a[m]) / b[m], noise[m] / b[m] ** 2, h))
    Q = np.diag([qn, qd])
    x, P = np.zeros(2), np.eye(2) * 1e3
    xf, xp = np.empty((T, 2)), np.empty((T, 2))
    Pf, Pp = np.empty((T, 2, 2)), np.empty((T, 2, 2))
    for t in range(T):
        if t:
            P = P + Q
        xp[t], Pp[t] = x, P
        for v, r, h in obs_at[t]:
            Ph = P @ h
            k = Ph / (h @ Ph + r)
            x = x + k * (v - h @ x)
            P = P - np.outer(k, Ph)
        xf[t], Pf[t] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    if smooth:
        for t in range(T - 2, -1, -1):
            C = Pf[t] @ np.linalg.pinv(Pp[t + 1])
            xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
            Ps[t] = Pf[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
    var_n = Ps[:, 0, 0]
    var_a = Ps[:, 0, 0] + Ps[:, 1, 1] + 2 * Ps[:, 0, 1]
    return xs[:, 0], xs[:, 1], var_n, var_a


def _shift(f: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return f
    return np.r_[np.full(lag, f[0]), f[:-lag]]


def bridge_fit(Y: np.ndarray, lags=(0, 1, 2), rounds: int = 6, online: bool = False,
               min_obs: int = 6, use_factor: bool = True, use_ar: bool = True,
               free_loadings: bool = True, n_primary: Optional[int] = None,
               helper_weight: float = 1.0, basis: bool = False) -> BridgeFit:
    """Ablation switches: use_factor=False drops the shared regional factor (each market bridges
    around its own mean); use_ar=False drops the market's own AR(1) bridge; free_loadings=False
    fixes every loading at 1; lags=(0,) removes lead-lag.

    Columns from `n_primary` on are helpers (other states' markets, state averages). Their weight in
    the factor is `helper_weight` (0 to 1): many dense helper series would otherwise set the factor
    on their own dynamics rather than Assam's."""
    T, N = Y.shape
    obs = ~np.isnan(Y)
    if not use_factor:
        a = np.nanmean(Y, axis=0)
        a = np.where(np.isnan(a), np.nanmean(Y), a)
        fit = BridgeFit(a, np.zeros(N), np.zeros(N, dtype=int), np.zeros(N), np.full(N, 0.01),
                        np.zeros(T), np.zeros(T), 0.0)
        _ar(Y, obs, fit, min_obs, use_ar)
        return fit
    a = np.nanmean(Y, axis=0)
    a = np.where(np.isnan(a), np.nanmean(Y), a)
    b = np.ones(N)
    lag = np.zeros(N, dtype=int)
    noise = np.full(N, 0.01)
    use_basis = basis and n_primary is not None and 0 < n_primary < N
    primary = np.arange(N) < (n_primary if n_primary is not None else N)
    # Start: factor = cross-sectional mean of de-meaned series.
    with np.errstate(all="ignore"):
        f0 = np.nanmean(Y - a, axis=1)
    f = pd.Series(f0).interpolate(limit_direction="both").fillna(0).to_numpy()
    q = max(float(np.var(np.diff(f))), 1e-5)
    F = np.tile(f[:, None], (1, N))          # the factor each series loads on
    FV = np.zeros((T, N))
    qd = 0.1 * q

    def factors(smooth: bool):
        if use_basis:
            fn, d, vn, va = _basis_factors(Y, a, b, lag, noise, n_primary, q, qd, smooth)
            Fm = np.where(primary[None, :], (fn + d)[:, None], fn[:, None])
            Vm = np.where(primary[None, :], va[:, None], vn[:, None])
            return Fm, Vm, fn, d
        fs, vs = _factor(Y, a, b, lag, _weighted(noise, n_primary, helper_weight), q, smooth=smooth)
        return np.tile(fs[:, None], (1, N)), np.tile(vs[:, None], (1, N)), fs, np.zeros(T)

    for _ in range(rounds):
        for m in range(N):
            y = Y[:, m]
            o = obs[:, m]
            fm = F[:, m]
            if o.sum() < min_obs:
                a[m], b[m], lag[m] = (np.nanmean(y) - fm[o].mean() if o.any() else 0.0), 1.0, 0
                continue
            best = None
            for L in lags:
                x = _shift(fm, L)[o]
                X = np.c_[np.ones(o.sum()), x]
                coef, *_ = np.linalg.lstsq(X, y[o], rcond=None)
                res = y[o] - X @ coef
                sse = float(res @ res)
                if best is None or sse < best[0] - 1e-12:
                    best = (sse, coef, L)
            _, coef, L = best
            if not free_loadings:
                coef = np.array([float(np.mean(y[o] - _shift(fm, L)[o])), 1.0])
            a[m], b[m], lag[m] = coef[0], float(np.clip(coef[1], 0.2, 3.0)), L
            res = y[o] - a[m] - b[m] * _shift(fm, L)[o]
            noise[m] = max(float(np.mean(res ** 2)), 1e-5)
        # Identify scale (mean loading 1): the factor is then re-estimated on that scale.
        b = b / float(np.mean(b))
        F, FV, fn, d = factors(smooth=True)
        # Identify level (factor mean 0), moving the level into each series' intercept.
        a = a + b * F.mean(axis=0)
        F = F - F.mean(axis=0)
        q = max(float(np.var(np.diff(fn))), 1e-5)
        if use_basis:
            qd = float(np.clip(np.var(np.diff(d)), 1e-6, q))
    if online:
        # Filter with the final a and b: on the same level as the smoothed fit, no re-centring.
        F, FV, _, _ = factors(smooth=False)
    fit = BridgeFit(a, b, lag, np.zeros(N), noise.copy(), F[:, 0], FV[:, 0], q, F, FV)
    _ar(Y, obs, fit, min_obs, use_ar)
    return fit


def _weighted(noise: np.ndarray, n_primary: Optional[int], w: float) -> np.ndarray:
    """Observation noise used for the factor: helpers count with weight w (noise / w)."""
    if n_primary is None or w >= 1.0:
        return noise
    out = noise.copy()
    out[n_primary:] = np.inf if w <= 0 else noise[n_primary:] / w
    return out


def _ar(Y: np.ndarray, obs: np.ndarray, fit: BridgeFit, min_obs: int, use_ar: bool) -> None:
    """AR(1) for each series' own deviation, from residual pairs at gaps of any length."""
    for m in range(Y.shape[1]):
        t_obs = np.where(obs[:, m])[0]
        if t_obs.size < min_obs:
            continue
        e = Y[t_obs, m] - fit.a[m] - fit.b[m] * _shift(fit.factor(m)[0], fit.lag[m])[t_obs]
        fit.sigma2[m] = max(float(np.var(e)), 1e-6)
        if not use_ar:
            continue
        num = den = 0.0
        for i in range(t_obs.size - 1):
            g = t_obs[i + 1] - t_obs[i]
            if g <= 4:
                num += e[i] * e[i + 1] / g
                den += e[i] ** 2 / g
        fit.rho[m] = float(np.clip(num / den, 0.0, 0.97)) if den > 0 else 0.0


def bridge(Y: np.ndarray, online: bool = False, return_var: bool = False, calibrate: bool = True,
           seed: int = 0, **kw):
    """Fill every missing cell. Offline: smoother + two-sided AR bridge. Online: filter + forward decay.

    With `calibrate`, the model variances are rescaled so that 80% intervals hold 80% of reports
    hidden from a second fit, separately by distance to the nearest report (model variances ignore
    parameter uncertainty and heavy-tailed price jumps, and the shortfall grows with gap length).
    Reports are hidden in the panel's own gap shapes (transplant masks), so calibration gaps look
    like real ones.
    """
    out, var = _bridge_fill(Y, online, **kw)
    if return_var and calibrate:
        rng = np.random.default_rng(seed)
        obs = ~np.isnan(Y)
        dense = [m for m in range(Y.shape[1]) if obs[:, m].sum() >= 20]
        masks = transplant_masks(obs, dense, rng)
        hide = np.zeros_like(obs)
        for m, h in masks.items():
            hide[:, m] = h
        if hide.sum() >= 30:
            Yc = np.where(hide, np.nan, Y)
            oc, vc = _bridge_fill(Yc, online, **kw)
            z = np.abs(oc[hide] - Y[hide]) / np.sqrt(np.maximum(vc[hide], 1e-12))
            gbin = np.digitize(_gap_length(Yc)[hide], CAL_BINS)
            pooled = (np.quantile(z, 0.8) / 1.2816) ** 2
            scale = {b: (np.quantile(z[gbin == b], 0.8) / 1.2816) ** 2 if (gbin == b).sum() >= 30 else pooled
                     for b in range(len(CAL_BINS) + 1)}
            cell_bin = np.digitize(_gap_length(Y), CAL_BINS)
            var = var * np.vectorize(scale.get)(cell_bin)
    return (out, var) if return_var else out


CAL_BINS = (2, 4, 9)   # gap-length bins for calibration: 1, 2-3, 4-8, 9+ weeks


def _bridge_fill(Y: np.ndarray, online: bool = False, idio: str = "ar", **kw):
    """Fill with the factor part plus a bridge of the market's own deviation e: "ar" (stationary AR(1),
    mean-reverting) or "rw" (random walk: a Brownian bridge, i.e. linear interpolation of e)."""
    fit = bridge_fit(Y, online=online, **kw)
    T, N = Y.shape
    out = Y.copy()
    var = np.zeros_like(Y)
    for m in range(N):
        fm = _shift(fit.factor(m)[0], fit.lag[m])
        fv = _shift(fit.factor(m)[1], fit.lag[m])
        base = fit.a[m] + fit.b[m] * fm
        o = ~np.isnan(Y[:, m])
        t_obs = np.where(o)[0]
        e = np.full(T, np.nan)
        e[o] = Y[o, m] - base[o]
        rho, s2 = fit.rho[m], fit.sigma2[m]
        if idio == "rw":
            dd, gg = np.diff(e[t_obs]), np.diff(t_obs)
            srw = max(float(np.mean(dd ** 2 / gg)) if dd.size else s2, 1e-8)   # variance per week of the walk
        for t in np.where(~o)[0]:
            prev = t_obs[t_obs < t]
            nxt = t_obs[t_obs > t] if not online else np.array([], dtype=int)
            if idio == "rw":
                if prev.size and nxt.size:
                    ta, tb = prev[-1], nxt[0]
                    ehat = e[ta] + (e[tb] - e[ta]) * (t - ta) / (tb - ta)
                    ev = srw * (t - ta) * (tb - t) / (tb - ta)
                elif prev.size:
                    ehat, ev = e[prev[-1]], srw * (t - prev[-1])
                elif nxt.size:
                    ehat, ev = e[nxt[0]], srw * (nxt[0] - t)
                else:
                    ehat, ev = 0.0, s2
                out[t, m] = base[t] + ehat
                var[t, m] = ev + fit.b[m] ** 2 * fv[t]
                continue
            if prev.size and nxt.size:
                ta, tb = prev[-1], nxt[0]
                ra, rb, rab = rho ** (t - ta), rho ** (tb - t), rho ** (tb - ta)
                den = 1 - rab ** 2
                if den < 1e-9:
                    ehat, ev = e[ta], 0.0
                else:
                    ehat = (ra * (1 - rb ** 2) * e[ta] + rb * (1 - ra ** 2) * e[tb]) / den
                    ev = s2 * (1 - ra ** 2) * (1 - rb ** 2) / den
            elif prev.size:
                ra = rho ** (t - prev[-1])
                ehat, ev = ra * e[prev[-1]], s2 * (1 - ra ** 2)
            elif nxt.size:
                rb = rho ** (nxt[0] - t)
                ehat, ev = rb * e[nxt[0]], s2 * (1 - rb ** 2)
            else:
                ehat, ev = 0.0, s2
            out[t, m] = base[t] + ehat
            var[t, m] = ev + fit.b[m] ** 2 * fv[t]
    return out, var


# --------------------------------------------------------------------------------------------
# Self-validating ensemble: choose how to borrow strength from the data itself

# Candidates differ in what the fill borrows: the state's own regional factor, plus the neighbouring
# states (ignored, pooled into the factor, or linked by a drifting basis), or only the market's own AR bridge.
AUTO_CANDIDATES = {
    "factor": dict(free_loadings=False, lags=(0,), helper_weight=0.0, basis=False),
    "factor+basis": dict(free_loadings=False, lags=(0,), helper_weight=1.0, basis=True),
    "factor+pooled": dict(free_loadings=False, lags=(0,), helper_weight=1.0, basis=False),
    "ar_only": dict(use_factor=False),
    "factor+rw": dict(free_loadings=False, lags=(0,), helper_weight=0.0, basis=False, idio="rw"),
    "rw_only": dict(use_factor=False, idio="rw"),
}


def bridge_auto(Y: np.ndarray, online: bool = False, return_var: bool = False, n_primary: Optional[int] = None,
                seed: int = 0, n_inner: int = 3, min_cells: int = 60, return_weights: bool = False,
                calibration: str = "legacy"):
    """Bridge with the way of borrowing strength chosen per panel by inner validation.

    Real reports of the target markets are hidden in the panel's own gap shapes (`n_inner` times, on
    data that already has the evaluation cells removed, so nothing hidden for the test leaks in). Every
    candidate in AUTO_CANDIDATES fills them, and non-negative stacking weights (summing to 1) are fitted
    on the hidden truth. The final fill is the weighted mixture of the candidates' fills; its variance
    adds the candidates' disagreement, and is then calibrated by gap length on the same hidden cells.
    With too few inner cells (tiny panels) the plain `factor` candidate is used.

    calibration="legacy" is the version of the saved study. "robust" is for short panels (origin-restricted
    fits): when the target markets give fewer than `min_cells` inner cells it also hides reports of sparser
    target markets (>= 8 reports) and then of helper markets, with up to 10 repetitions; it calibrates from
    20 inner cells on, bins by weeks since the previous report when `online` (the only report a one-sided
    fill sees), uses the finite-sample (conformal) 80% quantile and shrinks each bin's scale towards the
    pooled scale with a prior weight of 30 cells.
    """
    from scipy.optimize import nnls

    T, N = Y.shape
    obs = ~np.isnan(Y)
    has_helpers = n_primary is not None and 0 < n_primary < N
    names = [k for k in AUTO_CANDIDATES if has_helpers or k in ("factor", "ar_only", "factor+rw", "rw_only")]
    kw = {k: dict(AUTO_CANDIDATES[k], **({"n_primary": n_primary} if k.startswith("factor") else {})) for k in names}
    full = {k: _bridge_fill(Y, online, **kw[k]) for k in names}
    prim = range(n_primary if n_primary else N)
    robust = calibration == "robust"
    gap_fn = _back_gap if (robust and online) else _gap_length
    tiers = [(prim, 20, n_inner)]
    if robust:
        tiers += [(prim, 8, 10), (range(N), 8, 10)]

    truth, preds, pvars, gaps = [], [], [], []
    for tier, (cols, min_obs, reps) in enumerate(tiers):
        if tier and sum(len(t) for t in truth) >= min_cells:
            break
        dense = [m for m in cols if obs[:, m].sum() >= min_obs]
        for r in range(reps):
            masks = transplant_masks(obs, dense, np.random.default_rng(seed + r + 100 * tier)) if dense else {}
            hide = np.zeros_like(obs)
            for m, h in masks.items():
                hide[:, m] = h
            if hide.sum() < 20:
                continue
            Yc = np.where(hide, np.nan, Y)
            fits = [_bridge_fill(Yc, online, **kw[k]) for k in names]
            truth.append(Y[hide])
            preds.append(np.column_stack([f[0][hide] for f in fits]))
            pvars.append(np.column_stack([f[1][hide] for f in fits]))
            gaps.append(gap_fn(Yc)[hide])
            if tier and sum(len(t) for t in truth) >= min_cells:
                break
    w = np.zeros(len(names))
    if truth and sum(len(t) for t in truth) >= min_cells:
        P, y = np.vstack(preds), np.concatenate(truth)
        coef, _ = nnls(P, y)
        w = coef / coef.sum() if coef.sum() > 0 else w
    if w.sum() == 0:
        w[names.index("factor")] = 1.0

    out = sum(w[i] * full[k][0] for i, k in enumerate(names))
    out = np.where(obs, Y, out)
    if not return_var:
        return (out, w) if return_weights else out
    var = sum(w[i] * (full[k][1] + (full[k][0] - out) ** 2) for i, k in enumerate(names))
    var = np.where(obs, 0.0, var)
    if truth and sum(len(t) for t in truth) >= (20 if robust else min_cells):
        P, V, y, g = np.vstack(preds), np.vstack(pvars), np.concatenate(truth), np.concatenate(gaps)
        comb = P @ w
        cvar = (V + (P - comb[:, None]) ** 2) @ w
        z = np.abs(comb - y) / np.sqrt(np.maximum(cvar, 1e-12))
        gbin = np.digitize(g, CAL_BINS)
        if robust:
            def s80(x):   # finite-sample (split-conformal) 80% quantile, as a variance scale
                return (np.quantile(x, min(1.0, np.ceil(0.8 * (len(x) + 1)) / len(x))) / 1.2816) ** 2
            pooled = s80(z)
            scale = {}
            for b in range(len(CAL_BINS) + 1):
                nb = int((gbin == b).sum())
                sb = s80(z[gbin == b]) if nb >= 5 else pooled
                scale[b] = (nb * sb + 30 * pooled) / (nb + 30)
        else:
            pooled = (np.quantile(z, 0.8) / 1.2816) ** 2
            scale = {b: (np.quantile(z[gbin == b], 0.8) / 1.2816) ** 2 if (gbin == b).sum() >= 30 else pooled
                     for b in range(len(CAL_BINS) + 1)}
        var = var * np.vectorize(scale.get)(np.digitize(gap_fn(Y), CAL_BINS))
    return (out, var, w) if return_weights else (out, var)


# --------------------------------------------------------------------------------------------
# Baselines from the literature (optional dependencies, imported lazily)

def pchip(Y: np.ndarray) -> np.ndarray:
    """Monotone cubic interpolation per series: the closest standard method to the Stineman
    interpolation that tested best for Indian mandi prices in an earlier farmer-collective study."""
    # Interior gaps only: cubic extrapolation beyond the first/last report explodes, so ends are carried.
    return pd.DataFrame(Y).interpolate(method="pchip", limit_area="inside").ffill().bfill().to_numpy()


def chained_ml(Y: np.ndarray, max_iter: int = 5, seed: int = 0) -> np.ndarray:
    """Chained machine-learning regressions across markets (MICE with tree ensembles, missForest-style):
    the approach behind the World Bank's real-time food price imputation (Andree 2021, WPS 9886)."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    keep = ~np.isnan(Y).all(axis=0)
    out = locf(Y)
    imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, min_samples_leaf=2, random_state=seed,
                                                         n_jobs=1),
                           max_iter=max_iter, random_state=seed, initial_strategy="median")
    out[:, keep] = imp.fit_transform(Y[:, keep])
    return np.where(np.isnan(Y), out, Y)


def dfm_em(Y: np.ndarray, factors: int = 1, online: bool = False, min_obs: int = 10,
           max_series: int = 60) -> np.ndarray:
    """Dynamic factor model by EM with an arbitrary missing pattern and AR(1) idiosyncratic terms
    (Banbura and Modugno 2014, J. Applied Econometrics), via statsmodels DynamicFactorMQ.
    online=True reads the Kalman-filtered state (data up to each week) instead of the smoothed one.

    Series with fewer than `min_obs` reports are left out of the model (filled by carry-forward)
    and at most `max_series` of the best-reported series are used: EM is numerically unstable and
    memory-hungry on very sparse, very wide panels. If EM still fails, it retries with half as many."""
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    counts = (~np.isnan(Y)).sum(axis=0)
    order = [j for j in np.argsort(-counts) if counts[j] >= min_obs]
    out = locf(Y)
    while True:
        cols = sorted(order[:max_series])
        if len(cols) < 3:
            raise ValueError("Too few series for a factor model")
        sub = Y[:, cols]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        idx = pd.date_range("2024-01-01", periods=Y.shape[0], freq="W-MON")
        try:
            mod = DynamicFactorMQ(pd.DataFrame((sub - mu) / sd, index=idx), factors=factors, factor_orders=1,
                                  idiosyncratic_ar1=True, standardize=False)
            res = mod.fit(disp=False, maxiter=200)
            break
        except (np.linalg.LinAlgError, ValueError):
            if max_series <= 6:
                raise
            max_series //= 2
    if online:
        design = np.asarray(res.filter_results.design)[:, :, 0]
        fitted = (design @ np.asarray(res.filter_results.filtered_state)).T * sd + mu
    else:
        fitted = np.asarray(res.smoother_results.smoothed_forecasts).T * sd + mu
    out[:, cols] = fitted
    return np.where(np.isnan(Y), out, Y)


def _pypots(Y: np.ndarray, kind: str, window: int = 24, epochs: int = 50, seed: int = 0) -> np.ndarray:
    """Deep imputers from PyPOTS (as benchmarked in TSI-Bench): BRITS (Cao et al. 2018) and SAITS
    (Du et al. 2023). Trained on this commodity's own matrix, sliding windows, self-supervised."""
    import torch
    from pypots.imputation import BRITS, SAITS

    torch.set_num_threads(1)   # several commodities run in parallel processes
    torch.manual_seed(seed)
    keep = (~np.isnan(Y)).sum(axis=0) >= 3
    mu = np.nanmean(Y[:, keep], axis=0)
    sd = np.nanstd(Y[:, keep], axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Z = (Y[:, keep] - mu) / sd
    T, N = Z.shape
    window = min(window, T)
    starts = list(range(0, T - window + 1, 2)) + [T - window]
    X = np.stack([Z[s:s + window] for s in sorted(set(starts))]).astype("float32")
    common = dict(n_steps=window, n_features=N, epochs=epochs, batch_size=16, device="cpu", verbose=False)
    if kind == "brits":
        model = BRITS(rnn_hidden_size=64, **common)
    else:
        model = SAITS(n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=128, dropout=0.1, **common)
    model.fit({"X": X})
    imp = model.predict({"X": X})["imputation"]
    acc, cnt = np.zeros((T, N)), np.zeros((T, N))
    for i, s in enumerate(sorted(set(starts))):
        acc[s:s + window] += imp[i]
        cnt[s:s + window] += 1
    out = locf(Y)
    out[:, keep] = acc / np.maximum(cnt, 1) * sd + mu
    return np.where(np.isnan(Y), out, Y)


def brits(Y: np.ndarray) -> np.ndarray:
    return _pypots(Y, "brits")


def saits(Y: np.ndarray) -> np.ndarray:
    return _pypots(Y, "saits")


METHODS = {"locf": locf, "linear": linear, "pchip": pchip, "kalman": kalman, "softimpute": softimpute,
           "knn": knn, "chained_ml": chained_ml, "dfm_em": dfm_em, "brits": brits, "saits": saits,
           "bridge": bridge, "bridge_auto": bridge_auto}
TWO_SIDED = ("linear", "pchip", "softimpute", "knn", "chained_ml", "brits", "saits")


# --------------------------------------------------------------------------------------------
# Evaluation: hide real reports in realistic gap patterns

def transplant_masks(obs: np.ndarray, targets, rng: np.random.Generator, min_keep: int = 6) -> Dict[int, np.ndarray]:
    """For each target series, borrow the reporting pattern of another series (circularly shifted)
    and hide the target's reports where the donor had none. Gaps then look like real gaps."""
    T, N = obs.shape
    donors = [j for j in range(N) if 0.15 < obs[:, j].mean() < 0.9]
    masks = {}
    for m in targets:
        if not donors:
            break
        for _ in range(10):
            d = donors[rng.integers(len(donors))]
            pattern = np.roll(obs[:, d], rng.integers(T))
            hide = obs[:, m] & ~pattern
            if hide.any() and (obs[:, m] & pattern).sum() >= min_keep:
                masks[m] = hide
                break
    return masks


def block_masks(obs: np.ndarray, targets, length: int, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    T = obs.shape[0]
    masks = {}
    for m in targets:
        start = rng.integers(4, max(5, T - length - 4))
        hide = np.zeros(T, dtype=bool)
        hide[start:start + length] = True
        hide &= obs[:, m]
        if hide.any():
            masks[m] = hide
    return masks


def evaluate(Y: np.ndarray, masks: Dict[int, np.ndarray], methods=None, online: bool = False,
             bridge_kw: Optional[dict] = None, label: Optional[str] = None) -> pd.DataFrame:
    """Hide the masked cells, fill with each method, score on the hidden truth (log scale).

    `sd` is the method's own standard deviation for the fill (log scale), where it gives one.
    """
    methods = methods or list(METHODS)
    Yh = Y.copy()
    for m, hide in masks.items():
        Yh[hide, m] = np.nan
    gap = _gap_length(Yh)
    rows = []
    for name in methods:
        fn = METHODS[name]
        if name == "bridge":
            filled, var = bridge(Yh, online=online, return_var=True, **(bridge_kw or {}))
        elif name == "bridge_auto":
            filled, var = bridge_auto(Yh, online=online, return_var=True, n_primary=(bridge_kw or {}).get("n_primary"))
        elif name == "kalman":
            filled, var = kalman(Yh, online=online), None
        elif name == "dfm_em":
            filled, var = dfm_em(Yh, online=online), None
        elif online and name in TWO_SIDED:
            continue  # uses later weeks by construction
        else:
            filled, var = fn(Yh), None
        # Sanity rule, the same for every method: a fill that is not finite, or more than 10x away from
        # the nearest reported price of the same market, is replaced by carry-forward and counted.
        # (Real moves of 10x within a gap do not occur once entry errors are cleaned.)
        ref = locf(Yh)
        bad = ~np.isfinite(filled) | (np.abs(np.nan_to_num(filled, nan=1e9, posinf=1e9, neginf=-1e9) - ref) > np.log(10))
        if bad.any():
            filled = np.where(bad, ref, filled)
            if var is not None:
                var = np.where(bad, np.nan, var)
        for m, hide in masks.items():
            for t in np.where(hide)[0]:
                err = filled[t, m] - Y[t, m]
                sd = float(np.sqrt(max(var[t, m], 1e-12))) if var is not None else np.nan
                inside = bool(abs(err) <= 1.2816 * sd) if var is not None else None   # 80% interval
                rows.append((label or name, m, t, gap[t, m], float(np.exp(Y[t, m])), float(np.exp(filled[t, m])),
                             sd, inside, bool(bad[t, m])))
    return pd.DataFrame(rows, columns=["method", "series", "week", "gap", "actual", "filled", "sd", "in80", "fallback"])


def _gap_length(Yh: np.ndarray) -> np.ndarray:
    """Weeks from the nearest reported week of the same series (1 = next to a report)."""
    T, N = Yh.shape
    out = np.zeros((T, N), dtype=int)
    for m in range(N):
        t_obs = np.where(~np.isnan(Yh[:, m]))[0]
        if t_obs.size == 0:
            out[:, m] = T
            continue
        for t in range(T):
            out[t, m] = int(np.min(np.abs(t_obs - t)))
    return out


def _back_gap(Yh: np.ndarray) -> np.ndarray:
    """Weeks since the previous reported week of the same series (T if none): the distance a one-sided
    (online) fill bridges."""
    T, N = Yh.shape
    out = np.full((T, N), T, dtype=int)
    for m in range(N):
        last = None
        for t in range(T):
            if last is not None:
                out[t, m] = t - last
            if not np.isnan(Yh[t, m]):
                last = t
    return out


def summarise(ev: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    d = ev.copy()
    d["ape"] = (d["filled"] - d["actual"]).abs() / d["actual"]
    g = d.groupby(list(by))
    out = pd.DataFrame({"mape": g["ape"].mean(), "median_ape": g["ape"].median(), "n": g.size()})
    cov = d.dropna(subset=["in80"]).groupby(list(by))["in80"].mean()
    out["coverage80"] = cov
    return out
