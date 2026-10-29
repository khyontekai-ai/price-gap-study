"""Longer-range price intelligence for crop planning, from state-level monthly prices.

Question a farmer asks at sowing: what will this crop sell for at harvest, 3 to 6 months away?
Assam's own monthly history has gaps, so we also use comparable states (West Bengal, Meghalaya,
Bihar, Tripura, Uttar Pradesh), as the challenge suggests.

Methods compared, all out of sample (origins roll month by month):
  naive              today's price
  same_month_ly      the price in the target month last year
  seasonal_assam     today's price x Assam's own month-of-year profile
  seasonal_pooled    today's price x a month-of-year profile pooled across comparable states
                     (Assam's profile shrunk toward the pool by how many years Assam has)
  chronos_2          pretrained foundation model, zero-shot, on Assam's monthly series
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from .data import RAW

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                         "september", "october", "november", "december"], start=1)}
COMPARABLE = ("Assam", "West Bengal", "Meghalaya", "Bihar", "Tripura", "Uttar Pradesh")
_COL = re.compile(r"prices_([a-z]+)_(\d{4})$")


def load_statewise(path: Path = RAW / "statewise_monthly.jsonl") -> pd.DataFrame:
    """Long table commodity, state, month (Period), price. Each report also gives the previous month
    and the same month last year, which fill gaps; a month's own report wins."""
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        own = pd.Period(year=r["year"], month=r["month"], freq="M")
        for row in r["rows"]:
            for col, val in row.items():
                m = _COL.match(col)
                if m and val is not None and m.group(1) in MONTHS:
                    p = pd.Period(year=int(m.group(2)), month=MONTHS[m.group(1)], freq="M")
                    recs.append((r["commodity"], row.get("state"), p, float(val), p == own))
    df = pd.DataFrame(recs, columns=["commodity", "state", "month", "price", "own"])
    df = df[df["price"] > 0].sort_values("own", ascending=False)
    return df.drop_duplicates(["commodity", "state", "month"]).drop(columns="own").reset_index(drop=True)


def panel(df: pd.DataFrame, commodity: str, states: Sequence[str] = COMPARABLE) -> pd.DataFrame:
    d = df[(df["commodity"] == commodity) & df["state"].isin(states)]
    wide = d.pivot_table(index="month", columns="state", values="price")
    if wide.empty:
        return wide
    full = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    return wide.reindex(full)


def month_profile(s: pd.Series) -> tuple[np.ndarray, float]:
    """Month-of-year index (12 values, mean 1) from a monthly series, and years of evidence."""
    s = s.dropna()
    if len(s) < 6:
        return np.ones(12), 0.0
    level = s.rolling(12, center=True, min_periods=6).mean()
    ratio = (s / level).dropna()
    idx = np.ones(12)
    for m in range(1, 13):
        v = ratio[[p.month == m for p in ratio.index]]
        if len(v):
            idx[m - 1] = v.mean()
    return idx / idx.mean(), len(s) / 12


def pooled_profile(wide: pd.DataFrame, target: str = "Assam", prior_years: float = 2.0) -> np.ndarray:
    profiles, weights = [], []
    own, years = month_profile(wide[target]) if target in wide else (np.ones(12), 0.0)
    for st in wide.columns:
        if st == target:
            continue
        p, y = month_profile(wide[st])
        if y > 0:
            profiles.append(p)
            weights.append(y)
    pool = np.average(profiles, axis=0, weights=weights) if profiles else np.ones(12)
    shrunk = (years * own + prior_years * pool) / (years + prior_years)
    return shrunk / shrunk.mean()


def backtest(df: pd.DataFrame, commodities: Iterable[str], horizons=(3, 6), min_train: int = 18,
             target: str = "Assam", use_chronos: bool = True) -> pd.DataFrame:
    rows = []
    for c in commodities:
        wide = panel(df, c)
        if target not in wide or wide[target].notna().sum() < min_train:
            continue
        y = wide[target]
        for o in range(min_train, len(wide)):
            hist = wide.iloc[:o]
            ya = hist[target]
            if ya.notna().sum() < 6 or np.isnan(ya.iloc[-1]):
                continue  # need a price today to make a forecast today
            now, now_m = ya.iloc[-1], hist.index[-1]
            own, _ = month_profile(ya)
            pooled = pooled_profile(hist, target)
            fc = {}
            for h in horizons:
                if o - 1 + h >= len(wide) or np.isnan(y.iloc[o - 1 + h]):
                    continue
                tm = (now_m + h).month
                ly = ya.get(now_m + h - 12, np.nan)
                fc[h] = {"naive": now, "same_month_ly": ly if not np.isnan(ly) else now,
                         "seasonal_assam": now * own[tm - 1] / own[now_m.month - 1],
                         "seasonal_pooled": now * pooled[tm - 1] / pooled[now_m.month - 1]}
            if not fc:
                continue
            if use_chronos:
                ch = _chronos_monthly(ya.ffill().dropna().to_numpy(), max(horizons))
                for h in fc:
                    fc[h]["chronos_2"] = float(ch[h - 1])
            for h, f in fc.items():
                for model, v in f.items():
                    rows.append((c, o, h, str(now_m), model, now, float(y.iloc[o - 1 + h]), float(v)))
    return pd.DataFrame(rows, columns=["commodity", "origin", "horizon", "origin_month", "model", "now", "actual",
                                       "forecast"])


def _chronos_monthly(y: np.ndarray, h: int) -> np.ndarray:
    from .engine import ENGINE  # noqa: F401  (puts grdcast on the path)
    from grdcast.contenders import forecast_contender

    return forecast_contender("chronos_2", {("x", "x"): y}, h)[("x", "x")]


def score(bt: pd.DataFrame, by=("horizon", "model")) -> pd.DataFrame:
    d = bt.copy()
    d["ape"] = (d["forecast"] - d["actual"]).abs() / d["actual"]
    real, pred = np.sign(d["actual"] - d["now"]), np.sign(d["forecast"] - d["now"])
    d["dir_ok"] = (real == pred) & (real != 0)
    g = d.groupby(list(by))
    return pd.DataFrame({"mape": g["ape"].mean(), "direction": g["dir_ok"].mean(), "n": g.size()})


def coverage(df: pd.DataFrame, states: Sequence[str] = COMPARABLE) -> pd.DataFrame:
    """Months with a price, per commodity and state: how incomplete Assam is next to its neighbours."""
    d = df[df["state"].isin(states)]
    return d.pivot_table(index="commodity", columns="state", values="price", aggfunc="count").fillna(0).astype(int)
