"""Load downloaded Agmarknet days into tidy tables and weekly price series.

Daily rows: date, state, district, market, commodity, price (Rs/quintal, arrival-weighted modal),
arrival (tonnes or the commodity's own unit), reported (bool).
Weekly series: one per market x commodity, Monday-start weeks; the price is the arrival-weighted
mean of the reported days. Weeks with no report stay missing (NaN): the gaps are real and the
forecasting layer decides how to fill them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW


def load_days(folder: Path = RAW, states=None) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                rows.extend(json.loads(line)["rows"])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "state": df["state_name"], "district": df["district_name"], "market": df["mkt_name"],
        "market_id": df["mkt_id"], "commodity": df["cmdt_name"], "group": df["cmdt_grp_name"],
        "price": pd.to_numeric(df["modal_price_wt"], errors="coerce"),
        "arrival": pd.to_numeric(df["arrival_qty"], errors="coerce"),
        "arrival_unit": df["unit_name_arrival"], "price_unit": df["unit_name_price"],
        "reported": df["status"].astype(str).str.startswith("Reported"),
    })
    if states:
        df = df[df["state"].isin(states)]
    df = df[df["reported"] & (df["price"] > 0)]
    return df.drop_duplicates(["date", "market_id", "commodity"]).reset_index(drop=True)


def clean_outliers(days: pd.DataFrame, ratio: float = 3.0, window_days: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop reports that look like entry errors (wrong unit, misplaced digit).

    A report is dropped when it is more than `ratio` times above or below BOTH
      - the median of the same market and commodity within +/- `window_days`, and
      - the median of the same commodity across the state's markets in the same week,
    or more than `far` times off the commodity's long-run median across the state's markets
    (persistent unit errors, such as a market quoting per piece for weeks, look normal next to their
    own days, and are sometimes the only report that week).
    Real price spikes move many markets together and survive; a lone typo does not.
    Returns (clean days, dropped reports).
    """
    far = 8.0
    d = days.sort_values("date").copy()
    lr = np.log(d["price"])
    own = np.empty(len(d))
    for _, idx in d.groupby(["state", "market", "commodity"]).groups.items():
        g = d.loc[idx]
        s = pd.Series(np.log(g["price"].to_numpy()), index=g["date"].to_numpy())
        own[d.index.get_indexer(idx)] = s.rolling(f"{2 * window_days + 1}D", center=True, min_periods=1).median().to_numpy()
    week = d["date"].dt.to_period("W-SUN")
    state_med = lr.groupby([d["state"], d["commodity"], week]).transform("median")
    k = np.log(ratio)
    off_state = np.abs(lr - state_med).to_numpy()
    off_level = np.abs(lr - lr.groupby([d["state"], d["commodity"]]).transform("median")).to_numpy()
    bad = ((np.abs(lr.to_numpy() - own) > k) & (off_state > k)) | (off_level > np.log(far))
    return d[~bad].reset_index(drop=True), d[bad].reset_index(drop=True)


def weekly_prices(days: pd.DataFrame, level: str = "market") -> pd.DataFrame:
    """Wide weekly price table: index = week start (Monday), columns = (commodity, market or state).

    level="state" pools all markets of a state (arrival-weighted), the "comparable market" view.
    """
    d = days.copy()
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time
    d["w"] = d["arrival"].fillna(0).clip(lower=0) + 1e-6
    key = "market" if level == "market" else "state"
    g = d.groupby(["commodity", key, "week"])
    wk = (g.apply(lambda x: np.average(x["price"], weights=x["w"]), include_groups=False)
          .rename("price").reset_index())
    wide = wk.pivot_table(index="week", columns=["commodity", key], values="price")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="W-MON")
    return wide.reindex(full)


def weekly_arrivals(days: pd.DataFrame, level: str = "market") -> pd.DataFrame:
    d = days.copy()
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time
    key = "market" if level == "market" else "state"
    wide = d.pivot_table(index="week", columns=["commodity", key], values="arrival", aggfunc="sum")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="W-MON")
    return wide.reindex(full)


def coverage_table(days: pd.DataFrame) -> pd.DataFrame:
    """Per commodity x market: first and last report, reported days, share of weeks with a price."""
    d = days.copy()
    d["week"] = d["date"].dt.to_period("W-SUN")
    g = d.groupby(["commodity", "market"])
    out = g.agg(first=("date", "min"), last=("date", "max"), days=("date", "nunique"),
                weeks=("week", "nunique"), median_price=("price", "median"), arrival=("arrival", "sum"))
    span = ((out["last"] - out["first"]).dt.days // 7 + 1).clip(lower=1)
    out["week_coverage"] = out["weeks"] / span
    return out.sort_values("weeks", ascending=False).reset_index()
