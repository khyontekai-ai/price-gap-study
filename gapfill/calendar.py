"""Calendar dates for saved, panel-relative evaluation rows.

Reconstruct the original weekly panel and verify its prices against saved scoring truth before
using its index. This fails closed if local data or panel construction no longer reproduce a run.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from .controls import _setup
from .study import panel

ANCHOR = pd.Timestamp("1970-01-05")  # common Monday for all states and commodities


def calendar_blocks(dates, block_weeks=13):
    dates = pd.to_datetime(dates)
    if dates.isna().any() or (dates.dayofweek != 0).any():
        raise ValueError("calendar weeks must be non-missing Mondays")
    return np.asarray((dates - ANCHOR).days // (7 * block_weeks), dtype=int)


@lru_cache(maxsize=8)
def _state_inputs(state):
    days, monthly, _, _ = _setup(state)
    return days, monthly


def attach_calendar(ev):
    out = ev.copy().reset_index(drop=True)
    out["calendar_week"] = pd.NaT
    for (state, commodity), positions in out.groupby(["state", "commodity"]).groups.items():
        days, monthly = _state_inputs(state)
        built = panel(days, monthly, commodity, target=state)
        if built is None:
            raise ValueError(f"Cannot reconstruct {state}/{commodity}")
        Y, _, dates, n = built
        group = out.loc[positions]
        week = group["week"].to_numpy(dtype=int)
        series = group["series"].to_numpy(dtype=int)
        if ((week < 0) | (week >= len(dates)) | (series < 0) | (series >= n)).any():
            raise ValueError(f"Invalid saved indices for {state}/{commodity}")
        if not np.allclose(np.exp(Y[week, series]), group["actual"], rtol=1e-9, atol=1e-8):
            raise ValueError(f"Saved truth differs from reconstructed panel for {state}/{commodity}")
        out.loc[positions, "calendar_week"] = dates[week].to_numpy()
    return out
