import numpy as np
import pandas as pd
import pytest

from gapfill import calendar as cal
from gapfill.inference import _pairs
from gapfill.revision_controls import matched_predictions


def test_common_calendar_block_ignores_panel_relative_week():
    records = []
    for commodity, week in (("Tomato", 73), ("Carrot", 0)):
        for method, filled in (("bridge_auto", 101), ("locf", 110)):
            records.append(dict(state="Assam", commodity=commodity, week=week,
                                calendar_week=pd.Timestamp("2025-12-15"), protocol="transplant",
                                run=0, series=0, method=method, actual=100, filled=filled))
    paired = _pairs(pd.DataFrame(records), "bridge_auto", "locf")
    assert paired.block.nunique() == 1


def test_calendar_reconstruction_rejects_different_saved_truth(monkeypatch):
    monkeypatch.setattr(cal, "_state_inputs", lambda state: (None, None))
    monkeypatch.setattr(cal, "panel", lambda *a, **kw: (
        np.log(np.array([[100.], [101.]])), ["Assam:A"], pd.date_range("2025-01-06", periods=2, freq="W-MON"), 1))
    ev = pd.DataFrame([dict(state="Assam", commodity="Potato", week=1, series=0, actual=999)])
    with pytest.raises(ValueError, match="Saved truth differs"):
        cal.attach_calendar(ev)
    ev["actual"] = 101
    assert cal.attach_calendar(ev).calendar_week.iloc[0] == pd.Timestamp("2025-01-13")


def test_calendar_blocks_reject_non_monday_or_missing_dates():
    for dates in ([pd.Timestamp("2025-01-07")], [pd.NaT]):
        with pytest.raises(ValueError):
            cal.calendar_blocks(pd.DatetimeIndex(dates))


def test_matched_controls_reject_changed_truth_and_unmatched_cells():
    b = pd.DataFrame([dict(method="bridge", commodity="x", protocol="transplant", run=0,
                           series=0, week=2, actual=100, filled=101)])
    c = b.assign(filled=102)
    j = matched_predictions(b, c, "bridge")
    assert j.filled_c.iloc[0] == 102 and j.filled_b.iloc[0] == 101
    with pytest.raises(ValueError, match="truth differs"):
        matched_predictions(b, c.assign(actual=99), "bridge")
    with pytest.raises(ValueError, match="every C cell"):
        matched_predictions(b, c.assign(week=3), "bridge")
