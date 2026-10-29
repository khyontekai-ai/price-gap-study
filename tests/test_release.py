import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from scripts.release_safety import ROOT, checked_path, scan, verify_manifest
from scripts.reproduce import load_evaluation, with_calendar


def test_release_manifest():
    assert verify_manifest(ROOT, allow_runtime=True)["version"] == "0.1.0"


@pytest.mark.parametrize("name", ["../secrets", "/absolute", "C:/secret", "a\\b"])
def test_reject_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError):
        checked_path(tmp_path, name)


def test_manifest_rejects_edits_and_extras(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("safe", encoding="utf-8")
    manifest = {"kind": "code", "files": {"file.txt": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verify_manifest(tmp_path)
    (tmp_path / "private.txt").write_text("not approved", encoding="utf-8")
    with pytest.raises(ValueError, match="Unmanifested"):
        verify_manifest(tmp_path)
    (tmp_path / "private.txt").unlink()
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(tmp_path)


def test_secret_scan(tmp_path):
    path = tmp_path / "unsafe.txt"
    path.write_text("ghp_" + "z" * 36, encoding="utf-8")
    with pytest.raises(ValueError, match="credential"):
        scan(path)


def test_no_sibling_checkout_needed():
    from gapfill.sources import data, monthly
    assert "gapfill" in Path(data.__file__).parts
    assert "gapfill" in Path(monthly.__file__).parts


def test_calendar_mapping_rejects_wrong_truth():
    ev = pd.DataFrame([dict(state="Assam", commodity="x", series=0, week=0, actual=100)])
    lookup = ev.assign(calendar_week="2025-01-06")
    assert with_calendar(ev, lookup).calendar_week.iloc[0] == pd.Timestamp("2025-01-06")
    with pytest.raises(ValueError, match="truth differs"):
        with_calendar(ev, lookup.assign(actual=999))
    with pytest.raises(ValueError, match="Missing calendar"):
        with_calendar(ev, lookup.assign(week=9))


def test_evaluation_duplicate_and_nonpositive_truth(tmp_path):
    row = dict(method="bridge", commodity="x", protocol="block", run=0, series=0, week=1, actual=100, filled=101)
    path = tmp_path / "evaluation.csv"
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_evaluation(path)
    pd.DataFrame([dict(row, actual=0)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="positive"):
        load_evaluation(path)


@pytest.mark.parametrize("module,args", [
    ("scripts.upload_github", ["--repo", "example-owner/test-research"]),
    ("scripts.upload_github", ["--repo", "example-owner/test-research", "--public", "--confirm-publication"]),
])
def test_upload_dry_run_does_not_create_git(module, args, tmp_path):
    fresh = tmp_path / "release"
    shutil.copytree(ROOT, fresh, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "_reproduced"))
    result = subprocess.run([sys.executable, "-m", module, *args], cwd=fresh, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Dry run only" in result.stdout
    assert not (fresh / ".git").exists()


def test_upload_requires_confirmation():
    before = (ROOT / ".git").exists()
    result = subprocess.run([sys.executable, "-m", "scripts.upload_github", "--repo", "example-owner/test", "--public", "--execute"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "confirm-publication" in result.stderr
    assert (ROOT / ".git").exists() == before


def test_github_execution_uses_only_allowlist(monkeypatch, tmp_path):
    from scripts import upload_github as upload
    monkeypatch.setattr(upload, "ROOT", tmp_path)
    monkeypatch.setattr(upload, "verify_manifest", lambda *a, **k: {"files": {"README.md": "hash"}})
    monkeypatch.setattr(upload.shutil, "which", lambda name: name)
    monkeypatch.setattr(upload.subprocess, "run", lambda *a, **k: type("Result", (), {"returncode": 1})())
    calls = []
    monkeypatch.setattr(upload, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["upload_github", "--repo", "example-owner/research", "--execute"])
    upload.main()
    added = [call[-1] for call in calls if "add" in call]
    assert added == ["README.md", "release-manifest.json"]
    creation = next(call for call in calls if call[:3] == ("gh", "repo", "create"))
    assert "--private" in creation and "--public" not in creation
    assert not any("--all" in call for call in calls)


def test_hf_requires_rights_even_for_private(monkeypatch, tmp_path):
    from scripts import upload_hf as upload
    monkeypatch.setattr(upload, "verify_manifest", lambda *a, **k: {"files": {"README.md": "hash"}})
    monkeypatch.setattr(sys, "argv", ["upload_hf", "--repo", "example-owner/research", "--artifacts", str(tmp_path), "--execute"])
    with pytest.raises(SystemExit) as error:
        upload.main()
    assert error.value.code == 2


def test_hf_execution_uses_exact_files_and_private_default(monkeypatch, tmp_path):
    from scripts import upload_hf as upload
    from types import SimpleNamespace
    calls = []
    api = SimpleNamespace(whoami=lambda: calls.append("auth"),
                          create_repo=lambda *a, **k: calls.append(("create", a, k)),
                          upload_folder=lambda **k: calls.append(("upload", k)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    monkeypatch.setattr(upload, "verify_manifest", lambda *a, **k: {"files": {"README.md": "hash", "outputs/test.csv.gz": "hash"}})
    monkeypatch.setattr(sys, "argv", ["upload_hf", "--repo", "example-owner/research", "--artifacts", str(tmp_path), "--confirm-data-rights", "--execute"])
    upload.main()
    assert calls[0] == "auth"
    assert calls[1][2] == dict(repo_type="dataset", private=True, exist_ok=False)
    assert set(calls[2][1]["allow_patterns"]) == {"README.md", "outputs/test.csv.gz", "artifact-manifest.json"}


def test_daily_loader_filters_and_deduplicates(tmp_path):
    from gapfill.sources import data
    row = dict(date="2025-01-06", state_name="Assam", district_name="Test", mkt_name="A", mkt_id=1,
               cmdt_name="Potato", cmdt_grp_name="Vegetables", modal_price_wt=100, arrival_qty=2,
               unit_name_arrival="Tonnes", unit_name_price="Rs/Quintal", status="Reported")
    rows = [row, row, dict(row, mkt_id=2, status="Not Reported"), dict(row, mkt_id=3, modal_price_wt=0)]
    (tmp_path / "assam_2025.jsonl").write_text(json.dumps({"date": "2025-01-06", "rows": rows}) + "\n", encoding="utf-8")
    result = data.load_days(tmp_path, states=["Assam"])
    assert len(result) == 1 and result.price.iloc[0] == 100
    assert result.price_unit.iloc[0] == "Rs/Quintal"
    assert data.weekly_prices(result).iloc[0, 0] == 100


def test_monthly_loader_own_report_precedes_carryover(tmp_path):
    from gapfill.sources import monthly
    envelopes = [dict(commodity="Potato", year=2025, month=1, rows=[dict(state="Assam", prices_january_2025=10)]),
                 dict(commodity="Potato", year=2025, month=2, rows=[dict(state="Assam", prices_january_2025=999, prices_february_2025=20)])]
    path = tmp_path / "monthly.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in envelopes) + "\n", encoding="utf-8")
    result = monthly.load_statewise(path).set_index("month")
    assert result.loc[pd.Period("2025-01"), "price"] == 10
    assert result.loc[pd.Period("2025-02"), "price"] == 20
