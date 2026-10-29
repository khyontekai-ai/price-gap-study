"""Download daily market prices and arrivals from the public Agmarknet 2.0 API.

Uses the public `dashboard-data/` endpoint (market_details_sp), which returns every
market x commodity row for one day. No login or captcha. Requests are spaced out and
429 responses are retried with back-off, so the portal is not overloaded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.agmarknet.gov.in/v1/dashboard-data/"
ALL_COMMODITIES = 100001
# Assam plus comparable markets the challenge suggests using: neighbouring and supplying states.
STATES = {"Assam": 4, "West Bengal": 36, "Meghalaya": 22, "Bihar": 5, "Tripura": 33, "Uttar Pradesh": 34,
          # other data-poor target states for the transfer test
          "Manipur": 21, "Nagaland": 24, "Jharkhand": 15, "Goa": 10}
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}


def fetch_day(day: dt.date, state_id: int, pause: float = 1.6, retries: int = 10,
              commodities: list[int] | None = None) -> list[dict]:
    rows, page = [], 1
    while True:
        body = json.dumps({"dashboard": "market_details_sp", "commodity": commodities or [ALL_COMMODITIES],
                           "state": [state_id],
                           "date": day.isoformat(), "limit": 500, "page": page, "format": "json"}).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(API, data=body, headers=HEADERS, method="POST")
                with urllib.request.urlopen(req, timeout=120) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt == retries - 1:
                    raise
                time.sleep(min(300.0, pause * 2 ** (attempt + 1)) if getattr(e, "code", None) == 429 else pause * 3)
        data = payload.get("data")
        records = data.get("records", []) if isinstance(data, dict) else []
        rows.extend(dict(r, date=day.isoformat()) for r in records)
        pages = (payload.get("pagination") or {}).get("total_pages") or 1
        time.sleep(pause)
        if page >= pages:
            return rows
        page += 1


def download(out: Path, start: dt.date, end: dt.date, state: str = "Assam", pause: float = 1.6,
             commodities: list[int] | None = None) -> None:
    """Append one JSON line per day to <out>/<state>_<year>.jsonl; days already saved are skipped."""
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    prefix = state.lower().replace(" ", "_")
    for f in out.glob(f"{prefix}_*.jsonl"):
        done.update(json.loads(line)["date"] for line in f.read_text(encoding="utf-8").splitlines() if line)
    day = start
    while day <= end:
        if day.isoformat() not in done:
            rows = fetch_day(day, STATES[state], pause, commodities=commodities)
            with (out / f"{prefix}_{day.year}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"date": day.isoformat(), "rows": rows}, ensure_ascii=False) + "\n")
            print(day, len(rows), flush=True)
        day += dt.timedelta(days=1)


TREND = "https://api.agmarknet.gov.in/v1/price-trend/wholesale-prices-monthly"


def download_monthly_statewise(out: Path, commodities: dict[str, int], start_year: int = 2021,
                               end: dt.date | None = None, pause: float = 2.0) -> None:
    """State-level monthly prices for every state (one call per commodity-month), from 2021.

    Gives the longer, denser history of comparable markets that Assam's own daily records lack.
    Saved to <out>/statewise_monthly.jsonl; months already saved are skipped.
    """
    end = end or dt.date.today()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "statewise_monthly.jsonl"
    done = set()
    if path.exists():
        done = {(r["commodity"], r["year"], r["month"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}
    for name, cid in commodities.items():
        for year in range(start_year, end.year + 1):
            for month in range(1, 13):
                if (year, month) > (end.year, end.month) or (name, year, month) in done:
                    continue
                url = (f"{TREND}?report_mode=Statewise&commodity={cid}&year={year}&month={month}"
                       f"&state=0&district=0&export=false")
                for attempt in range(6):
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=180) as r:
                            payload = json.loads(r.read().decode("utf-8"))
                        break
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                        if attempt == 5:
                            raise
                        time.sleep(min(300.0, pause * 2 ** (attempt + 1)) if getattr(e, "code", None) == 429 else pause * 3)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"commodity": name, "year": year, "month": month,
                                        "rows": payload.get("rows", [])}, ensure_ascii=False) + "\n")
                print(name, year, month, len(payload.get("rows", [])), flush=True)
                time.sleep(pause)


MONTHLY_COMMODITIES = {"Potato": 24, "Onion": 23, "Rice": 3, "Paddy(Common)": 2, "Mustard": 12, "Tomato": 65,
                       "Ginger(Green)": 87, "Green Chilli": 73, "Jute": 16, "Maize": 4, "Turmeric": 35,
                       "Cabbage": 126, "Brinjal": 32, "Cauliflower": 31, "Capsicum": 136, "Pumpkin": 70,
                       "Lemon": 261, "Banana": 19}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/agmarknet")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    p.add_argument("--state", default="Assam")
    p.add_argument("--pause", type=float, default=1.6)
    p.add_argument("--commodities", default="", help="Comma-separated commodity ids (default: all)")
    p.add_argument("--monthly-statewise", action="store_true", help="Download state-level monthly prices instead")
    a = p.parse_args()
    if a.monthly_statewise:
        download_monthly_statewise(Path(a.out), MONTHLY_COMMODITIES, pause=a.pause)
        raise SystemExit
    ids = [int(x) for x in a.commodities.split(",") if x.strip()] or None
    download(Path(a.out), dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), a.state, a.pause, ids)
