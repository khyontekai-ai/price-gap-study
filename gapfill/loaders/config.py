"""Paths are independent of the working directory and neighbouring checkouts."""
import os
from pathlib import Path

RAW = Path(os.environ.get("PRICE_GAP_DATA_DIR", Path(__file__).resolve().parents[2] / "data/raw/agmarknet")).expanduser().resolve()
