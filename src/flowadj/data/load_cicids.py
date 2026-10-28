"""Load and clean the original CIC-IDS2017 per-day CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd

from flowadj.data.schema import (
    INT_COLS, NON_NUMERIC_COLS, REQUIRED_COLS, to_snake,
)


def clean_label(s: str) -> str:
    """Replace the broken dash character in labels like 'Web Attack - Brute Force'."""
    import re
    return re.sub(r"[^\x20-\x7e]+", "-", str(s)).strip()


def fix_timestamps(raw: pd.Series) -> pd.Series:
    """Parse d/m/Y H:M strings. The original files have no AM/PM marker.

    Heuristic: capture ran roughly 08:00-17:00, so hours 1-7 are really PM (+12h).
    PROVISIONAL: verify against the attack schedule on Day 3.
    """
    raw = raw.astype("string").str.strip()
    ts = pd.to_datetime(raw, format="%d/%m/%Y %H:%M", errors="coerce")
    bad = ts.isna() & raw.notna()
    if bad.any():  # some rows may include seconds or other variants
        ts[bad] = pd.to_datetime(raw[bad], dayfirst=True, format="mixed", errors="coerce")
    return ts.where(ts.dt.hour >= 8, ts + pd.Timedelta(hours=12))


def load_day(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one CSV, return (clean dataframe, stats dict)."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [to_snake(c) for c in df.columns]
    stats = {"file": path.name, "rows_raw": len(df)}

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")

    # strings
    for c in ["flow_id", "src_ip", "dst_ip", "label"]:
        df[c] = df[c].astype("string").str.strip()
    df["label"] = df["label"].map(lambda x: clean_label(x) if pd.notna(x) else x)

    # timestamps
    df["timestamp_raw"] = df["timestamp"].astype("string")
    df["timestamp"] = fix_timestamps(df["timestamp"])

    # numerics (everything else): coerce, replace +-inf with NaN
    num_cols = [c for c in df.columns if c not in NON_NUMERIC_COLS and c != "timestamp_raw"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    inf_mask = np.isinf(df[num_cols].to_numpy(dtype="float64", na_value=np.nan))
    stats["inf_values_replaced"] = int(inf_mask.sum())
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # drop rows missing identity/time (usually fully blank rows)
    key_cols = ["flow_id", "src_ip", "dst_ip", "timestamp", "label"] + INT_COLS
    keep = df[key_cols].notna().all(axis=1)
    stats["dropped_missing_key"] = int((~keep).sum())
    df = df[keep].copy()

    # integer types
    for c in INT_COLS:
        df[c] = df[c].astype("int32")

    # exact duplicate rows
    dup = df.duplicated()
    stats["dropped_duplicates"] = int(dup.sum())
    df = df[~dup].copy()

    # strings back to plain object for parquet friendliness
    for c in ["flow_id", "src_ip", "dst_ip", "label", "timestamp_raw"]:
        df[c] = df[c].astype(str)

    df["source_file"] = path.name
    df["capture_day"] = path.name.split("-")[0]
    df = df.sort_values("timestamp").reset_index(drop=True)

    stats["rows_kept"] = len(df)
    assert stats["rows_raw"] == stats["rows_kept"] + stats["dropped_missing_key"] + stats["dropped_duplicates"]
    return df, stats