"""Throwaway: print what the corrected CIC-IDS2017 files look like."""
import pandas as pd

from flowadj.config import CICIDS_CORRECTED_DIR

files = sorted(CICIDS_CORRECTED_DIR.rglob("*.csv"))
print(f"{len(files)} CSV files under {CICIDS_CORRECTED_DIR}")
for f in files:
    head = pd.read_csv(f, nrows=3, encoding="latin-1")
    cols = list(head.columns)
    print("\n=====", f.relative_to(CICIDS_CORRECTED_DIR))
    print("n_columns:", len(cols))
    print("first 12 columns:", cols[:12])
    print("last 4 columns:", cols[-4:])
    want = [c for c in cols if c.strip().lower() in ("label", "timestamp", "attempted category")]
    d = pd.read_csv(f, usecols=want, encoding="latin-1")
    print("rows:", len(d))
    ts = [c for c in want if c.strip().lower() == "timestamp"]
    if ts:
        print("timestamp examples:", d[ts[0]].head(2).tolist(), "...", d[ts[0]].tail(1).tolist())
    lab = [c for c in want if c.strip().lower() == "label"]
    if lab:
        print(d[lab[0]].value_counts().to_string())