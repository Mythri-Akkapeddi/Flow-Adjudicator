"""Convert CIC-IDS2017 CSVs into one canonical parquet file.

Usage:
    uv run python -m scripts.01_convert_to_parquet --which original
    uv run python -m scripts.01_convert_to_parquet --which corrected
"""
import argparse
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flowadj.config import CICIDS_CORRECTED_DIR, CICIDS_ORIGINAL_DIR, PROCESSED_DIR
from flowadj.data.load_cicids import load_day

SETTINGS = {
    "original": (CICIDS_ORIGINAL_DIR, "flows_original.parquet"),
    "corrected": (CICIDS_CORRECTED_DIR, "flows_corrected_raw.parquet"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=SETTINGS, required=True)
    args = ap.parse_args()
    src_dir, out_name = SETTINGS[args.which]

    files = sorted(src_dir.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs found under {src_dir}")
    print(f"Found {len(files)} CSV files:")
    for f in files:
        print("  ", f.relative_to(src_dir))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / out_name
    writer, schema, cols = None, None, None
    all_stats = []

    for f in files:
        print(f"\nProcessing {f.name} ...")
        df, stats = load_day(f, ts_mode=args.which)
        if cols is None:
            cols = list(df.columns)
        elif list(df.columns) != cols:
            print("  WARNING: column mismatch vs first file; aligning to first file")
            df = df.reindex(columns=cols)
        if writer is None:
            table = pa.Table.from_pandas(df, preserve_index=False)
            schema = table.schema
            writer = pq.ParquetWriter(out, schema, compression="zstd")
        else:
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        writer.write_table(table)
        all_stats.append(stats)
        print("  ", stats)
        del df, table

    writer.close()

    summary = pd.DataFrame(all_stats)
    print("\n" + summary.to_string(index=False))
    totals = summary.drop(columns="file").sum()
    print("\nTOTALS:\n", totals.to_string())
    with open(PROCESSED_DIR / f"conversion_report_{args.which}.json", "w") as fh:
        json.dump({"files": all_stats, "totals": totals.to_dict()}, fh, indent=2, default=int)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()