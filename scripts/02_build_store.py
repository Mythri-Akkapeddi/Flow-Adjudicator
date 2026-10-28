"""Merge corrected + original labels, write flows_corrected.parquet, build flows.duckdb."""
import json

import pandas as pd

from flowadj.config import PROCESSED_DIR, get_default
from flowadj.data.labels import attach_labels
from flowadj.data.store import DB_PATH, agent_columns, agent_query, build_store, eval_columns, eval_query


def main():
    print("Reading parquet files ...")
    orig = pd.read_parquet(
        PROCESSED_DIR / "flows_original.parquet",
        columns=["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "timestamp", "label"],
    )
    corr = pd.read_parquet(PROCESSED_DIR / "flows_corrected_raw.parquet")
    offset_h = get_default()["corrected_clock_offset_hours"]
    corr["timestamp"] = corr["timestamp"] - pd.Timedelta(hours=offset_h)
    print(f"  shifted corrected timestamps by -{offset_h}h to the capture clock")
    print(f"  original rows: {len(orig):,} | corrected rows: {len(corr):,}")

    merged, report = attach_labels(corr, orig)
    print("\nMerge report:", json.dumps(report, indent=2))
    out = PROCESSED_DIR / "flows_corrected.parquet"
    merged.to_parquet(out, compression="zstd", index=False)
    with open(PROCESSED_DIR / "label_merge_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")

    info = build_store(out, DB_PATH)
    print(f"\nBuilt {DB_PATH} with {info['rows']:,} rows")
    print("flows_agent columns:", len(agent_columns()))
    print("flows_eval columns :", eval_columns())

    bad = [c for c in agent_columns() if any(w in c.lower() for w in ("label", "family", "attempt", "source_file"))]
    print("Label-bearing columns visible to agent:", bad if bad else "NONE (good)")

    print("\nCorrected family counts (flows_eval):")
    print(eval_query("SELECT family_corrected, COUNT(*) AS n FROM flows_eval GROUP BY 1 ORDER BY n DESC").to_string(index=False))
    print("\nSample from flows_agent:")
    print(agent_query("SELECT * FROM flows_agent LIMIT 3").iloc[:, :10].to_string(index=False))


if __name__ == "__main__":
    main()