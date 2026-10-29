"""Build candidate cases + ground truth (Day 5); sample and freeze dev/test (Day 6).

    uv run python -m scripts.04_build_cases --stage pool
    uv run python -m scripts.04_build_cases --stage freeze
"""
import argparse
import json

import pandas as pd

from flowadj.cases.build_cases import ALERTS_PATH, build_clusters, load_alerts
from flowadj.cases.ground_truth import build_truth
from flowadj.cases.jsonl import CASES_DIR, write_jsonl
from flowadj.config import get_default


def stage_pool():
    pd.set_option("display.width", 200)
    cfg = get_default()
    cc, sp = cfg["cases"], cfg["splits"]
    print("Loading alerts ...")
    alerts = load_alerts()
    print(f"  {len(alerts):,} alert flows in dev/test blocks")
    pool, amap = build_clusters(alerts, cc, sp["block_minutes"], sp["buffer_minutes"])
    print(f"  {len(pool)} candidate cases (>= {cc['min_alerts']} alerts each)")
    amap.to_parquet(ALERTS_PATH, index=False)
    write_jsonl(CASES_DIR / "cases_pool.jsonl", pool)

    tc = build_truth(pool, amap, "family_corrected", "corrected", cc)
    to = build_truth(pool, amap, "family_original", "original", cc)
    write_jsonl(CASES_DIR / "ground_truth_corrected.jsonl", tc)
    write_jsonl(CASES_DIR / "ground_truth_original.jsonl", to)

    p = pd.DataFrame([{"case_id": r["input"]["case_id"], "split": r["split"], "block_id": r["block_id"]} for r in pool])
    c = pd.DataFrame(tc).merge(p, on="case_id")
    o = pd.DataFrame(to)[["case_id", "verdict", "family"]].rename(columns={"verdict": "verdict_orig", "family": "family_orig"})
    c = c.merge(o, on="case_id")

    print("\nCandidate cases: family x split (corrected truth)")
    print(pd.crosstab(c["family"], c["split"]).to_string())
    print("\nVerdict x split:")
    print(pd.crosstab(c["verdict"], c["split"]).to_string())
    print("\nFlag counts:")
    print(c["flags"].explode().value_counts(dropna=True).to_string())
    print(f"\nLow purity (<{cc['min_purity']}): {int((c['purity'] < cc['min_purity']).sum())}")
    print("Corrected vs original disagree on verdict:", int((c["verdict"] != c["verdict_orig"]).sum()),
          "| on family:", int((c["family"] != c["family_orig"]).sum()))
    print("Distinct blocks with cases:", c["block_id"].nunique())


def stage_freeze():
    from flowadj.cases.sampling import sample_and_freeze
    print(json.dumps(sample_and_freeze(), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pool", "freeze"], required=True)
    stage = ap.parse_args().stage
    stage_pool() if stage == "pool" else stage_freeze()