"""Run S0 on a frozen split and print scores.

    uv run python -m scripts.s0_run --split dev
    uv run python -m scripts.s0_run --split test
"""
import argparse

import pandas as pd

from flowadj.baselines.s0_detector_template import s0_reports
from flowadj.cases.jsonl import CASES_DIR, read_jsonl
from flowadj.config import ROOT
from flowadj.eval.score import aggregate, per_family, score_case

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], default="dev")
ap.add_argument("--gt", choices=["corrected", "original"], default="corrected")
a = ap.parse_args()
pd.set_option("display.width", 200)

ids = [c["case_id"] for c in read_jsonl(CASES_DIR / f"cases_{a.split}.jsonl")]
truth = {t["case_id"]: t for t in read_jsonl(CASES_DIR / f"ground_truth_{a.gt}.jsonl")}
reports = s0_reports(ids)
print(f"{len(ids)} cases, {len(reports)} S0 reports")

rows = pd.DataFrame([score_case(reports.get(cid), truth[cid]) for cid in ids])
print(f"\nS0 on {a.split} (truth: {a.gt})")
print(pd.Series(aggregate(rows)).to_string())
print("\nPer true family (what the detector alone already gets right):")
print(per_family(rows).to_string())
print("\nWrong cases:")
print(rows[~(rows.verdict_correct & rows.family_correct)][["case_id", "true_family", "pred_family"]].to_string(index=False))

out = ROOT / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
rows.to_csv(out / f"s0_{a.split}_{a.gt}.csv", index=False)
print(f"\nWrote results/tables/s0_{a.split}_{a.gt}.csv")