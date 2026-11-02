"""Run S1 on ONE dev case with the real local model and print everything.

    uv run python -m scripts.s1_one_case --idx 0
    uv run python -m scripts.s1_one_case --case case_444d274d
"""
import argparse
import json

from flowadj.agents.single import run_s1
from flowadj.cases.jsonl import CASES_DIR, read_jsonl
from flowadj.config import ROOT
from flowadj.eval.grounding import check_report
from flowadj.eval.score import score_case
from flowadj.llm.budget import Budget
from flowadj.trace import Tracer

ap = argparse.ArgumentParser()
ap.add_argument("--idx", type=int, default=0)
ap.add_argument("--case")
a = ap.parse_args()

cases = read_jsonl(CASES_DIR / "cases_dev.jsonl")
case = next(c for c in cases if c["case_id"] == a.case) if a.case else cases[a.idx]
truth = {t["case_id"]: t for t in read_jsonl(CASES_DIR / "ground_truth_corrected.jsonl")}[case["case_id"]]

budget = Budget.from_config()
tracer = Tracer(ROOT / "results" / "traces" / f"s1_{case['case_id']}_manual.jsonl", system="s1", case_id=case["case_id"])
print("Running S1 on", case["case_id"], "...")
res = run_s1(case, budget=budget, tracer=tracer)
tracer.close()

print("\nstatus:", res.status, "| error:", res.error, "| steps:", res.steps, "| forced_final:", res.forced_final,
      "| fallback tool calls:", res.fallback_tool_calls)
print("budget:", json.dumps(budget.to_dict()))
print("tools called:", [(t.call_id, t.name, t.ok) for t in res.tool_log])
if res.report:
    print("\nREPORT:\n" + json.dumps(res.report.model_dump(), indent=2))
    g = check_report(res.report, res.tool_log)
    print(f"\nGROUNDING: {g['n_supported']}/{g['n_claims']} claims supported")
    for d in g["details"]:
        if not d["supported"]:
            print("  UNSUPPORTED:", d["reason"], d["claim"])
else:
    print("\nRAW TEXT:\n", res.raw_text[:1500])

print("\nTRUTH:", json.dumps({k: truth[k] for k in ("verdict", "family", "attackers", "victims", "ports", "active_start", "active_end")}))
print("SCORE:", score_case(res.report, truth))
print("\nTrace saved to", tracer.path)