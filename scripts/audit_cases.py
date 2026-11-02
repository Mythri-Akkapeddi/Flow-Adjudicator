"""Print an audit sheet for sampled cases.

    uv run python -m scripts.audit_cases --n 20
    uv run python -m scripts.audit_cases --flagged
    uv run python -m scripts.audit_cases --case case_ab12cd34
"""
import argparse
import json
import random

import pandas as pd

from flowadj.cases.jsonl import CASES_DIR, read_jsonl
from flowadj.data.store import eval_query
from flowadj.tools import ToolContext, ToolRunner


def load():
    cases = {}
    for split in ("dev", "test"):
        for c in read_jsonl(CASES_DIR / f"cases_{split}.jsonl"):
            cases[c["case_id"]] = (split, c)
    tc = {t["case_id"]: t for t in read_jsonl(CASES_DIR / "ground_truth_corrected.jsonl")}
    to = {t["case_id"]: t for t in read_jsonl(CASES_DIR / "ground_truth_original.jsonl")}
    return cases, tc, to


def show(cid, split, inp, tc, to):
    ws, we = pd.Timestamp(inp["window_start"]), pd.Timestamp(inp["window_end"])
    print("=" * 100)
    print(f"{cid} [{split}] window {ws} -> {we} | alerts={inp['n_alerts']} score={inp['alert_score']}")
    print(f"seed: {inp['seed']}")
    t, o = tc[cid], to[cid]
    print(f"TRUTH corrected: {t['verdict']}/{t['family']} purity={t['purity']} flags={t['flags']}")
    print(f"      attackers={t['attackers']} victims={t['victims']} ports={t['ports']} "
          f"(distinct={t['n_distinct_ports']}) active {t['active_start']} -> {t['active_end']}")
    print(f"TRUTH original : {o['verdict']}/{o['family']} purity={o['purity']}")
    fams = eval_query(
        'SELECT family_corrected AS corrected, family_original AS original, COUNT(*) AS n FROM flows_eval '
        'WHERE "timestamp" >= ? AND "timestamp" <= ? GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 6',
        [ws.to_pydatetime(), we.to_pydatetime()])
    print("labels actually in window:\n" + fams.to_string(index=False))
    r = ToolRunner(ToolContext(window_start=ws, window_end=we)).call("case_overview")
    if r.ok:
        o = r.output
        print(f"agent sees: {o['n_flows']} flows, {o['n_src_ips']} src IPs, {o['n_dst_ips']} dst IPs, {o['n_dst_ports']} dst ports")
        print("  top_sources     :", json.dumps(o["top_sources"][:3]))
        print("  top_destinations:", json.dumps(o["top_destinations"][:3]))
        print("  top_dst_ports   :", json.dumps(o["top_dst_ports"][:3]))
    else:
        print("case_overview FAILED:", r.output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--flagged", action="store_true")
    ap.add_argument("--case")
    a = ap.parse_args()
    cases, tc, to = load()
    if a.case:
        ids = [a.case]
    elif a.flagged:
        ids = [c for c in sorted(cases) if tc[c]["flags"]]
    else:
        ids = sorted(cases)
        random.Random(42).shuffle(ids)
        ids = ids[: a.n]
    for cid in ids:
        show(cid, cases[cid][0], cases[cid][1], tc, to)
    print(f"\n{len(ids)} cases shown")