"""Stratified dev/test sampling and the freeze hash. EVALUATOR-SIDE."""
import hashlib
from collections import Counter

import pandas as pd

from flowadj.cases.jsonl import CASES_DIR, read_jsonl, write_jsonl
from flowadj.config import get_default

EXCLUDE_FILE = CASES_DIR / "audit_exclusions.txt"   # one case_id per line, '#' comments allowed
HASH_FILE = CASES_DIR / "test_ids.sha256"


def _exclusions() -> set:
    if not EXCLUDE_FILE.exists():
        return set()
    lines = (l.split("#")[0].strip() for l in EXCLUDE_FILE.read_text().splitlines())
    return {l for l in lines if l}


def _eligible(pool, truth, cc) -> pd.DataFrame:
    p = pd.DataFrame([{"case_id": r["input"]["case_id"], "split": r["split"], "block_id": r["block_id"]} for r in pool])
    df = p.merge(pd.DataFrame(truth), on="case_id")
    bad = set(cc["exclude_flags"])
    flagged = df["flags"].map(lambda fl: any(f in bad for f in fl))
    ok = (df["purity"] >= cc["min_purity"]) & ~flagged & ~df["case_id"].isin(_exclusions())
    return df[ok].copy()


def _pick(df: pd.DataFrame, n: int, max_per_block: int, seed: int, by_family: bool) -> list:
    if df.empty or n <= 0:
        return []
    df = df.sort_values("case_id").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["rank_in_block"] = df.groupby("block_id").cumcount()
    df = df.sort_values("rank_in_block", kind="stable")   # first case of every block before any second one
    groups = df.groupby("family") if by_family else [("all", df)]
    queues = {fam: list(g.itertuples()) for fam, g in groups}
    chosen, per_block = [], Counter()
    while len(chosen) < n and any(queues.values()):
        progressed = False
        for fam in sorted(queues):
            q = queues[fam]
            while q:
                r = q.pop(0)
                if per_block[r.block_id] < max_per_block:
                    chosen.append(r)
                    per_block[r.block_id] += 1
                    progressed = True
                    break
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return [r.case_id for r in chosen]


def sample_and_freeze() -> dict:
    cfg = get_default()
    cc, seed = cfg["cases"], cfg["seed"]
    pool = read_jsonl(CASES_DIR / "cases_pool.jsonl")
    truth = read_jsonl(CASES_DIR / "ground_truth_corrected.jsonl")
    el = _eligible(pool, truth, cc)
    inputs = {r["input"]["case_id"]: r["input"] for r in pool}
    report = {"pool": len(pool), "eligible": len(el), "excluded_by_audit": len(_exclusions()), "splits": {}}
    sel = {}
    for split in ("dev", "test"):
        sub = el[el["split"] == split]
        atk = _pick(sub[sub["verdict"] == "attack"], cc[split]["attack"], cc["max_per_block"], seed, True)
        ben = _pick(sub[sub["verdict"] == "benign"], cc[split]["benign"], cc["max_per_block"], seed + 1, False)
        sel[split] = sorted(atk + ben)
        chosen = sub[sub["case_id"].isin(sel[split])]
        report["splits"][split] = {
            "n": len(sel[split]), "attack": len(atk), "benign": len(ben),
            "target_attack": cc[split]["attack"], "target_benign": cc[split]["benign"],
            "families": chosen[chosen["verdict"] == "attack"]["family"].value_counts().to_dict(),
            "eligible_by_family": sub[sub["verdict"] == "attack"]["family"].value_counts().to_dict(),
            "shortfall": (len(atk) < cc[split]["attack"]) or (len(ben) < cc[split]["benign"]),
        }
        write_jsonl(CASES_DIR / f"cases_{split}.jsonl", [inputs[c] for c in sel[split]])
    digest = hashlib.sha256("\n".join(sel["test"]).encode()).hexdigest()
    HASH_FILE.write_text(f"{digest}  test_ids n={len(sel['test'])}\n")
    report["test_ids_sha256"] = digest
    (CASES_DIR / "sampling_report.json").write_text(__import__("json").dumps(report, indent=2))
    return report


def verify_frozen() -> bool:
    ids = sorted(r["case_id"] for r in read_jsonl(CASES_DIR / "cases_test.jsonl"))
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return digest == HASH_FILE.read_text().split()[0]