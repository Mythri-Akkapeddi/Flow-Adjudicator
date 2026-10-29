import json

import pandas as pd
import pytest

from flowadj.cases.build_cases import CASE_INPUT_KEYS, build_clusters
from flowadj.cases.ground_truth import truth_for_case
from flowadj.cases.jsonl import CASES_DIR, read_jsonl
from flowadj.cases.splits import EPOCH
from flowadj.config import get_default
from flowadj.data.store import eval_query

CFG = get_default()
CC = CFG["cases"]
BLOCK_S = CFG["splits"]["block_minutes"] * 60


# ---------- clustering (synthetic alerts) ----------
def _alerts():
    b = 3000
    t0 = EPOCH + pd.Timedelta(seconds=b * BLOCK_S)
    rows = []

    def add(i, src, dst, off, score):
        rows.append(dict(row_id=i, block_id=b, split="test", src_ip=src, dst_ip=dst, dst_port=80,
                         timestamp=t0 + pd.Timedelta(seconds=off), alert_score=score))
    for i in range(10):
        add(i, "1.1.1.1", "2.2.2.2", 200 + 5 * i, 0.99 if i == 4 else 0.9)
    for i in range(2):                      # too few alerts: dropped
        add(100 + i, "3.3.3.3", "4.4.4.4", 100 + i, 0.9)
    for i in range(4):                      # same pair, > gap later: second case
        add(200 + i, "2.2.2.2", "1.1.1.1", 500 + i, 0.8)
    return pd.DataFrame(rows), t0


def test_clustering():
    alerts, t0 = _alerts()
    pool, amap = build_clusters(alerts, CC, CFG["splits"]["block_minutes"], CFG["splits"]["buffer_minutes"])
    assert len(pool) == 2 and len(amap) == 14
    first = pool[0]["input"]
    assert set(first) == set(CASE_INPUT_KEYS)
    assert first["n_alerts"] == 10
    assert first["seed"]["timestamp"] == (t0 + pd.Timedelta(seconds=220)).isoformat(sep=" ")  # highest score
    for r in pool:
        ws, we = pd.Timestamp(r["input"]["window_start"]), pd.Timestamp(r["input"]["window_end"])
        assert t0 <= ws < we <= t0 + pd.Timedelta(seconds=BLOCK_S)


# ---------- ground truth (fixture) ----------
@pytest.fixture
def win(mini_db):
    return eval_query("SELECT row_id, src_ip, dst_ip, dst_port, timestamp, family_corrected, family_original FROM flows_eval",
                      db_path=mini_db)


def _truth(win, ids, eps):
    return truth_for_case("t", win, set(ids), eps, "family_corrected", CC)


def test_truth_portscan(win):
    ids = win.loc[win.family_corrected == "portscan", "row_id"]
    t = _truth(win, ids, ("172.16.0.1", "192.168.10.50"))
    assert t["verdict"] == "attack" and t["family"] == "portscan"
    assert t["attackers"] == ["172.16.0.1"] and t["victims"] == ["192.168.10.50"]
    assert t["ports"] == [] and t["n_distinct_ports"] == 100
    assert t["purity"] == 1.0


def test_truth_ssh(win):
    ids = win.loc[win.family_corrected == "ssh_bruteforce", "row_id"]
    t = _truth(win, ids, ("172.16.0.1", "192.168.10.50"))
    assert t["family"] == "ssh_bruteforce" and t["ports"] == [22] and t["n_distinct_ports"] == 1
    assert t["active_start"] == "2017-07-07 14:40:00" and t["active_end"] == "2017-07-07 14:44:55"


def test_truth_benign_is_flagged_when_attacks_nearby(win):
    ids = win.loc[win.family_corrected == "benign", "row_id"]
    t = _truth(win, ids, ("192.168.10.5", "192.168.10.3"))
    assert t["verdict"] == "benign" and t["family"] == "benign" and t["attackers"] == []
    assert "attack_traffic_elsewhere_in_window" in t["flags"]
    assert "benign_during_scheduled_attack" in t["flags"]   # fixture sits inside the Friday portscan slot


def test_truth_mixed_alerts_have_low_purity(win):
    a = win.loc[win.family_corrected == "portscan", "row_id"].head(50)
    b = win.loc[win.family_corrected == "benign", "row_id"].head(50)
    t = _truth(win, list(a) + list(b), ("172.16.0.1", "192.168.10.50"))
    assert t["verdict"] == "attack" and t["purity"] == pytest.approx(0.5)
    assert t["purity"] < CC["min_purity"]


# ---------- real artifacts ----------
POOL = CASES_DIR / "cases_pool.jsonl"
needs_pool = pytest.mark.skipif(not POOL.exists(), reason="run: uv run python -m scripts.04_build_cases --stage pool")


@needs_pool
def test_pool_inputs_have_only_allowed_keys():
    for r in read_jsonl(POOL):
        assert set(r["input"]) == set(CASE_INPUT_KEYS)
        assert set(r["input"]["seed"]) == {"src_ip", "dst_ip", "dst_port", "timestamp"}


@needs_pool
def test_pool_inputs_contain_no_label_words():
    text = json.dumps([r["input"] for r in read_jsonl(POOL)]).lower()
    for w in ("label", "family", "benign", "attack", "verdict", "patator", "portscan", "ddos", "botnet"):
        assert w not in text


@needs_pool
def test_case_ids_unique():
    ids = [r["input"]["case_id"] for r in read_jsonl(POOL)]
    assert len(ids) == len(set(ids))


@needs_pool
def test_windows_stay_inside_their_block():
    for r in read_jsonl(POOL):
        start = EPOCH + pd.Timedelta(seconds=r["block_id"] * BLOCK_S)
        assert pd.Timestamp(r["input"]["window_start"]) >= start
        assert pd.Timestamp(r["input"]["window_end"]) <= start + pd.Timedelta(seconds=BLOCK_S)


@needs_pool
def test_no_window_overlap_across_splits():
    spans = sorted((pd.Timestamp(r["input"]["window_start"]), pd.Timestamp(r["input"]["window_end"]), r["split"])
                   for r in read_jsonl(POOL))
    last_end = {}
    for s, e, sp in spans:
        for other, le in last_end.items():
            if other != sp:
                assert le <= s, f"{sp} window starting {s} overlaps a {other} window ending {le}"
        last_end[sp] = max(last_end.get(sp, e), e)