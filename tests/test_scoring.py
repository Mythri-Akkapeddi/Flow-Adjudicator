import math

import pandas as pd
import pytest

from flowadj.agents.schema import Report
from flowadj.eval.score import aggregate, jaccard, mcc, score_case, set_f1, window_iou

TRUTH = {"case_id": "c1", "verdict": "attack", "family": "ssh_bruteforce",
         "attackers": ["1.1.1.1"], "victims": ["2.2.2.2"], "ports": [22],
         "active_start": "2017-07-07 10:00:00", "active_end": "2017-07-07 10:10:00"}
BENIGN = {"case_id": "c2", "verdict": "benign", "family": "benign", "attackers": [], "victims": [],
          "ports": [], "active_start": None, "active_end": None}


def test_set_f1():
    assert set_f1(["a", "b"], ["a", "c"]) == pytest.approx(0.5)
    assert set_f1([], []) == 1.0
    assert set_f1([], ["a"]) == 0.0
    assert set_f1(["a"], ["b"]) == 0.0


def test_jaccard():
    assert jaccard([1, 2], [1, 3]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0


def test_window_iou():
    assert window_iou("2017-07-07 10:05:00", "2017-07-07 10:15:00",
                      "2017-07-07 10:00:00", "2017-07-07 10:10:00") == pytest.approx(1 / 3)
    assert window_iou("2017-07-07 10:00:00", "2017-07-07 10:10:00",
                      "2017-07-07 10:00:00", "2017-07-07 10:10:00") == 1.0
    assert window_iou(None, None, "2017-07-07 10:00:00", "2017-07-07 10:10:00") == 0.0
    assert window_iou("2017-07-07 11:00:00", "2017-07-07 11:10:00",
                      "2017-07-07 10:00:00", "2017-07-07 10:10:00") == 0.0


def test_mcc():
    assert mcc([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2 / math.sqrt(12))
    assert mcc([1, 0], [1, 0]) == 1.0
    assert mcc([1, 0], [0, 1]) == -1.0
    assert mcc([1, 1], [1, 1]) == 0.0  # undefined -> 0


def test_perfect_attack_report():
    r = Report(verdict="attack", family="ssh_bruteforce", attackers=["1.1.1.1"], victims=["2.2.2.2"],
               ports=[22], window_start="2017-07-07 10:00:00", window_end="2017-07-07 10:10:00")
    s = score_case(r, TRUTH)
    assert s["verdict_correct"] and s["family_correct"]
    assert s["attacker_f1"] == s["victim_f1"] == s["port_jaccard"] == s["window_iou"] == 1.0


def test_benign_case_has_no_entity_scores():
    r = Report(verdict="benign", family="benign")
    s = score_case(r, BENIGN)
    assert s["verdict_correct"] and s["family_correct"]
    assert math.isnan(s["attacker_f1"]) and math.isnan(s["window_iou"])


def test_false_positive_on_benign():
    s = score_case(Report(verdict="attack", family="dos"), BENIGN)
    assert not s["verdict_correct"] and not s["family_correct"]


def test_invalid_report_is_a_failure():
    s = score_case(None, TRUTH)
    assert not s["valid"] and not s["verdict_correct"] and not s["family_correct"]
    assert s["attacker_f1"] == 0.0 and s["port_jaccard"] == 0.0
    sb = score_case(None, BENIGN)
    assert not sb["verdict_correct"]


def test_empty_truth_ports_skip_port_metric():
    t = dict(TRUTH, ports=[], family="portscan")
    s = score_case(Report(verdict="attack", family="portscan", attackers=["1.1.1.1"]), t)
    assert math.isnan(s["port_jaccard"])


def test_aggregate():
    rows = pd.DataFrame([
        score_case(Report(verdict="attack", family="ssh_bruteforce", attackers=["1.1.1.1"]), TRUTH),
        score_case(Report(verdict="benign", family="benign"), BENIGN),
    ])
    a = aggregate(rows)
    assert a["n"] == 2 and a["mcc"] == 1.0 and a["family_acc"] == 1.0
    assert a["attacker_f1"] == 1.0   # only the attack case counts