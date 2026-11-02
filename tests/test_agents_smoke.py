import json

from flowadj.agents.single import run_s1
from flowadj.eval.grounding import check_report
from flowadj.llm.budget import Budget

CASE = {
    "case_id": "case_mock", "window_start": "2017-07-07 14:00:00", "window_end": "2017-07-07 15:00:00",
    "n_alerts": 10, "alert_score": 0.99,
    "seed": {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "dst_port": 22, "timestamp": "2017-07-07 14:40:00"},
}
REPORT = {
    "verdict": "attack", "family": "ssh_bruteforce", "attackers": ["172.16.0.1"],
    "victims": ["192.168.10.50"], "ports": [22],
    "window_start": "2017-07-07 14:40:00", "window_end": "2017-07-07 14:44:55", "summary": "mock",
    "claims": [
        {"type": "ip", "value": "172.16.0.1", "tool_call_id": "tc_002"},
        {"type": "count", "value": 160, "tool_call_id": "tc_002"},
        {"type": "ip", "value": "9.9.9.9", "tool_call_id": "tc_002"},
    ],
}
REPORT_TEXT = json.dumps(REPORT)


def test_s1_happy_path(mini_db, mock_backend_cls):
    backend = mock_backend_cls([{"calls": [("top_talkers", {"by": "src_ip"})]}, {"text": REPORT_TEXT}])
    res = run_s1(CASE, db_path=mini_db, backend=backend, budget=Budget(16, 24, 48000))
    assert res.status == "ok" and res.report.verdict == "attack"
    assert [t.call_id for t in res.tool_log] == ["tc_001", "tc_002"]   # overview + top_talkers
    assert res.steps == 2 and not res.forced_final
    g = check_report(res.report, res.tool_log)
    assert g["n_claims"] == 3 and g["n_supported"] == 2     # 9.9.9.9 is made up


def test_forced_final_when_llm_budget_runs_out(mini_db, mock_backend_cls):
    backend = mock_backend_cls([{"calls": [("top_talkers", {})]}] * 50, final={"text": REPORT_TEXT})
    b = Budget(max_llm_calls=4, max_tool_calls=24, max_total_tokens=48000)
    res = run_s1(CASE, db_path=mini_db, backend=backend, budget=b)
    assert res.status == "ok" and res.forced_final
    assert b.llm_calls <= 4
    assert backend.calls[-1]["tools_enabled"] is False


def test_forced_final_when_tool_budget_runs_out(mini_db, mock_backend_cls):
    backend = mock_backend_cls([{"calls": [("top_talkers", {})]}] * 50, final={"text": REPORT_TEXT})
    b = Budget(max_llm_calls=16, max_tool_calls=3, max_total_tokens=48000)
    res = run_s1(CASE, db_path=mini_db, backend=backend, budget=b)
    assert res.status == "ok" and res.forced_final
    assert b.tool_calls <= 3


def test_schema_failure_is_reported_not_raised(mini_db, mock_backend_cls):
    backend = mock_backend_cls([{"text": "not json"}], final={"text": "still not json"})
    res = run_s1(CASE, db_path=mini_db, backend=backend, budget=Budget(16, 24, 48000))
    assert res.status == "schema_failure" and res.report is None


def test_bad_tool_call_is_recoverable(mini_db, mock_backend_cls):
    backend = mock_backend_cls([{"calls": [("no_such_tool", {})]}, {"text": REPORT_TEXT}])
    res = run_s1(CASE, db_path=mini_db, backend=backend, budget=Budget(16, 24, 48000))
    assert res.status == "ok"
    assert res.tool_log[-1].ok is False


def test_tool_use_never_exceeds_caps(mini_db, mock_backend_cls):
    backend = mock_backend_cls([{"calls": [("top_talkers", {}), ("port_profile", {}), ("host_summary", {"host": "172.16.0.1"})]}] * 20,
                               final={"text": REPORT_TEXT})
    b = Budget(max_llm_calls=16, max_tool_calls=5, max_total_tokens=48000)
    res = run_s1(CASE, db_path=mini_db, backend=backend, budget=b)
    assert b.tool_calls <= 5 and res.status == "ok"