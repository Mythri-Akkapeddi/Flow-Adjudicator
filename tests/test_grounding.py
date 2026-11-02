from flowadj.agents.schema import Claim, Report
from flowadj.eval.grounding import check_report
from flowadj.tools.registry import ToolResult

LOG = [
    ToolResult("tc_001", "top_talkers", {}, True, {
        "rows": [{"src_ip": "172.16.0.1", "n_flows": 160, "share_of_flows": 0.4444}],
        "peak_bin_start": "2017-07-07 14:20:00"}),
    ToolResult("tc_002", "host_summary", {}, False, {"error": "boom"}),
    ToolResult("tc_003", "port_profile", {}, True, {"ports": [{"port": 22, "n_flows": 61}]}),
]


def rep(*claims):
    return Report(verdict="attack", family="dos", claims=list(claims))


def test_supported_claims():
    r = rep(Claim(type="ip", value="172.16.0.1", tool_call_id="tc_001"),
            Claim(type="count", value=160, tool_call_id="tc_001"),
            Claim(type="ratio", value=0.444, tool_call_id="tc_001"),
            Claim(type="ratio", value="44.4%", tool_call_id="tc_001"),
            Claim(type="time", value="2017-07-07 14:20:20", tool_call_id="tc_001"),
            Claim(type="port", value=22, tool_call_id="tc_003"))
    g = check_report(r, LOG)
    assert g["n_claims"] == 6 and g["n_supported"] == 6 and g["unsupported_rate"] == 0.0


def test_unsupported_reasons():
    r = rep(Claim(type="ip", value="9.9.9.9", tool_call_id="tc_001"),
            Claim(type="count", value=161, tool_call_id="tc_001"),
            Claim(type="ip", value="172.16.0.1"),
            Claim(type="ip", value="172.16.0.1", tool_call_id="tc_999"),
            Claim(type="ip", value="1.2.3.4", tool_call_id="tc_002"),
            Claim(type="port", value=80, tool_call_id="tc_003"),
            Claim(type="time", value="2017-07-07 15:00:00", tool_call_id="tc_001"))
    g = check_report(r, LOG)
    assert [d["reason"] for d in g["details"]] == [
        "value_not_in_output", "value_not_in_output", "no_citation", "unknown_call_id",
        "tool_error", "value_not_in_output", "value_not_in_output"]
    assert g["unsupported_rate"] == 1.0


def test_no_claims():
    g = check_report(rep(), LOG)
    assert g["n_claims"] == 0 and g["unsupported_rate"] is None


def test_none_report():
    assert check_report(None, LOG)["n_claims"] == 0


def test_dict_tool_log():
    log = [{"call_id": "tc_001", "ok": True, "output": {"x": 5}}]
    g = check_report(rep(Claim(type="count", value=5, tool_call_id="tc_001")), log)
    assert g["n_supported"] == 1