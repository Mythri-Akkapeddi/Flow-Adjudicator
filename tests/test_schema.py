from flowadj.agents.schema import Report, extract_report

GOOD = '{"verdict": "attack", "family": "portscan", "attackers": ["1.1.1.1"], "victims": ["2.2.2.2"], "ports": [], "window_start": "2017-07-07 14:20:00", "window_end": "2017-07-07 14:20:30", "summary": "scan", "claims": [{"type": "ip", "value": "1.1.1.1", "tool_call_id": "tc_001"}]}'


def test_plain_json():
    r, err = extract_report(GOOD)
    assert err is None and r.verdict == "attack" and r.claims[0].tool_call_id == "tc_001"


def test_fenced_with_think_and_prose():
    r, err = extract_report("<think>hmm</think>Here you go:\n```json\n" + GOOD + "\n```")
    assert r is not None


def test_wrapped_in_report_key():
    r, _ = extract_report('{"report": ' + GOOD + "}")
    assert r is not None


def test_case_and_numeric_ports_are_normalised():
    r, _ = extract_report('{"verdict": "ATTACK", "family": "DoS", "ports": ["80"]}')
    assert r.family == "dos" and r.ports == [80]


def test_invalid_family_fails():
    r, err = extract_report('{"verdict": "attack", "family": "malware"}')
    assert r is None and "family" in err


def test_garbage_fails():
    r, err = extract_report("I think it is an attack.")
    assert r is None and err


def test_defaults():
    r = Report(verdict="benign", family="benign")
    assert r.attackers == [] and r.claims == []