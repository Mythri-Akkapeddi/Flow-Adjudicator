from flowadj.llm.client import parse_tool_calls_from_text

NAMES = {"top_talkers", "port_profile"}


def test_plain_json():
    t = '{"name": "top_talkers", "arguments": {"n": 5}}'
    c = parse_tool_calls_from_text(t, NAMES)
    assert len(c) == 1 and c[0].name == "top_talkers" and c[0].arguments == {"n": 5}


def test_fenced_with_prose_and_think():
    t = "<think>hmm</think>Sure:\n```json\n{\"tool\": \"port_profile\", \"args\": {\"host\": \"1.2.3.4\"}}\n```"
    c = parse_tool_calls_from_text(t, NAMES)
    assert len(c) == 1 and c[0].name == "port_profile"


def test_list_and_openai_style():
    t = '[{"function": {"name": "top_talkers", "arguments": "{\\"n\\": 3}"}}]'
    c = parse_tool_calls_from_text(t, NAMES)
    assert c[0].arguments == {"n": 3}


def test_unknown_tool_rejected():
    assert parse_tool_calls_from_text('{"name": "rm_rf", "arguments": {}}', NAMES) == []


def test_no_json():
    assert parse_tool_calls_from_text("I think it is an attack.", NAMES) == []