import pytest

from flowadj.llm.budget import Budget, BudgetExceeded


def test_llm_call_limit():
    b = Budget(max_llm_calls=2, max_tool_calls=5, max_total_tokens=10_000)
    b.ensure_llm(); b.record_llm(10, 5)
    b.ensure_llm(); b.record_llm(10, 5)
    with pytest.raises(BudgetExceeded) as e:
        b.ensure_llm()
    assert e.value.resource == "llm_calls"


def test_token_limit():
    b = Budget(max_llm_calls=10, max_tool_calls=5, max_total_tokens=100)
    b.ensure_llm()
    b.record_llm(80, 30)  # 110 tokens, over the cap
    with pytest.raises(BudgetExceeded) as e:
        b.ensure_llm()
    assert e.value.resource == "total_tokens"
    assert b.exhausted


def test_tool_call_limit():
    b = Budget(max_llm_calls=10, max_tool_calls=2, max_total_tokens=10_000)
    b.record_tool()
    b.record_tool()
    with pytest.raises(BudgetExceeded) as e:
        b.record_tool()
    assert e.value.resource == "tool_calls"
    assert b.tool_calls == 2


def test_token_accounting():
    b = Budget()
    b.record_llm(100, 20)
    b.record_llm(50, 10)
    assert b.prompt_tokens == 150
    assert b.completion_tokens == 30
    assert b.total_tokens == 180
    assert b.llm_calls == 2


def test_split_never_exceeds_parent():
    b = Budget(max_llm_calls=16, max_tool_calls=24, max_total_tokens=48000)
    parts = b.split({"evidence": 0.6, "coordinator": 0.4})
    assert sum(p.max_llm_calls for p in parts.values()) <= 16
    assert sum(p.max_tool_calls for p in parts.values()) <= 24
    assert sum(p.max_total_tokens for p in parts.values()) <= 48000
    with pytest.raises(ValueError):
        b.split({"a": 0.7, "b": 0.7})


def test_from_config():
    b = Budget.from_config()
    assert b.max_llm_calls == 16