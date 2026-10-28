import pytest

from flowadj.llm.budget import Budget, BudgetExceeded
from flowadj.tools.registry import ToolContext, ToolRegistry, ToolRunner


def make_registry():
    reg = ToolRegistry()

    @reg.tool
    def echo(ctx: ToolContext, host: str, n: int = 5, flag: bool = False, hosts: list[str] | None = None):
        """Echo the arguments back.

        Args:
            host: a host IP
            n: how many rows
        """
        return {"host": host, "n": n, "flag": flag, "hosts": hosts}

    @reg.tool
    def many(ctx: ToolContext, k: int = 100):
        """Return k rows."""
        return {"rows": [{"i": i} for i in range(k)]}

    @reg.tool
    def huge(ctx: ToolContext):
        """Return a very large string."""
        return {"blob": "x" * 50_000}

    @reg.tool
    def boom(ctx: ToolContext):
        """Always fails."""
        raise RuntimeError("kaboom")

    return reg


def runner(reg=None, **kw):
    return ToolRunner(ToolContext(), reg=reg or make_registry(), **kw)


def test_schema_generation():
    r = runner()
    schemas = {s["function"]["name"]: s["function"] for s in r.schemas()}
    f = schemas["echo"]
    assert f["description"] == "Echo the arguments back."
    props = f["parameters"]["properties"]
    assert "ctx" not in props
    assert props["host"]["type"] == "string" and props["host"]["description"] == "a host IP"
    assert props["n"]["type"] == "integer" and props["n"]["default"] == 5
    assert props["flag"]["type"] == "boolean"
    assert props["hosts"] == {"type": "array", "items": {"type": "string"}}
    assert f["parameters"]["required"] == ["host"]


def test_registration_rules():
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        @reg.tool
        def bad(x: int):
            """no ctx"""
    with pytest.raises(ValueError):
        @reg.tool
        def nodoc(ctx):
            pass


def test_row_cap():
    res = runner(max_rows=20).call("many", {"k": 100})
    assert res.ok
    assert len(res.output["rows"]) == 20
    assert res.output["_truncated"]["original_lengths"]["rows"] == 100


def test_size_cap():
    res = runner(max_chars=1000).call("huge")
    assert not res.ok and "too large" in res.output["error"]


def test_allow_list():
    r = runner(allowed={"echo"})
    assert [s["function"]["name"] for s in r.schemas()] == ["echo"]
    res = r.call("many", {"k": 3})
    assert not res.ok and "not-allowed" in res.output["error"]
    assert r.call("echo", {"host": "1.2.3.4"}).ok


def test_call_ids_sequential_and_logged():
    r = runner()
    ids = [r.call("echo", {"host": "a"}).call_id, r.call("nope").call_id, r.call("many").call_id]
    assert ids == ["tc_001", "tc_002", "tc_003"]
    assert [x.call_id for x in r.log] == ids
    assert '"call_id": "tc_001"' in r.log[0].to_message()


def test_argument_validation_and_coercion():
    r = runner()
    assert not r.call("echo", {}).ok                                   # missing required
    assert not r.call("echo", {"host": "a", "bogus": 1}).ok            # unknown arg
    ok = r.call("echo", {"host": "a", "n": "7"})                       # "7" -> 7
    assert ok.ok and ok.output["n"] == 7
    assert not r.call("echo", {"host": "a", "n": "abc"}).ok            # not coercible


def test_tool_exception_becomes_error_result():
    res = runner().call("boom")
    assert not res.ok and "kaboom" in res.output["error"]


def test_budget_counts_every_call_and_raises():
    b = Budget(max_llm_calls=10, max_tool_calls=2, max_total_tokens=10_000)
    r = runner(budget=b)
    r.call("echo", {"host": "a"})
    r.call("nope")          # failed calls still count
    with pytest.raises(BudgetExceeded):
        r.call("echo", {"host": "a"})
    assert b.tool_calls == 2