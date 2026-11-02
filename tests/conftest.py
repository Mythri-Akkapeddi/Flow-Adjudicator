import runpy
from pathlib import Path

import pytest

from flowadj.data.store import build_store, close_all
from flowadj.llm.types import LLMResponse, ToolCall

FIXTURE = Path(__file__).parent / "fixtures" / "mini_flows.parquet"


@pytest.fixture(scope="session")
def mini_flows_path():
    if not FIXTURE.exists():
        runpy.run_path(str(FIXTURE.parent / "make_fixtures.py"), run_name="__main__")
    return FIXTURE


@pytest.fixture(scope="session")
def mini_db(tmp_path_factory, mini_flows_path):
    db = tmp_path_factory.mktemp("db") / "mini.duckdb"
    build_store(mini_flows_path, db)
    yield db
    close_all()


class MockBackend:
    """Scripted LLM. script = list of {"text": str, "calls": [(tool_name, args), ...]}.
    Items are used in order while tools are enabled; `final` is used when tools are disabled
    (forced final answer) or when the script runs out."""

    def __init__(self, script, final=None):
        self.script = list(script)
        self.final = final
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"n_messages": len(messages), "tools_enabled": bool(tools)})
        if tools and self.script:
            item = self.script.pop(0)
        elif self.final is not None:
            item = self.final
        else:
            raise AssertionError("mock LLM ran out of scripted responses")
        calls = [ToolCall(name=n, arguments=a, id=f"call_{i}") for i, (n, a) in enumerate(item.get("calls", []))]
        return LLMResponse(text=item.get("text", ""), tool_calls=calls,
                           prompt_tokens=100, completion_tokens=50, model="mock")


@pytest.fixture
def mock_backend_cls():
    return MockBackend