"""S1: one agent with all tools."""
from pathlib import Path

import flowadj.tools  # noqa: F401  (registers every tool)
from flowadj.agents.base import AgentResult, case_prompt, run_agent
from flowadj.data.store import DB_PATH
from flowadj.llm.budget import Budget
from flowadj.tools.registry import ToolContext, ToolRunner

PROMPT = Path(__file__).parent / "prompts" / "single.md"


def run_s1(case: dict, db_path=DB_PATH, backend=None, budget: Budget | None = None,
           tracer=None, model_key: str = "local") -> AgentResult:
    budget = budget or Budget.from_config()
    ctx = ToolContext(db_path=db_path, window_start=case["window_start"], window_end=case["window_end"])
    runner = ToolRunner(ctx, budget=budget)
    return run_agent(
        PROMPT.read_text(), case_prompt(case), runner, budget,
        backend=backend, model_key=model_key, tracer=tracer, agent_name="s1",
        preamble=[("case_overview", {})],
    )