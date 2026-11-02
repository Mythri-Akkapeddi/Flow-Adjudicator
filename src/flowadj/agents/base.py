"""Shared tool-calling loop used by S1, the evidence agents, coordinator and critic."""
import json
from dataclasses import asdict, dataclass, field

from flowadj.agents.schema import Report, extract_report
from flowadj.llm.budget import Budget, BudgetExceeded
from flowadj.llm.client import chat
from flowadj.tools.registry import ToolRunner
from flowadj.trace import NullTracer

FINAL_NUDGE = ("Your tool or call budget is almost used up. Do not call any more tools. "
               "Reply now with ONLY the final JSON report object.")


@dataclass
class AgentResult:
    report: Report | None
    status: str                      # ok | schema_failure | budget_exceeded
    error: str | None = None
    raw_text: str = ""
    forced_final: bool = False
    steps: int = 0
    fallback_tool_calls: int = 0
    budget: dict = field(default_factory=dict)
    tool_log: list = field(default_factory=list)   # list[ToolResult]

    def to_dict(self) -> dict:
        return {
            "status": self.status, "error": self.error, "raw_text": self.raw_text,
            "forced_final": self.forced_final, "steps": self.steps,
            "fallback_tool_calls": self.fallback_tool_calls, "budget": self.budget,
            "report": self.report.model_dump() if self.report else None,
            "tool_log": [asdict(t) for t in self.tool_log],
        }


def case_prompt(case: dict) -> str:
    return "Case to investigate:\n" + json.dumps(case, indent=2)


def _tool_event(res) -> dict:
    return {"call_id": res.call_id, "tool": res.name, "arguments": res.arguments,
            "ok": res.ok, "output": res.output}


def run_agent(
    system_prompt: str,
    user_prompt: str,
    runner: ToolRunner,
    budget: Budget,
    backend=None,
    model_key: str = "local",
    tracer=None,
    agent_name: str = "agent",
    preamble: list | None = None,
    schema_retries: int = 1,
    final_fraction: float = 0.85,
) -> AgentResult:
    tracer = tracer or NullTracer()
    runner.budget = budget
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    if preamble:
        parts = []
        for name, args in preamble:
            try:
                res = runner.call(name, args)
            except BudgetExceeded:
                break
            tracer.log("tool_call", agent=agent_name, **_tool_event(res))
            parts.append(res.to_message())
        if parts:
            messages[1]["content"] += "\n\nTool results already gathered for you:\n" + "\n".join(parts)

    forced, retries_left, steps, fallback_calls, last_text = False, schema_retries, 0, 0, ""

    def finish(report, status, error=None):
        r = AgentResult(report=report, status=status, error=error, raw_text=last_text,
                        forced_final=forced, steps=steps, fallback_tool_calls=fallback_calls,
                        budget=budget.to_dict(), tool_log=list(runner.log))
        tracer.log("final", agent=agent_name, status=status, error=error, budget=r.budget,
                   report=r.report.model_dump() if r.report else None)
        return r

    while True:
        if not forced and (
            budget.llm_calls >= budget.max_llm_calls - 1
            or budget.tool_calls >= budget.max_tool_calls
            or budget.total_tokens >= final_fraction * budget.max_total_tokens
        ):
            forced = True
            messages.append({"role": "user", "content": FINAL_NUDGE})
        try:
            budget.ensure_llm()
        except BudgetExceeded as e:
            return finish(None, "budget_exceeded", str(e))

        resp = chat(messages, None if forced else runner.schemas(), model_key=model_key, backend=backend)
        budget.record_llm(resp.prompt_tokens, resp.completion_tokens)
        steps += 1
        last_text = resp.text
        tracer.log("llm_call", agent=agent_name, step=steps, forced_final=forced, text=resp.text,
                   tool_calls=[{"name": t.name, "arguments": t.arguments} for t in resp.tool_calls],
                   prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                   latency_s=round(resp.latency_s, 2), used_fallback=resp.used_fallback)

        if resp.tool_calls and not forced:
            if resp.used_fallback:
                fallback_calls += len(resp.tool_calls)
            messages.append({
                "role": "assistant",
                "content": "" if resp.used_fallback else resp.text,
                "tool_calls": [{"function": {"name": t.name, "arguments": t.arguments}} for t in resp.tool_calls],
            })
            for t in resp.tool_calls:
                try:
                    res = runner.call(t.name, t.arguments)
                except BudgetExceeded:
                    messages.append({"role": "tool", "tool_name": t.name,
                                     "content": json.dumps({"ok": False, "error": "tool-call budget exhausted"})})
                    continue
                tracer.log("tool_call", agent=agent_name, **_tool_event(res))
                messages.append({"role": "tool", "tool_name": t.name, "content": res.to_message()})
            continue

        report, err = extract_report(resp.text)
        if report is not None:
            return finish(report, "ok")
        if retries_left > 0:
            retries_left -= 1
            forced = True   # tools off from now on
            messages.append({"role": "assistant", "content": resp.text})
            messages.append({"role": "user", "content":
                             f"That was not a valid report ({err}). Reply with ONLY the JSON report object, no other text."})
            continue
        return finish(None, "schema_failure", err)