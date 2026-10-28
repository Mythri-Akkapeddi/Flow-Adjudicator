"""Budget tracking: LLM calls, tool calls, tokens."""
from dataclasses import dataclass, asdict

from flowadj.config import get_default


class BudgetExceeded(Exception):
    def __init__(self, resource: str, used: int, limit: int):
        self.resource = resource
        self.used = used
        self.limit = limit
        super().__init__(f"Budget exceeded for {resource}: used {used}, limit {limit}")


@dataclass
class Budget:
    max_llm_calls: int = 16
    max_tool_calls: int = 24
    max_total_tokens: int = 48000
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_config(cls) -> "Budget":
        b = get_default()["budget"]
        return cls(
            max_llm_calls=b["max_llm_calls"],
            max_tool_calls=b["max_tool_calls"],
            max_total_tokens=b["max_total_tokens"],
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def ensure_llm(self) -> None:
        """Call BEFORE every LLM call. Raises if another call is not allowed."""
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded("llm_calls", self.llm_calls, self.max_llm_calls)
        if self.total_tokens >= self.max_total_tokens:
            raise BudgetExceeded("total_tokens", self.total_tokens, self.max_total_tokens)

    def record_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Call AFTER every LLM call with the token counts it reported."""
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def record_tool(self) -> None:
        """Call BEFORE executing a tool. Raises if the tool-call cap is reached."""
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("tool_calls", self.tool_calls, self.max_tool_calls)
        self.tool_calls += 1

    @property
    def exhausted(self) -> bool:
        return (
            self.llm_calls >= self.max_llm_calls
            or self.total_tokens >= self.max_total_tokens
        )

    def split(self, fractions: dict[str, float]) -> dict[str, "Budget"]:
        """Split this budget into sub-budgets, e.g. {'evidence': 0.5, 'coordinator': 0.5}."""
        if sum(fractions.values()) > 1.0 + 1e-9:
            raise ValueError("fractions must sum to at most 1.0")
        return {
            name: Budget(
                max_llm_calls=int(self.max_llm_calls * f),
                max_tool_calls=int(self.max_tool_calls * f),
                max_total_tokens=int(self.max_total_tokens * f),
            )
            for name, f in fractions.items()
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d