"""Local backend using Ollama's native chat API."""
import time

import ollama

from flowadj.llm.types import LLMResponse, ToolCall


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 16384,
        temperature: float = 0.3,
        seed: int = 42,
        think: bool | None = False,
    ):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.client = ollama.Client(host=host)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            messages=messages,
            options={
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        )
        if tools:
            kwargs["tools"] = tools
        if self.think is not None:
            kwargs["think"] = self.think

        t0 = time.perf_counter()
        resp = self.client.chat(**kwargs)
        latency = time.perf_counter() - t0

        msg = resp.message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            calls.append(
                ToolCall(
                    name=tc.function.name,
                    arguments=dict(tc.function.arguments or {}),
                    id=f"call_{i}",
                )
            )
        return LLMResponse(
            text=msg.content or "",
            tool_calls=calls,
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
            model=self.model,
            latency_s=latency,
            raw=resp,
        )