"""Provider-agnostic chat() with a JSON-in-text fallback for tool calls."""
import json
import re

from flowadj.config import get_model_cfg
from flowadj.llm.ollama_backend import OllamaBackend
from flowadj.llm.types import LLMResponse, ToolCall

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_backend(model_key: str = "local"):
    cfg = get_model_cfg(model_key)
    if cfg["provider"] == "ollama":
        return OllamaBackend(
            model=cfg["name"],
            host=cfg.get("host", "http://localhost:11434"),
            num_ctx=cfg.get("num_ctx", 16384),
            temperature=cfg.get("temperature", 0.3),
            seed=cfg.get("seed", 42),
            think=cfg.get("think", False),
        )
    raise NotImplementedError(f"Unknown provider: {cfg['provider']}")


def _json_candidates(text: str):
    """Yield every JSON object/list that can be decoded from the text."""
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                obj, end = dec.raw_decode(text, i)
                yield obj
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1


def _flatten(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from _flatten(x)
    elif isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            yield from _flatten(obj["tool_calls"])
        else:
            yield obj


def parse_tool_calls_from_text(text: str, allowed_names: set[str]) -> list[ToolCall]:
    """Fallback: find tool calls written as JSON inside the model's text.

    Accepts {"name": ..., "arguments": {...}} and common variants.
    Only accepts names that are in allowed_names.
    """
    text = _THINK_RE.sub("", text)
    calls: list[ToolCall] = []
    for obj in _json_candidates(text):
        for item in _flatten(obj):
            name = item.get("name") or item.get("tool")
            fn = item.get("function")
            args = item.get("arguments", item.get("args", item.get("parameters")))
            if isinstance(fn, dict):
                name = fn.get("name", name)
                args = fn.get("arguments", args)
            elif isinstance(fn, str) and not name:
                name = fn
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if name in allowed_names and isinstance(args, dict):
                calls.append(ToolCall(name=name, arguments=args, id=f"fb_{len(calls)}"))
    return calls


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model_key: str = "local",
    backend=None,
) -> LLMResponse:
    backend = backend or build_backend(model_key)
    resp = backend.chat(messages, tools)
    if tools and not resp.tool_calls:
        names = {t["function"]["name"] for t in tools}
        fallback = parse_tool_calls_from_text(resp.text, names)
        if fallback:
            resp.tool_calls = fallback
            resp.used_fallback = True
    return resp