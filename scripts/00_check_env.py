"""Environment check: packages, Ollama, model, and a real tool-call round trip."""
import sys

import ollama

from flowadj.config import get_model_cfg
from flowadj.llm.client import chat


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    cfg = get_model_cfg("local")
    try:
        models = [m.model for m in ollama.Client(host=cfg["host"]).list().models]
    except Exception as e:
        print(f"FAIL: cannot reach Ollama at {cfg['host']}: {e}")
        print("Open the Ollama app and try again.")
        return 1
    print("Models available:", models)
    if not any(m == cfg["name"] or m.startswith(cfg["name"]) for m in models):
        print(f"FAIL: model {cfg['name']} not pulled. Run: ollama pull {cfg['name']}")
        return 1

    tools = [{
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }]
    messages = [{"role": "user", "content": "What is 17 plus 25? You must use the add_numbers tool."}]
    resp = chat(messages, tools)
    print(f"Latency: {resp.latency_s:.1f}s | prompt tokens: {resp.prompt_tokens} | completion tokens: {resp.completion_tokens}")
    print(f"Tool calls: {resp.tool_calls} (via fallback: {resp.used_fallback})")
    print(f"Text: {resp.text[:200]!r}")

    ok = (
        len(resp.tool_calls) >= 1
        and resp.tool_calls[0].name == "add_numbers"
        and resp.prompt_tokens > 0
        and resp.completion_tokens > 0
    )
    print("PASS" if ok else "FAIL: no valid tool call or no token counts")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())