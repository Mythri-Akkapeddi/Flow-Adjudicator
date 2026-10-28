"""Tool framework: decorator registration, JSON-schema generation, per-agent allow-lists,
call IDs, argument validation, row/size caps, and call logging."""
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

import pandas as pd

from flowadj.data.store import DB_PATH, agent_query
from flowadj.llm.budget import Budget


@dataclass
class ToolContext:
    """Passed as the first argument to every tool. Tools read data ONLY via ctx.query()."""
    db_path: Path = DB_PATH
    window_start: Any = None
    window_end: Any = None

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return agent_query(sql, params, db_path=self.db_path)


# ---------- schema generation ----------

def _schema_for(hint) -> dict:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        return _schema_for(args[0])
    if origin in (list, typing.List):
        args = get_args(hint)
        return {"type": "array", "items": _schema_for(args[0]) if args else {}}
    if hint is bool:
        return {"type": "boolean"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is str:
        return {"type": "string"}
    raise TypeError(f"Unsupported tool parameter type: {hint}")


def _parse_doc(doc: str | None) -> tuple[str, dict]:
    doc = inspect.cleandoc(doc or "")
    parts = re.split(r"\n\s*Args:\s*\n", doc, maxsplit=1)
    desc = " ".join(parts[0].split())
    args = {}
    if len(parts) > 1:
        for line in parts[1].splitlines():
            m = re.match(r"\s*(\w+)\s*:\s*(.+)", line)
            if m:
                args[m.group(1)] = m.group(2).strip()
    return desc, args


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    properties: dict
    required: list

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                },
            },
        }


def build_spec(fn: Callable) -> ToolSpec:
    params = list(inspect.signature(fn).parameters.values())
    if not params or params[0].name != "ctx":
        raise ValueError(f"{fn.__name__}: first parameter must be named 'ctx'")
    hints = get_type_hints(fn)
    desc, arg_docs = _parse_doc(fn.__doc__)
    if not desc:
        raise ValueError(f"{fn.__name__}: tools need a docstring (it is the LLM's description)")
    props, required = {}, []
    for p in params[1:]:
        if p.name not in hints:
            raise ValueError(f"{fn.__name__}: parameter '{p.name}' needs a type hint")
        s = _schema_for(hints[p.name])
        if p.name in arg_docs:
            s["description"] = arg_docs[p.name]
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        elif p.default is not None:
            s["default"] = p.default
        props[p.name] = s
    return ToolSpec(fn.__name__, fn, desc, props, required)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, fn: Callable) -> Callable:
        spec = build_spec(fn)
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = spec
        return fn

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)


registry = ToolRegistry()  # global registry; tools modules decorate with @tool
tool = registry.tool


# ---------- running tools ----------

def _coerce(value, schema: dict):
    t = schema.get("type")
    if t == "integer":
        if isinstance(value, bool):
            raise ValueError("expected integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("expected integer")
        return int(value)
    if t == "number":
        if isinstance(value, bool):
            raise ValueError("expected number")
        return float(value)
    if t == "boolean":
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise ValueError("expected boolean")
        return bool(value)
    if t == "string":
        return str(value)
    if t == "array":
        if not isinstance(value, list):
            raise ValueError("expected list")
        return [_coerce(v, schema.get("items", {})) for v in value]
    return value


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def _cap_lists(obj, max_rows: int, truncated: dict, path: str = ""):
    if isinstance(obj, list):
        if len(obj) > max_rows:
            truncated[path or "result"] = len(obj)
            obj = obj[:max_rows]
        return [_cap_lists(x, max_rows, truncated, path + "[]") for x in obj]
    if isinstance(obj, dict):
        return {k: _cap_lists(v, max_rows, truncated, f"{path}.{k}" if path else k) for k, v in obj.items()}
    return obj


@dataclass
class ToolResult:
    call_id: str
    name: str
    arguments: dict
    ok: bool
    output: dict

    def to_message(self) -> str:
        """The string handed back to the LLM (call_id included so it can cite it)."""
        return json.dumps({"call_id": self.call_id, "tool": self.name, "ok": self.ok, "result": self.output})


class ToolRunner:
    def __init__(
        self,
        ctx: ToolContext,
        allowed: set[str] | None = None,
        reg: ToolRegistry = registry,
        budget: Budget | None = None,
        max_rows: int = 20,
        max_chars: int = 6000,
    ):
        self.ctx, self.allowed, self.reg = ctx, allowed, reg
        self.budget, self.max_rows, self.max_chars = budget, max_rows, max_chars
        self.log: list[ToolResult] = []
        self._n = 0

    def available(self) -> list[str]:
        return [n for n in self.reg.names() if self.allowed is None or n in self.allowed]

    def schemas(self) -> list[dict]:
        return [self.reg.get(n).schema() for n in self.available()]

    def _fail(self, call_id, name, args, msg) -> ToolResult:
        res = ToolResult(call_id, name, args, False, {"error": msg})
        self.log.append(res)
        return res

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        arguments = dict(arguments or {})
        self._n += 1
        call_id = f"tc_{self._n:03d}"
        if self.budget is not None:
            self.budget.record_tool()  # raises BudgetExceeded when the cap is reached

        spec = self.reg.get(name)
        if spec is None or (self.allowed is not None and name not in self.allowed):
            return self._fail(call_id, name, arguments, f"unknown or not-allowed tool '{name}'. Available: {self.available()}")
        unknown = sorted(set(arguments) - set(spec.properties))
        if unknown:
            return self._fail(call_id, name, arguments, f"unknown argument(s) {unknown}. Valid: {sorted(spec.properties)}")
        missing = [r for r in spec.required if r not in arguments]
        if missing:
            return self._fail(call_id, name, arguments, f"missing required argument(s): {missing}")
        try:
            kwargs = {k: _coerce(v, spec.properties[k]) for k, v in arguments.items()}
        except (ValueError, TypeError) as e:
            return self._fail(call_id, name, arguments, f"bad argument value: {e}")

        try:
            raw = spec.fn(self.ctx, **kwargs)
            out = json.loads(json.dumps(raw, default=_default))
        except Exception as e:  # tool bugs / forbidden queries become error results, not crashes
            return self._fail(call_id, name, arguments, f"{type(e).__name__}: {e}")

        if not isinstance(out, dict):
            out = {"result": out}
        truncated: dict = {}
        out = _cap_lists(out, self.max_rows, truncated)
        if truncated:
            out["_truncated"] = {"lists_cut_to_max_rows": self.max_rows, "original_lengths": truncated}
        size = len(json.dumps(out))
        if size > self.max_chars:
            return self._fail(call_id, name, arguments,
                              f"output too large ({size} chars > {self.max_chars}); narrow the query")
        res = ToolResult(call_id, name, arguments, True, out)
        self.log.append(res)
        return res