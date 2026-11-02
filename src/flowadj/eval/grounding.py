"""Grounding checker: is each claim backed by the tool output it cites?

Checks that the cited call exists, succeeded, and that the claimed value appears in its output.
It checks PRESENCE of the value, not whether the claim's meaning is right.
"""
import re

import pandas as pd

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _get(x, key):
    return x.get(key) if isinstance(x, dict) else getattr(x, key)


def _leaves(obj, strs: set, nums: list):
    if isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, strs, nums)
    elif isinstance(obj, list):
        for v in obj:
            _leaves(v, strs, nums)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        nums.append(float(obj))
        if float(obj).is_integer():
            strs.add(str(int(obj)))
    elif isinstance(obj, str):
        strs.add(obj.strip())


def _num(v):
    try:
        return float(str(v).replace(",", "").strip().rstrip("%"))
    except ValueError:
        return None


def _num_match(x: float, nums: list, abs_tol: float, rel_tol: float, percent: bool) -> bool:
    tol = max(abs_tol, rel_tol * abs(x))
    if any(abs(x - n) <= tol for n in nums):
        return True
    return percent and any(abs(x - n * 100) <= max(0.1, rel_tol * abs(x)) for n in nums)


def claim_supported(claim, output: dict, time_tol_s: int = 60) -> bool:
    strs, nums = set(), []
    _leaves(output, strs, nums)
    v, t = claim.value, claim.type
    if t == "time":
        try:
            target = pd.Timestamp(str(v))
        except (ValueError, TypeError):
            return False
        for s in strs:
            if _DATE.match(s):
                try:
                    if abs((pd.Timestamp(s) - target).total_seconds()) <= time_tol_s:
                        return True
                except (ValueError, TypeError):
                    pass
        return False
    if t == "port":
        x = _num(v)
        return x is not None and _num_match(x, nums, 0.0, 0.0, False)
    if t == "count":
        x = _num(v)
        return x is not None and _num_match(x, nums, 0.5, 0.0, False)
    if t == "ratio":
        x = _num(v)
        return x is not None and _num_match(x, nums, 0.01, 0.02, True)
    # ip / other: exact string, else numeric with tolerance
    if str(v).strip() in strs:
        return True
    x = _num(v)
    return x is not None and _num_match(x, nums, 0.01, 0.02, False)


def check_report(report, tool_log) -> dict:
    """report: Report (or None). tool_log: list of ToolResult (or dicts with call_id, ok, output)."""
    calls = {_get(t, "call_id"): t for t in tool_log}
    details = []
    for i, c in enumerate(report.claims if report else []):
        if not c.tool_call_id:
            ok, why = False, "no_citation"
        elif c.tool_call_id not in calls:
            ok, why = False, "unknown_call_id"
        elif not _get(calls[c.tool_call_id], "ok"):
            ok, why = False, "tool_error"
        elif claim_supported(c, _get(calls[c.tool_call_id], "output")):
            ok, why = True, "supported"
        else:
            ok, why = False, "value_not_in_output"
        details.append({"idx": i, "supported": ok, "reason": why, "claim": c.model_dump()})
    n = len(details)
    sup = sum(d["supported"] for d in details)
    return {
        "n_claims": n, "n_supported": sup, "n_unsupported": n - sup,
        "unsupported_rate": (n - sup) / n if n else None,
        "details": details,
    }