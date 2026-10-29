"""Helpers shared by tool modules."""
import math

import pandas as pd

TS = '"timestamp"'


def window(ctx):
    if ctx.window_start is None or ctx.window_end is None:
        raise ValueError("no case window set on the tool context")
    return pd.Timestamp(ctx.window_start).to_pydatetime(), pd.Timestamp(ctx.window_end).to_pydatetime()


def host_clause(host):
    """SQL fragment + params restricting to flows to or from a host (or nothing)."""
    return ("AND (src_ip = ? OR dst_ip = ?)", [host, host]) if host else ("", [])


def clean(v, nd: int = 4):
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat(sep=" ")
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return int(v) if v.is_integer() and abs(v) < 1e15 else round(v, nd)
    return v


def records(df, nd: int = 4) -> list:
    return [{k: clean(v, nd) for k, v in row.items()} for row in df.to_dict("records")]


def one(df, nd: int = 4) -> dict:
    r = records(df, nd)
    return r[0] if r else {}