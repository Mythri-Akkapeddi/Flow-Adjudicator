"""Time-structure tools (label-free)."""
from datetime import timedelta

import numpy as np
import pandas as pd

from flowadj.data.colmap import col, has
from flowadj.tools._common import TS, clean, host_clause, one, window
from flowadj.tools.registry import ToolContext, tool

W = f"{TS} >= ? AND {TS} <= ?"


@tool
def burst_profile(ctx: ToolContext, host: str | None = None, bin_seconds: int = 10):
    """How flow counts are distributed over time in the case window: busiest bins, peak versus median bin, and how many bins were active. Optionally restricted to one host.

    Args:
        host: optional IP address to restrict to
        bin_seconds: width of each time bin in seconds
    """
    t0, t1 = window(ctx)
    b = max(1, int(bin_seconds))
    hc, hp = host_clause(host)
    df = ctx.query(
        f"SELECT CAST(floor(epoch({TS}) / {b}) AS BIGINT) AS bin, COUNT(*) AS n FROM flows_agent "
        f"WHERE {W} {hc} GROUP BY 1 ORDER BY 1", [t0, t1] + hp)
    if df.empty:
        return {"n_flows": 0, "note": "no flows in window"}
    n_bins = max(1, int(np.ceil((t1 - t0).total_seconds() / b)))
    counts = df["n"].to_numpy()
    padded = np.concatenate([counts, np.zeros(max(0, n_bins - len(counts)))])
    med = float(np.median(padded))
    act_med = float(np.median(counts))
    peak_i = int(np.argmax(counts))
    start = lambda v: clean(pd.Timestamp(int(v) * b, unit="s")) # noqa: E731
    top = df.sort_values(["n", "bin"], ascending=[False, True]).head(5)
    return {
        "host": host, "bin_seconds": b, "n_flows": int(counts.sum()),
        "n_bins_in_window": n_bins, "n_active_bins": int(len(counts)),
        "median_flows_per_bin": med, "peak_flows_per_bin": int(counts[peak_i]),
        "peak_bin_start": start(df["bin"].iloc[peak_i]),
        "peak_to_median_ratio": round(float(counts[peak_i]) / med, 2) if med > 0 else None,
        "peak_to_median_active_ratio": round(float(counts[peak_i]) / act_med, 2) if act_med > 0 else None,
        "first_active_bin": start(df["bin"].iloc[0]), "last_active_bin": start(df["bin"].iloc[-1]),
        "busiest_bins": [{"start": start(r.bin), "n_flows": int(r.n)} for r in top.itertuples()],
    }


@tool
def beacon_periodicity(ctx: ToolContext, src_ip: str, dst_ip: str, dst_port: int | None = None):
    """Regularity of flow start times from one source to one destination (optionally one destination port): gap statistics and coefficient of variation. A very low coefficient of variation means timer-like, highly regular traffic.

    Args:
        src_ip: source IP address
        dst_ip: destination IP address
        dst_port: optional destination port
    """
    t0, t1 = window(ctx)
    pc, pp = ("AND dst_port = ?", [dst_port]) if dst_port is not None else ("", [])
    df = ctx.query(
        f"SELECT {TS} AS ts FROM flows_agent WHERE {W} AND src_ip = ? AND dst_ip = ? {pc} ORDER BY {TS}",
        [t0, t1, src_ip, dst_ip] + pp)
    n = len(df)
    if n < 4:
        return {"n_flows": n, "note": "fewer than 4 flows; periodicity not assessable"}
    gaps = df["ts"].diff().dt.total_seconds().dropna().to_numpy()
    mean, std, med = float(gaps.mean()), float(gaps.std(ddof=1)), float(np.median(gaps))
    return {
        "src_ip": src_ip, "dst_ip": dst_ip, "dst_port": dst_port, "n_flows": n,
        "mean_gap_s": round(mean, 3), "std_gap_s": round(std, 3), "median_gap_s": round(med, 3),
        "min_gap_s": round(float(gaps.min()), 3), "max_gap_s": round(float(gaps.max()), 3),
        "cv": round(std / mean, 4) if mean > 0 else None,
        "frac_gaps_within_10pct_of_median": round(float(np.mean(np.abs(gaps - med) <= 0.1 * med)), 4) if med > 0 else None,
    }


@tool
def interarrival_stats(ctx: ToolContext, host: str | None = None):
    """Statistics of the gaps between consecutive flow start times in the case window (mean, spread, median, 95th percentile), plus the median per-flow inter-arrival time when available. Optionally restricted to one host.

    Args:
        host: optional IP address to restrict to
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    r = one(ctx.query(
        f"""WITH t AS (SELECT {TS} AS ts FROM flows_agent WHERE {W} {hc}),
                 g AS (SELECT epoch(ts) - epoch(LAG(ts) OVER (ORDER BY ts)) AS gap FROM t)
            SELECT COUNT(gap) AS n_gaps, AVG(gap) AS mean_gap_s, STDDEV_SAMP(gap) AS std_gap_s,
                   MEDIAN(gap) AS median_gap_s, quantile_cont(gap, 0.95) AS p95_gap_s,
                   MIN(gap) AS min_gap_s, MAX(gap) AS max_gap_s
            FROM g WHERE gap IS NOT NULL""", [t0, t1] + hp), nd=6)
    if r.get("mean_gap_s"):
        r["cv_gap"] = round((r.get("std_gap_s") or 0) / r["mean_gap_s"], 4)
    extra = []
    if has(ctx, "iat_mean"):
        extra.append(f"MEDIAN({col(ctx, 'iat_mean')}) AS median_flow_iat_mean_us")
    if has(ctx, "iat_std"):
        extra.append(f"MEDIAN({col(ctx, 'iat_std')}) AS median_flow_iat_std_us")
    if extra:
        r.update(one(ctx.query(f"SELECT {', '.join(extra)} FROM flows_agent WHERE {W} {hc}", [t0, t1] + hp)))
    return {"host": host, **r}


@tool
def compare_to_host_baseline(ctx: ToolContext, host: str, baseline_minutes: int = 60):
    """Compare one host's activity in the case window with its own activity in the period just before the window (flows per minute, peers, ports, packets per flow, bytes per minute).

    Args:
        host: IP address of the host
        baseline_minutes: length of the baseline period before the window (5 to 240)
    """
    t0, t1 = window(ctx)
    bm = max(5, min(int(baseline_minutes), 240))
    pk = f"({col(ctx, 'fwd_pkts')} + {col(ctx, 'bwd_pkts')})"
    by = f"({col(ctx, 'fwd_bytes')} + {col(ctx, 'bwd_bytes')})"

    def stats(a, b, minutes):
        r = one(ctx.query(
            f"SELECT COUNT(*) AS n_flows, COUNT(DISTINCT CASE WHEN src_ip = ? THEN dst_ip ELSE src_ip END) AS n_peers, "
            f"COUNT(DISTINCT dst_port) AS n_dst_ports, AVG({pk}) AS mean_packets_per_flow, SUM({by}) AS n_bytes "
            f"FROM flows_agent WHERE {TS} >= ? AND {TS} <= ? AND (src_ip = ? OR dst_ip = ?)",
            [host, a, b, host, host]))
        r["flows_per_min"] = round(r["n_flows"] / minutes, 4)
        r["bytes_per_min"] = round((r.get("n_bytes") or 0) / minutes, 1)
        return r

    wm = max((t1 - t0).total_seconds() / 60.0, 1e-6)
    w = stats(t0, t1, wm)
    bl = stats(t0 - timedelta(minutes=bm), t0 - timedelta(microseconds=1), bm)
    ratios = None
    if bl["n_flows"] > 0:
        ratios = {
            "flows_per_min": round(w["flows_per_min"] / bl["flows_per_min"], 2),
            "n_peers": round(w["n_peers"] / bl["n_peers"], 2),
            "n_dst_ports": round(w["n_dst_ports"] / bl["n_dst_ports"], 2),
        }
    return {
        "host": host, "baseline_minutes": bm, "window": w, "baseline": bl,
        "ratios_window_over_baseline": ratios,
        "note": None if bl["n_flows"] else "no baseline flows for this host before the window; cannot judge whether this is unusual",
    }