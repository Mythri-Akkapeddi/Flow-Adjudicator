"""TCP behaviour tools (label-free). Only TCP flows (protocol 6) are considered."""
from flowadj.data.colmap import col, has
from flowadj.tools._common import TS, host_clause, one, records, window
from flowadj.tools.registry import ToolContext, tool

BASE = f"FROM flows_agent WHERE {TS} >= ? AND {TS} <= ? AND protocol = 6"
FLAGS = ["syn", "fin", "rst", "psh", "ack", "urg"]


def _failed(ctx):
    bwd, fwd, rst = col(ctx, "bwd_pkts"), col(ctx, "fwd_pkts"), col(ctx, "rst")
    return f"({bwd} = 0 OR ({rst} > 0 AND ({fwd} + {bwd}) <= 4))"


@tool
def flag_counts(ctx: ToolContext, host: str | None = None):
    """TCP flag totals and the share of TCP flows carrying each flag (SYN, FIN, RST, PSH, ACK, URG) in the case window. Optionally restricted to flows to or from one host.

    Args:
        host: optional IP address to restrict to
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    parts = ["COUNT(*) AS n_flows"]
    present = [f for f in FLAGS if has(ctx, f)]
    for f in present:
        c = col(ctx, f)
        parts.append(f"SUM({c}) AS {f}_packets")
        parts.append(f"AVG(CASE WHEN {c} > 0 THEN 1.0 ELSE 0.0 END) AS frac_{f}")
    row = one(ctx.query(f"SELECT {', '.join(parts)} {BASE} {hc}", [t0, t1] + hp))
    flags = {f: {"packets": row.get(f"{f}_packets"), "frac_flows_with_flag": row.get(f"frac_{f}")} for f in present}
    return {"host": host, "n_tcp_flows": row.get("n_flows"), "flags": flags}


@tool
def syn_ack_ratio(ctx: ToolContext, host: str | None = None):
    """SYN-to-ACK packet ratio for TCP flows, the share of flows with a SYN but no ACK, and the share of SYN flows that got no reply. Optionally restricted to one host.

    Args:
        host: optional IP address to restrict to
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    syn, ack, bwd = col(ctx, "syn"), col(ctx, "ack"), col(ctx, "bwd_pkts")
    r = one(ctx.query(
        f"SELECT COUNT(*) AS n_tcp_flows, SUM({syn}) AS syn_packets, SUM({ack}) AS ack_packets, "
        f"AVG(CASE WHEN {syn} > 0 AND {ack} = 0 THEN 1.0 ELSE 0.0 END) AS frac_syn_without_ack, "
        f"AVG(CASE WHEN {syn} > 0 AND {bwd} = 0 THEN 1.0 ELSE 0.0 END) AS frac_syn_no_reply "
        f"{BASE} {hc}", [t0, t1] + hp))
    ack_n = r.get("ack_packets") or 0
    r["syn_to_ack_ratio"] = round(r["syn_packets"] / ack_n, 4) if ack_n and r.get("syn_packets") is not None else None
    return {"host": host, **r}


@tool
def failed_connection_rate(ctx: ToolContext, host: str | None = None, port: int | None = None):
    """Share of TCP flows that look like failed connection attempts (no reply packets, or a reset after at most 4 packets), overall and for the destination ports with the most failures.

    Args:
        host: optional IP address; restrict to flows to or from this host
        port: optional destination port to restrict to
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    pc, pp = ("AND dst_port = ?", [port]) if port is not None else ("", [])
    f = _failed(ctx)
    bwd, fwd, rst = col(ctx, "bwd_pkts"), col(ctx, "fwd_pkts"), col(ctx, "rst")
    params = [t0, t1] + hp + pp
    tot = one(ctx.query(
        f"SELECT COUNT(*) AS n_tcp_flows, SUM(CASE WHEN {f} THEN 1 ELSE 0 END) AS n_failed, "
        f"AVG(CASE WHEN {bwd} = 0 THEN 1.0 ELSE 0.0 END) AS frac_no_reply, "
        f"AVG(CASE WHEN {rst} > 0 AND ({fwd} + {bwd}) <= 4 THEN 1.0 ELSE 0.0 END) AS frac_short_rst "
        f"{BASE} {hc} {pc}", params))
    n = tot.get("n_tcp_flows") or 0
    tot["failed_rate"] = round((tot.get("n_failed") or 0) / n, 4) if n else None
    ports = ctx.query(
        f"SELECT dst_port AS port, COUNT(*) AS n_flows, SUM(CASE WHEN {f} THEN 1 ELSE 0 END) AS n_failed "
        f"{BASE} {hc} {pc} GROUP BY dst_port HAVING SUM(CASE WHEN {f} THEN 1 ELSE 0 END) > 0 "
        f"ORDER BY n_failed DESC, port LIMIT 5", params)
    return {"host": host, "port": port, **tot, "top_failing_ports": records(ports)}


@tool
def rst_profile(ctx: ToolContext, host: str | None = None):
    """Flows that carry a TCP reset: how many, the share of TCP flows, average packets per such flow, and the top sources, destinations and destination ports among them.

    Args:
        host: optional IP address to restrict to
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    rst, fwd, bwd = col(ctx, "rst"), col(ctx, "fwd_pkts"), col(ctx, "bwd_pkts")
    params = [t0, t1] + hp
    tot = one(ctx.query(
        f"SELECT COUNT(*) AS n_tcp_flows, SUM(CASE WHEN {rst} > 0 THEN 1 ELSE 0 END) AS n_rst_flows, "
        f"AVG(CASE WHEN {rst} > 0 THEN {fwd} + {bwd} END) AS mean_packets_in_rst_flows {BASE} {hc}", params))
    n = tot.get("n_tcp_flows") or 0
    tot["frac_rst_flows"] = round((tot.get("n_rst_flows") or 0) / n, 4) if n else None

    def top(key, alias):
        return records(ctx.query(
            f"SELECT {key} AS {alias}, COUNT(*) AS n_rst_flows {BASE} {hc} AND {rst} > 0 "
            f"GROUP BY {key} ORDER BY n_rst_flows DESC, {alias} LIMIT 5", params))
    return {"host": host, **tot, "top_sources": top("src_ip", "ip"), "top_destinations": top("dst_ip", "ip"),
            "top_dst_ports": top("dst_port", "port")}


@tool
def short_flow_fraction(ctx: ToolContext, host: str | None = None, max_packets: int = 3):
    """Share of TCP flows with at most max_packets packets in total, and the destination ports and sources that account for most of them. Many tiny flows suggest probing or scanning.

    Args:
        host: optional IP address to restrict to
        max_packets: a flow counts as short if its total packet count is at most this value
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    mp = max(1, min(int(max_packets), 50))
    pk = f"({col(ctx, 'fwd_pkts')} + {col(ctx, 'bwd_pkts')})"
    params = [t0, t1] + hp
    tot = one(ctx.query(
        f"SELECT COUNT(*) AS n_tcp_flows, SUM(CASE WHEN {pk} <= {mp} THEN 1 ELSE 0 END) AS n_short_flows "
        f"{BASE} {hc}", params))
    n = tot.get("n_tcp_flows") or 0
    tot["frac_short_flows"] = round((tot.get("n_short_flows") or 0) / n, 4) if n else None

    def top(key, alias):
        return records(ctx.query(
            f"SELECT {key} AS {alias}, COUNT(*) AS n_short_flows {BASE} {hc} AND {pk} <= {mp} "
            f"GROUP BY {key} ORDER BY n_short_flows DESC, {alias} LIMIT 5", params))
    return {"host": host, "max_packets": mp, **tot, "top_dst_ports": top("dst_port", "port"),
            "top_sources": top("src_ip", "ip")}