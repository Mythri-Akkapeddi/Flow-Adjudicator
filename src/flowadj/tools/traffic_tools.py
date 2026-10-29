"""Volume and endpoint tools. Label-free: they read data only through ctx.query()."""
from flowadj.data.colmap import col
from flowadj.tools._common import TS, host_clause, one, records, window
from flowadj.tools.registry import ToolContext, tool

W = f"{TS} >= ? AND {TS} <= ?"


def _pk(ctx):
    return f"({col(ctx, 'fwd_pkts')} + {col(ctx, 'bwd_pkts')})"


def _by(ctx):
    return f"({col(ctx, 'fwd_bytes')} + {col(ctx, 'bwd_bytes')})"


def _iso(t):
    return t.isoformat(sep=" ")


@tool
def case_overview(ctx: ToolContext):
    """Summary of all traffic in the case window: flow, host and port counts, volume, protocol mix, and the top sources, destinations and destination ports."""
    t0, t1 = window(ctx)
    p, pk, by = [t0, t1], _pk(ctx), _by(ctx)
    tot = ctx.query(
        f"""SELECT COUNT(*) AS n_flows, COUNT(DISTINCT src_ip) AS n_src_ips,
                   COUNT(DISTINCT dst_ip) AS n_dst_ips, COUNT(DISTINCT dst_port) AS n_dst_ports,
                   SUM({pk}) AS n_packets, SUM({by}) AS n_bytes,
                   MEDIAN({col(ctx, 'duration')}) AS median_duration_us
            FROM flows_agent WHERE {W}""", p)
    protos = ctx.query(
        f"SELECT protocol, COUNT(*) AS n_flows FROM flows_agent WHERE {W} "
        f"GROUP BY protocol ORDER BY n_flows DESC, protocol LIMIT 4", p)
    src = ctx.query(
        f"SELECT src_ip AS ip, COUNT(*) AS n_flows, SUM({pk}) AS n_packets FROM flows_agent WHERE {W} "
        f"GROUP BY src_ip ORDER BY n_flows DESC, ip LIMIT 5", p)
    dst = ctx.query(
        f"SELECT dst_ip AS ip, COUNT(*) AS n_flows, SUM({pk}) AS n_packets FROM flows_agent WHERE {W} "
        f"GROUP BY dst_ip ORDER BY n_flows DESC, ip LIMIT 5", p)
    prt = ctx.query(
        f"SELECT dst_port AS port, COUNT(*) AS n_flows, COUNT(DISTINCT src_ip) AS n_src_ips "
        f"FROM flows_agent WHERE {W} GROUP BY dst_port ORDER BY n_flows DESC, port LIMIT 5", p)
    return {
        "window": {"start": _iso(t0), "end": _iso(t1)}, **one(tot),
        "protocols": records(protos), "top_sources": records(src),
        "top_destinations": records(dst), "top_dst_ports": records(prt),
    }


@tool
def top_talkers(ctx: ToolContext, by: str = "src_ip", metric: str = "flows", n: int = 5):
    """Rank endpoints in the case window by traffic volume.

    Args:
        by: what to group by, one of src_ip, dst_ip, pair
        metric: ranking metric, one of flows, packets, bytes
        n: how many rows to return (max 20)
    """
    keys = {"src_ip": "src_ip", "dst_ip": "dst_ip", "pair": "src_ip, dst_ip"}
    mcol = {"flows": "n_flows", "packets": "n_packets", "bytes": "n_bytes"}
    if by not in keys:
        raise ValueError(f"by must be one of {sorted(keys)}")
    if metric not in mcol:
        raise ValueError(f"metric must be one of {sorted(mcol)}")
    n = max(1, min(int(n), 20))
    t0, t1 = window(ctx)
    df = ctx.query(
        f"SELECT {keys[by]}, COUNT(*) AS n_flows, SUM({_pk(ctx)}) AS n_packets, SUM({_by(ctx)}) AS n_bytes "
        f"FROM flows_agent WHERE {W} GROUP BY {keys[by]} ORDER BY {mcol[metric]} DESC, {keys[by]} LIMIT {n}",
        [t0, t1])
    total = int(ctx.query(f"SELECT COUNT(*) AS n FROM flows_agent WHERE {W}", [t0, t1])["n"][0])
    rows = records(df)
    for r in rows:
        r["share_of_flows"] = round(r["n_flows"] / total, 4) if total else None
    return {"by": by, "metric": metric, "total_flows": total, "rows": rows}


@tool
def host_summary(ctx: ToolContext, host: str):
    """Activity of one host in the case window, as a source and as a destination, with the top ports it contacted and the top ports it served.

    Args:
        host: IP address of the host
    """
    t0, t1 = window(ctx)
    pk, by = _pk(ctx), _by(ctx)
    s = ctx.query(
        f"SELECT COUNT(*) AS n_flows, COUNT(DISTINCT dst_ip) AS n_peers, COUNT(DISTINCT dst_port) AS n_dst_ports, "
        f"SUM({pk}) AS n_packets, SUM({by}) AS n_bytes FROM flows_agent WHERE {W} AND src_ip = ?", [t0, t1, host])
    d = ctx.query(
        f"SELECT COUNT(*) AS n_flows, COUNT(DISTINCT src_ip) AS n_peers, COUNT(DISTINCT dst_port) AS n_dst_ports, "
        f"SUM({pk}) AS n_packets, SUM({by}) AS n_bytes FROM flows_agent WHERE {W} AND dst_ip = ?", [t0, t1, host])
    if int(s["n_flows"][0]) == 0 and int(d["n_flows"][0]) == 0:
        return {"host": host, "note": "no flows involving this host in the case window"}
    contacted = ctx.query(
        f"SELECT dst_port AS port, COUNT(*) AS n_flows, COUNT(DISTINCT dst_ip) AS n_peers FROM flows_agent "
        f"WHERE {W} AND src_ip = ? GROUP BY dst_port ORDER BY n_flows DESC, port LIMIT 5", [t0, t1, host])
    served = ctx.query(
        f"SELECT dst_port AS port, COUNT(*) AS n_flows, COUNT(DISTINCT src_ip) AS n_peers FROM flows_agent "
        f"WHERE {W} AND dst_ip = ? GROUP BY dst_port ORDER BY n_flows DESC, port LIMIT 5", [t0, t1, host])
    return {
        "host": host, "as_source": one(s), "as_destination": one(d),
        "top_ports_contacted": records(contacted), "top_ports_served": records(served),
    }


@tool
def port_profile(ctx: ToolContext, host: str | None = None, side: str = "dst", n: int = 8):
    """Top destination ports by flow count with distinct sources and mean duration, plus how many distinct ports were touched. Optionally restricted to one host.

    Args:
        host: optional IP address to restrict to
        side: with a host, dst keeps flows going to the host and src keeps flows coming from it
        n: how many ports to return (max 20)
    """
    t0, t1 = window(ctx)
    n = max(1, min(int(n), 20))
    cond, params = "", [t0, t1]
    if host:
        if side not in ("dst", "src"):
            raise ValueError("side must be dst or src")
        cond = f"AND {'dst_ip' if side == 'dst' else 'src_ip'} = ?"
        params.append(host)
    base = f"FROM flows_agent WHERE {W} {cond}"
    summ = one(ctx.query(f"SELECT COUNT(*) AS n_flows, COUNT(DISTINCT dst_port) AS n_distinct_ports {base}", params))
    top = ctx.query(
        f"SELECT dst_port AS port, COUNT(*) AS n_flows, COUNT(DISTINCT src_ip) AS n_src_ips, "
        f"COUNT(DISTINCT dst_ip) AS n_dst_ips, AVG({col(ctx, 'duration')}) AS mean_duration_us "
        f"{base} GROUP BY dst_port ORDER BY n_flows DESC, port LIMIT {n}", params)
    rows = records(top)
    total = summ.get("n_flows") or 0
    for r in rows:
        r["share_of_flows"] = round(r["n_flows"] / total, 4) if total else None
    return {"host": host, "side": side if host else None, **summ, "ports": rows}


@tool
def flow_size_distribution(ctx: ToolContext, host: str | None = None):
    """Distribution (10th, 50th, 90th, 99th percentile) of packets, bytes and duration per flow in the case window, plus the share of tiny flows (at most 2 packets). Optionally restricted to flows to or from one host.

    Args:
        host: optional IP address to restrict to
    """
    t0, t1 = window(ctx)
    hc, hp = host_clause(host)
    pk, by, dur = _pk(ctx), _by(ctx), col(ctx, "duration")
    q = (
        f"SELECT COUNT(*) AS n_flows, "
        f"quantile_cont({pk}, 0.1) AS packets_p10, quantile_cont({pk}, 0.5) AS packets_p50, "
        f"quantile_cont({pk}, 0.9) AS packets_p90, quantile_cont({pk}, 0.99) AS packets_p99, "
        f"quantile_cont({by}, 0.1) AS bytes_p10, quantile_cont({by}, 0.5) AS bytes_p50, "
        f"quantile_cont({by}, 0.9) AS bytes_p90, quantile_cont({by}, 0.99) AS bytes_p99, "
        f"quantile_cont({dur}, 0.5) AS duration_us_p50, quantile_cont({dur}, 0.9) AS duration_us_p90, "
        f"AVG(CASE WHEN {pk} <= 2 THEN 1.0 ELSE 0.0 END) AS frac_flows_le_2_packets "
        f"FROM flows_agent WHERE {W} {hc}"
    )
    return {"host": host, **one(ctx.query(q, [t0, t1] + hp))}