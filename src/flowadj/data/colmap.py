"""Resolve logical feature names to the real column names present in flows_agent."""
from flowadj.data.store import DB_PATH, agent_columns

CANDIDATES = {
    "fwd_pkts": ["total_fwd_packets", "total_fwd_packet", "tot_fwd_pkts"],
    "bwd_pkts": ["total_backward_packets", "total_bwd_packets", "total_bwd_packet", "tot_bwd_pkts"],
    "fwd_bytes": ["total_length_of_fwd_packets", "total_length_of_fwd_packet", "totlen_fwd_pkts"],
    "bwd_bytes": ["total_length_of_bwd_packets", "total_length_of_bwd_packet", "totlen_bwd_pkts"],
    "duration": ["flow_duration"],
    "syn": ["syn_flag_count"],
    "fin": ["fin_flag_count"],
    "rst": ["rst_flag_count"],
    "psh": ["psh_flag_count"],
    "ack": ["ack_flag_count"],
    "urg": ["urg_flag_count"],
    "iat_mean": ["flow_iat_mean"],
    "iat_std": ["flow_iat_std"],
}

_CACHE: dict = {}


def _cols(db_path) -> set:
    key = str(db_path)
    if key not in _CACHE:
        _CACHE[key] = set(agent_columns(db_path))
    return _CACHE[key]


def resolve(db_path, logical: str):
    present = _cols(db_path)
    for c in CANDIDATES[logical]:
        if c in present:
            return c
    return None


def resolve_all(db_path=DB_PATH) -> dict:
    return {k: resolve(db_path, k) for k in CANDIDATES}


def has(ctx, logical: str) -> bool:
    return resolve(ctx.db_path, logical) is not None


def col(ctx, logical: str) -> str:
    """Quoted real column name for a logical feature, or KeyError if the dataset lacks it."""
    c = resolve(ctx.db_path, logical)
    if c is None:
        raise KeyError(f"this dataset has no '{logical}' feature")
    return '"' + c + '"'