"""Canonical column naming for flow data."""
import re

# After snake_case conversion, these names get renamed to canonical ones.
CANONICAL_RENAMES = {
    "source_ip": "src_ip",
    "source_port": "src_port",
    "destination_ip": "dst_ip",
    "destination_port": "dst_port",
}

ID_COLS = ["flow_id", "src_ip", "dst_ip"]
INT_COLS = ["src_port", "dst_port", "protocol"]
REQUIRED_COLS = ["flow_id", "src_ip", "src_port", "dst_ip", "dst_port",
                 "protocol", "timestamp", "flow_duration", "label"]
# Everything that is not one of these is treated as a numeric feature.
NON_NUMERIC_COLS = set(ID_COLS) | {"timestamp", "label"}


def to_snake(name: str) -> str:
    """' Flow Bytes/s' -> 'flow_bytes_s'; 'Fwd Header Length.1' -> 'fwd_header_length_1'"""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return CANONICAL_RENAMES.get(s, s)