"""Detector inputs: numeric flow features only. No IPs, no timestamps, no ephemeral source port."""
import re

import numpy as np

from flowadj.data.store import DB_PATH, agent_query

EXCLUDE = {
    "row_id", "flow_id", "src_ip", "dst_ip", "src_port", "timestamp",
    "timestamp_raw", "source_file", "capture_day", "id",
}
_NUMERIC = re.compile(r"^(U?(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT)|FLOAT|DOUBLE|REAL|DECIMAL.*)$")


def feature_columns(db_path=DB_PATH) -> list[str]:
    d = agent_query("DESCRIBE flows_agent", db_path=db_path)
    return [
        c for c, t in zip(d["column_name"], d["column_type"])
        if _NUMERIC.match(str(t).upper()) and c not in EXCLUDE and not c.endswith("_ip")
    ]


def assert_clean(cols: list[str]) -> None:
    bad = [c for c in cols if c in EXCLUDE or c.endswith("_ip") or any(w in c for w in ("label", "family", "attempt"))]
    if bad:
        raise AssertionError(f"identifier or label-like detector features: {bad}")


def load_features(cols: list[str], lo: int, hi: int, db_path=DB_PATH):
    """Rows with lo <= row_id < hi as (row_ids, float32 matrix with NaN/inf replaced by 0)."""
    sel = ", ".join('"' + c + '"' for c in cols)
    df = agent_query(
        f"SELECT row_id, {sel} FROM flows_agent WHERE row_id >= ? AND row_id < ? ORDER BY row_id",
        [lo, hi], db_path=db_path,
    )
    ids = df["row_id"].to_numpy()
    X = df[cols].to_numpy(dtype="float32", na_value=np.nan)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return ids, X


def signed_log(X: np.ndarray) -> np.ndarray:
    """Heavy-tailed counts hurt the isolation forest; trees in the RF don't care."""
    return np.sign(X) * np.log1p(np.abs(X))