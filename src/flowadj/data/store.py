"""DuckDB store: a label-free `flows_agent` view and an evaluator-only `flows_eval` view.

Rule: tools and agents ONLY use agent_query(). It refuses SQL that mentions anything
label-related, and it opens the database read-only.
"""
import re
from pathlib import Path

import duckdb
import pandas as pd

from flowadj.config import PROCESSED_DIR

DB_PATH = PROCESSED_DIR / "flows.duckdb"

# Columns that must never be visible to an agent.
# source_file would leak the attack name (e.g. "Friday-...-PortScan.pcap_ISCX.csv").
HIDDEN_FROM_AGENT = {
    "label_original", "family_original", "label_corrected", "family_corrected",
    "attempted", "attempted_category", "timestamp_raw", "source_file",
    "id", "capture_day",
}
EVAL_EXTRA_COLS = [
    "label_original", "family_original", "label_corrected", "family_corrected", "attempted",
]
EVAL_ID_COLS = ["row_id", "flow_id", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "timestamp"]

_FORBIDDEN_SQL = re.compile(
    r"flows_all|flows_eval|label|family|attempted|source_file|timestamp_raw", re.IGNORECASE
)

_CONNS: dict = {}


def _conn(db_path, read_only: bool = True):
    key = (str(db_path), read_only)
    if key not in _CONNS:
        _CONNS[key] = duckdb.connect(str(db_path), read_only=read_only)
    return _CONNS[key]


def close_all() -> None:
    for c in _CONNS.values():
        try:
            c.close()
        except Exception:
            pass
    _CONNS.clear()


def _q(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def build_store(flows_parquet: Path, db_path: Path = DB_PATH) -> dict:
    """Create the database with table flows_all and the two views."""
    close_all()
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    p = str(flows_parquet).replace("'", "''")
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE TABLE flows_all AS SELECT * FROM read_parquet('{p}')")
        cols = [r[0] for r in con.execute("DESCRIBE flows_all").fetchall()]
        agent_cols = [c for c in cols if c not in HIDDEN_FROM_AGENT]
        eval_cols = [c for c in EVAL_ID_COLS + EVAL_EXTRA_COLS if c in cols]
        con.execute(f"CREATE VIEW flows_agent AS SELECT {', '.join(map(_q, agent_cols))} FROM flows_all")
        con.execute(f"CREATE VIEW flows_eval AS SELECT {', '.join(map(_q, eval_cols))} FROM flows_all")
        n = con.execute("SELECT COUNT(*) FROM flows_all").fetchone()[0]
    finally:
        con.close()
    return {"rows": n, "agent_cols": agent_cols, "eval_cols": eval_cols}


def agent_query(sql: str, params: list | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """The ONLY way tools may read data."""
    if _FORBIDDEN_SQL.search(sql):
        raise PermissionError("Query references a forbidden table or column.")
    return _conn(db_path, True).cursor().execute(sql, params or []).df()


def eval_query(sql: str, params: list | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Evaluator / ground-truth builder only. Never import this from tools/ or agents/."""
    return _conn(db_path, True).cursor().execute(sql, params or []).df()


def agent_columns(db_path: Path = DB_PATH) -> list[str]:
    cur = _conn(db_path, True).cursor()
    return [r[0] for r in cur.execute("DESCRIBE flows_agent").fetchall()]


def eval_columns(db_path: Path = DB_PATH) -> list[str]:
    cur = _conn(db_path, True).cursor()
    return [r[0] for r in cur.execute("DESCRIBE flows_eval").fetchall()]