"""Alert clustering -> candidate cases.

Agent-facing fields live ONLY under record["input"]. Everything else in the pool record
(split, block_id, endpoints) is evaluator metadata and is stripped when dev/test files are written.
"""
import hashlib

import numpy as np
import pandas as pd

from flowadj.cases.splits import EPOCH
from flowadj.config import PROCESSED_DIR
from flowadj.data.store import DB_PATH, agent_query

PRED_PATH = PROCESSED_DIR / "detector_predictions.parquet"
ALERTS_PATH = PROCESSED_DIR / "case_alerts.parquet"
CASE_INPUT_KEYS = ["case_id", "window_start", "window_end", "n_alerts", "alert_score", "seed"]


def load_alerts(db_path=DB_PATH, pred_path=PRED_PATH) -> pd.DataFrame:
    pred = pd.read_parquet(pred_path)
    pred = pred[pred["alert"] & pred["split"].isin(["dev", "test"])]
    ids = agent_query('SELECT row_id, src_ip, dst_ip, dst_port, "timestamp" FROM flows_agent', db_path=db_path)
    return pred.merge(ids, on="row_id", how="inner")


def make_case_id(block_id, ep_a, ep_b, first_row_id) -> str:
    h = hashlib.sha1(f"{block_id}|{ep_a}|{ep_b}|{first_row_id}".encode()).hexdigest()[:8]
    return f"case_{h}"


def _ts(x) -> str:
    return pd.Timestamp(x).isoformat(sep=" ")


def build_clusters(alerts: pd.DataFrame, cc: dict, block_minutes: int, buffer_minutes: int):
    """Return (pool records, alert_map DataFrame[case_id, row_id])."""
    if cc["pad_s"] > buffer_minutes * 60:
        raise ValueError("cases.pad_s must be <= splits.buffer_minutes * 60 (windows must stay inside their block)")
    a = alerts.copy()
    swap = (a["src_ip"] < a["dst_ip"]).to_numpy()
    a["ep_a"] = np.where(swap, a["src_ip"], a["dst_ip"])
    a["ep_b"] = np.where(swap, a["dst_ip"], a["src_ip"])
    a = a.sort_values(["block_id", "ep_a", "ep_b", "timestamp", "row_id"]).reset_index(drop=True)
    gap = a.groupby(["block_id", "ep_a", "ep_b"], sort=False)["timestamp"].diff().dt.total_seconds()
    a["cluster"] = (gap.isna() | (gap > cc["cluster_gap_s"])).cumsum()

    g = a.groupby("cluster")
    agg = g.agg(
        block_id=("block_id", "first"), split=("split", "first"),
        ep_a=("ep_a", "first"), ep_b=("ep_b", "first"),
        n_alerts=("row_id", "size"), first_row=("row_id", "min"),
        t0=("timestamp", "min"), t1=("timestamp", "max"), alert_score=("alert_score", "max"),
    )
    seed_idx = g["alert_score"].idxmax()
    seed = a.loc[seed_idx.to_numpy(), ["src_ip", "dst_ip", "dst_port", "timestamp"]]
    seed.index = seed_idx.index
    seed.columns = ["seed_src", "seed_dst", "seed_port", "seed_ts"]
    agg = agg.join(seed)
    agg = agg[agg["n_alerts"] >= cc["min_alerts"]].copy()

    block_s = block_minutes * 60
    bstart = EPOCH + pd.to_timedelta(agg["block_id"] * block_s, unit="s")
    bend = bstart + pd.Timedelta(seconds=block_s)
    pad = pd.Timedelta(seconds=cc["pad_s"])
    ws = agg["t0"] - pad
    we = agg["t1"] + pad
    agg["window_start"] = ws.where(ws >= bstart, bstart)
    agg["window_end"] = we.where(we <= bend, bend)

    pool, case_ids = [], []
    for cid, r in agg.iterrows():
        case_id = make_case_id(r["block_id"], r["ep_a"], r["ep_b"], r["first_row"])
        case_ids.append(case_id)
        pool.append({
            "input": {
                "case_id": case_id,
                "window_start": _ts(r["window_start"]),
                "window_end": _ts(r["window_end"]),
                "n_alerts": int(r["n_alerts"]),
                "alert_score": round(float(r["alert_score"]), 3),
                "seed": {"src_ip": str(r["seed_src"]), "dst_ip": str(r["seed_dst"]),
                         "dst_port": int(r["seed_port"]), "timestamp": _ts(r["seed_ts"])},
            },
            "split": str(r["split"]),
            "block_id": int(r["block_id"]),
            "endpoints": [str(r["ep_a"]), str(r["ep_b"])],
        })
    assert len(set(case_ids)) == len(case_ids), "case id collision"
    amap = a.loc[a["cluster"].isin(agg.index), ["cluster", "row_id"]].copy()
    amap["case_id"] = amap["cluster"].map(dict(zip(agg.index.tolist(), case_ids)))
    return pool, amap[["case_id", "row_id"]].reset_index(drop=True)