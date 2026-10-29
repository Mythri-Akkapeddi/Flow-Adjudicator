"""Time-block split assignment (detector-train / dev / test) with buffer gaps.

Evaluator-side module: it uses corrected labels ONLY to stratify blocks so that every attack
family appears in every split. Agents never see any of this.
"""
import numpy as np
import pandas as pd

from flowadj.config import PROCESSED_DIR, get_default
from flowadj.data.store import DB_PATH, eval_query

SPLITS_PATH = PROCESSED_DIR / "splits.parquet"
BLOCKS_PATH = PROCESSED_DIR / "blocks.parquet"
EPOCH = pd.Timestamp("2017-01-01")
NOT_ATTACK = {"benign", "unmatched", "unknown"}


def add_block_columns(df: pd.DataFrame, block_minutes: int) -> pd.DataFrame:
    block_s = block_minutes * 60
    secs = (df["timestamp"] - EPOCH).dt.total_seconds()
    out = df.copy()
    out["block_id"] = (secs // block_s).astype("int64")
    out["offset_s"] = secs - out["block_id"] * block_s
    return out


def block_families(df: pd.DataFrame, min_attack_flows: int) -> pd.DataFrame:
    """One row per block with its dominant attack family (or 'benign')."""
    atk = df[~df["family_corrected"].isin(NOT_ATTACK)]
    counts = atk.groupby(["block_id", "family_corrected"]).size().rename("n").reset_index()
    counts = counts.sort_values(["block_id", "n", "family_corrected"], ascending=[True, False, True])
    top = counts.drop_duplicates("block_id")
    top = top[top["n"] >= min_attack_flows]
    fam = dict(zip(top["block_id"].tolist(), top["family_corrected"].tolist()))
    ids = np.sort(df["block_id"].unique())
    blocks = pd.DataFrame({"block_id": ids})
    blocks["block_family"] = [fam.get(int(b), "benign") for b in ids]
    blocks["n_flows"] = df.groupby("block_id").size().reindex(ids).to_numpy()
    return blocks


def assign_block_splits(blocks: pd.DataFrame, dev_fraction: float, test_fraction: float, seed: int) -> dict:
    """Stratified by block_family: every family with >= 3 blocks gets blocks in all three splits."""
    rng = np.random.default_rng(seed)
    out = {}
    for fam in sorted(blocks["block_family"].unique()):
        ids = sorted(blocks.loc[blocks["block_family"] == fam, "block_id"].astype(int).tolist())
        ids = [ids[i] for i in rng.permutation(len(ids))]
        n = len(ids)
        if n == 1:
            plan = ["train"]
        elif n == 2:
            plan = ["train", "test"]
        else:
            n_dev = max(1, round(n * dev_fraction))
            n_test = max(1, round(n * test_fraction))
            plan = ["dev"] * n_dev + ["test"] * n_test
            plan += ["train"] * (n - len(plan))
        for b, s in zip(ids, plan):
            out[b] = s
    return out


def apply_buffers(df: pd.DataFrame, blocks: pd.DataFrame, block_minutes: int, buffer_minutes: int) -> pd.DataFrame:
    """Per-flow split. Flows near a boundary with a block of a DIFFERENT split become 'buffer'."""
    block_s, buf_s = block_minutes * 60, buffer_minutes * 60
    m = dict(zip(blocks["block_id"].astype(int).tolist(), blocks["split"].tolist()))
    out = df[["row_id", "block_id", "offset_s"]].copy()
    own = out["block_id"].map(m)
    prev = (out["block_id"] - 1).map(m)
    nxt = (out["block_id"] + 1).map(m)
    near_start = prev.notna() & (prev != own) & (out["offset_s"] < buf_s)
    near_end = nxt.notna() & (nxt != own) & (out["offset_s"] >= block_s - buf_s)
    out["split"] = own.where(~(near_start | near_end), "buffer")
    return out[["row_id", "block_id", "split"]]


def min_gap_seconds(ts: pd.Series, split: pd.Series) -> float:
    """Smallest time gap between two consecutive non-buffer flows that belong to different splits."""
    d = pd.DataFrame({"ts": ts.to_numpy(), "split": split.to_numpy()})
    d = d[d["split"] != "buffer"].sort_values("ts", kind="stable")
    change = d["split"] != d["split"].shift()
    gaps = d["ts"].diff().dt.total_seconds()[change].dropna()
    return float(gaps.min()) if len(gaps) else float("inf")


def build_splits(db_path=DB_PATH, write: bool = True):
    cfg = get_default()
    sp = cfg["splits"]
    df = eval_query("SELECT row_id, timestamp, family_corrected FROM flows_eval", db_path=db_path)
    df = df.sort_values("row_id").reset_index(drop=True)
    df = add_block_columns(df, sp["block_minutes"])
    blocks = block_families(df, sp["min_attack_flows_per_block"])
    mapping = assign_block_splits(blocks, sp["dev_fraction"], sp["test_fraction"], cfg["seed"])
    blocks["split"] = blocks["block_id"].map(mapping)
    blocks["start"] = EPOCH + pd.to_timedelta(blocks["block_id"] * sp["block_minutes"] * 60, unit="s")
    flows = apply_buffers(df, blocks, sp["block_minutes"], sp["buffer_minutes"])
    gap = min_gap_seconds(df["timestamp"], flows["split"])
    summary = pd.crosstab(df["family_corrected"], flows["split"])
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        flows.to_parquet(SPLITS_PATH, index=False)
        blocks.to_parquet(BLOCKS_PATH, index=False)
    return flows, blocks, summary, gap