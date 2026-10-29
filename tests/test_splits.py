import numpy as np
import pandas as pd

from flowadj.cases.splits import (
    add_block_columns, apply_buffers, assign_block_splits, block_families, min_gap_seconds,
)


def _df():
    ts = pd.date_range("2017-07-07 09:00:00", "2017-07-07 12:00:00", freq="5s")
    attack = (ts >= pd.Timestamp("2017-07-07 10:00:00")) & (ts < pd.Timestamp("2017-07-07 10:40:00"))
    return pd.DataFrame({"row_id": np.arange(len(ts)), "timestamp": ts,
                         "family_corrected": np.where(attack, "dos", "benign")})


def _run(seed=1):
    df = add_block_columns(_df(), 10)
    blocks = block_families(df, 10)
    blocks["split"] = blocks["block_id"].map(assign_block_splits(blocks, 0.2, 0.35, seed))
    return df, blocks, apply_buffers(df, blocks, 10, 2)


def test_buffers_separate_splits():
    df, _, flows = _run()
    assert {"train", "dev", "test", "buffer"} <= set(flows["split"])
    assert min_gap_seconds(df["timestamp"], flows["split"]) >= 120


def test_attack_family_appears_in_every_split():
    _, blocks, _ = _run()
    s = set(blocks.loc[blocks["block_family"] == "dos", "split"])
    assert s == {"train", "dev", "test"}


def test_deterministic():
    _, b1, _ = _run(seed=1)
    _, b2, _ = _run(seed=1)
    assert b1["split"].tolist() == b2["split"].tolist()