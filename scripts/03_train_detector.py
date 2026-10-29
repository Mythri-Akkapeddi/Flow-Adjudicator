"""split assignment -> detector training -> predictions.

    uv run python -m scripts.03_train_detector                 # all stages
    uv run python -m scripts.03_train_detector --stage splits  # or: train, predict
"""
import argparse

import pandas as pd

from flowadj.cases.splits import build_splits
from flowadj.config import get_default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "splits", "train", "predict"], default="all")
    stage = ap.parse_args().stage
    pd.set_option("display.width", 200)

    if stage in ("all", "splits"):
        flows, blocks, summary, gap = build_splits()
        print("\nFlows per family x split (corrected labels):")
        print(summary.to_string())
        print("\nBlocks per family x split:")
        print(pd.crosstab(blocks["block_family"], blocks["split"]).to_string())
        buf = get_default()["splits"]["buffer_minutes"] * 60
        print(f"\nmin gap between flows of different splits: {gap:.0f}s (must be >= {buf}s)")
        assert gap >= buf, "split buffers are broken"
    if stage in ("all", "train"):
        from flowadj.detector.train import train
        train()
    if stage in ("all", "predict"):
        from flowadj.detector.predict import run_predict
        run_predict()


if __name__ == "__main__":
    main()