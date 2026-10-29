"""Score every flow, define alerts, write detector_predictions.parquet."""
import json

import joblib
import numpy as np
import pandas as pd

from flowadj.cases.splits import SPLITS_PATH
from flowadj.config import PROCESSED_DIR, get_default
from flowadj.data.store import DB_PATH, eval_query
from flowadj.detector.features import load_features, signed_log
from flowadj.detector.train import MODEL_DIR

PRED_PATH = PROCESSED_DIR / "detector_predictions.parquet"


def run_predict(db_path=DB_PATH) -> pd.DataFrame:
    det = get_default()["detector"]
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    rf = joblib.load(MODEL_DIR / "rf.joblib")
    iso = joblib.load(MODEL_DIR / "iforest.joblib")
    cols = meta["features"]
    classes = list(rf.classes_)
    b_i = classes.index("benign")
    cls_arr = np.array(classes, dtype=object)
    q = np.array(meta["if_ref_quantiles"], dtype="float64")
    qs = np.linspace(0.0, 1.0, len(q))
    n = int(eval_query("SELECT COUNT(*) AS n FROM flows_eval", db_path=db_path)["n"][0])

    parts = []
    for lo in range(0, n, det["chunk_rows"]):
        hi = min(lo + det["chunk_rows"], n)
        ids, X = load_features(cols, lo, hi, db_path)
        P = rf.predict_proba(X)
        s = -iso.score_samples(signed_log(X))
        parts.append(pd.DataFrame({
            "row_id": ids,
            "rf_class": cls_arr[P.argmax(1)],
            "rf_p_attack": (1.0 - P[:, b_i]).astype("float32"),
            "rf_p_max": P.max(1).astype("float32"),
            "if_score": s.astype("float32"),
            "if_pct": np.interp(s, q, qs).astype("float32"),
        }))
        print(f"  scored {hi:,}/{n:,}")
    pred = pd.concat(parts, ignore_index=True)
    pred["alert_rf"] = pred["rf_p_attack"] >= det["rf_alert_threshold"]
    pred["alert_if"] = pred["if_score"] >= meta["if_threshold"]
    pred["alert"] = pred["alert_rf"] | pred["alert_if"]
    pred["alert_score"] = np.maximum(pred["rf_p_attack"], pred["if_pct"]).astype("float32")
    splits = pd.read_parquet(SPLITS_PATH)
    pred = pred.merge(splits[["row_id", "block_id", "split"]], on="row_id", how="left")
    pred.to_parquet(PRED_PATH, index=False)
    print(f"Wrote {PRED_PATH} ({len(pred):,} rows)")
    print(pred.groupby("split")[["alert_rf", "alert_if", "alert"]].mean().round(4))
    return pred