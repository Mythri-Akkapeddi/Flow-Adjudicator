"""Train a random forest (multiclass) and an isolation forest on detector-train blocks ONLY."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from flowadj.cases.splits import SPLITS_PATH
from flowadj.config import PROCESSED_DIR, get_default
from flowadj.data.store import DB_PATH, eval_query
from flowadj.detector.features import assert_clean, feature_columns, load_features, signed_log

MODEL_DIR = PROCESSED_DIR / "detector"


def _gather(mask, cols, chunk, db_path):
    ids_parts, x_parts = [], []
    n = len(mask)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        if not mask[lo:hi].any():
            continue
        ids, X = load_features(cols, lo, hi, db_path)
        keep = mask[ids]
        ids_parts.append(ids[keep])
        x_parts.append(X[keep])
    return np.concatenate(ids_parts), np.vstack(x_parts)


def train(db_path=DB_PATH) -> dict:
    t_start = time.time()
    cfg = get_default()
    det, seed = cfg["detector"], cfg["seed"]
    splits = pd.read_parquet(SPLITS_PATH).sort_values("row_id")
    split_arr = splits["split"].to_numpy(dtype=object)
    n = len(split_arr)
    assert int(splits["row_id"].iloc[-1]) == n - 1, "row_ids must be 0..n-1"
    lab = eval_query("SELECT row_id, family_corrected FROM flows_eval ORDER BY row_id",
                     db_path=db_path)["family_corrected"].to_numpy(dtype=object)
    cols = feature_columns(db_path)
    assert_clean(cols)
    print(f"{len(cols)} detector features; first 8: {cols[:8]}")

    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(split_arr == "train")
    train_lab = lab[train_idx]
    rf_parts = []
    for fam in sorted(set(train_lab.tolist())):
        if fam in ("unmatched", "unknown"):
            continue
        idx = train_idx[train_lab == fam]
        if len(idx) > det["rf_max_per_class"]:
            idx = rng.choice(idx, det["rf_max_per_class"], replace=False)
        rf_parts.append(idx)
    rf_idx = np.sort(np.concatenate(rf_parts))
    benign = rng.permutation(train_idx[train_lab == "benign"])
    fit_idx = np.sort(benign[: det["if_fit_rows"]])
    ref_idx = np.sort(benign[det["if_fit_rows"]: det["if_fit_rows"] + det["if_ref_rows"]])

    mask = np.zeros(n, dtype=bool)
    mask[rf_idx] = True
    mask[fit_idx] = True
    mask[ref_idx] = True
    print(f"gathering {int(mask.sum()):,} training rows ...")
    ids, X = _gather(mask, cols, det["chunk_rows"], db_path)

    def pick(want):
        return X[np.searchsorted(ids, want)]

    X_rf, y_rf = pick(rf_idx), lab[rf_idx]
    counts = pd.Series(y_rf).value_counts().to_dict()
    print("RF training rows per class:", counts)
    t = time.time()
    rf = RandomForestClassifier(
        n_estimators=det["rf_trees"], max_depth=18, min_samples_leaf=5,
        n_jobs=-1, random_state=seed,
    ).fit(X_rf, y_rf)
    print(f"RF trained in {time.time() - t:.0f}s")

    t = time.time()
    iso = IsolationForest(
        n_estimators=det["if_trees"], max_samples=1024, random_state=seed, n_jobs=-1,
    ).fit(signed_log(pick(fit_idx)))
    ref = -iso.score_samples(signed_log(pick(ref_idx)))  # higher = more anomalous
    thr = float(np.percentile(ref, det["if_alert_percentile"]))
    ref_q = np.quantile(ref, np.linspace(0, 1, 1001))
    print(f"IF trained in {time.time() - t:.0f}s; alert threshold (p{det['if_alert_percentile']}) = {thr:.4f}")

    importances = sorted(zip(cols, rf.feature_importances_), key=lambda x: -x[1])[:15]
    meta = {
        "features": cols,
        "classes": [str(c) for c in rf.classes_],
        "if_threshold": thr,
        "if_ref_quantiles": [float(v) for v in ref_q],
        "rf_train_counts": {str(k): int(v) for k, v in counts.items()},
        "n_if_fit": int(len(fit_idx)),
        "n_if_ref": int(len(ref_idx)),
        "top_importances": [(c, float(v)) for c, v in importances],
        "params": det,
        "seed": seed,
        "train_seconds": round(time.time() - t_start),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_DIR / "rf.joblib", compress=3)
    joblib.dump(iso, MODEL_DIR / "iforest.joblib", compress=3)
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("top RF importances:", [(c, round(v, 3)) for c, v in importances[:8]])
    return meta