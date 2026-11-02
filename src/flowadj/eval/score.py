"""Scoring of one report against hidden ground truth, plus aggregation."""
import math

import pandas as pd

NAN = float("nan")


def set_f1(pred, truth) -> float:
    p, t = set(pred), set(truth)
    if not p and not t:
        return 1.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(t)
    return 2 * prec * rec / (prec + rec)


def jaccard(pred, truth) -> float:
    p, t = set(pred), set(truth)
    if not p and not t:
        return 1.0
    return len(p & t) / len(p | t)


def _ts(x):
    if x is None:
        return None
    try:
        t = pd.Timestamp(x)
        return None if pd.isna(t) else t
    except (ValueError, TypeError):
        return None


def window_iou(pred_start, pred_end, true_start, true_end) -> float:
    ps, pe, ts, te = _ts(pred_start), _ts(pred_end), _ts(true_start), _ts(true_end)
    if None in (ps, pe, ts, te) or pe < ps or te < ts:
        return 0.0
    inter = max(pd.Timedelta(0), min(pe, te) - max(ps, ts))
    union = max(pe, te) - min(ps, ts)
    if union <= pd.Timedelta(0):
        return 1.0 if ps == ts else 0.0
    return inter / union


def mcc(y_true, y_pred) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denom == 0 else (tp * tn - fp * fn) / denom


def score_case(report, truth: dict) -> dict:
    """report: Report, dict, or None (= failed to produce a valid report)."""
    if report is not None and hasattr(report, "model_dump"):
        report = report.model_dump()
    valid = report is not None
    true_attack = truth["verdict"] == "attack"
    if valid:
        pred_attack = report["verdict"] == "attack"
    else:
        pred_attack = not true_attack          # failure counts as wrong
    row = {
        "case_id": truth["case_id"],
        "true_verdict": int(true_attack),
        "pred_verdict": int(pred_attack),
        "true_family": truth["family"],
        "pred_family": report["family"] if valid else None,
        "valid": valid,
        "verdict_correct": pred_attack == true_attack,
        "family_correct": bool(valid and report["family"] == truth["family"]),
        "attacker_f1": NAN, "victim_f1": NAN, "port_jaccard": NAN, "window_iou": NAN,
    }
    if true_attack:
        if valid:
            row["attacker_f1"] = set_f1(report["attackers"], truth["attackers"])
            row["victim_f1"] = set_f1(report["victims"], truth["victims"])
            row["window_iou"] = window_iou(report["window_start"], report["window_end"],
                                           truth["active_start"], truth["active_end"])
            if truth["ports"]:
                row["port_jaccard"] = jaccard(report["ports"], truth["ports"])
        else:
            row["attacker_f1"] = row["victim_f1"] = row["window_iou"] = 0.0
            if truth["ports"]:
                row["port_jaccard"] = 0.0
    return row


def aggregate(rows: pd.DataFrame) -> dict:
    return {
        "n": len(rows),
        "mcc": round(mcc(rows["true_verdict"].tolist(), rows["pred_verdict"].tolist()), 4),
        "verdict_acc": round(rows["verdict_correct"].mean(), 4),
        "family_acc": round(rows["family_correct"].mean(), 4),
        "attacker_f1": round(rows["attacker_f1"].mean(), 4),   # NaN rows are skipped
        "victim_f1": round(rows["victim_f1"].mean(), 4),
        "port_jaccard": round(rows["port_jaccard"].mean(), 4),
        "window_iou": round(rows["window_iou"].mean(), 4),
        "invalid_rate": round(1 - rows["valid"].mean(), 4),
    }


def per_family(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fam, g in rows.groupby("true_family"):
        out.append({"true_family": fam, **aggregate(g)})
    return pd.DataFrame(out).drop(columns=["mcc"]).set_index("true_family")