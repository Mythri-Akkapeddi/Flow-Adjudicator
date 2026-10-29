"""Hidden ground truth per case. EVALUATOR-SIDE ONLY: never import this from tools/ or agents/."""
import pandas as pd

from flowadj.data.attack_schedule import load_schedule, tolerance_minutes
from flowadj.data.store import DB_PATH, eval_query

NOT_ATTACK = {"benign", "unmatched", "unknown"}
ELSEWHERE_MIN_FLOWS = 20


def _top(s: pd.Series, min_share: float, max_n: int) -> list:
    vc = s.value_counts()
    keep = vc[vc / vc.sum() >= min_share].head(max_n)
    return sorted(keep.index.tolist())


def _iso(t):
    return pd.Timestamp(t).isoformat(sep=" ")


def truth_for_case(case_id, win: pd.DataFrame, alert_ids: set, endpoints, label_col: str, cc: dict) -> dict:
    """win: flows in the case window with row_id, src_ip, dst_ip, dst_port, timestamp, <label_col>."""
    alerts = win[win["row_id"].isin(alert_ids)]
    if len(alerts) == 0:
        raise ValueError(f"{case_id}: no alert flows found in window")
    atk = ~alerts[label_col].isin(NOT_ATTACK)
    share = float(atk.mean())
    verdict = "attack" if share >= 0.5 else "benign"
    tol = pd.Timedelta(minutes=tolerance_minutes())
    sched = load_schedule()
    out = {
        "case_id": case_id, "verdict": verdict, "family": "benign",
        "attackers": [], "victims": [], "ports": [], "n_distinct_ports": 0,
        "active_start": None, "active_end": None,
        "purity": round(share if verdict == "attack" else 1 - share, 4),
        "n_alert_flows": int(len(alerts)), "n_attack_alert_flows": int(atk.sum()),
        "flags": [], "label_col": label_col,
    }

    if verdict == "benign":
        t0, t1 = alerts["timestamp"].min(), alerts["timestamp"].max()
        if any(a.start - tol <= t1 and a.end + tol >= t0 for a in sched):
            out["flags"].append("benign_during_scheduled_attack")
        if int((~win[label_col].isin(NOT_ATTACK)).sum()) >= ELSEWHERE_MIN_FLOWS:
            out["flags"].append("attack_traffic_elsewhere_in_window")
        return out

    fam = alerts.loc[atk, label_col].value_counts().idxmax()
    eps = set(endpoints)
    rel = win[(win[label_col] == fam) & (win["src_ip"].isin(eps) | win["dst_ip"].isin(eps))]
    if rel.empty:
        rel = alerts[atk]
    thr, mx = cc["share_threshold"], cc["max_entities"]
    attackers = _top(rel["src_ip"], thr, mx)
    victims = _top(rel["dst_ip"], thr, mx)
    n_ports = int(rel["dst_port"].nunique())
    out.update(
        family=fam, attackers=attackers, victims=victims, n_distinct_ports=n_ports,
        ports=[] if n_ports > cc["many_ports"] else [int(p) for p in _top(rel["dst_port"], thr, mx)],
        active_start=_iso(rel["timestamp"].min()), active_end=_iso(rel["timestamp"].max()),
    )
    if set(attackers) & set(victims):
        out["flags"].append("attacker_victim_overlap")

    mid = rel["timestamp"].min() + (rel["timestamp"].max() - rel["timestamp"].min()) / 2
    active = [a for a in sched if a.start - tol <= mid <= a.end + tol]
    if not active:
        out["flags"].append("outside_schedule")
    elif fam not in {a.family for a in active}:
        out["flags"].append("family_mismatch_schedule")
    elif not (set(attackers) & {x for a in active if a.family == fam for x in a.attackers}):
        out["flags"].append("attacker_mismatch_schedule")
    return out


def build_truth(pool: list, alert_map: pd.DataFrame, label_col: str, source: str, cc: dict, db_path=DB_PATH) -> list:
    groups = {k: set(v) for k, v in alert_map.groupby("case_id")["row_id"]}
    out = []
    for i, rec in enumerate(pool):
        inp = rec["input"]
        win = eval_query(
            f'SELECT row_id, src_ip, dst_ip, dst_port, "timestamp", {label_col} FROM flows_eval '
            f'WHERE "timestamp" >= ? AND "timestamp" <= ?',
            [pd.Timestamp(inp["window_start"]).to_pydatetime(), pd.Timestamp(inp["window_end"]).to_pydatetime()],
            db_path=db_path,
        )
        t = truth_for_case(inp["case_id"], win, groups[inp["case_id"]], rec["endpoints"], label_col, cc)
        t["label_source"] = source
        out.append(t)
        if (i + 1) % 50 == 0:
            print(f"  truth[{source}] {i + 1}/{len(pool)}")
    return out