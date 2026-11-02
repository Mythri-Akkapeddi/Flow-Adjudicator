"""S0: detector + template. Fills the report schema with no LLM."""
import pandas as pd

from flowadj.agents.schema import Report
from flowadj.cases.build_cases import ALERTS_PATH, load_alerts
from flowadj.config import get_default


def _top(s: pd.Series, min_share: float, max_n: int) -> list:
    vc = s.value_counts()
    keep = vc[vc / vc.sum() >= min_share].head(max_n)
    return sorted(keep.index.tolist())


def _iso(t) -> str:
    return pd.Timestamp(t).isoformat(sep=" ")


def s0_reports(case_ids) -> dict[str, Report]:
    cc = get_default()["cases"]
    alerts = load_alerts()
    amap = pd.read_parquet(ALERTS_PATH)
    df = alerts.merge(amap, on="row_id")
    df = df[df["case_id"].isin(set(case_ids))]
    out = {}
    for cid, g in df.groupby("case_id"):
        rf = g[g["alert_rf"] & (g["rf_class"] != "benign")]
        if len(g) and len(rf) / len(g) >= 0.5:
            family = rf["rf_class"].value_counts().idxmax()
            fam_rows = rf[rf["rf_class"] == family]
            n_ports = int(fam_rows["dst_port"].nunique())
            out[cid] = Report(
                verdict="attack", family=family,
                attackers=_top(fam_rows["src_ip"], cc["share_threshold"], cc["max_entities"]),
                victims=_top(fam_rows["dst_ip"], cc["share_threshold"], cc["max_entities"]),
                ports=[] if n_ports > cc["many_ports"] else
                      [int(p) for p in _top(fam_rows["dst_port"], cc["share_threshold"], cc["max_entities"])],
                window_start=_iso(fam_rows["timestamp"].min()),
                window_end=_iso(fam_rows["timestamp"].max()),
                summary=f"S0 template: {len(rf)} of {len(g)} alert flows classified as {family} by the random forest.",
            )
        else:
            out[cid] = Report(
                verdict="benign", family="benign",
                summary=f"S0 template: only {len(rf)} of {len(g)} alert flows flagged by the random forest "
                        f"(alerts mostly from the isolation forest).",
            )
    return out