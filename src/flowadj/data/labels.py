"""Map raw labels to coarse families and attach corrected labels to flows."""
import numpy as np
import pandas as pd

FAMILIES = [
    "benign", "ftp_bruteforce", "ssh_bruteforce", "dos", "ddos",
    "portscan", "botnet", "web_attack", "other",
]  # 'other' = infiltration + heartbleed (very rare); 'unmatched'/'unknown' are bookkeeping

KEY = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "minute"]


def label_to_family(label: str) -> str:
    s = str(label).strip().lower()
    if s == "unmatched":
        return "unmatched"
    if s.startswith("benign"):
        return "benign"
    if "infiltration" in s and "portscan" in s:  # compromised host scanning the LAN: behaves as a portscan
        return "portscan"
    if "infiltration" in s or "heartbleed" in s:
        return "other"
    if "ftp" in s:
        return "ftp_bruteforce"
    if "ssh" in s:
        return "ssh_bruteforce"
    if "ddos" in s:  # before 'dos'
        return "ddos"
    if "dos" in s:
        return "dos"
    if "portscan" in s or "port scan" in s:
        return "portscan"
    if "bot" in s:
        return "botnet"
    if "web attack" in s or "xss" in s or "sql" in s or "brute force" in s:
        return "web_attack"
    return "unknown"


def attach_labels(corr: pd.DataFrame, orig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Base table = corrected (regenerated) flows. Original labels are looked up by
    (5-tuple + minute); flows with no match get label_original = 'UNMATCHED'.
    Returns (merged table with row_id, report dict).
    """
    corr = corr.rename(columns={"label": "label_corrected"}).copy()
    corr["family_corrected"] = corr["label_corrected"].map(label_to_family)
    corr["attempted"] = corr["label_corrected"].str.contains("attempted", case=False, na=False)
    corr["minute"] = corr["timestamp"].dt.floor("min")

    o = orig[["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "timestamp", "label"]].copy()
    o["minute"] = o["timestamp"].dt.floor("min")
    o["_benign"] = o["label"].str.lower().eq("benign")
    o = o.sort_values("_benign")  # attack labels first, so they win on duplicate keys
    o = o.drop_duplicates(KEY, keep="first")
    o = o.rename(columns={"label": "label_original"})[KEY + ["label_original"]]

    n_before = len(corr)
    merged = corr.merge(o, on=KEY, how="left")
    assert len(merged) == n_before, "merge changed the row count"

    matched = merged["label_original"].notna()
    merged["label_original"] = merged["label_original"].fillna("UNMATCHED")
    merged["family_original"] = merged["label_original"].map(label_to_family)
    merged = merged.drop(columns=["minute"])

    both = matched & (merged["family_original"] == merged["family_corrected"])
    report = {
        "rows": int(len(merged)),
        "matched_to_original": int(matched.sum()),
        "match_rate": float(matched.mean()),
        "family_agreement_among_matched": float(both.sum() / max(matched.sum(), 1)),
    }

    atk = ~merged["family_corrected"].isin(["benign", "unmatched", "unknown"])
    report["match_rate_attack_flows"] = float(matched[atk].mean())

    merged = merged.sort_values("timestamp", kind="stable").reset_index(drop=True)
    merged.insert(0, "row_id", np.arange(len(merged), dtype="int64"))
    return merged, report