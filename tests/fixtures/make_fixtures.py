"""Build a tiny deterministic flow table that mimics flows_corrected.parquet."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "mini_flows.parquet"


def build() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2017-07-07 14:00:00")
    rows = []

    def add(ts, src, sp, dst, dp, dur, fwd, bwd, syn, rst, ack, lo, fo, lc, fc, att, **extra):
        r = dict(
            flow_id=f"{dst}-{src}-{dp}-{sp}-6", src_ip=src, src_port=sp, dst_ip=dst, dst_port=dp,
            protocol=6, timestamp=ts, flow_duration=dur, total_fwd_packets=fwd,
            total_backward_packets=bwd, syn_flag_count=syn, rst_flag_count=rst, ack_flag_count=ack,
            total_length_of_fwd_packets=0, total_length_of_bwd_packets=0,
            fin_flag_count=0, psh_flag_count=0, urg_flag_count=0,
            flow_iat_mean=0.0, flow_iat_std=0.0, flow_iat_max=0.0, flow_iat_min=0.0,
            init_win_bytes_forward=-1, init_win_bytes_backward=-1,
            label_original=lo, family_original=fo, label_corrected=lc, family_corrected=fc,
            attempted=att, attempted_category=1 if att else -1,
            timestamp_raw=ts.strftime("%d/%m/%Y %H:%M"), source_file="Friday-fixture.csv",
            capture_day="Friday",
        )
        r.update(extra)
        rows.append(r)

    hosts = ["192.168.10.5", "192.168.10.8", "192.168.10.9", "192.168.10.14", "192.168.10.15"]
    for _ in range(200):  # benign
        fwd, bwd = int(rng.integers(2, 30)), int(rng.integers(2, 30))
        add(base + pd.Timedelta(seconds=int(rng.integers(0, 3600))), str(rng.choice(hosts)),
            int(rng.integers(30000, 60000)), str(rng.choice(["192.168.10.3", "8.8.8.8"])),
            int(rng.choice([53, 80, 443])), int(rng.integers(1_000, 5_000_000)),
            fwd, bwd, 1, 0, 1,
            "BENIGN", "benign", "BENIGN", "benign", False,
            total_length_of_fwd_packets=fwd * int(rng.integers(40, 1400)),
            total_length_of_bwd_packets=bwd * int(rng.integers(40, 1400)),
            fin_flag_count=1, psh_flag_count=int(rng.integers(0, 5)),
            flow_iat_mean=float(rng.integers(1_000, 500_000)), flow_iat_std=1000.0,
            flow_iat_max=600_000.0, flow_iat_min=10.0,
            init_win_bytes_forward=8192, init_win_bytes_backward=229)
    for i in range(100):  # portscan burst at 14:20, one attacker, many ports
        add(pd.Timestamp("2017-07-07 14:20:00") + pd.Timedelta(milliseconds=300 * i), "172.16.0.1",
            40000, "192.168.10.50", i + 1, 100, 1, 0, 1, 1, 0,
            "PortScan", "portscan", "PortScan", "portscan", False)
    for i in range(60):  # ssh brute force 14:40-14:45, regular spacing
        att = i < 10
        add(pd.Timestamp("2017-07-07 14:40:00") + pd.Timedelta(seconds=5 * i), "172.16.0.1",
            50000 + i, "192.168.10.50", 22, 2_000_000, 20, 18, 3, 0, 15,
            "SSH-Patator", "ssh_bruteforce",
            "SSH-Patator - Attempted" if att else "SSH-Patator", "ssh_bruteforce", att,
            total_length_of_fwd_packets=2000, total_length_of_bwd_packets=1800,
            fin_flag_count=1, psh_flag_count=10, flow_iat_mean=100_000.0, flow_iat_std=500.0,
            flow_iat_max=150_000.0, flow_iat_min=50_000.0,
            init_win_bytes_forward=29200, init_win_bytes_backward=235)

    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype="int64"))
    for c in ["src_port", "dst_port", "protocol"]:
        df[c] = df[c].astype("int32")
    return df


if __name__ == "__main__":
    df = build()
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")