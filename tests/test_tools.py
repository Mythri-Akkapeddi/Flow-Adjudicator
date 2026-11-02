import json

import pandas as pd
import pytest

from flowadj.tools import traffic_tools  # noqa: F401  (importing registers the tools)
from flowadj.tools.registry import ToolContext, ToolRunner, registry
from flowadj.tools import tcp_tools  # noqa: F401

START, END = "2017-07-07 14:00:00", "2017-07-07 15:00:00"


def run(mini_db, tool, args=None, start=START, end=END):
    ctx = ToolContext(db_path=mini_db, window_start=start, window_end=end)
    return ToolRunner(ctx).call(tool, args or {})


@pytest.fixture
def mini_df(mini_flows_path):
    return pd.read_parquet(mini_flows_path)


# ---------- traffic tools ----------
def test_case_overview_matches_pandas(mini_db, mini_df):
    r = run(mini_db, "case_overview")
    assert r.ok, r.output
    o = r.output
    assert o["n_flows"] == len(mini_df)
    assert o["n_src_ips"] == mini_df.src_ip.nunique()
    assert o["n_dst_ips"] == mini_df.dst_ip.nunique()
    assert o["top_sources"][0]["ip"] == "172.16.0.1" and o["top_sources"][0]["n_flows"] == 160


def test_case_overview_respects_window(mini_db, mini_df):
    s, e = "2017-07-07 14:20:00", "2017-07-07 14:20:30"
    o = run(mini_db, "case_overview", start=s, end=e).output
    exp = mini_df[(mini_df.timestamp >= s) & (mini_df.timestamp <= e)]
    assert o["n_flows"] == len(exp)


def test_top_talkers(mini_db):
    o = run(mini_db, "top_talkers", {"by": "src_ip", "n": 3}).output
    assert o["rows"][0]["src_ip"] == "172.16.0.1" and o["rows"][0]["n_flows"] == 160
    o = run(mini_db, "top_talkers", {"by": "dst_ip", "metric": "flows"}).output
    assert o["rows"][0]["dst_ip"] == "192.168.10.50" and o["rows"][0]["n_flows"] == 160
    o = run(mini_db, "top_talkers", {"by": "pair"}).output
    assert o["rows"][0]["src_ip"] == "172.16.0.1" and o["rows"][0]["dst_ip"] == "192.168.10.50"
    assert not run(mini_db, "top_talkers", {"by": "banana"}).ok


def test_host_summary(mini_db):
    o = run(mini_db, "host_summary", {"host": "192.168.10.50"}).output
    assert o["as_destination"]["n_flows"] == 160
    assert o["as_destination"]["n_peers"] == 1
    assert o["as_destination"]["n_dst_ports"] == 100
    assert o["as_source"]["n_flows"] == 0
    assert "note" in run(mini_db, "host_summary", {"host": "9.9.9.9"}).output


def test_port_profile(mini_db):
    o = run(mini_db, "port_profile", {"host": "192.168.10.50", "side": "dst"}).output
    assert o["n_distinct_ports"] == 100 and o["n_flows"] == 160
    assert o["ports"][0]["port"] == 22 and o["ports"][0]["n_flows"] == 61  # 60 ssh + 1 scan probe
    o = run(mini_db, "port_profile").output
    assert o["n_distinct_ports"] >= 100


def test_flow_size_distribution(mini_db):
    o = run(mini_db, "flow_size_distribution").output
    assert o["n_flows"] == 360
    assert o["frac_flows_le_2_packets"] == pytest.approx(100 / 360, abs=1e-3)
    assert run(mini_db, "flow_size_distribution", {"host": "192.168.10.50"}).output["n_flows"] == 160


# ---------- general ----------
def test_window_required(mini_db):
    r = ToolRunner(ToolContext(db_path=mini_db)).call("case_overview")
    assert not r.ok and "window" in r.output["error"]


def test_all_registered_schemas_are_clean():
    for name in registry.names():
        f = registry.get(name).schema()["function"]
        assert "ctx" not in f["parameters"]["properties"]
        assert f["description"]


# ---------- TCP tools ----------
def test_failed_connection_rate(mini_db):
    o = run(mini_db, "failed_connection_rate").output
    assert o["n_tcp_flows"] == 360 and o["n_failed"] == 100
    assert o["failed_rate"] == pytest.approx(100 / 360, abs=1e-3)
    o = run(mini_db, "failed_connection_rate", {"host": "192.168.10.50"}).output
    assert o["n_tcp_flows"] == 160 and o["n_failed"] == 100
    o = run(mini_db, "failed_connection_rate", {"port": 22}).output
    assert o["n_tcp_flows"] == 61 and o["n_failed"] == 1


def test_syn_ack_ratio(mini_db):
    o = run(mini_db, "syn_ack_ratio").output
    assert o["syn_packets"] == 480 and o["ack_packets"] == 1100  # 200+100+60*3 / 200+0+60*15
    assert o["syn_to_ack_ratio"] == pytest.approx(480 / 1100, abs=1e-3)
    assert o["frac_syn_without_ack"] == pytest.approx(100 / 360, abs=1e-3)
    assert o["frac_syn_no_reply"] == pytest.approx(100 / 360, abs=1e-3)


def test_flag_counts(mini_db):
    o = run(mini_db, "flag_counts").output
    assert o["n_tcp_flows"] == 360
    assert o["flags"]["rst"]["packets"] == 100
    assert o["flags"]["rst"]["frac_flows_with_flag"] == pytest.approx(100 / 360, abs=1e-3)
    assert o["flags"]["syn"]["packets"] == 480


def test_rst_profile(mini_db):
    o = run(mini_db, "rst_profile").output
    assert o["n_rst_flows"] == 100
    assert o["top_sources"][0]["ip"] == "172.16.0.1" and o["top_sources"][0]["n_rst_flows"] == 100
    assert o["top_destinations"][0]["ip"] == "192.168.10.50"


def test_short_flow_fraction(mini_db):
    o = run(mini_db, "short_flow_fraction", {"max_packets": 3}).output
    assert o["n_short_flows"] == 100
    assert o["frac_short_flows"] == pytest.approx(100 / 360, abs=1e-3)
    assert o["top_sources"][0]["ip"] == "172.16.0.1"


# ---------- temporal tools ----------
def test_burst_profile_finds_the_scan(mini_db):
    o = run(mini_db, "burst_profile", {"bin_seconds": 10}).output
    assert o["peak_flows_per_bin"] >= 33
    assert o["peak_bin_start"].startswith("2017-07-07 14:20")
    assert o["peak_to_median_active_ratio"] > 10


def test_beacon_periodicity_ssh_is_regular(mini_db):
    o = run(mini_db, "beacon_periodicity",
            {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50", "dst_port": 22},
            start="2017-07-07 14:39:00", end="2017-07-07 14:46:00").output
    assert o["n_flows"] == 60
    assert o["mean_gap_s"] == pytest.approx(5.0, abs=1e-3)
    assert o["cv"] < 0.01
    sparse = run(mini_db, "beacon_periodicity", {"src_ip": "9.9.9.9", "dst_ip": "8.8.8.8"}).output
    assert "note" in sparse


def test_interarrival_stats(mini_db):
    o = run(mini_db, "interarrival_stats", {"host": "192.168.10.50"},
            start="2017-07-07 14:39:00", end="2017-07-07 14:46:00").output
    assert o["n_gaps"] == 59
    assert o["median_gap_s"] == pytest.approx(5.0, abs=1e-3)
    assert o["median_flow_iat_mean_us"] == 100000


def test_compare_to_host_baseline(mini_db):
    o = run(mini_db, "compare_to_host_baseline", {"host": "192.168.10.50", "baseline_minutes": 60},
            start="2017-07-07 14:40:00", end="2017-07-07 14:46:00").output
    assert o["window"]["n_flows"] == 60 and o["baseline"]["n_flows"] == 100
    assert o["ratios_window_over_baseline"]["flows_per_min"] == pytest.approx(6.0, abs=0.05)
    o = run(mini_db, "compare_to_host_baseline", {"host": "192.168.10.50"}).output  # nothing before 14:00
    assert o["baseline"]["n_flows"] == 0 and o["ratios_window_over_baseline"] is None and o["note"]


# ---------- all tools: registration + no label leakage in outputs ----------
ALL_CALLS = [
    ("case_overview", {}), ("top_talkers", {"by": "pair"}), ("host_summary", {"host": "192.168.10.50"}),
    ("port_profile", {"host": "192.168.10.50"}), ("flow_size_distribution", {}),
    ("flag_counts", {}), ("syn_ack_ratio", {}), ("failed_connection_rate", {}), ("rst_profile", {}),
    ("short_flow_fraction", {}), ("burst_profile", {}),
    ("beacon_periodicity", {"src_ip": "172.16.0.1", "dst_ip": "192.168.10.50"}),
    ("interarrival_stats", {}), ("compare_to_host_baseline", {"host": "192.168.10.50"}),
]


def test_all_14_tools_registered():
    import flowadj.tools  # noqa: F401
    assert len(registry.names()) == 14
    assert {n for n, _ in ALL_CALLS} == set(registry.names())


@pytest.mark.parametrize("name,args", ALL_CALLS)
def test_tool_output_has_no_label_words(mini_db, name, args):
    r = run(mini_db, name, args)
    assert r.ok, r.output
    text = json.dumps(r.output).lower()
    for w in ("patator", "portscan", "ssh_bruteforce", "benign", "label", "family", "attempted"):
        assert w not in text