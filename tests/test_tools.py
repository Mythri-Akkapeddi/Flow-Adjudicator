import json

import pandas as pd
import pytest

from flowadj.tools import traffic_tools  # noqa: F401  (importing registers the tools)
from flowadj.tools.registry import ToolContext, ToolRunner, registry

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