import pytest

from flowadj.data.store import agent_columns, agent_query, eval_columns, eval_query

BAD_NAME_PARTS = ("label", "family", "attempt", "source_file", "timestamp_raw")
ATTACK_WORDS = ("patator", "portscan", "ddos", "benign", "slowloris", "hulk", "botnet", "xss")


def test_agent_view_has_no_label_columns(mini_db):
    bad = [c for c in agent_columns(mini_db) if any(p in c.lower() for p in BAD_NAME_PARTS)]
    assert bad == []


def test_eval_view_has_labels(mini_db):
    cols = eval_columns(mini_db)
    assert "label_corrected" in cols and "family_corrected" in cols and "row_id" in cols


def test_agent_view_keeps_needed_columns(mini_db):
    cols = agent_columns(mini_db)
    for c in ["row_id", "src_ip", "dst_ip", "dst_port", "timestamp", "syn_flag_count"]:
        assert c in cols


@pytest.mark.parametrize("sql", [
    "SELECT * FROM flows_eval",
    "SELECT * FROM flows_all",
    "SELECT label_corrected FROM flows_agent",
    "SELECT family_original FROM flows_agent",
    "SELECT source_file FROM flows_agent",
])
def test_agent_query_blocks_forbidden(mini_db, sql):
    with pytest.raises(PermissionError):
        agent_query(sql, db_path=mini_db)


def test_agent_query_works(mini_db):
    df = agent_query("SELECT COUNT(*) AS n FROM flows_agent", db_path=mini_db)
    assert int(df["n"][0]) == 360


def test_select_star_has_no_label_columns(mini_db):
    df = agent_query("SELECT * FROM flows_agent LIMIT 5", db_path=mini_db)
    assert not [c for c in df.columns if any(p in c.lower() for p in BAD_NAME_PARTS)]


def test_no_attack_words_in_string_values(mini_db):
    df = agent_query("SELECT * FROM flows_agent", db_path=mini_db)
    for col in df.select_dtypes(include="object").columns:
        joined = " ".join(df[col].astype(str).unique()).lower()
        assert not any(w in joined for w in ATTACK_WORDS), f"{col} leaks an attack word"


def test_eval_and_agent_rows_align(mini_db):
    a = agent_query("SELECT row_id FROM flows_agent ORDER BY row_id", db_path=mini_db)
    e = eval_query("SELECT row_id FROM flows_eval ORDER BY row_id", db_path=mini_db)
    assert a["row_id"].tolist() == e["row_id"].tolist()