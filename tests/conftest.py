import runpy
from pathlib import Path

import pytest

from flowadj.data.store import build_store, close_all

FIXTURE = Path(__file__).parent / "fixtures" / "mini_flows.parquet"


@pytest.fixture(scope="session")
def mini_flows_path():
    if not FIXTURE.exists():
        runpy.run_path(str(FIXTURE.parent / "make_fixtures.py"), run_name="__main__")
    return FIXTURE


@pytest.fixture(scope="session")
def mini_db(tmp_path_factory, mini_flows_path):
    db = tmp_path_factory.mktemp("db") / "mini.duckdb"
    build_store(mini_flows_path, db)
    yield db
    close_all()