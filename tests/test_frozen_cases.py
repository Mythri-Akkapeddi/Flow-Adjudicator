import json

import pandas as pd
import pytest

from flowadj.cases.build_cases import CASE_INPUT_KEYS
from flowadj.cases.jsonl import CASES_DIR, read_jsonl
from flowadj.cases.sampling import HASH_FILE, verify_frozen
from flowadj.config import get_default

needs_frozen = pytest.mark.skipif(not HASH_FILE.exists(), reason="run the freeze stage first")


@needs_frozen
def test_hash_matches():
    assert verify_frozen()


@needs_frozen
def test_dev_and_test_disjoint_and_sized():
    cc = get_default()["cases"]
    dev, test = read_jsonl(CASES_DIR / "cases_dev.jsonl"), read_jsonl(CASES_DIR / "cases_test.jsonl")
    assert not ({c["case_id"] for c in dev} & {c["case_id"] for c in test})
    assert len(dev) <= cc["dev"]["attack"] + cc["dev"]["benign"]
    assert len(test) <= cc["test"]["attack"] + cc["test"]["benign"]


@needs_frozen
def test_frozen_inputs_are_label_free_and_windows_do_not_overlap():
    dev, test = read_jsonl(CASES_DIR / "cases_dev.jsonl"), read_jsonl(CASES_DIR / "cases_test.jsonl")
    for c in dev + test:
        assert set(c) == set(CASE_INPUT_KEYS)
    text = json.dumps(dev + test).lower()
    for w in ("label", "family", "benign", "attack", "verdict", "split"):
        assert w not in text
    spans = [(pd.Timestamp(c["window_start"]), pd.Timestamp(c["window_end"])) for c in dev]
    for s, e in spans:
        for c in test:
            assert e <= pd.Timestamp(c["window_start"]) or s >= pd.Timestamp(c["window_end"])