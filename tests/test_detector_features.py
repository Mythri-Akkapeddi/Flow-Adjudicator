import numpy as np

from flowadj.detector.features import assert_clean, feature_columns, load_features

IDENTIFIERS = {"row_id", "flow_id", "src_ip", "dst_ip", "src_port", "timestamp"}


def test_no_identifier_features(mini_db):
    cols = feature_columns(mini_db)
    assert IDENTIFIERS.isdisjoint(cols)
    assert not [c for c in cols if any(w in c for w in ("label", "family", "attempt"))]
    assert "dst_port" in cols and "flow_duration" in cols
    assert_clean(cols)


def test_load_features_is_finite_float32(mini_db):
    cols = feature_columns(mini_db)
    ids, X = load_features(cols, 0, 100, mini_db)
    assert X.dtype == np.float32 and X.shape == (100, len(cols))
    assert np.isfinite(X).all()
    assert ids.tolist() == list(range(100))