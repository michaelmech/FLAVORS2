import numpy as np
import pytest

from flavors2.core import FLAVORS2, ensure_metric_protocol


def test_metric_feature_importances_are_preserved():
    def metric(X, y, sample_weight=None):
        return {
            "score": 0.75,
            "feature_importances": np.arange(1, X.shape[1] + 1, dtype=float),
        }

    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)
    result = FLAVORS2._compute_subset_score(
        [0, 2],
        X,
        y,
        None,
        [ensure_metric_protocol(metric)],
        False,
        X.shape[1],
    )

    np.testing.assert_array_equal(result["feature_importances"], [1.0, 2.0])


def test_metric_feature_importance_count_must_match_subset():
    def metric(X, y, sample_weight=None):
        return {"score": 0.75, "feature_importances": [1.0]}

    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)

    with pytest.raises(ValueError, match="length must match"):
        FLAVORS2._compute_subset_score(
            [0, 2],
            X,
            y,
            None,
            [ensure_metric_protocol(metric)],
            False,
            X.shape[1],
        )
