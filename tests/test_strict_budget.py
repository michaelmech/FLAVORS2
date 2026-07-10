import time

import numpy as np

from flavors2 import FLAVORS2FeatureSelector


def slow_metric(X, y, sample_weight=None):
    if X.shape[0] > 20:
        time.sleep(2.0)
    return {"score": 0.5}


def test_strict_budget_terminates_a_running_candidate():
    rng = np.random.RandomState(0)
    X = rng.randn(14_000, 50)
    y = (X[:, 0] > 0).astype(int)
    selector = FLAVORS2FeatureSelector(
        budget=0.25,
        metrics=[slow_metric],
        random_state=0,
        strict_budget=True,
    )

    started = time.monotonic()
    selector.fit(X, y)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert selector.selector.budget_exhausted_
    assert selector.selector.timed_out_evaluations_ == 1
    assert selector.selector.cache_misses_ == 1
    assert not selector.selector._score_cache
    assert selector.selected_indices_
