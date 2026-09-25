import time
import datetime

import numpy as np

from flavors2 import FLAVORS2, FLAVORS2FeatureSelector


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


def test_strict_candidate_respects_phase_deadline():
    rng = np.random.RandomState(3)
    X = rng.randn(40, 10)
    y = (X[:, 0] > 0).astype(int)
    selector = FLAVORS2(
        budget=0.01, metrics=[slow_metric], random_state=3, strict_budget=True
    ).fit(X, y)
    candidate = list(range(X.shape[1]))
    assert selector._normalize_subset_key(candidate) not in selector._score_cache

    now = datetime.datetime.now()
    selector._fit_deadline = now + datetime.timedelta(seconds=1)
    selector._worker_cleanup_reserve = 0.05
    phase_deadline = now + datetime.timedelta(seconds=0.15)

    started = time.monotonic()
    result = selector._evaluate_batch([candidate], deadline=phase_deadline)[0]
    elapsed = time.monotonic() - started

    assert result["timed_out"]
    assert elapsed < 0.6
    assert selector.timed_out_evaluations_ == 1
    assert selector._strict_evaluation_deadline(phase_deadline) == phase_deadline
    assert selector._normalize_subset_key(candidate) not in selector._score_cache
