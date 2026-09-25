"""Regressions reported by the GitHub review of persistent searches."""

from collections import Counter
import datetime
import time

import numpy as np
import pytest
from sklearn.base import clone

from flavors2 import FLAVORS2, FLAVORS2FeatureSelector
import flavors2.core as core_module


def slow_review_metric(X, y, sample_weight=None):
    if len(X) > 20:
        time.sleep(2)
    return {"score": 0.5}


def phase_sensitive_metric(X, y, sample_weight=None):
    if len(X) > 20 and X.shape[1] == 10:
        time.sleep(2)
    return {"score": 0.5}


def test_phase_timeout_allows_later_evaluation_without_fit_exhaustion():
    rng = np.random.RandomState(4)
    X = rng.randn(40, 10)
    y = (X[:, 0] > 0).astype(int)
    optimizer = FLAVORS2(
        budget=0.01, metrics=[phase_sensitive_metric], strict_budget=True
    ).fit(X, y)
    optimizer.budget_exhausted_ = False
    slow = list(range(10))
    fast = list(range(9))
    slow_key = optimizer._normalize_subset_key(slow)
    fast_key = optimizer._normalize_subset_key(fast)
    optimizer._candidate_strategies[slow_key] = "local"
    optimizer._candidate_eci_estimates[slow_key] = 0.8
    optimizer._candidate_strategies[fast_key] = "global"
    optimizer._candidate_eci_estimates[fast_key] = 0.8
    now = datetime.datetime.now()
    optimizer._fit_deadline = now + datetime.timedelta(seconds=5)
    optimizer._worker_cleanup_reserve = 0.05

    try:
        first = optimizer._evaluate_batch(
            [slow], deadline=now + datetime.timedelta(seconds=0.2)
        )[0]
        assert first["timed_out"]
        assert not optimizer.budget_exhausted_
        assert optimizer.timed_out_evaluations_ == 1
        assert optimizer.eci_history_[-1]["strategy"] == "local"

        second = optimizer._evaluate_batch(
            [fast], deadline=datetime.datetime.now() + datetime.timedelta(seconds=3)
        )[0]
        assert not second.get("timed_out", False)
        optimizer._update_after_evaluation(second, fast, slow)
        assert not optimizer.budget_exhausted_
        assert np.isfinite(optimizer.current_error)
        assert len(optimizer.performance_history) == 1
        assert optimizer.eci_history_[-1]["strategy"] == "global"

        incumbent_error = optimizer.current_error
        optimizer._candidate_strategies[slow_key] = "local"
        optimizer._candidate_eci_estimates[slow_key] = 0.8
        overall_deadline = datetime.datetime.now() + datetime.timedelta(seconds=0.25)
        optimizer._fit_deadline = overall_deadline
        third = optimizer._evaluate_batch(
            [slow], deadline=overall_deadline + datetime.timedelta(seconds=1)
        )[0]
        assert third["timed_out"]
        assert optimizer.budget_exhausted_
        assert optimizer.timed_out_evaluations_ == 2
        assert optimizer.current_error == incumbent_error
        assert len(optimizer.performance_history) == 1
    finally:
        optimizer._shutdown_evaluation_executor(kill_workers=True)


def test_budget_filter_removes_only_rejected_candidate_metadata(monkeypatch):
    rng = np.random.RandomState(5)
    X = rng.randn(40, 10)
    y = (X[:, 0] > 0).astype(int)
    optimizer = FLAVORS2(
        budget=0.01,
        metrics=[phase_sensitive_metric],
        n_jobs=4,
        strict_budget=False,
    ).fit(X, y)
    optimizer.cost_history = [1.0]
    proposals = [[0, 1], [2, 3], [4, 5], [6, 7]]
    strategies = ("local", "global", "ranked", "uncertain")
    for candidate, strategy in zip(proposals, strategies):
        key = optimizer._normalize_subset_key(candidate)
        optimizer._candidate_strategies[key] = strategy
        optimizer._candidate_eci_estimates[key] = 0.8

    now = datetime.datetime(2026, 1, 1)

    class FixedClock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(core_module.datetime, "datetime", FixedClock)
    admitted = optimizer._budget_limited_batch(
        proposals, now + datetime.timedelta(seconds=1.5)
    )

    assert admitted == [proposals[0]]
    admitted_key = optimizer._normalize_subset_key(admitted[0])
    assert optimizer._candidate_strategies == {admitted_key: "local"}
    assert optimizer._candidate_eci_estimates == {admitted_key: 0.8}

    result = optimizer._evaluate_batch(admitted, deadline=now + datetime.timedelta(seconds=1.5))[0]
    optimizer._update_after_evaluation(result, admitted[0], [])
    assert optimizer._proposal_stats["local"]["trials"] == 1
    assert optimizer.eci_history_[-1]["strategy"] == "local"
    assert not optimizer._candidate_strategies
    assert not optimizer._candidate_eci_estimates


def test_realized_strategy_chances_follow_eci_allocator(monkeypatch):
    optimizer = FLAVORS2(
        budget=1, metrics=[slow_review_metric], random_state=7, strict_budget=False
    )
    optimizer.n_feats = 24
    optimizer.unevaluated = set(range(24))
    optimizer.iters = 10
    optimizer.iters_best = 0
    counter = {"value": 0}

    def unique_candidate(*args):
        counter["value"] += 1
        return [index for index in range(24) if counter["value"] & (1 << index)]

    monkeypatch.setattr(
        optimizer,
        "_strategy_eci",
        lambda strategy: 1.0 if strategy == "global" else 1e9,
    )
    monkeypatch.setattr(optimizer, "_candidate_eci", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(optimizer, "_propose_candidate", unique_candidate)

    names, probabilities = optimizer._proposal_probabilities(0.1)
    assert dict(zip(names, probabilities))["local"] >= 0.05
    assert dict(zip(names, probabilities))["uncertain"] >= 0.05

    counts = Counter()
    for _ in range(6000):
        candidate = optimizer._generate_candidate_batch([0, 1, 2], 1, 1, 0.1)[0]
        key = optimizer._normalize_subset_key(candidate)
        counts[optimizer._candidate_strategies.pop(key)] += 1
        optimizer._candidate_eci_estimates.pop(key)

    assert counts["local"] / 6000 >= 0.04
    assert counts["uncertain"] / 6000 >= 0.04
    assert len({optimizer._proposal_target("global", [0, 1, 2], 1) for _ in range(40)}) > 1


def test_batched_strict_timeouts_charge_eci_once_without_a_score():
    rng = np.random.RandomState(3)
    X = rng.randn(40, 10)
    y = (X[:, 0] > 0).astype(int)
    optimizer = FLAVORS2(
        budget=0.01,
        metrics=[slow_review_metric],
        n_jobs=2,
        random_state=3,
        strict_budget=True,
    ).fit(X, y)
    candidates = [list(range(10)), list(range(9))]
    for candidate, strategy in zip(candidates, ("local", "uncertain")):
        key = optimizer._normalize_subset_key(candidate)
        optimizer._candidate_strategies[key] = strategy
        optimizer._candidate_eci_estimates[key] = 0.8

    initial_eci = {name: optimizer._strategy_eci(name) for name in ("local", "uncertain")}
    now = datetime.datetime.now()
    optimizer._fit_deadline = now + datetime.timedelta(seconds=1)
    optimizer._worker_cleanup_reserve = 0.05
    phase_deadline = now + datetime.timedelta(seconds=0.2)

    started = time.perf_counter()
    results = optimizer._evaluate_batch(
        [candidates[0], candidates[1], candidates[0]], deadline=phase_deadline
    )
    elapsed = time.perf_counter() - started

    assert all(result["timed_out"] for result in results)
    assert elapsed < 0.6
    assert len(optimizer.eci_history_) == 2
    charged = 0.0
    for name in ("local", "uncertain"):
        stats = optimizer._proposal_stats[name]
        assert stats["trials"] == 1
        assert optimizer._strategy_eci(name) > initial_eci[name]
        charged += stats["cost"]
    assert 0 < charged <= elapsed
    assert all(entry["error"] == float("inf") for entry in optimizer.eci_history_)
    assert not optimizer.performance_history
    assert not optimizer._score_cache
    assert optimizer.current_error == float("inf")


@pytest.mark.parametrize("estimator_type", [FLAVORS2, FLAVORS2FeatureSelector])
@pytest.mark.parametrize("seed", [3, 7, 21])
def test_refinement_evaluations_keep_the_adaptive_strategy_allocator(
    monkeypatch, estimator_type, seed
):
    rng = np.random.RandomState(13)
    X = rng.randn(40, 24)
    y = (X[:, 0] > 0).astype(int)
    epoch = datetime.datetime(2026, 1, 1)
    clock = {"ticks": 0}
    allocations = []
    tail_evaluations = []
    original_probabilities = core_module.FLAVORS2._proposal_probabilities
    original_evaluate = core_module.FLAVORS2._evaluate_batch

    class SearchClock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return epoch + datetime.timedelta(milliseconds=10 * clock["ticks"])

    def metric(X, y, sample_weight=None):
        if len(X) == 40:
            clock["ticks"] += 1
        return float(np.mean(X))

    def record_probabilities(self, remaining_budget_fraction=1.0):
        names, probabilities = original_probabilities(self, remaining_budget_fraction)
        allocations.append((clock["ticks"], dict(zip(names, probabilities))))
        return names, probabilities

    def record_evaluations(self, subsets, *args, **kwargs):
        if clock["ticks"] >= 90:
            tail_evaluations.append(
                (clock["ticks"], allocations[-1] if allocations else None)
            )
        return original_evaluate(self, subsets, *args, **kwargs)

    monkeypatch.setattr(core_module.datetime, "datetime", SearchClock)
    monkeypatch.setattr(
        core_module.FLAVORS2, "_proposal_probabilities", record_probabilities
    )
    monkeypatch.setattr(core_module.FLAVORS2, "_evaluate_batch", record_evaluations)
    estimator_type(
        budget=1.0, metrics=[metric], random_state=seed, strict_budget=False
    ).fit(X, y)

    assert tail_evaluations, "The public fit must reach the final search phase."
    for tick, allocation in tail_evaluations:
        assert allocation is not None and allocation[0] == tick, (
            "Every refinement batch must use the current adaptive probabilities."
        )
        assert set(allocation[1]) == {"local", "global", "ranked", "uncertain"}
        assert min(allocation[1].values()) >= 0.05


def test_resume_preserves_fit_budget_through_checkpoint_clone_and_refit(tmp_path):
    rng = np.random.RandomState(13)
    X = rng.randn(40, 24)
    y = (X[:, 0] > 0).astype(int)
    selector = FLAVORS2FeatureSelector(
        budget=0.1, metrics=[lambda X, y: float(np.mean(X))], strict_budget=False
    ).fit(X, y)
    selector.set_params(budget=0.2)
    original_core = selector.selector

    selector.resume(budget=0)

    assert selector.selector is original_core
    assert selector.get_params()["budget"] == 0.2
    assert clone(selector).budget == 0.2
    selector.save(tmp_path / "checkpoint")
    restored = FLAVORS2FeatureSelector.load(tmp_path / "checkpoint")
    assert restored.budget == 0.2
    restored_core = restored.selector
    restored.fit(X, y)
    assert restored.selector.budget == 0.2
    assert restored.selector is not restored_core
