"""Regressions reported by the GitHub review of persistent searches."""

import datetime

import numpy as np
import pytest
from sklearn.base import clone

from flavors2 import FLAVORS2, FLAVORS2FeatureSelector
import flavors2.core as core_module


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
