"""Exercise the public save/load/transform/resume workflow."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from flavors2 import FLAVORS2FeatureSelector
import flavors2.selector as selector_module


def make_fitted(*, default_metric=False, multiple_metrics=False):
    rng = np.random.RandomState(8)
    X = pd.DataFrame(rng.randn(40, 24), columns=[f"feature_{i}" for i in range(24)])
    y = (X.iloc[:, 0].to_numpy() > 0).astype(int)
    weights = np.linspace(1.0, 2.0, len(y))
    offset = 0.7

    # A local closure must survive loading in a process without this function.
    def metric(X, y, sample_weight=None):
        time.sleep(0.003)
        return {
            "score": offset + float(np.mean(X)),
            "feature_importances": np.ones(X.shape[1]),
        }

    metrics = None if default_metric else [metric]
    if multiple_metrics:
        metrics.append(lambda X, y: float(np.mean(X**2)))
    selector = FLAVORS2FeatureSelector(
        budget=0.1, metrics=metrics, random_state=9, strict_budget=False
    ).fit(X, y, sample_weight=weights)
    return selector, X, y, weights


@pytest.mark.parametrize(
    "default_metric,multiple_metrics", [(True, False), (False, False), (False, True)]
)
def test_roundtrip_preserves_selection_and_search_state(
    tmp_path, default_metric, multiple_metrics
):
    selector, X, _, weights = make_fitted(
        default_metric=default_metric, multiple_metrics=multiple_metrics
    )
    path = tmp_path / "selector.flavors2"
    selector.save(path)
    restored = FLAVORS2FeatureSelector.load(path)

    np.testing.assert_array_equal(restored.transform(X), selector.transform(X))
    np.testing.assert_array_equal(restored.get_support(), selector.get_support())
    np.testing.assert_array_equal(
        restored.get_feature_names_out(), selector.get_feature_names_out()
    )
    np.testing.assert_array_equal(restored.selector.sample_weight, weights)
    np.testing.assert_array_equal(
        restored.selector._rng.get_state()[1], selector.selector._rng.get_state()[1]
    )
    assert (
        restored.selector._rng.get_state()[2:] == selector.selector._rng.get_state()[2:]
    )
    assert (
        restored.selector.performance_history == selector.selector.performance_history
    )
    assert restored.selector._proposal_stats == selector.selector._proposal_stats
    assert restored.selector._elite_candidates == selector.selector._elite_candidates
    assert (
        restored.selector._score_cache.keys() == selector.selector._score_cache.keys()
    )
    assert restored.eci_history_ == selector.eci_history_
    assert restored.selector._evaluation_executor is None
    if multiple_metrics:
        assert restored.selector.pareto_history == selector.selector.pareto_history


def test_fresh_process_loads_closure_transforms_and_extends_search(tmp_path):
    selector, X, _, _ = make_fitted()
    path = tmp_path / "selector.flavors2"
    selector.save(path)
    script = """
import json
import sys
from flavors2 import FLAVORS2FeatureSelector
s = FLAVORS2FeatureSelector.load(sys.argv[1])
before = len(s.selector.performance_history)
selected = s.get_support(indices=True).tolist()
transformed_sum = float(s.transform(s.selector.X).sum())
s.resume(budget=0.15)
s.save(sys.argv[1])
print(json.dumps({'before': before, 'after': len(s.selector.performance_history),
                  'selected': selected, 'sum': transformed_sum}))
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["selected"] == selector.selected_indices_
    assert report["sum"] == pytest.approx(float(selector.transform(X).sum()))
    assert report["after"] > report["before"] > 0
    continued = FLAVORS2FeatureSelector.load(path)
    assert (
        continued.selector.performance_history[: report["before"]]
        == selector.selector.performance_history
    )
    assert continued.eci_history_[: len(selector.eci_history_)] == selector.eci_history_
    assert set(continued.selector._score_cache) >= set(selector.selector._score_cache)
    assert continued.selector.best_error <= selector.selector.best_error
    assert continued.selected_indices_ == list(continued.selector.leaderboard[0][1])
    assert continued.strategy_eci_ == continued.selector.strategy_eci_


def test_resume_keeps_incumbent_when_new_budget_cannot_evaluate(tmp_path, monkeypatch):
    selector, X, _, _ = make_fitted()
    selector.save(tmp_path / "selector")
    restored = FLAVORS2FeatureSelector.load(tmp_path / "selector")
    before = restored.selector.best_error
    history = list(restored.eci_history_)
    monkeypatch.setattr(restored.selector, "_fresh_eval_capacity", lambda deadline: 0)
    assert restored.resume(budget=0.1) is restored
    assert np.isfinite(before)
    assert restored.selector.best_error == before
    assert restored.eci_history_ == history
    np.testing.assert_array_equal(restored.transform(X), selector.transform(X))


def test_strict_worker_search_can_be_saved_and_resumed(tmp_path):
    rng = np.random.RandomState(5)
    X = rng.randn(50, 24)
    y = (X[:, 0] > 0).astype(int)
    selector = FLAVORS2FeatureSelector(
        # Allow cold Python/numpy/sklearn worker imports on Windows.
        budget=8,
        metrics=[lambda X, y: float(np.mean(X))],
        random_state=3,
        strict_budget=True,
    ).fit(X, y)
    assert selector.selector.performance_history
    selector.save(tmp_path / "selector")
    restored = FLAVORS2FeatureSelector.load(tmp_path / "selector")
    history = list(restored.selector.performance_history)
    restored.resume(budget=8)
    assert len(restored.selector.performance_history) > len(history)
    assert restored.selector.performance_history[: len(history)] == history
    assert restored.selector._evaluation_executor is None
    assert restored.selector._evaluation_worker_ready == []
    restored.save(tmp_path / "selector")
    np.testing.assert_array_equal(
        FLAVORS2FeatureSelector.load(tmp_path / "selector").transform(X),
        restored.transform(X),
    )


def test_fit_copies_inputs_and_resume_rejects_mutated_retained_data():
    selector, X, y, weights = make_fitted()
    original_X = selector.selector.X.copy()
    original_y = selector.selector.y.copy()
    original_weights = selector.selector.sample_weight.copy()
    X.iloc[:, :] = 0
    y[:] = 0
    weights[:] = 0
    selector.resume(budget=0)
    np.testing.assert_array_equal(selector.selector.X, original_X)
    np.testing.assert_array_equal(selector.selector.y, original_y)
    np.testing.assert_array_equal(selector.selector.sample_weight, original_weights)
    selector.selector.X = selector.selector.X.copy()
    selector.selector.X[0, 0] += 1
    with pytest.raises(ValueError, match="training data"):
        selector.resume(budget=0.1)


def test_loaded_selector_checks_feature_schema_and_fit_starts_fresh(tmp_path):
    selector, X, y, _ = make_fitted()
    selector.save(tmp_path / "selector")
    restored = FLAVORS2FeatureSelector.load(tmp_path / "selector")
    with pytest.raises(ValueError, match="names and order"):
        restored.transform(X.iloc[:, ::-1])
    with pytest.raises(ValueError, match="features"):
        restored.transform(X.to_numpy()[:, :-1])
    core = restored.selector
    restored.set_params(budget=0).fit(X.to_numpy(), y)
    assert restored.selector is not core
    assert restored.selector.performance_history == []
    assert restored.feature_names_in_ is None
    assert clone(restored).selector is None


def test_failed_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    selector, _, _, _ = make_fitted()
    path = tmp_path / "selector"
    selector.save(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(selector_module.cloudpickle, "dump", fail)
    with pytest.raises(RuntimeError, match="serialization failed"):
        selector.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "change,match", [("schema", "schema version"), ("versions", "environment")]
)
def test_incompatible_checkpoint_rejected_before_unpickling(
    tmp_path, monkeypatch, change, match
):
    header = copy.deepcopy(selector_module._checkpoint_header())
    header[change] = "unsupported"
    path = tmp_path / "selector"
    path.write_bytes((json.dumps(header) + "\n").encode() + b"invalid pickle")

    def fail(*args, **kwargs):
        pytest.fail("Incompatible payload must not be unpickled")

    monkeypatch.setattr(selector_module.cloudpickle, "load", fail)
    with pytest.raises(ValueError, match=match):
        FLAVORS2FeatureSelector.load(path)


def test_unfitted_and_busy_selectors_cannot_be_saved_or_resumed(tmp_path):
    selector = FLAVORS2FeatureSelector()
    with pytest.raises(NotFittedError):
        selector.save(tmp_path / "selector")
    with pytest.raises(NotFittedError):
        selector.resume(budget=1)
    with pytest.raises(NotFittedError):
        selector.transform(np.zeros((2, 2)))
    selector._operation_in_progress = True
    with pytest.raises(RuntimeError, match="current fit"):
        selector.save(tmp_path / "selector")


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), None, True])
def test_invalid_resume_budget_leaves_state_unchanged(budget):
    selector, _, _, _ = make_fitted()
    before = list(selector.selector.performance_history)
    with pytest.raises(ValueError, match="budget"):
        selector.resume(budget=budget)
    assert selector.selector.performance_history == before
