import numpy as np

import flavors2.core as core_module
from flavors2.core import FLAVORS2


def constant_metric(X, y, sample_weight=None):
    return {"score": 0.5}


def make_optimizer(n_features=12):
    optimizer = FLAVORS2(
        budget=1,
        metrics=[constant_metric],
        random_state=7,
        strict_budget=False,
    )
    optimizer.n_feats = n_features
    optimizer.feature_priors = np.linspace(1.0, 0.1, n_features)
    optimizer.feature_performance = np.zeros(n_features)
    optimizer.feature_counts = np.zeros(n_features)
    optimizer.feature_stability = np.zeros(n_features)
    optimizer.unevaluated = set(range(n_features))
    return optimizer


def test_importance_evidence_is_direction_free_and_quality_weighted():
    low_quality = FLAVORS2._feature_evidence([-4.0, 0.0, 2.0], 0.0)
    high_quality = FLAVORS2._feature_evidence([-4.0, 0.0, 2.0], 1.0)

    assert np.argmax(low_quality) == 0
    assert np.argmax(high_quality) == 0
    assert np.all(high_quality > low_quality)


def test_feature_updates_do_not_depend_on_dictionary_order():
    forward = make_optimizer(3)
    reverse = make_optimizer(3)
    scores = {0: 0.1, 1: 0.5, 2: 0.9}

    forward.update_feature_performance(scores)
    reverse.update_feature_performance(dict(reversed(list(scores.items()))))

    np.testing.assert_allclose(forward.feature_performance, reverse.feature_performance)
    np.testing.assert_allclose(forward.feature_counts, reverse.feature_counts)


def test_flaml_eci_combines_direct_improvement_and_catch_up_cost():
    best_sequence = FLAVORS2._calculate_eci(
        total_cost=12.0,
        latest_improvement_cost=10.0,
        previous_improvement_cost=4.0,
        improvement_delta=0.2,
        candidate_best_error=0.4,
        global_best_error=0.4,
        last_trial_cost=3.0,
        cost_growth=2.0,
    )
    trailing_sequence = FLAVORS2._calculate_eci(
        total_cost=12.0,
        latest_improvement_cost=10.0,
        previous_improvement_cost=4.0,
        improvement_delta=0.2,
        candidate_best_error=0.5,
        global_best_error=0.4,
        last_trial_cost=3.0,
        cost_growth=2.0,
    )

    assert best_sequence == 6.0
    assert np.isclose(trailing_sequence, 8.0)


def test_low_eci_strategy_receives_more_budget_without_starving_others():
    optimizer = make_optimizer()
    optimizer.best_error = 0.2
    optimizer.performance_history = [0.6, 0.5, 0.4, 0.2]
    optimizer._proposal_stats["global"].update({
        "trials": 3,
        "improvement": 0.4,
        "cost": 3.0,
        "last_cost": 1.0,
        "best_error": 0.2,
        "improvements": [(1.0, 0.5), (2.0, 0.2)],
    })
    for name in ("local", "ranked", "uncertain"):
        optimizer._proposal_stats[name].update({
            "trials": 3,
            "improvement": 0.2,
            "cost": 9.0,
            "last_cost": 3.0,
            "best_error": 0.4,
            "improvements": [(3.0, 0.6), (6.0, 0.4)],
        })

    names, probabilities = optimizer._proposal_probabilities()

    global_index = names.index("global")
    assert probabilities[global_index] == np.max(probabilities)
    assert np.all(probabilities >= 0.05)
    assert np.isclose(np.sum(probabilities), 1.0)


def test_strategy_eci_increases_when_cost_accumulates_without_improvement():
    optimizer = make_optimizer()
    optimizer.best_error = 0.4
    optimizer.performance_history = [0.5, 0.4]
    optimizer._proposal_stats["local"].update({
        "trials": 2,
        "improvement": 0.1,
        "cost": 2.0,
        "last_cost": 1.0,
        "best_error": 0.4,
        "improvements": [(1.0, 0.5), (2.0, 0.4)],
    })
    before = optimizer._strategy_eci("local")

    subset_key = (0, 1, 2)
    optimizer._candidate_strategies[subset_key] = "local"
    optimizer._candidate_eci_estimates[subset_key] = 0.8
    optimizer._record_proposal_result(
        subset_key,
        error=0.45,
        previous_size_best=0.4,
        eval_time=5.0,
    )

    after = optimizer._strategy_eci("local")
    assert after > before
    assert optimizer.eci_history_[-1]["strategy"] == "local"
    assert optimizer.eci_history_[-1]["candidate_eci"] == 0.8


def test_candidate_eci_penalizes_higher_predicted_cost_at_equal_potential():
    optimizer = make_optimizer(6)
    optimizer.feature_priors = np.ones(6)
    optimizer.feature_performance = np.full(6, 0.5)
    optimizer.feature_counts = np.ones(6)
    optimizer.feature_cost_history = {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 9.0,
        4: 9.0,
        5: 9.0,
    }

    cheap = optimizer._candidate_eci([0, 1, 2])
    expensive = optimizer._candidate_eci([3, 4, 5])

    assert cheap < expensive


def test_single_worker_search_does_not_initialize_loky_backend(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("single-worker search must not initialize loky")

    monkeypatch.setattr(core_module, "parallel_backend", fail_if_called)
    optimizer = FLAVORS2(
        budget=0.1,
        metrics=[constant_metric],
        n_jobs=1,
        random_state=7,
        strict_budget=False,
    )
    X = np.arange(80, dtype=float).reshape(20, 4)
    y = np.asarray([0, 1] * 10)

    optimizer.fit(X, y)

    assert optimizer.performance_history


def test_ranked_proposals_are_soft_samples_from_the_promising_pool():
    optimizer = make_optimizer(20)

    candidate = optimizer._ranked_candidate(5)

    assert len(candidate) == 5
    assert len(set(candidate)) == 5
    assert set(candidate).issubset(set(range(8)))


def test_elite_population_retains_non_incumbent_subsets():
    optimizer = make_optimizer(8)
    optimizer._record_elite_candidate([0, 1, 2], 0.20)
    optimizer._record_elite_candidate([3, 4, 5], 0.25)
    optimizer._record_elite_candidate([0, 1], 0.30)

    elites = optimizer._diverse_elite_subsets()

    assert [0, 1, 2] in elites
    assert [3, 4, 5] in elites
    assert [0, 1] in elites
