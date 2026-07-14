import numpy as np

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


def test_productive_proposal_strategy_receives_more_budget_without_starving_others():
    optimizer = make_optimizer()
    before_names, before = optimizer._proposal_probabilities()
    optimizer._proposal_stats["global"].update(
        {"trials": 3, "improvement": 0.3, "cost": 1.0}
    )

    after_names, after = optimizer._proposal_probabilities()

    assert before_names == after_names
    global_index = after_names.index("global")
    assert after[global_index] > before[global_index]
    assert np.all(after > 0)
    assert np.isclose(np.sum(after), 1.0)


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
