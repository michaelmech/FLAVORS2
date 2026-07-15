"""Basic smoke tests for the flavors2 package.

These tests ensure that the package can be imported and that the
expected public API attributes exist. They are not comprehensive
unit tests but provide a quick check that the package structure is
correct and that the module is functional at a high level.
"""

import numpy as np


def test_import():
    """Importing the top-level package should succeed."""
    import flavors2  # noqa: F401


def test_version_and_api():
    """The package should expose __version__ and FLAVORS2."""
    import flavors2

    assert hasattr(flavors2, "__version__")
    assert hasattr(flavors2, "FLAVORS2")
    assert hasattr(flavors2, "FLAVORS2FeatureSelector")
    assert hasattr(flavors2, "FLAVORS2Legacy")
    assert hasattr(flavors2, "FLAVORS2LegacyFeatureSelector")


def test_initial_fallback_uses_twice_square_root_feature_count():
    from flavors2 import FLAVORS2FeatureSelector

    rng = np.random.RandomState(0)
    X = rng.randn(30, 100)
    y = (X[:, 0] > 0).astype(int)
    selector = FLAVORS2FeatureSelector(budget=0, random_state=0, strict_budget=False)

    selector.fit(X, y)

    assert len(selector.selected_indices_) == 20
    assert isinstance(selector.strategy_eci_, dict)
    assert isinstance(selector.eci_history_, list)
