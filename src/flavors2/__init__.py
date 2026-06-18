"""Top-level package for flavors2.

This package provides the FLAVORS2 feature selection tool along with
helper functions. It exposes a clean public API for users to import
the `FLAVORS2` estimator directly from the package namespace.

Examples
--------
>>> from flavors2 import FLAVORS2
>>> fs = FLAVORS2(budget=10)
>>> fs.fit(X, y)
"""

from .core import FLAVORS2,FLAVORS2FeatureSelector
from .legacy import FLAVORS2 as FLAVORS2Legacy
from .legacy import FLAVORS2FeatureSelector as FLAVORS2LegacyFeatureSelector
from .__version__ import __version__  

__all__ = [
    "FLAVORS2",
    "FLAVORS2FeatureSelector",
    "FLAVORS2Legacy",
    "FLAVORS2LegacyFeatureSelector",
    "__version__",
]
