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

from .core import FLAVORS2  # noqa: F401
from .__version__ import __version__  # noqa: F401

__all__ = ["FLAVORS2", "__version__",'FLAVORS2FeatureSelector']
