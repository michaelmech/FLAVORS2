"""Public selector with persistent, resumable search sessions.

The search implementation in core.py is a generated notebook export.
Keep the user-facing session lifecycle here so it can evolve independently.
"""

import json
import os
from pathlib import Path
import platform
import tempfile

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import NotFittedError

from .__version__ import __version__
from .core import FLAVORS2FeatureSelector as _BaseSelector
from .search import FLAVORS2


def _checkpoint_header():
    return {
        "format": "flavors2-selector",
        "schema": 1,
        "versions": {
            "flavors2": __version__,
            "python": ".".join(platform.python_version_tuple()[:2]),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "cloudpickle": cloudpickle.__version__,
        },
    }


class FLAVORS2FeatureSelector(_BaseSelector):
    """Select features, save a completed search, and resume it later.

    Constructor parameters are the same as the core sklearn wrapper.
    ``fit`` starts a new search; ``resume(budget=...)`` extends the fitted one
    using its retained training data, metric, and search configuration.
    ``save`` includes those inputs, selected features, and learned search state.
    Checkpoints must be loaded in a compatible environment and must be trusted.
    """

    @staticmethod
    def _validate_budget(budget):
        if isinstance(budget, (bool, np.bool_)):
            raise ValueError("budget must be a finite, non-negative number of seconds.")
        try:
            value = float(budget)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "budget must be a finite, non-negative number of seconds."
            ) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError("budget must be a finite, non-negative number of seconds.")
        return value

    def _require_idle(self):
        if getattr(self, "_operation_in_progress", False):
            raise RuntimeError("Wait for the current fit or resume call to finish.")

    def _require_fitted(self):
        if self.selector is None or not self.selector.fitted:
            raise NotFittedError("Call fit before saving, transforming, or resuming.")

    def _training_fingerprint(self):
        return joblib.hash(
            tuple(
                None if value is None else np.ascontiguousarray(value)
                for value in (
                    self.selector.X,
                    self.selector.y,
                    self.selector.sample_weight,
                )
            ),
            hash_name="sha1",
        )

    def _check_training_data(self):
        if self._training_fingerprint() != self._training_fingerprint_:
            raise ValueError(
                "The retained training data or sample weights changed. "
                "Call fit to start a new search."
            )

    def _sync_search_results(self):
        if self.selector.leaderboard:
            self.selected_indices_ = list(map(int, self.selector.leaderboard[0][1]))
        else:
            self.selected_indices_ = list(range(self.n_features_in_))
        self.selected_indices = self.selected_indices_
        self.strategy_eci_ = dict(self.selector.strategy_eci_)
        self.eci_history_ = list(self.selector.eci_history_)

    def fit(self, X, y=None, sample_weight=None):
        """Start a fresh search and retain private copies of its training inputs."""
        self._require_idle()
        self._validate_budget(self.budget)
        self._operation_in_progress = True
        self.selector = None
        self.selected_indices_ = None
        self.selected_indices = None
        self.feature_names_in_ = None
        self.strategy_eci_ = None
        self.eci_history_ = None
        try:
            X = (
                X.copy(deep=True)
                if isinstance(X, pd.DataFrame)
                else np.array(X, copy=True)
            )
            y = None if y is None else np.array(y, copy=True)
            weights = (
                None if sample_weight is None else np.array(sample_weight, copy=True)
            )
            self.n_features_in_ = X.shape[1]
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = X.columns.astype(str).to_numpy()
            self._sample_weight = weights
            self.selector = FLAVORS2(**self.get_params(deep=False))
            self.selector.fit(X, y, sample_weight=weights)
            self._sync_search_results()
            self._training_fingerprint_ = self._training_fingerprint()
        finally:
            self._operation_in_progress = False
        return self

    def resume(self, *, budget):
        """Continue the fitted search for an additional ``budget`` seconds.

        Uses the retained training inputs and the fitted core's configuration.
        Changes to constructor parameters take effect on the next ``fit``.
        Prior evaluations, ECI statistics, population, and RNG state are reused.
        This is a new timed search phase, not an exact replay of one longer run.
        """
        self._require_idle()
        self._require_fitted()
        budget = self._validate_budget(budget)
        self._check_training_data()
        self._operation_in_progress = True
        previous_best = self.selector.best_error
        try:
            self.selector.fit(
                self.selector.X,
                self.selector.y,
                sample_weight=self.selector.sample_weight,
                budget=budget,
            )
        finally:
            # Retain the previous incumbent when a resumed phase cannot finish
            # a new evaluation.
            self.selector.best_error = min(previous_best, self.selector.best_error)
            self._sync_search_results()
            self._operation_in_progress = False
        return self

    def transform(self, X):
        """Apply the saved subset, checking named column order when available."""
        self._require_fitted()
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            if not np.array_equal(
                X.columns.astype(str).to_numpy(), self.feature_names_in_
            ):
                raise ValueError(
                    "DataFrame columns must match the fitted names and order."
                )
        return super().transform(X)

    def save(self, path):
        """Atomically save a completed selector, including its training data.

        Local metric functions and closures are serialized with cloudpickle.
        Existing checkpoints are replaced only after serialization succeeds.
        Do not save concurrently with fitting or resuming the same object.
        """
        self._require_idle()
        self._require_fitted()
        self._check_training_data()
        if self.selector._evaluation_executor is not None:
            raise RuntimeError("Cannot save while evaluation workers are active.")
        path = Path(path)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((json.dumps(_checkpoint_header()) + "\n").encode("utf-8"))
                cloudpickle.dump(self, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @classmethod
    def load(cls, path):
        """Load a trusted checkpoint for transform, get_support, or resume.

        Loading executes serialized Python code. Only load files you trust.
        The format checks Python major/minor and exact package versions before
        deserializing; migration between versions is not currently supported.
        """
        with Path(path).open("rb") as stream:
            try:
                header = json.loads(stream.readline(16384))
            except (ValueError, UnicodeError) as exc:
                raise ValueError(
                    "Invalid FLAVORS2 selector checkpoint header."
                ) from exc
            if (
                not isinstance(header, dict)
                or header.get("format") != "flavors2-selector"
            ):
                raise ValueError("Not a FLAVORS2 selector checkpoint.")
            expected = _checkpoint_header()
            if header.get("schema") != expected["schema"]:
                raise ValueError("Unsupported FLAVORS2 checkpoint schema version.")
            if header.get("versions") != expected["versions"]:
                raise ValueError(
                    "Checkpoint environment differs from this environment. "
                    f"Restore the saved versions: {header.get('versions')}."
                )
            selector = cloudpickle.load(stream)
        if not isinstance(selector, cls):
            raise ValueError(
                "Checkpoint does not contain a compatible feature selector."
            )
        selector._require_idle()
        selector._require_fitted()
        selector._check_training_data()
        selector._sync_search_results()
        return selector
