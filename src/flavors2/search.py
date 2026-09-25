"""Maintained search scheduling over the generated evaluation primitives."""

from contextlib import nullcontext
import datetime
import time

import numpy as np
from joblib import parallel_backend

from .core import FLAVORS2 as _GeneratedSearch


class FLAVORS2(_GeneratedSearch):
    """Allocate every search phase through the adaptive strategy portfolio."""

    def _size_exploration_probability(self, remaining_budget_fraction):
        return 0.0

    def _evaluate_batch(self, subsets, deadline=None):
        was_exhausted = self.budget_exhausted_
        started = time.perf_counter()
        results = super()._evaluate_batch(subsets, deadline=deadline)
        timed_out_keys = {
            result["subset_key"] for result in results if result.get("timed_out", False)
        }
        if timed_out_keys:
            fit_deadline = getattr(self, "_fit_deadline", deadline)
            cleanup_reserve = getattr(self, "_worker_cleanup_reserve", 0.05)
            fit_cutoff = fit_deadline - datetime.timedelta(seconds=cleanup_reserve)
            self.budget_exhausted_ = (
                was_exhausted
                or deadline >= fit_cutoff
                or datetime.datetime.now() >= fit_cutoff
            )
            trial_cost = (time.perf_counter() - started) / len(timed_out_keys)
            for key in timed_out_keys:
                self._record_proposal_result(key, float("inf"), float("inf"), trial_cost)
        return results

    def _budget_limited_batch(self, candidates, deadline):
        admitted = super()._budget_limited_batch(candidates, deadline)
        admitted_keys = {self._normalize_subset_key(candidate) for candidate in admitted}
        for candidate in candidates:
            key = self._normalize_subset_key(candidate)
            if key not in admitted_keys:
                self._candidate_strategies.pop(key, None)
                self._candidate_eci_estimates.pop(key, None)
        return admitted

    def _strict_evaluation_deadline(self, deadline):
        if not self.strict_budget:
            return deadline
        fit_deadline = getattr(self, "_fit_deadline", deadline)
        cleanup_reserve = getattr(self, "_worker_cleanup_reserve", 0.05)
        return min(
            deadline,
            fit_deadline - datetime.timedelta(seconds=cleanup_reserve),
        )

    def _ensure_search_fallback(self):
        # A warm start must retain its finite incumbent if no new trial fits.
        if not self.leaderboard:
            self.leaderboard.append((float("inf"), self._fallback_subset_from_priors()))

    def _reference_error(self, subset):
        cached = self._score_cache.get(self._normalize_subset_key(subset))
        if cached is None:
            return None
        error = cached.get("current_error")
        if error is None or not np.isfinite(error):
            return None
        return float(error)

    def _search_phase(
        self, current_subset, *, start_time, deadline, budget, refine=False,
        require_coverage=False,
    ):
        batch_size = self._worker_count()
        no_improvement_counter = 0
        reference_error = self._reference_error(current_subset)
        duration = max(1e-9, (deadline - start_time).total_seconds())
        while datetime.datetime.now() < deadline:
            remaining = max(
                0.0, (deadline - datetime.datetime.now()).total_seconds() / duration
            )
            needs_coverage = self._coverage_ratio() < self._coverage_target(budget)
            if require_coverage and not needs_coverage:
                break
            self._set_coverage_pressure(
                remaining, force=needs_coverage and (require_coverage or remaining > 0.2)
            )
            if refine and self.leaderboard:
                # Focus candidate mutations on the incumbent without restricting
                # the strategy portfolio that can spend the remaining budget.
                current_subset = list(self.leaderboard[0][1])
                reference_error = self._reference_error(current_subset)
            step_size = self.adjust_step_size(len(current_subset), remaining)
            if refine:
                step_size = 1
            candidates = self._generate_candidate_batch(
                current_subset, batch_size, step_size, remaining
            )
            candidates = self._budget_limited_batch(candidates, deadline)
            if not candidates:
                break

            best_error = float("inf") if reference_error is None else reference_error
            best_subset = None
            results = self._evaluate_batch(candidates, deadline=deadline)
            for subset, result in zip(candidates, results):
                self._update_after_evaluation(result, subset, current_subset)
                candidate_error = result.get("current_error")
                if (
                    not result.get("timed_out", False)
                    and candidate_error is not None
                    and np.isfinite(candidate_error)
                    and candidate_error < best_error
                ):
                    best_error = candidate_error
                    best_subset = subset
            if best_subset is not None:
                current_subset = list(best_subset)
                reference_error = float(best_error)
                self.num_features = len(current_subset)
                no_improvement_counter = 0
            else:
                no_improvement_counter += len(candidates)

            if not refine and self.iters > 20:
                restart_probability = min(0.1, no_improvement_counter / 50.0)
                if self._rng.rand() < restart_probability:
                    score = self._format_global_score(self._current_best_global_score())
                    print(f"Performing probabilistic restart... best_global_score={score}")
                    sizes = [len(subset) for _, subset in self.leaderboard[:5]]
                    size = int(self._rng.choice(sizes)) if sizes else self.num_features
                    current_subset = self.search_strategy(size)
                    reference_error = self._reference_error(current_subset)
                    self.num_features = len(current_subset)
                    no_improvement_counter = 0
                    # The new reference is explored by the next portfolio batch.
        return current_subset

    def select_best_feature_subset(self, budget):
        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=budget)
        main_end = end_time - datetime.timedelta(seconds=budget * 0.1)
        coverage_end = end_time - datetime.timedelta(seconds=budget * 0.03)
        if not hasattr(self, "feature_performance"):
            self.feature_performance = np.zeros(self.n_feats)
        if not hasattr(self, "feature_counts"):
            self.feature_counts = np.zeros(self.n_feats)
        if not hasattr(self, "num_features") or not self.fitted:
            self.num_features = self._initial_subset_size()

        self._set_coverage_pressure(1.0, force=True)
        current_subset = self.search_strategy(self.num_features)
        key = self._normalize_subset_key(current_subset)
        self._candidate_strategies[key] = "global"
        self._candidate_eci_estimates[key] = self._candidate_eci(key)
        if not self._budget_limited_batch([current_subset], end_time):
            self._ensure_search_fallback()
            return
        result = self._evaluate_batch([current_subset], deadline=end_time)[0]
        if result.get("timed_out", False):
            self._ensure_search_fallback()
            return
        self._update_after_evaluation(result, current_subset, [])
        if not self.fitted:
            self.best_error = self.current_error
            self.iters_best = 0

        context = parallel_backend("loky") if self._worker_count() > 1 else nullcontext()
        with context:
            current_subset = self._search_phase(
                current_subset, start_time=start_time, deadline=main_end, budget=budget
            )
            current_subset = self._search_phase(
                current_subset, start_time=start_time, deadline=coverage_end,
                budget=budget, require_coverage=True,
            )
            self._search_phase(
                current_subset, start_time=start_time, deadline=end_time,
                budget=budget, refine=True,
            )
        self._ensure_search_fallback()
