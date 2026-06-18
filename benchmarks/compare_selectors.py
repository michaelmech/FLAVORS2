"""Compare legacy and optimized FLAVORS2 selectors under shared seeds.

Example:
    python benchmarks/compare_selectors.py --budget 5 --repeats 5 --n-jobs 1
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flavors2 import FLAVORS2FeatureSelector, FLAVORS2LegacyFeatureSelector


def fixed_cv_accuracy(X, y, sample_weight=None):
    estimator = LogisticRegression(max_iter=1000, solver="liblinear", random_state=123)
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(3, counts.min())) if len(counts) else 0
    if n_splits < 2:
        estimator.fit(X, y)
        return {"score": float(estimator.score(X, y))}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
    return {"score": float(np.mean(scores))}


def load_datasets(seed):
    iris = load_iris()
    cancer = load_breast_cancer()
    synthetic_X, synthetic_y = make_classification(
        n_samples=600,
        n_features=80,
        n_informative=12,
        n_redundant=12,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
        shuffle=True,
    )
    return {
        "iris": (iris.data, iris.target),
        "breast_cancer": (cancer.data, cancer.target),
        "synthetic_80": (synthetic_X, synthetic_y),
    }


def run_once(selector_cls, X, y, budget, seed, n_jobs):
    random.seed(seed)
    np.random.seed(seed)

    selector = selector_cls(
        budget=budget,
        metrics=[fixed_cv_accuracy],
        n_jobs=n_jobs,
        random_state=seed,
    )

    started = time.perf_counter()
    selector.fit(X, y)
    elapsed = time.perf_counter() - started

    selected = selector.get_support(indices=True)
    best_marker = None
    cache_hits = None
    cache_misses = None
    n_evals = None
    fresh_evals = None
    loop_results = None

    if selector.selector is not None:
        if selector.selector.leaderboard:
            best_marker = selector.selector.leaderboard[0][0]
        cache_hits = getattr(selector.selector, "cache_hits_", None)
        cache_misses = getattr(selector.selector, "cache_misses_", None)
        n_evals = len(getattr(selector.selector, "performance_history", []))
        fresh_evals = cache_misses if cache_misses is not None else n_evals
        loop_results = getattr(selector.selector, "loop_results_", n_evals)

    return {
        "score_marker": best_marker,
        "n_selected": int(len(selected)),
        "selected": " ".join(map(str, selected)),
        "elapsed_sec": elapsed,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "fresh_evals": fresh_evals,
        "loop_results": loop_results,
        "n_evals": n_evals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks") / "selector_comparison.csv")
    args = parser.parse_args()

    selectors = {
        "legacy": FLAVORS2LegacyFeatureSelector,
        "optimized": FLAVORS2FeatureSelector,
    }

    rows = []
    for repeat in range(args.repeats):
        seed = 1000 + repeat
        for dataset_name, (X, y) in load_datasets(seed).items():
            for selector_name, selector_cls in selectors.items():
                result = run_once(selector_cls, X, y, args.budget, seed, args.n_jobs)
                row = {
                    "selector": selector_name,
                    "dataset": dataset_name,
                    "seed": seed,
                    "budget": args.budget,
                    "n_jobs": args.n_jobs,
                    **result,
                }
                rows.append(row)
                print(
                    f"{selector_name:9s} {dataset_name:14s} seed={seed} "
                    f"score_marker={result['score_marker']} "
                    f"n_selected={result['n_selected']} elapsed={result['elapsed_sec']:.2f}s"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
