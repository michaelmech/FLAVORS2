FLAVORS² — Adaptive Feature Selection with Pareto Search
Fast and Lightweight Assessment of Variables for Optimally-reduced subsets, Resource-aware and Searchable towards Feature Learning Automation with Variable-Objective, Resource Scheduling.

FLAVORS² is a flexible, metaheuristic framework for selecting informative subsets of features in supervised learning tasks. It supports single and multi-objective optimization, incorporates custom metrics, handles time budgets, accepts warm priors, and can exploit model-estimated feature importances during search.

🧩 Installation
Install the latest release from PyPI. Note that the package name on PyPI (flavors-squared) differs from the import name (flavors2).

pip install flavors-squared


Then import the selector from the flavors2 namespace:

from flavors2 import FLAVORS2FeatureSelector


🚀 Quick Start (Single Metric)
Below, we select features for a binary classification task using a custom ROC AUC metric.

Your custom metric should accept (X, y, sample_weight=None) and return either a number or a dictionary with at least {"score": float}. If you also return a fitted model as {"model": est}, FLAVORS² can use its coef_/feature_importances_ attribute to weight features more intelligently.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from flavors2 import FLAVORS2FeatureSelector

# --- 1) custom metric (higher = better) ---
def auc_metric(X, y, sample_weight=None):
    # simple train/holdout inside the metric
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    est = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    fit_kw = {}
    if sample_weight is not None:
        # Split sample weights to match X_tr/X_te split
        sw_tr, _ = train_test_split(sample_weight, test_size=0.25, random_state=42, stratify=y)
        fit_kw["sample_weight"] = sw_tr

    est.fit(X_tr, y_tr, **fit_kw)
    proba = est.predict_proba(X_te)[:, 1]
    score = roc_auc_score(y_te, proba)
    # returning the model lets FLAVORS² read coef_/feature_importances_
    return {"score": float(score), "model": est}

# --- 2) data ---
data = load_breast_cancer()
X, y = data.data, data.target

# --- 3) selector ---
selector = FLAVORS2FeatureSelector(
    budget=20,              # time budget in seconds
    metrics=[auc_metric],   # single objective
    n_jobs=1,               # set >1 for parallel batch evaluation
    boruta=False,           # set True to enable shadow-feature penalty
    random_state=42,
)

selector.fit(X, y)          # sample_weight optional
X_sel = selector.transform(X)

print("Selected indices:", selector.get_support(indices=True).tolist())


The call to fit will search for a subset of features that maximizes your metric within the time budget. If your metric returns a model, FLAVORS² leverages its feature importances to update the search more intelligently.

🎯 Multi-metric (Pareto) Example
Use multiple metrics to build a Pareto frontier. FLAVORS² assumes higher-is-better for each metric. To prefer fewer features, create a metric that rewards sparsity (e.g., negative subset size so higher is better).

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def accuracy_metric(X, y, sample_weight=None):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    est = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1)
    fit_kw = {}
    if sample_weight is not None:
        sw_tr, _ = train_test_split(sample_weight, test_size=0.25, random_state=0, stratify=y)
        fit_kw["sample_weight"] = sw_tr

    est.fit(X_tr, y_tr, **fit_kw)
    pred = est.predict(X_te)
    return {"score": float(accuracy_score(y_te, pred)), "model": est}

def sparsity_metric(X, y, sample_weight=None):
    # Higher is better: reward smaller subsets relative to full dimensionality.
    p_full = getattr(sparsity_metric, "_p_full", None)
    if p_full is None:
        raise RuntimeError("Set sparsity_metric._p_full = X.shape[1] of the FULL dataset before use.")
    p = X.shape[1]
    return {"score": float(p_full - p)}  # larger score => fewer features

# one-time setup for sparsity reference (using the data loaded above)
sparsity_metric._p_full = X.shape[1]

selector_pareto = FLAVORS2FeatureSelector(
    budget=25,
    metrics=[auc_metric, accuracy_metric, sparsity_metric],
    n_jobs=1,
    boruta=True,        # try Boruta-style shadow penalty per iteration
    random_state=0,
)

selector_pareto.fit(X, y)
X_sel_pareto = selector_pareto.transform(X)


After fitting, you can examine the Pareto frontier via selector_pareto.selector.pareto_history. Each point in the frontier is a non-dominated solution across all metrics.

⏱️ Time Budgeting
The budget argument (in seconds) caps the length of the search. FLAVORS² returns the best-found subset when the clock runs out:

# Tight 10-second budget for quick experiments
fast_selector = FLAVORS2FeatureSelector(budget=10, metrics=[auc_metric])
fast_selector.fit(X, y)


You can compute a time budget by measuring alternative selectors and passing the minimum duration into FLAVORS² for a fair benchmark.

🧠 Feature Priors (Warm Start)
Provide feature priors to bias the search. The feature_priors argument accepts a 1-D array of length n_features, with larger values indicating more promising features. Common choices include mutual information, domain heuristics, or previous run statistics.

from sklearn.feature_selection import mutual_info_classif

# Example: MI-based priors
priors = mutual_info_classif(X, y, random_state=42)
priors = (priors - priors.min()) / (priors.ptp() or 1.0)  # scale 0..1

selector_with_priors = FLAVORS2FeatureSelector(
    budget=20,
    metrics=[auc_metric],
    feature_priors=priors,
    n_jobs=1,
)

selector_with_priors.fit(X, y)


To warm-start across datasets, persist selector.selector.feature_performance from a previous run and feed it back as feature_priors.

⚖️ Importance-Weighted Updates
When your metric returns a model exposing coef_ (linear models) or feature_importances_ (tree ensembles), FLAVORS² weights each feature’s contribution by that model’s importance for the evaluated subset. This helps amplify strong signals and de-emphasize passengers.

Enabling Boruta-style shadow features via boruta=True compares real features to random shadows and penalizes those that underperform the strongest shadow.

selector_imp = FLAVORS2FeatureSelector(
    budget=20,
    metrics=[auc_metric],  # returns model => importances used
    boruta=True,
    n_jobs=1,
)

selector_imp.fit(X, y)


🔗 Pipeline Integration
FLAVORS² is compatible with scikit-learn pipelines and column selectors. If you work with pandas DataFrames, the get_feature_names_out() method will return the selected feature names in order.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

pipe = Pipeline([
    ("scale", StandardScaler(with_mean=False)),   # or RobustScaler
    ("select", FLAVORS2FeatureSelector(budget=15, metrics=[auc_metric], n_jobs=1)),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
])

pipe.fit(df, y)

# selected boolean mask
mask = pipe.named_steps["select"].get_support()
print("Selected mask:", mask.tolist())


💡 FAQ
Q: How do I pass sample weights?
A: Call selector.fit(X, y, sample_weight=w)—FLAVORS² will forward them into your metric if it accepts sample_weight.

Q: Do I need to make all objectives “maximize”?
A: Yes—return larger-is-better scores. If you conceptually want to minimize something, simply negate it before returning.

Q: What if I only return a single number?
A: Returning a plain float is fine; FLAVORS² will wrap it as {"score": float} internally. Returning {"score": float, "model": est} unlocks importance-weighted updates.

📄 License
This project is licensed under the terms of the Apache 2.0 License. See LICENSE for full details.
