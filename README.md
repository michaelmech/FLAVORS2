

---



**FLAVORS2 (Fast and Lightweight Assessment of Variables for Optimally-Reduced Subsets towards Feature Learning Automation with Variable-Objective, Resource Scheduling)** is an efficient feature selection library for machine learning. It automates the search for optimal feature subsets under time budgets, supporting custom metrics, multi-objective Pareto optimization, feature priors, and scikit-learn integration.

## Install

FLAVORS2 can be installed from [PyPI](https://pypi.org/project/flavors-squared):

```bash
pip install flavors-squared
```


## Feature Selection Example (scikit-learn compatible)

FLAVORS2 provides a scikit-learn compatible `FLAVORS2FeatureSelector` for seamless integration into ML pipelines. Here's an example using the Iris dataset:

```python
from sklearn.datasets import load_iris
from flavors2 import FLAVORS2FeatureSelector

# Load data
data = load_iris()
X, y = data.data, data.target

# Initialize and fit the selector with a 30-second budget
selector = FLAVORS2FeatureSelector(budget=30, random_state=42)
selector.fit(X, y)

# Transform the data to selected features
X_selected = selector.transform(X)

print(f"Selected feature indices: {selector.selected_indices_}")
```

This performs a budgeted search for the optimal subset using default metrics (AUC for classification or R² for regression).

## Advanced Usage

### Custom Metrics

Define custom metrics returning a dict with `'score'`:

```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from flavors2 import FLAVORS2FeatureSelector

def custom_accuracy(X, y, sample_weight=None):
    model = LogisticRegression(max_iter=200)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', fit_params={'sample_weight': sample_weight})
    return {'score': np.mean(scores)}

data = load_iris()
X, y = data.data, data.target

selector = FLAVORS2FeatureSelector(budget=20, metrics=[custom_accuracy], random_state=42)
selector.fit(X, y)
```

### Multiple Metrics with Pareto Optimization

Balance multiple objectives:

```python
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from flavors2 import FLAVORS2FeatureSelector

def accuracy_metric(X, y, sample_weight=None):
    model = LogisticRegression(max_iter=200)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', fit_params={'sample_weight': sample_weight})
    return {'score': np.mean(scores)}

def f1_metric(X, y, sample_weight=None):
    model = LogisticRegression(max_iter=200)
    scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(model, X, y, cv=5, scoring=scorer, fit_params={'sample_weight': sample_weight})
    return {'score': np.mean(scores)}

data = load_iris()
X, y = data.data, data.target

selector = FLAVORS2FeatureSelector(budget=30, metrics=[accuracy_metric, f1_metric], random_state=42)
selector.fit(X, y)

print("Pareto history:", selector.selector.pareto_history)
```

### Time Budgeting

Control search duration:

```python
from sklearn.datasets import load_breast_cancer
from flavors2 import FLAVORS2FeatureSelector

data = load_breast_cancer()
X, y = data.data, data.target

selector_short = FLAVORS2FeatureSelector(budget=10, random_state=42)
selector_short.fit(X, y)
print(f"Features selected in 10s: {len(selector_short.selected_indices_)}")
```

### Feature Priors

Guide selection with priors:

```python
from sklearn.datasets import load_iris
import numpy as np
from flavors2 import FLAVORS2FeatureSelector

data = load_iris()
X, y = data.data, data.target
priors = np.array([0.1, 0.1, 0.9, 0.9])

selector = FLAVORS2FeatureSelector(budget=20, feature_priors=priors, random_state=42)
selector.fit(X, y)
```

### Feature Importance Weighting with a Model

Incorporate model importances:

```python
# ✅ Correct way: return a fitted model so FLAVORS2 can read its importances.
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from flavors2 import FLAVORS2FeatureSelector
import numpy as np

def rf_metric_with_model(X, y, sample_weight=None):
    # 1) Fit once on the current subset to expose feature_importances_
    est = RandomForestClassifier(n_estimators=200, random_state=42)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    est.fit(X, y, **fit_kwargs)

    # 2) Score via CV (fresh clones); keep score pure (no “importance bonus”)
    cv_est = RandomForestClassifier(n_estimators=200, random_state=42)
    scores = cross_val_score(cv_est, X, y, cv=5, scoring="accuracy")
    return {"score": float(np.mean(scores)), "model": est}

selector = FLAVORS2FeatureSelector(
    budget=30,
    metrics=[rf_metric_with_model],
    boruta=True,            # optional: adds Boruta-style shadow checks
    random_state=42,
)


```

## Benchmarks

FLAVORS2 outperforms baselines in rankings and wins:

<p align="center">
  <img width="544" src="assets/boxplot_graph.png" />
</p>

<p align="center">
  <img width="544" src="assets/pie_graph.png" />
</p>

<p align="center">
  <img src="assets/distribution_graph.png" alt="Score Advantage" width="544"/>
</p>

**Figure 1.** Distribution of FLAVORS² score advantage over baselines like GJO, RFE, and Permutation.

To visualize the Pareto frontier of performance vs. runtime across datasets:

<p align="center">
  <img src="assets/pareto_graph.png" alt="Pareto Frontier" width="616"/>
</p>

**Figure 2.** Pareto frontier showing trade-off between performance and runtime across datasets.


See `assets/h2h_benchmark_summary.csv` for details.


## Documentation

- [Core Implementation](https://github.com/michaelmech/FLAVORS2/blob/main/src/flavors2/core.py)

## Citations

For use of FLAVORS2, cite the repository or relevant papers if applicable.

## License

MIT License

Copyright (c) 2025 Michael Mech

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

