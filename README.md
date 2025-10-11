import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from flavors2 import FLAVORS2FeatureSelector

# --- 1) Custom metric (higher = better)
def auc_metric(X, y, sample_weight=None):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    est = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    est.fit(X_tr, y_tr)
    score = roc_auc_score(y_te, est.predict_proba(X_te)[:, 1])
    return {"score": score, "model": est}

# --- 2) Load data
data = load_breast_cancer()
X, y = data.data, data.target

# --- 3) Run FLAVORS²
selector = FLAVORS2FeatureSelector(budget=20, metrics=[auc_metric], random_state=42)
selector.fit(X, y)

print("Selected indices:", selector.get_support(indices=True))
