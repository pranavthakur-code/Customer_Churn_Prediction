"""Metrics, cross-validation and threshold selection."""
import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn import config


def cross_validate_model(pipe, X, y):
    """Stratified k-fold CV. Preprocessing is re-fitted inside every fold."""
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    res = cross_validate(pipe, X, y, cv=cv,
                         scoring=["roc_auc", "recall", "precision"])
    return {k.replace("test_", ""): (v.mean(), v.std())
            for k, v in res.items() if k.startswith("test_")}


def best_threshold(y_true, proba, min_recall=0.70):
    """Highest-precision threshold that still reaches the recall target."""
    best_t, best_p = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (proba >= t).astype(int)
        if recall_score(y_true, pred) >= min_recall:
            p = precision_score(y_true, pred, zero_division=0)
            if p > best_p:
                best_t, best_p = float(round(t, 2)), p
    return best_t


def test_report(pipe, X_test, y_test, threshold=config.DECISION_THRESHOLD):
    """Final metrics on the untouched test set."""
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
