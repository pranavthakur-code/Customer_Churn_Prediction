"""Reusable evaluation helpers: one place for metrics, intervals and slices."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def _check(y_true, proba, threshold=0.5):
    """Validate inputs once so every metric fails with a clear message."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_true) != len(proba):
        raise ValueError(f"length mismatch: {len(y_true)} labels, "
                         f"{len(proba)} probabilities")
    if np.isnan(proba).any():
        raise ValueError("probabilities contain NaN")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true has a single class; AUC is undefined")
    return y_true, proba


def classification_summary(y_true, proba, threshold=0.5):
    """Threshold-based and threshold-free metrics in one dictionary."""
    y_true, proba = _check(y_true, proba, threshold)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y_true),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15)),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, proba, metric, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for any metric(y_true, proba) -> float.

    Captures test-set sampling noise only, not other sources of uncertainty.
    """
    y_true, proba = _check(y_true, proba)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:      # resample lost a class: skip
            continue
        stats.append(metric(y_true[i], proba[i]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, proba)), float(lo), float(hi)


def slice_report(frame, y_col, pred_col, by, min_n=40):
    """Error rate, recall and FP/FN counts per category of column `by`."""
    rows = []
    for value, g in frame.groupby(by):
        if len(g) < min_n:                     # too small to trust
            continue
        pos = g[g[y_col] == 1]
        rows.append({
            "segment": f"{by}={value}", "n": len(g),
            "positive_rate": g[y_col].mean(),
            "error_rate": (g[y_col] != g[pred_col]).mean(),
            "recall": pos[pred_col].mean() if len(pos) else np.nan,
            "fp": int(((g[pred_col] == 1) & (g[y_col] == 0)).sum()),
            "fn": int(((g[pred_col] == 0) & (g[y_col] == 1)).sum()),
        })
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)
