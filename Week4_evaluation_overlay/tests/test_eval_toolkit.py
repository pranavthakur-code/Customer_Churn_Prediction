"""Tests for churn.eval_toolkit using cases whose answers are known by hand."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churn import eval_toolkit as tk  # noqa: E402

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([.1, .2, .3, .6, .4, .7, .8, .9])   # hand-checked below


def test_confusion_counts_by_hand():
    s = tk.classification_summary(Y, P, 0.5)
    # predictions: 0 0 0 1 0 1 1 1 -> TN=3 FP=1 FN=1 TP=3
    assert (s["tn"], s["fp"], s["fn"], s["tp"]) == (3, 1, 1, 3)


def test_precision_recall_by_hand():
    s = tk.classification_summary(Y, P, 0.5)
    assert s["precision"] == pytest.approx(3 / 4)
    assert s["recall"] == pytest.approx(3 / 4)
    assert s["accuracy"] == pytest.approx(6 / 8)


def test_perfect_scores():
    s = tk.classification_summary([0, 0, 1, 1], [.1, .2, .8, .9])
    assert s["roc_auc"] == 1.0 and s["f1"] == 1.0 and s["mcc"] == 1.0


def test_threshold_extremes():
    assert tk.classification_summary(Y, P, 0.0)["recall"] == 1.0
    assert tk.classification_summary(Y, P, 1.0)["recall"] == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_bad_threshold_rejected(bad):
    with pytest.raises(ValueError, match="threshold"):
        tk.classification_summary(Y, P, bad)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        tk.classification_summary(Y, P[:-1])


def test_single_class_rejected():
    with pytest.raises(ValueError, match="single class"):
        tk.classification_summary([1, 1, 1], [.2, .5, .9])


def test_nan_probability_rejected():
    with pytest.raises(ValueError, match="NaN"):
        tk.classification_summary(Y, np.r_[P[:-1], np.nan])


def test_bootstrap_interval_contains_estimate_and_is_ordered():
    est, lo, hi = tk.bootstrap_ci(Y, P, roc_auc_score, n_boot=300)
    assert lo <= est <= hi


def test_bootstrap_is_reproducible():
    a = tk.bootstrap_ci(Y, P, roc_auc_score, n_boot=200, seed=3)
    b = tk.bootstrap_ci(Y, P, roc_auc_score, n_boot=200, seed=3)
    assert a == b


def test_slice_report_counts_add_up():
    f = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "y": [0, 1] * 50,
                      "pred": [0, 0] * 25 + [1, 1] * 25})
    r = tk.slice_report(f, "y", "pred", "g", min_n=10).set_index("segment")
    assert r.loc["g=a", "fn"] == 25 and r.loc["g=a", "fp"] == 0
    assert r.loc["g=b", "fp"] == 25 and r.loc["g=b", "fn"] == 0
    assert r["n"].sum() == 100


def test_small_slices_are_dropped():
    f = pd.DataFrame({"g": ["a"] * 60 + ["b"] * 5, "y": [0, 1] * 32 + [0],
                      "pred": [0] * 65})
    assert list(tk.slice_report(f, "y", "pred", "g", min_n=40).segment) == ["g=a"]
