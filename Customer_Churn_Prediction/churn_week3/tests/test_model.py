"""Integration tests: the whole pipeline on a small generated dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest

from churn import config, data, evaluate, model

ROOT = Path(__file__).parents[1]
CSV = ROOT / "data" / "telco_churn_sample.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def split():
    from sklearn.model_selection import train_test_split
    df = data.clean(data.load_raw(CSV))
    X = df.drop(columns=[config.ID_COLUMN, config.TARGET])
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def test_unknown_model_kind_rejected():
    with pytest.raises(ValueError, match="Unknown model kind"):
        model.build_pipeline("svm-typo")


def test_pipeline_fits_and_predicts_probabilities(split):
    X_tr, X_te, y_tr, _ = split
    pipe = model.build_pipeline("logistic").fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_no_nan_leaves_the_preprocessor(split):
    X_tr, X_te, y_tr, _ = split
    pipe = model.build_pipeline("logistic").fit(X_tr, y_tr)
    Z = pipe[:-1].transform(X_te)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    assert not np.isnan(Z).any()


def test_unseen_category_does_not_crash(split):
    X_tr, X_te, y_tr, _ = split
    pipe = model.build_pipeline("logistic").fit(X_tr, y_tr)
    row = X_te.iloc[[0]].copy()
    row["Contract"] = "Three year"          # never seen in training
    assert pipe.predict_proba(row).shape == (1, 2)


def test_same_seed_gives_identical_predictions(split):
    X_tr, X_te, y_tr, _ = split
    a = model.build_pipeline("tree").fit(X_tr, y_tr).predict_proba(X_te)
    b = model.build_pipeline("tree").fit(X_tr, y_tr).predict_proba(X_te)
    assert np.array_equal(a, b)


def test_model_beats_majority_baseline(split):
    X_tr, X_te, y_tr, y_te = split
    pipe = model.build_pipeline("logistic").fit(X_tr, y_tr)
    rep = evaluate.test_report(pipe, X_te, y_te)
    assert rep["roc_auc"] > 0.65     # a random guesser scores 0.50


def test_threshold_search_meets_recall_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(0.5 * y + rng.normal(0.25, 0.2, 500), 0, 1)
    t = evaluate.best_threshold(y, proba, min_recall=0.7)
    assert ((proba >= t)[y == 1]).mean() >= 0.7
