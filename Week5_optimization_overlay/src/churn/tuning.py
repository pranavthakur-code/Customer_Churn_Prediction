"""Hyper-parameter search spaces and ONE shared objective for every method.

Every search method (grid, random, Optuna/TPE, Hyperopt/TPE) calls cv_score()
with the same data, folds, metric and model, so differences between methods
come from how they choose configurations and nothing else.
"""
import itertools
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churn import config
from churn.features import add_features
from churn.model import build_preprocessor

FIXED = {"max_iter": 150}      # held constant in every experiment (a control)

# name: (kind, low, high, sample_on_log_scale)
SPACE = {
    "learning_rate": ("float", 0.01, 0.30, True),
    "max_depth": ("int", 2, 8, False),
    "min_samples_leaf": ("int", 5, 100, True),
    "l2_regularization": ("float", 1e-3, 10.0, True),
}
# Grid: 3 x 3 x 3 x 3 = 81 combinations covering the same ranges coarsely.
GRID = {
    "learning_rate": [0.03, 0.10, 0.30],
    "max_depth": [2, 4, 6],
    "min_samples_leaf": [10, 30, 90],
    "l2_regularization": [0.01, 0.3, 10.0],
}


def grid_points():
    """All grid configurations as a list of dicts, in a fixed order."""
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def sample_random(rng):
    """One configuration drawn uniformly (log-uniform where marked)."""
    cfg = {}
    for name, (kind, lo, hi, log) in SPACE.items():
        v = np.exp(rng.uniform(np.log(lo), np.log(hi))) if log else rng.uniform(lo, hi)
        cfg[name] = int(round(v)) if kind == "int" else float(v)
        if kind == "int":
            cfg[name] = int(min(max(cfg[name], lo), hi))
    return cfg


def validate(cfg):
    """Reject configurations outside the declared space (catches typos early)."""
    for name, v in cfg.items():
        if name not in SPACE:
            raise KeyError(f"unknown hyper-parameter: {name!r}")
        _, lo, hi, _ = SPACE[name]
        if not lo <= v <= hi:
            raise ValueError(f"{name}={v} outside [{lo}, {hi}]")


def build_hgb(cfg=None, **extra):
    """Pipeline: engineered features -> dense preprocessing -> gradient boosting."""
    cfg = dict(cfg or {})
    prep = build_preprocessor(scale=False)
    prep.set_params(sparse_threshold=0.0)        # boosting needs dense input
    est = HistGradientBoostingClassifier(
        random_state=config.RANDOM_STATE, **{**FIXED, **cfg, **extra})
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("prep", prep), ("model", est)])


def cv_score(cfg, X, y, cv, scoring="roc_auc", n_jobs=2):
    """Mean and std of the cross-validated score, plus wall-clock seconds."""
    validate(cfg)
    t0 = time.perf_counter()
    s = cross_val_score(build_hgb(cfg), X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return float(s.mean()), float(s.std()), time.perf_counter() - t0
