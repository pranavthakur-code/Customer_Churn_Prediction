"""Tests for churn.tuning: search spaces, sampling and the shared objective."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from churn import config, data, tuning  # noqa: E402

CSV = Path(__file__).parents[1] / "data" / "telco_churn_sample.csv"


def test_grid_has_81_points_all_inside_the_space():
    pts = tuning.grid_points()
    assert len(pts) == 81
    for p in pts:
        tuning.validate(p)                       # raises if out of bounds


def test_grid_points_are_unique():
    pts = {tuple(sorted(p.items())) for p in tuning.grid_points()}
    assert len(pts) == 81


def test_random_sampling_is_reproducible():
    a = [tuning.sample_random(np.random.default_rng(5)) for _ in range(3)]
    b = [tuning.sample_random(np.random.default_rng(5)) for _ in range(3)]
    assert a == b


def test_random_samples_respect_bounds_and_types():
    rng = np.random.default_rng(0)
    for _ in range(300):
        cfg = tuning.sample_random(rng)
        tuning.validate(cfg)
        assert isinstance(cfg["max_depth"], int)
        assert isinstance(cfg["min_samples_leaf"], int)
        assert isinstance(cfg["learning_rate"], float)


def test_log_scale_sampling_is_log_uniform():
    """On a log scale the median sits near the geometric mean of the range."""
    rng = np.random.default_rng(1)
    lr = np.array([tuning.sample_random(rng)["learning_rate"] for _ in range(3000)])
    geo_mean = (0.01 * 0.30) ** 0.5
    assert abs(np.median(lr) - geo_mean) / geo_mean < 0.15


def test_validate_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown hyper-parameter"):
        tuning.validate({"learnin_rate": 0.1})           # typo


@pytest.mark.parametrize("cfg", [{"learning_rate": 5.0}, {"max_depth": 1},
                                 {"l2_regularization": 100.0}])
def test_validate_rejects_out_of_range(cfg):
    with pytest.raises(ValueError, match="outside"):
        tuning.validate(cfg)


def test_build_hgb_applies_fixed_and_chosen_values():
    est = tuning.build_hgb({"learning_rate": 0.2, "max_depth": 3}).named_steps["model"]
    assert est.learning_rate == 0.2 and est.max_depth == 3
    assert est.max_iter == tuning.FIXED["max_iter"]


@pytest.mark.skipif(not CSV.exists(), reason="sample data missing")
def test_cv_score_is_valid_and_reproducible():
    df = data.clean(data.load_raw(CSV)).head(700)
    X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    cfg = {"learning_rate": 0.1, "max_depth": 3}
    m1, s1, t1 = tuning.cv_score(cfg, X, y, cv, n_jobs=1)
    m2, _, _ = tuning.cv_score(cfg, X, y, cv, n_jobs=1)
    assert 0.5 < m1 < 1.0 and s1 >= 0 and t1 > 0
    assert m1 == m2                                       # same seed -> same score
