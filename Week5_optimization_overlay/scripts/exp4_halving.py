"""Experiment 4: successive halving (multi-fidelity) versus plain random search."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
warnings.filterwarnings("ignore")
from scipy.stats import loguniform, randint
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split

from churn import config, data, tuning

df = data.clean(data.load_raw("data/telco_churn_sample.csv"))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)
dist = {"model__learning_rate": loguniform(0.01, 0.30), "model__max_depth": randint(2, 9),
        "model__min_samples_leaf": randint(5, 101), "model__l2_regularization": loguniform(1e-3, 10.0)}
hs = HalvingRandomSearchCV(tuning.build_hgb(), dist, n_candidates=81, factor=3,
                           resource="n_samples", min_resources=100, max_resources=len(X_tr),
                           scoring="roc_auc", cv=CV, random_state=0, n_jobs=2, refit=False)
t0 = time.perf_counter(); hs.fit(X_tr, y_tr); t = time.perf_counter() - t0
print("### E4. Successive halving (81 candidates, factor 3, resource = training rows)")
print("Rounds:", hs.n_iterations_)
print("Candidates per round:", hs.n_candidates_)
print("Training rows per round:", hs.n_resources_)
print("Total CV fits: %d (random search with 81 trials = %d fits)" % (sum(hs.n_candidates_) * 5, 81 * 5))
print("Wall-clock: %.1f s" % t)
bp = {k.replace("model__", ""): (int(v) if k.endswith(("depth", "leaf")) else float(v)) for k, v in hs.best_params_.items()}
m, s, tt = tuning.cv_score(bp, X_tr, y_tr, CV)
print("Best configuration:", {k: round(v, 4) for k, v in bp.items()})
print("Re-scored on the full development set with the same folds: CV AUC %.4f (sd %.4f)" % (m, s))
T = pd.read_csv("reports/trials_exp1.csv"); r = T[(T.method == "random") & (T.seed == 0)]
print("Random search, seed 0, 81 trials: best CV AUC %.4f, %.1f s of evaluation time" % (r.cv_auc.max(), r.seconds.sum()))
