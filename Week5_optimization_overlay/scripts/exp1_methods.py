"""Experiment 1: grid vs random vs Optuna(TPE) vs Hyperopt(TPE) at equal budget.

Controls (identical for every method): development data, 5 fixed CV folds,
metric (ROC-AUC), model (gradient boosting pipeline), fixed max_iter, and a
budget of 81 evaluations. Stochastic methods are repeated over 5 seeds.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
warnings.filterwarnings("ignore")
import optuna
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn import config, data, model, tuning

optuna.logging.set_verbosity(optuna.logging.WARNING)
BUDGET, SEEDS = 81, [0, 1, 2, 3, 4]
OUT = Path("reports")

df = data.clean(data.load_raw("data/telco_churn_sample.csv"))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=config.TEST_SIZE, stratify=y, random_state=42)   # as Weeks 3-4
CV = StratifiedKFold(5, shuffle=True, random_state=42)               # control: fixed folds
rows = []


def record(method, seed, i, cfg, res):
    rows.append({"method": method, "seed": seed, "trial": i, **cfg,
                 "cv_auc": res[0], "cv_std": res[1], "seconds": res[2]})


def run_grid():
    for i, cfg in enumerate(tuning.grid_points()):
        record("grid", 0, i, cfg, tuning.cv_score(cfg, X_tr, y_tr, CV))


def run_random(seed):
    rng = np.random.default_rng(seed)
    for i in range(BUDGET):
        cfg = tuning.sample_random(rng)
        record("random", seed, i, cfg, tuning.cv_score(cfg, X_tr, y_tr, CV))


def run_optuna(seed):
    def objective(trial):
        cfg = {}
        for name, (kind, lo, hi, log) in tuning.SPACE.items():
            cfg[name] = (trial.suggest_int(name, lo, hi, log=log) if kind == "int"
                         else trial.suggest_float(name, lo, hi, log=log))
        res = tuning.cv_score(cfg, X_tr, y_tr, CV)
        record("optuna_tpe", seed, trial.number, cfg, res)
        return res[0]
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    study.optimize(objective, n_trials=BUDGET)
    return study


def run_hyperopt(seed):
    space = {n: (hp.loguniform(n, np.log(lo), np.log(hi)) if log
                 else hp.uniform(n, lo, hi))
             for n, (k, lo, hi, log) in tuning.SPACE.items()}
    counter = {"i": 0}

    def objective(p):
        cfg = {}
        for n, (kind, lo, hi, log) in tuning.SPACE.items():
            v = float(p[n])
            v = min(max(v, lo), hi)                      # clip into the declared range
            cfg[n] = int(round(v)) if kind == "int" else v
        res = tuning.cv_score(cfg, X_tr, y_tr, CV)
        record("hyperopt_tpe", seed, counter["i"], cfg, res); counter["i"] += 1
        return {"loss": -res[0], "status": STATUS_OK}
    fmin(objective, space, algo=tpe.suggest, max_evals=BUDGET, trials=Trials(),
         rstate=np.random.default_rng(seed), show_progressbar=False)


t0 = time.time()
# --- reference points (no tuning) ---
ref = {}
hgb_default = tuning.build_hgb({}, max_iter=100)                # library defaults
ref["hgb_default"] = cross_val_score(hgb_default, X_tr, y_tr, cv=CV, scoring="roc_auc", n_jobs=2)
ref["logistic_C0.1"] = cross_val_score(model.build_pipeline("logistic", C=0.1),
                                       X_tr, y_tr, cv=CV, scoring="roc_auc", n_jobs=2)
ref = {k: [float(v.mean()), float(v.std())] for k, v in ref.items()}
print("Reference (no tuning), 5-fold CV ROC-AUC:", {k: round(v[0], 4) for k, v in ref.items()}, flush=True)

run_grid(); print(f"grid done {time.time()-t0:.0f}s", flush=True)
studies = {}
for s in SEEDS:
    run_random(s); studies[s] = run_optuna(s); run_hyperopt(s)
    print(f"seed {s} done {time.time()-t0:.0f}s", flush=True)

trials = pd.DataFrame(rows)
trials["best_so_far"] = trials.groupby(["method", "seed"])["cv_auc"].cummax()
trials.to_csv(OUT / "trials_exp1.csv", index=False)
json.dump({"reference": ref, "budget": BUDGET, "seeds": SEEDS,
           "total_seconds": time.time() - t0}, open(OUT / "exp1_meta.json", "w"), indent=2)

# parameter importance from the Optuna study of seed 0 (kept for Experiment 2 text)
imp = optuna.importance.get_param_importances(studies[0])
json.dump(imp, open(OUT / "optuna_importance_seed0.json", "w"), indent=2)
print("Done in %.0fs" % (time.time() - t0))
