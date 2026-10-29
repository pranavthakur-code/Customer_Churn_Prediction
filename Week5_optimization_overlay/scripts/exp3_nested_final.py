"""Experiment 3: honest estimate of the whole tuning procedure, then the one-time test.

A. Nested CV: tuning (random search, 25 trials, 3-fold) happens INSIDE each outer
   training fold, so each outer score comes from data that never influenced tuning.
B. Final model chosen by development CV only; sealed test set used once.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn import config, data, eval_toolkit as tk, model, tuning

OUT = Path("reports")
df = data.clean(data.load_raw("data/telco_churn_sample.csv"))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)

print("### E3-A. Nested cross-validation (5 outer folds x [25 random trials x 3 inner folds])")
OUTER = StratifiedKFold(5, shuffle=True, random_state=7)
INNER = StratifiedKFold(3, shuffle=True, random_state=0)
rows = []
for k, (i_a, i_v) in enumerate(OUTER.split(X_tr, y_tr)):
    Xa, ya, Xv, yv = X_tr.iloc[i_a], y_tr.iloc[i_a], X_tr.iloc[i_v], y_tr.iloc[i_v]
    rng = np.random.default_rng(100 + k); best_cfg, best_in = None, -1
    for _ in range(25):
        cfg = tuning.sample_random(rng)
        m, _, _ = tuning.cv_score(cfg, Xa, ya, INNER)
        if m > best_in:
            best_cfg, best_in = cfg, m
    auc = lambda est: roc_auc_score(yv, est.fit(Xa, ya).predict_proba(Xv)[:, 1])
    rows.append({"fold": k + 1, "inner_best_CV": best_in,
                 "tuned_HGB": auc(tuning.build_hgb(best_cfg)),
                 "default_HGB": auc(tuning.build_hgb({}, max_iter=100)),
                 "logistic_C0.1": auc(model.build_pipeline("logistic", C=0.1))})
nr = pd.DataFrame(rows)
print(nr.round(4).to_string(index=False))
mean = nr.drop(columns="fold").mean(); sd = nr.drop(columns="fold").std()
print("Mean over outer folds:", mean.round(4).to_dict())
print("Optimism of the tuning score (inner best minus outer score): %+.4f" % (mean.inner_best_CV - mean.tuned_HGB))
d1 = nr.tuned_HGB - nr.default_HGB; d2 = nr.tuned_HGB - nr["logistic_C0.1"]
print("Tuned HGB minus default HGB:  mean %+.4f, wins %d/5" % (d1.mean(), (d1 > 0).sum()))
print("Tuned HGB minus logistic:     mean %+.4f, wins %d/5" % (d2.mean(), (d2 > 0).sum()))

print("\n### E3-B. Final comparison on the sealed test set (used once)")
T = pd.read_csv(OUT / "trials_exp1.csv")
bestrow = T[T.method == "optuna_tpe"].sort_values("cv_auc", ascending=False).iloc[0]
final_cfg = {p: (int(bestrow[p]) if tuning.SPACE[p][0] == "int" else float(bestrow[p])) for p in tuning.SPACE}
print("Chosen from development CV only (best Optuna trial):",
      {k: round(v, 4) for k, v in final_cfg.items()}, "| CV AUC %.4f" % bestrow.cv_auc)
models = {"Logistic (Week 3 baseline)": model.build_pipeline("logistic", C=0.1),
          "Gradient boosting, defaults": tuning.build_hgb({}, max_iter=100),
          "Gradient boosting, tuned": tuning.build_hgb(final_cfg)}
res, probas = {}, {}
for name, pipe in models.items():
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=CV, method="predict_proba", n_jobs=2)[:, 1]
    ts = np.round(np.arange(.10, .91, .01), 2)
    from sklearn.metrics import precision_score, recall_score
    ok = [t for t in ts if recall_score(y_tr, oof >= t) >= .70]
    thr = max(ok, key=lambda t: precision_score(y_tr, oof >= t)) if ok else .5
    p = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]; probas[name] = p
    s = tk.classification_summary(y_te, p, float(thr))
    est, lo, hi = tk.bootstrap_ci(y_te, p, roc_auc_score, n_boot=1000, seed=0)
    res[name] = {"threshold": thr, "ROC-AUC": s["roc_auc"], "AUC 95% CI": f"{lo:.3f}-{hi:.3f}",
                 "PR-AUC": s["pr_auc"], "recall": s["recall"], "precision": s["precision"],
                 "F1": s["f1"], "Brier": s["brier"], "FN": s["fn"], "FP": s["fp"]}
print(pd.DataFrame(res).T.to_string(float_format=lambda v: f"{v:.3f}"))

a, b = probas["Gradient boosting, tuned"], probas["Logistic (Week 3 baseline)"]
rng = np.random.default_rng(1); diffs = []
yt = y_te.values
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    if yt[i].min() == yt[i].max():
        continue
    diffs.append(roc_auc_score(yt[i], a[i]) - roc_auc_score(yt[i], b[i]))
print("Paired bootstrap, AUC(tuned GBM) - AUC(logistic) on the test set: %+.4f, 95%% CI [%+.4f, %+.4f]"
      % (roc_auc_score(yt, a) - roc_auc_score(yt, b), *np.percentile(diffs, [2.5, 97.5])))
print("Cost of the search: %d trials, %.0f s of evaluation time" % (len(T), T.seconds.sum()))
