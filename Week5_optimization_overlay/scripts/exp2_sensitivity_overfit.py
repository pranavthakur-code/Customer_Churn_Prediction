"""Experiment 2: sensitivity, early stopping and the winner's curse (dev data only).

Uses the trial log of Experiment 1 plus a few cheap extra runs.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)

from churn import config, data, model, tuning

BLUE, ORANGE, GREY = "#2F5496", "#ED7D31", "#7F7F7F"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
OUT = Path("reports")
df = data.clean(data.load_raw("data/telco_churn_sample.csv"))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)
T = pd.read_csv(OUT / "trials_exp1.csv")

# ------------------------------------------------------------ A. logistic C sweep
print("### A. Logistic regression: sensitivity to C (5-fold CV, development data)")
rows = []
for C in np.logspace(-3, 2, 11):
    r = cross_validate(model.build_pipeline("logistic", C=float(C)), X_tr, y_tr, cv=CV,
                       scoring="roc_auc", return_train_score=True, n_jobs=2)
    rows.append({"C": C, "train_AUC": r["train_score"].mean(), "cv_AUC": r["test_score"].mean(),
                 "cv_sd": r["test_score"].std()})
lc = pd.DataFrame(rows)
print(lc.round(4).to_string(index=False))
print("Spread of CV AUC for C between 0.01 and 10: %.4f"
      % (lc[(lc.C >= .0099) & (lc.C <= 10.1)].cv_AUC.max() - lc[(lc.C >= .0099) & (lc.C <= 10.1)].cv_AUC.min()))

# ------------------------------------------------------------ B. iterations / early stopping
print("\n### B. Boosting rounds: when does adding trees start to hurt?")
a, b, ya, yb = train_test_split(X_tr, y_tr, test_size=.25, stratify=y_tr, random_state=1)
pipe = tuning.build_hgb({"learning_rate": 0.1, "max_depth": 4}, max_iter=400).fit(a, ya)
Za, Zb = pipe[:-1].transform(a), pipe[:-1].transform(b)
est = pipe.named_steps["model"]
its = list(range(10, 401, 10)); lv, lt = [], []
for k, (pa, pb) in enumerate(zip(est.staged_predict_proba(Za), est.staged_predict_proba(Zb)), 1):
    if k in its:
        lt.append(log_loss(ya, pa[:, 1])); lv.append(log_loss(yb, pb[:, 1]))
best_k = its[int(np.argmin(lv))]
print(f"Validation log loss is lowest at {best_k} rounds ({min(lv):.3f}); "
      f"at 400 rounds it is {lv[-1]:.3f} while training loss is {lt[-1]:.3f}")
auto = tuning.build_hgb({"learning_rate": 0.1, "max_depth": 4}, max_iter=400,
                        early_stopping=True, n_iter_no_change=10,
                        validation_fraction=0.15).fit(a, ya)
print("Built-in early stopping (patience 10) stopped at %d rounds; held-out AUC %.3f vs %.3f for 400 rounds"
      % (auto.named_steps["model"].n_iter_,
         roc_auc_score(yb, auto.predict_proba(b)[:, 1]),
         roc_auc_score(yb, pipe.predict_proba(b)[:, 1])))

# ------------------------------------------------------------ C. winner's curse
print("\n### C. The winner's curse: re-scoring the best configurations on fresh folds")
params = list(tuning.SPACE)
pool = T.sort_values("cv_auc", ascending=False).drop_duplicates(params).head(8)
FRESH = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=123)
rr = []
for _, r in pool.iterrows():
    cfg = {p: (int(r[p]) if tuning.SPACE[p][0] == "int" else float(r[p])) for p in params}
    s = cross_validate(tuning.build_hgb(cfg), X_tr, y_tr, cv=FRESH, scoring="roc_auc", n_jobs=2)["test_score"]
    rr.append({"method": r.method, "original_cv": r.cv_auc, "fresh_cv": s.mean()})
wc = pd.DataFrame(rr); wc["drop"] = wc.fresh_cv - wc.original_cv
print(wc.round(4).to_string(index=False))
print("Mean drop on fresh folds: %+.4f" % wc["drop"].mean())

# one-standard-error rule among Optuna seed-0 trials
o = T[(T.method == "optuna_tpe") & (T.seed == 0)]
best = o.loc[o.cv_auc.idxmax()]; se = best.cv_std / np.sqrt(5)
cand = o[o.cv_auc >= best.cv_auc - se].sort_values(["max_depth", "learning_rate"])
simple = cand.iloc[0]
print("\nOne-standard-error rule (Optuna, seed 0): best %.4f (SE %.4f, depth %d, lr %.3f)"
      % (best.cv_auc, se, best.max_depth, best.learning_rate))
print("  simplest config within 1 SE: AUC %.4f, depth %d, lr %.3f, min_leaf %d  (%d configs qualify)"
      % (simple.cv_auc, simple.max_depth, simple.learning_rate, simple.min_samples_leaf, len(cand)))

# ------------------------------------------------------------ D. sensitivity figures
print("\n### D. Which hyper-parameters matter? (from the 405 random-search trials and the 81-point grid)")
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
S = T[T.method.isin(["random"])]
for a_, p in zip(ax, params):
    a_.scatter(S[p], S.cv_auc, s=8, alpha=.45, color=BLUE)
    if tuning.SPACE[p][3]:
        a_.set_xscale("log")
    a_.set_xlabel(p); a_.axhline(.7311, color=ORANGE, lw=1, ls="--")
ax[0].set_ylabel("5-fold CV ROC-AUC"); ax[0].text(.0105, .7325, "library defaults", color=ORANGE, fontsize=7)
plt.suptitle("Random-search trials (5 seeds x 81): each dot is one configuration", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "fig_sensitivity.png", dpi=150); plt.close()
G = T[T.method == "grid"]
for p in params:
    print(f"  mean CV AUC by {p:<18}", G.groupby(p).cv_auc.mean().round(4).to_dict())
fig, ax = plt.subplots(figsize=(5, 3.4))
hm = G.pivot_table(index="max_depth", columns="learning_rate", values="cv_auc", aggfunc="max")
im = ax.imshow(hm.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(hm.columns)), [f"{c:g}" for c in hm.columns]); ax.set_yticks(range(len(hm.index)), hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax.text(j, i, f"{hm.values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if hm.values[i, j] > hm.values.mean() else "black")
ax.set(xlabel="learning_rate", ylabel="max_depth", title="Grid: best CV AUC per cell")
plt.tight_layout(); plt.savefig(OUT / "fig_heatmap.png", dpi=150); plt.close()

# iteration curve figure
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(its, lt, color=ORANGE, label="training log loss"); ax.plot(its, lv, color=BLUE, label="validation log loss")
ax.axvline(best_k, color="k", lw=.8, ls=":"); ax.text(best_k + 5, max(lv) * .995, f"best: {best_k}", fontsize=8)
ax.set(xlabel="boosting rounds", ylabel="log loss", title="Too many rounds overfit"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(OUT / "fig_rounds.png", dpi=150); plt.close()
print("saved figures")
