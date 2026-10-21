"""Week 4: evaluation and validation experiments on the Week 3 churn model.

Run from the project root:  python scripts/eval_experiments.py
Prints every result shown in the report and saves the figures to reports/.
The test set is split off first and used only once, in sections S4-S6.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score,
                             log_loss, matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, median_absolute_error)
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config, data, model

BLUE, ORANGE, GREY = "#2F5496", "#ED7D31", "#7F7F7F"
OUT = Path("reports")
RNG = np.random.default_rng(42)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

df = data.clean(data.load_raw("data/telco_churn_sample.csv"))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)       # same split as Week 3
print(f"Development set: {len(X_tr)} rows ({y_tr.mean():.1%} churn). "
      f"Sealed test set: {len(X_te)} rows ({y_te.mean():.1%} churn).")
logit = lambda: model.build_pipeline("logistic", C=0.1)
cv5 = StratifiedKFold(5, shuffle=True, random_state=42)

# ---------------------------------------------------------------- S1
print("\n### S1. Why accuracy alone misleads (development set, 5-fold OOF)")
oof = cross_val_predict(logit(), X_tr, y_tr, cv=cv5, method="predict_proba")[:, 1]
maj = np.zeros(len(y_tr), dtype=int)
rows = {}
for name, pred, proba in [("Always 'stays' (majority)", maj, np.full(len(y_tr), y_tr.mean())),
                          ("Logistic regression, t=0.50", (oof >= .5).astype(int), oof)]:
    rows[name] = {"accuracy": accuracy_score(y_tr, pred),
                  "recall": recall_score(y_tr, pred),
                  "precision": precision_score(y_tr, pred, zero_division=0),
                  "ROC-AUC": roc_auc_score(y_tr, proba)}
print(pd.DataFrame(rows).T.round(3).to_string())

print("\n### S2. Full classification metric panel (OOF on development set)")
pred = (oof >= .5).astype(int)
panel = {
    "accuracy": accuracy_score(y_tr, pred),
    "balanced accuracy": balanced_accuracy_score(y_tr, pred),
    "precision": precision_score(y_tr, pred),
    "recall": recall_score(y_tr, pred),
    "F1": f1_score(y_tr, pred),
    "F2 (recall-weighted)": fbeta_score(y_tr, pred, beta=2),
    "MCC": matthews_corrcoef(y_tr, pred),
    "ROC-AUC": roc_auc_score(y_tr, oof),
    "PR-AUC (avg precision)": average_precision_score(y_tr, oof),
    "log loss": log_loss(y_tr, oof),
    "Brier score": brier_score_loss(y_tr, oof),
}
print(pd.Series(panel).round(3).to_string())
print("(PR-AUC of a random model = churn rate = %.3f)" % y_tr.mean())

# ---------------------------------------------------------------- S3 regression
print("\n### S3. Regression metrics: predict MonthlyCharges")
d = df.dropna(subset=["MonthlyCharges"]).copy()
feat = ["tenure", "SeniorCitizen", "Contract", "InternetService",
        "OnlineSecurity", "TechSupport", "PaymentMethod", "Partner", "gender"]
# TotalCharges is deliberately excluded: it is ~ MonthlyCharges * tenure (leakage).
Xr, yr = d[feat], d["MonthlyCharges"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=.2, random_state=42)
numr, catr = ["tenure", "SeniorCitizen"], [c for c in feat if c not in ("tenure", "SeniorCitizen")]
prep = ColumnTransformer([("n", StandardScaler(), numr),
                          ("c", OneHotEncoder(handle_unknown="ignore"), catr)])
lin = Pipeline([("p", prep), ("m", LinearRegression())]).fit(Xr_tr, yr_tr)
dum = DummyRegressor().fit(Xr_tr, yr_tr)
res = {}
for name, m in [("Mean-only baseline", dum), ("Linear regression", lin)]:
    p = m.predict(Xr_te)
    res[name] = {"MAE": mean_absolute_error(yr_te, p),
                 "MedianAE": median_absolute_error(yr_te, p),
                 "MSE": mean_squared_error(yr_te, p),
                 "RMSE": mean_squared_error(yr_te, p) ** .5,
                 "R2": r2_score(yr_te, p),
                 "MAPE %": 100 * np.mean(np.abs((yr_te - p) / yr_te))}
print(pd.DataFrame(res).T.round(3).to_string())
p = lin.predict(Xr_te); err = np.abs(yr_te.values - p)
bad = err.copy(); bad[0] += 200                      # one 200-unit blunder
print("Effect of ONE extra 200-unit error on %d test rows:" % len(err))
print("  MAE  %.2f -> %.2f (x%.2f)" % (err.mean(), bad.mean(), bad.mean() / err.mean()))
print("  RMSE %.2f -> %.2f (x%.2f)" % ((err**2).mean()**.5, (bad**2).mean()**.5,
      (bad**2).mean()**.5 / (err**2).mean()**.5))

# ---------------------------------------------------------------- S4 fitting
print("\n### S4. Diagnosing under- and over-fitting")
depths = [1, 2, 3, 4, 6, 8, 10, 14]
tr_s, va_s = validation_curve(model.build_pipeline("tree"), X_tr, y_tr,
    param_name="model__max_depth", param_range=depths, cv=cv5, scoring="roc_auc")
vc = pd.DataFrame({"depth": depths, "train_AUC": tr_s.mean(1), "cv_AUC": va_s.mean(1)})
vc["gap"] = vc.train_AUC - vc.cv_AUC
print(vc.round(3).to_string(index=False))
sizes, ltr, lva = learning_curve(logit(), X_tr, y_tr, cv=cv5, scoring="roc_auc",
    train_sizes=[.1, .2, .4, .6, .8, 1.0], random_state=42)
lc = pd.DataFrame({"n_train": sizes, "train_AUC": ltr.mean(1), "cv_AUC": lva.mean(1)})
print("Learning curve, logistic regression:"); print(lc.round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(vc.depth, vc.train_AUC, "-o", color=ORANGE, label="training")
ax[0].plot(vc.depth, vc.cv_AUC, "-o", color=BLUE, label="cross-validation")
ax[0].set(title="Validation curve: tree depth", xlabel="max_depth", ylabel="ROC-AUC")
ax[0].text(1.15, .625, "underfit", color=GREY)
ax[0].text(8.2, .85, "overfit: gap grows", color=GREY); ax[0].legend(frameon=False)
ax[1].plot(lc.n_train, lc.train_AUC, "-o", color=ORANGE, label="training")
ax[1].plot(lc.n_train, lc.cv_AUC, "-o", color=BLUE, label="cross-validation")
ax[1].set(title="Learning curve: logistic regression", xlabel="training rows",
          ylabel="ROC-AUC"); ax[1].set_ylim(.70, .82); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(OUT / "fig_fitting.png", dpi=150); plt.close()

# ---------------------------------------------------------------- S5 validation
print("\n### S5. Validation strategies compared (development set only)")
hold = []
for s in range(30):
    a, b, c, e = train_test_split(X_tr, y_tr, test_size=.2, stratify=y_tr, random_state=s)
    hold.append(roc_auc_score(e, logit().fit(a, c).predict_proba(b)[:, 1]))
kf = [cross_val_score(logit(), X_tr, y_tr, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean() for s in range(30)]
rep = []
for s in range(10):
    sc = cross_val_score(logit(), X_tr, y_tr, scoring="roc_auc",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=s))
    rep.append(sc.mean())
tab = pd.DataFrame({
    "strategy": ["Single 80/20 hold-out", "5-fold stratified CV", "5x5 repeated stratified CV"],
    "fits per estimate": [1, 5, 25], "estimates": [30, 30, 10],
    "mean AUC": [np.mean(hold), np.mean(kf), np.mean(rep)],
    "sd across seeds": [np.std(hold), np.std(kf), np.std(rep)],
    "range": [f"{min(hold):.3f}-{max(hold):.3f}", f"{min(kf):.3f}-{max(kf):.3f}",
              f"{min(rep):.3f}-{max(rep):.3f}"]})
print(tab.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
bp = ax.boxplot([hold, kf, rep], vert=False, widths=.55, patch_artist=True,
                tick_labels=["Single hold-out", "5-fold CV", "5x5 repeated CV"])
for box, col in zip(bp["boxes"], [ORANGE, BLUE, BLUE]):
    box.set(facecolor=col, alpha=.55)
ax.set(xlabel="Estimated ROC-AUC (spread = luck of the split)",
       title="How much does the estimate move with the random seed?")
plt.tight_layout(); plt.savefig(OUT / "fig_validation.png", dpi=150); plt.close()

print("\nStratification on a small imbalanced sample (200 rows, 20 churners):")
idx = np.r_[RNG.choice(np.where(y_tr == 1)[0], 20, replace=False),
            RNG.choice(np.where(y_tr == 0)[0], 180, replace=False)]
ys = y_tr.values[idx]; Xs = X_tr.iloc[idx]
for name, splitter in [("plain KFold(5, shuffle)", KFold(5, shuffle=True, random_state=1)),
                       ("StratifiedKFold(5)", StratifiedKFold(5, shuffle=True, random_state=1))]:
    print(f"  {name:<24} churners per validation fold:",
          [int(ys[v].sum()) for _, v in splitter.split(Xs, ys)])

print("\nNested CV vs naive tuning (decision tree, grid of 27 settings):")
grid = {"model__max_depth": [2, 3, 4, 5, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 5, 20]}
naive = GridSearchCV(model.build_pipeline("tree"), grid, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                     scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
nested = cross_val_score(
    GridSearchCV(model.build_pipeline("tree"), grid, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                 scoring="roc_auc", n_jobs=-1),
    X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=7), scoring="roc_auc")
print("  best params (naive):", naive.best_params_)
print("  naive best CV AUC (used to pick AND report): %.3f" % naive.best_score_)
print("  nested CV AUC (outer folds never used for tuning): %.3f +/- %.3f"
      % (nested.mean(), nested.std()))
print("  optimism of naive estimate: %+.3f" % (naive.best_score_ - nested.mean()))

# ---------------------------------------------------------------- S6 threshold
print("\n### S6. Choosing the decision threshold from out-of-fold predictions")
ts = np.round(np.arange(.10, .91, .01), 2)
P = np.array([precision_score(y_tr, oof >= t, zero_division=0) for t in ts])
R = np.array([recall_score(y_tr, oof >= t) for t in ts])
F = np.array([f1_score(y_tr, oof >= t) for t in ts])
ok = R >= .70
t_recall = ts[ok][np.argmax(P[ok])]
COST_FN, COST_FP = 5.0, 1.0                 # ASSUMED: a missed churner costs 5x a wasted offer
cost = np.array([COST_FN * ((oof < t) & (y_tr == 1)).sum()
                 + COST_FP * ((oof >= t) & (y_tr == 0)).sum() for t in ts])
t_cost = ts[np.argmin(cost)]
t_f1 = ts[np.argmax(F)]
print("Rule                                   threshold  precision  recall   F1")
for name, t in [("Default 0.50", .50), ("Max F1", t_f1),
                ("Max precision s.t. recall>=70%", t_recall),
                (f"Min cost (FN={COST_FN:.0f}x FP)", t_cost)]:
    i = int(np.where(ts == t)[0][0])
    print(f"{name:<38} {t:>8.2f} {P[i]:>10.3f} {R[i]:>7.3f} {F[i]:>6.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ts, P, color=BLUE, label="precision"); ax[0].plot(ts, R, color=ORANGE, label="recall")
ax[0].plot(ts, F, color=GREY, ls="--", label="F1")
ax[0].axvline(t_recall, color="k", lw=.8, ls=":"); ax[0].text(t_recall + .01, .12, f"chosen {t_recall:.2f}")
ax[0].set(title="Metrics vs threshold (out-of-fold)", xlabel="threshold"); ax[0].legend(frameon=False)
ax[1].plot(ts, cost, color=BLUE); ax[1].axvline(t_cost, color="k", lw=.8, ls=":")
ax[1].set(title=f"Assumed cost curve (FN costs {COST_FN:.0f}x FP)", xlabel="threshold",
          ylabel="total cost (FP units)")
plt.tight_layout(); plt.savefig(OUT / "fig_threshold.png", dpi=150); plt.close()

# ---------------------------------------------------------------- S7 final test
print("\n### S7. Final, one-time evaluation on the sealed test set")
final = logit().fit(X_tr, y_tr)
pt = final.predict_proba(X_te)[:, 1]
THR = float(t_recall)
pred = (pt >= THR).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"Threshold {THR:.2f} chosen on out-of-fold training predictions (not in-sample)")
print("ROC-AUC %.3f | recall %.3f | precision %.3f | F1 %.3f | accuracy %.3f"
      % (roc_auc_score(y_te, pt), recall_score(y_te, pred), precision_score(y_te, pred),
         f1_score(y_te, pred), accuracy_score(y_te, pred)))
print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print("Per-class rates: false-positive rate %.3f | false-negative rate %.3f | specificity %.3f"
      % (fp / (fp + tn), fn / (fn + tp), tn / (tn + fp)))
B = 1000; bs = {"ROC-AUC": [], "recall": [], "precision": []}
for _ in range(B):
    i = RNG.integers(0, len(y_te), len(y_te)); yt, pp = y_te.values[i], pt[i]
    if yt.min() == yt.max():
        continue
    bs["ROC-AUC"].append(roc_auc_score(yt, pp))
    bs["recall"].append(recall_score(yt, pp >= THR))
    bs["precision"].append(precision_score(yt, pp >= THR, zero_division=0))
print("Bootstrap 95% confidence intervals (1,000 resamples of the test set):")
for k, v in bs.items():
    print(f"  {k:<10} {np.percentile(v, 2.5):.3f} - {np.percentile(v, 97.5):.3f}")

# ---------------------------------------------------------------- S8 errors
print("\n### S8. Error analysis")
res = X_te.copy(); res["y"] = y_te.values; res["p"] = pt; res["pred"] = pred
res["fp"] = ((res.pred == 1) & (res.y == 0)).astype(int)
res["fn"] = ((res.pred == 0) & (res.y == 1)).astype(int)
res["err"] = (res.pred != res.y).astype(int)
print(f"Overall error rate: {res.err.mean():.3f}")
seg_rows = []
for col in ["Contract", "InternetService", "tenure_group", "PaymentMethod"]:
    if col == "tenure_group":
        res[col] = pd.cut(res.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    for val, g in res.groupby(col):
        pos = g[g.y == 1]
        seg_rows.append({"segment": f"{col}={val}", "n": len(g),
            "churn %": 100 * g.y.mean(), "error %": 100 * g.err.mean(),
            "recall": pos.pred.mean() if len(pos) >= 10 else np.nan,
            "FP count": int(g.fp.sum()), "FN count": int(g.fn.sum())})
seg = pd.DataFrame(seg_rows)
print(seg.round({"churn %": 1, "error %": 1, "recall": 2}).to_string(index=False))
worst = seg[seg.n >= 40].sort_values("error %", ascending=False).head(3)
print("Segments with highest error rate (n>=40):", list(worst.segment))
print("\nFive most confident mistakes:")
cols = ["Contract", "InternetService", "tenure", "MonthlyCharges", "y", "p"]
mis = res[res.err == 1].assign(conf=lambda d: np.abs(d.p - .5)).sort_values("conf", ascending=False)
print(mis[cols].head(5).round(2).to_string())
print("\nCalibration (10 equal-count bins):")
fr, mp = calibration_curve(y_te, pt, n_bins=10, strategy="quantile")
cal = pd.DataFrame({"mean predicted": mp, "observed churn rate": fr}); print(cal.round(3).to_string(index=False))
cal_model = CalibratedClassifierCV(logit(), method="sigmoid", cv=5).fit(X_tr, y_tr)
pc = cal_model.predict_proba(X_te)[:, 1]
print("                      Brier   log loss   ROC-AUC")
print("Uncalibrated         %.3f    %.3f      %.3f" % (brier_score_loss(y_te, pt), log_loss(y_te, pt), roc_auc_score(y_te, pt)))
print("Sigmoid-calibrated   %.3f    %.3f      %.3f" % (brier_score_loss(y_te, pc), log_loss(y_te, pc), roc_auc_score(y_te, pc)))
print("Mean predicted churn: uncalibrated %.3f, calibrated %.3f, actual %.3f" % (pt.mean(), pc.mean(), y_te.mean()))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
pr, rc, _ = precision_recall_curve(y_te, pt)
ax[0].plot(rc, pr, color=BLUE); ax[0].axhline(y_te.mean(), color=GREY, ls="--", lw=1)
ax[0].plot(recall_score(y_te, pred), precision_score(y_te, pred), "o", color=ORANGE, ms=8)
ax[0].text(.02, y_te.mean() + .02, "random model", color=GREY, fontsize=8)
ax[0].set(title="Precision-recall curve (test)", xlabel="recall", ylabel="precision")
ax[1].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
ax[1].plot(mp, fr, "-o", color=BLUE, label="uncalibrated")
fr2, mp2 = calibration_curve(y_te, pc, n_bins=10, strategy="quantile")
ax[1].plot(mp2, fr2, "-o", color=ORANGE, label="sigmoid-calibrated")
ax[1].set(title="Calibration (reliability) curve", xlabel="mean predicted probability",
          ylabel="observed churn rate"); ax[1].legend(frameon=False, fontsize=8)
s2 = seg[seg.n >= 40].sort_values("error %")
ax[2].barh(s2.segment.str.replace("PaymentMethod", "Pay").str.replace("InternetService", "Net"),
           s2["error %"], color=[ORANGE if v >= res.err.mean() * 100 else BLUE for v in s2["error %"]])
ax[2].axvline(res.err.mean() * 100, color="k", lw=.8, ls=":")
ax[2].set(title="Error rate by segment (%)"); ax[2].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_errors.png", dpi=150); plt.close()

# ---------------------------------------------------------------- S9 comparison
print("\n### S9. Is model A really better than model B? (paired comparison, 5x3 CV)")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
a = cross_val_score(logit(), X_tr, y_tr, cv=rcv, scoring="roc_auc")
b = cross_val_score(model.build_pipeline("tree", max_depth=3), X_tr, y_tr, cv=rcv, scoring="roc_auc")
d_ = a - b
print("Logistic AUC %.3f | Tree(depth 3) AUC %.3f | mean paired difference %+.3f (sd %.3f)"
      % (a.mean(), b.mean(), d_.mean(), d_.std(ddof=1)))
print("Logistic wins in %d of %d paired folds" % ((d_ > 0).sum(), len(d_)))
c = cross_val_score(model.build_pipeline("logistic", C=1.0), X_tr, y_tr, cv=rcv, scoring="roc_auc")
e_ = a - c
print("Logistic C=0.1 vs C=1.0: mean difference %+.4f (sd %.4f), C=0.1 wins %d of %d folds"
      % (e_.mean(), e_.std(ddof=1), (e_ > 0).sum(), len(e_)))
print("\nDone. Figures saved to reports/.")
