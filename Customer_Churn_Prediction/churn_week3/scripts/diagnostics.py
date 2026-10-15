"""Debugging and verification demos used in the Week 3 report."""
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
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from churn import config, data, evaluate, model

CSV = "data/telco_churn_sample.csv"

print("### A. Error handling")
for label, call in [
    ("missing file", lambda: data.load_raw("nope.csv")),
    ("unknown model", lambda: model.build_pipeline("svm-typo")),
]:
    try:
        call()
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
Path("reports/bad.csv").write_text("customerID,tenure\nA,1\n")
try:
    data.load_raw("reports/bad.csv")
except data.DataValidationError as e:
    print(f"bad columns: {type(e).__name__}: {str(e)[:70]}...")
Path("reports/bad.csv").unlink()

df = data.clean(data.load_raw(CSV))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("\n### B. Sanity check: shuffled labels must score about 0.50")
# One shuffle is noisy (sd ~0.07 here), so average over 30 different shuffles.
aucs = []
for seed in range(30):
    y_shuf = pd.Series(np.random.default_rng(seed).permutation(y_tr.values),
                       index=y_tr.index)
    p = model.build_pipeline("logistic").fit(X_tr, y_shuf)
    aucs.append(roc_auc_score(y_te, p.predict_proba(X_te)[:, 1]))
print("ROC-AUC, 30 shuffled-label runs: mean %.3f  sd %.3f  (range %.3f-%.3f)"
      % (np.mean(aucs), np.std(aucs), min(aucs), max(aucs)))
print("Real labels give %.3f -> far outside the shuffled range" % 0.809)

print("\n### C. Over-fitting check: train vs held-out AUC")
for d in (3, 6, 12):
    p = model.build_pipeline("tree", max_depth=d).fit(X_tr, y_tr)
    a = roc_auc_score(y_tr, p.predict_proba(X_tr)[:, 1])
    b = roc_auc_score(y_te, p.predict_proba(X_te)[:, 1])
    print(f"tree depth={d:<2} train={a:.3f} test={b:.3f} gap={a-b:+.3f}")

print("\n### D. Missing-value stress test")
X_bad = X_te.copy()
X_bad.loc[X_bad.index[:50], "MonthlyCharges"] = np.nan   # 50 missing bills
X_bad.loc[X_bad.index[:50], "PaymentMethod"] = None      # 50 missing methods
fit = model.build_pipeline("logistic").fit(X_tr, y_tr)
p = fit.predict_proba(X_bad)[:, 1]
print("predictions made:", len(p), " NaN in output:", int(np.isnan(p).sum()))

print("\n### E. Interpretation: logistic regression coefficients")
final = model.build_pipeline("logistic", C=0.1).fit(X_tr, y_tr)
names = final.named_steps["prep"].get_feature_names_out()
coef = pd.Series(final.named_steps["model"].coef_[0], index=names)
coef.index = [n.split("__")[1] for n in coef.index]
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(8)
print(top.round(3).to_string())

print("\n### F. Figure")
rep = evaluate.test_report(final, X_te, y_te, 0.52)
proba = final.predict_proba(X_te)[:, 1]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
fpr, tpr, _ = roc_curve(y_te, proba)
ax[0].plot(fpr, tpr, color="#2F5496", lw=2, label=f"AUC = {rep['roc_auc']:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="gray", label="random")
ax[0].set(title="ROC curve (test set)", xlabel="False positive rate",
          ylabel="True positive rate"); ax[0].legend(loc="lower right")
ConfusionMatrixDisplay.from_predictions(
    y_te, (proba >= 0.52).astype(int), ax=ax[1], cmap="Blues",
    display_labels=["Stay", "Churn"], colorbar=False)
ax[1].set_title("Confusion matrix (threshold 0.52)")
t = top.sort_values()
ax[2].barh(t.index, t.values, color=["#C0504D" if v < 0 else "#2F5496" for v in t])
ax[2].set_title("Top coefficients"); ax[2].axvline(0, color="k", lw=.6)
ax[2].tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.savefig("reports/fig_results.png", dpi=150)
print("saved reports/fig_results.png")
