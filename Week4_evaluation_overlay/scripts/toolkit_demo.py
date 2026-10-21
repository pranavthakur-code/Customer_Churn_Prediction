"""Apply the reusable toolkit to the Week 3 model (sealed test set, run once)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from churn import config, data, eval_toolkit as tk, model

df = data.clean(data.load_raw("data/telco_churn_sample.csv"))
X = df.drop(columns=[config.ID_COLUMN, config.TARGET]); y = df[config.TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
proba = model.build_pipeline("logistic", C=0.1).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

s = tk.classification_summary(y_te, proba, threshold=0.51)
print(pd.Series({k: round(v, 3) for k, v in s.items()}).to_string())
est, lo, hi = tk.bootstrap_ci(y_te, proba, roc_auc_score)
print(f"\nROC-AUC {est:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
est, lo, hi = tk.bootstrap_ci(y_te, proba, lambda a, b: recall_score(a, b >= 0.51))
print(f"Recall  {est:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
frame = X_te.assign(y=y_te.values, pred=(proba >= 0.51).astype(int))
print("\nWorst slices by error rate:")
print(tk.slice_report(frame, "y", "pred", "Contract").round(3).to_string(index=False))
