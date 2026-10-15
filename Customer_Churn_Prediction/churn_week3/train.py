"""Command-line entry point: python train.py --data data/telco_churn_sample.csv"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from churn import __version__, config, data, evaluate, model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the churn model.")
    ap.add_argument("--data", required=True, help="path to the raw CSV")
    ap.add_argument("--out", default="reports", help="output folder")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = data.clean(data.load_raw(args.data))
    X = df.drop(columns=[config.ID_COLUMN, config.TARGET])
    y = df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)
    print(f"Train rows: {len(X_tr)}  Test rows: {len(X_te)}  "
          f"Churn rate (train): {y_tr.mean():.3f}")

    # --- experiments: every candidate scored by the same CV procedure ---
    candidates = {
        "majority-baseline": None,
        "logistic C=0.1": ("logistic", {"C": 0.1}),
        "logistic C=1.0": ("logistic", {"C": 1.0}),
        "tree depth=3": ("tree", {"max_depth": 3}),
        "tree depth=6": ("tree", {"max_depth": 6}),
    }
    rows = []
    for name, spec in candidates.items():
        if spec is None:
            rows.append({"model": name, "roc_auc": 0.5, "std": 0.0,
                         "recall": 0.0, "precision": 0.0})
            continue
        scores = evaluate.cross_validate_model(
            model.build_pipeline(spec[0], **spec[1]), X_tr, y_tr)
        rows.append({"model": name, "roc_auc": scores["roc_auc"][0],
                     "std": scores["roc_auc"][1],
                     "recall": scores["recall"][0],
                     "precision": scores["precision"][0]})
    table = pd.DataFrame(rows).set_index("model")
    print("\nCross-validated comparison (training data only):")
    print(table.round(3).to_string())

    # --- final model: best CV ROC-AUC; threshold tuned on TRAIN data only ---
    best_name = table["roc_auc"].idxmax()
    kind, params = candidates[best_name]
    pipe = model.build_pipeline(kind, **params)
    pipe.fit(X_tr, y_tr)
    thr = evaluate.best_threshold(y_tr, pipe.predict_proba(X_tr)[:, 1])
    rep = evaluate.test_report(pipe, X_te, y_te, thr)
    print(f"\nSelected model: {best_name}   tuned threshold: {thr}")
    print("Held-out test results:")
    for k in ("roc_auc", "recall", "precision", "f1"):
        print(f"  {k:<10}{rep[k]:.3f}")
    print(f"  confusion  TN={rep['tn']} FP={rep['fp']} "
          f"FN={rep['fn']} TP={rep['tp']}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(pipe, out / "churn_pipeline.joblib")
    meta = {"version": __version__, "model": best_name, "threshold": thr,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": config.RANDOM_STATE, "metrics": rep}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {out/'churn_pipeline.joblib'}, {out/'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
