# Customer Churn Prediction

A tested, versioned machine learning pipeline that predicts telecom customer
churn with logistic regression (decision tree and majority baseline for
comparison). Built as the Week 3 task of a Machine Learning Engineer
internship (Yuva Intern).

> **Data note:** results come from a *simulated* 3,000-customer dataset that
> follows the public IBM Telco Customer Churn schema, with deliberate data
> defects (duplicates, blanks, inconsistent labels, impossible values). They
> show the method works, not real-world performance. To use the real file,
> pass its path to `--data`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/make_sample_data.py                      # generate sample data
python -m pytest -v                                  # 24 tests
python train.py --data data/telco_churn_sample.csv   # train + evaluate
python scripts/diagnostics.py                        # debugging demos
```

## Results on the simulated sample (v0.1.0)

| Metric (held-out test set) | Value |
|---|---|
| ROC-AUC | 0.809 |
| Recall (churners) | 0.703 |
| Precision (churners) | 0.609 |

The Week 1 ROC-AUC target of 0.84 was **not** reached. The decision threshold
was tuned on training predictions, which is slightly optimistic.

## Layout

```
src/churn/   config, data, features, model, evaluate
tests/       pytest suite (unit + integration)
scripts/     diagnostics.py
train.py     command-line entry point
reports/     run outputs, metadata, figure
```

Design: one scikit-learn `Pipeline` holds feature creation, imputation,
scaling, encoding and the model, so training and prediction apply identical
steps and preprocessing is re-fitted inside every cross-validation fold
(no data leakage). All seeds and constants live in `src/churn/config.py`.
