
## Week 5: Model optimization and experimentation

Hyper-parameter search for a gradient boosting churn model: grid, random,
Optuna (TPE) and Hyperopt (TPE) with equal budgets, nested cross-validation,
successive halving and a one-time sealed test comparison.

```bash
pip install optuna hyperopt                 # added dependencies (see requirements_week5.txt)
python scripts/exp1_methods.py              # 1,296 trials, ~14 min on 2 cores
python scripts/analyze_exp1.py              # tables and convergence figure
python scripts/exp2_sensitivity_overfit.py  # sensitivity, early stopping, winner's curse
python scripts/exp3_nested_final.py         # nested CV and the one-time test
python scripts/exp4_halving.py              # successive halving
python -m pytest -q                         # Weeks 3, 4 and 5 tests
```

Key result: tuning lifted gradient boosting from 0.731 to 0.771 CV ROC-AUC, but
it only tied the untuned Week 3 logistic regression (0.7715 vs 0.7714) and was
slightly behind on the sealed test set. Data is simulated; see the Week 5 report.
