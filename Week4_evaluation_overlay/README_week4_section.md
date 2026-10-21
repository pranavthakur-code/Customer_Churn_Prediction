
## Week 4: Model evaluation and validation

Adds an evaluation toolkit and the experiments behind the Week 4 report
(metrics, cross-validation strategies, threshold selection, error analysis).

```bash
python scripts/eval_experiments.py   # all experiments, figures -> reports/
python scripts/toolkit_demo.py       # reusable toolkit on the Week 3 model
python -m pytest -q                  # Week 3 + Week 4 tests
```

| File | Purpose |
|---|---|
| `src/churn/eval_toolkit.py` | metric summary, bootstrap CI, slice report |
| `tests/test_eval_toolkit.py` | 13 tests with hand-computed expected values |
| `scripts/eval_experiments.py` | validation, threshold, calibration, error analysis |
| `scripts/make_split_diagram.py` | draws the splitting-strategy diagram |
| `reports/eval_output.txt` | recorded output of the experiments |

The decision threshold is now chosen from out-of-fold predictions (not
in-sample). Results are on simulated data and show the methods work, not
real-world performance.
