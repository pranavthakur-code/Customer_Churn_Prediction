"""Loading, validating and cleaning the raw customer table."""
import logging

import numpy as np
import pandas as pd

from churn import config

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input data cannot be used (clear, specific failure)."""


def load_raw(path):
    """Read the CSV and fail early with a helpful message if it is unusable."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataValidationError(f"Data file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Data file is empty: {path}") from exc

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    log.info("Loaded %d rows, %d columns from %s", *df.shape, path)
    return df


def clean(df):
    """Return a cleaned copy (types, labels, duplicates, impossible values).

    Imputation is NOT done here: it must be learned from training data only,
    so it lives inside the model pipeline (see model.py).
    """
    out = df.copy()

    n_before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    log.info("Dropped %d duplicate rows", n_before - len(out))

    # Blank strings become NaN, then new customers (tenure 0) were not billed.
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out.loc[out["tenure"] == 0, "TotalCharges"] = 0.0

    # " month-to-month " and "Month-to-month" are the same category.
    out["Contract"] = out["Contract"].str.strip().str.capitalize()

    # Impossible values are data errors -> missing (imputed later).
    bad = out["MonthlyCharges"] > config.MAX_VALID_MONTHLY
    out.loc[bad, "MonthlyCharges"] = np.nan
    log.info("Set %d impossible MonthlyCharges to NaN", int(bad.sum()))

    out[config.TARGET] = (out[config.TARGET] == "Yes").astype(int)
    return out
