"""Shared fixtures: a tiny, hand-built dataset so tests run in milliseconds."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


@pytest.fixture
def raw_df():
    """Six rows containing one example of each known data defect."""
    return pd.DataFrame({
        "customerID": ["A", "B", "C", "D", "E", "E"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 1],
        "Partner": ["Yes", "No", "No", "Yes", "No", "No"],
        "tenure": [0, 12, 30, 60, 5, 5],
        "Contract": [" month-to-month ", "One year", "Two year",
                     "Month-to-month", "ONE YEAR", "ONE YEAR"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "Yes", "No", "No"],
        "TechSupport": ["No", "No", "No internet service", "Yes", "No", "No"],
        "PaymentMethod": ["Mailed check", "Credit card (automatic)",
                          "Electronic check", None, "Mailed check",
                          "Mailed check"],
        "MonthlyCharges": [50.0, 80.0, 20.0, 999.99, 55.0, 55.0],
        "TotalCharges": [" ", "960.0", "600.0", "4000.0", "275.0", "275.0"],
        "Churn": ["No", "Yes", "No", "No", "Yes", "Yes"],
    })
