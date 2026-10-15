"""Feature engineering. One function, used identically in train and predict."""
import numpy as np
import pandas as pd


def add_features(df):
    """Create the engineered columns described in the Week 2 strategy."""
    d = df.copy()
    # Typical spend per month of tenure (clip avoids division by zero).
    d["avg_monthly_spend"] = d["TotalCharges"] / d["tenure"].clip(lower=1)
    # Positive when the current bill is above the customer's long-run average.
    d["charge_vs_avg"] = d["MonthlyCharges"] - d["avg_monthly_spend"]
    # Engagement: how many protective add-ons does the customer pay for?
    d["n_addons"] = ((d["OnlineSecurity"] == "Yes").astype(int)
                     + (d["TechSupport"] == "Yes").astype(int))
    d["is_autopay"] = d["PaymentMethod"].str.contains(
        "automatic", na=False).astype(int)
    d["tenure_group"] = pd.cut(
        d["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    # Square root reduces the right skew of TotalCharges (Week 2, step 8).
    d["sqrt_total_charges"] = np.sqrt(d["TotalCharges"])
    return d
