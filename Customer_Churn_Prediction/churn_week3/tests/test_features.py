"""Unit tests for churn.features with hand-computed expected values."""
import pytest

from churn import data, features


@pytest.fixture
def feats(raw_df):
    return features.add_features(data.clean(raw_df))


def test_avg_monthly_spend_is_hand_computed(feats):
    # customer B: 960 total over 12 months -> 80 per month
    assert feats.loc[feats.customerID == "B", "avg_monthly_spend"].iloc[0] == 80


def test_zero_tenure_does_not_divide_by_zero(feats):
    val = feats.loc[feats.customerID == "A", "avg_monthly_spend"].iloc[0]
    assert val == 0.0


def test_addon_count(feats):
    # customer D has both OnlineSecurity and TechSupport
    assert feats.loc[feats.customerID == "D", "n_addons"].iloc[0] == 2


def test_autopay_flag(feats):
    row = feats.set_index("customerID")
    assert row.loc["B", "is_autopay"] == 1 and row.loc["A", "is_autopay"] == 0


def test_missing_payment_method_is_handled(feats):
    assert feats.loc[feats.customerID == "D", "is_autopay"].iloc[0] == 0


def test_tenure_group_boundaries(feats):
    row = feats.set_index("customerID")["tenure_group"]
    assert row["A"] == "0-12" and row["B"] == "0-12" and row["C"] == "25-48"
    assert row["D"] == "49-72"


def test_input_not_modified(raw_df):
    cleaned = data.clean(raw_df)
    cols = list(cleaned.columns)
    features.add_features(cleaned)
    assert list(cleaned.columns) == cols
