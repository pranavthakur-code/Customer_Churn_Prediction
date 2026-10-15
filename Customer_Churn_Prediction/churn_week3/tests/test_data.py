"""Unit tests for churn.data: every cleaning rule gets its own test."""
import numpy as np
import pytest

from churn import data


def test_duplicates_removed(raw_df):
    assert not data.clean(raw_df).duplicated().any()


def test_blank_total_charges_for_new_customer_becomes_zero(raw_df):
    out = data.clean(raw_df)
    assert out.loc[out.customerID == "A", "TotalCharges"].iloc[0] == 0.0


def test_total_charges_is_numeric(raw_df):
    assert data.clean(raw_df)["TotalCharges"].dtype.kind == "f"


def test_contract_labels_standardised(raw_df):
    labels = set(data.clean(raw_df)["Contract"])
    assert labels == {"Month-to-month", "One year", "Two year"}


def test_impossible_monthly_charge_becomes_nan(raw_df):
    out = data.clean(raw_df)
    assert np.isnan(out.loc[out.customerID == "D", "MonthlyCharges"].iloc[0])


def test_target_is_binary_integer(raw_df):
    assert set(data.clean(raw_df)["Churn"]) <= {0, 1}


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    data.clean(raw_df)
    assert raw_df.equals(before)


def test_missing_file_gives_clear_error():
    with pytest.raises(data.DataValidationError, match="not found"):
        data.load_raw("no_such_file.csv")


def test_missing_columns_detected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("customerID,tenure\nA,1\n")
    with pytest.raises(data.DataValidationError, match="Missing required"):
        data.load_raw(p)


def test_empty_file_detected(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(data.DataValidationError, match="empty"):
        data.load_raw(p)
