"""Central configuration: every constant lives here, not scattered in code."""

RANDOM_STATE = 42          # one seed for splits, CV and models -> reproducible
TEST_SIZE = 0.20           # share of rows held out for the final test
CV_FOLDS = 5               # stratified folds for cross-validation
MAX_VALID_MONTHLY = 200.0  # business rule: bills above this are data errors
DECISION_THRESHOLD = 0.50  # default probability cut-off (tuned in evaluate.py)

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "sqrt_total_charges",
    "n_addons", "is_autopay", "SeniorCitizen", "charge_vs_avg",
]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Contract", "InternetService",
    "OnlineSecurity", "TechSupport", "PaymentMethod", "tenure_group",
]
REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "tenure", "Contract",
    "InternetService", "OnlineSecurity", "TechSupport", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
