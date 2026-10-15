"""Model construction: preprocessing + estimator in ONE pipeline object."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier

from churn import config
from churn.features import add_features


def build_preprocessor(scale=True):
    """Impute -> (scale) numeric columns, impute -> one-hot categoricals."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:                       # trees do not need scaling
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), config.NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     drop="if_binary")),
        ]), config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(kind="logistic", **params):
    """Return an unfitted Pipeline. `kind` selects the estimator."""
    if kind == "logistic":
        est = LogisticRegression(
            C=params.pop("C", 1.0), class_weight="balanced",
            max_iter=1000, random_state=config.RANDOM_STATE, **params)
        scale = True
    elif kind == "tree":
        est = DecisionTreeClassifier(
            max_depth=params.pop("max_depth", 4), class_weight="balanced",
            random_state=config.RANDOM_STATE, **params)
        scale = False
    else:
        raise ValueError(f"Unknown model kind: {kind!r} (use 'logistic'/'tree')")
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(scale=scale)),
        ("model", est),
    ])
