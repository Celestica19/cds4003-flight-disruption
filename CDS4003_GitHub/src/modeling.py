"""Training-fold-only preprocessing and chronological model comparisons."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from .features import feature_columns

def model_pipeline(name, seed=42, engineered=True):
    numeric, categorical = feature_columns(engineered)
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name != "hist_gradient_boosting":
        numeric_steps.append(("scale", StandardScaler()))
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=128, sparse_output=True)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=128, encoded_missing_value=np.nan)
    pre = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", encoder, categorical)
    ], sparse_threshold=1.0 if name != "hist_gradient_boosting" else 0.0)
    if name == "baseline":
        classifier = DummyClassifier(strategy="prior", random_state=seed)
    elif name == "logistic_sgd":
        classifier = SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1000,
                                   tol=0.001, random_state=seed, early_stopping=False)
    elif name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            categorical_features=[False] * len(numeric) + [True] * len(categorical),
            max_iter=150, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", pre), ("classifier", classifier)])

def parameter_space(name):
    if name == "logistic_sgd":
        return {"classifier__alpha": [0.00001, 0.0001, 0.001, 0.01],
                "classifier__class_weight": [None, "balanced"],
                "classifier__penalty": ["l2", "elasticnet"]}
    if name == "hist_gradient_boosting":
        return {"classifier__learning_rate": [0.04, 0.08, 0.12],
                "classifier__max_leaf_nodes": [15, 31, 63],
                "classifier__l2_regularization": [0.0, 1.0, 10.0],
                "classifier__min_samples_leaf": [30, 100]}
    return {}

def temporal_folds(dates, embargo_days=2):
    """Expanding months, with all flights on a date kept together."""
    dates = pd.to_datetime(dates)
    year = int(dates.dt.year.min())
    folds = []
    for month in [4, 6, 8]:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        train = np.flatnonzero((dates < start - pd.Timedelta(days=embargo_days)).to_numpy())
        validation = np.flatnonzero(((dates >= start) & (dates < end)).to_numpy())
        if not len(train) or not len(validation):
            raise ValueError(f"Insufficient data for temporal fold {month}; use the full year dataset")
        assert dates.iloc[train].max() < dates.iloc[validation].min() - pd.Timedelta(days=embargo_days)
        folds.append((train, validation))
    return folds
