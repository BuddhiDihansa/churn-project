from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "Telco_customer_churn.xlsx"
ARTIFACT_DIR = ROOT_DIR / "app" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.pkl"
METADATA_PATH = ARTIFACT_DIR / "churn_model_meta.json"

TARGET_COLUMN = "Churn Value"
FEATURE_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
CATEGORICAL_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in NUMERIC_FEATURES]


def _to_string_frame(values: pd.DataFrame) -> pd.DataFrame:
    return values.astype(str)


def load_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_excel(path)
    if "Total Charges" in dataframe.columns:
        dataframe["Total Charges"] = pd.to_numeric(dataframe["Total Charges"], errors="coerce")
    if "CLTV" in dataframe.columns:
        dataframe["CLTV"] = pd.to_numeric(dataframe["CLTV"], errors="coerce")
    return dataframe.dropna(subset=["Total Charges", "CLTV", TARGET_COLUMN]).copy()


def prepare_training_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in dataframe.columns]
    if missing_features:
        raise KeyError(f"Missing expected features: {missing_features}")

    X = dataframe[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    valid_mask = y.notna()
    return X.loc[valid_mask].copy(), y.loc[valid_mask].astype(int).copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(_to_string_frame)),
            ("one_hot", one_hot_encoder),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
    feature_specs: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        if feature in NUMERIC_FEATURES:
            series = pd.to_numeric(dataframe[feature], errors="coerce")
            feature_specs.append(
                {
                    "name": feature,
                    "kind": "numeric",
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "default": float(series.median()),
                    "step": 1.0 if feature == "Tenure Months" else 0.1,
                }
            )
        else:
            options = sorted(dataframe[feature].dropna().astype(str).unique().tolist())
            feature_specs.append(
                {
                    "name": feature,
                    "kind": "categorical",
                    "options": options,
                    "default": options[0] if options else "",
                }
            )

    return {
        "target": TARGET_COLUMN,
        "features": feature_specs,
    }
