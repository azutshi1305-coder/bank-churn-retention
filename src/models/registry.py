"""MLflow registry location and helpers shared by training, SHAP and the retention analysis."""
from pathlib import Path

import mlflow
import pandas as pd

from src.features.transform import AUDIT_COLUMNS, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "features.parquet"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MODEL_NAME = "bank-churn-model"
ALIAS = "champion"
TARGET = "Exited"


def load_champion():
    """The calibrated champion (a CalibratedClassifierCV) from the registry alias."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")


def load_model_version(version):
    """A specific registered version, e.g. 1 for the original (with-Gender) champion."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def load_split(name, audit=False):
    """(features, Exited array) for 'train' or 'test', in original file order.

    audit=True also returns the Gender audit columns as a third item. They are never model inputs.
    """
    part = pd.read_parquet(DATA_PATH).query("split == @name")
    X, y = part[FEATURE_COLUMNS].reset_index(drop=True), part[TARGET].to_numpy()
    return (X, y, part[AUDIT_COLUMNS].reset_index(drop=True)) if audit else (X, y)


def load_test_features(audit=False):
    """Test split; see load_split."""
    return load_split("test", audit=audit)
