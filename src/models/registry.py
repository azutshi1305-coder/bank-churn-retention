"""MLflow registry location and helpers shared by training, SHAP and the retention analysis."""
from pathlib import Path

import mlflow
import pandas as pd

from src.features.transform import FEATURE_COLUMNS

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


def load_test_features():
    """Test split: (features DataFrame, Exited array). Row order follows the original file."""
    df = pd.read_parquet(DATA_PATH)
    part = df[df["split"] == "test"]
    return part[FEATURE_COLUMNS].reset_index(drop=True), part[TARGET].to_numpy()
