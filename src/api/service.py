"""Scoring logic, independent of FastAPI. The model is loaded once; features come from src/features/transform.py."""
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.api.bands import clip_probabilities, risk_band
from src.config.economics import customer_value, expected_net_benefit, load_assumptions
from src.features.transform import FEATURE_COLUMNS, transform


N_DRIVERS = 3
ASSUMPTION_NOTE = ("Illustrative assumptions, not bank figures: the dataset has no treatment data, so the retention success rate "
                   "is assumed. Money is in units (no currency).")


class ChampionService:
    def __init__(self, model_path):
        path = Path(model_path)
        if not (path / "MLmodel").exists() or not (path / "metadata.json").exists():
            raise FileNotFoundError(f"no exported champion at {path}; run python -m src.models.export_champion")
        self.metadata = json.loads((path / "metadata.json").read_text())
        self.version = str(self.metadata["model_version"])
        self.features = [f["name"] for f in self.metadata["feature_schema"]["features"]]
        if self.features != FEATURE_COLUMNS:
            raise RuntimeError("exported model's features do not match src/features/transform.py; re-export the champion")
        self.model = mlflow.sklearn.load_model(str(path))
        self.boosters = [c.estimator.get_booster() for c in self.model.calibrated_classifiers_]
        for booster in self.boosters:  # trained with one thread; scoring (esp. exact contributions) is much faster in parallel
            booster.set_param({"nthread": int(os.environ.get("XGB_NTHREADS", os.cpu_count() or 1))})
        scenario = load_assumptions()["base"]
        self.assumptions = {"scenario": "base", **scenario, "units": "units", "note": ASSUMPTION_NOTE}

    def score(self, raw: pd.DataFrame, drivers: bool = True) -> list[dict]:
        """Score raw customers. drivers=False skips the exact per-customer contributions, which dominate the cost."""
        X = transform(raw)
        p = clip_probabilities(self.model.predict_proba(X)[:, 1])   # served probability; band and benefit use it too
        if drivers:
            dmatrix = xgb.DMatrix(X)
            contribs = np.mean([b.predict(dmatrix, pred_contribs=True)[:, :-1] for b in self.boosters], axis=0)
        a = self.assumptions
        value = customer_value(X["Balance"].to_numpy(), X["NumOfProducts"].to_numpy(), a["net_interest_margin"], a["annual_fee_per_product"])
        benefit = expected_net_benefit(p, value, a["retention_success_rate"], a["offer_cost"])
        results = []
        for i in range(len(X)):
            result = {"churn_probability": float(p[i]), "risk_band": risk_band(float(p[i])),
                      "estimated_annual_value": float(value[i]), "expected_net_benefit": float(benefit[i]), "top_drivers": None}
            if drivers:
                top = np.argsort(-np.abs(contribs[i]), kind="stable")[:N_DRIVERS]
                result["top_drivers"] = [{
                    "feature": self.features[j], "value": float(X.iloc[i, j]), "contribution": float(contribs[i, j]),
                    "direction": "increases churn risk" if contribs[i, j] > 0 else "decreases churn risk"} for j in top]
            results.append(result)
        return results
