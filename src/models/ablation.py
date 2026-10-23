"""Gender ablation: retrain the champion setup without the Gender features and compare on the test split.

Run from the repo root:  python -m src.models.ablation

Same train/test split, same XGBoost search space and seed, same calibration choice and bootstrap as
src/models/train.py. Logged to MLflow as its own run; the registered champion is not touched.
"""
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.models import evaluate as ev
from src.models.registry import ALIAS, MODEL_NAME, TRACKING_URI, load_champion, load_test_features
from src.models.train import (EXPERIMENT, N_BOOTSTRAP, N_SPLITS, SEED, choose_calibration, clean_params, cv_summary,
                              load_split, log_model_run, make_searches)

GENDER_COLUMNS = ["Gender_Female", "Gender_Male"]
RUN_NAME = "stage5-gender-ablation"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    champion_version_before = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version

    X_full, y_train = load_split("train")
    X_train = X_full.drop(columns=GENDER_COLUMNS)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    search = make_searches(float((y_train == 0).sum() / (y_train == 1).sum()), cv)["xgboost"]
    search.fit(X_train, y_train)
    cv_metrics = cv_summary(search)
    method, briers = choose_calibration(search.best_estimator_, X_train, y_train, cv)
    calibrated = CalibratedClassifierCV(clone(search.best_estimator_), method=method, cv=cv).fit(X_train, y_train)
    print(f"no-gender XGBoost: CV PR-AUC {cv_metrics['cv_pr_auc_mean']:.4f}; calibration={method}; params {clean_params(search.best_params_)}")

    X_test_full, y_test = load_test_features()       # first look at the test split for this ablation
    p_champion = load_champion().predict_proba(X_test_full)[:, 1]
    p_ablation = calibrated.predict_proba(X_test_full.drop(columns=GENDER_COLUMNS))[:, 1]
    draws = ev.bootstrap_metrics(y_test, {"champion": p_champion, "no_gender": p_ablation}, N_BOOTSTRAP, SEED)
    cis = {m: ev.percentile_ci(v) for m, v in draws["no_gender"].items()}

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags({"stage": "5", "role": "ablation"})
        mlflow.log_params({"dropped_features": ",".join(GENDER_COLUMNS), "compared_against": f"{MODEL_NAME}@champion (v{champion_version_before})",
                           "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED, "calibration_method": method})
        metrics_ablation, _ = log_model_run(
            "xgboost_calibrated_no_gender", calibrated, X_train, y_test, p_ablation,
            {"model_type": "xgboost", "calibrated": True, "calibration_method": method, "dropped_features": ",".join(GENDER_COLUMNS),
             **clean_params(search.best_params_)},
            cv_metrics, {"stage": "5", "role": "ablation", "model_type": "xgboost"}, cis,
            curves={"Champion": p_champion, "Without Gender": p_ablation})
        rows = []
        for m in ev.BOOTSTRAP_METRICS:
            diff = draws["no_gender"][m] - draws["champion"][m]
            lo, hi = ev.percentile_ci(diff)
            mlflow.log_metrics({f"paired_diff_{m}_mean": float(diff.mean()), f"paired_diff_{m}_ci_low": lo, f"paired_diff_{m}_ci_high": hi})
            rows.append({"metric": m, "champion": ev.score_all(y_test, p_champion)[m], "no_gender": metrics_ablation[m],
                         "diff (no gender - champion)": float(diff.mean()), "95% CI low": lo, "95% CI high": hi})

    assert client.get_model_version_by_alias(MODEL_NAME, ALIAS).version == champion_version_before, "champion alias changed"
    print(f"\nregistered champion unchanged: {MODEL_NAME} v{champion_version_before} (alias '{ALIAS}')")
    print(f"\nTest set, paired bootstrap ({N_BOOTSTRAP} resamples), model without Gender minus registered champion:")
    print(pd.DataFrame(rows).set_index("metric").round(4).to_string())


if __name__ == "__main__":
    main()
