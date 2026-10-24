"""Promote the no-Gender model (logged by src/models/ablation.py) to champion in the MLflow registry.

Run from the repo root:  python -m src.models.promote

Registers it as a NEW version of bank-churn-model and moves the 'champion' alias to it. Earlier versions
stay in the registry for comparison (v1 is the original with-Gender champion). Re-running is a no-op if
the alias already points at a version created from the same run.
"""
import mlflow
from mlflow import MlflowClient

from src.models.registry import ALIAS, MODEL_NAME, TRACKING_URI

SOURCE_RUN_NAME = "xgboost_calibrated_no_gender"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=["bank-churn"], filter_string=f"tags.mlflow.runName = '{SOURCE_RUN_NAME}'",
                              order_by=["start_time DESC"])
    if runs.empty:
        raise SystemExit(f"no run named {SOURCE_RUN_NAME}; run python -m src.models.ablation first")
    run_id = runs.iloc[0]["run_id"]

    current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    if current.run_id == run_id:
        print(f"already promoted: {MODEL_NAME} v{current.version} (alias '{ALIAS}')")
        return
    previous = current.version

    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.update_model_version(MODEL_NAME, version.version,
                                description="Calibrated XGBoost without Gender (16 features). Promoted in Stage 5 after an ablation "
                                            "showed no measurable loss; supersedes v1 (with Gender), which is kept for comparison.")
    client.set_model_version_tag(MODEL_NAME, version.version, "supersedes", str(previous))
    client.set_model_version_tag(MODEL_NAME, version.version, "features", "16, no Gender")
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    print(f"registered {MODEL_NAME} v{version.version} from run {run_id[:8]}; alias '{ALIAS}' moved v{previous} -> v{version.version}")
    print("versions in registry:", sorted(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'")))


if __name__ == "__main__":
    main()
