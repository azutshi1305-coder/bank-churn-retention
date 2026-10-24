"""Export the registered champion to artifacts/champion/ so the API runs without mlflow.db or mlruns/.

Run from the repo root:  python -m src.models.export_champion

Writes the model in MLflow model format plus metadata.json (version, run id, training date, feature schema,
test metrics, risk-band behaviour on the test split). Overwrites artifacts/champion/ on each run.
"""
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.api.bands import BAND_DESCRIPTIONS, PROB_CLIP, clip_probabilities, risk_band
from src.features.transform import FEATURE_COLUMNS
from src.models.evaluate import BOOTSTRAP_METRICS
from src.models.registry import ALIAS, MODEL_NAME, ROOT, TRACKING_URI, load_champion, load_test_features

OUT_DIR = ROOT / "artifacts" / "champion"
SCHEMA_PATH = ROOT / "src" / "features" / "feature_schema.json"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run = client.get_run(version.run_id)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}", dst_path=tmp)
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        shutil.copytree(downloaded, OUT_DIR)

    # MLflow records the absolute local path of the source run in MLmodel; it is unused at load time and would leak the machine's path
    mlmodel = OUT_DIR / "MLmodel"
    mlmodel.write_text("".join(line for line in mlmodel.read_text().splitlines(keepends=True) if not line.startswith("artifact_path:")))

    # the exported copy must load with no registry and give the registry model's exact predictions
    X_test, y_test = load_test_features()
    exported = mlflow.sklearn.load_model(str(OUT_DIR))
    p_exported = exported.predict_proba(X_test)[:, 1]
    p_registry = load_champion().predict_proba(X_test)[:, 1]
    max_diff = float(np.abs(p_exported - p_registry).max())
    assert max_diff == 0.0, f"exported model differs from the registry model (max diff {max_diff})"

    p_served = clip_probabilities(p_exported)
    before = {m: float(fn(y_test, p_exported)) for m, fn in BOOTSTRAP_METRICS.items()}
    after = {m: float(fn(y_test, p_served)) for m, fn in BOOTSTRAP_METRICS.items()}
    clip_info = {
        "low": PROB_CLIP[0], "high": PROB_CLIP[1],
        "applied_in": "the API service layer; the registered model and the exported model are unchanged",
        "reason": "isotonic calibration can return exactly 0 or 1, which overstates certainty about an individual customer",
        "test_rows_clipped_low": int((p_exported < PROB_CLIP[0]).sum()), "test_rows_clipped_high": int((p_exported > PROB_CLIP[1]).sum()),
        "test_rows": int(len(p_exported)), "test_metrics_before_clipping": before, "test_metrics_after_clipping": after,
    }
    bands = pd.Series([risk_band(p) for p in p_served])
    band_stats = {b: {"definition": BAND_DESCRIPTIONS[b], "share_of_test_customers": float((bands == b).mean()),
                      "observed_churn_rate_on_test": float(y_test[(bands == b).to_numpy()].mean())}
                  for b in ("low", "medium", "high")}

    schema = json.loads(SCHEMA_PATH.read_text())
    metrics = {k: v for k, v in run.data.metrics.items() if k.startswith(("test_", "cv_"))}
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": str(version.version),
        "alias": ALIAS,
        "run_id": version.run_id,
        "training_date": datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat(timespec="seconds"),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_description": version.description,
        "calibration_method": run.data.params.get("calibration_method"),
        "n_features": len(FEATURE_COLUMNS),
        "feature_schema": {"features": [{"name": f["name"], "dtype": f["dtype"]} for f in schema["features"]],
                           "excluded_from_model": schema["excluded_from_model"]},
        "metrics": metrics,
        "metrics_note": "Registered-model metrics, before serving-time clipping. test_* are on the held-out 20% test split "
                        "(2,000 customers); cv_* are 5-fold CV on train.",
        "probability_clip": clip_info,
        "risk_bands": band_stats,
    }
    assert [f["name"] for f in metadata["feature_schema"]["features"]] == FEATURE_COLUMNS
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"exported {MODEL_NAME} v{version.version} (run {version.run_id[:8]}) to {OUT_DIR.relative_to(ROOT)}")
    print(f"predictions identical to the registry model on {len(X_test):,} test rows (max diff {max_diff})")
    print(f"artifact size: {dir_size_mb(OUT_DIR):.2f} MB; files: {sorted(p.name for p in OUT_DIR.iterdir())}")
    print(f"served probabilities clipped to {PROB_CLIP}: {clip_info['test_rows_clipped_high']} high and {clip_info['test_rows_clipped_low']} low "
          f"of {clip_info['test_rows']:,} test rows")
    for m in before:
        print(f"  {m:8s} before {before[m]:.8f}  after {after[m]:.8f}  change {after[m] - before[m]:+.2e}")
    for b, s in band_stats.items():
        print(f"  {b:6s} {s['share_of_test_customers']:.1%} of test customers, observed churn {s['observed_churn_rate_on_test']:.1%}")


if __name__ == "__main__":
    main()
