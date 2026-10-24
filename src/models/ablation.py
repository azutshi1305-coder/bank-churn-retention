"""Gender ablation: the no-Gender XGBoost setup compared with registered v1 (the original with-Gender champion).

Run from the repo root:
    python -m src.models.ablation                 # logs to experiment "bank-churn"
    python -m src.models.ablation --no-register   # dry run: logs to "bank-churn-dryrun" instead and checks that the rebuilt
                                                  # no-Gender model reproduces the registered champion

Same train/test split, same XGBoost search space and seed, same calibration choice and bootstrap as
src/models/train.py. Logged to MLflow as its own run; registry aliases are not touched. This is the analysis
that led to promoting the no-Gender model (v2) to champion; src/models/promote.py does the promotion.
"""
import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.features.transform import AUDIT_COLUMNS, ENGINEERED, ONE_HOT, RAW_NUMERIC
from src.models import evaluate as ev
from src.models.registry import ALIAS, MODEL_NAME, load_champion, load_model_version, load_split
from src.models.train import (DRYRUN_EXPERIMENT, EXPERIMENT, N_BOOTSTRAP, N_SPLITS, SEED, choose_calibration, clean_params,
                              cv_summary, ensure_experiment, log_model_run, make_searches, reproduction_check)

GENDER_COLUMNS = AUDIT_COLUMNS
LEGACY_V1_COLUMNS = RAW_NUMERIC + ONE_HOT + AUDIT_COLUMNS + ENGINEERED  # the 18-column order v1 was trained on
RUN_NAME = "stage5-gender-ablation"


def main(register=True):
    ensure_experiment(EXPERIMENT if register else DRYRUN_EXPERIMENT)
    client = MlflowClient()
    alias_version_before = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version

    X_train, y_train = load_split("train")   # the served feature set: no Gender
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    search = make_searches(float((y_train == 0).sum() / (y_train == 1).sum()), cv)["xgboost"]
    search.fit(X_train, y_train)
    cv_metrics = cv_summary(search)
    method, briers = choose_calibration(search.best_estimator_, X_train, y_train, cv)
    calibrated = CalibratedClassifierCV(clone(search.best_estimator_), method=method, cv=cv).fit(X_train, y_train)
    print(f"no-gender XGBoost: CV PR-AUC {cv_metrics['cv_pr_auc_mean']:.4f}; calibration={method}; params {clean_params(search.best_params_)}")

    X_test, y_test, audit_test = load_split("test", audit=True)   # first look at the test split for this ablation
    p_champion = load_model_version(1).predict_proba(pd.concat([X_test, audit_test], axis=1)[LEGACY_V1_COLUMNS])[:, 1]
    p_ablation = calibrated.predict_proba(X_test)[:, 1]
    draws = ev.bootstrap_metrics(y_test, {"champion": p_champion, "no_gender": p_ablation}, N_BOOTSTRAP, SEED)
    cis = {m: ev.percentile_ci(v) for m, v in draws["no_gender"].items()}

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags({"stage": "5", "role": "ablation"})
        mlflow.log_params({"dropped_features": ",".join(GENDER_COLUMNS), "compared_against": f"{MODEL_NAME} v1 (with Gender)",
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
            rows.append({"metric": m, "v1 (with Gender)": ev.score_all(y_test, p_champion)[m], "no_gender": metrics_ablation[m],
                         "diff (no gender - v1)": float(diff.mean()), "95% CI low": lo, "95% CI high": hi})

    assert client.get_model_version_by_alias(MODEL_NAME, ALIAS).version == alias_version_before, "registry alias changed"
    print(f"\nregistry alias unchanged: {MODEL_NAME} v{alias_version_before} (alias '{ALIAS}')")
    print(f"\nTest set, paired bootstrap ({N_BOOTSTRAP} resamples), model without Gender minus v1 (with Gender):")
    print(pd.DataFrame(rows).set_index("metric").round(4).to_string())

    if not register:
        reproduction_check(y_test, p_ablation, load_champion().predict_proba(X_test)[:, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-register", action="store_true",
                        help='dry run: log to the separate "bank-churn-dryrun" experiment instead of "bank-churn" and check the '
                             "rebuilt no-Gender model against the registered champion")
    main(register=not parser.parse_args().no_register)
