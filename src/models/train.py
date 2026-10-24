"""Train, select, calibrate, evaluate and register the churn model. This is the documented way to rebuild the champion.

Run from the repo root:
    python -m src.models.train                 # logs to experiment "bank-churn", registers a new model version, moves the alias
    python -m src.models.train --no-register   # dry run: logs to "bank-churn-dryrun" only, registers nothing, moves no alias,
                                               # then checks the rebuilt champion against the registered one

Order matters: tuning and selection use the train split only. The test split is loaded inside
final_evaluation(), after the champion has been chosen and calibrated.
"""
import argparse
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models import evaluate as ev
from src.models.registry import ALIAS, MODEL_NAME, TRACKING_URI, load_champion, load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "bank-churn"
DRYRUN_EXPERIMENT = "bank-churn-dryrun"  # dry runs never touch EXPERIMENT: notebooks look runs up by name there

SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 1000
REFERENCE_DUMMIES = ["Geography_France"]
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
# skops (MLflow's default sklearn format) refuses unknown types. These are the only ones our own models use,
# reviewed from skops.io.get_untrusted_types; all come from scikit-learn / xgboost.
TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree", "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration", "sklearn.isotonic.IsotonicRegression",
    "sklearn.model_selection._split.StratifiedKFold",
    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
]
LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}


def ensure_experiment(name):
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)


def reproduction_check(y, p_rebuilt, p_registered, tol=5e-5):
    """Compare a rebuilt model with the registered champion on the same test rows. True if ROC-AUC, PR-AUC and Brier agree to 4 dp."""
    new, ref = ev.score_all(y, p_rebuilt), ev.score_all(y, p_registered)
    rows = {m: {"rebuilt": new[m], "registered": ref[m], "abs diff": abs(new[m] - ref[m])} for m in ("roc_auc", "pr_auc", "brier")}
    print("\nReproduction check against the registered champion (test split):")
    print(pd.DataFrame(rows).T.round(6).to_string())
    print(f"max |probability difference| = {np.abs(np.asarray(p_rebuilt) - np.asarray(p_registered)).max():.2e}")
    ok = all(r["abs diff"] < tol for r in rows.values())
    print("REPRODUCED" if ok else "DOES NOT REPRODUCE")
    return ok


def make_searches(scale_pos_weight, cv):
    logreg = Pipeline([
        ("prep", ColumnTransformer([("drop_reference", "drop", REFERENCE_DUMMIES)], remainder=StandardScaler())),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    forest = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)
    boost = XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                          scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=1)
    common = dict(scoring=SCORING, refit="pr_auc", cv=cv, n_jobs=-1)
    return {
        "logistic_regression": GridSearchCV(logreg, {"clf__C": np.logspace(-3, 2, 11)}, **common),
        "random_forest": RandomizedSearchCV(
            forest,
            {"n_estimators": randint(200, 501), "max_depth": [4, 6, 8, 10, 12],
             "min_samples_leaf": randint(1, 31), "max_features": ["sqrt", 0.5]},
            n_iter=12, random_state=SEED, **common),
        "xgboost": RandomizedSearchCV(
            boost,
            {"n_estimators": randint(150, 601), "max_depth": randint(2, 7), "learning_rate": loguniform(0.01, 0.2),
             "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
             "min_child_weight": randint(1, 11), "reg_lambda": loguniform(0.5, 10)},
            n_iter=20, random_state=SEED, **common),
    }


def cv_summary(search):
    i, r = search.best_index_, search.cv_results_
    return {"cv_pr_auc_mean": float(r["mean_test_pr_auc"][i]), "cv_pr_auc_std": float(r["std_test_pr_auc"][i]),
            "cv_roc_auc_mean": float(r["mean_test_roc_auc"][i])}


def clean_params(params):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def choose_calibration(estimator, X, y, cv):
    """Pick sigmoid vs isotonic by cross-validated Brier score on the train split."""
    briers = {}
    for method in ("sigmoid", "isotonic"):
        model = CalibratedClassifierCV(clone(estimator), method=method, cv=cv)
        briers[method] = float(-cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score", n_jobs=-1).mean())
    return min(briers, key=briers.get), briers


def log_model_run(name, estimator, X_sample, y, p, params, cv_metrics, tags, ci, curves=None, register=False):
    """One MLflow child run: params, CV + test metrics (with bootstrap CIs), plots, and the model with a signature."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tags(tags)
        mlflow.log_params({**params, "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED})
        mlflow.log_metrics(cv_metrics)
        test_metrics = ev.score_all(y, p)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"test_{m}_ci_{side}": v for m, (lo, hi) in ci.items()
                            for side, v in (("low", lo), ("high", hi))})
        figs = {"roc": ev.plot_roc(y, p), "pr": ev.plot_pr(y, p), "confusion_matrix": ev.plot_confusion(y, p),
                "calibration": ev.plot_calibration(y, curves or {"model": p})}
        for fig_name, fig in figs.items():
            mlflow.log_figure(fig, f"plots/{fig_name}.png")
            plt.close(fig)
        sample = X_sample.head(100)
        info = mlflow.sklearn.log_model(
            estimator, name="model", signature=infer_signature(sample, estimator.predict_proba(sample)),
            input_example=sample.head(5), pyfunc_predict_fn="predict_proba", skops_trusted_types=TRUSTED_TYPES,
            registered_model_name=MODEL_NAME if register else None)
        return test_metrics, info


def final_evaluation(fitted, champion_key, calibrated, calib_method, X_train, searches, summaries, register=True):
    """First and only use of the test split. With register=False nothing is registered and the alias is not moved;
    the rebuilt champion is checked against the registered one instead."""
    X_test, y_test = load_split("test")
    rows = []
    probs = {k: m.predict_proba(X_test)[:, 1] for k, m in fitted.items()}
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    cal_key = f"{champion_key}_calibrated"

    draws = ev.bootstrap_metrics(y_test, {**probs, cal_key: p_cal}, N_BOOTSTRAP, SEED)
    cis = {name: {m: ev.percentile_ci(v) for m, v in d.items()} for name, d in draws.items()}

    for key, model in fitted.items():
        metrics, _ = log_model_run(
            key, model, X_train, y_test, probs[key],
            {"model_type": key, **clean_params(searches[key].best_params_), "cv_folds": N_SPLITS,
             "calibrated": False},
            summaries[key], {"stage": "4", "role": "candidate", "model_type": key}, cis[key])
        rows.append({"model": LABELS[key], **summaries[key], **{f"test_{k}": v for k, v in metrics.items()},
                     **{f"test_{m}_ci_{s}": v for m, c in cis[key].items() for s, v in zip(("low", "high"), c)}})

    metrics, info = log_model_run(
        cal_key, calibrated, X_train, y_test, p_cal,
        {"model_type": champion_key, "calibrated": True, "calibration_method": calib_method,
         "calibration_cv_folds": N_SPLITS, **clean_params(searches[champion_key].best_params_)},
        summaries[champion_key], {"stage": "4", "role": "champion", "model_type": champion_key}, cis[cal_key],
        curves={"Before calibration": probs[champion_key], "After calibration": p_cal}, register=register)
    rows.append({"model": f"{LABELS[champion_key]} (calibrated)", **summaries[champion_key],
                 **{f"test_{k}": v for k, v in metrics.items()},
                 **{f"test_{m}_ci_{s}": v for m, c in cis[cal_key].items() for s, v in zip(("low", "high"), c)}})

    # paired bootstrap of champion minus runner-up (by CV PR-AUC): is the gap distinguishable from noise?
    runner_up = sorted(summaries, key=lambda k: summaries[k]["cv_pr_auc_mean"])[-2]
    paired = {}
    for m in ev.BOOTSTRAP_METRICS:
        diff = draws[champion_key][m] - draws[runner_up][m]
        lo, hi = ev.percentile_ci(diff)
        paired[m] = (float(diff.mean()), lo, hi)
        mlflow.log_metrics({f"paired_diff_{m}_mean": paired[m][0], f"paired_diff_{m}_ci_low": lo,
                            f"paired_diff_{m}_ci_high": hi})
    mlflow.log_param("paired_diff_pair", f"{champion_key} minus {runner_up}")
    print(f"\npaired bootstrap, {champion_key} minus {runner_up} (test, {N_BOOTSTRAP} resamples):")
    for m, (mean, lo, hi) in paired.items():
        print(f"  {m:8s} {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    if register:
        MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, info.registered_model_version)
        print(f"registered {MODEL_NAME} v{info.registered_model_version} with alias '{ALIAS}'")
    else:
        print(f"\ndry run: nothing registered, alias '{ALIAS}' not moved")
        reproduction_check(y_test, p_cal, load_champion().predict_proba(X_test)[:, 1])
    return pd.DataFrame(rows)


def main(register=True):
    ensure_experiment(EXPERIMENT if register else DRYRUN_EXPERIMENT)

    X_train, y_train = load_split("train")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"train rows={len(X_train)}, churn rate={y_train.mean():.4f}, scale_pos_weight={scale_pos_weight:.3f}")

    with mlflow.start_run(run_name="stage4-training"):
        searches = make_searches(scale_pos_weight, cv)
        summaries = {}
        for key, search in searches.items():
            search.fit(X_train, y_train)
            summaries[key] = cv_summary(search)
            print(f"{LABELS[key]:20s} CV PR-AUC {summaries[key]['cv_pr_auc_mean']:.4f} "
                  f"(+/- {summaries[key]['cv_pr_auc_std']:.4f})  best params: {clean_params(search.best_params_)}")

        champion_key = max(summaries, key=lambda k: summaries[k]["cv_pr_auc_mean"])
        fitted = {k: s.best_estimator_ for k, s in searches.items()}
        calib_method, briers = choose_calibration(fitted[champion_key], X_train, y_train, cv)
        calibrated = CalibratedClassifierCV(clone(fitted[champion_key]), method=calib_method, cv=cv).fit(X_train, y_train)
        print(f"champion by CV PR-AUC: {LABELS[champion_key]}; calibration={calib_method} (CV Brier {briers})")

        mlflow.log_params({"n_train": len(X_train), "seed": SEED, "cv_folds": N_SPLITS,
                           "scale_pos_weight": round(scale_pos_weight, 4), "selection_metric": "mean CV PR-AUC",
                           "selected_model": champion_key, "calibration_method": calib_method})
        mlflow.log_metrics({f"calibration_cv_brier_{m}": b for m, b in briers.items()})

        table = final_evaluation(fitted, champion_key, calibrated, calib_method, X_train, searches, summaries, register)

    shown = table.set_index("model")[[
        "cv_pr_auc_mean", "cv_roc_auc_mean", "test_roc_auc", "test_pr_auc", "test_brier",
        "test_precision_at_0.5", "test_recall_at_0.5", "test_lift_top10", "test_lift_top20"]]
    print("\n" + shown.round(4).to_string())

    ci_text = pd.DataFrame({
        label: table.apply(lambda r: f"{r[f'test_{m}']:.3f} [{r[f'test_{m}_ci_low']:.3f}, {r[f'test_{m}_ci_high']:.3f}]", axis=1)
        for label, m in (("test ROC-AUC", "roc_auc"), ("test PR-AUC", "pr_auc"), ("test Brier", "brier"))})
    ci_text.index = table["model"]
    print(f"\nTest metrics with bootstrap 95% CIs ({N_BOOTSTRAP} resamples of the test rows):\n" + ci_text.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-register", action="store_true",
                        help='dry run: log to the separate "bank-churn-dryrun" experiment, register nothing, do not move the '
                             "champion alias, and check the rebuilt champion against the registered one")
    main(register=not parser.parse_args().no_register)
