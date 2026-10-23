"""SHAP drivers of the champion model, and the lever each feature belongs to."""
import numpy as np
import shap

LEVERS = {
    "Customer": ["CreditScore", "Age", "Geography_France", "Geography_Germany", "Geography_Spain",
                 "Gender_Female", "Gender_Male", "is_senior", "EstimatedSalary"],
    "Product": ["NumOfProducts", "HasCrCard", "Balance", "zero_balance", "balance_to_salary", "products_per_tenure"],
    "Engagement": ["IsActiveMember", "Tenure", "active_x_products"],
}
LEVER_OF = {feature: lever for lever, features in LEVERS.items() for feature in features}


def champion_shap(champion, X):
    """SHAP values (log-odds) averaged over the XGBoost models inside the calibrated champion.

    The champion is a CalibratedClassifierCV: one XGBoost model per CV fold, each with its own isotonic
    calibrator. Calibration changes probabilities, not the log-odds the trees produce, so we explain the
    trees. Returns (values[n, p], base_value, mean_margin[n]); values.sum(1) + base_value == mean_margin.
    """
    values, bases, margins = [], [], []
    for model in (c.estimator for c in champion.calibrated_classifiers_):
        explainer = shap.TreeExplainer(model)
        values.append(explainer.shap_values(X))
        bases.append(float(np.ravel(explainer.expected_value)[0]))  # only valid after shap_values() has run
        margins.append(model.predict(X, output_margin=True))
    return np.mean(values, axis=0), float(np.mean(bases)), np.mean(margins, axis=0)
