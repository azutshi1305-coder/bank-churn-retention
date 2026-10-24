"""Pure pandas feature transform: the single source of truth for feature definitions.

The Spark job (build_features.py) imports the constants below and re-implements the same
formulas in Spark SQL functions; tests/test_feature_parity.py checks the two agree. The API
(Stage 6) calls transform() directly, so no Spark is needed per request.
"""
import pandas as pd

GEOGRAPHIES = ["France", "Germany", "Spain"]  # fixed lists: never learned from data
GENDERS = ["Female", "Male"]
SENIOR_AGE = 50

RAW_NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
ONE_HOT = [f"Geography_{g}" for g in GEOGRAPHIES]
ENGINEERED = [
    "zero_balance", "balance_to_salary", "products_per_tenure", "is_senior", "active_x_products",
]
# The served feature set. Gender is read from the raw data but is NOT a model input (Stage 5 ablation:
# dropping it cost nothing measurable). It is kept only as audit columns for fairness and proxy checks.
FEATURE_COLUMNS = RAW_NUMERIC + ONE_HOT + ENGINEERED
AUDIT_COLUMNS = [f"Gender_{g}" for g in GENDERS]

_FLOAT_FEATURES = {"Balance", "EstimatedSalary", "balance_to_salary", "products_per_tenure"}
FEATURE_DTYPES = {c: "float64" if c in _FLOAT_FEATURES else "int64" for c in FEATURE_COLUMNS}
AUDIT_DTYPES = {c: "int64" for c in AUDIT_COLUMNS}

DESCRIPTIONS = {
    "zero_balance": "1 if the account balance is exactly 0.",
    "balance_to_salary": "Balance / EstimatedSalary: how much of their income the customer keeps in the bank.",
    "products_per_tenure": "NumOfProducts / (Tenure + 1): product uptake per year of relationship; +1 keeps tenure 0 finite.",
    "is_senior": f"1 if Age >= {SENIOR_AGE} (the highest-churn age band in EDA).",
    "active_x_products": "IsActiveMember * NumOfProducts: product depth, counted only for active members.",
}


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer columns in, model features out (columns and dtypes as in FEATURE_COLUMNS/FEATURE_DTYPES).

    Extra columns (Exited, identifiers, Gender) are ignored. Row index is preserved. An unseen Geography
    gives all-zero one-hot columns. EstimatedSalary must be > 0 (min in the data is 11.58).
    """
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMERIC:
        out[c] = df[c]
    for g in GEOGRAPHIES:
        out[f"Geography_{g}"] = df["Geography"] == g
    out["zero_balance"] = df["Balance"] == 0
    out["balance_to_salary"] = df["Balance"] / df["EstimatedSalary"]
    out["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    out["is_senior"] = df["Age"] >= SENIOR_AGE
    out["active_x_products"] = df["IsActiveMember"] * df["NumOfProducts"]
    return out[FEATURE_COLUMNS].astype(FEATURE_DTYPES)


def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Gender one-hot columns for fairness audits and proxy checks. Never pass these to the model."""
    return pd.DataFrame({f"Gender_{g}": df["Gender"] == g for g in GENDERS}, index=df.index).astype(AUDIT_DTYPES)
