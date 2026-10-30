"""Tableau-ready CSVs in reports/tableau/. No identifiers (CustomerId, Surname, RowNumber) are ever written.

Run from the repo root:  python -m src.reporting.tableau

Everything money-related uses the ASSUMPTIONS in src/config/retention_assumptions.yaml (units, not a currency); the
drift file mixes one actual comparison with SIMULATED scenarios and says so in its columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.bands import HIGH_FROM, LOW_BELOW
from src.models.drivers import LEVER_OF, champion_shap
from src.models.registry import ROOT, TARGET, load_champion, load_split
from src.models.retention import AGE_BINS, AGE_LABELS, FORBIDDEN_COLUMNS, SCENARIOS, build_reports, capacity_rule, load_capacity
from src.monitoring.drift import drift_report

TABLEAU_DIR = ROOT / "reports" / "tableau"
DATA_PATH = ROOT / "data" / "processed" / "features.parquet"


def risk_band(p):
    """Same cut points as the API (src/api/bands.py), vectorised."""
    return np.select([p < LOW_BELOW, p < HIGH_FROM], ["low", "medium"], default="high")


def segment_summary():
    """Churn and balance at risk by Geography, age band, NumOfProducts and IsActiveMember, over all 10,000 customers.

    Balance at risk = balance held by customers who churned (the definition used in sql/07_balance_at_risk.sql).
    """
    df = pd.read_parquet(DATA_PATH)
    df["Geography"] = df[[c for c in df.columns if c.startswith("Geography_")]].idxmax(axis=1).str.replace("Geography_", "")
    df["age_band"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    df["IsActiveMember"] = np.where(df["IsActiveMember"] == 1, "Active", "Inactive")
    df["NumOfProducts"] = df["NumOfProducts"].astype(str)
    df["churned_balance"] = df["Balance"] * df[TARGET]
    total_balance = df["Balance"].sum()
    parts = []
    for dimension, column in (("Geography", "Geography"), ("Age band", "age_band"),
                              ("Number of products", "NumOfProducts"), ("Active member", "IsActiveMember")):
        g = df.groupby(column).agg(customers=(TARGET, "size"), churners=(TARGET, "sum"),
                                   segment_balance=("Balance", "sum"), balance_at_risk=("churned_balance", "sum")).reset_index()
        parts.append(g.rename(columns={column: "segment"}).assign(dimension=dimension))
    out = pd.concat(parts, ignore_index=True)
    out["churn_rate_pct"] = 100 * out["churners"] / out["customers"]
    out["pct_of_segment_balance_at_risk"] = 100 * out["balance_at_risk"] / out["segment_balance"].where(out["segment_balance"] > 0)
    out["share_of_total_balance_at_risk"] = 100 * out["balance_at_risk"] / total_balance
    cols = ["dimension", "segment", "customers", "churners", "churn_rate_pct", "segment_balance", "balance_at_risk",
            "pct_of_segment_balance_at_risk", "share_of_total_balance_at_risk"]
    return out[cols]


def drivers(champion, X):
    sv, _, _ = champion_shap(champion, X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns, name="mean_abs_shap").sort_values(ascending=False)
    out = imp.rename_axis("feature").reset_index()
    out["lever"] = out["feature"].map(LEVER_OF)
    out["share_pct"] = 100 * out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "feature", "lever", "mean_abs_shap", "share_pct"]]


def scored_customers(scored):
    """One row per test customer per cost scenario (long format, so Tableau filters by scenario instead of juggling columns)."""
    shares = load_capacity()[1]
    base = ["test_index", "risk", "churned_actual", "Geography", "Gender", "age_band", "Age", "NumOfProducts", "IsActiveMember",
            "HasCrCard", "zero_balance", "Balance", "Tenure", "CreditScore"]
    parts = []
    for s in SCENARIOS:
        part = scored[base].copy()
        part.insert(2, "risk_band", risk_band(scored["risk"].to_numpy()))
        part["scenario"] = s
        part["customer_value"] = scored[f"customer_value_{s}"]
        part["expected_net_benefit"] = scored[f"expected_net_benefit_{s}"]
        part["target_unconstrained"] = scored[f"target_cost_based_{s}"].astype(int)
        for share in shares:
            part[f"target_capacity_{round(share * 100)}pct"] = scored[f"target_{capacity_rule(share)}_{s}"].astype(int)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def scenarios(summary):
    """best/base/worst x capacity (10%, 15%, unconstrained). 'test_2000' rows are the test set; 'per_1000' rows are scaled."""
    shares = load_capacity()[1]
    labels = {"cost_based": "Unconstrained", **{capacity_rule(s): f"{round(s * 100)}%" for s in shares}}
    keep = summary[summary["rule"].isin(labels)].copy()
    keep["capacity"] = keep["rule"].map(labels)
    keep = keep.rename(columns={"scenario": "cost_scenario", "rule": "capacity_rule", "customers_targeted": "customers_contacted",
                                "customers_saved": "expected_customers_saved"})
    cols = ["cost_scenario", "capacity", "capacity_rule", "basis", "customers_contacted", "pct_targeted", "expected_churners_reached",
            "actual_churners_reached", "expected_customers_saved", "value_retained", "campaign_cost", "net_value", "roi_pct"]
    keep["cost_scenario"] = pd.Categorical(keep["cost_scenario"], categories=SCENARIOS, ordered=True)
    keep = keep.sort_values(["basis", "cost_scenario", "capacity_rule"])
    keep["cost_scenario"] = keep["cost_scenario"].astype(str)
    return keep[cols].reset_index(drop=True)


def drift_csv(champion):
    X_ref, _ = load_split("train")
    X_cur, _ = load_split("test")
    table, _ = drift_report(X_ref, X_cur, lambda X: champion.predict_proba(X)[:, 1])
    return table


def build():
    champion = load_champion()
    X_test, _ = load_split("test")
    reports = build_reports()
    return {"segment_summary.csv": segment_summary(),
            "drivers.csv": drivers(champion, X_test),
            "scored_customers.csv": scored_customers(reports["scored"]),
            "scenarios.csv": scenarios(reports["summary"]),
            "drift.csv": drift_csv(champion)}


def write(files, directory=TABLEAU_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in files.items():
        bad = FORBIDDEN_COLUMNS & set(df.columns)
        if bad:
            raise ValueError(f"{name} would contain identifier columns: {sorted(bad)}")
        out = df.copy()
        out[out.select_dtypes("bool").columns] = out.select_dtypes("bool").astype(int)
        out.round(6).to_csv(directory / name, index=False)
    return {name: directory / name for name in files}


if __name__ == "__main__":
    for name, path in write(build()).items():
        print(f"wrote {path} ({len(pd.read_csv(path))} rows)")
