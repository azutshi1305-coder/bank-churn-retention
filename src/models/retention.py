"""Retention economics: expected net benefit, targeting rules, campaign ROI and report exports.

Run from the repo root to (re)write the CSVs in reports/:  python -m src.models.retention

All money is in "units". The assumptions live in src/config/retention_assumptions.yaml and are
ASSUMPTIONS: in particular the retention success rate, because there is no treatment data. The model
predicts churn risk, not how a customer would respond to an offer.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.registry import ROOT, load_champion, load_test_features

CONFIG_PATH = ROOT / "src" / "config" / "retention_assumptions.yaml"
REPORTS_DIR = ROOT / "reports"
SCENARIOS = ("best", "base", "worst")
RISK_RULES = ("threshold_0.5", "top_10pct", "top_20pct")
RULE_LABELS = {"cost_based": "Unconstrained cost-based (net benefit > 0)", "threshold_0.5": "Risk >= 0.5",
               "top_10pct": "Top 10% by risk", "top_20pct": "Top 20% by risk"}
ADDITIVE = ["customers_targeted", "expected_churners_reached", "actual_churners_reached", "customers_saved",
            "value_retained", "campaign_cost", "net_value"]
FORBIDDEN_COLUMNS = {"CustomerId", "Surname", "RowNumber"}
AGE_BINS, AGE_LABELS = [0, 30, 40, 50, 60, 200], ["18-29", "30-39", "40-49", "50-59", "60+"]


def load_assumptions(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())["scenarios"]


def load_capacity(path=CONFIG_PATH):
    """(default_share, shown_shares) for the contact-capacity constraint."""
    cap = yaml.safe_load(Path(path).read_text())["capacity"]
    return float(cap["default_share"]), tuple(float(x) for x in cap["shown_shares"])


def capacity_rule(share):
    return f"capacity_{round(share * 100)}pct"


def rules_for(shares=None):
    shares = shares if shares is not None else load_capacity()[1]
    return ("cost_based", *[capacity_rule(s) for s in shares], *RISK_RULES)


def rule_label(rule):
    if rule.startswith("capacity_"):
        return f"Capacity cap {rule.split('_')[1][:-3]}% (ranked by net benefit)"
    return RULE_LABELS[rule]


def _k(n, share):
    return int(np.ceil(round(n * share, 6)))  # round first: 2000 * 0.15 must be 300, not 301


def select_capacity(benefit, share):
    """Contact at most `share` of customers: highest expected net benefit first, never a customer with benefit <= 0."""
    benefit = np.asarray(benefit)
    top = np.argsort(-benefit, kind="stable")[:_k(len(benefit), share)]
    mask = np.zeros(len(benefit), dtype=bool)
    mask[top] = benefit[top] > 0
    return mask


def customer_value(balance, n_products, net_interest_margin, annual_fee_per_product):
    """One year of value: Balance x margin + NumOfProducts x fee."""
    return balance * net_interest_margin + n_products * annual_fee_per_product


def expected_net_benefit(p_churn, value, retention_success_rate, offer_cost):
    return p_churn * retention_success_rate * value - offer_cost


def select_targets(rule, p, benefit):
    p = np.asarray(p)
    if rule == "cost_based":
        return np.asarray(benefit) > 0
    if rule == "threshold_0.5":
        return p >= 0.5
    if rule.startswith("capacity_"):
        return select_capacity(benefit, int(rule.split("_")[1][:-3]) / 100)
    if rule in ("top_10pct", "top_20pct"):
        mask = np.zeros(len(p), dtype=bool)
        mask[np.argsort(-p, kind="stable")[:_k(len(p), int(rule.split("_")[1][:-3]) / 100)]] = True
        return mask
    raise ValueError(f"unknown rule {rule}")


def campaign_summary(mask, p, value, churned, retention_success_rate, offer_cost):
    """Expected outcome of contacting the customers in `mask`, using the model's churn probabilities."""
    n = int(mask.sum())
    reached = float(p[mask].sum())
    retained = float((p[mask] * retention_success_rate * value[mask]).sum())
    cost = n * offer_cost
    return {"customers_targeted": n, "pct_targeted": 100 * n / len(p),
            "expected_churners_reached": reached, "actual_churners_reached": int(churned[mask].sum()),
            "customers_saved": reached * retention_success_rate, "value_retained": retained,
            "campaign_cost": cost, "net_value": retained - cost,
            "roi_pct": 100 * (retained - cost) / cost if cost > 0 else np.nan}


def net_value_curve(p, value, retention_success_rate, offer_cost, order="benefit"):
    """Cumulative net value as customers are contacted in descending order of `order` ('benefit' or 'risk')."""
    benefit = expected_net_benefit(p, value, retention_success_rate, offer_cost)
    idx = np.argsort(-(benefit if order == "benefit" else p), kind="stable")
    n = len(p)
    return pd.DataFrame({"customers_targeted": np.arange(0, n + 1), "pct_targeted": 100 * np.arange(0, n + 1) / n,
                         "cumulative_net_value": np.concatenate([[0.0], np.cumsum(benefit[idx])])})


def segment_fields(X):
    geography = X[[c for c in X.columns if c.startswith("Geography_")]].idxmax(axis=1).str.replace("Geography_", "")
    gender = X[[c for c in X.columns if c.startswith("Gender_")]].idxmax(axis=1).str.replace("Gender_", "")
    age_band = pd.cut(X["Age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    return pd.DataFrame({"Geography": geography, "Gender": gender, "age_band": age_band})


def score_test():
    """Test customers scored by the registered champion (no identifiers; test_index is an anonymous row number)."""
    X, y = load_test_features()
    p = load_champion().predict_proba(X)[:, 1]
    keep = ["Age", "NumOfProducts", "IsActiveMember", "HasCrCard", "zero_balance", "Balance", "Tenure", "CreditScore"]
    head = pd.DataFrame({"test_index": np.arange(len(X)), "risk": p, "churned_actual": y})
    return pd.concat([head, segment_fields(X), X[keep]], axis=1)


def build_reports(assumptions=None, capacity_shares=None):
    assumptions = assumptions or load_assumptions()
    capacity_shares = tuple(capacity_shares) if capacity_shares is not None else load_capacity()[1]
    rules = rules_for(capacity_shares)
    scored = score_test()
    p, y, n = scored["risk"].to_numpy(), scored["churned_actual"].to_numpy(), len(scored)
    scale = 1000 / n

    rows, curves = [], []
    for s in SCENARIOS:
        a = assumptions[s]
        value = customer_value(scored["Balance"].to_numpy(), scored["NumOfProducts"].to_numpy(),
                               a["net_interest_margin"], a["annual_fee_per_product"])
        benefit = expected_net_benefit(p, value, a["retention_success_rate"], a["offer_cost"])
        scored[f"customer_value_{s}"] = value
        scored[f"expected_net_benefit_{s}"] = benefit
        scored[f"target_cost_based_{s}"] = benefit > 0
        for share in capacity_shares:
            scored[f"target_{capacity_rule(share)}_{s}"] = select_capacity(benefit, share)
        for rule in rules:
            res = campaign_summary(select_targets(rule, p, benefit), p, value, y, a["retention_success_rate"], a["offer_cost"])
            rows.append({"scenario": s, "rule": rule, "basis": f"test_{n}", **res})
            rows.append({"scenario": s, "rule": rule, "basis": "per_1000",
                         **{k: (v * scale if k in ADDITIVE else v) for k, v in res.items()}})
        for order, label in (("benefit", "ranked by expected net benefit"), ("risk", "ranked by risk only")):
            curves.append(net_value_curve(p, value, a["retention_success_rate"], a["offer_cost"], order)
                          .assign(scenario=s, ranking=label))

    scored["target_threshold_0_5"] = select_targets("threshold_0.5", p, None)
    scored["target_top_10pct"] = select_targets("top_10pct", p, None)
    scored["target_top_20pct"] = select_targets("top_20pct", p, None)
    return {"scored": scored, "summary": pd.DataFrame(rows), "curves": pd.concat(curves, ignore_index=True),
            "assumptions": assumptions}


def write_reports(reports, directory=REPORTS_DIR):
    """Write the CSVs for the Tableau dashboard. Refuses to write identifier columns."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {"scored_test_customers.csv": reports["scored"], "scenario_summary.csv": reports["summary"],
             "net_value_curves.csv": reports["curves"]}
    for name, df in files.items():
        bad = FORBIDDEN_COLUMNS & set(df.columns)
        if bad:
            raise ValueError(f"{name} would contain identifier columns: {sorted(bad)}")
        out = df.copy()
        bool_cols = out.select_dtypes("bool").columns
        out[bool_cols] = out[bool_cols].astype(int)
        out.round(6).to_csv(directory / name, index=False)
    return {name: directory / name for name in files}


if __name__ == "__main__":
    reports = build_reports()
    for name, path in write_reports(reports).items():
        print(f"wrote {path} ({len(pd.read_csv(path))} rows)")
