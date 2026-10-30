"""Drift monitoring: Population Stability Index (PSI), simulated drift scenarios and a retrain rule.

The dataset has no time column, so there is no real "later" population to compare against. Two things follow:
  * "actual" drift is train (reference) vs test (current), a random split, so it should be ~0 by construction;
  * every other scenario here is SIMULATED by editing the test customers. Simulated results show how the
    monitoring would react and what the model would do under a shift; they are not observed behaviour.
Simulated churn and campaign figures assume the model's learned relationships still hold (covariate shift only).
They say nothing about concept drift, where the same customer would behave differently.
"""
import numpy as np
import pandas as pd

from src.config.economics import customer_value, expected_net_benefit
from src.features.transform import SENIOR_AGE
from src.models.retention import campaign_summary, select_targets

STABLE_BELOW, SIGNIFICANT_ABOVE = 0.10, 0.25   # the usual PSI bands: <0.1 stable, 0.1-0.25 moderate, >0.25 significant
EPS = 1e-4                                      # floor for a bin's share, so an empty bin does not give log(0)
N_BINS = 10
MAX_CATEGORIES = 12                             # features with this many distinct reference values or fewer are binned by value
SIMULATED = "SIMULATED"


def psi_band(value):
    if value < STABLE_BELOW:
        return "stable"
    return "moderate" if value <= SIGNIFICANT_ABOVE else "significant"


def _shares(counts):
    return np.clip(counts / counts.sum(), EPS, None)


def psi(reference, current, n_bins=N_BINS):
    """PSI = sum over bins of (current% - reference%) * ln(current% / reference%).

    Bins come from the reference only: its deciles for continuous inputs (repeated edges collapse, so a spike such as
    zero balance gets its own bin), or its distinct values when there are few (flags, product counts, tenure). Current
    values outside the reference range fall into the outer bins; a value never seen in the reference gets its own bin.
    Both shares are floored at EPS, so the result is finite but a brand-new category inflates it quickly.
    """
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("PSI needs at least one non-missing value in each sample")
    values = np.unique(ref)
    if len(values) <= MAX_CATEGORIES:
        cats = np.union1d(values, np.unique(cur))
        ref_counts = np.array([(ref == c).sum() for c in cats], dtype=float)
        cur_counts = np.array([(cur == c).sum() for c in cats], dtype=float)
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges = np.concatenate([[-np.inf], edges, [np.inf]])
        ref_counts = np.histogram(ref, edges)[0].astype(float)
        cur_counts = np.histogram(cur, edges)[0].astype(float)
    r, c = _shares(ref_counts), _shares(cur_counts)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_table(reference, current, columns=None):
    """PSI per column (reference and current are DataFrames), largest first."""
    columns = list(columns if columns is not None else reference.columns)
    out = pd.DataFrame({"feature": columns, "psi": [psi(reference[c], current[c]) for c in columns]})
    out["band"] = out["psi"].map(psi_band)
    return out.sort_values("psi", ascending=False, ignore_index=True)


# ------------------------------------------------------------------------------------------------ simulated drift
def _with_derived(X):
    """Recompute the features that are functions of ones we edit, so the frame stays internally consistent."""
    X = X.copy()
    X["is_senior"] = (X["Age"] >= SENIOR_AGE).astype("int64")
    X["active_x_products"] = (X["IsActiveMember"] * X["NumOfProducts"]).astype("int64")
    return X


def simulate_older_customers(X, years=5):
    """SIMULATED: every customer is `years` older (Age, and the Age-derived is_senior flag)."""
    out = X.copy()
    out["Age"] = (out["Age"] + years).astype("int64")
    return _with_derived(out)


def simulate_fewer_active(X, lapse_share=0.30, seed=0):
    """SIMULATED: `lapse_share` of the active members lapse to inactive (IsActiveMember 1 -> 0, active_x_products follows)."""
    out = X.copy()
    rng = np.random.default_rng(seed)
    lapse = (out["IsActiveMember"] == 1).to_numpy() & (rng.random(len(out)) < lapse_share)
    out.loc[lapse, "IsActiveMember"] = 0
    return _with_derived(out)


def simulate_germany_share(X, factor=2.0, seed=0):
    """SIMULATED: Germany's share of customers is multiplied by `factor` by resampling customers (same n, with replacement).

    Resampling keeps each customer's other attributes together, so Germany keeps its own profile (older, higher balances)
    rather than relabelling people. The resample adds a little sampling noise to the PSI of every feature.
    """
    share = float(X["Geography_Germany"].mean())
    target = factor * share
    if not 0 < target < 1:
        raise ValueError(f"cannot scale Germany's share {share:.2f} by {factor}")
    odds_ratio = target * (1 - share) / (share * (1 - target))   # weight of a German customer relative to others
    weights = np.where(X["Geography_Germany"] == 1, odds_ratio, 1.0)
    idx = np.random.default_rng(seed).choice(len(X), size=len(X), replace=True, p=weights / weights.sum())
    return X.iloc[idx].reset_index(drop=True)


SIMULATIONS = {
    "sim_older_customers": ("Older customer base (+5 years of Age)", simulate_older_customers),
    "sim_fewer_active_members": ("30% of active members lapse", simulate_fewer_active),
    "sim_germany_share_doubles": ("Germany's share of customers doubles", simulate_germany_share),
}


def drift_report(X_ref, X_cur, predict):
    """PSI of every model input and of the predicted score, for the actual comparison and each simulation.

    `predict` maps a feature frame to churn probabilities. The reference is X_ref (train) scored by the same model.
    Returns (long table, {scenario: (X, p)}) so callers can reuse the scored populations. The 'actual' scenario is
    train vs test, a random split, so it only checks that the monitoring reads ~0 when nothing has changed.
    """
    p_ref = predict(X_ref)
    populations = {"actual_train_vs_test": ("Actual: train (reference) vs test (current)", False, X_cur)}
    for name, (label, make) in SIMULATIONS.items():
        populations[name] = (f"{SIMULATED}: {label}", True, make(X_cur))
    rows, scored = [], {}
    for name, (label, simulated, X) in populations.items():
        p = predict(X)
        scored[name] = (X, p)
        table = psi_table(X_ref, X, list(X_ref.columns))
        table = table.assign(kind="feature").rename(columns={"feature": "name"})
        score = pd.DataFrame({"name": ["predicted_score"], "psi": [psi(p_ref, p)], "kind": "score"})
        score["band"] = score["psi"].map(psi_band)
        rows.append(pd.concat([table, score], ignore_index=True).assign(scenario=name, description=label, simulated=simulated))
    out = pd.concat(rows, ignore_index=True)
    return out[["scenario", "description", "simulated", "kind", "name", "psi", "band"]], scored


# ------------------------------------------------------------------------------------------------ impact under the model
def campaign_impact(X, p, assumptions, capacity_share=0.10):
    """Expected churn rate and campaign outcome per 1,000 customers for this population, using the model's probabilities.

    One row per scenario (best/base/worst) with the capacity-capped campaign. `actual_*` is not available for simulated
    populations (no real outcomes), so it is not computed here.
    """
    n = len(X)
    rows = []
    for name, a in assumptions.items():
        value = customer_value(X["Balance"].to_numpy(), X["NumOfProducts"].to_numpy(),
                               a["net_interest_margin"], a["annual_fee_per_product"])
        benefit = expected_net_benefit(p, value, a["retention_success_rate"], a["offer_cost"])
        res = campaign_summary(select_targets(f"capacity_{round(capacity_share * 100)}pct", p, benefit), p, value,
                               np.zeros(n, dtype=int), a["retention_success_rate"], a["offer_cost"])
        rows.append({"cost_scenario": name, "expected_churn_rate_pct": 100 * float(np.mean(p)),
                     "contacted_per_1000": res["customers_targeted"] * 1000 / n,
                     "value_retained_per_1000": res["value_retained"] * 1000 / n,
                     "net_value_per_1000": res["net_value"] * 1000 / n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ retrain rule
def retrain_decision(feature_psi, score_psi, key_features):
    """Turn PSI results into an action. The rule, in order:

      RETRAIN      score PSI > 0.25, or any key feature PSI > 0.25
      INVESTIGATE  score PSI 0.10-0.25, or any key feature PSI 0.10-0.25, or any other feature PSI > 0.25
      OK           otherwise

    `feature_psi` is the output of psi_table; `key_features` are the model's strongest drivers (e.g. top 5 by mean |SHAP|).
    PSI only says the inputs or scores moved. Once real outcomes arrive, a fall in ROC-AUC or PR-AUC on recent customers
    is a retrain trigger in its own right; that cannot be tested here because the dataset has no time or follow-up.
    Returns (decision, reasons).
    """
    feature_psi = feature_psi.rename(columns={"name": "feature"})   # drift_report labels the column "name"
    key = feature_psi[feature_psi["feature"].isin(key_features)]
    other = feature_psi[~feature_psi["feature"].isin(key_features)]
    reasons = []
    if score_psi > SIGNIFICANT_ABOVE:
        reasons.append(f"score PSI {score_psi:.2f} > {SIGNIFICANT_ABOVE}")
    reasons += [f"key feature {r.feature} PSI {r.psi:.2f} > {SIGNIFICANT_ABOVE}" for r in key.itertuples() if r.psi > SIGNIFICANT_ABOVE]
    if reasons:
        return "RETRAIN", reasons
    if STABLE_BELOW <= score_psi <= SIGNIFICANT_ABOVE:
        reasons.append(f"score PSI {score_psi:.2f} is moderate")
    reasons += [f"key feature {r.feature} PSI {r.psi:.2f} is moderate" for r in key.itertuples() if r.psi >= STABLE_BELOW]
    reasons += [f"other feature {r.feature} PSI {r.psi:.2f} > {SIGNIFICANT_ABOVE}" for r in other.itertuples() if r.psi > SIGNIFICANT_ABOVE]
    return ("INVESTIGATE" if reasons else "OK"), reasons
