import numpy as np
import pandas as pd
import pytest

from src.features.transform import FEATURE_COLUMNS
from src.monitoring import drift as d


def test_identical_distributions_give_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert d.psi(x, x.copy()) == pytest.approx(0, abs=1e-12)


def test_two_samples_from_the_same_distribution_are_stable():
    rng = np.random.default_rng(1)
    assert d.psi(rng.normal(size=8000), rng.normal(size=2000)) < d.STABLE_BELOW


def test_known_shift_matches_the_hand_computed_value():
    """Binary flag moving from 25% to 50%: (0.5-0.25)ln(0.5/0.25) + (0.5-0.75)ln(0.5/0.75) = 0.2747."""
    ref = np.array([1] * 25 + [0] * 75)
    cur = np.array([1] * 50 + [0] * 50)
    assert d.psi(ref, cur) == pytest.approx(0.25 * np.log(2) + 0.25 * np.log(1.5))


def test_half_sd_mean_shift_is_moderate_to_significant():
    """For equal-variance normals PSI is about shift^2 (0.25 for half an SD); decile bins land just under that."""
    rng = np.random.default_rng(2)
    value = d.psi(rng.normal(0, 1, 50_000), rng.normal(0.5, 1, 50_000))
    assert 0.18 < value < 0.30


def test_psi_is_larger_for_a_larger_shift_and_is_never_negative():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=20_000)
    values = [d.psi(ref, rng.normal(shift, 1, 20_000)) for shift in (0.0, 0.2, 0.5, 1.0)]
    assert values == sorted(values) and min(values) >= 0


def test_unseen_category_and_empty_bin_stay_finite():
    ref = np.array([0] * 90 + [1] * 10)
    cur = np.array([0] * 50 + [2] * 50)   # 1 vanished, 2 never seen in the reference
    assert np.isfinite(d.psi(ref, cur)) and d.psi(ref, cur) > d.SIGNIFICANT_ABOVE


def test_values_outside_the_reference_range_land_in_the_outer_bins():
    ref = np.linspace(0, 1, 1000)
    assert d.psi(ref, np.full(1000, 50.0)) > d.SIGNIFICANT_ABOVE


def test_spike_in_the_reference_does_not_break_binning():
    """36% of balances are exactly 0: repeated quantile edges must collapse, not crash or double count."""
    rng = np.random.default_rng(4)
    ref = np.where(rng.random(10_000) < 0.36, 0.0, rng.normal(120_000, 30_000, 10_000))
    assert d.psi(ref, ref[::-1].copy()) == pytest.approx(0, abs=1e-12)


def test_missing_values_are_ignored_and_empty_input_is_an_error():
    assert d.psi([1, 2, 3, np.nan] * 10, [1, 2, 3] * 10) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        d.psi([np.nan], [1.0])


@pytest.mark.parametrize("value,band", [(0.0, "stable"), (0.099, "stable"), (0.10, "moderate"), (0.25, "moderate"),
                                        (0.2501, "significant")])
def test_bands(value, band):
    assert d.psi_band(value) == band


def _frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    geo = rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25])
    X = pd.DataFrame({c: np.zeros(n, dtype="int64") for c in FEATURE_COLUMNS})
    X["Age"], X["IsActiveMember"], X["NumOfProducts"] = age, rng.integers(0, 2, n), rng.integers(1, 4, n)
    X["Balance"] = rng.normal(80_000, 30_000, n).clip(0)
    for g in ("France", "Germany", "Spain"):
        X[f"Geography_{g}"] = (geo == g).astype("int64")
    return d._with_derived(X)


def test_psi_table_is_sorted_and_banded():
    X = _frame()
    Y = d.simulate_older_customers(X, years=10)
    t = d.psi_table(X, Y, ["Age", "Balance", "IsActiveMember"])
    assert t["feature"].iloc[0] == "Age" and t["band"].iloc[0] == "significant"
    assert t["psi"].is_monotonic_decreasing
    assert t.set_index("feature").loc["Balance", "psi"] == pytest.approx(0, abs=1e-12)


def test_older_customers_shifts_age_and_recomputes_is_senior_without_touching_the_input():
    X = _frame()
    before = X.copy()
    Y = d.simulate_older_customers(X, years=5)
    pd.testing.assert_frame_equal(X, before)
    assert (Y["Age"] - X["Age"]).eq(5).all()
    assert (Y["is_senior"] == (Y["Age"] >= 50)).all() and Y["is_senior"].sum() > X["is_senior"].sum()


def test_fewer_active_members_only_lapses_actives_and_updates_the_interaction():
    X = _frame()
    Y = d.simulate_fewer_active(X, lapse_share=0.3, seed=1)
    assert (Y["IsActiveMember"] <= X["IsActiveMember"]).all()
    assert Y["IsActiveMember"].sum() / X["IsActiveMember"].sum() == pytest.approx(0.7, abs=0.03)
    assert (Y["active_x_products"] == Y["IsActiveMember"] * Y["NumOfProducts"]).all()


def test_germany_share_doubles_and_rows_stay_intact():
    X = _frame(n=20_000)
    Y = d.simulate_germany_share(X, factor=2.0, seed=1)
    assert len(Y) == len(X)
    assert Y["Geography_Germany"].mean() == pytest.approx(2 * X["Geography_Germany"].mean(), abs=0.02)
    assert (Y[["Geography_France", "Geography_Germany", "Geography_Spain"]].sum(axis=1) == 1).all()
    with pytest.raises(ValueError):
        d.simulate_germany_share(X, factor=10)


def _psi_frame(**values):
    return pd.DataFrame({"feature": list(values), "psi": list(values.values())})


def test_retrain_rule():
    key = ["Age", "NumOfProducts"]
    assert d.retrain_decision(_psi_frame(Age=0.02, NumOfProducts=0.01, Tenure=0.0), 0.01, key)[0] == "OK"
    assert d.retrain_decision(_psi_frame(Age=0.02, NumOfProducts=0.01), 0.30, key)[0] == "RETRAIN"          # score
    assert d.retrain_decision(_psi_frame(Age=0.34, NumOfProducts=0.01), 0.01, key)[0] == "RETRAIN"          # key feature
    assert d.retrain_decision(_psi_frame(Age=0.02, Tenure=0.40), 0.01, key)[0] == "INVESTIGATE"             # non-key feature
    assert d.retrain_decision(_psi_frame(Age=0.15, Tenure=0.0), 0.01, key)[0] == "INVESTIGATE"              # key, moderate
    decision, reasons = d.retrain_decision(_psi_frame(Age=0.02), 0.15, key)
    assert decision == "INVESTIGATE" and "score PSI" in reasons[0]


def test_campaign_impact_scales_to_1000_customers_and_reports_churn_rate():
    X = _frame(n=2000)
    p = np.full(len(X), 0.2)
    a = {"base": {"net_interest_margin": 0.025, "annual_fee_per_product": 40, "offer_cost": 100, "retention_success_rate": 0.15}}
    out = d.campaign_impact(X, p, a, capacity_share=0.10)
    assert out["expected_churn_rate_pct"].iloc[0] == pytest.approx(20)
    assert out["contacted_per_1000"].iloc[0] <= 100 + 1e-9   # capacity 10% of customers
