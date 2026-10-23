import numpy as np
import pandas as pd
import pytest

from src.models import retention as r


def test_value_and_benefit_formulas():
    assert r.customer_value(10_000, 2, 0.02, 50) == 10_000 * 0.02 + 2 * 50
    assert r.expected_net_benefit(0.5, 1000, 0.2, 50) == pytest.approx(50)  # 0.5 * 0.2 * 1000 - 50


def test_campaign_summary_hand_computed():
    p = np.array([0.5, 0.1])
    value = np.array([1000.0, 100.0])
    churned = np.array([1, 0])
    benefit = r.expected_net_benefit(p, value, 0.2, 50)           # [50, -48]
    mask = r.select_targets("cost_based", p, benefit)
    assert mask.tolist() == [True, False]
    s = r.campaign_summary(mask, p, value, churned, 0.2, 50)
    assert s["customers_targeted"] == 1
    assert s["expected_churners_reached"] == pytest.approx(0.5)
    assert s["customers_saved"] == pytest.approx(0.1)
    assert s["value_retained"] == pytest.approx(100)
    assert s["campaign_cost"] == 50 and s["net_value"] == pytest.approx(50)
    assert s["roi_pct"] == pytest.approx(100)
    assert s["actual_churners_reached"] == 1


def test_rules_and_empty_campaign():
    p = np.linspace(0.01, 0.99, 100)
    assert r.select_targets("top_10pct", p, None).sum() == 10
    assert r.select_targets("top_20pct", p, None).sum() == 20
    assert r.select_targets("threshold_0.5", p, None).sum() == (p >= 0.5).sum()
    s = r.campaign_summary(np.zeros(100, dtype=bool), p, np.ones(100), np.zeros(100), 0.2, 50)
    assert s["customers_targeted"] == 0 and np.isnan(s["roi_pct"])


def test_capacity_ranks_by_benefit_and_skips_unprofitable():
    benefit = np.array([-5.0, 40.0, 30.0, 10.0, -1.0])
    assert r.select_capacity(benefit, 0.4).tolist() == [False, True, True, False, False]   # k = 2, best two by benefit
    assert r.select_capacity(benefit, 0.8).sum() == 3                                       # k = 4 but only 3 are profitable
    assert r.select_capacity(benefit, 1.0).tolist() == (benefit > 0).tolist()               # capacity never forces a loss
    p = np.array([0.99, 0.2, 0.3, 0.1, 0.9])                                               # highest risk is customer 0 ...
    assert not r.select_targets("capacity_20pct", p, benefit)[0]                            # ... but it is not unprofitable-contacted


def test_capacity_size_does_not_overshoot_on_float_rounding():
    benefit = np.arange(2000, 0, -1, dtype=float)
    assert r.select_capacity(benefit, 0.10).sum() == 200
    assert r.select_capacity(benefit, 0.15).sum() == 300
    assert r.rules_for((0.10, 0.15)) == ("cost_based", "capacity_10pct", "capacity_15pct", "threshold_0.5", "top_10pct", "top_20pct")


def test_capacity_never_beats_unconstrained_optimum():
    rng = np.random.default_rng(1)
    p, value = rng.uniform(0, 1, 1000), rng.uniform(0, 4000, 1000)
    benefit = r.expected_net_benefit(p, value, 0.15, 100)
    churned = (rng.uniform(size=1000) < p).astype(int)
    free = r.campaign_summary(r.select_targets("cost_based", p, benefit), p, value, churned, 0.15, 100)["net_value"]
    capped = r.campaign_summary(r.select_capacity(benefit, 0.10), p, value, churned, 0.15, 100)["net_value"]
    assert capped <= free + 1e-9


def test_curve_optimum_matches_cost_based_rule():
    rng = np.random.default_rng(0)
    p, value = rng.uniform(0, 1, 500), rng.uniform(0, 5000, 500)
    curve = r.net_value_curve(p, value, 0.2, 60)
    benefit = r.expected_net_benefit(p, value, 0.2, 60)
    best = curve.loc[curve["cumulative_net_value"].idxmax()]
    assert best["customers_targeted"] == (benefit > 0).sum()
    assert best["cumulative_net_value"] == pytest.approx(benefit[benefit > 0].sum())
    assert curve.iloc[0]["cumulative_net_value"] == 0


def test_write_reports_refuses_identifiers(tmp_path):
    bad = {"scored": pd.DataFrame({"CustomerId": [1]}), "summary": pd.DataFrame({"a": [1]}), "curves": pd.DataFrame({"a": [1]})}
    with pytest.raises(ValueError, match="identifier"):
        r.write_reports(bad, tmp_path)
