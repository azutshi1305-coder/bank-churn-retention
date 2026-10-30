import numpy as np
import pandas as pd
import pytest

from src.api.bands import risk_band as api_band
from src.models.registry import DATA_PATH
from src.models.retention import FORBIDDEN_COLUMNS
from src.reporting import tableau as t

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="needs data/processed (not tracked in git)")


def test_vectorised_band_matches_the_api_bands_at_the_cut_points():
    p = np.array([0.0, 0.1999, 0.20, 0.4999, 0.50, 0.99])
    assert t.risk_band(p).tolist() == [api_band(x) for x in p]


def test_write_refuses_identifier_columns(tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        t.write({"x.csv": pd.DataFrame({"CustomerId": [1], "a": [2]})}, tmp_path)


def test_write_stores_booleans_as_integers(tmp_path):
    t.write({"x.csv": pd.DataFrame({"flag": [True, False]})}, tmp_path)
    assert (tmp_path / "x.csv").read_text().split() == ["flag", "1", "0"]


def test_scenarios_has_every_scenario_by_capacity_and_drops_other_rules():
    rows = [{"scenario": s, "rule": r, "basis": b, "customers_targeted": 1, "pct_targeted": 1, "expected_churners_reached": 1,
             "actual_churners_reached": 1, "customers_saved": 1, "value_retained": 1, "campaign_cost": 1, "net_value": 1, "roi_pct": 1}
            for s in ("best", "base", "worst") for r in ("cost_based", "capacity_10pct", "capacity_15pct", "top_10pct", "threshold_0.5")
            for b in ("test_2000", "per_1000")]
    out = t.scenarios(pd.DataFrame(rows))
    assert len(out) == 3 * 3 * 2
    assert set(out["capacity"]) == {"Unconstrained", "10%", "15%"}
    assert out["cost_scenario"].tolist()[:3] == ["best", "best", "best"] and out["cost_scenario"].iloc[-1] == "worst"


@needs_data
def test_segment_summary_dimensions_each_cover_all_customers_and_have_no_identifiers():
    s = t.segment_summary()
    assert not FORBIDDEN_COLUMNS & set(s.columns)
    assert s.groupby("dimension")["customers"].sum().eq(10_000).all()
    assert s.groupby("dimension")["churners"].sum().eq(s.groupby("dimension")["churners"].sum().iloc[0]).all()
    assert s.groupby("dimension")["share_of_total_balance_at_risk"].sum().round(6).nunique() == 1
    assert s["churn_rate_pct"].between(0, 100).all()
