import shutil
import xml.etree.ElementTree as ET

import pytest

from src.reporting import tableau_workbook as tw

pytestmark = pytest.mark.skipif(not all((tw.CSV_DIR / f).exists() for f in ("segment_summary.csv", "drivers.csv", "scored_customers.csv",
                                                                           "scenarios.csv", "drift.csv")),
                                reason="needs reports/tableau/*.csv; run python -m src.reporting.tableau")


@pytest.fixture
def built(tmp_path):
    return tw.write(tmp_path / "tableau" / "wb.twb")


def test_generated_workbook_is_well_formed_and_every_reference_resolves(built):
    assert tw.validate(built, tw.CSV_DIR) == []


def test_committed_workbook_matches_what_the_generator_produces(built):
    committed = tw.OUT_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert committed == built.read_bytes().replace(b"\r\n", b"\n"), "regenerate with: python -m src.reporting.tableau_workbook"


def test_connections_are_relative_and_point_at_real_files():
    root = ET.parse(tw.OUT_PATH).getroot()
    conns = root.findall(".//named-connection/connection")
    assert len(conns) == 5
    for c in conns:
        assert c.get("class") == "textscan" and not c.get("directory").startswith(("/", "C:", "c:"))
        assert (tw.OUT_PATH.parent / c.get("directory") / c.get("filename")).exists()


def test_parameters_have_the_requested_domains_and_defaults():
    root = ET.parse(tw.OUT_PATH).getroot()
    params = {c.get("caption"): c for c in root.find("datasources/datasource[@name='Parameters']").findall("column")}
    cost, cap = params["Cost scenario"], params["Capacity"]
    assert [m.get("value") for m in cost.find("members")] == ['"best"', '"base"', '"worst"'] and cost.get("value") == '"base"'
    assert [m.get("value") for m in cap.find("members")] == ['"10%"', '"15%"', '"Unconstrained"'] and cap.get("value") == '"10%"'


def test_capacity_choices_match_the_values_in_scenarios_csv():
    import pandas as pd
    assert set(pd.read_csv(tw.CSV_DIR / "scenarios.csv")["capacity"]) == {"10%", "15%", "Unconstrained"}
    assert set(pd.read_csv(tw.CSV_DIR / "scenarios.csv")["cost_scenario"]) == {"best", "base", "worst"}


def test_dashboard_has_both_parameter_controls_and_every_sheet():
    root = ET.parse(tw.OUT_PATH).getroot()
    sheets = {w.get("name") for w in root.find("worksheets")}
    assert sheets == {"KPI Churn rate", "KPI Balance at risk", "KPI Net value", "KPI ROI", "Churn by segment", "Churn drivers",
                      "Scenario comparison", "Risk vs value", "Drift"}
    dash = root.find("dashboards/dashboard")
    assert dash.get("name") == "Bank Churn & Retention"
    assert {z.get("param") for z in dash.iter("zone") if z.get("type-v2") == "paramctrl"} == {"[Parameters].[Cost scenario]", "[Parameters].[Capacity]"}
    assert {z.get("name") for z in dash.iter("zone") if z.get("name")} == sheets


def test_drift_sheet_is_labelled_simulated_and_has_both_reference_lines():
    root = ET.parse(tw.OUT_PATH).getroot()
    drift = root.find("worksheets/worksheet[@name='Drift']")
    assert "SIMULATED" in "".join(drift.find("layout-options/title").itertext())
    lines = drift.findall("table/panes/pane/reference-line")
    assert [l.get("value-column") for l in lines] == ["[Parameters].[PSI moderate threshold]", "[Parameters].[PSI significant threshold]"]


def test_validator_catches_a_renamed_csv_column(tmp_path, built):
    shutil.copytree(tw.CSV_DIR, tmp_path / "csv")
    path = tmp_path / "csv" / "drift.csv"
    path.write_text(path.read_text().replace("scenario,description", "scenario_id,description", 1))
    assert any("drift.csv" in p for p in tw.validate(built, tmp_path / "csv"))


def test_validator_catches_a_formula_with_a_missing_field_and_an_undeclared_parameter(tmp_path, built):
    text = built.read_text(encoding="utf-8")
    broken = text.replace("THEN [net_value] END", "THEN [net_valu] END", 1)
    p = tmp_path / "broken.twb"
    p.write_text(broken, encoding="utf-8")
    assert any("net_valu" in x for x in tw.validate(p, tw.CSV_DIR))
    root = ET.fromstring(text)
    ws = root.find("worksheets/worksheet[@name='KPI Net value']")
    dep = [d for d in ws.findall("table/view/datasource-dependencies") if d.get("datasource") == "Parameters"][0]
    ws.find("table/view").remove(dep)
    q = tmp_path / "noparam.twb"
    ET.ElementTree(root).write(q, encoding="utf-8")
    assert any("does not declare" in x for x in tw.validate(q, tw.CSV_DIR))


def test_not_well_formed_xml_is_reported(tmp_path):
    p = tmp_path / "bad.twb"
    p.write_text("<workbook><unclosed></workbook>")
    assert tw.validate(p)[0].startswith("not well-formed")
