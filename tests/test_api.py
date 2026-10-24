import json
import logging

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.bands import HIGH_FROM, LOW_BELOW, PROB_CLIP, clip_probabilities, risk_band
from src.api.main import DEFAULT_MODEL_PATH, app
from src.api.schemas import MAX_BATCH
from src.features.transform import transform
from src.models.registry import DATA_PATH, ROOT

pytestmark = pytest.mark.skipif(not (DEFAULT_MODEL_PATH / "MLmodel").exists(),
                                reason="no exported champion; run python -m src.models.export_champion")

VALID = {"CreditScore": 619, "Geography": "France", "Age": 42, "Tenure": 2, "Balance": 83807.86,
         "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88}
RAW_FIELDS = list(VALID)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def with_(**changes):
    return {**VALID, **changes}


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "2"}
    info = client.get("/model-info").json()
    assert info["model_name"] == "bank-churn-model" and info["model_version"] == "2"
    assert info["training_date"] and info["run_id"] and info["calibration_method"] == "isotonic"
    assert info["n_features"] == 16 == len(info["features"])
    assert not any("Gender" in f for f in info["features"])
    assert 0.86 < info["metrics"]["test_roc_auc"] < 0.87
    assert set(info["risk_bands"]) == {"low", "medium", "high"}
    clip = info["probability_clip"]
    assert (clip["low"], clip["high"]) == PROB_CLIP == (0.005, 0.995)
    assert set(clip["test_metrics_before_clipping"]) == set(clip["test_metrics_after_clipping"]) == {"roc_auc", "pr_auc", "brier"}
    assert "service layer" in clip["applied_in"]


def test_docs_and_capacity_note_are_published(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    for path in ("/predict", "/predict/batch"):
        text = spec["paths"][path]["post"]["description"]
        assert "not a contact decision" in text and "portfolio-level" in text
    assert "low" in spec["info"]["description"] and "50%" in spec["info"]["description"]
    assert "clipped" in spec["info"]["description"] and "0.005" in spec["info"]["description"] and "0.995" in spec["info"]["description"]


def test_predict_valid_request(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    p = body["churn_probability"]
    assert 0 < p < 1 and body["risk_band"] == risk_band(p)
    assert body["model_version"] == "2" and body["request_id"] == r.headers["X-Request-ID"]
    a = body["assumptions"]
    value = VALID["Balance"] * a["net_interest_margin"] + VALID["NumOfProducts"] * a["annual_fee_per_product"]
    assert body["estimated_annual_value"] == pytest.approx(value)
    assert body["expected_net_benefit"] == pytest.approx(p * a["retention_success_rate"] * value - a["offer_cost"])
    drivers = body["top_drivers"]
    assert len(drivers) == 3
    mags = [abs(d["contribution"]) for d in drivers]
    assert mags == sorted(mags, reverse=True)
    assert all(d["direction"] == ("increases churn risk" if d["contribution"] > 0 else "decreases churn risk") for d in drivers)
    assert "not a contact decision" in body["note"]


def test_clip_probabilities_unit():
    raw = np.array([0.0, 0.001, 0.005, 0.5, 0.995, 0.999, 1.0])
    np.testing.assert_array_equal(clip_probabilities(raw), [0.005, 0.005, 0.005, 0.5, 0.995, 0.995, 0.995])


def test_api_clips_an_overconfident_prediction_and_uses_it_downstream(client):
    high = {"CreditScore": 502, "Geography": "Germany", "Age": 52, "Tenure": 5, "Balance": 116497.31,
            "NumOfProducts": 3, "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 113931.57}
    raw = mlflow.sklearn.load_model(str(DEFAULT_MODEL_PATH)).predict_proba(transform(pd.DataFrame([high])))[:, 1][0]
    assert raw > PROB_CLIP[1]                      # the model itself is overconfident here (it returns 1.0) ...
    body = client.post("/predict", json=high).json()
    assert body["churn_probability"] == PROB_CLIP[1] and body["risk_band"] == "high"        # ... the API serves the clipped value
    a = body["assumptions"]                        # and the expected benefit is computed from the clipped probability
    value = body["estimated_annual_value"]
    assert body["expected_net_benefit"] == pytest.approx(PROB_CLIP[1] * a["retention_success_rate"] * value - a["offer_cost"])


def test_risk_band_cutoffs():
    assert [risk_band(x) for x in (0.0, LOW_BELOW - 1e-9, LOW_BELOW, HIGH_FROM - 1e-9, HIGH_FROM, 1.0)] == \
        ["low", "low", "medium", "medium", "high", "high"]


@pytest.mark.parametrize("field,bad", [
    ("CreditScore", 299), ("CreditScore", 901), ("CreditScore", "abc"), ("CreditScore", 600.5),
    ("Geography", "Italy"), ("Geography", "france"), ("Geography", None),
    ("Age", 17), ("Age", 101), ("Tenure", -1), ("Tenure", 11),
    ("Balance", -0.01), ("Balance", "lots"),
    ("NumOfProducts", 0), ("NumOfProducts", 5),
    ("HasCrCard", 2), ("HasCrCard", -1), ("IsActiveMember", 2), ("IsActiveMember", 0.5),
    ("EstimatedSalary", 0), ("EstimatedSalary", -1.0),
])
def test_each_invalid_field_gives_422(client, field, bad):
    r = client.post("/predict", json=with_(**{field: bad}))
    assert r.status_code == 422
    assert [field] in [e["loc"][1:] for e in r.json()["detail"]]
    assert all(set(e) == {"loc", "msg", "type"} for e in r.json()["detail"])   # never echoes the rejected input


@pytest.mark.parametrize("field", RAW_FIELDS)
def test_missing_field_gives_422(client, field):
    body = {k: v for k, v in VALID.items() if k != field}
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "missing"


@pytest.mark.parametrize("extra", ["Gender", "CustomerId", "Surname", "RowNumber", "Exited"])
def test_gender_identifiers_and_target_are_rejected(client, extra):
    r = client.post("/predict", json={**VALID, extra: "Female" if extra == "Gender" else 1})
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("literal", ["NaN", "Infinity"])
def test_non_finite_numbers_are_rejected(client, literal):
    raw = json.dumps(VALID).replace("83807.86", literal)
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_boundary_values_are_accepted(client):
    for changes in ({"CreditScore": 300, "Age": 18, "Tenure": 0, "NumOfProducts": 1, "Balance": 0, "EstimatedSalary": 0.01},
                    {"CreditScore": 900, "Age": 100, "Tenure": 10, "NumOfProducts": 4, "HasCrCard": 0, "IsActiveMember": 0}):
        assert client.post("/predict", json=with_(**changes)).status_code == 200


def test_batch_limit_and_shape(client):
    ok = client.post("/predict/batch", json={"customers": [VALID] * MAX_BATCH})
    assert ok.status_code == 200 and ok.json()["count"] == MAX_BATCH == len(ok.json()["predictions"])
    assert all(p["top_drivers"] is None for p in ok.json()["predictions"])      # drivers are opt-in for batch
    assert client.post("/predict/batch", json={"customers": [VALID] * (MAX_BATCH + 1)}).status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    bad = client.post("/predict/batch", json={"customers": [VALID, with_(Age=5)]})
    assert bad.status_code == 422 and bad.json()["detail"][0]["loc"][:3] == ["body", "customers", 1]


def test_batch_matches_single_predictions(client):
    customers = [VALID, with_(Geography="Germany", Age=55, IsActiveMember=0, NumOfProducts=3), with_(Balance=0.0, Tenure=0)]
    batch = client.post("/predict/batch", params={"include_drivers": True}, json={"customers": customers}).json()["predictions"]
    for c, b in zip(customers, batch):
        single = client.post("/predict", json=c).json()
        assert single["churn_probability"] == b["churn_probability"] and single["top_drivers"] == b["top_drivers"]
        assert len(b["top_drivers"]) == 3


@pytest.fixture(scope="module")
def raw_test_rows():
    raw_path = ROOT / "data" / "raw" / "Churn_Modelling.csv"
    if not raw_path.exists() or not DATA_PATH.exists():
        pytest.skip("needs data/raw and data/processed (not tracked in git)")
    raw = pd.read_csv(raw_path)
    split = pd.read_parquet(DATA_PATH, columns=["split"])["split"].to_numpy()
    return raw[split == "test"].head(50).reset_index(drop=True)


def test_api_probabilities_match_offline_champion_on_50_test_rows(client, raw_test_rows):
    offline_model = mlflow.sklearn.load_model(str(DEFAULT_MODEL_PATH))
    offline = clip_probabilities(offline_model.predict_proba(transform(raw_test_rows))[:, 1])   # same transform code as training
    payload = json.loads(raw_test_rows[RAW_FIELDS].to_json(orient="records"))      # raw fields only: no Gender, no ids
    r = client.post("/predict/batch", json={"customers": payload})
    assert r.status_code == 200
    api = np.array([p["churn_probability"] for p in r.json()["predictions"]])
    assert len(api) == 50
    np.testing.assert_allclose(api, offline, rtol=0, atol=1e-12)
    with_drivers = client.post("/predict/batch", params={"include_drivers": True}, json={"customers": payload}).json()["predictions"]
    np.testing.assert_allclose([p["churn_probability"] for p in with_drivers], offline, rtol=0, atol=1e-12)
    assert all(len(p["top_drivers"]) == 3 for p in with_drivers)


def test_whole_test_split_matches_clipped_offline_predictions(client):
    raw_path = ROOT / "data" / "raw" / "Churn_Modelling.csv"
    if not raw_path.exists() or not DATA_PATH.exists():
        pytest.skip("needs data/raw and data/processed (not tracked in git)")
    split = pd.read_parquet(DATA_PATH, columns=["split"])["split"].to_numpy()
    rows = pd.read_csv(raw_path)[split == "test"].reset_index(drop=True)
    assert len(rows) == 2000
    raw_p = mlflow.sklearn.load_model(str(DEFAULT_MODEL_PATH)).predict_proba(transform(rows))[:, 1]
    api = []
    for start in range(0, len(rows), MAX_BATCH):
        chunk = json.loads(rows[RAW_FIELDS].iloc[start:start + MAX_BATCH].to_json(orient="records"))
        api += [p["churn_probability"] for p in client.post("/predict/batch", json={"customers": chunk}).json()["predictions"]]
    api = np.array(api)
    np.testing.assert_allclose(api, np.clip(raw_p, *PROB_CLIP), rtol=0, atol=1e-12)
    assert api.min() >= PROB_CLIP[0] and api.max() <= PROB_CLIP[1]
    assert (raw_p > PROB_CLIP[1]).sum() > 0 and (raw_p < PROB_CLIP[0]).sum() > 0     # both ends are actually exercised
    assert (api == PROB_CLIP[1]).sum() == (raw_p > PROB_CLIP[1]).sum() and (api == PROB_CLIP[0]).sum() == (raw_p < PROB_CLIP[0]).sum()


def test_logs_have_request_id_latency_version_but_never_input_values(client, caplog):
    secret_balance, secret_salary, bad_balance = 777123.456, 98765.4321, -4242.4242
    with caplog.at_level(logging.INFO, logger="churn_api"):
        ok = client.post("/predict", json=with_(Balance=secret_balance, EstimatedSalary=secret_salary))
        bad = client.post("/predict", json=with_(Balance=bad_balance))
        client.post("/predict/batch", json={"customers": [with_(Balance=secret_balance)]})
    text = caplog.text
    assert ok.json()["request_id"] in text and bad.status_code == 422
    assert "latency_ms=" in text and "model_version=2" in text and "path=/predict" in text and "status=422" in text
    for value in ("777123.456", "98765.4321", "4242.4242", "Germany", "France"):
        assert value not in text


def test_startup_fails_clearly_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="export_champion"):
        with TestClient(app):
            pass
