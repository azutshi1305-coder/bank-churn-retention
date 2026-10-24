"""Smoke test for a running churn API (local uvicorn or the container). Standard library only.

Usage:  python scripts/smoke_test.py [--base-url http://127.0.0.1:8000]
Exit code 0 if every check passes, 1 otherwise.
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

VALID = {"CreditScore": 619, "Geography": "France", "Age": 42, "Tenure": 2, "Balance": 83807.86,
         "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88}


def call(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, raw, headers = resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as err:
        status, raw, headers = err.code, err.read(), err.headers
    return status, json.loads(raw), headers, (time.perf_counter() - start) * 1000


def random_customers(n, seed=0):
    rng = random.Random(seed)
    return [{"CreditScore": rng.randint(350, 850), "Geography": rng.choice(["France", "Germany", "Spain"]), "Age": rng.randint(18, 90),
             "Tenure": rng.randint(0, 10), "Balance": rng.choice([0.0, round(rng.uniform(1e4, 2.5e5), 2)]),
             "NumOfProducts": rng.randint(1, 4), "HasCrCard": rng.randint(0, 1), "IsActiveMember": rng.randint(0, 1),
             "EstimatedSalary": round(rng.uniform(1e4, 2e5), 2)} for _ in range(n)]


def check_health(base):
    status, body, _, ms = call(base, "GET", "/health")
    assert status == 200 and body["status"] == "ok" and body["model_version"], body
    return f"model_version={body['model_version']}", ms


def check_model_info(base):
    status, body, _, ms = call(base, "GET", "/model-info")
    assert status == 200, status
    assert body["n_features"] == len(body["features"]) == 16 and not any("Gender" in f for f in body["features"]), body["features"]
    assert 0.8 < body["metrics"]["test_roc_auc"] < 0.95 and body["probability_clip"]["low"] == 0.005
    return f"v{body['model_version']}, test ROC-AUC {body['metrics']['test_roc_auc']:.4f}, trained {body['training_date']}", ms


def check_valid_predict(base):
    status, body, headers, ms = call(base, "POST", "/predict", VALID)
    assert status == 200, (status, body)
    assert 0.005 <= body["churn_probability"] <= 0.995 and body["risk_band"] in {"low", "medium", "high"}
    assert len(body["top_drivers"]) == 3 and headers["X-Request-ID"] == body["request_id"]
    return f"p={body['churn_probability']:.4f} band={body['risk_band']} top driver={body['top_drivers'][0]['feature']}", ms


def check_invalid_is_422(base):
    status, body, _, ms = call(base, "POST", "/predict", {**VALID, "Age": 17, "Gender": "Female"})
    assert status == 422, (status, body)
    assert all(set(e) == {"loc", "msg", "type"} for e in body["detail"]), body      # never echoes the rejected input
    fields = sorted(e["loc"][-1] for e in body["detail"])
    assert fields == ["Age", "Gender"], fields
    return f"rejected {fields}", ms


def check_batch_of_100(base):
    status, body, _, ms = call(base, "POST", "/predict/batch", {"customers": random_customers(100)})
    assert status == 200, (status, body)
    preds = body["predictions"]
    assert body["count"] == len(preds) == 100 and all(0.005 <= p["churn_probability"] <= 0.995 for p in preds)
    assert all(p["top_drivers"] is None for p in preds)                              # drivers are opt-in on batch
    bands = {b: sum(p["risk_band"] == b for p in preds) for b in ("low", "medium", "high")}
    return f"count=100 bands={bands}", ms


CHECKS = [("GET  /health", check_health), ("GET  /model-info", check_model_info), ("POST /predict (valid)", check_valid_predict),
          ("POST /predict (invalid -> 422)", check_invalid_is_422), ("POST /predict/batch (100)", check_batch_of_100)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    base = parser.parse_args().base_url.rstrip("/")
    failures = 0
    print(f"smoke test against {base}")
    for name, check in CHECKS:
        try:
            detail, ms = check(base)
            print(f"  PASS  {name:32s} {ms:7.0f} ms  {detail}")
        except Exception as exc:  # report and keep going
            failures += 1
            print(f"  FAIL  {name:32s} {type(exc).__name__}: {exc}")
    print("ALL CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
