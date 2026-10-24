"""Compare two running churn APIs (e.g. the container and a local uvicorn) on the same 50 held-out test customers.

Usage:  python scripts/parity_check.py http://127.0.0.1:8000 http://127.0.0.1:8001
Needs data/raw and data/processed on the host (they are not in the image). Exit code 0 if the probabilities agree.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FIELDS = ["CreditScore", "Geography", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
PROB_TOLERANCE = 1e-9
CONTRIB_TOLERANCE = 1e-6


def test_rows(n=50):
    raw = pd.read_csv(ROOT / "data" / "raw" / "Churn_Modelling.csv")
    split = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet", columns=["split"])["split"].to_numpy()
    return raw[split == "test"].head(n).reset_index(drop=True)


def predict_batch(base, customers):
    req = urllib.request.Request(f"{base.rstrip('/')}/predict/batch?include_drivers=true", method="POST",
                                 data=json.dumps({"customers": customers}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url_a")
    parser.add_argument("url_b")
    args = parser.parse_args()
    customers = json.loads(test_rows()[RAW_FIELDS].to_json(orient="records"))   # raw fields only: no Gender, no identifiers
    a, b = predict_batch(args.url_a, customers), predict_batch(args.url_b, customers)
    pa, pb = a["predictions"], b["predictions"]
    assert len(pa) == len(pb) == len(customers)

    prob_diff = max(abs(x["churn_probability"] - y["churn_probability"]) for x, y in zip(pa, pb))
    same_bands = all(x["risk_band"] == y["risk_band"] for x, y in zip(pa, pb))
    same_drivers = all([d["feature"] for d in x["top_drivers"]] == [d["feature"] for d in y["top_drivers"]] for x, y in zip(pa, pb))
    contrib_diff = max(abs(d1["contribution"] - d2["contribution"]) for x, y in zip(pa, pb) for d1, d2 in zip(x["top_drivers"], y["top_drivers"]))
    benefit_diff = max(abs(x["expected_net_benefit"] - y["expected_net_benefit"]) for x, y in zip(pa, pb))

    print(f"A = {args.url_a}  (model v{a['model_version']})\nB = {args.url_b}  (model v{b['model_version']})\n{len(customers)} test customers")
    print(f"  max |probability difference|        {prob_diff:.3e}   (tolerance {PROB_TOLERANCE:.0e})")
    print(f"  max |driver contribution difference| {contrib_diff:.3e}   (tolerance {CONTRIB_TOLERANCE:.0e})")
    print(f"  max |expected net benefit difference| {benefit_diff:.3e}")
    print(f"  identical risk bands: {same_bands} | identical top-3 driver features and order: {same_drivers}")
    ok = a["model_version"] == b["model_version"] and prob_diff <= PROB_TOLERANCE and contrib_diff <= CONTRIB_TOLERANCE and same_bands and same_drivers
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
