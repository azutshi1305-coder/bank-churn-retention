import json

import pandas as pd
import pytest

from src.features import build_features as bf
from src.features.transform import (
    AUDIT_COLUMNS, FEATURE_COLUMNS, FEATURE_DTYPES, GEOGRAPHIES, ONE_HOT, audit_columns, transform,
)


@pytest.fixture(scope="module")
def spark():
    session = bf.get_spark()
    yield session
    session.stop()


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(bf.RAW_PATH)


def test_spark_and_pandas_features_are_identical(spark, raw):
    spark_feats = (bf.add_features(bf.read_raw(spark).withColumn("CustomerId", bf.F.col("CustomerId").cast("long")))
                   .select("CustomerId", *FEATURE_COLUMNS).toPandas()
                   .set_index("CustomerId").sort_index())
    pandas_feats = transform(raw.set_index("CustomerId")).sort_index()

    assert list(spark_feats.columns) == list(pandas_feats.columns) == FEATURE_COLUMNS
    assert spark_feats.shape == (10000, len(FEATURE_COLUMNS))
    pd.testing.assert_frame_equal(spark_feats, pandas_feats, check_exact=True)


def test_spark_and_pandas_audit_columns_are_identical(spark, raw):
    spark_audit = (bf.add_features(bf.read_raw(spark).withColumn("CustomerId", bf.F.col("CustomerId").cast("long")))
                   .select("CustomerId", *AUDIT_COLUMNS).toPandas().set_index("CustomerId").sort_index())
    pandas_audit = audit_columns(raw.set_index("CustomerId")).sort_index()
    pd.testing.assert_frame_equal(spark_audit, pandas_audit, check_exact=True)


def test_gender_is_not_a_model_feature(raw):
    assert not any(c.startswith("Gender") for c in FEATURE_COLUMNS)
    assert not any(c.startswith("Gender") for c in transform(raw).columns)
    assert AUDIT_COLUMNS == ["Gender_Female", "Gender_Male"]


def test_dtypes_match_schema_file(raw):
    schema = json.loads(bf.SCHEMA_PATH.read_text())
    assert [f["name"] for f in schema["features"]] == FEATURE_COLUMNS
    assert {f["name"]: f["dtype"] for f in schema["features"]} == FEATURE_DTYPES
    out = transform(raw)
    assert {c: str(t) for c, t in out.dtypes.items()} == FEATURE_DTYPES


def test_tenure_zero_is_finite_and_formulas(raw):
    row = pd.DataFrame([{
        "CreditScore": 600, "Geography": "Germany", "Gender": "Female", "Age": 50, "Tenure": 0,
        "Balance": 0.0, "NumOfProducts": 3, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000.0,
    }])
    f = transform(row).iloc[0]
    assert f["products_per_tenure"] == 3.0          # 3 / (0 + 1), not a division by zero
    assert f["zero_balance"] == 1 and f["balance_to_salary"] == 0.0
    assert f["is_senior"] == 1                      # Age >= 50 is inclusive
    assert f["active_x_products"] == 3
    assert f["Geography_Germany"] == 1 and f["Geography_France"] == 0
    assert "Gender_Female" not in f.index          # Gender is read upstream and ignored
    assert transform(raw)["products_per_tenure"].replace([float("inf")], float("nan")).notna().all()


def test_unseen_category_gives_all_zero_one_hot():
    row = pd.DataFrame([{
        "CreditScore": 600, "Geography": "Italy", "Gender": "Other", "Age": 40, "Tenure": 2,
        "Balance": 1000.0, "NumOfProducts": 1, "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 40000.0,
    }])
    assert transform(row)[ONE_HOT].iloc[0].sum() == 0
    assert len(ONE_HOT) == len(GEOGRAPHIES)


def test_split_is_exact_stratified_and_deterministic(spark):
    def assignment():
        df = bf.add_split(bf.read_raw(spark))
        return df.select("CustomerId", "Exited", "split").toPandas().set_index("CustomerId").sort_index()

    first, second = assignment(), assignment()
    counts = first.groupby(["Exited", "split"]).size().to_dict()
    assert counts == {(0, "train"): 6370, (0, "test"): 1593, (1, "train"): 1630, (1, "test"): 407}
    assert first.equals(second)
