"""Spark feature pipeline: raw CSV -> data/processed/features.parquet + src/features/feature_schema.json.

Run from the repo root:  python -m src.features.build_features
No scaling here; scaling is fitted on the train split only in Stage 4.
"""
import json
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.features.transform import (
    DESCRIPTIONS, ENGINEERED, FEATURE_COLUMNS, FEATURE_DTYPES, GENDERS, GEOGRAPHIES,
    ONE_HOT, RAW_NUMERIC, SENIOR_AGE,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "Churn_Modelling.csv"
OUT_PATH = ROOT / "data" / "processed" / "features.parquet"
SCHEMA_PATH = ROOT / "src" / "features" / "feature_schema.json"

TARGET = "Exited"
SPLIT_COL = "split"
SEED = 42
TRAIN_FRACTION = 0.8
IDENTIFIERS = ["RowNumber", "CustomerId", "Surname"]
SPARK_TYPE = {"int64": "long", "float64": "double"}


def get_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    return (SparkSession.builder.master("local[*]").appName("churn-features")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate())


def read_raw(spark: SparkSession) -> DataFrame:
    return spark.read.csv(str(RAW_PATH), header=True, inferSchema=True)


def add_split(df: DataFrame) -> DataFrame:
    """Exact stratified 80/20 split on Exited, reproducible without relying on Spark partitioning.

    Within each class, rows are ordered by a seeded hash of CustomerId and the first 80% are train.
    Must run before identifiers are dropped.
    """
    by_class = Window.partitionBy(TARGET)
    ordered = by_class.orderBy(F.xxhash64(F.lit(SEED), F.col("CustomerId")), F.col("CustomerId"))
    return (df.withColumn("_rn", F.row_number().over(ordered))
              .withColumn("_n", F.count(F.lit(1)).over(by_class))
              .withColumn(SPLIT_COL, F.when(F.col("_rn") <= F.round(F.col("_n") * TRAIN_FRACTION), "train")
                                      .otherwise("test"))
              .drop("_rn", "_n"))


def add_features(df: DataFrame) -> DataFrame:
    """Adds every feature in FEATURE_COLUMNS (same formulas as transform.transform) and casts them to the schema dtypes."""
    for g in GEOGRAPHIES:
        df = df.withColumn(f"Geography_{g}", F.col("Geography") == g)
    for g in GENDERS:
        df = df.withColumn(f"Gender_{g}", F.col("Gender") == g)
    df = (df.withColumn("zero_balance", F.col("Balance") == 0)
            .withColumn("balance_to_salary", F.col("Balance") / F.col("EstimatedSalary"))
            .withColumn("products_per_tenure", F.col("NumOfProducts") / (F.col("Tenure") + 1))
            .withColumn("is_senior", F.col("Age") >= SENIOR_AGE)
            .withColumn("active_x_products", F.col("IsActiveMember") * F.col("NumOfProducts")))
    for c in FEATURE_COLUMNS:
        df = df.withColumn(c, F.col(c).cast(SPARK_TYPE[FEATURE_DTYPES[c]]))
    return df


def write_schema(out: DataFrame) -> None:
    actual = {name: ("int64" if t == "bigint" else "float64" if t == "double" else t) for name, t in out.dtypes}
    for c in FEATURE_COLUMNS:
        assert actual[c] == FEATURE_DTYPES[c], f"{c}: spark gave {actual[c]}, expected {FEATURE_DTYPES[c]}"

    def kind(c):
        return "raw" if c in RAW_NUMERIC else "one_hot" if c in ONE_HOT else "engineered"

    schema = {
        "features": [{"name": c, "dtype": FEATURE_DTYPES[c], "kind": kind(c),
                      **({"description": DESCRIPTIONS[c]} if c in ENGINEERED else {})}
                     for c in FEATURE_COLUMNS],
        "target": TARGET,
        "split_column": SPLIT_COL,
        "split": {"method": "exact stratified on Exited, ordered by seeded hash of CustomerId",
                  "train_fraction": TRAIN_FRACTION, "seed": SEED, "values": ["train", "test"]},
        "categories": {"Geography": GEOGRAPHIES, "Gender": GENDERS},
        "notes": "Identifiers dropped. No scaling applied; fit scalers on the train split only.",
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n")


def build() -> DataFrame:
    spark = get_spark()
    df = add_features(add_split(read_raw(spark)))
    # back to original file order (the split window sorts by class); then drop identifiers
    out = (df.orderBy("RowNumber").drop(*IDENTIFIERS).withColumn(TARGET, F.col(TARGET).cast("long"))
             .select(*FEATURE_COLUMNS, TARGET, SPLIT_COL))
    write_schema(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Spark's own parquet writer needs Hadoop's winutils.exe on Windows; at 10k rows, collecting and
    # writing a single file with pyarrow is simpler and portable. Use out.write.parquet on a cluster.
    out.toPandas().to_parquet(OUT_PATH, index=False)
    return out


if __name__ == "__main__":
    result = build()
    print(f"wrote {OUT_PATH} and {SCHEMA_PATH}")
    print(f"rows={result.count()}, columns={len(result.columns)} "
          f"({len(FEATURE_COLUMNS)} features + {TARGET} + {SPLIT_COL})")
    (result.groupBy(SPLIT_COL).agg(F.count("*").alias("rows"), F.round(F.avg(TARGET) * 100, 2).alias("churn_pct"))
           .orderBy(SPLIT_COL).show())
