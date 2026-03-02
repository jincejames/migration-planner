from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, upper, lower, lit

COMPLEXITY_WEIGHTS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "complex": 4,
    "very_complex": 7,
}


def load_stream_table_dependencies(spark: SparkSession, path: str) -> DataFrame:
    """Load the ETL stream-to-table dependency CSV (header=true)."""
    return spark.read.format("csv").option("header", "true").load(path)


def load_outofscope_streams(spark: SparkSession, path: str) -> list[str]:
    """Load the out-of-scope stream names CSV and return as a plain list."""
    rows = (
        spark.read.format("csv").option("header", "true").load(path)
        .select(col("stream_name"))
        .collect()
    )
    return [row["stream_name"] for row in rows]


def load_complexity_scores(spark: SparkSession, path: str) -> DataFrame:
    """
    Load the per-stream complexity breakdown CSV (semicolon-delimited) and
    return a DataFrame with an added `complexity_score` column.
    Score = low*1 + medium*2 + complex*4 + very_complex*7
    """
    raw = (
        spark.read.format("csv")
        .option("header", "true")
        .option("delimiter", ";")
        .load(path)
    )
    return raw.withColumn(
        "complexity_score",
        (col("low").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS["low"])) +
        (col("medium").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS["medium"])) +
        (col("complex").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS["complex"])) +
        (col("very_complex").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS["very_complex"]))
    ).select(
        col("stream_name"),
        col("low").cast("double").cast("int").alias("low"),
        col("medium").cast("double").cast("int").alias("medium"),
        col("complex").cast("double").cast("int").alias("complex"),
        col("very_complex").cast("double").cast("int").alias("very_complex"),
        col("complexity_score"),
    )


def load_report_dependencies(spark: SparkSession, path: str) -> DataFrame:
    """
    Load the report-to-table dependency CSV; rename columns, mark all tables
    as Src type, and exclude corona / gdpr reports.
    """
    return (
        spark.read.format("csv").option("header", "true").load(path)
        .select(
            col("report_name").alias("stream_name"),
            upper(col("table_name")).alias("table_name"),
            lit("Src").alias("table_type"),
        )
        .filter(
            ~lower(col("stream_name")).contains("corona")
            & ~lower(col("stream_name")).contains("gdpr")
        )
    )


def load_table_sizes(spark: SparkSession, path: str) -> DataFrame:
    """Load the table-size CSV; return table_name (upper) and size columns."""
    return (
        spark.read.format("csv").option("header", "true").load(path)
        .select(
            upper(col("DB_Table_Name")).alias("table_name"),
            col("SPACE_IN_GB").alias("size"),
        )
    )


def load_static_tables(spark: SparkSession, path: str) -> set[str]:
    """Load the static-tables-for-report CSV and return a set of table names."""
    rows = (
        spark.read.option("header", True).csv(path)
        .select("table_name")
        .collect()
    )
    return {row["table_name"] for row in rows}
