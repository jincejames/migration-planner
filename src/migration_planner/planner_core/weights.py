from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    ceil,
    col,
    lit,
    log1p,
    lower,
    max as spark_max,
    min as spark_min,
    sum as spark_sum,
    when,
)

# ---------------------------------------------------------------------------
# Weight method identifiers — used in PlannerConfig and as display labels
# ---------------------------------------------------------------------------

WEIGHT_METHOD_FACTOR: str = "factor"
WEIGHT_METHOD_SCALED: str = "scaled"

_VALID_WEIGHT_METHODS: tuple[str, ...] = (WEIGHT_METHOD_FACTOR, WEIGHT_METHOD_SCALED)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def calculate_table_weights(
    stream_stream_dependency_df: DataFrame,
    method: str = WEIGHT_METHOD_FACTOR,
    use_log: bool = False,
    scaled_min: float = 0.0,
    scaled_max: float = 1.0,
    tableau_multiplier: float = 1.0,
) -> DataFrame:
    """
    Add a ``table_weight`` column using the requested weight method.

    Parameters
    ----------
    stream_stream_dependency_df:
        Input DataFrame with at least columns ``to``, ``size``.
    method:
        Weight calculation strategy.  One of ``"factor"`` (default) or
        ``"scaled"``.  Unknown values raise :exc:`ValueError`.
    use_log:
        Apply a log₁₊ transform to raw sizes before scaling (``"scaled"``
        method only).
    scaled_min, scaled_max:
        Target range for the min-max scaler (``"scaled"`` method only).
    tableau_multiplier:
        Multiplier applied to JSON-destination weights after scaling
        (``"scaled"`` method only).

    Returns
    -------
    DataFrame
        Input DataFrame with an added ``table_weight`` column.
    """
    if method == WEIGHT_METHOD_SCALED:
        return _calculate_scaled_weights(
            stream_stream_dependency_df,
            use_log=use_log,
            scaled_min=scaled_min,
            scaled_max=scaled_max,
            tableau_multiplier=tableau_multiplier,
        )
    if method == WEIGHT_METHOD_FACTOR:
        return _calculate_factor_weights(stream_stream_dependency_df)
    raise ValueError(
        f"Unknown weight method {method!r}. "
        f"Valid values: {_VALID_WEIGHT_METHODS}"
    )


# ---------------------------------------------------------------------------
# Method implementations (private)
# ---------------------------------------------------------------------------


def _calculate_factor_weights(stream_stream_dependency_df: DataFrame) -> DataFrame:
    """
    Factor-based weights (Option 1):

    - JSON stream destinations → ``table_weight = 2``
    - all others               → ``ceil(size / 100)``, minimum 1
    """
    return (
        stream_stream_dependency_df
        .withColumn(
            "table_weight",
            when(lower(col("to")).contains("json"), 2)
            .otherwise(ceil(col("size").cast("double") / 100).cast("int")),
        )
        .withColumn(
            "table_weight",
            when(col("table_weight") < 1, 1).otherwise(col("table_weight")),
        )
    )


def _calculate_scaled_weights(
    stream_stream_dependency_df: DataFrame,
    use_log: bool = False,
    scaled_min: float = 0.0,
    scaled_max: float = 1.0,
    tableau_multiplier: float = 1.0,
) -> DataFrame:
    """
    Min-max scaled weights (Option 2):

    1. Raw value = size (optionally log-transformed).
    2. Min-max scale to [``scaled_min``, ``scaled_max``].
    3. Apply a multiplier to JSON-destination streams.
    4. Ensure final weight >= ``scaled_min``.
    """
    df = stream_stream_dependency_df.withColumn(
        "raw_table_weight",
        col("size").cast("double"),
    )

    # Guard against null / negative sizes
    df = df.withColumn(
        "raw_table_weight",
        when(
            col("raw_table_weight").isNull() | (col("raw_table_weight") < 0),
            lit(0.0),
        ).otherwise(col("raw_table_weight")),
    )

    # Optional log transform
    if use_log:
        df = df.withColumn("raw_for_scaling", log1p(col("raw_table_weight")))
    else:
        df = df.withColumn("raw_for_scaling", col("raw_table_weight"))

    # Compute min / max on the cluster — required for normalisation
    stats = df.agg(
        spark_min("raw_for_scaling").alias("mn"),
        spark_max("raw_for_scaling").alias("mx"),
    ).collect()[0]

    mn = float(stats["mn"]) if stats["mn"] is not None else 0.0
    mx = float(stats["mx"]) if stats["mx"] is not None else 0.0
    den = (mx - mn) if (mx - mn) != 0 else 1.0

    df = df.withColumn("mm01", (col("raw_for_scaling") - lit(mn)) / lit(den))

    df = df.withColumn(
        "table_weight",
        lit(scaled_min) + (lit(scaled_max - scaled_min) * col("mm01")),
    )

    # Ensure weight >= scaled_min
    df = df.withColumn(
        "table_weight",
        when(col("table_weight") < scaled_min, lit(scaled_min))
        .otherwise(col("table_weight")),
    )

    # JSON-destination boost
    df = df.withColumn(
        "table_weight",
        when(
            lower(col("to")).contains("json"),
            col("table_weight") * lit(tableau_multiplier),
        ).otherwise(col("table_weight")),
    )

    return df


# ---------------------------------------------------------------------------
# Shared post-processing steps (used by both methods)
# ---------------------------------------------------------------------------


def deduplicate_table_weights(table_weight_df: DataFrame) -> DataFrame:
    """Drop duplicate (from, to, table) rows and rename ``table_weight`` → ``weight``."""
    return (
        table_weight_df
        .dropDuplicates(["from", "to", "table"])
        .withColumnRenamed("table_weight", "weight")
    )


def aggregate_edge_weights(unique_table_weights: DataFrame) -> DataFrame:
    """Sum per-table weights into a single (from, to, weight) edge record."""
    return unique_table_weights.groupBy("from", "to").agg(
        spark_sum("weight").alias("weight")
    )
