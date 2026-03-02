from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    array,
    col,
    lit,
    sort_array,
    upper,
    when,
)

from migration_planner.community_detector.weights import (
    aggregate_edge_weights,
    calculate_table_weights,
    deduplicate_table_weights,
)


# ---------------------------------------------------------------------------
# Step 1 — filter & normalise the raw dependency table
# ---------------------------------------------------------------------------


def filter_admin_streams(
    dependency_df_full: DataFrame,
    outofscope_stream_names_list: list[str],
) -> DataFrame:
    """Remove ARCHIVE, GDPR, HOUSEKEEPING, and caller-supplied out-of-scope streams."""
    return dependency_df_full.filter(
        ~upper(col("stream_name")).contains("ARCHIVE")
        & ~upper(col("stream_name")).contains("GDPR")
        & ~upper(col("stream_name")).contains("HOUSEKEEPING")
        & ~upper(col("stream_name")).isin(outofscope_stream_names_list)
    )


def treat_tgt_as_src(dependency_df_filtered: DataFrame) -> DataFrame:
    """Duplicate TGT/TGT_TRNS rows as SRC/SRC_TRNS, union, deduplicate."""
    tgt_as_source = (
        dependency_df_filtered
        .filter(upper(col("table_type")).contains("TGT"))
        .replace({"Tgt": "Src", "Tgt_Trns": "Src_Trns"}, subset=["table_type"])
    )
    return dependency_df_filtered.union(tgt_as_source).distinct()


# ---------------------------------------------------------------------------
# Step 2 — cross-stream dependency formation
# ---------------------------------------------------------------------------


def form_stream_stream_dependencies(
    dependency_df: DataFrame,
    report_dependency_df: DataFrame | None,
    table_size_df: DataFrame | None,
) -> DataFrame:
    """
    Self-join dependency_df on table_name to find cross-stream table links.

    Joins table sizes when available; produces a null ``size`` column otherwise.
    ``report_dependency_df`` is accepted for future activation of the report-join
    branch (currently a TODO in leiden.py) — the function does not crash when it
    is None.

    Returns
    -------
    DataFrame
        Columns: from, to, table, size
    """
    dep = dependency_df.select(
        "stream_name",
        col("DB_Table_Name").alias("table_name"),
        "table_type",
    ).distinct()

    cross = (
        dep.alias("df1")
        .join(dep.alias("df2"), col("df1.table_name") == col("df2.table_name"))
        .filter(col("df1.stream_name") != col("df2.stream_name"))
    )

    if table_size_df is not None:
        joined = (
            cross.join(
                table_size_df.alias("table_size"),
                col("df1.table_name") == col("table_size.table_name"),
                "left",
            ).select("df1.*", "df2.*", col("table_size.size"))
        )
    else:
        joined = cross.select("df1.*", "df2.*", lit(None).cast("double").alias("size"))

    tgt_types = ("TGT", "TGT_TRNS", "FILE")
    src_types = ("SRC", "SRC_TRNS", "FILE")

    src_tgt = joined.filter(
        upper(col("df1.table_type").cast("string")).isin(tgt_types)
        & upper(col("df2.table_type").cast("string")).isin(src_types)
    )

    return src_tgt.select(
        col("df1.stream_name").alias("from"),
        col("df2.stream_name").alias("to"),
        col("df1.table_name").alias("table"),
        col("size"),
    )


# ---------------------------------------------------------------------------
# Step 4 — merge bidirectional edges
# ---------------------------------------------------------------------------


def merge_bidirectional_edges(
    weighted_stream_stream_dependency_df: DataFrame,
) -> DataFrame:
    """
    Combine A→B and B→A into a single undirected edge by summing weights.
    Unidirectional edges are kept as-is.

    Returns
    -------
    DataFrame
        Columns: streamA, streamB, weight
    """
    w = weighted_stream_stream_dependency_df

    bidir = (
        w.alias("forward")
        .join(
            w.alias("backward"),
            (col("forward.from") == col("backward.to"))
            & (col("forward.to") == col("backward.from")),
        )
        .select(
            when(col("forward.weight") > col("backward.weight"), col("forward.from"))
            .otherwise(col("backward.from")).alias("streamA"),
            when(col("forward.weight") > col("backward.weight"), col("forward.to"))
            .otherwise(col("backward.to")).alias("streamB"),
            (col("forward.weight") + col("backward.weight")).alias("weight"),
            sort_array(array(col("forward.from"), col("forward.to"))).alias("pair_key"),
        )
        .dropDuplicates(["pair_key"])
        .select("streamA", "streamB", "weight")
    )

    unidir = (
        w.alias("forward")
        .join(
            w.alias("backward"),
            (col("forward.from") == col("backward.to"))
            & (col("forward.to") == col("backward.from")),
            "left_anti",
        )
        .select(
            col("forward.from").alias("streamA"),
            col("forward.to").alias("streamB"),
            col("forward.weight").alias("weight"),
        )
    )

    return bidir.union(unidir)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_stream_dependencies(
    dependency_df_full: DataFrame,
    outofscope_stream_names_list: list[str],
    report_dependency_df: DataFrame | None,
    table_size_df: DataFrame | None,
    weight_method: str = "factor",
) -> tuple[DataFrame, DataFrame]:
    """
    Run the complete preprocessing pipeline.

    Applies stream filtering, TGT-as-SRC normalisation, cross-stream dependency
    formation, weight calculation (method selected by ``weight_method``),
    deduplication, aggregation, and bidirectional edge merging.

    Parameters
    ----------
    dependency_df_full:
        Raw stream-table dependency table.
    outofscope_stream_names_list:
        Stream names to exclude from the analysis.
    report_dependency_df:
        Optional report-to-table dependency table (reserved for future use).
    table_size_df:
        Optional table size lookup; produces null size when absent.
    weight_method:
        Weight calculation strategy — ``"factor"`` (default) or ``"scaled"``.
        Passed through to :func:`weights.calculate_table_weights`.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        ``(dependency_df, merged_dependency_df)`` where

        * ``dependency_df`` – filtered and TGT-as-SRC normalised stream-table
          dependency table; required by :func:`graph_builder.find_isolated_streams`.
        * ``merged_dependency_df`` – weighted, undirected edge list with columns
          ``streamA``, ``streamB``, ``weight``; ready for graph construction.
    """
    filtered = filter_admin_streams(dependency_df_full, outofscope_stream_names_list)
    dependency_df = treat_tgt_as_src(filtered)
    stream_deps = form_stream_stream_dependencies(dependency_df, report_dependency_df, table_size_df)
    table_weights = calculate_table_weights(stream_deps, method=weight_method)
    unique_weights = deduplicate_table_weights(table_weights)
    weighted = aggregate_edge_weights(unique_weights)
    merged_dependency_df = merge_bidirectional_edges(weighted)
    return dependency_df, merged_dependency_df
