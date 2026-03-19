"""
Analysis functions for Leiden community detection results.

All functions in this module operate on pre-computed data passed as
explicit parameters — no values are recomputed here.
"""
from __future__ import annotations

import itertools
import os
import time
from collections.abc import Iterable
from datetime import datetime
from math import factorial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


# ---------------------------------------------------------------------------
# Module-level helpers (reused across multiple public functions)
# ---------------------------------------------------------------------------


def split_bi_etl(streams: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split stream names into (bi_reports, etl_streams) by ``'json'`` in name.

    Streams whose name contains ``'json'`` (case-insensitive) are classified
    as BI reports; all others are ETL streams.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(bi_reports, etl_streams)``
    """
    bi = [s for s in streams if "json" in s.lower()]
    etl = [s for s in streams if "json" not in s.lower()]
    return bi, etl


def _coerce_size(df: pd.DataFrame, col: str = "size") -> pd.DataFrame:
    """Return a *copy* of *df* with *col* coerced to ``float``; NaN → 0.0."""
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _safe_pct(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator * 100``, or 0.0 when *denominator* is zero."""
    return numerator / denominator * 100.0 if denominator else 0.0


def _global_stream_totals(
    leiden_df: pd.DataFrame,
    complexity_pdf: pd.DataFrame | None = None,
) -> tuple[int, int, int, float]:
    """Return ``(total_streams, total_bi, total_etl, total_complexity)``.

    Parameters
    ----------
    leiden_df:
        Must have a ``stream`` column.
    complexity_pdf:
        Optional DataFrame with a ``complexity_score`` column; ``total_complexity``
        is 0.0 when *None*.
    """
    all_streams = set(leiden_df["stream"].tolist())
    bi, etl = split_bi_etl(all_streams)
    total_complexity = (
        float(complexity_pdf["complexity_score"].sum())
        if complexity_pdf is not None
        else 0.0
    )
    return len(all_streams), len(bi), len(etl), total_complexity


def _classify_tables_by_consumer_type(
    table_dep_df: pd.DataFrame,
    consumer_col: str = "to",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify tables as BI or ETL based on the consuming stream name.

    A table consumed by *both* BI and ETL streams is classified as ETL
    (ETL-wins rule).

    Parameters
    ----------
    table_dep_df:
        DataFrame with at least columns ``table``, *consumer_col*, ``size``.
    consumer_col:
        Column containing the consuming stream name (default ``"to"``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(bi_tables, etl_tables)`` — each has columns
        ``table``, ``is_bi``, ``is_etl``, ``size``, ``category``.
    """
    if table_dep_df.empty:
        empty = pd.DataFrame(columns=["table", "is_bi", "is_etl", "size", "category"])
        return empty, empty

    df = table_dep_df.copy()
    df["is_bi"] = df[consumer_col].str.lower().str.contains("json")
    df["is_etl"] = ~df["is_bi"]

    classified = (
        df.groupby("table")
        .agg({"is_bi": "any", "is_etl": "any", "size": "first"})
        .reset_index()
    )
    classified["category"] = classified.apply(
        lambda row: "ETL" if row["is_etl"] else "BI", axis=1
    )

    bi = classified[classified["category"] == "BI"]
    etl = classified[classified["category"] == "ETL"]
    return bi, etl


def _filter_boundary_tables(
    stream_table_pdf: pd.DataFrame,
    streams_in_community: set,
    direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(all_rows, unique_by_table)`` for community-boundary tables.

    Parameters
    ----------
    direction : ``{"incoming", "outgoing"}``
        ``"incoming"`` — ``from`` ∉ community **and** ``to`` ∈ community.
        ``"outgoing"`` — ``from`` ∈ community **and** ``to`` ∉ community.
    """
    cols = ["table", "from", "to", "size"]
    if direction == "incoming":
        mask = (
            ~stream_table_pdf["from"].isin(streams_in_community)
            & stream_table_pdf["to"].isin(streams_in_community)
        )
    else:
        mask = (
            stream_table_pdf["from"].isin(streams_in_community)
            & ~stream_table_pdf["to"].isin(streams_in_community)
        )
    all_rows = stream_table_pdf[mask][cols].sort_values("table")
    unique_rows = all_rows.drop_duplicates(subset=["table"])
    return all_rows, unique_rows


def _format_table_entry(
    table: str,
    table_instances: pd.DataFrame,
    writer_label: str = "Written by (outside)",
    reader_label: str = "Read by (inside)",
    suffix: str = "",
) -> str:
    """Return a formatted 3-line table entry for report text.

    Parameters
    ----------
    table:
        Table name to display.
    table_instances:
        Rows for this table; must have ``size``, ``from``, and ``to`` columns.
    writer_label:
        Label for the producer side.
    reader_label:
        Label for the consumer side.
    suffix:
        Optional marker appended after the size on the first line (e.g.
        ``" [MULTI-PRODUCER & MULTI-CONSUMER]"``).
    """
    size = table_instances["size"].iloc[0]
    producers = sorted(table_instances["from"].unique())
    consumers = sorted(table_instances["to"].unique())
    return (
        f"\n  Table: {table} ({size:.2f} GB){suffix}\n"
        f"    - {writer_label}: {', '.join(producers)}\n"
        f"    - {reader_label}: {', '.join(consumers)}\n"
    )


def build_stream_produces_mapping(dependency_df: "DataFrame") -> dict[str, set]:
    """Return ``{stream_name -> set of uppercase TGT/TGT_TRNS table names}``.

    Extracts which tables each stream *produces* (writes) by filtering
    ``table_type`` to ``TGT`` and ``TGT_TRNS`` rows.

    Parameters
    ----------
    dependency_df:
        Preprocessed Spark DataFrame with columns ``stream_name``,
        ``DB_Table_Name``, ``table_type``.
    """
    from pyspark.sql.functions import col, upper  # noqa: PLC0415

    pdf = (
        dependency_df.filter(
            (upper(col("table_type")) == "TGT")
            | (upper(col("table_type")) == "TGT_TRNS")
        )
        .select(
            col("stream_name"),
            upper(col("DB_Table_Name")).alias("table_name"),
        )
        .distinct()
        .toPandas()
    )
    return pdf.groupby("stream_name")["table_name"].apply(set).to_dict()


def build_report_required_tables(report_dependency_df: "DataFrame") -> dict[str, set]:
    """Return ``{report_name -> set of uppercase required table names}``.

    Parameters
    ----------
    report_dependency_df:
        Spark DataFrame with columns ``stream_name`` (report name) and
        ``table_name``.
    """
    from pyspark.sql.functions import col, upper  # noqa: PLC0415

    pdf = (
        report_dependency_df.select(
            col("stream_name").alias("report_name"),
            upper(col("table_name")).alias("table_name"),
        )
        .distinct()
        .toPandas()
    )
    return pdf.groupby("report_name")["table_name"].apply(set).to_dict()


# ---------------------------------------------------------------------------
# Membership helpers
# ---------------------------------------------------------------------------


def membership_to_leiden_df(membership: list, igraph_names) -> pd.DataFrame:
    """Convert a membership array to a pandas DataFrame.

    Parameters
    ----------
    membership:
        List or array of integer community IDs, one per node.
    igraph_names:
        Array of node names corresponding to the membership indices.

    Returns
    -------
    pd.DataFrame
        Columns: ``stream``, ``community``.
    """
    return pd.DataFrame(
        {
            "stream": np.array(igraph_names),
            "community": np.array(membership, dtype=int),
        }
    )


def get_leiden_df(
    resolution: float,
    rep_by_res: dict,
    igraph_names,
) -> tuple[pd.DataFrame, dict]:
    """Return the Leiden DataFrame and metadata for a given resolution.

    Parameters
    ----------
    resolution:
        Leiden resolution parameter.
    rep_by_res:
        Mapping of resolution → representative run dict (keys: ``membership``,
        ``seed``, ``quality``, optional ``resolution``).
    igraph_names:
        Array of node names (same order as membership arrays).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(leiden_df, meta)`` where ``leiden_df`` has columns
        ``stream`` / ``community`` and ``meta`` contains scalar
        ``resolution``, ``seed``, and ``quality``.
    """
    rep = rep_by_res[resolution]
    leiden_df = membership_to_leiden_df(rep["membership"], igraph_names)
    meta = {
        "resolution": float(rep.get("resolution", resolution)),
        "seed": int(rep["seed"]),
        "quality": float(rep.get("quality", np.nan)),
    }
    return leiden_df, meta


# ---------------------------------------------------------------------------
# Community split
# ---------------------------------------------------------------------------


def split_communities_topN(
    leiden_df: pd.DataFrame,
    unique_table_weights_df: "DataFrame",
    spark: "SparkSession",
    top_n: int = 10,
) -> tuple[list, list, dict]:
    """Split communities into top-N heaviest and the rest.

    Heaviness is measured as total incoming table weight — the sum of weights
    for tables read by a community's streams but written by streams outside.

    Parameters
    ----------
    leiden_df:
        DataFrame with columns ``stream`` and ``community``.
    unique_table_weights_df:
        Spark DataFrame with columns ``from``, ``to``, ``table``, ``weight``.
        This is the deduplicated stream-stream-table weight table produced by
        :func:`~migration_planner.planner_core.weights.deduplicate_table_weights`.
    spark:
        Active :class:`~pyspark.sql.SparkSession`.
    top_n:
        Number of top communities to select (default 10).

    Returns
    -------
    tuple[list, list, dict]
        * ``top_n_ids`` – community IDs for the top-N heaviest communities.
        * ``rest_ids`` – community IDs for the remaining communities.
        * ``community_weights`` – mapping of community ID → total incoming weight.
    """
    from pyspark.sql.functions import broadcast
    from pyspark.sql.functions import col
    from pyspark.sql.functions import sum as spark_sum

    print(f"Splitting communities into top {top_n} by incoming table weight...")

    # Convert leiden_df to Spark DataFrame for distributed join
    leiden_spark = spark.createDataFrame(leiden_df)

    # Broadcast the small leiden mapping for efficient joins
    leiden_broadcast = broadcast(leiden_spark)

    # Join unique_table_weights with community assignments for both 'from' and 'to' streams
    edges_with_communities = (
        unique_table_weights_df
        .select("from", "to", "table", "weight")
        .join(
            leiden_broadcast.selectExpr("stream as from", "community as from_community"),
            on="from",
            how="inner",
        )
        .join(
            leiden_broadcast.selectExpr("stream as to", "community as to_community"),
            on="to",
            how="inner",
        )
    )

    # Filter for incoming edges: tables read by streams in a community but written outside
    incoming_edges = edges_with_communities.filter(
        col("from_community") != col("to_community")
    )

    # Aggregate: sum weight by to_community
    community_weights_spark = (
        incoming_edges
        .groupBy("to_community")
        .agg(spark_sum("weight").alias("total_incoming_weight"))
        .orderBy(col("total_incoming_weight").desc())
    )

    # Convert only the small aggregated result to pandas
    community_weights_pdf = community_weights_spark.toPandas()

    # Create dictionary mapping community -> weight
    community_weights = dict(
        zip(community_weights_pdf["to_community"], community_weights_pdf["total_incoming_weight"])
    )

    # Handle communities with no incoming edges (weight = 0)
    all_communities = set(leiden_df["community"].unique())
    for comm in all_communities:
        if comm not in community_weights:
            community_weights[comm] = 0.0

    # Sort communities by weight (descending)
    sorted_comms = sorted(community_weights.items(), key=lambda x: x[1], reverse=True)
    top_n_actual = min(top_n, len(sorted_comms))
    top_n_ids = [comm_id for comm_id, _ in sorted_comms[:top_n_actual]]
    rest_ids = [comm_id for comm_id, _ in sorted_comms[top_n_actual:]]

    print(f"\n=== Community Split (top_n={top_n}) ===")
    print(f"Total communities: {len(sorted_comms)}")
    print(f"Top {top_n_actual} communities: {top_n_ids}")
    print(f"Remaining communities: {len(rest_ids)}")
    print(f"\nTop {top_n_actual} community incoming weights:")
    for comm_id, weight in sorted_comms[:top_n_actual]:
        num_streams = len(leiden_df[leiden_df["community"] == comm_id])
        print(f"  Community {comm_id}: incoming_weight={weight:.2f}, streams={num_streams}")

    print(f"\nCommunities split into top {top_n_actual} communities: {top_n_ids} and {len(rest_ids)} others")
    return top_n_ids, rest_ids, community_weights


# ---------------------------------------------------------------------------
# Brute-force community ordering
# ---------------------------------------------------------------------------


class BruteForceCommunityOrdering:
    """
    Brute-force search over all community orderings for a given subset.

    Cost = SUM of per-table ``weight`` that must be synced at each step:

    * ``to_sync(step) = incoming_tables[community] - available_tables_so_far``
    * ``step_cost = sum(table_weight[t] for t in to_sync)``
    * ``total_cost = sum(step_cost)``

    Supports pre-available tables: when optimising a subset of communities you
    can specify other communities whose produced tables should be considered
    already available (e.g. when optimising top-N after rest communities).

    Optimised with:

    * Bitmask representation for table sets (bitwise OR/AND instead of Python sets)
    * Numpy weight vector for vectorised cost calculation
    * Branch-and-bound pruning to skip permutations that exceed the current best
    * Lazy ``step_costs`` (only computed for new best solutions)
    """

    def __init__(
        self,
        dep_pdf: pd.DataFrame,
        leiden_df: pd.DataFrame,
        communities_subset: list | None = None,
        pre_available_communities: list | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dep_pdf:
            Pre-collected pandas DataFrame with columns ``from``, ``to``,
            ``table``, ``weight``.  Avoids repeated ``.toPandas()`` calls
            when multiple instances are created from the same Spark source.
        leiden_df:
            DataFrame with columns ``stream``, ``community``.
        communities_subset:
            Community IDs to optimise.  ``None`` optimises all communities.
        pre_available_communities:
            Community IDs whose produced tables are considered already available
            at the start of the optimisation.
        """
        self.dep = dep_pdf.copy()

        self.dep["from"] = self.dep["from"].astype(str)
        self.dep["to"] = self.dep["to"].astype(str)
        self.dep["table"] = self.dep["table"].astype(str)
        self.dep["weight"] = pd.to_numeric(self.dep["weight"], errors="coerce").fillna(0.0)

        # table → single weight (max)
        self.table_weight = (
            self.dep.groupby("table")["weight"].max().to_dict()
        )

        # choose communities to optimise
        all_comms = sorted(leiden_df["community"].unique().astype(int).tolist())
        if communities_subset is None:
            comms = all_comms
        else:
            comms = sorted([int(c) for c in communities_subset])

        # Build community_streams for ALL communities (needed for pre-available calc)
        all_community_streams = {
            int(c): set(leiden_df.loc[leiden_df["community"] == c, "stream"].astype(str).tolist())
            for c in all_comms
        }

        # Store only the subset we're optimising
        self.community_streams = {c: all_community_streams[c] for c in comms}

        # produced tables per community (for communities we're optimising)
        self.produced_tables: dict[int, set] = {}
        for c, streams in self.community_streams.items():
            self.produced_tables[c] = set(
                self.dep[self.dep["from"].isin(streams)]["table"].unique()
            )

        # incoming tables per community (for communities we're optimising)
        self.incoming_tables: dict[int, set] = {}
        for c, streams in self.community_streams.items():
            incoming = self.dep[
                (~self.dep["from"].isin(streams))
                & (self.dep["to"].isin(streams))
            ]["table"].unique()
            self.incoming_tables[c] = set(incoming)

        self.communities = list(self.community_streams.keys())

        # Calculate pre-available tables from other communities
        self.pre_available_tables: set = set()
        if pre_available_communities is not None:
            print(f"  Pre-available communities: {sorted(pre_available_communities)}")
            for c in pre_available_communities:
                if c in all_community_streams:
                    streams = all_community_streams[c]
                    tables = set(
                        self.dep[self.dep["from"].isin(streams)]["table"].unique()
                    )
                    self.pre_available_tables |= tables
            print(f"  Pre-available tables count: {len(self.pre_available_tables)}")

        # --- Build bitmask representation for fast evaluation ---
        all_tables = sorted(self.table_weight.keys())
        self._table_to_idx = {t: i for i, t in enumerate(all_tables)}
        self._n_tables = len(all_tables)

        self._weight_vec = np.zeros(self._n_tables, dtype=np.float64)
        for t, w in self.table_weight.items():
            self._weight_vec[self._table_to_idx[t]] = w

        self._incoming_mask: dict[int, np.ndarray] = {}
        for c in self.communities:
            mask = np.zeros(self._n_tables, dtype=np.bool_)
            for t in self.incoming_tables[c]:
                if t in self._table_to_idx:
                    mask[self._table_to_idx[t]] = True
            self._incoming_mask[c] = mask

        self._produced_mask: dict[int, np.ndarray] = {}
        for c in self.communities:
            mask = np.zeros(self._n_tables, dtype=np.bool_)
            for t in self.produced_tables[c]:
                if t in self._table_to_idx:
                    mask[self._table_to_idx[t]] = True
            self._produced_mask[c] = mask

        self._pre_available_mask = np.zeros(self._n_tables, dtype=np.bool_)
        for t in self.pre_available_tables:
            if t in self._table_to_idx:
                self._pre_available_mask[self._table_to_idx[t]] = True

        print(f"  Bitmask optimization: {self._n_tables} unique tables indexed")

    def evaluate_ordering_cost(
        self,
        ordering: list | tuple,
        best_cost: float = float("inf"),
        return_step_costs: bool = False,
    ) -> tuple[float, list | None]:
        """Evaluate total sync cost with bitmask + numpy, with branch-and-bound pruning.

        Parameters
        ----------
        ordering:
            Sequence of community IDs.
        best_cost:
            Current best cost for pruning.  Returns ``(inf, None)`` early if
            the running total reaches or exceeds this value.
        return_step_costs:
            If ``True``, also compute per-step costs.

        Returns
        -------
        tuple[float, list | None]
            ``(total_cost, step_costs or None)``
        """
        available = self._pre_available_mask.copy()
        total = 0.0
        step_costs: list | None = [] if return_step_costs else None

        for c in ordering:
            to_sync = self._incoming_mask[c] & ~available
            step = float(np.dot(to_sync, self._weight_vec))
            total += step
            if return_step_costs:
                step_costs.append(step)
            if total >= best_cost:
                return float("inf"), None
            available |= self._produced_mask[c]

        return total, step_costs

    def brute_force(self, log_every: int = 5000, label: str = "subset") -> dict:
        """Perform a brute-force search over all permutations.

        Uses bitmask + numpy for fast cost evaluation and branch-and-bound
        pruning to skip permutations that exceed the current best cost.

        Parameters
        ----------
        log_every:
            Print progress every N permutations.
        label:
            Label used in progress messages.

        Returns
        -------
        dict
            Keys: ``best_cost``, ``best_order``, ``best_step_costs``,
            ``total_time_sec``, ``total_perms``.
        """
        n = len(self.communities)
        total_perms = factorial(n)

        best_cost = float("inf")
        best_order = None
        best_step_costs = None
        pruned_count = 0

        start = time.time()

        print(f"\n=== Brute force ({label}) | communities={n} | perms={total_perms} ===")
        print("  Optimization objective: Minimize total sync cost across all steps")
        print("  Optimizations: bitmask + numpy + branch-and-bound pruning")
        if self.pre_available_tables:
            print(f"  Starting with {len(self.pre_available_tables)} pre-available tables")

        for i, perm in enumerate(itertools.permutations(self.communities), 1):
            cost, _ = self.evaluate_ordering_cost(perm, best_cost=best_cost)

            if cost < best_cost:
                best_cost = cost
                best_order = perm
                _, best_step_costs = self.evaluate_ordering_cost(
                    perm, return_step_costs=True,
                )
                print(f"[NEW BEST] {i}/{total_perms} cost={best_cost:.6f} order={list(best_order)}")
            elif cost == float("inf"):
                pruned_count += 1

            if i % log_every == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = total_perms - i
                eta = remaining / rate if rate > 0 else float("inf")
                prune_pct = (pruned_count / i * 100) if i > 0 else 0
                print(
                    f"[PROGRESS] {i}/{total_perms} "
                    f"({100 * i / total_perms:.2f}%) | "
                    f"best={best_cost:.6f} | "
                    f"{rate:.1f} perms/sec | "
                    f"pruned={prune_pct:.1f}% | "
                    f"elapsed={elapsed / 60:.2f} min | "
                    f"eta={eta / 60:.2f} min"
                )

        total_time = time.time() - start
        print(f"\nDONE ({label}) in {total_time / 60:.2f} min")
        print(f"BEST COST: {best_cost:.6f}")
        print(f"BEST ORDER: {list(best_order)}")
        print(f"Pruned: {pruned_count}/{total_perms} ({pruned_count / total_perms * 100:.1f}%)")

        return {
            "best_cost": float(best_cost),
            "best_order": list(best_order),
            "best_step_costs": best_step_costs,
            "total_time_sec": total_time,
            "total_perms": total_perms,
        }


# ---------------------------------------------------------------------------
# Execution metadata logging
# ---------------------------------------------------------------------------


def append_execution_metadata(
    spark: "SparkSession",
    weight_method: str,
    top_n: int,
    resolutions_dict: dict,
) -> str:
    """Append execution metadata to a managed Unity Catalog Delta table.

    Creates the table if it does not exist.

    Parameters
    ----------
    spark:
        Active :class:`~pyspark.sql.SparkSession`.
    weight_method:
        Weight calculation method label (e.g. ``"Factor based"``).
    top_n:
        Number of top communities used for ordering optimisation.
    resolutions_dict:
        Mapping of ``resolution → (total_sync_cost, num_communities)``.

    Returns
    -------
    str
        Name of the metadata table.
    """
    table_name = "odp_adw_utilities_n.planning.execution_metadata"

    execution_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for resolution, (total_cost, num_communities) in sorted(resolutions_dict.items()):
        rows.append(
            {
                "execution_datetime": execution_datetime,
                "weight_calculation_method": weight_method,
                "top_n": int(top_n),
                "resolution": float(resolution),
                "number_of_communities": int(num_communities),
                "total_sync_cost_gb": float(total_cost),
            }
        )

    metadata_df = spark.createDataFrame(rows)

    table_exists = spark.catalog.tableExists(table_name)
    if table_exists:
        metadata_df.write.mode("append").saveAsTable(table_name)
        print(f"Execution metadata appended to table: {table_name}")
    else:
        metadata_df.write.mode("overwrite").saveAsTable(table_name)
        print(f"Execution metadata table created: {table_name}")

    print(f"Appended {len(rows)} rows to {table_name}")
    return table_name


# ---------------------------------------------------------------------------
# Community analysis report generation
# ---------------------------------------------------------------------------


def generate_community_analysis(
    leiden_df: pd.DataFrame,
    stream_table_pdf: pd.DataFrame,
    merged_edges_pdf: pd.DataFrame,
    complexity_pdf: pd.DataFrame,
    resolution: float,
    outdir: str = "leiden_community_plots",
) -> list[str]:
    """Write a structured analysis text file for every community.

    For each community, the analysis covers: stream inventory (BI vs ETL),
    complexity breakdown, incoming/outgoing table dependencies, aggregated
    stream connections, and a summary block.

    Parameters
    ----------
    leiden_df:
        DataFrame with columns ``['stream', 'community']``.
    stream_table_pdf:
        Pre-converted pandas DataFrame of stream-table dependencies.
        Expected columns: ``from``, ``to``, ``table``, ``size``.
        The DataFrame is not modified in place.
    merged_edges_pdf:
        Pre-converted pandas DataFrame of merged stream-stream edges.
        Expected columns: ``streamA``, ``streamB``, ``weight``.
    complexity_pdf:
        Pre-converted pandas DataFrame of stream complexity scores.
        Expected columns: ``stream_name``, ``complexity_score``.
    resolution:
        Leiden resolution written into each analysis file header.
    outdir:
        Root output directory.  One ``community_<N>/`` subdirectory is
        created per community.

    Returns
    -------
    list[str]
        Paths of ``.txt`` analysis files written (one per community).
    """
    os.makedirs(outdir, exist_ok=True)

    stream_table_pdf = _coerce_size(stream_table_pdf)
    total_streams_global, total_bi_reports_global, total_etl_streams_global, total_complexity_global = (
        _global_stream_totals(leiden_df, complexity_pdf)
    )

    communities = sorted(leiden_df["community"].unique())
    saved_files = []

    for c in communities:
        comm_dir = os.path.join(outdir, f"community_{c}")
        os.makedirs(comm_dir, exist_ok=True)

        streams_in_community = set(leiden_df[leiden_df["community"] == c]["stream"].tolist())

        streams_list = sorted(streams_in_community)
        bi_reports, etl_streams = split_bi_etl(streams_list)

        pct_total = _safe_pct(len(streams_list), total_streams_global)
        pct_bi = _safe_pct(len(bi_reports), total_bi_reports_global)
        pct_etl = _safe_pct(len(etl_streams), total_etl_streams_global)

        community_complexity_df = complexity_pdf[
            complexity_pdf['stream_name'].isin(streams_in_community)
        ].copy()
        stream_complexity_map = dict(
            zip(community_complexity_df['stream_name'], community_complexity_df['complexity_score'])
        )
        total_community_complexity = community_complexity_df['complexity_score'].sum()
        pct_complexity = _safe_pct(total_community_complexity, total_complexity_global)
        bi_complexity_df = community_complexity_df[
            community_complexity_df['stream_name'].isin(bi_reports)
        ]
        etl_complexity_df = community_complexity_df[
            community_complexity_df['stream_name'].isin(etl_streams)
        ]
        total_bi_complexity = bi_complexity_df['complexity_score'].sum()
        total_etl_complexity = etl_complexity_df['complexity_score'].sum()

        tables_src_inside_tgt_outside_all, tables_src_inside_tgt_outside_unique = (
            _filter_boundary_tables(stream_table_pdf, streams_in_community, "incoming")
        )
        tables_incoming_bi, tables_incoming_etl = _classify_tables_by_consumer_type(
            tables_src_inside_tgt_outside_all
        )
        total_size_incoming_bi = tables_incoming_bi['size'].sum()
        total_size_incoming_etl = tables_incoming_etl['size'].sum()
        total_size_incoming = total_size_incoming_bi + total_size_incoming_etl

        tables_tgt_inside_src_outside_all, tables_tgt_inside_src_outside_unique = (
            _filter_boundary_tables(stream_table_pdf, streams_in_community, "outgoing")
        )
        total_size_outgoing = tables_tgt_inside_src_outside_unique['size'].sum()

        outgoing_edges = merged_edges_pdf[
            (merged_edges_pdf['streamA'].isin(streams_in_community))
            & (~merged_edges_pdf['streamB'].isin(streams_in_community))
        ]
        incoming_edges = merged_edges_pdf[
            (~merged_edges_pdf['streamA'].isin(streams_in_community))
            & (merged_edges_pdf['streamB'].isin(streams_in_community))
        ]

        analysis_content = f"""Community {c} Analysis (Resolution γ={resolution})
{'='*80}

1. STREAMS IN COMMUNITY ({len(streams_list)} total streams, {pct_total:.2f}% of all streams):
{'-'*80}

1a. BI REPORTS/STREAMS ({len(bi_reports)} streams with 'json' in name, {pct_bi:.2f}% of all BI reports):
{'-'*80}
"""
        if len(bi_reports) > 0:
            for i, stream in enumerate(bi_reports, 1):
                complexity = stream_complexity_map.get(stream, 0.0)
                analysis_content += f"{i}. {stream} (complexity: {complexity:.2f})\n"
        else:
            analysis_content += "  (No BI reports/streams found)\n"

        analysis_content += (
            f"\n1b. ETL STREAMS ({len(etl_streams)} streams without 'json' in name, "
            f"{pct_etl:.2f}% of all ETL streams):\n{'-'*80}\n"
        )
        if len(etl_streams) > 0:
            for i, stream in enumerate(etl_streams, 1):
                complexity = stream_complexity_map.get(stream, 0.0)
                analysis_content += f"{i}. {stream} (complexity: {complexity:.2f})\n"
        else:
            analysis_content += "  (No ETL streams found)\n"

        analysis_content += f"""
2. COMPLEXITY ANALYSIS:
{'-'*80}
Total Community Complexity: {total_community_complexity:.2f}
Percentage of Total Complexity: {pct_complexity:.2f}%
  - BI Reports Complexity: {total_bi_complexity:.2f}
  - ETL Streams Complexity: {total_etl_complexity:.2f}
"""

        analysis_content += f"""
3. TABLES - SRC OF STREAMS INSIDE, TGT OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community READ FROM, but are WRITTEN BY streams outside.
(Dependencies flowing INTO the community - SYNC REQUIREMENTS)
Total: {len(tables_src_inside_tgt_outside_unique)} unique tables, {total_size_incoming:.2f} GB
  - For BI Reports: {len(tables_incoming_bi)} tables, {total_size_incoming_bi:.2f} GB
  - For ETL Streams: {len(tables_incoming_etl)} tables, {total_size_incoming_etl:.2f} GB
Total Instances: {len(tables_src_inside_tgt_outside_all)}\n\n"""

        if len(tables_src_inside_tgt_outside_all) > 0:
            if len(tables_incoming_bi) > 0:
                analysis_content += (
                    f"\n3a. FOR BI REPORTS ({len(tables_incoming_bi)} tables, "
                    f"{total_size_incoming_bi:.2f} GB):\n{'-'*80}\n"
                )
                for table in tables_incoming_bi['table']:
                    table_instances = tables_src_inside_tgt_outside_all[
                        tables_src_inside_tgt_outside_all['table'] == table
                    ]
                    analysis_content += _format_table_entry(table, table_instances)
            if len(tables_incoming_etl) > 0:
                analysis_content += (
                    f"\n3b. FOR ETL STREAMS ({len(tables_incoming_etl)} tables, "
                    f"{total_size_incoming_etl:.2f} GB):\n{'-'*80}\n"
                )
                for table in tables_incoming_etl['table']:
                    table_instances = tables_src_inside_tgt_outside_all[
                        tables_src_inside_tgt_outside_all['table'] == table
                    ]
                    analysis_content += _format_table_entry(table, table_instances)
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""
4. TABLES - TGT OF STREAMS INSIDE, SRC OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community WRITE TO, but are READ BY streams outside.
(Dependencies flowing OUT OF the community)
Total: {len(tables_tgt_inside_src_outside_unique)} unique tables
Total Size: {total_size_outgoing:.2f} GB
Total Instances: {len(tables_tgt_inside_src_outside_all)}\n\n"""

        if len(tables_tgt_inside_src_outside_all) > 0:
            for table in sorted(tables_tgt_inside_src_outside_unique['table'].unique()):
                table_instances = tables_tgt_inside_src_outside_all[
                    tables_tgt_inside_src_outside_all['table'] == table
                ]
                analysis_content += _format_table_entry(
                    table, table_instances,
                    writer_label="Written by (inside)",
                    reader_label="Read by (outside)",
                )
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""
5. AGGREGATED STREAM CONNECTIONS:
{'-'*80}

5a. Outgoing Stream Connections (Inside → Outside):
"""
        if len(outgoing_edges) > 0:
            analysis_content += f"Total: {len(outgoing_edges)} connections\n\n"
            for _, row in outgoing_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No outgoing connections)\n"

        analysis_content += "\n5b. Incoming Stream Connections (Outside → Inside):\n"
        if len(incoming_edges) > 0:
            analysis_content += f"Total: {len(incoming_edges)} connections\n\n"
            for _, row in incoming_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No incoming connections)\n"

        analysis_content += f"""
6. SUMMARY:
{'-'*80}
  - Total streams in community: {len(streams_list)} ({pct_total:.2f}% of {total_streams_global} total streams)
    * BI Reports/Streams (with 'json'): {len(bi_reports)} ({pct_bi:.2f}% of {total_bi_reports_global} total BI reports)
    * ETL Streams (without 'json'): {len(etl_streams)} ({pct_etl:.2f}% of {total_etl_streams_global} total ETL streams)
  - Total Community Complexity: {total_community_complexity:.2f} ({pct_complexity:.2f}% of total complexity)
    * BI Reports Complexity: {total_bi_complexity:.2f}
    * ETL Streams Complexity: {total_etl_complexity:.2f}
  - Tables flowing INTO community (SYNC REQUIREMENTS): {len(tables_src_inside_tgt_outside_unique)} unique tables, {total_size_incoming:.2f} GB
    * For BI Reports: {len(tables_incoming_bi)} tables, {total_size_incoming_bi:.2f} GB
    * For ETL Streams: {len(tables_incoming_etl)} tables, {total_size_incoming_etl:.2f} GB
    * Total instances: {len(tables_src_inside_tgt_outside_all)}
  - Tables flowing OUT OF community: {len(tables_tgt_inside_src_outside_unique)} unique tables, {total_size_outgoing:.2f} GB ({len(tables_tgt_inside_src_outside_all)} total instances)
  - Aggregated outgoing stream connections: {len(outgoing_edges)}
  - Aggregated incoming stream connections: {len(incoming_edges)}
"""

        analysis_file = os.path.join(comm_dir, f"community_{c}_analysis.txt")
        with open(analysis_file, 'w') as f:
            f.write(analysis_content)
        print(f"  Saved analysis: {analysis_file}")
        saved_files.append(analysis_file)

    return saved_files


# ---------------------------------------------------------------------------
# Migration order analysis
# ---------------------------------------------------------------------------


def generate_migration_order_analysis(
    leiden_df: pd.DataFrame,
    stream_table_dependency_df: "DataFrame",
    merged_edges_df: "DataFrame",
    optimized_order: list,
    resolution: float,
    complexity_scores_df: "DataFrame",
    outdir: str = "migration_order_analysis",
) -> tuple[str, float]:
    """Generate a detailed migration analysis report.

    Shows sync requirements for each community in the optimised migration order.

    Parameters
    ----------
    leiden_df:
        DataFrame with columns ``stream`` and ``community``.
    stream_table_dependency_df:
        Spark DataFrame of stream-table dependencies.
    merged_edges_df:
        Spark DataFrame of aggregated stream-stream edges.
    optimized_order:
        Ordered list of community IDs for migration.
    resolution:
        Leiden resolution parameter.
    complexity_scores_df:
        Spark DataFrame with columns ``stream_name``, ``complexity_score``.
    outdir:
        Output directory for reports.

    Returns
    -------
    tuple[str, float]
        ``(report_file_path, total_sync_cost_gb)`` where
        ``total_sync_cost_gb`` is the total cumulative sync cost in GB.
    """
    os.makedirs(outdir, exist_ok=True)

    # Convert to pandas
    stream_table_pdf = stream_table_dependency_df.toPandas()
    merged_edges_pdf = merged_edges_df.toPandas()
    complexity_pdf = complexity_scores_df.select("stream_name", "complexity_score").toPandas()

    # Coerce size column (returns a copy — no in-place mutation)
    stream_table_pdf = _coerce_size(stream_table_pdf)

    # Merge leiden_df with complexity scores
    leiden_with_complexity = leiden_df.merge(
        complexity_pdf,
        left_on="stream",
        right_on="stream_name",
        how="left",
    )
    leiden_with_complexity["complexity_score"] = leiden_with_complexity["complexity_score"].fillna(0)

    # Global totals (complexity taken from leiden_with_complexity to include merged scores)
    total_streams_global, total_bi_reports_global, total_etl_streams_global, _ = (
        _global_stream_totals(leiden_df)
    )
    total_complexity_global = leiden_with_complexity["complexity_score"].sum()

    # Build community -> streams mapping with complexity
    community_streams: dict = {}
    community_complexity: dict = {}
    for c in optimized_order:
        streams = leiden_df[leiden_df["community"] == c]["stream"].tolist()
        community_streams[c] = set(streams)
        community_complexity[c] = leiden_with_complexity[
            leiden_with_complexity["community"] == c
        ]["complexity_score"].sum()

    # Track already migrated / available tables
    available_tables: set = set()
    migrated_streams: set = set()

    # Track cumulative outstanding sync requirements
    cumulative_outstanding_sync_tables: set = set()

    # Cumulative metrics
    cumulative_streams = 0
    cumulative_bi = 0
    cumulative_etl = 0
    cumulative_sync_size = 0.0
    cumulative_sync_size_bi = 0.0
    cumulative_sync_size_etl = 0.0
    cumulative_multi_producer_consumer_tables = 0
    cumulative_complexity = 0.0

    # Lists to collect output rows
    sync_details_rows: list = []
    stream_ordering_rows: list = []

    # Generate report header
    report_content = f"""{'=' * 100}
MIGRATION ORDER ANALYSIS REPORT
Resolution γ={resolution}
Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 100}

OPTIMIZED MIGRATION ORDER: {optimized_order}

Total Communities: {len(optimized_order)}
Total Streams: {total_streams_global}
  - BI Reports: {total_bi_reports_global}
  - ETL Streams: {total_etl_streams_global}
Total Complexity Score: {total_complexity_global:.0f}

{'=' * 100}
"""

    for step, community_id in enumerate(optimized_order, 1):
        streams_in_community = community_streams[community_id]
        community_complexity_score = community_complexity[community_id]

        # Get stream-level complexity for this community
        stream_complexity_map = leiden_with_complexity[
            leiden_with_complexity["community"] == community_id
        ].set_index("stream")["complexity_score"].to_dict()

        # Add stream-to-community-to-execution_order mapping
        for stream in streams_in_community:
            stream_ordering_rows.append(
                {
                    "resolution": float(resolution),
                    "stream_name": str(stream),
                    "community_id": int(community_id),
                    "execution_order": int(step),
                }
            )

        # Separate BI and ETL
        bi_reports, etl_streams = split_bi_etl(streams_in_community)

        # Calculate percentages
        pct_total = _safe_pct(len(streams_in_community), total_streams_global)
        pct_bi = _safe_pct(len(bi_reports), total_bi_reports_global)
        pct_etl = _safe_pct(len(etl_streams), total_etl_streams_global)

        # Incoming and outgoing boundary tables
        incoming_tables_all, incoming_tables_unique = _filter_boundary_tables(
            stream_table_pdf, streams_in_community, "incoming"
        )

        # Tables that need to be synced (not yet available)
        tables_to_sync = incoming_tables_unique[
            ~incoming_tables_unique["table"].isin(available_tables)
        ]

        # Detailed sync rows for tables that are needed by this community
        tables_to_sync_details = incoming_tables_all[
            incoming_tables_all["table"].isin(tables_to_sync["table"])
        ].copy()

        # Collect sync details for CSV
        for _, row in tables_to_sync_details.iterrows():
            table_name = row["table"]
            size_gb = row["size"]
            producer_stream = row["from"]
            consumer_stream = row["to"]
            sync_details_rows.append(
                {
                    "resolution": float(resolution),
                    "execution_order": int(step),
                    "community_id": int(community_id),
                    "table_name": str(table_name),
                    "size_gb": float(size_gb),
                    "stream_name": str(producer_stream),
                    "handling_type": "written outside",
                }
            )
            sync_details_rows.append(
                {
                    "resolution": float(resolution),
                    "execution_order": int(step),
                    "community_id": int(community_id),
                    "table_name": str(table_name),
                    "size_gb": float(size_gb),
                    "stream_name": str(consumer_stream),
                    "handling_type": "read inside",
                }
            )

        # Classify each table as BI or ETL based on the consuming stream (ETL-wins rule)
        tables_to_sync_bi, tables_to_sync_etl = _classify_tables_by_consumer_type(
            tables_to_sync_details
        )
        sync_size_bi = tables_to_sync_bi["size"].sum()
        sync_size_etl = tables_to_sync_etl["size"].sum()
        sync_size = sync_size_bi + sync_size_etl

        # Tables with multiple producers AND multiple consumers
        multi_producer_consumer_tables = []
        multi_producer_consumer_size = 0.0
        for table in tables_to_sync["table"]:
            table_instances = tables_to_sync_details[tables_to_sync_details["table"] == table]
            producers = table_instances["from"].unique()
            consumers = table_instances["to"].unique()
            if len(producers) > 1 and len(consumers) > 1:
                multi_producer_consumer_tables.append(table)
                multi_producer_consumer_size += table_instances["size"].iloc[0]

        num_multi_producer_consumer = len(multi_producer_consumer_tables)
        cumulative_multi_producer_consumer_tables += num_multi_producer_consumer

        # Add tables to cumulative outstanding sync
        cumulative_outstanding_sync_tables.update(tables_to_sync["table"].tolist())

        # Tables already available (produced by previous communities)
        tables_already_available = incoming_tables_unique[
            incoming_tables_unique["table"].isin(available_tables)
        ]
        available_size = tables_already_available["size"].sum()

        # Tables this community produces (outgoing)
        outgoing_tables_all, outgoing_tables_unique = _filter_boundary_tables(
            stream_table_pdf, streams_in_community, "outgoing"
        )
        outgoing_size = outgoing_tables_unique["size"].sum()

        # Update available tables BEFORE calculating cumulative outstanding sync
        produced_tables = stream_table_pdf[
            stream_table_pdf["from"].isin(streams_in_community)
        ]["table"].unique()
        available_tables.update(produced_tables)

        # Remove newly available tables from cumulative outstanding sync
        cumulative_outstanding_sync_tables -= set(produced_tables)

        # Update migrated streams
        migrated_streams.update(streams_in_community)

        # Cumulative outstanding sync: only count deps where consuming stream is migrated
        cumulative_outstanding_sync_df = stream_table_pdf[
            (stream_table_pdf["table"].isin(cumulative_outstanding_sync_tables))
            & (stream_table_pdf["to"].isin(migrated_streams))
        ][["table", "to", "size"]].drop_duplicates(subset=["table", "to"])

        cumulative_outstanding_bi, cumulative_outstanding_etl = _classify_tables_by_consumer_type(
            cumulative_outstanding_sync_df
        )
        cumulative_outstanding_sync_count_bi = len(cumulative_outstanding_bi)
        cumulative_outstanding_sync_count_etl = len(cumulative_outstanding_etl)
        cumulative_outstanding_sync_count = (
            cumulative_outstanding_sync_count_bi + cumulative_outstanding_sync_count_etl
        )
        cumulative_outstanding_sync_size_bi = cumulative_outstanding_bi["size"].sum()
        cumulative_outstanding_sync_size_etl = cumulative_outstanding_etl["size"].sum()
        cumulative_outstanding_sync_size = (
            cumulative_outstanding_sync_size_bi + cumulative_outstanding_sync_size_etl
        )

        # Update cumulative metrics
        cumulative_streams += len(streams_in_community)
        cumulative_bi += len(bi_reports)
        cumulative_etl += len(etl_streams)
        cumulative_sync_size += sync_size
        cumulative_sync_size_bi += sync_size_bi
        cumulative_sync_size_etl += sync_size_etl
        cumulative_complexity += community_complexity_score

        # Calculate remaining
        remaining_streams = total_streams_global - cumulative_streams
        remaining_bi = total_bi_reports_global - cumulative_bi
        remaining_etl = total_etl_streams_global - cumulative_etl
        remaining_complexity = total_complexity_global - cumulative_complexity

        # Progress percentages
        progress_pct = (cumulative_streams / total_streams_global * 100) if total_streams_global > 0 else 0.0
        complexity_progress_pct = (
            cumulative_complexity / total_complexity_global * 100
        ) if total_complexity_global > 0 else 0.0

        # Write community section
        report_content += f"""\n{'=' * 100}
STEP {step}/{len(optimized_order)}: COMMUNITY {community_id}
{'=' * 100}

--- COMMUNITY COMPOSITION ---
Streams in this community: {len(streams_in_community)} ({pct_total:.2f}% of total)
  - BI Reports: {len(bi_reports)} ({pct_bi:.2f}% of all BI reports)
  - ETL Streams: {len(etl_streams)} ({pct_etl:.2f}% of all ETL streams)
Community Complexity Score: {community_complexity_score:.0f}

--- STREAMS LIST ---
"""

        if len(bi_reports) > 0:
            report_content += f"\n  ** BI REPORTS ({len(bi_reports)}) **\n"
            for stream in sorted(bi_reports):
                complexity = stream_complexity_map.get(stream, 0)
                report_content += f"    - {stream} (complexity: {complexity:.0f})\n"

        if len(etl_streams) > 0:
            report_content += f"\n  ** ETL STREAMS ({len(etl_streams)}) **\n"
            for stream in sorted(etl_streams):
                complexity = stream_complexity_map.get(stream, 0)
                report_content += f"    - {stream} (complexity: {complexity:.0f})\n"

        report_content += f"""\n--- MIGRATION PROGRESS ---
Cumulative streams migrated: {cumulative_streams}/{total_streams_global} ({progress_pct:.2f}%)
  - BI Reports migrated: {cumulative_bi}/{total_bi_reports_global}
  - ETL Streams migrated: {cumulative_etl}/{total_etl_streams_global}
Cumulative complexity migrated: {cumulative_complexity:.0f}/{total_complexity_global:.0f} ({complexity_progress_pct:.2f}%)
Remaining streams: {remaining_streams}
  - BI Reports remaining: {remaining_bi}
  - ETL Streams remaining: {remaining_etl}
Remaining complexity: {remaining_complexity:.0f}

--- SYNC REQUIREMENTS FOR THIS STEP ---
Tables to SYNC (not yet available): {len(tables_to_sync)} unique tables, {sync_size:.2f} GB
  - For BI Reports: {len(tables_to_sync_bi)} tables, {sync_size_bi:.2f} GB
  - For ETL Streams: {len(tables_to_sync_etl)} tables, {sync_size_etl:.2f} GB
  - With multiple producers AND consumers: {num_multi_producer_consumer} tables, {multi_producer_consumer_size:.2f} GB

Tables ALREADY AVAILABLE (from previous migrations): {len(tables_already_available)} unique tables, {available_size:.2f} GB
Total incoming dependencies: {len(incoming_tables_unique)} unique tables, {incoming_tables_unique['size'].sum():.2f} GB

Sync efficiency: {(available_size / (available_size + sync_size) * 100) if (available_size + sync_size) > 0 else 0:.2f}% of dependencies already available

--- CUMULATIVE OUTSTANDING SYNC AT THIS STAGE ---
Total tables still to be synced (this + earlier communities): {cumulative_outstanding_sync_count} unique tables, {cumulative_outstanding_sync_size:.2f} GB
  - For BI Reports: {cumulative_outstanding_sync_count_bi} tables, {cumulative_outstanding_sync_size_bi:.2f} GB
  - For ETL Streams: {cumulative_outstanding_sync_count_etl} tables, {cumulative_outstanding_sync_size_etl:.2f} GB

--- TABLES TO SYNC (NEW) ---
"""

        if len(tables_to_sync) > 0:
            if len(tables_to_sync_bi) > 0:
                report_content += (
                    f"\n  ** FOR BI REPORTS ({len(tables_to_sync_bi)} tables, {sync_size_bi:.2f} GB) **\n"
                )
                for table in tables_to_sync_bi["table"]:
                    table_instances = tables_to_sync_details[tables_to_sync_details["table"] == table]
                    suffix = " [MULTI-PRODUCER & MULTI-CONSUMER]" if table in multi_producer_consumer_tables else ""
                    report_content += _format_table_entry(table, table_instances, suffix=suffix)

            if len(tables_to_sync_etl) > 0:
                report_content += (
                    f"\n  ** FOR ETL STREAMS ({len(tables_to_sync_etl)} tables, {sync_size_etl:.2f} GB) **\n"
                )
                for table in tables_to_sync_etl["table"]:
                    table_instances = tables_to_sync_details[tables_to_sync_details["table"] == table]
                    suffix = " [MULTI-PRODUCER & MULTI-CONSUMER]" if table in multi_producer_consumer_tables else ""
                    report_content += _format_table_entry(table, table_instances, suffix=suffix)
        else:
            report_content += "  (No new tables to sync - all dependencies already available!)\n"

        report_content += "\n--- TABLES ALREADY AVAILABLE (FROM PREVIOUS STEPS) ---\n"

        if len(tables_already_available) > 0:
            tables_available_all = incoming_tables_all[
                incoming_tables_all["table"].isin(tables_already_available["table"])
            ]
            for table in sorted(tables_already_available["table"].unique()):
                table_instances = tables_available_all[tables_available_all["table"] == table]
                report_content += _format_table_entry(table, table_instances)
        else:
            report_content += "  (No dependencies from previous migrations)\n"

        report_content += f"""\n--- TABLES PRODUCED BY THIS COMMUNITY ---
This community produces: {len(outgoing_tables_unique)} unique tables, {outgoing_size:.2f} GB
(These will be available for subsequent communities)\n\n"""

        if len(outgoing_tables_unique) > 0:
            for table in sorted(outgoing_tables_unique["table"].unique()):
                table_instances = outgoing_tables_all[outgoing_tables_all["table"] == table]
                report_content += _format_table_entry(
                    table, table_instances,
                    writer_label="Written by (inside)",
                    reader_label="Read by (outside)",
                )

    # Final summary
    report_content += f"""\n\n{'=' * 100}
FINAL MIGRATION SUMMARY
{'=' * 100}

Total communities migrated: {len(optimized_order)}
Total streams migrated: {cumulative_streams}/{total_streams_global} (100%)
  - BI Reports: {cumulative_bi}/{total_bi_reports_global}
  - ETL Streams: {cumulative_etl}/{total_etl_streams_global}
Total complexity migrated: {cumulative_complexity:.0f}/{total_complexity_global:.0f} (100%)

Total cumulative sync cost: {cumulative_sync_size:.2f} GB
  - For BI Reports: {cumulative_sync_size_bi:.2f} GB
  - For ETL Streams: {cumulative_sync_size_etl:.2f} GB

Total tables with multiple producers AND consumers to sync: {cumulative_multi_producer_consumer_tables}
Total tables produced: {len(available_tables)} unique tables

Migration order: {optimized_order}

{'=' * 100}
"""

    # Save report
    report_file = os.path.join(outdir, f"migration_order_analysis_gamma_{resolution}.txt")
    with open(report_file, "w") as f:
        f.write(report_content)

    print(f"Migration order analysis saved: {report_file}")
    print(f"Total sync cost for this order: {cumulative_sync_size:.2f} GB")

    # Save sync details CSV
    if sync_details_rows:
        sync_details_df = pd.DataFrame(sync_details_rows)
        sync_details_csv = os.path.join(outdir, f"community_sync_details_gamma_{resolution}.csv")
        sync_details_df.to_csv(sync_details_csv, index=False)
        print(f"Community sync details CSV saved: {sync_details_csv}")
        print(f"Total sync detail rows: {len(sync_details_rows)}")

    # Save stream-to-community ordering CSV
    if stream_ordering_rows:
        stream_ordering_df = pd.DataFrame(stream_ordering_rows)
        stream_ordering_csv = os.path.join(outdir, f"stream_community_ordering_gamma_{resolution}.csv")
        stream_ordering_df.to_csv(stream_ordering_csv, index=False)
        print(f"Stream community ordering CSV saved: {stream_ordering_csv}")
        print(f"Total streams mapped: {len(stream_ordering_rows)}")

    return report_file, cumulative_sync_size
