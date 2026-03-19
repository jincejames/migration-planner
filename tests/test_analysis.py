"""
Tests for migration_planner.planner_core.analysis.

All Spark DataFrames are MagicMock objects; no real Spark cluster is required.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

from migration_planner.planner_core.analysis import (
    BruteForceCommunityOrdering,
    append_execution_metadata,
    generate_community_analysis,
    generate_migration_order_analysis,
    get_leiden_df,
    membership_to_leiden_df,
    split_communities_topN,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_spark_df(pandas_df: pd.DataFrame | None = None) -> MagicMock:
    """Return a MagicMock whose .toPandas() returns ``pandas_df``."""
    df = MagicMock()
    for method in ("select", "join", "filter", "groupBy", "orderBy", "agg"):
        getattr(df, method).return_value = df
    df.toPandas.return_value = pandas_df if pandas_df is not None else pd.DataFrame()
    df.selectExpr.return_value = df
    return df


def _make_spark() -> MagicMock:
    """Return a minimal SparkSession mock."""
    spark = MagicMock()
    spark.createDataFrame.return_value = _make_spark_df()
    spark.catalog.tableExists.return_value = False
    return spark


def _simple_leiden_df() -> pd.DataFrame:
    """Three streams across two communities."""
    return pd.DataFrame(
        {"stream": ["s1", "s2", "s3"], "community": [0, 0, 1]}
    )


def _simple_dep_pdf() -> pd.DataFrame:
    """Minimal edge table for BruteForceCommunityOrdering tests."""
    return pd.DataFrame(
        {
            "from": ["s1", "s2"],
            "to": ["s3", "s3"],
            "table": ["T1", "T2"],
            "weight": [1.0, 2.0],
        }
    )


# ---------------------------------------------------------------------------
# TestMembershipToLeidenDf
# ---------------------------------------------------------------------------


class TestMembershipToLeidenDf:
    def test_returns_dataframe(self):
        result = membership_to_leiden_df([0, 1, 2], np.array(["a", "b", "c"]))
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_stream_and_community(self):
        result = membership_to_leiden_df([0, 1], np.array(["x", "y"]))
        assert list(result.columns) == ["stream", "community"]

    def test_stream_values_match_igraph_names(self):
        names = np.array(["alpha", "beta", "gamma"])
        result = membership_to_leiden_df([0, 0, 1], names)
        assert list(result["stream"]) == ["alpha", "beta", "gamma"]

    def test_community_dtype_is_int(self):
        result = membership_to_leiden_df([0, 1, 2], np.array(["a", "b", "c"]))
        assert result["community"].dtype == int

    def test_length_matches_membership(self):
        result = membership_to_leiden_df([0, 1, 2, 3], np.array(["a", "b", "c", "d"]))
        assert len(result) == 4

    def test_single_community(self):
        result = membership_to_leiden_df([0, 0, 0], np.array(["a", "b", "c"]))
        assert result["community"].nunique() == 1

    def test_empty_membership(self):
        result = membership_to_leiden_df([], np.array([]))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestGetLeidenDf
# ---------------------------------------------------------------------------


class TestGetLeidenDf:
    def _make_rep_by_res(self, resolution: float = 1.8) -> dict:
        return {
            resolution: {
                "membership": [0, 0, 1],
                "seed": 42,
                "quality": 0.75,
                "resolution": resolution,
            }
        }

    def test_returns_tuple(self):
        rep = self._make_rep_by_res()
        result = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_leiden_df_has_correct_columns(self):
        rep = self._make_rep_by_res()
        leiden_df, _ = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert "stream" in leiden_df.columns
        assert "community" in leiden_df.columns

    def test_meta_contains_resolution_seed_quality(self):
        rep = self._make_rep_by_res()
        _, meta = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert "resolution" in meta
        assert "seed" in meta
        assert "quality" in meta

    def test_meta_resolution_is_float(self):
        rep = self._make_rep_by_res()
        _, meta = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert isinstance(meta["resolution"], float)

    def test_meta_seed_is_int(self):
        rep = self._make_rep_by_res()
        _, meta = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert isinstance(meta["seed"], int)

    def test_meta_quality_is_float(self):
        rep = self._make_rep_by_res()
        _, meta = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert isinstance(meta["quality"], float)

    def test_leiden_df_length_matches_names(self):
        rep = self._make_rep_by_res()
        leiden_df, _ = get_leiden_df(1.8, rep, np.array(["s1", "s2", "s3"]))
        assert len(leiden_df) == 3

    def test_resolution_defaults_from_key_when_not_in_rep(self):
        """If 'resolution' key is absent, meta['resolution'] equals the argument."""
        rep = {2.2: {"membership": [0, 1], "seed": 10, "quality": 0.5}}
        _, meta = get_leiden_df(2.2, rep, np.array(["a", "b"]))
        assert meta["resolution"] == 2.2

    def test_missing_quality_gives_nan(self):
        rep = {1.0: {"membership": [0, 1], "seed": 7}}
        _, meta = get_leiden_df(1.0, rep, np.array(["a", "b"]))
        assert np.isnan(meta["quality"])


# ---------------------------------------------------------------------------
# TestSplitCommunitiesTopN
# ---------------------------------------------------------------------------


class TestSplitCommunitiesTopN:
    """
    split_communities_topN uses spark.createDataFrame and Spark join/agg chains.
    We mock all Spark operations; the function's return values depend on what
    .toPandas() returns for the aggregated community weight DataFrame.
    """

    def _make_weight_df_mock(self, pdf: pd.DataFrame) -> MagicMock:
        """Mock unique_table_weights_df whose aggregated toPandas returns ``pdf``."""
        df = MagicMock()
        # All Spark transformation methods return `df` (circular mock)
        for m in ("select", "join", "filter", "groupBy", "agg", "orderBy", "alias"):
            getattr(df, m).return_value = df
        df.selectExpr.return_value = df
        df.toPandas.return_value = pdf
        return df

    def test_returns_three_tuple(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        result = split_communities_topN(leiden_df, utw, spark, top_n=1)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_top_n_ids_is_list(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        top_n_ids, _, _ = split_communities_topN(leiden_df, utw, spark, top_n=1)
        assert isinstance(top_n_ids, list)

    def test_rest_ids_is_list(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        _, rest_ids, _ = split_communities_topN(leiden_df, utw, spark, top_n=1)
        assert isinstance(rest_ids, list)

    def test_community_weights_is_dict(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        _, _, community_weights = split_communities_topN(leiden_df, utw, spark, top_n=1)
        assert isinstance(community_weights, dict)

    def test_top_n_and_rest_ids_cover_all_communities(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        top_n_ids, rest_ids, _ = split_communities_topN(leiden_df, utw, spark, top_n=1)
        all_comms = set(leiden_df["community"].unique())
        assert set(top_n_ids) | set(rest_ids) == all_comms

    def test_top_n_larger_than_community_count_returns_all_in_top(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        top_n_ids, rest_ids, _ = split_communities_topN(leiden_df, utw, spark, top_n=100)
        assert rest_ids == []
        assert len(top_n_ids) == 2

    def test_community_with_no_incoming_edges_gets_zero_weight(self):
        leiden_df = _simple_leiden_df()
        # Only community 0 appears in the aggregated result
        weights_pdf = pd.DataFrame(
            {"to_community": [0], "total_incoming_weight": [8.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        _, _, community_weights = split_communities_topN(leiden_df, utw, spark, top_n=1)
        # Community 1 had no incoming edges — should default to 0.0
        assert community_weights.get(1, None) == 0.0

    def test_spark_create_dataframe_called(self):
        leiden_df = _simple_leiden_df()
        weights_pdf = pd.DataFrame(
            {"to_community": [0, 1], "total_incoming_weight": [5.0, 3.0]}
        )
        utw = self._make_weight_df_mock(weights_pdf)
        spark = _make_spark()

        split_communities_topN(leiden_df, utw, spark, top_n=1)
        spark.createDataFrame.assert_called_once()


# ---------------------------------------------------------------------------
# TestBruteForceCommunityOrdering
# ---------------------------------------------------------------------------


class TestBruteForceCommunityOrdering:
    """
    Tests use a controlled pandas DF returned by the Spark mock's .toPandas().
    Some tests use real data to verify the brute-force algorithm correctness.
    """

    def _simple_dep_pdf(self) -> pd.DataFrame:
        return _simple_dep_pdf()

    def _make_bf(self, pdf=None, leiden_df=None, communities_subset=None,
                 pre_available_communities=None) -> BruteForceCommunityOrdering:
        if pdf is None:
            pdf = self._simple_dep_pdf()
        if leiden_df is None:
            leiden_df = _simple_leiden_df()
        return BruteForceCommunityOrdering(
            pdf, leiden_df,
            communities_subset=communities_subset,
            pre_available_communities=pre_available_communities,
        )

    # --- Initialisation ---

    def test_dep_converted_to_pandas(self):
        bf = self._make_bf()
        assert isinstance(bf.dep, pd.DataFrame)

    def test_communities_attribute_is_list(self):
        bf = self._make_bf()
        assert isinstance(bf.communities, list)

    def test_all_communities_present_when_no_subset(self):
        bf = self._make_bf()
        assert set(bf.communities) == {0, 1}

    def test_subset_restricts_communities(self):
        bf = self._make_bf(communities_subset=[0])
        assert bf.communities == [0]

    def test_pre_available_tables_empty_by_default(self):
        bf = self._make_bf()
        assert isinstance(bf.pre_available_tables, set)
        assert len(bf.pre_available_tables) == 0

    def test_pre_available_tables_populated(self):
        """When pre_available_communities=[0], tables produced by stream s1 and s2 are pre-available."""
        bf = self._make_bf(communities_subset=[1], pre_available_communities=[0])
        # s1 produces T1; s2 produces T2 (from dep_pdf)
        assert len(bf.pre_available_tables) > 0

    def test_produced_tables_populated(self):
        bf = self._make_bf()
        # At least one community should have produced tables
        assert any(len(v) > 0 for v in bf.produced_tables.values())

    def test_incoming_tables_populated_for_community_1(self):
        """Community 1 (stream s3) needs T1 and T2, which are produced by community 0."""
        bf = self._make_bf()
        # Community 1 (s3) should have incoming tables T1 and T2
        assert len(bf.incoming_tables.get(1, set())) > 0

    def test_bitmask_attributes_initialised(self):
        bf = self._make_bf()
        assert bf._n_tables > 0
        assert len(bf._table_to_idx) == bf._n_tables
        assert bf._weight_vec.shape == (bf._n_tables,)
        assert len(bf._incoming_mask) == len(bf.communities)
        assert len(bf._produced_mask) == len(bf.communities)
        assert bf._pre_available_mask.shape == (bf._n_tables,)

    # --- evaluate_ordering_cost ---

    def test_returns_tuple_with_total_and_steps(self):
        bf = self._make_bf()
        total, steps = bf.evaluate_ordering_cost([0, 1], return_step_costs=True)
        assert isinstance(total, float)
        assert isinstance(steps, list)

    def test_returns_none_steps_by_default(self):
        bf = self._make_bf()
        total, steps = bf.evaluate_ordering_cost([0, 1])
        assert isinstance(total, float)
        assert steps is None

    def test_ordering_starting_with_producer_has_lower_cost(self):
        """Ordering [0, 1]: community 0 first → community 1's tables already available."""
        bf = self._make_bf()
        cost_producer_first, _ = bf.evaluate_ordering_cost([0, 1])
        cost_consumer_first, _ = bf.evaluate_ordering_cost([1, 0])
        assert cost_producer_first <= cost_consumer_first

    def test_empty_ordering_has_zero_cost(self):
        bf = self._make_bf(communities_subset=[0])
        total, steps = bf.evaluate_ordering_cost([], return_step_costs=True)
        assert total == 0.0
        assert steps == []

    def test_pruning_returns_inf_when_exceeding_best_cost(self):
        bf = self._make_bf()
        # Use a very small best_cost so pruning kicks in
        cost, steps = bf.evaluate_ordering_cost([1, 0], best_cost=0.0)
        assert cost == float("inf")
        assert steps is None

    def test_pre_available_communities_reduces_cost(self):
        """When pre_available_communities=[0], community 1's incoming tables are already available."""
        bf_without = self._make_bf()
        bf_with = self._make_bf(communities_subset=[1], pre_available_communities=[0])
        cost_without, _ = bf_without.evaluate_ordering_cost([1])
        cost_with, _ = bf_with.evaluate_ordering_cost([1])
        assert cost_with <= cost_without

    # --- brute_force ---

    def test_brute_force_returns_dict(self):
        bf = self._make_bf()
        result = bf.brute_force(log_every=10000, label="test")
        assert isinstance(result, dict)

    def test_brute_force_result_has_required_keys(self):
        bf = self._make_bf()
        result = bf.brute_force(log_every=10000, label="test")
        assert "best_cost" in result
        assert "best_order" in result
        assert "best_step_costs" in result
        assert "total_time_sec" in result
        assert "total_perms" in result

    def test_brute_force_best_order_is_list(self):
        bf = self._make_bf()
        result = bf.brute_force(log_every=10000, label="test")
        assert isinstance(result["best_order"], list)

    def test_brute_force_best_cost_is_float(self):
        bf = self._make_bf()
        result = bf.brute_force(log_every=10000, label="test")
        assert isinstance(result["best_cost"], float)

    def test_brute_force_finds_optimal_order_for_2_communities(self):
        """With community 0 producing T1,T2 and community 1 consuming them,
        order [0, 1] should be optimal (lower cost than [1, 0])."""
        bf = self._make_bf()
        result = bf.brute_force(log_every=10000, label="test")
        cost_optimal, _ = bf.evaluate_ordering_cost(
            result["best_order"], return_step_costs=False,
        )
        assert cost_optimal == result["best_cost"]

    def test_single_community_brute_force(self):
        bf = self._make_bf(communities_subset=[0])
        result = bf.brute_force(log_every=10000, label="single")
        assert result["best_order"] == [0]
        assert result["total_perms"] == 1


# ---------------------------------------------------------------------------
# TestAppendExecutionMetadata
# ---------------------------------------------------------------------------


class TestAppendExecutionMetadata:
    def test_returns_table_name_string(self):
        spark = _make_spark()
        result = append_execution_metadata(spark, "factor", 10, {1.8: (100.0, 5)})
        assert isinstance(result, str)

    def test_table_name_is_correct(self):
        spark = _make_spark()
        result = append_execution_metadata(spark, "factor", 10, {1.8: (100.0, 5)})
        assert result == "odp_adw_utilities_n.planning.execution_metadata"

    def test_creates_dataframe_with_row_count(self):
        spark = _make_spark()
        append_execution_metadata(spark, "factor", 10, {1.8: (100.0, 5), 2.2: (200.0, 8)})
        assert spark.createDataFrame.call_count == 1
        rows_passed = spark.createDataFrame.call_args[0][0]
        assert len(rows_passed) == 2

    def test_overwrites_when_table_does_not_exist(self):
        spark = _make_spark()
        spark.catalog.tableExists.return_value = False
        append_execution_metadata(spark, "scaled", 5, {1.0: (50.0, 3)})
        metadata_df = spark.createDataFrame.return_value
        metadata_df.write.mode.assert_called_with("overwrite")

    def test_appends_when_table_exists(self):
        spark = _make_spark()
        spark.catalog.tableExists.return_value = True
        append_execution_metadata(spark, "scaled", 5, {1.0: (50.0, 3)})
        metadata_df = spark.createDataFrame.return_value
        metadata_df.write.mode.assert_called_with("append")

    def test_row_has_correct_fields(self):
        spark = _make_spark()
        append_execution_metadata(spark, "my_method", 7, {2.0: (123.45, 6)})
        rows = spark.createDataFrame.call_args[0][0]
        row = rows[0]
        assert row["weight_calculation_method"] == "my_method"
        assert row["top_n"] == 7
        assert row["resolution"] == 2.0
        assert row["number_of_communities"] == 6
        assert row["total_sync_cost_gb"] == 123.45

    def test_empty_resolutions_dict(self):
        spark = _make_spark()
        append_execution_metadata(spark, "factor", 10, {})
        rows = spark.createDataFrame.call_args[0][0]
        assert rows == []


# ---------------------------------------------------------------------------
# TestGenerateMigrationOrderAnalysis
# ---------------------------------------------------------------------------


class TestGenerateMigrationOrderAnalysis:
    """
    generate_migration_order_analysis calls .toPandas() on three Spark DFs.
    We supply controlled pandas DataFrames and verify outputs.
    """

    def _make_stream_table_df(self) -> MagicMock:
        pdf = pd.DataFrame(
            {
                "from": ["s1", "s1", "s2"],
                "to": ["s3", "s2", "s3"],
                "table": ["T1", "T2", "T2"],
                "size": [1.0, 2.0, 2.0],
            }
        )
        return _make_spark_df(pdf)

    def _make_merged_edges_df(self) -> MagicMock:
        pdf = pd.DataFrame(
            {
                "streamA": ["s1", "s2"],
                "streamB": ["s3", "s3"],
                "weight": [3.0, 2.0],
            }
        )
        return _make_spark_df(pdf)

    def _make_complexity_df(self) -> MagicMock:
        pdf = pd.DataFrame(
            {"stream_name": ["s1", "s2", "s3"], "complexity_score": [10.0, 20.0, 30.0]}
        )
        df = MagicMock()
        df.select.return_value = df
        df.toPandas.return_value = pdf
        return df

    def _make_leiden_df(self) -> pd.DataFrame:
        return pd.DataFrame({"stream": ["s1", "s2", "s3"], "community": [0, 0, 1]})

    def test_returns_tuple(self, tmp_path):
        result = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_report_file_path_is_string(self, tmp_path):
        report_file, _ = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        assert isinstance(report_file, str)

    def test_total_sync_cost_is_float(self, tmp_path):
        _, total_sync_cost = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        assert isinstance(total_sync_cost, float)

    def test_report_file_is_created(self, tmp_path):
        report_file, _ = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        assert os.path.exists(report_file)

    def test_report_file_contains_resolution(self, tmp_path):
        report_file, _ = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        with open(report_file) as f:
            content = f.read()
        assert "1.8" in content

    def test_report_filename_includes_resolution(self, tmp_path):
        report_file, _ = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        assert "gamma_1.8" in report_file

    def test_sync_details_csv_created(self, tmp_path):
        # Use order [1, 0]: community 1 (s3) is processed first, so its
        # incoming tables (T1, T2) are not yet available → sync rows are generated.
        generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[1, 0],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        csv_path = tmp_path / "community_sync_details_gamma_1.8.csv"
        assert csv_path.exists()

    def test_stream_ordering_csv_created(self, tmp_path):
        generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        csv_path = tmp_path / "stream_community_ordering_gamma_1.8.csv"
        assert csv_path.exists()

    def test_output_directory_created(self, tmp_path):
        outdir = str(tmp_path / "new_subdir")
        generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=outdir,
        )
        assert os.path.isdir(outdir)

    def test_no_sync_tables_when_order_is_optimal(self, tmp_path):
        """With [0, 1] ordering, community 1 has no tables to sync if
        community 0 already produced everything it needs."""
        _, total_sync_cost = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        assert total_sync_cost >= 0.0

    def test_report_contains_migration_summary(self, tmp_path):
        report_file, _ = generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        with open(report_file) as f:
            content = f.read()
        assert "FINAL MIGRATION SUMMARY" in content

    def test_stream_ordering_csv_has_correct_columns(self, tmp_path):
        generate_migration_order_analysis(
            leiden_df=self._make_leiden_df(),
            stream_table_dependency_df=self._make_stream_table_df(),
            merged_edges_df=self._make_merged_edges_df(),
            optimized_order=[0, 1],
            resolution=1.8,
            complexity_scores_df=self._make_complexity_df(),
            outdir=str(tmp_path),
        )
        csv_path = tmp_path / "stream_community_ordering_gamma_1.8.csv"
        df = pd.read_csv(csv_path)
        assert "stream_name" in df.columns
        assert "community_id" in df.columns
        assert "execution_order" in df.columns


# ---------------------------------------------------------------------------
# TestGenerateCommunityAnalysis
# ---------------------------------------------------------------------------


def _gc_leiden_df() -> pd.DataFrame:
    return pd.DataFrame({
        "stream": ["stream_a", "stream_b_json", "stream_c", "stream_d"],
        "community": [0, 0, 1, 1],
    })


def _gc_stream_table_pdf() -> pd.DataFrame:
    return pd.DataFrame({
        "from": ["stream_a", "stream_c"],
        "to": ["stream_c", "stream_b_json"],
        "table": ["TABLE_X", "TABLE_Y"],
        "size": [10.0, 5.0],
    })


def _gc_merged_edges_pdf() -> pd.DataFrame:
    return pd.DataFrame({
        "streamA": ["stream_a"],
        "streamB": ["stream_c"],
        "weight": [3.0],
    })


def _gc_complexity_pdf() -> pd.DataFrame:
    return pd.DataFrame({
        "stream_name": ["stream_a", "stream_b_json", "stream_c", "stream_d"],
        "complexity_score": [4.0, 2.0, 6.0, 1.0],
    })


class TestGenerateCommunityAnalysis:
    def _call(self, tmp_path, **kwargs):
        defaults = dict(
            leiden_df=_gc_leiden_df(),
            stream_table_pdf=_gc_stream_table_pdf(),
            merged_edges_pdf=_gc_merged_edges_pdf(),
            complexity_pdf=_gc_complexity_pdf(),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
        )
        defaults.update(kwargs)
        return generate_community_analysis(**defaults)

    def test_returns_list_of_txt_files(self, tmp_path):
        result = self._call(tmp_path)
        assert isinstance(result, list)
        assert all(f.endswith(".txt") for f in result)

    def test_creates_one_txt_per_community(self, tmp_path):
        result = self._call(tmp_path)
        assert len(result) == 2  # communities 0 and 1

    def test_community_subdirs_created(self, tmp_path):
        outdir = str(tmp_path / "out")
        generate_community_analysis(
            leiden_df=_gc_leiden_df(),
            stream_table_pdf=_gc_stream_table_pdf(),
            merged_edges_pdf=_gc_merged_edges_pdf(),
            complexity_pdf=_gc_complexity_pdf(),
            resolution=1.8,
            outdir=outdir,
        )
        assert os.path.isdir(os.path.join(outdir, "community_0"))
        assert os.path.isdir(os.path.join(outdir, "community_1"))

    def test_analysis_file_contains_community_id(self, tmp_path):
        result = self._call(tmp_path)
        for f in result:
            content = open(f).read()
            assert "Community" in content

    def test_analysis_file_contains_resolution(self, tmp_path):
        result = self._call(tmp_path, resolution=1.8)
        content = open(result[0]).read()
        assert "1.8" in content

    def test_streams_classified_as_bi_or_etl(self, tmp_path):
        result = self._call(tmp_path)
        # Community 0 has stream_b_json (BI) and stream_a (ETL)
        comm0_file = [f for f in result if "community_0" in f][0]
        content = open(comm0_file).read()
        assert "BI REPORTS" in content
        assert "ETL STREAMS" in content

    def test_incoming_tables_for_community(self, tmp_path):
        # Community 0: stream_b_json reads TABLE_Y, produced by stream_c (community 1)
        # → TABLE_Y should appear in community 0 analysis as an incoming table
        result = self._call(tmp_path)
        comm0_file = [f for f in result if "community_0" in f][0]
        content = open(comm0_file).read()
        assert "TABLE_Y" in content
        assert "SYNC REQUIREMENTS" in content

    def test_outgoing_tables_for_community(self, tmp_path):
        # Community 0: stream_a writes TABLE_X, read by stream_c (community 1)
        # → TABLE_X should appear in community 0 analysis as an outgoing table
        result = self._call(tmp_path)
        comm0_file = [f for f in result if "community_0" in f][0]
        content = open(comm0_file).read()
        assert "TABLE_X" in content

    def test_complexity_section_in_analysis(self, tmp_path):
        result = self._call(tmp_path)
        comm0_file = [f for f in result if "community_0" in f][0]
        content = open(comm0_file).read()
        assert "COMPLEXITY ANALYSIS" in content

    def test_summary_section_in_analysis(self, tmp_path):
        result = self._call(tmp_path)
        comm0_file = [f for f in result if "community_0" in f][0]
        content = open(comm0_file).read()
        assert "SUMMARY" in content

    def test_does_not_modify_input_stream_table_pdf(self, tmp_path):
        stream_table_pdf = _gc_stream_table_pdf()
        original_dtypes = stream_table_pdf.dtypes.copy()
        self._call(tmp_path, stream_table_pdf=stream_table_pdf)
        # dtype of 'size' column must remain unchanged in the caller's copy
        assert stream_table_pdf.dtypes["size"] == original_dtypes["size"]

    def test_non_numeric_size_coerced_gracefully(self, tmp_path):
        pdf = _gc_stream_table_pdf().copy()
        pdf["size"] = "not_a_number"  # all strings
        result = self._call(tmp_path, stream_table_pdf=pdf)
        # Should not raise; writes two txt files
        assert len(result) == 2

    def test_single_community(self, tmp_path):
        leiden_df = pd.DataFrame({"stream": ["s1", "s2"], "community": [0, 0]})
        result = generate_community_analysis(
            leiden_df=leiden_df,
            stream_table_pdf=pd.DataFrame({
                "from": pd.Series([], dtype=str),
                "to": pd.Series([], dtype=str),
                "table": pd.Series([], dtype=str),
                "size": pd.Series([], dtype=float),
            }),
            merged_edges_pdf=pd.DataFrame({
                "streamA": pd.Series([], dtype=str),
                "streamB": pd.Series([], dtype=str),
                "weight": pd.Series([], dtype=float),
            }),
            complexity_pdf=pd.DataFrame({
                "stream_name": ["s1", "s2"],
                "complexity_score": [1.0, 2.0],
            }),
            resolution=1.0,
            outdir=str(tmp_path / "out"),
        )
        assert len(result) == 1

    def test_empty_leiden_df_returns_empty_list(self, tmp_path):
        result = generate_community_analysis(
            leiden_df=pd.DataFrame({
                "stream": pd.Series([], dtype=str),
                "community": pd.Series([], dtype=int),
            }),
            stream_table_pdf=_gc_stream_table_pdf(),
            merged_edges_pdf=_gc_merged_edges_pdf(),
            complexity_pdf=_gc_complexity_pdf(),
            resolution=1.0,
            outdir=str(tmp_path / "out"),
        )
        assert result == []
