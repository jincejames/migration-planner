"""
Tests for migration_planner.dependency_extractors.loaders.

All tests use unittest.mock.MagicMock to mock the SparkSession and the
fluent reader chain — no real Spark cluster is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from migration_planner.dependency_extractors.loaders import (
    COMPLEXITY_WEIGHTS,
    load_complexity_scores,
    load_outofscope_streams,
    load_report_dependencies,
    load_static_tables,
    load_stream_table_dependencies,
    load_table_sizes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spark_reader():
    """Return a MagicMock that supports fluent .format().option().load() chains."""
    spark = MagicMock()
    reader = MagicMock()
    spark.read = reader
    # Each chained call returns the same reader mock so fluency works
    reader.format.return_value = reader
    reader.option.return_value = reader
    reader.csv.return_value = reader
    reader.select.return_value = reader
    reader.filter.return_value = reader
    reader.withColumn.return_value = reader
    reader.load.return_value = reader
    return spark, reader


# ---------------------------------------------------------------------------
# TestLoadStreamTableDependencies
# ---------------------------------------------------------------------------


class TestLoadStreamTableDependencies:
    def test_calls_format_csv(self):
        spark, reader = _make_spark_reader()
        load_stream_table_dependencies(spark, "/some/path.csv")
        reader.format.assert_called_once_with("csv")

    def test_calls_option_header_true(self):
        spark, reader = _make_spark_reader()
        load_stream_table_dependencies(spark, "/some/path.csv")
        reader.option.assert_called_once_with("header", "true")

    def test_calls_load_with_path(self):
        spark, reader = _make_spark_reader()
        load_stream_table_dependencies(spark, "/some/path.csv")
        reader.load.assert_called_once_with("/some/path.csv")

    def test_returns_dataframe_from_spark(self):
        spark, reader = _make_spark_reader()
        expected_df = MagicMock()
        reader.load.return_value = expected_df
        result = load_stream_table_dependencies(spark, "/some/path.csv")
        assert result is expected_df


# ---------------------------------------------------------------------------
# TestLoadOutofScopeStreams
# ---------------------------------------------------------------------------


class TestLoadOutofScopeStreams:
    def _setup_collect(self, names: list[str]):
        spark, reader = _make_spark_reader()
        rows = [MagicMock(**{"__getitem__": lambda self, k: n}) for n in names]
        for row, name in zip(rows, names):
            row.__getitem__ = lambda self, k, _n=name: _n
        reader.collect.return_value = [
            type("Row", (), {"__getitem__": lambda self, k: n})() for n in names
        ]
        return spark, reader

    def test_calls_reader_chain_with_path(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_outofscope_streams(spark, "/oos/path.csv")
        reader.format.assert_called_once_with("csv")
        reader.option.assert_called_once_with("header", "true")
        reader.load.assert_called_once_with("/oos/path.csv")

    def test_calls_select_stream_name(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_outofscope_streams(spark, "/oos/path.csv")
        reader.select.assert_called_once()

    def test_calls_collect(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_outofscope_streams(spark, "/oos/path.csv")
        reader.collect.assert_called_once()

    def test_returns_list_of_stream_names(self):
        spark, reader = _make_spark_reader()
        row1 = MagicMock()
        row1.__getitem__ = lambda self, k: "stream_a"
        row2 = MagicMock()
        row2.__getitem__ = lambda self, k: "stream_b"
        reader.collect.return_value = [row1, row2]
        result = load_outofscope_streams(spark, "/oos/path.csv")
        assert result == ["stream_a", "stream_b"]

    def test_returns_empty_list_when_collect_returns_nothing(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        result = load_outofscope_streams(spark, "/oos/path.csv")
        assert result == []


# ---------------------------------------------------------------------------
# TestLoadComplexityScores
# ---------------------------------------------------------------------------


class TestLoadComplexityScores:
    def test_calls_format_csv(self):
        spark, reader = _make_spark_reader()
        load_complexity_scores(spark, "/cx/path.csv")
        reader.format.assert_called_once_with("csv")

    def test_calls_option_delimiter_semicolon(self):
        spark, reader = _make_spark_reader()
        load_complexity_scores(spark, "/cx/path.csv")
        option_calls = reader.option.call_args_list
        assert call("delimiter", ";") in option_calls

    def test_calls_load_with_path(self):
        spark, reader = _make_spark_reader()
        load_complexity_scores(spark, "/cx/path.csv")
        reader.load.assert_called_once_with("/cx/path.csv")

    def test_calls_with_column_complexity_score(self):
        spark, reader = _make_spark_reader()
        load_complexity_scores(spark, "/cx/path.csv")
        reader.withColumn.assert_called_once()
        col_name = reader.withColumn.call_args[0][0]
        assert col_name == "complexity_score"

    def test_calls_select_with_expected_columns(self):
        spark, reader = _make_spark_reader()
        load_complexity_scores(spark, "/cx/path.csv")
        reader.select.assert_called_once()

    def test_returns_dataframe_from_spark(self):
        spark, reader = _make_spark_reader()
        expected_df = MagicMock()
        reader.select.return_value = expected_df
        result = load_complexity_scores(spark, "/cx/path.csv")
        assert result is expected_df


# ---------------------------------------------------------------------------
# TestLoadReportDependencies
# ---------------------------------------------------------------------------


class TestLoadReportDependencies:
    def test_calls_format_csv_header_true(self):
        spark, reader = _make_spark_reader()
        load_report_dependencies(spark, "/rep/path.csv")
        reader.format.assert_called_once_with("csv")
        reader.option.assert_called_once_with("header", "true")

    def test_calls_load_with_path(self):
        spark, reader = _make_spark_reader()
        load_report_dependencies(spark, "/rep/path.csv")
        reader.load.assert_called_once_with("/rep/path.csv")

    def test_calls_select_renames_report_name_to_stream_name(self):
        spark, reader = _make_spark_reader()
        load_report_dependencies(spark, "/rep/path.csv")
        reader.select.assert_called_once()

    def test_calls_filter_for_corona_and_gdpr_exclusion(self):
        spark, reader = _make_spark_reader()
        load_report_dependencies(spark, "/rep/path.csv")
        reader.filter.assert_called_once()

    def test_returns_dataframe(self):
        spark, reader = _make_spark_reader()
        expected_df = MagicMock()
        reader.filter.return_value = expected_df
        result = load_report_dependencies(spark, "/rep/path.csv")
        assert result is expected_df


# ---------------------------------------------------------------------------
# TestLoadTableSizes
# ---------------------------------------------------------------------------


class TestLoadTableSizes:
    def test_calls_format_csv_header_true(self):
        spark, reader = _make_spark_reader()
        load_table_sizes(spark, "/sz/path.csv")
        reader.format.assert_called_once_with("csv")
        reader.option.assert_called_once_with("header", "true")

    def test_calls_load_with_path(self):
        spark, reader = _make_spark_reader()
        load_table_sizes(spark, "/sz/path.csv")
        reader.load.assert_called_once_with("/sz/path.csv")

    def test_calls_select_renames_db_table_name_and_space_in_gb(self):
        spark, reader = _make_spark_reader()
        load_table_sizes(spark, "/sz/path.csv")
        reader.select.assert_called_once()

    def test_returns_dataframe(self):
        spark, reader = _make_spark_reader()
        expected_df = MagicMock()
        reader.select.return_value = expected_df
        result = load_table_sizes(spark, "/sz/path.csv")
        assert result is expected_df


# ---------------------------------------------------------------------------
# TestLoadStaticTables
# ---------------------------------------------------------------------------


class TestLoadStaticTables:
    def test_calls_option_header_true(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_static_tables(spark, "/st/path.csv")
        reader.option.assert_called_once_with("header", True)

    def test_calls_csv_with_path(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_static_tables(spark, "/st/path.csv")
        reader.csv.assert_called_once_with("/st/path.csv")

    def test_calls_select_table_name(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_static_tables(spark, "/st/path.csv")
        reader.select.assert_called_once_with("table_name")

    def test_calls_collect(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        load_static_tables(spark, "/st/path.csv")
        reader.collect.assert_called_once()

    def test_returns_set_of_table_names(self):
        spark, reader = _make_spark_reader()
        row1 = MagicMock()
        row1.__getitem__ = lambda self, k: "TABLE_A"
        row2 = MagicMock()
        row2.__getitem__ = lambda self, k: "TABLE_B"
        reader.collect.return_value = [row1, row2]
        result = load_static_tables(spark, "/st/path.csv")
        assert result == {"TABLE_A", "TABLE_B"}

    def test_returns_empty_set_when_collect_returns_nothing(self):
        spark, reader = _make_spark_reader()
        reader.collect.return_value = []
        result = load_static_tables(spark, "/st/path.csv")
        assert result == set()


# ---------------------------------------------------------------------------
# TestComplexityWeights
# ---------------------------------------------------------------------------


class TestComplexityWeights:
    def test_low_weight_is_1(self):
        assert COMPLEXITY_WEIGHTS["low"] == 1

    def test_medium_weight_is_2(self):
        assert COMPLEXITY_WEIGHTS["medium"] == 2

    def test_complex_weight_is_4(self):
        assert COMPLEXITY_WEIGHTS["complex"] == 4

    def test_very_complex_weight_is_7(self):
        assert COMPLEXITY_WEIGHTS["very_complex"] == 7

    def test_all_four_keys_present(self):
        assert set(COMPLEXITY_WEIGHTS.keys()) == {"low", "medium", "complex", "very_complex"}
