"""
Tests for migration_planner.community_detector.weights.

All Spark DataFrames are MagicMock objects; no real Spark cluster is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from migration_planner.community_detector.weights import (
    WEIGHT_METHOD_FACTOR,
    WEIGHT_METHOD_SCALED,
    _VALID_WEIGHT_METHODS,
    aggregate_edge_weights,
    calculate_table_weights,
    deduplicate_table_weights,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_df() -> MagicMock:
    """Return a *circular* MagicMock that mimics a PySpark DataFrame.

    Circular returns (every method returns ``df`` itself) are needed for the
    factor-method tests so that chained ``.withColumn`` calls all increment
    the *same* mock's call counter.
    """
    df = MagicMock()
    for method in (
        "filter", "select", "join", "union", "distinct",
        "withColumn", "replace", "alias", "dropDuplicates",
        "groupBy", "agg", "withColumnRenamed", "cast",
    ):
        getattr(df, method).return_value = df
    df.count.return_value = 0
    df.collect.return_value = []
    return df


def _make_df_for_scaled() -> MagicMock:
    """Return a MagicMock configured for scaled-weight tests.

    * ``withColumn`` and ``agg`` return ``df`` itself (circular) so the
      chain stays on a single mock.
    * ``collect()`` returns a one-element list with a mock stats row where
      ``mn = 0.0`` and ``mx = 100.0``.
    """
    df = MagicMock()
    for method in ("withColumn", "agg"):
        getattr(df, method).return_value = df

    mock_row = MagicMock()
    mock_row.__getitem__ = MagicMock(
        side_effect=lambda k: 0.0 if k == "mn" else 100.0
    )
    df.collect.return_value = [mock_row]
    return df


# ---------------------------------------------------------------------------
# TestWeightMethodConstants
# ---------------------------------------------------------------------------


class TestWeightMethodConstants:
    def test_factor_method_identifier(self):
        assert WEIGHT_METHOD_FACTOR == "factor"

    def test_scaled_method_identifier(self):
        assert WEIGHT_METHOD_SCALED == "scaled"

    def test_valid_methods_contains_both(self):
        assert WEIGHT_METHOD_FACTOR in _VALID_WEIGHT_METHODS
        assert WEIGHT_METHOD_SCALED in _VALID_WEIGHT_METHODS

    def test_constants_are_strings(self):
        assert isinstance(WEIGHT_METHOD_FACTOR, str)
        assert isinstance(WEIGHT_METHOD_SCALED, str)


# ---------------------------------------------------------------------------
# TestCalculateTableWeightsDispatcher
# ---------------------------------------------------------------------------


class TestCalculateTableWeightsDispatcher:
    def test_default_method_is_factor(self):
        """Calling with no method should use factor: 2 withColumn calls."""
        df = _make_df()
        calculate_table_weights(df)
        assert df.withColumn.call_count == 2

    def test_explicit_factor_method_also_works(self):
        df = _make_df()
        calculate_table_weights(df, method=WEIGHT_METHOD_FACTOR)
        assert df.withColumn.call_count == 2

    def test_unknown_method_raises_value_error(self):
        df = _make_df()
        with pytest.raises(ValueError, match="Unknown weight method"):
            calculate_table_weights(df, method="invalid_method")

    def test_scaled_method_dispatched_correctly(self):
        """Scaled method calls agg for min/max stats; factor method does not."""
        df = _make_df_for_scaled()
        calculate_table_weights(df, method=WEIGHT_METHOD_SCALED)
        df.agg.assert_called_once()

    def test_returns_dataframe(self):
        df = _make_df()
        result = calculate_table_weights(df)
        assert result is not None


# ---------------------------------------------------------------------------
# TestCalculateTableWeightsFactorMethod
# ---------------------------------------------------------------------------


class TestCalculateTableWeightsFactorMethod:
    def test_calls_with_column_twice(self):
        """Two withColumn calls: one for initial weight, one for the min=1 guard."""
        df = _make_df()
        calculate_table_weights(df, method=WEIGHT_METHOD_FACTOR)
        assert df.withColumn.call_count == 2

    def test_first_with_column_name_is_table_weight(self):
        df = _make_df()
        calculate_table_weights(df, method=WEIGHT_METHOD_FACTOR)
        first_call_col_name = df.withColumn.call_args_list[0][0][0]
        assert first_call_col_name == "table_weight"

    def test_second_with_column_name_is_table_weight(self):
        df = _make_df()
        calculate_table_weights(df, method=WEIGHT_METHOD_FACTOR)
        second_call_col_name = df.withColumn.call_args_list[1][0][0]
        assert second_call_col_name == "table_weight"

    def test_returns_result_of_second_with_column(self):
        df = _make_df()
        result = calculate_table_weights(df, method=WEIGHT_METHOD_FACTOR)
        assert result is df.withColumn.return_value.withColumn.return_value

    def test_does_not_call_agg(self):
        """Factor method computes weights locally — no Spark agg action needed."""
        df = _make_df()
        calculate_table_weights(df, method=WEIGHT_METHOD_FACTOR)
        df.agg.assert_not_called()


# ---------------------------------------------------------------------------
# TestCalculateTableWeightsScaledMethod
# ---------------------------------------------------------------------------


class TestCalculateTableWeightsScaledMethod:
    def test_calls_agg_for_min_max_stats(self):
        df = _make_df_for_scaled()
        calculate_table_weights(df, method=WEIGHT_METHOD_SCALED)
        df.agg.assert_called_once()

    def test_calls_collect_on_agg_result(self):
        df = _make_df_for_scaled()
        calculate_table_weights(df, method=WEIGHT_METHOD_SCALED)
        df.collect.assert_called_once()

    def test_calls_withcolumn_multiple_times(self):
        """Scaled method has at least 5 transformation steps."""
        df = _make_df_for_scaled()
        calculate_table_weights(df, method=WEIGHT_METHOD_SCALED)
        assert df.withColumn.call_count >= 5

    def test_adds_table_weight_column(self):
        df = _make_df_for_scaled()
        calculate_table_weights(df, method=WEIGHT_METHOD_SCALED)
        col_names = [c[0][0] for c in df.withColumn.call_args_list]
        assert "table_weight" in col_names

    def test_returns_dataframe(self):
        df = _make_df_for_scaled()
        result = calculate_table_weights(df, method=WEIGHT_METHOD_SCALED)
        assert result is not None

    def test_use_log_flag_does_not_crash(self):
        df = _make_df_for_scaled()
        result = calculate_table_weights(df, method=WEIGHT_METHOD_SCALED, use_log=True)
        assert result is not None

    def test_custom_scaled_range_accepted(self):
        df = _make_df_for_scaled()
        result = calculate_table_weights(
            df, method=WEIGHT_METHOD_SCALED, scaled_min=0.1, scaled_max=0.9
        )
        assert result is not None


# ---------------------------------------------------------------------------
# TestDeduplicateTableWeights
# ---------------------------------------------------------------------------


class TestDeduplicateTableWeights:
    def test_calls_drop_duplicates_with_correct_columns(self):
        df = _make_df()
        deduplicate_table_weights(df)
        df.dropDuplicates.assert_called_once_with(["from", "to", "table"])

    def test_renames_table_weight_to_weight(self):
        df = _make_df()
        deduplicate_table_weights(df)
        df.dropDuplicates.return_value.withColumnRenamed.assert_called_once_with(
            "table_weight", "weight"
        )

    def test_returns_result_of_rename(self):
        df = _make_df()
        result = deduplicate_table_weights(df)
        assert result is df.dropDuplicates.return_value.withColumnRenamed.return_value

    def test_returns_dataframe(self):
        df = _make_df()
        result = deduplicate_table_weights(df)
        assert result is not None


# ---------------------------------------------------------------------------
# TestAggregateEdgeWeights
# ---------------------------------------------------------------------------


class TestAggregateEdgeWeights:
    def test_calls_group_by_from_and_to(self):
        df = _make_df()
        aggregate_edge_weights(df)
        df.groupBy.assert_called_once_with("from", "to")

    def test_calls_agg_on_group_by_result(self):
        df = _make_df()
        aggregate_edge_weights(df)
        df.groupBy.return_value.agg.assert_called_once()

    def test_returns_result_of_agg(self):
        df = _make_df()
        expected = MagicMock()
        df.groupBy.return_value.agg.return_value = expected
        result = aggregate_edge_weights(df)
        assert result is expected

    def test_returns_dataframe(self):
        df = _make_df()
        result = aggregate_edge_weights(df)
        assert result is not None
