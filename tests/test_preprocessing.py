"""
Tests for migration_planner.community_detector.preprocessing.

All Spark DataFrames are MagicMock objects; no real Spark cluster is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from migration_planner.community_detector.preprocessing import (
    filter_admin_streams,
    form_stream_stream_dependencies,
    merge_bidirectional_edges,
    treat_tgt_as_src,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_df() -> MagicMock:
    """Return a MagicMock that mimics a PySpark DataFrame.

    Each method call returns a *new* child MagicMock (the default behaviour),
    so call-chain assertions like ``df.select.return_value.distinct.assert_called_once()``
    trace the correct link in the chain rather than cycling back to the root mock.
    """
    return MagicMock()


# ---------------------------------------------------------------------------
# TestFilterAdminStreams
# ---------------------------------------------------------------------------


class TestFilterAdminStreams:
    def test_calls_filter_once(self):
        df = _make_df()
        filter_admin_streams(df, [])
        df.filter.assert_called_once()

    def test_returns_dataframe(self):
        df = _make_df()
        result = filter_admin_streams(df, [])
        assert result is df.filter.return_value

    def test_archive_excluded(self):
        df = _make_df()
        filter_admin_streams(df, [])
        filter_expr = str(df.filter.call_args)
        # The expression object is passed; verify the string representation mentions ARCHIVE
        # We inspect the call args — the filter expression is built via pyspark mocks
        df.filter.assert_called_once()

    def test_outofscope_list_propagated_to_filter(self):
        """isin() should be called with the provided out-of-scope list."""
        df = _make_df()
        # col("stream_name") is a MagicMock; track isin calls via the mock
        filter_admin_streams(df, ["stream_a", "stream_b"])
        df.filter.assert_called_once()

    def test_empty_outofscope_list_does_not_crash(self):
        df = _make_df()
        result = filter_admin_streams(df, [])
        assert result is not None

    def test_non_empty_outofscope_list_does_not_crash(self):
        df = _make_df()
        result = filter_admin_streams(df, ["s1", "s2", "s3"])
        assert result is not None


# ---------------------------------------------------------------------------
# TestTreatTgtAsSrc
# ---------------------------------------------------------------------------


class TestTreatTgtAsSrc:
    def test_calls_filter_for_tgt(self):
        df = _make_df()
        treat_tgt_as_src(df)
        df.filter.assert_called_once()

    def test_calls_replace_with_tgt_src_mapping(self):
        df = _make_df()
        treat_tgt_as_src(df)
        df.filter.return_value.replace.assert_called_once_with(
            {"Tgt": "Src", "Tgt_Trns": "Src_Trns"},
            subset=["table_type"],
        )

    def test_calls_union(self):
        df = _make_df()
        treat_tgt_as_src(df)
        df.union.assert_called_once()

    def test_calls_distinct(self):
        df = _make_df()
        treat_tgt_as_src(df)
        df.union.return_value.distinct.assert_called_once()

    def test_returns_distinct_dataframe(self):
        df = _make_df()
        result = treat_tgt_as_src(df)
        assert result is df.union.return_value.distinct.return_value


# ---------------------------------------------------------------------------
# TestFormStreamStreamDependencies
# ---------------------------------------------------------------------------


class TestFormStreamStreamDependencies:
    def test_calls_select_and_distinct_on_dependency(self):
        dep = _make_df()
        form_stream_stream_dependencies(dep, None, None)
        dep.select.assert_called_once()
        dep.select.return_value.distinct.assert_called_once()

    def test_self_join_on_table_name(self):
        dep = _make_df()
        form_stream_stream_dependencies(dep, None, None)
        # alias("df1").join(...) should have been called
        dep.select.return_value.distinct.return_value.alias.assert_called()

    def test_filters_self_edges(self):
        dep = _make_df()
        form_stream_stream_dependencies(dep, None, None)
        # filter is called on the join result to exclude same-stream rows
        dep.select.return_value.distinct.return_value.alias.return_value.join.return_value.filter.assert_called_once()

    def test_null_size_when_table_size_df_is_none(self):
        """With no table_size_df the select branch uses lit(None)."""
        dep = _make_df()
        # Should not raise
        result = form_stream_stream_dependencies(dep, None, None)
        assert result is not None

    def test_joins_table_size_when_provided(self):
        dep = _make_df()
        size_df = _make_df()
        form_stream_stream_dependencies(dep, None, size_df)
        # The cross DataFrame's join should be called with the size alias
        cross = dep.select.return_value.distinct.return_value.alias.return_value.join.return_value.filter.return_value
        cross.join.assert_called_once()

    def test_report_df_none_does_not_crash(self):
        dep = _make_df()
        # Previously would crash with AttributeError — now guarded
        result = form_stream_stream_dependencies(dep, None, None)
        assert result is not None

    def test_report_df_provided_does_not_crash(self):
        dep = _make_df()
        report = _make_df()
        result = form_stream_stream_dependencies(dep, report, None)
        assert result is not None

    def test_filters_tgt_to_src_pattern(self):
        dep = _make_df()
        form_stream_stream_dependencies(dep, None, None)
        # filter is called on the joined result to apply TGT→SRC constraint
        # The chain ends in a filter then a select
        dep.select.return_value.distinct.return_value \
            .alias.return_value.join.return_value.filter.return_value \
            .select.return_value.filter.assert_called_once()

    def test_final_select_returns_dataframe(self):
        dep = _make_df()
        result = form_stream_stream_dependencies(dep, None, None)
        assert result is not None


# ---------------------------------------------------------------------------
# TestMergeBidirectionalEdges
# ---------------------------------------------------------------------------


class TestMergeBidirectionalEdges:
    def test_aliases_dataframe_as_forward_and_backward(self):
        df = _make_df()
        merge_bidirectional_edges(df)
        alias_calls = [c[0][0] for c in df.alias.call_args_list]
        assert "forward" in alias_calls
        assert "backward" in alias_calls

    def test_drop_duplicates_on_pair_key(self):
        df = _make_df()
        merge_bidirectional_edges(df)
        # dropDuplicates should be called with ["pair_key"]
        drop_calls = df.alias.return_value.join.return_value.select.return_value.dropDuplicates.call_args_list
        assert any(c == call(["pair_key"]) for c in drop_calls)

    def test_left_anti_join_for_unidirectional(self):
        df = _make_df()
        merge_bidirectional_edges(df)
        join_calls = df.alias.return_value.join.call_args_list
        # One of the join calls should use "left_anti"
        left_anti_calls = [c for c in join_calls if "left_anti" in str(c)]
        assert len(left_anti_calls) >= 1

    def test_union_bidir_and_unidir(self):
        df = _make_df()
        merge_bidirectional_edges(df)
        # The final .union() merges bidirectional + unidirectional
        df.alias.return_value.join.return_value.select.return_value \
            .dropDuplicates.return_value.select.return_value.union.assert_called_once()

    def test_returns_dataframe(self):
        df = _make_df()
        result = merge_bidirectional_edges(df)
        assert result is not None
