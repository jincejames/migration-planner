"""
Tests for migration_planner.visualization.community_plots.

All functions that write files use a tmp_path fixture so nothing is left on
disk after the test run.  Matplotlib is forced to use the non-interactive Agg
backend so the tests run without a display server.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from migration_planner.visualization.community_plots import (
    edge_style,
    plot_communities_with_analysis_safe,
    plot_leiden_resolutions,
    precompute_edge_style,
    precompute_layout,
    select_resolutions,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _simple_graph() -> nx.Graph:
    """Return a small weighted NetworkX graph for use in tests."""
    G = nx.Graph()
    G.add_edge("s1", "s2", weight=1.0)
    G.add_edge("s2", "s3", weight=2.0)
    G.add_edge("s3", "s4", weight=3.0)
    return G


def _sample_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "resolution": [0.4, 0.8, 1.2, 1.8],
        "stability_ari": [1.0, 1.0, 0.95, 0.93],
        "largest_comm_share_avg": [0.98, 0.40, 0.18, 0.15],
        "n_communities_avg": [3.0, 5.0, 10.0, 12.0],
    })


def _mock_spark_df(pandas_df: pd.DataFrame) -> MagicMock:
    """Return a mock Spark DataFrame whose toPandas() returns *pandas_df*."""
    mock_df = MagicMock()
    mock_df.toPandas.return_value = pandas_df
    return mock_df


# ---------------------------------------------------------------------------
# TestSelectResolutions
# ---------------------------------------------------------------------------


class TestSelectResolutions:
    def test_filters_by_ari(self):
        summary = _sample_summary()
        _, resolutions = select_resolutions(summary, ari_target=0.95, max_largest_comm_share=1.0)
        for res in resolutions:
            row = summary[summary["resolution"] == res].iloc[0]
            assert row["stability_ari"] >= 0.95

    def test_filters_by_largest_comm_share(self):
        summary = _sample_summary()
        _, resolutions = select_resolutions(summary, ari_target=0.0, max_largest_comm_share=0.20)
        for res in resolutions:
            row = summary[summary["resolution"] == res].iloc[0]
            assert row["largest_comm_share_avg"] <= 0.20

    def test_returns_dataframe_and_list(self):
        summary = _sample_summary()
        selected, resolutions = select_resolutions(summary)
        assert isinstance(selected, pd.DataFrame)
        assert isinstance(resolutions, list)

    def test_empty_result_when_no_rows_match(self):
        summary = _sample_summary()
        selected, resolutions = select_resolutions(summary, ari_target=1.1)
        assert len(selected) == 0
        assert resolutions == []

    def test_result_sorted_by_resolution(self):
        summary = _sample_summary().sample(frac=1, random_state=0)  # shuffle
        selected, resolutions = select_resolutions(summary, ari_target=0.0, max_largest_comm_share=1.0)
        assert resolutions == sorted(resolutions)

    def test_custom_sort_by(self):
        summary = _sample_summary()
        selected, _ = select_resolutions(
            summary, ari_target=0.0, max_largest_comm_share=1.0, sort_by="n_communities_avg"
        )
        vals = selected["n_communities_avg"].tolist()
        assert vals == sorted(vals)

    def test_returned_resolutions_match_dataframe(self):
        summary = _sample_summary()
        selected, resolutions = select_resolutions(summary, ari_target=0.0, max_largest_comm_share=1.0)
        assert selected["resolution"].tolist() == resolutions


# ---------------------------------------------------------------------------
# TestPrecomputeLayout
# ---------------------------------------------------------------------------


class TestPrecomputeLayout:
    def test_returns_dict(self):
        G = _simple_graph()
        pos = precompute_layout(G)
        assert isinstance(pos, dict)

    def test_every_node_has_position(self):
        G = _simple_graph()
        pos = precompute_layout(G)
        for node in G.nodes():
            assert node in pos

    def test_positions_are_2d(self):
        G = _simple_graph()
        pos = precompute_layout(G)
        for _, coords in pos.items():
            assert len(coords) == 2

    def test_deterministic_with_same_seed(self):
        G = _simple_graph()
        pos1 = precompute_layout(G, seed=7)
        pos2 = precompute_layout(G, seed=7)
        for node in G.nodes():
            np.testing.assert_array_almost_equal(pos1[node], pos2[node])

    def test_empty_graph_returns_empty_dict(self):
        pos = precompute_layout(nx.Graph())
        assert pos == {}


# ---------------------------------------------------------------------------
# TestPrecomputeEdgeStyle
# ---------------------------------------------------------------------------


class TestPrecomputeEdgeStyle:
    def test_returns_two_arrays(self):
        G = _simple_graph()
        widths, alphas = precompute_edge_style(G)
        assert isinstance(widths, np.ndarray)
        assert isinstance(alphas, np.ndarray)

    def test_length_matches_edge_count(self):
        G = _simple_graph()
        widths, alphas = precompute_edge_style(G)
        assert len(widths) == G.number_of_edges()
        assert len(alphas) == G.number_of_edges()

    def test_empty_graph_returns_empty_arrays(self):
        widths, alphas = precompute_edge_style(nx.Graph())
        assert widths.size == 0
        assert alphas.size == 0

    def test_widths_within_expected_range(self):
        G = _simple_graph()
        widths, _ = precompute_edge_style(G, width_min=0.2, width_scale=3.0)
        assert float(widths.min()) >= 0.2
        assert float(widths.max()) <= 0.2 + 3.0

    def test_alphas_within_expected_range(self):
        G = _simple_graph()
        _, alphas = precompute_edge_style(G, alpha_min=0.05, alpha_scale=0.45)
        assert float(alphas.min()) >= 0.05
        assert float(alphas.max()) <= 0.05 + 0.45

    def test_uniform_weights_all_same_width(self):
        G = nx.Graph()
        G.add_edge("a", "b", weight=5.0)
        G.add_edge("b", "c", weight=5.0)
        widths, _ = precompute_edge_style(G, width_min=0.5, width_scale=2.0)
        # All weights equal → normalised weight = 0 → all widths = width_min
        np.testing.assert_array_almost_equal(widths, [0.5, 0.5])

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge("x", "y", weight=10.0)
        widths, alphas = precompute_edge_style(G)
        assert len(widths) == 1
        assert len(alphas) == 1


# ---------------------------------------------------------------------------
# TestEdgeStyle
# ---------------------------------------------------------------------------


class TestEdgeStyle:
    def test_returns_two_sequences(self):
        G = _simple_graph()
        widths, alphas = edge_style(G)
        assert len(widths) == G.number_of_edges()
        assert len(alphas) == G.number_of_edges()

    def test_empty_graph_returns_empty_lists(self):
        widths, alphas = edge_style(nx.Graph())
        assert widths == []
        assert alphas == []

    def test_widths_in_min_max_range(self):
        G = _simple_graph()
        widths, _ = edge_style(G, min_w=0.5, max_w=4.0)
        for w in widths:
            assert 0.5 <= float(w) <= 4.0

    def test_alphas_between_0_and_1(self):
        G = _simple_graph()
        _, alphas = edge_style(G)
        for a in alphas:
            assert 0.0 <= float(a) <= 1.0

    def test_heavier_edge_wider_than_lighter(self):
        G = nx.Graph()
        G.add_edge("a", "b", weight=1.0)
        G.add_edge("b", "c", weight=10.0)
        edges = list(G.edges())
        widths, _ = edge_style(G)
        idx_light = edges.index(("a", "b")) if ("a", "b") in edges else edges.index(("b", "a"))
        idx_heavy = edges.index(("b", "c")) if ("b", "c") in edges else edges.index(("c", "b"))
        assert float(widths[idx_heavy]) > float(widths[idx_light])

    def test_single_edge_returns_min_values(self):
        G = nx.Graph()
        G.add_edge("x", "y", weight=5.0)
        widths, alphas = edge_style(G, min_w=0.3, max_w=3.0)
        assert len(widths) == 1
        # With a single edge, w_min == w_max, so normalised weight = 0
        assert float(widths[0]) == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# TestPlotLeidenResolutions
# ---------------------------------------------------------------------------


class TestPlotLeidenResolutions:
    def _make_rep_by_res(self) -> dict:
        return {
            1.0: {
                "membership": [0, 0, 1, 1],
                "quality": 0.5,
                "seed": 42,
                "resolution": 1.0,
            }
        }

    def _make_membership_fn(self):
        def fn(membership):
            return pd.DataFrame({
                "stream": ["s1", "s2", "s3", "s4"],
                "community": list(membership),
            })
        return fn

    def test_saves_file_to_outdir(self, tmp_path):
        G = _simple_graph()
        G.add_node("s4")
        pos = {n: (i, 0) for i, n in enumerate(G.nodes())}
        widths = np.array([1.0, 1.0, 1.0])
        alphas = np.array([0.5, 0.5, 0.5])
        g_igraph = MagicMock()
        g_igraph.vcount.return_value = 4

        files = plot_leiden_resolutions(
            G=G,
            g_igraph=g_igraph,
            selected_resolutions=[1.0],
            rep_by_res=self._make_rep_by_res(),
            membership_to_leiden_df=self._make_membership_fn(),
            pos=pos,
            edge_widths=widths,
            edge_alphas=alphas,
            outdir=str(tmp_path / "leiden_plots"),
            save=True,
            show=False,
            draw_labels=False,
            use_adjust_text=False,
            dpi=10,
            figsize=(4, 3),
        )
        assert len(files) == 1
        assert os.path.exists(files[0])

    def test_no_files_when_save_false(self, tmp_path):
        G = _simple_graph()
        G.add_node("s4")
        pos = {n: (i, 0) for i, n in enumerate(G.nodes())}
        widths = np.array([1.0, 1.0, 1.0])
        alphas = np.array([0.5, 0.5, 0.5])
        g_igraph = MagicMock()
        g_igraph.vcount.return_value = 4

        files = plot_leiden_resolutions(
            G=G,
            g_igraph=g_igraph,
            selected_resolutions=[1.0],
            rep_by_res=self._make_rep_by_res(),
            membership_to_leiden_df=self._make_membership_fn(),
            pos=pos,
            edge_widths=widths,
            edge_alphas=alphas,
            outdir=str(tmp_path / "leiden_plots"),
            save=False,
            show=False,
            draw_labels=False,
            use_adjust_text=False,
            dpi=10,
            figsize=(4, 3),
        )
        assert files == []

    def test_empty_resolution_list_returns_empty(self, tmp_path):
        G = _simple_graph()
        files = plot_leiden_resolutions(
            G=G,
            g_igraph=MagicMock(),
            selected_resolutions=[],
            rep_by_res={},
            membership_to_leiden_df=self._make_membership_fn(),
            outdir=str(tmp_path / "leiden_plots"),
            save=True,
            show=False,
        )
        assert files == []

    def test_computes_layout_when_pos_is_none(self, tmp_path):
        G = _simple_graph()
        G.add_node("s4")
        g_igraph = MagicMock()
        g_igraph.vcount.return_value = 4

        # Should not raise even when pos=None
        plot_leiden_resolutions(
            G=G,
            g_igraph=g_igraph,
            selected_resolutions=[1.0],
            rep_by_res=self._make_rep_by_res(),
            membership_to_leiden_df=self._make_membership_fn(),
            pos=None,
            edge_widths=None,
            edge_alphas=None,
            outdir=str(tmp_path / "leiden_plots"),
            save=False,
            show=False,
            draw_labels=False,
            use_adjust_text=False,
            dpi=10,
            figsize=(4, 3),
        )

    def test_output_filename_contains_resolution(self, tmp_path):
        G = _simple_graph()
        G.add_node("s4")
        pos = {n: (i, 0) for i, n in enumerate(G.nodes())}
        widths = np.array([1.0, 1.0, 1.0])
        alphas = np.array([0.5, 0.5, 0.5])
        g_igraph = MagicMock()
        g_igraph.vcount.return_value = 4

        files = plot_leiden_resolutions(
            G=G,
            g_igraph=g_igraph,
            selected_resolutions=[1.0],
            rep_by_res=self._make_rep_by_res(),
            membership_to_leiden_df=self._make_membership_fn(),
            pos=pos,
            edge_widths=widths,
            edge_alphas=alphas,
            outdir=str(tmp_path / "leiden_plots"),
            save=True,
            show=False,
            draw_labels=False,
            use_adjust_text=False,
            dpi=10,
            figsize=(4, 3),
        )
        assert "leiden_rb_gamma_1.0" in files[0]


# ---------------------------------------------------------------------------
# TestPlotCommunitiesWithAnalysisSafe
# ---------------------------------------------------------------------------


def _make_leiden_df() -> pd.DataFrame:
    return pd.DataFrame({
        "stream": ["stream_a", "stream_b_json", "stream_c", "stream_d"],
        "community": [0, 0, 1, 1],
    })


def _make_stream_table_pdf() -> pd.DataFrame:
    return pd.DataFrame({
        "from": ["stream_a", "stream_c"],
        "to": ["stream_c", "stream_b_json"],
        "table": ["TABLE_X", "TABLE_Y"],
        "size": [10.0, 5.0],
    })


def _make_merged_edges_pdf() -> pd.DataFrame:
    return pd.DataFrame({
        "streamA": ["stream_a"],
        "streamB": ["stream_c"],
        "weight": [3.0],
    })


def _make_complexity_pdf() -> pd.DataFrame:
    return pd.DataFrame({
        "stream_name": ["stream_a", "stream_b_json", "stream_c", "stream_d"],
        "complexity_score": [4.0, 2.0, 6.0, 1.0],
    })


class TestPlotCommunitiesWithAnalysisSafe:
    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_edge("stream_a", "stream_b_json", weight=1.0)
        G.add_edge("stream_c", "stream_d", weight=2.0)
        return G

    def test_creates_analysis_txt_per_community(self, tmp_path):
        G = self._build_graph()
        leiden_df = _make_leiden_df()
        dep_spark = _mock_spark_df(_make_stream_table_pdf())
        merged_spark = _mock_spark_df(_make_merged_edges_pdf())
        complexity_spark = _mock_spark_df(_make_complexity_pdf())

        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=leiden_df,
            stream_table_dependency_df=dep_spark,
            merged_edges_df=merged_spark,
            complexity_scores_df=complexity_spark,
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )

        txt_files = [f for f in saved if f.endswith(".txt")]
        assert len(txt_files) == 2  # one per community

    def test_analysis_file_contains_community_id(self, tmp_path):
        G = self._build_graph()
        leiden_df = _make_leiden_df()

        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=leiden_df,
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )

        for txt_file in (f for f in saved if f.endswith(".txt")):
            content = open(txt_file).read()
            assert "Community" in content

    def test_analysis_file_contains_resolution(self, tmp_path):
        G = self._build_graph()
        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )
        txt_files = [f for f in saved if f.endswith(".txt")]
        content = open(txt_files[0]).read()
        assert "1.8" in content

    def test_outdir_subdirs_created(self, tmp_path):
        G = self._build_graph()
        outdir = str(tmp_path / "communities")

        plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=outdir,
            enable_plotting=False,
        )

        assert os.path.isdir(os.path.join(outdir, "community_0"))
        assert os.path.isdir(os.path.join(outdir, "community_1"))

    def test_no_png_when_plotting_disabled(self, tmp_path):
        G = self._build_graph()
        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )
        png_files = [f for f in saved if f.endswith(".png")]
        assert png_files == []

    def test_png_saved_when_plotting_enabled(self, tmp_path):
        G = self._build_graph()
        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=True,
            save=True,
            show=False,
            dpi=10,
            figsize=(4, 3),
        )
        png_files = [f for f in saved if f.endswith(".png")]
        assert len(png_files) == 2  # one per community

    def test_returns_list(self, tmp_path):
        G = self._build_graph()
        result = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )
        assert isinstance(result, list)

    def test_streams_classified_as_bi_or_etl(self, tmp_path):
        G = self._build_graph()
        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=_mock_spark_df(_make_stream_table_pdf()),
            merged_edges_df=_mock_spark_df(_make_merged_edges_pdf()),
            complexity_scores_df=_mock_spark_df(_make_complexity_pdf()),
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )
        # Community 0 has stream_b_json (BI) and stream_a (ETL)
        comm0_file = [
            f for f in saved if f.endswith(".txt") and "community_0" in f
        ][0]
        content = open(comm0_file).read()
        assert "BI REPORTS" in content
        assert "ETL STREAMS" in content

    def test_toPandas_called_on_spark_dfs(self, tmp_path):
        G = self._build_graph()
        dep_spark = _mock_spark_df(_make_stream_table_pdf())
        merged_spark = _mock_spark_df(_make_merged_edges_pdf())
        complexity_spark = _mock_spark_df(_make_complexity_pdf())

        plot_communities_with_analysis_safe(
            G=G,
            leiden_df=_make_leiden_df(),
            stream_table_dependency_df=dep_spark,
            merged_edges_df=merged_spark,
            complexity_scores_df=complexity_spark,
            resolution=1.8,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )

        dep_spark.toPandas.assert_called()
        merged_spark.toPandas.assert_called()
        complexity_spark.toPandas.assert_called()

    def test_single_community_graph(self, tmp_path):
        """Handles a degenerate case where all streams are in one community."""
        G = nx.Graph()
        G.add_edge("s1", "s2", weight=1.0)
        leiden_df = pd.DataFrame({"stream": ["s1", "s2"], "community": [0, 0]})
        # Use typed empty DataFrames so pandas .str accessor doesn't fail on float column
        dep = _mock_spark_df(
            pd.DataFrame({"from": pd.Series([], dtype=str), "to": pd.Series([], dtype=str),
                          "table": pd.Series([], dtype=str), "size": pd.Series([], dtype=float)})
        )
        merged = _mock_spark_df(
            pd.DataFrame({"streamA": pd.Series([], dtype=str), "streamB": pd.Series([], dtype=str),
                          "weight": pd.Series([], dtype=float)})
        )
        complexity = _mock_spark_df(
            pd.DataFrame({"stream_name": ["s1", "s2"], "complexity_score": [1.0, 2.0]})
        )

        saved = plot_communities_with_analysis_safe(
            G=G,
            leiden_df=leiden_df,
            stream_table_dependency_df=dep,
            merged_edges_df=merged,
            complexity_scores_df=complexity,
            resolution=1.0,
            outdir=str(tmp_path / "out"),
            enable_plotting=False,
        )

        txt_files = [f for f in saved if f.endswith(".txt")]
        assert len(txt_files) == 1
