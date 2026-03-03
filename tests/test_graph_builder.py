"""
Tests for migration_planner.community_detector.graph_builder.

graph_builder.py uses real igraph and networkx (in dev deps) for build_igraph /
build_networkx_graph, and MagicMock Spark DataFrames for find_isolated_streams.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import igraph as ig
import networkx as nx
import pandas as pd
import pytest

from migration_planner.community_detector.graph_builder import (
    build_igraph,
    build_networkx_graph,
    find_isolated_streams,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_df_with_rows(rows: list[dict], col: str) -> MagicMock:
    """Return a mock DF whose collect() yields Row-like objects for *col*."""
    df = MagicMock()
    mock_rows = [type("Row", (), {"__getitem__": lambda self, k, _v=v: _v})() for v in [r[col] for r in rows]]
    df.select.return_value = df
    df.distinct.return_value = df
    df.collect.return_value = mock_rows
    return df


def _sample_edges() -> pd.DataFrame:
    return pd.DataFrame({
        "streamA": ["s1", "s2", "s3"],
        "streamB": ["s2", "s3", "s4"],
        "weight": [1.0, 2.0, 3.0],
    })


# ---------------------------------------------------------------------------
# TestFindIsolatedStreams
# ---------------------------------------------------------------------------


class TestFindIsolatedStreams:
    def _build_dep_df(self, stream_names: list[str]) -> MagicMock:
        df = MagicMock()
        mock_rows = [
            type("Row", (), {"__getitem__": lambda self, k, _n=n: _n})()
            for n in stream_names
        ]
        inner = MagicMock()
        inner.collect.return_value = mock_rows
        df.select.return_value = inner
        inner.distinct.return_value = inner
        return df

    def _build_merged_df(self, streamA_names: list[str], streamB_names: list[str]) -> MagicMock:
        df = MagicMock()

        def _select_side_effect(col_name):
            names = streamA_names if col_name == "streamA" else streamB_names
            mock_rows = [
                type("Row", (), {"__getitem__": lambda self, k, _n=n: _n})()
                for n in names
            ]
            inner = MagicMock()
            inner.distinct.return_value = inner
            inner.collect.return_value = mock_rows
            return inner

        df.select.side_effect = _select_side_effect
        return df

    def test_returns_streams_not_in_any_edge(self):
        dep = self._build_dep_df(["s1", "s2", "s3", "s4"])
        merged = self._build_merged_df(["s1", "s2"], ["s2", "s3"])
        result = find_isolated_streams(dep, merged)
        assert "s4" in result
        assert "s1" not in result

    def test_empty_when_all_streams_have_edges(self):
        dep = self._build_dep_df(["s1", "s2"])
        merged = self._build_merged_df(["s1"], ["s2"])
        result = find_isolated_streams(dep, merged)
        assert result == []

    def test_calls_collect_on_dependency_df(self):
        dep = self._build_dep_df([])
        merged = self._build_merged_df([], [])
        find_isolated_streams(dep, merged)
        dep.select.return_value.distinct.return_value.collect.assert_called_once()

    def test_returns_list(self):
        dep = self._build_dep_df(["s1"])
        merged = self._build_merged_df(["s1"], [])
        result = find_isolated_streams(dep, merged)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestBuildIgraph
# ---------------------------------------------------------------------------


class TestBuildIgraph:
    def test_node_count_equals_edge_nodes_plus_isolated(self):
        edges = _sample_edges()
        # unique nodes in edges: s1, s2, s3, s4 (4 nodes)
        g = build_igraph(edges, ["s5"])
        assert g.vcount() == 5

    def test_edge_count_matches_input(self):
        edges = _sample_edges()
        g = build_igraph(edges, [])
        assert g.ecount() == 3

    def test_edge_weights_preserved(self):
        edges = _sample_edges()
        g = build_igraph(edges, [])
        weights = g.es["weight"]
        assert set(weights) == {1.0, 2.0, 3.0}

    def test_graph_is_undirected(self):
        edges = _sample_edges()
        g = build_igraph(edges, [])
        assert not g.is_directed()

    def test_isolated_nodes_added(self):
        edges = _sample_edges()
        g = build_igraph(edges, ["isolated_a", "isolated_b"])
        node_names = g.vs["name"]
        assert "isolated_a" in node_names
        assert "isolated_b" in node_names

    def test_empty_isolated_streams_no_extra_nodes(self):
        edges = _sample_edges()
        g_with = build_igraph(edges, [])
        g_without = build_igraph(edges, [])
        assert g_with.vcount() == g_without.vcount()

    def test_returns_igraph_graph(self):
        edges = _sample_edges()
        g = build_igraph(edges, [])
        assert isinstance(g, ig.Graph)


# ---------------------------------------------------------------------------
# TestBuildNetworkxGraph
# ---------------------------------------------------------------------------


class TestBuildNetworkxGraph:
    def test_node_count_equals_edge_nodes_plus_isolated(self):
        edges = _sample_edges()
        G = build_networkx_graph(edges, ["s5"])
        assert G.number_of_nodes() == 5

    def test_edge_count_matches_input(self):
        edges = _sample_edges()
        G = build_networkx_graph(edges, [])
        assert G.number_of_edges() == 3

    def test_edge_weights_preserved(self):
        edges = _sample_edges()
        G = build_networkx_graph(edges, [])
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        assert set(weights) == {1.0, 2.0, 3.0}

    def test_isolated_nodes_added(self):
        edges = _sample_edges()
        G = build_networkx_graph(edges, ["iso_x"])
        assert "iso_x" in G.nodes

    def test_is_undirected_graph(self):
        edges = _sample_edges()
        G = build_networkx_graph(edges, [])
        assert isinstance(G, nx.Graph)
        assert not isinstance(G, nx.DiGraph)

    def test_returns_networkx_graph(self):
        edges = _sample_edges()
        G = build_networkx_graph(edges, [])
        assert isinstance(G, nx.Graph)
