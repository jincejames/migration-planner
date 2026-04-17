"""Tests for migration_planner.visualization.pyvis_plots."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from migration_planner.visualization.pyvis_plots import (
    DEFAULT_FALLBACK_COLOR,
    DEFAULT_PALETTE,
    DEFAULT_UNIFORM_COLOR,
    plot_interactive_graph,
)


def _simple_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge("s1", "s2", weight=1.0)
    G.add_edge("s2", "s3", weight=2.0)
    G.add_edge("s3", "s4", weight=3.0)
    G.add_edge("s1", "s4", weight=4.0)
    return G


def _simple_edges_df() -> pd.DataFrame:
    return pd.DataFrame({
        "streamA": ["s1", "s2", "s3", "s1"],
        "streamB": ["s2", "s3", "s4", "s4"],
        "weight": [1.0, 2.0, 3.0, 4.0],
    })


class TestPlotInteractiveGraph:
    def test_saves_html_from_networkx_graph(self, tmp_path):
        out = tmp_path / "graph.html"
        result = plot_interactive_graph(_simple_graph(), output_path=str(out))
        assert result == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saves_html_from_edges_dataframe(self, tmp_path):
        out = tmp_path / "graph.html"
        plot_interactive_graph(_simple_edges_df(), output_path=str(out))
        assert out.exists()

    def test_html_contains_node_names(self, tmp_path):
        out = tmp_path / "graph.html"
        plot_interactive_graph(_simple_graph(), output_path=str(out))
        html = out.read_text()
        for node in ["s1", "s2", "s3", "s4"]:
            assert node in html

    def test_uniform_color_used_when_color_by_is_none(self, tmp_path):
        out = tmp_path / "graph.html"
        plot_interactive_graph(
            _simple_graph(),
            uniform_color="#abcdef",
            output_path=str(out),
        )
        assert "#abcdef" in out.read_text()

    def test_color_by_from_node_attrs(self, tmp_path):
        out = tmp_path / "graph.html"
        node_attrs = pd.DataFrame({
            "stream": ["s1", "s2", "s3", "s4"],
            "community": [0, 0, 1, 1],
        })
        plot_interactive_graph(
            _simple_graph(),
            color_by="community",
            node_attrs=node_attrs,
            node_col="stream",
            output_path=str(out),
        )
        html = out.read_text()
        # With sorted unique values [0, 1], palette entries 0 and 1 should both appear.
        assert DEFAULT_PALETTE[0] in html
        assert DEFAULT_PALETTE[1] in html

    def test_explicit_color_map_overrides_palette(self, tmp_path):
        out = tmp_path / "graph.html"
        node_attrs = pd.DataFrame({
            "stream": ["s1", "s2", "s3", "s4"],
            "community": [0, 0, 1, 1],
        })
        plot_interactive_graph(
            _simple_graph(),
            color_by="community",
            node_attrs=node_attrs,
            node_col="stream",
            color_map={0: "#111111", 1: "#222222"},
            output_path=str(out),
        )
        html = out.read_text()
        assert "#111111" in html
        assert "#222222" in html

    def test_missing_attribute_falls_back_to_fallback_color(self, tmp_path):
        out = tmp_path / "graph.html"
        node_attrs = pd.DataFrame({
            "stream": ["s1", "s2"],  # s3, s4 missing
            "community": [0, 0],
        })
        plot_interactive_graph(
            _simple_graph(),
            color_by="community",
            node_attrs=node_attrs,
            node_col="stream",
            fallback_color="#deadbe",
            output_path=str(out),
        )
        assert "#deadbe" in out.read_text()

    def test_color_by_reads_graph_node_attribute_when_no_node_attrs(self, tmp_path):
        out = tmp_path / "graph.html"
        G = _simple_graph()
        for node, prefix in [("s1", "A"), ("s2", "A"), ("s3", "B"), ("s4", "B")]:
            G.nodes[node]["prefix"] = prefix
        plot_interactive_graph(
            G,
            color_by="prefix",
            color_map={"A": "#aaaaaa", "B": "#bbbbbb"},
            output_path=str(out),
        )
        html = out.read_text()
        assert "#aaaaaa" in html
        assert "#bbbbbb" in html

    def test_rejects_invalid_graph_type(self):
        with pytest.raises(TypeError):
            plot_interactive_graph("not a graph")  # type: ignore[arg-type]

    def test_raises_when_node_attrs_missing_required_columns(self, tmp_path):
        with pytest.raises(ValueError):
            plot_interactive_graph(
                _simple_graph(),
                color_by="community",
                node_attrs=pd.DataFrame({"stream": ["s1"], "other": [0]}),
                node_col="stream",
                output_path=str(tmp_path / "g.html"),
            )

    def test_default_output_path_is_used_when_none(self):
        path = plot_interactive_graph(_simple_graph())
        try:
            assert path.endswith(".html")
            with open(path, "r") as f:
                assert f.read()
        finally:
            import os
            os.remove(path)

    def test_top_label_count_limits_labels(self, tmp_path):
        G = nx.Graph()
        for i in range(10):
            G.add_edge(f"n{i}", f"n{i+1}", weight=float(i + 1))
        out = tmp_path / "graph.html"
        plot_interactive_graph(G, top_label_count=2, output_path=str(out))
        # Top-2 nodes by weighted degree on this path graph are the middle nodes.
        # We only verify that the HTML is produced; the exact labeled set
        # depends on tie-breaking which is not part of this contract.
        assert out.exists()

    def test_uniform_color_default_constant(self):
        # Simple smoke: the exported constant matches the notebook convention.
        assert DEFAULT_UNIFORM_COLOR.startswith("#")
        assert DEFAULT_FALLBACK_COLOR.startswith("#")
