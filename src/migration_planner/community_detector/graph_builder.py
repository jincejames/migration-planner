from __future__ import annotations

from typing import TYPE_CHECKING

import igraph as ig
import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def find_isolated_streams(
    dependency_df: DataFrame,
    merged_dependency_df: DataFrame,
) -> list[str]:
    """Return streams that have no inter-stream edge (only intra-stream deps)."""
    all_streams = [
        row["stream_name"]
        for row in dependency_df.select("stream_name").distinct().collect()
    ]
    in_edges = {
        row["streamA"]
        for row in merged_dependency_df.select("streamA").distinct().collect()
    } | {
        row["streamB"]
        for row in merged_dependency_df.select("streamB").distinct().collect()
    }
    return [s for s in all_streams if s not in in_edges]


def build_igraph(edges: pd.DataFrame, isolated_streams: list[str]) -> ig.Graph:
    """
    Build an undirected weighted igraph from a pandas edges DataFrame
    (columns: streamA, streamB, weight) and add any isolated stream nodes.
    """
    g = ig.Graph.TupleList(
        edges[["streamA", "streamB", "weight"]].itertuples(index=False, name=None),
        directed=False,
        edge_attrs=["weight"],
    )
    if isolated_streams:
        g.add_vertices(isolated_streams)
    return g


def build_networkx_graph(edges: pd.DataFrame, isolated_streams: list[str]) -> nx.Graph:
    """Build a NetworkX Graph from the same edge list, adding any isolated nodes."""
    G = nx.from_pandas_edgelist(
        edges, "streamA", "streamB", edge_attr="weight", create_using=nx.Graph()
    )
    if isolated_streams:
        G.add_nodes_from(isolated_streams)
    return G
