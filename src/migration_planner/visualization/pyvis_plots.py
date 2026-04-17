"""Interactive PyVis visualizations for the migration-planner graph.

The main entry point is :func:`plot_interactive_graph`, which renders any
weighted graph (NetworkX or edges DataFrame) to an interactive HTML file.
Nodes can be a single uniform color, or coloured by any column in an
optional node-attribute table (e.g. Leiden community, stream prefix).
"""
from __future__ import annotations

import os
import tempfile
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults (kept at module level so callers can import and override)
# ---------------------------------------------------------------------------

DEFAULT_UNIFORM_COLOR = "#f4b942"
DEFAULT_FALLBACK_COLOR = "#B0B0B0"
DEFAULT_BG_COLOR = "#1b3139"
DEFAULT_FONT_COLOR = "white"

# 25-colour palette, stable across runs. Leading entries match the
# prefix-colouring used in the reference notebook so the look stays consistent.
DEFAULT_PALETTE: tuple[str, ...] = (
    "#114b5f", "#1a936f", "#f4b942", "#780116", "#f3e9d2",
    "#4D96FF", "#FF6B6B", "#6BCB77", "#B892FF", "#43AA8B",
    "#577590", "#F3722C", "#277DA1", "#90BE6D", "#F9C74F",
    "#C77DFF", "#00BBF9", "#E76F51", "#2A9D8F", "#A8DADC",
    "#D62828", "#7B2CBF", "#3A86FF", "#8338EC", "#FF006E",
)

DEFAULT_PHYSICS: dict = {
    "enabled": True,
    "solver": "barnesHut",
    "barnesHut": {
        "theta": 0.5,
        "gravitationalConstant": -3000,
        "centralGravity": 0.2,
        "springLength": 140,
        "springConstant": 0.04,
        "damping": 0.09,
        "avoidOverlap": 0,
    },
    "maxVelocity": 11,
    "minVelocity": 0.75,
    "timestep": 0.5,
    "wind": {"x": 0, "y": 0},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_to_graph(
    graph: nx.Graph | pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str,
) -> nx.Graph:
    """Return a NetworkX graph from either an existing graph or an edges DataFrame."""
    if isinstance(graph, nx.Graph):
        return graph
    if not isinstance(graph, pd.DataFrame):
        raise TypeError(
            f"graph must be nx.Graph or pandas DataFrame, got {type(graph).__name__}"
        )
    edges = graph.copy()
    edges[source_col] = edges[source_col].astype(str)
    edges[target_col] = edges[target_col].astype(str)
    edges[weight_col] = pd.to_numeric(edges[weight_col], errors="coerce").fillna(0.0)
    return nx.from_pandas_edgelist(
        edges,
        source=source_col,
        target=target_col,
        edge_attr=weight_col,
        create_using=nx.Graph(),
    )


def _scale_node_sizes(
    weighted_degree: dict,
    size_min: float,
    size_range: float,
) -> dict:
    """Min-max scale weighted degrees into [size_min, size_min + size_range]."""
    values = np.array(list(weighted_degree.values()), dtype=float)
    if values.size == 0:
        return {}
    v_min, v_max = float(values.min()), float(values.max())
    den = (v_max - v_min) if (v_max - v_min) > 0 else 1.0
    return {n: size_min + size_range * ((float(v) - v_min) / den)
            for n, v in weighted_degree.items()}


def _scale_edge_widths(
    G: nx.Graph,
    weight_attr: str,
    width_min: float,
    width_range: float,
) -> dict:
    """Return {(u, v): width} mapping based on normalised edge weights."""
    weights = [float(d.get(weight_attr, 1.0)) for _, _, d in G.edges(data=True)]
    if not weights:
        return {}
    w_min, w_max = min(weights), max(weights)
    den = (w_max - w_min) if (w_max - w_min) > 0 else 1.0
    return {
        (u, v): width_min + width_range * ((float(d.get(weight_attr, 1.0)) - w_min) / den)
        for u, v, d in G.edges(data=True)
    }


def _build_color_resolver(
    G: nx.Graph,
    color_by: str | None,
    node_attrs: pd.DataFrame | None,
    node_col: str,
    color_map: dict | None,
    palette: Iterable[str],
    uniform_color: str,
    fallback_color: str,
):
    """Return (resolver, value_getter) where resolver(node) -> hex color.

    ``value_getter(node)`` returns the raw coloring value (for tooltip use) or
    None when uniform coloring is in effect.
    """
    if color_by is None:
        return (lambda _n: uniform_color), (lambda _n: None)

    if node_attrs is not None:
        if node_col not in node_attrs.columns:
            raise ValueError(
                f"node_attrs must contain a {node_col!r} column (got {list(node_attrs.columns)})"
            )
        if color_by not in node_attrs.columns:
            raise ValueError(
                f"node_attrs must contain the color_by={color_by!r} column "
                f"(got {list(node_attrs.columns)})"
            )
        node_to_value = dict(zip(
            node_attrs[node_col].astype(str),
            node_attrs[color_by],
        ))
        getter = lambda n: node_to_value.get(str(n))
    else:
        # Fall back to reading the attribute straight off the graph node.
        getter = lambda n: G.nodes[n].get(color_by)

    if color_map is None:
        palette_list = list(palette)
        unique_values = []
        seen: set = set()
        for n in G.nodes():
            v = getter(n)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if v not in seen:
                seen.add(v)
                unique_values.append(v)
        # Sort when homogeneous & comparable — keeps colours stable across runs.
        try:
            unique_values = sorted(unique_values)
        except TypeError:
            pass
        color_map = {
            v: palette_list[i % len(palette_list)]
            for i, v in enumerate(unique_values)
        }

    def resolver(n):
        v = getter(n)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return fallback_color
        return color_map.get(v, fallback_color)

    return resolver, getter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_interactive_graph(
    graph: nx.Graph | pd.DataFrame,
    *,
    color_by: str | None = None,
    node_attrs: pd.DataFrame | None = None,
    node_col: str = "node",
    color_map: dict | None = None,
    palette: Iterable[str] = DEFAULT_PALETTE,
    uniform_color: str = DEFAULT_UNIFORM_COLOR,
    fallback_color: str = DEFAULT_FALLBACK_COLOR,
    source_col: str = "streamA",
    target_col: str = "streamB",
    weight_col: str = "weight",
    top_label_count: int = 40,
    node_size_min: float = 10.0,
    node_size_range: float = 40.0,
    edge_width_min: float = 1.0,
    edge_width_range: float = 8.0,
    height: str = "1500px",
    width: str = "100%",
    bgcolor: str = DEFAULT_BG_COLOR,
    font_color: str = DEFAULT_FONT_COLOR,
    physics_options: dict | None = None,
    output_path: str | None = None,
    cdn_resources: str = "in_line",
) -> str:
    """Render a weighted graph as an interactive PyVis HTML file.

    Parameters
    ----------
    graph
        Either a ``networkx.Graph`` or a pandas edges DataFrame with
        ``source_col`` / ``target_col`` / ``weight_col`` columns.
    color_by
        Optional coloring key. When ``None`` all nodes use ``uniform_color``.
        When provided, the value is looked up either on ``node_attrs`` (if
        given) or directly on each graph node's attribute dict.
    node_attrs
        Optional DataFrame with a ``node_col`` column plus the ``color_by``
        column (e.g. Leiden output with columns ``stream`` and ``community``
        — then pass ``node_col="stream"``, ``color_by="community"``).
    color_map
        Explicit ``value -> hex`` map. When ``None`` colors are assigned from
        ``palette`` in sorted order of the distinct coloring values.
    output_path
        Where to write the HTML. When ``None`` a temp file is used.

    Returns
    -------
    str
        Path of the saved HTML file.
    """
    # Imported lazily so the rest of the package can be imported without pyvis.
    from pyvis.network import Network

    G = _coerce_to_graph(graph, source_col, target_col, weight_col).copy()

    weighted_degree = dict(G.degree(weight=weight_col))
    plain_degree = dict(G.degree())
    node_sizes = _scale_node_sizes(weighted_degree, node_size_min, node_size_range)
    edge_widths = _scale_edge_widths(G, weight_col, edge_width_min, edge_width_range)

    top_nodes = set(
        pd.Series(weighted_degree)
        .sort_values(ascending=False)
        .head(top_label_count)
        .index
        .tolist()
    )

    color_of, value_of = _build_color_resolver(
        G,
        color_by=color_by,
        node_attrs=node_attrs,
        node_col=node_col,
        color_map=color_map,
        palette=palette,
        uniform_color=uniform_color,
        fallback_color=fallback_color,
    )

    for n in G.nodes():
        wd = weighted_degree.get(n, 0.0)
        deg = plain_degree.get(n, 0)
        G.nodes[n]["label"] = str(n) if n in top_nodes else ""
        G.nodes[n]["size"] = node_sizes.get(n, node_size_min)
        G.nodes[n]["color"] = color_of(n)

        tooltip = [f"<b>{n}</b>"]
        if color_by is not None:
            tooltip.append(f"{color_by}: {value_of(n)}")
        tooltip.append(f"Degree: {deg}")
        tooltip.append(f"Weighted degree: {wd:.2f}")
        G.nodes[n]["title"] = "<br>".join(tooltip)

    for u, v, d in G.edges(data=True):
        w = float(d.get(weight_col, 1.0))
        d["value"] = edge_widths.get((u, v), edge_width_min)
        d["title"] = f"{u} ↔ {v}<br>weight={w:.3f}"

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        cdn_resources=cdn_resources,
    )
    net.from_nx(G)
    net.set_options("var options = " + _physics_options_json(physics_options or DEFAULT_PHYSICS))

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".html", prefix="graph_")
        os.close(fd)
    else:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    net.save_graph(output_path)
    return output_path


def _physics_options_json(physics: dict) -> str:
    """Serialize physics settings into the JSON blob PyVis' set_options expects."""
    import json
    return json.dumps({"physics": physics, "configure": {"enabled": True, "filter": ["physics"]}})
