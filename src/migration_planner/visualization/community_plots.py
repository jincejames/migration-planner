"""Visualization helpers for the Migration Planner community-detection pipeline.

Functions
---------
select_resolutions
    Filter a resolution scan summary table to candidate resolutions.
precompute_layout
    Compute a spring layout for a NetworkX graph (deterministic).
precompute_edge_style
    Compute per-edge widths and opacities from edge weights.
plot_leiden_resolutions
    Render one high-resolution PNG per selected resolution, coloring nodes
    by community assignment.
edge_style
    Compute per-edge widths and alphas for a community sub-graph.
plot_communities_with_analysis_safe
    For every community at a given resolution: delegate analysis text-file
    writing to ``planner_core.analysis.generate_community_analysis`` and
    (optionally) save a sub-graph PNG.
"""
from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from migration_planner.planner_core.analysis import generate_community_analysis


def select_resolutions(
    summary: pd.DataFrame,
    ari_target: float = 0.93,
    max_largest_comm_share: float = 0.2,
    sort_by: str = "resolution",
) -> tuple[pd.DataFrame, list[float]]:
    """
    Select resolutions from the summary table using simple filters.

    Parameters
    ----------
    summary : pd.DataFrame
        Resolution scan summary produced by ``scan_resolutions()``.
    ari_target : float
        Keep resolutions whose stability_ari is >= this value (default 0.93).
    max_largest_comm_share : float
        Keep resolutions with avg largest community share <= this threshold
        (default 0.20).
    sort_by : str
        Column to sort the result by (default "resolution").

    Returns
    -------
    selected : pd.DataFrame
        Filtered summary rows.
    selected_resolutions : list[float]
        Resolution values that passed the filters.
    """
    selected = summary[
        (summary["stability_ari"] >= ari_target)
        & (summary["largest_comm_share_avg"] <= max_largest_comm_share)
    ].sort_values(sort_by)
    selected_resolutions = selected["resolution"].tolist()
    return selected, selected_resolutions


def precompute_layout(
    G: nx.Graph,
    seed: int = 42,
    k: float = 3.0,
) -> dict:
    """
    Precompute a single spring layout so that plots across resolutions are
    spatially comparable.

    Parameters
    ----------
    G : nx.Graph
        Graph to lay out.
    seed : int
        Random seed for a deterministic layout.
    k : float
        Spring layout distance parameter. Higher values spread nodes further
        apart.

    Returns
    -------
    pos : dict
        Mapping of node -> (x, y) position.
    """
    print(f"Computing layout with k={k} for better node spacing...")
    pos = nx.spring_layout(G, seed=seed, k=k, iterations=100)
    return pos


def precompute_edge_style(
    G: nx.Graph,
    weight_attr: str = "weight",
    width_min: float = 0.2,
    width_scale: float = 3.0,
    alpha_min: float = 0.05,
    alpha_scale: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-edge widths and alpha values from edge weights.

    All weights are min-max normalised before scaling so the result is
    independent of the absolute weight range.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edges will be styled.
    weight_attr : str
        Edge attribute name to use as weight (default ``"weight"``).
    width_min, width_scale : float
        ``width = width_min + width_scale * normalised_weight``
    alpha_min, alpha_scale : float
        ``alpha = alpha_min + alpha_scale * normalised_weight``

    Returns
    -------
    edge_widths : np.ndarray
        Width per edge in ``G.edges()`` order. Empty array when G has no edges.
    edge_alphas : np.ndarray
        Alpha per edge in ``G.edges()`` order. Empty array when G has no edges.
    """
    w = np.array([G[u][v].get(weight_attr, 1.0) for u, v in G.edges()], dtype=float)

    if w.size == 0:
        return np.array([]), np.array([])

    w_min, w_max = float(w.min()), float(w.max())
    den = (w_max - w_min) if (w_max - w_min) > 0 else 1.0
    w_norm = (w - w_min) / den

    edge_widths = width_min + width_scale * w_norm
    edge_alphas = alpha_min + alpha_scale * w_norm
    return edge_widths, edge_alphas


def plot_leiden_resolutions(
    G: nx.Graph,
    g_igraph,
    selected_resolutions: list,
    rep_by_res: dict,
    membership_to_leiden_df: Callable,
    pos: dict = None,
    edge_widths=None,
    edge_alphas=None,
    outdir: str = "leiden_plots",
    figsize: tuple = (50, 40),
    dpi: int = 400,
    label_fontsize: int = 11,
    node_size: int = 800,
    cmap=plt.cm.tab20,
    draw_labels: bool = True,
    save: bool = True,
    show: bool = True,
    use_adjust_text: bool = True,
) -> list[str]:
    """
    Render one high-resolution PNG for each selected Leiden resolution,
    coloring every node by its community assignment.

    Parameters
    ----------
    G : nx.Graph
        Full NetworkX graph (all streams).
    g_igraph : igraph.Graph
        The igraph used for Leiden; only ``g_igraph.vcount()`` is called here.
    selected_resolutions : list
        Resolution values to plot.
    rep_by_res : dict
        Mapping ``resolution -> dict`` with keys ``"membership"``,
        ``"quality"``, ``"seed"``.
    membership_to_leiden_df : callable
        Maps a membership array to a DataFrame with columns
        ``["stream", "community"]``.
    pos : dict or None
        Pre-computed node positions. Computed via spring layout if ``None``.
    edge_widths, edge_alphas : array-like or None
        Pre-computed per-edge styling. Computed from weights if ``None``.
    outdir : str
        Directory where PNGs are saved.
    figsize : tuple
        Matplotlib figure size.
    dpi : int
        Output resolution in dots per inch.
    label_fontsize : int
        Font size for node labels.
    node_size : int
        Node marker size.
    cmap : matplotlib colormap
        Colormap used to color communities.
    draw_labels : bool
        Whether to draw node labels.
    save : bool
        Save each figure to *outdir*.
    show : bool
        Call ``plt.show()`` after each figure.
    use_adjust_text : bool
        Use the *adjustText* library to prevent label overlap. Requires
        ``adjustText`` to be installed.

    Returns
    -------
    outputs : list[str]
        Paths of saved PNG files (empty when *save* is ``False``).
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=3.0, iterations=100)

    if edge_widths is None or edge_alphas is None:
        edge_widths, edge_alphas = precompute_edge_style(G)

    os.makedirs(outdir, exist_ok=True)
    outputs = []

    for res in selected_resolutions:
        print(f"\nGenerating plot for resolution γ={res}...")
        rep = rep_by_res[res]
        membership = rep["membership"]
        quality = rep["quality"]

        leiden_df = membership_to_leiden_df(membership)

        node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))
        node_colors = [node_to_comm.get(n, -1) for n in G.nodes()]

        counts = np.bincount(membership)
        n_comms = len(counts)
        largest_share = counts.max() / g_igraph.vcount()
        tiny_lt5 = int((counts < 5).sum())

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        print(f"  Drawing {G.number_of_edges()} edges...")
        for (u, v), lw, a in zip(G.edges(), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=float(lw),
                alpha=float(a),
                ax=ax,
                edge_color='gray',
            )

        print(f"  Drawing {G.number_of_nodes()} nodes...")
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_size,
            cmap=cmap,
            linewidths=2.0,
            edgecolors='black',
            ax=ax,
        )

        if draw_labels:
            if use_adjust_text:
                from adjustText import adjust_text  # optional dep
                print("  Adding labels with overlap prevention...")
                texts = []
                for node, (x, y) in pos.items():
                    texts.append(
                        ax.text(
                            x, y, str(node),
                            fontsize=label_fontsize,
                            ha='center',
                            va='center',
                            fontweight='bold',
                            bbox=dict(
                                boxstyle='round,pad=0.4',
                                facecolor='white',
                                edgecolor='gray',
                                linewidth=0.5,
                                alpha=0.85,
                            ),
                        )
                    )
                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6),
                    expand_points=(2.0, 2.0),
                    expand_text=(1.5, 1.5),
                    force_points=(0.8, 0.8),
                    force_text=(0.8, 0.8),
                    ax=ax,
                    lim=500,
                )
            else:
                for node, (x, y) in pos.items():
                    ax.text(
                        x, y, str(node),
                        fontsize=label_fontsize,
                        ha='center',
                        va='center',
                        fontweight='bold',
                        bbox=dict(
                            boxstyle='round,pad=0.4',
                            facecolor='white',
                            edgecolor='gray',
                            linewidth=0.5,
                            alpha=0.85,
                        ),
                    )

        title = (
            f"Leiden (RBConfiguration) — resolution γ={res} (seed={rep['seed']})\n"
            f"#communities={n_comms}, largest_comm_share={largest_share:.3f}, "
            f"small_comms<5={tiny_lt5}, quality={quality:.2f}"
        )
        ax.set_title(title, fontsize=22, fontweight='bold', pad=30)
        ax.axis("off")
        ax.margins(0.1)

        plt.tight_layout()

        if save:
            outfile = os.path.join(outdir, f"leiden_rb_gamma_{res}.png")
            print("  Saving high-resolution plot...")
            plt.savefig(outfile, bbox_inches="tight", dpi=dpi, facecolor='white')
            outputs.append(outfile)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return outputs


def edge_style(
    sub_g: nx.Graph,
    weight_attr: str = "weight",
    min_w: float = 0.2,
    max_w: float = 3.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-edge widths and alpha values for a community sub-graph.

    Parameters
    ----------
    sub_g : nx.Graph
        Sub-graph whose edges will be styled.
    weight_attr : str
        Edge attribute name to use as weight (default ``"weight"``).
    min_w, max_w : float
        Minimum and maximum rendered edge width.

    Returns
    -------
    widths : np.ndarray
        Width per edge (empty list when *sub_g* has no edges).
    alphas : np.ndarray
        Alpha (opacity) per edge (empty list when *sub_g* has no edges).
    """
    if sub_g.number_of_edges() == 0:
        return [], []
    w = np.array([sub_g[u][v].get(weight_attr, 1.0) for u, v in sub_g.edges()], dtype=float)
    w_min, w_max = float(w.min()), float(w.max())
    widths = min_w + (max_w - min_w) * (w - w_min) / (w_max - w_min + 1e-9)
    alphas = 0.10 + 0.60 * (w - w_min) / (w_max - w_min + 1e-9)
    return widths, alphas


def plot_communities_with_analysis_safe(
    G: nx.Graph,
    leiden_df: pd.DataFrame,
    stream_table_dependency_df,
    merged_edges_df,
    complexity_scores_df,
    resolution: float,
    outdir: str = "leiden_community_plots",
    layout_seed: int = 42,
    layout_k=None,
    layout_iterations: int = 30,
    weight_attr: str = "weight",
    figsize: tuple = (24, 18),
    dpi: int = 220,
    node_size: int = 220,
    font_size: int = 8,
    cmap=plt.cm.tab20,
    show: bool = False,
    save: bool = True,
    filename_prefix: str = "community",
    max_labels: int = 120,
    enable_plotting: bool = False,
) -> list[str]:
    """
    For every community at a given Leiden resolution: write a structured
    analysis text file and (optionally) save a community sub-graph PNG.

    Analysis text files are always written regardless of *enable_plotting*.
    The analysis logic is delegated to
    :func:`~migration_planner.planner_core.analysis.generate_community_analysis`;
    this function is responsible only for the optional graph plots.

    Parameters
    ----------
    G : nx.Graph
        Full NetworkX graph (all streams).
    leiden_df : pd.DataFrame
        DataFrame with columns ``['stream', 'community']``.
    stream_table_dependency_df : pyspark.sql.DataFrame
        Full stream-to-table dependency Spark DataFrame.  Converted to pandas
        once via ``.toPandas()`` before being passed to
        ``generate_community_analysis``.
    merged_edges_df : pyspark.sql.DataFrame
        Merged, bidirectional edge-list Spark DataFrame.  Converted to pandas
        once via ``.toPandas()`` before being passed to
        ``generate_community_analysis``.
    complexity_scores_df : pyspark.sql.DataFrame
        Complexity scores Spark DataFrame.  Converted to pandas once via
        ``.toPandas()`` before being passed to ``generate_community_analysis``.
    resolution : float
        Leiden resolution used (for labelling output files only).
    outdir : str
        Root output directory. Created if it does not exist.
    layout_seed : int
        Random seed for the spring layout.
    layout_k : float or None
        Spring layout distance parameter passed to NetworkX.
    layout_iterations : int
        Number of spring-layout iterations.
    weight_attr : str
        Edge attribute used as weight for rendering.
    figsize : tuple
        Matplotlib figure size.
    dpi : int
        Output resolution.
    node_size : int
        Node marker size.
    font_size : int
        Label font size.
    cmap : matplotlib colormap
        Colormap for community colours.
    show : bool
        Call ``plt.show()`` after each plot.
    save : bool
        Save PNG plots to *outdir*.
    filename_prefix : str
        Prefix for PNG filenames.
    max_labels : int
        Maximum number of node labels to render per community.
    enable_plotting : bool
        When ``False`` only the analysis text files are written; no plots are
        generated.

    Returns
    -------
    saved_files : list[str]
        Paths of all files written (analysis ``.txt`` files and PNG plots).
    """
    os.makedirs(outdir, exist_ok=True)

    node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))
    labeled_nodes = [n for n in G.nodes() if n in node_to_comm]

    pos_global = None
    if enable_plotting:
        pos_global = nx.spring_layout(
            G.subgraph(labeled_nodes),
            seed=layout_seed,
            k=layout_k,
            iterations=layout_iterations,
        )

    # Convert Spark DFs to pandas once — no recomputation downstream
    stream_table_pdf = stream_table_dependency_df.toPandas()
    merged_edges_pdf = merged_edges_df.toPandas()
    complexity_pdf = complexity_scores_df.toPandas()

    # Delegate all analysis (text file writing) to planner_core
    saved_files = generate_community_analysis(
        leiden_df=leiden_df,
        stream_table_pdf=stream_table_pdf,
        merged_edges_pdf=merged_edges_pdf,
        complexity_pdf=complexity_pdf,
        resolution=resolution,
        outdir=outdir,
    )

    if not enable_plotting:
        return saved_files

    # ---- PLOTTING (only if enabled) ----
    communities = sorted(leiden_df["community"].unique())

    for c in communities:
        comm_dir = os.path.join(outdir, f"community_{c}")
        os.makedirs(comm_dir, exist_ok=True)

        comm_nodes = [n for n in labeled_nodes if node_to_comm[n] == c]
        h_graph = G.subgraph(comm_nodes).copy()
        print(f"Community {c}: nodes={h_graph.number_of_nodes()}, edges={h_graph.number_of_edges()}")

        pos = {n: pos_global[n] for n in h_graph.nodes() if n in pos_global}

        missing = [n for n in h_graph.nodes() if n not in pos]
        if missing:
            pos_local = nx.spring_layout(
                h_graph, seed=layout_seed, k=layout_k, iterations=20
            )
            for n in missing:
                pos[n] = pos_local[n]

        widths, alphas = edge_style(h_graph, weight_attr=weight_attr)

        plt.figure(figsize=figsize, dpi=dpi)

        nx.draw_networkx_nodes(
            h_graph, pos,
            node_size=node_size,
            node_color=[c] * h_graph.number_of_nodes(),
            cmap=cmap,
        )

        for (u, v), lw, a in zip(h_graph.edges(), widths, alphas):
            nx.draw_networkx_edges(
                h_graph, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a)
            )

        label_nodes = [n for n in h_graph.nodes() if n in pos]
        if len(label_nodes) <= max_labels:
            labels = {n: n for n in label_nodes}
            nx.draw_networkx_labels(h_graph, pos, labels=labels, font_size=font_size)
        else:
            subset = label_nodes[:max_labels]
            labels = {n: n for n in subset}
            nx.draw_networkx_labels(h_graph, pos, labels=labels, font_size=font_size)

        plt.title(
            f"Community {c} — nodes={h_graph.number_of_nodes()} "
            f"edges={h_graph.number_of_edges()}",
            fontsize=16,
        )
        plt.axis("off")
        plt.tight_layout()

        if save:
            plot_file = os.path.join(comm_dir, f"{filename_prefix}_{c}.png")
            plt.savefig(plot_file, bbox_inches="tight")
            saved_files.append(plot_file)

        if show:
            plt.show()
        else:
            plt.close()

    return saved_files
