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
    For every community at a given resolution: write an analysis text file
    and (optionally) save a sub-graph PNG.
"""
from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


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

    Analysis is always written regardless of *enable_plotting*.

    Parameters
    ----------
    G : nx.Graph
        Full NetworkX graph (all streams).
    leiden_df : pd.DataFrame
        DataFrame with columns ``['stream', 'community']``.
    stream_table_dependency_df : pyspark.sql.DataFrame
        Full stream-to-table dependency Spark DataFrame.  Converted to pandas
        internally via ``.toPandas()``.
    merged_edges_df : pyspark.sql.DataFrame
        Merged, bidirectional edge-list Spark DataFrame.  Converted to pandas
        internally via ``.toPandas()``.
    complexity_scores_df : pyspark.sql.DataFrame
        Complexity scores Spark DataFrame.  Converted to pandas internally
        via ``.toPandas()``.
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

    stream_table_pdf = stream_table_dependency_df.toPandas()
    merged_edges_pdf = merged_edges_df.toPandas()
    complexity_pdf = complexity_scores_df.toPandas()

    stream_table_pdf['size'] = pd.to_numeric(stream_table_pdf['size'], errors='coerce').fillna(0.0)

    all_streams = set(leiden_df["stream"].tolist())
    total_streams_global = len(all_streams)
    bi_reports_global = [s for s in all_streams if "json" in s.lower()]
    etl_streams_global = [s for s in all_streams if "json" not in s.lower()]
    total_bi_reports_global = len(bi_reports_global)
    total_etl_streams_global = len(etl_streams_global)
    total_complexity_global = complexity_pdf['complexity_score'].sum()

    communities = sorted(leiden_df["community"].unique())
    saved_files = []

    for c in communities:
        comm_dir = os.path.join(outdir, f"community_{c}")
        os.makedirs(comm_dir, exist_ok=True)

        comm_nodes = [n for n in labeled_nodes if node_to_comm[n] == c]
        streams_in_community = set(comm_nodes)

        if enable_plotting:
            h_graph = G.subgraph(comm_nodes).copy()
            print(f"Community {c}: nodes={h_graph.number_of_nodes()}, edges={h_graph.number_of_edges()}")

        # ---- ANALYSIS (always runs regardless of plotting) ----

        streams_list = sorted(list(streams_in_community))
        bi_reports = [s for s in streams_list if "json" in s.lower()]
        etl_streams = [s for s in streams_list if "json" not in s.lower()]

        pct_total = (len(streams_list) / total_streams_global * 100) if total_streams_global > 0 else 0.0
        pct_bi = (len(bi_reports) / total_bi_reports_global * 100) if total_bi_reports_global > 0 else 0.0
        pct_etl = (len(etl_streams) / total_etl_streams_global * 100) if total_etl_streams_global > 0 else 0.0

        community_complexity_df = complexity_pdf[
            complexity_pdf['stream_name'].isin(streams_in_community)
        ].copy()
        stream_complexity_map = dict(
            zip(community_complexity_df['stream_name'], community_complexity_df['complexity_score'])
        )
        total_community_complexity = community_complexity_df['complexity_score'].sum()
        pct_complexity = (
            (total_community_complexity / total_complexity_global * 100)
            if total_complexity_global > 0
            else 0.0
        )
        bi_complexity_df = community_complexity_df[
            community_complexity_df['stream_name'].isin(bi_reports)
        ]
        etl_complexity_df = community_complexity_df[
            community_complexity_df['stream_name'].isin(etl_streams)
        ]
        total_bi_complexity = bi_complexity_df['complexity_score'].sum()
        total_etl_complexity = etl_complexity_df['complexity_score'].sum()

        tables_src_inside_tgt_outside_all = stream_table_pdf[
            (~stream_table_pdf['from'].isin(streams_in_community))
            & (stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].sort_values('table')
        tables_src_inside_tgt_outside_unique = tables_src_inside_tgt_outside_all.drop_duplicates(
            subset=['table']
        )

        tables_incoming_details = tables_src_inside_tgt_outside_all.copy()
        tables_incoming_details['is_bi'] = (
            tables_incoming_details['to'].str.lower().str.contains('json')
        )
        tables_incoming_details['is_etl'] = ~tables_incoming_details['is_bi']

        table_classification = tables_incoming_details.groupby('table').agg(
            {'is_bi': 'any', 'is_etl': 'any', 'size': 'first'}
        ).reset_index()
        table_classification['category'] = table_classification.apply(
            lambda row: 'ETL' if row['is_etl'] else 'BI', axis=1
        )
        tables_incoming_bi = table_classification[table_classification['category'] == 'BI']
        tables_incoming_etl = table_classification[table_classification['category'] == 'ETL']
        total_size_incoming_bi = tables_incoming_bi['size'].sum() if len(tables_incoming_bi) > 0 else 0.0
        total_size_incoming_etl = (
            tables_incoming_etl['size'].sum() if len(tables_incoming_etl) > 0 else 0.0
        )
        total_size_incoming = total_size_incoming_bi + total_size_incoming_etl

        tables_tgt_inside_src_outside_all = stream_table_pdf[
            (stream_table_pdf['from'].isin(streams_in_community))
            & (~stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].sort_values('table')
        tables_tgt_inside_src_outside_unique = tables_tgt_inside_src_outside_all.drop_duplicates(
            subset=['table']
        )
        total_size_outgoing = (
            tables_tgt_inside_src_outside_unique['size'].sum()
            if len(tables_tgt_inside_src_outside_unique) > 0
            else 0.0
        )

        outgoing_edges = merged_edges_pdf[
            (merged_edges_pdf['streamA'].isin(streams_in_community))
            & (~merged_edges_pdf['streamB'].isin(streams_in_community))
        ]
        incoming_edges = merged_edges_pdf[
            (~merged_edges_pdf['streamA'].isin(streams_in_community))
            & (merged_edges_pdf['streamB'].isin(streams_in_community))
        ]

        analysis_content = f"""Community {c} Analysis (Resolution γ={resolution})
{'='*80}

1. STREAMS IN COMMUNITY ({len(streams_list)} total streams, {pct_total:.2f}% of all streams):
{'-'*80}

1a. BI REPORTS/STREAMS ({len(bi_reports)} streams with 'json' in name, {pct_bi:.2f}% of all BI reports):
{'-'*80}
"""
        if len(bi_reports) > 0:
            for i, stream in enumerate(bi_reports, 1):
                complexity = stream_complexity_map.get(stream, 0.0)
                analysis_content += f"{i}. {stream} (complexity: {complexity:.2f})\n"
        else:
            analysis_content += "  (No BI reports/streams found)\n"

        analysis_content += (
            f"\n1b. ETL STREAMS ({len(etl_streams)} streams without 'json' in name, "
            f"{pct_etl:.2f}% of all ETL streams):\n{'-'*80}\n"
        )
        if len(etl_streams) > 0:
            for i, stream in enumerate(etl_streams, 1):
                complexity = stream_complexity_map.get(stream, 0.0)
                analysis_content += f"{i}. {stream} (complexity: {complexity:.2f})\n"
        else:
            analysis_content += "  (No ETL streams found)\n"

        analysis_content += f"""
2. COMPLEXITY ANALYSIS:
{'-'*80}
Total Community Complexity: {total_community_complexity:.2f}
Percentage of Total Complexity: {pct_complexity:.2f}%
  - BI Reports Complexity: {total_bi_complexity:.2f}
  - ETL Streams Complexity: {total_etl_complexity:.2f}
"""

        analysis_content += f"""
3. TABLES - SRC OF STREAMS INSIDE, TGT OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community READ FROM, but are WRITTEN BY streams outside.
(Dependencies flowing INTO the community - SYNC REQUIREMENTS)
Total: {len(tables_src_inside_tgt_outside_unique)} unique tables, {total_size_incoming:.2f} GB
  - For BI Reports: {len(tables_incoming_bi)} tables, {total_size_incoming_bi:.2f} GB
  - For ETL Streams: {len(tables_incoming_etl)} tables, {total_size_incoming_etl:.2f} GB
Total Instances: {len(tables_src_inside_tgt_outside_all)}\n\n"""

        if len(tables_src_inside_tgt_outside_all) > 0:
            if len(tables_incoming_bi) > 0:
                analysis_content += (
                    f"\n3a. FOR BI REPORTS ({len(tables_incoming_bi)} tables, "
                    f"{total_size_incoming_bi:.2f} GB):\n{'-'*80}\n"
                )
                for table in tables_incoming_bi['table']:
                    table_instances = tables_incoming_details[
                        tables_incoming_details['table'] == table
                    ]
                    size = table_instances['size'].iloc[0]
                    producers = sorted(table_instances['from'].unique())
                    consumers = sorted(table_instances['to'].unique())
                    analysis_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                    analysis_content += f"    - Written by (outside): {', '.join(producers)}\n"
                    analysis_content += f"    - Read by (inside): {', '.join(consumers)}\n"
            if len(tables_incoming_etl) > 0:
                analysis_content += (
                    f"\n3b. FOR ETL STREAMS ({len(tables_incoming_etl)} tables, "
                    f"{total_size_incoming_etl:.2f} GB):\n{'-'*80}\n"
                )
                for table in tables_incoming_etl['table']:
                    table_instances = tables_incoming_details[
                        tables_incoming_details['table'] == table
                    ]
                    size = table_instances['size'].iloc[0]
                    producers = sorted(table_instances['from'].unique())
                    consumers = sorted(table_instances['to'].unique())
                    analysis_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                    analysis_content += f"    - Written by (outside): {', '.join(producers)}\n"
                    analysis_content += f"    - Read by (inside): {', '.join(consumers)}\n"
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""
4. TABLES - TGT OF STREAMS INSIDE, SRC OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community WRITE TO, but are READ BY streams outside.
(Dependencies flowing OUT OF the community)
Total: {len(tables_tgt_inside_src_outside_unique)} unique tables
Total Size: {total_size_outgoing:.2f} GB
Total Instances: {len(tables_tgt_inside_src_outside_all)}\n\n"""

        if len(tables_tgt_inside_src_outside_all) > 0:
            for table in sorted(tables_tgt_inside_src_outside_unique['table'].unique()):
                table_instances = tables_tgt_inside_src_outside_all[
                    tables_tgt_inside_src_outside_all['table'] == table
                ]
                size = table_instances['size'].iloc[0]
                producers = sorted(table_instances['from'].unique())
                consumers = sorted(table_instances['to'].unique())
                analysis_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                analysis_content += f"    - Written by (inside): {', '.join(producers)}\n"
                analysis_content += f"    - Read by (outside): {', '.join(consumers)}\n"
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""
5. AGGREGATED STREAM CONNECTIONS:
{'-'*80}

5a. Outgoing Stream Connections (Inside → Outside):
"""
        if len(outgoing_edges) > 0:
            analysis_content += f"Total: {len(outgoing_edges)} connections\n\n"
            for _, row in outgoing_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No outgoing connections)\n"

        analysis_content += "\n5b. Incoming Stream Connections (Outside → Inside):\n"
        if len(incoming_edges) > 0:
            analysis_content += f"Total: {len(incoming_edges)} connections\n\n"
            for _, row in incoming_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No incoming connections)\n"

        analysis_content += f"""
6. SUMMARY:
{'-'*80}
  - Total streams in community: {len(streams_list)} ({pct_total:.2f}% of {total_streams_global} total streams)
    * BI Reports/Streams (with 'json'): {len(bi_reports)} ({pct_bi:.2f}% of {total_bi_reports_global} total BI reports)
    * ETL Streams (without 'json'): {len(etl_streams)} ({pct_etl:.2f}% of {total_etl_streams_global} total ETL streams)
  - Total Community Complexity: {total_community_complexity:.2f} ({pct_complexity:.2f}% of total complexity)
    * BI Reports Complexity: {total_bi_complexity:.2f}
    * ETL Streams Complexity: {total_etl_complexity:.2f}
  - Tables flowing INTO community (SYNC REQUIREMENTS): {len(tables_src_inside_tgt_outside_unique)} unique tables, {total_size_incoming:.2f} GB
    * For BI Reports: {len(tables_incoming_bi)} tables, {total_size_incoming_bi:.2f} GB
    * For ETL Streams: {len(tables_incoming_etl)} tables, {total_size_incoming_etl:.2f} GB
    * Total instances: {len(tables_src_inside_tgt_outside_all)}
  - Tables flowing OUT OF community: {len(tables_tgt_inside_src_outside_unique)} unique tables, {total_size_outgoing:.2f} GB ({len(tables_tgt_inside_src_outside_all)} total instances)
  - Aggregated outgoing stream connections: {len(outgoing_edges)}
  - Aggregated incoming stream connections: {len(incoming_edges)}
"""

        analysis_file = os.path.join(comm_dir, f"community_{c}_analysis.txt")
        with open(analysis_file, 'w') as f:
            f.write(analysis_content)
        print(f"  Saved analysis: {analysis_file}")
        saved_files.append(analysis_file)

        # ---- PLOTTING (only if enabled) ----
        if enable_plotting:
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
        else:
            print(f"  Skipping plot generation (enable_plotting=False)")

    return saved_files
