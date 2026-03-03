# Databricks notebook source
# MAGIC %md
# MAGIC ## Libraries and Parameters
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install required libraries

!pip install python-igraph adjustText infomap netgraph networkx python-louvain igraph leidenalg
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Library imports
import pandas as pd

import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs
import igraph as ig
import leidenalg as la
import random
from datetime import datetime
import os
import numpy as np

%matplotlib inline
from networkx.algorithms.centrality import degree_centrality
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from netgraph import Graph
from infomap import Infomap
from pyspark.sql.functions import col, lit, when, upper, lower
from sklearn.metrics import adjusted_rand_score
from collections import Counter
from pyspark.sql.functions import ceil, sum, when, array, sort_array

# COMMAND ----------

# DBTITLE 1,Load configuration
from migration_planner.utils.config import load_config
from migration_planner.dependency_extractors.loaders import (
    load_stream_table_dependencies,
    load_outofscope_streams,
    load_complexity_scores,
    load_report_dependencies,
    load_table_sizes,
    load_static_tables,
)
from migration_planner.planner_core.preprocessing import (
    preprocess_stream_dependencies,
)
from migration_planner.planner_core.weights import (
    WEIGHT_METHOD_FACTOR,
    WEIGHT_METHOD_SCALED,
)
from migration_planner.community_detector.graph_builder import (
    find_isolated_streams,
    build_igraph,
    build_networkx_graph,
)
from migration_planner.community_detector.algorithm import (
    run_leiden_rb,
    stability_ari,
    scan_resolutions,
)
from migration_planner.visualization.community_plots import (
    select_resolutions,
    precompute_layout,
    precompute_edge_style,
    plot_leiden_resolutions,
    edge_style,
    plot_communities_with_analysis_safe,
)
cfg = load_config()

volume_path            = cfg.volume_name
dependency_input_path  = cfg.dependency_input_path
outofscope_stream_path = cfg.outofscope_stream_path
report_dependency      = cfg.report_dependency_path
table_size             = cfg.table_size_path
complexity_path        = cfg.complexity_path
static_tables_path     = cfg.static_tables_path
output_path            = cfg.output_path
latest_path            = cfg.latest_path
weight_method          = cfg.weight_method  # "factor" (default) or "scaled"

# COMMAND ----------

# DBTITLE 1,Output directory operation
# Move existing folders from 'latest' one level up (to volume_path)
folders = [f.name for f in dbutils.fs.ls(latest_path) if f.isDir()]
for folder in folders:
    dbutils.fs.mv(latest_path + folder, volume_path + folder, recurse=True)

# Create the new output directory under 'latest'
dbutils.fs.mkdirs(output_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Datasets:

# COMMAND ----------

# DBTITLE 1,Reading input stream - table dependency file
dependency_df_full = load_stream_table_dependencies(spark, dependency_input_path)

# COMMAND ----------

# DBTITLE 1,Reading out of scope stream names
outofscope_stream_names_list = (
    load_outofscope_streams(spark, outofscope_stream_path)
    if outofscope_stream_path is not None else []
)

# COMMAND ----------

# DBTITLE 1,Read complexity by stream
complexity_scores_df = load_complexity_scores(spark, complexity_path)
total_complexity = complexity_scores_df.agg({"complexity_score": "sum"}).collect()[0][0]
print(f"Total complexity score across all streams: {total_complexity}")
print("\nTop 10 streams by complexity score:")
display(complexity_scores_df.orderBy(col("complexity_score").desc()).limit(10))

# COMMAND ----------

# DBTITLE 1,Reading report to table dependency
report_dependency_df = (
    load_report_dependencies(spark, report_dependency)
    if report_dependency is not None else None
)

# COMMAND ----------

# DBTITLE 1,Reading table size in GB records
table_size_df = (
    load_table_sizes(spark, table_size)
    if table_size is not None else None
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing: Filter → Weight → Merge

# COMMAND ----------

# DBTITLE 1,Run full preprocessing pipeline
# Filters admin/out-of-scope streams, normalises TGT-as-SRC, forms cross-stream
# dependencies, applies factor-based weight calculation (WEIGHT_METHOD), and
# merges bidirectional edges into a single undirected weighted edge list.
dependency_df, merged_dependency_df = preprocess_stream_dependencies(
    dependency_df_full,
    outofscope_stream_names_list,
    report_dependency_df,
    table_size_df,
    weight_method=weight_method,
)
print(f"Weight method: {weight_method}")
print(f"Total edges after merging: {merged_dependency_df.count()}")
display(merged_dependency_df)

# COMMAND ----------

# DBTITLE 1,Create Edges of the graph in Pandas
edges_df = merged_dependency_df.toPandas()
edges_df.to_csv(f"{output_path}edges.csv", index=False)

# COMMAND ----------

# DBTITLE 1,Identify isolated streams (only intra-stream dependencies)
isolated_streams = find_isolated_streams(dependency_df, merged_dependency_df)
print(f"Isolated streams (only intra-stream dependencies): {len(isolated_streams)}")

if len(isolated_streams) > 0:
    print(f"\nFirst 10 isolated streams: {isolated_streams[:10]}")
    isolated_streams_df = pd.DataFrame({'stream_name': isolated_streams})
    isolated_streams_df.to_csv(f"{output_path}isolated_streams.csv", index=False)
    print(f"\nIsolated streams saved to: {output_path}isolated_streams.csv")
else:
    print("\nNo isolated streams found - all streams have inter-stream dependencies.")

# COMMAND ----------

# DBTITLE 1,Aggregate weights - safety step
edges = edges_df.groupby(['streamA', 'streamB'])['weight'].sum().reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC # Leiden Algorithm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Algorithm used: Leiden + RBConfiguration (resolution γ)
# MAGIC ##
# MAGIC We're running the Leiden algorithm, which is a community detection method that searches for a partition of the graph into communities by optimizing an objective function.
# MAGIC
# MAGIC In this case the objective is RBConfiguration (RBConfigurationVertexPartition), which is a modularity-like objective that includes a resolution parameter γ (called resolution_parameter):
# MAGIC
# MAGIC - Lower γ → fewer, larger communities (coarser clustering)
# MAGIC - Higher γ → more, smaller communities (finer clustering)
# MAGIC
# MAGIC Leiden is stochastic (it can produce different local optima depending on random initialization and tie-breaking), so we test stability by re-running with many seeds.

# COMMAND ----------

# DBTITLE 1,Community Detection
g = build_igraph(edges, isolated_streams)
G = build_networkx_graph(edges, isolated_streams)
igraph_names = np.array(g.vs["name"])
print(f"\nigraph summary: {g.summary()}")
print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# COMMAND ----------

# DBTITLE 1,Resolution scan
resolutions = [0.4, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.2, 2.4, 2.6]
seeds = [1, 3, 7, 43, 99, 123, 11, 28, 37, 45, 672, 42, 10, 100, 178]
plot_seed = 42
summary, rep_by_res = scan_resolutions(g, resolutions, seeds, plot_seed)

# Final summary table
summary



# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretation of Leiden (RBConfiguration) Stability Results
# MAGIC
# MAGIC The above table summarizes how the Leiden algorithm behaves across different resolution (γ) values, using multiple random seeds per resolution. Each row describes the **solution landscape** at a fixed γ.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What the columns tell us 
# MAGIC - **n_communities_avg / min / max**  
# MAGIC   Number of communities found across seeds.  
# MAGIC   Equality of min = max indicates a fixed scale; divergence indicates multiple competing solutions.
# MAGIC
# MAGIC - **largest_comm_share_avg**  
# MAGIC   Fraction of nodes in the largest community (detects overly coarse “giant blob” solutions).
# MAGIC
# MAGIC - **small_comms_lt5_avg**  
# MAGIC   Average number of very small communities (<5 nodes), indicating fragmentation.
# MAGIC
# MAGIC - **quality_avg**  
# MAGIC   Average value of the RBConfiguration objective (comparable only *within* a given γ).
# MAGIC
# MAGIC - **stability_ari (Adjusted Rand Index)**  
# MAGIC   Measures node-level consistency across seeds (1.0 = identical partitions).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Resolution regimes and their meaning
# MAGIC
# MAGIC ### **Low resolution: γ = 0.2–0.4**
# MAGIC - **3–5 communities**, always identical in count (`min = max`)
# MAGIC - **ARI ≈ 1.0** (or very close)
# MAGIC - Very large dominant community at γ = 0.2 (≈98% of nodes)
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC The graph supports a very coarse, highly stable structure. At γ = 0.2 the solution is almost trivial (one giant cluster plus a few small ones). γ = 0.3–0.4 reveals more structure while remaining extremely stable.
# MAGIC But due to high number of nodes in each community these are not practical
# MAGIC ---
# MAGIC
# MAGIC ### **Moderate resolution: γ = 0.5–1.0**
# MAGIC - **5–9 communities**, perfectly consistent across seeds
# MAGIC - **ARI = 1.0** at all these resolutions
# MAGIC - Largest community shrinks substantially (≈26–40%)
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC This is the **most stable and interpretable regime**. The solution landscape is unimodal:
# MAGIC - every run converges to the same partition,
# MAGIC - node assignments are identical across seeds,
# MAGIC - communities are neither too coarse nor excessively fragmented.
# MAGIC
# MAGIC This range represents the strongest evidence for a “natural” community structure in the graph. Yet due to high number of nodes in each community these are not practical
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Degenerate regime: γ = 1.2 - 1.8**
# MAGIC - **10–12 communities** across seeds
# MAGIC - **ARI > 0.92**
# MAGIC - Low fragmentation, but multiple distinct partitions
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC The solution landscape becomes **degenerate**:
# MAGIC - several competing near-optimal partitions exist,
# MAGIC - differences are no longer limited to a few nodes,
# MAGIC - no single partition can be considered uniquely “correct.”
# MAGIC
# MAGIC At this resolution, results should be treated as *one of several plausible clusterings*.
# MAGIC Due to manageble number of nodes in each community and acceptable stability, this could be the most practical clustering for migration.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC ### **Very high resolution: γ = 2.2–3.0**
# MAGIC - **24–37 communities**
# MAGIC - **ARI ≈ 0.89–0.97**
# MAGIC - Rapid growth in tiny communities (<5 nodes)
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC The partitions are mostly stable but increasingly fragmented. While node assignments are fairly consistent, interpretability declines due to many very small clusters. These resolutions may be useful for exploratory or hierarchical analysis, but not for a single clean partition.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Plots

# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the bigger clustered graph for the chosen resolutions where each node is colored in the color of the community it has been clustered into. 
# MAGIC Results are saved under the volume path/leiden_plots

# COMMAND ----------

# DBTITLE 1,Overall graph plot
# Visualization functions (select_resolutions, precompute_layout,
# precompute_edge_style, plot_leiden_resolutions) have been moved to
# migration_planner.visualization.community_plots and are imported at the top of this
# notebook.  To generate overall resolution graphs, call them as shown:
#
#   selected, selected_resolutions = select_resolutions(
#       summary, ari_target=0.93, max_largest_comm_share=0.25
#   )
#   pos          = precompute_layout(G, seed=42, k=3.0)
#   edge_widths, edge_alphas = precompute_edge_style(G)
#   output_dir   = os.path.join(output_path, "leiden_plots")
#   plot_leiden_resolutions(
#       G=G, g_igraph=g,
#       selected_resolutions=selected_resolutions,
#       rep_by_res=rep_by_res,
#       membership_to_leiden_df=membership_to_leiden_df,
#       pos=pos, edge_widths=edge_widths, edge_alphas=edge_alphas,
#       outdir=output_dir, figsize=(50, 40), dpi=400,
#       save=True, show=False, use_adjust_text=True,
#   )


# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the subgraphs / clusters, by choosing the preferred resolution**

# COMMAND ----------

# DBTITLE 1,Choosing resolutions to progress with
resolutions = [1.8]

# COMMAND ----------

# DBTITLE 1,Merge single-stream communities for all resolutions
# Define helper function to convert membership to DataFrame
def membership_to_leiden_df(membership, igraph_names):
    return pd.DataFrame({
        "stream": np.array(igraph_names),
        "community": np.array(membership, dtype=int),
    })

# Choose which solution to visualize
def get_leiden_df(resolution, rep_by_res, igraph_names):
    rep = rep_by_res[resolution]
    leiden_df = membership_to_leiden_df(rep["membership"], igraph_names)
    meta = {
        "resolution": float(rep.get("resolution", resolution)),
        "seed": int(rep["seed"]),
        "quality": float(rep.get("quality", np.nan)),
    }
    return leiden_df, meta

# Merge single-stream communities into one consolidated community
# This operation is performed for all chosen resolutions before analysis

print("=" * 100)
print("MERGING SINGLE-STREAM COMMUNITIES")
print("=" * 100)

for resolution in resolutions:
    print(f"\n=== Processing resolution: {resolution} ===")
    
    # Get the leiden dataframe for this resolution
    leiden_df, meta = get_leiden_df(resolution, rep_by_res, igraph_names)
    
    # Get number of communities BEFORE merging
    num_communities_before = leiden_df['community'].nunique()
    print(f"Number of communities (before merging): {num_communities_before}")
    
    # Identify communities with only one stream
    community_stream_counts = leiden_df.groupby('community').size()
    single_stream_communities = community_stream_counts[community_stream_counts == 1].index.tolist()
    
    if len(single_stream_communities) > 1:
        # Get the smallest community ID from single-stream communities
        merged_community_id = min(single_stream_communities)
        
        print(f"\n--- Merging {len(single_stream_communities)} single-stream communities into community {merged_community_id} ---")
        print(f"Single-stream communities to merge: {sorted(single_stream_communities)}")
        
        # Reassign all single-stream communities to the merged community ID
        leiden_df.loc[leiden_df['community'].isin(single_stream_communities), 'community'] = merged_community_id
        
        # Update the rep_by_res dictionary with the modified membership
        # Create a mapping from stream name to new community ID
        stream_to_community = leiden_df.set_index('stream')['community'].to_dict()
        
        # Update the membership array in rep_by_res
        # The membership array corresponds to igraph_names order
        updated_membership = [stream_to_community[stream] for stream in igraph_names]
        rep_by_res[resolution]["membership"] = updated_membership
        
        # Verify the merge
        num_communities_after = leiden_df['community'].nunique()
        merged_community_size = len(leiden_df[leiden_df['community'] == merged_community_id])
        print(f"Merged community {merged_community_id} now contains {merged_community_size} streams")
        print(f"Number of communities (after merging): {num_communities_after}")
        print(f"Reduction: {num_communities_before - num_communities_after} communities merged")
        
    elif len(single_stream_communities) == 1:
        print(f"\n--- Only 1 single-stream community found (community {single_stream_communities[0]}), no merging needed ---")
    else:
        print(f"\n--- No single-stream communities found, no merging needed ---")

print(f"\n{'=' * 100}")
print("MERGING COMPLETE FOR ALL RESOLUTIONS")
print(f"{'=' * 100}\n")

# COMMAND ----------

# DBTITLE 1,Analysis & Plotting
# edge_style and plot_communities_with_analysis_safe are imported from
# migration_planner.visualization.community_plots


# ---- Calling analysis and plotting function  ----
for res in resolutions:
    leiden_df, meta = get_leiden_df(res, rep_by_res, igraph_names)
    print(meta)

    saved = plot_communities_with_analysis_safe(
        G=G,
        leiden_df=leiden_df,
        stream_table_dependency_df=stream_stream_dependency_df,
        merged_edges_df=merged_dependency_df,
        complexity_scores_df=complexity_scores_df,
        resolution=res,
        outdir=f"{output_path}leiden_community_plots_gamma_{res}",
        layout_seed=42,
        layout_k=None,
        layout_iterations=30,
        figsize=(24, 18),
        dpi=220,
        node_size=220,
        font_size=8,
        show=False,
        save=True,
        filename_prefix=f"gamma_{res}_comm",
        max_labels=120,
        enable_plotting=False,
    )
    print(f"Saved {len(saved)} files")

# COMMAND ----------

# MAGIC %md
# MAGIC # Community Ordering Optimization
# MAGIC
# MAGIC This module determines an execution order for communities at a fixed Leiden resolution to **minimize total data synchronization effort** between communities.
# MAGIC
# MAGIC The goal is to ensure that, as communities are processed sequentially, the amount of external data that must be synchronized into each community is as small as possible.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Order communities so that the **cumulative synchronization cost** across all communities is minimized.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Sync Cost Definition
# MAGIC
# MAGIC For a given ordering of communities:
# MAGIC
# MAGIC - For each community in sequence, identify the set of tables that **streams inside the community read from** but are **written by streams outside**.
# MAGIC - Tables already produced by previously ordered communities are considered **already available** and do **not** incur any cost.
# MAGIC - The **sync cost for a community** is defined as:
# MAGIC
# MAGIC   $$
# MAGIC   \mathrm{SyncCost(community)} = \sum_{t \in \mathrm{TablesToSync}} \mathrm{weight}(t)
# MAGIC   $$
# MAGIC
# MAGIC - The **total cost of an ordering** is the sum of sync costs over all communities in that order.
# MAGIC
# MAGIC This formulation naturally penalizes both:
# MAGIC - **Many tables** (large cardinality)
# MAGIC - **Large tables** (high data volume)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Optimization Strategy
# MAGIC
# MAGIC Because the search space grows factorially, the optimization is performed in two independent phases:
# MAGIC
# MAGIC 1. **Brute-force optimization of the N largest communities**, guaranteeing an optimal ordering for the most impactful portion of the graph.
# MAGIC 2. **Brute-force optimization of the remaining communities**.
# MAGIC
# MAGIC The final execution order is obtained by concatenating both optimal orderings:
# MAGIC
# MAGIC   $$
# MAGIC   \mathrm{FinalOrder} = \mathrm{Order}_{\mathrm{Rest}} \;\Vert\; \mathrm{Order}_{\mathrm{TopN}}
# MAGIC   $$
# MAGIC
# MAGIC This split preserves optimality where it matters most while keeping the computation tractable.

# COMMAND ----------

# DBTITLE 1,Split communities by total weight 
def split_communities_topN(leiden_df, top_n=10):
    """
    Split communities into top N heaviest and the rest based on total incoming table weight
    (sum of table weights for tables read by streams in the community, but written by streams outside).
    
    Parameters
    ----------
    leiden_df : pd.DataFrame
        DataFrame with columns ['stream', 'community']
    top_n : int
        Number of top communities to select (default 9)
    
    Returns
    -------
    top_n_ids : list
        Community IDs for the top N heaviest communities
    rest_ids : list
        Community IDs for the remaining communities
    community_weights : dict
        Dictionary mapping community ID to total incoming table weight
    """
    from pyspark.sql.functions import col, sum as spark_sum, broadcast
    
    print(f"Splitting communities into top {top_n} by incoming table weight...")

    # Convert leiden_df to Spark DataFrame for distributed join
    leiden_spark = spark.createDataFrame(leiden_df)
    
    # Broadcast the small leiden mapping for efficient joins
    leiden_broadcast = broadcast(leiden_spark)
    
    # Join unique_table_weights with community assignments for both 'from' and 'to' streams
    # This adds community labels to each edge
    edges_with_communities = (
        unique_table_weights
        .select("from", "to", "table", "weight")
        .join(
            leiden_broadcast.selectExpr("stream as from", "community as from_community"),
            on="from",
            how="inner"
        )
        .join(
            leiden_broadcast.selectExpr("stream as to", "community as to_community"),
            on="to",
            how="inner"
        )
    )
    
    # Filter for incoming edges: tables read by streams in a community but written by streams outside
    # (from_community != to_community)
    incoming_edges = edges_with_communities.filter(
        col("from_community") != col("to_community")
    )
    
    # Aggregate: sum weight by to_community (the community receiving the data)
    community_weights_spark = (
        incoming_edges
        .groupBy("to_community")
        .agg(spark_sum("weight").alias("total_incoming_weight"))
        .orderBy(col("total_incoming_weight").desc())
    )
    
    # Convert only the small aggregated result to pandas
    community_weights_pdf = community_weights_spark.toPandas()
    
    # Create dictionary mapping community -> weight
    community_weights = dict(
        zip(community_weights_pdf['to_community'], community_weights_pdf['total_incoming_weight'])
    )
    
    # Handle communities with no incoming edges (weight = 0)
    all_communities = set(leiden_df['community'].unique())
    for comm in all_communities:
        if comm not in community_weights:
            community_weights[comm] = 0.0
    
    # Sort communities by weight (descending)
    sorted_comms = sorted(community_weights.items(), key=lambda x: x[1], reverse=True)
    top_n_actual = min(top_n, len(sorted_comms))
    top_n_ids = [comm_id for comm_id, _ in sorted_comms[:top_n_actual]]
    rest_ids = [comm_id for comm_id, _ in sorted_comms[top_n_actual:]]

    print(f"\n=== Community Split (top_n={top_n}) ===")
    print(f"Total communities: {len(sorted_comms)}")
    print(f"Top {top_n_actual} communities: {top_n_ids}")
    print(f"Remaining communities: {len(rest_ids)}")
    print(f"\nTop {top_n_actual} community incoming weights:")
    for comm_id, weight in sorted_comms[:top_n_actual]:
        num_streams = len(leiden_df[leiden_df['community'] == comm_id])
        print(f"  Community {comm_id}: incoming_weight={weight:.2f}, streams={num_streams}")

    print(f"\nCommunities split into top {top_n_actual} communities: {top_n_ids} and {len(rest_ids)} others")
    return top_n_ids, rest_ids, community_weights

# COMMAND ----------

# DBTITLE 1,BruteForceCommunityOrdering with pre-available tables support
import itertools
import time
from math import factorial
import builtins
import pandas as pd
import numpy as np


class BruteForceCommunityOrdering:
    """
    Brute force search over all community orderings (for a given subset of communities).

    Cost = SUM of per-table 'weight' that must be synced at each step:
      to_sync(step) = incoming_tables[community] - available_tables_so_far
      step_cost     = sum(table_weight[table] for table in to_sync)
      total_cost    = sum(step_cost)
      
    Supports pre-available tables: when optimizing a subset of communities,
    you can specify other communities whose produced tables should be considered
    already available (e.g., when optimizing top N after rest communities).
    """

    def __init__(self, stream_table_dependency_df_scaled, leiden_df, communities_subset=None, 
                 pre_available_communities=None):
        """
        Parameters
        ----------
        stream_table_dependency_df_scaled : Spark DataFrame
            DataFrame with columns: from, to, table, weight
        leiden_df : pd.DataFrame
            DataFrame with columns: stream, community
        communities_subset : list or None
            List of community IDs to optimize. If None, uses all communities.
        pre_available_communities : list or None
            List of community IDs whose produced tables should be considered
            already available at the start of optimization. Useful when optimizing
            top N communities after rest communities have been ordered.
        """

        self.dep = (
            stream_table_dependency_df_scaled
            .select("from", "to", "table", "weight")
            .toPandas()
            .copy()
        )

        self.dep["from"] = self.dep["from"].astype(str)
        self.dep["to"] = self.dep["to"].astype(str)
        self.dep["table"] = self.dep["table"].astype(str)
        self.dep["weight"] = pd.to_numeric(self.dep["weight"], errors="coerce").fillna(0.0)

        # table → single weight (max)
        self.table_weight = (
            self.dep.groupby("table")["weight"].max().to_dict()
        )

        # choose communities to optimize
        all_comms = sorted(leiden_df["community"].unique().astype(int).tolist())
        if communities_subset is None:
            comms = all_comms
        else:
            comms = sorted([int(c) for c in communities_subset])

        # Build community_streams for ALL communities (needed for pre-available calculation)
        all_community_streams = {
            int(c): set(leiden_df.loc[leiden_df["community"] == c, "stream"].astype(str).tolist())
            for c in all_comms
        }
        
        # Store only the subset we're optimizing
        self.community_streams = {c: all_community_streams[c] for c in comms}

        # produced tables per community (for communities we're optimizing)
        self.produced_tables = {}
        for c, streams in self.community_streams.items():
            self.produced_tables[c] = set(
                self.dep[self.dep["from"].isin(streams)]["table"].unique()
            )

        # incoming tables per community (for communities we're optimizing)
        self.incoming_tables = {}
        for c, streams in self.community_streams.items():
            incoming = self.dep[
                (~self.dep["from"].isin(streams)) &
                (self.dep["to"].isin(streams))
            ]["table"].unique()
            self.incoming_tables[c] = set(incoming)

        self.communities = list(self.community_streams.keys())
        
        # Calculate pre-available tables from other communities
        self.pre_available_tables = set()
        if pre_available_communities is not None:
            print(f"  Pre-available communities: {sorted(pre_available_communities)}")
            for c in pre_available_communities:
                if c in all_community_streams:
                    streams = all_community_streams[c]
                    tables = set(
                        self.dep[self.dep["from"].isin(streams)]["table"].unique()
                    )
                    self.pre_available_tables |= tables
            print(f"  Pre-available tables count: {len(self.pre_available_tables)}")

    def evaluate_ordering_cost(self, ordering, initial_available=None):
        """
        Evaluate the total sync cost for a given ordering.
        
        At each step, calculates the immediate sync cost for the current community:
        - Tables needed by this community that are not yet available
        
        The total cost is the sum of immediate sync costs across all steps.
        This minimizes the total amount of data that needs to be synced.
        
        Parameters
        ----------
        ordering : tuple or list
            Sequence of community IDs
        initial_available : set or None
            Set of tables that are already available at the start.
            If None, uses self.pre_available_tables.
        
        Returns
        -------
        total : float
            Total sync cost (sum of immediate sync costs at each step)
        step_costs : list
            Immediate sync cost at each step
        """
        if initial_available is None:
            available = self.pre_available_tables.copy()
        else:
            available = initial_available.copy()
            
        total = 0.0
        step_costs = []

        for c in ordering:
            # Calculate immediate sync cost for this community
            to_sync = self.incoming_tables[c] - available
            step = float(builtins.sum(self.table_weight.get(t, 0.0) for t in to_sync))
            total += step
            step_costs.append(step)
            
            # Update available tables with what this community produces
            available |= self.produced_tables[c]

        return total, step_costs

    def brute_force(self, log_every=5000, label="subset"):
        """
        Perform brute force search over all permutations.
        
        Parameters
        ----------
        log_every : int
            Print progress every N permutations
        label : str
            Label for logging
        
        Returns
        -------
        dict
            Results including best_cost, best_order, best_step_costs, total_time_sec, total_perms
        """
        n = len(self.communities)
        total_perms = factorial(n)

        best_cost = float("inf")
        best_order = None
        best_step_costs = None

        start = time.time()

        print(f"\n=== Brute force ({label}) | communities={n} | perms={total_perms} ===")
        print(f"  Optimization objective: Minimize total sync cost across all steps")
        if len(self.pre_available_tables) > 0:
            print(f"  Starting with {len(self.pre_available_tables)} pre-available tables")

        for i, perm in enumerate(itertools.permutations(self.communities), 1):

            cost, step_costs = self.evaluate_ordering_cost(perm)

            if cost < best_cost:
                best_cost = cost
                best_order = perm
                best_step_costs = step_costs
                print(f"[NEW BEST] {i}/{total_perms} cost={best_cost:.6f} order={list(best_order)}")

            if i % log_every == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = total_perms - i
                eta = remaining / rate if rate > 0 else float("inf")

                print(f"[PROGRESS] {i}/{total_perms} "
                      f"({100*i/total_perms:.2f}%) | "
                      f"best={best_cost:.6f} | "
                      f"{rate:.1f} perms/sec | "
                      f"elapsed={elapsed/60:.2f} min | "
                      f"eta={eta/60:.2f} min")

        total_time = time.time() - start
        print(f"\nDONE ({label}) in {total_time/60:.2f} min")
        print(f"BEST COST: {best_cost:.6f}")
        print(f"BEST ORDER: {list(best_order)}")

        return {
            "best_cost": float(best_cost),
            "best_order": list(best_order),
            "best_step_costs": best_step_costs,
            "total_time_sec": total_time,
            "total_perms": total_perms
        }

# COMMAND ----------

# DBTITLE 1,Execution metadata logging
def append_execution_metadata(weight_method, top_n, resolutions_dict):
    """
    Append execution metadata to a managed table in Unity Catalog.
    Creates the table if it doesn't exist.
    
    Parameters
    ----------
    weight_method : str
        Weight calculation method: "Factor based", "Min-max based", or "Log based"
    top_n : int
        Number of top communities used for ordering optimization
    resolutions_dict : dict
        Dictionary mapping resolution -> (total_sync_cost, num_communities)
        Example: {2.2: (1234.56, 15), 2.5: (987.65, 18)}
    
    Returns
    -------
    str
        Name of the metadata table
    """
    table_name = "odp_adw_mvp_n.migration.execution_metadata"
    
    # Current timestamp
    execution_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create rows for each resolution
    rows = []
    for resolution, (total_cost, num_communities) in sorted(resolutions_dict.items()):
        rows.append({
            'execution_datetime': execution_datetime,
            'weight_calculation_method': weight_method,
            'top_n': int(top_n),
            'resolution': float(resolution),
            'number_of_communities': int(num_communities),
            'total_sync_cost_gb': float(total_cost)
        })
    
    # Convert to Spark DataFrame
    metadata_df = spark.createDataFrame(rows)
    
    # Check if table exists
    table_exists = spark.catalog.tableExists(table_name)
    
    if table_exists:
        # Append to existing table
        metadata_df.write.mode("append").saveAsTable(table_name)
        print(f"Execution metadata appended to table: {table_name}")
    else:
        # Create new table
        metadata_df.write.mode("overwrite").saveAsTable(table_name)
        print(f"Execution metadata table created: {table_name}")
    
    print(f"Appended {len(rows)} rows to {table_name}")
    return table_name

# COMMAND ----------

# DBTITLE 1,Migration Order Analysis Function
def generate_migration_order_analysis(
    leiden_df,
    stream_table_dependency_df,
    merged_edges_df,
    optimized_order,
    resolution,
    complexity_scores_df,
    outdir="migration_order_analysis",
):
    """
    Generate detailed migration analysis report showing sync requirements
    for each community in the optimized migration order.
    
    Parameters
    ----------
    leiden_df : pd.DataFrame
        DataFrame with columns ['stream', 'community']
    stream_table_dependency_df : Spark DataFrame
        Stream-table dependencies
    merged_edges_df : Spark DataFrame
        Aggregated stream-stream edges
    optimized_order : list
        Ordered list of community IDs for migration
    resolution : float
        Leiden resolution parameter
    complexity_scores_df : Spark DataFrame
        DataFrame with columns ['stream_name', 'complexity_score']
    outdir : str
        Output directory for reports
    
    Returns
    -------
    tuple
        (report_file_path, total_sync_cost_gb)
        - report_file_path: Path to the generated report file
        - total_sync_cost_gb: Total cumulative sync cost in GB for the entire migration order
    """
    import os
    
    os.makedirs(outdir, exist_ok=True)
    
    # Convert to pandas
    stream_table_pdf = stream_table_dependency_df.toPandas()
    merged_edges_pdf = merged_edges_df.toPandas()
    complexity_pdf = complexity_scores_df.select('stream_name', 'complexity_score').toPandas()
    
    # Convert size to numeric
    stream_table_pdf['size'] = pd.to_numeric(stream_table_pdf['size'], errors='coerce').fillna(0.0)
    
    # Merge leiden_df with complexity scores
    leiden_with_complexity = leiden_df.merge(
        complexity_pdf,
        left_on='stream',
        right_on='stream_name',
        how='left'
    )
    leiden_with_complexity['complexity_score'] = leiden_with_complexity['complexity_score'].fillna(0)
    
    # Calculate global totals
    all_streams = set(leiden_df["stream"].tolist())
    total_streams_global = len(all_streams)
    bi_reports_global = [s for s in all_streams if "json" in s.lower()]
    etl_streams_global = [s for s in all_streams if "json" not in s.lower()]
    total_bi_reports_global = len(bi_reports_global)
    total_etl_streams_global = len(etl_streams_global)
    
    # Calculate total complexity across all communities
    total_complexity_global = leiden_with_complexity['complexity_score'].sum()
    
    # Build community -> streams mapping with complexity
    community_streams = {}
    community_complexity = {}
    for c in optimized_order:
        streams = leiden_df[leiden_df['community'] == c]['stream'].tolist()
        community_streams[c] = set(streams)
        # Calculate community complexity
        community_complexity[c] = leiden_with_complexity[
            leiden_with_complexity['community'] == c
        ]['complexity_score'].sum()
    
    # Track already migrated/available tables
    available_tables = set()
    migrated_streams = set()
    
    # Track cumulative outstanding sync requirements
    cumulative_outstanding_sync_tables = set()
    
    # Cumulative metrics
    cumulative_streams = 0
    cumulative_bi = 0
    cumulative_etl = 0
    cumulative_sync_size = 0.0
    cumulative_sync_size_bi = 0.0
    cumulative_sync_size_etl = 0.0
    cumulative_multi_producer_consumer_tables = 0
    cumulative_complexity = 0.0
    
    # List to collect sync details for CSV
    sync_details_rows = []
    
    # List to collect stream-to-community-to-execution_order mapping for CSV
    stream_ordering_rows = []
    
    # Generate report
    report_content = f"""{'='*100}
MIGRATION ORDER ANALYSIS REPORT
Resolution γ={resolution}
Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

OPTIMIZED MIGRATION ORDER: {optimized_order}

Total Communities: {len(optimized_order)}
Total Streams: {total_streams_global}
  - BI Reports: {total_bi_reports_global}
  - ETL Streams: {total_etl_streams_global}
Total Complexity Score: {total_complexity_global:.0f}

{'='*100}
"""
    
    for step, community_id in enumerate(optimized_order, 1):
        streams_in_community = community_streams[community_id]
        community_complexity_score = community_complexity[community_id]
        
        # Get stream-level complexity for this community
        stream_complexity_map = leiden_with_complexity[
            leiden_with_complexity['community'] == community_id
        ].set_index('stream')['complexity_score'].to_dict()
        
        # Add stream-to-community-to-execution_order mapping for CSV
        for stream in streams_in_community:
            stream_ordering_rows.append({
                'resolution': float(resolution),
                'stream_name': str(stream),
                'community_id': int(community_id),
                'execution_order': int(step)
            })
        
        # Separate BI and ETL
        bi_reports = [s for s in streams_in_community if "json" in s.lower()]
        etl_streams = [s for s in streams_in_community if "json" not in s.lower()]
        
        # Calculate percentages
        pct_total = (len(streams_in_community) / total_streams_global * 100) if total_streams_global > 0 else 0.0
        pct_bi = (len(bi_reports) / total_bi_reports_global * 100) if total_bi_reports_global > 0 else 0.0
        pct_etl = (len(etl_streams) / total_etl_streams_global * 100) if total_etl_streams_global > 0 else 0.0
        
        # Find tables this community needs (incoming dependencies)
        incoming_tables_all = stream_table_pdf[
            (~stream_table_pdf['from'].isin(streams_in_community)) & 
            (stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].sort_values('table')
        
        # Unique incoming tables
        incoming_tables_unique = incoming_tables_all.drop_duplicates(subset=['table'])
        
        # Tables that need to be synced (not yet available)
        tables_to_sync = incoming_tables_unique[~incoming_tables_unique['table'].isin(available_tables)]
        
        # Separate sync requirements by BI vs ETL
        # For each table to sync, check if it's needed by BI reports or ETL streams
        tables_to_sync_details = incoming_tables_all[incoming_tables_all['table'].isin(tables_to_sync['table'])].copy()
        
        # Collect sync details for CSV
        for _, row in tables_to_sync_details.iterrows():
            table_name = row['table']
            size_gb = row['size']
            producer_stream = row['from']
            consumer_stream = row['to']
            
            # Row for producer (written outside)
            sync_details_rows.append({
                'resolution': float(resolution),
                'execution_order': int(step),
                'community_id': int(community_id),
                'table_name': str(table_name),
                'size_gb': float(size_gb),
                'stream_name': str(producer_stream),
                'handling_type': 'written outside'
            })
            
            # Row for consumer (read inside)
            sync_details_rows.append({
                'resolution': float(resolution),
                'execution_order': int(step),
                'community_id': int(community_id),
                'table_name': str(table_name),
                'size_gb': float(size_gb),
                'stream_name': str(consumer_stream),
                'handling_type': 'read inside'
            })
        
        # Classify each table dependency as BI or ETL based on the 'to' stream
        tables_to_sync_details['is_bi'] = tables_to_sync_details['to'].str.lower().str.contains('json')
        tables_to_sync_details['is_etl'] = ~tables_to_sync_details['is_bi']
        
        # For each unique table, determine if it's needed by BI, ETL, or both
        table_classification = tables_to_sync_details.groupby('table').agg({
            'is_bi': 'any',
            'is_etl': 'any',
            'size': 'first'  # Size is same for all instances of a table
        }).reset_index()
        
        # If a table is needed by both, classify as ETL (per user requirement)
        table_classification['category'] = table_classification.apply(
            lambda row: 'ETL' if row['is_etl'] else 'BI',
            axis=1
        )
        
        # Calculate sync requirements by category
        tables_to_sync_bi = table_classification[table_classification['category'] == 'BI']
        tables_to_sync_etl = table_classification[table_classification['category'] == 'ETL']
        
        sync_size_bi = tables_to_sync_bi['size'].sum() if len(tables_to_sync_bi) > 0 else 0.0
        sync_size_etl = tables_to_sync_etl['size'].sum() if len(tables_to_sync_etl) > 0 else 0.0
        sync_size = sync_size_bi + sync_size_etl
        
        # Calculate tables with multiple producers AND multiple consumers
        multi_producer_consumer_tables = []
        multi_producer_consumer_size = 0.0
        
        for table in tables_to_sync['table']:
            table_instances = tables_to_sync_details[tables_to_sync_details['table'] == table]
            producers = table_instances['from'].unique()
            consumers = table_instances['to'].unique()
            
            if len(producers) > 1 and len(consumers) > 1:
                multi_producer_consumer_tables.append(table)
                multi_producer_consumer_size += table_instances['size'].iloc[0]
        
        num_multi_producer_consumer = len(multi_producer_consumer_tables)
        cumulative_multi_producer_consumer_tables += num_multi_producer_consumer
        
        # Add these tables to cumulative outstanding sync
        cumulative_outstanding_sync_tables.update(tables_to_sync['table'].tolist())
        
        # Tables already available (produced by previous communities)
        tables_already_available = incoming_tables_unique[incoming_tables_unique['table'].isin(available_tables)]
        available_size = tables_already_available['size'].sum() if len(tables_already_available) > 0 else 0.0
        
        # Tables this community produces (outgoing)
        outgoing_tables_all = stream_table_pdf[
            stream_table_pdf['from'].isin(streams_in_community) & 
            (~stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].sort_values('table')
        
        outgoing_tables_unique = outgoing_tables_all.drop_duplicates(subset=['table'])
        outgoing_size = outgoing_tables_unique['size'].sum() if len(outgoing_tables_unique) > 0 else 0.0
        
        # Update available tables BEFORE calculating cumulative outstanding sync
        # This ensures we have the most up-to-date view of what's available
        produced_tables = stream_table_pdf[
            stream_table_pdf['from'].isin(streams_in_community)
        ]['table'].unique()
        available_tables.update(produced_tables)
        
        # Remove newly available tables from cumulative outstanding sync
        cumulative_outstanding_sync_tables -= set(produced_tables)
        
        # Update migrated streams
        migrated_streams.update(streams_in_community)
        
        # Calculate cumulative outstanding sync (tables still needed but not yet available)
        # CRITICAL FIX: Only consider dependencies where the consuming stream is in migrated communities
        cumulative_outstanding_sync_df = stream_table_pdf[
            (stream_table_pdf['table'].isin(cumulative_outstanding_sync_tables)) &
            (stream_table_pdf['to'].isin(migrated_streams))  # Only count dependencies for migrated streams
        ][['table', 'to', 'size']].drop_duplicates(subset=['table', 'to'])
        
        # Classify cumulative outstanding sync by BI vs ETL
        cumulative_outstanding_sync_df['is_bi'] = cumulative_outstanding_sync_df['to'].str.lower().str.contains('json')
        cumulative_outstanding_sync_df['is_etl'] = ~cumulative_outstanding_sync_df['is_bi']
        
        cumulative_classification = cumulative_outstanding_sync_df.groupby('table').agg({
            'is_bi': 'any',
            'is_etl': 'any',
            'size': 'first'
        }).reset_index()
        
        cumulative_classification['category'] = cumulative_classification.apply(
            lambda row: 'ETL' if row['is_etl'] else 'BI',
            axis=1
        )
        
        cumulative_outstanding_bi = cumulative_classification[cumulative_classification['category'] == 'BI']
        cumulative_outstanding_etl = cumulative_classification[cumulative_classification['category'] == 'ETL']
        
        cumulative_outstanding_sync_count_bi = len(cumulative_outstanding_bi)
        cumulative_outstanding_sync_count_etl = len(cumulative_outstanding_etl)
        cumulative_outstanding_sync_count = cumulative_outstanding_sync_count_bi + cumulative_outstanding_sync_count_etl
        
        cumulative_outstanding_sync_size_bi = cumulative_outstanding_bi['size'].sum() if len(cumulative_outstanding_bi) > 0 else 0.0
        cumulative_outstanding_sync_size_etl = cumulative_outstanding_etl['size'].sum() if len(cumulative_outstanding_etl) > 0 else 0.0
        cumulative_outstanding_sync_size = cumulative_outstanding_sync_size_bi + cumulative_outstanding_sync_size_etl
        
        # Update cumulative metrics
        cumulative_streams += len(streams_in_community)
        cumulative_bi += len(bi_reports)
        cumulative_etl += len(etl_streams)
        cumulative_sync_size += sync_size
        cumulative_sync_size_bi += sync_size_bi
        cumulative_sync_size_etl += sync_size_etl
        cumulative_complexity += community_complexity_score
        
        # Calculate remaining
        remaining_streams = total_streams_global - cumulative_streams
        remaining_bi = total_bi_reports_global - cumulative_bi
        remaining_etl = total_etl_streams_global - cumulative_etl
        remaining_complexity = total_complexity_global - cumulative_complexity
        
        # Progress percentages
        progress_pct = (cumulative_streams / total_streams_global * 100) if total_streams_global > 0 else 0.0
        complexity_progress_pct = (cumulative_complexity / total_complexity_global * 100) if total_complexity_global > 0 else 0.0
        
        # Write community section
        report_content += f"""\n{'='*100}
STEP {step}/{len(optimized_order)}: COMMUNITY {community_id}
{'='*100}

--- COMMUNITY COMPOSITION ---
Streams in this community: {len(streams_in_community)} ({pct_total:.2f}% of total)
  - BI Reports: {len(bi_reports)} ({pct_bi:.2f}% of all BI reports)
  - ETL Streams: {len(etl_streams)} ({pct_etl:.2f}% of all ETL streams)
Community Complexity Score: {community_complexity_score:.0f}

--- STREAMS LIST ---
"""
        
        # Add list of streams in the community with complexity scores
        if len(bi_reports) > 0:
            report_content += f"\n  ** BI REPORTS ({len(bi_reports)}) **\n"
            for stream in sorted(bi_reports):
                complexity = stream_complexity_map.get(stream, 0)
                report_content += f"    - {stream} (complexity: {complexity:.0f})\n"
        
        if len(etl_streams) > 0:
            report_content += f"\n  ** ETL STREAMS ({len(etl_streams)}) **\n"
            for stream in sorted(etl_streams):
                complexity = stream_complexity_map.get(stream, 0)
                report_content += f"    - {stream} (complexity: {complexity:.0f})\n"
        
        report_content += f"""\n--- MIGRATION PROGRESS ---
Cumulative streams migrated: {cumulative_streams}/{total_streams_global} ({progress_pct:.2f}%)
  - BI Reports migrated: {cumulative_bi}/{total_bi_reports_global}
  - ETL Streams migrated: {cumulative_etl}/{total_etl_streams_global}
Cumulative complexity migrated: {cumulative_complexity:.0f}/{total_complexity_global:.0f} ({complexity_progress_pct:.2f}%)
Remaining streams: {remaining_streams}
  - BI Reports remaining: {remaining_bi}
  - ETL Streams remaining: {remaining_etl}
Remaining complexity: {remaining_complexity:.0f}

--- SYNC REQUIREMENTS FOR THIS STEP ---
Tables to SYNC (not yet available): {len(tables_to_sync)} unique tables, {sync_size:.2f} GB
  - For BI Reports: {len(tables_to_sync_bi)} tables, {sync_size_bi:.2f} GB
  - For ETL Streams: {len(tables_to_sync_etl)} tables, {sync_size_etl:.2f} GB
  - With multiple producers AND consumers: {num_multi_producer_consumer} tables, {multi_producer_consumer_size:.2f} GB

Tables ALREADY AVAILABLE (from previous migrations): {len(tables_already_available)} unique tables, {available_size:.2f} GB
Total incoming dependencies: {len(incoming_tables_unique)} unique tables, {incoming_tables_unique['size'].sum():.2f} GB

Sync efficiency: {(available_size / (available_size + sync_size) * 100) if (available_size + sync_size) > 0 else 0:.2f}% of dependencies already available

--- CUMULATIVE OUTSTANDING SYNC AT THIS STAGE ---
Total tables still to be synced (this + earlier communities): {cumulative_outstanding_sync_count} unique tables, {cumulative_outstanding_sync_size:.2f} GB
  - For BI Reports: {cumulative_outstanding_sync_count_bi} tables, {cumulative_outstanding_sync_size_bi:.2f} GB
  - For ETL Streams: {cumulative_outstanding_sync_count_etl} tables, {cumulative_outstanding_sync_size_etl:.2f} GB

--- TABLES TO SYNC (NEW) ---
"""
        
        if len(tables_to_sync) > 0:
            # Show BI tables first
            if len(tables_to_sync_bi) > 0:
                report_content += f"\n  ** FOR BI REPORTS ({len(tables_to_sync_bi)} tables, {sync_size_bi:.2f} GB) **\n"
                for table in tables_to_sync_bi['table']:
                    table_instances = tables_to_sync_details[tables_to_sync_details['table'] == table]
                    size = table_instances['size'].iloc[0]
                    # Get unique sets of producers and consumers
                    producers = sorted(table_instances['from'].unique())
                    consumers = sorted(table_instances['to'].unique())
                    
                    # Mark if this table has multiple producers AND consumers
                    multi_marker = " [MULTI-PRODUCER & MULTI-CONSUMER]" if table in multi_producer_consumer_tables else ""
                    
                    report_content += f"\n  Table: {table} ({size:.2f} GB){multi_marker}\n"
                    report_content += f"    - Written by (outside): {', '.join(producers)}\n"
                    report_content += f"    - Read by (inside): {', '.join(consumers)}\n"
            
            # Show ETL tables
            if len(tables_to_sync_etl) > 0:
                report_content += f"\n  ** FOR ETL STREAMS ({len(tables_to_sync_etl)} tables, {sync_size_etl:.2f} GB) **\n"
                for table in tables_to_sync_etl['table']:
                    table_instances = tables_to_sync_details[tables_to_sync_details['table'] == table]
                    size = table_instances['size'].iloc[0]
                    # Get unique sets of producers and consumers
                    producers = sorted(table_instances['from'].unique())
                    consumers = sorted(table_instances['to'].unique())
                    
                    # Mark if this table has multiple producers AND consumers
                    multi_marker = " [MULTI-PRODUCER & MULTI-CONSUMER]" if table in multi_producer_consumer_tables else ""
                    
                    report_content += f"\n  Table: {table} ({size:.2f} GB){multi_marker}\n"
                    report_content += f"    - Written by (outside): {', '.join(producers)}\n"
                    report_content += f"    - Read by (inside): {', '.join(consumers)}\n"
        else:
            report_content += "  (No new tables to sync - all dependencies already available!)\n"
        
        report_content += f"""\n--- TABLES ALREADY AVAILABLE (FROM PREVIOUS STEPS) ---
"""
        
        if len(tables_already_available) > 0:
            # Get all instances
            tables_available_all = incoming_tables_all[incoming_tables_all['table'].isin(tables_already_available['table'])]
            # Group by table to get unique producers and consumers
            for table in sorted(tables_already_available['table'].unique()):
                table_instances = tables_available_all[tables_available_all['table'] == table]
                size = table_instances['size'].iloc[0]
                producers = sorted(table_instances['from'].unique())
                consumers = sorted(table_instances['to'].unique())
                report_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                report_content += f"    - Written by (outside): {', '.join(producers)}\n"
                report_content += f"    - Read by (inside): {', '.join(consumers)}\n"
        else:
            report_content += "  (No dependencies from previous migrations)\n"
        
        report_content += f"""\n--- TABLES PRODUCED BY THIS COMMUNITY ---
This community produces: {len(outgoing_tables_unique)} unique tables, {outgoing_size:.2f} GB
(These will be available for subsequent communities)\n\n"""
        
        if len(outgoing_tables_unique) > 0:
            # Group by table to show unique producers and consumers
            for table in sorted(outgoing_tables_unique['table'].unique()):
                table_instances = outgoing_tables_all[outgoing_tables_all['table'] == table]
                size = table_instances['size'].iloc[0]
                producers = sorted(table_instances['from'].unique())
                consumers = sorted(table_instances['to'].unique())
                report_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                report_content += f"    - Written by (inside): {', '.join(producers)}\n"
                report_content += f"    - Read by (outside): {', '.join(consumers)}\n"
    
    # Final summary
    report_content += f"""\n\n{'='*100}
FINAL MIGRATION SUMMARY
{'='*100}

Total communities migrated: {len(optimized_order)}
Total streams migrated: {cumulative_streams}/{total_streams_global} (100%)
  - BI Reports: {cumulative_bi}/{total_bi_reports_global}
  - ETL Streams: {cumulative_etl}/{total_etl_streams_global}
Total complexity migrated: {cumulative_complexity:.0f}/{total_complexity_global:.0f} (100%)

Total cumulative sync cost: {cumulative_sync_size:.2f} GB
  - For BI Reports: {cumulative_sync_size_bi:.2f} GB
  - For ETL Streams: {cumulative_sync_size_etl:.2f} GB

Total tables with multiple producers AND consumers to sync: {cumulative_multi_producer_consumer_tables}
Total tables produced: {len(available_tables)} unique tables

Migration order: {optimized_order}

{'='*100}
"""
    
    # Save report
    report_file = os.path.join(outdir, f"migration_order_analysis_gamma_{resolution}.txt")
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"Migration order analysis saved: {report_file}")
    print(f"Total sync cost for this order: {cumulative_sync_size:.2f} GB")
    
    # Save sync details to CSV file
    if sync_details_rows:
        sync_details_df = pd.DataFrame(sync_details_rows)
        sync_details_csv = os.path.join(outdir, f"community_sync_details_gamma_{resolution}.csv")
        sync_details_df.to_csv(sync_details_csv, index=False)
        print(f"Community sync details CSV saved: {sync_details_csv}")
        print(f"Total sync detail rows: {len(sync_details_rows)}")
    
    # Save stream-to-community ordering mapping to CSV file
    if stream_ordering_rows:
        stream_ordering_df = pd.DataFrame(stream_ordering_rows)
        stream_ordering_csv = os.path.join(outdir, f"stream_community_ordering_gamma_{resolution}.csv")
        stream_ordering_df.to_csv(stream_ordering_csv, index=False)
        print(f"Stream community ordering CSV saved: {stream_ordering_csv}")
        print(f"Total streams mapped: {len(stream_ordering_rows)}")
    
    # Return both the report file path and the total sync cost
    return report_file, cumulative_sync_size

# COMMAND ----------

# DBTITLE 1,Run order optimization
order_rows = []
resolutions_costs = {}  # Track resolution -> (total_cost, num_communities) for metadata CSV

for resolution in resolutions:
    print(f"\n=== Processing resolution: {resolution} ===")
    leiden_df, meta = get_leiden_df(resolution, rep_by_res, igraph_names)
    
    # Get number of communities for this resolution
    num_communities = leiden_df['community'].nunique()
    print(f"Number of communities at resolution {resolution}: {num_communities}")

    # Check if rest_ids needs to be split into batches
    top_n = 10  # Same as used in split_communities_topN

    topN_ids, rest_ids_desc, community_weights = split_communities_topN(leiden_df, top_n=top_n)
    
    # REVERSE rest_ids to get ASCENDING order (lowest weight first)
    # split_communities_topN returns rest_ids in DESCENDING order (highest weight first)
    # We want to migrate batches with lowest weight first
    rest_ids = list(reversed(rest_ids_desc))
    print(f"\n--- REST communities reordered: ASCENDING by weight (lowest first) ---")

    dep_scaled_df = unique_table_weights.select("from", "to", "table", "weight")
    
    # Step 1: Optimize REST communities as a single batch
    # But identify communities with 0 sync requirement to migrate first
    print(f"\n--- Step 1: Optimizing REST communities ({len(rest_ids)} communities) ---")
    
    # Identify communities with 0 sync requirement (no incoming dependencies from outside)
    zero_sync_communities = []
    non_zero_sync_communities = []
    
    for comm_id in rest_ids:
        # Get streams in this community
        streams_in_community = set(leiden_df[leiden_df['community'] == comm_id]['stream'].tolist())
        
        # Check if there are any incoming dependencies from outside this community
        # Convert to pandas for easier filtering
        dep_pdf = dep_scaled_df.toPandas()
        
        # Find incoming dependencies: tables consumed by this community but produced outside
        incoming_deps = dep_pdf[
            (~dep_pdf['from'].isin(streams_in_community)) & 
            (dep_pdf['to'].isin(streams_in_community))
        ]
        
        if len(incoming_deps) == 0:
            zero_sync_communities.append(comm_id)
        else:
            non_zero_sync_communities.append(comm_id)
    
    print(f"Communities with 0 sync requirement: {len(zero_sync_communities)} - {zero_sync_communities}")
    print(f"Communities with non-zero sync requirement: {len(non_zero_sync_communities)}")
    
    # Optimize non-zero sync communities
    if len(non_zero_sync_communities) > 0:
        bf_rest = BruteForceCommunityOrdering(
            dep_scaled_df, 
            leiden_df, 
            non_zero_sync_communities,
            pre_available_communities=None  # No pre-available tables for rest
        )
        res_rest = bf_rest.brute_force(log_every=5000, label=f"REST_gamma_{resolution}")
        optimized_rest_order = res_rest["best_order"]
        rest_cost = res_rest['best_cost']
    else:
        optimized_rest_order = []
        rest_cost = 0.0
    
    # Final REST order: zero-sync communities first, then optimized non-zero communities
    rest_order = zero_sync_communities + optimized_rest_order
    
    print(f"\n--- REST order (0-sync first, then optimized): {rest_order} ---")
    print(f"--- REST cost (optimization metric): {rest_cost:.2f} ---")

    # Step 2: Optimize TOP N communities with all REST communities as pre-available
    print(f"\n--- Step 2: Optimizing TOP N communities (with all REST tables pre-available) ---")
    bf_top = BruteForceCommunityOrdering(
        dep_scaled_df, 
        leiden_df, 
        topN_ids,
        pre_available_communities=rest_ids  # All REST communities are already available
    )
    res_top = bf_top.brute_force(log_every=7500, label=f"TOPN_gamma_{resolution}")

    # Final order: REST first (0-sync + optimized), then TOP N
    final_order = rest_order + res_top["best_order"]
    
    print(f"\n=== FINAL MERGED ORDER for γ={resolution} ===")
    print(f"  REST order (0-sync first, then optimized): {rest_order}")
    print(f"  TOP N order: {res_top['best_order']}")
    print(f"  FINAL order: {final_order}")
    print(f"  REST cost (optimization metric): {rest_cost:.2f}")
    print(f"  TOP N cost (with REST pre-available, optimization metric): {res_top['best_cost']:.2f}")

# COMMAND ----------

# DBTITLE 1,Migration Order Analysis
# Generate detailed migration analysis report
print(f"\n--- Generating migration order analysis report ---")
report_file, total_sync_cost = generate_migration_order_analysis(
    leiden_df=leiden_df,
    stream_table_dependency_df=stream_stream_dependency_df,
    merged_edges_df=merged_dependency_df,
    optimized_order=final_order,
    resolution=resolution,
    complexity_scores_df=complexity_scores_df,
    outdir=f"{output_path}migration_order_analysis",
)
print(f"Report saved to: {report_file}")

# Store resolution, total sync cost, and number of communities 
resolutions_costs[resolution] = (total_sync_cost, num_communities)

print(f"\n=== ACTUAL TOTAL SYNC COST (from analysis): {total_sync_cost:.2f} GB ===")
print(f"=== NUMBER OF COMMUNITIES: {num_communities} ===")

# Format output with costs
order_line = (
    f"gamma={resolution}: {final_order} | "
    f"REST_cost={rest_cost:.2f}, "
    f"TOPN_cost={res_top['best_cost']:.2f}, "
    f"ACTUAL_TOTAL_SYNC_COST={total_sync_cost:.2f}, "
    f"NUM_COMMUNITIES={num_communities}"
)
order_rows.append(order_line)

# Write all orders to a text file in output_path
orders_output_path = output_path + "community_execution_orders.txt"
print(f"\n=== Writing results to: {orders_output_path} ===")

# Create new content with header and timestamp
header = "\n" + "="*100 + "\n"
header += f"Community Execution Orders - Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
header += "="*100 + "\n"
new_content = header + "\n".join(order_rows) + "\n"

# Write to file (creates new or overwrites with complete new content)
dbutils.fs.put(orders_output_path, new_content, overwrite=True)

print(f"All community execution orders written to: {orders_output_path}")
print(f"Total resolutions processed: {len(order_rows)}")
print("\nWritten content:")
for row in order_rows:
    print(f"  {row}")

# Append execution metadata to table
metadata_file = append_execution_metadata(
    weight_method=WEIGHT_METHOD,  # Use the variable set by the active weight calculation cell
    top_n=top_n,  # Include the top_n value used for ordering
    resolutions_dict=resolutions_costs  # Now contains (total_cost, num_communities) tuples
)

# COMMAND ----------

# DBTITLE 1,Generate migration timeline with efficiency gains
from datetime import datetime, timedelta
import pandas as pd
import math
import builtins

print("\n" + "="*100)
print("GENERATING MIGRATION TIMELINE")
print("="*100)

# Start date for migration
START_DATE = datetime(2026, 2, 23)

# Load the sync details and stream ordering CSVs generated by generate_migration_order_analysis
sync_details_file = f"{output_path}migration_order_analysis/community_sync_details_gamma_{resolution}.csv"
stream_ordering_file = f"{output_path}migration_order_analysis/stream_community_ordering_gamma_{resolution}.csv"

try:
    sync_details_pd = pd.read_csv(sync_details_file)
    stream_ordering_pd = pd.read_csv(stream_ordering_file)
except FileNotFoundError as e:
    print(f"ERROR: Required CSV files not found. Ensure cell 44 has completed successfully.")
    print(f"Missing file: {e.filename}")
    raise

# Get stream-to-squad mapping from dependency_df_full
stream_squad_mapping_df = dependency_df_full.select(
    col('STREAM_NAME').alias('stream_name'),
    col('SquadID').alias('squad')
).distinct().toPandas()

# Create a dictionary for quick lookup
stream_to_squad = stream_squad_mapping_df.set_index('stream_name')['squad'].to_dict()

print(f"Loaded stream-to-squad mapping for {len(stream_to_squad)} streams")

# Get total complexity for percentage calculations
total_complexity = complexity_scores_df.agg({'complexity_score': 'sum'}).collect()[0][0]
total_streams = leiden_df.shape[0]

print(f"Total complexity score: {total_complexity:.2f}")
print(f"Total streams: {total_streams}")

missing_static_tables = (
    load_static_tables(spark, static_tables_path)
    if static_tables_path is not None else set()
)

# Prepare timeline data
timeline_rows = []
cumulative_sync_gb = 0.0
cumulative_complexity = 0.0
cumulative_streams = 0
cumulative_weeks = 0
current_date = START_DATE
available_tables = missing_static_tables
bi_reports_ready = []  # Track BI reports ready from previous stage

# Get stream-to-table production mapping (TGT tables)
stream_produces_tables_df = dependency_df.filter(
    (upper(col('table_type')) == 'TGT') |
    (upper(col('table_type')) == 'TGT_TRNS')
).select(
    col('stream_name'),
    upper(col('DB_Table_Name')).alias('table_name')
).distinct().toPandas()

stream_produces = stream_produces_tables_df.groupby('stream_name')['table_name'].apply(set).to_dict()

# Get report-to-table dependencies if available
try:
    report_to_tables_pd = report_dependency_df.select(
        col('stream_name').alias('report_name'),
        upper(col('table_name')).alias('table_name')
    ).distinct().toPandas()
    report_required_tables = report_to_tables_pd.groupby('report_name')['table_name'].apply(set).to_dict()
    has_report_data = True
except:
    report_required_tables = {}
    has_report_data = False
    print("Note: Report dependency data not available, skipping BI report readiness analysis")

for idx, community_id in enumerate(final_order, start=1):
    # Get streams in this community
    community_streams = leiden_df[leiden_df['community'] == community_id]['stream'].tolist()
    num_streams = len(community_streams)
    
    # Separate BI reports from ETL streams
    bi_reports = [s for s in community_streams if 'json' in s.lower()]
    etl_streams = [s for s in community_streams if 'json' not in s.lower()]
    
    # Group streams by squad
    squad_streams = {}
    streams_without_squad = []
    
    for stream in community_streams:
        squad = stream_to_squad.get(stream)
        if squad:
            if squad not in squad_streams:
                squad_streams[squad] = []
            squad_streams[squad].append(stream)
        else:
            streams_without_squad.append(stream)
    
    # Create squad-based summary string for CSV (just counts)
    squad_summary_parts = []
    for squad in sorted(squad_streams.keys()):
        streams_in_squad = squad_streams[squad]
        squad_summary_parts.append(f"{squad} ({len(streams_in_squad)} streams)")
    
    if streams_without_squad:
        squad_summary_parts.append(f"Unknown Squad ({len(streams_without_squad)} streams)")
    
    squad_summary = ", ".join(squad_summary_parts)
    
    # Create detailed squad-based stream list for CSV (with stream names)
    squad_details_parts = []
    for squad in sorted(squad_streams.keys()):
        streams = sorted(squad_streams[squad])
        squad_details_parts.append(f"{squad}: {', '.join(streams)}")
    
    if streams_without_squad:
        squad_details_parts.append(f"Unknown Squad: {', '.join(sorted(streams_without_squad))}")
    
    squad_based_streams = " | ".join(squad_details_parts)
    
    # Get complexity for this community
    community_complexity_df = complexity_scores_df.filter(
        col('stream_name').isin(community_streams)
    )
    community_complexity = community_complexity_df.agg({'complexity_score': 'sum'}).collect()[0][0] or 0.0
    complexity_pct = (community_complexity / total_complexity * 100) if total_complexity > 0 else 0.0
    
    # Get sync requirements for this community
    sync_tables = sync_details_pd[sync_details_pd['community_id'] == community_id]
    sync_gb = sync_tables['size_gb'].sum() if len(sync_tables) > 0 else 0.0
    sync_table_list = sync_tables['table_name'].unique().tolist() if len(sync_tables) > 0 else []
    
    # Get target tables produced by this community
    target_tables = set()
    for stream in community_streams:
        if stream in stream_produces:
            target_tables.update(stream_produces[stream])
    num_target_tables = len(target_tables)
    
    # Update cumulative metrics FIRST
    cumulative_sync_gb += sync_gb
    cumulative_complexity += community_complexity
    cumulative_streams += num_streams
    cumulative_complexity_pct = (cumulative_complexity / total_complexity * 100) if total_complexity > 0 else 0.0
    cumulative_streams_pct = (cumulative_streams / total_streams * 100) if total_streams > 0 else 0.0
    
    # Calculate efficiency gain based on CURRENT cumulative complexity (absolute)
    # 4% for each 10% of cumulative complexity migrated, max 25%
    efficiency_gain = min(0.04 * math.floor(cumulative_complexity_pct / 10), 0.25)
    
    # Apply efficiency gain to complexity
    effective_complexity_pct = complexity_pct * (1 - efficiency_gain)
    
    # Calculate weeks: 1 week per 2% effective complexity, rounded up to full weeks
    weeks_for_community = math.ceil(effective_complexity_pct / 2.0)
    cumulative_weeks += weeks_for_community
    
    # Calculate dates
    end_date = START_DATE + timedelta(weeks=cumulative_weeks)
    
    # Check which BI reports can be migrated after this community
    # (reports whose required tables are now all available)
    if has_report_data:
        available_tables.update(target_tables)
        newly_ready_reports = []
        for report, required_tables in report_required_tables.items():
            if report not in bi_reports_ready and required_tables.issubset(available_tables):
                newly_ready_reports.append(report)
                bi_reports_ready.append(report)
    else:
        newly_ready_reports = []
    
    # Add BI reports from previous stage to this community's timeline
    bi_reports_in_timeline = bi_reports if idx == 1 else bi_reports + newly_ready_reports
    
    timeline_rows.append({
        'Stage': idx,
        'Community_ID': community_id,
        'Original_Community_IDs': [community_id],  # Track original communities
        'Num_Streams': num_streams,
        'Squad_Based_Streams': squad_based_streams,  # Detailed squad-based list for CSV
        'Squad_Summary': squad_summary,  # Summary with counts
        'ETL_Streams': len(etl_streams),
        'BI_Reports_Migrated': len(bi_reports_in_timeline),
        'BI_Reports_List': ', '.join(bi_reports_in_timeline) if bi_reports_in_timeline else 'None',
        'Sync_Tables_Count': len(sync_table_list),
        'Sync_Tables_GB': f"{sync_gb:.2f}",
        'Cumulative_Sync_GB': f"{cumulative_sync_gb:.2f}",
        'Complexity_Pct': f"{complexity_pct:.2f}%",
        'Cumulative_Complexity_Pct': f"{cumulative_complexity_pct:.2f}%",
        'Cumulative_Streams_Pct': f"{cumulative_streams_pct:.2f}%",
        'Weeks_For_Stage': weeks_for_community,
        'Cumulative_Weeks': cumulative_weeks,
        'Start_Date': current_date.strftime('%Y-%m-%d'),
        'End_Date': end_date.strftime('%Y-%m-%d'),
        'Target_Tables_Created': num_target_tables,
        # Keep these for text report generation (not in CSV)
        '_squad_streams_dict': squad_streams,
        '_streams_without_squad': streams_without_squad
    })
    
    current_date = end_date

# Create DataFrame
timeline_df = pd.DataFrame(timeline_rows)

print(f"\nOriginal timeline has {len(timeline_df)} stages")

# Club consecutive communities with 2 or fewer weeks
print("\nClubbing consecutive communities with ≤2 weeks...")
clubbed_rows = []
i = 0
clubbed_stage_num = 1

while i < len(timeline_df):
    current_row = timeline_df.iloc[i].to_dict()
    
    # Check if current stage has ≤2 weeks
    if current_row['Weeks_For_Stage'] <= 2:
        # Start a clubbed group
        clubbed_communities = [current_row['Community_ID']]
        clubbed_original_ids = current_row['Original_Community_IDs'].copy()
        combined_streams = current_row['Num_Streams']
        combined_etl = current_row['ETL_Streams']
        combined_bi = current_row['BI_Reports_Migrated']
        combined_sync_count = current_row['Sync_Tables_Count']
        combined_sync_gb = float(current_row['Sync_Tables_GB'])
        combined_target_tables = current_row['Target_Tables_Created']
        combined_weeks = current_row['Weeks_For_Stage']
        
        # Collect squad streams
        all_squad_streams = current_row['_squad_streams_dict'].copy()
        all_streams_without_squad = current_row['_streams_without_squad'].copy()
        all_bi_reports = [r.strip() for r in current_row['BI_Reports_List'].split(',') if r.strip() != 'None']
        
        # Look ahead for more consecutive stages with ≤2 weeks
        j = i + 1
        while j < len(timeline_df) and timeline_df.iloc[j]['Weeks_For_Stage'] <= 2:
            next_row = timeline_df.iloc[j].to_dict()
            clubbed_communities.append(next_row['Community_ID'])
            clubbed_original_ids.extend(next_row['Original_Community_IDs'])
            combined_streams += next_row['Num_Streams']
            combined_etl += next_row['ETL_Streams']
            combined_bi += next_row['BI_Reports_Migrated']
            combined_sync_count += next_row['Sync_Tables_Count']
            combined_sync_gb += float(next_row['Sync_Tables_GB'])
            combined_target_tables += next_row['Target_Tables_Created']
            combined_weeks += next_row['Weeks_For_Stage']
            
            # Merge squad streams
            for squad, streams in next_row['_squad_streams_dict'].items():
                if squad not in all_squad_streams:
                    all_squad_streams[squad] = []
                all_squad_streams[squad].extend(streams)
            all_streams_without_squad.extend(next_row['_streams_without_squad'])
            
            # Merge BI reports
            next_bi = [r.strip() for r in next_row['BI_Reports_List'].split(',') if r.strip() != 'None']
            all_bi_reports.extend(next_bi)
            
            j += 1
        
        # Create clubbed row
        if len(clubbed_communities) > 1:
            # Multiple communities clubbed together
            community_id_str = f"Clubbed ({', '.join(map(str, clubbed_communities))})"
            print(f"  Clubbing communities {clubbed_communities} into stage {clubbed_stage_num}")
        else:
            # Single community, keep as is
            community_id_str = str(clubbed_communities[0])
        
        # Rebuild squad summary and details
        squad_summary_parts = []
        squad_details_parts = []
        for squad in sorted(all_squad_streams.keys()):
            streams = sorted(set(all_squad_streams[squad]))  # Remove duplicates
            squad_summary_parts.append(f"{squad} ({len(streams)} streams)")
            squad_details_parts.append(f"{squad}: {', '.join(streams)}")
        
        if all_streams_without_squad:
            unique_unknown = sorted(set(all_streams_without_squad))
            squad_summary_parts.append(f"Unknown Squad ({len(unique_unknown)} streams)")
            squad_details_parts.append(f"Unknown Squad: {', '.join(unique_unknown)}")
        
        squad_summary = ", ".join(squad_summary_parts)
        squad_based_streams = " | ".join(squad_details_parts)
        
        # Use the last row's cumulative metrics
        last_row = timeline_df.iloc[j-1]
        
        clubbed_rows.append({
            'Stage': clubbed_stage_num,
            'Community_ID': community_id_str,
            'Original_Community_IDs': clubbed_original_ids,
            'Num_Streams': combined_streams,
            'Squad_Based_Streams': squad_based_streams,
            'Squad_Summary': squad_summary,
            'ETL_Streams': combined_etl,
            'BI_Reports_Migrated': combined_bi,
            'BI_Reports_List': ', '.join(sorted(set(all_bi_reports))) if all_bi_reports else 'None',
            'Sync_Tables_Count': combined_sync_count,
            'Sync_Tables_GB': f"{combined_sync_gb:.2f}",
            'Cumulative_Sync_GB': last_row['Cumulative_Sync_GB'],
            'Complexity_Pct': f"{builtins.sum([float(timeline_df.iloc[k]['Complexity_Pct'].rstrip('%')) for k in range(i, j)]):.2f}%",
            'Cumulative_Complexity_Pct': last_row['Cumulative_Complexity_Pct'],
            'Cumulative_Streams_Pct': last_row['Cumulative_Streams_Pct'],
            'Weeks_For_Stage': combined_weeks,
            'Cumulative_Weeks': last_row['Cumulative_Weeks'],
            'Start_Date': current_row['Start_Date'],
            'End_Date': last_row['End_Date'],
            'Target_Tables_Created': combined_target_tables,
            '_squad_streams_dict': all_squad_streams,
            '_streams_without_squad': all_streams_without_squad
        })
        
        i = j  # Move to next unchecked stage
    else:
        # Stage has >2 weeks, keep as is
        current_row['Stage'] = clubbed_stage_num
        current_row['Community_ID'] = str(current_row['Community_ID'])
        clubbed_rows.append(current_row)
        i += 1
    
    clubbed_stage_num += 1

# Create clubbed DataFrame
timeline_df = pd.DataFrame(clubbed_rows)

print(f"After clubbing: {len(timeline_df)} stages\n")

# Remove internal columns before saving to CSV
csv_columns = [column for column in timeline_df.columns if not column.startswith('_') and column != 'Original_Community_IDs']
timeline_csv_df = timeline_df[csv_columns]

# Write to CSV
timeline_csv_path = f"{output_path}migration_order_analysis/migration_timeline_gamma_{resolution}.csv"
timeline_csv_df.to_csv(timeline_csv_path, index=False)
print(f"\nMigration timeline CSV saved to: {timeline_csv_path}")

# Generate concise text report
timeline_txt_path = f"{output_path}migration_order_analysis/migration_timeline_gamma_{resolution}.txt"

with open(timeline_txt_path, 'w') as f:
    f.write("="*120 + "\n")
    f.write(f"MIGRATION TIMELINE - Resolution γ={resolution}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Start Date: {START_DATE.strftime('%Y-%m-%d')}\n")
    f.write(f"Total Duration: {timeline_df.iloc[-1]['Cumulative_Weeks']} weeks\n")
    f.write("="*120 + "\n\n")
    
    for _, row in timeline_df.iterrows():
        # Show which communities were clubbed
        if 'Clubbed' in str(row['Community_ID']):
            original_ids = ', '.join(map(str, row['Original_Community_IDs']))
            f.write(f"STAGE {row['Stage']}: Communities {original_ids} (Clubbed) | {row['Start_Date']} to {row['End_Date']} ({row['Weeks_For_Stage']} weeks)\n")
        else:
            f.write(f"STAGE {row['Stage']}: Community {row['Community_ID']} | {row['Start_Date']} to {row['End_Date']} ({row['Weeks_For_Stage']} weeks)\n")
        
        f.write(f"  Streams: {row['Num_Streams']} ({row['ETL_Streams']} ETL, {row['BI_Reports_Migrated']} BI) | Complexity: {row['Complexity_Pct']}\n")
        f.write(f"  Sync: {row['Sync_Tables_Count']} tables, {row['Sync_Tables_GB']} GB | Cumulative: {row['Cumulative_Sync_GB']} GB\n")
        f.write(f"  Target Tables: {row['Target_Tables_Created']} | Cumulative Progress: {row['Cumulative_Complexity_Pct']} complexity, {row['Cumulative_Streams_Pct']} streams\n")
        
        # Add squad-based stream details with stream names
        f.write(f"\n  Squads:\n")
        squad_streams = row['_squad_streams_dict']
        for squad in sorted(squad_streams.keys()):
            streams = sorted(squad_streams[squad])
            f.write(f"    {squad} ({len(streams)} streams):\n")
            for stream in streams:
                f.write(f"      - {stream}\n")
        
        # Add streams without squad if any
        streams_without_squad = row['_streams_without_squad']
        if streams_without_squad:
            f.write(f"    Unknown Squad ({len(streams_without_squad)} streams):\n")
            for stream in sorted(streams_without_squad):
                f.write(f"      - {stream}\n")
        
        if row['BI_Reports_List'] != 'None':
            f.write(f"\n  BI Reports: {row['BI_Reports_List']}\n")
        f.write("\n")
    
    f.write("="*120 + "\n")
    f.write(f"SUMMARY\n")
    f.write(f"  Total Stages (after clubbing): {len(timeline_df)}\n")
    f.write(f"  Total Streams: {total_streams}\n")
    f.write(f"  Total Sync Required: {timeline_df.iloc[-1]['Cumulative_Sync_GB']} GB\n")
    f.write(f"  Total Duration: {timeline_df.iloc[-1]['Cumulative_Weeks']} weeks ({timeline_df.iloc[-1]['Cumulative_Weeks']/4:.1f} months)\n")
    f.write(f"  Completion Date: {timeline_df.iloc[-1]['End_Date']}\n")
    if has_report_data:
        f.write(f"  BI Reports Ready: {len(bi_reports_ready)}\n")
    f.write("="*120 + "\n")

print(f"Migration timeline report saved to: {timeline_txt_path}")

# Display summary
print(f"\n{'='*120}")
print(f"MIGRATION TIMELINE SUMMARY")
print(f"{'='*120}")
print(f"Total Stages (after clubbing): {len(timeline_df)}")
print(f"Total Streams: {total_streams}")
print(f"Total Sync Required: {timeline_df.iloc[-1]['Cumulative_Sync_GB']} GB")
print(f"Start Date: {START_DATE.strftime('%Y-%m-%d')}")
print(f"End Date: {timeline_df.iloc[-1]['End_Date']}")
print(f"Total Duration: {timeline_df.iloc[-1]['Cumulative_Weeks']} weeks ({timeline_df.iloc[-1]['Cumulative_Weeks']/4:.1f} months)")
if has_report_data:
    print(f"BI Reports Ready: {len(bi_reports_ready)}")
print(f"{'='*120}\n")

# Display first few rows (without internal columns)
print("\nFirst 5 stages:")
display(timeline_csv_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Report Migration Readiness Analysis
# MAGIC
# MAGIC Determining which reports can be migrated at each stage of the execution order based on:
# MAGIC * Tables produced by migrated streams (TGT tables)
# MAGIC * Tables that are currently synced (incoming dependencies)
# MAGIC * Report-to-table dependencies from cell 9

# COMMAND ----------

# DBTITLE 1,Select resolution for report analysis
# Select which resolution to use for report migration readiness analysis
# This should match one of the resolutions processed in cell 40

SELECTED_RESOLUTION = 1.8  # Change this to the desired resolution


print(f"Selected resolution for report analysis: {SELECTED_RESOLUTION}")
print(f"Available resolutions: {resolutions}")

if SELECTED_RESOLUTION not in resolutions:
    print(f"\nWARNING: Selected resolution {SELECTED_RESOLUTION} was not processed in cell 40!")
    print(f"Please choose from: {resolutions}")
else:
    print(f"✓ Resolution {SELECTED_RESOLUTION} is valid")

# COMMAND ----------

# DBTITLE 1,Load required data for report readiness analysis
# Use the selected resolution to load the correct CSV file
stream_ordering_file = f"{output_path}migration_order_analysis/stream_community_ordering_gamma_{SELECTED_RESOLUTION}.csv"

try:
    stream_ordering_pd = pd.read_csv(stream_ordering_file)
    print(f"Loaded stream ordering from: {stream_ordering_file}")
    print(f"Resolution: {SELECTED_RESOLUTION}")
    print(f"Columns: {stream_ordering_pd.columns.tolist()}")
    print(f"Total streams: {len(stream_ordering_pd)}")
    display(stream_ordering_pd.head(10))
except FileNotFoundError:
    print(f"ERROR: File not found: {stream_ordering_file}")
    print(f"Please ensure cell 40 has completed and generated output for resolution {SELECTED_RESOLUTION}")
    print(f"\nAvailable files:")
    !ls -t {output_path}migration_order_analysis/stream_community_ordering_gamma_*.csv 2>/dev/null

# COMMAND ----------

# DBTITLE 1,Prepare report-to-table dependencies
# Convert report_dependency_df to Pandas for easier manipulation
report_to_tables_pd = report_dependency_df.select(
    col('stream_name').alias('report_name'),
    upper(col('table_name')).alias('table_name')
).distinct().toPandas()

print(f"Total report-to-table dependencies: {len(report_to_tables_pd)}")
print(f"Unique reports: {report_to_tables_pd['report_name'].nunique()}")
print(f"Unique tables required by reports: {report_to_tables_pd['table_name'].nunique()}")

# Group by report to get all tables required per report
report_required_tables = report_to_tables_pd.groupby('report_name')['table_name'].apply(set).to_dict()

print(f"\nExample - First 3 reports and their required tables:")
for i, (report, tables) in enumerate(list(report_required_tables.items())[:3]):
    print(f"  {report}: {len(tables)} tables - {list(tables)[:5]}{'...' if len(tables) > 5 else ''}")

# COMMAND ----------

# DBTITLE 1,Extract table production by streams
# Get which tables are produced (TGT) by which streams
# From the original dependency_df, extract TGT tables per stream
stream_produces_tables_df = dependency_df.filter(
    (upper(col('table_type')) == 'TGT') |
    (upper(col('table_type')) == 'TGT_TRNS')
).select(
    col('stream_name'),
    upper(col('DB_Table_Name')).alias('table_name')
).distinct()

stream_produces_tables_pd = stream_produces_tables_df.toPandas()
print(f"Total stream-produces-table mappings: {len(stream_produces_tables_pd)}")

# Group by stream to get all tables produced per stream
stream_produces = stream_produces_tables_pd.groupby('stream_name')['table_name'].apply(set).to_dict()

print(f"Total streams that produce tables: {len(stream_produces)}")
print(f"\nExample - First 3 streams and tables they produce:")
for i, (stream, tables) in enumerate(list(stream_produces.items())[:3]):
    print(f"  {stream}: {len(tables)} tables - {list(tables)[:5]}{'...' if len(tables) > 5 else ''}")

# COMMAND ----------

# DBTITLE 1,Load sync requirements from migration analysis
# Read the community sync details CSV that was generated in cell 48
# Use the selected resolution to load the correct CSV file
sync_details_file = f"{output_path}migration_order_analysis/community_sync_details_gamma_{SELECTED_RESOLUTION}.csv"

try:
    sync_details_pd = pd.read_csv(sync_details_file)
    print(f"Loaded sync details from: {sync_details_file}")
    print(f"Resolution: {SELECTED_RESOLUTION}")
    print(f"Columns: {sync_details_pd.columns.tolist()}")
    print(f"Total sync requirements: {len(sync_details_pd)}")
    
    # Extract tables that need to be synced for each community
    # These are the incoming dependencies (tables needed but not yet available)
    community_sync_tables = sync_details_pd.groupby('community_id')['table_name'].apply(set).to_dict()
    
    print(f"\nCommunities with sync requirements: {len(community_sync_tables)}")
    display(sync_details_pd.head(10))
except FileNotFoundError:
    print(f"WARNING: File not found: {sync_details_file}")
    print(f"Will assume all incoming tables are synced.")
    print(f"\nAvailable files:")
    !ls -t {output_path}migration_order_analysis/community_sync_details_gamma_*.csv 2>/dev/null
    community_sync_tables = {}

# COMMAND ----------

# DBTITLE 1,Calculate report readiness at each execution stage
# Get execution orders in the EXACT order they appear (preserving optimization order)
# Do NOT sort - the order from the CSV reflects the optimized community ordering
execution_stages = stream_ordering_pd['execution_order'].unique().tolist()

print(f"Total execution stages: {len(execution_stages)}")
print(f"Execution order (optimized): {execution_stages}")

missing_static_tables = (
    load_static_tables(spark, static_tables_path)
    if static_tables_path is not None else set()
)


# Initialize tracking
available_tables = missing_static_tables  # Tables available from ALL migrated streams (cumulative) and initialized with already available static tables
synced_tables = set()  # Tables that are synced (incoming dependencies, cumulative)
reports_ready_by_stage = {}  # stage -> list of reports ready
reports_migrated = set()  # Track which reports have been marked as ready

# Process each execution stage IN THE OPTIMIZED ORDER
for stage in execution_stages:
    # Get communities being migrated at this stage
    communities_at_stage = stream_ordering_pd[stream_ordering_pd['execution_order'] == stage]['community_id'].unique()
    
    # Get streams being migrated at this stage
    streams_at_stage = stream_ordering_pd[stream_ordering_pd['execution_order'] == stage]['stream_name'].tolist()
    
    # Track new tables added at this stage
    new_synced_tables = set()
    new_produced_tables = set()
    
    # Add synced tables for these communities (incoming dependencies)
    for comm_id in communities_at_stage:
        if comm_id in community_sync_tables:
            new_synced_tables.update(community_sync_tables[comm_id])
    
    # Add tables produced by streams at this stage
    for stream in streams_at_stage:
        if stream in stream_produces:
            new_produced_tables.update(stream_produces[stream])
    
    # Update cumulative sets
    synced_tables.update(new_synced_tables)
    available_tables.update(new_produced_tables)
    
    # Combine all available tables: produced by migrated streams + synced tables
    all_available_tables = available_tables.union(synced_tables)
    
    # Check which reports are now ready (all required tables available)
    reports_ready_at_stage = []
    for report, required_tables in report_required_tables.items():
        if report not in reports_migrated:  # Only check reports not yet migrated
            if required_tables.issubset(all_available_tables):
                reports_ready_at_stage.append(report)
                reports_migrated.add(report)
    
    reports_ready_by_stage[stage] = reports_ready_at_stage
    
    print(f"\nStage {stage}:")
    print(f"  Communities: {list(communities_at_stage)}")
    print(f"  Streams migrated at this stage: {len(streams_at_stage)}")
    print(f"  New tables produced: {len(new_produced_tables)}")
    print(f"  New tables synced: {len(new_synced_tables)}")
    print(f"  Cumulative tables from migrated streams: {len(available_tables)}")
    print(f"  Cumulative tables synced: {len(synced_tables)}")
    print(f"  Total tables available: {len(all_available_tables)}")
    print(f"  Reports ready at this stage: {len(reports_ready_at_stage)}")
    if reports_ready_at_stage:
        print(f"  Report names: {reports_ready_at_stage[:5]}{'...' if len(reports_ready_at_stage) > 5 else ''}")

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"Total reports analyzed: {len(report_required_tables)}")
print(f"Total reports ready after all stages: {len(reports_migrated)}")
print(f"Reports not ready: {len(report_required_tables) - len(reports_migrated)}")

# COMMAND ----------

# DBTITLE 1,Generate detailed report readiness output
# Create detailed output DataFrame
report_readiness_data = []

for stage in execution_stages:
    for report in reports_ready_by_stage[stage]:
        report_readiness_data.append({
            'execution_order': stage,
            'report_name': report,
            'num_required_tables': len(report_required_tables[report]),
            'required_tables': ', '.join(sorted(list(report_required_tables[report])[:10])) + 
                             ('...' if len(report_required_tables[report]) > 10 else '')
        })

report_readiness_df = pd.DataFrame(report_readiness_data)

# Save to CSV in the migration_order_analysis subdirectory with resolution in filename
report_readiness_file = f"{output_path}migration_order_analysis/report_migration_readiness_gamma_{SELECTED_RESOLUTION}.csv"
report_readiness_df.to_csv(report_readiness_file, index=False)
print(f"Report readiness saved to: {report_readiness_file}")

# Display summary statistics
print(f"\nReport Readiness by Execution Stage:")
stage_summary = report_readiness_df.groupby('execution_order').agg({
    'report_name': 'count'
}).rename(columns={'report_name': 'reports_ready'}).reset_index()

stage_summary['cumulative_reports'] = stage_summary['reports_ready'].cumsum()

display(stage_summary)

print(f"\nFirst 20 reports ready:")
display(report_readiness_df.head(20))

# COMMAND ----------

# DBTITLE 1,Generate detailed text report
# Generate a detailed text report similar to the migration analysis
report_readiness_text_file = f"{output_path}migration_order_analysis/report_migration_readiness_analysis_gamma_{SELECTED_RESOLUTION}.txt"

with open(report_readiness_text_file, 'w') as f:
    f.write("="*100 + "\n")
    f.write("REPORT MIGRATION READINESS ANALYSIS\n")
    f.write("="*100 + "\n\n")
    
    f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
    f.write(f"Resolution (gamma): {SELECTED_RESOLUTION}\n")
    # f.write(f"Weight Method: {WEIGHT_METHOD}\n")
    f.write(f"Total Reports Analyzed: {len(report_required_tables)}\n")
    f.write(f"Total Reports Ready: {len(reports_migrated)}\n")
    f.write(f"Total Execution Stages: {len(execution_stages)}\n\n")
    
    f.write("="*100 + "\n")
    f.write("REPORT READINESS BY EXECUTION STAGE\n")
    f.write("="*100 + "\n\n")
    
    cumulative_reports = 0
    for stage in execution_stages:
        reports_at_stage = reports_ready_by_stage[stage]
        cumulative_reports += len(reports_at_stage)
        
        communities_at_stage = stream_ordering_pd[stream_ordering_pd['execution_order'] == stage]['community_id'].unique()
        streams_at_stage = stream_ordering_pd[stream_ordering_pd['execution_order'] == stage]['stream_name'].tolist()
        
        f.write(f"\n{'─'*100}\n")
        f.write(f"EXECUTION STAGE {stage}\n")
        f.write(f"{'─'*100}\n")
        f.write(f"Communities Migrated: {list(communities_at_stage)}\n")
        f.write(f"Number of Streams: {len(streams_at_stage)}\n")
        f.write(f"Reports Ready at This Stage: {len(reports_at_stage)}\n")
        f.write(f"Cumulative Reports Ready: {cumulative_reports}\n\n")
        
        if reports_at_stage:
            f.write(f"Reports Ready:\n")
            for report in sorted(reports_at_stage):
                required_tables = report_required_tables[report]
                f.write(f"  • {report}\n")
                f.write(f"    Required Tables ({len(required_tables)}): {', '.join(sorted(list(required_tables)[:5]))}")
                if len(required_tables) > 5:
                    f.write(f" ... and {len(required_tables) - 5} more")
                f.write(f"\n")
        else:
            f.write(f"  No reports ready at this stage.\n")
    
    # Reports not ready
    reports_not_ready = set(report_required_tables.keys()) - reports_migrated
    if reports_not_ready:
        f.write(f"\n\n{'='*100}\n")
        f.write(f"REPORTS NOT READY AFTER ALL STAGES ({len(reports_not_ready)})\n")
        f.write(f"{'='*100}\n\n")
        
        for report in sorted(reports_not_ready):
            required_tables = report_required_tables[report]
            missing_tables = required_tables - available_tables.union(synced_tables)
            f.write(f"  • {report}\n")
            f.write(f"    Required Tables: {len(required_tables)}\n")
            f.write(f"    Missing Tables: {len(missing_tables)}\n")
            if missing_tables:
                f.write(f"    Missing: {', '.join(sorted(list(missing_tables)[:10]))}")
                if len(missing_tables) > 10:
                    f.write(f" ... and {len(missing_tables) - 10} more")
                f.write(f"\n")

print(f"\nDetailed text report saved to: {report_readiness_text_file}")
print(f"\nAnalysis complete!")
print(f"\nOutput files:")
print(f"  1. {report_readiness_file}")
print(f"  2. {report_readiness_text_file}")

# COMMAND ----------

# DBTITLE 1,Report migration readiness output
missing_tables = set()
section_found = False
with open(f"{output_path}migration_order_analysis/report_migration_readiness_analysis_gamma_1.8.txt", "r") as f:
    for line in f:
        if "REPORTS NOT READY AFTER ALL STAGES" in line:
            section_found = True
        elif section_found and line.strip().startswith("Missing:"):
            tables_str = line.strip().split("Missing:")[1].split("...")[0]
            tables = [t.strip() for t in tables_str.split(",") if t.strip()]
            missing_tables.update(tables)
        elif section_found and line.strip() == "":
            continue

missing_tables = sorted(missing_tables)
print(",\n".join(missing_tables))

pd.DataFrame({"table name": missing_tables}).to_csv(
    f"{output_path}migration_order_analysis/missing_tables_for_reports.csv",
    index=False
)
