# Databricks notebook source
# MAGIC %md
# MAGIC ## Libraries and Parameters
# MAGIC

# COMMAND ----------

!pip install python-igraph adjustText infomap netgraph networkx python-louvain igraph leidenalg

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import libraries
import pandas as pd  # For reading dataset files
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
from pyspark.sql.functions import col, lit, when
from sklearn.metrics import adjusted_rand_score
from collections import Counter



# COMMAND ----------

current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# COMMAND ----------

dbutils.widgets.text(
    "input_path",
    "/Volumes/users/jince_james/lufthansa/community_detection/ETL-table-dependencies_20251201_1345.csv",
    "Input CSV Path"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Datasets:

# COMMAND ----------

input_path = dbutils.widgets.get("input_path")

# COMMAND ----------

dependency_df = spark.read.format("csv").option("header", "true").load(input_path)

# COMMAND ----------

from pyspark.sql.functions import col, upper

# Removing streams associated with acrchiving, GDPR and housekeeping
dependency_df = dependency_df.filter(
    ~upper(col("stream_name")).contains("ARCHIVE") & 
    ~upper(col("stream_name")).contains("GDPR") &
    ~upper(col("stream_name")).contains("HOUSEKEEPING")
)

# COMMAND ----------

display(dependency_df)

# COMMAND ----------

# DBTITLE 1,analysis
print(f"Total number of records is {dependency_df.count()}")
print(f"Total number of distinct records is {dependency_df.distinct().count()}")

# There are no duplicate records in the input dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forming Table to Stream Dependecies

# COMMAND ----------

# DBTITLE 1,analysis: Check if there are any trns src or tgt tables part of non transactional DBs
cleaned_dependency_df = dependency_df.select('stream_name', col('ref_object_name').alias('table_name'),col('ref_object_dbname').alias('db_name'), 'table_type')

display(cleaned_dependency_df.filter(col('table_type').contains('Trns') & ~ col('db_name').contains('TRANS')))

# There are no transactional tables in the dataset that are part of a non transactional DB - Verified

# COMMAND ----------

# DBTITLE 1,analysis: #ToDo : Check if the DDL of tables with same name, same across 2 different DBs
duplicate_tables = (
    cleaned_dependency_df.alias("cdf1")
    .join(cleaned_dependency_df.alias("cdf2"), col("cdf1.table_name") == col("cdf2.table_name"))
    .filter(col("cdf1.db_name") != col("cdf2.db_name"))
    .filter(~col("cdf1.stream_name").contains("ARCHIVE") & ~col("cdf2.stream_name").contains("ARCHIVE"))
    .select(
        col("cdf1.db_name").alias("db_name1"),
        col("cdf1.table_name").alias("table_name1"),
        col("cdf2.db_name").alias("db_name2"),
        col("cdf2.table_name").alias("table_name2")
    ).distinct()
)
duplicate_tables.groupBy("table_name1", "db_name1").count().display()
# duplicate_tables.display(False)

# There are tables with same table name, are the DDLs same for duplicates? Are they 2 different instances? or same?
# RESULT -> If they are present in 2 different DBs, they should be also different table instances. 

# COMMAND ----------

# DBTITLE 1,Filtering Transactional Tables
dependency_without_transactional_df = (
    dependency_df.filter(~col('table_type').contains('Trns'))
    .select('stream_name', col('DB_Table_Name').alias('table_name'), 'table_type')
    .distinct()
)


# Self join to find dependencies and filter intra stream dependency
filtered_self_join_result = (
    dependency_without_transactional_df.alias("df1")
    .join(dependency_without_transactional_df.alias("df2"), col("df1.table_name") == col("df2.table_name"))
    .filter(col("df1.stream_name") != col("df2.stream_name"))
)

display(filtered_self_join_result)

# COMMAND ----------

# DBTITLE 1,Analysis: See results count with transactional tables filtered self join
filtered_self_join_result.count()

# COMMAND ----------

# DBTITLE 1,Analysis: Without filtering transactional tables
dependency_with_transactional_df = dependency_df.select(
    'stream_name', col('DB_Table_Name').alias('table_name'), 'table_type'
).distinct()

# Self join to find dependencies and filter intra stream dependency (table to table within same stream)
# This will basically result all cases where the exact table is a src or tgt of 2 different streams
non_filtered_self_join_result = (
    dependency_with_transactional_df.alias("df1")
    .join(dependency_with_transactional_df.alias("df2"), col("df1.table_name") == col("df2.table_name"))
    .filter(col("df1.stream_name") != col("df2.stream_name"))
)
display(non_filtered_self_join_result)

# COMMAND ----------

# DBTITLE 1,analysis: See results count with transactional count dependencies
non_filtered_self_join_result.count()

# RESULT: There are ~120 inter stream dependencies more with transactional tables included. So we will use dataset without filtering
# Result -> Except for archiving streams, this shouldn't be the case. Dig deeper. Send ex to micheal if there are outliers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forming Stream - Stream Dependencies

# COMMAND ----------

src_tgt_dependencies = non_filtered_self_join_result.filter(
    (
        (upper(col("df1.table_type").cast("string")) == "TGT") |
        (upper(col("df1.table_type").cast("string")) == "TGT_TRNS") |
        (upper(col("df1.table_type").cast("string")) == "FILE")
    ) &
    (
        (upper(col("df2.table_type").cast("string")) == "SRC") |
        (upper(col("df2.table_type").cast("string")) == "SRC_TRNS") |
        (upper(col("df2.table_type").cast("string")) == "FILE")
    )
)

stream_stream_dependency_df = src_tgt_dependencies.select(
    col("df1.stream_name").alias("from"),
    col("df2.stream_name").alias("to"),
    col("df1.table_name").alias("table"),
    lit('dependent').alias('relation'),
)

display(stream_stream_dependency_df)

# COMMAND ----------

# DBTITLE 1,Analysis: Stream-stream dependency count
stream_stream_dependency_df.count()

# COMMAND ----------

# Group by 'from' and 'to', count occurrences, and compute weights
weighted_stream_stream_dependency_df = stream_stream_dependency_df.groupBy('from', 'to').count().select('from', 'to', col('count').alias('weight'))
display(weighted_stream_stream_dependency_df)

#<ToDo> Add table size into weight calculation

#<ToDo> CLarify why ARCHIVE is used as a source stream (read that is read from) / which usecases use this? -> This should be only archive to archive. Check if there are any other instances. This also could be the case some cleanup is happening.

# RESULT: There are duplicated archivals

# COMMAND ----------

# DBTITLE 1,Merge bidirectional dependencies with summed weights
from pyspark.sql.functions import when, array, sort_array

# Step 1: Identify bidirectional pairs and sum their weights, keeping only one direction
bidirectional_merged = (
    weighted_stream_stream_dependency_df.alias("forward")
    .join(
        weighted_stream_stream_dependency_df.alias("backward"),
        (col("forward.from") == col("backward.to")) & 
        (col("forward.to") == col("backward.from"))
    )
    .select(
        # If forward.weight > backward.weight, use forward.from/to, else use backward.from/to
        when(col("forward.weight") > col("backward.weight"), col("forward.from"))
            .otherwise(col("backward.from")).alias("streamA"),
        when(col("forward.weight") > col("backward.weight"), col("forward.to"))
            .otherwise(col("backward.to")).alias("streamB"),
        (col("forward.weight") + col("backward.weight")).alias("weight"),
        sort_array(array(col("forward.from"), col("forward.to"))).alias("pair_key")
    )
    .dropDuplicates(["pair_key"])  # Only keep one row per bidirectional pair
    .select("streamA", "streamB", "weight")
)

print(f"Bidirectional pairs (merged): {bidirectional_merged.count()}")

# Step 2: Identify unidirectional relationships (no reverse edge exists)
unidirectional = (
    weighted_stream_stream_dependency_df.alias("forward")
    .join(
        weighted_stream_stream_dependency_df.alias("backward"),
        (col("forward.from") == col("backward.to")) & 
        (col("forward.to") == col("backward.from")),
        "left_anti"  # Keep only rows from forward that don't have a match in backward
    )
    .select(
        col("forward.from").alias("streamA"),
        col("forward.to").alias("streamB"),
        col("forward.weight").alias("weight")
    )
)

print(f"Unidirectional relationships: {unidirectional.count()}")

# Step 3: Combine bidirectional (merged) and unidirectional
merged_dependency_df = bidirectional_merged.union(unidirectional)

print(f"\nTotal edges after merging: {merged_dependency_df.count()}")
print(f"Original edges: {weighted_stream_stream_dependency_df.count()}")
print(f"Reduction: {weighted_stream_stream_dependency_df.count() - merged_dependency_df.count()} edges")

display(merged_dependency_df.orderBy("streamA", "streamB"))

# COMMAND ----------

# DBTITLE 1,Analysis: Src Streams for INT_IDM_INVENTORY Stream. To Verify Correctness
# display(stream_stream_dependency_df.filter((col("to") == "INT_IDM_INVENTORY")).sort("from"))

display(merged_dependency_df.filter((col("streamA") == "INT_IDM_REF_PRODUCT_COSMA")|(col("streamB") == "INT_IDM_REF_PRODUCT_COSMA")).sort("streamA","streamB"))
# Result -> Verified with Micheal for 2 streams.

# COMMAND ----------

# DBTITLE 1,Analysis: Non IMP_INV* Src Streams for INT_IDM_INVENTORY stream. To Verify Correctness
display(stream_stream_dependency_df.filter((col("to") == "INT_IDM_INVENTORY") & ~col("from").contains("IMP_INVENTORY")).select("from","table").distinct().sort("from"))

# COMMAND ----------

# output = output.union(depend_df)
# # Show the output
# result = output.groupBy('from','to').count().select('from','to',col('count').alias('weight'))
# result1 = result.filter(col('from') != col('to'))
# edges = result1.toPandas()
# edges = edges.groupby(['from', 'to'])['weight'].sum().reset_index()
# G_directed = nx.from_pandas_edgelist(edges, source='from', target='to',edge_attr=True, create_using=nx.DiGraph())

# COMMAND ----------

# DBTITLE 1,Analysis: Total number of stream to stream connections
merged_dependency_df.filter(col("weight")<=3).count()

# Total number of connecttions = 3800
# Stream - stream dependecies where weight <= 3, aka only 3 or less tables has connection between 2 streams = 2861
# ~75% stream - stream connections are loosely coupled


# COMMAND ----------

# Only use for small data!
edges_df = merged_dependency_df.toPandas()
edges_df.to_csv("/Volumes/users/jince_james/lufthansa/community_detection/edges.csv", index=False)

# COMMAND ----------

# Aggregate weights if necessary (depends on whether the grouping already handles this)
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


# Build an undirected weighted igraph graph from the edge list
# edges is a pandas DataFrame with columns:
#   - streamA: source node label (string/int)
#   - streamB: target node label (string/int)
#   - weight: edge weight (numeric, strength of relationship)
#
# Graph.TupleList will:
#   - create vertices implicitly from unique labels in streamA/streamB
#   - create edges between them
#   - attach edge attribute "weight"
g = ig.Graph.TupleList(
    edges[["streamA", "streamB", "weight"]].itertuples(index=False, name=None),
    directed=False,
    edge_attrs=["weight"]
)

# Also build a NetworkX graph for plotting (separate from igraph/Leiden)
G = nx.from_pandas_edgelist(
    edges, "streamA", "streamB",
    edge_attr="weight",
    create_using=nx.Graph()
)

# Quick sanity summary check
print(g.summary())
print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


#Run Leiden using the RBConfiguration objective at a fixed resolution (gamma)
def run_leiden_rb(g, resolution, seed, weights="weight", n_iterations=100):
    """
    Run Leiden community detection on graph g using the RBConfiguration objective.

    Parameters
    ----------
    g : igraph.Graph
        The graph to cluster.
    resolution : float
        The resolution (gamma) parameter controlling community granularity.
        Higher -> more/smaller communities; lower -> fewer/larger communities.
    seed : int
        Random seed controlling stochastic aspects of Leiden (important for stability testing).
    weights : str or None
        Name of edge attribute to use as weights (default "weight").
    n_iterations : int
        Number of Leiden iterations. More iterations can improve quality but costs time.

    Returns
    -------
    dict with:
        - membership: community assignment per vertex
        - quality: value of RBConfiguration objective for this partition
        - diagnostics about community size distribution
    """

    # find_partition runs Leiden optimization for the chosen partition type/objective.
    # RBConfigurationVertexPartition = modularity-like objective with a resolution parameter.
    # seed=... is crucial: it ensures runs start from different random initializations across seeds.
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )

    # part.membership is a list of length |V|, giving cluster id for each vertex
    membership = np.array(part.membership, dtype=int)

    # Compute community sizes by counting how many nodes each community label got
    counts = np.bincount(membership)

    # Sort community sizes descending (largest community first)
    counts_sorted = np.sort(counts)[::-1]

    return {
        "resolution": resolution,
        "seed": seed,

        # number of communities found in this run
        "n_communities": len(counts),

        # fraction of all nodes contained in the largest community
        # (useful to detect "giant blob" solutions)
        "largest_comm_share": counts_sorted[0] / g.vcount(),

        # count of "tiny" communities (here <5 nodes), a fragmentation indicator
        "small_comms_lt5": int((counts < 5).sum()),

        # objective value for the partition (RBConfiguration quality)
        "quality": float(part.quality()),

        # raw node-to-community assignment
        "membership": membership,
    }


# Stability metric: average pairwise Adjusted Rand Index (ARI)
def stability_ari(memberships):
    """
    Compute average pairwise Adjusted Rand Index (ARI) across partitions.

    ARI compares two partitions of the same nodes:
      - 1.0: identical clustering
      - ~0: no better than random agreement

    Here we average ARI over all pairs of runs for a given resolution.
    """
    if len(memberships) < 2:
        return 1.0

    aris = []
    for i in range(len(memberships)):
        for j in range(i + 1, len(memberships)):
            aris.append(adjusted_rand_score(memberships[i], memberships[j]))

    return float(np.mean(aris)) if aris else 1.0


# Helper to map igraph membership -> (stream -> community)
# igraph vertex "name" will be the original labels from TupleList
igraph_names = np.array(g.vs["name"])

def membership_to_leiden_df(membership):
    return pd.DataFrame({
        "stream": igraph_names,
        "community": membership
    })


#Scan over resolutions (gamma) and many random seeds per resolution
#Resolution grid: controls granularity (higher -> more communities)
resolutions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0]

#Multiple random restarts at each resolution to test robustness
seeds = [1, 7, 43, 99, 123, 11, 28, 37, 45, 672, 42, 10, 100, 178, 2, 3, 5]

# Store one representative run per resolution (so plotting does not re-run Leiden)
# If ARI==1.0 at a resolution, any seed is equivalent; otherwise this still gives a consistent choice.
plot_seed = 42
rep_by_res = {}  # resolution -> dict returned by run_leiden_rb (includes membership, quality, etc.)

rows = []
for res in resolutions:
    # Run Leiden many times at the same resolution (different seeds)
    runs = [run_leiden_rb(g, res, s) for s in seeds]

    # Extract memberships for ARI stability computation
    memberships = [r["membership"] for r in runs]

    # Aggregate diagnostics across seeds for this resolution
    n_comms = [r["n_communities"] for r in runs]
    largest = [r["largest_comm_share"] for r in runs]
    small_lt5 = [r["small_comms_lt5"] for r in runs]
    quality = [r["quality"] for r in runs]

    rows.append({
        "resolution": res,

        # community count statistics across seeds at this resolution
        "n_communities_avg": np.mean(n_comms),
        "n_communities_min": np.min(n_comms),
        "n_communities_max": np.max(n_comms),

        # size/fragmentation diagnostics
        "largest_comm_share_avg": np.mean(largest),
        "small_comms_lt5_avg": np.mean(small_lt5),

        # average objective value across seeds
        "quality_avg": np.mean(quality),

        # average pairwise partition similarity across seeds
        "stability_ari": stability_ari(memberships),
    })

    # Keep a representative partition for plotting later (no re-run)
    rep = next((r for r in runs if r["seed"] == plot_seed), runs[0])
    rep_by_res[res] = rep


summary = pd.DataFrame(rows)

# Summary table
summary = summary.sort_values("resolution").reset_index(drop=True)

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
# MAGIC ## What the columns tell us (brief reminder)
# MAGIC
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
# MAGIC
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
# MAGIC This range represents the strongest evidence for a “natural” community structure in the graph.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Near-transition resolution: γ = 1.2**
# MAGIC - **8 communities**, fixed count
# MAGIC - **ARI ≈ 0.99**
# MAGIC - Minor node reassignment across seeds
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC Still highly stable, but approaching a structural transition. A small number of boundary nodes begin to move, though the global structure remains intact.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Degenerate regime: γ = 1.5**
# MAGIC - **10–12 communities** across seeds
# MAGIC - **ARI ≈ 0.86**
# MAGIC - Low fragmentation, but multiple distinct partitions
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC The solution landscape becomes **degenerate**:
# MAGIC - several competing near-optimal partitions exist,
# MAGIC - differences are no longer limited to a few nodes,
# MAGIC - no single partition can be considered uniquely “correct.”
# MAGIC
# MAGIC At this resolution, results should be treated as *one of several plausible clusterings*.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Higher resolution: γ = 1.8**
# MAGIC - **14–15 communities**
# MAGIC - **ARI ≈ 0.97**
# MAGIC - Instability is localized (mostly small splits/merges)
# MAGIC
# MAGIC **Interpretation:**  
# MAGIC Stability largely returns at a finer scale. Most node assignments are consistent; variability is confined to small communities. This resolution is defensible if fine granularity is required.
# MAGIC
# MAGIC ---
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
# MAGIC ## Overall conclusions
# MAGIC
# MAGIC - **Most stable and well-supported resolutions:**  
# MAGIC   **γ ≈ 0.5–1.0**  
# MAGIC   (perfect node-level stability and fixed community counts)
# MAGIC
# MAGIC - **Transition / ambiguity zone:**  
# MAGIC   **γ ≈ 1.5**  
# MAGIC   (multiple competing partitions; no unique solution)
# MAGIC
# MAGIC - **Fine-scale but fragmented structure:**  
# MAGIC   **γ ≥ 1.8**  
# MAGIC   (mostly stable, but many small communities)
# MAGIC
# MAGIC In summary, your graph exhibits a **clear, reproducible community structure at moderate resolutions**, with instability emerging only when forcing finer-grained partitions than the data naturally supports.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Recommended next steps
# MAGIC
# MAGIC - Select a representative γ in the **0.5–1.0 range** for a single, defensible partition.
# MAGIC - PLotting
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Plots

# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the bigger clustered graph for the chosen resolutions where each node is colored in the color of the community it has been clustered into. 
# MAGIC Results are saved under ./leiden_plots**

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def select_resolutions(
    summary: pd.DataFrame,
    ari_target: float = 1.0,
    max_largest_comm_share: float = 0.50,
    sort_by: str = "resolution",
):
    """
    Select resolutions from the summary table using simple filters.

    Parameters
    ----------
    summary : pd.DataFrame
    ari_target : float
        Keep resolutions whose stability_ari equals this value (default 1.0).
    max_largest_comm_share : float
        Keep resolutions with average largest community share <= this threshold (default 0.50).
    sort_by : str
        Column to sort the result by (default "resolution").

    Returns
    -------
    selected : pd.DataFrame
        Filtered summary.
    selected_resolutions : list[float]
        Resolution values to plot.
    """
    selected = summary[
        (summary["stability_ari"] == ari_target) &
        (summary["largest_comm_share_avg"] <= max_largest_comm_share)
    ].sort_values(sort_by)

    selected_resolutions = selected["resolution"].tolist()
    return selected, selected_resolutions


def precompute_layout(
    G: nx.Graph,
    seed: int = 42,
    k=None,
):
    """
    Precompute a single spring layout (so plots are comparable).

    Parameters
    ----------
    G : nx.Graph
        Graph to layout.
    seed : int
        Random seed for deterministic layout.
    k : float or None
        Spring layout distance parameter. None uses NetworkX default.

    Returns
    -------
    pos : dict
        Node -> (x, y) layout positions.
    """
    pos = nx.spring_layout(G, seed=seed, k=k)
    return pos


def precompute_edge_style(
    G: nx.Graph,
    weight_attr: str = "weight",
    width_min: float = 0.2,
    width_scale: float = 3.0,
    alpha_min: float = 0.05,
    alpha_scale: float = 0.45,
):
    """
    Precompute edge widths and alphas from edge weights.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edges will be styled.
    weight_attr : str
        Edge attribute name to use as weights.
    width_min, width_scale : float
        width = width_min + width_scale * normalized_weight
    alpha_min, alpha_scale : float
        alpha = alpha_min + alpha_scale * normalized_weight

    Returns
    -------
    edge_widths : np.ndarray
        Width per edge in the order of G.edges().
    edge_alphas : np.ndarray
        Alpha per edge in the order of G.edges().
    """
    w = np.array([G[u][v].get(weight_attr, 1.0) for u, v in G.edges()], dtype=float)

    # Handle empty graph or single-weight graph safely
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
    selected_resolutions,
    rep_by_res: dict,
    membership_to_leiden_df,
    pos: dict = None,
    edge_widths=None,
    edge_alphas=None,
    outdir: str = "leiden_plots",
    figsize=(30, 24),
    dpi: int = 300,
    label_fontsize: int = 6,
    node_size: int = 110,
    cmap=plt.cm.tab20,
    draw_labels: bool = True,
    save: bool = True,
    show: bool = True,
):
    """
    Plot each selected resolution using stored memberships.

    Parameters
    ----------
    G : nx.Graph
        Graph to plot (NetworkX).
    g_igraph : igraph.Graph
        Graph used for Leiden (only used here for g_igraph.vcount()).
    selected_resolutions : list
        Resolutions to plot.
    rep_by_res : dict
        resolution -> dict containing at least: "membership", "quality", "seed"
    membership_to_leiden_df : callable
        Function that maps membership -> DataFrame with columns ["stream","community"]
    pos : dict or None
        Node positions. If None, will compute nx.spring_layout(G, seed=42).
    edge_widths, edge_alphas : array-like or None
        If None, will compute from weights with defaults.
    outdir : str
        Output directory for PNGs.
    figsize, dpi, label_fontsize : plotting params
    node_size, cmap : node styling
    draw_labels : bool
        Whether to draw node labels.
    save, show : bool
        Save figures and/or display them.

    Returns
    -------
    outputs : list[str]
        Paths of saved files (may be empty if save=False).
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=None)

    if edge_widths is None or edge_alphas is None:
        edge_widths, edge_alphas = precompute_edge_style(G)

    os.makedirs(outdir, exist_ok=True)
    outputs = []

    for res in selected_resolutions:
        rep = rep_by_res[res]
        membership = rep["membership"]
        quality = rep["quality"]

        leiden_df = membership_to_leiden_df(membership)

        # Map stream -> community
        node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))

        # Color nodes by community id
        node_colors = [node_to_comm.get(n, -1) for n in G.nodes()]

        # Compute a few plot annotations
        counts = np.bincount(membership)
        n_comms = len(counts)
        largest_share = counts.max() / g_igraph.vcount()
        tiny_lt5 = int((counts < 5).sum())

        # --- Create figure ---
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_size,
            cmap=cmap,
            linewidths=0.0,
            ax=ax
        )

        # Draw edges (per-edge alpha by looping)
        for (u, v), lw, a in zip(G.edges(), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=float(lw),
                alpha=float(a),
                ax=ax
            )

        # Labels
        if draw_labels:
            nx.draw_networkx_labels(
                G, pos,
                font_size=label_fontsize,
                ax=ax
            )

        title = (
            f"Leiden (RBConfiguration) — resolution γ={res} (seed={rep['seed']})\n"
            f"#comms={n_comms}, largest_comm_share={largest_share:.3f}, "
            f"small_comms<5={tiny_lt5}, quality={quality:.2f}"
        )
        ax.set_title(title)
        ax.axis("off")

        if save:
            outfile = os.path.join(outdir, f"leiden_rb_gamma_{res}.png")
            plt.savefig(outfile, bbox_inches="tight")
            outputs.append(outfile)

        if show:
            plt.show()
        else:
            plt.close(fig)

        if save:
            print(f"Saved: {outputs[-1]}")

    return outputs


# Example usage (matches your current flow):

selected, selected_resolutions = select_resolutions(summary, ari_target=1.0, max_largest_comm_share=0.50)
print("Selected resolutions:", selected_resolutions)
print(selected[["resolution", "largest_comm_share_avg", "stability_ari", "n_communities_avg"]])

pos = precompute_layout(G, seed=42, k=None)
edge_widths, edge_alphas = precompute_edge_style(G)

plot_leiden_resolutions(
    G=G,
    g_igraph=g,
    selected_resolutions=selected_resolutions,
    rep_by_res=rep_by_res,
    membership_to_leiden_df=membership_to_leiden_df,
    pos=pos,
    edge_widths=edge_widths,
    edge_alphas=edge_alphas,
    outdir="leiden_plots",
    figsize=(30, 24),
    dpi=300,
    label_fontsize=6,
)


# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the subgraphs / clusters, by choosing the preferred resolution**

# COMMAND ----------



# Helper to build leiden_df from a stored membership array (no re-run)
def membership_to_leiden_df(membership, igraph_names):
    return pd.DataFrame({
        "stream": np.array(igraph_names),
        "community": np.array(membership, dtype=int),
    })


# Choose which solution to visualize (from rep_by_res only; no Leiden re-run)
def get_leiden_df(resolution, rep_by_res, igraph_names):
    """
    Get leiden_df for a chosen resolution from already-computed results (rep_by_res).
    """
    rep = rep_by_res[resolution]
    leiden_df = membership_to_leiden_df(rep["membership"], igraph_names)

    meta = {
        "resolution": float(rep.get("resolution", resolution)),
        "seed": int(rep["seed"]),
        "quality": float(rep.get("quality", np.nan)),
    }
    return leiden_df, meta


# Helper: scale edge widths/alphas for a given graph
def edge_style(subG, weight_attr="weight", min_w=0.2, max_w=3.2):
    if subG.number_of_edges() == 0:
        return [], []
    w = np.array([subG[u][v].get(weight_attr, 1.0) for u, v in subG.edges()], dtype=float)
    w_min, w_max = float(w.min()), float(w.max())
    widths = min_w + (max_w - min_w) * (w - w_min) / (w_max - w_min + 1e-9)
    alphas = 0.10 + 0.60 * (w - w_min) / (w_max - w_min + 1e-9)
    return widths, alphas


def plot_communities_induced_subgraphs(
    G,
    leiden_df,
    outdir="leiden_community_plots",
    layout_seed=42,
    layout_k=None,
    weight_attr="weight",
    figsize=(24, 18),
    dpi=220,
    node_size=220,
    font_size=8,
    cmap=plt.cm.tab20,
    show=True,
    save=True,
    filename_prefix="community",
):
    """
    Plot each community as its induced subgraph (and optionally save each figure).
    Uses an existing Leiden solution.
    """
    os.makedirs(outdir, exist_ok=True)

    # Map node -> community
    node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))

    # Keep only nodes that actually have a community label
    labeled_nodes = [n for n in G.nodes() if n in node_to_comm]

    # Precompute a global layout ONCE so every community plot is comparable
    pos_global = nx.spring_layout(G.subgraph(labeled_nodes), seed=layout_seed, k=layout_k)

    communities = sorted(leiden_df["community"].unique())
    saved_files = []

    for c in communities:
        # Nodes in this community
        comm_nodes = [n for n in labeled_nodes if node_to_comm[n] == c]

        # Induced subgraph: only edges between nodes in the community
        H = G.subgraph(comm_nodes).copy()

        print(f"Community {c}: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")

        # Use global positions (consistent across plots)
        pos = {n: pos_global[n] for n in H.nodes()}

        widths, alphas = edge_style(H, weight_attr=weight_attr)

        plt.figure(figsize=figsize, dpi=dpi)

        # Draw nodes
        nx.draw_networkx_nodes(
            H, pos,
            node_size=node_size,
            node_color=[c] * H.number_of_nodes(),
            cmap=cmap
        )

        # Draw edges with per-edge alpha
        for (u, v), lw, a in zip(H.edges(), widths, alphas):
            nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a))

        # Labels
        nx.draw_networkx_labels(H, pos, font_size=font_size)

        plt.title(f"Community {c} — nodes={H.number_of_nodes()} edges={H.number_of_edges()}", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        if save:
            outfile = os.path.join(outdir, f"{filename_prefix}_{c}.png")
            plt.savefig(outfile, bbox_inches="tight")
            saved_files.append(outfile)
            print(f"Saved: {outfile}")

        if show:
            plt.show()
        else:
            plt.close()

    return saved_files


# Example usage

resolution = 1.2
leiden_df, meta = get_leiden_df(resolution, rep_by_res, igraph_names)
print(meta)

saved = plot_communities_induced_subgraphs(
    G=G,
    leiden_df=leiden_df,
    outdir=f"leiden_community_plots_gamma_{resolution}",
    layout_seed=42,
    layout_k=None,
    show=True,
    save=True,
    filename_prefix=f"gamma_{resolution}_comm"
)
print("Saved files:")