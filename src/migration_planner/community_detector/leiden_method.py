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
import random
from datetime import datetime

%matplotlib inline
from networkx.algorithms.centrality import degree_centrality
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score

# Build an undirected weighted igraph graph from your edge list
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

# Quick sanity summary (number of vertices/edges, etc.)
print(g.summary())


# ------------------------------------------------------------
# 2) Run Leiden using the RBConfiguration objective at a fixed resolution (gamma)
# ------------------------------------------------------------
def run_leiden_rb(g, resolution, seed, weights="weight", n_iterations=10):
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
    # seed=... is crucial: it ensures runs differ across seeds (true stability check).
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

    # Compute community sizes by counting how many nodes got each community label
    # (np.bincount assumes labels are 0..K-1)
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


# ------------------------------------------------------------
# 3) Stability metric: average pairwise Adjusted Rand Index (ARI)
# ------------------------------------------------------------
def stability_ari(memberships):
    """
    Compute average pairwise Adjusted Rand Index (ARI) across partitions.

    ARI compares two partitions of the same nodes:
      - 1.0: identical clustering
      - ~0: no better than random agreement
      - <0: worse than random (possible, uncommon)

    Here we average ARI over all pairs of runs for a given resolution.
    """
    if len(memberships) < 2:
        return 1.0

    aris = []
    for i in range(len(memberships)):
        for j in range(i + 1, len(memberships)):
            aris.append(adjusted_rand_score(memberships[i], memberships[j]))

    return float(np.mean(aris)) if aris else 1.0


# ------------------------------------------------------------
# 4) Scan over resolutions (gamma) and many random seeds per resolution
# ------------------------------------------------------------
# Resolution grid: controls granularity (higher -> more communities)
resolutions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0]

# Multiple random restarts at each resolution to test robustness
seeds = [1, 7, 43, 99, 123, 11, 28, 37, 45, 672, 42, 10, 100, 178, 2, 3, 5]

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

import os
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Assumptions:
# - edges: pandas df with columns ["streamA", "streamB", "weight"]
# - summary: pandas df with columns:
#     ["resolution", "largest_comm_share_avg", "stability_ari", ...]
# ------------------------------------------------------------

# ----------------------------
# 1) Build graphs (igraph for Leiden, networkx for plotting)
# ----------------------------
g = ig.Graph.TupleList(
    edges[["streamA", "streamB", "weight"]].itertuples(index=False, name=None),
    directed=False,
    edge_attrs=["weight"]
)

G = nx.from_pandas_edgelist(
    edges, "streamA", "streamB",
    edge_attr="weight",
    create_using=nx.Graph()
)

print(g.summary())
print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ----------------------------
# 2) Leiden runner (RBConfiguration)
# ----------------------------
def run_leiden_rb_membership(g, resolution, seed=42, weights="weight", n_iterations=10):
    """
    Returns:
      - membership array of length |V| with community id per vertex
      - part object (optional use: quality, etc.)
    """
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    membership = np.array(part.membership, dtype=int)
    return membership, part

# ----------------------------
# 3) Select resolutions: ARI == 1.0 and largest blob <= 50%
# ----------------------------
selected = summary[
    (summary["stability_ari"] == 1.0) &
    (summary["largest_comm_share_avg"] <= 0.50)
].sort_values("resolution")

selected_resolutions = selected["resolution"].tolist()
print("Selected resolutions:", selected_resolutions)
print(selected[["resolution", "largest_comm_share_avg", "stability_ari", "n_communities_avg"]])

# ----------------------------
# 4) Precompute a single layout once (so plots are comparable)
# ----------------------------
# Note: for dense graphs, try k=0.15 or 0.2. For sparse, leave k=None.
pos = nx.spring_layout(G, seed=42, k=0.15)

# ----------------------------
# 5) Precompute edge styling once (weight -> width/alpha)
# ----------------------------
w = np.array([G[u][v].get("weight", 1.0) for u, v in G.edges()], dtype=float)

# Avoid divide-by-zero if all weights equal
w_min, w_max = float(w.min()), float(w.max())
den = (w_max - w_min) if (w_max - w_min) > 0 else 1.0

edge_widths = 0.2 + 3.0 * (w - w_min) / den          # ~0.2 .. 3.2
edge_alphas = 0.05 + 0.45 * (w - w_min) / den        # ~0.05 .. 0.5

# ----------------------------
# 6) Helper to map igraph membership -> (stream -> community)
# ----------------------------
# igraph vertex "name" will be the original labels from TupleList
igraph_names = np.array(g.vs["name"])
name_to_comm = None  # per resolution

def membership_to_leiden_df(membership):
    return pd.DataFrame({
        "stream": igraph_names,
        "community": membership
    })

# ----------------------------
# 7) Plot each selected resolution at high quality
# ----------------------------
outdir = "leiden_plots"
os.makedirs(outdir, exist_ok=True)

# Large canvas + high DPI output for zooming
FIGSIZE = (30, 24)      # inches (big)
DPI = 300               # high-res
LABEL_FONTSIZE = 6      # readable-ish; 380 nodes will still be dense

for res in selected_resolutions:
    membership, part = run_leiden_rb_membership(g, res, seed=42, n_iterations=10)
    leiden_df = membership_to_leiden_df(membership)

    # Map stream -> community
    node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))

    # Color nodes by community id
    node_colors = [node_to_comm.get(n, -1) for n in G.nodes()]

    # Compute a few plot annotations
    counts = np.bincount(membership)
    n_comms = len(counts)
    largest_share = counts.max() / g.vcount()
    tiny_lt5 = int((counts < 5).sum())
    quality = float(part.quality())

    # --- Create figure ---
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = plt.gca()

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=110,
        cmap=plt.cm.tab20,
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

    # Labels (will be crowded, but you requested zoomable readability)
    nx.draw_networkx_labels(
        G, pos,
        font_size=LABEL_FONTSIZE,
        ax=ax
    )

    title = (
        f"Leiden (RBConfiguration) — resolution γ={res}\n"
        f"#comms={n_comms}, largest_comm_share={largest_share:.3f}, small_comms<5={tiny_lt5}, quality={quality:.2f}"
    )
    ax.set_title(title)
    ax.axis("off")

    # Save and show
    outfile = os.path.join(outdir, f"leiden_rb_gamma_{res}.png")
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

    print(f"Saved: {outfile}")


# COMMAND ----------

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# edges: pd.DataFrame with streamA, streamB, weight
# leiden_df: pd.DataFrame with columns ["stream", "community"]

# --- Build full graph once
G = nx.from_pandas_edgelist(
    edges, "streamA", "streamB",
    edge_attr="weight",
    create_using=nx.Graph()
)

# NetworkX 3.x replacement for nx.info(G)
print(f"Graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")

# --- Map node -> community
node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))

# Keep only nodes that actually have a community label
labeled_nodes = [n for n in G.nodes() if n in node_to_comm]

# Precompute a global layout ONCE so every community plot is comparable
# (you can switch to a per-community layout if you prefer)
pos_global = nx.spring_layout(G.subgraph(labeled_nodes), seed=42)

# --- Helper: scale edge widths for a given graph
def edge_style(subG, min_w=0.2, max_w=3.2):
    if subG.number_of_edges() == 0:
        return [], []
    w = np.array([subG[u][v].get("weight", 1.0) for u, v in subG.edges()], dtype=float)
    w_min, w_max = float(w.min()), float(w.max())
    widths = min_w + (max_w - min_w) * (w - w_min) / (w_max - w_min + 1e-9)
    alphas = 0.10 + 0.60 * (w - w_min) / (w_max - w_min + 1e-9)
    return widths, alphas

# --- Plot each community as its own induced subgraph
communities = sorted(leiden_df["community"].unique())

for c in communities:
    # Nodes in this community
    comm_nodes = [n for n in labeled_nodes if node_to_comm.get(n) == c]

    # Induced subgraph: only edges between nodes in the community
    H = G.subgraph(comm_nodes).copy()

    # If you also want to show isolated nodes (no internal edges), this will still plot them.
    print(f"Community {c}: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")

    # Use global positions (consistent across plots) but only keep positions for these nodes
    pos = {n: pos_global[n] for n in H.nodes() if n in pos_global}

    # If a community has nodes not in pos (rare), fall back to a local layout
    if len(pos) != H.number_of_nodes():
        pos = nx.spring_layout(H, seed=42)

    widths, alphas = edge_style(H)

    # BIG + high DPI so you can zoom/read labels
    plt.figure(figsize=(24, 18), dpi=220)

    # Draw nodes
    nx.draw_networkx_nodes(
        H, pos,
        node_size=220,
        node_color=[c] * H.number_of_nodes(),  # single color per community (still uses cmap)
        cmap=plt.cm.tab20
    )

    # Draw edges with per-edge alpha
    if H.number_of_edges() > 0:
        for (u, v), lw, a in zip(H.edges(), widths, alphas):
            nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a))

    # Labels (bigger than your full-graph version since this is smaller)
    nx.draw_networkx_labels(H, pos, font_size=8)

    plt.title(f"Community {c} — nodes={H.number_of_nodes()} edges={H.number_of_edges()}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
