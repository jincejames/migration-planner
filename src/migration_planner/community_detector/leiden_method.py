# Databricks notebook source
# MAGIC %md
# MAGIC ## Libraries and Parameters
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install required libraries
# MAGIC %skip
# MAGIC !pip install python-igraph adjustText infomap netgraph networkx python-louvain igraph leidenalg
# MAGIC dbutils.library.restartPython()

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

# DBTITLE 1,Widgets
dbutils.widgets.text(
    "volume_name",
    "/Volumes/odp_adw_mvp_n/migration/utilities/community_detection/",
    "Input Volume Path"
)
dbutils.widgets.text(
    "input_dependency_name",
    "ETL-table-dependencies_20251223_1218.csv",
    "Input CSV file name"
)
dbutils.widgets.text(
    "outofscope_stream_file_name",
    "out-of-scopte-streams.csv",
    "Out of scope streams file name"
)
dbutils.widgets.text(
    "report_dependency_file_name",
    "stream_to_report_mapping_new.csv",
    "report Dependency file name"
)

dbutils.widgets.text(
    "table_size",
    "table-space-in-gb_20251201_1352.csv",
    "Table size file name"
)

# COMMAND ----------

# DBTITLE 1,Common variables & output directory operation
volume_path = dbutils.widgets.get("volume_name")
dependency_input_path = volume_path + dbutils.widgets.get("input_dependency_name")
outofscope_stream_path = volume_path + dbutils.widgets.get("outofscope_stream_file_name")
report_dependency = volume_path + dbutils.widgets.get("report_dependency_file_name")
table_size = volume_path + dbutils.widgets.get("table_size")

# Output path with date and hour
output_dir_name = "community_detection_output_" + datetime.now().strftime("%d%m%Y_%H")
output_path = volume_path + "community_detection_output_latest/" + output_dir_name + "/"

# Move existing folders from 'latest' one level up (to volume_path)
latest_path = volume_path + "community_detection_output_latest/"
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
dependency_df_full = spark.read.format("csv").option("header", "true").load(dependency_input_path)

# COMMAND ----------

# DBTITLE 1,Reading out of scope stream names
outofscope_stream_names_df = spark.read.format("csv").option("header","true").load(outofscope_stream_path).select(col("stream_name"))
outofscope_stream_names_rows_list = outofscope_stream_names_df.collect()
outofscope_stream_names_list = [x['stream_name'] for x in outofscope_stream_names_rows_list]

# COMMAND ----------

# DBTITLE 1,Reading report to table dependency
# Read report to stream dependency and standardize values and column names
# Each table is marked as a Src, since reports only create tables and do not write to tables (or only exceptions)
# Filtering two reports wrt corona and gdpr, since they skew the community creation due to high number of dependencies
report_dependency_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(report_dependency)
    .select(
        col("report_name").alias("stream_name"),
        upper(col("table_name")).alias("table_name"),
        lit("Src").alias('table_type'),
    )
    .filter(~lower(col("stream_name")).contains("corona") & ~lower(col("stream_name")).contains("gdpr"))
)

# COMMAND ----------

# DBTITLE 1,Reading table size in GB records
table_size_df = spark.read.format("csv").option("header","true").load(table_size).select(upper(col("DB_Table_Name")).alias("table_name"), col("SPACE_IN_GB").alias("size"))

# COMMAND ----------

# Removing streams associated with acrchiving, GDPR, housekeeping and out of scope streams
dependency_df_filtered = dependency_df_full.filter(
    ~upper(col("stream_name")).contains("ARCHIVE") & 
    ~upper(col("stream_name")).contains("GDPR") &
    ~upper(col("stream_name")).contains("HOUSEKEEPING") &
    ~upper(col("stream_name")).isin(outofscope_stream_names_list)
)

# COMMAND ----------

# DBTITLE 1,Considering all TGT tables as SRC
# Considering all TGT tables as SRC as well due to a gap in ODAT output
tgt_as_source = dependency_df_filtered.filter(upper(col('table_type')).contains("TGT")).replace({"Tgt" : "Src", "Tgt_Trns" : "Src_Trns"}, subset=["table_type"])
dependency_df = dependency_df_filtered.union(tgt_as_source).distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forming Table to Stream Dependecies

# COMMAND ----------

# DBTITLE 1,Filter out intra stream (self) dependency
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


# COMMAND ----------

# DBTITLE 1,Joining reports to the stream data
report_join_result = (
    report_dependency_df.alias("df2")
    .join(dependency_with_transactional_df.alias("df1"), upper(col("df1.table_name")) == upper(col("df2.table_name")))
)

# COMMAND ----------

# DBTITLE 1,Merging reports and streams
self_join_result_without_size = report_join_result.union(non_filtered_self_join_result)

# COMMAND ----------

# DBTITLE 1,Adding table size information
self_join_result = self_join_result_without_size.join(
    table_size_df.alias("table_size"),
    col("df1.table_name") == col("table_size.table_name")
).select(
    "df1.*",
    "df2.*",
    col("table_size.size")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forming Stream - Stream Dependencies

# COMMAND ----------

# DBTITLE 1,Forming Stream - Stream dependency
src_tgt_dependencies = self_join_result.filter(
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
    col("table_size.size").alias("size")
)

stream_stream_dependency_df.toPandas().to_csv(output_path + "stream_stream_dependencies.csv", index=False)

# COMMAND ----------

# DBTITLE 1,Option 1 - Weight Calculation
# For each (from, to, table), compute table_weight = ceil(size/100), min 1
# If "to" contains "json", set table_weight = 2 regardless of size, this is to give slightly high preference to reports
table_weight_df = stream_stream_dependency_df.withColumn(
    "table_weight",
    when(
        lower(col("to")).contains("json"),
        2
    ).otherwise(
        ceil((col("size").cast("double") / 100)).cast("int")
    )
).withColumn(
    "table_weight",
    when(col("table_weight") < 1, 1).otherwise(col("table_weight"))
)

# Remove duplicate (from, to, table) combinations
unique_table_weights = table_weight_df.dropDuplicates(["from", "to", "table"])

# Group by (from, to), sum the table_weight as the edge weight
weighted_stream_stream_dependency_df = unique_table_weights.groupBy("from", "to").agg(
    sum("table_weight").alias("weight")
)

display(weighted_stream_stream_dependency_df)


# COMMAND ----------

# DBTITLE 1,[Backup] Option 2 - Scaling and weight calculation
# MAGIC %skip
# MAGIC from pyspark.sql.functions import (
# MAGIC     col, when, lower, lit,
# MAGIC     min as spark_min, max as spark_max, sum as spark_sum,
# MAGIC     log1p
# MAGIC )
# MAGIC
# MAGIC # ---- CONFIG ----
# MAGIC SCALED_MIN = 0.0001  # Minimum weight to ensure non-zero weights for existing connections
# MAGIC SCALED_MAX = 1.0
# MAGIC TABLEAU_MULTIPLIER = 1.5
# MAGIC USE_LOG = True   # <-- set False to disable log transform
# MAGIC
# MAGIC # 1) Raw per-table value = size
# MAGIC df = stream_stream_dependency_df.withColumn("raw_table_weight", col("size").cast("double"))
# MAGIC
# MAGIC # (optional) guard against null/negative sizes (log1p requires >= 0)
# MAGIC df = df.withColumn(
# MAGIC     "raw_table_weight",
# MAGIC     when(col("raw_table_weight").isNull() | (col("raw_table_weight") < 0), lit(0.0))
# MAGIC     .otherwise(col("raw_table_weight"))
# MAGIC )
# MAGIC
# MAGIC # 2) Optional log transform (log1p)
# MAGIC if USE_LOG:
# MAGIC     df = df.withColumn("raw_for_scaling", log1p(col("raw_table_weight")))
# MAGIC else:
# MAGIC     df = df.withColumn("raw_for_scaling", col("raw_table_weight"))
# MAGIC
# MAGIC # 3) Min–Max stats on the scaling column
# MAGIC stats = df.agg(
# MAGIC     spark_min("raw_for_scaling").alias("mn"),
# MAGIC     spark_max("raw_for_scaling").alias("mx")
# MAGIC ).collect()[0]
# MAGIC
# MAGIC mn = float(stats["mn"])
# MAGIC mx = float(stats["mx"])
# MAGIC den = (mx - mn) if (mx - mn) != 0 else 1.0
# MAGIC
# MAGIC # 4) Min–Max scale into [0, 1] then map to [SCALED_MIN, SCALED_MAX]
# MAGIC df = df.withColumn("mm01", (col("raw_for_scaling") - lit(mn)) / lit(den))
# MAGIC
# MAGIC df = df.withColumn(
# MAGIC     "scaled_table_weight",
# MAGIC     lit(SCALED_MIN) + (lit(SCALED_MAX - SCALED_MIN) * col("mm01"))
# MAGIC )
# MAGIC
# MAGIC # Ensure scaled_table_weight is never less than SCALED_MIN (guard against float error)
# MAGIC df = df.withColumn(
# MAGIC     "scaled_table_weight",
# MAGIC     when(col("scaled_table_weight") < SCALED_MIN, lit(SCALED_MIN)).otherwise(col("scaled_table_weight"))
# MAGIC )
# MAGIC
# MAGIC # 5) Tableau/BI report boost AFTER scaling
# MAGIC df = df.withColumn(
# MAGIC     "scaled_table_weight",
# MAGIC     when(lower(col("to")).contains("json"),
# MAGIC          col("scaled_table_weight") * lit(TABLEAU_MULTIPLIER)
# MAGIC     ).otherwise(col("scaled_table_weight"))
# MAGIC )
# MAGIC
# MAGIC # 6) Drop duplicates per (from, to, table)
# MAGIC unique_table_weights = df.dropDuplicates(["from", "to", "table"])
# MAGIC
# MAGIC # 7) Aggregate stream–stream edge weights
# MAGIC weighted_stream_stream_dependency_df = (
# MAGIC     unique_table_weights
# MAGIC     .groupBy("from", "to")
# MAGIC     .agg(spark_sum("scaled_table_weight").alias("weight"))
# MAGIC )
# MAGIC
# MAGIC # 8) Final safeguard: ensure aggregated weight is never 0 for existing connections
# MAGIC weighted_stream_stream_dependency_df = weighted_stream_stream_dependency_df.withColumn(
# MAGIC     "weight",
# MAGIC     when(col("weight") < SCALED_MIN, lit(SCALED_MIN)).otherwise(col("weight"))
# MAGIC )
# MAGIC
# MAGIC display(weighted_stream_stream_dependency_df)

# COMMAND ----------

# DBTITLE 1,Merge bidirectional dependencies with summed weights

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

# COMMAND ----------

# DBTITLE 1,Create Edges of the graph in Pandas
edges_df = merged_dependency_df.toPandas()
edges_df.to_csv(f"{output_path}edges.csv", index=False)

# COMMAND ----------

# DBTITLE 1,Aggregate weights - safety step
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

# DBTITLE 1,Community Detection

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
resolutions = [0.4, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.6]

#Multiple random restarts at each resolution to test robustness
seeds = [1, 3, 7, 43, 99, 123, 11, 28, 37, 45, 672, 42, 10, 100, 178]

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

# MAGIC %skip
# MAGIC from adjustText import adjust_text
# MAGIC
# MAGIC def select_resolutions(
# MAGIC     summary: pd.DataFrame,
# MAGIC     ari_target: float = 0.93,
# MAGIC     max_largest_comm_share: float = 0.2,
# MAGIC     sort_by: str = "resolution",
# MAGIC ):
# MAGIC     """
# MAGIC     Select resolutions from the summary table using simple filters.
# MAGIC
# MAGIC     Parameters
# MAGIC     ----------
# MAGIC     summary : pd.DataFrame
# MAGIC     ari_target : float
# MAGIC         Keep resolutions whose stability_ari equals this value (default 0.93).
# MAGIC     max_largest_comm_share : float
# MAGIC         Keep resolutions with average largest community share <= this threshold (default 0.20).
# MAGIC     sort_by : str
# MAGIC         Column to sort the result by (default "resolution").
# MAGIC
# MAGIC     Returns
# MAGIC     -------
# MAGIC     selected : pd.DataFrame
# MAGIC         Filtered summary.
# MAGIC     selected_resolutions : list[float]
# MAGIC         Resolution values to plot.
# MAGIC     """
# MAGIC     selected = summary[
# MAGIC         (summary["stability_ari"] >= ari_target) &
# MAGIC         (summary["largest_comm_share_avg"] <= max_largest_comm_share)
# MAGIC     ].sort_values(sort_by)
# MAGIC
# MAGIC     selected_resolutions = selected["resolution"].tolist()
# MAGIC     return selected, selected_resolutions
# MAGIC
# MAGIC
# MAGIC def precompute_layout(
# MAGIC     G: nx.Graph,
# MAGIC     seed: int = 42,
# MAGIC     k=3.0,  # Increased for more spacing
# MAGIC ):
# MAGIC     """
# MAGIC     Precompute a single spring layout (so plots are comparable).
# MAGIC
# MAGIC     Parameters
# MAGIC     ----------
# MAGIC     G : nx.Graph
# MAGIC         Graph to layout.
# MAGIC     seed : int
# MAGIC         Random seed for deterministic layout.
# MAGIC     k : float or None
# MAGIC         Spring layout distance parameter. Higher = more spacing.
# MAGIC
# MAGIC     Returns
# MAGIC     -------
# MAGIC     pos : dict
# MAGIC         Node -> (x, y) layout positions.
# MAGIC     """
# MAGIC     print(f"Computing layout with k={k} for better node spacing...")
# MAGIC     pos = nx.spring_layout(G, seed=seed, k=k, iterations=100)
# MAGIC     return pos
# MAGIC
# MAGIC
# MAGIC def precompute_edge_style(
# MAGIC     G: nx.Graph,
# MAGIC     weight_attr: str = "weight",
# MAGIC     width_min: float = 0.2,
# MAGIC     width_scale: float = 3.0,
# MAGIC     alpha_min: float = 0.05,
# MAGIC     alpha_scale: float = 0.45,
# MAGIC ):
# MAGIC     """
# MAGIC     Precompute edge widths and alphas from edge weights.
# MAGIC
# MAGIC     Parameters
# MAGIC     ----------
# MAGIC     G : nx.Graph
# MAGIC         Graph whose edges will be styled.
# MAGIC     weight_attr : str
# MAGIC         Edge attribute name to use as weights.
# MAGIC     width_min, width_scale : float
# MAGIC         width = width_min + width_scale * normalized_weight
# MAGIC     alpha_min, alpha_scale : float
# MAGIC         alpha = alpha_min + alpha_scale * normalized_weight
# MAGIC
# MAGIC     Returns
# MAGIC     -------
# MAGIC     edge_widths : np.ndarray
# MAGIC         Width per edge in the order of G.edges().
# MAGIC     edge_alphas : np.ndarray
# MAGIC         Alpha per edge in the order of G.edges().
# MAGIC     """
# MAGIC     w = np.array([G[u][v].get(weight_attr, 1.0) for u, v in G.edges()], dtype=float)
# MAGIC
# MAGIC     # Handle empty graph or single-weight graph safely
# MAGIC     if w.size == 0:
# MAGIC         return np.array([]), np.array([])
# MAGIC
# MAGIC     w_min, w_max = float(w.min()), float(w.max())
# MAGIC     den = (w_max - w_min) if (w_max - w_min) > 0 else 1.0
# MAGIC     w_norm = (w - w_min) / den
# MAGIC
# MAGIC     edge_widths = width_min + width_scale * w_norm
# MAGIC     edge_alphas = alpha_min + alpha_scale * w_norm
# MAGIC     return edge_widths, edge_alphas
# MAGIC
# MAGIC
# MAGIC def plot_leiden_resolutions(
# MAGIC     G: nx.Graph,
# MAGIC     g_igraph,
# MAGIC     selected_resolutions,
# MAGIC     rep_by_res: dict,
# MAGIC     membership_to_leiden_df,
# MAGIC     pos: dict = None,
# MAGIC     edge_widths=None,
# MAGIC     edge_alphas=None,
# MAGIC     outdir: str = "leiden_plots",
# MAGIC     figsize=(50, 40),  # Much larger figure
# MAGIC     dpi: int = 400,  # Higher resolution
# MAGIC     label_fontsize: int = 11,  # Larger font
# MAGIC     node_size: int = 800,  # Larger nodes
# MAGIC     cmap=plt.cm.tab20,
# MAGIC     draw_labels: bool = True,
# MAGIC     save: bool = True,
# MAGIC     show: bool = True,
# MAGIC     use_adjust_text: bool = True,  # Use adjustText for label positioning
# MAGIC ):
# MAGIC     """
# MAGIC     Plot each selected resolution using stored memberships.
# MAGIC
# MAGIC     Parameters
# MAGIC     ----------
# MAGIC     G : nx.Graph
# MAGIC         Graph to plot (NetworkX).
# MAGIC     g_igraph : igraph.Graph
# MAGIC         Graph used for Leiden (only used here for g_igraph.vcount()).
# MAGIC     selected_resolutions : list
# MAGIC         Resolutions to plot.
# MAGIC     rep_by_res : dict
# MAGIC         resolution -> dict containing at least: "membership", "quality", "seed"
# MAGIC     membership_to_leiden_df : callable
# MAGIC         Function that maps membership -> DataFrame with columns ["stream","community"]
# MAGIC     pos : dict or None
# MAGIC         Node positions. If None, will compute nx.spring_layout(G, seed=42).
# MAGIC     edge_widths, edge_alphas : array-like or None
# MAGIC         If None, will compute from weights with defaults.
# MAGIC     outdir : str
# MAGIC         Output directory for PNGs.
# MAGIC     figsize, dpi, label_fontsize : plotting params
# MAGIC     node_size, cmap : node styling
# MAGIC     draw_labels : bool
# MAGIC         Whether to draw node labels.
# MAGIC     save, show : bool
# MAGIC         Save figures and/or display them.
# MAGIC     use_adjust_text : bool
# MAGIC         Use adjustText library to prevent label overlap.
# MAGIC
# MAGIC     Returns
# MAGIC     -------
# MAGIC     outputs : list[str]
# MAGIC         Paths of saved files (may be empty if save=False).
# MAGIC     """
# MAGIC     if pos is None:
# MAGIC         pos = nx.spring_layout(G, seed=42, k=3.0, iterations=100)
# MAGIC
# MAGIC     if edge_widths is None or edge_alphas is None:
# MAGIC         edge_widths, edge_alphas = precompute_edge_style(G)
# MAGIC
# MAGIC     os.makedirs(outdir, exist_ok=True)
# MAGIC     outputs = []
# MAGIC
# MAGIC     for res in selected_resolutions:
# MAGIC         print(f"\nGenerating plot for resolution γ={res}...")
# MAGIC         rep = rep_by_res[res]
# MAGIC         membership = rep["membership"]
# MAGIC         quality = rep["quality"]
# MAGIC
# MAGIC         leiden_df = membership_to_leiden_df(membership)
# MAGIC
# MAGIC         # Map stream -> community
# MAGIC         node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))
# MAGIC
# MAGIC         # Color nodes by community id
# MAGIC         node_colors = [node_to_comm.get(n, -1) for n in G.nodes()]
# MAGIC
# MAGIC         # Compute a few plot annotations
# MAGIC         counts = np.bincount(membership)
# MAGIC         n_comms = len(counts)
# MAGIC         largest_share = counts.max() / g_igraph.vcount()
# MAGIC         tiny_lt5 = int((counts < 5).sum())
# MAGIC
# MAGIC         # --- Create figure ---
# MAGIC         fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
# MAGIC
# MAGIC         # Draw edges first (behind nodes)
# MAGIC         print(f"  Drawing {G.number_of_edges()} edges...")
# MAGIC         for (u, v), lw, a in zip(G.edges(), edge_widths, edge_alphas):
# MAGIC             nx.draw_networkx_edges(
# MAGIC                 G, pos,
# MAGIC                 edgelist=[(u, v)],
# MAGIC                 width=float(lw),
# MAGIC                 alpha=float(a),
# MAGIC                 ax=ax,
# MAGIC                 edge_color='gray'
# MAGIC             )
# MAGIC
# MAGIC         # Draw nodes
# MAGIC         print(f"  Drawing {G.number_of_nodes()} nodes...")
# MAGIC         nx.draw_networkx_nodes(
# MAGIC             G, pos,
# MAGIC             node_color=node_colors,
# MAGIC             node_size=node_size,
# MAGIC             cmap=cmap,
# MAGIC             linewidths=2.0,
# MAGIC             edgecolors='black',
# MAGIC             ax=ax
# MAGIC         )
# MAGIC
# MAGIC         # Draw labels
# MAGIC         if draw_labels:
# MAGIC             if use_adjust_text:
# MAGIC                 print(f"  Adding labels with overlap prevention...")
# MAGIC                 texts = []
# MAGIC                 for node, (x, y) in pos.items():
# MAGIC                     texts.append(
# MAGIC                         ax.text(
# MAGIC                             x, y, str(node),
# MAGIC                             fontsize=label_fontsize,
# MAGIC                             ha='center',
# MAGIC                             va='center',
# MAGIC                             fontweight='bold',
# MAGIC                             bbox=dict(
# MAGIC                                 boxstyle='round,pad=0.4',
# MAGIC                                 facecolor='white',
# MAGIC                                 edgecolor='gray',
# MAGIC                                 linewidth=0.5,
# MAGIC                                 alpha=0.85
# MAGIC                             )
# MAGIC                         )
# MAGIC                     )
# MAGIC                 
# MAGIC                 # Adjust text positions to avoid overlap
# MAGIC                 adjust_text(
# MAGIC                     texts,
# MAGIC                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6),
# MAGIC                     expand_points=(2.0, 2.0),
# MAGIC                     expand_text=(1.5, 1.5),
# MAGIC                     force_points=(0.8, 0.8),
# MAGIC                     force_text=(0.8, 0.8),
# MAGIC                     ax=ax,
# MAGIC                     lim=500
# MAGIC                 )
# MAGIC             else:
# MAGIC                 # Standard labels with background
# MAGIC                 for node, (x, y) in pos.items():
# MAGIC                     ax.text(
# MAGIC                         x, y, str(node),
# MAGIC                         fontsize=label_fontsize,
# MAGIC                         ha='center',
# MAGIC                         va='center',
# MAGIC                         fontweight='bold',
# MAGIC                         bbox=dict(
# MAGIC                             boxstyle='round,pad=0.4',
# MAGIC                             facecolor='white',
# MAGIC                             edgecolor='gray',
# MAGIC                             linewidth=0.5,
# MAGIC                             alpha=0.85
# MAGIC                         )
# MAGIC                     )
# MAGIC
# MAGIC         title = (
# MAGIC             f"Leiden (RBConfiguration) — resolution γ={res} (seed={rep['seed']})\n"
# MAGIC             f"#communities={n_comms}, largest_comm_share={largest_share:.3f}, "
# MAGIC             f"small_comms<5={tiny_lt5}, quality={quality:.2f}"
# MAGIC         )
# MAGIC         ax.set_title(title, fontsize=22, fontweight='bold', pad=30)
# MAGIC         ax.axis("off")
# MAGIC         ax.margins(0.1)  # Add margin around plot
# MAGIC         
# MAGIC         plt.tight_layout()
# MAGIC
# MAGIC         if save:
# MAGIC             outfile = os.path.join(outdir, f"leiden_rb_gamma_{res}.png")
# MAGIC             print(f"  Saving high-resolution plot...")
# MAGIC             plt.savefig(outfile, bbox_inches="tight", dpi=dpi, facecolor='white')
# MAGIC             outputs.append(outfile)
# MAGIC             print(f"  Saved: {outfile}")
# MAGIC
# MAGIC         if show:
# MAGIC             plt.show()
# MAGIC         else:
# MAGIC             plt.close(fig)
# MAGIC
# MAGIC     return outputs
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC selected, selected_resolutions = select_resolutions(summary, ari_target=0.93, max_largest_comm_share=0.25)
# MAGIC print("Selected resolutions:", selected_resolutions)
# MAGIC print(selected[["resolution", "largest_comm_share_avg", "stability_ari", "n_communities_avg"]])
# MAGIC
# MAGIC print("\nGenerating high-resolution plots with improved readability...")
# MAGIC print("This may take several minutes depending on graph size.\n")
# MAGIC
# MAGIC # Ensure we're using the volume path from the widget parameter
# MAGIC output_dir = os.path.join(output_path, "leiden_plots")
# MAGIC print(f"Output directory: {output_dir}")
# MAGIC print(f"Saving plots to volume location (not git folder)\n")
# MAGIC
# MAGIC pos = precompute_layout(G, seed=42, k=3.0)  # More spacing
# MAGIC edge_widths, edge_alphas = precompute_edge_style(G)
# MAGIC
# MAGIC plot_leiden_resolutions(
# MAGIC     G=G,
# MAGIC     g_igraph=g,
# MAGIC     selected_resolutions=selected_resolutions,
# MAGIC     rep_by_res=rep_by_res,
# MAGIC     membership_to_leiden_df=membership_to_leiden_df,
# MAGIC     pos=pos,
# MAGIC     edge_widths=edge_widths,
# MAGIC     edge_alphas=edge_alphas,
# MAGIC     outdir=output_dir,  # Using volume path
# MAGIC     figsize=(50, 40),  # Very large figure
# MAGIC     dpi=400,  # High resolution
# MAGIC     label_fontsize=11,  # Larger font
# MAGIC     node_size=800,  # Larger nodes
# MAGIC     draw_labels=True,
# MAGIC     save=True,
# MAGIC     show=False,
# MAGIC     use_adjust_text=True,  # Prevent label overlap
# MAGIC )
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC print("COMPLETE: All plots saved with high resolution (400 DPI)")
# MAGIC print(f"Output directory: {output_dir}")
# MAGIC print("Plots can be zoomed in with full clarity.")
# MAGIC print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the subgraphs / clusters, by choosing the preferred resolution**

# COMMAND ----------

# DBTITLE 1,Analysis & Plotting
# Helper to build leiden_df from a stored membership array
def membership_to_leiden_df(membership, igraph_names):
    return pd.DataFrame(
        {
            "stream": np.array(igraph_names),
            "community": np.array(membership, dtype=int),
        }
    )


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


# Helper: scale edge widths/alphas for a given graph
def edge_style(subG, weight_attr="weight", min_w=0.2, max_w=3.2):
    if subG.number_of_edges() == 0:
        return [], []
    w = np.array([subG[u][v].get(weight_attr, 1.0) for u, v in subG.edges()], dtype=float)
    w_min, w_max = float(w.min()), float(w.max())
    widths = min_w + (max_w - min_w) * (w - w_min) / (w_max - w_min + 1e-9)
    alphas = 0.10 + 0.60 * (w - w_min) / (w_max - w_min + 1e-9)
    return widths, alphas


def plot_communities_with_analysis_safe(
    G,
    leiden_df,
    stream_table_dependency_df,
    merged_edges_df,
    resolution,
    outdir="leiden_community_plots",
    layout_seed=42,
    layout_k=None,
    layout_iterations=30,
    weight_attr="weight",
    figsize=(24, 18),
    dpi=220,
    node_size=220,
    font_size=8,
    cmap=plt.cm.tab20,
    show=False,
    save=True,
    filename_prefix="community",
    max_labels=120,
    enable_plotting=False,
):

    os.makedirs(outdir, exist_ok=True)

    node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))
    labeled_nodes = [n for n in G.nodes() if n in node_to_comm]

    # Global layout (only compute if plotting is enabled)
    pos_global = None
    if enable_plotting:
        pos_global = nx.spring_layout(
            G.subgraph(labeled_nodes), seed=layout_seed, k=layout_k, iterations=layout_iterations
        )

    stream_table_pdf = stream_table_dependency_df.toPandas().copy()
    merged_edges_pdf = merged_edges_df.toPandas().copy()

    communities = sorted(leiden_df["community"].unique())
    saved_files = []

    # Convert DataFrames to pandas for analysis
    stream_table_pdf = stream_table_dependency_df.toPandas()
    merged_edges_pdf = merged_edges_df.toPandas()

    for c in communities:
        comm_dir = os.path.join(outdir, f"community_{c}")
        os.makedirs(comm_dir, exist_ok=True)

        comm_nodes = [n for n in labeled_nodes if node_to_comm[n] == c]
        streams_in_community = set(comm_nodes)

        if enable_plotting:
            H = G.subgraph(comm_nodes).copy()
            print(f"Community {c}: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")

        # ---- ANALYSIS (always runs regardless of plotting) ----

        # === Generate Analysis ===

        # 1. List of streams in community
        streams_list = sorted(list(streams_in_community))

        # 2. Find tables that are SRC of streams inside but TGT of streams outside
        # In stream_table_dependency_df:
        #   - 'from' stream writes to 'table' (table is TGT of 'from' stream)
        #   - 'to' stream reads from 'table' (table is SRC of 'to' stream)
        # So we want: 'from' OUTSIDE community AND 'to' INSIDE community

        # Tables that are SRC of inside streams, TGT of outside streams
        # (produced outside, consumed inside)
        tables_src_inside_tgt_outside = stream_table_pdf[
            (~stream_table_pdf['from'].isin(streams_in_community)) & (stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].drop_duplicates(subset=['table'])

        # Tables that are TGT of inside streams, SRC of outside streams
        # (produced inside, consumed outside)
        tables_tgt_inside_src_outside = stream_table_pdf[
            (stream_table_pdf['from'].isin(streams_in_community)) & (~stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].drop_duplicates(subset=['table'])

        # Get aggregated edge connections (for summary)
        outgoing_edges = merged_edges_pdf[
            (merged_edges_pdf['streamA'].isin(streams_in_community))
            & (~merged_edges_pdf['streamB'].isin(streams_in_community))
        ]

        incoming_edges = merged_edges_pdf[
            (~merged_edges_pdf['streamA'].isin(streams_in_community))
            & (merged_edges_pdf['streamB'].isin(streams_in_community))
        ]

        # Create analysis output
        analysis_content = f"""Community {c} Analysis (Resolution γ={resolution})
{'='*80}

1. STREAMS IN COMMUNITY ({len(streams_list)} streams):
{'-'*80}
"""
        for i, stream in enumerate(streams_list, 1):
            analysis_content += f"{i}. {stream}\n"

        analysis_content += f"""\n2. TABLES - SRC OF STREAMS INSIDE, TGT OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community READ FROM, but are WRITTEN BY streams outside.
(Dependencies flowing INTO the community)
Total: {len(tables_src_inside_tgt_outside)} unique tables\n\n"""

        if len(tables_src_inside_tgt_outside) > 0:
            for idx, row in tables_src_inside_tgt_outside.iterrows():
                analysis_content += f"  Table: {row['table']}\n"
                analysis_content += f"    - Written by (outside): {row['from']}\n"
                analysis_content += f"    - Read by (inside): {row['to']}\n"
                analysis_content += f"    - Size: {row['size']} GB\n\n"
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""\n3. TABLES - TGT OF STREAMS INSIDE, SRC OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community WRITE TO, but are READ BY streams outside.
(Dependencies flowing OUT OF the community)
Total: {len(tables_tgt_inside_src_outside)} unique tables\n\n"""

        if len(tables_tgt_inside_src_outside) > 0:
            for idx, row in tables_tgt_inside_src_outside.iterrows():
                analysis_content += f"  Table: {row['table']}\n"
                analysis_content += f"    - Written by (inside): {row['from']}\n"
                analysis_content += f"    - Read by (outside): {row['to']}\n"
                analysis_content += f"    - Size: {row['size']} GB\n\n"

        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""\n4. AGGREGATED STREAM CONNECTIONS:
{'-'*80}

4a. Outgoing Stream Connections (Inside → Outside):
"""
        if len(outgoing_edges) > 0:
            analysis_content += f"Total: {len(outgoing_edges)} connections\n\n"
            for idx, row in outgoing_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No outgoing connections)\n"

        analysis_content += f"""\n4b. Incoming Stream Connections (Outside → Inside):
"""
        if len(incoming_edges) > 0:
            analysis_content += f"Total: {len(incoming_edges)} connections\n\n"
            for idx, row in incoming_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No incoming connections)\n"

        analysis_content += f"""\n5. SUMMARY:
{'-'*80}
  - Total streams in community: {len(streams_list)}
  - Tables flowing INTO community (SRC inside, TGT outside): {len(tables_src_inside_tgt_outside)}
  - Tables flowing OUT OF community (TGT inside, SRC outside): {len(tables_tgt_inside_src_outside)}
  - Aggregated outgoing stream connections: {len(outgoing_edges)}
  - Aggregated incoming stream connections: {len(incoming_edges)}
"""

        # Save analysis file
        analysis_file = os.path.join(comm_dir, f"community_{c}_analysis.txt")
        with open(analysis_file, 'w') as f:
            f.write(analysis_content)
        print(f"  Saved analysis: {analysis_file}")
        saved_files.append(analysis_file)

        # ---- PLOTTING (only if enabled) ----
        if enable_plotting:
            # Build pos from global layout where possible
            pos = {n: pos_global[n] for n in H.nodes() if n in pos_global}

            # Patch missing positions with local layout (prevents KeyError)
            missing = [n for n in H.nodes() if n not in pos]
            if missing:
                pos_local = nx.spring_layout(H, seed=layout_seed, k=layout_k, iterations=20)
                for n in missing:
                    pos[n] = pos_local[n]

            widths, alphas = edge_style(H, weight_attr=weight_attr)

            plt.figure(figsize=figsize, dpi=dpi)

            nx.draw_networkx_nodes(H, pos, node_size=node_size, node_color=[c] * H.number_of_nodes(), cmap=cmap)

            for (u, v), lw, a in zip(H.edges(), widths, alphas):
                nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a))

            # Label only nodes we definitely have positions for
            label_nodes = [n for n in H.nodes() if n in pos]
            if len(label_nodes) <= max_labels:
                # Create labels dict with only nodes that have positions
                labels = {n: n for n in label_nodes}
                nx.draw_networkx_labels(H, pos, labels=labels, font_size=font_size)
            else:
                subset = label_nodes[:max_labels]
                labels = {n: n for n in subset}
                nx.draw_networkx_labels(H, pos, labels=labels, font_size=font_size)

            plt.title(f"Community {c} — nodes={H.number_of_nodes()} edges={H.number_of_edges()}", fontsize=16)
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


# ---- Example usage  ----
resolution = [1.4, 1.7]
for res in resolution:
    leiden_df, meta = get_leiden_df(res, rep_by_res, igraph_names)
    print(meta)

    saved = plot_communities_with_analysis_safe(
        G=G,
        leiden_df=leiden_df,
        stream_table_dependency_df=stream_stream_dependency_df,
        merged_edges_df=merged_dependency_df,
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
def split_communities_topN(leiden_df, top_n=9):
    """
    Split communities into top N heaviest and the rest based on total incoming table weight
    (sum of table weights for tables read by streams in the community, but written by streams outside).
    
    OPTIMIZED VERSION: Uses Spark distributed computing instead of pandas loops.
    
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
        .select("from", "to", "table", "table_weight")
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
        .agg(spark_sum("table_weight").alias("total_incoming_weight"))
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
            DataFrame with columns: from, to, table, table_weight
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
            .select("from", "to", "table", "table_weight")
            .toPandas()
            .copy()
        )

        self.dep["from"] = self.dep["from"].astype(str)
        self.dep["to"] = self.dep["to"].astype(str)
        self.dep["table"] = self.dep["table"].astype(str)
        self.dep["table_weight"] = pd.to_numeric(self.dep["table_weight"], errors="coerce").fillna(0.0)

        # table → single weight (max)
        self.table_weight = (
            self.dep.groupby("table")["table_weight"].max().to_dict()
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
            Total sync cost
        step_costs : list
            Cost at each step
        """
        if initial_available is None:
            available = self.pre_available_tables.copy()
        else:
            available = initial_available.copy()
            
        total = 0.0
        step_costs = []

        for c in ordering:
            to_sync = self.incoming_tables[c] - available
            step = float(builtins.sum(self.table_weight.get(t, 0.0) for t in to_sync))
            total += step
            step_costs.append(step)
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

# DBTITLE 1,Run order optimization for selected resolutions
resolutions = [1.4, 1.7]

order_rows = []

for resolution in resolutions:
    print(f"\n=== Processing resolution: {resolution} ===")
    leiden_df, meta = get_leiden_df(resolution, rep_by_res, igraph_names)

    topN_ids, rest_ids, _ = split_communities_topN(leiden_df, top_n=10)

    dep_scaled_df = unique_table_weights.select("from", "to", "table", "table_weight")

    # Step 1: Optimize REST communities (no pre-available tables)
    print(f"\n--- Step 1: Optimizing REST communities ---")
    bf_rest = BruteForceCommunityOrdering(
        dep_scaled_df, 
        leiden_df, 
        rest_ids,
        pre_available_communities=None  # No pre-available tables for rest
    )
    res_rest = bf_rest.brute_force(log_every=5000, label=f"REST_gamma_{resolution}")

    # Step 2: Optimize TOP N communities with REST tables as pre-available
    print(f"\n--- Step 2: Optimizing TOP N communities (with REST tables pre-available) ---")
    bf_top = BruteForceCommunityOrdering(
        dep_scaled_df, 
        leiden_df, 
        topN_ids,
        pre_available_communities=rest_ids  # REST tables are already available
    )
    res_top = bf_top.brute_force(log_every=5000, label=f"TOPN_gamma_{resolution}")

    # Final order: REST first, then TOP N
    final_order = res_rest["best_order"] + res_top["best_order"]
    total_cost = res_rest['best_cost'] + res_top['best_cost']

    print(f"\n=== FINAL MERGED ORDER for γ={resolution} ===")
    print(f"  REST order: {res_rest['best_order']}")
    print(f"  TOP N order: {res_top['best_order']}")
    print(f"  FINAL order: {final_order}")
    print(f"  REST cost: {res_rest['best_cost']:.2f}")
    print(f"  TOP N cost (with REST pre-available): {res_top['best_cost']:.2f}")
    print(f"  Total cost: {total_cost:.2f}")
    
    # Format output with costs
    order_line = (
        f"gamma={resolution}: {final_order} | "
        f"REST_cost={res_rest['best_cost']:.2f}, "
        f"TOPN_cost={res_top['best_cost']:.2f}, "
        f"TOTAL_cost={total_cost:.2f}"
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
