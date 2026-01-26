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

# DBTITLE 1,Read complexity by stream
# Read complexity by stream with semicolon delimiter
complexity_by_stream_df = spark.read.format("csv").option("header", "true").option("delimiter", ";").load(f"{volume_path}Complexity_by_Stream.csv")

# Define complexity score weights
COMPLEXITY_WEIGHTS = {
    'low': 1,
    'medium': 2,
    'complex': 4,
    'very_complex': 7
}

# Calculate complexity score for each stream
# Score = (low * 1) + (medium * 2) + (complex * 4) + (very_complex * 7)
# Using double to handle decimal values, then casting to int
complexity_scores_df = complexity_by_stream_df.withColumn(
    "complexity_score",
    (col("low").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['low'])) +
    (col("medium").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['medium'])) +
    (col("complex").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['complex'])) +
    (col("very_complex").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['very_complex']))
).select(
    col("stream_name"),
    col("low").cast("double").cast("int").alias("low"),
    col("medium").cast("double").cast("int").alias("medium"),
    col("complex").cast("double").cast("int").alias("complex"),
    col("very_complex").cast("double").cast("int").alias("very_complex"),
    col("complexity_score")
)

# Calculate total complexity across all streams for percentage calculations
total_complexity = complexity_scores_df.agg({"complexity_score": "sum"}).collect()[0][0]
print(f"Total complexity score across all streams: {total_complexity}")

print("\nTop 10 streams by complexity score:")
display(complexity_scores_df.orderBy(col("complexity_score").desc()).limit(10))

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

# DBTITLE 1,Remove admin streams
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
#<TODO> revert back to include report dependencies
self_join_result = non_filtered_self_join_result.join(
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
# ===== WEIGHT CALCULATION METHOD =====
WEIGHT_METHOD = "Factor based"
# =====================================

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

# Remove duplicate (from, to, table) combinations and alias as "weight" for consistency
unique_table_weights = (
    table_weight_df
    .dropDuplicates(["from", "to", "table"])
    .withColumnRenamed("table_weight", "weight")
)

# Group by (from, to), sum the weight as the edge weight
weighted_stream_stream_dependency_df = unique_table_weights.groupBy("from", "to").agg(
    sum("weight").alias("weight")
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
# MAGIC SCALED_MIN = 0.0000  
# MAGIC SCALED_MAX = 1.0
# MAGIC TABLEAU_MULTIPLIER = 1
# MAGIC USE_LOG = False   # <-- set False to disable log transform
# MAGIC
# MAGIC # ===== WEIGHT CALCULATION METHOD =====
# MAGIC WEIGHT_METHOD = "Log based" if USE_LOG else "Min-max based"
# MAGIC # =====================================
# MAGIC
# MAGIC # Learning - Without log function it gives few extremely large clusters and too many small clusters for the current input, so set USE_LOG to True
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
# MAGIC # 6) Drop duplicates per (from, to, table) and alias as "weight" for consistency
# MAGIC unique_table_weights = (
# MAGIC     df
# MAGIC     .dropDuplicates(["from", "to", "table"])
# MAGIC     .withColumnRenamed("scaled_table_weight", "weight")
# MAGIC )
# MAGIC
# MAGIC # 7) Aggregate stream–stream edge weights
# MAGIC weighted_stream_stream_dependency_df = (
# MAGIC     unique_table_weights
# MAGIC     .groupBy("from", "to")
# MAGIC     .agg(spark_sum("weight").alias("weight"))
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

# DBTITLE 1,Identify isolated streams (only intra-stream dependencies)
# Get all unique stream names from the original dependency data (before filtering)
all_streams_in_data = [row['stream_name'] for row in dependency_df.select('stream_name').distinct().collect()]

# Get all streams that appear in the edges (have inter-stream dependencies)
streams_in_edges = set([row['streamA'] for row in merged_dependency_df.select('streamA').distinct().collect()]) | \
                   set([row['streamB'] for row in merged_dependency_df.select('streamB').distinct().collect()])

# Find isolated streams (streams with only intra-stream dependencies)
isolated_streams = [s for s in all_streams_in_data if s not in streams_in_edges]

print(f"Total streams in original data: {len(all_streams_in_data)}")
print(f"Streams with inter-stream dependencies: {len(streams_in_edges)}")
print(f"Isolated streams (only intra-stream dependencies): {len(isolated_streams)}")

if len(isolated_streams) > 0:
    print(f"\nFirst 10 isolated streams: {isolated_streams[:10]}")
    
    # Save isolated streams to CSV for reference
    isolated_streams_df = pd.DataFrame({'stream_name': isolated_streams})
    isolated_streams_df.to_csv(f"{output_path}isolated_streams.csv", index=False)
    print(f"\nIsolated streams saved to: {output_path}isolated_streams.csv")
else:
    print("\nNo isolated streams found - all streams have inter-stream dependencies.")

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

# Add isolated streams as nodes (streams with only intra-stream dependencies)
if len(isolated_streams) > 0:
    print(f"\nAdding {len(isolated_streams)} isolated streams as nodes to the graph...")
    g.add_vertices(isolated_streams)
    print(f"Graph now has {g.vcount()} nodes (including {len(isolated_streams)} isolated nodes)")

# Also build a NetworkX graph for plotting (separate from igraph/Leiden)
G = nx.from_pandas_edgelist(
    edges, "streamA", "streamB",
    edge_attr="weight",
    create_using=nx.Graph()
)

# Add isolated streams to NetworkX graph as well
if len(isolated_streams) > 0:
    G.add_nodes_from(isolated_streams)
    print(f"NetworkX graph now has {G.number_of_nodes()} nodes (including {len(isolated_streams)} isolated nodes)")

# Quick sanity summary check
print(f"\nigraph summary: {g.summary()}")
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
resolutions = [0.4, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6,1.8, 1.9, 2.2, 2.4, 2.6]

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

# DBTITLE 1,Overall graph plot
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

# DBTITLE 1,Choosing resolutions to progress with
resolutions = [1.8]

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
    complexity_scores_df,
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
    complexity_pdf = complexity_scores_df.toPandas()

    # Convert size column to numeric to avoid string concatenation issues
    stream_table_pdf['size'] = pd.to_numeric(stream_table_pdf['size'], errors='coerce').fillna(0.0)

    # Calculate global totals for percentage calculations
    all_streams = set(leiden_df["stream"].tolist())
    total_streams_global = len(all_streams)
    
    # Separate global BI reports and ETL streams
    bi_reports_global = [s for s in all_streams if "json" in s.lower()]
    etl_streams_global = [s for s in all_streams if "json" not in s.lower()]
    total_bi_reports_global = len(bi_reports_global)
    total_etl_streams_global = len(etl_streams_global)
    
    # Calculate total complexity across all streams
    total_complexity_global = complexity_pdf['complexity_score'].sum()

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

        # 1. List of streams in community - separate BI reports from ETL streams
        streams_list = sorted(list(streams_in_community))
        
        # Separate BI reports (containing "json") from ETL streams
        bi_reports = [s for s in streams_list if "json" in s.lower()]
        etl_streams = [s for s in streams_list if "json" not in s.lower()]

        # Calculate percentages
        pct_total = (len(streams_list) / total_streams_global * 100) if total_streams_global > 0 else 0.0
        pct_bi = (len(bi_reports) / total_bi_reports_global * 100) if total_bi_reports_global > 0 else 0.0
        pct_etl = (len(etl_streams) / total_etl_streams_global * 100) if total_etl_streams_global > 0 else 0.0
        
        # Calculate complexity scores for streams in this community
        community_complexity_df = complexity_pdf[complexity_pdf['stream_name'].isin(streams_in_community)].copy()
        
        # Create a mapping of stream to complexity score
        stream_complexity_map = dict(zip(community_complexity_df['stream_name'], community_complexity_df['complexity_score']))
        
        # Calculate total community complexity
        total_community_complexity = community_complexity_df['complexity_score'].sum()
        
        # Calculate percentage of total complexity
        pct_complexity = (total_community_complexity / total_complexity_global * 100) if total_complexity_global > 0 else 0.0
        
        # Separate complexity by BI and ETL
        bi_complexity_df = community_complexity_df[community_complexity_df['stream_name'].isin(bi_reports)]
        etl_complexity_df = community_complexity_df[community_complexity_df['stream_name'].isin(etl_streams)]
        
        total_bi_complexity = bi_complexity_df['complexity_score'].sum()
        total_etl_complexity = etl_complexity_df['complexity_score'].sum()

        # 2. Find tables that are SRC of streams inside but TGT of streams outside
        # In stream_table_dependency_df:
        #   - 'from' stream writes to 'table' (table is TGT of 'from' stream)
        #   - 'to' stream reads from 'table' (table is SRC of 'to' stream)
        # So we want: 'from' OUTSIDE community AND 'to' INSIDE community

        # Tables that are SRC of inside streams, TGT of outside streams
        # (produced outside, consumed inside)
        # For detailed listing: keep all instances, sorted by table name
        tables_src_inside_tgt_outside_all = stream_table_pdf[
            (~stream_table_pdf['from'].isin(streams_in_community)) & (stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].sort_values('table')
        
        # For metrics: unique tables only
        tables_src_inside_tgt_outside_unique = tables_src_inside_tgt_outside_all.drop_duplicates(subset=['table'])

        # Classify incoming tables by BI vs ETL based on which streams need them
        tables_incoming_details = tables_src_inside_tgt_outside_all.copy()
        tables_incoming_details['is_bi'] = tables_incoming_details['to'].str.lower().str.contains('json')
        tables_incoming_details['is_etl'] = ~tables_incoming_details['is_bi']
        
        # For each unique table, determine if it's needed by BI, ETL, or both
        table_classification = tables_incoming_details.groupby('table').agg({
            'is_bi': 'any',
            'is_etl': 'any',
            'size': 'first'  # Size is same for all instances of a table
        }).reset_index()
        
        # If a table is needed by both, classify as ETL (per user requirement)
        table_classification['category'] = table_classification.apply(
            lambda row: 'ETL' if row['is_etl'] else 'BI',
            axis=1
        )
        
        # Separate tables by category
        tables_incoming_bi = table_classification[table_classification['category'] == 'BI']
        tables_incoming_etl = table_classification[table_classification['category'] == 'ETL']
        
        total_size_incoming_bi = tables_incoming_bi['size'].sum() if len(tables_incoming_bi) > 0 else 0.0
        total_size_incoming_etl = tables_incoming_etl['size'].sum() if len(tables_incoming_etl) > 0 else 0.0
        total_size_incoming = total_size_incoming_bi + total_size_incoming_etl

        # Tables that are TGT of inside streams, SRC of outside streams
        # (produced inside, consumed outside)
        # For detailed listing: keep all instances, sorted by table name
        tables_tgt_inside_src_outside_all = stream_table_pdf[
            (stream_table_pdf['from'].isin(streams_in_community)) & (~stream_table_pdf['to'].isin(streams_in_community))
        ][['table', 'from', 'to', 'size']].sort_values('table')
        
        # For metrics: unique tables only
        tables_tgt_inside_src_outside_unique = tables_tgt_inside_src_outside_all.drop_duplicates(subset=['table'])

        # Calculate total sizes (using unique tables)
        total_size_outgoing = tables_tgt_inside_src_outside_unique['size'].sum() if len(tables_tgt_inside_src_outside_unique) > 0 else 0.0

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

        analysis_content += f"""\n1b. ETL STREAMS ({len(etl_streams)} streams without 'json' in name, {pct_etl:.2f}% of all ETL streams):
{'-'*80}
"""
        if len(etl_streams) > 0:
            for i, stream in enumerate(etl_streams, 1):
                complexity = stream_complexity_map.get(stream, 0.0)
                analysis_content += f"{i}. {stream} (complexity: {complexity:.2f})\n"
        else:
            analysis_content += "  (No ETL streams found)\n"

        analysis_content += f"""\n2. COMPLEXITY ANALYSIS:
{'-'*80}
Total Community Complexity: {total_community_complexity:.2f}
Percentage of Total Complexity: {pct_complexity:.2f}%
  - BI Reports Complexity: {total_bi_complexity:.2f}
  - ETL Streams Complexity: {total_etl_complexity:.2f}
"""

        analysis_content += f"""\n3. TABLES - SRC OF STREAMS INSIDE, TGT OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community READ FROM, but are WRITTEN BY streams outside.
(Dependencies flowing INTO the community - SYNC REQUIREMENTS)
Total: {len(tables_src_inside_tgt_outside_unique)} unique tables, {total_size_incoming:.2f} GB
  - For BI Reports: {len(tables_incoming_bi)} tables, {total_size_incoming_bi:.2f} GB
  - For ETL Streams: {len(tables_incoming_etl)} tables, {total_size_incoming_etl:.2f} GB
Total Instances: {len(tables_src_inside_tgt_outside_all)}\n\n"""

        if len(tables_src_inside_tgt_outside_all) > 0:
            # Show BI tables first
            if len(tables_incoming_bi) > 0:
                analysis_content += f"\n3a. FOR BI REPORTS ({len(tables_incoming_bi)} tables, {total_size_incoming_bi:.2f} GB):\n{'-'*80}\n"
                for table in tables_incoming_bi['table']:
                    table_instances = tables_incoming_details[tables_incoming_details['table'] == table]
                    size = table_instances['size'].iloc[0]
                    # Get unique sets of producers and consumers
                    producers = sorted(table_instances['from'].unique())
                    consumers = sorted(table_instances['to'].unique())
                    analysis_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                    analysis_content += f"    - Written by (outside): {', '.join(producers)}\n"
                    analysis_content += f"    - Read by (inside): {', '.join(consumers)}\n"
            
            # Show ETL tables
            if len(tables_incoming_etl) > 0:
                analysis_content += f"\n3b. FOR ETL STREAMS ({len(tables_incoming_etl)} tables, {total_size_incoming_etl:.2f} GB):\n{'-'*80}\n"
                for table in tables_incoming_etl['table']:
                    table_instances = tables_incoming_details[tables_incoming_details['table'] == table]
                    size = table_instances['size'].iloc[0]
                    # Get unique sets of producers and consumers
                    producers = sorted(table_instances['from'].unique())
                    consumers = sorted(table_instances['to'].unique())
                    analysis_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                    analysis_content += f"    - Written by (outside): {', '.join(producers)}\n"
                    analysis_content += f"    - Read by (inside): {', '.join(consumers)}\n"
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""\n4. TABLES - TGT OF STREAMS INSIDE, SRC OF STREAMS OUTSIDE:
{'-'*80}
These are tables that streams in this community WRITE TO, but are READ BY streams outside.
(Dependencies flowing OUT OF the community)
Total: {len(tables_tgt_inside_src_outside_unique)} unique tables
Total Size: {total_size_outgoing:.2f} GB
Total Instances: {len(tables_tgt_inside_src_outside_all)}\n\n"""

        if len(tables_tgt_inside_src_outside_all) > 0:
            # Group by table to show unique producers and consumers
            for table in sorted(tables_tgt_inside_src_outside_unique['table'].unique()):
                table_instances = tables_tgt_inside_src_outside_all[tables_tgt_inside_src_outside_all['table'] == table]
                size = table_instances['size'].iloc[0]
                # Get unique sets of producers and consumers
                producers = sorted(table_instances['from'].unique())
                consumers = sorted(table_instances['to'].unique())
                analysis_content += f"\n  Table: {table} ({size:.2f} GB)\n"
                analysis_content += f"    - Written by (inside): {', '.join(producers)}\n"
                analysis_content += f"    - Read by (outside): {', '.join(consumers)}\n"
        else:
            analysis_content += "  (No such tables found)\n"

        analysis_content += f"""\n5. AGGREGATED STREAM CONNECTIONS:
{'-'*80}

5a. Outgoing Stream Connections (Inside → Outside):
"""
        if len(outgoing_edges) > 0:
            analysis_content += f"Total: {len(outgoing_edges)} connections\n\n"
            for idx, row in outgoing_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No outgoing connections)\n"

        analysis_content += f"""\n5b. Incoming Stream Connections (Outside → Inside):
"""
        if len(incoming_edges) > 0:
            analysis_content += f"Total: {len(incoming_edges)} connections\n\n"
            for idx, row in incoming_edges.iterrows():
                analysis_content += f"  {row['streamA']} → {row['streamB']} (weight: {row['weight']})\n"
        else:
            analysis_content += "  (No incoming connections)\n"

        analysis_content += f"""\n6. SUMMARY:
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

# DBTITLE 1,Migration Order Analysis Function
import os
from datetime import datetime

def append_execution_metadata(output_path, weight_method, top_n, resolutions_dict):
    """
    Append execution metadata to a managed table in Unity Catalog.
    Creates the table if it doesn't exist.
    
    Parameters
    ----------
    output_path : str
        Base output directory path (not used for table, kept for compatibility)
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


def generate_migration_order_analysis(
    leiden_df,
    stream_table_dependency_df,
    merged_edges_df,
    optimized_order,
    resolution,
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
    
    # Convert size to numeric
    stream_table_pdf['size'] = pd.to_numeric(stream_table_pdf['size'], errors='coerce').fillna(0.0)
    
    # Calculate global totals
    all_streams = set(leiden_df["stream"].tolist())
    total_streams_global = len(all_streams)
    bi_reports_global = [s for s in all_streams if "json" in s.lower()]
    etl_streams_global = [s for s in all_streams if "json" not in s.lower()]
    total_bi_reports_global = len(bi_reports_global)
    total_etl_streams_global = len(etl_streams_global)
    
    # Build community -> streams mapping
    community_streams = {}
    for c in optimized_order:
        streams = leiden_df[leiden_df['community'] == c]['stream'].tolist()
        community_streams[c] = set(streams)
    
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

{'='*100}
"""
    
    for step, community_id in enumerate(optimized_order, 1):
        streams_in_community = community_streams[community_id]
        
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
        
        # Calculate remaining
        remaining_streams = total_streams_global - cumulative_streams
        remaining_bi = total_bi_reports_global - cumulative_bi
        remaining_etl = total_etl_streams_global - cumulative_etl
        
        # Progress percentages
        progress_pct = (cumulative_streams / total_streams_global * 100) if total_streams_global > 0 else 0.0
        
        # Write community section
        report_content += f"""\n{'='*100}
STEP {step}/{len(optimized_order)}: COMMUNITY {community_id}
{'='*100}

--- COMMUNITY COMPOSITION ---
Streams in this community: {len(streams_in_community)} ({pct_total:.2f}% of total)
  - BI Reports: {len(bi_reports)} ({pct_bi:.2f}% of all BI reports)
  - ETL Streams: {len(etl_streams)} ({pct_etl:.2f}% of all ETL streams)

--- STREAMS LIST ---
"""
        
        # Add list of streams in the community
        if len(bi_reports) > 0:
            report_content += f"\n  ** BI REPORTS ({len(bi_reports)}) **\n"
            for stream in sorted(bi_reports):
                report_content += f"    - {stream}\n"
        
        if len(etl_streams) > 0:
            report_content += f"\n  ** ETL STREAMS ({len(etl_streams)}) **\n"
            for stream in sorted(etl_streams):
                report_content += f"    - {stream}\n"
        
        report_content += f"""\n--- MIGRATION PROGRESS ---
Cumulative streams migrated: {cumulative_streams}/{total_streams_global} ({progress_pct:.2f}%)
  - BI Reports migrated: {cumulative_bi}/{total_bi_reports_global}
  - ETL Streams migrated: {cumulative_etl}/{total_etl_streams_global}
Remaining streams: {remaining_streams}
  - BI Reports remaining: {remaining_bi}
  - ETL Streams remaining: {remaining_etl}

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

# DBTITLE 1,Run order optimization and generate reports
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

    topN_ids, rest_ids, community_weights = split_communities_topN(leiden_df, top_n=top_n)

    dep_scaled_df = unique_table_weights.select("from", "to", "table", "weight")
    
    if len(rest_ids) > top_n:
        print(f"\n--- REST communities ({len(rest_ids)}) exceed top_n ({top_n}), splitting into batches ---")
        
        # IMPORTANT: rest_ids is already sorted by weight (descending) from split_communities_topN
        # Verify the ordering is preserved
        rest_weights = [community_weights.get(comm_id, 0.0) for comm_id in rest_ids]
        print(f"REST communities weights (first 10): {rest_weights[:10]}")
        print(f"Weights are in descending order: {rest_weights == sorted(rest_weights, reverse=True)}")
        
        # Split rest_ids into batches of size top_n
        # Each batch will contain communities in weight-descending order
        rest_batches = []
        for i in range(0, len(rest_ids), top_n):
            batch = rest_ids[i:i+top_n]
            rest_batches.append(batch)
        
        print(f"Created {len(rest_batches)} batches: {[len(b) for b in rest_batches]}")
        
        # Print weight ranges for each batch and identify zero-weight batches
        zero_weight_batches = []
        non_zero_weight_batches = []
        
        for batch_idx, batch in enumerate(rest_batches):
            batch_weights = [community_weights.get(comm_id, 0.0) for comm_id in batch]
            max_weight = max(batch_weights)
            min_weight = min(batch_weights)
            print(f"  Batch {batch_idx + 1}: weight range [{min_weight:.2f} - {max_weight:.2f}]")
            
            # Check if this is a zero-weight batch (max weight is 0)
            if max_weight == 0.0:
                zero_weight_batches.append((batch_idx, batch))
                print(f"    -> Zero-weight batch, will be migrated FIRST without optimization")
            else:
                non_zero_weight_batches.append((batch_idx, batch))
                print(f"    -> Non-zero weight batch, will be optimized")
        
        print(f"\nBatch classification:")
        print(f"  Zero-weight batches: {len(zero_weight_batches)} (migrate first, no optimization)")
        print(f"  Non-zero weight batches: {len(non_zero_weight_batches)} (optimize)")
        
        # Optimize only non-zero weight batches in reverse order (last batch first)
        # This ensures earlier batches can use later batches' tables as pre-available
        batch_results = []  # Will store results in order: batch3, batch2, batch1
        cumulative_pre_available = []  # Communities whose tables are already available
        
        # Process non-zero weight batches in reverse order
        for batch_idx, batch in reversed(non_zero_weight_batches):
            print(f"\n--- Optimizing REST batch {batch_idx + 1}/{len(rest_batches)} (communities: {batch}) ---")
            
            # Pre-available communities are all batches processed so far (later batches)
            pre_available = cumulative_pre_available if cumulative_pre_available else None
            
            bf_batch = BruteForceCommunityOrdering(
                dep_scaled_df,
                leiden_df,
                batch,
                pre_available_communities=pre_available
            )
            res_batch = bf_batch.brute_force(log_every=5000, label=f"REST_batch_{batch_idx+1}_gamma_{resolution}")
            
            batch_results.append({
                'order': res_batch['best_order'],
                'cost': res_batch['best_cost'],
                'batch_idx': batch_idx
            })
            
            # Add this batch to cumulative pre-available for next iteration
            cumulative_pre_available.extend(batch)
            
            print(f"Batch {batch_idx + 1} optimized: order={res_batch['best_order']}, cost={res_batch['best_cost']:.2f}")
        
        # Combine results: zero-weight batches first (in original order), then optimized batches
        rest_order = []
        rest_cost = 0.0
        
        # Add zero-weight batches first (in their original order, which is descending by batch index)
        for batch_idx, batch in zero_weight_batches:
            rest_order.extend(batch)
            print(f"\nAdding zero-weight batch {batch_idx + 1} to order (no optimization): {batch}")
        
        # Add optimized batches (already in reverse order from optimization)
        for result in batch_results:
            rest_order.extend(result['order'])
            rest_cost += result['cost']
        
        print(f"\n--- Combined REST order (zero-weight batches → optimized batches): {rest_order} ---")
        print(f"--- Combined REST cost (optimization metric, zero-weight batches excluded): {rest_cost:.2f} ---")
        
    else:
        # Original logic: optimize REST communities as a single batch
        print(f"\n--- Step 1: Optimizing REST communities ({len(rest_ids)} communities) ---")
        bf_rest = BruteForceCommunityOrdering(
            dep_scaled_df, 
            leiden_df, 
            rest_ids,
            pre_available_communities=None  # No pre-available tables for rest
        )
        res_rest = bf_rest.brute_force(log_every=5000, label=f"REST_gamma_{resolution}")
        rest_order = res_rest["best_order"]
        rest_cost = res_rest['best_cost']

    # Step 2: Optimize TOP N communities with all REST communities as pre-available
    print(f"\n--- Step 2: Optimizing TOP N communities (with all REST tables pre-available) ---")
    bf_top = BruteForceCommunityOrdering(
        dep_scaled_df, 
        leiden_df, 
        topN_ids,
        pre_available_communities=rest_ids  # All REST communities are already available
    )
    res_top = bf_top.brute_force(log_every=7500, label=f"TOPN_gamma_{resolution}")

    # Final order: REST first (zero-weight batches + optimized batches), then TOP N
    final_order = rest_order + res_top["best_order"]
    
    print(f"\n=== FINAL MERGED ORDER for γ={resolution} ===")
    print(f"  REST order: {rest_order}")
    print(f"  TOP N order: {res_top['best_order']}")
    print(f"  FINAL order: {final_order}")
    print(f"  REST cost (optimization metric): {rest_cost:.2f}")
    print(f"  TOP N cost (with REST pre-available, optimization metric): {res_top['best_cost']:.2f}")
    
    # Generate detailed migration analysis report
    # This will calculate the TRUE total sync cost for the entire order
    print(f"\n--- Generating migration order analysis report ---")
    report_file, total_sync_cost = generate_migration_order_analysis(
        leiden_df=leiden_df,
        stream_table_dependency_df=stream_stream_dependency_df,
        merged_edges_df=merged_dependency_df,
        optimized_order=final_order,
        resolution=resolution,
        outdir=f"{output_path}migration_order_analysis",
    )
    print(f"Report saved to: {report_file}")
    
    # Store resolution, total sync cost, and number of communities for metadata CSV
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

# Append execution metadata to CSV
print(f"\n=== Appending execution metadata to CSV ===")
metadata_file = append_execution_metadata(
    output_path="/Volumes/odp_adw_mvp_n/migration/utilities/community_detection/",
    weight_method=WEIGHT_METHOD,  # Use the variable set by the active weight calculation cell
    top_n=top_n,  # Include the top_n value used for ordering
    resolutions_dict=resolutions_costs  # Now contains (total_cost, num_communities) tuples
)
print(f"Metadata CSV updated: {metadata_file}")
print(f"Weight calculation method: {WEIGHT_METHOD}")
print(f"Top N value: {top_n}")
print(f"Resolutions and costs: {resolutions_costs}")

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
    print(f"\n⚠️ WARNING: Selected resolution {SELECTED_RESOLUTION} was not processed in cell 40!")
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

# Initialize tracking
available_tables = set()  # Tables available from ALL migrated streams (cumulative)
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
    f.write(f"Weight Method: {WEIGHT_METHOD}\n")
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
