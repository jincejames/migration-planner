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
from pyspark.sql.functions import col, lit, when, upper, lower
from sklearn.metrics import adjusted_rand_score
from collections import Counter
from pyspark.sql.functions import ceil, sum, when, array, sort_array

# COMMAND ----------

current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# COMMAND ----------

dbutils.widgets.text(
    "volume_name",
    "/Volumes/users/jince_james/lufthansa/community_detection/",
    "Input Volume Path"
)

# COMMAND ----------

dbutils.widgets.text(
    "input_dependency_name",
    "ETL-table-dependencies_20251223_1218.csv",
    "Input CSV file name"
)

# COMMAND ----------

dbutils.widgets.text(
    "outofscope_stream_file_name",
    "out-of-scopte-streams.csv",
    "Out of scope streams file name"
)

# COMMAND ----------

dbutils.widgets.text(
    "report_dependency_file_name",
    "stream_to_report_mapping.csv",
    "report Dependency file name"
)

dbutils.widgets.text(
    "table_size",
    "table-space-in-gb_20251201_1352.csv",
    "Table size file name"
)

# COMMAND ----------

volume_path = dbutils.widgets.get("volume_name")
dependency_input_path = volume_path + dbutils.widgets.get("input_dependency_name")
outofscope_stream_path = volume_path + dbutils.widgets.get("outofscope_stream_file_name")
report_dependency = volume_path + dbutils.widgets.get("report_dependency_file_name")
table_size = volume_path + dbutils.widgets.get("table_size")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Datasets:

# COMMAND ----------

# DBTITLE 1,Reading input stream - table dependency file
dependency_df_full = spark.read.format("csv").option("header", "true").load(dependency_input_path)
display(dependency_df_full)

# COMMAND ----------

# DBTITLE 1,Reading out of scope stream names
outofscope_stream_names_df = spark.read.format("csv").option("header","true").load(outofscope_stream_path).select(col("stream_name"))
outofscope_stream_names_rows_list = outofscope_stream_names_df.collect()
outofscope_stream_names_list = [x['stream_name'] for x in outofscope_stream_names_rows_list]

# COMMAND ----------

print(outofscope_stream_names_list)

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
display(report_dependency_df.dropDuplicates(["stream_name"]).count())
display(report_dependency_df)

# COMMAND ----------

# DBTITLE 1,Reading table size in GB records
table_size_df = spark.read.format("csv").option("header","true").load(table_size).select(upper(col("DB_Table_Name")).alias("table_name"), col("SPACE_IN_GB").alias("size"))
display(table_size_df)

# COMMAND ----------

# Removing streams associated with acrchiving, GDPR, housekeeping and out of scope streams
dependency_df_filtered = dependency_df_full.filter(
    ~upper(col("stream_name")).contains("ARCHIVE") & 
    ~upper(col("stream_name")).contains("GDPR") &
    ~upper(col("stream_name")).contains("HOUSEKEEPING") &
    ~upper(col("stream_name")).isin(outofscope_stream_names_list)
)

# COMMAND ----------

dependency_df_filtered.count()

# COMMAND ----------

dependency_df_filtered.filter(upper(col('table_type')).contains("TGT")).count()

# COMMAND ----------

# DBTITLE 1,Considering all TGT tables as SRC
# Considering all TGT tables as SRC as well due to a gap in ODAT output
tgt_as_source = dependency_df_filtered.filter(upper(col('table_type')).contains("TGT")).replace({"TGT" : "SRC", "Tgt_Trns" : "Src_Trns"}, subset=["table_type"])
dependency_df = dependency_df_filtered.union(tgt_as_source)
dependency_df.count()

# COMMAND ----------

display(dependency_df)

# COMMAND ----------

# DBTITLE 1,analysis
print(f"Total number of records is {dependency_df.count()}")
print(f"Total number of distinct records is {dependency_df.distinct().count()}")

# There are duplicate records in the input dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forming Table to Stream Dependecies

# COMMAND ----------

# DBTITLE 1,analysis: Check if there are any trns src or tgt tables part of non transactional DBs
cleaned_dependency_df = dependency_df.distinct().select('stream_name', col('ref_object_name').alias('table_name'),col('ref_object_dbname').alias('db_name'), 'table_type')

display(cleaned_dependency_df.filter(col('table_type').contains('Trns') & ~ col('db_name').contains('TRANS')))

# There are no transactional tables in the dataset that are part of a non transactional DB - Verified

# COMMAND ----------

# DBTITLE 1,analysis: Check if the DDL of tables with same name, same across 2 different DBs
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

# DBTITLE 1,Without filtering transactional tables
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

# DBTITLE 1,analysis: See results count with transactional count dependencies
non_filtered_self_join_result.count()

# RESULT: There are ~100 inter stream dependencies more with transactional tables included. So we will use dataset without filtering
# Result -> Except for archiving streams, this shouldn't be the case. Dig deeper. Send ex to micheal if there are outliers.

# COMMAND ----------

report_join_result = (
    report_dependency_df.alias("df2")
    .join(dependency_with_transactional_df.alias("df1"), upper(col("df1.table_name")) == upper(col("df2.table_name")))
)
display(report_join_result)

# COMMAND ----------

# DBTITLE 1,Merging reports and streams
self_join_result_without_size = report_join_result.union(non_filtered_self_join_result)
display(self_join_result_without_size)

# COMMAND ----------

self_join_result = self_join_result_without_size.join(
    table_size_df.alias("table_size"),
    col("df1.table_name") == col("table_size.table_name")
).select(
    "df1.*",
    "df2.*",
    col("table_size.size")
)
display(self_join_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forming Stream - Stream Dependencies

# COMMAND ----------

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

display(stream_stream_dependency_df)

# COMMAND ----------

# DBTITLE 1,Analysis: Verify created community
stream_list = [
    "EXP_PARTNERS_LG",
    "EXP_PARTNERS_LHG",
    "EXP_PARTNERS_LO",
    "EXP_PARTNERS_OS",
    "EXP_PARTNER_MAM_REVENUE",
    "EXP_PGR_FEEDBACK",
    "EXP_PGR_PROFILE",
    "EXP_RESPONSYS",
    "IMP_ANTITERROR_LIST",
    "IMP_EMAIL_TRACKING",
    "IMP_FFP_BIT",
    "IMP_KFM",
    "IMP_MAM_AIA",
    "IMP_SAMBA",
    "INT_FFP_DAY",
    "INT_FFP_MON",
    "INT_FFP_REP",
    "INT_KFM_FACT",
    "INT_UNIFIED_REPORTING_DAILY",
    "INT_WEBQUERY",
]
display(
    stream_stream_dependency_df.filter((col("to").isin(stream_list)) & (~col("from").isin(stream_list))).sort(
        "from", "to"
    )
)

# COMMAND ----------

# DBTITLE 1,Analysis: Stream-stream dependency count
stream_stream_dependency_df.count()

# COMMAND ----------

# DBTITLE 1,Analysis: Number of distinct streams


# Number of distinct streams in the dependency map
total_distinct_streams = stream_stream_dependency_df.select("from").union(stream_stream_dependency_df.select("to")).distinct()

display(total_distinct_streams)
print(f"Total distinct streams, reports (from + to): {total_distinct_streams.count()}")

# COMMAND ----------

# # For each (from, to, table), compute table_weight = ceil(size/100), min 1
# # If "to" contains "tableau", set table_weight = 2 regardless of size, this is to give slightly high preference to reports
# table_weight_df = stream_stream_dependency_df.withColumn(
#     "table_weight",
#     when(
#         lower(col("to")).contains("tableau"),
#         2
#     ).otherwise(
#         ceil((col("size").cast("double") / 100)).cast("int")
#     )
# ).withColumn(
#     "table_weight",
#     when(col("table_weight") < 1, 1).otherwise(col("table_weight"))
# )

# # Remove duplicate (from, to, table) combinations
# unique_table_weights = table_weight_df.dropDuplicates(["from", "to", "table"])

# # Group by (from, to), sum the table_weight as the edge weight
# weighted_stream_stream_dependency_df = unique_table_weights.groupBy("from", "to").agg(
#     sum("table_weight").alias("weight")
# )

# display(weighted_stream_stream_dependency_df)

# #<ToDo> CLarify why ARCHIVE is used as a source stream (read that is read from) / which usecases use this? -> This should be only archive to archive. Check if there are any other instances. This also could be the case some cleanup is happening.

# COMMAND ----------

# DBTITLE 1,Scaling and weight calculation
from pyspark.sql.functions import (
    col, when, lower, lit,
    min as spark_min, max as spark_max, sum as spark_sum,
    log1p
)

# ---- CONFIG ----
SCALED_MIN = 0.0
SCALED_MAX = 1.0
TABLEAU_MULTIPLIER = 1.5
USE_LOG = True   # <-- set False to disable log transform

# 1) Raw per-table value = size
df = stream_stream_dependency_df.withColumn("raw_table_weight", col("size").cast("double"))

# (optional) guard against null/negative sizes (log1p requires >= 0)
df = df.withColumn(
    "raw_table_weight",
    when(col("raw_table_weight").isNull() | (col("raw_table_weight") < 0), lit(0.0))
    .otherwise(col("raw_table_weight"))
)

# 2) Optional log transform (log1p)
if USE_LOG:
    df = df.withColumn("raw_for_scaling", log1p(col("raw_table_weight")))
else:
    df = df.withColumn("raw_for_scaling", col("raw_table_weight"))

# 3) Min–Max stats on the scaling column
stats = df.agg(
    spark_min("raw_for_scaling").alias("mn"),
    spark_max("raw_for_scaling").alias("mx")
).collect()[0]

mn = float(stats["mn"])
mx = float(stats["mx"])
den = (mx - mn) if (mx - mn) != 0 else 1.0

# 4) Min–Max scale into [0, 1] then map to [SCALED_MIN, SCALED_MAX]
df = df.withColumn("mm01", (col("raw_for_scaling") - lit(mn)) / lit(den))

df = df.withColumn(
    "scaled_table_weight",
    lit(SCALED_MIN) + (lit(SCALED_MAX - SCALED_MIN) * col("mm01"))
)

# 5) Tableau boost AFTER scaling
df = df.withColumn(
    "scaled_table_weight",
    when(lower(col("to")).contains("tableau"),
         col("scaled_table_weight") * lit(TABLEAU_MULTIPLIER)
    ).otherwise(col("scaled_table_weight"))
)

# 6) Drop duplicates per (from, to, table)
unique_table_weights = df.dropDuplicates(["from", "to", "table"])

# 7) Aggregate stream–stream edge weights
weighted_stream_stream_dependency_df = (
    unique_table_weights
    .groupBy("from", "to")
    .agg(spark_sum("scaled_table_weight").alias("weight"))
)

display(weighted_stream_stream_dependency_df)


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

display(merged_dependency_df.orderBy("streamA", "streamB"))

# COMMAND ----------

# DBTITLE 1,Analysis: Src Streams for INT_IDM_INVENTORY Stream. To Verify Correctness
# display(stream_stream_dependency_df.filter((col("to") == "INT_IDM_INVENTORY")).sort("from"))

stream_list = ["IMP_APLUS_SDH", "IMP_CAFE_CURRENCY", "IMP_TCURRENCY", "INT_IDM_REF_CURRENCY"]
display(merged_dependency_df.filter((col("streamA").isin(stream_list))|(col("streamB").isin(stream_list))).sort("streamA","streamB"))
# Result -> Verified with Micheal for 2 streams.

# COMMAND ----------

# DBTITLE 1,Analysis: Non IMP_INV* Src Streams for INT_IDM_INVENTORY stream. To Verify Correctness
display(stream_stream_dependency_df.filter((col("to") == "INT_IDM_INVENTORY") & ~col("from").contains("IMP_INVENTORY")).select("from","table").distinct().sort("from"))

# COMMAND ----------

# DBTITLE 1,Analysis: Total number of stream to stream connections
# merged_dependency_df.filter(col("weight")<=4).count()

# Total number of connections = 4093
# Stream - stream dependecies where weight <= 4, aka only 4 or less tables (2 incase of reports) has connection between 2 streams = 2628
# ~75% stream - stream connections are loosely coupled


# COMMAND ----------

# Only use for small data!
edges_df = merged_dependency_df.toPandas()
edges_df.to_csv(f"{volume_path}edges.csv", index=False)

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
# MAGIC - Select a representative γ in the **0.8–2 range** for a single, defensible partition.
# MAGIC - PLotting
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Plots

# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the bigger clustered graph for the chosen resolutions where each node is colored in the color of the community it has been clustered into. 
# MAGIC Results are saved under the volume path/leiden_plots

# COMMAND ----------

from adjustText import adjust_text

def select_resolutions(
    summary: pd.DataFrame,
    ari_target: float = 0.93,
    max_largest_comm_share: float = 0.2,
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
        (summary["stability_ari"] >= ari_target) &
        (summary["largest_comm_share_avg"] <= max_largest_comm_share)
    ].sort_values(sort_by)

    selected_resolutions = selected["resolution"].tolist()
    return selected, selected_resolutions


def precompute_layout(
    G: nx.Graph,
    seed: int = 42,
    k=3.0,  # Increased for more spacing
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
        Spring layout distance parameter. Higher = more spacing.

    Returns
    -------
    pos : dict
        Node -> (x, y) layout positions.
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
    figsize=(50, 40),  # Much larger figure
    dpi: int = 400,  # Higher resolution
    label_fontsize: int = 11,  # Larger font
    node_size: int = 800,  # Larger nodes
    cmap=plt.cm.tab20,
    draw_labels: bool = True,
    save: bool = True,
    show: bool = True,
    use_adjust_text: bool = True,  # Use adjustText for label positioning
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
    use_adjust_text : bool
        Use adjustText library to prevent label overlap.

    Returns
    -------
    outputs : list[str]
        Paths of saved files (may be empty if save=False).
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
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Draw edges first (behind nodes)
        print(f"  Drawing {G.number_of_edges()} edges...")
        for (u, v), lw, a in zip(G.edges(), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=float(lw),
                alpha=float(a),
                ax=ax,
                edge_color='gray'
            )

        # Draw nodes
        print(f"  Drawing {G.number_of_nodes()} nodes...")
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_size,
            cmap=cmap,
            linewidths=2.0,
            edgecolors='black',
            ax=ax
        )

        # Draw labels
        if draw_labels:
            if use_adjust_text:
                print(f"  Adding labels with overlap prevention...")
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
                                alpha=0.85
                            )
                        )
                    )
                
                # Adjust text positions to avoid overlap
                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.6),
                    expand_points=(2.0, 2.0),
                    expand_text=(1.5, 1.5),
                    force_points=(0.8, 0.8),
                    force_text=(0.8, 0.8),
                    ax=ax,
                    lim=500
                )
            else:
                # Standard labels with background
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
                            alpha=0.85
                        )
                    )

        title = (
            f"Leiden (RBConfiguration) — resolution γ={res} (seed={rep['seed']})\n"
            f"#communities={n_comms}, largest_comm_share={largest_share:.3f}, "
            f"small_comms<5={tiny_lt5}, quality={quality:.2f}"
        )
        ax.set_title(title, fontsize=22, fontweight='bold', pad=30)
        ax.axis("off")
        ax.margins(0.1)  # Add margin around plot
        
        plt.tight_layout()

        if save:
            outfile = os.path.join(outdir, f"leiden_rb_gamma_{res}.png")
            print(f"  Saving high-resolution plot...")
            plt.savefig(outfile, bbox_inches="tight", dpi=dpi, facecolor='white')
            outputs.append(outfile)
            print(f"  Saved: {outfile}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return outputs




selected, selected_resolutions = select_resolutions(summary, ari_target=0.93, max_largest_comm_share=0.25)
print("Selected resolutions:", selected_resolutions)
print(selected[["resolution", "largest_comm_share_avg", "stability_ari", "n_communities_avg"]])

print("\nGenerating high-resolution plots with improved readability...")
print("This may take several minutes depending on graph size.\n")

# Ensure we're using the volume path from the widget parameter
output_dir = os.path.join(volume_path, "leiden_plots")
print(f"Output directory: {output_dir}")
print(f"Saving plots to volume location (not git folder)\n")

pos = precompute_layout(G, seed=42, k=3.0)  # More spacing
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
    outdir=output_dir,  # Using volume path
    figsize=(50, 40),  # Very large figure
    dpi=400,  # High resolution
    label_fontsize=11,  # Larger font
    node_size=800,  # Larger nodes
    draw_labels=True,
    save=True,
    show=False,
    use_adjust_text=True,  # Prevent label overlap
)

print("\n" + "="*80)
print("COMPLETE: All plots saved with high resolution (400 DPI)")
print(f"Output directory: {output_dir}")
print("Plots can be zoomed in with full clarity.")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC **Plotting the subgraphs / clusters, by choosing the preferred resolution**

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Helper to build leiden_df from a stored membership array
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
):
    import os
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    node_to_comm = dict(zip(leiden_df["stream"], leiden_df["community"]))
    labeled_nodes = [n for n in G.nodes() if n in node_to_comm]

    # Global layout (works for most nodes)
    pos_global = nx.spring_layout(
        G.subgraph(labeled_nodes),
        seed=layout_seed,
        k=layout_k,
        iterations=layout_iterations
    )

    stream_table_pdf = stream_table_dependency_df.toPandas().copy()
    merged_edges_pdf = merged_edges_df.toPandas().copy()

    communities = sorted(leiden_df["community"].unique())
    saved_files = []

    for c in communities:
        comm_dir = os.path.join(outdir, f"community_{c}")
        os.makedirs(comm_dir, exist_ok=True)

        comm_nodes = [n for n in labeled_nodes if node_to_comm[n] == c]
        streams_in_comm = set(comm_nodes)

        H = G.subgraph(comm_nodes).copy()
        print(f"Community {c}: nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")

        # ---- analysis (same as before) ----
        incoming_rows = stream_table_pdf[
            (~stream_table_pdf["from"].isin(streams_in_comm)) &
            (stream_table_pdf["to"].isin(streams_in_comm))
        ][["table", "from", "to"]].drop_duplicates()

        outgoing_rows = stream_table_pdf[
            (stream_table_pdf["from"].isin(streams_in_comm)) &
            (~stream_table_pdf["to"].isin(streams_in_comm))
        ][["table", "from", "to"]].drop_duplicates()

        incoming_tables = sorted(incoming_rows["table"].unique().tolist())
        outgoing_tables = sorted(outgoing_rows["table"].unique().tolist())

        cross_edges = merged_edges_pdf[
            (merged_edges_pdf["streamA"].isin(streams_in_comm) & ~merged_edges_pdf["streamB"].isin(streams_in_comm)) |
            (~merged_edges_pdf["streamA"].isin(streams_in_comm) & merged_edges_pdf["streamB"].isin(streams_in_comm))
        ].copy()

        streams_sorted = sorted(list(streams_in_comm))

        analysis_lines = []
        analysis_lines.append(f"Community {c} Analysis (Resolution γ={resolution})")
        analysis_lines.append("=" * 80)
        analysis_lines.append("")
        analysis_lines.append(f"1) STREAMS IN COMMUNITY ({len(streams_sorted)}):")
        analysis_lines.append("-" * 80)
        for i, s in enumerate(streams_sorted, 1):
            analysis_lines.append(f"{i}. {s}")

        analysis_lines.append("")
        analysis_lines.append("2) TABLES FLOWING INTO COMMUNITY (produced outside → consumed inside)")
        analysis_lines.append("-" * 80)
        analysis_lines.append(f"Unique tables: {len(incoming_tables)}")
        if len(incoming_rows) > 0:
            for _, r in incoming_rows.iterrows():
                analysis_lines.append(
                    f"  Table: {r['table']} | Produced by (outside): {r['from']} | Consumed by (inside): {r['to']}"
                )
        else:
            analysis_lines.append("  (None)")

        analysis_lines.append("")
        analysis_lines.append("3) TABLES FLOWING OUT OF COMMUNITY (produced inside → consumed outside)")
        analysis_lines.append("-" * 80)
        analysis_lines.append(f"Unique tables: {len(outgoing_tables)}")
        if len(outgoing_rows) > 0:
            for _, r in outgoing_rows.iterrows():
                analysis_lines.append(
                    f"  Table: {r['table']} | Produced by (inside): {r['from']} | Consumed by (outside): {r['to']}"
                )
        else:
            analysis_lines.append("  (None)")

        analysis_lines.append("")
        analysis_lines.append("4) CROSS-COMMUNITY STREAM CONNECTIONS (from merged edges)")
        analysis_lines.append("-" * 80)
        analysis_lines.append(f"Cross edges: {len(cross_edges)}")
        if len(cross_edges) > 0:
            cross_edges_sorted = cross_edges.sort_values("weight", ascending=False)
            for _, r in cross_edges_sorted.iterrows():
                analysis_lines.append(f"  {r['streamA']} — {r['streamB']} (weight={r['weight']})")
        else:
            analysis_lines.append("  (None)")

        analysis_lines.append("")
        analysis_lines.append("5) SUMMARY")
        analysis_lines.append("-" * 80)
        analysis_lines.append(f"Streams: {len(streams_sorted)}")
        analysis_lines.append(f"Internal edges: {H.number_of_edges()}")
        analysis_lines.append(f"Incoming unique tables: {len(incoming_tables)}")
        analysis_lines.append(f"Outgoing unique tables: {len(outgoing_tables)}")
        analysis_lines.append(f"Cross-community edges: {len(cross_edges)}")

        analysis_file = os.path.join(comm_dir, f"{filename_prefix}_{c}_analysis.txt")
        with open(analysis_file, "w") as f:
            f.write("\n".join(analysis_lines))
        saved_files.append(analysis_file)

        # ---- plot ----
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

        nx.draw_networkx_nodes(
            H, pos,
            node_size=node_size,
            node_color=[c] * H.number_of_nodes(),
            cmap=cmap
        )

        for (u, v), lw, a in zip(H.edges(), widths, alphas):
            nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a))

        # Label only nodes we definitely have positions for
        label_nodes = [n for n in H.nodes() if n in pos]
        if len(label_nodes) <= max_labels:
            nx.draw_networkx_labels(H, pos, font_size=font_size)
        else:
            subset = label_nodes[:max_labels]
            nx.draw_networkx_labels(H, {n: pos[n] for n in subset}, font_size=font_size)

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

    return saved_files


# ---- Example usage  ----
resolution = 2.6
leiden_df, meta = get_leiden_df(resolution, rep_by_res, igraph_names)
print(meta)

saved = plot_communities_with_analysis_safe(
    G=G,
    leiden_df=leiden_df,
    stream_table_dependency_df=stream_stream_dependency_df,
    merged_edges_df=merged_dependency_df,
    resolution=resolution,
    outdir=f"{volume_path}leiden_community_plots_gamma_{resolution}",
    layout_seed=42,
    layout_k=None,                 # same as your working code
    layout_iterations=30,          # safer than 100
    figsize=(24, 18),              # safer than (40,32)
    dpi=220,                       # safer than 300
    node_size=220,
    font_size=8,
    show=False,
    save=True,
    filename_prefix=f"gamma_{resolution}_comm",
    max_labels=120                 # guardrail
)
print(f"Saved {len(saved)} files")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Community Ordering Optimization
# MAGIC
# MAGIC This module determines an execution order for communities at a fixed Leiden resolution in order to **minimize total data synchronization effort** between communities.
# MAGIC
# MAGIC The goal is to ensure that, as communities are processed sequentially, the amount of external data that must be synchronized into each community is as small as possible.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Order communities such that the **cumulative synchronization cost** across all communities is minimized.
# MAGIC
# MAGIC Unlike earlier versions, the optimization is performed on the **same scaled table weights** that are used to construct the community graph itself, ensuring full consistency between clustering and execution ordering.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Sync Cost Definition
# MAGIC
# MAGIC For a given ordering of communities:
# MAGIC
# MAGIC * For each community in sequence, identify the set of tables that  
# MAGIC   **streams inside the community READ FROM** but are **WRITTEN BY streams outside**.
# MAGIC
# MAGIC * Tables that have already been produced by previously ordered communities are considered **already available** and do **not** incur any cost.
# MAGIC
# MAGIC * The **sync cost for a community** is defined as:
# MAGIC
# MAGIC \[
# MAGIC \text{SyncCost(community)} = \sum_{t \in \text{TablesToSync}} \text{ScaledTableWeight}(t)
# MAGIC \]
# MAGIC
# MAGIC where `ScaledTableWeight` is the **min–max scaled and Tableau-adjusted table weight** used in the Leiden graph construction.
# MAGIC
# MAGIC * The **total cost of an ordering** is the sum of sync costs over all communities in that order.
# MAGIC
# MAGIC This formulation naturally penalizes both:
# MAGIC - **many tables** (large cardinality), and
# MAGIC - **large tables** (high data volume),
# MAGIC
# MAGIC without requiring separate weighting terms.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Optimization Strategy
# MAGIC
# MAGIC Because the search space grows factorially, the optimization is performed in two independent phases:
# MAGIC
# MAGIC 1. **Brute-force optimization of the 10 largest communities**, guaranteeing an optimal ordering for the most impactful portion of the graph.
# MAGIC 2. **Brute-force optimization of the remaining communities**.
# MAGIC
# MAGIC The final execution order is obtained by concatenating both optimal orderings:
# MAGIC
# MAGIC \[
# MAGIC \text{FinalOrder} = \text{Order}_{\text{Top10}} \;\Vert\; \text{Order}_{\text{Rest}}
# MAGIC \]
# MAGIC
# MAGIC This split preserves optimality where it matters most while keeping the computation tractable.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Properties
# MAGIC
# MAGIC * Cost function is **fully aligned with community detection weights**  
# MAGIC * No heuristic mixing of size and table count  
# MAGIC * Deterministic optimal solutions within each optimization block  
# MAGIC * Directly reflects actual data movement requirements
# MAGIC
# MAGIC

# COMMAND ----------

import itertools
import time
from math import factorial
import builtins
import pandas as pd
import numpy as np


class BruteForceCommunityOrderingScaledWeight:
    """
    Brute force search over all community orderings (for a given subset of communities).

    Cost = SUM of per-table 'scaled_table_weight' that must be synced at each step:
      to_sync(step) = incoming_tables[community] - available_tables_so_far
      step_cost     = sum(table_weight[table] for table in to_sync)
      total_cost    = sum(step_cost)
    """

    def __init__(self, stream_table_dependency_df_scaled, leiden_df, communities_subset=None):

        self.dep = (
            stream_table_dependency_df_scaled
            .select("from", "to", "table", "scaled_table_weight")
            .toPandas()
            .copy()
        )

        self.dep["from"] = self.dep["from"].astype(str)
        self.dep["to"] = self.dep["to"].astype(str)
        self.dep["table"] = self.dep["table"].astype(str)
        self.dep["scaled_table_weight"] = pd.to_numeric(self.dep["scaled_table_weight"], errors="coerce").fillna(0.0)

        # table → single weight (max)
        self.table_weight = (
            self.dep.groupby("table")["scaled_table_weight"].max().to_dict()
        )

        # choose communities
        all_comms = sorted(leiden_df["community"].unique().astype(int).tolist())
        if communities_subset is None:
            comms = all_comms
        else:
            comms = sorted([int(c) for c in communities_subset])

        self.community_streams = {
            int(c): set(leiden_df.loc[leiden_df["community"] == c, "stream"].astype(str).tolist())
            for c in comms
        }

        # produced tables per community
        self.produced_tables = {}
        for c, streams in self.community_streams.items():
            self.produced_tables[c] = set(
                self.dep[self.dep["from"].isin(streams)]["table"].unique()
            )

        # incoming tables per community
        self.incoming_tables = {}
        for c, streams in self.community_streams.items():
            incoming = self.dep[
                (~self.dep["from"].isin(streams)) &
                (self.dep["to"].isin(streams))
            ]["table"].unique()
            self.incoming_tables[c] = set(incoming)

        self.communities = list(self.community_streams.keys())

    def evaluate_ordering_cost(self, ordering):
        available = set()
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

        n = len(self.communities)
        total_perms = factorial(n)

        best_cost = float("inf")
        best_order = None
        best_step_costs = None

        start = time.time()

        print(f"\n=== Brute force ({label}) | communities={n} | perms={total_perms} ===")

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

resolution = 1.8
leiden_df, meta = get_leiden_df(resolution, rep_by_res, igraph_names)

top10_ids, rest_ids, _ = split_communities_topN(leiden_df, top_n=10)

dep_scaled_df = unique_table_weights.select("from", "to", "table", "scaled_table_weight")

bf_top = BruteForceCommunityOrderingScaledWeight(dep_scaled_df, leiden_df, top10_ids)
res_top = bf_top.brute_force(log_every=5000, label="TOP10")

bf_rest = BruteForceCommunityOrderingScaledWeight(dep_scaled_df, leiden_df, rest_ids)
res_rest = bf_rest.brute_force(log_every=5000, label="REST")

final_order = res_top["best_order"] + res_rest["best_order"]

print("\nFINAL MERGED ORDER:", final_order)


# COMMAND ----------

# DBTITLE 1,Community Ordering Functions
# import itertools
# from typing import Dict, List, Tuple, Set
# import pandas as pd
# import numpy as np

# class CommunityOrderingOptimizer:
#     """
#     Optimizes the ordering of communities to minimize data synchronization costs.
#     """
    
#     def __init__(self, stream_table_dependency_df, merged_edges_df, leiden_df, size_weight=0.75, count_weight=0.25):
#         """
#         Initialize the optimizer.
        
#         Parameters:
#         -----------
#         stream_table_dependency_df : DataFrame
#             Spark DataFrame with columns [from, to, table, size]
#             'from' writes to 'table', 'to' reads from 'table'
#         merged_edges_df : DataFrame
#             Spark DataFrame with columns [streamA, streamB, weight]
#         leiden_df : DataFrame
#             Pandas DataFrame with columns [stream, community]
#         size_weight : float
#             Weight for table size in cost calculation (default 0.75)
#         count_weight : float
#             Weight for table count in cost calculation (default 0.25)
#         """
#         self.stream_table_pdf = stream_table_dependency_df.toPandas().copy()
#         self.merged_edges_pdf = merged_edges_df.toPandas().copy()
#         self.leiden_df = leiden_df
#         self.size_weight = size_weight
#         self.count_weight = count_weight
        
#         # Build community -> streams mapping
#         self.community_streams = {}
#         for comm in leiden_df['community'].unique():
#             self.community_streams[comm] = set(leiden_df[leiden_df['community'] == comm]['stream'].tolist())
        
#         # Precompute dependencies for each community
#         self._precompute_dependencies()
    
#     def _precompute_dependencies(self):
#         """
#         Precompute for each community:
#         - Tables that need to be synced (read by inside, written by outside)
#         - Which communities produce those tables
#         """
#         self.community_dependencies = {}
        
#         for comm, streams_in_comm in self.community_streams.items():
#             # Find tables that are SRC of streams inside but TGT of streams outside
#             # (Dependencies flowing INTO the community)
#             # Convert set to list for .isin() compatibility
#             streams_list = list(streams_in_comm)
#             incoming_tables = self.stream_table_pdf[
#                 (~self.stream_table_pdf['from'].isin(streams_list)) & 
#                 (self.stream_table_pdf['to'].isin(streams_list))
#             ].copy()
            
#             # For each table, find which community produces it
#             table_info = {}
#             for _, row in incoming_tables.iterrows():
#                 table = row['table']
#                 producer_stream = row['from']
#                 size = float(row['size']) if pd.notna(row['size']) else 0.0
                
#                 # Find which community the producer belongs to
#                 producer_comm = None
#                 for c, streams in self.community_streams.items():
#                     if producer_stream in streams:
#                         producer_comm = c
#                         break
                
#                 if table not in table_info:
#                     table_info[table] = {
#                         'size': size,
#                         'producer_communities': set(),
#                         'producer_streams': set()
#                     }
                
#                 if producer_comm is not None:
#                     table_info[table]['producer_communities'].add(producer_comm)
#                 table_info[table]['producer_streams'].add(producer_stream)
            
#             self.community_dependencies[comm] = table_info
    
#     def calculate_sync_cost(self, community, already_available_tables: Set[str]) -> Tuple[float, int, float, List[str]]:
#         """
#         Calculate sync cost for a community given already available tables.
        
#         Returns:
#         --------
#         (total_cost, num_tables_to_sync, total_size_to_sync, tables_to_sync)
#         """
#         table_info = self.community_dependencies[community]
        
#         tables_to_sync = []
#         total_size = 0.0
        
#         for table, info in table_info.items():
#             if table not in already_available_tables:
#                 tables_to_sync.append(table)
#                 total_size += info['size']
        
#         num_tables = len(tables_to_sync)
        
#         # Normalize for scoring (to make size and count comparable)
#         # Use simple weighted sum
#         cost = (self.size_weight * total_size) + (self.count_weight * num_tables)
        
#         return cost, num_tables, total_size, tables_to_sync
    
#     def get_tables_produced_by_community(self, community) -> Set[str]:
#         """
#         Get all tables that are produced (written) by streams in this community.
#         """
#         streams_in_comm = self.community_streams[community]
        
#         # Convert set to list for .isin() compatibility
#         streams_list = list(streams_in_comm)
#         # Tables where 'from' is in this community
#         produced_tables = self.stream_table_pdf[
#             self.stream_table_pdf['from'].isin(streams_list)
#         ]['table'].unique()
        
#         return set(produced_tables)
    
#     def calculate_benefit_to_remaining(self, candidate_comm, remaining_comms, available_tables):
#         """
#         Calculate how much selecting candidate_comm would reduce sync costs for remaining communities.
        
#         Returns:
#         --------
#         (total_benefit, num_communities_helped, benefit_details)
#         """
#         # Get tables that would become available if we select candidate_comm
#         tables_produced = self.get_tables_produced_by_community(candidate_comm)
#         new_available = available_tables.union(tables_produced)
        
#         total_benefit = 0.0
#         communities_helped = 0
#         benefit_details = []
        
#         for remaining_comm in remaining_comms:
#             # Calculate cost with current available tables
#             cost_before, _, _, _ = self.calculate_sync_cost(remaining_comm, available_tables)
            
#             # Calculate cost with new available tables (after selecting candidate)
#             cost_after, _, _, _ = self.calculate_sync_cost(remaining_comm, new_available)
            
#             benefit = cost_before - cost_after
            
#             if benefit > 0:
#                 communities_helped += 1
#                 total_benefit += benefit
#                 benefit_details.append({
#                     'community': remaining_comm,
#                     'benefit': benefit,
#                     'cost_before': cost_before,
#                     'cost_after': cost_after
#                 })
        
#         return total_benefit, communities_helped, benefit_details
    
#     def lookahead_greedy_ordering(self, initial_top_k=3, lookahead_top_k=4, min_communities_helped=1, max_communities_helped=3) -> Tuple[List[int], Dict]:
#         """
#         Lookahead greedy strategy:
#         1. Start with top K communities with least sync cost
#         2. Then select from top M lowest cost communities, choosing the one that
#            maximizes benefit to 1-3 remaining communities
        
#         Parameters:
#         -----------
#         initial_top_k : int
#             Number of initial communities to select based purely on lowest cost (default 3)
#         lookahead_top_k : int
#             Number of candidates to consider at each step (default 4)
#         min_communities_helped : int
#             Minimum number of communities that should benefit (default 1)
#         max_communities_helped : int
#             Maximum number of communities to consider for benefit calculation (default 3)
        
#         Returns:
#         --------
#         (ordered_communities, metrics)
#         """
#         communities = list(self.community_streams.keys())
#         ordered = []
#         remaining = set(communities)
#         available_tables = set()
        
#         metrics = {
#             'ordering': [],
#             'costs': [],
#             'num_tables': [],
#             'sizes': [],
#             'cumulative_cost': 0.0,
#             'cumulative_tables': 0,
#             'cumulative_size': 0.0,
#             'details': []
#         }
        
#         # Phase 1: Select initial top K communities with lowest cost
#         print(f"  Phase 1: Selecting top {initial_top_k} communities with lowest sync cost...")
        
#         for step in range(min(initial_top_k, len(remaining))):
#             # Calculate costs for all remaining
#             costs = {}
#             for comm in remaining:
#                 cost, num_tables, size, tables = self.calculate_sync_cost(comm, available_tables)
#                 costs[comm] = (cost, num_tables, size, tables)
            
#             # Select the one with lowest cost
#             best_comm = min(costs.keys(), key=lambda c: costs[c][0])
#             cost, num_tables, size, tables = costs[best_comm]
            
#             # Add to ordering
#             ordered.append(best_comm)
#             remaining.remove(best_comm)
            
#             # Update available tables
#             produced_tables = self.get_tables_produced_by_community(best_comm)
#             available_tables.update(produced_tables)
            
#             # Record metrics
#             metrics['ordering'].append(best_comm)
#             metrics['costs'].append(cost)
#             metrics['num_tables'].append(num_tables)
#             metrics['sizes'].append(size)
#             metrics['cumulative_cost'] += cost
#             metrics['cumulative_tables'] += num_tables
#             metrics['cumulative_size'] += size
            
#             metrics['details'].append({
#                 'community': best_comm,
#                 'num_streams': len(self.community_streams[best_comm]),
#                 'tables_to_sync': tables,
#                 'num_tables_to_sync': num_tables,
#                 'size_to_sync_gb': size,
#                 'cost': cost,
#                 'selection_reason': f'Initial top-{initial_top_k} (lowest cost)'
#             })
        
#         # Phase 2: Lookahead selection
#         print(f"  Phase 2: Lookahead selection (considering top {lookahead_top_k} candidates, optimizing for benefit to {min_communities_helped}-{max_communities_helped} communities)...")
        
#         while remaining:
#             # Calculate costs for all remaining
#             costs = {}
#             for comm in remaining:
#                 cost, num_tables, size, tables = self.calculate_sync_cost(comm, available_tables)
#                 costs[comm] = (cost, num_tables, size, tables)
            
#             # Get top M candidates with lowest cost
#             sorted_candidates = sorted(costs.keys(), key=lambda c: costs[c][0])
#             top_candidates = sorted_candidates[:min(lookahead_top_k, len(sorted_candidates))]
            
#             # For each candidate, calculate benefit to remaining communities
#             best_candidate = None
#             best_score = -float('inf')
#             best_info = None
            
#             for candidate in top_candidates:
#                 # Calculate benefit to remaining communities (excluding this candidate)
#                 remaining_others = remaining - {candidate}
#                 total_benefit, num_helped, benefit_details = self.calculate_benefit_to_remaining(
#                     candidate, remaining_others, available_tables
#                 )
                
#                 # Sort benefit details by benefit amount
#                 benefit_details_sorted = sorted(benefit_details, key=lambda x: x['benefit'], reverse=True)
                
#                 # Consider benefit to top 1-3 communities that would benefit most
#                 top_beneficiaries = benefit_details_sorted[:max_communities_helped]
#                 top_benefit = sum(b['benefit'] for b in top_beneficiaries)
                
#                 # Score: prioritize candidates that help at least min_communities_helped
#                 # and maximize benefit to top beneficiaries
#                 if num_helped >= min_communities_helped:
#                     # Score = benefit to top beneficiaries - own cost (to break ties)
#                     score = top_benefit - (costs[candidate][0] * 0.1)  # Small penalty for own cost
#                 else:
#                     # If doesn't help enough communities, just use negative cost
#                     score = -costs[candidate][0]
                
#                 if score > best_score:
#                     best_score = score
#                     best_candidate = candidate
#                     best_info = {
#                         'total_benefit': total_benefit,
#                         'num_helped': num_helped,
#                         'top_beneficiaries': top_beneficiaries,
#                         'top_benefit': top_benefit
#                     }
            
#             # Select best candidate
#             cost, num_tables, size, tables = costs[best_candidate]
            
#             # Add to ordering
#             ordered.append(best_candidate)
#             remaining.remove(best_candidate)
            
#             # Update available tables
#             produced_tables = self.get_tables_produced_by_community(best_candidate)
#             available_tables.update(produced_tables)
            
#             # Record metrics
#             metrics['ordering'].append(best_candidate)
#             metrics['costs'].append(cost)
#             metrics['num_tables'].append(num_tables)
#             metrics['sizes'].append(size)
#             metrics['cumulative_cost'] += cost
#             metrics['cumulative_tables'] += num_tables
#             metrics['cumulative_size'] += size
            
#             # Create selection reason
#             if best_info and best_info['num_helped'] >= min_communities_helped:
#                 helped_comms = [b['community'] for b in best_info['top_beneficiaries']]
#                 selection_reason = f"Lookahead: helps {best_info['num_helped']} communities (top benefit to {helped_comms[:3]}), benefit={best_info['top_benefit']:.2f}"
#             else:
#                 selection_reason = f"Lookahead: lowest cost among top-{lookahead_top_k}"
            
#             metrics['details'].append({
#                 'community': best_candidate,
#                 'num_streams': len(self.community_streams[best_candidate]),
#                 'tables_to_sync': tables,
#                 'num_tables_to_sync': num_tables,
#                 'size_to_sync_gb': size,
#                 'cost': cost,
#                 'selection_reason': selection_reason
#             })
        
#         return ordered, metrics
    
#     def greedy_ordering(self) -> Tuple[List[int], Dict]:
#         """
#         Find a good ordering using greedy algorithm.
#         At each step, choose the community with lowest sync cost given current available tables.
        
#         Returns:
#         --------
#         (ordered_communities, metrics)
#         """
#         communities = list(self.community_streams.keys())
#         ordered = []
#         remaining = set(communities)
#         available_tables = set()
        
#         metrics = {
#             'ordering': [],
#             'costs': [],
#             'num_tables': [],
#             'sizes': [],
#             'cumulative_cost': 0.0,
#             'cumulative_tables': 0,
#             'cumulative_size': 0.0,
#             'details': []  # Added details list
#         }
        
#         while remaining:
#             # Calculate cost for each remaining community
#             best_comm = None
#             best_cost = float('inf')
#             best_num_tables = 0
#             best_size = 0.0
#             best_tables_to_sync = []
            
#             for comm in remaining:
#                 cost, num_tables, size, tables_to_sync = self.calculate_sync_cost(comm, available_tables)
                
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_comm = comm
#                     best_num_tables = num_tables
#                     best_size = size
#                     best_tables_to_sync = tables_to_sync
            
#             # Add best community to ordering
#             ordered.append(best_comm)
#             remaining.remove(best_comm)
            
#             # Update available tables
#             produced_tables = self.get_tables_produced_by_community(best_comm)
#             available_tables.update(produced_tables)
            
#             # Record metrics
#             metrics['ordering'].append(best_comm)
#             metrics['costs'].append(best_cost)
#             metrics['num_tables'].append(best_num_tables)
#             metrics['sizes'].append(best_size)
#             metrics['cumulative_cost'] += best_cost
#             metrics['cumulative_tables'] += best_num_tables
#             metrics['cumulative_size'] += best_size
            
#             # Add detailed information
#             metrics['details'].append({
#                 'community': best_comm,
#                 'num_streams': len(self.community_streams[best_comm]),
#                 'tables_to_sync': best_tables_to_sync,
#                 'num_tables_to_sync': best_num_tables,
#                 'size_to_sync_gb': best_size,
#                 'cost': best_cost
#             })
        
#         return ordered, metrics
    
#     def evaluate_ordering(self, ordering: List[int]) -> Dict:
#         """
#         Evaluate a given ordering and return detailed metrics.
#         """
#         available_tables = set()
        
#         metrics = {
#             'ordering': ordering,
#             'costs': [],
#             'num_tables': [],
#             'sizes': [],
#             'cumulative_cost': 0.0,
#             'cumulative_tables': 0,
#             'cumulative_size': 0.0,
#             'details': []
#         }
        
#         for comm in ordering:
#             cost, num_tables, size, tables_to_sync = self.calculate_sync_cost(comm, available_tables)
            
#             metrics['costs'].append(cost)
#             metrics['num_tables'].append(num_tables)
#             metrics['sizes'].append(size)
#             metrics['cumulative_cost'] += cost
#             metrics['cumulative_tables'] += num_tables
#             metrics['cumulative_size'] += size
            
#             metrics['details'].append({
#                 'community': comm,
#                 'num_streams': len(self.community_streams[comm]),
#                 'tables_to_sync': tables_to_sync,
#                 'num_tables_to_sync': num_tables,
#                 'size_to_sync_gb': size,
#                 'cost': cost
#             })
            
#             # Update available tables
#             produced_tables = self.get_tables_produced_by_community(comm)
#             available_tables.update(produced_tables)
        
#         return metrics
    
#     def try_multiple_strategies(self) -> Dict[str, Tuple[List[int], Dict]]:
#         """
#         Try multiple ordering strategies and return results.
#         """
#         results = {}
        
#         # Strategy 1: Greedy (lowest cost first)
#         print("Running greedy strategy...")
#         ordered, metrics = self.greedy_ordering()
#         results['greedy'] = (ordered, metrics)
        
#         # Strategy 2: Lookahead greedy (new strategy)
#         print("Running lookahead-greedy strategy...")
#         ordered_lookahead, metrics_lookahead = self.lookahead_greedy_ordering(
#             initial_top_k=3,
#             lookahead_top_k=4,
#             min_communities_helped=1,
#             max_communities_helped=3
#         )
#         results['lookahead_greedy'] = (ordered_lookahead, metrics_lookahead)
        
#         # Strategy 3: Smallest communities first
#         print("Running smallest-first strategy...")
#         communities = list(self.community_streams.keys())
#         ordered_by_size = sorted(communities, key=lambda c: len(self.community_streams[c]))
#         metrics_size = self.evaluate_ordering(ordered_by_size)
#         results['smallest_first'] = (ordered_by_size, metrics_size)
        
#         # Strategy 4: Largest communities first
#         print("Running largest-first strategy...")
#         ordered_by_size_desc = sorted(communities, key=lambda c: len(self.community_streams[c]), reverse=True)
#         metrics_size_desc = self.evaluate_ordering(ordered_by_size_desc)
#         results['largest_first'] = (ordered_by_size_desc, metrics_size_desc)
        
#         # Strategy 5: Random baseline (for comparison)
#         print("Running random baseline...")
#         import random
#         random_order = communities.copy()
#         random.shuffle(random_order)
#         metrics_random = self.evaluate_ordering(random_order)
#         results['random'] = (random_order, metrics_random)
        
#         return results

# print("Community ordering optimizer module loaded successfully.")

# COMMAND ----------

# DBTITLE 1,Run Optimization for Selected Resolutions
# # Helper function to get leiden_df from stored results
# def get_leiden_df_for_optimization(resolution, rep_by_res, igraph_names):
#     """
#     Get leiden_df for a chosen resolution from already-computed results (rep_by_res).
#     """
#     rep = rep_by_res[resolution]
#     leiden_df = pd.DataFrame({
#         "stream": np.array(igraph_names),
#         "community": np.array(rep["membership"], dtype=int),
#     })
    
#     meta = {
#         "resolution": float(rep.get("resolution", resolution)),
#         "seed": int(rep["seed"]),
#         "quality": float(rep.get("quality", np.nan)),
#     }
#     return leiden_df, meta

# # Run optimization for selected resolutions
# resolutions_to_optimize = [1.2, 1.8, 2.2]

# optimization_results = {}

# for res in resolutions_to_optimize:
#     print(f"\n{'='*80}")
#     print(f"OPTIMIZING COMMUNITY ORDERING FOR RESOLUTION γ={res}")
#     print(f"{'='*80}\n")
    
#     # Get leiden_df for this resolution
#     leiden_df, meta = get_leiden_df_for_optimization(res, rep_by_res, igraph_names)
    
#     print(f"Resolution: {res}")
#     print(f"Number of communities: {leiden_df['community'].nunique()}")
#     print(f"Number of streams: {len(leiden_df)}")
#     print()
    
#     # Create optimizer
#     optimizer = CommunityOrderingOptimizer(
#         stream_table_dependency_df=stream_stream_dependency_df,
#         merged_edges_df=merged_dependency_df,
#         leiden_df=leiden_df,
#         size_weight=0.75,
#         count_weight=0.25
#     )
    
#     # Try multiple strategies
#     results = optimizer.try_multiple_strategies()
    
#     # Store results
#     optimization_results[res] = results
    
#     # Print comparison
#     print(f"\n{'='*80}")
#     print(f"STRATEGY COMPARISON FOR γ={res}")
#     print(f"{'='*80}\n")
    
#     comparison_data = []
#     for strategy_name, (ordering, metrics) in results.items():
#         comparison_data.append({
#             'Strategy': strategy_name,
#             'Total Cost': f"{metrics['cumulative_cost']:.2f}",
#             'Total Tables to Sync': metrics['cumulative_tables'],
#             'Total Size to Sync (GB)': f"{metrics['cumulative_size']:.2f}",
#             'Ordering': str(ordering[:5]) + '...' if len(ordering) > 5 else str(ordering)
#         })
    
#     comparison_df = pd.DataFrame(comparison_data)
#     print(comparison_df.to_string(index=False))
#     print()
    
#     # Find best strategy
#     best_strategy = min(results.items(), key=lambda x: x[1][1]['cumulative_cost'])
#     print(f"\n✓ BEST STRATEGY: {best_strategy[0].upper()}")
#     print(f"  Total Cost: {best_strategy[1][1]['cumulative_cost']:.2f}")
#     print(f"  Total Tables: {best_strategy[1][1]['cumulative_tables']}")
#     print(f"  Total Size: {best_strategy[1][1]['cumulative_size']:.2f} GB")
#     print()

# print(f"\n{'='*80}")
# print("OPTIMIZATION COMPLETE FOR ALL RESOLUTIONS")
# print(f"{'='*80}")

# COMMAND ----------

# DBTITLE 1,Generate Detailed Reports
# # Generate detailed reports for each resolution

# for res in resolutions_to_optimize:
#     print(f"\n{'='*80}")
#     print(f"DETAILED REPORT FOR RESOLUTION γ={res}")
#     print(f"{'='*80}\n")
    
#     results = optimization_results[res]
#     best_strategy_name = min(results.items(), key=lambda x: x[1][1]['cumulative_cost'])[0]
#     ordering, metrics = results[best_strategy_name]
    
#     print(f"Strategy: {best_strategy_name.upper()}")
#     print(f"Number of communities: {len(ordering)}")
#     print()
    
#     # Create detailed report DataFrame
#     report_data = []
#     for detail in metrics['details']:
#         report_data.append({
#             'Order': len(report_data) + 1,
#             'Community': detail['community'],
#             'Streams': detail['num_streams'],
#             'Tables to Sync': detail['num_tables_to_sync'],
#             'Size (GB)': f"{detail['size_to_sync_gb']:.2f}",
#             'Cost': f"{detail['cost']:.2f}"
#         })
    
#     report_df = pd.DataFrame(report_data)
#     print("\nCOMMUNITY ORDERING WITH SYNC REQUIREMENTS:")
#     print(report_df.to_string(index=False))
    
#     # Summary statistics
#     print(f"\n\nSUMMARY STATISTICS:")
#     print(f"{'─'*80}")
#     print(f"Total communities: {len(ordering)}")
#     print(f"Total tables to sync: {metrics['cumulative_tables']}")
#     print(f"Total size to sync: {metrics['cumulative_size']:.2f} GB")
#     print(f"Total cost: {metrics['cumulative_cost']:.2f}")
#     print(f"Average tables per community: {metrics['cumulative_tables']/len(ordering):.2f}")
#     print(f"Average size per community: {metrics['cumulative_size']/len(ordering):.2f} GB")
#     print(f"Average cost per community: {metrics['cumulative_cost']/len(ordering):.2f}")
    
#     # Communities with highest sync requirements
#     print(f"\n\nTOP 5 COMMUNITIES BY SYNC COST:")
#     print(f"{'─'*80}")
#     sorted_details = sorted(metrics['details'], key=lambda x: x['cost'], reverse=True)[:5]
#     for i, detail in enumerate(sorted_details, 1):
#         print(f"{i}. Community {detail['community']}: {detail['num_tables_to_sync']} tables, "
#               f"{detail['size_to_sync_gb']:.2f} GB, cost={detail['cost']:.2f}")
    
#     # Communities with no sync requirements
#     no_sync = [d for d in metrics['details'] if d['num_tables_to_sync'] == 0]
#     print(f"\n\nCOMMUNITIES WITH NO SYNC REQUIREMENTS: {len(no_sync)}")
#     if no_sync:
#         print(f"Communities: {[d['community'] for d in no_sync]}")
    
#     print(f"\n{'─'*80}\n")

# print(f"\n{'='*80}")
# print("ALL REPORTS GENERATED SUCCESSFULLY")
# print(f"{'='*80}")

# COMMAND ----------

# DBTITLE 1,Visualize Ordering Results
# # Create visualizations for the optimization results

# for res in resolutions_to_optimize:
#     results = optimization_results[res]
#     best_strategy_name = min(results.items(), key=lambda x: x[1][1]['cumulative_cost'])[0]
#     ordering, metrics = results[best_strategy_name]
    
#     # Create figure with multiple subplots
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#     fig.suptitle(f'Community Ordering Optimization Results (γ={res}, Strategy: {best_strategy_name})', 
#                  fontsize=16, fontweight='bold')
    
#     # Plot 1: Sync cost per community
#     ax1 = axes[0, 0]
#     communities = [d['community'] for d in metrics['details']]
#     costs = [d['cost'] for d in metrics['details']]
#     colors = plt.cm.viridis(np.linspace(0, 1, len(communities)))
#     ax1.bar(range(len(communities)), costs, color=colors)
#     ax1.set_xlabel('Order Position', fontsize=12)
#     ax1.set_ylabel('Sync Cost', fontsize=12)
#     ax1.set_title('Sync Cost per Community (in order)', fontsize=14, fontweight='bold')
#     ax1.grid(axis='y', alpha=0.3)
    
#     # Plot 2: Cumulative cost
#     ax2 = axes[0, 1]
#     cumulative_costs = np.cumsum(costs)
#     ax2.plot(range(len(communities)), cumulative_costs, marker='o', linewidth=2, markersize=6)
#     ax2.fill_between(range(len(communities)), cumulative_costs, alpha=0.3)
#     ax2.set_xlabel('Order Position', fontsize=12)
#     ax2.set_ylabel('Cumulative Sync Cost', fontsize=12)
#     ax2.set_title('Cumulative Sync Cost', fontsize=14, fontweight='bold')
#     ax2.grid(alpha=0.3)
    
#     # Plot 3: Tables to sync per community
#     ax3 = axes[1, 0]
#     num_tables = [d['num_tables_to_sync'] for d in metrics['details']]
#     ax3.bar(range(len(communities)), num_tables, color=colors)
#     ax3.set_xlabel('Order Position', fontsize=12)
#     ax3.set_ylabel('Number of Tables', fontsize=12)
#     ax3.set_title('Tables to Sync per Community', fontsize=14, fontweight='bold')
#     ax3.grid(axis='y', alpha=0.3)
    
#     # Plot 4: Size to sync per community
#     ax4 = axes[1, 1]
#     sizes = [d['size_to_sync_gb'] for d in metrics['details']]
#     ax4.bar(range(len(communities)), sizes, color=colors)
#     ax4.set_xlabel('Order Position', fontsize=12)
#     ax4.set_ylabel('Size (GB)', fontsize=12)
#     ax4.set_title('Data Size to Sync per Community', fontsize=14, fontweight='bold')
#     ax4.grid(axis='y', alpha=0.3)
    
#     plt.tight_layout()
    
#     # Save figure
#     viz_file = os.path.join(volume_path, f"community_ordering_visualization_gamma_{res}.png")
#     plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"✓ Visualization saved: {viz_file}")
#     plt.show()

# print(f"\n{'='*80}")
# print("ALL VISUALIZATIONS GENERATED")
# print(f"{'='*80}")