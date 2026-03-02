# Databricks notebook source
# MAGIC %md
# MAGIC ## Libraries and Parameters
# MAGIC

# COMMAND ----------

!pip install python-igraph adjustText infomap netgraph

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

# dbutils.fs.ls("file:///Workspace/Users/aspram.grigoryan@databricks.com/LH_Stream_Community_Detection")

# COMMAND ----------

# dbutils.fs.cp("file:/Workspace/Users/aspram.grigoryan@databricks.com/LH_Stream_Community_Detection/Stream-Table-dependencies-All-ADW-Streams-Tabelle1.csv",
#               "dbfs:/tmp/new_data.csv")

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

# Build the directed graph
# <TODO> Not valid anymore, since the edges have been merged where there are bidirectional dependencies
G_directed = nx.from_pandas_edgelist(edges, source='streamA', target='streamB', edge_attr=True, create_using=nx.DiGraph())

# COMMAND ----------

# MAGIC %md
# MAGIC # Community Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ## Infomap
# MAGIC - works through random walk and information flow from one node to another. 
# MAGIC - minimizes amount of information needed to describe a random walk over the network.
# MAGIC - works with directed and weighted graphs
# MAGIC - requires a lot of memory

# COMMAND ----------

# ==============================================
# Robust Infomap sweep over Markov times
# ==============================================

from infomap import Infomap
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- PARAMETERS -----------------------
markov_times = np.linspace(0.01, 1.0, 10)  # 10 values
n_trials = 1000  # runs per Markov time
distance_threshold = 0.02  # for hierarchical clustering

# ----------- GRAPH → index mappings ----------
node_mapping = {node: i for i, node in enumerate(G_directed.nodes())}
inverse_mapping = {i: node for node, i in node_mapping.items()}
n_nodes = len(node_mapping)


# ----------- helper: Jaccard distance --------
def jaccard_distance(u, v):
    return 1.0 - np.sum(u == v) / len(u)


# ----------- helper: analyse one sweep -------
def analyse_partitions(partitions_df, codelengths, threshold):
    n = partitions_df.shape[1]

    # --- pairwise Jaccard distances ------------
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(partitions_df.iloc[:, i], partitions_df.iloc[:, j])
            dist[i, j] = dist[j, i] = d

    # --- clustering ---------------------------
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold
    )
    labels = clustering.fit_predict(dist)

    # --- t-SNE embedding ----------------------
    embed = TSNE(n_components=2, metric="precomputed", init="random", perplexity=30, random_state=42).fit_transform(
        dist
    )

    # --- landscape dataframe ------------------
    df = pd.DataFrame(
        {"X": embed[:, 0], "Y": embed[:, 1], "Cluster": labels, "Codelength": codelengths, "PartitionID": range(n)}
    )
    return df


# ----------- storage for summary -------------
summary_rows = []
best_global = {"cluster_size": -1}  # keep track of overall best

# =============================================
# MAIN LOOP:  markov_time  →  replicate runs
# =============================================
for m_time in markov_times:
    print(f"\n=== Markov time {m_time:.2f} ===")
    partitions = np.zeros((n_nodes, n_trials), dtype=np.int32)
    codelengths = np.zeros(n_trials)

    # ---------- run Infomap repeatedly --------
    for run in range(n_trials):
        seed = 123 + run
        im = Infomap(f"--directed --markov-time {m_time} --silent -s {seed}")

        for s, t, data in G_directed.edges(data=True):
            im.addLink(node_mapping[s], node_mapping[t], data.get("weight", 1.0))

        im.run()
        modules = im.get_modules()
        partitions[:, run] = [modules.get(i, -1) for i in range(n_nodes)]
        codelengths[run] = im.codelength

    # ---------- analyse solution space --------
    part_df = pd.DataFrame(partitions)
    landscape_df = analyse_partitions(part_df, codelengths, distance_threshold)

    # ---------- plot landscape ----------------
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=landscape_df,
        x="X",
        y="Y",
        hue="Cluster",
        size=landscape_df.groupby("Cluster")["Cluster"].transform("count"),
        palette="tab10",
        sizes=(30, 250),
        alpha=0.85,
        legend=False,
    )
    plt.title(f"Solution landscape  •  Markov-time {m_time:.2f}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

    # ---------- cluster stats -----------------
    stats = (
        landscape_df.groupby("Cluster")
        .agg(Size=("Cluster", "size"), MeanCL=("Codelength", "mean"))
        .sort_values(["Size", "MeanCL"], ascending=[False, True])
    )
    top_cluster_id = stats.index[0]

    summary_rows.append(
        {
            "MarkovTime": m_time,
            "LargestClusterSize": int(stats.iloc[0]["Size"]),
            "LargestClusterMeanCL": float(stats.iloc[0]["MeanCL"]),
            "TopClusterID": int(top_cluster_id),
            "partitions_df": part_df,  # ✅ this enables later analysis
        }
    )

    # ---------- track global best -------------
    if stats.iloc[0]["Size"] > best_global.get("cluster_size", -1):
        best_global.update(
            {
                "cluster_size": int(stats.iloc[0]["Size"]),
                "markov_time": m_time,
                "cluster_id": int(top_cluster_id),
                "partitions_df": part_df,
                "landscape_df": landscape_df,
            }
        )

# =============================================
#  SUMMARY  -----------------------------------
summary_df = pd.DataFrame(summary_rows)
print("\nStability / quality summary:")
print(summary_df.sort_values("LargestClusterSize", ascending=False).to_string(index=False))


# --- Dual-axis line plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = "tab:blue"
ax1.set_xlabel("Markov Time")
ax1.set_ylabel("Largest Cluster Size", color=color1)
ax1.plot(summary_df["MarkovTime"], summary_df["LargestClusterSize"], marker='o', color=color1, label="Cluster Size")
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Mean Codelength", color=color2)
ax2.plot(
    summary_df["MarkovTime"], summary_df["LargestClusterMeanCL"], marker='s', color=color2, label="Mean Codelength"
)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("Markov Time vs Cluster Stability and Codelength")
fig.tight_layout()
plt.show()


# =============================================
#  FINAL BEST PARTITION
# =============================================
print(
    f"\nSelected best Markov time: {best_global['markov_time']:.2f} " f"(cluster size = {best_global['cluster_size']})"
)

# representative partition: first run in the winning cluster
win_df = best_global["landscape_df"]
rep_pid = int(win_df[win_df["Cluster"] == best_global["cluster_id"]].iloc[0]["PartitionID"])
best_partition_vec = best_global["partitions_df"].iloc[:, rep_pid]

final_partition = {inverse_mapping[i]: int(best_partition_vec.iloc[i]) for i in range(n_nodes)}
nx.set_node_attributes(G_directed, final_partition, "best_community")
print("✅ Assigned 'best_community' to G_directed nodes.")

# COMMAND ----------

# Count number of communities for every trial at each Markov time
all_counts = []

for row in summary_rows:
    m_time = row["MarkovTime"]
    part_df = row["partitions_df"]

    # For each trial/column, count how many unique community IDs it contains
    counts = part_df.apply(lambda col: col.nunique(), axis=0)

    for trial_id, n_comms in enumerate(counts):
        all_counts.append({"MarkovTime": m_time, "TrialID": trial_id, "NumCommunities": n_comms})

# Create DataFrame for plotting
community_counts_df = pd.DataFrame(all_counts)

# COMMAND ----------

plt.figure(figsize=(10, 6))
sns.boxplot(data=community_counts_df, x="MarkovTime", y="NumCommunities")
plt.title("Number of Communities per Trial vs Markov Time")
plt.xlabel("Markov Time")
plt.ylabel("Number of Communities")
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Markov time stretches or compresses the amount of "flow" the algorithm simulates on the graph before optimizing community structure.
# MAGIC
# MAGIC
# MAGIC At Markov time = 0.01, the walker essentially only considers immediate neighbors and maybe second-order neighbors.
# MAGIC
# MAGIC The flow stays highly localized.
# MAGIC
# MAGIC The resulting partition tends to group tightly connected small regions (e.g., cliques or star structures).
# MAGIC
# MAGIC How “Short” Is Short?
# MAGIC Its relative to:
# MAGIC
# MAGIC - The graph's topology (density, connectivity, cycles)
# MAGIC
# MAGIC - The stationary distribution of the random walk
# MAGIC
# MAGIC - The expected mixing time (i.e., how long until the walk loses memory of its starting point)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The largest cluster with the lowest average codelength is chosen.

# COMMAND ----------

best_partition = nx.get_node_attributes(G_directed, "best_community")
num_communities = len(set(best_partition.values()))
print(f"Number of communities: {num_communities}")

# COMMAND ----------

part_df = best_global["partitions_df"]  # shape: (n_nodes, n_trials)
landscape_df = best_global["landscape_df"]  # contains PartitionID

# Compute number of unique communities for each trial
num_communities_per_trial = part_df.apply(lambda col: col.nunique(), axis=0)

# Merge into a DataFrame for plotting
trial_stats = pd.DataFrame(
    {"PartitionID": num_communities_per_trial.index, "NumCommunities": num_communities_per_trial.values}
).merge(landscape_df[["PartitionID", "Cluster"]], on="PartitionID")

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Get community assignments
node_colors = [G_directed.nodes[n]["best_community"] for n in G_directed.nodes()]
uniform_node_size = 100

# Step 2: Compute layout
pos = nx.spring_layout(G_directed, seed=2)

# Step 3: Plot
plt.figure(figsize=(12, 8))

nx.draw_networkx_edges(G_directed, pos, edge_color="gray")
nodes = nx.draw_networkx_nodes(
    G_directed,
    pos,
    node_size=uniform_node_size,
    node_color=node_colors,
    cmap=plt.cm.Spectral,  # or `viridis`, `tab20`, etc.
)

plt.title("Graph Visualization with Best Infomap Communities")
plt.axis("off")
plt.savefig("/Volumes/users/jince_james/lufthansa/community_detection/svg_plots/graph_best_infomap_communities.svg", format="svg", bbox_inches='tight')

plt.show()

# COMMAND ----------

from collections import defaultdict

# Group nodes by their assigned community
communities = defaultdict(list)
for node, data in G_directed.nodes(data=True):
    comm_id = data["best_community"]
    communities[comm_id].append(node)

# Draw each community's subgraph
for community_id, nodes in communities.items():
    subgraph = G_directed.subgraph(nodes)
    pos = nx.spring_layout(subgraph, seed=2, k=2)

    plt.figure(figsize=(8, 6))
    nx.draw(
        subgraph, pos, node_size=300, node_color=[community_id] * len(nodes), cmap=plt.cm.Spectral, with_labels=False
    )

    # Optional: add node labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color="black")

    # Optional: edge weights
    edge_labels = {(u, v): d['weight'] for u, v, d in subgraph.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Community {int(community_id)}")
    plt.axis("off")
    plt.savefig(
        f"/Volumes/users/jince_james/lufthansa/community_detection/svg_plots//graph_visualization_infomap_clusters_{community_id}.svg", format="svg", bbox_inches='tight'
    )
    plt.show()
