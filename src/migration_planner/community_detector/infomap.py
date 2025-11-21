# Databricks notebook source
!pip install python-igraph adjustText infomap netgraph

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Libraries needed:
# MAGIC

# COMMAND ----------

# Import libraries
import pandas as pd #For reading dataset files
import networkx as nx #For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt #For plotting graphs
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

# MAGIC %md
# MAGIC # Loading Datasets:

# COMMAND ----------

dbutils.fs.ls("file:///Workspace/Users/aspram.grigoryan@databricks.com/LH_Stream_Community_Detection")


# COMMAND ----------

# dbutils.fs.cp("file:/Workspace/Users/aspram.grigoryan@databricks.com/LH_Stream_Community_Detection/Stream-Table-dependencies-All-ADW-Streams-Tabelle1.csv",
#               "dbfs:/tmp/new_data.csv")


# COMMAND ----------

df2 = spark.read.format("csv").option("header", "true").load("dbfs:/tmp/new_data.csv")

# COMMAND ----------

# df2.count()

# COMMAND ----------


# df2 = spark.read.format("csv").option("header", "true").load("dbfs:/tmp/source_transform_target_tables_per_stream.csv")

depend_df = spark.table('odp_adw_mvp_n.db_cascade.cascade_dependants').withColumn('weight', lit('')).withColumn('type', lit('casacade_dependent'))

df = df2.filter(col('table_type') != 'Trns').select('stream_name', col('DB_Table_Name').alias('table_name'), 'table_type')
display(df)

# COMMAND ----------

# Self join to find dependencies
result = df.alias("df1").join(
    df.alias("df2"),
    col("df1.table_name") == col("df2.table_name")
)

display(result.filter(col("df2.stream_name").contains("IMP_LOCATIONS") == True))

# COMMAND ----------

# Filter only Src -> Tgt dependencies
filtered_result = result.filter(
    (col("df1.table_type") == "Src") & (col("df2.table_type") == "Tgt")
)

# Select required columns and rename them
output = filtered_result.select(
    col("df1.stream_name").alias("from"),
    col("df2.stream_name").alias("to"),
    col("df1.table_name").alias("table"),
    lit('dependent').alias('')
)

output.show()

# COMMAND ----------

# output = output.union(depend_df)
# # Show the output
# result = output.groupBy('from','to').count().select('from','to',col('count').alias('weight'))
# result1 = result.filter(col('from') != col('to'))
# edges = result1.toPandas()
# edges = edges.groupby(['from', 'to'])['weight'].sum().reset_index()
# G_directed = nx.from_pandas_edgelist(edges, source='from', target='to',edge_attr=True, create_using=nx.DiGraph())

# COMMAND ----------

# Group by 'from' and 'to', count occurrences, and compute weights
result = output.groupBy('from', 'to').count().select('from', 'to', col('count').alias('weight'))

# Filter out self-loops (where 'from' == 'to')
result1 = result.filter(col('from') != col('to'))




# COMMAND ----------

# Only use for small data!
edges_df = result1.toPandas()
edges_df.to_csv("/Volumes/migration_planner/default/dependency_data/edges.csv", index=False)


# COMMAND ----------

# Aggregate weights if necessary (depends on whether the grouping already handles this)
edges = edges_df.groupby(['from', 'to'])['weight'].sum().reset_index()

# Build the directed graph
G_directed = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True, create_using=nx.DiGraph())


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
markov_times = np.linspace(0.01, 1.0, 10)   # 10 values
n_trials     = 1000                         # runs per Markov time
distance_threshold = 0.02                   # for hierarchical clustering

# ----------- GRAPH → index mappings ----------
node_mapping   = {node: i for i, node in enumerate(G_directed.nodes())}
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
        n_clusters=None, metric="precomputed",
        linkage="average", distance_threshold=threshold
    )
    labels = clustering.fit_predict(dist)

    # --- t-SNE embedding ----------------------
    embed = TSNE(
        n_components=2, metric="precomputed",
        init="random", perplexity=30, random_state=42
    ).fit_transform(dist)

    # --- landscape dataframe ------------------
    df = pd.DataFrame({
        "X": embed[:, 0],
        "Y": embed[:, 1],
        "Cluster": labels,
        "Codelength": codelengths,
        "PartitionID": range(n)
    })
    return df

# ----------- storage for summary -------------
summary_rows = []
best_global   = {"cluster_size": -1}        # keep track of overall best

# =============================================
# MAIN LOOP:  markov_time  →  replicate runs
# =============================================
for m_time in markov_times:
    print(f"\n=== Markov time {m_time:.2f} ===")
    partitions   = np.zeros((n_nodes, n_trials), dtype=np.int32)
    codelengths  = np.zeros(n_trials)

    # ---------- run Infomap repeatedly --------
    for run in range(n_trials):
        seed = 123 + run
        im = Infomap(f"--directed --markov-time {m_time} --silent -s {seed}")

        for s, t, data in G_directed.edges(data=True):
            im.addLink(node_mapping[s], node_mapping[t], data.get("weight", 1.0))

        im.run()
        modules = im.get_modules()
        partitions[:, run] = [modules.get(i, -1) for i in range(n_nodes)]
        codelengths[run]   = im.codelength

    # ---------- analyse solution space --------
    part_df  = pd.DataFrame(partitions)
    landscape_df = analyse_partitions(part_df, codelengths, distance_threshold)

    # ---------- plot landscape ----------------
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=landscape_df, x="X", y="Y",
        hue="Cluster",
        size=landscape_df.groupby("Cluster")["Cluster"].transform("count"),
        palette="tab10", sizes=(30, 250), alpha=0.85, legend=False
    )
    plt.title(f"Solution landscape  •  Markov-time {m_time:.2f}")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

    # ---------- cluster stats -----------------
    stats = (landscape_df
             .groupby("Cluster")
             .agg(Size=("Cluster", "size"), MeanCL=("Codelength", "mean"))
             .sort_values(["Size", "MeanCL"], ascending=[False, True]))
    top_cluster_id = stats.index[0]



    summary_rows.append({
    "MarkovTime": m_time,
    "LargestClusterSize": int(stats.iloc[0]["Size"]),
    "LargestClusterMeanCL": float(stats.iloc[0]["MeanCL"]),
    "TopClusterID": int(top_cluster_id),
    "partitions_df": part_df  # ✅ this enables later analysis
})



    # ---------- track global best -------------
    if stats.iloc[0]["Size"] > best_global.get("cluster_size", -1):
        best_global.update({
            "cluster_size": int(stats.iloc[0]["Size"]),
            "markov_time" : m_time,
            "cluster_id"  : int(top_cluster_id),
            "partitions_df": part_df,
            "landscape_df" : landscape_df
        })

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
ax1.plot(summary_df["MarkovTime"], summary_df["LargestClusterSize"],
         marker='o', color=color1, label="Cluster Size")
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Mean Codelength", color=color2)
ax2.plot(summary_df["MarkovTime"], summary_df["LargestClusterMeanCL"],
         marker='s', color=color2, label="Mean Codelength")
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("Markov Time vs Cluster Stability and Codelength")
fig.tight_layout()
plt.show()




# =============================================
#  FINAL BEST PARTITION
# =============================================
print(f"\nSelected best Markov time: {best_global['markov_time']:.2f} "
      f"(cluster size = {best_global['cluster_size']})")

# representative partition: first run in the winning cluster
win_df = best_global["landscape_df"]
rep_pid = int(win_df[win_df["Cluster"] == best_global["cluster_id"]]
              .iloc[0]["PartitionID"])
best_partition_vec = best_global["partitions_df"].iloc[:, rep_pid]

final_partition = {inverse_mapping[i]: int(best_partition_vec.iloc[i])
                   for i in range(n_nodes)}
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
        all_counts.append({
            "MarkovTime": m_time,
            "TrialID": trial_id,
            "NumCommunities": n_comms
        })

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
trial_stats = pd.DataFrame({
    "PartitionID": num_communities_per_trial.index,
    "NumCommunities": num_communities_per_trial.values
}).merge(landscape_df[["PartitionID", "Cluster"]], on="PartitionID")


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
    cmap=plt.cm.Spectral  # or `viridis`, `tab20`, etc.
)

plt.title("Graph Visualization with Best Infomap Communities")
plt.axis("off")
plt.savefig("svg_plots_latest/graph_best_infomap_communities.svg", format="svg", bbox_inches='tight')
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
        subgraph,
        pos,
        node_size=300,
        node_color=[community_id] * len(nodes),
        cmap=plt.cm.Spectral,
        with_labels=False
    )

    # Optional: add node labels
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color="black")

    # Optional: edge weights
    edge_labels = {(u, v): d['weight'] for u, v, d in subgraph.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Community {int(community_id)}")
    plt.axis("off")
    plt.savefig(f"svg_plots_latest/graph_visualization_infomap_clusters_{community_id}.svg", format="svg", bbox_inches='tight')
    plt.show()
