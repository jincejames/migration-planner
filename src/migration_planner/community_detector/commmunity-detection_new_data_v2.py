# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC + **Network Communities using networkx/igraph**
# MAGIC   1. Edge betweenness (Girvan–Newman)
# MAGIC   2. Modularity maximization
# MAGIC   3. Label propagation
# MAGIC   4. Fast community unfolding (Louvian)
# MAGIC   5. Walktrap
# MAGIC   6. k-clique community

# COMMAND ----------

!pip install python-louvain python-igraph adjustText infomap netgraph

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
import community as community_louvain
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

dbutils.fs.cp("file:/Workspace/Users/aspram.grigoryan@databricks.com/LH_Stream_Community_Detection/Stream-Table-dependencies-All-ADW-Streams-Tabelle1.csv",
              "dbfs:/tmp/new_data.csv")


# COMMAND ----------

# df2 = spark.read.format("csv").option("header", "true").load("dbfs:/tmp/new_data.csv")

# COMMAND ----------

# df2.count()

# COMMAND ----------


df2 = spark.read.format("csv").option("header", "true").load("dbfs:/tmp/source_transform_target_tables_per_stream.csv")

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

# Convert to Pandas DataFrame for networkx processing
edges = result1.toPandas()

# Aggregate weights if necessary (depends on whether the grouping already handles this)
edges = edges.groupby(['from', 'to'])['weight'].sum().reset_index()

# Build the directed graph
G_directed = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True, create_using=nx.DiGraph())



# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's visualize the Stream Dependencies by using the centrality measure of the streams

# COMMAND ----------

# Calculate centrality
centrality = degree_centrality(G_directed)

# Sort nodes by centrality (largest to smallest) to ensure big nodes are on top
sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=False)
node_sizes = [v * 3000 for _, v in sorted_nodes]
node_colors = [v for _, v in sorted_nodes]
sorted_nodes_list = [n for n, _ in sorted_nodes]  # Reorder nodes

# Create position layout
pos = nx.spring_layout(G_directed, seed = 2)

# Draw the graph
plt.figure(figsize=(12, 8))

nx.draw_networkx_edges(G_directed, pos, edge_color='gray')  # Draw edges first
nodes = nx.draw_networkx_nodes(
    G_directed, pos,
    nodelist=sorted_nodes_list,  # Use sorted nodes for correct layering
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.Reds  # Use a color map for centrality
)
# Disable labels for simplicity
nx.draw_networkx_labels(G_directed, pos, labels={}, font_size=8)


# Add title and show the plot
plt.title("Stream Dependency Graph with Centrality")

plt.savefig("svg_plots_latest/graph_visualization_stream_dependency_with_centrality.svg", format="svg", bbox_inches='tight')


plt.show()


# COMMAND ----------

sorted_nodes
# Convert the sorted list of tuples to a DataFrame
df = pd.DataFrame(sorted_nodes, columns=["node", "centrality"])

output_path = "./centrality_sorted_nodes_latest/sorted_nodes_overall_centrality.csv"  # Save file in the workspace directory

# Save DataFrame to CSV without the index column
df.to_csv(output_path, index=False)

print(f"CSV saved to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## In-degree Centrality

# COMMAND ----------

# MAGIC %md
# MAGIC Measures the number of incoming connections a node has.

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

# Calculate in-degree centrality
in_degree_centrality = nx.in_degree_centrality(G_directed)

# Sort nodes by in-degree centrality
sorted_in_nodes = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=False)
node_sizes_in = [v * 3000 for _, v in sorted_in_nodes]
node_colors_in = [v for _, v in sorted_in_nodes]
sorted_nodes_list_in = [n for n, _ in sorted_in_nodes]  # Reorder nodes

# Set a threshold for labeling (e.g., nodes with centrality >= 0.1)
label_threshold = 0.2
labels = {n: n if in_degree_centrality[n] >= label_threshold else '' for n in G_directed.nodes()}

# Create position layout
pos = nx.spring_layout(G_directed, seed = 2)

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G_directed, pos, edge_color='gray')
nodes = nx.draw_networkx_nodes(
    G_directed, pos,
    nodelist=sorted_nodes_list_in,
    node_size=node_sizes_in,
    node_color=node_colors_in,
    cmap=plt.cm.Blues
)

# Add labels with adjustment
texts = []
for node, (x, y) in pos.items():
    if labels[node]:  # Only add labels for nodes above the threshold
        texts.append(plt.text(x, y, labels[node], fontsize=6, color='black'))

# Adjust text to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# Add title and colorbar
plt.title("Stream Dependency Graph with In-Degree Centrality")
plt.colorbar(nodes, label="In-Degree Centrality")

plt.savefig("svg_plots_latest/graph_visualization_stream_dependency_with_in_centrality.svg", format="svg", bbox_inches='tight')

plt.show()


# COMMAND ----------


# Convert the sorted list of tuples to a DataFrame
df = pd.DataFrame(sorted_in_nodes, columns=["node", "centrality"])

output_path = "./centrality_sorted_nodes_latest/sorted_nodes_indegree_centrality.csv"  # Save file in the workspace directory

# Save DataFrame to CSV without the index column
df.to_csv(output_path, index=False)

print(f"CSV saved to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Out-Degree Centrality

# COMMAND ----------

# MAGIC %md
# MAGIC Measures the number of outgoing connections a node has.

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

# Calculate out-degree centrality
out_degree_centrality = nx.out_degree_centrality(G_directed)

# Sort nodes by out-degree centrality
sorted_out_nodes = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=False)
node_sizes_out = [v * 3000 for _, v in sorted_out_nodes]
node_colors_out = [v for _, v in sorted_out_nodes]
sorted_nodes_list_out = [n for n, _ in sorted_out_nodes]  # Reorder nodes

# Set a threshold for labeling (e.g., nodes with centrality >= 0.1)
label_threshold = 0.12
labels = {n: n if out_degree_centrality[n] >= label_threshold else '' for n in G_directed.nodes()}

# Create position layout
pos = nx.spring_layout(G_directed, seed = 5)

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G_directed, pos, edge_color='gray')
nodes = nx.draw_networkx_nodes(
    G_directed, pos,
    nodelist=sorted_nodes_list_out,
    node_size=node_sizes_out,
    node_color=node_colors_out,
    cmap=plt.cm.Greens
)

# Add labels with adjustment
texts = []
for node, (x, y) in pos.items():
    if labels[node]:  # Only add labels for nodes above the threshold
        texts.append(plt.text(x, y, labels[node], fontsize=6, color='black'))

# Adjust text to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# Add title and colorbar
plt.title("Stream Dependency Graph with Out-Degree Centrality")
plt.colorbar(nodes, label="Out-Degree Centrality")
plt.savefig("svg_plots_latest/graph_visualization_stream_dependency_with_out_centrality.svg", format="svg", bbox_inches='tight')

plt.show()


# COMMAND ----------


# Convert the sorted list of tuples to a DataFrame
df = pd.DataFrame(sorted_out_nodes, columns=["node", "centrality"])

output_path = "./centrality_sorted_nodes_latest/sorted_nodes_outdegree_centrality.csv"  # Save file in the workspace directory

# Save DataFrame to CSV without the index column
df.to_csv(output_path, index=False)

print(f"CSV saved to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC Streams with a high degree of connectivity (many incoming or outgoing edges) are critical because they influence or depend on multiple other streams.   
# MAGIC Streams with many outgoing edges are producers (important data sources).  
# MAGIC Streams with many incoming edges are consumers (dependent on multiple upstream processes).  
# MAGIC As next steps we need to: 
# MAGIC - The graph shows how streams depend on each other. Migration must respect these dependencies, need to migrate producer streams (with outgoing edges) before their dependent consumer streams (with incoming edges).
# MAGIC - Analyze clusters of tightly connected nodes (streams with interdependencies) to identify complex sections of the pipeline. Cycles (if present) indicate circular dependencies that might complicate migration or testing.
# MAGIC - Streams with higher edge weights indicate more significant dependencies (e.g., many shared tables). These require thorough testing and validation during migration. Streams with low connectivity might be easier to migrate in isolation.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Community Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fast community unfolding (Louvian)
# MAGIC **can play with the resolution parameter to force a bigger number of clusters. However, cluster uniformity cannot be controlled.  
# MAGIC **does not take into account the directedness of the graph
# MAGIC ***the algo is not deterministic

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs
import igraph as ig
import random

G_undirected = nx.from_pandas_edgelist(
    edges, source="from", target="to", edge_attr=True
)

# Use Louvain community detection algorithm to find communities
lst_m = community_louvain.best_partition(G_undirected, resolution=2.5)

# Extract cluster information
clusters = {}
for node, cluster_id in lst_m.items():
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(node)

# Create a DataFrame for tabular representation
cluster_data = {"Cluster ID": [], "Number of Nodes": [], "List of Nodes": []}

for cluster_id, nodes in clusters.items():
    cluster_data["Cluster ID"].append(cluster_id)
    cluster_data["Number of Nodes"].append(len(nodes))
    cluster_data["List of Nodes"].append(", ".join(nodes))

# Convert dictionary to DataFrame
df_clusters = pd.DataFrame(cluster_data)

# Display the DataFrame
display(df_clusters)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consensus Clustering

# COMMAND ----------

# MAGIC %md
# MAGIC determined by finding the most frequent community assignment for each node across all runs.

# COMMAND ----------

import pandas as pd
import networkx as nx
import community as community_louvain
import numpy as np
from sklearn.metrics import adjusted_rand_score
from collections import Counter

# Number of runs
num_runs = 1000

# Store partitions from multiple runs
partitions = []

for i in range(num_runs):
    partition = community_louvain.best_partition(G_undirected, resolution=4)
    partitions.append(partition)

# Convert partitions to a node-community matrix
nodes = list(G_undirected.nodes())
partition_matrix = np.zeros((len(nodes), num_runs))

for run_idx, partition in enumerate(partitions):
    for node_idx, node in enumerate(nodes):
        partition_matrix[node_idx, run_idx] = partition[node]

# Determine consensus partition (most common assignment for each node)
consensus_partition = {}
for node_idx, node in enumerate(nodes):
    cluster_assignments = partition_matrix[node_idx, :]
    consensus_partition[node] = Counter(cluster_assignments).most_common(1)[0][0]

# Organize consensus partition into clusters
clusters = {}
for node, cluster_id in consensus_partition.items():
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(node)

# Create a DataFrame for tabular representation
cluster_data = {"Cluster ID": [], "Number of Nodes": [], "List of Nodes": []}

for cluster_id, nodes in clusters.items():
    cluster_data["Cluster ID"].append(cluster_id)
    cluster_data["Number of Nodes"].append(len(nodes))
    cluster_data["List of Nodes"].append(", ".join(map(str, nodes)))

# Convert dictionary to DataFrame
df_clusters = pd.DataFrame(cluster_data).sort_values("Number of Nodes")

# Display the DataFrame
display(df_clusters)


# COMMAND ----------


# Get consensus cluster IDs as node attributes
consensus_colors = [consensus_partition[node] for node in G_undirected.nodes()]  # Consensus Cluster IDs
uniform_node_size = 100  # Same size for all nodes

# Create position layout
pos = nx.spring_layout(G_undirected, seed=2)  # Use a fixed seed for reproducibility

# Draw the graph
plt.figure(figsize=(12, 8))

# Draw edges first
nx.draw_networkx_edges(G_undirected, pos, edge_color="gray")

# Draw nodes with uniform size and colors based on consensus clusters
nodes = nx.draw_networkx_nodes(
    G_undirected,
    pos,
    node_size=uniform_node_size,
    node_color=consensus_colors,
    cmap=plt.cm.Spectral,  # Colormap for distinct cluster colors
)

# Optionally, add labels
# nx.draw_networkx_labels(G_undirected, pos, font_size=8, font_color="black")

# Add color bar for clusters
cbar = plt.colorbar(nodes, label="Cluster ID")

# Add title and show the plot
plt.title("Graph Visualization with Consensus Clusters")

plt.savefig("svg_plots_latest/graph_visualization_consensus_clusters.svg", format="svg", bbox_inches='tight')

plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotting Consensus Clusters Individually

# COMMAND ----------

# Organize nodes by consensus cluster
consensus_clusters = {}
for node, cluster_id in consensus_partition.items():
    if cluster_id not in consensus_clusters:
        consensus_clusters[cluster_id] = []
    consensus_clusters[cluster_id].append(node)

# Draw each cluster separately
for cluster_id, nodes in consensus_clusters.items():
    subgraph = G_undirected.subgraph(nodes)  # Create a subgraph for the cluster
    pos = nx.spring_layout(subgraph, seed=2, k=2)  # Layout for the subgraph (with seed for consistency)
    
    plt.figure(figsize=(8, 6))
    
    # Draw nodes and edges
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=300,
        node_color=[cluster_id] * len(nodes),  # Assign the cluster ID as the color
        cmap=plt.cm.Spectral  # Use a colormap for distinct colors
    )

    # Draw edge labels (weights)
    edge_labels = {(u, v): d['weight'] for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=10)

    plt.title(f"Consensus Cluster {int(cluster_id)}")
    plt.savefig(f"svg_plots_latest/graph_consensus_clusters_louvain_{cluster_id}.svg", format="svg", bbox_inches='tight')

    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Infomap
# MAGIC works through random walk and information flow from one node to another. minimizes amount of information needed to describe a random walk over the network.

# COMMAND ----------

from infomap import Infomap
import networkx as nx

# Step 2: Map node labels to unique integers
original_nodes = list(G_directed.nodes())
node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
inverse_mapping = {idx: node for node, idx in node_mapping.items()}

# Step 3: Initialize Infomap
im = Infomap("--markov-time 0.07 --num-trials 10")

# Step 4: Add edges to Infomap using integer IDs
for source, target, data in G_directed.edges(data=True):
    source_id = node_mapping[source]  # Using integer node ID from mapping
    target_id = node_mapping[target]  # Using integer node ID from mapping
    weight = data.get('weight', 1.0)
    im.addLink(source_id, target_id, weight)

# Step 5: Run Infomap
im.run()

# Step 6: Extract partition results
partition = {}
for node_id, module_id in im.modules:  # Iterate over the dictionary of module assignments
    # Ensure that the mapping from node_id (integer ID) back to the original node is correct
    original_node = inverse_mapping[node_id]
    partition[original_node] = module_id

# Step 7: Add partition as a node attribute in NetworkX graph
nx.set_node_attributes(G_directed, partition, 'community')

# Step 9: Group nodes by communities
communities = {}
for node, community in partition.items():
    if community not in communities:
        communities[community] = []
    communities[community].append(node)

# Step 10: Display nodes in each community
# print("Nodes grouped by communities:")
# for community, nodes in communities.items():
#     print(f"Community {community}: {nodes}")


# Create a DataFrame for tabular representation
community_data = {"Cluster ID": [], "Number of Nodes": [], "List of Nodes": []}

for cluster_id, nodes in communities.items():
    community_data["Cluster ID"].append(cluster_id)
    community_data["Number of Nodes"].append(len(nodes))
    community_data["List of Nodes"].append(", ".join(map(str, nodes)))

# Convert dictionary to DataFrame
df_communities = pd.DataFrame(community_data)

# Display the DataFrame
display(df_communities)


# COMMAND ----------

# MAGIC %md
# MAGIC Hierarchical Clustering in Infomap
# MAGIC Infomap's algorithm works hierarchically:
# MAGIC
# MAGIC - First Level (Level 1): The graph is divided into broad clusters to minimize the codelength.
# MAGIC - Further Refinement (Level 2 and beyond): Each Level 1 community is further split into sub-communities, creating a hierarchy of clusters. This process can go deeper (e.g., Level 3, Level 4), but for most networks, Level 2 provides sufficient granularity.

# COMMAND ----------

# import matplotlib.pyplot as plt
# import networkx as nx

# Get community assignments from the Infomap partition
node_colors = [partition[node] for node in G_directed.nodes()]  # Community IDs as colors
uniform_node_size = 100  # All nodes will have the same size

# Create position layout
pos = nx.spring_layout(G_directed, seed=2)  # Use a fixed seed for reproducibility

# Draw the graph
plt.figure(figsize=(12, 8))

# Draw edges first
nx.draw_networkx_edges(G_directed, pos, edge_color="gray")

# Draw nodes with uniform size and colors based on communities
nodes = nx.draw_networkx_nodes(
    G_directed,
    pos,
    node_size=uniform_node_size,
    node_color=node_colors,
    cmap=plt.cm.Spectral,  # Use a colormap for distinct cluster colors
)

# Optionally, add labels
# nx.draw_networkx_labels(G_directed, pos, font_size=8, font_color="black")


# Add title and show the plot
plt.title("Graph Visualization with Infomap Communities")
plt.savefig("svg_plots_latest/graph_visualization_infomap_communities.svg", format="svg", bbox_inches='tight')
plt.show()


# COMMAND ----------

# Map node IDs back to original names and assign community IDs
flat_partition = {inverse_mapping[node_id]: module_id for node_id, module_id in im.modules}


# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx

# Organize nodes by community
communities = {}
for node, community_id in flat_partition.items():  # flat_partition contains node-to-community mapping
    if community_id not in communities:
        communities[community_id] = []
    communities[community_id].append(node)

# Draw each community separately
for community_id, nodes in communities.items():
    # Create a subgraph for the community
    subgraph = G_directed.subgraph(nodes)
    
    # Generate positions for the subgraph
    pos = nx.spring_layout(subgraph, seed=2, k=2)  # Use a fixed seed for consistency, Increase k for more spacing
    
    # Draw the subgraph
    plt.figure(figsize=(8, 6))
    nx.draw(
        subgraph,
        pos,
        with_labels=False,  # Disable default labels
        node_size=300,
        node_color=[community_id] * len(nodes),  # Assign the community ID as the color
        cmap=plt.cm.Spectral  # Use a colormap for distinct colors
    )
    
    # Add labels with smaller font size
    nx.draw_networkx_labels(
        subgraph,
        pos,
        font_size=8,  # Adjust this value to control label size
        font_color="black"  # Optional: Set label color
    )

    # Add edge labels (weights)
    edge_labels = {(u, v): d['weight'] for u, v, d in subgraph.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Community {int(community_id)}")
    plt.savefig(f"svg_plots_latest/graph_visualization_infomap_clusters_{community_id}.svg", format="svg", bbox_inches='tight')

    plt.show()
