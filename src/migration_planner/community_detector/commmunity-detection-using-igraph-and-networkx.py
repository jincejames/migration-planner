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

from IPython.display import Image

Image("../input/image-community/Community.jpeg")  # https://healthcommcapacity.org/

# COMMAND ----------

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Libraries needed:
# MAGIC
# MAGIC **We need following libraries: networkx**
# MAGIC

# COMMAND ----------

# Import libraries
import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs
import igraph as ig
import random
from datetime import datetime


# COMMAND ----------

current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading Datasets:

# COMMAND ----------

from pyspark.sql.functions import col, lit

df2 = (
    spark.read.format("csv")
    .option("header", "true")
    .load(
        "/Workspace/Users/aspram.grigoryan@databricks.com/(Clone) LH Stream Community Detection/source_transform_target_tables_per_stream.csv"
    )
)
depend_df = (
    spark.table('odp_adw_mvp_n.db_cascade.cascade_dependants')
    .withColumn('weight', lit(''))
    .withColumn('type', lit('casacade_dependent'))
)

df = df2.filter(col('table_type') != 'Trns').select(
    'stream_name', col('DB_Table_Name').alias('table_name'), 'table_type'
)
df.show()

# COMMAND ----------

# Self join to find dependencies
result = df.alias("df1").join(df.alias("df2"), col("df1.table_name") == col("df2.table_name"))

result.show()

# COMMAND ----------

# Filter only Src -> Tgt dependencies
filtered_result = result.filter((col("df1.table_type") == "Src") & (col("df2.table_type") == "Tgt"))

# Select required columns and rename them
output = filtered_result.select(
    col("df1.stream_name").alias("from"),
    col("df2.stream_name").alias("to"),
    col("df1.table_name").alias("table"),
    lit('dependent').alias(''),
)

output.show()

# COMMAND ----------

output = output.union(depend_df)
# Show the output
result = output.groupBy('from', 'to').count().select('from', 'to', col('count').alias('weight'))
result1 = result.filter(col('from') != col('to'))
edges = result1.toPandas()
edges = edges.groupby(['from', 'to'])['weight'].sum().reset_index()
G_directed = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True, create_using=nx.DiGraph())

# COMMAND ----------

nx.draw_networkx(G_directed)

# COMMAND ----------

# MAGIC %md
# MAGIC # Community Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Edge betweenness (Girvan–Newman)

# COMMAND ----------

# Modularity for different communities
lst_b = nx.community.girvan_newman(G_directed)
modularity = []
for x in lst_b:
    modularity.append(community.modularity(G_directed, x))

# COMMAND ----------

# Plot modularity
plt.plot(modularity, 'o')
plt.xlabel('# of clusters')
plt.ylabel('modularity')
plt.show()

# COMMAND ----------


# Function to generate a random color in hex format
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


plt.figure(figsize=(50, 50))

# Apply Girvan-Newman algorithm to find communities
lst_b = community.girvan_newman(G_directed)

# Extract the community with the maximum modularity
index = modularity.index(max(modularity))
temp = None
for i, v in enumerate(lst_b):
    if i == index:
        temp = v
        break

# Generate colors dynamically based on the number of communities
num_communities = len(list(temp))  # Get the number of splits in the community structure
colors = [generate_random_color() for _ in range(num_communities)]

# Generate positions for nodes using the spring layout
pos = nx.spring_layout(G_directed, seed=42)

# Assign colors to nodes based on their community
color_map_b = {node: "black" for node in G_directed.nodes()}
for counter, community_nodes in enumerate(temp):
    for node in community_nodes:
        color_map_b[node] = colors[counter]

# Draw edges of the graph
nx.draw_networkx_edges(G_directed, pos, alpha=0.5)

# Display edge weights on the graph
edge_labels = nx.get_edge_attributes(G_directed, "weight")
nx.draw_networkx_edge_labels(G_directed, pos, edge_labels=edge_labels)

# Draw nodes of the graph with assigned colors
nx.draw_networkx_nodes(G_directed, pos, node_color=list(color_map_b.values()), node_size=700)

# Add labels to nodes
nx.draw_networkx_labels(G_directed, pos, font_size=10)

# Remove axes for a cleaner visualization
plt.axis("off")

# Save the graph as an SVG file
output_path = f"/Volumes/odp_adb_ctdb_testing_catalog/testing/lh_metadata_files/SVGS/full_lineage/Girvan_Newman/Girvan_Newman_{current_timestamp}.svg"
plt.savefig(output_path, format="svg", bbox_inches="tight")

# Display the graph
plt.show()

# COMMAND ----------

import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs

# Apply Girvan-Newman algorithm to find communities
lst_b = community.girvan_newman(G_directed)

# Extract the community with the maximum modularity
index = modularity.index(max(modularity))
temp = None
for i, v in enumerate(lst_b):
    if i == index:
        temp = v
        break

# Extract community information
community_data = []
for idx, community_nodes in enumerate(temp):
    community_id = idx + 1
    num_streams = len(community_nodes)
    list_of_streams = ", ".join(community_nodes)
    community_data.append(
        {
            "Community ID": community_id,
            "Number of Streams in Community": num_streams,
            "List of Streams": list_of_streams,
        }
    )

# Convert to a pandas DataFrame for tabular representation
community_df = pd.DataFrame(community_data)

# Display the table
display(community_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Modularity maximization

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs

# Initialize data for the table
data = []

# Create an undirected graph for modularity maximization
G_directed = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True)

# Apply Greedy Modularity Maximization algorithm to find communities
lst_b = community.greedy_modularity_communities(G_directed)

# Process the communities identified by the algorithm
for idx, community_nodes in enumerate(lst_b):
    community_id = idx + 1  # Assign a unique ID to each community
    number_of_streams = len(community_nodes)
    streams_list = ", ".join(community_nodes)

    # Append data for this community
    data.append({"Community ID": community_id, "Number of Streams": number_of_streams, "List of Streams": streams_list})

# Create a DataFrame from the collected data
community_table = pd.DataFrame(data)
display(community_table)

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs


# Function to generate a random color in hex format
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"


# Generate random colors dynamically for communities
def generate_colors_for_communities(community_count):
    return [generate_random_color() for _ in range(community_count)]


plt.figure(figsize=(50, 50))


# Create an undirected graph for modularity maximization
G_directed = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True)

# Generate positions for nodes using the spring layout
pos = nx.spring_layout(G_directed, seed=42)

# Apply Greedy Modularity Maximization algorithm to find communities
lst_b = community.greedy_modularity_communities(G_directed)

# Dynamically assign colors to communities
community_count = len(lst_b)
colors = generate_colors_for_communities(community_count)

# Assign colors to nodes based on their community
color_map_b = {node: "black" for node in G_directed.nodes()}
for idx, community_nodes in enumerate(lst_b):
    for node in community_nodes:
        color_map_b[node] = colors[idx]

# Draw edges and display weights
nx.draw_networkx_edges(G_directed, pos, alpha=0.5)
edge_labels = nx.get_edge_attributes(G_directed, "weight")
nx.draw_networkx_edge_labels(G_directed, pos, edge_labels=edge_labels)

# Draw nodes with assigned community colors
nx.draw_networkx_nodes(G_directed, pos, node_color=list(color_map_b.values()), node_size=700)

# Add labels to nodes
nx.draw_networkx_labels(G_directed, pos, font_size=10)

# Remove axes for cleaner visualization
plt.axis("off")

# Graph(G_directed,
#       node_color=color_map_b, node_edge_width=0, edge_alpha=0.1,
#       node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
#       edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
# )

# Save the graph as an SVG file
output_path = f"/Volumes/odp_adb_ctdb_testing_catalog/testing/lh_metadata_files/SVGS/full_lineage/Modularity_maximization/Modularity_maximization_{current_timestamp}.svg"
plt.savefig(output_path, format="svg", bbox_inches="tight")

# Display the graph
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Label propagation

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs

# Create an undirected graph from the pandas edgelist
G_undirected = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True)

# Use Label Propagation algorithm to find communities
lst_m = community.label_propagation_communities(G_undirected)

# Prepare data for the tabular output
community_data = []
for idx, community_nodes in enumerate(lst_m):
    community_id = idx
    no_of_streams = len(community_nodes)
    streams_list = ", ".join(map(str, community_nodes))  # Convert nodes to a comma-separated string
    community_data.append({"Community ID": community_id, "No of Streams": no_of_streams, "Streams": streams_list})

# Convert the community data into a DataFrame
community_df = pd.DataFrame(community_data)

# Display the DataFrame
display(community_df)

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs


# Function to generate a random color in hex format
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"


# Create an undirected graph from the pandas edgelist
G_undirected = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True)

# Create a larger figure for better visualization
plt.figure(figsize=(50, 50))

# Generate positions for nodes using spring layout
pos = nx.spring_layout(G_undirected, k=0.5, seed=42)  # Adjust 'k' for node spacing

# Use Label Propagation algorithm to find communities
lst_m = community.label_propagation_communities(G_undirected)

# Generate random colors for communities dynamically
community_count = len(lst_m)
colors = [generate_random_color() for _ in range(community_count)]

# Initialize a color map with black as the default color
color_map_b = {node: "black" for node in G_undirected.nodes()}

# Assign colors to nodes based on their community
for counter, community_nodes in enumerate(lst_m):
    for node in community_nodes:
        color_map_b[node] = colors[counter]

# Draw edges with transparency for clarity
nx.draw_networkx_edges(G_undirected, pos, alpha=0.5, edge_color="gray")

# Draw nodes with assigned colors
nx.draw_networkx_nodes(G_undirected, pos, node_color=list(color_map_b.values()), node_size=800)  # Adjust size as needed

# Add labels to the nodes
nx.draw_networkx_labels(G_undirected, pos, font_size=10, font_color="black")

# Extract edge weights to display them as labels
edge_labels = nx.get_edge_attributes(
    G_undirected, 'weight'
)  # Replace 'weight' with the actual attribute name if different
nx.draw_networkx_edge_labels(G_undirected, pos, edge_labels=edge_labels, font_size=8)

# Remove axes for a cleaner visualization
plt.axis("off")

# Save the graph as an SVG file
output_path = f"/Volumes/odp_adb_ctdb_testing_catalog/testing/lh_metadata_files/SVGS/full_lineage/label_propagation_communities/label_propagation_communities_{current_timestamp}.svg"
plt.savefig(output_path, format="svg", bbox_inches="tight")

# Display the graph
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fast community unfolding (Louvian)

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import matplotlib.pyplot as plt  # For plotting graphs

G_undirected = nx.from_pandas_edgelist(edges, source="from", target="to", edge_attr=True)

# Use Louvain community detection algorithm to find communities
lst_m = community_louvain.best_partition(G_undirected)

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

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs


# Function to generate a random color in hex format
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"


# Create a figure with a larger size for better visualization
plt.figure(figsize=(50, 50))

# Create an undirected graph from the pandas edgelist
G_undirected = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True)

# Generate positions for nodes using the spring layout
pos = nx.spring_layout(G_undirected)

# Use Louvain community detection algorithm to find communities
lst_m = community_louvain.best_partition(G_undirected)

# Generate random colors for communities dynamically
community_count = max(lst_m.values()) + 1
colors = [generate_random_color() for _ in range(community_count)]

# Initialize a color map with black as the default color
color_map_b = {node: "black" for node in G_undirected.nodes()}

# Assign colors to nodes based on their community
for node, community in lst_m.items():
    color_map_b[node] = colors[community]

# Draw edges of the graph
nx.draw_networkx_edges(G_undirected, pos, alpha=0.6, edge_color="gray")

# Draw nodes of the graph with assigned colors
nx.draw_networkx_nodes(G_undirected, pos, node_color=list(color_map_b.values()), node_size=500)

# Add labels to the nodes
nx.draw_networkx_labels(G_undirected, pos, font_size=10, font_color="black")

# Extract edge weights to display them as labels
edge_labels = nx.get_edge_attributes(
    G_undirected, 'weight'
)  # Replace 'weight' with the actual edge attribute name if different
nx.draw_networkx_edge_labels(G_undirected, pos, edge_labels=edge_labels, font_size=8)

# Remove axes for a cleaner visualization
plt.axis("off")

# Optional: Save the graph as an SVG file (uncomment if needed)
output_path = f"/Volumes/odp_adb_ctdb_testing_catalog/testing/lh_metadata_files/SVGS/full_lineage/community_louvain/community_louvain_{current_timestamp}.svg"
plt.savefig(output_path, format="svg", bbox_inches="tight")

# Display the graph
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Walktrap

# COMMAND ----------

import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs


G = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True, create_using=nx.DiGraph())
g = ig.Graph.TupleList(G.edges(), directed=False)
wtrap = g.community_walktrap(steps=20)
clust = wtrap.as_clustering()
max(set(clust.membership)) + 1


def find_community(step):
    wtrap = g.community_walktrap(steps=step)
    clust = wtrap.as_clustering()
    return max(set(clust.membership)) + 1


for step in range(1, 20):
    print("############# Step Test : %d ################" % step)
    rst_com = find_community(step)
    print("Count of Community being found：%d" % rst_com)

# COMMAND ----------

# MAGIC %md
# MAGIC can change step values for different outputs

# COMMAND ----------

step = 10  # Number of steps

# COMMAND ----------

# DBTITLE 1,Tabular output for Walktrap
import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs
import igraph as ig

# Extract clusters from the walktrap result
wtrap = g.community_walktrap(steps=step)
clust = wtrap.as_clustering()

clusters = clust.membership

# Create a dictionary to store clusters by community
cluster_dict = {}

# Map the indices of nodes in 'G' to the actual node labels
node_labels = list(G.nodes)

for idx, community_id in enumerate(clusters):
    node_name = node_labels[idx]  # Get the actual node label (stream name or ID)
    if community_id not in cluster_dict:
        cluster_dict[community_id] = []
    cluster_dict[community_id].append(node_name)

# Convert cluster information into a pandas DataFrame
cluster_df = pd.DataFrame(
    [
        (community_id, ', '.join(map(str, nodes)), len(nodes))  # Add the length of nodes for stream count
        for community_id, nodes in cluster_dict.items()
    ],
    columns=["Community_ID", "Nodes", "Stream_Count"],  # Include 'Stream_Count' column
)
# Convert to JSON if needed
cluster_json = cluster_dict  # You can save this to a JSON file if required
# Display the DataFrame
cluster_df.display()

# COMMAND ----------

# DBTITLE 1,walktrap algo png write
import igraph as ig
import networkx as nx

# Convert the DataFrame to a NetworkX graph
edges = result1.toPandas()
edges = edges.groupby(['from', 'to'])['weight'].sum().reset_index()
G = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True, create_using=nx.DiGraph())

# Convert NetworkX graph to iGraph
g = ig.Graph.TupleList(G.edges(data=True), directed=False)

# Add weights to iGraph edges
weights = [edge['weight'] for edge in G.edges.values()]
g.es['weight'] = weights

# Perform the Walktrap community detection
wtrap = g.community_walktrap(steps=1)
clust = wtrap.as_clustering()

# Add vertex labels (node names) for visualization
g.vs['label'] = list(G.nodes)

# Add edge labels (weights) for visualization
edge_labels = [str(int(w)) for w in g.es['weight']]
g.es['label'] = edge_labels

# Plot the graph with weights and communities
plot = ig.plot(
    clust,
    mark_groups=True,
    bbox=(5000, 5000),
    vertex_label=g.vs['label'],  # Node labels
    edge_label=g.es['label'],  # Edge weights as labels
    vertex_size=50,
    edge_width=[max(1, w) for w in g.es['weight']],
)

# Save the plot to a file
plot.save('/Volumes/odp_adb_ctdb_testing_catalog/testing/lh_metadata_files/SVGS/full_lineage/Walktrap/Walktrap.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. k-clique community

# COMMAND ----------

# DBTITLE 1,k_clique_communities
import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs
import igraph as ig


def find_community(graph, k):
    return list(community.k_clique_communities(graph, k))


for k in range(2, 10):
    print("############# k-Clique: %d ################" % k)
    rst_com = find_community(G_undirected, k)
    print("Count of Community being found：%d" % len(rst_com))

# COMMAND ----------

# MAGIC %md
# MAGIC can change K values for different outputs

# COMMAND ----------

# Find k-clique communities
k = 14  # Number of cliques

# COMMAND ----------

# DBTITLE 1,Graph output for k_clique_communities
import pandas as pd  # For reading dataset files
import networkx as nx  # For network creation/analysis
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt  # For plotting graphs
import igraph as ig

# Create a weighted graph from the edges DataFrame
G_undirected = nx.from_pandas_edgelist(edges, source='from', target='to', edge_attr=True)

# Use spring layout for better spacing
pos = nx.spring_layout(G_undirected, k=0.8)  # Adjust 'k' for spacing

lst_m = list(community.k_clique_communities(G_undirected, k))

# Generate random colors for each community
colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(lst_m))]

# Initialize color map for nodes
color_map_b = {node: "black" for node in G_undirected.nodes()}

# Assign random colors to each community
for counter, community_nodes in enumerate(lst_m):
    for node in community_nodes:
        color_map_b[node] = colors[counter % len(colors)]

# Scale edge weights for visualization
edge_weights = [G_undirected[u][v].get('weight', 1) for u, v in G_undirected.edges()]
max_weight = max(edge_weights)
min_weight = min(edge_weights)
scaled_weights = [(5 * (w - min_weight) / (max_weight - min_weight) + 0.5) for w in edge_weights]

# Draw the graph
plt.figure(figsize=(25, 20))
nx.draw_networkx_edges(G_undirected, pos, width=scaled_weights, alpha=0.7, edge_color="gray")
nx.draw_networkx_nodes(G_undirected, pos, node_color=list(color_map_b.values()), node_size=600, edgecolors="black")
nx.draw_networkx_labels(G_undirected, pos, font_size=10, font_color="black")

edge_labels = nx.get_edge_attributes(G_undirected, "weight")
nx.draw_networkx_edge_labels(G_undirected, pos, edge_labels=edge_labels, font_size=10, font_color="red")
# Add legend for community colors

for idx, color in enumerate(colors):
    plt.plot([], [], marker="o", color=color, label=f"Community {idx+1}", linestyle="None")

plt.legend(loc="best", fontsize="small", frameon=False)
plt.title(f"Weighted Graph with {k}-Clique Communities")
plt.axis("off")

# Save as SVG file
output_path = f"/Volumes/odp_adb_ctdb_testing_catalog/testing/lh_metadata_files/SVGS/full_lineage/k_clique_community/k_clique_community_{current_timestamp}.svg"  # Databricks file path
plt.savefig(output_path, format="svg", bbox_inches="tight")
print(f"SVG saved to: {output_path}")

# Show the graph
plt.show()

# COMMAND ----------

# DBTITLE 1,Tabular output for k_clique_communities
# Initialize a dictionary to store community details
community_details = []
# k = 14
# 13, 14 some what even distribution but communities are less 5,6
lst_m = community.k_clique_communities(G_undirected, k)

# Iterate over the communities
for idx, comnty in enumerate(lst_m):
    community_name = f"Community {idx + 1}"  # Name the community
    stream_count = len(comnty)  # Count nodes in the community
    streams = ", ".join(comnty)  # Join node names (streams) as a single string
    community_details.append(
        {"Community": community_name, "Stream Count": stream_count, "Streams": streams}
    )  # Add all details to the list

# Convert the list of dictionaries to a DataFrame
community_df = pd.DataFrame(community_details)

# Display the DataFrame
spark.createDataFrame(community_df).display()
