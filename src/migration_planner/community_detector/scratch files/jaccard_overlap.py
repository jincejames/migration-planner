# Databricks notebook source
!pip install python-igraph adjustText infomap netgraph

# COMMAND ----------

!pip install networkx

# COMMAND ----------

dbutils.library.restartPython()

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

df2 = spark.read.format("csv").option("header", "true").load("/Volumes/users/aspram_grigoryan/community_detection/Stream-Table-dependencies-All-ADW-Streams-Tabelle1 (2).csv")
df2.display()

# COMMAND ----------

from pyspark.sql import functions as F

# Start from df2
df = df2.select(
    "STREAM_NAME",
    "STREAM_TYPE",
    "script_name",
    F.col("DB_Table_Name").alias("table_name"),
    "table_type"
)


# COMMAND ----------

# Sources: tables read by the script
src = (
    df.filter(F.col("table_type") == "Src")
      .select(
          "STREAM_NAME",
          "STREAM_TYPE",
          "script_name",
          F.col("table_name").alias("src_table")
      )
)

# Targets: tables written by the script
# IMPORTANT: include both 'Trns' and 'Tgt'
tgt = (
    df.filter(F.col("table_type").isin("Trns", "Tgt"))
      .select(
          "STREAM_NAME",
          "STREAM_TYPE",
          "script_name",
          F.col("table_name").alias("tgt_table")
      )
)


# COMMAND ----------

table_edges_sdf = (
    src.join(
        tgt,
        on=["STREAM_NAME", "STREAM_TYPE", "script_name"],
        how="inner"
    )
    .select(
        "STREAM_NAME",
        "STREAM_TYPE",
        "script_name",
        "src_table",
        "tgt_table"
    )
)

table_edges_sdf.show(20, truncate=False)


# COMMAND ----------

table_edges_weighted_sdf = (
    table_edges_sdf
    .groupBy("src_table", "tgt_table")
    .count()
    .withColumnRenamed("count", "weight")
)

table_edges_weighted_sdf.show(20, truncate=False)

# If you want it in pandas for NetworkX
edges_pdf = table_edges_weighted_sdf.toPandas()


# COMMAND ----------

G_directed = nx.from_pandas_edgelist(
    edges_pdf,          # from your script-based edges
    source="src_table",
    target="tgt_table",
    edge_attr=True,
    create_using=nx.DiGraph()
)


# COMMAND ----------

# DBTITLE 1,finding leaf nodes / golden layer reports that do not produce anything downstream
# "Reports" are nodes with out-degree == 0 (nothing depends on them)

report_nodes = [
    n for n in G_directed.nodes()
    if G_directed.out_degree(n) == 0 and G_directed.in_degree(n) > 0
]
# Conversely, "root/source tables" are those with in-degree == 0
source_nodes = [n for n, deg in G_directed.in_degree() if deg == 0]

print(f"Reports (no outgoing edges): {len(report_nodes)}")
print(f"Root/source tables (no incoming edges): {len(source_nodes)}")

print("Example report tables:")
print(report_nodes[:10])


# COMMAND ----------

# DBTITLE 1,finding the lineage of each table
import networkx as nx

def lineage_closure(G, leaf):
    """All tables that can reach `leaf` (ancestors) plus `leaf` itself."""
    return nx.ancestors(G, leaf) | {leaf}

# lineage for each report / golden table
report_to_lineage = {
    r: lineage_closure(G_directed, r)
    for r in report_nodes
}

# quick sanity check
example_report = report_nodes[0]
print("Example report:", example_report)
print("Number of tables in its lineage:", len(report_to_lineage[example_report]))
print("Sample tables in its lineage:", list(report_to_lineage[example_report])[:10])


# COMMAND ----------

# DBTITLE 1,completely independent reports - not sharing any lineage
from collections import Counter

# Count how many report lineages each table appears in
table_usage = Counter()
for lineage in report_to_lineage.values():
    for t in lineage:
        table_usage[t] += 1

# A report is "completely independent" if every table in its lineage
# is only ever used in that one report's lineage
independent_reports = [
    r for r, lineage in report_to_lineage.items()
    if all(table_usage[t] == 1 for t in lineage)
]

print(f"Number of completely independent reports: {len(independent_reports)}")
print("Sample independent reports:", independent_reports[:10])


# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_report_lineage(G, report, lineage_nodes, figsize=(10, 8), save_path=None):
    """
    G: full lineage DiGraph
    report: report table name (leaf)
    lineage_nodes: set of tables in its lineage (from lineage_closure)
    """
    # Subgraph of just this report's lineage
    H = G.subgraph(lineage_nodes).copy()

    # Identify roles within this lineage
    roots = [n for n in H.nodes() if H.in_degree(n) == 0]
    leaf = report
    intermediates = [n for n in H.nodes() if n not in roots and n != leaf]

    # Assign colors
    colors = []
    for n in H.nodes():
        if n == leaf:
            colors.append("tab:red")      # report
        elif n in roots:
            colors.append("tab:green")    # source tables
        else:
            colors.append("tab:blue")     # intermediate tables

    # Layout: spring layout is OK for ~10–100 nodes
    k = 1 / np.sqrt(len(H.nodes()) + 1)
    pos = nx.spring_layout(H, seed=42, k=k)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=300, alpha=0.9)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="->", arrowsize=10, width=0.8, alpha=0.6)

    # Optional labels (comment out if too busy)
    nx.draw_networkx_labels(H, pos, font_size=7)

    plt.title(f"Lineage for report: {report}")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# COMMAND ----------

# Plot all independent reports (or just the first N)
N = 100  # change if you want more/less

for r in independent_reports[:N]:
    print(f"Plotting lineage for independent report: {r}")
    lineage_nodes = report_to_lineage[r]
    plot_report_lineage(G_directed, r, lineage_nodes, save_path=None)
    # or save to file:
    # plot_report_lineage(G_directed, r, lineage_nodes,
    #                     save_path=f"plots/lineage_{r.replace('.', '_')}.png")


# COMMAND ----------

# MAGIC %md
# MAGIC to abstract away this similarity calculation to stream

# COMMAND ----------

import itertools
import networkx as nx

def jaccard(a, b):
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union

report_list = list(report_to_lineage.keys())
R = len(report_list)

j_threshold = 0.05  # similarity threshold; tune this

report_graph = nx.Graph()
report_graph.add_nodes_from(report_list)

for i in range(R):
    r1 = report_list[i]
    s1 = report_to_lineage[r1]
    for j in range(i + 1, R):
        r2 = report_list[j]
        s2 = report_to_lineage[r2]
        j = jaccard(s1, s2)
        if j >= j_threshold:
            report_graph.add_edge(r1, r2, weight=j)

print(
    f"Report similarity graph: {report_graph.number_of_nodes()} reports, "
    f"{report_graph.number_of_edges()} edges (Jaccard ≥ {j_threshold})"
)


# COMMAND ----------

# Print top pairs with biggest overlap
overlaps = []
for u, v, data in report_graph.edges(data=True):
    overlaps.append((data["weight"], u, v))

overlaps.sort(reverse=True)
print("Top 10 most overlapping report pairs:")
for w, u, v in overlaps[:10]:
    print(f"{u} -- {v}  (Jaccard={w:.3f})")
