# Databricks notebook source
# MAGIC %md
# MAGIC ## Libraries and Parameters
# MAGIC

# COMMAND ----------

# DBTITLE 1,Library imports
import pandas as pd

from datetime import datetime
import os
import numpy as np

from pyspark.sql.functions import col, lit, when, upper, lower
from pyspark.sql.functions import ceil, sum, when, array, sort_array

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text(
    "volume_name", "/Volumes/odp_adw_mvp_n/migration/utilities/community_detection/", "Input Volume Path"
)
dbutils.widgets.text("input_dependency_name", "ETL-table-dependencies_20251223_1218.csv", "Input CSV file name")
dbutils.widgets.text("outofscope_stream_file_name", "out-of-scopte-streams.csv", "Out of scope streams file name")
dbutils.widgets.text("report_dependency_file_name", "stream_to_report_mapping_new.csv", "report Dependency file name")

dbutils.widgets.text("table_size", "table-space-in-gb_20251201_1352.csv", "Table size file name")

# COMMAND ----------

# DBTITLE 1,Common variables & output directory operation
volume_path = dbutils.widgets.get("volume_name")
dependency_input_path = volume_path + dbutils.widgets.get("input_dependency_name")
outofscope_stream_path = volume_path + dbutils.widgets.get("outofscope_stream_file_name")
report_dependency = volume_path + dbutils.widgets.get("report_dependency_file_name")
table_size = volume_path + dbutils.widgets.get("table_size")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Datasets:

# COMMAND ----------

# DBTITLE 1,Analysis: Old Reading input stream - table dependency file
old_dependency_df_full = spark.read.format("csv").option("header", "true").load(dependency_input_path)

# COMMAND ----------

# DBTITLE 1,Analysis : Get stream for particular tables
filtered_df = old_dependency_df_full.filter(
    col("DB_Table_Name").isin("DB_STREAM_DATA.CTRL_RSE_IMPORTS", "DB_STREAM_DATA.CTRL_RSE_AIRL_SUB_CD")
)
display(filtered_df)

# COMMAND ----------

# DBTITLE 1,Analysis: Current stream - table dependency file
current_dependency_df_full = (
    spark.read.format("csv")
    .option("header", "true")
    .option("delimiter", ";")
    .load("/Volumes/odp_adw_mvp_n/migration/utilities/community_detection/ETL-table-dependencies_11022026.csv")
)

# COMMAND ----------

# DBTITLE 1,Analysis: See previous sources and targets of IMP_JV_RSE stream
old_imp_jv_rse_filtered = (
    old_dependency_df_full.filter(
        (col("STREAM_NAME") == "IMP_JV_RSE") & (upper(col("table_type")).isin(["TGT", "SRC"]))
    )
    .select("SquadID", "STREAM_NAME", "DB_Table_Name", "table_type")
    .distinct()
)
display(old_imp_jv_rse_filtered)

# COMMAND ----------

# DBTITLE 1,Reading table size in GB records
table_size_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(table_size)
    .select(upper(col("DB_Table_Name")).alias("table_name"), col("SPACE_IN_GB").alias("size"))
)

# COMMAND ----------

# DBTITLE 1,Analysis: IMP_JV_RSE table sizes
old_imp_jv_rse_with_size = old_imp_jv_rse_filtered.join(
    table_size_df, upper(old_imp_jv_rse_filtered.DB_Table_Name) == table_size_df.table_name, "left"
).select(old_imp_jv_rse_filtered["*"], table_size_df["size"])
display(old_imp_jv_rse_with_size)

# COMMAND ----------

# DBTITLE 1,Current inputs and outputs of imp_jv_rse with table size
current_imp_jv_rse_filtered = (
    current_dependency_df_full.filter(
        (col("STREAM_NAME") == "IMP_JV_RSE") & (upper(col("table_type")).isin(["TGT", "SRC"]))
    )
    .select("SquadID", "STREAM_NAME", "DB_Table_Name", "table_type")
    .distinct()
)

current_imp_jv_rse_with_size = current_imp_jv_rse_filtered.join(
    table_size_df, upper(current_imp_jv_rse_filtered.DB_Table_Name) == table_size_df.table_name, "left"
).select(current_imp_jv_rse_filtered["*"], table_size_df["size"])

display(current_imp_jv_rse_with_size)

# COMMAND ----------

# DBTITLE 1,Find upstream dependencies (streams that IMP_JV_RSE depends on)
# Get source tables from already filtered results
old_imp_jv_rse_sources = (
    old_imp_jv_rse_filtered.filter(upper(col("table_type")) == "SRC").select("DB_Table_Name").distinct()
)
current_imp_jv_rse_sources = (
    current_imp_jv_rse_filtered.filter(upper(col("table_type")) == "SRC").select("DB_Table_Name").distinct()
)

# Find streams that produce these tables as targets (upstream dependencies) - OLD data
old_upstream_deps = (
    old_dependency_df_full.alias("dep")
    .join(
        old_imp_jv_rse_sources.alias("src"), upper(col("dep.DB_Table_Name")) == upper(col("src.DB_Table_Name")), "inner"
    )
    .filter((upper(col("dep.table_type")) == "TGT") & (col("dep.STREAM_NAME") != "IMP_JV_RSE"))
    .select(
        col("dep.SquadID"),
        col("dep.STREAM_NAME"),
        col("dep.DB_Table_Name"),
        lit("upstream_of_IMP_JV_RSE").alias("dependency_type"),
    )
    .distinct()
)

# Find streams that produce these tables as targets (upstream dependencies) - CURRENT data
current_upstream_deps = (
    current_dependency_df_full.alias("dep")
    .join(
        current_imp_jv_rse_sources.alias("src"),
        upper(col("dep.DB_Table_Name")) == upper(col("src.DB_Table_Name")),
        "inner",
    )
    .filter((upper(col("dep.table_type")) == "TGT") & (col("dep.STREAM_NAME") != "IMP_JV_RSE"))
    .select(
        col("dep.SquadID"),
        col("dep.STREAM_NAME"),
        col("dep.DB_Table_Name"),
        lit("upstream_of_IMP_JV_RSE").alias("dependency_type"),
    )
    .distinct()
)

print(f"Old upstream dependencies: {old_upstream_deps.count()}")
print(f"Current upstream dependencies: {current_upstream_deps.count()}")

# COMMAND ----------

# DBTITLE 1,Find downstream dependencies (streams that depend on IMP_JV_RSE)
# Get target tables from already filtered results
old_imp_jv_rse_targets = (
    old_imp_jv_rse_filtered.filter(upper(col("table_type")) == "TGT").select("DB_Table_Name").distinct()
)
current_imp_jv_rse_targets = (
    current_imp_jv_rse_filtered.filter(upper(col("table_type")) == "TGT").select("DB_Table_Name").distinct()
)

# Find streams that use these tables as sources (downstream dependencies) - OLD data
old_downstream_deps = (
    old_dependency_df_full.alias("dep")
    .join(
        old_imp_jv_rse_targets.alias("tgt"), upper(col("dep.DB_Table_Name")) == upper(col("tgt.DB_Table_Name")), "inner"
    )
    .filter((upper(col("dep.table_type")) == "SRC") & (col("dep.STREAM_NAME") != "IMP_JV_RSE"))
    .select(
        col("dep.SquadID"),
        col("dep.STREAM_NAME"),
        col("dep.DB_Table_Name"),
        lit("downstream_of_IMP_JV_RSE").alias("dependency_type"),
    )
    .distinct()
)

# Find streams that use these tables as sources (downstream dependencies) - CURRENT data
current_downstream_deps = (
    current_dependency_df_full.alias("dep")
    .join(
        current_imp_jv_rse_targets.alias("tgt"),
        upper(col("dep.DB_Table_Name")) == upper(col("tgt.DB_Table_Name")),
        "inner",
    )
    .filter((upper(col("dep.table_type")) == "SRC") & (col("dep.STREAM_NAME") != "IMP_JV_RSE"))
    .select(
        col("dep.SquadID"),
        col("dep.STREAM_NAME"),
        col("dep.DB_Table_Name"),
        lit("downstream_of_IMP_JV_RSE").alias("dependency_type"),
    )
    .distinct()
)

print(f"Old downstream dependencies: {old_downstream_deps.count()}")
print(f"Current downstream dependencies: {current_downstream_deps.count()}")

# COMMAND ----------

# DBTITLE 1,Combine all dependencies and display results
# Combine old and current dependencies separately
old_all_deps = old_upstream_deps.union(old_downstream_deps)
current_all_deps = current_upstream_deps.union(current_downstream_deps)

# Display OLD dependencies
print("\n=== OLD Input - IMP_JV_RSE Dependencies ===")
print(
    f"Total unique dependencies: {old_all_deps.select('STREAM_NAME', 'DB_Table_Name', 'dependency_type').distinct().count()}"
)
print(f"\nBreakdown by dependency type:")
old_all_deps.groupBy("dependency_type").count().orderBy("dependency_type").show()

print("\n=== OLD Dependencies (SquadID, Stream Name, Table Name, Dependency Type) ===")
old_all_deps_sorted = old_all_deps.orderBy("dependency_type", "STREAM_NAME", "DB_Table_Name")
display(old_all_deps_sorted)

# Display CURRENT dependencies
print("\n=== CURRENT Input - IMP_JV_RSE Dependencies ===")
print(
    f"Total unique dependencies: {current_all_deps.select('STREAM_NAME', 'DB_Table_Name', 'dependency_type').distinct().count()}"
)
print(f"\nBreakdown by dependency type:")
current_all_deps.groupBy("dependency_type").count().orderBy("dependency_type").show()

print("\n=== CURRENT Dependencies (SquadID, Stream Name, Table Name, Dependency Type) ===")
current_all_deps_sorted = current_all_deps.orderBy("dependency_type", "STREAM_NAME", "DB_Table_Name")
display(current_all_deps_sorted)

# COMMAND ----------

# DBTITLE 1,Reading out of scope stream names
outofscope_stream_names_df = (
    spark.read.format("csv").option("header", "true").load(outofscope_stream_path).select(col("stream_name"))
)
outofscope_stream_names_rows_list = outofscope_stream_names_df.collect()
outofscope_stream_names_list = [x['stream_name'] for x in outofscope_stream_names_rows_list]

# COMMAND ----------

# DBTITLE 1,Read complexity by stream
# Read complexity by stream with semicolon delimiter
complexity_by_stream_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("delimiter", ";")
    .load(f"{volume_path}Complexity_by_Stream.csv")
)

# Define complexity score weights
COMPLEXITY_WEIGHTS = {'low': 1, 'medium': 2, 'complex': 4, 'very_complex': 7}

# Calculate complexity score for each stream
# Score = (low * 1) + (medium * 2) + (complex * 4) + (very_complex * 7)
# Using double to handle decimal values, then casting to int
complexity_scores_df = complexity_by_stream_df.withColumn(
    "complexity_score",
    (col("low").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['low']))
    + (col("medium").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['medium']))
    + (col("complex").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['complex']))
    + (col("very_complex").cast("double").cast("int") * lit(COMPLEXITY_WEIGHTS['very_complex'])),
).select(
    col("stream_name"),
    col("low").cast("double").cast("int").alias("low"),
    col("medium").cast("double").cast("int").alias("medium"),
    col("complex").cast("double").cast("int").alias("complex"),
    col("very_complex").cast("double").cast("int").alias("very_complex"),
    col("complexity_score"),
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

# <TODO> Filtering two reports wrt corona and gdpr, since they skew the community creation due to high number of dependencies
report_dependency_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(report_dependency)
    .select(
        lit("Business").alias("SquadID"),
        col("report_name").alias("STREAM_NAME"),
        lit("REPORT").alias("STREAM_TYPE"),
        upper(col("table_name")).alias("DB_Table_Name"),
        lit("SRC").alias("table_type"),
        lit("REPORT").alias("script_name"),
    )
)

# COMMAND ----------

# DBTITLE 1,Take stream, table info from table-dependency file
current_dependency_df = current_dependency_df_full.select(
    "SquadID", "STREAM_NAME", "STREAM_TYPE", "DB_Table_Name", "table_type", "script_name"
).distinct()

# COMMAND ----------

# DBTITLE 1,union current table dependency with reports
report_current_dependency_df_full = current_dependency_df.union(report_dependency_df).filter(
    ~upper(col("stream_name")).contains("ARCHIVE")
    & ~upper(col("stream_name")).contains("GDPR")
    & ~upper(col("stream_name")).contains("HOUSEKEEPING")
)

display(report_current_dependency_df_full)

# COMMAND ----------

display(report_current_dependency_df_full.filter(col("STREAM_TYPE") == "REPORT"))

# COMMAND ----------

# DBTITLE 1,Find all unused tables
# Find all tables where upper(table_type) is "TGT" and there is no corresponding row with "SRC" or "SRC_TRNS"
from pyspark.sql.functions import upper, count

tgt_tables_df = (
    report_current_dependency_df_full.filter(upper(col("table_type")) == "TGT")
    .select("SquadID", "STREAM_NAME", "STREAM_TYPE", "script_name", "DB_Table_Name")
    .distinct()
)

src_tables_df = (
    report_current_dependency_df_full.filter(upper(col("table_type")).isin("SRC", "SRC_TRNS"))
    .select("DB_Table_Name")
    .distinct()
)

tgt_only_tables_df = (
    tgt_tables_df.alias("tgt")
    .join(src_tables_df.alias("src"), upper(col("tgt.DB_Table_Name")) == upper(col("src.DB_Table_Name")), "left_anti")
    .select(col("tgt.*"))
    .distinct()
)

# Exclude tables that appear as TGT multiple times (either in multiple streams or more than once within the same stream)
multi_instance_tgt_tables = tgt_only_tables_df.groupBy("DB_Table_Name").count().filter(col("count") > 1)

tgt_only_tables_df = (
    tgt_only_tables_df.alias("tgt")
    .join(
        multi_instance_tgt_tables.alias("multi"),
        upper(col("tgt.DB_Table_Name")) == upper(col("multi.DB_Table_Name")),
        "left_anti",
    )
    .select(col("tgt.*"))
)

tgt_only_tables_df_sorted = tgt_only_tables_df.orderBy("SquadID", "STREAM_TYPE", "STREAM_NAME")

display(tgt_only_tables_df_sorted)

# COMMAND ----------

tgt_only_tables_df_sorted.select("STREAM_NAME").distinct().count()

# COMMAND ----------

# DBTITLE 1,Identify streams that only produce unused target tables
# Get all target tables by stream
all_tgt_by_stream = (
    report_current_dependency_df_full.filter(upper(col("table_type")) == "TGT")
    .select("SquadID", "STREAM_NAME", "STREAM_TYPE", "DB_Table_Name")
    .distinct()
)

# Get the set of unused tables
unused_tables = tgt_only_tables_df_sorted.select("DB_Table_Name").distinct()

# Mark each target table as used or unused
tgt_with_usage = (
    all_tgt_by_stream.alias("tgt")
    .join(unused_tables.alias("unused"), upper(col("tgt.DB_Table_Name")) == upper(col("unused.DB_Table_Name")), "left")
    .withColumn("is_unused", when(col("unused.DB_Table_Name").isNotNull(), lit(1)).otherwise(lit(0)))
    .select(col("tgt.*"), col("is_unused"))
)

# Count total and unused targets per stream
stream_target_stats = tgt_with_usage.groupBy("SquadID", "STREAM_NAME", "STREAM_TYPE").agg(
    count("*").alias("total_targets"), sum("is_unused").alias("unused_targets")
)

# Find streams where ALL targets are unused (total_targets == unused_targets)
streams_only_unused = stream_target_stats.filter(col("total_targets") == col("unused_targets")).orderBy(
    "SquadID", "STREAM_TYPE", "STREAM_NAME"
)

print(f"Total streams that only produce unused target tables: {streams_only_unused.count()}")
print(f"\nBreakdown by Squad and Stream Type:")

display(streams_only_unused)

# COMMAND ----------

# DBTITLE 1,Filter Unused Streams Excluding Out of Scope Names
in_scope_streams_only_unused = streams_only_unused.filter(
    ~col("STREAM_NAME").isin(outofscope_stream_names_list)
).orderBy("SquadID", "STREAM_TYPE")

display(in_scope_streams_only_unused)

# COMMAND ----------

# DBTITLE 1,SquadID and STREAM_TYPE Count with Percentage ...
from pyspark.sql.functions import count, round as spark_round

# Count per SquadID and STREAM_TYPE
stats_df = tgt_only_tables_df_sorted.groupBy("SquadID", "STREAM_TYPE").agg(count("*").alias("count"))

# Count per STREAM_TYPE only
stats_by_streamtype_df = tgt_only_tables_df_sorted.groupBy("STREAM_TYPE").agg(count("*").alias("count"))

# Total number of these tables
total_tables = tgt_only_tables_df_sorted.count()

# Add total and percentage columns for SquadID + STREAM_TYPE
stats_with_pct = (
    stats_df.withColumn("total", lit(total_tables))
    .withColumn("percentage", spark_round((col("count") / col("total")) * 100, 2))
    .select("SquadID", "STREAM_TYPE", "count", "total", "percentage")
    .orderBy("SquadID", "STREAM_TYPE")
)

# Add total and percentage columns for STREAM_TYPE only
stats_by_streamtype_with_pct = (
    stats_by_streamtype_df.withColumn("total", lit(total_tables))
    .withColumn("percentage", spark_round((col("count") / col("total")) * 100, 2))
    .select("STREAM_TYPE", "count", "total", "percentage")
    .orderBy("STREAM_TYPE")
)

display(stats_with_pct)
display(stats_by_streamtype_with_pct)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from IDENTIFIER('catalog_name' || :n || '.schema.tablename');

# COMMAND ----------
