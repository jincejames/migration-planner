# Databricks notebook source
# DBTITLE 1,Library imports
import pandas as pd
import random
from datetime import datetime
import os
import numpy as np
from pyspark.sql.functions import col, lit, when, upper, lower
from pyspark.sql.functions import ceil, sum, when, array, sort_array,explode,udf
from pyspark.sql.types import ArrayType, StringType

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text(
    "volume_name",
    "/Volumes/odp_adw_utilities_n/planning/utilities/community_detection/",
    "Input Volume Path"
)
dbutils.widgets.text(
    "input_dependency_name",
    "ODAT__202603301324.csv",
    "Input CSV file name"
)
dbutils.widgets.text(
    "outofscope_stream_file_name",
    "out-of-scopte-streams.csv",
    "Out of scope streams file name"
)
dbutils.widgets.text(
    "report_table_dependency_file_name",
    "Unique_Reports_With_Queries_and_Tables.csv",
    "report to table dependency file name"
)
dbutils.widgets.text(
    "view_table_association",
    "View-Table-Association_20260401.csv",
    "View Table dependencies"
)
dbutils.widgets.text(
    "table_size",
    "table-space-in-gb_20251201_1352.csv",
    "Table size file name"
)
dbutils.widgets.text(
    "community_mapping",
    "ETL_Scripts_20260106_Master_Communities.csv",
    "Stream community mapping"
)

# COMMAND ----------

# DBTITLE 1,Common variables & output directory operation
volume_path = dbutils.widgets.get("volume_name")
dependency_input_path = volume_path + dbutils.widgets.get("input_dependency_name")
outofscope_stream_path = volume_path + dbutils.widgets.get("outofscope_stream_file_name")
report_to_table_dependency = volume_path + dbutils.widgets.get("report_table_dependency_file_name")
view_table_association = volume_path + dbutils.widgets.get("view_table_association")
community_mapping = volume_path + dbutils.widgets.get("community_mapping")
table_size = volume_path + dbutils.widgets.get("table_size")

# Output path with date and hour
output_dir_name = "bi_migration_analysis_output" + datetime.now().strftime("%d%m%Y_%H")
output_path = volume_path + "bi_migration_analysis_output_latest/" + output_dir_name + "/"

# Move existing folders from 'latest' one level up (to volume_path)
latest_path = volume_path + "bi_migration_analysis_output_latest/"
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

# DBTITLE 1,Read view table dependencies and standardize
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Read the csv file
view_dependency_df = spark.read.format("csv").option("header", "true").load(view_table_association)



# COMMAND ----------

# DBTITLE 1,Reading report to table dependency
# Read report to stream dependency and standardize values and column names
# Each table is marked as a Src, since reports only create tables and do not write to tables (or only exceptions)
# Filtering two reports wrt corona and gdpr, since they skew the community creation due to high number of dependencies
report_table_dependency_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(report_to_table_dependency)
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
    print(f"\nWARNING: Selected resolution {SELECTED_RESOLUTION} was not processed in cell 40!")
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

# DBTITLE 1,Calculate report readiness at each execution stage
# Get execution orders in the EXACT order they appear (preserving optimization order)
# Do NOT sort - the order from the CSV reflects the optimized community ordering
execution_stages = stream_ordering_pd['execution_order'].unique().tolist()

print(f"Total execution stages: {len(execution_stages)}")
print(f"Execution order (optimized): {execution_stages}")

missing_static_tables_df = spark.read.option("header", True).csv("/Volumes/odp_adw_mvp_n/migration/utilities/community_detection/static_tables_for_report.csv").select("table_name")
missing_static_tables = set(row["table_name"] for row in missing_static_tables_df.collect())


# Initialize tracking
available_tables = missing_static_tables  # Tables available from ALL migrated streams (cumulative) and initialized with already available static tables
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
    # f.write(f"Weight Method: {WEIGHT_METHOD}\n")
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

# DBTITLE 1,Report migration readiness output
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
