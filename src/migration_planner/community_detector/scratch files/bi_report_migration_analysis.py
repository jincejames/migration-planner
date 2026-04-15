# Databricks notebook source
# DBTITLE 1,Library imports
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text(
    "volume_name",
    "/Volumes/odp_adw_utilities_n/planning/utilities/community_detection/",
    "Input Volume Path",
)
dbutils.widgets.text(
    "input_dependency_name",
    "ODAT__202603301324.csv",
    "Input CSV file name",
)
dbutils.widgets.text(
    "outofscope_stream_file_name",
    "out-of-scopte-streams.csv",
    "Out of scope streams file name",
)
dbutils.widgets.text(
    "report_table_dependency_file_name",
    "Unique_Reports_With_Queries_and_Tables.csv",
    "Report to table dependency file name",
)
dbutils.widgets.text(
    "view_table_association",
    "View-Table-Association_20260401.csv",
    "View Table dependencies",
)
dbutils.widgets.text(
    "community_mapping",
    "ETL_Scripts_20260106_Master_Communities.csv",
    "Stream community mapping",
)
dbutils.widgets.text(
    "static_tables_file_name",
    "static_tables_for_report.csv",
    "Static tables available from start",
)
dbutils.widgets.text(
    "prd_views_mapping",
    "PRDViews_to_Table_mapping.csv",
    "PRD Views to Table mapping file name",
)

# COMMAND ----------

# DBTITLE 1,Common variables & output directory operation
volume_path = dbutils.widgets.get("volume_name")
dependency_input_path = volume_path + dbutils.widgets.get("input_dependency_name")
outofscope_stream_path = volume_path + dbutils.widgets.get("outofscope_stream_file_name")
report_dependency_path = volume_path + dbutils.widgets.get("report_table_dependency_file_name")
view_table_association = volume_path + dbutils.widgets.get("view_table_association")
community_mapping = volume_path + dbutils.widgets.get("community_mapping")
static_tables_path = volume_path + dbutils.widgets.get("static_tables_file_name")
prd_views_mapping_path = volume_path + dbutils.widgets.get("prd_views_mapping")

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
# MAGIC ## Loading Datasets

# COMMAND ----------

# DBTITLE 1,Reading input stream-table dependency file
# Normalize column names to lowercase for consistent referencing
_dep_raw = spark.read.format("csv").option("header", "true").load(dependency_input_path)
dependency_df_full = _dep_raw.select(
    [F.col(c).alias(c.lower()) for c in _dep_raw.columns]
)

# Filter out rows with empty DB_Table_Name or table_type (admin streams with no tables)
dependency_df_full = dependency_df_full.filter(
    F.col("db_table_name").isNotNull()
    & (F.trim(F.col("db_table_name")) != "")
    & F.col("table_type").isNotNull()
    & (F.trim(F.col("table_type")) != "")
)

# COMMAND ----------

# DBTITLE 1,Reading out-of-scope stream names and filtering dependency data
outofscope_stream_names_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(outofscope_stream_path)
    .select(F.col("stream_name"))
)

# Filter out out-of-scope streams from the dependency data
dependency_df_full = dependency_df_full.join(
    outofscope_stream_names_df,
    dependency_df_full["stream_name"] == outofscope_stream_names_df["stream_name"],
    "left_anti",
)

# COMMAND ----------

# DBTITLE 1,Read view-table dependencies and resolve views to source tables
view_dependency_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(view_table_association)
)

# Resolve each view to its ultimate source TABLEs iteratively:
#   - rows where dep_object_type_cd = 'TABLE' are already resolved
#   - rows where dep_object_type_cd = 'VIEW'  need another hop
# We stop when no VIEW dependencies remain (all paths end at TABLEs).

# Standardise to upper-case and build fully-qualified names (DATABASE.OBJECT)
view_dep = view_dependency_df.select(
    F.concat_ws(".", F.upper(F.col("DATABASE_NAME")), F.upper(F.col("object_name"))).alias("view_fqn"),
    F.upper(F.col("dep_object_type_cd")).alias("dep_type"),
    F.concat_ws(".", F.upper(F.col("dep_database_name")), F.upper(F.col("dep_object_name"))).alias("dep_fqn"),
)

# Split into already-resolved (TABLE deps) and still-to-resolve (VIEW deps)
resolved = (
    view_dep.filter(F.col("dep_type") == "TABLE")
    .select(F.col("view_fqn").alias("view_name"), F.col("dep_fqn").alias("source_table"))
)
unresolved = (
    view_dep.filter(F.col("dep_type") == "VIEW")
    .select(F.col("view_fqn").alias("view_name"), F.col("dep_fqn").alias("intermediate_view"))
)

# Iteratively replace each intermediate VIEW with its own dependencies
MAX_VIEW_DEPTH = 20
for i in range(MAX_VIEW_DEPTH):
    if unresolved.count() == 0:
        break

    next_hop = unresolved.join(
        view_dep,
        unresolved["intermediate_view"] == view_dep["view_fqn"],
        "inner",
    ).select(
        unresolved["view_name"],
        view_dep["dep_fqn"].alias("dep_name"),
        view_dep["dep_type"],
    )

    newly_resolved = (
        next_hop.filter(F.col("dep_type") == "TABLE")
        .select("view_name", F.col("dep_name").alias("source_table"))
    )
    resolved = resolved.union(newly_resolved).distinct()

    unresolved = (
        next_hop.filter(F.col("dep_type") == "VIEW")
        .select("view_name", F.col("dep_name").alias("intermediate_view"))
    )

# Final view -> source-table mapping (one row per view-table pair)
view_to_source_tables_df = resolved.dropDuplicates(["view_name", "source_table"])

print(f"Resolved {view_to_source_tables_df.count()} view -> source-table mappings")
print(f"Unique views:  {view_to_source_tables_df.select('view_name').distinct().count()}")
print(f"Unique tables: {view_to_source_tables_df.select('source_table').distinct().count()}")
display(view_to_source_tables_df)

# Save iterative resolution output before merging (for comparison analysis)
iterative_view_to_tables_df = view_to_source_tables_df

# COMMAND ----------

# DBTITLE 1,Reading PRD views-to-table mapping (supplementary)
prd_views_raw_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(prd_views_mapping_path)
    .select(
        F.upper(F.col("view_name")).alias("view_name"),
        F.col("dependent_tables"),
    )
)

# Filter to views with resolved table dependencies
prd_views_with_tables = prd_views_raw_df.filter(
    F.col("dependent_tables").isNotNull()
    & (F.trim(F.col("dependent_tables")) != "")
)

# Explode pipe-separated dependent_tables into individual rows
prd_view_to_tables_df = (
    prd_views_with_tables
    .withColumn("source_table", F.explode(F.split(F.col("dependent_tables"), r"\s*\|\s*")))
    .select(
        F.col("view_name"),
        F.upper(F.trim(F.col("source_table"))).alias("source_table"),
    )
    .filter(F.col("source_table") != "")
    .dropDuplicates(["view_name", "source_table"])
)

print(f"PRD view-to-table mappings: {prd_view_to_tables_df.count()}")
print(f"Unique views (PRD):  {prd_view_to_tables_df.select('view_name').distinct().count()}")
print(f"Unique tables (PRD): {prd_view_to_tables_df.select('source_table').distinct().count()}")
print(f"PRD views with no direct table deps: {prd_views_raw_df.count() - prd_views_with_tables.count()}")
display(prd_view_to_tables_df)

# COMMAND ----------

# DBTITLE 1,Merge iterative and PRD view-to-table mappings
# Union source tables from both datasets, deduplicate:
#   - Views in both: union of their source tables
#   - Views in only one: kept as-is
view_to_source_tables_df = (
    iterative_view_to_tables_df
    .union(prd_view_to_tables_df)
    .dropDuplicates(["view_name", "source_table"])
)

print(f"Merged view -> source-table mappings: {view_to_source_tables_df.count()}")
print(f"Unique views (merged):  {view_to_source_tables_df.select('view_name').distinct().count()}")
print(f"Unique tables (merged): {view_to_source_tables_df.select('source_table').distinct().count()}")
display(view_to_source_tables_df)

# COMMAND ----------

# DBTITLE 1,Analysis: Iterative vs PRD view-to-table mapping comparison
iter_pd = iterative_view_to_tables_df.toPandas()
prd_pd = prd_view_to_tables_df.toPandas()

iter_view_tables = iter_pd.groupby("view_name")["source_table"].apply(set).to_dict()
prd_view_tables = prd_pd.groupby("view_name")["source_table"].apply(set).to_dict()

iter_views = set(iter_view_tables.keys())
prd_views = set(prd_view_tables.keys())

# All PRD view names (including those with no dependent_tables)
prd_all_views = set(
    prd_views_raw_df.select("view_name").distinct().toPandas()["view_name"]
)

common_views = iter_views & prd_views
only_in_iterative = iter_views - prd_all_views
only_in_prd = prd_views - iter_views
# Views present in both datasets, but PRD has no table deps for them
in_both_but_prd_no_tables = (iter_views & prd_all_views) - prd_views

# Classify common views (both have source tables)
exact_match = []
varying_tables = []
for v in sorted(common_views):
    if iter_view_tables[v] == prd_view_tables[v]:
        exact_match.append(v)
    else:
        varying_tables.append(v)

print("=" * 80)
print("VIEW-TO-TABLE MAPPING COMPARISON: Iterative Resolution vs PRD Mapping")
print("=" * 80)
print(f"\nTotal views in Iterative resolution:            {len(iter_views)}")
print(f"Total views in PRD mapping (with tables):        {len(prd_views)}")
print(f"Total views in PRD mapping (all):                {len(prd_all_views)}")
print()
print(f"Common views (both have source tables):          {len(common_views)}")
print(f"  - Exact same source tables:                    {len(exact_match)}")
print(f"  - Different source table lists:                {len(varying_tables)}")
print(f"Tables in Iterative only (no tables in PRD):     {len(in_both_but_prd_no_tables)}")
print(f"Views only in Iterative (not in PRD at all):     {len(only_in_iterative)}")
print(f"Views only in PRD (not in Iterative):            {len(only_in_prd)}")

MAX_EXAMPLES = 5

if exact_match:
    print(f"\n{'─' * 80}")
    print(f"EXAMPLES: Views with EXACT SAME source tables ({min(MAX_EXAMPLES, len(exact_match))} of {len(exact_match)})")
    print(f"{'─' * 80}")
    for v in exact_match[:MAX_EXAMPLES]:
        tables = sorted(iter_view_tables[v])
        print(f"  {v}")
        print(f"    Tables: {', '.join(tables[:5])}{' ...' if len(tables) > 5 else ''}")

if varying_tables:
    print(f"\n{'─' * 80}")
    print(f"EXAMPLES: Views with DIFFERENT source tables ({min(MAX_EXAMPLES, len(varying_tables))} of {len(varying_tables)})")
    print(f"{'─' * 80}")
    for v in varying_tables[:MAX_EXAMPLES]:
        iter_only = sorted(iter_view_tables[v] - prd_view_tables[v])
        prd_only = sorted(prd_view_tables[v] - iter_view_tables[v])
        shared = sorted(iter_view_tables[v] & prd_view_tables[v])
        print(f"  {v}")
        if shared:
            print(f"    Shared:          {', '.join(shared[:3])}{' ...' if len(shared) > 3 else ''}")
        if iter_only:
            print(f"    Only Iterative:  {', '.join(iter_only[:3])}{' ...' if len(iter_only) > 3 else ''}")
        if prd_only:
            print(f"    Only PRD:        {', '.join(prd_only[:3])}{' ...' if len(prd_only) > 3 else ''}")

if in_both_but_prd_no_tables:
    print(f"\n{'─' * 80}")
    print(f"EXAMPLES: Views with tables in Iterative but NO tables in PRD ({min(MAX_EXAMPLES, len(in_both_but_prd_no_tables))} of {len(in_both_but_prd_no_tables)})")
    print(f"{'─' * 80}")
    for v in sorted(in_both_but_prd_no_tables)[:MAX_EXAMPLES]:
        tables = sorted(iter_view_tables[v])
        print(f"  {v}")
        print(f"    Iterative tables: {', '.join(tables[:5])}{' ...' if len(tables) > 5 else ''}")

if only_in_iterative:
    print(f"\n{'─' * 80}")
    print(f"EXAMPLES: Views ONLY in Iterative ({min(MAX_EXAMPLES, len(only_in_iterative))} of {len(only_in_iterative)})")
    print(f"{'─' * 80}")
    for v in sorted(only_in_iterative)[:MAX_EXAMPLES]:
        tables = sorted(iter_view_tables[v])
        print(f"  {v}")
        print(f"    Tables: {', '.join(tables[:5])}{' ...' if len(tables) > 5 else ''}")

if only_in_prd:
    print(f"\n{'─' * 80}")
    print(f"EXAMPLES: Views ONLY in PRD ({min(MAX_EXAMPLES, len(only_in_prd))} of {len(only_in_prd)})")
    print(f"{'─' * 80}")
    for v in sorted(only_in_prd)[:MAX_EXAMPLES]:
        tables = sorted(prd_view_tables[v])
        print(f"  {v}")
        print(f"    Tables: {', '.join(tables[:5])}{' ...' if len(tables) > 5 else ''}")

# Summary DataFrame for display
comparison_summary = pd.DataFrame([
    {"Category": "Common views - exact same tables", "Count": len(exact_match)},
    {"Category": "Common views - different table lists", "Count": len(varying_tables)},
    {"Category": "Tables in Iterative only (PRD has view but no tables)", "Count": len(in_both_but_prd_no_tables)},
    {"Category": "Views only in Iterative (not in PRD)", "Count": len(only_in_iterative)},
    {"Category": "Views only in PRD (not in Iterative)", "Count": len(only_in_prd)},
])
display(comparison_summary)

# COMMAND ----------

# DBTITLE 1,Reading report-to-table dependency (with view resolution)
# Read report CSV: "Workbook Name" -> report, "fullName" -> table reference.
# fullName comes in two formats: "schema.table" or "[SCHEMA].[TABLE]"
# Strip brackets and upper-case to get a consistent DB_NAME.TABLE_NAME format,
# then replace any table that is actually a view with its resolved source tables.
raw_report_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .load(report_dependency_path)
    .select(
        F.col("Workbook Name").alias("report_name"),
        F.upper(F.regexp_replace(F.col("fullName"), r"[\[\]']", "")).alias("table_name"),
    )
    .distinct()
    .filter(
        ~F.lower(F.col("report_name")).contains("corona")
        & ~F.lower(F.col("report_name")).contains("gdpr")
    )
    # Filter out malformed table names: must have non-empty schema AND table parts
    # e.g. reject ".TABLE", "SCHEMA.", ".", "", or null
    .filter(
        F.col("table_name").isNotNull()
        & ~F.col("table_name").startswith(".")
        & ~F.col("table_name").endswith(".")
        & (F.col("table_name") != "")
    )
)

# Split: rows whose table_name matches a known view vs those that don't
report_with_views = raw_report_df.join(
    view_to_source_tables_df,
    raw_report_df["table_name"] == view_to_source_tables_df["view_name"],
    "inner",
).select(
    raw_report_df["report_name"],
    view_to_source_tables_df["source_table"].alias("table_name"),
)

report_without_views = raw_report_df.join(
    view_to_source_tables_df,
    raw_report_df["table_name"] == view_to_source_tables_df["view_name"],
    "left_anti",
).select("report_name", "table_name")

# Combine: views replaced with source tables + tables that were already tables
report_table_dependency_df = (
    report_with_views
    .union(report_without_views)
    .select(
        F.col("report_name"),
        F.col("table_name"),
        F.lit("Src").alias("table_type"),
    )
    .dropDuplicates(["report_name", "table_name"])
)

print(f"Report-table dependencies (after view resolution): {report_table_dependency_df.count()}")
print(f"Unique reports: {report_table_dependency_df.select('report_name').distinct().count()}")
print(f"Unique tables:  {report_table_dependency_df.select('table_name').distinct().count()}")

# COMMAND ----------

# DBTITLE 1,Reading community mapping and building community order
community_raw_df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("sep", ";")
    .load(community_mapping)
)

print("Detected columns:", community_raw_df.columns)

# Replace #N/A strings with null for date columns
DATE_COLS = [
    "Code Freeze Start",
    "Code Conv. Start Date",
    "Code Conv. End Date",
    "Code Freeze End",
]

community_df = community_raw_df.select(
    F.col("`Community_Number(Old)`").alias("community_old"),
    F.col("Updated_Community_Number").alias("community_new"),
    F.col("`Stream Name`").alias("stream_name"),
    F.col("`Scope Status`").alias("scope_status"),
    *[
        F.when(F.col(f"`{c}`") == "#N/A", None)
        .otherwise(F.col(f"`{c}`"))
        .alias(c)
        for c in DATE_COLS
    ],
)

community_df = (
    community_df
    .filter(~F.upper(F.col("scope_status")).contains("OUT OF SCOPE"))
    .withColumn(
        "community",
        F.when(
            F.col("community_new").isNotNull() & (F.trim(F.col("community_new")) != ""),
            F.col("community_new"),
        ).otherwise(F.col("community_old")),
    )
    .select(
        "community",
        "stream_name",
        F.col("`Code Freeze Start`").alias("code_freeze_start"),
        F.col("`Code Conv. Start Date`").alias("code_conv_start"),
        F.col("`Code Conv. End Date`").alias("code_conv_end"),
        F.col("`Code Freeze End`").alias("code_freeze_end"),
    )
)

print(f"Community-stream mappings (in-scope): {community_df.count()}")
print(f"Unique communities: {community_df.select('community').distinct().count()}")
display(community_df)

# Build community execution order based on earliest Code Freeze Start date.
# Parse all date columns before aggregating so min/max are chronological, not lexicographic.
DATE_FMT = F.lit("d-MMM-yy")

community_order_df = (
    community_df
    .withColumn("_freeze_start", F.try_to_timestamp(F.col("code_freeze_start"), DATE_FMT))
    .withColumn("_conv_start", F.try_to_timestamp(F.col("code_conv_start"), DATE_FMT))
    .withColumn("_conv_end", F.try_to_timestamp(F.col("code_conv_end"), DATE_FMT))
    .withColumn("_freeze_end", F.try_to_timestamp(F.col("code_freeze_end"), DATE_FMT))
    .groupBy("community")
    .agg(
        F.min("_freeze_start").alias("_freeze_start"),
        F.min("_conv_start").alias("_conv_start"),
        F.max("_conv_end").alias("_conv_end"),
        F.max("_freeze_end").alias("_freeze_end"),
    )
    .withColumn("code_freeze_start", F.date_format(F.col("_freeze_start"), "d-MMM-yy"))
    .withColumn("code_conv_start", F.date_format(F.col("_conv_start"), "d-MMM-yy"))
    .withColumn("code_conv_end", F.date_format(F.col("_conv_end"), "d-MMM-yy"))
    .withColumn("code_freeze_end", F.date_format(F.col("_freeze_end"), "d-MMM-yy"))
    .withColumn(
        "has_date",
        F.when(F.col("_freeze_start").isNotNull(), F.lit(0)).otherwise(F.lit(1)),
    )
)

# Communities with no freeze date go last
order_window = Window.orderBy("has_date", "_freeze_start", "community")
community_order_df = (
    community_order_df
    .withColumn("execution_order", F.row_number().over(order_window))
    .select(
        "community",
        "execution_order",
        "code_freeze_start",
        "code_conv_start",
        "code_conv_end",
        "code_freeze_end",
    )
)

print(f"\nCommunity execution order ({community_order_df.count()} communities):")
display(community_order_df)

# COMMAND ----------

# DBTITLE 1,Reading static tables (available from start)
static_tables_df = (
    spark.read.format("csv")
    .option("header", "true")
    .load(static_tables_path)
    .select(F.upper(F.col("table_name")).alias("table_name"))
)
static_tables = set(row["table_name"] for row in static_tables_df.collect())
print(f"Static tables available from start: {len(static_tables)}")

# COMMAND ----------

# DBTITLE 1,Considering all TGT tables as SRC
# Every TGT table is also a potential SRC (a gap in ODAT output).
# If a table is written by multiple streams, it is both SRC and TGT in each.
tgt_as_source = (
    dependency_df_full
    .filter(F.upper(F.col("table_type")).contains("TGT"))
    .replace({"Tgt": "Src", "Tgt_Trns": "Src_Trns"}, subset=["table_type"])
)
dependency_df = dependency_df_full.union(tgt_as_source).distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Report Migration Readiness Analysis
# MAGIC
# MAGIC A report is ready to migrate when **all** its source tables are available.
# MAGIC A source table becomes available when **every** stream that produces it (TGT/TGT_TRNS)
# MAGIC belongs to a community that has completed migration.
# MAGIC Static tables are available from the start.

# COMMAND ----------

# DBTITLE 1,Build table-to-stream production mapping
# Include TGT, TGT_TRNS (tables written by streams) and File (file-based loads)
stream_produces_df = (
    dependency_df
    .filter(
        (F.upper(F.col("table_type")) == "TGT")
        | (F.upper(F.col("table_type")) == "TGT_TRNS")
        | (F.upper(F.col("table_type")) == "FILE")
    )
    .select(
        F.col("stream_name"),
        F.upper(F.col("db_table_name")).alias("table_name"),
    )
    .distinct()
)

print(f"Stream-produces-table mappings: {stream_produces_df.count()}")
print(f"Unique producing streams: {stream_produces_df.select('stream_name').distinct().count()}")
print(f"Unique produced tables:   {stream_produces_df.select('table_name').distinct().count()}")

# COMMAND ----------

# DBTITLE 1,Map tables to their producing communities
# For each table, find the latest community (by execution order) that produces it.
# That is when the table becomes fully available.
table_to_community_df = (
    stream_produces_df
    .join(community_df.select("stream_name", "community"), on="stream_name", how="inner")
    .select("table_name", "community")
    .distinct()
)

table_community_order_df = (
    table_to_community_df
    .join(community_order_df, on="community", how="inner")
    .select("table_name", "community", "execution_order", "code_freeze_end")
)

table_max_order_df = (
    table_community_order_df
    .groupBy("table_name")
    .agg(F.max("execution_order").alias("available_after_order"))
)

# Join back to get community name and code_freeze_end for the max execution_order
table_availability_df = (
    table_max_order_df.alias("ta")
    .join(
        table_community_order_df.alias("tco"),
        (F.col("ta.table_name") == F.col("tco.table_name"))
        & (F.col("ta.available_after_order") == F.col("tco.execution_order")),
        "inner",
    )
    .select(
        F.col("ta.table_name"),
        F.col("tco.community").alias("available_after_community"),
        F.col("ta.available_after_order"),
        F.col("tco.code_freeze_end").alias("available_after_date"),
    )
    .distinct()
)

print(f"Tables with community availability: {table_availability_df.count()}")
display(table_availability_df.orderBy("available_after_order"))

# COMMAND ----------

# DBTITLE 1,Determine report readiness
# For each report, find when it becomes ready based on its table dependencies.
# A report is ready after the LATEST of its table dependencies becomes available.
# Static tables are available from the start (order = 0).

table_avail_pd = table_availability_df.toPandas()
table_avail_map = dict(
    zip(
        table_avail_pd["table_name"],
        zip(
            table_avail_pd["available_after_community"],
            table_avail_pd["available_after_order"],
            table_avail_pd["available_after_date"],
        ),
    )
)

comm_order_pd = community_order_df.toPandas()

# Report to tables, dropping any null values
report_tables_pd = (
    report_table_dependency_df
    .select("report_name", "table_name")
    .filter(F.col("table_name").isNotNull() & F.col("report_name").isNotNull())
    .distinct()
    .toPandas()
)

report_required = report_tables_pd.groupby("report_name")["table_name"].apply(set).to_dict()

print(f"Total reports: {len(report_required)}")
print(f"Total unique tables required: {report_tables_pd['table_name'].nunique()}")

# Classify each report as ready or not-ready
all_report_rows = []

for report, tables in report_required.items():
    max_order = 0
    max_community = "Static (available from start)"
    max_date = ""
    missing_tables = []
    all_resolved = True

    for tbl in tables:
        if tbl in static_tables:
            continue
        elif tbl in table_avail_map:
            comm, order, date = table_avail_map[tbl]
            if order > max_order:
                max_order = order
                max_community = comm
                max_date = date if date else ""
        else:
            all_resolved = False
            missing_tables.append(tbl)

    # Ideal End Date = code_freeze_end + 20 days buffer
    ideal_end = ""
    if max_date:
        try:
            freeze_end_dt = pd.to_datetime(max_date, format="%d-%b-%y", dayfirst=True)
            ideal_end = (freeze_end_dt + timedelta(days=20)).strftime("%-d-%b-%y")
        except (ValueError, TypeError):
            ideal_end = ""

    all_report_rows.append({
        "report_name": report,
        "status": "Ready" if all_resolved else "Not Ready",
        "Ready After": max_community,
        "Migrate After Execution Order": max_order,
        "num_required_tables": len(tables),
        "Ready to migrate from": max_date,
        "Ideal End Date": ideal_end,
        "num_missing_tables": len(missing_tables),
        "missing_tables": ", ".join(sorted(missing_tables)),
        "required_tables": ", ".join(sorted(tables)),
    })

# Split for per-status views, then combine: ready first (by order), not-ready last (by name)
all_reports_df = pd.DataFrame(all_report_rows)
readiness_df = (
    all_reports_df[all_reports_df["status"] == "Ready"]
    .sort_values(["Migrate After Execution Order", "report_name"])
    .reset_index(drop=True)
)
not_ready_df = (
    all_reports_df[all_reports_df["status"] == "Not Ready"]
    .sort_values("report_name")
    .reset_index(drop=True)
)
all_reports_df = pd.concat([readiness_df, not_ready_df], ignore_index=True)

print(f"\nTotal reports analyzed: {len(all_reports_df)}")
print(f"Reports ready after migration: {len(readiness_df)}")
print(f"Reports NOT ready (missing tables): {len(not_ready_df)}")
display(all_reports_df.head(20))

# COMMAND ----------

# DBTITLE 1,Save report readiness CSV
readiness_file = f"{output_path}report_migration_readiness.csv"
all_reports_df.to_csv(readiness_file, index=False)
print(f"Readiness CSV saved to: {readiness_file}")
print(f"  Ready reports: {len(readiness_df)}, Not-ready reports: {len(not_ready_df)}")

display(all_reports_df)

# COMMAND ----------

# DBTITLE 1,Readiness summary by community
summary = (
    readiness_df
    .groupby(["Migrate After Execution Order", "Ready After"])
    .agg(reports_ready=("report_name", "count"))
    .reset_index()
    .sort_values("Migrate After Execution Order")
)
summary["cumulative_reports"] = summary["reports_ready"].cumsum()
display(summary)

# COMMAND ----------

# DBTITLE 1,Generate detailed text report
report_text_file = f"{output_path}report_migration_readiness_analysis.txt"

with open(report_text_file, "w") as f:
    f.write("=" * 100 + "\n")
    f.write("REPORT MIGRATION READINESS ANALYSIS\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
    f.write(f"Total Reports Analyzed: {len(report_required)}\n")
    f.write(f"Reports Ready After All Communities: {len(readiness_df)}\n")
    f.write(f"Reports Not Ready: {len(not_ready_df)}\n")
    f.write(f"Static Tables (available from start): {len(static_tables)}\n")
    f.write(f"Total Communities: {len(comm_order_pd)}\n\n")

    # Reports ready immediately (only static table dependencies)
    static_ready = readiness_df[readiness_df["Migrate After Execution Order"] == 0]
    if len(static_ready) > 0:
        f.write("=" * 100 + "\n")
        f.write(
            f"REPORTS READY IMMEDIATELY — only static table dependencies ({len(static_ready)})\n"
        )
        f.write("=" * 100 + "\n\n")
        for _, rpt in static_ready.iterrows():
            f.write(f"  - {rpt['report_name']} ({rpt['num_required_tables']} tables)\n")
        f.write("\n")

    f.write("=" * 100 + "\n")
    f.write("REPORT READINESS BY COMMUNITY (EXECUTION ORDER)\n")
    f.write("=" * 100 + "\n")

    cumulative = len(static_ready)
    for _, row in comm_order_pd.sort_values("execution_order").iterrows():
        comm = row["community"]
        order = row["execution_order"]
        freeze_start = row["code_freeze_start"] if pd.notna(row["code_freeze_start"]) else "N/A"
        freeze_end = row["code_freeze_end"] if pd.notna(row["code_freeze_end"]) else "N/A"

        reports_at = readiness_df[readiness_df["Migrate After Execution Order"] == order]
        cumulative += len(reports_at)

        f.write(f"\n{'─' * 100}\n")
        f.write(f"EXECUTION ORDER {order}: {comm}\n")
        f.write(f"{'─' * 100}\n")
        f.write(f"Code Freeze: {freeze_start} - {freeze_end}\n")
        f.write(f"Reports Ready at This Stage: {len(reports_at)}\n")
        f.write(f"Cumulative Reports Ready: {cumulative}\n\n")

        if len(reports_at) > 0:
            for _, rpt in reports_at.iterrows():
                table_list = rpt["required_tables"].split(", ")[:5]
                f.write(f"  - {rpt['report_name']}\n")
                f.write(f"    Tables ({rpt['num_required_tables']}): {', '.join(table_list)}")
                if rpt["num_required_tables"] > 5:
                    f.write(f" ... and {rpt['num_required_tables'] - 5} more")
                f.write("\n")
        else:
            f.write("  No new reports ready at this stage.\n")

    # Reports not ready
    if len(not_ready_df) > 0:
        f.write(f"\n\n{'=' * 100}\n")
        f.write(f"REPORTS NOT READY AFTER ALL COMMUNITIES ({len(not_ready_df)})\n")
        f.write(f"{'=' * 100}\n\n")
        f.write("These reports depend on tables not produced by any known stream/community.\n\n")

        for _, rpt in not_ready_df.iterrows():
            f.write(f"  - {rpt['report_name']}\n")
            f.write(f"    Required Tables: {rpt['num_required_tables']}\n")
            f.write(
                f"    Missing Tables ({rpt['num_missing_tables']}): {rpt['missing_tables']}\n\n"
            )

print(f"Text report saved to: {report_text_file}")
print("Analysis complete!")
