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

# COMMAND ----------

# DBTITLE 1,Common variables & output directory operation
volume_path = dbutils.widgets.get("volume_name")
dependency_input_path = volume_path + dbutils.widgets.get("input_dependency_name")
outofscope_stream_path = volume_path + dbutils.widgets.get("outofscope_stream_file_name")
report_dependency_path = volume_path + dbutils.widgets.get("report_table_dependency_file_name")
view_table_association = volume_path + dbutils.widgets.get("view_table_association")
community_mapping = volume_path + dbutils.widgets.get("community_mapping")
static_tables_path = volume_path + dbutils.widgets.get("static_tables_file_name")

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
readiness_rows = []
not_ready_rows = []

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

    if all_resolved:
        # Ideal End Date = code_freeze_end + 20 days buffer
        ideal_end = ""
        if max_date:
            try:
                freeze_end_dt = pd.to_datetime(max_date, format="%d-%b-%y", dayfirst=True)
                ideal_end = (freeze_end_dt + timedelta(days=20)).strftime("%-d-%b-%y")
            except (ValueError, TypeError):
                ideal_end = ""

        readiness_rows.append({
            "Ready After": max_community,
            "Migrate After Execution Order": max_order,
            "report_name": report,
            "num_required_tables": len(tables),
            "Ready to migrate from": max_date,
            "Ideal End Date": ideal_end,
            "required_tables": ", ".join(sorted(tables)),
        })
    else:
        not_ready_rows.append({
            "report_name": report,
            "num_required_tables": len(tables),
            "num_missing_tables": len(missing_tables),
            "missing_tables": ", ".join(sorted(missing_tables)),
            "required_tables": ", ".join(sorted(tables)),
        })

readiness_df = (
    pd.DataFrame(readiness_rows)
    .sort_values(["Migrate After Execution Order", "report_name"])
    .reset_index(drop=True)
)

not_ready_df = (
    pd.DataFrame(not_ready_rows)
    .sort_values("report_name")
    .reset_index(drop=True)
)

print(f"\nReports ready after migration: {len(readiness_df)}")
print(f"Reports NOT ready (missing tables): {len(not_ready_df)}")
display(readiness_df.head(20))

# COMMAND ----------

# DBTITLE 1,Save report readiness CSV
readiness_file = f"{output_path}report_migration_readiness.csv"
readiness_df.to_csv(readiness_file, index=False)
print(f"Readiness CSV saved to: {readiness_file}")

if len(not_ready_df) > 0:
    not_ready_file = f"{output_path}reports_not_ready.csv"
    not_ready_df.to_csv(not_ready_file, index=False)
    print(f"Not-ready CSV saved to: {not_ready_file}")

display(readiness_df)

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
