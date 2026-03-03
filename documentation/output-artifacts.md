# Output Artifacts

This document describes every file and table that Migration Planner produces, explains how each is generated, and provides guidance on how to read and act on the results.

---

## Table of Contents

- [Output Directory Layout](#output-directory-layout)
- [1. Edge List — `edges.csv`](#1-edge-list--edgescsv)
- [2. Isolated Streams — `isolated_streams.csv`](#2-isolated-streams--isolated_streamscsv)
- [3. Resolution Scan Summary](#3-resolution-scan-summary)
- [4. Per-Community Analysis Reports](#4-per-community-analysis-reports)
- [5. Migration Order Report](#5-migration-order-report)
- [6. Sync Details — `sync_details_gamma_<γ>.csv`](#6-sync-details--sync_details_gamma_γcsv)
- [7. Stream Ordering — `stream_ordering_gamma_<γ>.csv`](#7-stream-ordering--stream_ordering_gamma_γcsv)
- [8. Execution Metadata — Delta Table](#8-execution-metadata--delta-table)
- [9. Graph Visualizations (optional)](#9-graph-visualizations-optional)
- [How to Choose a Resolution](#how-to-choose-a-resolution)
- [How to Plan a Migration from the Outputs](#how-to-plan-a-migration-from-the-outputs)

---

## Output Directory Layout

Every run creates a timestamped subdirectory under the volume:

```
<volume_name>/
├── community_detection_output_latest/             ← always points to the newest run
│   └── community_detection_output_<ddmmyyyy_HH>/
│       ├── edges.csv
│       ├── isolated_streams.csv
│       ├── leiden_community_plots_gamma_<γ>/
│       │   ├── community_0/
│       │   │   └── community_0_analysis.txt
│       │   ├── community_1/
│       │   │   └── community_1_analysis.txt
│       │   └── ...
│       └── migration_order_analysis/
│           ├── migration_order_report_gamma_<γ>.txt
│           ├── sync_details_gamma_<γ>.csv
│           └── stream_ordering_gamma_<γ>.csv
│
├── community_detection_output_<ddmmyyyy_HH_prev>/  ← archived previous run
└── ...
```

Archived runs are preserved under `<volume_name>/` so that historical analysis results are not overwritten.

---

## 1. Edge List — `edges.csv`

**Path:** `<output_path>/edges.csv`
**When produced:** After the preprocessing and weight calculation pipeline completes.

### Schema

| Column | Type | Description |
|---|---|---|
| `streamA` | string | First endpoint of the undirected edge |
| `streamB` | string | Second endpoint of the undirected edge |
| `weight` | float/int | Total migration sync weight for this edge |

### How it is produced

The merged, weighted, undirected edge DataFrame (`merged_dependency_df`) is collected to pandas and written to CSV:

```python
edges_df = merged_dependency_df.toPandas()
edges_df.to_csv(f"{output_path}edges.csv", index=False)
```

Edge weights are the sum of per-table weights across all shared tables between the two streams. Bidirectional edges (A→B and B→A) are merged into a single row by summing their individual weights.

### How to interpret it

- Each row represents a pair of ETL streams (or BI reports) that share at least one database table.
- A higher `weight` means more migration effort is required if the two streams are placed in different migration phases — more or larger tables need to be synchronized between phases.
- Streams that do not appear in this file at all are isolated (see `isolated_streams.csv`).
- This file is also used internally to build the igraph and NetworkX graphs for community detection.

### How to use it

- Visualize the dependency network using any graph tool (Gephi, NetworkX, D3.js) by treating `streamA`/`streamB` as node pairs and `weight` as edge thickness.
- Sort by weight descending to find the most tightly coupled stream pairs — these are candidates that should ideally be placed in the same migration phase.

---

## 2. Isolated Streams — `isolated_streams.csv`

**Path:** `<output_path>/isolated_streams.csv`
**When produced:** After isolated stream detection; not written if no isolated streams exist.

### Schema

| Column | Type | Description |
|---|---|---|
| `stream_name` | string | Name of the isolated stream |

### How it is produced

A stream is isolated when it does not appear as either `streamA` or `streamB` in the merged edge list — meaning it has no inter-stream table dependencies. It may still have intra-stream dependencies (it reads and writes its own tables, but no other stream reads its outputs or writes its inputs).

```python
isolated_streams = find_isolated_streams(dependency_df, merged_dependency_df)
```

Isolated streams are added as disconnected nodes to both the igraph and NetworkX graphs, so they still receive community assignments (they each form their own singleton community under fine-grained resolutions, or are merged into a shared "isolated" community by the single-stream community consolidation step).

### How to interpret it

- Isolated streams can be migrated **independently** and in **any order** — they have no external table sync requirements.
- They represent the easiest migration candidates: no coordination with other teams or phases is needed.
- A large number of isolated streams is typical for utility or maintenance jobs (data quality checks, archival processes).

---

## 3. Resolution Scan Summary

**Format:** Pandas DataFrame displayed inline in the Databricks notebook.
**When produced:** After `scan_resolutions()` completes.

### Schema

| Column | Type | Description |
|---|---|---|
| `resolution` | float | Leiden resolution parameter (γ) |
| `n_communities_avg` | float | Mean community count across seeds |
| `n_communities_min` | int | Minimum community count across seeds |
| `n_communities_max` | int | Maximum community count across seeds |
| `largest_comm_share_avg` | float | Mean fraction of nodes in the largest community |
| `small_comms_lt5_avg` | float | Mean count of communities with fewer than 5 streams |
| `quality_avg` | float | Mean RBConfiguration objective value |
| `stability_ari` | float | Average pairwise Adjusted Rand Index across seeds |

### How it is produced

For each resolution value in `[0.4, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.2, 2.4, 2.6]`, Leiden is run 15 times with different random seeds. Metrics are aggregated across seeds into one summary row.

### How to interpret each column

**`n_communities_avg / min / max`**

The typical number of migration phases (communities) at this resolution.
- `min == max`: every seed found the same number of communities — high structural confidence.
- `min < max`: multiple competing partition structures exist at this resolution.

**`largest_comm_share_avg`**

The fraction of all streams assigned to the single largest community. Values above 0.5 indicate one dominant cluster that is likely too large for a single migration phase.

**`small_comms_lt5_avg`**

Communities with fewer than 5 streams are typically too small to be actionable phases. A rising value at higher resolutions indicates over-fragmentation.

**`quality_avg`**

The RBConfiguration modularity objective. Comparable only within the same resolution — a higher value at γ = 1.2 does not mean the partition is better than a lower value at γ = 0.8.

**`stability_ari` (most important signal)**

Values close to 1.0 indicate that the community structure at this resolution is robust — every random seed converges to essentially the same partition, meaning the detected communities are a genuine feature of the graph rather than an artifact of randomness.

| ARI range | Interpretation |
|---|---|
| 1.0 | Perfectly stable — every seed gives identical partitions |
| 0.95–0.99 | Highly stable — minor boundary differences between seeds |
| 0.85–0.94 | Moderately stable — several competing but similar solutions |
| < 0.85 | Unstable — resolution is too fine for this graph's structure |

### Resolution regimes

| γ range | Communities | ARI | Practical meaning |
|---|---|---|---|
| 0.2–0.4 | 3–5 | ≈ 1.0 | Very coarse; each community has hundreds of streams — not actionable |
| 0.5–1.0 | 5–9 | 1.0 | Perfectly stable; good for understanding high-level structure |
| 1.2–1.8 | 10–12 | > 0.92 | **Best for practical migration planning** — manageable phase sizes |
| 2.2–3.0 | 24–37 | 0.89–0.97 | Fine-grained; many tiny communities increase coordination overhead |

**Recommended starting point:** γ = 1.8 — typically 10–12 communities, ARI > 0.92, and manageable community sizes.

---

## 4. Per-Community Analysis Reports

**Path:** `<output_path>/leiden_community_plots_gamma_<γ>/community_<N>/community_<N>_analysis.txt`
**When produced:** After `plot_communities_with_analysis_safe()` is called for each chosen resolution.

One text file is generated per community per resolution. These are the primary planning documents for migration teams.

### Structure of `community_<N>_analysis.txt`

```
Community N Analysis (Resolution γ=1.8)
================================================================================

1. STREAMS IN COMMUNITY (42 total streams, 14.52% of all streams):
   1a. BI REPORTS/STREAMS (7 streams with 'json' in name, 11.11% of all BI reports):
       1. reporting_orders_json (complexity: 6.00)
       ...
   1b. ETL STREAMS (35 streams without 'json' in name, 15.22% of all ETL streams):
       1. ORDERS_STREAM (complexity: 12.00)
       ...

2. COMPLEXITY ANALYSIS:
   Total Community Complexity: 284.00
   Percentage of Total Complexity: 13.87%
     - BI Reports Complexity: 42.00
     - ETL Streams Complexity: 242.00

3. TABLES - SRC OF STREAMS INSIDE, TGT OF STREAMS OUTSIDE:
   (Tables produced outside, consumed inside → SYNC REQUIREMENTS)
   Total: 18 unique tables, 1247.50 GB
   ...

4. TABLES - TGT OF STREAMS INSIDE, SRC OF STREAMS OUTSIDE:
   (Tables produced inside, consumed outside)
   Total: 12 unique tables
   Total Size: 843.20 GB
   ...

5. AGGREGATED STREAM CONNECTIONS:
   5a. Outgoing Stream Connections (Inside → Outside): 6 connections
   5b. Incoming Stream Connections (Outside → Inside): 4 connections

6. SUMMARY:
   ...
```

### How to interpret each section

#### Section 1 — Streams in community

Lists every stream assigned to this community, split by type:
- **BI REPORTS/STREAMS** — streams whose names contain `json` (typically Tableau or similar BI endpoints). These are read-only consumers; migrating them requires all their source tables to be available.
- **ETL STREAMS** — all other streams. These both read and write tables.

The percentages show how much of the total estate is covered by this community.

#### Section 2 — Complexity analysis

The complexity score (weighted sum of object counts: low×1 + medium×2 + complex×4 + very_complex×7) gives a proxy for development effort required to migrate each stream. Use this to:
- Estimate relative migration effort across communities.
- Balance complexity across phases when you have flexibility in community ordering.
- Identify communities that need larger migration teams or longer timelines.

#### Section 3 — Incoming tables (sync requirements)

These are the **critical external dependencies** for this community. Before or during migration of this community, these tables must be synchronized from an outside-community source.

For each table:
```
Table: DW.FACT_ORDERS (142.50 GB)
  - Written by (outside): ORDERS_STREAM
  - Read by (inside):     REPORTING_ORDERS_JSON
```

- **For BI Reports:** Tables needed only by BI consumers — relatively easy to sync (no write-back required).
- **For ETL Streams:** Tables needed as inputs for ETL processing — timing is critical; sync must complete before the ETL runs.

**Action:** Work with the teams owning the outside streams to establish a data synchronization plan (replication, dual-write, or CDC) for these tables.

#### Section 4 — Outgoing tables

Tables produced by this community that are consumed by streams in other communities. These define the **downstream dependencies** — other communities must wait for these tables to be produced before they can run.

**Action:** Communicate production SLAs for these tables to the teams managing dependent communities.

#### Section 5 — Aggregated stream connections

High-level view of how many streams in other communities this community exchanges data with, and in which direction. Useful for identifying communities that are well-isolated (few connections) versus central (many connections).

#### Section 6 — Summary block

A one-page summary combining all key metrics for quick reference and stakeholder communication.

---

## 5. Migration Order Report

**Path:** `<output_path>/migration_order_analysis/migration_order_report_gamma_<γ>.txt`
**When produced:** After `generate_migration_order_analysis()` is called with the optimized ordering.

This is the primary deliverable for migration project management.

### Structure

```
====================================================================================================
MIGRATION ORDER ANALYSIS REPORT
Resolution γ=1.8
Generated at 2025-12-23 12:18:45
====================================================================================================

OPTIMIZED MIGRATION ORDER: [5, 2, 0, 7, 3, 1, 8, 4, 6, 9]

Total Communities: 10
Total Streams: 289
  - BI Reports: 63
  - ETL Streams: 226
Total Complexity Score: 2048

====================================================================================================

STEP 1 — Community 5
────────────────────────────────────────────────────────────────────────────────
Streams: 28 total (9.69% of all streams)
  - BI Reports: 5 (7.94% of total BI)
  - ETL Streams: 23 (10.18% of total ETL)
Complexity: 217.00 (10.59% of total)

Tables to sync at this step: 3 unique tables, 47.80 GB
  → Sync requirements (not yet available):
     Table: REFERENCE.COUNTRY_CODES (0.50 GB)
       - Written by: REFERENCE_LOADER (outside)
       - Read by: ORDERS_STREAM (inside)
     ...

Tables produced by this community: 18 tables
Outgoing connections: 4 stream connections to other communities

Cumulative progress after step 1:
  Streams migrated: 28 / 289 (9.69%)
  Cumulative sync size: 47.80 GB
  ...
```

### How to interpret it

**Optimized migration order** — the sequence of community IDs that minimizes total cumulative synchronization cost. Communities listed first produce tables that are needed by later communities, reducing the sync burden progressively.

**Per-step sync requirements** — the list of tables that must be synchronized from external sources at each step. Tables already produced by previously migrated communities are marked as available and do not appear in this list.

**Cumulative metrics** — running totals after each step show how the migration estate is progressively covered:
- `Streams migrated` — progress toward complete migration
- `Cumulative sync size` — total data volume that has been synchronized as external dependencies
- `Cumulative complexity` — proportion of total migration effort completed

**Key insight:** The first community in the optimized order typically has the fewest external dependencies (it does not depend on other communities' outputs). Later communities benefit from earlier communities having already migrated their tables.

---

## 6. Sync Details — `sync_details_gamma_<γ>.csv`

**Path:** `<output_path>/migration_order_analysis/sync_details_gamma_<γ>.csv`
**When produced:** Alongside the migration order report.

Machine-readable complement to the migration order report.

### Schema

| Column | Type | Description |
|---|---|---|
| `resolution` | float | Leiden resolution used |
| `execution_order` | int | Step number in the migration order (1-based) |
| `community_id` | int | Community being migrated at this step |
| `table_name` | string | Table that requires synchronization |
| `size_gb` | float | Table size in gigabytes |
| `stream_name` | string | Stream involved in the dependency |
| `handling_type` | string | `written outside` or `read inside` |

Each table sync requirement produces **two rows** — one for the producing stream (outside) and one for the consuming stream (inside).

### How to use it

- Filter by `handling_type = 'written outside'` to identify the teams/systems that must provide data.
- Filter by `handling_type = 'read inside'` to identify which streams in each community are blocked until sync completes.
- Group by `execution_order` and sum `size_gb` (deduplicated by `table_name`) to calculate the total GB that must be synced before each migration step can begin.
- Use as input to a data synchronization scheduling tool.

---

## 7. Stream Ordering — `stream_ordering_gamma_<γ>.csv`

**Path:** `<output_path>/migration_order_analysis/stream_ordering_gamma_<γ>.csv`
**When produced:** Alongside the migration order report.

Maps every stream to its community and execution order — the master stream-to-phase assignment.

### Schema

| Column | Type | Description |
|---|---|---|
| `resolution` | float | Leiden resolution used |
| `stream_name` | string | ETL stream or BI report name |
| `community_id` | int | Leiden community the stream belongs to |
| `execution_order` | int | Migration phase number (1 = first to migrate) |

### How to use it

- This is the **primary handoff document** for migration teams. Each team takes their list of streams (filtered by `execution_order`) and begins migration work.
- Sort by `execution_order` then `stream_name` to produce a phased migration backlog.
- Join to your project management tool (Jira, Azure DevOps) to auto-create migration tickets per phase.
- Join back to the complexity scores to estimate effort per phase.

Example query (pandas):

```python
import pandas as pd

order = pd.read_csv("stream_ordering_gamma_1.8.csv")
complexity = pd.read_csv("stream_complexity_scores.csv", delimiter=";")
complexity["complexity_score"] = (
    complexity["low"] * 1 + complexity["medium"] * 2 +
    complexity["complex"] * 4 + complexity["very_complex"] * 7
)

phase_plan = (
    order
    .merge(complexity[["stream_name", "complexity_score"]], on="stream_name", how="left")
    .groupby("execution_order")
    .agg(
        streams=("stream_name", "count"),
        total_complexity=("complexity_score", "sum"),
        community_ids=("community_id", lambda x: sorted(x.unique()))
    )
    .reset_index()
)
print(phase_plan)
```

---

## 8. Execution Metadata — Delta Table

**Table:** `<catalog>.<schema>.execution_metadata`
**When produced:** After each run if `append_execution_metadata()` is called.

Logs key parameters and results of every Migration Planner execution for audit and comparison.

### Schema

| Column | Type | Description |
|---|---|---|
| `execution_datetime` | string | Timestamp of the run (`YYYY-MM-DD HH:MM:SS`) |
| `weight_calculation_method` | string | Weight method used: `"Factor based"`, `"Min-max based"`, or `"Log based"` |
| `top_n` | int | Number of top communities used in the ordering optimization |
| `resolution` | float | Leiden resolution parameter |
| `number_of_communities` | int | Number of communities found at this resolution |
| `total_sync_cost_gb` | float | Total sync cost (GB) for the optimized migration order |

One row is inserted per resolution per run.

### How to use it

- Compare `total_sync_cost_gb` across runs to measure the impact of changing input data, weight methods, or resolution parameters.
- Track `number_of_communities` over time to detect structural changes in the dependency graph as the EDW evolves.
- Use `execution_datetime` + `resolution` as a composite key for point-in-time analysis.

Example SQL:

```sql
SELECT
  execution_datetime,
  weight_calculation_method,
  resolution,
  number_of_communities,
  total_sync_cost_gb,
  total_sync_cost_gb - LAG(total_sync_cost_gb) OVER (
    PARTITION BY resolution ORDER BY execution_datetime
  ) AS sync_cost_delta_gb
FROM odp_adw_mvp_n.migration.execution_metadata
WHERE resolution = 1.8
ORDER BY execution_datetime DESC;
```

---

## 9. Graph Visualizations (optional)

**Path:** `<output_path>/leiden_community_plots_gamma_<γ>/community_<N>/community_<N>.png`
**Path:** `<output_path>/leiden_community_plots_gamma_<γ>/leiden_rb_gamma_<γ>.png`
**When produced:** Only when `enable_plotting=True` is set in `plot_communities_with_analysis_safe()` or the overall graph plot cell is un-skipped.

Visualizations are disabled by default (they can be slow to generate for large graphs).

### Overall graph plot

A single PNG showing all streams as nodes, colored by community, with edges weighted by migration sync cost. Node positions are computed using NetworkX spring layout.

- Node color = community assignment
- Edge width/opacity = normalized edge weight (thicker = more shared data)
- Resolution and quality statistics shown in the title

### Per-community subgraph plots

Individual PNG plots for each community, showing only the streams within that community and their internal connections. Useful for detailed structural review by the team responsible for that migration phase.

### How to enable them

Un-comment the overall graph plotting cell, or set `enable_plotting=True` in the `plot_communities_with_analysis_safe()` call. For large graphs (> 200 nodes), set `dpi=200` and `figsize=(30, 24)` to balance quality and generation time.

---

## How to Choose a Resolution

Use the resolution scan summary table to select the resolution for final analysis:

**Step 1 — Filter by stability.**
Keep only resolutions where `stability_ari >= 0.93`. Below this threshold, the community structure is not reliably reproducible.

**Step 2 — Filter by community size.**
Eliminate resolutions where `largest_comm_share_avg > 0.25`. A community containing more than 25% of all streams is typically too large for a practical migration phase.

**Step 3 — Filter by fragmentation.**
Avoid resolutions where `small_comms_lt5_avg > 3`. Many tiny communities increase coordination overhead without providing useful planning granularity.

**Step 4 — Choose the resolution that gives the desired number of phases.**
Among the remaining candidates, select the resolution whose `n_communities_avg` best matches your project's capacity — typically 8–15 phases for a large EDW migration.

**Step 5 — Validate the representative run.**
Review the per-community analysis files for the chosen resolution to confirm that the community assignments make intuitive sense for your domain (e.g., streams that process related business domains should be together).

---

## How to Plan a Migration from the Outputs

1. **Confirm the resolution** — use the stability and community-size heuristics above to select γ.

2. **Review `stream_ordering_gamma_<γ>.csv`** — this is your migration backlog. Each unique `execution_order` is a migration phase.

3. **For each phase, review `community_<N>_analysis.txt`** — identify the sync requirements (Section 3) and arrange data synchronization contracts with upstream teams.

4. **Schedule phases** — phases with zero sync requirements can start immediately. Subsequent phases can begin once their incoming tables are available (produced by a previous phase or synchronized from the source system).

5. **Track progress** — as phases complete, update the `available_tables` set in the ordering model to refine remaining sync cost estimates.

6. **Re-run as needed** — if the dependency graph changes significantly (new streams, deprecated tables, ETL restructuring), re-run Migration Planner and compare the new `execution_metadata` table entries to the previous run to assess structural drift.
