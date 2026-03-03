# Architecture and Data Flow

This document describes the module architecture of Migration Planner and traces how data moves through every stage of the pipeline — from raw CSV files on a Databricks volume to the weighted graph that drives community detection.

---

## Table of Contents

- [System Context](#system-context)
- [Module Map](#module-map)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 0 — Configuration](#stage-0--configuration)
  - [Stage 1 — Data Loading](#stage-1--data-loading)
  - [Stage 2 — Stream Filtering](#stage-2--stream-filtering)
  - [Stage 3 — TGT-as-SRC Normalization](#stage-3--tgt-as-src-normalization)
  - [Stage 4 — Cross-Stream Dependency Formation](#stage-4--cross-stream-dependency-formation)
  - [Stage 5 — Edge Weight Calculation](#stage-5--edge-weight-calculation)
  - [Stage 6 — Edge Deduplication and Aggregation](#stage-6--edge-deduplication-and-aggregation)
  - [Stage 7 — Bidirectional Edge Merging](#stage-7--bidirectional-edge-merging)
  - [Stage 8 — Graph Construction](#stage-8--graph-construction)
  - [Stage 9 — Resolution Scan and Stability Analysis](#stage-9--resolution-scan-and-stability-analysis)
  - [Stage 10 — Community Analysis](#stage-10--community-analysis)
  - [Stage 11 — Community Ordering Optimization](#stage-11--community-ordering-optimization)
- [Key Design Decisions](#key-design-decisions)
- [Dependency Graph Between Modules](#dependency-graph-between-modules)

---

## System Context

Migration Planner runs as a **Databricks notebook** (`leiden.py`) on an Apache Spark cluster. It requires:

- **PySpark** — all preprocessing and weight calculations operate as distributed Spark DataFrame transformations
- **igraph** + **leidenalg** — community detection on the collected, in-memory graph
- **NetworkX** — auxiliary graph for visualization and subgraph analysis
- **pandas** — edge list bridge between Spark and igraph
- **scikit-learn** — Adjusted Rand Index for stability measurement
- **Unity Catalog volume** — all input CSV files are read from and all outputs are written to a volume path

```
┌──────────────────────────────────────────────────────────┐
│                    Databricks Cluster                    │
│                                                          │
│  ┌─────────────┐    ┌──────────────────┐                 │
│  │  PySpark    │───▶│  migration_      │                 │
│  │  (driver + │    │  planner package │                 │
│  │   workers) │    │                  │                 │
│  └─────────────┘    │  ├─ config.py   │                 │
│                     │  ├─ loaders.py  │                 │
│  ┌─────────────┐    │  ├─ preproc.py  │                 │
│  │  igraph /   │◀───│  ├─ weights.py  │                 │
│  │  leidenalg  │    │  ├─ graph_b.py  │                 │
│  └─────────────┘    │  ├─ algorithm.py│                 │
│                     │  └─ leiden.py   │                 │
│  ┌─────────────┐    └──────────────────┘                 │
│  │  Unity Cat. │◀──── reads CSVs / writes outputs        │
│  │  Volume     │                                          │
│  └─────────────┘                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Module Map

```
src/migration_planner/
│
├── utils/
│   └── config.py               PlannerConfig dataclass
│                               load_config() — YAML + CLI arg parsing
│
├── dependency_extractors/
│   └── loaders.py              6 CSV reader functions
│                               COMPLEXITY_WEIGHTS constant
│
└── community_detector/
    ├── preprocessing.py        filter_admin_streams()
    │                           treat_tgt_as_src()
    │                           form_stream_stream_dependencies()
    │                           merge_bidirectional_edges()
    │                           preprocess_stream_dependencies()  ← pipeline entry
    │
    ├── weights.py              calculate_table_weights()  ← dispatcher
    │                           _calculate_factor_weights()
    │                           _calculate_scaled_weights()
    │                           deduplicate_table_weights()
    │                           aggregate_edge_weights()
    │
    ├── graph_builder.py        find_isolated_streams()
    │                           build_igraph()
    │                           build_networkx_graph()
    │
    ├── algorithm.py            run_leiden_rb()
    │                           stability_ari()
    │                           scan_resolutions()
    │
    └── leiden.py               Databricks notebook orchestrator
                                Community analysis functions
                                Migration ordering optimization
                                Execution metadata logging
```

---

## Pipeline Stages

### Stage 0 — Configuration

**Module:** `utils/config.py`
**Entry point:** `load_config(args=None) → PlannerConfig`

Configuration is resolved in two layers; CLI always wins over YAML:

```
CLI flags  ──┐
             ├──▶  PlannerConfig  ──▶  resolved paths
YAML file  ──┘
```

`PlannerConfig` computes derived path properties at access time:

| Property | Formula |
|---|---|
| `dependency_input_path` | `volume_name + input_dependency_name` |
| `output_path` | `volume_name + "community_detection_output_latest/" + "<ddmmyyyy_HH>/"` |
| `latest_path` | `volume_name + "community_detection_output_latest/"` |
| `outofscope_stream_path` | `volume_name + outofscope_stream_file_name` (or `None`) |
| `report_dependency_path` | `volume_name + report_dependency_file_name` (or `None`) |
| `table_size_path` | `volume_name + table_size_file_name` (or `None`) |
| `complexity_path` | `volume_name + complexity_file_name` (or `None`) |
| `static_tables_path` | `volume_name + static_tables_file_name` (or `None`) |

Mandatory fields (`volume_name`, `input_dependency_name`) raise `ValueError` if absent.

---

### Stage 1 — Data Loading

**Module:** `dependency_extractors/loaders.py`

Six loader functions read CSV files from the Unity Catalog volume using the Spark `DataFrameReader`. All paths come from the resolved `PlannerConfig` properties.

```
volume/
 ├─ etl_dependencies.csv   ──▶  load_stream_table_dependencies()  ──▶  dependency_df_full  (Spark DF)
 ├─ out_of_scope.csv        ──▶  load_outofscope_streams()          ──▶  oos_list            (list[str])
 ├─ complexity.csv          ──▶  load_complexity_scores()           ──▶  complexity_df       (Spark DF)
 ├─ report_deps.csv         ──▶  load_report_dependencies()         ──▶  report_df           (Spark DF)
 ├─ table_sizes.csv         ──▶  load_table_sizes()                 ──▶  table_size_df       (Spark DF)
 └─ static_tables.csv       ──▶  load_static_tables()               ──▶  static_tables_set   (set[str])
```

**Complexity score formula** (applied at load time):

```
complexity_score = (low × 1) + (medium × 2) + (complex × 4) + (very_complex × 7)
```

**Report dependency normalization** (applied at load time):
- Renames `report_name → stream_name`, uppercases `table_name`
- Sets `table_type = "Src"` for all rows
- Filters out rows where `stream_name` contains `corona` or `gdpr`

---

### Stage 2 — Stream Filtering

**Function:** `preprocessing.filter_admin_streams(dependency_df_full, outofscope_list)`

Removes streams whose names match any of the following (case-insensitive):
- Contains `ARCHIVE`
- Contains `GDPR`
- Contains `HOUSEKEEPING`
- Is in the caller-supplied `outofscope_list`

```
dependency_df_full  ──▶  [filter rows]  ──▶  filtered_df
```

**Data shape after this stage:**

Same columns as input (`stream_name`, `DB_Table_Name`, `table_type`), but with administrative and out-of-scope streams removed.

---

### Stage 3 — TGT-as-SRC Normalization

**Function:** `preprocessing.treat_tgt_as_src(filtered_df)`

A stream that writes a table (type `Tgt` or `Tgt_Trns`) implicitly also reads it. This stage duplicates every TGT row as a SRC row so that the self-join in Stage 4 can find the correct directional dependency.

```
filtered_df
  ├─ TGT rows  ──▶  copy with table_type: Tgt→Src, Tgt_Trns→Src_Trns
  └─ union + distinct
```

**Why this matters:**
If stream A writes table T and stream B reads table T, there is a dependency A→B. Without this step the self-join would not find that A depends on T as a source.

---

### Stage 4 — Cross-Stream Dependency Formation

**Function:** `preprocessing.form_stream_stream_dependencies(dependency_df, report_df, size_df)`

Performs a self-join on `table_name` to find pairs of streams that share a table:

```
dependency_df  (df1)
    │
    └── self-join on table_name ──▶  cross_df  (df1.stream ≠ df2.stream)
                                          │
                 optional left-join ◀─────┘
                   with table_size_df
                          │
                     filter: df1.table_type ∈ {TGT, TGT_TRNS, FILE}
                           AND df2.table_type ∈ {SRC, SRC_TRNS, FILE}
                          │
                          ▼
                stream_stream_deps_df
                  columns: from, to, table, size
```

A row `(from=A, to=B, table=T, size=...)` means: stream A writes table T, stream B reads table T → A depends on B during migration.

If `table_size_df` is absent, the `size` column is `null`.

---

### Stage 5 — Edge Weight Calculation

**Module:** `community_detector/weights.py`
**Dispatcher:** `calculate_table_weights(df, method, ...)`

Adds a `table_weight` column to the dependency DataFrame. Two methods are available:

#### Factor-based (`method="factor"`)

```
table_weight =
  if "json" in lower(to):  → 2
  else:                     → max(1, ceil(size / 100))
```

Tables with `null` size receive weight 1 (minimum). JSON destinations (BI/Tableau endpoints) are assigned a fixed weight of 2 to avoid being collapsed with low-weight pure-ETL edges.

#### Min-max scaled (`method="scaled"`)

```
1. raw = size (optionally apply log₁₊ if use_log=True)
2. mm01 = (raw - min) / (max - min)
3. table_weight = scaled_min + (scaled_max - scaled_min) × mm01
4. if "json" in lower(to): table_weight × tableau_multiplier
5. table_weight = max(table_weight, scaled_min)
```

Min and max are computed in a distributed Spark aggregation, then broadcast back to apply the linear transform.

---

### Stage 6 — Edge Deduplication and Aggregation

**Functions:** `weights.deduplicate_table_weights()`, `weights.aggregate_edge_weights()`

```
table_weight_df
  │
  ├─ dropDuplicates(["from", "to", "table"])   ← remove duplicate rows per shared table
  │  rename table_weight → weight
  │
  └─ groupBy("from", "to").agg(sum("weight"))  ← sum per-table weights into one edge weight
         │
         ▼
  weighted_df
    columns: from, to, weight
```

One row per directed stream-stream edge, with weight equal to the total migration effort of all shared tables.

---

### Stage 7 — Bidirectional Edge Merging

**Function:** `preprocessing.merge_bidirectional_edges(weighted_df)`

The graph must be undirected. If both A→B and B→A exist, they are merged into a single edge by summing their weights. Unidirectional edges are kept as-is.

```
weighted_df (directed)
  │
  ├── self-join: (forward.from == backward.to AND forward.to == backward.from)
  │     → bidir_df  (combined weight = forward.weight + backward.weight)
  │       dropDuplicates on sort_array([from, to])
  │
  └── left_anti join (no reverse edge found)
        → unidir_df  (kept as-is)
  │
  └── union(bidir_df, unidir_df)
         │
         ▼
  merged_df
    columns: streamA, streamB, weight
```

`streamA` is chosen as the endpoint with the higher individual weight (for determinism).

---

### Stage 8 — Graph Construction

**Module:** `community_detector/graph_builder.py`

Two parallel graph objects are built from the merged edge list, collected to pandas first:

```
merged_df (Spark)
  │
  └── .toPandas()  ──▶  edges_df (pandas)
            │
            ├── build_igraph(edges_df, isolated_streams)   ──▶  g  (igraph.Graph)
            │    TupleList from (streamA, streamB, weight)
            │    + add_vertices(isolated_streams)
            │
            └── build_networkx_graph(edges_df, isolated_streams)  ──▶  G  (nx.Graph)
                 from_pandas_edgelist + add_nodes_from
```

**Isolated stream detection** (`find_isolated_streams`):

```
all_streams   = dependency_df.select("stream_name").distinct().collect()
in_edges      = {streamA} ∪ {streamB} from merged_df
isolated      = all_streams − in_edges
```

Isolated streams (no inter-stream dependencies) are added as disconnected nodes to both graphs. They form their own singleton community.

---

### Stage 9 — Resolution Scan and Stability Analysis

**Module:** `community_detector/algorithm.py`
**Entry point:** `scan_resolutions(g, resolutions, seeds, plot_seed)`

For each resolution γ × each seed, `run_leiden_rb` is called:

```
for resolution in [0.4, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.2, 2.4, 2.6]:
  for seed in [1, 3, 7, 43, 99, 123, 11, 28, 37, 45, 672, 42, 10, 100, 178]:
    run_leiden_rb(g, resolution, seed)
      → {resolution, seed, n_communities, largest_comm_share,
         small_comms_lt5, quality, membership}

  stability_ari(all memberships for this resolution)
  aggregate → one summary row
```

**run_leiden_rb internals:**

```
leidenalg.find_partition(
  g,
  RBConfigurationVertexPartition,
  weights="weight",
  resolution_parameter=γ,
  n_iterations=100,
  seed=seed
)
→ membership array (int, length = n_nodes)
→ quality = partition.quality()
```

**stability_ari:**

Computes the average of all C(N,2) pairwise Adjusted Rand Index scores across N membership arrays for a given resolution. ARI = 1.0 means every seed produced identical partitions.

```
ARI(m1, m2) = adjusted_rand_score(m1, m2)   (sklearn)
stability   = mean(ARI for all pairs)
```

The output `summary` DataFrame has one row per resolution; `rep_by_res` maps each resolution to the representative run (seed = `plot_seed`, or first run if not in seeds list).

---

### Stage 10 — Community Analysis

**Function:** `leiden.py / plot_communities_with_analysis_safe()`

For each chosen resolution and each community:

```
leiden_df (stream → community_id)
  │
  ├── Separate BI reports (stream contains "json") vs ETL streams
  ├── Calculate complexity scores from complexity_scores_df
  │
  ├── Incoming tables:  stream_table_pdf WHERE (from ∉ community AND to ∈ community)
  │   → tables produced outside, consumed inside → SYNC REQUIREMENTS
  │   → classify by whether consuming stream is BI or ETL
  │
  ├── Outgoing tables:  stream_table_pdf WHERE (from ∈ community AND to ∉ community)
  │   → tables produced inside, consumed outside
  │
  └── Write community_<N>_analysis.txt per community
```

The analysis text file contains six sections:
1. Streams in community (BI vs ETL breakdown with complexity)
2. Complexity analysis (totals and percentages of global)
3. Incoming tables — sync requirements (by BI / ETL, with producer/consumer lists)
4. Outgoing tables (consumed by other communities)
5. Aggregated stream connections (in/out stream-level edges)
6. Summary block

---

### Stage 11 — Community Ordering Optimization

**Class:** `leiden.py / CommunityOrderOptimizer`

Finds the migration sequence that minimizes total data synchronization cost. The search space grows factorially with the number of communities, so it is split into two parts:

```
All communities
  │
  ├── Top-N heaviest (by total incoming table weight)
  │    └── brute_force() → optimal ordering of top-N communities
  │
  └── Remaining communities
       └── brute_force() → optimal ordering of rest
  │
  └── FinalOrder = concat(rest_order, top_order)
```

**Sync cost definition:**

```
For ordering [C1, C2, ..., Cn]:
  available_tables = pre_available  (tables from previously migrated data)

  for Ci in ordering:
    to_sync = incoming_tables(Ci) − available_tables
    step_cost = Σ weight(t) for t in to_sync
    available_tables += produced_tables(Ci)
    total_cost += step_cost
```

The cost naturally penalizes both large numbers of tables and large table sizes. An ordering that places a community later (when upstream tables are already available) incurs lower sync cost.

---

## Key Design Decisions

### Two DataFrames returned from preprocessing

`preprocess_stream_dependencies()` returns a tuple `(dependency_df, merged_dependency_df)`:

- `dependency_df` — filtered and normalized stream-table dependency table. Needed by `find_isolated_streams()` to enumerate all streams, including those with no cross-stream dependencies.
- `merged_dependency_df` — weighted, undirected edge list. Used directly for graph construction.

Returning both avoids a redundant re-computation of the filtered dependency table after graph building.

### PySpark only in preprocessing and weights

The Spark DataFrame API is used for all preprocessing and weight calculation steps (Stages 2–7) to allow distributed execution on large dependency tables. The graph itself is collected into memory as a pandas DataFrame and passed to igraph, which runs on the driver node.

### Plain class for test stubs, not MagicMock

`tests/conftest.py` defines `_SparkColumn` and `_SparkFunctions` as plain Python classes rather than `MagicMock` subclasses. In Python 3.10, `MagicMock._get_child_mock` always returns plain `MagicMock` instances for comparison operators (`>`, `<`, `&`, `~`), causing `TypeError: NotImplemented`. Plain classes with explicit operator implementations avoid this entirely.

### Weight method chosen at config time

The weight method (`"factor"` or `"scaled"`) is stored in `PlannerConfig.weight_method` and passed through the `preprocess_stream_dependencies()` pipeline to `calculate_table_weights()`. This means the choice affects the entire edge weight computation, not just individual steps.

### Output directory rotation

Every run rotates the output directory:
1. Existing folders under `latest_path/` are moved up to `volume_name/` (archival).
2. A new timestamped folder `community_detection_output_<ddmmyyyy_HH>/` is created under `latest_path/`.

This ensures the latest results are always at a predictable path while preserving historical runs.

---

## Dependency Graph Between Modules

```
leiden.py (orchestrator)
  │
  ├── utils/config.py
  │     └── (no internal imports)
  │
  ├── dependency_extractors/loaders.py
  │     └── (no internal imports)
  │
  └── community_detector/
        ├── preprocessing.py
        │     └── imports: weights.py
        │
        ├── weights.py
        │     └── (no internal imports from this package)
        │
        ├── graph_builder.py
        │     └── (no internal imports from this package)
        │
        └── algorithm.py
              └── (no internal imports from this package)
```

All modules depend only downward; there are no circular imports. `preprocessing.py` is the only module that imports from a sibling module (`weights.py`).
