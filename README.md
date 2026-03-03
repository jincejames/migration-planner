# Migration Planner

> **Intelligent EDW migration planning through graph-based community detection**

Migration Planner analyzes the dependency graph of an existing data warehouse ecosystem and clusters ETL streams and BI reports into cohesive migration phases. It uses the [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z) — a state-of-the-art community detection method — to find natural groupings of assets that minimize cross-phase data synchronization costs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build System: Hatch](https://img.shields.io/badge/build-hatch-purple)](https://hatch.pypa.io/)

---

## What it does

1. **Ingests** CSV files describing which ETL streams read from and write to which database tables, plus optional metadata (table sizes, stream complexity scores, out-of-scope lists).
2. **Builds** a weighted undirected graph where nodes are streams/reports and edge weights reflect the volume of shared data.
3. **Detects communities** by running the Leiden algorithm across a configurable resolution grid with multiple random seeds to measure stability.
4. **Analyzes** each community: stream inventory (ETL vs BI), complexity breakdown, incoming/outgoing table dependencies, and sync requirements.
5. **Optimizes** the migration order across communities to minimize total data synchronization cost using brute-force permutation search.
6. **Outputs** structured CSV files, per-community analysis reports, an execution-order report, and optional graph visualizations — all saved to a Databricks Unity Catalog volume.

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [YAML Config File](#yaml-config-file)
  - [CLI Flags](#cli-flags)
  - [Priority Rules](#priority-rules)
- [Input Files](#input-files)
- [Weight Methods](#weight-methods)
- [Resolution Selection](#resolution-selection)
- [Outputs](#outputs)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Capability | Description |
|---|---|
| Dependency graph construction | Builds igraph + NetworkX graphs from stream-table CSV data |
| Configurable edge weights | Factor-based (ceil size/100) or min-max scaled, with optional log transform |
| Community detection | Leiden RBConfiguration with resolution grid scan |
| Stability measurement | Average pairwise Adjusted Rand Index (ARI) across multiple seeds |
| Isolated stream detection | Identifies streams with no cross-stream dependencies |
| Per-community analysis | Stream inventory, complexity, incoming/outgoing table dependencies |
| Migration order optimization | Brute-force permutation search minimizing total sync cost |
| Execution metadata logging | Appends run results to a Unity Catalog Delta table for tracking |
| Graph visualization | Optional matplotlib/NetworkX plots colored by community (high-res PNG) |
| YAML + CLI configuration | Two-layer config with CLI overriding YAML |

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- [Hatch](https://hatch.pypa.io/) build tool
- Access to a **Databricks** workspace with:
  - A Unity Catalog volume containing your input CSV files
  - A cluster with PySpark available (the notebook runs as a Databricks job or interactive notebook)
  - The following Python libraries (installed by the notebook at runtime): `python-igraph`, `leidenalg`, `networkx`, `netgraph`, `adjustText`

### Installation

**1. Clone the repository**

```shell
git clone https://github.com/jincejames/migration-planner.git
cd migration-planner
```

**2. Install Hatch** (macOS)

```shell
brew install hatch
```

On Linux/Windows, see the [Hatch installation guide](https://hatch.pypa.io/latest/install/).

**3. Create the virtual environment and install all development dependencies**

```shell
make dev
```

This runs `hatch env create`, placing the `.venv` directory inside the project folder so that VS Code and PyCharm pick it up automatically.

**4. Verify everything works**

```shell
make test
```

All 223 unit tests should pass.

### Quick Start

**Step 1 — Prepare your config file**

Copy the example config and edit it to point at your Databricks volume:

```shell
cp config.yaml my_config.yaml
```

Edit `my_config.yaml`:

```yaml
# Migration Planner - Community Detection Config
volume_name: /Volumes/my_catalog/my_schema/my_volume/community_detection/
input_dependency_name: etl_stream_table_dependencies.csv
outofscope_stream_file_name: out_of_scope_streams.csv    # optional
table_size_file_name: table_sizes_gb.csv                  # optional
complexity_file_name: stream_complexity_scores.csv        # optional
# weight_method: factor   # "factor" (default) or "scaled"
```

**Step 2 — Upload the notebook to Databricks**

Upload `src/migration_planner/community_detector/leiden.py` to your Databricks workspace as a notebook.

**Step 3 — Run the notebook**

Pass the config path as a widget or CLI argument:

```
--config /Volumes/my_catalog/my_schema/my_volume/community_detection/my_config.yaml
```

Or override individual fields directly:

```
--volume-name /Volumes/my_catalog/my_schema/my_volume/
--input-dependency-name etl_stream_table_dependencies.csv
--weight-method scaled
```

The notebook will:
- Load and validate configuration
- Read all input files from the volume
- Run the preprocessing and graph-building pipeline
- Scan resolutions `[0.4, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.2, 2.4, 2.6]` across 15 seeds
- Save edges, isolated streams, community analyses, and the migration order report to the output path

---

## Configuration

### YAML Config File

```yaml
# Required
volume_name: /Volumes/catalog/schema/volume/community_detection/
input_dependency_name: etl_table_dependencies.csv

# Optional — omit entirely to skip the corresponding feature
outofscope_stream_file_name: out_of_scope_streams.csv
report_dependency_file_name: stream_to_report_mapping.csv
table_size_file_name: table_sizes_gb.csv
complexity_file_name: stream_complexity_scores.csv
static_tables_file_name: static_tables_for_report.csv

# Weight method — defaults to "factor" if omitted
# weight_method: factor   # Options: "factor" or "scaled"
```

### CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | path | — | Path to YAML config file |
| `--volume-name` | string | **required** | Base path for all input/output on the Databricks volume |
| `--input-dependency-name` | string | **required** | CSV filename of stream-table dependencies |
| `--outofscope-stream-file-name` | string | `None` | CSV filename of streams to exclude |
| `--report-dependency-file-name` | string | `None` | CSV filename of report-to-table dependencies |
| `--table-size-file-name` | string | `None` | CSV filename of table sizes in GB |
| `--complexity-file-name` | string | `None` | CSV filename of per-stream complexity scores |
| `--static-tables-file-name` | string | `None` | CSV filename of static (non-migrated) tables |
| `--weight-method` | `factor`\|`scaled` | `factor` | Edge weight calculation method |

### Priority Rules

Configuration layers are merged with the following precedence (highest wins):

```
CLI flags  >  YAML file  >  dataclass defaults
```

Mandatory fields (`volume_name`, `input_dependency_name`) raise a `ValueError` if absent in both layers. Optional fields silently remain `None` when not provided, disabling the corresponding feature.

---

## Input Files

All files are read from `volume_name + <file_name>` on the Unity Catalog volume.

### 1. Stream-Table Dependencies (required)

**Config key:** `input_dependency_name`

Comma-delimited CSV describing which ETL streams read from / write to which tables.

| Column | Type | Description |
|---|---|---|
| `stream_name` | string | ETL stream or BI report name |
| `DB_Table_Name` | string | Fully-qualified database table name |
| `table_type` | string | Role: `Src`, `Src_Trns`, `Tgt`, `Tgt_Trns`, `File` |

```
stream_name,DB_Table_Name,table_type
ORDERS_STREAM,DW.FACT_ORDERS,Tgt
ORDERS_STREAM,STG.STG_ORDERS,Src
SHIPMENT_STREAM,DW.FACT_ORDERS,Src
SHIPMENT_STREAM,DW.FACT_SHIPMENTS,Tgt
```

### 2. Out-of-Scope Streams (optional)

**Config key:** `outofscope_stream_file_name`

CSV listing stream names that should be excluded from analysis (in addition to the automatic exclusion of streams containing ARCHIVE, GDPR, or HOUSEKEEPING).

| Column | Type | Description |
|---|---|---|
| `stream_name` | string | Stream to exclude |

```
stream_name
LEGACY_ARCHIVE_STREAM
TEST_LOAD_STREAM
```

### 3. Table Sizes (optional)

**Config key:** `table_size_file_name`

Comma-delimited CSV with table storage sizes. Used to compute edge weights (factor method) and to report sync cost in GB.

| Column | Type | Description |
|---|---|---|
| `DB_Table_Name` | string | Table name (matched case-insensitively) |
| `SPACE_IN_GB` | float | Table size in gigabytes |

```
DB_Table_Name,SPACE_IN_GB
DW.FACT_ORDERS,142.5
DW.FACT_SHIPMENTS,87.3
```

### 4. Stream Complexity Scores (optional)

**Config key:** `complexity_file_name`

Semicolon-delimited CSV with per-stream complexity breakdown. Used only for analysis and reporting; does not affect graph structure.

| Column | Type | Description |
|---|---|---|
| `stream_name` | string | ETL stream name |
| `low` | int | Count of low-complexity objects |
| `medium` | int | Count of medium-complexity objects |
| `complex` | int | Count of complex objects |
| `very_complex` | int | Count of very-complex objects |

Computed score: `low×1 + medium×2 + complex×4 + very_complex×7`

```
stream_name;low;medium;complex;very_complex
ORDERS_STREAM;5;3;1;0
SHIPMENT_STREAM;2;4;2;1
```

### 5. Report-to-Table Dependencies (optional)

**Config key:** `report_dependency_file_name`

Maps BI report names to the database tables they read from. Reports named after `corona` or `gdpr` are automatically excluded.

| Column | Type | Description |
|---|---|---|
| `report_name` | string | BI report name |
| `table_name` | string | Table the report reads |

### 6. Static Tables (optional)

**Config key:** `static_tables_file_name`

CSV listing tables that are not part of the migration (e.g., reference/lookup tables). Used for filtering during report analysis.

| Column | Type | Description |
|---|---|---|
| `table_name` | string | Table name to treat as static |

---

## Weight Methods

Edge weights reflect the migration effort required to keep a shared table in sync between two communities. Two methods are available:

### Factor-based (default — `"factor"`)

Simple, interpretable, and robust to missing size data.

| Condition | Weight |
|---|---|
| Destination stream contains `"json"` (BI/Tableau endpoint) | 2 |
| All others | `ceil(size_GB / 100)`, minimum 1 |

A 250 GB table gets weight 3; a table with unknown size gets weight 1.

### Min-max scaled (`"scaled"`)

Normalizes all table sizes to a `[0, 1]` range (or any `[scaled_min, scaled_max]`) for smoother, proportional weighting.

Optional parameters (set in code, not exposed via CLI):
- `use_log=True` — apply log₁₊ transform before scaling (reduces skew from very large tables)
- `scaled_min` / `scaled_max` — target range (default `0.0` to `1.0`)
- `tableau_multiplier` — additional boost for JSON-destination streams

**When to choose which:**

| Scenario | Recommended method |
|---|---|
| Many tables lack size data | `factor` |
| Uniform size distribution | `factor` |
| Wide size range (outlier tables) | `scaled` with `use_log=True` |
| Precise proportional weighting needed | `scaled` |

---

## Resolution Selection

The `scan_resolutions` function tests every resolution (γ) value across 15 random seeds and produces a summary table to guide resolution selection:

| Column | Meaning |
|---|---|
| `n_communities_avg/min/max` | Range of community counts across seeds |
| `largest_comm_share_avg` | Fraction of nodes in the largest community |
| `small_comms_lt5_avg` | Average number of communities with fewer than 5 streams |
| `quality_avg` | Average RBConfiguration objective value |
| `stability_ari` | Average pairwise Adjusted Rand Index — `1.0` = identical partitions |

**Typical resolution regimes:**

| γ range | Communities | ARI | Practical use |
|---|---|---|---|
| 0.2–0.4 | 3–5 | ≈ 1.0 | Highly stable but too coarse for most migrations |
| 0.5–1.0 | 5–9 | 1.0 | Most stable; good starting point |
| 1.2–1.8 | 10–12 | > 0.92 | Best balance of granularity and stability for practical migration |
| 2.2–3.0 | 24–37 | 0.89–0.97 | Fine-grained but increasingly fragmented |

A resolution of **γ = 1.8** is used as the default for the community analysis and ordering steps, balancing manageable community sizes with acceptable stability.

---

## Outputs

All outputs are written to `<volume_name>/community_detection_output_latest/<timestamp>/`.

| Artifact | File | Description |
|---|---|---|
| Edge list | `edges.csv` | Weighted undirected stream-stream edges |
| Isolated streams | `isolated_streams.csv` | Streams with no cross-stream dependencies |
| Resolution scan | (displayed in notebook) | Summary DataFrame — one row per γ value |
| Community analysis | `leiden_community_plots_gamma_<γ>/community_<N>/community_<N>_analysis.txt` | Per-community stream inventory, complexity, and table dependency report |
| Migration order report | `migration_order_analysis/migration_order_report_gamma_<γ>.txt` | Step-by-step migration order with cumulative sync costs |
| Sync details | `migration_order_analysis/sync_details_gamma_<γ>.csv` | Table-level sync requirements per migration step |
| Stream ordering | `migration_order_analysis/stream_ordering_gamma_<γ>.csv` | Stream → community → execution order mapping |
| Execution metadata | Delta table `<catalog>.<schema>.execution_metadata` | Run history with weight method, resolution, and total sync cost |

For full details on interpreting each artifact, see [`documentation/output-artifacts.md`](documentation/output-artifacts.md).

---

## Development

### Prerequisites

- [Hatch](https://hatch.pypa.io/) — `brew install hatch`
- Python 3.10+

### Common commands

```shell
make dev        # Create .venv and install all dev dependencies
make fmt        # Auto-format: black, ruff --fix, mypy, pylint
make lint       # Verify formatting (no modifications)
make test       # Run all 223 unit tests (parallel, 30 s timeout)
make coverage   # Generate HTML coverage report and open in browser
make clean      # Remove .venv, build artifacts, and caches
```

### Project structure

```
migration-planner/
├── src/migration_planner/
│   ├── utils/
│   │   └── config.py                   # PlannerConfig dataclass + load_config()
│   ├── dependency_extractors/
│   │   └── loaders.py                  # Six CSV loader functions
│   └── community_detector/
│       ├── preprocessing.py            # Full preprocessing pipeline
│       ├── weights.py                  # Factor & scaled weight methods
│       ├── graph_builder.py            # igraph + NetworkX graph builders
│       ├── algorithm.py                # Leiden RB + stability (ARI) + resolution scan
│       └── leiden.py                   # Databricks orchestration notebook
├── tests/                              # 223 unit tests (pytest)
├── documentation/
│   ├── architecture-and-data-flow.md   # Module architecture and pipeline stages
│   └── output-artifacts.md             # All output files with interpretation guide
├── config.yaml                         # Example configuration
├── pyproject.toml                      # Build, lint, and test configuration
├── Makefile                            # Developer shortcuts
└── CONTRIBUTING.md                     # Contribution workflow
```

For a detailed walkthrough of the architecture and data flow, see [`documentation/architecture-and-data-flow.md`](documentation/architecture-and-data-flow.md).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow.

In brief:

1. Fork and clone the repository
2. `make dev` to set up the environment
3. Create a feature branch
4. `make fmt && make lint && make test` before committing
5. Open a pull request with a meaningful title

---

## License

[MIT](LICENSE)
