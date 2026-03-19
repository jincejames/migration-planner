"""
Cutover readiness analysis for migration planning.

Provides two analysis modes:

* **Recursive upstream**: a stream is cutover-ready when ALL its transitive
  upstream dependencies are migrated (downstream state does not block cutover).
* **Immediate dependency**: a stream is cutover-ready when all its immediate
  upstream neighbours are migrated, subject to a sync-size threshold and a
  multi-producer safety rule.
"""
from __future__ import annotations

import builtins
import os
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from migration_planner.planner_core.analysis import split_bi_etl

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_all_recursive_upstreams(
    stream: str, upstream_map: dict[str, set]
) -> set[str]:
    """BFS to find all transitive upstream dependencies of *stream*.

    Returns the set of upstream stream names, **excluding** *stream* itself.
    """
    visited: set[str] = set()
    queue = [stream]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for up in upstream_map.get(current, set()):
            if up not in visited:
                queue.append(up)
    visited.discard(stream)
    return visited


def compute_stream_sync_tables(
    stream: str,
    migrated_set: set[str],
    upstream_nbrs: dict[str, set],
    edge_tbl_lookup: dict[tuple, list],
) -> dict[str, float]:
    """BFS upstream within migrated streams from *stream*.

    At each unmigrated boundary, collect the SRC tables that need syncing.

    Returns
    -------
    dict[str, float]
        ``{table_name: size_gb}``
    """
    sync: dict[str, float] = {}
    visited: set[str] = set()
    queue = [stream]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for up in upstream_nbrs.get(current, set()):
            if up in migrated_set:
                queue.append(up)
            else:
                for tbl, sz in edge_tbl_lookup.get((up, current), []):
                    if tbl not in sync or sz > sync[tbl]:
                        sync[tbl] = sz
    return sync


def has_multi_unmigrated_producer(
    sync_tables: set[str] | dict,
    migrated_set: set[str],
    tbl_producers: dict[str, set],
) -> tuple[bool, list]:
    """Check whether any sync table has more than one unmigrated producer.

    Returns
    -------
    tuple[bool, list]
        ``(has_problem, [(table, unmigrated_producers_set), ...])``
    """
    problematic = []
    for tbl in sync_tables:
        unmigrated = tbl_producers.get(tbl, set()) - migrated_set
        if len(unmigrated) > 1:
            problematic.append((tbl, unmigrated))
    return len(problematic) > 0, problematic


# ---------------------------------------------------------------------------
# Recursive upstream cutover analysis
# ---------------------------------------------------------------------------


def generate_recursive_cutover_analysis(
    leiden_df: pd.DataFrame,
    stream_stream_dependency_df: "DataFrame",
    dependency_df: "DataFrame",
    timeline_df: pd.DataFrame,
    stream_produces: dict[str, set],
    missing_static_tables: set[str],
    bi_stream_required_tables: dict[str, set],
    resolution: float,
    start_date: datetime,
    output_path: str,
) -> tuple[set, set, list, dict]:
    """Recursive upstream cutover analysis.

    A stream is cutover-ready when **all** its transitive upstream dependencies
    are migrated.  Downstream state does not block cutover.

    Returns
    -------
    tuple
        ``(cutover_ready_etl, cutover_ready_bi, stage_summaries,
        recursive_upstream)``
    """
    from pyspark.sql.functions import col, lower, upper  # noqa: E402

    outdir = os.path.join(output_path, "migration_order_analysis")
    os.makedirs(outdir, exist_ok=True)

    all_streams_set = set(leiden_df["stream"].tolist())
    bi_streams_all, etl_streams_all_list = split_bi_etl(all_streams_set)
    bi_streams_all = set(bi_streams_all)
    etl_streams_all = set(etl_streams_all_list)

    # --- Build directed upstream map ---
    deps_pd = stream_stream_dependency_df.select("from", "to").distinct().toPandas()
    etl_deps = deps_pd[
        ~deps_pd["from"].str.lower().str.contains("json")
        & ~deps_pd["to"].str.lower().str.contains("json")
    ].copy()

    recursive_upstream: dict[str, set] = defaultdict(set)
    for _, row in etl_deps.iterrows():
        src, tgt = row["from"], row["to"]
        if src in etl_streams_all and tgt in etl_streams_all:
            recursive_upstream[tgt].add(src)

    streams_with_upstream = builtins.sum(1 for s in etl_streams_all if recursive_upstream[s])
    streams_no_upstream = len(etl_streams_all) - streams_with_upstream

    print(f"Total streams: {len(all_streams_set)}")
    print(f"ETL streams: {len(etl_streams_all)}")
    print(f"BI/JSON streams: {len(bi_streams_all)}")
    print(f"ETL streams with upstream deps: {streams_with_upstream}")
    print(f"ETL source streams (no upstream): {streams_no_upstream}")

    # --- Get BI stream table requirements ---
    bi_src_df = dependency_df.filter(
        (lower(col("stream_name")).contains("json"))
        & (upper(col("table_type")).isin(["SRC", "SRC_TRNS"]))
    ).select(
        col("stream_name"),
        upper(col("DB_Table_Name")).alias("table_name"),
    ).distinct().toPandas()

    bi_req_tables = bi_src_df.groupby(
        "stream_name", group_keys=False
    )["table_name"].apply(set).to_dict()

    stream_to_community = leiden_df.set_index("stream")["community"].to_dict()

    # --- Process stages ---
    migrated = set()
    cutover_ready_etl: set[str] = set()
    cutover_ready_bi: set[str] = set()
    bi_available = missing_static_tables.copy()

    cutover_rows: list[dict] = []
    stage_summaries: list[dict] = []

    for _, stage_row in timeline_df.iterrows():
        stage_num = stage_row["Stage"]
        stage_end = stage_row["End_Date"]
        stage_start = stage_row["Start_Date"]
        stage_comms = stage_row["Original_Community_IDs"]
        stage_label = stage_row["Community_ID"]

        for comm_id in stage_comms:
            cs = set(leiden_df[leiden_df["community"] == comm_id]["stream"].tolist())
            migrated.update(cs)
            for s in cs:
                if s in stream_produces:
                    bi_available.update(stream_produces[s])

        newly_etl: list[str] = []
        for stream in sorted((etl_streams_all & migrated) - cutover_ready_etl):
            if get_all_recursive_upstreams(stream, recursive_upstream).issubset(migrated):
                newly_etl.append(stream)
                cutover_ready_etl.add(stream)

        newly_bi: list[str] = []
        for bs in sorted(bi_streams_all - cutover_ready_bi):
            req = bi_req_tables.get(bs, set())
            if not req:
                if bs in migrated:
                    newly_bi.append(bs)
                    cutover_ready_bi.add(bs)
            elif req.issubset(bi_available):
                newly_bi.append(bs)
                cutover_ready_bi.add(bs)

        for stream in newly_etl:
            up = get_all_recursive_upstreams(stream, recursive_upstream)
            cutover_rows.append({
                "stage": stage_num, "end_date": stage_end,
                "community_id": int(stream_to_community.get(stream, -1)),
                "stream_name": stream, "stream_type": "ETL",
                "recursive_upstream_count": len(up),
            })
        for stream in newly_bi:
            cutover_rows.append({
                "stage": stage_num, "end_date": stage_end,
                "community_id": int(stream_to_community.get(stream, -1)),
                "stream_name": stream, "stream_type": "BI",
                "recursive_upstream_count": -1,
            })

        total_cutover = len(cutover_ready_etl) + len(cutover_ready_bi)
        total_all = len(all_streams_set)
        stage_summaries.append({
            "stage": stage_num, "community_label": stage_label,
            "start_date": stage_start, "end_date": stage_end,
            "newly_cutover_etl": len(newly_etl), "newly_cutover_bi": len(newly_bi),
            "newly_cutover_etl_streams": newly_etl,
            "newly_cutover_bi_streams": newly_bi,
            "cumulative_cutover_etl": len(cutover_ready_etl),
            "cumulative_cutover_bi": len(cutover_ready_bi),
            "cumulative_cutover_total": total_cutover,
            "cumulative_cutover_pct": (total_cutover / total_all * 100) if total_all else 0.0,
            "cumulative_migrated": len(migrated),
        })
        print(f"\nStage {stage_num} ({stage_end}):")
        print(f"  Newly cutover-ready: {len(newly_etl)} ETL, {len(newly_bi)} BI")
        print(f"  Cumulative cutover: {total_cutover}/{total_all} ({total_cutover / total_all * 100:.1f}%)")

    # --- Save CSV ---
    csv_df = pd.DataFrame(cutover_rows)
    csv_path = os.path.join(outdir, f"cutover_readiness_gamma_{resolution}.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"\nCutover readiness CSV saved: {csv_path}")

    # --- Save TXT ---
    txt_path = os.path.join(outdir, f"cutover_readiness_analysis_gamma_{resolution}.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write(f"CUTOVER READINESS ANALYSIS (Recursive Upstream)\nResolution γ={resolution}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 120 + "\n\n")
        f.write("A stream is cutover-ready when:\n")
        f.write("  - ETL: ALL recursive upstream dependencies are migrated\n")
        f.write("    (downstream does not block cutover)\n")
        f.write("  - BI/JSON: all required source tables available (produced + static)\n\n")
        f.write(f"Total ETL: {len(etl_streams_all)} | BI: {len(bi_streams_all)}\n")
        f.write(f"Upstream deps: {streams_with_upstream} | Source streams: {streams_no_upstream}\n")
        f.write(f"Start Date: {start_date.strftime('%Y-%m-%d')}\n\n")

        for ss in stage_summaries:
            f.write(f"{'=' * 120}\n")
            lbl = ss["community_label"]
            if "Clubbed" in str(lbl):
                f.write(f"STAGE {ss['stage']}: {lbl} | {ss['start_date']} to {ss['end_date']}\n")
            else:
                f.write(f"STAGE {ss['stage']}: Community {lbl} | {ss['start_date']} to {ss['end_date']}\n")
            f.write(f"{'=' * 120}\n\n")
            f.write(f"Newly Cutover-Ready ETL: {ss['newly_cutover_etl']} | BI: {ss['newly_cutover_bi']}\n")
            f.write(f"Cumulative: {ss['cumulative_cutover_total']}/{len(all_streams_set)} ({ss['cumulative_cutover_pct']:.2f}%)\n")
            f.write(f"  ETL: {ss['cumulative_cutover_etl']}/{len(etl_streams_all)} | BI: {ss['cumulative_cutover_bi']}/{len(bi_streams_all)}\n\n")

            if ss["newly_cutover_etl_streams"]:
                f.write("--- ETL Streams Ready for Cutover ---\n")
                for stream in ss["newly_cutover_etl_streams"]:
                    up = get_all_recursive_upstreams(stream, recursive_upstream)
                    if up:
                        f.write(f"    - {stream} ({len(up)} recursive upstreams, all migrated)\n")
                    else:
                        f.write(f"    - {stream} (source stream)\n")
            else:
                f.write("--- ETL Streams Ready for Cutover ---\n  (None)\n")

            if ss["newly_cutover_bi_streams"]:
                f.write("\n--- BI Streams Ready for Cutover ---\n")
                for stream in ss["newly_cutover_bi_streams"]:
                    f.write(f"    - {stream}\n")
            else:
                f.write("\n--- BI Streams Ready for Cutover ---\n  (None)\n")
            f.write("\n")

        # Summary
        f.write(f"\n{'=' * 120}\nCUTOVER READINESS SUMMARY\n{'=' * 120}\n\n")
        fin = stage_summaries[-1] if stage_summaries else {}
        er = fin.get("cumulative_cutover_etl", 0)
        br = fin.get("cumulative_cutover_bi", 0)
        f.write(f"ETL cutover-ready: {er}/{len(etl_streams_all)}")
        f.write(f" ({er / len(etl_streams_all) * 100:.1f}%)\n" if etl_streams_all else "\n")
        f.write(f"BI cutover-ready: {br}/{len(bi_streams_all)}")
        f.write(f" ({br / len(bi_streams_all) * 100:.1f}%)\n" if bi_streams_all else "\n")
        f.write(f"Total: {er + br}/{len(all_streams_set)} ({fin.get('cumulative_cutover_pct', 0):.1f}%)\n\n")

        not_etl = etl_streams_all - cutover_ready_etl
        not_bi = bi_streams_all - cutover_ready_bi
        if not_etl or not_bi:
            f.write("--- Streams NOT Cutover-Ready After All Stages ---\n\n")
            if not_etl:
                f.write(f"ETL ({len(not_etl)}):\n")
                for stream in sorted(not_etl):
                    unmig = sorted(get_all_recursive_upstreams(stream, recursive_upstream) - migrated)
                    f.write(f"  - {stream} ({len(unmig)} unmigrated upstream)\n")
                    if unmig:
                        f.write(f"    {', '.join(unmig[:10])}")
                        if len(unmig) > 10:
                            f.write(f" ... +{len(unmig) - 10}")
                        f.write("\n")
            if not_bi:
                f.write(f"\nBI ({len(not_bi)}):\n")
                for stream in sorted(not_bi):
                    missing = bi_req_tables.get(stream, set()) - bi_available
                    f.write(f"  - {stream} ({len(missing)} missing tables)\n")
        f.write(f"\n{'=' * 120}\n")

    print(f"Cutover readiness report saved: {txt_path}")
    return cutover_ready_etl, cutover_ready_bi, stage_summaries, recursive_upstream


# ---------------------------------------------------------------------------
# Immediate dependency cutover analysis
# ---------------------------------------------------------------------------


def generate_immediate_cutover_analysis(
    leiden_df: pd.DataFrame,
    stream_stream_dependency_df: "DataFrame",
    dependency_df: "DataFrame",
    timeline_df: pd.DataFrame,
    stream_produces: dict[str, set],
    missing_static_tables: set[str],
    bi_stream_required_tables: dict[str, set],
    resolution: float,
    start_date: datetime,
    output_path: str,
    recursive_upstream: dict[str, set],
    cutover_ready_etl_recursive: set[str],
    cutover_ready_bi_recursive: set[str],
    migrated_streams_recursive: set[str],
    sync_threshold_gb: float = 500.0,
) -> tuple[set, set, list]:
    """Immediate dependency cutover analysis with sync threshold gating.

    A stream is a cutover candidate when all its immediate **upstream**
    neighbours are migrated.  Candidates are greedily selected to keep total
    sync under *sync_threshold_gb*, and blocked if any sync table has >1
    unmigrated producer.

    Returns
    -------
    tuple
        ``(cutover_ready_etl_imm, cutover_ready_bi_imm, stage_summaries_imm)``
    """
    from pyspark.sql.functions import col, lower, upper  # noqa: E402

    outdir = os.path.join(output_path, "migration_order_analysis")
    os.makedirs(outdir, exist_ok=True)

    all_streams_set = set(leiden_df["stream"].tolist())
    bi_streams_all_list, etl_streams_all_list = split_bi_etl(all_streams_set)
    bi_streams_all = set(bi_streams_all_list)
    etl_streams_all = set(etl_streams_all_list)

    # --- Directed neighbour maps ---
    deps_dir = stream_stream_dependency_df.select("from", "to").distinct().toPandas()
    etl_dir = deps_dir[
        ~deps_dir["from"].str.lower().str.contains("json")
        & ~deps_dir["to"].str.lower().str.contains("json")
    ].copy()

    upstream_neighbors: dict[str, set] = defaultdict(set)
    downstream_neighbors: dict[str, set] = defaultdict(set)
    for _, row in etl_dir.iterrows():
        src, tgt = row["from"], row["to"]
        if src in etl_streams_all and tgt in etl_streams_all:
            upstream_neighbors[tgt].add(src)
            downstream_neighbors[src].add(tgt)

    # --- Edge table lookup ---
    deps_with_tables = stream_stream_dependency_df.select(
        "from", "to", "table", "size"
    ).distinct().toPandas()
    deps_with_tables["size"] = pd.to_numeric(deps_with_tables["size"], errors="coerce").fillna(0.0)

    edge_tables_lookup: dict[tuple, list] = defaultdict(list)
    for _, row in deps_with_tables.iterrows():
        edge_tables_lookup[(row["from"], row["to"])].append((row["table"], row["size"]))
    for key in edge_tables_lookup:
        d: dict[str, float] = {}
        for tbl, sz in edge_tables_lookup[key]:
            if tbl not in d or sz > d[tbl]:
                d[tbl] = sz
        edge_tables_lookup[key] = [(t, s) for t, s in d.items()]

    # --- Table → all producers ---
    table_to_all_producers: dict[str, set] = defaultdict(set)
    for _, row in deps_with_tables.iterrows():
        table_to_all_producers[row["table"]].add(row["from"])

    # --- BI table requirements ---
    bi_src_df = dependency_df.filter(
        (lower(col("stream_name")).contains("json"))
        & (upper(col("table_type")).isin(["SRC", "SRC_TRNS"]))
    ).select(
        col("stream_name"), upper(col("DB_Table_Name")).alias("table_name"),
    ).distinct().toPandas()
    bi_req = bi_src_df.groupby("stream_name", group_keys=False)["table_name"].apply(set).to_dict()

    stream_to_community = leiden_df.set_index("stream")["community"].to_dict()

    print(f"Directed ETL edges: {len(etl_dir)}")
    print(f"Edge-to-table mappings: {len(edge_tables_lookup)}")
    print(f"Sync threshold: {sync_threshold_gb:.0f} GB")

    # --- Process stages ---
    migrated_imm: set[str] = set()
    cutover_etl: set[str] = set()
    cutover_bi: set[str] = set()
    bi_avail = missing_static_tables.copy()

    cutover_rows: list[dict] = []
    stage_summaries: list[dict] = []

    for _, stage_row in timeline_df.iterrows():
        stage_num = stage_row["Stage"]
        stage_end = stage_row["End_Date"]
        stage_start = stage_row["Start_Date"]
        stage_comms = stage_row["Original_Community_IDs"]
        stage_label = stage_row["Community_ID"]

        for comm_id in stage_comms:
            cs = set(leiden_df[leiden_df["community"] == comm_id]["stream"].tolist())
            migrated_imm.update(cs)
            for s in cs:
                if s in stream_produces:
                    bi_avail.update(stream_produces[s])

        # Baseline sync for already-cutover streams
        baseline_sync: dict[str, float] = {}
        for stream in cutover_etl:
            for tbl, sz in compute_stream_sync_tables(
                stream, migrated_imm, upstream_neighbors, edge_tables_lookup
            ).items():
                if tbl not in baseline_sync or sz > baseline_sync[tbl]:
                    baseline_sync[tbl] = sz
        baseline_total = builtins.sum(baseline_sync.values())

        # Candidates: upstream-only check
        candidates = []
        for stream in sorted((etl_streams_all & migrated_imm) - cutover_etl):
            if upstream_neighbors[stream].issubset(migrated_imm):
                candidates.append(stream)

        # Per-candidate sync
        cand_sync: dict[str, dict] = {}
        for stream in candidates:
            cand_sync[stream] = compute_stream_sync_tables(
                stream, migrated_imm, upstream_neighbors, edge_tables_lookup
            )

        # Greedy selection
        cand_sorted = sorted(candidates, key=lambda s: builtins.sum(cand_sync[s].values()))
        sel_sync = dict(baseline_sync)
        sel_total = baseline_total
        newly_etl: list[str] = []
        held_back: list[str] = []
        multi_blocked: list[tuple] = []

        for stream in cand_sorted:
            ss_sync = cand_sync[stream]
            has_mp, prob = has_multi_unmigrated_producer(
                ss_sync.keys(), migrated_imm, table_to_all_producers
            )
            if has_mp:
                held_back.append(stream)
                multi_blocked.append((stream, prob))
                continue
            marginal = {t: s for t, s in ss_sync.items() if t not in sel_sync}
            mc = builtins.sum(marginal.values())
            if sel_total + mc < sync_threshold_gb:
                newly_etl.append(stream)
                cutover_etl.add(stream)
                sel_sync.update(ss_sync)
                sel_total += mc
            else:
                held_back.append(stream)

        held_back_count = len(held_back)
        mp_set = set(s for s, _ in multi_blocked)
        held_reasons = {
            s: ("multi_producer" if s in mp_set else "sync_threshold")
            for s in held_back
        }
        held_costs = {s: builtins.sum(cand_sync[s].values()) for s in held_back}

        # Stage sync tables (for the selected cutover set)
        stage_sync: dict[str, dict] = {}
        for stream in cutover_etl:
            for tbl, sz in compute_stream_sync_tables(
                stream, migrated_imm, upstream_neighbors, edge_tables_lookup
            ).items():
                if tbl not in stage_sync:
                    stage_sync[tbl] = {"size": sz, "producers": set(), "consumers": set()}
                visited_inner: set[str] = set()
                q = [stream]
                while q:
                    cur = q.pop(0)
                    if cur in visited_inner:
                        continue
                    visited_inner.add(cur)
                    for up in upstream_neighbors.get(cur, set()):
                        if up in migrated_imm:
                            q.append(up)
                        else:
                            for t, _ in edge_tables_lookup.get((up, cur), []):
                                if t == tbl:
                                    stage_sync[tbl]["producers"].add(up)
                                    stage_sync[tbl]["consumers"].add(cur)

        sync_count = len(stage_sync)
        sync_size = builtins.sum(i["size"] for i in stage_sync.values())

        # BI cutover
        newly_bi: list[str] = []
        for bs in sorted(bi_streams_all - cutover_bi):
            req = bi_req.get(bs, set())
            if not req:
                if bs in migrated_imm:
                    newly_bi.append(bs)
                    cutover_bi.add(bs)
            elif req.issubset(bi_avail):
                newly_bi.append(bs)
                cutover_bi.add(bs)

        for stream in newly_etl:
            cutover_rows.append({
                "stage": stage_num, "end_date": stage_end,
                "community_id": int(stream_to_community.get(stream, -1)),
                "stream_name": stream, "stream_type": "ETL",
                "immediate_upstream_count": len(upstream_neighbors[stream]),
                "immediate_downstream_count": len(downstream_neighbors[stream]),
                "recursive_upstream_count": len(get_all_recursive_upstreams(stream, recursive_upstream)),
            })
        for stream in newly_bi:
            cutover_rows.append({
                "stage": stage_num, "end_date": stage_end,
                "community_id": int(stream_to_community.get(stream, -1)),
                "stream_name": stream, "stream_type": "BI",
                "immediate_upstream_count": -1, "immediate_downstream_count": -1,
                "recursive_upstream_count": -1,
            })

        total_co = len(cutover_etl) + len(cutover_bi)
        total_all = len(all_streams_set)
        stage_summaries.append({
            "stage": stage_num, "community_label": stage_label,
            "start_date": stage_start, "end_date": stage_end,
            "newly_cutover_etl": len(newly_etl), "newly_cutover_bi": len(newly_bi),
            "newly_cutover_etl_streams": newly_etl, "newly_cutover_bi_streams": newly_bi,
            "cumulative_cutover_etl": len(cutover_etl),
            "cumulative_cutover_bi": len(cutover_bi),
            "cumulative_cutover_total": total_co,
            "cumulative_cutover_pct": (total_co / total_all * 100) if total_all else 0.0,
            "cumulative_migrated": len(migrated_imm),
            "sync_table_count": sync_count, "sync_total_size_gb": sync_size,
            "sync_tables_detail": stage_sync,
            "cutover_allowed": held_back_count == 0,
            "held_back_count": held_back_count,
            "held_back_streams": held_back,
            "held_back_sync_costs": held_costs,
            "held_back_reasons": held_reasons,
            "multi_producer_blocked": multi_blocked,
        })

        mp_n = len(multi_blocked)
        th_n = held_back_count - mp_n
        status = f"ALL ({len(newly_etl)})" if held_back_count == 0 else f"PARTIAL ({len(newly_etl)} sel, {held_back_count} held)"
        print(f"\nStage {stage_num} ({stage_end}):")
        print(f"  Sync: {sync_count} tables, {sync_size:.2f} GB (threshold: {sync_threshold_gb:.0f} GB)")
        print(f"  Cutover: {status}")
        print(f"  Newly cutover-ready: {len(newly_etl)} ETL, {len(newly_bi)} BI")
        if mp_n:
            print(f"  Blocked (multi-producer): {mp_n}")
        if th_n > 0:
            print(f"  Blocked (sync threshold): {th_n}")
        print(f"  Cumulative: {total_co}/{total_all} ({total_co / total_all * 100:.1f}%)")

    # --- Save CSVs ---
    csv_df = pd.DataFrame(cutover_rows)
    csv_path = os.path.join(outdir, f"cutover_readiness_immediate_gamma_{resolution}.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"\nImmediate cutover CSV saved: {csv_path}")

    sync_rows = []
    for ss in stage_summaries:
        for tbl in sorted(ss["sync_tables_detail"].keys()):
            info = ss["sync_tables_detail"][tbl]
            sync_rows.append({
                "stage": ss["stage"], "end_date": ss["end_date"],
                "table_name": tbl, "size_gb": info["size"],
                "unmigrated_producers": ", ".join(sorted(info["producers"])),
                "migrated_consumers": ", ".join(sorted(info["consumers"])),
            })
    sync_csv = pd.DataFrame(sync_rows)
    sync_csv_path = os.path.join(outdir, f"cutover_sync_details_immediate_gamma_{resolution}.csv")
    sync_csv.to_csv(sync_csv_path, index=False)
    print(f"Sync details CSV saved: {sync_csv_path}")

    # --- Save TXT ---
    txt_path = os.path.join(outdir, f"cutover_readiness_immediate_analysis_gamma_{resolution}.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write(f"IMMEDIATE DEPENDENCY CUTOVER READINESS ANALYSIS\nResolution γ={resolution}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 120 + "\n\n")
        f.write("Cutover-ready when:\n")
        f.write("  - ETL: all immediate upstream neighbours migrated\n")
        f.write(f"  - Sync threshold: < {sync_threshold_gb:.0f} GB\n")
        f.write("  - Multi-producer rule: sync table must have at most 1 unmigrated producer\n")
        f.write("  - BI/JSON: required source tables available\n\n")
        f.write(f"Total ETL: {len(etl_streams_all)} | BI: {len(bi_streams_all)}\n")
        f.write(f"Threshold: {sync_threshold_gb:.0f} GB\n\n")

        for ss in stage_summaries:
            f.write(f"{'=' * 120}\n")
            lbl = ss["community_label"]
            if "Clubbed" in str(lbl):
                f.write(f"STAGE {ss['stage']}: {lbl} | {ss['start_date']} to {ss['end_date']}\n")
            else:
                f.write(f"STAGE {ss['stage']}: Community {lbl} | {ss['start_date']} to {ss['end_date']}\n")
            f.write(f"{'=' * 120}\n\n")

            if ss["cutover_allowed"]:
                f.write(f"Cutover: ALL ELIGIBLE ({ss['sync_total_size_gb']:.2f} GB sync)\n")
            else:
                f.write(f"Cutover: PARTIAL — {ss['newly_cutover_etl']} selected, {ss['held_back_count']} held back\n")
                f.write(f"  Sync for subset: {ss['sync_total_size_gb']:.2f} GB\n")
                mp_bl = ss.get("multi_producer_blocked", [])
                if mp_bl:
                    f.write(f"\n  Held back — MULTI-PRODUCER ({len(mp_bl)}):\n")
                    for hb, prob in mp_bl:
                        cost = ss["held_back_sync_costs"].get(hb, 0.0)
                        f.write(f"    - {hb} (sync: {cost:.2f} GB)\n")
                        for tbl, unmig in prob:
                            f.write(f"        Table: {tbl} — unmigrated: {', '.join(sorted(unmig))}\n")
                th_bl = [s for s in ss["held_back_streams"] if ss["held_back_reasons"].get(s) == "sync_threshold"]
                if th_bl:
                    f.write(f"\n  Held back — SYNC THRESHOLD ({len(th_bl)}):\n")
                    for hb in sorted(th_bl):
                        f.write(f"    - {hb} (sync: {ss['held_back_sync_costs'].get(hb, 0.0):.2f} GB)\n")

            f.write(f"\nNewly Cutover-Ready ETL: {ss['newly_cutover_etl']} | BI: {ss['newly_cutover_bi']}\n")
            f.write(f"Cumulative: {ss['cumulative_cutover_total']}/{len(all_streams_set)} ({ss['cumulative_cutover_pct']:.2f}%)\n\n")

            sd = ss["sync_tables_detail"]
            f.write(f"--- Sync Requirements ---\nTables: {ss['sync_table_count']} | Size: {ss['sync_total_size_gb']:.2f} GB\n\n")
            if sd:
                for tbl in sorted(sd.keys()):
                    info = sd[tbl]
                    f.write(f"  Table: {tbl} ({info['size']:.2f} GB)\n")
                    f.write(f"    Produced by (unmigrated): {', '.join(sorted(info['producers']))}\n")
                    f.write(f"    Consumed by (migrated):   {', '.join(sorted(info['consumers']))}\n")
            else:
                f.write("  (No sync required)\n")

            if ss["newly_cutover_etl_streams"]:
                f.write(f"\n--- ETL Streams Ready ---\n")
                for stream in ss["newly_cutover_etl_streams"]:
                    up = sorted(upstream_neighbors[stream])
                    dn = sorted(downstream_neighbors[stream])
                    f.write(f"    - {stream}\n")
                    if up:
                        f.write(f"        Upstream ({len(up)}): {', '.join(up)}\n")
                    if dn:
                        f.write(f"        Downstream ({len(dn)}): {', '.join(dn)}\n")
                    if not up and not dn:
                        f.write("        (Isolated)\n")

            if ss["newly_cutover_bi_streams"]:
                f.write(f"\n--- BI Streams Ready ---\n")
                for stream in ss["newly_cutover_bi_streams"]:
                    f.write(f"    - {stream}\n")
            f.write("\n")

        # Summary
        f.write(f"\n{'=' * 120}\nIMMEDIATE CUTOVER SUMMARY\n{'=' * 120}\n\n")
        fin = stage_summaries[-1] if stage_summaries else {}
        er = fin.get("cumulative_cutover_etl", 0)
        br = fin.get("cumulative_cutover_bi", 0)
        tr = fin.get("cumulative_cutover_total", 0)
        f.write(f"ETL: {er}/{len(etl_streams_all)}")
        f.write(f" ({er / len(etl_streams_all) * 100:.1f}%)\n" if etl_streams_all else "\n")
        f.write(f"BI: {br}/{len(bi_streams_all)}")
        f.write(f" ({br / len(bi_streams_all) * 100:.1f}%)\n" if bi_streams_all else "\n")
        f.write(f"Total: {tr}/{len(all_streams_set)} ({fin.get('cumulative_cutover_pct', 0):.1f}%)\n\n")

        f.write(f"Threshold: {sync_threshold_gb:.0f} GB\n\n")
        f.write(f"{'Stage':<8} {'Date':<14} {'Sync Tbl':>9} {'Sync GB':>10} {'Status':>10} {'Held':>6} {'MProd':>6} {'Thresh':>6}\n")
        f.write(f"{'-' * 70}\n")
        for ss in stage_summaries:
            st = "ALL" if ss["cutover_allowed"] else "PARTIAL"
            h = str(ss["held_back_count"]) if ss["held_back_count"] else "-"
            mp = str(len(ss.get("multi_producer_blocked", []))) or "-"
            th = str(ss["held_back_count"] - len(ss.get("multi_producer_blocked", [])))
            if th == "0":
                th = "-"
            f.write(f"{ss['stage']:<8} {ss['end_date']:<14} {ss['sync_table_count']:>9} {ss['sync_total_size_gb']:>10.2f} {st:>10} {h:>6} {mp:>6} {th:>6}\n")

        # Comparison
        f.write(f"\n--- Immediate vs Recursive ---\n\n")
        rec_etl = len(cutover_ready_etl_recursive)
        rec_bi = len(cutover_ready_bi_recursive)
        f.write(f"{'Metric':<35} {'Immediate':>10} {'Recursive':>10} {'Diff':>8}\n")
        f.write(f"{'-' * 63}\n")
        f.write(f"{'ETL cutover-ready':<35} {er:>10} {rec_etl:>10} {er - rec_etl:>+8}\n")
        f.write(f"{'BI cutover-ready':<35} {br:>10} {rec_bi:>10} {br - rec_bi:>+8}\n")
        f.write(f"{'Total':<35} {tr:>10} {rec_etl + rec_bi:>10} {tr - rec_etl - rec_bi:>+8}\n")

        # Immediate-only streams
        imm_only = cutover_etl - cutover_ready_etl_recursive
        if imm_only:
            f.write(f"\n--- Immediate-Ready but NOT Recursive-Ready ({len(imm_only)}) ---\n")
            for stream in sorted(imm_only):
                unmig = sorted(get_all_recursive_upstreams(stream, recursive_upstream) - migrated_streams_recursive)
                f.write(f"  - {stream} ({len(unmig)} unmigrated recursive upstreams)\n")
                if unmig:
                    f.write(f"    {', '.join(unmig[:10])}")
                    if len(unmig) > 10:
                        f.write(f" ... +{len(unmig) - 10}")
                    f.write("\n")

        # Not ready
        not_etl = etl_streams_all - cutover_etl
        not_bi = bi_streams_all - cutover_bi
        if not_etl or not_bi:
            f.write(f"\n--- NOT Cutover-Ready (Immediate) ---\n\n")
            if not_etl:
                f.write(f"ETL ({len(not_etl)}):\n")
                for stream in sorted(not_etl):
                    unmig_up = sorted(upstream_neighbors[stream] - migrated_imm)
                    unmig_dn = sorted(downstream_neighbors[stream] - migrated_imm)
                    f.write(f"  - {stream}\n")
                    if unmig_up:
                        f.write(f"    Unmigrated upstream: {', '.join(unmig_up)}\n")
                    if unmig_dn:
                        f.write(f"    Unmigrated downstream: {', '.join(unmig_dn)}\n")
            if not_bi:
                f.write(f"\nBI ({len(not_bi)}):\n")
                for stream in sorted(not_bi):
                    missing = bi_req.get(stream, set()) - bi_avail
                    f.write(f"  - {stream} ({len(missing)} missing tables)\n")

        f.write(f"\n{'=' * 120}\n")

    print(f"Immediate cutover report saved: {txt_path}")
    return cutover_etl, cutover_bi, stage_summaries
