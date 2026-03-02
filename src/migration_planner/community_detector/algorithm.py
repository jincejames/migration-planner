from __future__ import annotations

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def run_leiden_rb(
    g: ig.Graph,
    resolution: float,
    seed: int,
    weights: str = "weight",
    n_iterations: int = 100,
) -> dict:
    """
    Run Leiden community detection on *g* using the RBConfiguration objective.

    Parameters
    ----------
    g:
        Undirected, weighted igraph.Graph.
    resolution:
        The gamma parameter controlling community granularity.
        Higher → more/smaller communities; lower → fewer/larger.
    seed:
        Random seed for reproducible / stability-test runs.
    weights:
        Name of the edge attribute to use as weights (default ``"weight"``).
    n_iterations:
        Number of Leiden optimisation iterations.

    Returns
    -------
    dict
        Keys: resolution, seed, n_communities, largest_comm_share,
        small_comms_lt5, quality, membership.
    """
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    membership = np.array(part.membership, dtype=int)
    counts = np.bincount(membership)
    counts_sorted = np.sort(counts)[::-1]
    return {
        "resolution": resolution,
        "seed": seed,
        "n_communities": len(counts),
        "largest_comm_share": counts_sorted[0] / g.vcount(),
        "small_comms_lt5": int((counts < 5).sum()),
        "quality": float(part.quality()),
        "membership": membership,
    }


def stability_ari(memberships: list[np.ndarray]) -> float:
    """
    Average pairwise Adjusted Rand Index across a list of membership arrays.

    Returns 1.0 when fewer than two membership arrays are supplied (degenerate
    case: perfect agreement by definition).
    """
    if len(memberships) < 2:
        return 1.0
    aris = [
        adjusted_rand_score(memberships[i], memberships[j])
        for i in range(len(memberships))
        for j in range(i + 1, len(memberships))
    ]
    return float(np.mean(aris)) if aris else 1.0


def scan_resolutions(
    g: ig.Graph,
    resolutions: list[float],
    seeds: list[int],
    plot_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Run Leiden at every resolution × seed combination and collect diagnostics.

    Parameters
    ----------
    g:
        Input graph.
    resolutions:
        Grid of gamma values to evaluate.
    seeds:
        Random seeds for stability testing at each resolution.
    plot_seed:
        Preferred seed to use as the representative run for each resolution.
        Falls back to the first run when ``plot_seed`` is not in ``seeds``.

    Returns
    -------
    summary : pd.DataFrame
        One row per resolution with aggregated metrics.
    rep_by_res : dict[float, dict]
        Maps each resolution to the representative run dict.
    """
    rows: list[dict] = []
    rep_by_res: dict[float, dict] = {}

    for res in resolutions:
        runs = [run_leiden_rb(g, res, s) for s in seeds]
        memberships = [r["membership"] for r in runs]
        n_comms = [r["n_communities"] for r in runs]
        largest = [r["largest_comm_share"] for r in runs]
        small_lt5 = [r["small_comms_lt5"] for r in runs]
        quality = [r["quality"] for r in runs]

        rows.append({
            "resolution": res,
            "n_communities_avg": np.mean(n_comms),
            "n_communities_min": np.min(n_comms),
            "n_communities_max": np.max(n_comms),
            "largest_comm_share_avg": np.mean(largest),
            "small_comms_lt5_avg": np.mean(small_lt5),
            "quality_avg": np.mean(quality),
            "stability_ari": stability_ari(memberships),
        })

        rep = next((r for r in runs if r["seed"] == plot_seed), runs[0])
        rep_by_res[res] = rep

    summary = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)
    return summary, rep_by_res
