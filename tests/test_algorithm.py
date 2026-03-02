"""
Tests for migration_planner.community_detector.algorithm.

leidenalg is stubbed in conftest.py.
stability_ari is tested with real numpy arrays and sklearn.
scan_resolutions patches run_leiden_rb to avoid needing a real graph.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from migration_planner.community_detector.algorithm import (
    run_leiden_rb,
    scan_resolutions,
    stability_ari,
)


# ---------------------------------------------------------------------------
# TestStabilityAri
# ---------------------------------------------------------------------------


class TestStabilityAri:
    def test_empty_list_returns_1(self):
        assert stability_ari([]) == 1.0

    def test_single_membership_returns_1(self):
        m = np.array([0, 1, 1, 2])
        assert stability_ari([m]) == 1.0

    def test_identical_memberships_returns_1(self):
        m = np.array([0, 0, 1, 1, 2])
        result = stability_ari([m, m, m])
        assert abs(result - 1.0) < 1e-9

    def test_different_memberships_returns_less_than_1(self):
        m1 = np.array([0, 0, 1, 1])
        m2 = np.array([0, 1, 0, 1])
        result = stability_ari([m1, m2])
        assert result < 1.0

    def test_average_over_all_pairs(self):
        """With 3 memberships there are 3 pairs; the mean must be a valid ARI in [-1, 1]."""
        m1 = np.array([0, 0, 1, 1])
        m2 = np.array([0, 1, 0, 1])
        m3 = np.array([0, 0, 0, 1])
        result = stability_ari([m1, m2, m3])
        assert -1.0 <= result <= 1.0

    def test_returns_float(self):
        m = np.array([0, 1])
        assert isinstance(stability_ari([m, m]), float)


# ---------------------------------------------------------------------------
# TestRunLeidenRb
# ---------------------------------------------------------------------------


class TestRunLeidenRb:
    """run_leiden_rb calls la.find_partition, which is stubbed via conftest."""

    def _make_mock_part(self, n_nodes: int = 4, n_communities: int = 2) -> MagicMock:
        """Return a mock leidenalg partition with a plausible membership list."""
        part = MagicMock()
        # membership: 4 nodes split into 2 communities
        membership = [i % n_communities for i in range(n_nodes)]
        part.membership = membership
        part.quality.return_value = 0.42
        return part

    def test_calls_find_partition(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        part = self._make_mock_part()
        la.find_partition.return_value = part

        run_leiden_rb(mock_g, resolution=1.0, seed=42)
        la.find_partition.assert_called_once()

    def test_passes_resolution_to_find_partition(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        run_leiden_rb(mock_g, resolution=1.5, seed=7)
        kwargs = la.find_partition.call_args[1]
        assert kwargs["resolution_parameter"] == 1.5

    def test_passes_seed_to_find_partition(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        run_leiden_rb(mock_g, resolution=1.0, seed=99)
        kwargs = la.find_partition.call_args[1]
        assert kwargs["seed"] == 99

    def test_returns_dict_with_all_expected_keys(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        result = run_leiden_rb(mock_g, resolution=1.0, seed=1)
        expected_keys = {
            "resolution", "seed", "n_communities",
            "largest_comm_share", "small_comms_lt5", "quality", "membership",
        }
        assert set(result.keys()) == expected_keys

    def test_membership_is_numpy_array(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        result = run_leiden_rb(mock_g, resolution=1.0, seed=1)
        assert isinstance(result["membership"], np.ndarray)

    def test_resolution_in_result(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        result = run_leiden_rb(mock_g, resolution=0.8, seed=1)
        assert result["resolution"] == 0.8

    def test_seed_in_result(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        result = run_leiden_rb(mock_g, resolution=1.0, seed=77)
        assert result["seed"] == 77

    def test_quality_is_float(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part()

        result = run_leiden_rb(mock_g, resolution=1.0, seed=1)
        assert isinstance(result["quality"], float)

    def test_n_communities_is_positive_int(self):
        import leidenalg as la

        mock_g = MagicMock()
        mock_g.vcount.return_value = 4
        la.find_partition.return_value = self._make_mock_part(n_nodes=4, n_communities=2)

        result = run_leiden_rb(mock_g, resolution=1.0, seed=1)
        assert isinstance(result["n_communities"], int)
        assert result["n_communities"] > 0


# ---------------------------------------------------------------------------
# TestScanResolutions
# ---------------------------------------------------------------------------


_FAKE_RUN = {
    "resolution": 1.0,
    "seed": 42,
    "n_communities": 3,
    "largest_comm_share": 0.5,
    "small_comms_lt5": 1,
    "quality": 0.9,
    "membership": np.array([0, 1, 2, 0]),
}


def _make_run(resolution: float, seed: int) -> dict:
    return {**_FAKE_RUN, "resolution": resolution, "seed": seed}


class TestScanResolutions:
    def test_calls_run_leiden_for_each_resolution_and_seed(self):
        resolutions = [0.5, 1.0]
        seeds = [1, 2]
        mock_g = MagicMock()

        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ) as mock_run:
            scan_resolutions(mock_g, resolutions, seeds, plot_seed=1)
            assert mock_run.call_count == len(resolutions) * len(seeds)

    def test_summary_has_one_row_per_resolution(self):
        resolutions = [0.5, 1.0, 1.5]
        seeds = [1, 2]
        mock_g = MagicMock()

        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ):
            summary, _ = scan_resolutions(mock_g, resolutions, seeds, plot_seed=1)

        assert len(summary) == len(resolutions)

    def test_summary_columns_present(self):
        resolutions = [1.0]
        seeds = [1]
        mock_g = MagicMock()

        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ):
            summary, _ = scan_resolutions(mock_g, resolutions, seeds, plot_seed=1)

        expected_cols = {
            "resolution", "n_communities_avg", "n_communities_min",
            "n_communities_max", "largest_comm_share_avg",
            "small_comms_lt5_avg", "quality_avg", "stability_ari",
        }
        assert expected_cols.issubset(set(summary.columns))

    def test_rep_by_res_keys_match_resolutions(self):
        resolutions = [0.4, 1.2]
        seeds = [1]
        mock_g = MagicMock()

        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ):
            _, rep_by_res = scan_resolutions(mock_g, resolutions, seeds, plot_seed=1)

        assert set(rep_by_res.keys()) == set(resolutions)

    def test_uses_plot_seed_as_representative_when_present(self):
        resolutions = [1.0]
        seeds = [1, 42, 99]
        mock_g = MagicMock()

        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ):
            _, rep_by_res = scan_resolutions(mock_g, resolutions, seeds, plot_seed=42)

        assert rep_by_res[1.0]["seed"] == 42

    def test_falls_back_to_first_run_when_plot_seed_absent(self):
        resolutions = [1.0]
        seeds = [1, 2, 3]
        mock_g = MagicMock()

        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ):
            _, rep_by_res = scan_resolutions(mock_g, resolutions, seeds, plot_seed=999)

        # plot_seed=999 not in seeds; falls back to first run (seed=1)
        assert rep_by_res[1.0]["seed"] == 1

    def test_returns_dataframe_and_dict(self):
        mock_g = MagicMock()
        with patch(
            "migration_planner.community_detector.algorithm.run_leiden_rb",
            side_effect=lambda g, res, seed, **kw: _make_run(res, seed),
        ):
            result = scan_resolutions(mock_g, [1.0], [1], plot_seed=1)

        summary, rep_by_res = result
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(rep_by_res, dict)
