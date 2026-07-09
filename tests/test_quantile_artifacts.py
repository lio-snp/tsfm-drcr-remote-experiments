import unittest

import numpy as np

from low_snr_tsfm.quantile_artifacts import (
    forecast_quantile_columns,
    nearest_quantile_index,
    orient_quantile_matrix,
    quantile_column_name,
    quantile_level_from_column_name,
    quantile_matrix_from_rows,
    quantile_row_values,
    quantile_triplet_from_matrix,
)


class QuantileArtifactTests(unittest.TestCase):
    def test_quantile_column_names_keep_legacy_percent_style(self) -> None:
        self.assertEqual(quantile_column_name(0.1), "forecast_q10")
        self.assertEqual(quantile_column_name(0.5), "forecast_q50")
        self.assertEqual(quantile_column_name(0.9), "forecast_q90")
        self.assertEqual(quantile_column_name(0.025), "forecast_q2p5")
        self.assertAlmostEqual(quantile_level_from_column_name("forecast_q2p5"), 0.025)
        self.assertEqual(
            forecast_quantile_columns({"forecast_q90", "forecast_mean", "forecast_q10"}),
            [(0.1, "forecast_q10"), (0.9, "forecast_q90")],
        )

    def test_orient_quantile_matrix_accepts_both_orientations(self) -> None:
        levels = [0.1, 0.5, 0.9]
        horizon_by_level = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float)
        level_by_horizon = horizon_by_level.T
        np.testing.assert_allclose(orient_quantile_matrix(horizon_by_level, levels), horizon_by_level)
        np.testing.assert_allclose(orient_quantile_matrix(level_by_horizon, levels), horizon_by_level)

    def test_triplet_uses_nearest_levels(self) -> None:
        levels = [0.05, 0.5, 0.95]
        values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float)
        q10, q50, q90 = quantile_triplet_from_matrix(values, levels)
        np.testing.assert_allclose(q10, [1, 4])
        np.testing.assert_allclose(q50, [2, 5])
        np.testing.assert_allclose(q90, [3, 6])
        self.assertEqual(nearest_quantile_index(levels, 0.9), 2)

    def test_quantile_row_values_preserve_all_levels(self) -> None:
        levels = [0.1, 0.25, 0.5, 0.9]
        values = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        self.assertEqual(
            quantile_row_values(values, levels, 1),
            {
                "forecast_q10": 5.0,
                "forecast_q25": 6.0,
                "forecast_q50": 7.0,
                "forecast_q90": 8.0,
            },
        )

    def test_quantile_matrix_from_rows_uses_complete_columns_only(self) -> None:
        rows = [
            {"forecast_q10": "1", "forecast_q50": "2", "forecast_q90": "3", "forecast_q95": ""},
            {"forecast_q10": "4", "forecast_q50": "5", "forecast_q90": "6", "forecast_q95": "7"},
        ]
        levels, values = quantile_matrix_from_rows(rows)
        self.assertEqual(levels, [0.1, 0.5, 0.9])
        np.testing.assert_allclose(values, [[1, 2, 3], [4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
