import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_remote_q9_main_table.py"
SPEC = importlib.util.spec_from_file_location("build_remote_q9_main_table", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def raw_row(series: str, actual: float, model: float, baseline: float) -> dict[str, str]:
    row = {
        "series_id": series,
        "actual": str(actual),
        "forecast_mean": str(model),
        "baseline_forecast": str(baseline),
    }
    for level in range(1, 10):
        row[f"forecast_q{level}0"] = str(actual)
    return row


class RemoteQ9MainTableTests(unittest.TestCase):
    def test_pooled_mae_and_rmse_ratios(self):
        rows = [raw_row("A", 1, 2, 1), raw_row("A", 3, 1, 5)]
        result = MODULE.pooled_point_metrics(rows)
        self.assertAlmostEqual(result["model_mae"], 1.5)
        self.assertAlmostEqual(result["baseline_mae"], 1.0)
        self.assertAlmostEqual(result["relmae"], 1.5)
        self.assertAlmostEqual(result["relrmse"], math.sqrt(1.25))

    def test_avg_relmae_uses_error_count_weights(self):
        rows = [
            raw_row("A", 0, 1, 2),
            raw_row("A", 0, 1, 2),
            raw_row("B", 0, 2, 1),
        ]
        result = MODULE.series_comparison_metrics(rows)
        expected = math.exp((2 * math.log(0.5) + math.log(2.0)) / 3)
        self.assertAlmostEqual(result["avg_relmae"], expected)
        self.assertEqual(result["percent_better_mae"], 50.0)
        self.assertEqual(result["sign_test_pvalue"], 1.0)

    def test_avg_relmae_is_na_for_zero_series_error(self):
        rows = [raw_row("A", 0, 1, 0), raw_row("B", 0, 1, 2)]
        result = MODULE.series_comparison_metrics(rows)
        self.assertIsNone(result["avg_relmae"])
        self.assertEqual(result["avg_relmae_status"], "undefined_nonpositive_series_mae")
        self.assertEqual(result["zero_baseline_mae_series"], 1)

    def test_q9_wql_and_coverage(self):
        result = MODULE.q9_metrics([raw_row("A", 10, 10, 9)])
        self.assertEqual(result["q9_wql"], 0.0)
        self.assertEqual(result["q10_q90_coverage"], 1.0)
        self.assertAlmostEqual(result["q10_q90_coverage_abs_error"], 0.2)

    def test_seasonal_scale_has_no_epsilon_fallback(self):
        self.assertEqual(MODULE.seasonal_scale([1.0, 2.0, 3.0], 1), 1.0)
        self.assertIsNone(MODULE.seasonal_scale([1.0, 1.0, 1.0], 1))

    def test_benjamini_hochberg_preserves_missing_values(self):
        adjusted = MODULE.benjamini_hochberg([0.01, 0.04, 0.03, None])
        self.assertEqual(adjusted[-1], None)
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)


if __name__ == "__main__":
    unittest.main()
