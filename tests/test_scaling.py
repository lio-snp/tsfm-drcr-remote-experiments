import math
import unittest

import context  # noqa: F401
from low_snr_tsfm.scaling import bootstrap_interactions, linear_slope, model_rate_slope


class ScalingTests(unittest.TestCase):
    def test_linear_slope_ignores_nonfinite_values(self):
        self.assertAlmostEqual(linear_slope([1, 2, 3], [2, 4, 6]), 2.0)
        self.assertTrue(math.isnan(linear_slope([1], [2])))

    def test_model_rate_slope_aggregates_by_model(self):
        rows = [
            {"model": "tiny", "log10_params_m": 1.0, "excess_variance": 0},
            {"model": "tiny", "log10_params_m": 1.0, "excess_variance": 1},
            {"model": "base", "log10_params_m": 2.0, "excess_variance": 1},
            {"model": "base", "log10_params_m": 2.0, "excess_variance": 1},
        ]
        self.assertAlmostEqual(model_rate_slope(rows, outcome="excess_variance"), 0.5)

    def test_bootstrap_interactions_preserve_units_across_sizes(self):
        rows = []
        for target, values in {"covid": [0, 1], "solar": [0, 0]}.items():
            for unit in ["a", "b"]:
                for model, x, value in [("tiny", 1.0, values[0]), ("base", 2.0, values[1])]:
                    rows.append(
                        {
                            "target_key": target,
                            "paired_unit_id": unit,
                            "model": model,
                            "log10_params_m": x,
                            "excess_variance": value,
                        }
                    )
        result = bootstrap_interactions(
            rows,
            failure_target_key="covid",
            control_target_keys=["solar"],
            outcome="excess_variance",
            n_bootstrap=20,
        )
        self.assertEqual(result["outcome"], "excess_variance")
        self.assertGreater(
            result["interactions"]["covid_minus_solar"]["point_slope_difference"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
