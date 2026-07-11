import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.features import autocorrelation_strength, feature_vector, spectral_entropy
from low_snr_tsfm.benefit_selective import frozen_action_grids, pre_origin_feature_vector
from low_snr_tsfm.stats import benjamini_hochberg, christoffersen_independence, diebold_mariano, kupiec_pof


class StatsFeatureTests(unittest.TestCase):
    def test_bh_rejects_small_p_values(self):
        rejected, q_values = benjamini_hochberg(np.array([0.001, 0.02, 0.5]), alpha=0.05)
        self.assertTrue(rejected[0])
        self.assertLessEqual(q_values[0], q_values[2])

    def test_dm_detects_lower_model_loss(self):
        model = np.array([0.8, 0.7, 0.9, 0.8, 0.75])
        baseline = np.array([1.2, 1.1, 1.0, 1.3, 1.2])
        result = diebold_mariano(model, baseline, alternative="less")
        self.assertLess(result.statistic, 0)
        self.assertLess(result.p_value, 0.05)

    def test_coverage_tests_return_valid_p_values(self):
        exceptions = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        kupiec = kupiec_pof(exceptions, alpha=0.1)
        christoffersen = christoffersen_independence(exceptions)
        self.assertGreaterEqual(kupiec.p_value, 0.0)
        self.assertLessEqual(christoffersen.p_value, 1.0)

    def test_features_have_expected_direction(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(size=500)
        periodic = np.sin(np.arange(500) * 2 * np.pi / 24)
        self.assertGreaterEqual(spectral_entropy(noise), spectral_entropy(periodic))
        self.assertGreater(autocorrelation_strength(periodic), 0.5)
        features = feature_vector(periodic, horizon=24, context_length=128, period=24)
        self.assertIn("horizon_context_ratio", features)
        self.assertIn("seasonality_strength", features)

    def test_external_features_and_actions_are_outcome_blind(self):
        levels = [level / 10 for level in range(1, 10)]
        context = np.sin(np.arange(96) * 2 * np.pi / 12)
        grid = np.column_stack([np.full(6, level) for level in levels])
        method = {
            "base_cpr_policy": {
                "min_active": 2, "weight": 0.6, "factor_step": 0.5, "max_weight": 1.0,
                "conflict_threshold": 0.25, "shield_cap": 0.125,
                "degeneracy_hcr_threshold": 0.3, "degeneracy_trend_threshold": 0.4,
            },
            "smooth_interval_head": {
                "low_scale": 0.906, "high_scale": 1.0, "native_width_ratio_threshold": 0.1,
                "temperature": 0.05, "structured_guard_hcr_threshold": 0.05,
                "structured_guard_trend_threshold": 0.05, "structured_scale": 2.0,
                "structured_weight_cap": 0.125,
            },
        }
        taxonomy = {
            "minimum_active_factors": 2,
            "thresholds": {
                "horizon_context_ratio": 0.1, "seasonality_strength": 0.15,
                "trend_strength": 0.1, "severe_weak_seasonality": 0.075,
                "spike_frequency": 0.022, "changepoint_density": 0.08,
                "zero_ratio": 0.1, "coefficient_of_variation": 1.5,
                "kurtosis_excess": 8.0, "spectral_entropy": 0.85,
            },
        }
        features = pre_origin_feature_vector(context, 6, levels, grid, "chronos", method["smooth_interval_head"])
        self.assertNotIn("flatness_score", features)
        actions, diagnostics = frozen_action_grids(
            np.full(6, 0.5), np.full(6, 1.0), levels, grid, features, taxonomy, method
        )
        self.assertEqual(set(actions), {"native_tsfm", "drcr_point", "drcr_expert_pull_1.25_cap_1.10"})
        self.assertEqual(actions["native_tsfm"].shape, (6, 9))
        self.assertIn("low_structure_factor_count", diagnostics)

        altered_future = np.asarray([1e9, -1e9, 3e9, -3e9, 5e9, -5e9])
        repeated_features = pre_origin_feature_vector(
            context, altered_future.size, levels, grid, "chronos", method["smooth_interval_head"]
        )
        repeated_actions, repeated_diagnostics = frozen_action_grids(
            np.full(6, 0.5), np.full(6, 1.0), levels, grid, repeated_features, taxonomy, method
        )
        self.assertEqual(features, repeated_features)
        self.assertEqual(diagnostics, repeated_diagnostics)
        for action in actions:
            np.testing.assert_allclose(actions[action], repeated_actions[action])


if __name__ == "__main__":
    unittest.main()
