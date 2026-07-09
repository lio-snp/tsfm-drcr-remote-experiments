import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.features import autocorrelation_strength, feature_vector, spectral_entropy
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


if __name__ == "__main__":
    unittest.main()
