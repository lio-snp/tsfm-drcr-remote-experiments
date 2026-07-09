import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.synthetic import ar1, default_snr_sweep, garch11, regime_switching_ar, white_noise


class SyntheticTests(unittest.TestCase):
    def test_white_noise_length_and_mean(self):
        values = white_noise(10_000, sigma=1.0, seed=1)
        self.assertEqual(values.shape, (10_000,))
        self.assertLess(abs(float(np.mean(values))), 0.05)

    def test_ar1_autocorrelation_increases_with_phi(self):
        weak = ar1(2_000, phi=0.1, seed=2)
        strong = ar1(2_000, phi=0.8, seed=2)
        weak_ac = np.corrcoef(weak[:-1], weak[1:])[0, 1]
        strong_ac = np.corrcoef(strong[:-1], strong[1:])[0, 1]
        self.assertGreater(strong_ac, weak_ac + 0.4)

    def test_garch_is_finite(self):
        values = garch11(500, seed=3)
        self.assertTrue(np.isfinite(values).all())

    def test_regime_switching_returns_regimes(self):
        values, regimes = regime_switching_ar(500, 0.1, 0.8, 0.1, seed=4)
        self.assertEqual(values.shape, regimes.shape)
        self.assertTrue(set(np.unique(regimes)).issubset({0, 1}))

    def test_default_sweep_has_multiple_processes(self):
        names = {spec.process for spec in default_snr_sweep(length=128)}
        self.assertIn("white_noise", names)
        self.assertIn("ar1", names)
        self.assertIn("seasonal_ar", names)


if __name__ == "__main__":
    unittest.main()
