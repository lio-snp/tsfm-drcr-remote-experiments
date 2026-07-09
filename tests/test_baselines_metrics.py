import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.baselines import (
    linear_ar_forecast,
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    zero_forecast,
)
from low_snr_tsfm.metrics import (
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    mean_weighted_quantile_loss,
    pinball_loss,
    prediction_amplitude_ratio,
    sample_crps,
    spike_recall,
)


class BaselineMetricTests(unittest.TestCase):
    def test_basic_baselines(self):
        context = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(zero_forecast(context, 3), [0, 0, 0])
        np.testing.assert_allclose(mean_forecast(context, 2), [2.5, 2.5])
        np.testing.assert_allclose(naive_forecast(context, 2), [4, 4])
        np.testing.assert_allclose(seasonal_naive_forecast(context, 5, 2), [3, 4, 3, 4, 3])

    def test_linear_ar_forecast_shape(self):
        context = np.sin(np.arange(50) / 5)
        pred = linear_ar_forecast(context, horizon=7, lags=5)
        self.assertEqual(pred.shape, (7,))
        self.assertTrue(np.isfinite(pred).all())

    def test_accuracy_metrics(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 1.0, 5.0])
        self.assertAlmostEqual(mae(y, f), 1.0)

    def test_degradation_metrics(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        flat = np.array([0.5, 0.5, 0.5, 0.5])
        noisy = np.array([0.0, 3.0, -3.0, 3.0])
        self.assertGreater(flatness_score(y, flat), 0.9)
        self.assertGreater(prediction_amplitude_ratio(y, noisy), 2.0)
        self.assertGreater(forecast_variance_ratio(y, noisy), 4.0)
        self.assertEqual(spike_recall(y, y, k=1), 1.0)

    def test_probabilistic_metrics(self):
        y = np.array([0.0, 1.0])
        self.assertEqual(empirical_coverage(y, np.array([-1.0, 0.5]), np.array([0.5, 2.0])), 1.0)
        self.assertAlmostEqual(pinball_loss(y, np.array([0.0, 0.0]), tau=0.5), 0.25)
        quantiles = np.asarray([[-1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]], dtype=float)
        self.assertGreaterEqual(mean_weighted_quantile_loss(y, quantiles, [0.0, 0.1, 0.9, 1.0]), 0.0)
        samples = np.array([[-1.0, 0.0], [1.0, 2.0], [0.0, 1.0]])
        self.assertGreaterEqual(sample_crps(samples, y), 0.0)


if __name__ == "__main__":
    unittest.main()
