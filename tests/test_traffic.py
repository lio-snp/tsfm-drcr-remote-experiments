import tempfile
import unittest
from pathlib import Path

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.traffic import (
    TrafficRegimeConfig,
    historical_conditional_forecast,
    historical_conditional_samples,
    label_traffic_regime,
    load_traffic_matrix,
)


class TrafficTests(unittest.TestCase):
    def test_label_traffic_regime(self):
        cfg = TrafficRegimeConfig(low_speed=30.0, high_speed=55.0, transition_range=15.0)
        self.assertEqual(label_traffic_regime(np.array([60.0, 62.0]), cfg), "free_flow")
        self.assertEqual(label_traffic_regime(np.array([20.0, 25.0]), cfg), "congested")
        self.assertEqual(label_traffic_regime(np.array([25.0, 60.0]), cfg), "transition")

    def test_historical_conditional_samples_shape(self):
        context = np.arange(24.0)
        samples = historical_conditional_samples(context, horizon=3, period=6, max_samples=3)
        self.assertEqual(samples.shape, (3, 3))
        np.testing.assert_allclose(samples[-1], [18.0, 19.0, 20.0])

    def test_historical_conditional_forecast(self):
        forecast = historical_conditional_forecast(np.arange(24.0), horizon=2, period=6)
        self.assertIn("mean", forecast)
        self.assertIn("q10", forecast)
        self.assertEqual(forecast["mean"].shape, (2,))

    def test_load_npz_time_sensor_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traffic.npz"
            np.savez(path, data=np.arange(30.0).reshape(10, 3))
            matrix = load_traffic_matrix(path)
        self.assertEqual(matrix.shape, (10, 3))


if __name__ == "__main__":
    unittest.main()
