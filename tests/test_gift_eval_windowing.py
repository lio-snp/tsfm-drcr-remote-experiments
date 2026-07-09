import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.gift_eval_windowing import (
    canonical_freq_unit,
    forward_fill_nan,
    prediction_length,
    test_windows,
    window_count,
)


class GiftEvalWindowingTests(unittest.TestCase):
    def test_prediction_length_matches_gift_eval_maps(self):
        self.assertEqual(canonical_freq_unit("10S"), "S")
        self.assertEqual(canonical_freq_unit("5min"), "T")
        self.assertEqual(prediction_length("bizitobs_application", "10S", "short"), 60)
        self.assertEqual(prediction_length("m4_hourly", "H", "short"), 48)

    def test_window_count_uses_gift_eval_bounds(self):
        self.assertEqual(window_count(100, 60), 1)
        self.assertEqual(window_count(12_000, 60), 20)

    def test_test_windows_are_last_non_overlapping_windows(self):
        values = np.arange(20, dtype=float)
        windows = test_windows(values, pred_length=4, windows=2)
        self.assertEqual([w.origin for w in windows], [12, 16])
        np.testing.assert_array_equal(windows[0].target, [12, 13, 14, 15])
        np.testing.assert_array_equal(windows[1].target, [16, 17, 18, 19])

    def test_forward_fill_nan_handles_leading_and_internal_gaps(self):
        values = forward_fill_nan(np.array([np.nan, 1.0, np.nan, 3.0]))
        np.testing.assert_array_equal(values, [1.0, 1.0, 1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
