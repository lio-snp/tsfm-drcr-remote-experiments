import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.repair import adaptive_reference_weight, blended_interval, convex_mixture, hull_interval


class RepairTests(unittest.TestCase):
    def test_adaptive_reference_weight_respects_gate_and_cap(self):
        self.assertEqual(adaptive_reference_weight(0.25, 1, 2, 0.25, 0.75), 0.0)
        self.assertEqual(adaptive_reference_weight(0.25, 2, 2, 0.25, 0.75), 0.25)
        self.assertEqual(adaptive_reference_weight(0.25, 3, 2, 0.25, 0.75), 0.50)
        self.assertEqual(adaptive_reference_weight(0.25, 8, 2, 0.25, 0.75), 0.75)

    def test_adaptive_reference_weight_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            adaptive_reference_weight(1.1, 2, 2)
        with self.assertRaises(ValueError):
            adaptive_reference_weight(0.5, 2, 2, -0.1)
        with self.assertRaises(ValueError):
            adaptive_reference_weight(0.5, 2, 2, 0.1, 1.1)

    def test_convex_mixture(self):
        model = np.array([0.0, 2.0])
        reference = np.array([2.0, 0.0])
        np.testing.assert_allclose(convex_mixture(model, reference, 0.25), [0.5, 1.5])

    def test_convex_mixture_rejects_bad_weight(self):
        with self.assertRaises(ValueError):
            convex_mixture(np.array([1.0]), np.array([2.0]), 1.1)

    def test_blended_interval_keeps_order(self):
        lower = np.array([0.0, 10.0])
        upper = np.array([2.0, 12.0])
        reference = np.array([10.0, 0.0])
        lo, hi = blended_interval(lower, upper, reference, 0.5)
        self.assertTrue(np.all(lo <= hi))

    def test_hull_interval_contains_reference_and_model_bounds(self):
        lower = np.array([0.0, 10.0])
        upper = np.array([2.0, 12.0])
        reference = np.array([5.0, 5.0])
        lo, hi = hull_interval(lower, upper, reference)
        np.testing.assert_allclose(lo, [0.0, 5.0])
        np.testing.assert_allclose(hi, [5.0, 12.0])


if __name__ == "__main__":
    unittest.main()
