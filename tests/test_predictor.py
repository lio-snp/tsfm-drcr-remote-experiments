import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.predictor import (
    apply_threshold,
    balanced_accuracy,
    best_threshold,
    evaluate_threshold,
    fit_depth2_threshold_tree,
    fit_balanced_logistic_regression,
    predict_depth2_threshold_tree,
    predict_logistic_regression,
)


class PredictorTests(unittest.TestCase):
    def test_balanced_accuracy(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(balanced_accuracy(labels, preds), 0.75)

    def test_best_threshold_and_evaluate(self):
        rows = [
            {"failure_delta_005": 0, "x": 0.1},
            {"failure_delta_005": 0, "x": 0.2},
            {"failure_delta_005": 1, "x": 0.8},
            {"failure_delta_005": 1, "x": 0.9},
        ]
        threshold = best_threshold(rows, ["x"])
        self.assertEqual(threshold["feature"], "x")
        metrics = evaluate_threshold(rows, threshold)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_apply_threshold_rejects_unknown_direction(self):
        with self.assertRaises(ValueError):
            apply_threshold(np.array([1.0]), 0.0, "bad")

    def test_numpy_logistic_regression_separates_simple_signal(self):
        features = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        model = fit_balanced_logistic_regression(features, labels, max_iter=1500)
        preds = predict_logistic_regression(model, features)
        self.assertGreaterEqual(balanced_accuracy(labels, preds), 0.95)

    def test_depth2_threshold_tree_separates_staged_pattern(self):
        rows = [
            {"failure_delta_005": 0, "x": 0.0, "z": 0.0},
            {"failure_delta_005": 1, "x": 0.0, "z": 1.0},
            {"failure_delta_005": 1, "x": 1.0, "z": 0.0},
            {"failure_delta_005": 1, "x": 1.0, "z": 1.0},
        ]
        model = fit_depth2_threshold_tree(rows, ["x", "z"])
        preds = predict_depth2_threshold_tree(model, rows)
        labels = np.asarray([row["failure_delta_005"] for row in rows])
        self.assertEqual(balanced_accuracy(labels, preds), 1.0)


if __name__ == "__main__":
    unittest.main()
