import importlib.util
import unittest
from pathlib import Path


def load_pooled_predictor_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_pooled_failure_predictor.py"
    spec = importlib.util.spec_from_file_location("build_pooled_failure_predictor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load build_pooled_failure_predictor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PooledFailurePredictorTests(unittest.TestCase):
    def test_within_domain_grouped_holdout_detects_local_signal(self):
        module = load_pooled_predictor_module()
        rows = []
        for series_id in ["s1", "s2", "s3"]:
            for value, label in [(0.0, 0), (1.0, 1)]:
                row = {
                    "domain": "TestDomain",
                    "source": "synthetic",
                    "dataset": "toy",
                    "series_id": series_id,
                    "failure_delta_005": label,
                    "relative_error_ratio": 1.0 + value,
                }
                for feature in module.FEATURE_NAMES:
                    row[feature] = 0.0
                row["trend_strength"] = value
                rows.append(row)

        result = module.within_domain_grouped_holdout(rows, ["TestDomain"])

        self.assertEqual(result[0]["status"], "ok")
        self.assertEqual(result[0]["n_groups"], 3)
        self.assertEqual(result[0]["threshold_balanced_accuracy"], 1.0)
        self.assertGreaterEqual(result[0]["logistic_balanced_accuracy"], 0.95)
        self.assertEqual(result[0]["tree_balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
