import unittest

import context  # noqa: F401
from low_snr_tsfm.gift_eval_results import (
    build_failure_rows,
    shared_failure_rows,
    summarize_failures,
)


def row(dataset, key, family, mase, domain="Test"):
    return {
        "dataset": dataset,
        "local_model_key": key,
        "model_family": family,
        "model": key,
        "domain": domain,
        "eval_metrics/MASE[0.5]": str(mase),
        "eval_metrics/mean_weighted_sum_quantile_loss": str(mase / 10),
    }


class GiftEvalResultsTests(unittest.TestCase):
    def test_build_failure_rows_uses_best_available_baseline(self):
        rows = [
            row("toy/A/short", "naive", "Baseline", 2.0),
            row("toy/A/short", "auto_arima", "Baseline", 1.0),
            row("toy/A/short", "timesfm_2_5", "TimesFM", 1.2),
        ]
        failures = build_failure_rows(
            rows,
            model_keys={"timesfm_2_5"},
            baseline_keys={"naive", "auto_arima"},
            dataset_to_regime={"toy/A/short": "locked"},
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["best_baseline"], "auto_arima")
        self.assertAlmostEqual(failures[0]["mase_relative_error_ratio"], 1.2)
        self.assertEqual(failures[0]["failure_delta_005"], 1)
        self.assertEqual(failures[0]["locked_slice"], 1)

    def test_shared_failure_rows_requires_distinct_families(self):
        rows = [
            row("toy/A/short", "naive", "Baseline", 1.0),
            row("toy/A/short", "timesfm_2_5", "TimesFM", 1.2),
            row("toy/A/short", "moirai2", "Moirai", 1.4),
            row("toy/B/short", "naive", "Baseline", 1.0),
            row("toy/B/short", "timesfm_2_5", "TimesFM", 1.3),
        ]
        failures = build_failure_rows(
            rows,
            model_keys={"timesfm_2_5", "moirai2"},
            baseline_keys={"naive"},
        )
        shared = shared_failure_rows(failures, min_failed_families=2)
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["dataset"], "toy/A/short")
        self.assertEqual(shared[0]["n_failed_families"], 2)

    def test_summarize_failures_reports_rates(self):
        rows = [
            row("toy/A/short", "naive", "Baseline", 1.0),
            row("toy/A/short", "moirai2", "Moirai", 1.4),
            row("toy/B/short", "naive", "Baseline", 1.0),
            row("toy/B/short", "moirai2", "Moirai", 0.9),
        ]
        failures = build_failure_rows(
            rows,
            model_keys={"moirai2"},
            baseline_keys={"naive"},
        )
        summary = summarize_failures(failures)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_datasets"], 2)
        self.assertAlmostEqual(summary[0]["failure_rate_delta_005"], 0.5)


if __name__ == "__main__":
    unittest.main()
