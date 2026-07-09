import unittest

import context  # noqa: F401
from low_snr_tsfm.failure_selection import representative_cases


def metric_row(name, rer, *, failed=1, flat=0.0, fvr=0.0, par=0.0, coverage=0.9, spike=1.0, regime="low"):
    return {
        "dataset": f"toy/{name}/short",
        "model": "chronos_bolt_small",
        "baseline": "naive",
        "series_id": name,
        "origin": "10",
        "window_index": name,
        "relative_error_ratio": str(rer),
        "failure_delta_005": str(failed),
        "flatness_score": str(flat),
        "forecast_variance_ratio": str(fvr),
        "prediction_amplitude_ratio": str(par),
        "empirical_coverage_90": str(coverage),
        "spike_recall": str(spike),
        "over_smoothing": "1" if flat > 0.5 else "0",
        "excess_variance": "1" if fvr > 1.0 or par > 1.0 else "0",
        "regime": regime,
        "mae": str(rer),
    }


class FailureSelectionTests(unittest.TestCase):
    def test_representative_cases_cover_predefined_selectors(self):
        rows = [
            metric_row("a", 1.2, flat=0.2, fvr=0.5, coverage=0.8, spike=0.5),
            metric_row("b", 3.0, flat=0.1, fvr=0.2, coverage=0.7, spike=0.4),
            metric_row("c", 1.8, flat=0.95, fvr=0.1, coverage=0.6, spike=0.3),
            metric_row("d", 1.6, flat=0.0, fvr=4.0, par=3.0, coverage=0.9, spike=0.2),
            metric_row("e", 1.4, flat=0.0, fvr=0.0, coverage=0.1, spike=0.8),
            metric_row("win", 0.4, failed=0, coverage=1.0, regime="seasonal_high_structure"),
        ]
        cases = representative_cases(rows)
        by_selector = {row["selector"]: row for row in cases}

        self.assertEqual(by_selector["largest_relative_error_failure"]["series_id"], "b")
        self.assertEqual(by_selector["largest_absolute_error_gap_failure"]["series_id"], "b")
        self.assertEqual(by_selector["largest_over_smoothing_failure"]["series_id"], "c")
        self.assertEqual(by_selector["largest_excess_variance_failure"]["series_id"], "d")
        self.assertEqual(by_selector["worst_coverage_failure"]["series_id"], "e")
        self.assertEqual(by_selector["strongest_positive_control_win"]["series_id"], "win")
        self.assertIn("median_relative_error_failure", by_selector)


if __name__ == "__main__":
    unittest.main()
