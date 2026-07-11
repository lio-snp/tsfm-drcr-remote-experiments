import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.benefit_selective import frozen_action_grids, pre_origin_feature_vector


ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)


class BenefitSelectiveActionParityTests(unittest.TestCase):
    def test_frozen_actions_match_historical_rows_across_families(self) -> None:
        protocol = json.loads((ROOT / "configs/benefit_selective_drcr_external_protocol.json").read_text())
        with (ROOT / "results/aaai_stress/remote_q9_final_main_replacements.csv").open(newline="") as handle:
            replacements = list(csv.DictReader(handle))
        with (ROOT / "results/aaai_stress/final_main_figure_windows.csv").open(newline="") as handle:
            final_rows = list(csv.DictReader(handle))
        selected = {}
        for replacement in replacements:
            if replacement["ready_for_final_main_refresh"] == "1":
                selected.setdefault(replacement["family"], replacement)
        chronos_source = "chronos_bolt_small_fullgrid9_scaling_covid_deaths_d_short_auto_ets"
        selected["chronos"] = {
            "rerun_slug": chronos_source,
            "raw_path": f"results/raw_forecasts/{chronos_source}.csv",
            "history_path": f"results/raw_forecasts/{chronos_source}_history_context.csv",
        }
        self.assertEqual(set(selected), {"chronos", "moirai", "timesfm"})

        levels = [level / 10 for level in range(1, 10)]
        for family, replacement in selected.items():
            with (ROOT / replacement["raw_path"]).open(newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with (ROOT / replacement["history_path"]).open(newline="") as handle:
                history = next(csv.DictReader(handle))
            rows = [
                row for row in raw_rows
                if row["series_id"] == history["series_id"] and row["window_index"] == history["window_index"]
            ]
            rows.sort(key=lambda row: int(row["horizon_index"]))
            grid = np.asarray(
                [[float(row[f"forecast_q{level:02d}"]) for level in range(10, 100, 10)] for row in rows]
            )
            actual = np.asarray([float(row["actual"]) for row in rows])
            native_mean = np.asarray([float(row["forecast_mean"]) for row in rows])
            baseline = np.asarray([float(row["baseline_forecast"]) for row in rows])
            context_values = json.loads(history["context_values"])
            history_context = np.asarray(
                [np.nan if value is None else float(value) for value in context_values], dtype=float
            )
            features = pre_origin_feature_vector(
                history_context,
                len(rows),
                levels,
                grid,
                family,
                protocol["method"]["smooth_interval_head"],
            )
            grids, _ = frozen_action_grids(
                native_mean,
                baseline,
                levels,
                grid,
                features,
                protocol["low_structure_taxonomy"],
                protocol["method"],
            )
            for action in protocol["method"]["repair_actions"]:
                expected = next(
                    row for row in final_rows
                    if row["source"] == replacement["rerun_slug"]
                    and row["series_id"] == history["series_id"]
                    and row["window_index"] == history["window_index"]
                    and row["candidate_id"] == action
                )
                action_grid = grids[action]
                errors = actual[:, None] - action_grid
                taus = np.asarray(levels)[None, :]
                pinball = float(np.sum(np.maximum(taus * errors, (taus - 1.0) * errors)))
                repair_mae = float(np.mean(np.abs(actual - action_grid[:, 4])))
                coverage = float(np.mean((actual >= action_grid[:, 0]) & (actual <= action_grid[:, -1])))
                self.assertAlmostEqual(pinball, float(expected["repair_pinball_sum"]), places=8)
                self.assertAlmostEqual(repair_mae, float(expected["repair_mae"]), places=10)
                self.assertAlmostEqual(coverage, float(expected["repair_coverage_q10_q90"]), places=12)


if __name__ == "__main__":
    unittest.main()
