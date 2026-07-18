from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "aaai_stress" / "benefit_selective_external_recovery_jobs.csv"


class BenefitSelectiveExternalRecoveryTest(unittest.TestCase):
    def test_recovery_manifest_is_frozen_subset(self) -> None:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 33)
        self.assertEqual(len({row["job_id"] for row in rows}), 33)
        self.assertEqual(
            Counter(row["category"] for row in rows),
            {
                "timeout": 15,
                "baseline_infeasible": 3,
                "timesfm_nonfinite": 5,
                "runner_complete_contract_mismatch": 9,
                "outcome_invalid": 1,
            },
        )
        self.assertTrue(all(row["force_required"] == "1" for row in rows))
        self.assertTrue(all("--job-id" in row["recovery_command"] for row in rows))
        self.assertTrue(all("--force" in row["recovery_command"] for row in rows))


if __name__ == "__main__":
    unittest.main()
