import unittest

import numpy as np

from scripts.evaluate_benefit_selective_external import mase_scale


class BenefitSelectiveExternalEvaluatorTests(unittest.TestCase):
    def test_mase_scale_matches_seasonal_naive_differences(self) -> None:
        context = np.asarray([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        self.assertAlmostEqual(mase_scale(context, 2), 1.0)

    def test_mase_scale_ignores_missing_context_values(self) -> None:
        context = np.asarray([1.0, np.nan, 2.0, 3.0])
        self.assertAlmostEqual(mase_scale(context, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
