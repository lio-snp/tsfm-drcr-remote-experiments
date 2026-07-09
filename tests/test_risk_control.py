import unittest

import numpy as np

import context  # noqa: F401
from low_snr_tsfm.risk_control import (
    binomial_lower_tail_p_value,
    hoeffding_upper_bound,
    ltt_risk_tests,
    select_highest_utility_certified,
)


class RiskControlTests(unittest.TestCase):
    def test_binomial_tail_certifies_zero_harm_with_enough_samples(self):
        p_value = binomial_lower_tail_p_value(0, 80, 0.10)
        self.assertLess(p_value, 0.001)

    def test_hoeffding_upper_bound_is_in_unit_interval(self):
        bound = hoeffding_upper_bound(0.05, 100, 0.1)
        self.assertGreaterEqual(bound, 0.05)
        self.assertLessEqual(bound, 1.0)

    def test_ltt_accepts_low_risk_and_rejects_high_risk_policy(self):
        losses = {
            "safe": np.zeros(90),
            "risky": np.r_[np.ones(30), np.zeros(60)],
        }
        tests = {row.policy_id: row for row in ltt_risk_tests(losses, alpha=0.10, delta=0.10)}
        self.assertTrue(tests["safe"].accepted)
        self.assertFalse(tests["risky"].accepted)

    def test_select_highest_utility_among_certified(self):
        losses = {
            "fallback": np.zeros(90),
            "useful": np.zeros(90),
            "risky": np.r_[np.ones(25), np.zeros(65)],
        }
        tests = ltt_risk_tests(losses, alpha=0.10, delta=0.10)
        selected = select_highest_utility_certified(
            tests,
            {"fallback": 0.0, "useful": 0.2, "risky": 0.5},
            fallback_policy_id="fallback",
        )
        self.assertEqual(selected, "useful")


if __name__ == "__main__":
    unittest.main()
