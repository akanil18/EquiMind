import unittest
from equimind.quantitative.alpha_lab import (
    AlphaResearchLab,
    AlphaFactor,
    FactorCategory,
)


class TestAlphaResearchLab(unittest.TestCase):

    def setUp(self):
        self.factor_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.fwd_returns = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        self.random_returns = [0.05, -0.02, 0.01, -0.04, 0.02, -0.01, 0.03, -0.05, 0.01, -0.02]

    def test_ic_and_rank_ic_calculation(self):
        ic = AlphaResearchLab.calculate_ic(self.factor_vals, self.fwd_returns)
        rank_ic = AlphaResearchLab.calculate_rank_ic(self.factor_vals, self.fwd_returns)

        self.assertEqual(ic, 1.0)
        self.assertEqual(rank_ic, 1.0)

    def test_alpha_factor_evaluation(self):
        factor = AlphaResearchLab.evaluate_alpha_factor(
            factor_name="DeveloperVelocityAlpha",
            category=FactorCategory.DEVELOPER_VELOCITY,
            description="GitHub commit velocity 30d acceleration",
            factor_values=self.factor_vals,
            forward_returns=self.fwd_returns,
            half_life_days=21.0,
        )

        self.assertIsInstance(factor, AlphaFactor)
        self.assertEqual(factor.name, "DeveloperVelocityAlpha")
        self.assertEqual(factor.information_coefficient, 1.0)
        self.assertTrue(factor.is_statistically_significant)

    def test_alpha_factor_ranking(self):
        f1 = AlphaResearchLab.evaluate_alpha_factor(
            "StrongAlpha", FactorCategory.MOMENTUM, "Strong signal", self.factor_vals, self.fwd_returns
        )
        f2 = AlphaResearchLab.evaluate_alpha_factor(
            "WeakAlpha", FactorCategory.VALUE, "Weak signal", self.factor_vals, self.random_returns
        )

        ranked = AlphaResearchLab.rank_alpha_factors([f2, f1])
        self.assertEqual(ranked[0].name, "StrongAlpha")
        self.assertEqual(ranked[1].name, "WeakAlpha")


if __name__ == "__main__":
    unittest.main()
