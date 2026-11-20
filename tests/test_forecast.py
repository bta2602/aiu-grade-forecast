"""Regression checks for stage boundaries and the same predictions served by the UI."""

import unittest

from grade_forecast.model import formula_grade, load_bundle, predict


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_early_forecast_uses_only_first_rating(self):
        estimate = predict(self.bundle, 74.5)
        self.assertEqual(estimate["stage"], "early")
        self.assertIsNone(estimate["exam_score"])
        self.assertEqual(set(estimate["explanation"]["contributions"]), {"r1"})
        self.assertLessEqual(0, estimate["final_grade"])
        self.assertLessEqual(estimate["final_grade"], 100)

    def test_explanations_reconstruct_predictions(self):
        for r1, r2 in [(0, None), (75.5, None), (70, 80), (100, 100)]:
            with self.subTest(r1=r1, r2=r2):
                result = predict(self.bundle, r1, r2)
                explanation = result["explanation"]
                self.assertAlmostEqual(
                    explanation["base"] + sum(explanation["contributions"].values()),
                    explanation["raw_prediction"], places=5,
                )
                if r2 is not None:
                    self.assertAlmostEqual(
                        result["formula_grade"], formula_grade(r1, r2, result["exam_score"])
                    )

    def test_admission_and_exam_thresholds(self):
        def fixed_exam(score):
            return {"stages": {"late": {"name": "linear", "weights": [0, 0],
                                        "intercept": score, "center": [70, 70]}}}

        for r1, r2, exam, status in [
            (40, 59, 75, "rating_below_50"),
            (60, 60, 49.99, "exam_below_50"),
            (40, 60, 50, "estimated"),
        ]:
            with self.subTest(r1=r1, r2=r2, exam=exam):
                result = predict(fixed_exam(exam), r1, r2)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["final_grade"] is None, status != "estimated")

    def test_invalid_numeric_inputs(self):
        for invalid in [True, -1, 101, float("inf"), float("nan"), "н.п."]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    predict(self.bundle, invalid)
                with self.assertRaises(ValueError):
                    predict(self.bundle, 70, invalid)


if __name__ == "__main__":
    unittest.main()
