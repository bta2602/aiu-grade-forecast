"""Regression checks for stage boundaries and the same predictions served by the UI."""

import unittest

import pandas as pd

from grade_forecast.history import previous_average
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
                with self.assertRaises(ValueError):
                    predict(self.bundle, 70, prior_average=invalid)

    def test_history_contains_only_the_students_own_earlier_courses(self):
        frame = pd.DataFrame(
            [(1, 3, 99), (1, 1, 50), (2, 1, 100), (1, 2, 80),
             (1, 1, 70), (2, 2, 60), (1, 3, 11)],
            columns=["student", "semester", "final"],
        )
        prior = previous_average(frame)
        self.assertTrue(prior.loc[[1, 2, 4]].isna().all())
        self.assertEqual(prior[3], 60)
        self.assertEqual(prior[5], 100)
        for index in (0, 6):
            self.assertAlmostEqual(prior[index], 200 / 3)

    def test_optional_history_and_contributions(self):
        for second in (None, 80):
            reference = predict(self.bundle, 70, second)
            result = predict(self.bundle, 70, second, prior_average=85)
            self.assertFalse(reference["history_used"])
            self.assertTrue(result["history_used"])
            explain = result["explanation"]
            self.assertIn("prior_average", explain["contributions"])
            self.assertAlmostEqual(explain["base"] + sum(explain["contributions"].values()),
                                   explain["raw_prediction"], places=5)
            without_history = {"stages": self.bundle["stages"]}
            self.assertEqual(predict(without_history, 70, second, prior_average=85), reference)

    def test_historical_average_does_not_change_formal_admission(self):
        fixed = {"name": "linear", "weights": [0, 0, 0], "intercept": 75, "center": [0, 0, 0]}
        bundle = {"history_stages": {"late": fixed}}
        low = predict(bundle, 40, 59, prior_average=100)
        self.assertEqual(low["status"], "rating_below_50")
        eligible = predict(bundle, 40, 60, prior_average=0)
        self.assertEqual(eligible["status"], "estimated")
        self.assertEqual(eligible["final_grade"], formula_grade(40, 60, 75))


if __name__ == "__main__":
    unittest.main()
