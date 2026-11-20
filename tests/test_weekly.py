"""Checks specific to the AIU journal notation and the temporal cutoff."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from grade_forecast.validation import COMPONENT_WEIGHTS, engagement_by_course, weekly_features


class WeeklyTests(unittest.TestCase):
    def test_an_absence_counts_and_an_unscheduled_week_does_not(self):
        marks = ["90", "н", "40", "н.п.", "70", "н.п.", "60"]
        row = {"course_id": "example", "activity_code": "SRSP"}
        row.update({f"week_{week:02}": value for week, value in enumerate(marks, start=1)})
        features = engagement_by_course(pd.DataFrame([row]), 7).iloc[0]
        self.assertEqual(features["eligible"], 5)
        self.assertEqual(features["recorded"], 5)
        self.assertEqual(features["absences"], 1)
        self.assertAlmostEqual(features["self_work_mean"], 260 / 5)

    def test_unposted_grade_is_missing(self):
        row = {"course_id": "example", "activity_code": "SRSP", "week_01": "90", "week_02": ""}
        features = engagement_by_course(pd.DataFrame([row]), 2).iloc[0]
        self.assertEqual(features["recorded"], 1)
        self.assertEqual(features["absences"], 0)
        self.assertEqual(features["eligible"], 2)
        self.assertEqual(features["self_work_mean"], 90)

    def test_later_grades_cannot_affect_the_first_checkpoint(self):
        row = {f"{activity}_week_{week:02}": 75 for activity in COMPONENT_WEIGHTS for week in range(1, 16)}
        row.update({"MC1": 70, "Rat1": 74, "MC2": 92, "Rat2": 87, "E": 60, "FG": 72})
        early, baseline, _ = weekly_features(pd.DataFrame([row]), 7)
        future = {f"{activity}_week_{week:02}": 0 for activity in COMPONENT_WEIGHTS for week in range(8, 16)}
        future.update({"MC2": 0, "Rat2": 0, "E": 0, "FG": 0})
        changed, changed_baseline, _ = weekly_features(pd.DataFrame([{**row, **future}]), 7)
        assert_frame_equal(early, changed)
        assert_series_equal(baseline, changed_baseline)


if __name__ == "__main__":
    unittest.main()
