"""Regression tests for user-provided gradebooks and the two upload workflows."""

import io
import json
import unittest
from pathlib import Path

from openpyxl import Workbook
from streamlit.testing.v1 import AppTest

from grade_forecast.model import DEFAULT_ARTIFACTS, load_bundle, predict
from grade_forecast.uploads import (
    excel_sheets, guess_columns, rating_forecasts, read_grade_file,
    results_csv, sample_file, single_transcript_columns, transcript_history, weekly_forecasts,
)


class GradebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.weekly = json.loads((DEFAULT_ARTIFACTS / "weekly.json").read_text(encoding="utf-8-sig"))

    def test_cp1251_csv_stages_and_invalid_rows(self):
        data = ("Предмет;Р1;Р2;Экзамен\nАлгебра;78,5;;0\n\nФизика;90;95;1\n"
                "Химия;74;н.п.;50\nМеханика;0;0;99\n").encode("cp1251")
        frame = read_grade_file("grades.csv", data)
        output = rating_forecasts(frame, guess_columns(frame, ["course", "r1", "r2"]), self.bundle)
        self.assertEqual(output["Строка"].tolist(), [2, 4, 5, 6])
        self.assertAlmostEqual(output.at[0, "Итог"], predict(self.bundle, 78.5)["final_grade"])
        self.assertAlmostEqual(output.at[1, "Экзамен"], predict(self.bundle, 90, 95)["exam_score"])
        self.assertIn("Р2", output.at[2, "Статус"])
        self.assertEqual(output.at[3, "Статус"], "Средний рейтинг ниже 50")
        self.assertTrue(output.loc[[2, 3], "Итог"].isna().all())

    def test_distinguishes_rating_and_control_columns(self):
        frame = read_grade_file("test.csv", "РК1,РК2,Итог\n80,75,77".encode())
        mapping = guess_columns(frame, ["r1", "r2"])
        self.assertIsNone(mapping["r1"])
        with self.assertRaisesRegex(ValueError, "Р1"):
            rating_forecasts(frame, mapping, self.bundle)

    def test_xlsx_sheet_selection_and_export(self):
        book = Workbook()
        book.active.title = "Описание"
        second = book.create_sheet("Ведомость")
        second.append(["Дисциплина", "Р1", "Р2"])
        second.append(["+test", 77.5, None])
        buffer = io.BytesIO()
        book.save(buffer)
        book.close()
        payload = buffer.getvalue()
        self.assertEqual(excel_sheets(payload), ["Описание", "Ведомость"])
        frame = read_grade_file("marks.XLSX", payload, "Ведомость")
        output = rating_forecasts(frame, guess_columns(frame, ["course", "r1", "r2"]), self.bundle)
        exported = results_csv(output).decode("utf-8-sig")
        self.assertIn("'+test", exported)
        self.assertEqual(output.at[0, "Дисциплина"], "+test")
        self.assertAlmostEqual(output.at[0, "Итог"], predict(self.bundle, 77.5)["final_grade"])

    def test_file_errors(self):
        for name, payload in [("marks.xlsx", b"broken"), ("marks.csv", b"r1;r1\n4;8"),
                              ("marks.csv", b""), ("marks.csv", b"r1\n"), ("marks.xls", b"old")]:
            with self.subTest(name=name, payload=payload), self.assertRaises(ValueError):
                read_grade_file(name, payload)

    def test_weekly_grades_cutoff_and_duplicate(self):
        data = sample_file(weekly=True).decode("utf-8-sig")
        data += "Химия;1;75;;;;\nХимия;1;80;;;;\n"
        frame = read_grade_file("weekly.csv", data.encode())
        mapping = guess_columns(frame, ["course", "week", "lecture", "lab", "self_work", "mc1", "mc2"])
        before = weekly_forecasts(frame, mapping, self.weekly, 3)
        self.assertEqual(before.at[0, "Отметок"], 8)
        self.assertIn("две записи", before.at[1, "Статус"])
        frame.loc[frame["Неделя"] == "15", "Лекции"] = "неизвестно"
        after = weekly_forecasts(frame, mapping, self.weekly, 3)
        self.assertEqual(before.at[0, "Итог"], after.at[0, "Итог"])
        all_weeks = weekly_forecasts(frame, mapping, self.weekly, 15)
        self.assertIn("неделю 15", all_weeks.at[0, "Статус"])

    def test_history_blanks_and_results_match_manual(self):
        if not self.bundle.get("history_stages"):
            self.skipTest("No history model in this artifact")
        frame = read_grade_file("history.csv", sample_file(history=True))
        mapping = guess_columns(frame, ["course", "r1", "r2", "prior_average"])
        result = rating_forecasts(frame, mapping, self.bundle)
        self.assertEqual(result.at[0, "История учтена"], "Нет")
        self.assertEqual(result.at[1, "История учтена"], "Да")
        self.assertAlmostEqual(result.at[1, "Итог"], predict(self.bundle, 85, 90, prior_average=83)["final_grade"])

    def test_transcript_uses_only_completed_prior_terms_and_all_their_known_finals(self):
        data = ("course,semester,status,r1,final_grade,prior_average\n"
                "Future,3,completed,90,100,\nFirst,1,completed,,60,\n"
                "Second,1,completed,80,80,\nNever sat,1,in_progress,40,0,\n"
                "Current,2,in_progress,75,95,\nSame term,2,completed,71,71,65\n")
        frame = read_grade_file("transcript.csv", data.encode())
        history = transcript_history(frame)
        self.assertTrue(history.loc[[3, 4, 5]].isna().all())
        self.assertEqual(history.at[6], 70)
        self.assertAlmostEqual(history.at[2], (60 + 80 + 71) / 3)
        mapping = guess_columns(frame, ["course", "r1", "r2", "prior_average"])
        forecasts = rating_forecasts(frame, mapping, self.bundle, derived_history=history)
        self.assertEqual(forecasts.at[4, "Средний прошлых семестров"], 70)
        self.assertEqual(forecasts.at[5, "Средний прошлых семестров"], 65)
        frame.loc[2, "final_grade"] = 0
        self.assertEqual(transcript_history(frame).at[6], 70)
        frame["student_id"] = ["Second", "One", "One", "One", "One", "One"]
        self.assertIsNone(single_transcript_columns(frame))
        with self.assertRaisesRegex(ValueError, "одного студента"):
            transcript_history(frame)

    def test_streamlit_switches_between_samples_and_manual(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
        app.toggle(key="demo_False").set_value(True).run()
        self.assertEqual(len(app.dataframe[0].value), 3)
        app.radio(key="upload_mode").set_value("Недельные оценки").run()
        app.toggle(key="demo_True").set_value(True).run()
        app.radio(key="upload_horizon").set_value(15).run()
        self.assertEqual(app.dataframe[0].value.at[0, "Статус"], "Прогноз готов")
        app.radio(key="section_nav").set_value("Ручной ввод").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.number_input)
        if self.bundle.get("history_stages"):
            next(e for e in app.number_input if e.label == "Средний балл прошлых семестров").set_value(80).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Учтён средний" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
