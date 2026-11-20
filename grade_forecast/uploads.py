"""Read grade tables in memory and apply the same predictors as the manual forms."""

import csv
import io
import math
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from xml.etree.ElementTree import ParseError

import pandas as pd

from grade_forecast.history import previous_average
from grade_forecast.model import predict
from grade_forecast.validation import COMPONENT_WEIGHTS, numeric_mark, predict_weekly


MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000
MAX_COLUMNS = 150
ALIASES = {
    "course": ["Дисциплина", "Предмет", "Курс", "course_name", "course", "course_id"],
    "r1": ["Р1", "R1", "Рейтинг 1", "Рейтинг за первый период", "Rat1", "Rating 1"],
    "r2": ["Р2", "R2", "Рейтинг 2", "Рейтинг за второй период", "Rat2", "Rating 2"],
    "prior_average": ["Средний балл прошлых семестров", "prior_average"],
    "week": ["Неделя", "Week"],
    "lecture": ["Лекции", "Лекция", "lecture"],
    "lab": ["Лабораторные", "Лабораторная", "lab"],
    "self_work": ["Самостоятельные", "Самостоятельная работа", "self_work", "СРС"],
    "mc1": ["РК1", "RK1", "MC1"],
    "mc2": ["РК2", "RK2", "MC2"],
    "semester": ["semester", "Семестр", "Номер семестра"],
    "status": ["status", "Статус", "Статус дисциплины"],
    "final": ["final_grade", "Итог", "Итоговая оценка"],
}
STUDENT_KEYS = ("student", "student_id", "student_code", "student_name", "student number",
                "student identifier", "learner_id", "Студент", "ID студента", "ИИН", "ФИО", "ФИО студента")
COMPLETED = {"completed", "завершен", "завершена", "завершено"}
STATUS = {
    "estimated": "Прогноз готов",
    "rating_below_50": "Средний рейтинг ниже 50",
    "exam_below_50": "Прогноз экзамена ниже 50",
}
RATING_SAMPLE = [
    ["Дисциплина", "Р1", "Р2"],
    ["Базы данных", 78, ""],
    ["Математический анализ", 85, 90],
    ["Физика", 64, 75],
]


def sample_file(weekly=False, history=False):
    if weekly:
        rows = [["Дисциплина", "Неделя", "Лекции", "Лабораторные", "Самостоятельные", "РК1", "РК2"]]
        for i in range(1, 16):
            rows.append(["Основы Java", i, {2: "н", 3: "н.п."}.get(i, 78 + i % 6),
                         72 + i, 75 + i % 5, 82 if i == 7 else "", 86 if i == 15 else ""])
    elif history:
        rows = [row + [value] for row, value in zip(
            RATING_SAMPLE, ["Средний балл прошлых семестров", "", 83, 81], strict=True
        )]
    else:
        rows = RATING_SAMPLE
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=";").writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _text(value):
    return "" if value is None else str(value).strip()


def _name(value):
    return re.sub(r"[\s_.()-]+", "", _text(value).casefold())


def guess_columns(frame, fields):
    available = {_name(column): column for column in frame.columns}
    return {field: next((available[_name(a)] for a in ALIASES[field] if _name(a) in available), None)
            for field in fields}


def _excel(payload):
    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException

    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
                raise ValueError("Лист слишком большой. Сохрани нужные столбцы отдельно.")
        return load_workbook(io.BytesIO(payload), read_only=True, data_only=True, keep_links=False)
    except (BadZipFile, InvalidFileException, OSError, KeyError, ParseError) as exc:
        raise ValueError("Не удалось прочитать книгу. Сохрани её в формате XLSX или CSV.") from exc


def _size(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError("Файл превышает 5 МБ. Выбери только лист или строки с нужными оценками.")
    if not payload:
        raise ValueError("Файл пуст.")


def excel_sheets(payload):
    _size(payload)
    book = _excel(payload)
    try:
        sheets = [sheet.title for sheet in book.worksheets if sheet.sheet_state == "visible"]
        if not sheets:
            raise ValueError("В книге нет видимых листов с данными.")
        return sheets
    finally:
        book.close()


def _table(rows):
    rows = iter(rows)
    header = [_text(h) for h in next(rows, [])]
    while header and not header[-1]:
        header.pop()
    if not header or len(header) > MAX_COLUMNS or any(not h for h in header):
        raise ValueError("В первой строке нужны названия всех столбцов (до 150).")
    if len(set(map(_name, header))) != len(header):
        raise ValueError("Два столбца имеют одинаковые имена. Переименуй один из них.")
    records, numbers = [], []
    for line, record in enumerate(rows, 2):
        if line > MAX_ROWS + 1:
            raise ValueError("Можно обработать до 5 000 строк за одну загрузку.")
        if not any(_text(value) for value in record):
            continue
        if any(_text(value) for value in record[len(header):]):
            raise ValueError(f"В строке {line} больше значений, чем названий столбцов.")
        values = list(record[:len(header)])
        records.append(values + [None] * (len(header) - len(values)))
        numbers.append(line)
    if not records:
        raise ValueError("Под заголовками пока нет оценок.")
    return pd.DataFrame(records, columns=header, index=numbers, dtype=object)


def read_grade_file(filename, payload, sheet=None):
    _size(payload)
    extension = Path(filename).suffix.lower()
    if extension == ".xlsx":
        book = _excel(payload)
        try:
            page = book[sheet] if sheet else book.worksheets[0]
            if (page.max_column or 0) > MAX_COLUMNS or (page.max_row or 0) > MAX_ROWS + 1:
                raise ValueError("Лист превышает 5 000 строк или 150 столбцов. Скопируй заполненную область на отдельный лист.")
            return _table(page.values)
        finally:
            book.close()
    if extension != ".csv":
        raise ValueError("Выбери файл CSV или XLSX.")
    try:
        raw = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            raw = payload.decode("cp1251")
        except UnicodeDecodeError as exc:
            raise ValueError("Сохрани CSV в кодировке UTF-8 или Windows-1251.") from exc
    first = raw.splitlines()[0] if raw.splitlines() else ""
    delimiter = max((";", ",", "\t"), key=lambda d: len(next(csv.reader([first], delimiter=d))))
    try:
        return _table(csv.reader(io.StringIO(raw, newline=""), delimiter=delimiter, strict=True))
    except csv.Error as exc:
        raise ValueError("Не удалось разобрать CSV. Проверь разделители и кавычки.") from exc


def _score(value, label):
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return numeric_mark(value)
    except ValueError as exc:
        raise ValueError(f"{label} должен содержать число от 0 до 100.") from exc


def _cell(row, mapping, field):
    column = mapping.get(field)
    return row[column] if column else None


def single_transcript_columns(frame):
    fields = guess_columns(frame, ("semester", "status", "final"))
    if not all(fields.values()):
        return None
    student_keys = set(map(_name, STUDENT_KEYS))
    for column in frame.columns:
        if _name(column) in student_keys:
            values = frame[column].map(_text)
            if values.eq("").any() or values.nunique() != 1:
                return None
    return fields


def transcript_history(frame):
    columns = single_transcript_columns(frame)
    if columns is None:
        raise ValueError("Для расчёта нужна ведомость одного студента со столбцами семестра, статуса и итога.")
    term = pd.to_numeric(frame[columns["semester"]], errors="coerce")
    if term.isna().any() or not (term.ge(0) & term.mod(1).eq(0)).all():
        raise ValueError("Укажи целый сквозной номер семестра в каждой строке ведомости.")
    year = next((c for c in frame if _name(c) in {"academicyear", "учебныйгод"}), None)
    if year and frame[year].map(_text).groupby(term).nunique().gt(1).any():
        raise ValueError("Номер семестра повторяется в разных учебных годах. Укажи сквозную нумерацию семестров.")
    completed = frame[columns["status"]].map(lambda s: _text(s).casefold().replace("ё", "е")).isin(COMPLETED)
    final = pd.Series(float("nan"), index=frame.index)
    for number, value in frame.loc[completed, columns["final"]].items():
        if _text(value):
            final.at[number] = _score(value, f"Итог в строке {number}")
    transcript = pd.DataFrame({"student": 1, "semester": term, "final": final})
    return previous_average(transcript)


def rating_forecasts(frame, mapping, bundle, *, derived_history=None):
    if not mapping.get("r1"):
        raise ValueError("Укажи столбец с рейтингом первого периода (Р1).")
    rows = []
    for number, row in frame.iterrows():
        course = _text(_cell(row, mapping, "course"))
        output = {"Строка": number, "Дисциплина": course, "Этап": None, "Итог": None, "Экзамен": None}
        try:
            r1 = _score(_cell(row, mapping, "r1"), "Р1")
            r2 = _cell(row, mapping, "r2")
            r2 = _score(r2, "Р2") if _text(r2) else None
            prior = _cell(row, mapping, "prior_average")
            kwargs = {}
            if _text(prior):
                kwargs["prior_average"] = _score(prior, "Средний балл прошлых семестров")
            elif derived_history is not None and pd.notna(derived_history.at[number]):
                kwargs["prior_average"] = float(derived_history.at[number])
            result = predict(bundle, r1, r2, **kwargs)
            output.update({"Этап": "После РК1" if r2 is None else "После РК2",
                           "Итог": result["final_grade"], "Экзамен": result["exam_score"],
                           "Статус": STATUS[result["status"]]})
            if mapping.get("prior_average") or derived_history is not None:
                output["История учтена"] = "Да" if result.get("history_used") else "Нет"
                output["Средний прошлых семестров"] = kwargs.get("prior_average")
        except ValueError as exc:
            output["Статус"] = str(exc)
        rows.append(output)
    return pd.DataFrame(rows)


def weekly_forecasts(frame, mapping, report, horizon):
    if not mapping.get("course") or not mapping.get("week"):
        raise ValueError("Укажи столбцы с дисциплиной и номером недели.")
    if not any(mapping.get(k) for k in COMPONENT_WEIGHTS):
        raise ValueError("Укажи хотя бы один столбец с оценками за занятия.")
    for field, week in (("mc1", 7), ("mc2", 15)):
        if horizon >= week and not mapping.get(field):
            raise ValueError(f"Для этого прогноза нужен столбец с оценкой за РК{1 if week == 7 else 2}.")
    results = []
    for course, group in frame.groupby(frame[mapping["course"]].map(_text), sort=False):
        output = {"Строка": int(group.index[0]), "Дисциплина": course, "Этап": f"Неделя {horizon}",
                  "Итог": None, "Отметок": None}
        try:
            if not course:
                raise ValueError("Укажи название дисциплины.")
            marks = {activity: [None] * horizon for activity in COMPONENT_WEIGHTS}
            written = set()
            controls = {}
            for number, row in group.iterrows():
                try:
                    week = float(_text(_cell(row, mapping, "week")).replace(",", "."))
                    if not math.isfinite(week) or not week.is_integer() or not 1 <= week <= 15:
                        raise ValueError
                except ValueError:
                    raise ValueError(f"Строка {number}. Нужен номер недели от 1 до 15.") from None
                week = int(week)
                if week > horizon:
                    continue
                if week in written:
                    raise ValueError(f"Строка {number}. У дисциплины две записи за неделю {week}.")
                written.add(week)
                for activity in COMPONENT_WEIGHTS:
                    value = _text(_cell(row, mapping, activity)).lower()
                    if value not in ("", "н", "н.п.", "a/np"):
                        value = _score(value, f"Балл «{ALIASES[activity][0]}» за неделю {week}")
                    marks[activity][week - 1] = value
                if week in (7, 15):
                    field = "mc1" if week == 7 else "mc2"
                    controls[field] = _score(_cell(row, mapping, field), ALIASES[field][0])
            for field, week in (("mc1", 7), ("mc2", 15)):
                if horizon >= week and field not in controls:
                    raise ValueError(f"Добавь строку за неделю {week} с баллом за {ALIASES[field][0]}.")
            result = predict_weekly(report, horizon, marks, **controls)
            output["Итог"], output["Отметок"] = result["prediction"], result["observations"]
            output["Статус"] = "Прогноз готов" if result["prediction"] is not None else "Нет оценок к выбранной неделе"
        except ValueError as exc:
            output["Статус"] = str(exc)
        results.append(output)
    return pd.DataFrame(results)


def results_csv(results):
    # An uploaded name may resemble a spreadsheet formula when the export is opened.
    safe = results.copy()
    safe["Дисциплина"] = safe["Дисциплина"].map(
        lambda name: "'" + name if name.lstrip().startswith(("=", "+", "-", "@")) else name
    )
    return safe.to_csv(index=False, sep=";", decimal=",", float_format="%.2f").encode("utf-8-sig")
