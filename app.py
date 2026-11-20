"""Streamlit interface for the AIU grade forecasting project."""

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from grade_forecast.model import DEFAULT_ARTIFACTS, load_bundle, predict


ARTIFACTS = Path(DEFAULT_ARTIFACTS)
MODEL_NAMES = {
    "baseline": "Базовое правило",
    "linear": "Линейная регрессия",
    "catboost": "CatBoost",
    "ridge": "Гребневая регрессия",
}
STAGES = {"После РК1": "early", "После РК2": "late"}
ACCENT = "#167568"


def decimal(number, digits=1):
    return f"{number:,.{digits}f}".replace(",", " ").replace(".", ",")


def read_report(filename):
    return json.loads((ARTIFACTS / filename).read_text(encoding="utf-8-sig"))


@st.cache_resource(show_spinner=False)
def cached_bundle(file_stamp):
    return load_bundle(ARTIFACTS)


def result_card(label, value, note, unit=""):
    st.markdown(
        f"""
        <section class="forecast-card" aria-label="{escape(label)}">
          <div class="forecast-label">{escape(label)}</div>
          <div class="forecast-value">{escape(value)} <small>{escape(unit)}</small></div>
          <div class="forecast-note">{escape(note)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def forecast_page(bundle, report):
    controls, result = st.columns([1, 1.12], gap="large")
    with controls:
        stage_label = st.radio("Момент прогноза", list(STAGES), horizontal=True, key="stage")
        stage = STAGES[stage_label]
        st.caption("Введи рейтинги периода из журнала Platonus. Они учитывают текущие работы и рубежный контроль.")
        r1 = st.number_input("Рейтинг за первый период (Р1)", 0.0, 100.0, 75.0, 0.5, format="%.2f")
        r2 = None
        if stage == "late":
            r2 = st.number_input("Рейтинг за второй период (Р2)", 0.0, 100.0, 80.0, 0.5, format="%.2f")

    # The prediction code accepts only the inputs available at the selected date.
    prediction = predict(bundle, r1, r2)
    method = prediction["model"]
    stage_report = report["stages"][stage]
    winner_metrics = next(row for row in stage_report["models"] if row["name"] == method)
    error = winner_metrics["test"]["mae"]
    with result:
        if prediction["status"] == "estimated":
            note = (
                "По рейтингу первого периода. После второго контроля прогноз можно обновить."
                if stage == "early"
                else f"Прогноз экзамена — {decimal(prediction['exam_score'])} из 100. "
                "Итог рассчитан при успешной сдаче."
            )
            result_card("Прогноз итоговой оценки", decimal(prediction["final_grade"]), note, "/ 100")
        elif prediction["status"] == "rating_below_50":
            result_card("Допуск к экзамену", "Рейтинг ниже 50",
                        "Среднее введённых рейтингов ниже порога допуска. Официальный допуск указан в Platonus.")
        else:
            exam = prediction["exam_score"]
            shown_exam = "< 50" if round(exam, 1) >= 50 else decimal(exam)
            result_card(
                "Прогноз экзамена", shown_exam,
                "Ожидаемый балл ниже порога 50. По правилам AIU при FX/F итог по формуле не выставляется.",
                "/ 100",
            )
        target = "итоговой оценки" if stage == "early" else "экзамена"
        st.caption(f"{MODEL_NAMES.get(method, method)} · Средняя ошибка прогноза {target} на открытом тесте — {decimal(error, 2)} балла.")

    details = prediction.get("explanation")
    if details:
        with st.expander("Как получился прогноз"):
            target = "итоговой оценки" if stage == "early" else "экзамена"
            st.write(f"Модель начинает с базового прогноза {target}. Вклады рейтингов показывают, на сколько баллов каждый из них меняет оценку для введённых значений.")
            terms = [{"Слагаемое": "Базовый прогноз", "Баллы": float(details["base"])}]
            for name, value in details["contributions"].items():
                terms.append({"Слагаемое": {"r1": "Вклад Р1", "r2": "Вклад Р2"}.get(name, name), "Баллы": float(value)})
            terms.append({"Слагаемое": "Сумма до ограничения шкалы", "Баллы": float(details["raw_prediction"])})
            st.dataframe(pd.DataFrame(terms), hide_index=True, width="stretch",
                         column_config={"Баллы": st.column_config.NumberColumn(format="%.2f")})
            if stage == "late":
                st.markdown("При допуске и успешной сдаче экзамена AIU рассчитывает итог по формуле "
                            "**0,30 × Р1 + 0,30 × Р2 + 0,40 × экзамен**.")
                st.markdown("[Академическая политика AIU](https://aiu.kz/students/documents/academic-policy?lang=ru)")


def model_evaluation(report):
    stage_label = st.radio("Этап проверки", list(STAGES), horizontal=True, key="test_stage")
    stage = STAGES[stage_label]
    info = report["stages"][stage]
    native_target = "итоговой оценки" if stage == "early" else "экзамена"
    st.subheader("Сравнение моделей")
    st.write(f"Модель выбиралась по средней абсолютной ошибке прогноза {native_target} на валидации. "
             "Ошибки приведены в баллах по шкале от 0 до 100.")
    comparison = []
    for row in info["models"]:
        item = {
            "Метод": MODEL_NAMES.get(row["name"], row["name"]),
            "MAE валидации": row["validation"]["mae"],
            "MAE теста": row["test"]["mae"],
            "RMSE теста": row["test"]["rmse"],
        }
        if stage == "late":
            item["MAE итога по формуле"] = row["final_test"]["mae"]
        comparison.append(item)
    config = {k: st.column_config.NumberColumn(format="%.2f") for k in comparison[0] if k != "Метод"}
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch", column_config=config)
    st.caption(f"Выбранная модель — {MODEL_NAMES.get(info['winner'], info['winner'])}. "
               "MAE — средняя абсолютная ошибка. RMSE сильнее учитывает большие промахи.")
    if stage == "late":
        st.caption("MAE итога по формуле рассчитана на всём открытом тесте. В кабинете AIU выставление итога зависит от допуска и сдачи экзамена.")
        shown = info.get("displayed_final_test")
        if shown and shown.get("mae") is not None:
            st.write(f"При применении порогов допуска и экзамена числовой итог выдан для "
                     f"**{shown['rows']} из {shown['total_rows']}** тестовых записей. "
                     f"На них средняя ошибка итогового балла — **{decimal(shown['mae'], 2)}**.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Ошибки по диапазонам")
        st.caption(f"Средняя ошибка по фактическому баллу {native_target} на тесте.")
        bands = pd.DataFrame(info["errors"]["bands"])
        bands = bands.loc[bands["count"].gt(0)].rename(columns={"label": "Диапазон", "mae": "MAE", "count": "Наблюдения"})
        if not bands.empty:
            st.vega_lite_chart(bands, {
                "mark": {"type": "bar", "color": ACCENT, "cornerRadiusEnd": 3},
                "height": 200,
                "encoding": {
                    "x": {"field": "Диапазон", "type": "ordinal", "sort": None, "axis": {"labelAngle": 0, "title": "Фактический балл"}},
                    "y": {"field": "MAE", "type": "quantitative", "title": "Ошибка, баллы"},
                    "tooltip": [{"field": "Диапазон"}, {"field": "Наблюдения"}, {"field": "MAE", "format": ".2f"}],
                },
            }, width="stretch")
    with right:
        st.subheader("Разделение данных")
        split_labels = {"train": "Обучение", "validation": "Валидация", "test": "Тест"}
        counts = [{"Выборка": label, "Студенты": report["split"][key]["students"], "Записи": report["split"][key]["rows"]}
                  for key, label in split_labels.items()]
        st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
        st.caption("Тест состоит из более позднего семестра. Один студент встречается только в одной из трёх выборок.")
    with st.expander("Примеры наибольших ошибок"):
        examples = pd.DataFrame(info["errors"]["examples"])
        if stage == "early":
            examples = examples.drop(columns=["r2"], errors="ignore")
        columns = {"r1": "Р1", "r2": "Р2", "actual": "Факт", "predicted": "Прогноз", "absolute_error": "Абсолютная ошибка"}
        examples = examples[[k for k in columns if k in examples]].rename(columns=columns)
        st.dataframe(examples, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.2f") for k in examples})
    with st.expander("Вклад рейтингов в прогноз CatBoost (SHAP)"):
        shap = info["catboost_shap"]
        st.write(f"Средний абсолютный вклад каждого рейтинга в прогноз CatBoost по {shap['rows']} "
                 "валидационным записям. Единица измерения — балл целевой оценки.")
        factors = [{"Рейтинг": k.upper().replace("R", "Р"), "Средний вклад": v}
                   for k, v in shap["mean_abs_effect"].items()]
        st.dataframe(pd.DataFrame(factors), hide_index=True, width="stretch",
                     column_config={"Средний вклад": st.column_config.NumberColumn(format="%.2f")})
    with st.expander("Распределение оценок в обучающей выборке"):
        histogram = pd.DataFrame(report["eda"]["histogram"]).rename(
            columns={"label": "Итоговый балл", "count": "Записи"}
        )
        st.vega_lite_chart(histogram, {
            "height": 200,
            "mark": {"type": "bar", "color": ACCENT},
            "encoding": {
                "x": {"field": "Итоговый балл", "type": "ordinal", "sort": None,
                      "axis": {"labelAngle": 0}},
                "y": {"field": "Записи", "type": "quantitative"},
                "tooltip": [{"field": "Итоговый балл"}, {"field": "Записи"}],
            },
        }, width="stretch")
    st.markdown("[Данные исследования](https://github.com/Mikhail31415/cgpa-prediction/tree/main/data/final_grade_prediction_dataset)")


@st.cache_data(show_spinner=False)
def java_gradebook():
    directory = Path(__file__).resolve().parent / "datasets"
    lectures = pd.read_csv(directory / "kz_java_15_weeks.csv")
    assignments = pd.read_csv(directory / "kz_java_assignments.csv")
    return lectures.merge(assignments, on="student_id", validate="one_to_one").set_index("student_id")


def weekly_prediction(report):
    from grade_forecast.validation import predict_weekly

    week = st.radio("Неделя прогноза", [h["week"] for h in report["horizons"]],
                    key="new_week", horizontal=True)
    student = java_gradebook().iloc[0]
    activity_names = {"lecture": "Лекции", "lab": "Лабораторные", "self_work": "Самостоятельные"}
    sample = {"Неделя": list(range(1, week + 1))}
    for prefix, label in activity_names.items():
        values = [student[f"{prefix}_week_{i:02d}"] for i in sample["Неделя"]]
        sample[label] = ["" if pd.isna(value) else f"{value:g}" for value in values]
    st.caption("В таблице пример из открытой ведомости Java. Можно заменить оценки своими. "
               "Введи балл 0–100, «н» за отсутствие или «н.п.» за неделю без оценивания. "
               "Пустая клетка означает, что балл ещё неизвестен.")
    edited = st.data_editor(
        pd.DataFrame(sample), hide_index=True, num_rows="fixed", disabled=["Неделя"],
        key=f"new_marks_{week}", width="stretch",
        column_config={label: st.column_config.TextColumn(label) for label in activity_names.values()},
    )
    mc1 = mc2 = None
    if week >= 7:
        left, right = st.columns(2)
        mc1 = left.number_input("Балл за РК1", 0.0, 100.0, float(student["MC1"]), 0.5,
                                format="%.2f", key=f"weekly_mc1_{week}")
        if week == 15:
            mc2 = right.number_input("Балл за РК2", 0.0, 100.0, float(student["MC2"]), 0.5,
                                     format="%.2f", key="weekly_mc2")
    marks = {
        prefix: [None if pd.isna(value) else str(value).strip().replace(",", ".") for value in edited[label]]
        for prefix, label in activity_names.items()
    }
    try:
        result = predict_weekly(report, week, marks, mc1=mc1, mc2=mc2)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if result["prediction"] is None:
        st.info("Чтобы рассчитать прогноз, внеси хотя бы одну выставленную оценку или отметку «н».")
        return
    horizon = next(h for h in report["horizons"] if h["week"] == week)
    study_mae = next(m["mae"] for m in horizon["models"] if m["model"] == "ridge")
    st.metric("Прогноз итоговой оценки по недельным данным", f"{decimal(result['prediction'])} / 100")
    st.caption(f"Учтено отметок — {result['observations']}. Средняя ошибка на отложенных студентах "
               f"курса Java для недели {week} — {decimal(study_mae, 2)} балла.")


def weekly_example(horizons):
    st.subheader("Динамика одного студента")
    student_input, week_input = st.columns(2, gap="large")
    with student_input:
        student_id = st.selectbox("Студент в открытом наборе Java",
                                  [r["student_id"] for r in horizons[0]["predictions"]],
                                  format_func=lambda value: f"Студент № {value}")
    with week_input:
        week = st.radio("Доступны оценки по неделю включительно",
                        [h["week"] for h in horizons], horizontal=True, key="example_week")
    horizon = next(h for h in horizons if h["week"] == week)
    item = next(r for r in horizon["predictions"] if r["student_id"] == student_id)
    columns = st.columns(3)
    columns[0].metric("Прогноз модели", decimal(item["predicted"]))
    columns[1].metric("Базовое правило", decimal(item["baseline"]))
    if st.checkbox("Раскрыть итог из ведомости"):
        columns[2].metric("Фактический итог", decimal(item["actual"]))
        st.caption(f"Абсолютная ошибка модели — {decimal(abs(item['predicted'] - item['actual']))} балла.")
    with st.expander("Какие оценки уже были известны"):
        student = java_gradebook().loc[student_id]
        marks = pd.DataFrame({
            "Неделя": range(1, week + 1),
            **{label: [student[f"{prefix}_week_{i:02d}"] for i in range(1, week + 1)]
               for label, prefix in (("Лекции", "lecture"), ("Лабораторные", "lab"),
                                     ("Самостоятельные", "self_work"))},
        })
        st.dataframe(marks, hide_index=True, width="stretch")
        if week >= 7:
            controls = [f"РК1 — {decimal(student['MC1'])}"]
            if week >= 15:
                controls.append(f"РК2 — {decimal(student['MC2'])}")
            st.write(" · ".join(controls))


def weekly_evaluation():
    report = read_report("weekly.json")
    action = st.radio("Недельные данные",
                      ["Прогноз по ведомости", "Проверка на курсе Java"], horizontal=True, key="weekly_action")
    if action == "Прогноз по ведомости":
        st.subheader("Прогноз по текущим работам")
        st.write(f"Модель обучена на {report['n_students']} студентах казахстанского курса Java. "
                 "Она использует оценки за лекции, лабораторные, самостоятельные работы и доступные РК.")
        weekly_prediction(report)
        st.markdown("[Исследование и исходные таблицы](https://doi.org/10.1080/2331186X.2025.2481008)")
        return
    st.subheader("Когда оценки начинают помогать прогнозу")
    st.write(f"Открытая ведомость по курсу Java содержит {report['n_students']} студентов. "
             f"Для каждого срока прогноз проверен по {report['n_splits']} частям выборки. "
             "Каждый студент получил прогноз от модели, обученной на других студентах.")
    weekly_example(report["horizons"])
    st.subheader("Ошибка по всему потоку")
    rows = [{"Неделя": h["week"], "Метод": MODEL_NAMES.get(m["model"], m["model"]), "MAE": m["mae"], "RMSE": m["rmse"]}
            for h in report["horizons"] for m in h["models"]]
    df = pd.DataFrame(rows)
    st.vega_lite_chart(df, {
        "height": 245,
        "mark": {"type": "line", "point": {"filled": True, "size": 85}, "strokeWidth": 3},
        "encoding": {
            "x": {"field": "Неделя", "type": "quantitative", "axis": {"values": [h["week"] for h in report["horizons"]]}, "title": "Учебная неделя"},
            "y": {"field": "MAE", "type": "quantitative", "title": "Средняя ошибка, баллы"},
            "color": {"field": "Метод", "type": "nominal", "scale": {"range": ["#8b6544", ACCENT]}, "legend": {"orient": "bottom", "title": None}},
            "tooltip": [{"field": "Неделя"}, {"field": "Метод"}, {"field": "MAE", "format": ".2f"}],
        },
    }, width="stretch")
    with st.expander("Численные результаты"):
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("MAE", "RMSE")})
        for horizon in report["horizons"]:
            if "baseline_rule" in horizon:
                st.write(f"**Неделя {horizon['week']}.** {horizon['baseline_rule']}")
    st.caption("Это один учебный поток. Точность для других курсов нужно оценивать отдельно.")
    st.markdown("[Исследование и таблицы по неделям](https://doi.org/10.1080/2331186X.2025.2481008)")


st.set_page_config(page_title="Прогноз итоговой оценки", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1080px; padding-top: 2.5rem; padding-bottom: 3.5rem;}
h1,h2,h3 {font-family: 'Source Sans 3','Segoe UI',sans-serif; letter-spacing:-.025em;}
.forecast-card {border:1px solid #19524e; background:#143f3c; color:#f4fcf9;
  border-radius:16px; padding:1.45rem 1.6rem; margin:.2rem 0 .55rem;}
.forecast-label {font-size:1.02rem; font-weight:600; color:#f1faf7;}
.forecast-value {font-size:clamp(2.2rem,5vw,4.1rem); font-weight:650; line-height:1.15;
  margin:.45rem 0; font-variant-numeric:tabular-nums; letter-spacing:-.035em;}
.forecast-value small {font-size:.3em; font-weight:500; color:#b7d8cf; letter-spacing:0;}
.forecast-note {font-size:.95rem; color:#d8ebe5; line-height:1.45;}
@media (max-width: 700px) {.block-container {padding-top:1.4rem;} }
</style>
""", unsafe_allow_html=True)
st.title("Прогноз итоговой оценки")
st.caption("Машинное обучение · 15-недельный семестр AIU")
bundle_file = ARTIFACTS / "forecast.json"
if not bundle_file.is_file() or not (ARTIFACTS / "public_metrics.json").is_file():
    st.info("Для первого запуска подготовь данные и обучи модели по инструкции в README проекта.")
    st.stop()
bundle = cached_bundle(bundle_file.stat().st_mtime_ns)
report = read_report("public_metrics.json")
weekly_available = (ARTIFACTS / "weekly.json").is_file()
tabs = st.tabs(["Прогноз", "Проверка модели"] + (["Недельные оценки"] if weekly_available else []))
with tabs[0]:
    forecast_page(bundle, report)
with tabs[1]:
    model_evaluation(report)
if weekly_available:
    with tabs[2]:
        weekly_evaluation()
