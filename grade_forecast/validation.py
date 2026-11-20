"""Weekly Java cohort experiment and independent single-student AIU validation."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
COMPONENT_WEIGHTS = {"lecture": 0.2, "lab": 0.5, "self_work": 0.3}
METRIC_SEED = 42


def error_metrics(actual, predicted):
    """Compute errors in points on rows with a numerical prediction."""
    residuals = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    return {
        "n": len(residuals),
        "mae": round(float(np.abs(residuals).mean()), 4),
        "rmse": round(float(np.sqrt(np.square(residuals).mean())), 4),
        "bias": round(float(residuals.mean()), 4),
    }


def weekly_scores(raw):
    """An absence earns zero. Unassessed weeks and blank entries have no score."""
    cleaned = raw.mask(raw.isin(["н.п.", "a/np", ""])).mask(raw.eq("н"), 0)
    scores = cleaned.apply(pd.to_numeric, errors="raise")
    if ((scores < 0) | (scores > 100)).any().any():
        raise ValueError("Недельная оценка за пределами шкалы 0–100.")
    return scores


def mark_features(students, week):
    """Return the weekly predictors, available weighted mean and observed mark count."""
    averages = {}
    records = []
    for activity in COMPONENT_WEIGHTS:
        columns = [f"{activity}_week_{i:02}" for i in range(1, week + 1)]
        marks = weekly_scores(students[columns])
        averages[activity] = marks.mean(axis=1)
        records.append(marks)
    features = pd.DataFrame(averages)
    # Reweight when no marks exist for a particular type of activity.
    weights = pd.Series(COMPONENT_WEIGHTS)
    available_weight = features.notna().mul(weights).sum(axis=1)
    baseline = features.mul(weights).sum(axis=1) / available_weight.replace(0, np.nan)
    entries = pd.concat(records, axis=1)
    observed = entries.count(axis=1)
    features["zero_share"] = entries.eq(0).sum(axis=1) / observed.replace(0, np.nan)
    if week >= 7:
        features["mc1"] = pd.to_numeric(students["MC1"])
    if week >= 15:
        features["mc2"] = pd.to_numeric(students["MC2"])
    return features, baseline, observed


def weekly_features(students, week):
    features, baseline, _ = mark_features(students, week)
    rule = "Среднее по имеющимся оценкам с весами курса 20%, 50% и 30%"
    if week >= 7:
        baseline = pd.to_numeric(students["Rat1"])
        rule = "Первый рейтинг дисциплины"
    if week >= 15:
        baseline = (baseline + pd.to_numeric(students["Rat2"])) / 2
        rule = "Среднее первого и второго рейтингов"
    return features, baseline, rule


def ridge_pipeline():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=10))


def fitted_scores(predictor, features):
    x = features[predictor["features"]].to_numpy(dtype=float)
    filled = np.where(np.isnan(x), predictor["medians"], x)
    z = (filled - predictor["centers"]) / predictor["scales"]
    return np.clip(z @ predictor["weights"] + predictor["intercept"], 0, 100)


def export_ridge(features, target):
    model = ridge_pipeline().fit(features, target)
    imputer, scaler, ridge = (step for _, step in model.steps)
    predictor = {
        "features": list(features.columns),
        "medians": imputer.statistics_.tolist(),
        "centers": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "weights": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    }
    np.testing.assert_allclose(fitted_scores(predictor, features),
                               np.clip(model.predict(features), 0, 100), atol=1e-9, rtol=0)
    return predictor


def numeric_mark(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Оценка должна быть числом от 0 до 100.")
    try:
        grade = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Оценка должна быть числом от 0 до 100.") from exc
    if not np.isfinite(grade) or not 0 <= grade <= 100:
        raise ValueError("Оценка должна быть числом от 0 до 100.")
    return grade


def predict_weekly(report, week, marks, mc1=None, mc2=None):
    """Forecast an unseen student's FG using the model refit after OOF evaluation."""
    if not isinstance(week, (int, np.integer)) or week not in (3, 7, 15):
        raise ValueError("Выбери неделю 3, 7 или 15.")
    if (week < 7 and mc1 is not None) or (week < 15 and mc2 is not None):
        raise ValueError("Этот рубежный контроль ещё недоступен на выбранной неделе.")
    if not isinstance(marks, dict) or set(marks) != set(COMPONENT_WEIGHTS):
        raise ValueError("Заполни три ряда: лекции, лабораторные, самостоятельные работы.")
    row = {}
    observed = 0
    for activity in COMPONENT_WEIGHTS:
        if not isinstance(marks[activity], (tuple, list)) or len(marks[activity]) != week:
            raise ValueError(f"Для каждого вида занятий нужно передать {week} недель.")
        for i, value in enumerate(marks[activity], start=1):
            if isinstance(value, str):
                value = value.strip().lower()
            is_blank = value is None or value is pd.NA or isinstance(value, (float, np.floating)) and np.isnan(value)
            if is_blank or isinstance(value, str) and value in ("", "н.п.", "a/np"):
                score = np.nan
            else:
                score = 0.0 if value == "н" else numeric_mark(value)
                observed += 1
            row[f"{activity}_week_{i:02}"] = score
    if not observed:
        return {"prediction": None, "observations": 0, "features": {}}
    if week >= 7:
        row["MC1"] = numeric_mark(mc1)
    if week >= 15:
        row["MC2"] = numeric_mark(mc2)
    features, _, _ = mark_features(pd.DataFrame([row]), week)
    horizon = next(h for h in report["horizons"] if h["week"] == week)
    score = float(fitted_scores(horizon["predictor"], features)[0])
    observed_features = {
        key: None if pd.isna(val) else float(val) for key, val in features.iloc[0].items()
    }
    return {"prediction": score, "observations": observed, "features": observed_features}


def weekly_report():
    marks = pd.read_csv(DATA / "kz_java_15_weeks.csv")
    assignments = pd.read_csv(DATA / "kz_java_assignments.csv")
    students = marks.merge(assignments, how="outer", on="student_id",
                           validate="one_to_one", indicator=True)
    if students["student_id"].isna().any() or not students["_merge"].eq("both").all():
        raise ValueError("В двух недельных таблицах должны совпадать ID студентов.")
    target = students["FG"].to_numpy(dtype=float)
    if not np.isfinite(target).all() or ((target < 0) | (target > 100)).any():
        raise ValueError("Отсутствует итоговая оценка или нарушена шкала 0–100.")
    kfold = KFold(n_splits=5, shuffle=True, random_state=METRIC_SEED)
    horizons = []
    for week in (3, 7, 15):
        features, baseline, rule = weekly_features(students, week)
        ridge_predictions = np.empty(len(students))
        baseline_predictions = np.empty(len(students))
        for train, held_out in kfold.split(students):
            ridge = ridge_pipeline()
            ridge.fit(features.iloc[train], target[train])
            ridge_predictions[held_out] = np.clip(ridge.predict(features.iloc[held_out]), 0, 100)
            baseline_predictions[held_out] = baseline.iloc[held_out].fillna(target[train].mean())
        horizons.append({
            "week": week,
            "n_evaluated": len(students),
            "features": list(features.columns),
            "baseline_rule": rule,
            "predictor": export_ridge(features, target),
            "models": [
                {"model": "baseline", **error_metrics(target, baseline_predictions)},
                {"model": "ridge", **error_metrics(target, ridge_predictions)},
            ],
            "predictions": [
                {"student_id": int(student_id), "predicted": round(float(ridge_score), 3),
                 "baseline": round(float(naive_score), 3), "actual": float(actual)}
                for student_id, ridge_score, naive_score, actual in zip(
                    students["student_id"], ridge_predictions, baseline_predictions, target, strict=True
                )
            ],
        })
    return {
        "n_students": len(students),
        "n_splits": kfold.n_splits,
        "seed": METRIC_SEED,
        "target": "FG",
        "target_range": [float(target.min()), float(target.max())],
        "available_components": list(COMPONENT_WEIGHTS),
        "horizons": horizons,
    }


def engagement_by_course(weekly, week):
    """Measure posted grades and explicit absences using only weeks up to the cutoff."""
    dates = [f"week_{i:02}" for i in range(1, week + 1)]
    marks = weekly[dates]
    scores = weekly_scores(marks)
    attendance = pd.DataFrame({
        "course_id": weekly["course_id"],
        "absences": marks.eq("н").sum(axis=1),
        "recorded": scores.count(axis=1),
        "eligible": marks.ne("н.п.").sum(axis=1),
    })
    result = attendance.groupby("course_id")[["absences", "recorded", "eligible"]].sum()
    result["numeric_coverage"] = (result["recorded"] - result["absences"]) / result["eligible"].replace(0, np.nan)
    self_work = weekly["activity_code"].eq("SRSP")
    means = scores.loc[self_work].mean(axis=1)
    result["self_work_mean"] = pd.Series(means.to_numpy(), index=weekly.loc[self_work, "course_id"])
    return result


def private_ablation(data, counts, stage):
    """Forward tests on semesters 3 and 4 of one student; separate from transfer testing."""
    ratings = ["r1"] if stage == "early" else ["r1", "r2"]
    objectives = {
        "ratings_only": ratings,
        "ratings_and_weekly": ratings + ["absences", "numeric_coverage", "self_work_mean"],
    }
    observations = data.set_index("course_id").join(counts, validate="one_to_one")
    observations = observations.loc[observations["recorded"].gt(0)]
    results = {model: [] for model in objectives}
    actuals = []
    splits = []
    for semester in (3, 4):
        train = observations.loc[observations["semester"] < semester]
        test = observations.loc[observations["semester"] == semester]
        if len(train) < 10 or test.empty:
            continue
        splits.append({"holdout_semester": semester, "train": len(train), "test": len(test)})
        actuals.extend(test["final_grade"])
        for label, columns in objectives.items():
            ridge = ridge_pipeline()
            ridge.fit(train[columns], train["final_grade"])
            results[label].extend(np.clip(ridge.predict(test[columns]), 0, 100))
    return {
        "n_students": 1,
        "target": "final_grade",
        "splits": splits,
        "features": objectives,
        "models": {k: error_metrics(actuals, v) for k, v in results.items()},
    }


def private_report(directory, artifact_dir):
    from grade_forecast.model import load_bundle, predict

    courses = pd.read_csv(directory / "courses.csv")
    data = courses.loc[courses["status"].eq("completed")].dropna(
        subset=["rk1", "rk2", "r1", "r2", "exam_score", "final_grade"]
    )
    if data.empty:
        raise ValueError("В AIU нет завершённых дисциплин с обоими рейтингами и экзаменом.")
    if data["course_id"].duplicated().any():
        raise ValueError("Каждый ключ дисциплины должен встречаться один раз.")
    bundle = load_bundle(artifact_dir)
    report = {
        "n_students": 1,
        "n_courses": len(data),
        "n_completed_before_filter": int(courses["status"].eq("completed").sum()),
        "input_mapping": {"r1": "r1", "r2": "r2"},
        "stages": {},
    }
    target = data["final_grade"].to_numpy(dtype=float)
    weekly_file = directory / "weekly_grades.csv"
    weekly = pd.read_csv(weekly_file, keep_default_na=False, dtype=str) if weekly_file.exists() else None
    if weekly is not None and weekly.duplicated(["course_id", "activity_code"]).any():
        raise ValueError("Найдены дубли строк вида занятия в недельной ведомости.")
    for stage, week in (("early", 7), ("late", 15)):
        predictions = [
            predict(bundle, row.r1, row.r2 if stage == "late" else None)
            for row in data.itertuples(index=False)
        ]
        scores = np.asarray([p["final_grade"] for p in predictions], dtype=float)
        baseline = data["r1"] if stage == "early" else (data["r1"] + data["r2"]) / 2
        stage_report = {
            "model": predictions[0]["model"],
            "final_grade": error_metrics(target, scores),
            "baseline_final_grade": error_metrics(target[np.isfinite(scores)], baseline[np.isfinite(scores)]),
            "statuses": pd.Series([p["status"] for p in predictions]).value_counts().to_dict(),
        }
        if stage == "late":
            stage_report["exam_score"] = error_metrics(data["exam_score"], [p["exam_score"] for p in predictions])
            stage_report["exam_baseline_mean_rating"] = error_metrics(data["exam_score"], baseline)
        if weekly is not None:
            counts = engagement_by_course(weekly, week).reindex(data["course_id"])
            available = counts["recorded"].fillna(0).to_numpy() > 0
            absences = counts["absences"].fillna(0).to_numpy() > 0
            stage_report["absence_audit"] = {
                label: error_metrics(target[mask], scores[mask])
                for label, mask in [
                    ("explicit_absence", available & absences),
                    ("no_explicit_absence", available & ~absences),
                ]
            }
            report.setdefault("separate_local_ablation", {})[stage] = private_ablation(data, counts, stage)
        report["stages"][stage] = stage_report
    return report


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
        temp = Path(output.name)
    try:
        temp.write_text(payload, encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Проверки прогноза итоговой оценки")
    parser.add_argument("experiment", choices=["weekly", "aiu"])
    args = parser.parse_args()
    if args.experiment == "weekly":
        path = ROOT / "artifacts" / "weekly.json"
        report = weekly_report()
    else:
        directory = DATA / "aiu_platonus"
        if not (directory / "courses.csv").is_file():
            print("Локальная выгрузка AIU отсутствует. Проверка пропущена.")
            return
        path = directory / "model_validation.json"
        report = private_report(directory, ROOT / "artifacts")
    write_report(path, report)
    print(f"Сохранено: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
