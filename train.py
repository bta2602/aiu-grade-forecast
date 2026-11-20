"""Train and evaluate two time-specific grade forecasts on public Kazakhstan data."""

import argparse
import base64
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from grade_forecast.history import previous_average
from grade_forecast.model import DEFAULT_ARTIFACTS, FEATURES, ROOT, formula_grade, score_matrix
from grade_forecast.report_plot import save_test_plot

PUBLIC = ROOT / "datasets" / "public_kz"
SEED = 716
SPLITS = {"train": ("train", [0, 1, 2]), "validation": ("val", [3]), "test": ("test", [4])}
RENAME = {
    "Student ID": "student", "Course": "course", "Semester": "semester",
    "Midterm 1": "r1", "Midterm 2": "r2", "Rating": "rating", "Exam": "exam", "Final Grade": "final",
}
GRADES = ["r1", "r2", "exam", "final"]
CAT_CONFIGS = [
    {"depth": 3, "iterations": 180, "loss_function": "RMSE"},
    {"depth": 4, "iterations": 250, "loss_function": "MAE"},
    {"depth": 5, "iterations": 200, "loss_function": "RMSE"},
]


def load_data(data_dir):
    data, split_report, audit = {}, {}, {}
    seen_students = set()
    for name, (file_stem, semesters) in SPLITS.items():
        file = data_dir / f"{file_stem}.csv"
        if not file.exists():
            raise FileNotFoundError(f"Не найдены публичные данные: {file}")
        raw = pd.read_csv(file, usecols=list(RENAME)).rename(columns=RENAME)
        for col in RENAME.values():
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        required = ["student", "course", "semester", *GRADES]
        valid = raw[required].notna().all(axis=1) & raw[GRADES].ge(0).all(axis=1) & raw[GRADES].le(100).all(axis=1)
        clean = raw.loc[valid].drop_duplicates()
        key = ["student", "course", "semester"]
        if clean.duplicated(key).any():
            raise ValueError(f"Противоречивые дубликаты дисциплины у студента: {file.name}")
        students = set(clean["student"])
        if seen_students & students:
            raise ValueError("У исходных train/val/test есть общие студенты")
        seen_students.update(students)
        clean = clean.assign(prior_average=previous_average(clean))
        current = clean.loc[clean["semester"].isin(semesters)].reset_index(drop=True)
        if current.empty or sorted(current["semester"].unique()) != semesters:
            raise ValueError(f"Отсутствует ожидаемый семестр: {file.name}")
        data[name] = current
        split_report[name] = {
            "rows": len(current), "students": int(current["student"].nunique()), "semesters": semesters,
        }
        audit[file_stem] = {
            "source_rows": len(raw), "invalid_rows": int((~valid).sum()),
            "exact_duplicates": int(valid.sum() - len(clean)),
            "other_semesters_excluded": len(clean) - len(current),
        }
    return data, split_report, audit


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "r2": float(r2_score(actual, predicted)),
    }


def fit_model(name, x, y, config=None):
    if name == "catboost":
        model = CatBoostRegressor(
            **config, random_seed=SEED, learning_rate=0.07, thread_count=2,
            allow_writing_files=False, verbose=False,
        ).fit(x, y)
        return {"name": name, "estimator": model, "parameters": config}
    if name == "linear":
        fit = LinearRegression().fit(x, y)
        weights, intercept = fit.coef_, float(fit.intercept_)
    else:
        # The latest single rating (early) or the mean of both ratings (late).
        weights, intercept = np.full(x.shape[1], 1 / x.shape[1]), 0.0
    return {"name": name, "weights": weights.tolist(), "intercept": intercept,
            "center": x.mean(axis=0).tolist()}


def predict_scored(model, x):
    return np.clip(score_matrix(model, x), 0, 100)


def predict_targets(model, stage, data, features):
    native = predict_scored(model, data[features].to_numpy(dtype=float))
    final = native if stage == "early" else formula_grade(data.r1.to_numpy(), data.r2.to_numpy(), native)
    return native, final


def paired_final_comparison(data, reference, candidate, alpha=0.025):
    """Resample whole students, retaining all their courses, for paired MAE differences."""
    actual = data.final.to_numpy()
    errors = np.abs(actual - reference) - np.abs(actual - candidate)
    _, student = np.unique(data.student, return_inverse=True)
    totals = np.bincount(student, weights=errors)
    counts = np.bincount(student)
    sample = np.random.default_rng(SEED).integers(len(counts), size=(4000, len(counts)))
    improvement = totals[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return {
        "students": len(counts), "rows": len(data),
        "reference_mae": float(np.abs(actual - reference).mean()),
        "candidate_mae": float(np.abs(actual - candidate).mean()),
        "mae_gain": float(errors.mean()), "ci_level": 1 - 2 * alpha,
        "gain_interval": np.quantile(improvement, [alpha, 1 - alpha]).tolist(),
    }


def load_external(path):
    if path is None:
        return {}
    frame = pd.read_csv(path, usecols=["source", *GRADES])
    if frame.empty or frame.source.isna().any():
        raise ValueError("Each external row must name a source")
    for key in GRADES:
        values = pd.to_numeric(frame[key], errors="raise")
        if not values.dropna().between(0, 100).all():
            raise ValueError(f"External grades outside 0-100: {key}")
        frame[key] = values
    return {source: rows for source, rows in frame.groupby("source")}


def select_extensions(stage, selection, train, val, external):
    features = FEATURES[stage]
    target = "final" if stage == "early" else "exam"
    original = selection["models"][selection["winner"]]
    config = selection["models"]["catboost"]["parameters"]
    _, reference = predict_targets(original, stage, val, features)
    report = {"transfer": {}, "selected_external_source": None}
    compatible = {source: rows.dropna(subset=[*features, target]) for source, rows in external.items()}
    trials = sum(not rows.empty for rows in compatible.values())
    best_gain = 0
    rating_model = original
    for source, extra in compatible.items():
        if extra.empty:
            report["transfer"][source] = {"training_rows_added": 0, "reason": f"missing {target} or ratings"}
            continue
        pooled = pd.concat([train, extra], ignore_index=True)
        candidate = fit_model("catboost", pooled[features].to_numpy(), pooled[target].to_numpy(), config)
        native, final = predict_targets(candidate, stage, val, features)
        # Bonferroni-adjust the paired bootstrap interval for the sources tried on this stage.
        comparison = paired_final_comparison(val, reference, final, alpha=0.025 / trials)
        comparison["training_rows_added"] = len(extra)
        comparison["native_validation_mae"] = float(mean_absolute_error(val[target], native))
        report["transfer"][source] = comparison
        if comparison["gain_interval"][0] > 0 and comparison["mae_gain"] > best_gain:
            best_gain, rating_model = comparison["mae_gain"], candidate
            report["selected_external_source"] = source
    if report["selected_external_source"] is not None:
        selection["models"]["catboost"] = rating_model
        selection["winner"] = "catboost"
        pred, _ = predict_targets(rating_model, stage, val, features)
        selection["validation"]["catboost"] = metrics(val[target], pred)

    # Keep the no-history model available for unobserved history and new students.
    history_features = [*features, "prior_average"]
    history_train = train.dropna(subset=["prior_average"])
    history_val = val.dropna(subset=["prior_average"])
    if history_train.empty or history_val.empty:
        report["history"] = {"selected": False, "training_rows": len(history_train), "rows": len(history_val)}
        return None, report
    history_model = fit_model(
        "catboost", history_train[history_features].to_numpy(), history_train[target].to_numpy(), config,
    )
    _, reference = predict_targets(rating_model, stage, history_val, features)
    native, final = predict_targets(history_model, stage, history_val, history_features)
    history = paired_final_comparison(history_val, reference, final)
    history["selected"] = history["gain_interval"][0] > 0
    history["training_rows"] = len(history_train)
    history["native_validation_mae"] = float(mean_absolute_error(history_val[target], native))
    report["history"] = history
    return history_model if history["selected"] else None, report


def grade_bands(actual, predicted):
    bands = []
    for low, high in [(0, 50), (50, 70), (70, 90), (90, 101)]:
        mask = (actual >= low) & (actual < high)
        n = int(mask.sum())
        bands.append({
            "label": f"{low}–{min(high - 1, 100)}", "count": n,
            "mae": float(np.mean(np.abs(predicted[mask] - actual[mask]))) if n else None,
            "bias": float(np.mean(predicted[mask] - actual[mask])) if n else None,
        })
    return bands


def direct_final_benchmark(x, data):
    """A public-source diagnostic: fit Final Grade directly after both periods."""
    actual = {split: frame["final"].to_numpy() for split, frame in data.items()}
    models = {
        name: fit_model(name, x["train"], actual["train"])
        for name in ("baseline", "linear")
    }
    val = {
        name: metrics(actual["validation"], predict_scored(model, x["validation"]))
        for name, model in models.items()
    }
    winner = min(val, key=lambda name: val[name]["mae"])
    return {
        "target": "final", "features": FEATURES["late"], "selection_metric": "validation MAE",
        "winner": winner,
        "models": [{
            "name": name, "validation": val[name],
            "test": metrics(actual["test"], predict_scored(model, x["test"])),
        } for name, model in models.items()],
    }


def select_stage(stage, train, val):
    feature_names = FEATURES[stage]
    target = "final" if stage == "early" else "exam"
    x_train, x_val = train[feature_names].to_numpy(dtype=float), val[feature_names].to_numpy(dtype=float)
    y_train, y_val = train[target].to_numpy(dtype=float), val[target].to_numpy(dtype=float)
    fitted = {
        "baseline": fit_model("baseline", x_train, y_train),
        "linear": fit_model("linear", x_train, y_train),
    }
    tuning = []
    cats = []
    for config in CAT_CONFIGS:
        cat = fit_model("catboost", x_train, y_train, config)
        validation_mae = mean_absolute_error(y_val, predict_scored(cat, x_val))
        cats.append(cat)
        tuning.append({"parameters": config, "validation_mae": float(validation_mae)})
    fitted["catboost"] = cats[min(range(len(cats)), key=lambda i: tuning[i]["validation_mae"])]
    validation = {
        name: metrics(y_val, predict_scored(model, x_val))
        for name, model in fitted.items()
    }
    winner = min(validation, key=lambda name: validation[name]["mae"])
    return {"models": fitted, "winner": winner, "validation": validation, "tuning": tuning}


def evaluate_stage(stage, data, selection):
    feature_names = FEATURES[stage]
    target = "final" if stage == "early" else "exam"
    x = {key: df[feature_names].to_numpy(dtype=float) for key, df in data.items()}
    y = {key: df[target].to_numpy(dtype=float) for key, df in data.items()}
    fitted, winner = selection["models"], selection["winner"]
    validation, tuning = selection["validation"], selection["tuning"]
    test = data["test"]
    final = test["final"].to_numpy()
    summary = []
    for name, model in fitted.items():
        pred = predict_scored(model, x["test"])
        final_pred = pred if stage == "early" else formula_grade(test["r1"].to_numpy(), test["r2"].to_numpy(), pred)
        summary.append({
            "name": name, "validation": validation[name], "test": metrics(y["test"], pred),
            "final_test": metrics(final, final_pred),
        })
        if name == winner:
            selected_pred, selected_final = pred, final_pred
    residual = selected_pred - y["test"]
    examples = []
    for idx in np.argsort(-np.abs(residual), kind="stable")[:5]:
        examples.append({
            "r1": float(test.iloc[idx]["r1"]),
            "r2": float(test.iloc[idx]["r2"]) if stage == "late" else None,
            "actual": float(y["test"][idx]), "predicted": float(selected_pred[idx]),
            "absolute_error": abs(float(residual[idx])),
        })
    cat_shap = fitted["catboost"]["estimator"].get_feature_importance(
        Pool(x["validation"]), type="ShapValues", thread_count=2
    )
    report = {
        "target": target, "features": feature_names, "winner": winner, "selection_metric": "validation MAE",
        "models": summary, "catboost_tuning": tuning,
        "catboost_shap": {
            "basis": "validation", "rows": len(x["validation"]),
            "mean_abs_effect": dict(zip(feature_names, np.abs(cat_shap[:, :-1]).mean(axis=0).tolist())),
        },
        "errors": {
            "bands": grade_bands(y["test"], selected_pred), "examples": examples,
            "final_bands": grade_bands(final, selected_final),
            "final_error_quantiles": {str(q): float(np.quantile(np.abs(final-selected_final), q)) for q in (0.5, 0.9, 0.95)},
        },
    }
    if stage == "late":
        eligible = (x["test"].mean(axis=1) >= 50) & (selected_pred >= 50)
        count = int(eligible.sum())
        report["displayed_final_test"] = {
            "rows": count, "total_rows": len(final), "coverage": count / len(final),
            "mae": float(mean_absolute_error(final[eligible], selected_final[eligible])) if count else None,
        }
        report["direct_final_benchmark"] = direct_final_benchmark(x, data)
    return fitted[winner], report


def describe(data):
    train = data["train"]
    rating = train.loc[train["rating"].between(0, 100)]
    rating_difference = rating["rating"] - (rating["r1"] + rating["r2"]) / 2
    spans = [[low, low + 10] for low in range(0, 100, 10)]
    spans[-1][1] = 101
    return {
        "basis": "train", "histogram": [
            {"label": f"{lo}–{min(hi-1, 100)}",
             "count": int(((train.final >= lo) & (train.final < hi)).sum())} for lo, hi in spans
        ],
        "target_stats": {field: float(value) for field, value in train.final.describe().items()},
        "correlations_with_final": {key: float(train[key].corr(train.final)) for key in GRADES if key != "final"},
        "below_50": {key: int((df.final < 50).sum()) for key, df in data.items()},
        "midterms_vs_rating": {
            "rows": len(rating),
            "within_one_point": int(rating_difference.abs().le(1).sum()),
            "fraction_within_one_point": float(rating_difference.abs().le(1).mean()) if len(rating) else None,
            "mean_absolute_difference": float(rating_difference.abs().mean()) if len(rating) else None,
        },
    }


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp") as tmp:
        json.dump(content, tmp, indent=2, ensure_ascii=False, allow_nan=False)
        tmp.write("\n")
    os.replace(tmp.name, path)


def export_model(model):
    if model["name"] != "catboost":
        return model
    with TemporaryDirectory() as tmp:
        cbm = Path(tmp) / "model.cbm"
        model["estimator"].save_model(str(cbm))
        return {"name": "catboost", "parameters": model["parameters"], "cbm": base64.b64encode(cbm.read_bytes()).decode("ascii")}


def train(data_dir=PUBLIC, artifacts=DEFAULT_ARTIFACTS, external_path=None):
    data, splits, audit = load_data(Path(data_dir))
    external = load_external(external_path)
    bundle = {"version": 1, "stages": {}, "history_stages": {}}
    report = {"version": 1, "seed": SEED, "split": splits, "cleaning": audit, "stages": {}, "extensions": {}}
    selections, histories = {}, {}
    for stage in FEATURES:
        selection = select_stage(stage, data["train"], data["validation"])
        histories[stage], experiment = select_extensions(stage, selection, data["train"], data["validation"], external)
        selections[stage] = selection
        report["extensions"][stage] = experiment

    # All training and model selection have finished before any test metrics are calculated.
    report["eda"] = describe(data)
    for stage in FEATURES:
        selected, stage_report = evaluate_stage(stage, data, selections[stage])
        bundle["stages"][stage] = export_model(selected)
        report["stages"][stage] = stage_report
        if histories[stage] is not None:
            history_model = histories[stage]
            bundle["history_stages"][stage] = export_model(history_model)
            rows = data["test"].dropna(subset=["prior_average"])
            if not rows.empty:
                native, final = predict_targets(history_model, stage, rows, [*FEATURES[stage], "prior_average"])
                _, ref = predict_targets(selected, stage, rows, FEATURES[stage])
                target = "final" if stage == "early" else "exam"
                comparison = paired_final_comparison(rows, ref, final)
                comparison["native_test"] = metrics(rows[target], native)
                comparison["within_5"] = int((np.abs(rows.final - final) <= 5).sum())
                comparison["final_bands"] = grade_bands(rows.final.to_numpy(), final)
                if stage == "late":
                    eligible = ((rows.r1 + rows.r2) / 2 >= 50) & (native >= 50)
                    comparison["displayed_final_test"] = {
                        "rows": int(eligible.sum()), "total_rows": len(rows),
                        "mae": float(mean_absolute_error(rows.final[eligible], final[eligible])) if eligible.any() else None,
                    }
                report["extensions"][stage]["history"]["test"] = comparison
        result = next(r for r in stage_report["models"] if r["name"] == stage_report["winner"])
        print(f"{stage}: {stage_report['winner']}, test MAE={result['test']['mae']:.3f}, final MAE={result['final_test']['mae']:.3f}", flush=True)
    write_json(Path(artifacts) / "forecast.json", bundle)
    write_json(Path(artifacts) / "public_metrics.json", report)
    save_test_plot(report, Path(artifacts) / "test_mae.png")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PUBLIC)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--external", type=Path, help="Normalized external training sources (optional CSV)")
    options = parser.parse_args()
    train(options.data_dir, options.artifacts, options.external)
