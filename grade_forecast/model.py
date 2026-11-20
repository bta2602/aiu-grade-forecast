"""Inference shared by the dashboard, public evaluation and local AIU check."""

import base64
import json
import math
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "artifacts"
FEATURES = {"early": ["r1"], "late": ["r1", "r2"]}


def load_bundle(path=DEFAULT_ARTIFACTS):
    """Read one self-contained model file, without deserializing Python objects."""
    return decode_bundle((Path(path) / "forecast.json").read_bytes())


def decode_bundle(content: bytes):
    bundle = json.loads(content)
    if bundle["version"] != 1:
        raise ValueError("Неподдерживаемая версия обученной модели")
    for model in (*bundle["stages"].values(), *bundle.get("history_stages", {}).values()):
        if model["name"] == "catboost":
            model["estimator"] = CatBoostRegressor().load_model(blob=base64.b64decode(model.pop("cbm")))
    return bundle


def score_matrix(model, x):
    """Raw predictions for a rectangular numeric array in FEATURES order."""
    if model["name"] == "catboost":
        return np.asarray(model["estimator"].predict(x, thread_count=1), dtype=float)
    return np.asarray(x) @ np.asarray(model["weights"]) + model["intercept"]


def formula_grade(r1, r2, exam):
    return 0.30 * r1 + 0.30 * r2 + 0.40 * exam


def _validate_grade(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Балл должен быть числом от 0 до 100")
    try:
        grade = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Балл должен быть числом от 0 до 100") from exc
    if not math.isfinite(grade) or not 0 <= grade <= 100:
        raise ValueError("Балл должен быть числом от 0 до 100")
    return grade


def predict(bundle, r1, r2=None, *, prior_average=None):
    """Forecast with the same feature order and clipping used for evaluation.

    Contributions add to the raw model output. Scores displayed to users are
    constrained to the grade scale. The late explanation is for the exam.
    """
    stage = "early" if r2 is None else "late"
    names = FEATURES[stage]
    values = [_validate_grade(r1)]
    if r2 is not None:
        values.append(_validate_grade(r2))
    history_used = prior_average is not None and stage in bundle.get("history_stages", {})
    if prior_average is not None:
        prior_average = _validate_grade(prior_average)
    if history_used:
        names = [*names, "prior_average"]
        values.append(prior_average)
    x = np.array([values], dtype=float)
    model = bundle["history_stages" if history_used else "stages"][stage]
    raw = float(score_matrix(model, x)[0])
    estimate = float(np.clip(raw, 0, 100))
    if model["name"] == "catboost":
        shap = model["estimator"].get_feature_importance(
            Pool(x), type="ShapValues", thread_count=1
        )[0]
        base, effects = float(shap[-1]), shap[:-1]
    else:
        weights = np.array(model["weights"])
        center = np.array(model["center"])
        base = float(model["intercept"] + weights @ center)
        effects = weights * (x[0] - center)
    exam, formula, final = None, None, estimate
    status = "estimated"
    if stage == "late":
        exam = estimate
        formula = float(formula_grade(values[0], values[1], exam))
        if np.mean(values[:2]) < 50:
            status = "rating_below_50"
        elif exam < 50:
            status = "exam_below_50"
        final = formula if status == "estimated" else None
    return {
        "stage": stage, "model": model["name"], "status": status, "history_used": history_used,
        "final_grade": final, "exam_score": exam, "formula_grade": formula,
        "explanation": {
            "target": "final_grade" if stage == "early" else "exam_score",
            "base": base, "contributions": dict(zip(names, map(float, effects))),
            "raw_prediction": raw,
        },
    }
