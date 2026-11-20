"""Normalize the UTB 2024-I cohort and evaluate the frozen forecast across universities."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import hashlib

import numpy as np
import pandas as pd

from grade_forecast.model import DEFAULT_ARTIFACTS, ROOT, load_bundle, predict
from grade_forecast.validation import error_metrics, write_report

SOURCE = "https://zenodo.org/api/records/22308083/files/Dato%20compartido%202024-I.xlsx/content"
SHA256 = "d78d6678ca1c03a7b83add5a5a5200d866268bad4ba738063746cb99249bce66"
DATA_DIR = ROOT / "datasets" / "external"


def read_cohort(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "utb_2024_i.xlsx"
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(SOURCE, timeout=40) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError(f"Не совпала контрольная сумма книги UTB: {path}")
    if not path.exists():
        path.write_bytes(content)
    sheets = pd.read_excel(BytesIO(content), sheet_name=None)
    frame = pd.concat(sheets.values(), ignore_index=True)
    columns = {"PI": "r1", "PII": "r2", "PIII": "third_period",
               "Final grade": "final", "GPA": "prior_gpa_scaled"}
    withdrawals = {"Withdrawn after PI": "withdrew_after_1", "Withdrawn after PII": "withdrew_after_2"}
    status = frame["Final grade"].map(withdrawals).fillna("graded")
    # Withdrawal text describes an outcome, never a score of zero.
    raw_scores = frame[list(columns)]
    scores = raw_scores.mask(raw_scores.isin(list(withdrawals))).apply(pd.to_numeric, errors="raise")
    if ((scores < 0) | (scores > 5)).any().any():
        raise ValueError("Балл UTB вне шкалы 0–5")
    data = scores.rename(columns=columns).mul(20)
    data.insert(0, "status", status)
    data.to_csv(directory / "utb_2024_i.csv", index=False, float_format="%.6f")
    return data


def transfer_report(data, bundle):
    graded = data.loc[data.status.eq("graded")]
    if graded[["r1", "r2", "final"]].isna().any().any():
        raise ValueError("У завершённого курса UTB отсутствует оценка")
    actual = graded.final.to_numpy()
    stages = {}
    for stage in ("early", "late"):
        predictions = [
            predict(bundle, row.r1, row.r2 if stage == "late" else None)
            for row in graded.itertuples(index=False)
        ]
        # Report numeric forecasts conditional on exactly what the UI would display.
        scores = np.asarray([p["final_grade"] for p in predictions], dtype=float)
        visible = np.isfinite(scores)
        rule = graded.r1.to_numpy() if stage == "early" else (graded.r1 + graded.r2).to_numpy() / 2
        stages[stage] = {
            "displayed": int(visible.sum()),
            "eligible": len(actual),
            "model": error_metrics(actual[visible], scores[visible]),
            "baseline_on_displayed": error_metrics(actual[visible], rule[visible]),
        }
        if stage == "late":
            formula = [p["formula_grade"] for p in predictions]
            stages[stage]["formula_all_graded"] = error_metrics(actual, formula)
            stages[stage]["baseline_all_graded"] = error_metrics(actual, rule)
    return {"source": "https://doi.org/10.5281/zenodo.22308083", "scale_multiplier": 20,
            "rows_by_outcome": data.status.value_counts().to_dict(), "stages": stages}


def main():
    data = read_cohort(DATA_DIR)
    report = transfer_report(data, load_bundle())
    target = DEFAULT_ARTIFACTS / "utb_transfer.json"
    write_report(target, report)
    print(f"{len(data)} observations; comparison saved to {target.name}")


if __name__ == "__main__":
    main()
