"""Download two published grade datasets and align their observed scores on a 0–100 scale."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "datasets" / "external" / "normalized.csv"
OPENINTRO_URL = (
    "https://raw.githubusercontent.com/OpenIntroStat/openintro/"
    "f6372de1039e5075e87bd8e1e2ab194819a57894/data-raw/exam_grades/exam_grades.txt"
)
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
COLUMNS = ["source", "course", "term", "r1", "r2", "exam", "final"]


def fetch(url: str, expected_sha: str) -> BytesIO:
    request = Request(url, headers={"User-Agent": "aiu-grade-forecast"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
    if sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"Source file changed, verify mapping and SHA-256: {url}")
    return BytesIO(data)


def openintro() -> pd.DataFrame:
    raw = fetch(OPENINTRO_URL, "cb85c275251d4c1d5dfa18bf9ae35d9db1f40520ec2dfb2a02fb5468e1d8dbb5")
    grades = pd.read_csv(raw, sep="\t", usecols=["semester", "exam1", "exam2", "exam3", "course_grade"])
    return grades.rename(columns={
        "semester": "term", "exam1": "r1", "exam2": "r2", "exam3": "exam", "course_grade": "final",
    }).assign(source="openintro_statistics", course="statistics")[COLUMNS]


def portugal() -> pd.DataFrame:
    raw = fetch(UCI_URL, "82ae9d66437b9808df42e8c89d2bb179c46e9cfbcf06f38abc1d20b3b747e177")
    courses = []
    with ZipFile(raw) as outer, ZipFile(BytesIO(outer.read("student.zip"))) as dataset:
        for filename, course in [("student-mat.csv", "mathematics"), ("student-por.csv", "portuguese")]:
            with dataset.open(filename) as stream:
                scores = pd.read_csv(stream, sep=";", usecols=["G1", "G2", "G3"])
            scores = scores.mul(5).rename(columns={"G1": "r1", "G2": "r2", "G3": "final"})
            courses.append(scores.assign(source="uci_portugal", course=course, term=None, exam=float("nan")))
    return pd.concat(courses, ignore_index=True)[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    output = parser.parse_args().output
    rows = pd.concat([openintro(), portugal()], ignore_index=True)
    for col in ("r1", "r2", "exam", "final"):
        if not rows[col].dropna().between(0, 100).all():
            raise ValueError(f"Observed grades outside 0–100 in {col}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        rows.to_csv(temporary_path, index=False, lineterminator="\n", encoding="utf-8")
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    counts = rows.groupby(["source", "course"]).agg(
        rows=("source", "size"), first_scores=("r1", "count"), exams=("exam", "count"), finals=("final", "count")
    )
    print(counts)
    print(output)


if __name__ == "__main__":
    main()
