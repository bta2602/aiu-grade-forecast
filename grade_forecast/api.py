"""Local HTTP adapter for the same AIU forecast served by Streamlit."""

from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from grade_forecast.model import DEFAULT_ARTIFACTS, decode_bundle, predict

Grade = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]


class Ratings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r1: Grade = Field(description="Первый рейтинг периода по шкале 0–100.")
    r2: Grade | None = Field(default=None, description="Второй рейтинг, если уже опубликован.")
    prior_average: Grade | None = Field(default=None, description=(
        "Среднее итоговых баллов за дисциплины завершённых предыдущих семестров по шкале 0–100."
    ))


class Forecast(BaseModel):
    stage: Literal["early", "late"]
    status: Literal["estimated", "rating_below_50", "exam_below_50"]
    exam_score: Grade | None
    final_grade: Grade | None
    history_used: bool
    model_revision: str = Field(description="SHA-256 файла обученной модели.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    content = (DEFAULT_ARTIFACTS / "forecast.json").read_bytes()
    app.state.bundle = decode_bundle(content)
    app.state.revision = sha256(content).hexdigest()
    yield


app = FastAPI(title="Прогноз оценок AIU", version="1", lifespan=lifespan,
              description="Прогноз итоговой оценки по дисциплине после первого или второго периода.")


@app.post("/v1/forecast", response_model=Forecast, summary="Оценить будущий итог по дисциплине")
def forecast(ratings: Ratings, request: Request):
    result = predict(request.app.state.bundle, ratings.r1, ratings.r2, prior_average=ratings.prior_average)
    return Forecast(stage=result["stage"], status=result["status"], exam_score=result["exam_score"],
                    final_grade=result["final_grade"], history_used=result["history_used"],
                    model_revision=request.app.state.revision)
