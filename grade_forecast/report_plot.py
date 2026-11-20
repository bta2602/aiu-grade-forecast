"""Render the same final-grade metric for both decision dates and all methods."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_test_plot(report, destination: Path):
    figure = Figure(figsize=(8.6, 3.8), dpi=150, facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    methods = [("baseline", "Базовое правило"), ("linear", "Линейная регрессия"), ("catboost", "CatBoost")]
    maximum = 0
    for stage, label, offset, color in [
        ("early", "После РК1", -0.17, "#305687"),
        ("late", "После РК2", 0.17, "#128274"),
    ]:
        models = {model["name"]: model for model in report["stages"][stage]["models"]}
        values = [models[name]["final_test"]["mae"] for name, _ in methods]
        maximum = max(maximum, *values)
        bars = ax.barh([i + offset for i in range(len(methods))], values, height=0.29, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in values], padding=5, fontsize=10)
    ax.set_yticks(range(len(methods)), [label for _, label in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.17)
    ax.set_xlabel("Средняя абсолютная ошибка итоговой оценки, баллы")
    ax.set_title("Ошибка прогноза на отложенном семестре", loc="left", fontweight="bold", pad=14)
    ax.xaxis.grid(True, color="#e1e5e7")
    ax.set_axisbelow(True)
    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=5)
    figure.legend(loc="outside lower center", ncols=2, frameon=False)
    figure.savefig(destination, dpi=150)
