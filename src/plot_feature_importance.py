"""A4(b): plot existing attribution scores without recomputing them."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a4"

def draw(axes: plt.Axes, scores: pd.DataFrame, title: str, colour: str) -> None:
    scores = scores.sort_values("mean_absolute_attribution_sgd", ascending=False)
    axes.barh(scores["feature"], scores["mean_absolute_attribution_sgd"], color=colour)
    axes.invert_yaxis()
    axes.set(title=title, xlabel="Mean absolute attribution (SGD)")
    axes.set_xlim(0, scores["mean_absolute_attribution_sgd"].max() * 1.2)
    axes.grid(axis="x", alpha=0.25)
    axes.set_axisbelow(True)
    index: int
    value: float
    for index, value in enumerate(scores["mean_absolute_attribution_sgd"]):
        axes.text(value + 500, index, f"{value:,.0f}", va="center", fontsize=8)

def main() -> None:
    scores: pd.DataFrame = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    coefficient: float = json.loads((OUTPUT_DIR / "metadata.json").read_text())["wide_lambda"]
    labels: tuple[tuple[str, str, str], ...] = (
        ("baseline", "Baseline (seed 42)", "#4477aa"),
        ("wide", f"Wide-and-deep (seed 42, λ={coefficient:g})", "#228866"),
    )
    figure: plt.Figure
    axes: plt.Axes
    label: str
    title: str
    colour: str
    for label, title, colour in labels:
        figure, axes = plt.subplots(figsize=(8, 5))
        draw(axes, scores[scores["model"] == label], title, colour)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"{label}_importance.png", dpi=200)
        plt.close(figure)
    figure, panels = plt.subplots(1, 2, figsize=(15, 5.5))
    index: int
    for index, (label, title, colour) in enumerate(labels):
        draw(panels[index], scores[scores["model"] == label], title, colour)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "importance_comparison.png", dpi=200)
    plt.close(figure)
    print(f"Wrote three descending, labelled importance plots to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
