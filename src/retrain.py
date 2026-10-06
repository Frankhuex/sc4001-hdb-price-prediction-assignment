"""A1(c)--(d): plot search history and retrain without test-based selection."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import DATA_PATH, train_one_epoch
from config import Config
from data import PreparedData, prepare_data
from model import PriceModel

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a1_d"

class TestMetrics(TypedDict):
    rmse: float
    r2: float

class RetrainingEpoch(TypedDict):
    epoch: int
    train_rmse: float
    test_rmse: float

@torch.no_grad()
def evaluate(model: PriceModel, dataset: TensorDataset, batch_size: int) -> TestMetrics:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    categorical: torch.Tensor
    continuous: torch.Tensor
    target: torch.Tensor
    for categorical, continuous, target in DataLoader(dataset, batch_size=batch_size):
        predictions.append(model.predict_price(categorical, continuous).double())
        targets.append(target.double())
    prediction: torch.Tensor = torch.cat(predictions)
    actual: torch.Tensor = torch.cat(targets)
    residual_sum: torch.Tensor = (prediction - actual).square().sum()
    total_sum: torch.Tensor = (actual - actual.mean()).square().sum()
    return {"rmse": (residual_sum / len(actual)).sqrt().item(),
            "r2": (1 - residual_sum / total_sum).item()}

def plot_curves(history: pd.DataFrame, second: str, path: Path) -> None:
    figure: plt.Figure
    axes: plt.Axes
    figure, axes = plt.subplots(figsize=(6, 3.5))
    axes.plot(history["epoch"], history["train_rmse"], label="Training", marker=".")
    axes.plot(history["epoch"], history[f"{second}_rmse"], label=second.capitalize(), marker=".")
    axes.set(xlabel="Epoch", ylabel="RMSE (SGD)")
    axes.set_xticks(range(1, len(history) + 1, 2))
    axes.grid(alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)

def main() -> None:
    selected_dir: Path = PROJECT_ROOT / "outputs/a1_b/selected"
    metadata: dict[str, Any] = json.loads((selected_dir / "metadata.json").read_text())
    settings: dict[str, Any] = metadata["config"]
    settings["train_years"] = tuple(settings["train_years"])
    settings["categorical_features"] = tuple(settings["categorical_features"])
    settings["continuous_features"] = tuple(settings["continuous_features"])
    settings["embedding_dims"] = tuple(settings["embedding_dims"])
    config: Config = Config(**settings)
    epochs: int = int(metadata["best_epoch"])
    if not 1 <= epochs <= config.max_epochs:
        raise ValueError("Selected epoch must be within the assignment epoch limit.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    search_history: pd.DataFrame = pd.read_csv(selected_dir / "history.csv")
    plot_curves(search_history, "validation", selected_dir / "rmse_curves.png")

    # Refit every preprocessing statistic using 2017--2021 only.
    data: PreparedData = prepare_data(PROJECT_ROOT / DATA_PATH, config,
                                     combine_train_validation=True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model: PriceModel = PriceModel(data["cardinalities"], data["target_mean"],
                                   data["target_std"], config)
    loader: DataLoader = DataLoader(data["train"], batch_size=config.batch_size, shuffle=True)
    optimiser: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function: nn.MSELoss = nn.MSELoss()
    history: list[RetrainingEpoch] = []
    print(f"Selected width={config.hidden_width}, embeddings={config.embedding_dims}, seed={config.seed}")
    print(f"Combined training rows={len(data['train'])}; test rows={len(data['test'])}")
    print(f"Training for {epochs} epochs fixed by validation search; test metrics do not affect training.")
    epoch: int
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, loader, optimiser, loss_function)
        train_metrics: TestMetrics = evaluate(model, data["train"], config.batch_size)
        test_metrics: TestMetrics = evaluate(model, data["test"], config.batch_size)
        history.append({"epoch": epoch, "train_rmse": train_metrics["rmse"],
                        "test_rmse": test_metrics["rmse"]})
        print(f"epoch {epoch:02d} | train RMSE {train_metrics['rmse']:,.2f} | test RMSE {test_metrics['rmse']:,.2f} SGD", flush=True)
    with (OUTPUT_DIR / "history.csv").open("w", newline="") as stream:
        writer: csv.DictWriter = csv.DictWriter(stream, fieldnames=list(RetrainingEpoch.__annotations__))
        writer.writeheader()
        writer.writerows(history)
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    final_metadata: dict[str, object] = {
        "config": asdict(config), "epochs": epochs,
        "training_years": [*config.train_years, config.validation_year],
        "training_rows": len(data["train"]), "test_rows": len(data["test"]),
        "cardinalities": data["cardinalities"],
        "continuous_mean": data["continuous_mean"].tolist(),
        "continuous_std": data["continuous_std"].tolist(),
        "target_mean": data["target_mean"], "target_std": data["target_std"],
        "final_train_rmse": train_metrics["rmse"],
        "final_test_rmse": test_metrics["rmse"], "final_test_r2": test_metrics["r2"],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(final_metadata, indent=2) + "\n")
    plot_curves(pd.DataFrame(history), "test", OUTPUT_DIR / "rmse_curves.png")
    print(f"Final test RMSE={test_metrics['rmse']:,.2f} SGD; R2={test_metrics['r2']:.6f}")
    print(f"Artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
