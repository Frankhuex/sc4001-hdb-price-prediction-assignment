"""A2: paired five-seed ablations, with one final test evaluation per run."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline import DATA_PATH, calculate_rmse, train_one_epoch
from config import Config
from data import PreparedData, prepare_data
from model import PriceModel
from retrain import TestMetrics, evaluate

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a2"
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

class Variant(TypedDict):
    name: str
    use_embeddings: bool
    activation: Literal["relu", "sigmoid"]

class RunResult(TypedDict):
    model: str
    seed: int
    epochs: int
    train_rmse: float
    test_rmse: float
    test_r2: float
    parameters: int

VARIANTS: tuple[Variant, ...] = (
    {"name": "Baseline", "use_embeddings": True, "activation": "relu"},
    {"name": "Without embeddings", "use_embeddings": False, "activation": "relu"},
    {"name": "Sigmoid", "use_embeddings": True, "activation": "sigmoid"},
)

def plot_summary(summary: pd.DataFrame, metric: str, ylabel: str) -> None:
    figure: plt.Figure
    axes: plt.Axes
    figure, axes = plt.subplots(figsize=(6, 3.5))
    axes.bar(summary["model"], summary[f"{metric}_mean"],
             yerr=summary[f"{metric}_std"], capsize=5,
             color=["#4477aa", "#ee9933", "#228866"])
    axes.set_ylabel(ylabel)
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{metric}.png", dpi=200)
    plt.close(figure)

def main() -> None:
    metadata: dict[str, Any] = json.loads(
        (PROJECT_ROOT / "outputs/a1_b/selected/metadata.json").read_text()
    )
    settings: dict[str, Any] = metadata["config"]
    key: str
    for key in ("train_years", "categorical_features", "continuous_features", "embedding_dims"):
        settings[key] = tuple(settings[key])
    base_config: Config = Config(**settings)
    epochs: int = int(metadata["best_epoch"])
    if not 1 <= epochs <= base_config.max_epochs:
        raise ValueError("Selected epoch is outside the assignment limit.")
    data: PreparedData = prepare_data(PROJECT_ROOT / DATA_PATH, base_config,
                                     combine_train_validation=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    results: list[RunResult] = []
    print(f"A2: seeds={SEEDS}; epochs={epochs}; width={base_config.hidden_width}; dimensions={base_config.embedding_dims}", flush=True)
    print(f"Training rows={len(data['train'])}; test rows={len(data['test'])}; test evaluated once per run.", flush=True)
    variant: Variant
    seed: int
    for variant in VARIANTS:
        for seed in SEEDS:
            config: Config = replace(base_config, seed=seed)
            torch.manual_seed(seed)
            model: PriceModel = PriceModel(
                data["cardinalities"], data["target_mean"], data["target_std"], config,
                use_embeddings=variant["use_embeddings"], activation=variant["activation"],
            )
            # Model initialization cannot change the paired minibatch order.
            generator: torch.Generator = torch.Generator().manual_seed(seed)
            loader: DataLoader = DataLoader(data["train"], batch_size=config.batch_size,
                                             shuffle=True, generator=generator)
            optimiser: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            loss_function: nn.MSELoss = nn.MSELoss()
            history: list[dict[str, int | float]] = []
            epoch: int
            print(f"Starting {variant['name']}, seed={seed}", flush=True)
            for epoch in range(1, epochs + 1):
                train_one_epoch(model, loader, optimiser, loss_function)
                train_rmse: float = calculate_rmse(model, data["train"], config.batch_size)
                history.append({"epoch": epoch, "train_rmse": train_rmse})
                print(f"{variant['name']} | seed {seed} | epoch {epoch:02d} | train RMSE {train_rmse:,.2f} SGD", flush=True)
            test: TestMetrics = evaluate(model, data["test"], config.batch_size)
            run_dir: Path = OUTPUT_DIR / variant["name"].lower().replace(" ", "_") / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            torch.save(model.state_dict(), run_dir / "final_model.pt")
            result: RunResult = {
                "model": variant["name"], "seed": seed, "epochs": epochs,
                "train_rmse": train_rmse, "test_rmse": test["rmse"], "test_r2": test["r2"],
                "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            }
            results.append(result)
            run_metadata: dict[str, Any] = {"config": asdict(config), "variant": variant, "result": result}
            (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
            pd.DataFrame(results).to_csv(OUTPUT_DIR / "results.csv", index=False)
            print(f"Completed {variant['name']}, seed={seed}: test RMSE={test['rmse']:,.2f} SGD, R2={test['r2']:.6f}", flush=True)
    frame: pd.DataFrame = pd.DataFrame(results)
    summary: pd.DataFrame = frame.groupby("model", sort=False).agg(
        runs=("seed", "count"), test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"), test_r2_mean=("test_r2", "mean"),
        test_r2_std=("test_r2", "std"), train_rmse_mean=("train_rmse", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    preprocessing: dict[str, Any] = {
        "config": asdict(base_config), "seeds": SEEDS, "epochs": epochs,
        "training_years": [*base_config.train_years, base_config.validation_year],
        "training_rows": len(data["train"]), "test_rows": len(data["test"]),
        "cardinalities": data["cardinalities"], "target_mean": data["target_mean"],
        "target_std": data["target_std"], "continuous_mean": data["continuous_mean"].tolist(),
        "continuous_std": data["continuous_std"].tolist(),
        "std_ddof": 1, "shuffle": "independent generator seeded per run",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(preprocessing, indent=2) + "\n")
    plot_summary(summary, "test_rmse", "Test RMSE (SGD), mean ± SD")
    plot_summary(summary, "test_r2", "Test R², mean ± SD")
    print(summary.to_string(index=False), flush=True)
    print(f"Completed 15 runs. Results saved to {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
