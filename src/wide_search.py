"""A3(a): validation search with fixed A1 hyperparameters."""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from baseline import DATA_PATH, TrainingResult, train_model
from config import Config
from data import PreparedData, prepare_data
from model import PriceModel

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a3_a"
LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)

def main() -> None:
    selected: dict[str, Any] = json.loads((PROJECT_ROOT / "outputs/a1_b/selected/metadata.json").read_text())
    settings: dict[str, Any] = selected["config"]
    key: str
    for key in ("train_years", "categorical_features", "continuous_features", "embedding_dims"):
        settings[key] = tuple(settings[key])
    config: Config = Config(**settings)
    data: PreparedData = prepare_data(PROJECT_ROOT / DATA_PATH, config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    best_rmse: float = float("inf")
    coefficient: float
    print(f"A3(a): width={config.hidden_width}, dimensions={config.embedding_dims}, seed={config.seed}", flush=True)
    print("2017--2020 training; 2021 validation; no test evaluation. Identical initialization and independently seeded batch order.", flush=True)
    for coefficient in LAMBDAS:
        torch.manual_seed(config.seed)
        model: PriceModel = PriceModel(data["cardinalities"], data["target_mean"], data["target_std"], config, wide_lambda=coefficient)
        print(f"Starting lambda={coefficient}", flush=True)
        result: TrainingResult = train_model(model, data["train"], data["validation"], config, shuffle_seed=config.seed)
        directory: Path = OUTPUT_DIR / f"lambda_{coefficient:g}"
        directory.mkdir(exist_ok=True)
        metadata: dict[str, Any] = {
            "config": asdict(config), "wide_lambda": coefficient,
            "a1_best_epoch": selected["best_epoch"],
            "best_epoch": result.best_epoch, "best_validation_rmse": result.best_validation_rmse,
            "epochs_run": len(result.history), "cardinalities": data["cardinalities"],
            "target_mean": data["target_mean"], "target_std": data["target_std"],
            "continuous_mean": data["continuous_mean"].tolist(), "continuous_std": data["continuous_std"].tolist(),
            "shuffle_seed": config.seed,
        }
        destination: Path
        destinations: list[Path] = [directory]
        if result.best_validation_rmse < best_rmse:
            best_rmse = result.best_validation_rmse
            destinations.append(OUTPUT_DIR / "selected")
        for destination in destinations:
            destination.mkdir(exist_ok=True)
            torch.save(model.state_dict(), destination / "best_model.pt")
            pd.DataFrame(result.history).to_csv(destination / "history.csv", index=False)
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summaries.append({"wide_lambda": coefficient, "best_validation_rmse": result.best_validation_rmse, "best_epoch": result.best_epoch, "epochs_run": len(result.history)})
        pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "search_results.csv", index=False)
        print(f"Completed lambda={coefficient}: best validation RMSE={result.best_validation_rmse:,.2f} SGD, epoch={result.best_epoch}", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    winner: dict[str, Any] = json.loads((OUTPUT_DIR / "selected/metadata.json").read_text())
    print(f"Selected lambda={winner['wide_lambda']}; best validation RMSE={winner['best_validation_rmse']:,.2f} SGD", flush=True)

if __name__ == "__main__":
    main()
