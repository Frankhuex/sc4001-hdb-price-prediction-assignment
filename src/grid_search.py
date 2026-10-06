import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import pandas as pd
import torch

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
import ray
from ray import tune

from baseline import DATA_PATH, EpochMetrics, TrainingResult, train_model
from config import Config
from data import PreparedData, prepare_data
from model import PriceModel

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a1_b"
HIDDEN_WIDTHS: list[int] = [16, 32, 64, 128, 256]
EMBEDDING_DIMS: list[int] = [2, 4, 8, 12, 16]

class TrialConfig(TypedDict):
    hidden_width: int
    embedding_dim: int

class TrialSummary(TypedDict):
    hidden_width: int
    embedding_dim: int
    best_validation_rmse: float
    best_epoch: int
    epochs_run: int
    trial_path: str

def train_trial(
    trial_config: TrialConfig,
    data: PreparedData,
    base_config: Config,
    cpus_per_trial: int,
) -> None:
    config: Config = Config(
        **{
            **asdict(base_config),
            "hidden_width": trial_config["hidden_width"],
            "embedding_dims": (trial_config["embedding_dim"],) * 4,
        }
    )
    torch.set_num_threads(cpus_per_trial)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model: PriceModel = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )

    def report_epoch(metrics: EpochMetrics) -> None:
        tune.report(dict(metrics))

    result: TrainingResult = train_model(
        model, data["train"], data["validation"], config, report_epoch
    )
    # The model has already been restored to its best validation checkpoint.
    with TemporaryDirectory() as directory:
        checkpoint_dir: Path = Path(directory)
        torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        with (checkpoint_dir / "history.csv").open("w", newline="") as stream:
            writer: csv.DictWriter = csv.DictWriter(
                stream, fieldnames=list(EpochMetrics.__annotations__)
            )
            writer.writeheader()
            writer.writerows(result.history)
        metadata: dict[str, object] = {
            "config": asdict(config),
            "cardinalities": data["cardinalities"],
            "target_mean": data["target_mean"],
            "target_std": data["target_std"],
            "continuous_mean": data["continuous_mean"].tolist(),
            "continuous_std": data["continuous_std"].tolist(),
            "best_epoch": result.best_epoch,
            "best_validation_rmse": result.best_validation_rmse,
            "epochs_run": len(result.history),
        }
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        # Final report attaches all artifacts; history.csv has one row per epoch.
        tune.report(
            dict(result.history[-1]),
            checkpoint=tune.Checkpoint.from_directory(directory),
        )

def positive_int(value: str) -> int:
    number: int = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="A1(b): Ray grid search over hidden width and embedding dimension."
    )
    parser.add_argument("--max-concurrent-trials", type=positive_int, default=1)
    parser.add_argument("--cpus-per-trial", type=positive_int, default=1)
    args: argparse.Namespace = parser.parse_args()
    config: Config = Config()
    data: PreparedData = prepare_data(PROJECT_ROOT / DATA_PATH, config)
    output_dir: Path = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ray.init(
        address="local", include_dashboard=False,
        runtime_env={
            "py_executable": sys.executable,
            "env_vars": {"PYTHONPATH": str(PROJECT_ROOT / "src")},
        },
    )
    tuner: tune.Tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_trial, data=data, base_config=config,
                cpus_per_trial=args.cpus_per_trial,
            ),
            resources={"cpu": args.cpus_per_trial},
        ),
        param_space={
            "hidden_width": tune.grid_search(HIDDEN_WIDTHS),
            "embedding_dim": tune.grid_search(EMBEDDING_DIMS),
        },
        tune_config=tune.TuneConfig(
            metric="best_validation_rmse", mode="min",
            max_concurrent_trials=args.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name="trials", storage_path=str(output_dir), verbose=1,
            checkpoint_config=tune.CheckpointConfig(num_to_keep=1),
        ),
    )
    try:
        results: tune.ResultGrid = tuner.fit()
    finally:
        ray.shutdown()
    if results.errors:
        raise RuntimeError(
            f"{len(results.errors)} trial(s) failed; inspect {output_dir / 'trials'}."
        )

    summaries: list[TrialSummary] = []
    trial: tune.Result
    for trial in results:
        summaries.append({
            "hidden_width": trial.config["hidden_width"],
            "embedding_dim": trial.config["embedding_dim"],
            "best_validation_rmse": trial.metrics["best_validation_rmse"],
            "best_epoch": trial.metrics["best_epoch"],
            "epochs_run": trial.metrics["epoch"],
            "trial_path": trial.path,
        })
    summary: pd.DataFrame = pd.DataFrame(summaries).sort_values(
        ["hidden_width", "embedding_dim"]
    )
    summary.to_csv(output_dir / "search_results.csv", index=False)
    summary.pivot(
        index="hidden_width", columns="embedding_dim", values="best_validation_rmse"
    ).to_csv(output_dir / "validation_rmse_grid.csv")

    best: tune.Result = results.get_best_result(
        metric="best_validation_rmse", mode="min", scope="last"
    )
    assert best.checkpoint is not None
    selected_dir: Path = output_dir / "selected"
    selected_dir.mkdir(exist_ok=True)
    with best.checkpoint.as_directory() as directory:
        filename: str
        for filename in ("best_model.pt", "history.csv", "metadata.json"):
            shutil.copyfile(Path(directory) / filename, selected_dir / filename)
    print(
        f"Selected hidden_width={best.config['hidden_width']}, "
        f"embedding_dim={best.config['embedding_dim']}, "
        f"best_epoch={best.metrics['best_epoch']}, "
        f"validation RMSE={best.metrics['best_validation_rmse']:,.2f} SGD"
    )
    print(f"Wrote search results and selected model to {output_dir}")

if __name__ == "__main__":
    main()
