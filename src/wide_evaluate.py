"""A3(b): five-seed final evaluation of the validation-selected wide model."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ablation import SEEDS, RunResult
from baseline import DATA_PATH, calculate_rmse, train_one_epoch
from config import Config
from data import PreparedData, prepare_data
from model import PriceModel
from retrain import TestMetrics, evaluate

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a3_b"

def plot_comparison(summary: pd.DataFrame, metric: str, reference: float) -> None:
    figure: plt.Figure
    axes: plt.Axes
    figure, axes = plt.subplots(figsize=(6, 3.5))
    axes.bar(summary["model"], summary[f"{metric}_mean"],
             yerr=summary[f"{metric}_std"], capsize=5, color=["#4477aa", "#228866"])
    axes.axhline(reference, color="gray", linestyle="--", label="A1(d), single seed")
    axes.set_ylabel("Test RMSE (SGD)" if metric == "test_rmse" else "Test R²")
    axes.grid(axis="y", alpha=0.25)
    axes.set_axisbelow(True)
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{metric}.png", dpi=200)
    plt.close(figure)

def main() -> None:
    selected: dict[str, Any] = json.loads((PROJECT_ROOT / "outputs/a3_a/selected/metadata.json").read_text())
    settings: dict[str, Any] = selected["config"]
    key: str
    for key in ("train_years", "categorical_features", "continuous_features", "embedding_dims"):
        settings[key] = tuple(settings[key])
    base_config: Config = Config(**settings)
    coefficient: float = float(selected["wide_lambda"])
    epochs: int = int(selected["a1_best_epoch"])
    if not 1 <= epochs <= base_config.max_epochs:
        raise ValueError("A1-selected epoch count exceeds the assignment limit.")
    baseline_metadata: dict[str, Any] = json.loads((PROJECT_ROOT / "outputs/a2/metadata.json").read_text())
    if baseline_metadata["config"] != json.loads(json.dumps(asdict(base_config))):
        raise ValueError("Stored baseline uses different hyperparameters.")
    if baseline_metadata["epochs"] != epochs or baseline_metadata["seeds"] != list(SEEDS):
        raise ValueError("Stored baseline uses different epochs or seeds.")
    baseline: pd.DataFrame = pd.read_csv(PROJECT_ROOT / "outputs/a2/results.csv", float_precision="round_trip")
    baseline = baseline[baseline["model"] == "Baseline"].copy()
    if len(baseline) != len(SEEDS) or set(baseline["seed"]) != set(SEEDS):
        raise ValueError("Five paired baseline runs are required.")
    data: PreparedData = prepare_data(PROJECT_ROOT / DATA_PATH, base_config, combine_train_validation=True)
    for key in ("target_mean", "target_std"):
        if data[key] != baseline_metadata[key]:
            raise ValueError(f"Stored baseline preprocessing differs: {key}")
    for key in ("continuous_mean", "continuous_std"):
        if data[key].tolist() != baseline_metadata[key]:
            raise ValueError(f"Stored baseline preprocessing differs: {key}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(OUTPUT_DIR / "baseline_reference.csv", index=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    results: list[RunResult] = []
    print(f"A3(b): lambda={coefficient}; seeds={SEEDS}; fixed A1 duration={epochs} epochs", flush=True)
    print(f"Combined training rows={len(data['train'])}; test rows={len(data['test'])}. Test evaluated once per run.", flush=True)
    seed: int
    for seed in SEEDS:
        config: Config = replace(base_config, seed=seed)
        torch.manual_seed(seed)
        model: PriceModel = PriceModel(data["cardinalities"], data["target_mean"], data["target_std"], config, wide_lambda=coefficient)
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        loader: DataLoader = DataLoader(data["train"], batch_size=config.batch_size, shuffle=True, generator=generator)
        optimiser: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_function: nn.MSELoss = nn.MSELoss()
        history: list[dict[str, int | float]] = []
        epoch: int
        for epoch in range(1, epochs + 1):
            train_one_epoch(model, loader, optimiser, loss_function)
            train_rmse: float = calculate_rmse(model, data["train"], config.batch_size)
            history.append({"epoch": epoch, "train_rmse": train_rmse})
            print(f"seed {seed} | epoch {epoch:02d} | train RMSE {train_rmse:,.2f} SGD", flush=True)
        test: TestMetrics = evaluate(model, data["test"], config.batch_size)
        result: RunResult = {"model": "Wide-and-deep", "seed": seed, "epochs": epochs,
            "train_rmse": train_rmse, "test_rmse": test["rmse"], "test_r2": test["r2"],
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        results.append(result)
        run_dir: Path = OUTPUT_DIR / f"seed_{seed}"
        run_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), run_dir / "final_model.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        (run_dir / "metadata.json").write_text(json.dumps({"config": asdict(config), "wide_lambda": coefficient, "result": result}, indent=2) + "\n")
        pd.DataFrame(results).to_csv(OUTPUT_DIR / "results.csv", index=False)
        print(f"Completed seed={seed}: test RMSE={test['rmse']:,.2f} SGD; R2={test['r2']:.6f}", flush=True)
        if coefficient == 0:
            baseline_state: dict[str, torch.Tensor] = torch.load(PROJECT_ROOT / f"outputs/a2/baseline/seed_{seed}/final_model.pt", weights_only=True)
            if not all(torch.equal(value, model.state_dict()[name]) for name, value in baseline_state.items()):
                raise RuntimeError("Lambda-zero model differs from the paired baseline checkpoint.")
    frame: pd.DataFrame = pd.DataFrame(results)
    comparison: pd.DataFrame = pd.concat([baseline, frame], ignore_index=True)
    summary: pd.DataFrame = comparison.groupby("model", sort=False).agg(
        runs=("seed", "count"), test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"), test_r2_mean=("test_r2", "mean"), test_r2_std=("test_r2", "std"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    paired: pd.DataFrame = frame.merge(baseline, on="seed", suffixes=("_wide", "_baseline"))
    paired["test_rmse_difference"] = paired["test_rmse_wide"] - paired["test_rmse_baseline"]
    paired["test_r2_difference"] = paired["test_r2_wide"] - paired["test_r2_baseline"]
    paired.to_csv(OUTPUT_DIR / "paired_comparison.csv", index=False)
    a1_reference: dict[str, Any] = json.loads((PROJECT_ROOT / "outputs/a1_d/metadata.json").read_text())
    metadata: dict[str, Any] = {**baseline_metadata, "wide_lambda": coefficient,
        "baseline_source": "outputs/a2/results.csv (A1 architecture, matched protocol)",
        "a1_single_seed": {"test_rmse": a1_reference["final_test_rmse"], "test_r2": a1_reference["final_test_r2"]}}
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_comparison(summary, "test_rmse", a1_reference["final_test_rmse"])
    plot_comparison(summary, "test_r2", a1_reference["final_test_r2"])
    print(summary.to_string(index=False), flush=True)
    print(f"Completed five seeds; artifacts saved to {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
