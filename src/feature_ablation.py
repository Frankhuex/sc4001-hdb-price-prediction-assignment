"""A4(a): per-sample FeatureAblation in SGD for ten original features."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from captum.attr import FeatureAblation

from baseline import DATA_PATH
from config import Config
from data import PreparedData, prepare_data
from model import PriceModel

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs/a4"
SEED: int = 42
BATCH_SIZE: int = 512

@torch.no_grad()
def attribute_samples(
    model: PriceModel, categorical: torch.Tensor, continuous: torch.Tensor,
    references: tuple[torch.Tensor, torch.Tensor],
    mask: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    explainer: FeatureAblation = FeatureAblation(model.predict_price)
    categorical_attr: torch.Tensor
    continuous_attr: torch.Tensor
    categorical_attr, continuous_attr = explainer.attribute(
        (categorical, continuous), baselines=references, feature_mask=mask,
        perturbations_per_eval=1,
    )
    return torch.cat((categorical_attr, continuous_attr), dim=1)

@torch.no_grad()
def verify_manual(
    model: PriceModel, categorical: torch.Tensor, continuous: torch.Tensor,
    references: tuple[torch.Tensor, torch.Tensor],
    mask: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual: torch.Tensor = attribute_samples(model, categorical, continuous, references, mask)
    original: torch.Tensor = model.predict_price(categorical, continuous)
    index: int
    for index in range(10):
        changed_cat: torch.Tensor = categorical.clone()
        changed_cont: torch.Tensor = continuous.clone()
        if index < 4:
            changed_cat[:, index] = references[0][0, index]
        else:
            changed_cont[:, index - 4] = references[1][0, index - 4]
        expected: torch.Tensor = original - model.predict_price(changed_cat, changed_cont)
        torch.testing.assert_close(actual[:, index], expected, rtol=0, atol=0.0625)
    print("Captum matches manual prediction differences for all ten features on eight samples.", flush=True)

def main() -> None:
    metadata: dict[str, Any] = json.loads((PROJECT_ROOT / "outputs/a3_b/metadata.json").read_text())
    settings: dict[str, Any] = metadata["config"]
    key: str
    for key in ("train_years", "categorical_features", "continuous_features", "embedding_dims"):
        settings[key] = tuple(settings[key])
    config: Config = Config(**settings)
    coefficient: float = float(metadata["wide_lambda"])
    data: PreparedData = prepare_data(PROJECT_ROOT / DATA_PATH, config, combine_train_validation=True)
    for key in ("target_mean", "target_std"):
        if data[key] != metadata[key]:
            raise ValueError(f"Checkpoint preprocessing mismatch: {key}")
    for key in ("continuous_mean", "continuous_std"):
        if data[key].tolist() != metadata[key]:
            raise ValueError(f"Checkpoint preprocessing mismatch: {key}")
    torch.set_num_threads(1)
    train_cat: torch.Tensor = data["train"].tensors[0]
    category_ids: torch.Tensor = torch.tensor([
        torch.bincount(train_cat[:, index]).argmax().item() for index in range(4)
    ], dtype=torch.long).unsqueeze(0)
    references: tuple[torch.Tensor, torch.Tensor] = (category_ids, torch.zeros(1, 6))
    mask: tuple[torch.Tensor, torch.Tensor] = (torch.arange(4).unsqueeze(0), torch.arange(4, 10).unsqueeze(0))
    feature_names: list[str] = [*config.categorical_features, *config.continuous_features]
    raw: pd.DataFrame = pd.read_csv(PROJECT_ROOT / DATA_PATH)
    train_raw: pd.DataFrame = raw[raw["year"].isin([*config.train_years, config.validation_year])]
    reference_values: dict[str, Any] = {}
    index: int
    name: str
    for index, name in enumerate(config.categorical_features):
        categories: list[Any] = sorted(set(train_raw[name]))
        reference_values[name] = {"category_id": int(category_ids[0, index]),
                                  "value": categories[int(category_ids[0, index]) - 1]}
    for index, name in enumerate(config.continuous_features):
        reference_values[name] = {"standardized_value": 0.0, "raw_value": data["continuous_mean"][index].item()}
    checkpoint_paths: dict[str, Path] = {
        "baseline": PROJECT_ROOT / f"outputs/a2/baseline/seed_{SEED}/final_model.pt",
        "wide": PROJECT_ROOT / f"outputs/a3_b/seed_{SEED}/final_model.pt",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_cat: torch.Tensor = data["test"].tensors[0]
    test_cont: torch.Tensor = data["test"].tensors[1]
    csv_rows: list[int] = (raw.index[raw["year"] == config.test_year] + 2).tolist()
    matrices: dict[str, torch.Tensor] = {}
    importance_rows: list[dict[str, str | float]] = []
    print(f"A4(a): seed={SEED}; lambda={coefficient}; test samples={len(test_cat)}; features={feature_names}", flush=True)
    print(f"References: {json.dumps(reference_values)}", flush=True)
    label: str
    checkpoint: Path
    for label, checkpoint in checkpoint_paths.items():
        run_metadata: dict[str, Any] = json.loads(checkpoint.with_name("metadata.json").read_text())
        if run_metadata["config"] != json.loads(json.dumps(asdict(config))) or run_metadata["result"]["epochs"] != metadata["epochs"]:
            raise ValueError("Checkpoints must use the same seed and duration.")
        model: PriceModel = PriceModel(data["cardinalities"], data["target_mean"], data["target_std"], config,
                                       wide_lambda=coefficient if label == "wide" else None)
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()
        verify_manual(model, test_cat[:8], test_cont[:8], references, mask)
        chunks: list[torch.Tensor] = []
        start: int
        for start in range(0, len(test_cat), BATCH_SIZE):
            chunk: torch.Tensor = attribute_samples(model, test_cat[start:start+BATCH_SIZE], test_cont[start:start+BATCH_SIZE], references, mask)
            if not torch.isfinite(chunk).all():
                raise ValueError("Non-finite attributions.")
            chunks.append(chunk.double())
            if start % (BATCH_SIZE * 10) == 0 or start + BATCH_SIZE >= len(test_cat):
                print(f"{label}: processed {min(start+BATCH_SIZE, len(test_cat))}/{len(test_cat)}", flush=True)
        matrix: torch.Tensor = torch.cat(chunks)
        assert matrix.shape == (len(test_cat), 10)
        matrices[label] = matrix
        frame: pd.DataFrame = pd.DataFrame(matrix.numpy(), columns=feature_names)
        frame.insert(0, "source_csv_row", csv_rows)
        frame.insert(0, "sample_index", range(len(test_cat)))
        frame.to_csv(OUTPUT_DIR / f"{label}_attributions.csv", index=False)
        scores: torch.Tensor = matrix.abs().mean(dim=0)
        for index, name in enumerate(feature_names):
            importance_rows.append({"model": label, "feature": name, "mean_absolute_attribution_sgd": scores[index].item()})
    difference: float = (matrices["baseline"] - matrices["wide"]).abs().max().item()
    if coefficient == 0 and difference != 0:
        raise RuntimeError("Lambda-zero attribution differs from matched baseline.")
    importance: pd.DataFrame = pd.DataFrame(importance_rows)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    output_metadata: dict[str, Any] = {
        "config": asdict(config), "seed": SEED, "wide_lambda": coefficient,
        "test_samples": len(test_cat), "feature_names": feature_names,
        "reference_values": reference_values, "units": "SGD",
        "aggregation": "mean absolute per-sample attribution", "perturbations_per_eval": 1,
        "source_csv_row": "1-based file line, including header; sample_index is 0-based test order",
        "max_model_attribution_difference_sgd": difference,
        "checkpoint_sha256": {label: hashlib.sha256(checkpoint.read_bytes()).hexdigest() for label, checkpoint in checkpoint_paths.items()},
        "dataset_sha256": hashlib.sha256((PROJECT_ROOT / DATA_PATH).read_bytes()).hexdigest(),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(output_metadata, indent=2) + "\n")
    print(importance.to_string(index=False), flush=True)
    print(f"Completed; maximum attribution difference={difference} SGD. Artifacts: {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
