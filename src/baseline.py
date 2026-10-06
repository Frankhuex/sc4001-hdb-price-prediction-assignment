import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data import PreparedData, prepare_data
from model import PriceModel

DATA_PATH: str = "hdb_price_prediction.csv"
OUTPUT_DIR: Path = Path("outputs/baseline")


@torch.no_grad()
def calculate_rmse(
    model: PriceModel, dataset: TensorDataset, batch_size: int
) -> float:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    categorical: torch.Tensor
    continuous: torch.Tensor
    target: torch.Tensor
    for categorical, continuous, target in DataLoader(dataset, batch_size=batch_size):
        predictions.append(model.predict_price(categorical, continuous))
        targets.append(target)

    prediction: torch.Tensor = torch.cat(predictions)
    target = torch.cat(targets)
    return (prediction - target).square().mean().sqrt().item()


def train_model(
    model: PriceModel,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    config: Config,
) -> int:
    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        train_data, config.batch_size, shuffle=True
    )
    loss_function: nn.MSELoss = nn.MSELoss()
    optimiser: torch.optim.Adam = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )

    best_rmse: float = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    epoch: int
    categorical: torch.Tensor
    continuous: torch.Tensor
    target: torch.Tensor
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        squared_error: float = 0.0
        row_count: int = 0

        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()
            standard_prediction: torch.Tensor = model(categorical, continuous)
            standard_target: torch.Tensor = (
                target - model.target_mean
            ) / model.target_std
            loss: torch.Tensor = loss_function(standard_prediction, standard_target)
            loss.backward()
            optimiser.step()

            prediction: torch.Tensor = (
                standard_prediction.detach() * model.target_std + model.target_mean
            )
            squared_error += (prediction - target).square().sum().item()
            row_count += len(target)

        train_rmse: float = (squared_error / row_count) ** 0.5
        validation_rmse: float = calculate_rmse(
            model, validation_data, config.batch_size
        )
        print(
            f"epoch {epoch:02d} | train RMSE {train_rmse:,.0f} "
            f"| validation RMSE {validation_rmse:,.0f}"
        )

        if (
            best_weights is None
            or validation_rmse < best_rmse - config.early_stopping_threshold
        ):
            best_rmse = validation_rmse
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return best_epoch


def main() -> None:
    config: Config = Config()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    data: PreparedData = prepare_data(DATA_PATH, config)

    model: PriceModel = PriceModel(
        data["cardinalities"],
        data["target_mean"],
        data["target_std"],
        config,
    )

    best_epoch: int = train_model(model, data["train"], data["validation"], config)
    train_rmse: float = calculate_rmse(model, data["train"], config.batch_size)
    validation_rmse: float = calculate_rmse(
        model, data["validation"], config.batch_size
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path: Path = OUTPUT_DIR / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"\nbest epoch: {best_epoch}")
    print(f"train RMSE: {train_rmse:,.0f}")
    print(f"validation RMSE: {validation_rmse:,.0f}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
