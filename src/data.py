import pandas as pd
import torch
from torch.utils.data import TensorDataset

from config import Config


def make_dataset(rows, category_maps, means, stds, config):
    categorical = torch.tensor(
        [
            [
                category_maps[name].get(row[name], 0)
                for name in config.categorical_features
            ]
            for row in rows
        ],
        dtype=torch.long,
    )
    continuous = torch.tensor(
        [[float(row[name]) for name in config.continuous_features] for row in rows],
        dtype=torch.float32,
    )
    continuous = (continuous - means) / stds

    target = torch.tensor([float(row["resale_price"]) for row in rows])
    return TensorDataset(categorical, continuous, target)


def prepare_data(csv_path, config: Config):
    data = pd.read_csv(csv_path)
    rows = data.to_dict(orient="records")

    train_rows = [row for row in rows if int(row["year"]) in config.train_years]
    validation_rows = [
        row for row in rows if int(row["year"]) == config.validation_year
    ]
    test_rows = [row for row in rows if int(row["year"]) == config.test_year]

    # Fit categorical encoders on training data only. ID 0 means "unseen category".
    category_maps = {}
    for name in config.categorical_features:
        values = sorted({row[name] for row in train_rows})
        category_maps[name] = {value: index + 1 for index, value in enumerate(values)}

    # Fit scaling statistics on training data only.
    train_continuous = torch.tensor(
        [
            [float(row[name]) for name in config.continuous_features]
            for row in train_rows
        ]
    )
    means = train_continuous.mean(dim=0)
    stds = train_continuous.std(dim=0).clamp_min(1e-8)

    train_target = torch.tensor([float(row["resale_price"]) for row in train_rows])
    target_mean = train_target.mean().item()
    target_std = train_target.std(correction=0).clamp_min(1e-8).item()

    return {
        "train": make_dataset(train_rows, category_maps, means, stds, config),
        "validation": make_dataset(validation_rows, category_maps, means, stds, config),
        "test": make_dataset(test_rows, category_maps, means, stds, config),
        "cardinalities": [
            len(category_maps[name]) + 1 for name in config.categorical_features
        ],
        "continuous_mean": means,
        "continuous_std": stds,
        "target_mean": target_mean,
        "target_std": target_std,
    }
