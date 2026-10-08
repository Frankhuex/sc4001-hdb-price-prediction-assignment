from typing import Literal

import torch
from torch import nn

from config import Config

class PriceModel(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        target_mean: float,
        target_std: float,
        config: Config,
        *,
        use_embeddings: bool = True,
        activation: Literal["relu", "sigmoid"] = "relu",
        wide_lambda: float | None = None,
    ) -> None:
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
            if use_embeddings
        )

        input_width: int = len(config.continuous_features)
        if use_embeddings:
            input_width += sum(config.embedding_dims)
        if activation not in ("relu", "sigmoid"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Linear(config.hidden_width, 1),
        )

        self.wide_lambda: float | None = wide_lambda
        self.wide: nn.Linear | None = nn.Linear(input_width, 1) if wide_lambda is not None else None

        self.target_mean: float = target_mean
        self.target_std: float = target_std

    def forward(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        embedded: list[torch.Tensor] = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features: torch.Tensor = torch.cat(embedded + [continuous], dim=1)
        output: torch.Tensor = self.mlp(features).squeeze(1)
        if self.wide is not None and self.wide_lambda is not None:
            output = output + self.wide_lambda * self.wide(features).squeeze(1)
        return output

    def predict_price(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        return self(categorical, continuous) * self.target_std + self.target_mean
