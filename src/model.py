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
    ) -> None:
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
        )

        input_width: int = sum(config.embedding_dims) + len(config.continuous_features)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.ReLU(),
            nn.Linear(config.hidden_width, 1),
        )

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
        return self.mlp(features).squeeze(1)

    def predict_price(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        return self(categorical, continuous) * self.target_std + self.target_mean
