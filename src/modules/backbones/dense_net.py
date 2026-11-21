import torch
from torch import nn
from typing import Union, Sequence
from jaxtyping import Float

from src.modules.components.blocks import LinearBlock


class DenseNet(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Union[int, Sequence[int]] = 256,
        compile: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]

        features = [in_features, *hidden_features, out_features]
        self.model = nn.Sequential(
            *[LinearBlock(i, o) for i, o in zip(features[:-1], features[1:])]
        )

    def forward(self, x: Float[torch.Tensor, "batch *features"]) -> Float[torch.Tensor, "batch out_features"]:
        if x.ndim < 2:
            raise ValueError(f"DenseNet expects a batch dimension and features, got shape {tuple(x.shape)}")
        batch_size = x.size()[0]

        x = x.view(batch_size, -1)
        expected = self.model[0].net[0].in_features
        if x.shape[1] != expected:
            raise ValueError(f"DenseNet expected {expected} input features per sample, got {x.shape[1]}")

        return self.model(x)
