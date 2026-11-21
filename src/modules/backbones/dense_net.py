import torch
from torch import nn
from typing import Union, Sequence

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]

        x = x.view(batch_size, -1)

        return self.model(x)
