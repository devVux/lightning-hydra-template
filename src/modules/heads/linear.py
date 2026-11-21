import torch
from torch import nn

from src.modules.components.blocks import LinearBlock


class LinearHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.model = LinearBlock(
            in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]

        x = x.view(batch_size, -1)

        return self.model(x)
