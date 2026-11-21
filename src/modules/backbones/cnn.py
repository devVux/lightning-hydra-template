import torch
from torch import nn
from typing import Sequence

from src.modules.components.blocks import ConvBlock


class CNN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int] = (16, 32),
        compile: bool = False,
    ) -> None:
        super().__init__()

        channels = [in_channels, *hidden_channels, out_channels]
        self.model = nn.Sequential(
            *[ConvBlock(i, o) for i, o in zip(channels[:-1], channels[1:])]
        )
        # Keep the backbone output size independent of the input image resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.model(x)).flatten(start_dim=1)
