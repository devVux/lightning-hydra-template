import torch
from torch import nn
from typing import Sequence
from jaxtyping import Float

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

    def forward(self, x: Float[torch.Tensor, "batch channels height width"]) -> Float[torch.Tensor, "batch out_channels"]:
        if x.ndim != 4:
            raise ValueError(
                f"CNN expects a BCHW tensor, got shape {tuple(x.shape)}")

        expected = self.model[0].net[0].in_channels

        if x.shape[1] != expected:
            raise ValueError(
                f"CNN expected {expected} input channels, got {x.shape[1]}")

        if min(x.shape[-2:]) < 2 ** len(self.model):
            raise ValueError(
                "CNN input height and width are too small for its pooling layers")

        return self.pool(self.model(x)).flatten(start_dim=1)
