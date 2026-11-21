import torch
from torch import nn
from jaxtyping import Float


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Float[torch.Tensor, "batch in_features"]) -> Float[torch.Tensor, "batch out_features"]:
        if x.ndim != 2 or x.shape[-1] != self.net[0].in_features:
            raise ValueError(
                f"LinearBlock expected shape (batch, {self.net[0].in_features}), got {tuple(x.shape)}"
            )

        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
            nn.AvgPool2d(pool_size),
        )

    def forward(self, x: Float[torch.Tensor, "batch in_channels height width"]) -> Float[torch.Tensor, "batch out_channels height width"]:
        conv = self.net[0]

        if x.ndim != 4 or x.shape[1] != conv.in_channels:
            raise ValueError(
                f"ConvBlock expected BCHW input with {conv.in_channels} channels, got {tuple(x.shape)}"
            )

        return self.net(x)
