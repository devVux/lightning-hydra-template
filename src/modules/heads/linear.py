import torch
from torch import nn
from jaxtyping import Float

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

    def forward(self, x: Float[torch.Tensor, "batch *features"]) -> Float[torch.Tensor, "batch out_features"]:
        if x.ndim < 2:
            raise ValueError(f"LinearHead expects a batch dimension and features, got shape {tuple(x.shape)}")
        batch_size = x.size()[0]

        x = x.view(batch_size, -1)
        expected = self.model.net[0].in_features
        if x.shape[1] != expected:
            raise ValueError(f"LinearHead expected {expected} input features per sample, got {x.shape[1]}")

        return self.model(x)
