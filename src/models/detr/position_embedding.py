"""Positional embedding utilities for 2D feature maps."""

import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """Create sine/cosine positional embeddings for 2D feature maps."""

    def __init__(
        self,
        num_pos_features: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ) -> None:
        """Initialize the positional-embedding frequency bands.

        Args:
            num_pos_features: Number of frequency features for each spatial
                axis. The returned embedding has twice this many channels.
            temperature: Temperature used to form the frequency wavelengths.
            normalize: Whether to normalize cumulative x/y coordinates before
                applying ``scale``.
            scale: Coordinate scale in radians. Defaults to ``2 * pi``.
        """
        super().__init__()
        if num_pos_features <= 0 or num_pos_features % 2 != 0:
            raise ValueError("num_pos_features must be a positive even integer")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(
        self, tensor: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Build a positional encoding for a 2D feature map.

        Args:
            tensor: Feature map with shape [B, C, H, W].
            mask: Optional padding mask with shape [B, H, W]. True indicates padded
                positions.

        Returns:
            Positional encoding with shape
            ``[B, 2 * num_pos_features, H, W]``.
        """
        if tensor.dim() != 4:
            raise ValueError(f"Expected input tensor to be 4D, but got {tensor.dim()}D")

        batch_size, _, height, width = tensor.shape
        if mask is None:
            mask = torch.zeros(
                (batch_size, height, width), dtype=torch.bool, device=tensor.device
            )
        valid = ~mask

        y_embed = valid.cumsum(dim=1, dtype=torch.float32)
        x_embed = valid.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_features, dtype=torch.float32, device=tensor.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_features
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
