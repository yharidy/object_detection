"""Backbone utilities for multi-scale feature extraction."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter

from .position_embedding import PositionEmbeddingSine


class BackboneWithPositionEmbedding(nn.Module):
    """Extract projected multi-scale ResNet-18 features and 2D positions.

    The first ``num_levels`` ResNet stages are exposed. Each stage is projected
    to ``hidden_dim`` channels, and its padding mask is downsampled to the
    corresponding feature-map resolution.
    """

    def __init__(
        self, hidden_dim: int = 256, pretrained: bool = True, num_levels: int = 4
    ) -> None:
        """Initialize the backbone and per-level input projections.

        Args:
            hidden_dim: Number of channels produced at every feature level.
                Must be positive and even because it is also used by the
                positional embedding.
            pretrained: Whether to initialize ResNet-18 with torchvision's
                default pretrained weights.
            num_levels: Number of intermediate ResNet stages to expose. The
                implementation provides projections for at most four stages.
        """
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be a positive even integer")

        self.hidden_dim = hidden_dim
        self.pretrained = pretrained
        self.num_levels = num_levels
        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.levels = {f"layer{i}": f"layer{i}" for i in range(1, num_levels + 1)}
        self.backbone_with_interm_layers = IntermediateLayerGetter(
            self.backbone,
            return_layers=self.levels,
        )
        self.input_projections = nn.ModuleList(
            [
                nn.Conv2d(64, self.hidden_dim, kernel_size=1),
                nn.Conv2d(128, self.hidden_dim, kernel_size=1),
                nn.Conv2d(256, self.hidden_dim, kernel_size=1),
                nn.Conv2d(512, self.hidden_dim, kernel_size=1),
            ]
        )
        self.position_embedding = PositionEmbeddingSine(
            num_pos_features=hidden_dim // 2
        )

    def forward(
        self, images: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Run the backbone and return one tensor per feature level.

        Args:
            images: Input images with shape [B, C, H, W].
            mask: Optional padding mask with shape [B, H, W]. True indicates padded
                pixels.

        Returns:
            A tuple ``(features, positional_encodings, masks)``. Each list has
            ``num_levels`` tensors. Feature and positional tensors have shape
            ``[B, hidden_dim, H_i, W_i]``; masks have shape ``[B, H_i, W_i]``.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected images to be 4D, but got {images.dim()}D")

        batch_size, _, image_height, image_width = images.shape
        if mask is None:
            mask = torch.zeros(
                batch_size,
                image_height,
                image_width,
                dtype=torch.bool,
                device=images.device,
            )
        elif mask.shape != (batch_size, image_height, image_width):
            raise ValueError(
                "mask must have shape [batch_size, image_height, image_width]"
            )
        elif mask.dtype != torch.bool:
            raise ValueError("mask must have dtype torch.bool")
        elif mask.device != images.device:
            raise ValueError("mask and images must be on the same device")

        outputs = self.backbone_with_interm_layers(images)
        feature_maps = [
            self.input_projections[i](outputs[layer])
            for i, layer in enumerate(self.levels.keys())
        ]
        masks = [
            F.interpolate(
                mask[:, None].float(), size=outputs[layer].shape[-2:], mode="nearest"
            ).to(torch.bool)[:, 0]
            for layer in self.levels
        ]
        positional_encodings = [
            self.position_embedding(feature_map, level_mask)
            for feature_map, level_mask in zip(feature_maps, masks)
        ]
        return feature_maps, positional_encodings, masks
