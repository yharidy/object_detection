"""Backbone utilities for multi-scale feature extraction."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter

from .position_embedding import PositionEmbeddingSine


class BackboneWithPositionEmbedding(nn.Module):
    """ResNet18 backbone that exposes multi-scale feature maps and positional encodings."""

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be a positive even integer")

        self.hidden_dim = hidden_dim
        self.pretrained = pretrained
        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backbone_with_interm_layers = IntermediateLayerGetter(
            self.backbone,
            return_layers={
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
            },
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
        """Run the backbone and return multi-scale features, encodings, and masks.

        Args:
            images: Input images with shape [B, C, H, W].
            mask: Optional padding mask with shape [B, H, W]. True indicates padded
                pixels.

        Returns:
            A tuple of (features, positional_encodings, masks), where each item is a
            list of tensors for the ResNet stages.
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
            for i, layer in enumerate(["layer1", "layer2", "layer3", "layer4"])
        ]
        masks = [
            F.interpolate(
                mask[:, None].float(), size=outputs[layer].shape[-2:], mode="nearest"
            ).to(torch.bool)[:, 0]
            for layer in ["layer1", "layer2", "layer3", "layer4"]
        ]
        positional_encodings = [
            self.position_embedding(feature_map, level_mask)
            for feature_map, level_mask in zip(feature_maps, masks)
        ]
        return feature_maps, positional_encodings, masks
