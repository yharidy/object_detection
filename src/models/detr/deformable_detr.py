import torch
from torch import nn

from .backbone import BackboneWithPositionEmbedding
from .deformable_transformer_decoder import DeformableTransformerDecoder
from .deformable_transformer_encoder import DeformableTransformerEncoder
from .feature_utils import flatten_multi_scale_features


class DeformableDetr(nn.Module):
    """End-to-end Deformable DETR backbone, encoder, and decoder model."""

    def __init__(
        self,
        num_levels: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_classes: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        num_queries: int = 100,
        pretrained_backbone: bool = True,
    ):
        """Construct the detector and all of its transformer components.

        Args:
            num_levels: Number of multi-scale feature levels.
            num_encoder_layers: Number of deformable encoder layers.
            num_decoder_layers: Number of decoder layers and box-refinement
                stages.
            num_classes: Number of foreground object classes. The decoder adds
                one no-object class to its logits.
            num_heads: Number of attention heads.
            hidden_dim: Shared feature and transformer dimension.
            num_queries: Number of learned object queries.
            pretrained_backbone: Whether to use torchvision's pretrained
                ResNet-18 weights.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.backbone = BackboneWithPositionEmbedding(
            hidden_dim=self.hidden_dim,
            pretrained=pretrained_backbone,
            num_levels=num_levels,
        )
        self.encoder = DeformableTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_layers=num_encoder_layers,
        )
        self.decoder = DeformableTransformerDecoder(
            n_queries=num_queries,
            query_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            num_levels=num_levels,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor, masks: torch.Tensor | None):
        """Run images through the backbone, encoder, and decoder.

        Args:
            images: Input images with shape [B, C, H, W].
            masks: Optional padding mask with shape ``[B, H, W]``. ``True``
                indicates padded pixels.

        Returns:
            A tuple ``(decoder_output, encoder_memory)``. ``decoder_output``
            is itself ``(queries, boxes, class_logits)``. Boxes are normalized
            ``cxcywh`` values with shape ``[B, num_queries, 4]`` and logits
            have shape ``[B, num_queries, num_classes + 1]``.
        """
        features, pos_encodings, feature_masks = self.backbone(images, masks)
        src_flatten, pos_flatten, mask_flatten, spatial_shapes, level_start_index = (
            flatten_multi_scale_features(features, pos_encodings, feature_masks)
        )
        encoder_memory = self.encoder(
            src_flatten,
            pos_flatten,
            spatial_shapes,
            level_start_index,
            mask_flatten,
        )
        decoder_output = self.decoder(
            encoder_memory,
            spatial_shapes,
            level_start_index,
            mask_flatten,
        )
        return decoder_output, encoder_memory
