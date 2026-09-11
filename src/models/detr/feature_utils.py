"""Helpers for flattening and reconstructing multi-scale feature maps."""

import torch


def flatten_multi_scale_features(
    features: list[torch.Tensor],
    positions: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten aligned feature levels into deformable-attention metadata.

    Args:
        features: List of feature maps with shape [B, C, H_i, W_i].
        positions: List of positional encodings shaped like the feature maps.
        masks: List of boolean masks with shape [B, H_i, W_i].

    Returns:
        A tuple ``(src_flatten, pos_flatten, mask_flatten, spatial_shapes,
        level_start_index)`` with shapes ``[B, S, C]``, ``[B, S, C]``,
        ``[B, S]``, ``[num_levels, 2]``, and ``[num_levels]`` respectively.
        ``S`` is the sum of ``H_i * W_i`` across all levels.
    """
    for feat in features:
        if feat.dim() != 4:
            raise ValueError(
                f"Expected each feature map to be 4D, but got {feat.dim()}D"
            )

    if len(features) != len(positions) or len(features) != len(masks):
        raise ValueError(
            "features, positions, and masks must have the same number of levels"
        )

    for feat, pos, mask in zip(features, positions, masks):
        if feat.shape != pos.shape:
            raise ValueError(
                f"Feature map and positional encoding must have the same shape, but got {feat.shape} and {pos.shape}"
            )
        if feat.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Feature map and mask must have the same batch size, but got {feat.shape[0]} and {mask.shape[0]}"
            )
        if feat.shape[2:] != mask.shape[1:]:
            raise ValueError(
                f"Feature map and mask must have the same spatial dimensions, but got {feat.shape[2:]} and {mask.shape[1:]}"
            )

    resized_features = [feat.flatten(2).transpose(1, 2) for feat in features]
    resized_positions = [pos.flatten(2).transpose(1, 2) for pos in positions]
    resized_masks = [mask.flatten(1) for mask in masks]

    src_flatten = torch.cat(resized_features, dim=1)
    pos_flatten = torch.cat(resized_positions, dim=1)
    mask_flatten = torch.cat(resized_masks, dim=1)

    spatial_shapes = torch.tensor(
        [(feat.shape[2], feat.shape[3]) for feat in features],
        dtype=torch.long,
        device=features[0].device,
    )
    level_start_index = torch.cat(
        (
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1],
        )
    )
    return src_flatten, pos_flatten, mask_flatten, spatial_shapes, level_start_index
