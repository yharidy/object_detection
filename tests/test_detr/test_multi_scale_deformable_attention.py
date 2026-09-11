import pytest
import torch

from src.models.detr.feature_utils import flatten_multi_scale_features
from src.models.detr.multi_scale_deformable_attention import (
    MultiScaleDeformableAttention,
)


def _build_attention_inputs(
    batch_size: int = 1,
    hidden_dim: int = 8,
    num_levels: int = 2,
):
    features = [
        torch.randn(batch_size, hidden_dim, 2, 2),
        torch.randn(batch_size, hidden_dim, 1, 2),
    ]
    positions = [torch.randn_like(feature) for feature in features]
    masks = [
        torch.zeros(batch_size, 2, 2, dtype=torch.bool),
        torch.zeros(batch_size, 1, 2, dtype=torch.bool),
    ]

    src_flatten, pos_flatten, mask_flatten, spatial_shapes, level_start_index = (
        flatten_multi_scale_features(features, positions, masks)
    )

    reference_points = torch.zeros(batch_size, 3, num_levels, 2)
    return {
        "query": torch.randn(batch_size, 3, hidden_dim),
        "input_flatten": src_flatten,
        "reference_points": reference_points,
        "spatial_shapes": spatial_shapes,
        "level_start_index": level_start_index,
        "input_padding_mask": mask_flatten,
    }


def test_multi_scale_deformable_attention_returns_expected_shape_and_finite_values():
    attention = MultiScaleDeformableAttention(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    inputs = _build_attention_inputs(batch_size=1, hidden_dim=8, num_levels=2)

    out = attention(**inputs)

    assert out.shape == (1, 3, 8)
    assert torch.isfinite(out).all()


def test_multi_scale_deformable_attention_zero_weights_produce_zero_output():
    attention = MultiScaleDeformableAttention(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    for module in (attention.sampling_offsets, attention.attention_weights):
        with torch.no_grad():
            module.weight.zero_()
            module.bias.zero_()

    inputs = _build_attention_inputs(batch_size=1, hidden_dim=8, num_levels=2)
    inputs["query"] = torch.zeros_like(inputs["query"])
    inputs["input_flatten"] = torch.zeros_like(inputs["input_flatten"])
    inputs["reference_points"] = torch.zeros_like(inputs["reference_points"])

    out = attention(**inputs)

    assert torch.allclose(out, torch.zeros_like(out))


def test_multi_scale_deformable_attention_rejects_bad_reference_points_shape():
    attention = MultiScaleDeformableAttention(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    inputs = _build_attention_inputs(batch_size=1, hidden_dim=8, num_levels=2)
    inputs["reference_points"] = torch.zeros(1, 3, 1, 2)

    with pytest.raises(ValueError, match="reference_points"):
        attention(**inputs)


def test_multi_scale_deformable_attention_works_with_multiple_queries():
    attention = MultiScaleDeformableAttention(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    inputs = _build_attention_inputs(batch_size=2, hidden_dim=8, num_levels=2)

    out = attention(**inputs)

    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()


def test_multi_scale_deformable_attention_keeps_query_and_head_order_aligned():
    attention = MultiScaleDeformableAttention(
        hidden_dim=2,
        num_heads=2,
        num_levels=1,
        num_points=1,
    )
    with torch.no_grad():
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.zero_()
        attention.attention_weights.weight.zero_()
        attention.attention_weights.bias.zero_()
        attention.value_proj.weight.copy_(torch.eye(2))
        attention.value_proj.bias.zero_()
        attention.output_proj.weight.copy_(torch.eye(2))
        attention.output_proj.bias.zero_()

    output = attention(
        query=torch.zeros(1, 2, 2),
        input_flatten=torch.tensor([[[1.0, 10.0]]]),
        reference_points=torch.tensor([[[[0.5, 0.5]], [[0.5, 0.5]]]]),
        spatial_shapes=torch.tensor([[1, 1]]),
        level_start_index=torch.tensor([0]),
        input_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
    )

    expected = torch.tensor([[[1.0, 10.0], [1.0, 10.0]]])
    assert torch.allclose(output, expected)
