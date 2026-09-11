import pytest
import torch

from src.models.detr.deformable_transformer_decoder import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
)


def _decoder_inputs():
    return {
        "encoder_memory": torch.randn(2, 6, 8),
        "queries": torch.randn(2, 3, 8),
        "reference_points": torch.full((2, 3, 2, 2), 0.5),
        "spatial_shapes": torch.tensor([[2, 2], [1, 2]], dtype=torch.long),
        "level_start_index": torch.tensor([0, 4], dtype=torch.long),
    }


def test_decoder_layer_returns_expected_shape_without_padding_mask():
    layer = DeformableTransformerDecoderLayer(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        ffn_dim=16,
        dropout=0.0,
    )

    output = layer(**_decoder_inputs())

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()


def test_decoder_stack_returns_queries_boxes_and_class_logits():
    decoder = DeformableTransformerDecoder(
        n_queries=3,
        query_dim=8,
        num_layers=2,
        num_heads=2,
        num_levels=2,
        num_points=2,
        ffn_dim=16,
        dropout=0.0,
    )
    encoder_memory = torch.randn(2, 6, 8)
    spatial_shapes = torch.tensor([[2, 2], [1, 2]], dtype=torch.long)
    level_start_index = torch.tensor([0, 4], dtype=torch.long)

    queries, boxes, class_logits = decoder(
        encoder_output=encoder_memory,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    assert queries.shape == (2, 3, 8)
    assert boxes.shape == (2, 3, 4)
    assert class_logits.shape == (2, 3, 92)
    assert (boxes >= 0).all()
    assert (boxes <= 1).all()
    assert torch.isfinite(queries).all()
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(class_logits).all()


def test_decoder_layer_rejects_invalid_activation():
    with pytest.raises(ValueError, match="activation"):
        DeformableTransformerDecoderLayer(
            hidden_dim=8,
            num_heads=2,
            num_levels=2,
            num_points=2,
            activation="tanh",
        )
