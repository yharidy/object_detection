import torch

from src.models.detr.deformable_transformer_encoder import (
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
)


def test_get_reference_points_returns_expected_shape_and_values():
    encoder_layer = DeformableTransformerEncoderLayer(
        hidden_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
    )

    spatial_shapes = torch.tensor(
        [[80, 80], [40, 40], [20, 20], [10, 10]], dtype=torch.int64
    )
    reference_points = encoder_layer._get_reference_points(spatial_shapes)

    # Check the shape of the reference points
    assert reference_points.shape == (80 * 80 + 40 * 40 + 20 * 20 + 10 * 10, 2)

    # Check that the reference points are within the range [0, 1]
    assert (reference_points >= 0).all() and (reference_points <= 1).all()


def test_get_reference_points_uses_xy_cell_centers_and_level_order():
    encoder_layer = DeformableTransformerEncoderLayer(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
    )
    spatial_shapes = torch.tensor([[2, 3], [1, 1]], dtype=torch.int64)

    reference_points = encoder_layer._get_reference_points(spatial_shapes)

    expected = torch.tensor(
        [
            [1 / 6, 1 / 4],
            [3 / 6, 1 / 4],
            [5 / 6, 1 / 4],
            [1 / 6, 3 / 4],
            [3 / 6, 3 / 4],
            [5 / 6, 3 / 4],
            [1 / 2, 1 / 2],
        ]
    )

    assert torch.allclose(reference_points, expected)


def test_encoder_layer_returns_expected_shape_and_finite_values():
    encoder_layer = DeformableTransformerEncoderLayer(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        ffn_dim=16,
        dropout=0.0,
    )
    spatial_shapes = torch.tensor([[2, 2], [1, 2]], dtype=torch.int64)
    level_start_index = torch.tensor([0, 4], dtype=torch.int64)
    input_flatten = torch.randn(2, 6, 8)
    positional_encodings = torch.randn_like(input_flatten)
    input_padding_mask = torch.zeros(2, 6, dtype=torch.bool)

    output = encoder_layer(
        input_flatten=input_flatten,
        positional_encodings=positional_encodings,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        input_padding_mask=input_padding_mask,
    )

    assert output.shape == input_flatten.shape
    assert torch.isfinite(output).all()


def test_encoder_stack_returns_expected_shape_and_finite_values():
    encoder = DeformableTransformerEncoder(
        hidden_dim=8,
        num_heads=2,
        num_levels=2,
        num_points=2,
        ffn_dim=16,
        dropout=0.0,
        num_layers=2,
    )
    spatial_shapes = torch.tensor([[2, 2], [1, 2]], dtype=torch.int64)
    level_start_index = torch.tensor([0, 4], dtype=torch.int64)
    input_flatten = torch.randn(2, 6, 8)
    positional_encodings = torch.randn_like(input_flatten)
    input_padding_mask = torch.zeros(2, 6, dtype=torch.bool)

    output = encoder(
        input_flatten=input_flatten,
        positional_encodings=positional_encodings,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        input_padding_mask=input_padding_mask,
    )

    assert output.shape == input_flatten.shape
    assert torch.isfinite(output).all()
