import pytest
import torch

from src.models.detr.position_embedding import PositionEmbeddingSine


def test_position_embedding_shape_and_finite_values():
    features = torch.zeros(2, 64, 20, 30)

    position = PositionEmbeddingSine(num_pos_features=32)(features)

    assert position.shape == (2, 64, 20, 30)
    assert torch.isfinite(position).all()


def test_position_embedding_changes_across_spatial_locations():
    features = torch.zeros(1, 64, 20, 30)

    position = PositionEmbeddingSine(num_pos_features=32)(features)

    assert not torch.allclose(position[:, :, 0, 0], position[:, :, 1, 0])
    assert not torch.allclose(position[:, :, 0, 0], position[:, :, 0, 1])


def test_position_embedding_is_independent_of_batch_index():
    features = torch.zeros(2, 64, 20, 30)

    position = PositionEmbeddingSine(num_pos_features=32)(features)

    assert torch.allclose(position[0], position[1])


def test_position_embedding_rejects_non_4d_input():
    features = torch.zeros(64, 20, 30)

    with pytest.raises(ValueError, match="Expected input tensor to be 4D"):
        PositionEmbeddingSine(num_pos_features=32)(features)
