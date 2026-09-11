import torch

from src.models.detr.feature_utils import flatten_multi_scale_features


def test_flatten_multi_scale_features_returns_expected_shapes_and_metadata():
    sizes = [(8, 8), (4, 4), (2, 2), (1, 1)]
    features = [torch.randn(2, 4, height, width) for height, width in sizes]
    positions = [torch.randn_like(feature) for feature in features]
    masks = [torch.zeros(2, height, width, dtype=torch.bool) for height, width in sizes]

    src_flatten, pos_flatten, mask_flatten, spatial_shapes, level_start_index = (
        flatten_multi_scale_features(features, positions, masks)
    )

    assert src_flatten.shape == (2, 85, 4)
    assert pos_flatten.shape == (2, 85, 4)
    assert mask_flatten.shape == (2, 85)
    assert spatial_shapes.tolist() == [[8, 8], [4, 4], [2, 2], [1, 1]]
    assert level_start_index.tolist() == [0, 64, 80, 84]


def test_flatten_multi_scale_features_preserves_level_order():
    features = [
        torch.full((1, 1, 2, 2), 1.0),
        torch.full((1, 1, 1, 2), 2.0),
    ]
    positions = [torch.zeros_like(feature) for feature in features]
    masks = [
        torch.zeros(1, 2, 2, dtype=torch.bool),
        torch.zeros(1, 1, 2, dtype=torch.bool),
    ]

    src_flatten, _, _, _, _ = flatten_multi_scale_features(features, positions, masks)

    assert src_flatten[0, :, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]


def test_flatten_multi_scale_features_rejects_incompatible_shapes():
    feature = torch.randn(2, 4, 4, 4)
    position = torch.randn(2, 4, 4, 5)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)

    try:
        flatten_multi_scale_features([feature], [position], [mask])
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_flatten_multi_scale_features_rejects_non_4d_feature():
    feature = torch.randn(2, 4, 4)
    position = torch.randn(2, 4, 4, 4)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)

    try:
        flatten_multi_scale_features([feature], [position], [mask])
    except ValueError as exc:
        assert "each feature map to be 4D" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
