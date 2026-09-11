import torch

from src.models.detr.backbone import BackboneWithPositionEmbedding


def test_backbone_outputs_multiscale_features_positions_and_masks():
    images = torch.randn(2, 3, 320, 320)

    backbone = BackboneWithPositionEmbedding(hidden_dim=256, pretrained=False)
    features, positions, masks = backbone(images)

    expected_sizes = [(80, 80), (40, 40), (20, 20), (10, 10)]
    assert len(features) == len(positions) == len(masks) == 4

    for feature, position, mask, size in zip(
        features, positions, masks, expected_sizes
    ):
        assert feature.shape == (2, 256, *size)
        assert position.shape == (2, 256, *size)
        assert mask.shape == (2, *size)
        assert torch.isfinite(feature).all()
        assert torch.isfinite(position).all()
