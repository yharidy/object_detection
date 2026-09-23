import torch

from src.models.detr.deformable_detr import DeformableDetr


def test_deformable_detr_forward_returns_decoder_outputs_and_memory():
    model = DeformableDetr(
        num_levels=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_classes=3,
        num_heads=2,
        hidden_dim=8,
        num_queries=5,
        pretrained_backbone=False,
    )

    decoder_outputs, encoder_memory = model(
        images=torch.randn(1, 3, 64, 64),
        masks=torch.zeros(1, 64, 64, dtype=torch.bool),
    )
    queries, intermediate_boxes, intermediate_logits = decoder_outputs

    assert queries.shape == (1, 5, 8)
    assert torch.isfinite(queries).all()
    for boxes in intermediate_boxes:
        assert boxes.shape == (1, 5, 4)
        assert torch.isfinite(boxes).all()
    for class_logits in intermediate_logits:
        assert class_logits.shape == (1, 5, 4)
        assert torch.isfinite(class_logits).all()
    assert encoder_memory.shape == (1, 340, 8)
    assert torch.isfinite(encoder_memory).all()
