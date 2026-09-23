import torch

from src.models.detr.detr_loss import DETRLoss


def test_detr_loss_returns_scalar_for_single_image():
    loss_fn = DETRLoss(
        image_width=100,
        image_height=100,
        num_classes=2,
        class_loss_weight=1.0,
        box_loss_weight=1.0,
        giou_loss_weight=1.0,
    )

    pred_logits = torch.tensor(
        [
            [[2.0, 0.5, 0.2], [0.4, 2.0, 0.9], [1.0, 0.3, 0.7]],
        ],
        dtype=torch.float32,
    )
    pred_boxes = torch.tensor(
        [
            [
                [50.0, 50.0, 20.0, 20.0],
                [60.0, 60.0, 10.0, 10.0],
                [30.0, 30.0, 15.0, 10.0],
            ]
        ],
        dtype=torch.float32,
    )
    gt_labels = [torch.tensor([0, 1], dtype=torch.long)]
    gt_boxes = [
        torch.tensor(
            [
                [50.0, 50.0, 20.0, 20.0],
                [60.0, 60.0, 10.0, 10.0],
            ],
            dtype=torch.float32,
        )
    ]

    mean_total_loss, _ = loss_fn(pred_logits, pred_boxes, gt_labels, gt_boxes)

    assert torch.is_tensor(mean_total_loss)
    assert mean_total_loss.ndim == 0
    assert torch.isfinite(mean_total_loss)


def test_detr_loss_handles_empty_ground_truth():
    loss_fn = DETRLoss(
        image_width=100,
        image_height=100,
        num_classes=2,
    )

    pred_logits = torch.randn(1, 4, 3)
    pred_boxes = torch.randn(1, 4, 4)
    gt_labels = [torch.empty(0, dtype=torch.long)]
    gt_boxes = [torch.empty((0, 4), dtype=torch.float32)]

    mean_total_loss, _ = loss_fn(pred_logits, pred_boxes, gt_labels, gt_boxes)

    assert torch.is_tensor(mean_total_loss)
    assert mean_total_loss.ndim == 0
    assert torch.isfinite(mean_total_loss)
