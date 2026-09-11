import pytest
import torch

from src.models.detr.matcher import Matcher


def test_calculate_class_cost_returns_prediction_gt_matrix():
    matcher = Matcher(image_width=100, image_height=100)
    logits = torch.tensor(
        [
            [2.0, 0.5, 1.0],
            [0.2, 1.5, 0.8],
        ]
    )
    gt_labels = torch.tensor([1, 2], dtype=torch.long)

    cost = matcher._calculate_class_cost(logits, gt_labels)

    expected = -logits.softmax(dim=-1)[:, gt_labels]
    assert cost.shape == (2, 2)
    assert torch.allclose(cost, expected)


def test_match_returns_hungarian_assignments_for_one_image():
    matcher = Matcher(image_width=100, image_height=100)
    pred_logits = torch.tensor(
        [
            [[2.0, 0.5, 0.2], [0.4, 2.0, 0.3]],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [
                [50.0, 50.0, 20.0, 20.0],
                [60.0, 60.0, 10.0, 10.0],
            ]
        ]
    )
    target_labels = [torch.tensor([0, 1], dtype=torch.long)]
    target_boxes = [
        torch.tensor(
            [
                [50.0, 50.0, 20.0, 20.0],
                [60.0, 60.0, 10.0, 10.0],
            ]
        )
    ]

    matches = matcher.match(pred_logits, pred_boxes, target_labels, target_boxes)

    assert len(matches) == 1
    pred_idx, gt_idx = matches[0]
    assert pred_idx.shape == gt_idx.shape
    assert pred_idx.numel() == 2
    assert gt_idx.numel() == 2
    assert set(pred_idx.tolist()) == {0, 1}
    assert set(gt_idx.tolist()) == {0, 1}


def test_match_handles_empty_ground_truth_for_an_image():
    matcher = Matcher(image_width=100, image_height=100)
    pred_logits = torch.randn(1, 3, 3)
    pred_boxes = torch.randn(1, 3, 4)
    target_labels = [torch.empty(0, dtype=torch.long)]
    target_boxes = [torch.empty((0, 4), dtype=torch.float32)]

    matches = matcher.match(pred_logits, pred_boxes, target_labels, target_boxes)

    assert len(matches) == 1
    pred_idx, gt_idx = matches[0]
    assert pred_idx.numel() == 0
    assert gt_idx.numel() == 0


def test_giou_cost_returns_finite_values_for_valid_boxes():
    matcher = Matcher(image_width=100, image_height=100)
    pred = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 4.0, 4.0],
        ]
    )
    target = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [2.0, 2.0, 5.0, 5.0],
        ]
    )

    cost = matcher._calculate_giou_cost(pred, target)

    assert cost.shape == (2, 2)
    assert torch.isfinite(cost).all()
