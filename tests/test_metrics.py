import torch

from src.utils.metrics import DetectionRecord, compute_detection_metrics


def record(boxes, labels, scores=None):
    return DetectionRecord(
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor(labels, dtype=torch.long),
        scores=(
            torch.tensor(scores, dtype=torch.float32) if scores is not None else None
        ),
    )


def test_perfect_detection_metrics():
    predictions = [record([[0, 0, 10, 10]], [0], [0.9])]
    targets = [record([[0, 0, 10, 10]], [0])]

    metrics = compute_detection_metrics(predictions, targets, num_classes=1)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["tp_rate"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["ap50"] == 1.0
    assert metrics["ap75"] == 1.0
    assert metrics["map"] == 1.0


def test_metrics_count_class_and_iou_mismatches():
    predictions = [
        record(
            [[0, 0, 10, 10], [20, 20, 30, 30]],
            [0, 1],
            [0.9, 0.8],
        )
    ]
    targets = [record([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0])]

    metrics = compute_detection_metrics(predictions, targets, num_classes=2)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
