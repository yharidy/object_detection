"""Detection metrics and evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .box_utils import cxcywh_to_xyxy


@dataclass
class DetectionRecord:
    """Predictions or targets for one image in pixel-space ``xyxy`` format."""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor | None = None


def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for ``xyxy`` boxes with shapes ``[N, 4]`` and ``[M, 4]``."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)


def _match_image(
    prediction: DetectionRecord,
    target: DetectionRecord,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Greedily match one image's score-sorted predictions to targets."""
    if prediction.scores is None:
        raise ValueError("predictions must include scores")

    order = prediction.scores.argsort(descending=True)
    matched_targets = torch.zeros(
        target.boxes.shape[0], dtype=torch.bool, device=prediction.boxes.device
    )
    true_positive = torch.zeros(
        prediction.boxes.shape[0], dtype=torch.bool, device=prediction.boxes.device
    )

    if target.boxes.shape[0] == 0:
        return true_positive, ~true_positive, 0

    ious = _box_iou(prediction.boxes, target.boxes)
    for prediction_index in order.tolist():
        same_class = target.labels == prediction.labels[prediction_index]
        candidate_ious = ious[prediction_index].masked_fill(~same_class, -1)
        candidate_ious = candidate_ious.masked_fill(matched_targets, -1)
        best_iou, best_target = candidate_ious.max(dim=0)
        if best_iou >= iou_threshold:
            true_positive[prediction_index] = True
            matched_targets[best_target] = True

    false_positive = ~true_positive
    return true_positive, false_positive, int(target.boxes.shape[0])


def _average_precision(
    true_positive: torch.Tensor, scores: torch.Tensor, num_targets: int
) -> float:
    """Compute area under the interpolated precision-recall curve."""
    if num_targets == 0:
        return float("nan")
    if true_positive.numel() == 0:
        return 0.0

    order = scores.argsort(descending=True)
    true_positive = true_positive[order].float()
    false_positive = 1.0 - true_positive
    precision = true_positive.cumsum(0) / torch.arange(
        1, true_positive.numel() + 1, device=true_positive.device
    )
    recall = true_positive.cumsum(0) / num_targets

    precision = torch.cat((precision.new_zeros(1), precision, precision.new_zeros(1)))
    recall = torch.cat((recall.new_zeros(1), recall, recall.new_ones(1)))
    for index in range(precision.numel() - 1, 0, -1):
        precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
    change_points = torch.where(recall[1:] != recall[:-1])[0]
    area = torch.sum(
        (recall[change_points + 1] - recall[change_points])
        * precision[change_points + 1]
    )
    return float(area.item())


def _compute_map(
    predictions: list[DetectionRecord],
    targets: list[DetectionRecord],
    num_classes: int,
    iou_threshold: float,
) -> float:
    class_aps = []
    for class_id in range(num_classes):
        class_predictions: list[tuple[float, bool]] = []
        num_targets = 0
        for prediction, target in zip(predictions, targets):
            prediction_mask = prediction.labels == class_id
            target_mask = target.labels == class_id
            filtered_prediction = DetectionRecord(
                boxes=prediction.boxes[prediction_mask],
                labels=prediction.labels[prediction_mask],
                scores=(
                    prediction.scores[prediction_mask]
                    if prediction.scores is not None
                    else None
                ),
            )
            filtered_target = DetectionRecord(
                boxes=target.boxes[target_mask],
                labels=target.labels[target_mask],
            )
            image_tp, image_fp, image_targets = _match_image(
                filtered_prediction, filtered_target, iou_threshold
            )
            num_targets += image_targets
            if filtered_prediction.scores is not None:
                class_predictions.extend(
                    zip(
                        filtered_prediction.scores.detach().cpu().tolist(),
                        image_tp.detach().cpu().tolist(),
                    )
                )

        if num_targets == 0:
            continue
        if not class_predictions:
            class_aps.append(0.0)
            continue
        scores = torch.tensor([item[0] for item in class_predictions])
        true_positive = torch.tensor([item[1] for item in class_predictions])
        class_aps.append(_average_precision(true_positive, scores, num_targets))

    return float(sum(class_aps) / len(class_aps)) if class_aps else 0.0


def compute_detection_metrics(
    predictions: list[DetectionRecord],
    targets: list[DetectionRecord],
    num_classes: int,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    map_iou_thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    """Compute fixed-threshold KPIs and AP/mAP from image-level detections.

    Boxes must be pixel-space ``xyxy``. Predictions must include one score per
    box; targets should leave ``scores`` as ``None``.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal lengths")
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    filtered_predictions = []
    for prediction in predictions:
        if prediction.scores is None:
            raise ValueError("predictions must include scores")
        keep = prediction.scores >= score_threshold
        filtered_predictions.append(
            DetectionRecord(
                boxes=prediction.boxes[keep],
                labels=prediction.labels[keep],
                scores=prediction.scores[keep],
            )
        )

    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction, target in zip(filtered_predictions, targets):
        image_tp, image_fp, target_count = _match_image(
            prediction, target, iou_threshold
        )
        true_positive += int(image_tp.sum())
        false_positive += int(image_fp.sum())
        false_negative += target_count - int(image_tp.sum())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    thresholds = list(
        map_iou_thresholds
        or (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    )
    aps = [
        _compute_map(predictions, targets, num_classes, threshold)
        for threshold in thresholds
    ]
    return {
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "precision": precision,
        "recall": recall,
        "tp_rate": recall,
        "f1": f1,
        "ap50": _compute_map(predictions, targets, num_classes, 0.50),
        "ap75": _compute_map(predictions, targets, num_classes, 0.75),
        "map": sum(aps) / len(aps),
    }


def evaluate_detr_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    image_width: float,
    image_height: float,
    num_classes: int,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    use_amp: bool = False,
) -> dict[str, float]:
    """Run a Deformable DETR model and compute detection metrics."""
    was_training = model.training
    model.eval()
    predictions = []
    targets = []
    scale = torch.tensor(
        [image_width, image_height, image_width, image_height], device=device
    )

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                decoder_output, _ = model(images, masks=None)
                _, pred_boxes, pred_logits = decoder_output
            probabilities = pred_logits.float().softmax(dim=-1)
            scores, labels = probabilities[..., :-1].max(dim=-1)
            for index in range(images.shape[0]):
                predictions.append(
                    DetectionRecord(
                        boxes=cxcywh_to_xyxy(pred_boxes[index]) * scale,
                        labels=labels[index],
                        scores=scores[index],
                    )
                )
                targets.append(
                    DetectionRecord(
                        boxes=cxcywh_to_xyxy(batch["boxes"][index].to(device)),
                        labels=batch["labels"][index].to(device),
                    )
                )

    if was_training:
        model.train()
    return compute_detection_metrics(
        predictions,
        targets,
        num_classes=num_classes,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
