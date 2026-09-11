import scipy
import torch

from ...utils.box_utils import cxcywh_to_xyxy


class Matcher:
    """Find minimum-cost prediction/target assignments with Hungarian matching.

    Matching combines classification, L1 box, and generalized-IoU costs. The
    public ``match_one`` method operates on one image; ``match`` applies it to
    every image in a batch.
    """

    def __init__(
        self,
        image_width: float,
        image_height: float,
        class_cost_weight: float = 1.0,
        bbox_cost_weight: float = 1.0,
        giou_cost_weight: float = 1.0,
    ):
        """Initialize image dimensions and matching-cost weights.

        Args:
            image_width: Width of the input images.
            image_height: Height of the input images.
            class_cost_weight: Weight for classification cost.
            bbox_cost_weight: Weight for L1 bounding-box cost.
            giou_cost_weight: Weight for generalized-IoU cost.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.class_cost_weight = class_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.giou_cost_weight = giou_cost_weight

    def _calculate_class_cost(
        self, image_logits: torch.Tensor, image_target_labels: torch.Tensor
    ):
        """Return the ``[num_predictions, num_targets]`` class-cost matrix."""
        probabilities = image_logits.softmax(dim=-1)  # [n_queries, n_classes]
        return -probabilities[:, image_target_labels]  # [n_pred, n_gt]

    def _calculate_l1_box_cost(
        self, image_boxes: torch.Tensor, image_target_boxes: torch.Tensor
    ):
        """Return pairwise L1 distances between prediction and target boxes."""
        return torch.cdist(image_boxes, image_target_boxes, p=1.0)

    def _calculate_giou_cost(self, pred_boxes_xyxy, target_boxes_xyxy):
        """Compute pairwise generalized IoU for valid ``xyxy`` boxes.

        GIoU extends IoU to non-overlapping boxes by penalizing the area of the
        smallest enclosing box. Inputs must have strictly positive width and
        height.

        Args:
            pred_boxes_xyxy: Prediction boxes with shape ``[N, 4]``.
            target_boxes_xyxy: Target boxes with shape ``[M, 4]``.

        Returns:
            Pairwise GIoU matrix with shape ``[N, M]``.
        """
        if not (pred_boxes_xyxy[:, 2:] >= pred_boxes_xyxy[:, :2]).all():
            raise ValueError("pred boxes have zero or negative width/height")
        if not (target_boxes_xyxy[:, 2:] >= target_boxes_xyxy[:, :2]).all():
            raise ValueError("target boxes have zero or negative width/height")
        if (pred_boxes_xyxy[:, 2] <= pred_boxes_xyxy[:, 0]).any() or (
            pred_boxes_xyxy[:, 3] <= pred_boxes_xyxy[:, 1]
        ).any():
            raise ValueError("pred boxes have zero or negative size")

        if (target_boxes_xyxy[:, 2] <= target_boxes_xyxy[:, 0]).any() or (
            target_boxes_xyxy[:, 3] <= target_boxes_xyxy[:, 1]
        ).any():
            raise ValueError("target boxes have zero or negative size")

        def _box_iou(boxes1, boxes2):
            area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
            rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
            wh = (rb - lt).clamp(min=0)  # [N,M,2]
            inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

            union = area1[:, None] + area2 - inter
            iou = torch.zeros_like(union)
            valid = union > 0
            iou[valid] = inter[valid] / union[valid]
            return iou, union

        iou, union = _box_iou(pred_boxes_xyxy, target_boxes_xyxy)

        lt_out = torch.min(pred_boxes_xyxy[:, None, :2], target_boxes_xyxy[:, :2])
        rb_out = torch.max(pred_boxes_xyxy[:, None, 2:], target_boxes_xyxy[:, 2:])
        wh = (rb_out - lt_out).clamp(min=0)
        enclosed_area = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        return iou - (enclosed_area - union) / enclosed_area

    def match_one(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ):
        """Match predictions and targets for one image.

        Args:
            logits: Query logits with shape ``[Q, num_classes + 1]``.
            boxes: Predicted normalized ``cxcywh`` boxes with shape ``[Q, 4]``.
            gt_labels: Target foreground class indices with shape ``[N]``.
            gt_boxes: Target pixel-space ``cxcywh`` boxes with shape ``[N, 4]``.

        Returns:
            Two long-index tensors ``(prediction_indices, target_indices)``.
            Both are empty when the image has no targets.
        """
        if gt_labels.numel() == 0 or gt_boxes.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=logits.device),
                torch.empty(0, dtype=torch.long, device=logits.device),
            )

        # classification cost
        cost_class = self._calculate_class_cost(logits, gt_labels)

        # L1 box loss
        normalized_target_boxes = gt_boxes / torch.tensor(
            [
                self.image_width,
                self.image_height,
                self.image_width,
                self.image_height,
            ],
            device=gt_boxes.device,
            dtype=gt_boxes.dtype,
        )

        cost_bbox = self._calculate_l1_box_cost(boxes, normalized_target_boxes)

        # GIOU cost
        image_boxes_xyxy = cxcywh_to_xyxy(boxes)
        image_target_boxes_normalized_xyxy = cxcywh_to_xyxy(normalized_target_boxes)
        cost_giou = -self._calculate_giou_cost(
            image_boxes_xyxy, image_target_boxes_normalized_xyxy
        )

        # total cost
        cost_matrix = (
            self.class_cost_weight * cost_class
            + self.bbox_cost_weight * cost_bbox
            + self.giou_cost_weight * cost_giou
        )  # [N,M]
        pred_idx, gt_idx = scipy.optimize.linear_sum_assignment(
            cost_matrix.detach().cpu().numpy()
        )
        return (
            torch.as_tensor(pred_idx, dtype=torch.long, device=logits.device),
            torch.as_tensor(gt_idx, dtype=torch.long, device=logits.device),
        )

    def match(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match each image in a batch independently.

        Args:
            pred_logits: Class logits with shape ``[B, Q, num_classes + 1]``.
            pred_boxes: Normalized ``cxcywh`` boxes with shape ``[B, Q, 4]``.
            target_labels: List of target-label tensors, one ``[N_i]`` tensor
                per image.
            target_boxes: List of pixel-space ``cxcywh`` target boxes, one
                ``[N_i, 4]`` tensor per image.

        Returns:
            A list of ``(prediction_indices, target_indices)`` pairs, one per
            image.
        """
        if len(target_boxes) != len(target_labels):
            raise ValueError(
                f" Mismatching length of target labels ({len(target_labels)}) and target boxes ({len(target_boxes)})"
            )
        matched_indices = []
        for i in range(len(target_boxes)):
            image_logits = pred_logits[i]
            image_boxes = pred_boxes[i]
            gt_labels = target_labels[i]
            gt_boxes = target_boxes[i]

            if gt_labels.numel() == 0:
                matched_indices.append(
                    (
                        torch.empty(0, dtype=torch.long, device=pred_boxes.device),
                        torch.empty(0, dtype=torch.long, device=pred_boxes.device),
                    )
                )
                continue
            pred_idx, gt_idx = self.match_one(
                image_logits, image_boxes, gt_labels, gt_boxes
            )

            matched_indices.append(
                (
                    torch.as_tensor(
                        pred_idx, dtype=torch.long, device=pred_logits.device
                    ),
                    torch.as_tensor(
                        gt_idx, dtype=torch.long, device=pred_logits.device
                    ),
                )
            )

        return matched_indices
