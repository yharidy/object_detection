import torch
from torch.nn import functional as F

from ...utils.box_utils import cxcywh_to_xyxy
from .matcher import Matcher


class DETRLoss:
    """Compute image-wise DETR classification and box regression losses.

    Ground-truth boxes are expected in pixel-space ``cxcywh`` format. Model
    boxes are expected in normalized ``cxcywh`` format. Matching is performed
    independently for each image using classification, L1, and GIoU costs.
    """

    def __init__(
        self,
        image_width: float,
        image_height: float,
        num_classes: int,
        class_loss_weight: float = 1.0,
        box_loss_weight: float = 1.0,
        giou_loss_weight: float = 1.0,
    ):
        """Initialize loss weights, image scaling, and the matcher.

        Args:
            image_width: Width of the transformed training image in pixels.
            image_height: Height of the transformed training image in pixels.
            num_classes: Number of foreground classes; the next index is the
                background/no-object class.
            class_loss_weight: Weight for cross-entropy classification loss.
            box_loss_weight: Weight for matched-box L1 loss.
            giou_loss_weight: Weight for matched-box GIoU loss.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.num_classes = num_classes
        self.class_loss_weight = class_loss_weight
        self.box_loss_weight = box_loss_weight
        self.giou_loss_weight = giou_loss_weight
        self.matcher = Matcher(image_width=image_width, image_height=image_height)

    def _normalize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert pixel-space ``cxcywh`` boxes to normalized ``cxcywh``."""
        scale = torch.tensor(
            [
                self.image_width,
                self.image_height,
                self.image_width,
                self.image_height,
            ],
            device=boxes.device,
            dtype=boxes.dtype,
        )
        return boxes / scale

    def compute_detr_loss(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ):
        """Compute the DETR loss for one image.

        Args:
            logits: Query logits with shape ``[Q, num_classes + 1]``.
            boxes: Predicted normalized ``cxcywh`` boxes with shape ``[Q, 4]``.
            gt_labels: Foreground class indices with shape ``[N]``.
            gt_boxes: Ground-truth pixel-space ``cxcywh`` boxes with shape
                ``[N, 4]``.

        Returns:
            total_loss, class_loss, box_loss, giou_loss
        """
        if logits.shape[-1] != self.num_classes + 1:
            raise ValueError(
                f"Expected logits to have {self.num_classes + 1} classes "
                f"(object classes + background), got {logits.shape[-1]}."
            )

        pred_idx, gt_idx = self.matcher.match_one(logits, boxes, gt_labels, gt_boxes)

        background_class = self.num_classes
        target_classes = torch.full(
            (logits.shape[0],),
            background_class,
            dtype=torch.long,
            device=logits.device,
        )
        if pred_idx.numel() > 0:
            for p, gt in zip(pred_idx, gt_idx):
                target_classes[p] = gt_labels[gt]

        class_loss = F.cross_entropy(logits, target_classes, reduction="mean")

        if pred_idx.numel() == 0:
            box_loss = boxes.new_zeros(())
            giou_loss = boxes.new_zeros(())
        else:
            matched_boxes = boxes[pred_idx]
            matched_gt_boxes = gt_boxes[gt_idx]
            matched_gt_boxes_norm = self._normalize_boxes(matched_gt_boxes)

            box_loss = F.l1_loss(matched_boxes, matched_gt_boxes_norm, reduction="mean")

            pred_xyxy = cxcywh_to_xyxy(matched_boxes)
            gt_xyxy = cxcywh_to_xyxy(matched_gt_boxes_norm)
            giou = self.matcher._calculate_giou_cost(pred_xyxy, gt_xyxy)
            giou_loss = 1.0 - giou.mean()

        total = (
            self.class_loss_weight * class_loss
            + self.box_loss_weight * box_loss
            + self.giou_loss_weight * giou_loss
        )
        return total, class_loss.detach(), box_loss.detach(), giou_loss.detach()

    def __call__(self, pred_logits, pred_boxes, target_labels, target_boxes):
        """Average the per-image DETR loss across a batch.

        Args:
            pred_logits: Batched logits with shape ``[B, Q, num_classes + 1]``.
            pred_boxes: Batched normalized ``cxcywh`` boxes with shape
                ``[B, Q, 4]``.
            target_labels: List of ``[N_i]`` label tensors.
            target_boxes: List of pixel-space ``[N_i, 4]`` target tensors.

        Returns:
            Mean scalar loss over the batch.
            dict containing loss components.
        """
        totals, class_losses, box_losses, giou_losses = [], [], [], []
        for i in range(len(target_labels)):
            total, c, b, g = self.compute_detr_loss(
                self, pred_logits[i], pred_boxes[i], target_labels[i], target_boxes[i]
            )
            totals.append(total)
            class_losses.append(c)
            box_losses.append(b)
            giou_losses.append(g)
        mean_total = torch.stack(totals).mean()
        components = {
            "class": torch.stack(class_losses).mean().item(),
            "box": torch.stack(box_losses).mean().item(),
            "giou": torch.stack(giou_losses).mean().item(),
        }
        return mean_total, components
