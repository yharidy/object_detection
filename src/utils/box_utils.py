import torch
from torch import Tensor


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert bounding boxes from (center_x, center_y, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        boxes: A tensor of shape (N, 4) where each row is a bounding box in (cx, cy, w, h) format.
    Returns:
        A tensor of shape (N, 4) where each row is the corresponding bounding box in (x_min, y_min, x_max, y_max) format.
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
