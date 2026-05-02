import cv2
import numpy as np
import torch

from src.domain.boxes import Box2D


class Camera2DTransform:
    """Transformation class for 2D camera data."""

    def __init__(
        self,
        target_image_size: tuple[int, int],
        norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.target_image_size = target_image_size
        self.norm_mean = torch.tensor(norm_mean).view(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).view(3, 1, 1)

    def _rescale_image_and_labels(
        self, image: np.ndarray, boxes: list[Box2D]
    ) -> tuple[np.ndarray, list[Box2D]]:
        """Rescale image and bounding boxes to the target size.

        Args:
            image: Input image as a numpy array of shape (H, W, C).
            boxes: List of Box2D objects representing bounding boxes in the original image.
        Returns:
            A tuple containing:
            - Rescaled image as a PyTorch Tensor of shape (C, target_height, target_width).
            - Rescaled bounding boxes as a PyTorch Tensor of shape (N, 4) where N is the number of boxes, and each box is represented as (center_x, center_y, width, height) in the rescaled image coordinates.
        """
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.target_image_size

        # Compute scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Rescale the image
        rescaled_image = cv2.resize(
            image, self.target_image_size, interpolation=cv2.INTER_LINEAR
        )

        # Rescale the bounding boxes
        rescaled_boxes = []
        for box in boxes:
            rescaled_center_x = box.center_x * scale_x
            rescaled_center_y = box.center_y * scale_y
            rescaled_width = box.width * scale_x
            rescaled_height = box.height * scale_y
            rescaled_boxes.append(
                Box2D(
                    rescaled_center_x,
                    rescaled_center_y,
                    rescaled_width,
                    rescaled_height,
                )
            )
        return rescaled_image, rescaled_boxes

    def __call__(self, sample: dict) -> dict:
        resized_image, rescaled_boxes = self._rescale_image_and_labels(
            sample["image"], sample["boxes"]
        )

        image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
        image = (image - self.norm_mean) / self.norm_std

        boxes = (
            torch.tensor(
                [[b.center_x, b.center_y, b.width, b.height] for b in rescaled_boxes],
                dtype=torch.float32,
            )
            if rescaled_boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels = (
            torch.tensor([label.value for label in sample["labels"]], dtype=torch.long)
            if sample["labels"]
            else torch.zeros((0,), dtype=torch.long)
        )

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "meta": sample["meta"],
        }
