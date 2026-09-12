"""Utility functions for data, boxes, logging, and evaluation."""

from .metrics import DetectionRecord, compute_detection_metrics, evaluate_detr_model

__all__ = [
    "DetectionRecord",
    "compute_detection_metrics",
    "evaluate_detr_model",
]
