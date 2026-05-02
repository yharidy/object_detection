import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)


def build_faster_rcnn(num_classes: int) -> nn.Module:
    """Build a Faster R-CNN model with a ResNet-50 backbone for 2D object detection.

    Args:
        num_classes: The number of object classes (including background) that the model should predict.
    Returns:
        An instance of torchvision's Faster R-CNN model configured for the specified number of classes.
    """
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
