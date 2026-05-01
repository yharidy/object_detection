from dataclasses import dataclass

import numpy as np

from src.domain.boxes import Box2D, Box3D
from src.domain.enums import ObjectClass


@dataclass
class CameraLabel:
    """Annotated object label in camera image space."""

    object_id: int
    box_2d: Box2D
    object_class: ObjectClass


@dataclass
class LidarLabel:
    """Annotated object label in LiDAR space."""

    object_id: int
    box_3d: Box3D
    object_class: ObjectClass
    speed: np.ndarray | None = None  # [speed_x, speed_y, speed_z]
    acceleration: np.ndarray | None = None  # [accel_x, accel_y, accel_z]
