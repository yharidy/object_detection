from dataclasses import dataclass

import numpy as np

from src.domain.enums import CameraPosition


@dataclass
class CameraImage:
    """Loaded camera image and associated metadata."""

    camera: CameraPosition
    timestamp_micros: int
    image: np.ndarray
