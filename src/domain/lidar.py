from dataclasses import dataclass

import numpy as np

from src.domain.enums import LidarPosition


@dataclass
class LidarRangeImage:
    """Loaded LiDAR range image representation."""

    lidar: LidarPosition
    timestamp_micros: int
    range_image: np.ndarray  # HxWx4 [range, intensity, elongation, no_label_zone]
    return_count: int


@dataclass
class LidarPointCloud:
    """Point cloud converted from a LiDAR range image."""

    lidar: LidarPosition
    timestamp_micros: int
    point_cloud: np.ndarray  # Nx4 [x, y,z, intensity]
    return_count: int
