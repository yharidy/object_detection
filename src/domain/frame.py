"""Data models used by the Waymo data loader and parser."""

from dataclasses import dataclass, field

from src.domain.camera import CameraImage
from src.domain.enums import CameraPosition, LidarPosition
from src.domain.labels import CameraLabel, LidarLabel
from src.domain.lidar import LidarPointCloud, LidarRangeImage


@dataclass
class Frame:
    """Parsed frame payload containing camera and LiDAR observations."""

    timestamp_micros: int
    camera_images: dict[CameraPosition, CameraImage] = field(default_factory=dict)
    lidar_range_images: dict[LidarPosition, list[LidarRangeImage]] = field(
        default_factory=dict
    )  # as stored in the dataset, may have multiple returns per lidar
    lidar_point_clouds: dict[LidarPosition, list[LidarPointCloud]] = field(
        default_factory=dict
    )  # converted from range images
    camera_labels: dict[CameraPosition, list[CameraLabel]] = field(default_factory=dict)
    lidar_labels: list[LidarLabel] = field(default_factory=list)
