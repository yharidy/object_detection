from enum import Enum


class CameraPosition(Enum):
    """Camera IDs in the Waymo Open Dataset."""

    FRONT = "front"
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"


class LidarPosition(Enum):
    """Lidar IDs in the Waymo Open Dataset."""

    TOP = "top"
    FRONT = "front"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    REAR = "rear"


class ObjectClass(Enum):
    """Object class IDs."""

    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    SIGN = 3
    CYCLIST = 4
