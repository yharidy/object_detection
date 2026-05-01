"""Enums for Waymo Open Dataset components."""

from enum import Enum

from src.domain.enums import CameraPosition, LidarPosition, ObjectClass


class WaymoCamera(Enum):
    """Camera IDs in the Waymo Open Dataset."""

    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5


WAYMO_TO_DOMAIN_CAMERA_MAP = {
    WaymoCamera.FRONT: CameraPosition.FRONT,
    WaymoCamera.FRONT_LEFT: CameraPosition.FRONT_LEFT,
    WaymoCamera.FRONT_RIGHT: CameraPosition.FRONT_RIGHT,
    WaymoCamera.SIDE_LEFT: CameraPosition.SIDE_LEFT,
    WaymoCamera.SIDE_RIGHT: CameraPosition.SIDE_RIGHT,
}


class WaymoLidar(Enum):
    """Lidar IDs in the Waymo Open Dataset."""

    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5


WAYMO_TO_DOMAIN_LIDAR_MAP = {
    WaymoLidar.TOP: LidarPosition.TOP,
    WaymoLidar.FRONT: LidarPosition.FRONT,
    WaymoLidar.SIDE_LEFT: LidarPosition.SIDE_LEFT,
    WaymoLidar.SIDE_RIGHT: LidarPosition.SIDE_RIGHT,
    WaymoLidar.REAR: LidarPosition.REAR,
}


class ClassID(Enum):
    """Object class IDs in the Waymo dataset."""

    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3


WAYMO_TO_DOMAIN_CLASS_MAP = {
    ClassID.UNKNOWN: ObjectClass.UNKNOWN,
    ClassID.VEHICLE: ObjectClass.VEHICLE,
    ClassID.PEDESTRIAN: ObjectClass.PEDESTRIAN,
    ClassID.CYCLIST: ObjectClass.CYCLIST,
}
