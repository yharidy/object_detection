"""Enums for Waymo Open Dataset components."""

from enum import Enum


class Camera(Enum):
    """Camera IDs in the Waymo Open Dataset."""

    FRONT = 1
    FRONT_LEFT = 2
    FRONT_RIGHT = 3
    SIDE_LEFT = 4
    SIDE_RIGHT = 5


class Lidar(Enum):
    """Lidar IDs in the Waymo Open Dataset."""

    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5


class ClassID(Enum):
    """Object class IDs in the Waymo dataset."""

    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
