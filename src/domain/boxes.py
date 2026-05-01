from dataclasses import dataclass


@dataclass
class Box2D:
    """2D bounding box in image coordinates."""

    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class Box3D:
    """3D bounding box in the vehicle coordinate frame."""

    center_x: float
    center_y: float
    center_z: float
    width: float
    length: float
    height: float
    heading: float  # rotation around z-axis in radians, counter-clockwise positive
