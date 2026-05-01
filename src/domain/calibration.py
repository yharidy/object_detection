from dataclasses import dataclass, field

import numpy as np

from src.domain.enums import CameraPosition, LidarPosition


@dataclass
class CameraIntrinsicsPinhole:
    """Pinhole camera intrinsics used for image projection."""

    # u = focal_length_u * x / z + principal_point_u
    # v = focal_length_v * y / z + principal_point_v
    width: int
    height: int
    focal_length_u: float
    focal_length_v: float
    principal_point_u: float
    principal_point_v: float


@dataclass
class CameraIntrinsicsBrownConrady:
    """Brown–Conrady camera intrinsics model for distortion-aware projection."""

    # 1. x_n = x / z, y_n = y / z
    # 2. r^2 = x_n^2 + y_n^2
    # 3. x_d = x_r * (1 + k1*r^2 + k2*r^4 + k3*r^6), y_d = y_n * (1 + k1*r^2 + k2*r^4 + k3*r^6)  # radial distortion
    # 4. x_d = x_r + 2*p1*x_n*y_n + p2*(r^2 + 2*x_n^2), y_d = y_r + p1*(r^2 + 2*y_n^2) + 2*p2*x_n*y_n  # tangential distortion
    # 5. u = focal_length_u * x_d + principal_point_u, v = focal_length_v * y_d + principal_point_v
    width: int
    height: int
    focal_length_u: float
    focal_length_v: float
    principal_point_u: float
    principal_point_v: float
    radial_distortion_k1: float
    radial_distortion_k2: float
    radial_distortion_k3: float
    tangential_distortion_p1: float
    tangential_distortion_p2: float


@dataclass
class CameraCalibration:
    """Camera calibration parameters including intrinsics and extrinsics."""

    camera: CameraPosition
    intrinsic_matrix: CameraIntrinsicsPinhole | CameraIntrinsicsBrownConrady
    extrinsic_matrix: np.ndarray  # 4x4


@dataclass
class LidarCalibration:
    """LiDAR calibration parameters including beam inclinations and extrinsics."""

    lidar: LidarPosition
    extrinsic_matrix: np.ndarray  # 4x4 lidar_frame_to_vehicle_frame
    beam_inclination_min: float
    beam_inclination_max: float
    beam_inclinations: np.ndarray  # shape (num_beams,)


@dataclass
class SensorRig:
    """Sensor rig definition for a dataset segment."""

    # segment-level set of sensors
    cameras: dict[CameraPosition, CameraCalibration] = field(default_factory=dict)
    lidars: dict[LidarPosition, LidarCalibration] = field(default_factory=dict)
