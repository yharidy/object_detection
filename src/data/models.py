from dataclasses import dataclass, field

import numpy as np
from enums import Camera, ClassID, Lidar


@dataclass
class Box2D:
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class Box3D:
    center_x: float
    center_y: float
    center_z: float
    width: float
    length: float
    height: float
    heading: float  # rotation around z-axis in radians, counter-clockwise positive


@dataclass
class CameraLabel:
    objecd_id: int
    box_2d: Box2D
    class_id: ClassID


@dataclass
class LidarLabel:
    object_id: int
    box_3d: Box3D
    class_id: ClassID
    speed: np.ndarray | None = None  # [speed_x, speed_y, speed_z]
    acceleration: np.ndarray | None = None  # [accel_x, accel_y, accel_z]


@dataclass
class CameraImage:
    camera_name: Camera
    timestamp_micros: int
    image: np.ndarray


@dataclass
class LidarRangeImage:
    lidar_name: Lidar
    timestamp_micros: int
    range_image: np.ndarray  # HxWx4 [range, intensity, elongation, no_label_zone]
    return_count: int


@dataclass
class LidarPointCloud:
    lidar_name: Lidar
    timestamp_micros: int
    point_cloud: np.ndarray  # Nx4 [x, y, z, intensity]


@dataclass
class CameraIntrinsicsPinhole:
    # u = focal_length_u * x / z + principal_point_u
    # v = focal_length_v * y / z + principal_point_v
    focal_length_u: float
    focal_length_v: float
    principal_point_u: float
    principal_point_v: float


@dataclass
class CameraIntrinsicsBrownConrady:
    # 1. x_n = x / z, y_n = y / z
    # 2. r^2 = x_n^2 + y_n^2
    # 3. x_d = x_r * (1 + k1*r^2 + k2*r^4 + k3*r^6), y_d = y_n * (1 + k1*r^2 + k2*r^4 + k3*r^6)  # radial distortion
    # 4. x_d = x_r + 2*p1*x_n*y_n + p2*(r^2 + 2*x_n^2), y_d = y_r + p1*(r^2 + 2*y_n^2) + 2*p2*x_n*y_n  # tangential distortion
    # 5. u = focal_length_u * x_d + principal_point_u, v = focal_length_v * y_d + principal_point_v
    focal_length_u: float
    focal_length_v: float
    principal_point_u: float
    principal_point_v: float
    radial_disortion_k1: float
    radial_disortion_k2: float
    radial_distortion_k3: float
    tangential_distortion_p1: float
    tangential_distortion_p2: float


@dataclass
class CameraCalibration:
    camera_name: Camera
    intrinsic_matrix: CameraIntrinsicsPinhole | CameraIntrinsicsBrownConrady
    extrinsic_matrix: np.ndarray  # 4x4


@dataclass
class LidarCalibration:
    lidar_name: Lidar
    extrinsic_matrix: np.ndarray  # 4x4 lidar_frame_to_vehicle_frame


@dataclass
class SensorRig:
    # segment-level set of sensors
    cameras: dict[Camera, CameraCalibration] = field(default_factory=dict)
    lidars: dict[Lidar, LidarCalibration] = field(default_factory=dict)


@dataclass
class Frame:
    timestamp_micros: int
    camera_images: dict[Camera, CameraImage] = field(default_factory=dict)
    lidar_range_images: dict[Lidar, list[LidarRangeImage]] = field(
        default_factory=dict
    )  # as stored in the dataset, may have multiple returns per lidar
    lidar_point_clouds: dict[Lidar, LidarPointCloud] = field(
        default_factory=dict
    )  # converted from range images
