"""Rerun visualization helper for Waymo dataset frames."""

import time

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

from src.domain.calibration import SensorRig
from src.domain.frame import Frame
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RerunVisualizer:
    """Visualize frames in Rerun with cameras, LiDAR, and labels."""

    def __init__(
        self,
        vis_point_clouds: bool = False,
        vis_range_images: bool = False,
        vis_camera_labels: bool = False,
        vis_lidar_labels: bool = False,
        app_name: str = "waymo_rerun_visualizer",
    ):
        """Initialize the Rerun visualizer application."""
        self.vis_point_clouds = vis_point_clouds
        self.vis_range_images = vis_range_images
        self.vis_camera_labels = vis_camera_labels
        self.vis_lidar_labels = vis_lidar_labels
        rr.init(app_name, spawn=False)
        self._transforms_logged = False

    def setup(self, sensor_rig: SensorRig):
        """Start the Rerun server and log the static sensor transforms."""
        rr.serve_web(open_browser=False)
        logger.info(
            "Run 'rerun+http://localhost:9876/proxy' to view the visualization in Rerun."
        )
        time.sleep(2)
        self._log_sensor_transforms(sensor_rig)
        self._transforms_logged = True

    def log_frame(self, frame_idx: int, frame: Frame):
        """Log a single frame's camera images, LiDAR data, and labels to Rerun."""
        assert (
            self._transforms_logged
        ), "Call setup() with the sensor rig before logging frames."
        rr.set_time(
            "frame",
            timestamp=np.datetime64(int(frame.timestamp_micros), "us"),
        )
        self._log_camera_images(frame)
        if self.vis_range_images:
            self._log_lidar_range_images(frame)
        if self.vis_point_clouds:
            self._log_lidar_point_clouds(frame)
        if self.vis_camera_labels:
            self._log_camera_labels(frame)
        if self.vis_lidar_labels:
            self._log_lidar_labels(frame)

    def _extrinsic_to_transform3d(
        self, extrinsic_matrix, offset_deg: list[float] | None = None
    ):
        """Convert a 4x4 extrinsic matrix into a Rerun Transform3D object."""
        rotation = np.array(extrinsic_matrix[:3, :3], dtype=float)
        rotation = R.from_matrix(rotation)
        if offset_deg is not None:
            assert len(offset_deg) == 3, "Offset must be a list of 3 floats."
            offset_matrix = R.from_euler("xyz", offset_deg, degrees=True).as_matrix()
            rotation = R.from_matrix(rotation.as_matrix() @ offset_matrix)

        translation = np.array(extrinsic_matrix[:3, 3], dtype=float)
        quaternion = rotation.as_quat()
        return rr.Transform3D(
            translation=translation.tolist(),
            quaternion=quaternion.tolist(),
            axis_length=0.1,
        )

    def _camera_intrinsic_to_pinhole(self, intrinsic):
        """Convert camera intrinsics into a Rerun Pinhole camera model."""
        return rr.Pinhole(
            resolution=(intrinsic.width, intrinsic.height),
            focal_length=(intrinsic.focal_length_u, intrinsic.focal_length_v),
            principal_point=(intrinsic.principal_point_u, intrinsic.principal_point_v),
        )

    def _log_sensor_transforms(self, sensor_rig: SensorRig):
        """Log the world, vehicle, camera, and lidar coordinate transforms."""
        rr.log(
            "world/vehicle",
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                quaternion=[0.0, 0.0, 0.0, 1.0],
                axis_length=1.0,
            ),
            static=True,
        )
        for camera, camera_calibration in sensor_rig.cameras.items():
            rr.log("world", rr.ViewCoordinates.FLU, static=True)
            rr.log(
                f"world/vehicle/camera/{camera.name.lower()}",
                self._extrinsic_to_transform3d(
                    camera_calibration.extrinsic_matrix,
                    offset_deg=[
                        -90.0,
                        0.0,
                        -90.0,
                    ],  # Rerun projects camera image on XY plane
                ),
                static=True,
            )
            rr.log(
                f"world/vehicle/camera/{camera.name.lower()}",
                self._camera_intrinsic_to_pinhole(camera_calibration.intrinsic_matrix),
                static=True,
            )

        for lidar, lidar_calibration in sensor_rig.lidars.items():
            rr.log(
                f"world/vehicle/lidar/{lidar.name.lower()}",
                self._extrinsic_to_transform3d(lidar_calibration.extrinsic_matrix),
                static=True,
            )
            rr.log(
                f"world/vehicle/lidar/{lidar.name.lower()}_points",  # only applies transformation since lidar points are aligned with vehicle frame
                rr.Transform3D(
                    translation=np.array(
                        lidar_calibration.extrinsic_matrix[:3, 3], dtype=float
                    ).tolist(),
                    quaternion=np.array(
                        [0, 0, 0, 1], dtype=float
                    ).tolist(),  # point clouds are aligned with vehicle frame
                    axis_length=0.1,
                ),
                static=True,
            )

    def _log_camera_images(self, frame: Frame):
        """Log camera images for the current frame."""
        for camera, camera_image in frame.camera_images.items():
            logger.debug(f"Logging camera image for camera: {camera.name}")
            rr.log(
                f"world/vehicle/camera/{camera.name.lower()}/image",
                rr.Image(camera_image.image),
            )

    def _log_lidar_range_images(self, frame: Frame):
        """Log LiDAR range images for each return in the frame."""
        for lidar, range_images in frame.lidar_range_images.items():
            logger.debug(f"Logging range images for LiDAR: {lidar.name}")
            for range_image in range_images:
                range_channel = range_image.range_image[:, :, 0]
                r_min, r_max = range_channel.min(), range_channel.max()
                normalized = (
                    (range_channel - r_min) / (r_max - r_min + 1e-6) * 255
                ).astype(np.uint8)
                # colormap
                colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
                colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
                rr.log(
                    f"lidar_2d/{lidar.name.lower()}/return{range_image.return_count}",
                    rr.Image(colorized),
                )

    def _log_lidar_point_clouds(self, frame: Frame):
        """Log LiDAR point clouds with intensity colorization."""
        for lidar, point_clouds in frame.lidar_point_clouds.items():
            logger.debug(f"Logging point clouds for LiDAR: {lidar.name}")
            for point_cloud in point_clouds:
                intensity = point_cloud.point_cloud[
                    :, 3
                ]  # intensity is the 4th channel
                p_low, p_high = np.percentile(
                    intensity, [2, 98]
                )  # clip to 2-98 percentile to remove outliers
                intensity_clipped = np.clip(intensity, p_low, p_high)
                grey = (
                    (intensity_clipped - p_low) / (p_high - p_low + 1e-6) * 255
                ).astype(np.uint8)
                colors = np.stack([grey, grey, grey], axis=-1)  # (N, 3)
                rr.log(
                    f"world/vehicle/lidar/{lidar.name.lower()}_points/return{point_cloud.return_count}",
                    rr.Points3D(
                        positions=point_cloud.point_cloud[:, :3],
                        colors=colors,
                    ),
                )

    def _log_camera_labels(self, frame: Frame):
        """Log 2D camera bounding boxes for the current frame."""
        for camera, camera_boxes in frame.camera_labels.items():
            if not camera_boxes:
                continue
            logger.debug(f"Logging camera bounding boxes for camera: {camera.name}")
            centers = np.array(
                [[b.box_2d.center_x, b.box_2d.center_y] for b in camera_boxes]
            )
            sizes = np.array([[b.box_2d.width, b.box_2d.height] for b in camera_boxes])
            class_ids = np.array([b.object_class.value for b in camera_boxes])

            rr.log(
                f"world/vehicle/camera/{camera.name.lower()}/boxes",
                rr.Boxes2D(
                    centers=centers,
                    sizes=sizes,
                    class_ids=class_ids,
                ),
            )

    def _log_lidar_labels(self, frame: Frame):
        """Log 3D LiDAR bounding boxes for the current frame."""
        if not frame.lidar_labels:
            return
        logger.debug("Logging LiDAR bounding boxes.")

        centers = np.array(
            [
                [b.box_3d.center_x, b.box_3d.center_y, b.box_3d.center_z]
                for b in frame.lidar_labels
            ]
        )
        sizes = np.array(
            [
                [b.box_3d.width, b.box_3d.length, b.box_3d.height]
                for b in frame.lidar_labels
            ]
        )
        class_ids = np.array([b.object_class.value for b in frame.lidar_labels])
        half_yaws = np.array([b.box_3d.heading for b in frame.lidar_labels]) / 2

        # xyzw order as plain numpy array
        quaternions = np.stack(
            [
                np.zeros_like(half_yaws),  # x
                np.zeros_like(half_yaws),  # y
                np.sin(half_yaws),  # z
                np.cos(half_yaws),  # w
            ],
            axis=-1,
        )  # (N, 4) in xyzw order

        rr.log(
            "world/vehicle/lidar/top_boxes",
            rr.Boxes3D(
                centers=centers,
                sizes=sizes,
                class_ids=class_ids,
                quaternions=quaternions,
            ),
        )
