"""Module for loading and representing a Waymo Open Dataset segment."""

from pathlib import Path

from enums import Camera, Lidar
from frame_parser import WaymoFrameParser
from store import WaymoDatasetV2Store

from models import Frame


class WaymoSegment:
    """Class representing a Waymo Open Dataset segment. Provides methods to load and access frames from the segment, as well as the sensor rig information."""

    def __init__(
        self,
        segment_name: str,
        root_dir: Path,
        cameras: list[Camera] | None,
        lidars: list[Lidar] | None,
    ):
        self.store = WaymoDatasetV2Store(root_dir=root_dir, segment_name=segment_name)
        self.parser = WaymoFrameParser()
        self.cameras = cameras
        self.lidars = lidars

    def load(self):
        self.sensor_rig = self.parser.parse_sensor_rig(
            camera_calibration_df=self.store.load_camera_calibrations(self.cameras),
            lidar_calibration_df=self.store.load_lidar_calibrations(self.lidars),
        )
        index_df = self.store.load_camera_images(
            columns=["key.frame_timestamp_micros", "key.camera_name"]
        )
        self.timestamps = sorted(index_df["key.frame_timestamp_micros"].unique())

    def __get_item__(self, idx) -> Frame:
        timestamp = self.timestamps[idx]

        camera_images_df = self.store.load_camera_images(
            cameras=self.cameras,
            filters=[("key.frame_timestamp_micros", "==", timestamp)],
        )
        lidar_data_df = self.store.load_lidar_data(
            lidars=self.lidars,
            filters=[("key.frame_timestamp_micros", "==", timestamp)],
        )
        frame = self.parser.parse_frame(
            timestamp_micros=timestamp,
            camera_images_df=camera_images_df,
            lidar_data_df=lidar_data_df,
            sensor_rig=self.sensor_rig,
        )
        return frame
