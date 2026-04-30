"""Module for loading and representing a Waymo Open Dataset segment."""

from pathlib import Path

from ..utils.logging import get_logger
from .enums import Camera, Lidar
from .frame_parser import WaymoFrameParser
from .models import Frame
from .store import WaymoDatasetV2Store

logger = get_logger(__name__)


class WaymoSegment:
    """Class representing a Waymo Open Dataset segment.

    A WaymoSegment acts as the dataset entry point for a single segment. It
    initializes the Parquet-backed store and the frame parser, loads the sensor
    rig information, and provides indexed access to parsed frames.
    """

    def __init__(
        self,
        segment_name: str,
        root_dir: Path,
        cameras: list[Camera] | None,
        lidars: list[Lidar] | None,
        load_point_clouds: bool = False,
        lidar_returns: list[int] | None = None,
        load_camera_labels: bool = False,
        load_lidar_labels: bool = False,
    ):
        """Initialize the WaymoSegment.

        Args:
            segment_name: Name of the segment to load.
            root_dir: Root directory where the segment data is stored in Parquet format.
            cameras: List of Camera enums to load.
            lidars: List of Lidar enums to load.
            load_point_clouds: Whether to load point clouds for the lidars. If True,
                the point clouds will be converted from the range images using the provided calibrations.
            lidar_returns: List of return indices to load for each lidar. If None, loads only the first return.
            load_camera_labels: Whether to load camera bounding boxes for each frame.
            load_lidar_labels: Whether to load LiDAR bounding boxes for each frame.
        """
        logger.info(
            f"Initializing WaymoSegment with segment_name='{segment_name}', root_dir='{root_dir}', cameras={cameras}, lidars={lidars}, load_point_clouds={load_point_clouds}, lidar_returns={lidar_returns}"
        )
        self.store = WaymoDatasetV2Store(root_dir=root_dir, segment_name=segment_name)
        self.parser = WaymoFrameParser()
        self.cameras = cameras
        self.lidars = lidars
        self.load_point_clouds = load_point_clouds
        self.lidar_returns = lidar_returns
        self.load_camera_labels = load_camera_labels
        self.load_lidar_labels = load_lidar_labels
        if not self.cameras and not self.lidars:
            raise ValueError("At least one camera or lidar must be specified.")
        self._load()

    def _load(self):
        """Load the segment sensor rig and timestamp index."""
        # load calibrations
        camera_calib_df = (
            self.store.load_camera_calibrations(self.cameras) if self.cameras else None
        )
        lidar_calib_df = (
            self.store.load_lidar_calibrations(self.lidars) if self.lidars else None
        )

        self.sensor_rig = self.parser.parse_sensor_rig(
            camera_calibration_df=camera_calib_df,
            lidar_calibration_df=lidar_calib_df,
        )
        # load frame timestamps
        if self.cameras:
            index_df = self.store.load_camera_images(
                columns=["key.frame_timestamp_micros"],
                cameras=[self.cameras[0]],  # use first camera timestamps for indexing
            )

        else:
            index_df = self.store.load_lidar_data(
                columns=["key.frame_timestamp_micros"],
                lidars=[self.lidars[0]],  # use first lidar timestamps for indexing
            )
        self.timestamps = sorted(index_df["key.frame_timestamp_micros"].unique())

    def __getitem__(self, idx) -> Frame:
        """Return the parsed Frame at the given index."""
        timestamp = self.timestamps[idx]
        if self.cameras:
            camera_images_df = self.store.load_camera_images(
                cameras=self.cameras,
                filters=[("key.frame_timestamp_micros", "==", timestamp)],
            )
        else:
            camera_images_df = None

        if self.lidars:
            lidar_data_df = self.store.load_lidar_data(
                lidars=self.lidars,
                filters=[("key.frame_timestamp_micros", "==", timestamp)],
            )
        else:
            lidar_data_df = None

        if self.cameras and self.load_camera_labels:
            camera_labels_df = self.store.load_camera_bboxes(
                cameras=self.cameras,
                filters=[("key.frame_timestamp_micros", "==", timestamp)],
            )
        else:
            camera_labels_df = None
        if self.lidars and self.load_lidar_labels:
            lidar_labels_df = self.store.load_lidar_bboxes(
                filters=[("key.frame_timestamp_micros", "==", timestamp)],
            )
        else:
            lidar_labels_df = None

        frame = self.parser.parse_frame(
            timestamp_micros=timestamp,
            camera_images_df=camera_images_df,
            lidar_range_images_df=lidar_data_df,
            lidar_calibrations=self.sensor_rig.lidars,
            lidar_returns=self.lidar_returns,
            load_point_clouds=self.load_point_clouds,
            camera_labels_df=camera_labels_df,
            lidar_labels_df=lidar_labels_df,
        )
        return frame

    def __len__(self) -> int:
        """Return the number of frames in the segment."""
        return len(self.timestamps)

    def __iter__(self):
        """Iterate over parsed frames in the segment."""
        for idx in range(len(self)):
            yield self[idx]
