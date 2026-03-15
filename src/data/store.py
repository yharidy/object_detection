"""Module for loading data from a Waymo Open Dataset v2 segment stored in Parquet format."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from enums import Camera, Lidar


class WaymoDatasetV2Store:
    """Class for loading data from a Waymo Open Dataset v2 segment stored in Parquet
    format.

    The dataset is expected to be organized in folders for different components
    (e.g., camera images, LiDAR data, calibrations, bounding boxes). Each folder
    contains a Parquet file for each segment, named according to the segment name.
    """

    def __init__(self, root_dir: Path, segment_name: str):
        self.root_dir = root_dir
        self.segment_name = (
            segment_name
            if segment_name.endswith(".parquet")
            else f"{segment_name}.parquet"
        )

    def _load_component(
        self,
        component: str,
        columns: list[str] | None = None,
        filters: list | None = None,
        as_pandas: bool = True,
    ) -> pd.DataFrame | pq.Table:
        component_path = self.root_dir / component / self.segment_name
        table = pq.read_table(component_path, columns=columns, filters=filters)
        return table.to_pandas() if as_pandas else table

    def load_camera_images(
        self,
        cameras: list[Camera] | None = None,
        columns: list[str] | None = None,
        filters: list | None = None,
        as_pandas: bool = True,
    ) -> pd.DataFrame | pa.Table:
        """Load camera images for the segment.

        Args:
            camera: Optional camera to filter by. If None, loads images from all cameras.
            columns: Optional list of columns to load. If None, loads all columns.
            filters: Optional list of filters to apply when loading the data. Each filter should be a tuple of (column, operator, value).
            as_pandas: Whether to return the data as a pandas DataFrame. If False, returns a pyarrow Table.
        Returns:
            A pandas DataFrame or pyarrow Table containing the camera images for the segment, filtered by the specified camera and other filters if provided.
        """
        if cameras is not None:
            filters = list(filters) if filters else []
            filters.extend([("key.camera_name", "==", cam.value) for cam in cameras])
        return self._load_component(
            "camera_image", columns=columns, filters=filters, as_pandas=as_pandas
        )

    def load_lidar_data(
        self,
        lidars: list[Lidar] | None = None,
        columns: list[str] | None = None,
        filters: list | None = None,
        as_pandas: bool = True,
    ) -> pd.DataFrame | pa.Table:
        """Load LiDAR data for the segment.

        Args:
            lidars: Optional list of LiDARs to filter by. If None, loads data from all LiDARs.
            columns: Optional list of columns to load. If None, loads all columns.
            filters: Optional list of filters to apply when loading the data. Each filter should be a tuple of (column, operator, value).
            as_pandas: Whether to return the data as a pandas DataFrame. If False, returns a pyarrow Table.
        Returns:
            A pandas DataFrame or pyarrow Table containing the LiDAR data for the segment, filtered by the specified LiDAR and other filters if provided.
        """
        if lidars is not None:
            filters = list(filters) if filters else []
            filters.extend([("key.laser_name", "==", lidar.value) for lidar in lidars])
        return self._load_component(
            "lidar", columns=columns, filters=filters, as_pandas=as_pandas
        )

    def load_camera_calibrations(
        self, cameras: list[Camera] | None = None
    ) -> pd.DataFrame:
        """Load camera calibrations for the segment, optionally filtered by camera."""
        if cameras:
            filters = [("key.camera_name", "==", cam.value) for cam in cameras]
        return self._load_component(
            "camera_calibration", filters=filters, as_pandas=True
        )

    def load_lidar_calibrations(
        self, lidars: list[Lidar] | None = None
    ) -> pd.DataFrame:
        """Load LiDAR calibrations for the segment, optionally filtered by LiDAR."""
        if lidars:
            filters = [("key.laser_name", "==", lidar.value) for lidar in lidars]
        return self._load_component(
            "lidar_calibration", filters=filters, as_pandas=True
        )

    def load_camera_bboxes(self, camera: Camera | None = None) -> pd.DataFrame:
        """Load camera bounding boxes for the segment, optionally filtered by camera."""
        filters = (
            [("key.camera_name", "==", camera.value)] if camera is not None else None
        )
        return self._load_component("camera_box", filters=filters, as_pandas=True)

    def load_lidar_bboxes(self, lidar: Lidar | None = None) -> pd.DataFrame:
        """Load LiDAR bounding boxes for the segment, optionally filtered by LiDAR."""
        filters = [("key.laser_name", "==", lidar.value)] if lidar is not None else None
        return self._load_component("lidar_box", filters=filters, as_pandas=True)
