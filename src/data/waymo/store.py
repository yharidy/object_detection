"""Module for loading data from a Waymo Open Dataset v2 segment stored in Parquet format."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.waymo.enums import WaymoCamera, WaymoLidar
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WaymoDatasetV2Store:
    """Class for loading data from a Waymo Open Dataset v2 segment stored in Parquet
    format.

    The dataset is expected to be organized in folders for different components
    (e.g., camera images, LiDAR data, calibrations, bounding boxes). Each folder
    contains a Parquet file for each segment, named according to the segment name.
    """

    def __init__(self, root_dir: Path, segment_name: str):
        """Create a new WaymoDatasetV2Store for a given segment.

        Args:
            root_dir: Root directory containing the Parquet dataset components.
            segment_name: Name of the segment file or segment base name.
        """
        logger.info(
            f"Initializing WaymoDatasetV2Store for segment '{segment_name}' at '{root_dir}'"
        )
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
    ) -> pd.DataFrame | pa.Table:
        """Load a component table from disk and return it as pandas or pyarrow.

        Args:
            component: Component folder name (e.g. camera_image, lidar, camera_calibration).
            columns: Optional column subset to read.
            filters: Optional pyarrow-compatible filters to apply.
            as_pandas: Whether to return a pandas DataFrame.

        Returns:
            A pandas DataFrame or pyarrow Table for the requested component.
        """
        component_path = self.root_dir / component / self.segment_name
        table = pq.read_table(component_path, columns=columns, filters=filters)
        return table.to_pandas() if as_pandas else table

    def load_camera_images(
        self,
        cameras: list[WaymoCamera] | None = None,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
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
        logger.debug(
            f"Loading camera images for segment '{self.segment_name}' with filters: cameras={cameras}, columns={columns}, additional_filters={filters}"
        )
        if cameras is not None:
            extra_tuples = filters or []
            camera_filters = [
                [("key.camera_name", "==", cam.value)] + extra_tuples for cam in cameras
            ]
        else:
            camera_filters = [[f] for f in filters] if filters else []
        return self._load_component(
            "camera_image",
            columns=columns,
            filters=camera_filters or None,
            as_pandas=as_pandas,
        )

    def load_lidar_data(
        self,
        lidars: list[WaymoLidar] | None = None,
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
        logger.debug(
            f"Loading LiDAR data for segment '{self.segment_name}' with filters: lidars={lidars}, columns={columns}, additional_filters={filters}"
        )
        if lidars is not None:
            extra_tuples = filters or []
            lidar_filters = [
                [("key.laser_name", "==", lidar.value)] + extra_tuples
                for lidar in lidars
            ]
        else:
            lidar_filters = [[f] for f in filters] if filters else []
        return self._load_component(
            "lidar", columns=columns, filters=lidar_filters or None, as_pandas=as_pandas
        )

    def load_camera_calibrations(
        self, cameras: list[WaymoCamera] | None = None
    ) -> pd.DataFrame:
        """Load camera calibrations for the segment, optionally filtered by camera."""
        logger.debug(
            f"Loading camera calibrations for segment '{self.segment_name}' with filters: cameras={cameras}"
        )
        filters = (
            [[("key.camera_name", "==", cam.value)] for cam in cameras]
            if cameras
            else None
        )
        return self._load_component(
            "camera_calibration", filters=filters, as_pandas=True
        )

    def load_lidar_calibrations(
        self, lidars: list[WaymoLidar] | None = None
    ) -> pd.DataFrame:
        """Load LiDAR calibrations for the segment, optionally filtered by LiDAR."""
        logger.debug(
            f"Loading LiDAR calibrations for segment '{self.segment_name}' with filters: lidars={lidars}"
        )
        filters = (
            [[("key.laser_name", "==", lidar.value)] for lidar in lidars]
            if lidars
            else None
        )
        return self._load_component(
            "lidar_calibration", filters=filters, as_pandas=True
        )

    def load_camera_bboxes(
        self, cameras: list[WaymoCamera] | None = None, filters: list | None = None
    ) -> pd.DataFrame:
        """Load camera bounding boxes for the segment, optionally filtered by camera."""
        logger.debug(
            f"Loading camera bounding boxes for segment '{self.segment_name}' with filters: camera={cameras}, additional_filters={filters}"
        )
        if cameras is not None:
            extra_tuples = filters or []
            camera_filters = [
                [("key.camera_name", "==", cam.value)] + extra_tuples for cam in cameras
            ]
        else:
            camera_filters = [[f] for f in filters] if filters else []

        return self._load_component(
            "camera_box", filters=camera_filters, as_pandas=True
        )

    def load_lidar_bboxes(self, filters: list | None = None) -> pd.DataFrame:
        """Load LiDAR bounding boxes for the segment, optionally filtered by LiDAR."""
        logger.debug(
            f"Loading LiDAR bounding boxes for segment '{self.segment_name}' with additional_filters={filters}"
        )

        lidar_filters = [[f] for f in filters] if filters else []

        return self._load_component("lidar_box", filters=lidar_filters, as_pandas=True)
