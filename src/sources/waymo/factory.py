from pathlib import Path

import gcsfs

from src.domain.enums import CameraPosition
from src.sources.waymo.enums import DOMAIN_CAMERA_TO_WAYMO, WaymoCamera
from src.sources.waymo.segment import WaymoSegment
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_waymo_loaders(
    data_root: str,
    split: str,
    cameras: list[CameraPosition] | None = None,
    load_camera_labels: bool = True,
    num_segments: int = -1,
    **kwargs,
) -> dict[str, WaymoSegment]:
    """Factory function to build WaymoSegment loaders for a given split.

    Args:
        data_root: Root directory of the Waymo Open Dataset.
        split: Dataset split to load (e.g., 'train', 'val', 'test').
        cameras: List of Camera enums to load for each segment. Defaults to [WaymoCamera.FRONT].
        load_camera_labels: Whether to load camera bounding boxes for each frame. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the WaymoSegment constructor.
            This can include options like lidars, load_point_clouds, lidar_returns, load_lidar_labels, etc.
    Returns:
        A dictionary mapping segment names to initialized WaymoSegment loaders.
    """
    if cameras is None:
        cameras = [WaymoCamera.FRONT]

    # Convert from domain CameraPosition to source WaymoCamera enums
    cameras = [DOMAIN_CAMERA_TO_WAYMO[cam] for cam in cameras]

    # use one subfolder to get the list of segments, assuming all components have the same segment files
    if data_root.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        camera_image_dir = f"{data_root}/{split}/camera_image"
        segment_files = fs.glob(f"{camera_image_dir}/*.parquet")
        segment_names = [
            f.split("/")[-1].replace(".parquet", "") for f in segment_files
        ]
    else:
        # local path — use pathlib as before
        camera_image_dir = Path(data_root) / split / "camera_image"
        segment_files = list(camera_image_dir.glob("*.parquet"))
        segment_names = [f.stem for f in segment_files]

    logger.info(f"Scanning: {camera_image_dir}")
    logger.info(f"Found {len(segment_names)} segment files")

    if num_segments > 0:
        segment_names = segment_names[:num_segments]

    return {
        segment_name: WaymoSegment(
            segment_name=segment_name,
            root_dir=f"{data_root}/{split}",
            cameras=cameras,
            load_camera_labels=load_camera_labels,
            **kwargs,
        )
        for segment_name in segment_names
    }
