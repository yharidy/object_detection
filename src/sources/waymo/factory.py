from pathlib import Path

from src.domain.enums import CameraPosition
from src.sources.waymo.enums import DOMAIN_CAMERA_TO_WAYMO, WaymoCamera
from src.sources.waymo.segment import WaymoSegment


def build_waymo_loaders(
    data_root: Path,
    split: str,
    cameras: list[CameraPosition] | None = None,
    load_camera_labels: bool = True,
    **kwargs
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
    split_dir = data_root / split
    # use one subfolder to get the list of segments, assuming all components have the same segment files
    segment_files = (split_dir / "camera_image").glob("*.parquet")
    segment_names = [f.stem for f in segment_files]
    return {
        segment_name: WaymoSegment(
            segment_name=segment_name,
            root_dir=split_dir,
            cameras=cameras,
            load_camera_labels=load_camera_labels,
            **kwargs
        )
        for segment_name in segment_names
    }
