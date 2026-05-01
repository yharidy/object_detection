import sys
from pathlib import Path

# Add the parent directory to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.data.waymo.enums import WaymoCamera, WaymoLidar
from src.data.waymo.segment import WaymoSegment
from src.utils.logging import get_logger, setup_logging
from src.viz.rerun_visualizer import RerunVisualizer


@click.command()
@click.option(
    "--data-root",
    type=Path,
    required=True,
    help="Root directory of the Waymo Open Dataset.",
)
@click.option(
    "--segment-name", type=str, required=True, help="Name of the segment to visualize."
)
@click.option(
    "--cameras",
    "-c",
    multiple=True,
    type=click.Choice([e.value for e in WaymoCamera]),
    default=[WaymoCamera.FRONT.value],
    help="Cameras to load and visualize.",
)
@click.option(
    "--lidars",
    "-l",
    multiple=True,
    type=click.Choice([e.value for e in WaymoLidar]),
    default=[WaymoLidar.TOP.value],
    help="LiDARs to load and visualize.",
)
@click.option(
    "--vis-range-images",
    is_flag=True,
    default=False,
    help="Whether to visualize range images for the lidars instead of point clouds.",
)
@click.option(
    "--vis-point-clouds",
    is_flag=True,
    default=True,
    help="Whether to visualize point clouds for the lidars. If True, the point clouds will be converted from the range images using the provided calibrations.",
)
@click.option(
    "--lidar-returns",
    multiple=True,
    type=int,
    help="List of return indices to load for each lidar. If not specified, loads only the first return.",
)
@click.option(
    "--vis-camera-labels",
    is_flag=True,
    help="Whether to visualize camera bounding boxes for each frame.",
)
@click.option(
    "--vis-lidar-labels",
    is_flag=True,
    help="Whether to visualize LiDAR bounding boxes for each frame.",
)
@click.option(
    "--log-level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO)."
)
def main(
    data_root: Path,
    segment_name: str,
    cameras: list[str],
    lidars: list[str],
    vis_range_images: bool,
    vis_point_clouds: bool,
    lidar_returns: list[int],
    vis_camera_labels: bool,
    vis_lidar_labels: bool,
    log_level: str,
):

    setup_logging(level=log_level)
    logger = get_logger("visualize_segment")
    logger.info(f"Visualizing segment '{segment_name}' from data root '{data_root}'")
    logger.info(f"Cameras: {cameras}, LiDARs: {lidars}")
    logger.info(
        f"Visualization options - Range Images: {vis_range_images}, Point Clouds: {vis_point_clouds}, Lidar Returns: {lidar_returns}, Camera Labels: {vis_camera_labels}, LiDAR Labels: {vis_lidar_labels}"
    )

    segment = WaymoSegment(
        root_dir=Path(data_root),
        segment_name=segment_name,
        cameras=[WaymoCamera(camera) for camera in cameras],
        lidars=[WaymoLidar(lidar) for lidar in lidars],
        lidar_returns=lidar_returns if lidar_returns else None,
        load_point_clouds=vis_point_clouds,
        load_camera_labels=vis_camera_labels,
        load_lidar_labels=vis_lidar_labels,
    )

    viz = RerunVisualizer(
        vis_point_clouds=vis_point_clouds,
        vis_range_images=vis_range_images,
        vis_camera_labels=vis_camera_labels,
        vis_lidar_labels=vis_lidar_labels,
    )
    viz.setup(segment.sensor_rig)
    for frame_idx, frame in enumerate(segment):
        viz.log_frame(frame_idx, frame)


if __name__ == "__main__":
    main()
