import sys
from pathlib import Path

# Add the parent directory to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.data.models import Camera, Lidar
from src.data.segment import WaymoSegment
from src.utils.logging import get_logger, setup_logging
from src.viz.rerun_visualizer import WaymoRerunVisualizer


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
    "--log-level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO)."
)
def main(data_root: Path, segment_name: str, log_level: str):

    setup_logging(level=log_level)
    logger = get_logger("visualize_segment")
    logger.info(f"Visualizing segment '{segment_name}' from data root '{data_root}'")

    segment = WaymoSegment(
        root_dir=Path(data_root),
        segment_name=segment_name,
        cameras=[Camera.FRONT, Camera.FRONT_LEFT, Camera.FRONT_RIGHT],
        lidars=[Lidar.TOP, Lidar.SIDE_LEFT, Lidar.SIDE_RIGHT],
        load_point_clouds=True,
        load_camera_labels=True,
        load_lidar_labels=True,
    )

    viz = WaymoRerunVisualizer()
    viz.setup(segment.sensor_rig)
    for frame_idx, frame in enumerate(segment):
        viz.log_frame(frame_idx, frame)


if __name__ == "__main__":
    main()
