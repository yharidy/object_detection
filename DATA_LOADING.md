# Data Loading Architecture

This document describes the data-loading architecture for the Waymo dataset pipeline in `src/data`.
It explains how a segment is represented, how Parquet-backed dataset components are loaded, how frames are parsed, and which underlying model classes are used.

## High-level flow

1. `WaymoSegment` is the user-facing segment wrapper.
2. `WaymoDatasetV2Store` provides Parquet-backed access to dataset components.
3. `WaymoFrameParser` converts loaded DataFrames into structured model objects.
4. Parsed objects are exposed through the `Frame` data model.

## Component responsibilities

### WaymoDatasetV2Store (`src/data/store.py`)

`WaymoDatasetV2Store` is responsible for reading Parquet files from disk. It assumes the dataset is stored like:

- `camera_image/<segment>.parquet`
- `lidar/<segment>.parquet`
- `camera_calibration/<segment>.parquet`
- `lidar_calibration/<segment>.parquet`
- `camera_box/<segment>.parquet`
- `lidar_box/<segment>.parquet`

The store exposes methods such as:

- `load_camera_images(...)`
- `load_lidar_data(...)`
- `load_camera_calibrations(...)`
- `load_lidar_calibrations(...)`
- `load_camera_bboxes(...)`
- `load_lidar_bboxes(...)`

Each loader supports column selection, filters, and optional pandas/pyarrow return types.

### WaymoSegment (`src/data/segment.py`)

`WaymoSegment` is the segment-level API for loading and iterating frames.
It:

- initializes a `WaymoDatasetV2Store`
- constructs a `WaymoFrameParser`
- loads camera and LiDAR calibrations
- parses a `SensorRig` from calibration data
- collects frame timestamps for indexing
- loads frames on demand via `__getitem__`

Optional behavior includes:

- loading point clouds from LiDAR range images
- loading camera labels
- loading LiDAR labels

### WaymoFrameParser (`src/data/frame_parser.py`)

`WaymoFrameParser` is responsible for converting raw DataFrame rows into structured objects.
This includes:

- parsing `SensorRig` calibration data for cameras and LiDARs
- converting camera image records into `CameraImage` objects
- converting LiDAR range image rows into `LidarRangeImage` objects
- converting range images into `LidarPointCloud` objects when requested
- parsing camera and LiDAR labels into `CameraLabel` and `LidarLabel`

The parser encapsulates the logic that maps raw parquet schema fields into domain models.

## Underlying data models (`src/data/models.py`)

The data-loading pipeline is built around the following model classes:

- `SensorRig`: Holds camera and LiDAR calibration data for a segment.
- `Frame`: Holds parsed observations for a single timestamp.
- `CameraImage`: Stores camera name, timestamp, and pixel data.
- `LidarRangeImage`: Stores a LiDAR range image and return count.
- `LidarPointCloud`: Stores a converted point cloud from a range image.
- `CameraCalibration`: Stores camera intrinsics and extrinsics.
- `LidarCalibration`: Stores LiDAR extrinsics and beam inclination metadata.
- `CameraLabel` and `LidarLabel`: Represent ground-truth annotations.

## Range image conversion

Point clouds are generated from range images using `src/data/transforms.py`.
The key conversion step is:

1. extract range and intensity channels from `LidarRangeImage`
2. compute azimuth angles for each column
3. compute elevation/inclination angles from LiDAR calibration data
4. project each range measurement into Cartesian `x, y, z`
5. output an `Nx4` point cloud containing `[x, y, z, intensity]`

## Typical usage

```python
from pathlib import Path
from src.data.segment import WaymoSegment
from src.data.enums import Camera, Lidar

segment = WaymoSegment(
    segment_name="10017090168044687777_6380_000_6400_000",
    root_dir=Path("data/waymo/raw/training"),
    cameras=[Camera.FRONT],
    lidars=[Lidar.TOP],
    load_point_clouds=True,
    load_camera_labels=True,
)

frame = segment[0]
print(frame.timestamp_micros)
print(frame.camera_images)
print(frame.lidar_point_clouds)
```

## Notes

- `WaymoSegment` supports both camera-only, lidar-only, and mixed sensor setups.
- Calibration data is parsed once during segment initialization and reused for frame parsing.
- Label loading is optional to avoid unnecessary overhead when only raw observations are needed.
