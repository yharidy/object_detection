# sources/

This directory contains data source handlers for loading data from different datasets.

## Contents

- `base.py`: Protocol definitions for frame loaders.
- `waymo/`: Waymo Open Dataset loader implementation.
  - `segment.py`: WaymoSegment class for loading segment data.
  - `store.py`: Parquet-backed data store for Waymo.
  - `frame_parser.py`: Parser for converting raw data to domain models.
  - `lidar_transforms.py`: LiDAR-specific transformations.
  - `enums.py`: Waymo-specific enumerations.
  - `factory.py`: Factory functions for creating loaders.