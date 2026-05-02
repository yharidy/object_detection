# datasets/

This directory contains PyTorch `Dataset` implementations for loading and preprocessing data for object detection tasks.

## Contents

- `base.py`: Abstract base dataset class that handles frame loading from multiple sources.
- `camera_2d/`: Dataset for 2D camera-based object detection.
  - `dataset.py`: Camera2DDataset class.
  - `transforms.py`: Data augmentation and preprocessing transforms.
  - `collate.py`: Custom collate functions for batching.