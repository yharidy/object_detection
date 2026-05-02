from typing import Callable

import numpy as np

from src.datasets.base import BaseDataset
from src.domain.boxes import Box2D
from src.domain.enums import CameraPosition
from src.domain.frame import Frame
from src.domain.labels import ObjectClass
from src.sources.base import FrameLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Camera2DDataset(BaseDataset):
    def __init__(
        self,
        loaders: dict[str, FrameLoader],
        cameras: list[CameraPosition],
        transform: Callable[[dict], dict] | None = None,
    ):
        super().__init__(loaders, transform)
        self.cameras = cameras
        # rebuild index with camera dimension
        self.index = [
            (segment_id, frame_idx, camera)
            for segment_id, loader in loaders.items()
            for frame_idx in range(loader.get_frame_count())
            for camera in cameras
        ]

    def __getitem__(self, i: int) -> dict:
        segment_id, frame_idx, camera = self.index[i]
        frame = self.loaders[segment_id].load_frame(frame_idx)
        if camera not in frame.camera_images:
            logger.warning(
                f"Camera {camera} not found in frame at timestamp {frame.timestamp_micros}. Skipping frame."
            )
            return self.__getitem__(
                (i + 1) % len(self)
            )  # skip frames that don't have the desired camera view
        sample = self._extract_sample(frame, segment_id, camera)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _extract_sample(
        self, frame: Frame, segment_name: str, camera: CameraPosition
    ) -> dict:
        if camera not in frame.camera_images:
            raise ValueError(
                f"Camera {camera} not found in frame at timestamp {frame.timestamp_micros}"
            )

        image: np.ndarray = frame.camera_images[camera].image

        # empty labels is valid — frame may have no objects in view
        camera_labels = frame.camera_labels.get(camera, [])
        boxes: list[Box2D] = [label.box_2d for label in camera_labels]
        labels: list[ObjectClass] = [label.object_class for label in camera_labels]

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "meta": {
                "segment_id": segment_name,
                "timestamp": frame.timestamp_micros,
                "camera": camera,
            },
        }
