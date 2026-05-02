from abc import ABC, abstractmethod
from typing import Callable

from torch.utils.data import Dataset

from src.domain.frame import Frame
from src.sources.base import FrameLoader


class BaseDataset(Dataset, ABC):
    """Abstract base class for datasets that load frames from multiple sources.

    This class provides a common interface for PyTorch datasets that need to load
    and transform frames from various data sources. It handles indexing across
    multiple segments and applies optional transformations.
    """

    def __init__(self, loaders: dict[str, FrameLoader], transform: Callable = None):
        """Base dataset class for loading and transforming frames from multiple sources.

        Args:
            loaders: A dictionary mapping source names to FrameLoader instances.
            transform: An optional callable that takes a dictionary of loaded frame data
                and returns a transformed version. This can be used for data augmentation
                or preprocessing.
        """
        self.loaders = loaders
        self.transform = transform
        self.index = [
            (segment_name, frame_idx)
            for segment_name, loader in loaders.items()
            for frame_idx in range(loader.get_frame_count())
        ]

    def __len__(self):
        # tells pytorch how many samples are in the dataset, i.e. how many samples in an epoch
        return len(self.index)

    def __getitem__(self, idx):
        # tells pytorch how to get a sample by index, i.e. how to load and transform a frame
        segment_name, frame_idx = self.index[idx]
        loader = self.loaders[segment_name]
        frame: Frame = loader.load_frame(frame_idx)
        sample = self._extract_sample(frame, segment_name)
        if self.transform:
            sample = self.transform(sample)
        return sample

    @abstractmethod
    def _extract_sample(self, frame: Frame, segment_name: str) -> dict:
        """Extract a sample dictionary from a loaded Frame."""
        raise NotImplementedError
