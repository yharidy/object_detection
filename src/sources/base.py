from typing import Protocol


class FrameLoader(Protocol):
    """Protocol for loading frames from a data source.

    Implementations should provide methods to load individual frames by index
    and report the total number of available frames.
    """

    def load_frame(self, frame_idx: int):
        """Load a single frame by its index."""
        ...

    def get_frame_count(self) -> int:
        """Return the total number of frames available."""
        ...
