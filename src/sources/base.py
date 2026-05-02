from typing import Protocol


class FrameLoader(Protocol):
    def load_frame(self, frame_idx: int):
        """Load a single frame by its index."""
        ...

    def get_frame_count(self) -> int:
        """Return the total number of frames available."""
        ...
