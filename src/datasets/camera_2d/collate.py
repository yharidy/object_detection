import torch


def camera_2d_collate_fn(samples: list[dict]) -> dict:
    """Collate a list of camera 2D samples into a batch.

    Args:
        samples: List of sample dicts from Camera2DDataset.__getitem__,
                 each containing 'image', 'boxes', 'labels', and 'meta'.
    Returns:
        A batched dict where images are stacked into a single tensor,
        and boxes/labels are kept as lists due to variable length.
    """
    return {
        "image": torch.stack([s["image"] for s in samples]),  # Tensor[B, C, H, W]
        "boxes": [s["boxes"] for s in samples],  # list of B Tensor[N, 4]
        "labels": [s["labels"] for s in samples],  # list of B Tensor[N]
        "meta": [s["meta"] for s in samples],  # list of B dicts
    }
