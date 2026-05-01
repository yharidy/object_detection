import numpy as np


def range_image_to_point_cloud(
    range_image: np.ndarray,
    beam_inclinations: np.ndarray,
    beam_inclination_min: float,
    beam_inclination_max: float,
) -> np.ndarray:
    """Convert a range image to a point cloud.

    Args:
        range_image: HxWx4 array containing range, intensity, elongation, and no_label_zone.
        beam_inclinations: Array of beam inclination angles in radians, shape (num_beams,).
        beam_inclination_min: Minimum beam inclination angle in radians.
        beam_inclination_max: Maximum beam inclination angle in radians.
    Returns:
        Nx4 array containing x, y, z, and intensity of the points in the point
    """
    height, width = range_image.shape[:2]

    u = np.arange(width)
    azimuth = np.pi - u * (2 * np.pi / width)

    azimuth = azimuth[np.newaxis, :]

    # top lidar: non-uniform per-beam inclinations
    # side lidars: uniform distribution between min and max
    if not np.isnan(beam_inclinations).all():
        inclination = beam_inclinations[::-1, np.newaxis]
    else:
        inclination = np.linspace(
            beam_inclination_max, beam_inclination_min, height, dtype=np.float32
        )  # top-to-bottom = max-to-min
        inclination = inclination[:, np.newaxis]
    # convert to Cartesian coordinates
    range_values = range_image[:, :, 0]
    valid_mask = range_values.reshape(-1) > 0  # range > 0 means valid measurement
    x = range_values * np.cos(inclination) * np.cos(azimuth)
    y = range_values * np.cos(inclination) * np.sin(azimuth)
    z = range_values * np.sin(inclination)
    intensity = range_image[:, :, 1]
    points = np.stack([x, y, z, intensity], axis=-1).reshape(-1, 4)
    return points[valid_mask]
