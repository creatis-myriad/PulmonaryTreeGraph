"""
Utility functions for mathematical and statistical calculations.
"""

from typing import Any

import numpy as np
from skimage.measure import euler_number, label  # pylint: disable=no-name-in-module


def compute_betti_numbers(mask: np.ndarray) -> tuple[int, int, int]:
    """
    Compute the Betti numbers for a 3D binary mask (morphological).

    Returns:
        tuple: (betti_0, betti_1, betti_2) representing connected components, cycles, and cavities
    """
    # Transform the mask to int
    mask = mask.astype(np.int32)
    euler_num = euler_number(mask)
    _, num_cc = label(mask, return_num=True)
    # Betti 0: Number of Connected Components
    betti_0 = num_cc
    # Betti 2: Number of Connected Components in the Inverted Image
    inverted_data = np.ones(mask.shape) - mask
    _, num_cc_inverted = label(inverted_data, return_num=True, connectivity=1)
    betti_2 = (
        num_cc_inverted - 1
    )  # -1 because background is counted as a connected component
    # Betti 1: Number of cycles
    betti_1 = betti_0 + betti_2 - euler_num
    return betti_0, betti_1, betti_2


def calculate_volume_from_voxels(
    voxel_count: int, spacing: tuple[float, float, float]
) -> dict[str, float]:
    """
    Calculate volume from voxel count and spacing.

    Parameters:
    - voxel_count: Number of voxels
    - spacing: Voxel spacing (x, y, z) in mm

    Returns:
    - Dictionary with volume information
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = voxel_count * voxel_volume_mm3

    return {
        "volume_mm3": float(volume_mm3),
        "volume_cm3": float(volume_mm3 / 1000.0),
        "voxel_volume_mm3": float(voxel_volume_mm3),
    }


def calculate_occupancy_metrics(
    occupied_pixels: int, total_pixels: int
) -> dict[str, float]:
    """
    Calculate occupancy metrics.

    Parameters:
    - occupied_pixels: Number of occupied pixels
    - total_pixels: Total number of pixels

    Returns:
    - Dictionary with occupancy metrics
    """
    if total_pixels == 0:
        return {"occupancy_percent": 0.0, "ratio": 0.0}

    ratio = occupied_pixels / total_pixels
    return {"occupancy_percent": float(ratio * 100), "ratio": float(ratio)}


def calculate_bounding_box(coords: np.ndarray) -> dict[str, tuple[int, int]]:
    """
    Calculate bounding box for a set of coordinates.

    Parameters:
    - coords: Array of coordinates

    Returns:
    - Dictionary with min/max coordinates for each axis
    """
    if len(coords) == 0:
        return {"x": (0, 0), "y": (0, 0), "z": (0, 0)}

    return {
        "x": (int(coords[:, 0].min()), int(coords[:, 0].max())),
        "y": (int(coords[:, 1].min()), int(coords[:, 1].max())),
        "z": (int(coords[:, 2].min()), int(coords[:, 2].max())),
    }


def calculate_centroid(coords: np.ndarray) -> np.ndarray:
    """
    Calculate centroid of coordinates.

    Parameters:
    - coords: Array of coordinates

    Returns:
    - Centroid coordinates or None if empty
    """
    return np.mean(coords, axis=0) if len(coords) > 0 else None


def calculate_min_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate minimum distance between two sets of coordinates.

    Parameters:
    - coords1, coords2: Arrays of coordinates

    Returns:
    - Minimum distance between any two points
    """
    min_dist = float("inf")

    for point1 in coords1:
        for point2 in coords2:
            dist = np.linalg.norm(point1 - point2)
            min_dist = min(min_dist, dist)

    return min_dist


def calculate_min_distance_with_spacing(
    coords1: np.ndarray, coords2: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    """
    Calculate minimum distance between two coordinate sets considering voxel spacing.

    Parameters:
    - coords1: First set of coordinates
    - coords2: Second set of coordinates
    - spacing: Voxel spacing (x, y, z) in mm

    Returns:
    - Minimum distance in voxel units (normalized by spacing)
    """
    if len(coords1) == 0 or len(coords2) == 0:
        return float("inf")

    try:
        # Convert to physical coordinates (mm)
        coords1_mm = coords1 * np.array(spacing)
        coords2_mm = coords2 * np.array(spacing)

        # Calculate all pairwise distances in mm using scipy
        from scipy.spatial.distance import cdist

        distances_mm = cdist(coords1_mm, coords2_mm)
        min_distance_mm = np.min(distances_mm)

        # Convert back to normalized voxel units using minimum spacing
        min_spacing = min(spacing)
        min_distance_voxels = min_distance_mm / min_spacing

        return min_distance_voxels

    except (ImportError, ValueError):
        return calculate_min_distance(coords1, coords2)


def calculate_total_volume(
    objects_with_volumes: list[dict[str, Any]], volume_key: str = "volume_mm3"
) -> tuple[float, float]:
    """
    Calculate total volume from a list of objects containing volume data.

    Parameters:
    - objects_with_volumes: List of dictionaries with volume data
    - volume_key: Key for volume data in mm³

    Returns:
    - Tuple of (total_volume_mm3, total_volume_cm3)
    """
    if not objects_with_volumes:
        return 0.0, 0.0

    try:
        valid_volumes = [
            obj[volume_key]
            for obj in objects_with_volumes
            if obj.get(volume_key) is not None and obj[volume_key] > 0
        ]

        total_mm3 = sum(valid_volumes)
        total_cm3 = total_mm3 / 1000.0

        return total_mm3, total_cm3

    except (ValueError, KeyError, TypeError):
        return 0.0, 0.0
