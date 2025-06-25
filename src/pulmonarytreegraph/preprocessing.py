"""
Preprocessing functions for pulmonary tree graph analysis.
This module provides functions to create a skeleton and a distance map from a binary mask.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def create_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Create a skeleton from a binary mask.

    Parameters:
    - mask: A 3D numpy array representing the binary mask.

    Returns:
    - A 3D numpy array representing the skeleton of the mask.
    """
    skeleton = skeletonize(mask, method="lee")

    return skeleton


def create_distance_map(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """
    Create a distance map from a mask.

    Parameters:
    - mask: A 3D numpy array representing the mask.

    Returns:
    - A 3D numpy array representing the distance map.
    """
    distance_map = distance_transform_edt(mask, sampling=spacing)

    return distance_map


def create_skeleton_and_distance_map(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build skeleton and distance map from a binary mask.

    Parameters:
    - mask: A 3D numpy array representing the binary mask.
    - spacing: A tuple representing the voxel spacing in each dimension.

    Returns:
    - A tuple containing the skeleton and distance map as 3D numpy arrays.
    """
    skeleton = create_skeleton(mask)
    distance_map = create_distance_map(mask, spacing)

    return skeleton, distance_map
