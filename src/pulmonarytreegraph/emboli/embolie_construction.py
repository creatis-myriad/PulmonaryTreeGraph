"""
Embolism Construction Module

This module provides functions for detecting, analyzing, and processing
pulmonary embolisms from medical imaging masks.

Core Functions:
- separate_embolisms(): Detect individual embolism regions
- create_embolism_objects(): Create structured embolism data
"""

import logging
from typing import Any

import numpy as np
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module

from pulmonarytreegraph.utils.metrics import (
    calculate_min_distance_with_spacing,
    calculate_volume_from_voxels,
)

logger = logging.getLogger(__name__)


def separate_embolisms(
    mask_data: np.ndarray,
    spacing: tuple[float, float, float],
    method: str = "connected_components",
    fusion_threshold_mm: float = 10.0,
) -> list[np.ndarray]:
    """
    Separate individual embolism regions from a binary mask.

    Parameters:
    - mask_data: Binary mask array containing embolisms
    - spacing: Voxel spacing (x, y, z) in mm
    - method: Method for separation ("connected_components" or "connected_components_with_threshold")
    - fusion_threshold_mm: Distance threshold for fusing nearby components (in millimeters)

    Returns:
    - List of coordinate arrays for each separate embolism

    Raises:
    - ValueError: If mask_data is empty or invalid method specified
    - RuntimeError: If component separation fails
    """
    logger.info("Separating embolisms using method: %s", method)

    if mask_data is None or mask_data.size == 0:
        logger.error("Empty or invalid mask data provided")
        raise ValueError("Mask data cannot be empty")

    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        logger.error("Invalid spacing provided: %s", spacing)
        raise ValueError("Spacing must be a 3-tuple of positive values")

    if method not in ["connected_components", "connected_components_with_threshold"]:
        logger.warning("Unknown method: %s, using 'connected_components'", method)
        method = "connected_components"

    try:
        # Find connected components
        logger.debug("Finding connected components in mask")
        connected_components = label(mask_data)
        properties = regionprops(connected_components)
        embolisms = [prop.coords for prop in properties]

        logger.debug("Found %d initial connected components", len(embolisms))

        # Apply fusion if requested
        if method == "connected_components_with_threshold":
            # Convert mm threshold to voxels using minimum spacing for conservative approach
            min_spacing = min(spacing)
            fusion_threshold_voxels = fusion_threshold_mm / min_spacing

            logger.debug(
                "Applying fusion with threshold: %.1f mm (%.1f voxels, spacing: %s)",
                fusion_threshold_mm,
                fusion_threshold_voxels,
                spacing,
            )

            embolisms = _fuse_nearby_components(
                embolisms, fusion_threshold_voxels, spacing
            )
            logger.debug("After fusion: %d separate embolisms", len(embolisms))

        logger.info("Successfully separated %d embolisms", len(embolisms))
        return embolisms

    except Exception as e:
        logger.error("Error separating embolisms: %s", str(e))
        raise RuntimeError("Failed to separate embolisms") from e


def _fuse_nearby_components(
    embolisms: list[np.ndarray],
    distance_threshold_voxels: float,
    spacing: tuple[float, float, float],
) -> list[np.ndarray]:
    """
    Fuse connected components that are close to each other.

    Parameters:
    - embolisms: List of coordinate arrays for each embolism
    - distance_threshold_voxels: Maximum distance between components to fuse (in voxels)
    - spacing: Voxel spacing (x, y, z) in mm

    Returns:
    - List of fused embolism coordinate arrays
    """
    logger.debug(
        "Starting component fusion with distance threshold: %.1f voxels",
        distance_threshold_voxels,
    )

    if not embolisms:
        logger.debug("No embolisms to fuse")
        return []

    # Ensure we have a list of arrays
    if isinstance(embolisms, np.ndarray):
        embolisms = list(embolisms)

    initial_count = len(embolisms)
    fusion_iterations = 0
    max_iterations = 100  # Prevent infinite loops

    fused = True
    while fused and fusion_iterations < max_iterations:
        fused = False
        fusion_iterations += 1
        new_embolisms = []

        logger.debug(
            "Fusion iteration %d, processing %d components",
            fusion_iterations,
            len(embolisms),
        )

        while embolisms:
            current_embolism = embolisms.pop(0)
            to_fuse_indices = []

            # Find nearby components to fuse
            for i, other_embolism in enumerate(embolisms):
                try:
                    # Calculate minimum distance considering voxel spacing
                    min_distance_voxels = calculate_min_distance_with_spacing(
                        current_embolism, other_embolism, spacing
                    )

                    if min_distance_voxels <= distance_threshold_voxels:
                        to_fuse_indices.append(i)
                        fused = True
                        logger.debug(
                            "Components will be fused (distance: %.2f <= %.1f voxels)",
                            min_distance_voxels,
                            distance_threshold_voxels,
                        )

                except ValueError as e:
                    logger.warning(
                        "Error calculating distance between components: %s", str(e)
                    )
                    continue

            # Fuse components (remove in reverse order to maintain indices)
            for i in sorted(to_fuse_indices, reverse=True):
                try:
                    fuse_embolism = embolisms.pop(i)
                    current_embolism = np.vstack((current_embolism, fuse_embolism))
                    logger.debug("Fused component with %d voxels", len(fuse_embolism))
                except ValueError as e:
                    logger.warning("Error fusing components: %s", str(e))
                    continue

            new_embolisms.append(current_embolism)

        embolisms = new_embolisms

    if fusion_iterations >= max_iterations:
        logger.warning("Fusion stopped after maximum iterations (%d)", max_iterations)

    final_count = len(embolisms)
    logger.debug(
        "Fusion complete after %d iterations: %d → %d components",
        fusion_iterations,
        initial_count,
        final_count,
    )

    return embolisms


def _create_objects_from_coordinates(
    embolism_coords: list[np.ndarray],
    patient_name: str,
    spacing: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Helper function to create objects from already separated coordinates."""
    if not embolism_coords:
        logger.warning("No embolism coordinates provided")
        return []

    if not patient_name or not patient_name.strip():
        logger.warning("Empty patient name provided")
        patient_name = "unknown"

    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        logger.error("Invalid spacing provided: %s", spacing)
        raise ValueError("Spacing must be a 3-tuple of positive values")

    logger.info(
        "Creating %d embolism objects for patient: %s",
        len(embolism_coords),
        patient_name,
    )

    embolism_objects = []

    for i, coords in enumerate(embolism_coords, start=1):
        try:
            if coords is None or len(coords) == 0:
                logger.warning("Empty coordinates for embolism %d, skipping", i)
                continue

            # Calculate voxel count
            voxel_count = len(coords)

            # Calculate volumes using the metrics utility function
            volume_info = calculate_volume_from_voxels(voxel_count, spacing)

            embolism = {
                "id": i,
                "patient_name": patient_name,
                "coordinates": coords,
                "voxel_count": voxel_count,
                "spacing": spacing,
                "volume_mm3": volume_info["volume_mm3"],
                "volume_cm3": volume_info["volume_cm3"],
            }

            embolism_objects.append(embolism)
            logger.debug(
                "Created embolism %d: %d voxels, %.1f mm³ (%.4f cm³)",
                i,
                voxel_count,
                volume_info["volume_mm3"],
                volume_info["volume_cm3"],
            )

        except ValueError as e:
            logger.error("Error creating embolism object %d: %s", i, str(e))
            continue

    logger.info("Successfully created %d embolism objects", len(embolism_objects))
    return embolism_objects


def create_embolism_objects(
    mask_data: np.ndarray,
    patient_name: str,
    spacing: tuple[float, float, float],
    separation_method: str = "connected_components",
    fusion_threshold_mm: float = 10.0,
) -> list[dict[str, Any]]:
    """
    Create structured embolism objects from mask data.

    This function does the complete pipeline:
    1. Separates embolisms from mask
    2. Creates structured objects with volumes

    Parameters:
    - mask_data: Binary mask array containing embolisms
    - patient_name: Patient identifier
    - spacing: Voxel spacing for volume calculations
    - separation_method: Method for separation ("connected_components" or "connected_components_with_threshold")
    - fusion_threshold_mm: Distance threshold for component fusion

    Returns:
    - List of embolism dictionaries with metadata
    """
    logger.info("Creating embolism objects for patient: %s", patient_name)

    # Step 1: Separate embolisms from mask
    embolism_coords = separate_embolisms(
        mask_data,
        spacing,
        method=separation_method,
        fusion_threshold_mm=fusion_threshold_mm,
    )

    if not embolism_coords:
        logger.warning("No embolism coordinates found")
        return []

    # Step 2: Create objects from coordinates
    return _create_objects_from_coordinates(embolism_coords, patient_name, spacing)
