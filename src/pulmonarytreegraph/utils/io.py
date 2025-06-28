"""Utility functions for loading and saving masks and graph."""

import json
import logging
import os
from typing import Any

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels

from pulmonarytreegraph.utils.metrics import compute_betti_numbers

logger = logging.getLogger(__name__)


def load_niftii(path: str) -> np.ndarray:
    """
    Load a NIfTI mask file.
    Parameter:
    - path: The file path to the NIfTI mask file.
    """
    img = nib.load(path)
    data = img.get_fdata()

    validate_mask(data, mask_name=os.path.basename(path))

    header = img.header
    spacing = header.get_zooms()
    affine_matrix = img.affine

    return data, header, spacing, affine_matrix


def save_mask(
    data: np.ndarray,
    header: nib.nifti1.Nifti1Header,
    affine_matrix: np.ndarray,
    path: str,
) -> None:
    """
    Save a NIfTI mask file.
    Parameters:
    - data: A 3D numpy array representing the mask data.
    - header: A NIfTI header object containing metadata.
    - affine_matrix: A 4x4 numpy array representing the affine transformation matrix.
    - path: The file path where the mask will be saved.
    """
    img = nib.Nifti1Image(data, affine_matrix, header)
    nib.save(img, path)


def validate_mask(mask: np.ndarray, mask_name: str = "mask") -> None:
    """
    Validate a mask array for preprocessing operations.

    Parameters:
    - mask: A numpy array representing the mask
    - mask_name: Name of the mask for error messages

    Raises:
    - ValueError: If mask fails validation checks
    - TypeError: If mask is not a numpy array
    """
    # Check type
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{mask_name} must be a numpy array, got {type(mask)}")

    # Check dimensionality
    if mask.ndim != 3:
        raise ValueError(
            f"{mask_name} must be 3D, got {mask.ndim}D array with shape {mask.shape}"
        )

    # Check if mask is empty
    if mask.size == 0:
        raise ValueError(f"{mask_name} is empty (size = 0)")

    # Check for NaN or infinite values
    if not np.isfinite(mask).all():
        raise ValueError(f"{mask_name} contains NaN or infinite values")

    # Check if mask contains any foreground voxels
    if not np.any(mask > 0):
        raise ValueError(f"{mask_name} contains no foreground voxels (all values <= 0)")


def generate_mask_metadata(
    data: np.ndarray,
    spacing: tuple[float, float, float],
    affine_matrix: np.ndarray,
    path: str,
) -> dict[str, Any]:
    "Generate metadata for a mask file."

    betti_0, betti_1, betti_2 = compute_betti_numbers(data)
    metadata = {
        "mask": {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "spacing": spacing,
            "affine_matrix": affine_matrix.tolist(),
            "path": path,
        },
        "metrics": {
            "betti_0": betti_0,
            "betti_1": betti_1,
            "betti_2": betti_2,
            "volume_voxels": int(np.sum(data)),
            "total_voxels": int(data.size),
        },
    }
    return metadata


def _ndarray_to_list(data):
    """
    Recursively convert numpy arrays to lists in nested data structures.

    Parameters:
    - data: Data structure that may contain numpy arrays

    Returns:
    - Data structure with numpy arrays converted to lists
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: _ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_ndarray_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_ndarray_to_list(item) for item in data)
    else:
        return data


def save_graph(graph, path: str) -> None:
    """
    Save a NetworkX graph (directed or undirected) to JSON format.

    Parameters:
    - graph: NetworkX graph to save
    - path: Output file path (e.g., 'graph.json')
    """
    # Convert graph to node-link data format
    data = nx.readwrite.json_graph.node_link_data(graph, edges="links")

    # Convert all numpy arrays to lists recursively
    data_list = _ndarray_to_list(data)

    # Save to JSON file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)


def _convert_if_numeric_list(value):
    """Convert a list to numpy array if it contains only numbers."""
    if isinstance(value, list) and len(value) > 0:
        # Check if all elements are numbers (int or float)
        if all(isinstance(x, int | float) for x in value):
            return np.array(value)
        # Handle 2D lists (like coordinates)
        elif all(
            isinstance(x, list) and all(isinstance(y, int | float) for y in x)
            for x in value
            if isinstance(x, list)
        ):
            return np.array(value)
    return value


def apply_affine_to_graph(graph: nx.Graph, affine: np.ndarray) -> nx.Graph:
    """
    Apply affine transformation to convert voxel coordinates to real-world coordinates.

    Parameters:
    - graph: NetworkX graph with voxel coordinates
    - affine: 4x4 affine transformation matrix

    Returns:
    - graph_copy: Graph with real-world coordinates
    """
    graph_copy = graph.copy()

    # Convert node positions
    for node in graph_copy.nodes:
        if "pos" in graph_copy.nodes[node]:
            pos = graph_copy.nodes[node]["pos"]
            real_pos = nib.affines.apply_affine(affine, pos)
            graph_copy.nodes[node]["pos"] = real_pos.tolist()

    # Convert edge centerlines
    for edge in graph_copy.edges:
        edge_data = graph_copy.edges[edge]
        if "centerline" in edge_data:
            centerline = edge_data["centerline"]
            real_centerline = [
                nib.affines.apply_affine(affine, point).tolist() for point in centerline
            ]
            edge_data["centerline"] = real_centerline

    return graph_copy


def save_graph_with_real_coordinates(
    graph: nx.Graph, path: str, reference_skeleton_path: str
) -> None:
    """
    Save a NetworkX graph with coordinates converted to real-world coordinates.

    Parameters:
    - graph: NetworkX graph to save (with voxel coordinates)
    - path: Output file path (e.g., 'graph_real_coords.json')
    - reference_skeleton_path: Path to reference skeleton for affine matrix
    """
    # Load reference skeleton to get affine matrix
    _, _, _, affine = load_niftii(reference_skeleton_path)

    # Apply affine transformation to convert coordinates
    graph_real_coords = apply_affine_to_graph(graph, affine)

    # Save the graph with real coordinates
    save_graph(graph_real_coords, path)


def save_graph_dual_format(
    graph: nx.Graph, base_path: str, reference_skeleton_path: str
) -> tuple[str, str]:
    """
    Save graph in both voxel and real-world coordinate formats.

    Parameters:
    - graph: NetworkX graph to save (with voxel coordinates)
    - base_path: Base path without extension (e.g., 'results/0001/cleaned_graph')
    - reference_skeleton_path: Path to reference skeleton for affine matrix

    Returns:
    - Tuple of (voxel_path, real_coords_path)
    """
    voxel_path = f"{base_path}.json"
    real_coords_path = f"{base_path}_real_coords.json"

    # Save voxel coordinates version
    save_graph(graph, voxel_path)

    # Save real coordinates version
    save_graph_with_real_coordinates(graph, real_coords_path, reference_skeleton_path)

    return voxel_path, real_coords_path


def _lists_to_arrays(graph: nx.Graph) -> None:
    """
    Convert lists that should be numpy arrays back to arrays in graph data.

    This function automatically detects lists that are likely numpy arrays
    (lists of numbers) and converts them back.
    """
    # Convert node attributes
    for _, data in graph.nodes(data=True):
        for attr, value in data.items():
            data[attr] = _convert_if_numeric_list(value)

    # Convert edge attributes
    for _, _, data in graph.edges(data=True):
        for attr, value in data.items():
            data[attr] = _convert_if_numeric_list(value)


def load_graph(path: str) -> nx.Graph:
    """
    Load a NetworkX graph from JSON format.

    Parameters:
    - path: Input file path (e.g., 'graph.json')

    Returns:
    - NetworkX graph
    """
    with open(path, encoding="utf-8") as f:
        graph_data = json.load(f)

    graph = nx.node_link_graph(graph_data, edges="links")

    # Convert lists back to numpy arrays automatically
    _lists_to_arrays(graph)

    return graph


def recreate_skeleton_from_graph(graph: nx.Graph, reference_shape: tuple) -> np.ndarray:
    """
    Recreate skeleton array from graph without saving to file.

    Parameters:
    - graph: NetworkX graph containing centerline data
    - reference_shape: Shape of the original skeleton (height, width, depth)

    Returns:
    - skeleton: 3D numpy array representing the skeleton as uint8
    """
    # Create skeleton as uint8 from the start
    skeleton = np.zeros(reference_shape, dtype=np.uint8)

    # Fill skeleton from graph edges
    for _, _, edge_data in graph.edges(data=True):
        if "centerline" in edge_data:
            centerline = edge_data["centerline"]
            # Handle both numpy arrays and lists
            if isinstance(centerline, np.ndarray):
                points = centerline
            else:
                points = np.array(centerline)

            # Mark all points in the centerline
            for point in points:
                # Ensure coordinates are integers and within bounds
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                if (
                    0 <= x < skeleton.shape[0]
                    and 0 <= y < skeleton.shape[1]
                    and 0 <= z < skeleton.shape[2]
                ):
                    skeleton[x, y, z] = 1  # This will be uint8(1)

    # Ensure return type is definitely uint8
    return skeleton.astype(np.uint8)


def save_skeleton_from_graph(
    graph: nx.Graph, original_skeleton_path: str, output_path: str
) -> None:
    """
    Recreate and save a skeleton NIfTI file from a NetworkX graph.

    Parameters:
    - graph: NetworkX graph containing centerline data
    - original_skeleton_path: Path to original skeleton file (for header/affine info)
    - output_path: Path where the new skeleton will be saved
    """
    # Load original skeleton to get metadata
    original_data, header, _, affine = load_niftii(original_skeleton_path)

    # Use recreate_skeleton_from_graph to generate the skeleton
    skeleton = recreate_skeleton_from_graph(graph, original_data.shape)

    # Ensure skeleton is binary integer type
    skeleton = skeleton.astype(np.uint8)

    # Update header to reflect correct data type
    header.set_data_dtype(np.uint8)

    # Save the recreated skeleton
    save_mask(skeleton, header, affine, output_path)


def compare_skeletons(original: np.ndarray, recreated: np.ndarray) -> dict[str, Any]:
    """
    Compare original and recreated skeletons to validate graph construction.

    Parameters:
    - original: Original skeleton array
    - recreated: Recreated skeleton from graph

    Returns:
    - Dictionary with comparison metrics
    """
    # Ensure both are binary
    original_binary = (original > 0).astype(np.uint8)
    recreated_binary = (recreated > 0).astype(np.uint8)

    # Basic statistics
    original_voxels = np.sum(original_binary)
    recreated_voxels = np.sum(recreated_binary)

    # Intersection and union
    intersection = np.sum(original_binary & recreated_binary)

    # Calculate metrics
    dice_coefficient = (
        (2 * intersection) / (original_voxels + recreated_voxels)
        if (original_voxels + recreated_voxels) > 0
        else 0.0
    )

    # Missing and extra voxels
    missing_voxels = np.sum(original_binary & ~recreated_binary)
    extra_voxels = np.sum(~original_binary & recreated_binary)

    # Coverage percentages
    coverage_original = intersection / original_voxels if original_voxels > 0 else 0.0
    coverage_recreated = (
        intersection / recreated_voxels if recreated_voxels > 0 else 0.0
    )

    return {
        "original_voxels": int(original_voxels),
        "recreated_voxels": int(recreated_voxels),
        "intersection_voxels": int(intersection),
        "missing_voxels": int(missing_voxels),
        "extra_voxels": int(extra_voxels),
        "dice_coefficient": float(dice_coefficient),
        "coverage_original": float(coverage_original),
        "coverage_recreated": float(coverage_recreated),
        "perfect_match": missing_voxels == 0 and extra_voxels == 0,
    }


def validate_graph_skeleton(
    graph: nx.Graph, original_skeleton: np.ndarray
) -> dict[str, Any]:
    """
    Validate that the graph preserves skeleton information by recreating and comparing.

    Parameters:
    - graph: NetworkX graph to validate
    - original_skeleton: Original skeleton array

    Returns:
    - Validation metrics dictionary
    """
    # Recreate skeleton from graph
    recreated_skeleton = recreate_skeleton_from_graph(graph, original_skeleton.shape)

    # Compare skeletons
    comparison = compare_skeletons(original_skeleton, recreated_skeleton)

    return {"validation": comparison, "recreated_skeleton": recreated_skeleton}


def recreate_colored_skeleton_from_graph(
    graph: nx.Graph, reference_shape: tuple, color_by: str = "edge_id"
) -> np.ndarray:
    """
    Recreate skeleton array from graph with different colors/labels for different properties.

    Parameters:
    - graph: NetworkX graph containing centerline data
    - reference_shape: Shape of the original skeleton (height, width, depth)
    - color_by: What to color by ('edge_id', 'level', 'lobe')

    Returns:
    - skeleton: 3D numpy array with different integer values for different properties
    """
    skeleton = np.zeros(reference_shape, dtype=np.uint16)

    for _, _, edge_data in graph.edges(data=True):
        if "centerline" in edge_data:
            centerline = edge_data["centerline"]
            if isinstance(centerline, np.ndarray):
                points = centerline
            else:
                points = np.array(centerline)

            # Determine the color/label value based on the specified property
            if color_by == "edge_id":
                # Use the actual edge ID from the graph (already starts from 1)
                color_value = edge_data.get("id", 1)  # Default to 1 if no ID
            elif color_by == "level":
                color_value = edge_data.get("level", 1)  # Default to 1
            elif color_by == "lobe":
                lobe = edge_data.get("lobe", "unknown")
                color_value = _lobe_to_value(lobe)
            else:
                color_value = 1  # Default binary

            # Mark all points in the centerline with the determined color
            for point in points:
                x, y, z = int(point[0]), int(point[1]), int(point[2])
                if (
                    0 <= x < skeleton.shape[0]
                    and 0 <= y < skeleton.shape[1]
                    and 0 <= z < skeleton.shape[2]
                ):
                    skeleton[x, y, z] = color_value

    return skeleton


def _lobe_to_value(lobe: str) -> int:
    """Convert lobe name to integer value for coloring."""
    lobe_map = {
        "RUL": 1,
        "RML": 2,
        "RLL": 3,
        "LUL": 4,
        "LLL": 5,
        "interlobar": 6,
        "unknown": 7,
    }
    return lobe_map.get(lobe, 7)


def save_colored_skeleton_from_graph(
    graph: nx.Graph,
    reference_shape: tuple,
    header,
    affine,
    output_path: str,
    color_by: str = "edge_id",
) -> None:
    """
    Save a colored skeleton from graph with different labels.

    Parameters:
    - graph: NetworkX graph
    - reference_shape: Original skeleton shape
    - header: NIfTI header
    - affine: Affine transformation
    - output_path: Where to save the skeleton
    - color_by: What property to color by ('edge_id', 'level', 'lobe')
    """
    colored_skeleton = recreate_colored_skeleton_from_graph(
        graph, reference_shape, color_by
    )

    # Create header copy with appropriate data type
    skeleton_header = header.copy()
    skeleton_header.set_data_dtype(np.uint16)  # Support more labels

    save_mask(colored_skeleton, skeleton_header, affine, output_path)


def save_embolism_masks(
    embolism_objects: list[dict[str, Any]], original_mask_path: str, output_dir: str
) -> list[str]:
    """
    Save individual embolism masks as NIfTI files.

    Parameters:
    - embolism_objects: List of embolism dictionaries
    - original_mask_path: Path to original mask file (for header/affine)
    - output_dir: Directory to save individual masks

    Returns:
    - List of saved file paths
    """
    # Load original mask for metadata
    original_data, header, _, affine = load_niftii(original_mask_path)

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for embolism in embolism_objects:
        try:
            # Create individual mask
            mask = np.zeros_like(original_data, dtype=np.uint8)
            coords = embolism["coordinates"]

            # Set embolism voxels
            for coord in coords:
                x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                if (
                    0 <= x < mask.shape[0]
                    and 0 <= y < mask.shape[1]
                    and 0 <= z < mask.shape[2]
                ):
                    mask[x, y, z] = 1

            # Save mask
            patient_name = embolism["patient_name"]
            embolism_number = embolism["number"]
            output_path = os.path.join(
                output_dir, f"{patient_name}_embolism_{embolism_number}.nii.gz"
            )

            save_mask(mask, header, affine, output_path)
            saved_paths.append(output_path)

        except Exception as e:
            # Re-raise exception to be handled by calling function
            raise RuntimeError(
                f"Error saving embolism {embolism['number']}: {e}"
            ) from e

    return saved_paths


def save_embolism_metrics(
    embolism_objects: list[dict[str, Any]], output_path: str
) -> None:
    """
    Save embolism metrics to CSV file.

    Parameters:
    - embolism_objects: List of embolism dictionaries
    - output_path: Path to save CSV file
    """
    try:
        # Prepare data for CSV
        csv_data = []

        for embolism in embolism_objects:
            row = {
                "embolism_number": embolism["number"],
                "patient_name": embolism["patient_name"],
                "voxel_count": embolism["voxel_count"],
                "volume_mm3": embolism["volume_mm3"],
                "volume_cm3": embolism["volume_cm3"],
            }

            csv_data.append(row)

        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# No embolism data available\n")

    except Exception as e:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Error occurred: {str(e)}\n")
                f.write("embolism_number,volume_mm3,volume_cm3\n")
                for emb in embolism_objects:
                    f.write(
                        f"{emb['number']},{emb.get('volume_mm3', 0):.2f},{emb.get('volume_cm3', 0):.4f}\n"
                    )
        except OSError:
            pass  # If even fallback fails, let original exception propagate
        raise RuntimeError(f"Error saving embolism metrics: {e}") from e


def create_colored_embolism_masks(
    embolism_objects: list[dict[str, Any]],
    original_mask_path: str,
    vessel_mask_path: str,
    output_dir: str,
    color_by: str = "vessel_id",
) -> list[str]:
    """
    Create colored embolism masks based on vessel or level labels.

    Parameters:
    - embolism_objects: List of embolism dictionaries
    - original_mask_path: Path to original embolism mask
    - vessel_mask_path: Path to labeled vessel mask
    - output_dir: Directory to save colored masks
    - color_by: What to color by ("vessel_id" or "level_id")

    Returns:
    - List of saved colored mask paths
    """
    try:
        # Load masks
        original_data, header, _, affine = load_niftii(original_mask_path)
        vessel_data, _, _, _ = load_niftii(vessel_mask_path)

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for embolism in embolism_objects:
            # Create colored mask
            colored_mask = np.zeros_like(original_data, dtype=np.uint16)
            coords = embolism["coordinates"]

            for coord in coords:
                x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                if (
                    0 <= x < vessel_data.shape[0]
                    and 0 <= y < vessel_data.shape[1]
                    and 0 <= z < vessel_data.shape[2]
                ):
                    if color_by == "vessel_id":
                        color_value = int(vessel_data[x, y, z])
                    else:
                        color_value = 1  # Default binary

                    colored_mask[x, y, z] = color_value

            # Save colored mask
            patient_name = embolism["patient_name"]
            embolism_number = embolism["number"]
            output_path = os.path.join(
                output_dir,
                f"{patient_name}_embolism_{embolism_number}_{color_by}.nii.gz",
            )

            # Update header for larger data type
            colored_header = header.copy()
            colored_header.set_data_dtype(np.uint16)

            save_mask(colored_mask, colored_header, affine, output_path)
            saved_paths.append(output_path)

        return saved_paths

    except Exception as e:
        raise RuntimeError(f"Error creating colored embolism masks: {e}") from e


def _iterative_vessel_expansion_from_skeleton(
    labeled_skeleton: np.ndarray,
    arteries_mask: np.ndarray,
    initial_distance: int = 5,
    subsequent_distance: int = 2,
    max_iterations: int = 20,
    force_complete_coverage: bool = True,
) -> np.ndarray:
    """
    Expand vessel labels from skeleton by iteratively growing labels into artery volume.
    Guarantees 100% coverage of the artery mask when force_complete_coverage=True.

    Parameters:
    - labeled_skeleton: Labeled skeleton array
    - arteries_mask: Binary mask of arteries volume
    - initial_distance: Distance for initial expansion
    - subsequent_distance: Distance for subsequent expansions
    - max_iterations: Maximum number of expansion iterations
    - force_complete_coverage: If True, continues expansion until 100% coverage

    Returns:
    - Expanded vessel labels as numpy array
    """

    try:
        # Validate inputs using your validation function
        validate_mask(labeled_skeleton, "labeled_skeleton")
        validate_mask(arteries_mask, "arteries_mask")

        # Additional shape compatibility check
        if labeled_skeleton.shape != arteries_mask.shape:
            raise ValueError(
                f"Shape mismatch: skeleton {labeled_skeleton.shape} vs arteries {arteries_mask.shape}"
            )

        logger.debug("Expanding labels with initial distance=%d", initial_distance)
        expanded = expand_labels(labeled_skeleton, distance=initial_distance)
        expanded[arteries_mask == 0] = 0

        # Calculate initial coverage
        valid_pixels = np.sum(arteries_mask > 0)
        labeled_pixels = np.sum(expanded > 0)
        coverage = (labeled_pixels / valid_pixels) * 100
        logger.debug("Initial expansion covers %.2f%% of artery mask", coverage)

        iteration = 0
        previous_coverage = coverage

        while iteration < max_iterations and coverage < 100.0:
            distance = subsequent_distance
            logger.debug(
                "Iteration %d: Expanding with distance=%d", iteration + 1, distance
            )

            expanded = expand_labels(expanded, distance=distance)
            expanded[arteries_mask == 0] = 0

            labeled_pixels = np.sum(expanded > 0)
            coverage = (labeled_pixels / valid_pixels) * 100
            coverage_improvement = coverage - previous_coverage

            logger.debug(
                "Iteration %d: Now covers %.2f%% of artery mask (+%.2f%%)",
                iteration + 1,
                coverage,
                coverage_improvement,
            )

            if coverage == 100.0:
                logger.debug("Achieved 100%% coverage in %d iterations", iteration + 1)
                break

            # Break if we're not making progress (avoid infinite loop)
            if coverage_improvement <= 0:
                if force_complete_coverage:
                    logger.warning(
                        "Coverage not improving after %d iterations. Increasing distance to ensure complete coverage.",
                        iteration + 1,
                    )
                    # Increase distance aggressively to ensure coverage
                    distance = subsequent_distance * 2
                    expanded = expand_labels(expanded, distance=distance)
                    expanded[arteries_mask == 0] = 0

                    labeled_pixels = np.sum(expanded > 0)
                    coverage = (labeled_pixels / valid_pixels) * 100
                    logger.debug(
                        "After aggressive expansion: %.2f%% coverage", coverage
                    )
                else:
                    logger.warning(
                        "Coverage not improving after %d iterations. Final coverage: %.2f%%",
                        iteration + 1,
                        coverage,
                    )
                    break

            previous_coverage = coverage
            iteration += 1

        # Final check for complete coverage requirement
        if force_complete_coverage and coverage < 100.0:
            logger.warning("Forcing complete coverage from %.2f%%", coverage)
            # Use distance transform approach for remaining uncovered areas
            remaining_mask = (arteries_mask > 0) & (expanded == 0)
            if np.any(remaining_mask):
                # Find nearest labeled voxel for each remaining voxel
                labeled_coords = np.where(expanded > 0)
                remaining_coords = np.where(remaining_mask)

                # Simple nearest neighbor assignment for remaining voxels
                for i in range(len(remaining_coords[0])):
                    z, y, x = (
                        remaining_coords[0][i],
                        remaining_coords[1][i],
                        remaining_coords[2][i],
                    )
                    # Find closest labeled voxel
                    distances = np.sqrt(
                        (labeled_coords[0] - z) ** 2
                        + (labeled_coords[1] - y) ** 2
                        + (labeled_coords[2] - x) ** 2
                    )
                    closest_idx = np.argmin(distances)
                    closest_label = expanded[
                        labeled_coords[0][closest_idx],
                        labeled_coords[1][closest_idx],
                        labeled_coords[2][closest_idx],
                    ]
                    expanded[z, y, x] = closest_label

                final_coverage = (np.sum(expanded > 0) / valid_pixels) * 100
                logger.info("Forced complete coverage achieved: %.2f%%", final_coverage)

        # Guarantee 100% coverage verification
        final_labeled_pixels = np.sum(expanded > 0)
        final_coverage = (final_labeled_pixels / valid_pixels) * 100

        if force_complete_coverage and final_coverage < 100.0:
            raise RuntimeError(
                f"Failed to achieve 100% coverage. Final coverage: {final_coverage:.2f}%"
            )

        logger.info("Expansion completed with %.2f%% coverage", final_coverage)
        return expanded

    except (ValueError, RuntimeError, OSError) as e:
        logger.error("Error expanding vessel labels: %s", e)
        raise


def save_colored_vessel_volume_from_graph(
    graph: nx.Graph,
    reference_skeleton_path: str,
    arteries_mask_path: str,
    output_path: str,
    color_by: str = "edge_id",
    colored_skeleton_path: str | None = None,
) -> None:
    """
    Save colored vessel volume from graph or from pre-existing colored skeleton.

    Parameters:
    - graph: NetworkX graph with edge IDs and centerlines
    - reference_skeleton_path: Path to original skeleton file
    - arteries_mask_path: Path to arteries mask file
    - output_path: Path to save the colored vessel volume
    - color_by: What to color by ('edge_id', 'level', 'lobe')
    - colored_skeleton_path: Path to pre-existing colored skeleton (optional)
    """

    try:
        # Validate input paths
        if not os.path.exists(reference_skeleton_path):
            raise FileNotFoundError(
                f"Reference skeleton file not found: {reference_skeleton_path}"
            )
        if not os.path.exists(arteries_mask_path):
            raise FileNotFoundError(
                f"Arteries mask file not found: {arteries_mask_path}"
            )

        # Validate color_by parameter
        valid_color_options = ["edge_id", "level", "lobe"]
        if color_by not in valid_color_options:
            raise ValueError(
                f"Invalid color_by option '{color_by}'. Must be one of: {valid_color_options}"
            )

        # Load input data
        logger.debug("Loading skeleton data from: %s", reference_skeleton_path)
        skeleton_data, skeleton_header, _, skeleton_affine = load_niftii(
            reference_skeleton_path
        )

        logger.debug("Loading arteries mask from: %s", arteries_mask_path)
        arteries_data, _, _, _ = load_niftii(arteries_mask_path)

        # Validate data compatibility
        if skeleton_data.shape != arteries_data.shape:
            raise ValueError(
                f"Shape mismatch: skeleton {skeleton_data.shape} vs arteries {arteries_data.shape}"
            )

        # Use existing colored skeleton or create from graph
        if colored_skeleton_path and os.path.exists(colored_skeleton_path):
            logger.debug(
                "Using pre-existing colored skeleton: %s", colored_skeleton_path
            )
            colored_skeleton_data, _, _, _ = load_niftii(colored_skeleton_path)

            # Validate colored skeleton compatibility
            if colored_skeleton_data.shape != skeleton_data.shape:
                raise ValueError(
                    f"Shape mismatch: colored skeleton {colored_skeleton_data.shape} vs reference {skeleton_data.shape}"
                )
        else:
            logger.debug(
                "Creating colored skeleton from graph (color_by='%s')", color_by
            )
            colored_skeleton_data = recreate_colored_skeleton_from_graph(
                graph, skeleton_data.shape, color_by=color_by
            )

            # Validate that skeleton was created successfully
            if np.sum(colored_skeleton_data > 0) == 0:
                logger.warning("No colored skeleton points were created from graph")

        # Perform expansion with guaranteed 100% coverage
        logger.debug("Starting vessel expansion with guaranteed complete coverage")
        expanded_volume = _iterative_vessel_expansion_from_skeleton(
            colored_skeleton_data, arteries_data, force_complete_coverage=True
        )

        # Simple validation - expansion should always be complete now
        skeleton_points = np.sum(colored_skeleton_data > 0)
        expanded_points = np.sum(expanded_volume > 0)
        artery_points = np.sum(arteries_data > 0)

        logger.info(
            "Expansion complete: %s skeleton \u2192 %s expanded voxels",
            f"{skeleton_points:,}",
            f"{expanded_points:,}",
        )

        # Validate that expansion was successful
        if expanded_points == 0:
            raise RuntimeError("Expansion failed - no voxels in expanded volume")

        if expanded_points != artery_points:
            raise RuntimeError(
                f"Expansion incomplete: {expanded_points} != {artery_points} artery voxels"
            )

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.debug("Created output directory: %s", output_dir)

        # Update header for proper data type
        output_header = skeleton_header.copy()

        # Choose appropriate data type based on max label value
        max_label = np.max(expanded_volume)
        if max_label <= 255:
            output_header.set_data_dtype(np.uint8)
        elif max_label <= 65535:
            output_header.set_data_dtype(np.uint16)
        else:
            output_header.set_data_dtype(np.uint32)

        logger.debug("Set output data type to support max label: %s", max_label)

        # Save result
        save_mask(expanded_volume, output_header, skeleton_affine, output_path)
        logger.info("Successfully saved colored vessel volume to: %s", output_path)

    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        logger.error("Failed to create colored vessel volume: %s", e)
        raise


def validate_vessel_processing_inputs(
    labeled_skeleton: np.ndarray,
    arteries_mask: np.ndarray,
    graph: nx.Graph | None = None,
) -> bool:
    """
    Validate inputs for vessel processing functions.

    Parameters:
    - labeled_skeleton: Labeled skeleton array
    - arteries_mask: Binary mask of arteries volume
    - graph: Optional NetworkX graph

    Returns:
    - True if all inputs are valid, False otherwise
    """

    try:
        # Check array shapes
        if labeled_skeleton.shape != arteries_mask.shape:
            logger.error(
                "Shape mismatch: skeleton %s vs arteries %s",
                labeled_skeleton.shape,
                arteries_mask.shape,
            )
            return False

        # Check for empty arrays
        if labeled_skeleton.size == 0 or arteries_mask.size == 0:
            logger.error("Empty input arrays")
            return False

        # Check data types
        if not np.issubdtype(labeled_skeleton.dtype, np.integer):
            logger.warning(
                "Skeleton should be integer type, got %s", labeled_skeleton.dtype
            )

        if not np.issubdtype(arteries_mask.dtype, np.integer):
            logger.warning(
                "Arteries mask should be integer type, got %s", arteries_mask.dtype
            )

        # Check for valid skeleton labels
        skeleton_labels = np.unique(labeled_skeleton[labeled_skeleton > 0])
        if len(skeleton_labels) == 0:
            logger.warning("No labeled skeleton points found")
        else:
            logger.debug("Found %d unique skeleton labels", len(skeleton_labels))

        # Check for valid arteries mask
        if np.sum(arteries_mask > 0) == 0:
            logger.warning("No artery mask points found")

        # Validate graph if provided
        if graph is not None:
            if not isinstance(graph, nx.Graph):
                logger.error("Graph must be a NetworkX Graph object")
                return False

            if len(graph.nodes) == 0:
                logger.warning("Graph has no nodes")

            if len(graph.edges) == 0:
                logger.warning("Graph has no edges")

        return True

    except (ValueError, TypeError, OSError) as e:
        logger.error("Error validating inputs: %s", e)
        return False
