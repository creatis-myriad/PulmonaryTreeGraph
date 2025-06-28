"""Core pipeline functions extracted from scripts."""

import logging
import os
from typing import Any

from ..emboli.embolie_analysis import analyze_embolisms_with_visualization
from ..graph.graph_cleaning import clean_graph
from ..graph.graph_construction import mask_to_branch_graph
from ..graph.graph_labelling import label_graph
from ..graph.graph_orienting import orient_graph
from ..graph_embolism_integration import (
    analyze_embolism_by_levels,
    analyze_embolism_vessel_interactions,
    compute_embolism_vessel_metrics,
    enrich_vessel_graph_with_embolisms,
    get_embolism_vessel_statistics,
    get_vessels_with_embolisms,
    save_analysis_report,
    save_level_analysis_results,
    save_vessel_embolism_analysis,
)
from ..preprocessing import create_skeleton_and_distance_map
from ..utils.io import (
    load_graph,
    load_niftii,
    recreate_skeleton_from_graph,
    save_colored_skeleton_from_graph,
    save_colored_vessel_volume_from_graph,
    save_graph,
    save_graph_dual_format,
    save_graph_with_real_coordinates,
    save_mask,
    validate_graph_skeleton,
)

logger = logging.getLogger(__name__)


def preprocess_artery_mask(
    artery_path: str, output_dir: str | None = None
) -> tuple[str, str]:
    """
    Preprocess an artery mask: create skeleton and distance map.

    Parameters:
    - artery_path: Path to artery mask file
    - output_dir: Output directory (defaults to results/patient_name)

    Returns:
    - Tuple of (skeleton_path, distance_map_path)
    """
    logger.info("Loading mask: %s", artery_path)
    mask, header, spacing, affine = load_niftii(artery_path)

    # Extract patient name and create output directory
    if output_dir is None:
        patient_name = os.path.basename(artery_path).replace("_arteries.nii.gz", "")
        output_dir = os.path.join("results", patient_name)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Creating skeleton and distance map...")
    skeleton, distance_map = create_skeleton_and_distance_map(mask, spacing)
    skeleton_path = os.path.join(output_dir, "skeleton.nii.gz")
    save_mask(skeleton.astype(mask.dtype), header, affine, skeleton_path)
    logger.info("Skeleton saved: %s", skeleton_path)
    distance_map_path = os.path.join(output_dir, "distance_map.nii.gz")
    save_mask(distance_map.astype(mask.dtype), header, affine, distance_map_path)
    logger.info("Distance map saved: %s", distance_map_path)

    return skeleton_path, distance_map_path


def build_graph_from_skeleton(
    skeleton_path: str,
    distance_map_path: str,
    output_dir: str | None = None,
    coordinate_format: str = "both",
    save_skeleton: bool = False,
    detailed_validation: bool = False,
) -> dict[str, Any]:
    """
    Build a graph from skeleton and distance map.

    Parameters:
    - skeleton_path: Path to skeleton file
    - distance_map_path: Path to distance map file
    - output_dir: Output directory (defaults to results/patient_name)
    - coordinate_format: "voxel", "real", or "both"
    - save_skeleton: Whether to save both binary and edge_id colored skeleton from graph
    - detailed_validation: Whether to show detailed validation

    Returns:
    - Dictionary with graph paths and results
    """
    if output_dir is None:
        patient_name = skeleton_path.split("/")[-2]
        output_dir = os.path.join("results", patient_name)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading skeleton: %s", skeleton_path)
    skeleton, header, spacing, affine = load_niftii(skeleton_path)

    logger.info("Loading distance map: %s", distance_map_path)
    distance_map, _, _, _ = load_niftii(distance_map_path)

    logger.info("Building branch graph...")
    branch_graph = mask_to_branch_graph(skeleton, spacing, distance_map)

    logger.info(
        "Graph statistics: %d nodes, %d edges",
        len(branch_graph.nodes),
        len(branch_graph.edges),
    )

    # Save branch graph in requested coordinate format(s)
    graph_paths = {}

    if coordinate_format == "both":
        base_path = os.path.join(output_dir, "branch_graph")
        voxel_path, real_path = save_graph_dual_format(
            branch_graph, base_path, skeleton_path
        )
        graph_paths["voxel"] = voxel_path
        graph_paths["real"] = real_path
        logger.info("Branch graph (voxel coords) saved: %s", voxel_path)
        logger.info("Branch graph (real coords) saved: %s", real_path)

    elif coordinate_format == "real":
        branch_graph_path = os.path.join(output_dir, "branch_graph_real_coords.json")
        save_graph_with_real_coordinates(branch_graph, branch_graph_path, skeleton_path)
        graph_paths["real"] = branch_graph_path
        logger.info("Branch graph (real coords) saved: %s", branch_graph_path)

    else:  # voxel
        branch_graph_path = os.path.join(output_dir, "branch_graph.json")
        save_graph(branch_graph, branch_graph_path)
        graph_paths["voxel"] = branch_graph_path
        logger.info("Branch graph (voxel coords) saved: %s", branch_graph_path)

    # Save skeletons from graph (both binary and edge_id)
    skeleton_paths = {}
    if save_skeleton:
        # Save binary skeleton
        binary_skeleton_path = os.path.join(output_dir, "branch_graph_skeleton.nii.gz")
        recreated_skeleton = recreate_skeleton_from_graph(branch_graph, skeleton.shape)
        header_copy = header.copy()
        header_copy.set_data_dtype(skeleton.dtype)
        save_mask(recreated_skeleton, header_copy, affine, binary_skeleton_path)
        skeleton_paths["binary"] = binary_skeleton_path
        logger.info("Branch graph skeleton (binary) saved: %s", binary_skeleton_path)

        # Save edge_id colored skeleton
        edge_id_skeleton_path = os.path.join(
            output_dir, "branch_graph_skeleton_edge_id.nii.gz"
        )
        save_colored_skeleton_from_graph(
            branch_graph,
            skeleton.shape,
            header,
            affine,
            edge_id_skeleton_path,
            "edge_id",
        )
        skeleton_paths["edge_id"] = edge_id_skeleton_path
        logger.info("Branch graph skeleton (edge_id) saved: %s", edge_id_skeleton_path)

    # Validate graph
    logger.info("Validating graph by recreating skeleton...")
    validation_result = validate_graph_skeleton(branch_graph, skeleton)
    metrics = validation_result["validation"]

    logger.info("Validation summary: Perfect match=%s", metrics["perfect_match"])

    if detailed_validation:
        logger.info("Detailed Validation Results:")
        logger.info("  Original voxels: %d", metrics["original_voxels"])
        logger.info("  Recreated voxels: %d", metrics["recreated_voxels"])
        logger.info("  Dice Coefficient: %.4f", metrics["dice_coefficient"])
        logger.info(
            "  Coverage of original: %.2f%%", metrics["coverage_original"] * 100
        )
        logger.info("  Missing voxels: %d", metrics["missing_voxels"])
        logger.info("  Extra voxels: %d", metrics["extra_voxels"])

    if metrics["missing_voxels"] > 0:
        logger.warning(
            "%d voxels from original skeleton missing in graph representation",
            metrics["missing_voxels"],
        )

    return {
        "graph_paths": graph_paths,
        "skeleton_paths": skeleton_paths,
        "validation": metrics,
        "graph_object": branch_graph,
    }


def clean_vessel_graph(
    graph_path: str,
    output_dir: str | None = None,
    original_skeleton_path: str | None = None,
    arteries_mask_path: str | None = None,
    radius_ratio_threshold: float = 0.3,
    coordinate_format: str = "both",
    save_skeleton: bool = False,
    save_colored_volumes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Clean a graph by removing suspicious edges and short/thin branches.

    Parameters:
    - graph_path: Path to input graph file
    - output_dir: Output directory (defaults to results/patient_name)
    - original_skeleton_path: Path to original skeleton file
    - arteries_mask_path: Path to arteries mask file (required for vessel volumes)
    - distance_map_path: Path to pre-computed distance map (optional)
    - radius_ratio_threshold: Threshold for removing thin branches
    - coordinate_format: "voxel", "real", or "both"
    - save_skeleton: Whether to save both binary and edge_id colored skeleton from graph
    - save_colored_volumes: List of volume types to save ["edge_id", "level", "lobe"]

    Returns:
    - Dictionary with cleaned graph paths and info
    """
    if output_dir is None:
        patient_name = graph_path.split("/")[-2]
        output_dir = os.path.join("results", patient_name)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading graph: %s", graph_path)
    graph = load_graph(graph_path)

    logger.info(
        "Original graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges)
    )

    logger.info("Cleaning graph...")
    cleaned_graph = clean_graph(graph, radius_ratio_threshold=radius_ratio_threshold)

    logger.info(
        "Cleaned graph: %d nodes, %d edges",
        len(cleaned_graph.nodes),
        len(cleaned_graph.edges),
    )

    # Save cleaned graph
    graph_paths = {}

    if coordinate_format == "both":
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when using coordinate_format 'both'"
            )

        base_path = os.path.join(output_dir, "cleaned_graph")
        voxel_path, real_path = save_graph_dual_format(
            cleaned_graph, base_path, original_skeleton_path
        )
        graph_paths["voxel"] = voxel_path
        graph_paths["real"] = real_path
        logger.info("Cleaned graph (voxel coords) saved: %s", voxel_path)
        logger.info("Cleaned graph (real coords) saved: %s", real_path)

    elif coordinate_format == "real":
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when using coordinate_format 'real'"
            )

        cleaned_graph_path = os.path.join(output_dir, "cleaned_graph_real_coords.json")
        save_graph_with_real_coordinates(
            cleaned_graph, cleaned_graph_path, original_skeleton_path
        )
        graph_paths["real"] = cleaned_graph_path
        logger.info("Cleaned graph (real coords) saved: %s", cleaned_graph_path)

    else:  # voxel
        cleaned_graph_path = os.path.join(output_dir, "cleaned_graph.json")
        save_graph(cleaned_graph, cleaned_graph_path)
        graph_paths["voxel"] = cleaned_graph_path
        logger.info("Cleaned graph (voxel coords) saved: %s", cleaned_graph_path)

    # Save skeletons (both binary and edge_id)
    skeleton_paths = {}
    if save_skeleton:
        if not original_skeleton_path:
            raise ValueError("original_skeleton_path required when save_skeleton=True")

        skeleton_data, header, _, affine = load_niftii(original_skeleton_path)

        # Save binary skeleton
        binary_skeleton_path = os.path.join(output_dir, "cleaned_graph_skeleton.nii.gz")
        recreated_skeleton = recreate_skeleton_from_graph(
            cleaned_graph, skeleton_data.shape
        )
        header_copy = header.copy()
        header_copy.set_data_dtype(skeleton_data.dtype)
        save_mask(recreated_skeleton, header_copy, affine, binary_skeleton_path)
        skeleton_paths["binary"] = binary_skeleton_path
        logger.info("Cleaned graph skeleton (binary) saved: %s", binary_skeleton_path)

        # Save edge_id colored skeleton
        edge_id_skeleton_path = os.path.join(
            output_dir, "cleaned_graph_skeleton_edge_id.nii.gz"
        )
        save_colored_skeleton_from_graph(
            cleaned_graph,
            skeleton_data.shape,
            header,
            affine,
            edge_id_skeleton_path,
            "edge_id",
        )
        skeleton_paths["edge_id"] = edge_id_skeleton_path
        logger.info("Cleaned graph skeleton (edge_id) saved: %s", edge_id_skeleton_path)

    # Save vessel volumes (if requested)
    volume_paths = {}
    if save_colored_volumes:
        if not arteries_mask_path:
            raise ValueError(
                "arteries_mask_path required when save_colored_volumes is specified"
            )
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when save_colored_volumes is specified"
            )

        logger.info("Creating vessel volumes...")

        # Validate volume options - only allow edge_id, level, lobe
        valid_options = ["edge_id", "level", "lobe"]
        for color_by in save_colored_volumes:
            if color_by not in valid_options:
                logger.warning(
                    "Invalid volume option '%s'. Valid options: %s",
                    color_by,
                    valid_options,
                )
                continue

            # Determine colored skeleton path if it exists
            colored_skeleton_path = None
            if color_by == "edge_id" and "edge_id" in skeleton_paths:
                colored_skeleton_path = skeleton_paths["edge_id"]
            elif save_skeleton:
                # Check if skeleton would be saved with this color_by
                potential_skeleton_path = os.path.join(
                    output_dir, f"cleaned_graph_skeleton_{color_by}.nii.gz"
                )
                if os.path.exists(potential_skeleton_path):
                    colored_skeleton_path = potential_skeleton_path

            volume_output = os.path.join(
                output_dir,
                f"cleaned_graph_vessel_volume_{color_by}.nii.gz",
            )
            save_colored_vessel_volume_from_graph(
                graph=cleaned_graph,
                reference_skeleton_path=original_skeleton_path,
                arteries_mask_path=arteries_mask_path,
                output_path=volume_output,
                color_by=color_by,
                colored_skeleton_path=colored_skeleton_path,  # Pass the colored skeleton path
            )
            volume_paths[color_by] = volume_output
            logger.info("Vessel volume (%s) saved: %s", color_by, volume_output)

    return {
        "graph_paths": graph_paths,
        "skeleton_paths": skeleton_paths,
        "volume_paths": volume_paths,
        "graph_object": cleaned_graph,
    }


def orient_vessel_graph(
    graph_path: str,
    output_dir: str | None = None,
    original_skeleton_path: str | None = None,
    arteries_mask_path: str | None = None,
    coordinate_format: str = "both",
    save_skeleton: bool = False,
    save_colored_volumes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Orient a graph by starting from the end branch with maximum mean radius.

    Parameters:
    - graph_path: Path to input graph file
    - output_dir: Output directory (defaults to results/patient_name)
    - original_skeleton_path: Path to original skeleton file
    - arteries_mask_path: Path to arteries mask file (required for vessel volumes)
    - distance_map_path: Path to pre-computed distance map (optional)
    - coordinate_format: "voxel", "real", or "both"
    - save_skeleton: Whether to save both binary and edge_id colored skeleton from graph
    - save_colored_volumes: List of volume types to save ["edge_id", "level", "lobe"]

    Returns:
    - Dictionary with oriented graph paths and info
    """
    if output_dir is None:
        patient_name = graph_path.split("/")[-2]
        output_dir = os.path.join("results", patient_name)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading graph: %s", graph_path)
    graph = load_graph(graph_path)

    logger.info(
        "Original graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges)
    )

    logger.info("Orienting graph...")
    oriented_graph = orient_graph(graph)
    logger.info(
        "Oriented graph: %d nodes, %d edges",
        len(oriented_graph.nodes),
        len(oriented_graph.edges),
    )

    # Save oriented graph
    graph_paths = {}

    if coordinate_format == "both":
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when using coordinate_format 'both'"
            )

        base_path = os.path.join(output_dir, "oriented_graph")
        voxel_path, real_path = save_graph_dual_format(
            oriented_graph, base_path, original_skeleton_path
        )
        graph_paths["voxel"] = voxel_path
        graph_paths["real"] = real_path
        logger.info("Oriented graph (voxel coords) saved: %s", voxel_path)
        logger.info("Oriented graph (real coords) saved: %s", real_path)

    elif coordinate_format == "real":
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when using coordinate_format 'real'"
            )

        oriented_graph_path = os.path.join(
            output_dir, "oriented_graph_real_coords.json"
        )
        save_graph_with_real_coordinates(
            oriented_graph, oriented_graph_path, original_skeleton_path
        )
        graph_paths["real"] = oriented_graph_path
        logger.info("Oriented graph (real coords) saved: %s", oriented_graph_path)

    else:  # voxel
        oriented_graph_path = os.path.join(output_dir, "oriented_graph.json")
        save_graph(oriented_graph, oriented_graph_path)
        graph_paths["voxel"] = oriented_graph_path
        logger.info("Oriented graph (voxel coords) saved: %s", oriented_graph_path)

    # Save skeletons (both binary and edge_id)
    skeleton_paths = {}
    if save_skeleton:
        if not original_skeleton_path:
            raise ValueError("original_skeleton_path required when save_skeleton=True")

        skeleton_data, header, _, affine = load_niftii(original_skeleton_path)

        # Save binary skeleton
        binary_skeleton_path = os.path.join(
            output_dir, "oriented_graph_skeleton.nii.gz"
        )
        recreated_skeleton = recreate_skeleton_from_graph(
            oriented_graph, skeleton_data.shape
        )
        header_copy = header.copy()
        header_copy.set_data_dtype(skeleton_data.dtype)
        save_mask(recreated_skeleton, header_copy, affine, binary_skeleton_path)
        skeleton_paths["binary"] = binary_skeleton_path
        logger.info("Oriented graph skeleton (binary) saved: %s", binary_skeleton_path)

        # Save edge_id colored skeleton
        edge_id_skeleton_path = os.path.join(
            output_dir, "oriented_graph_skeleton_edge_id.nii.gz"
        )
        save_colored_skeleton_from_graph(
            oriented_graph,
            skeleton_data.shape,
            header,
            affine,
            edge_id_skeleton_path,
            "edge_id",
        )
        skeleton_paths["edge_id"] = edge_id_skeleton_path
        logger.info(
            "Oriented graph skeleton (edge_id) saved: %s", edge_id_skeleton_path
        )

    # Save vessel volumes (if requested)
    volume_paths = {}
    if save_colored_volumes:
        if not arteries_mask_path:
            raise ValueError(
                "arteries_mask_path required when save_colored_volumes is specified"
            )
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when save_colored_volumes is specified"
            )

        logger.info("Creating vessel volumes...")

        # Validate volume options - only allow edge_id, level, lobe
        valid_options = ["edge_id", "level", "lobe"]
        for color_by in save_colored_volumes:
            if color_by not in valid_options:
                logger.warning(
                    "Invalid volume option '%s'. Valid options: %s",
                    color_by,
                    valid_options,
                )
                continue

            # Determine colored skeleton path if it exists
            colored_skeleton_path = None
            if color_by == "edge_id" and "edge_id" in skeleton_paths:
                colored_skeleton_path = skeleton_paths["edge_id"]
            elif save_skeleton:
                # Check if skeleton would be saved with this color_by
                potential_skeleton_path = os.path.join(
                    output_dir, f"oriented_graph_skeleton_{color_by}.nii.gz"
                )
                if os.path.exists(potential_skeleton_path):
                    colored_skeleton_path = potential_skeleton_path

            volume_output = os.path.join(
                output_dir,
                f"oriented_graph_vessel_volume_{color_by}.nii.gz",
            )
            save_colored_vessel_volume_from_graph(
                graph=oriented_graph,
                reference_skeleton_path=original_skeleton_path,
                arteries_mask_path=arteries_mask_path,
                output_path=volume_output,
                color_by=color_by,
                colored_skeleton_path=colored_skeleton_path,  # Pass the colored skeleton path
            )
            volume_paths[color_by] = volume_output
            logger.info("Vessel volume (%s) saved: %s", color_by, volume_output)

    return {
        "graph_paths": graph_paths,
        "skeleton_paths": skeleton_paths,
        "volume_paths": volume_paths,
        "graph_object": oriented_graph,
    }


def label_vessel_graph(
    graph_path: str,
    lung_masks_dir: str,
    output_dir: str | None = None,
    original_skeleton_path: str | None = None,
    arteries_mask_path: str | None = None,
    coordinate_format: str = "both",
    save_skeleton: bool = False,
    save_colored_volumes: list[str] | None = None,
    diameter_threshold: float = 0.7,
    length_threshold: float = 2.0,
) -> dict[str, Any]:
    """
    Label a graph with anatomical information (lobes, levels, laterality).

    Parameters:
    - graph_path: Path to input graph file
    - lung_masks_dir: Directory containing lung mask files
    - output_dir: Output directory (defaults to results/patient_name)
    - original_skeleton_path: Path to original skeleton file
    - arteries_mask_path: Path to arteries mask file (required for vessel volumes)
    - distance_map_path: Path to pre-computed distance map (optional)
    - coordinate_format: "voxel", "real", or "both"
    - save_skeleton: Whether to save both binary and edge_id colored skeleton from graph
    - save_colored_volumes: List of volume types to save ["edge_id", "level", "lobe"]
    - diameter_threshold: Minimum diameter ratio for trunk vessels (default: 0.7)
    - length_threshold: Maximum length/diameter ratio for trunk vessels (default: 2)

    Returns:
    - Dictionary with labeled graph paths and info
    """
    if output_dir is None:
        patient_name = graph_path.split("/")[-2]
        output_dir = os.path.join("results", patient_name)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading graph: %s", graph_path)
    graph = load_graph(graph_path)

    logger.info(
        "Original graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges)
    )

    # Load lung masks
    logger.info("Loading lung masks from: %s", lung_masks_dir)
    mask_files = {
        "lung_right": "lung_right.nii.gz",
        "lung_left": "lung_left.nii.gz",
        "lung_upper_lobe_right": "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right": "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right": "lung_lower_lobe_right.nii.gz",
        "lung_upper_lobe_left": "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left": "lung_lower_lobe_left.nii.gz",
    }

    lungs_dict = {}
    for mask_name, filename in mask_files.items():
        mask_path = os.path.join(lung_masks_dir, filename)
        if os.path.exists(mask_path):
            mask_data, _, _, _ = load_niftii(mask_path)
            lungs_dict[mask_name] = mask_data
            logger.info("✓ Loaded mask: %s", mask_name)
        else:
            logger.warning("✗ Mask file not found: %s", mask_path)

    if not lungs_dict:
        raise ValueError("No lung masks loaded. Cannot proceed with labeling.")

    logger.info("Successfully loaded %d/%d masks", len(lungs_dict), len(mask_files))

    logger.info("Labeling graph...")
    labeled_graph = label_graph(
        graph,
        lungs_dict,
        diameter_threshold=diameter_threshold,
        length_threshold=length_threshold,
    )

    logger.info(
        "Labeled graph: %d nodes, %d edges",
        len(labeled_graph.nodes),
        len(labeled_graph.edges),
    )

    # Save labeled graph
    graph_paths = {}

    if coordinate_format == "both":
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when using coordinate_format 'both'"
            )

        base_path = os.path.join(output_dir, "labeled_graph")
        voxel_path, real_path = save_graph_dual_format(
            labeled_graph, base_path, original_skeleton_path
        )
        graph_paths["voxel"] = voxel_path
        graph_paths["real"] = real_path
        logger.info("Labeled graph (voxel coords) saved: %s", voxel_path)
        logger.info("Labeled graph (real coords) saved: %s", real_path)

    elif coordinate_format == "real":
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when using coordinate_format 'real'"
            )

        labeled_graph_path = os.path.join(output_dir, "labeled_graph_real_coords.json")
        save_graph_with_real_coordinates(
            labeled_graph, labeled_graph_path, original_skeleton_path
        )
        graph_paths["real"] = labeled_graph_path
        logger.info("Labeled graph (real coords) saved: %s", labeled_graph_path)

    else:  # voxel
        labeled_graph_path = os.path.join(output_dir, "labeled_graph.json")
        save_graph(labeled_graph, labeled_graph_path)
        graph_paths["voxel"] = labeled_graph_path
        logger.info("Labeled graph (voxel coords) saved: %s", labeled_graph_path)

    # Save skeletons (both binary and edge_id)
    skeleton_paths = {}
    if save_skeleton:
        if not original_skeleton_path:
            raise ValueError("original_skeleton_path required when save_skeleton=True")

        skeleton_data, header, _, affine = load_niftii(original_skeleton_path)

        # Save binary skeleton
        binary_skeleton_path = os.path.join(output_dir, "labeled_graph_skeleton.nii.gz")
        recreated_skeleton = recreate_skeleton_from_graph(
            labeled_graph, skeleton_data.shape
        )
        header_copy = header.copy()
        header_copy.set_data_dtype(skeleton_data.dtype)
        save_mask(recreated_skeleton, header_copy, affine, binary_skeleton_path)
        skeleton_paths["binary"] = binary_skeleton_path
        logger.info("Labeled graph skeleton (binary) saved: %s", binary_skeleton_path)

        # Save edge_id colored skeleton
        edge_id_skeleton_path = os.path.join(
            output_dir, "labeled_graph_skeleton_edge_id.nii.gz"
        )
        save_colored_skeleton_from_graph(
            labeled_graph,
            skeleton_data.shape,
            header,
            affine,
            edge_id_skeleton_path,
            "edge_id",
        )
        skeleton_paths["edge_id"] = edge_id_skeleton_path
        logger.info("Labeled graph skeleton (edge_id) saved: %s", edge_id_skeleton_path)

        # Save level colored skeleton
        level_skeleton_path = os.path.join(
            output_dir, "labeled_graph_skeleton_level.nii.gz"
        )
        save_colored_skeleton_from_graph(
            labeled_graph,
            skeleton_data.shape,
            header,
            affine,
            level_skeleton_path,
            "level",
        )
        skeleton_paths["level"] = level_skeleton_path
        logger.info("Labeled graph skeleton (level) saved: %s", level_skeleton_path)

    # Save vessel volumes (if requested)
    volume_paths = {}
    if save_colored_volumes:
        if not arteries_mask_path:
            raise ValueError(
                "arteries_mask_path required when save_colored_volumes is specified"
            )
        if not original_skeleton_path:
            raise ValueError(
                "original_skeleton_path required when save_colored_volumes is specified"
            )

        logger.info("Creating labeled vessel volumes...")

        # Validate volume options - only allow edge_id, level, lobe
        valid_options = ["edge_id", "level", "lobe"]
        for color_by in save_colored_volumes:
            if color_by not in valid_options:
                logger.warning(
                    "Invalid volume option '%s'. Valid options: %s",
                    color_by,
                    valid_options,
                )
                continue

            # Determine colored skeleton path if it exists
            colored_skeleton_path = None
            if color_by == "edge_id" and "edge_id" in skeleton_paths:
                colored_skeleton_path = skeleton_paths["edge_id"]
            elif save_skeleton:
                # Check if skeleton would be saved with this color_by
                potential_skeleton_path = os.path.join(
                    output_dir, f"labeled_graph_skeleton_{color_by}.nii.gz"
                )
                if os.path.exists(potential_skeleton_path):
                    colored_skeleton_path = potential_skeleton_path

            volume_output = os.path.join(
                output_dir,
                f"labeled_graph_vessel_volume_{color_by}.nii.gz",
            )
            save_colored_vessel_volume_from_graph(
                graph=labeled_graph,
                reference_skeleton_path=original_skeleton_path,
                arteries_mask_path=arteries_mask_path,
                output_path=volume_output,
                color_by=color_by,
                colored_skeleton_path=colored_skeleton_path,  # Pass the colored skeleton path
            )
            volume_paths[color_by] = volume_output
            logger.info("Vessel volume (%s) saved: %s", color_by, volume_output)

    return {
        "graph_paths": graph_paths,
        "skeleton_paths": skeleton_paths,
        "volume_paths": volume_paths,
        "graph_object": labeled_graph,
    }


def analyze_embolism_regions_with_graph_enrichment(
    mask_path: str,
    patient_name: str,
    output_dir: str,
    vessel_graph_path: str,
    vessel_mask_path: str,
    level_mask_path: str | None = None,
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
    include_colored_masks: bool = True,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Complete pipeline for embolism analysis with vessel graph enrichment.

    This function combines basic embolism analysis with vessel interaction analysis
    and graph enrichment to provide comprehensive embolism-vessel relationship data.

    Parameters:
    - mask_path: Path to embolism mask file
    - patient_name: Patient identifier
    - output_dir: Output directory for results
    - vessel_graph_path: Path to vessel graph file (NetworkX format)
    - vessel_mask_path: Path to labeled vessel mask file
    - level_mask_path: Path to labeled level mask (optional)
    - separation_method: Method for separating embolisms
    - fusion_threshold: Distance threshold for component fusion
    - include_colored_masks: Whether to create colored visualization masks
    - save_results: Whether to save results to files

    Returns:
    - Dictionary with comprehensive analysis results including:
      - Basic embolism analysis results
      - Vessel interaction results
      - Enriched vessel graph
      - Comprehensive statistics
      - File paths to saved results
    """
    logger.info(
        "Starting comprehensive embolism analysis with graph enrichment for patient: %s",
        patient_name,
    )
    logger.info("Embolism mask: %s", mask_path)
    logger.info("Vessel graph: %s", vessel_graph_path)
    logger.info("Vessel mask: %s", vessel_mask_path)
    logger.info("Output directory: %s", output_dir)

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Basic embolism analysis with visualization
        logger.info("Step 1: Performing basic embolism analysis")
        basic_results = analyze_embolisms_with_visualization(
            mask_path=mask_path,
            patient_name=patient_name,
            output_dir=output_dir,
            separation_method=separation_method,
            fusion_threshold=fusion_threshold,
            include_colored_masks=include_colored_masks,
        )

        if basic_results["num_embolisms"] == 0:
            logger.warning("No embolisms found - returning basic results only")
            return basic_results

        embolism_objects = basic_results["embolism_objects"]
        logger.info(
            "Found %d embolisms for vessel interaction analysis", len(embolism_objects)
        )

        # Step 2: Load vessel data
        logger.info("Step 2: Loading vessel graph and mask data")

        # Load vessel graph
        try:
            vessel_graph = load_graph(vessel_graph_path)
            logger.info(
                "Loaded vessel graph with %d nodes and %d edges",
                len(vessel_graph.nodes),
                len(vessel_graph.edges),
            )
        except Exception as e:
            logger.error("Failed to load vessel graph: %s", e)
            raise RuntimeError(
                f"Failed to load vessel graph from {vessel_graph_path}"
            ) from e

        # Load vessel mask
        try:
            vessel_mask_data, _, vessel_spacing, _ = load_niftii(vessel_mask_path)
            logger.info(
                "Loaded vessel mask with shape: %s and spacing: %s",
                vessel_mask_data.shape,
                vessel_spacing,
            )
        except Exception as e:
            logger.error("Failed to load vessel mask: %s", e)
            raise RuntimeError(
                f"Failed to load vessel mask from {vessel_mask_path}"
            ) from e

        # Load level mask if provided
        level_mask_data = None
        if level_mask_path:
            try:
                level_mask_data, _, _, _ = load_niftii(level_mask_path)
                logger.info("Loaded level mask with shape: %s", level_mask_data.shape)
            except Exception as e:
                logger.error("Failed to load level mask: %s", e)
                raise RuntimeError(
                    f"Failed to load level mask from {level_mask_path}"
                ) from e

        # Step 3: Analyze vessel interactions
        logger.info("Step 3: Analyzing embolism-vessel interactions")
        interaction_results = analyze_embolism_vessel_interactions(
            embolism_objects=embolism_objects,
            vessel_mask=vessel_mask_data,
            spacing=vessel_spacing,
        )

        # Step 4: Compute detailed vessel metrics
        logger.info("Step 4: Computing detailed vessel metrics")
        metrics_results = compute_embolism_vessel_metrics(
            embolism_objects=embolism_objects,
            vessel_mask=vessel_mask_data,
            include_vessel_analysis=True,
        )

        # Step 5: Enrich vessel graph with embolism data
        logger.info("Step 5: Enriching vessel graph with embolism data")
        enriched_graph = enrich_vessel_graph_with_embolisms(
            vessel_graph=vessel_graph,
            embolism_objects=embolism_objects,
            vessel_mask=vessel_mask_data,
            interaction_results=interaction_results,
        )

        # Step 6: Generate comprehensive statistics
        logger.info("Step 6: Generating comprehensive statistics")
        vessel_embolism_stats = get_embolism_vessel_statistics(
            interaction_results, metrics_results
        )

        # Step 7: Add level-based analysis if level mask is available
        level_analysis = {}
        if level_mask_data is not None:
            logger.info("Step 7: Performing level-based analysis")
            try:
                level_analysis = analyze_embolism_by_levels(
                    embolism_objects=embolism_objects,
                    level_mask=level_mask_data,
                )
                logger.info(
                    "Level analysis completed for %d levels",
                    len(level_analysis.get("level_stats", {})),
                )
            except Exception as e:
                logger.warning("Level analysis failed: %s", e)
                level_analysis = {"error": str(e)}

        # Step 8: Combine all results
        logger.info("Step 8: Combining analysis results")
        results = {
            # Basic embolism analysis results
            "patient_name": patient_name,
            "num_embolisms": basic_results["num_embolisms"],
            "total_volume_mm3": basic_results["total_volume_mm3"],
            "total_volume_cm3": basic_results["total_volume_cm3"],
            "embolism_objects": embolism_objects,
            # Vessel interaction results
            "vessel_interactions": interaction_results,
            "vessel_metrics": metrics_results,
            "enriched_vessel_graph": enriched_graph,
            # Statistics and summaries
            "vessel_embolism_statistics": vessel_embolism_stats,
            "level_analysis": level_analysis,
            # Summary statistics
            "summary": {
                "total_embolisms": len(embolism_objects),
                "embolisms_affecting_vessels": sum(
                    1 for r in interaction_results if r["num_affected_vessels"] > 0
                ),
                "total_vessel_interactions": sum(
                    r["num_affected_vessels"] for r in interaction_results
                ),
                "unique_affected_vessels": len(
                    set().union(
                        *[
                            r["affected_vessel_ids"]
                            for r in interaction_results
                            if r["affected_vessel_ids"]
                        ]
                    )
                ),
                "vessels_in_graph": enriched_graph.number_of_nodes(),
                "vessels_with_embolisms": len(
                    get_vessels_with_embolisms(enriched_graph)
                ),
                "percentage_vessels_affected": 0.0,
            },
            # File outputs
            "output_files": basic_results.get("output_files", {}),
        }

        # Calculate percentage of affected vessels
        if results["summary"]["vessels_in_graph"] > 0:
            results["summary"]["percentage_vessels_affected"] = (
                results["summary"]["vessels_with_embolisms"]
                / results["summary"]["vessels_in_graph"]
                * 100
            )

        # Step 9: Save comprehensive results if requested
        if save_results:
            logger.info("Step 9: Saving comprehensive analysis results")
            try:
                # Save vessel-embolism analysis results
                vessel_files = save_vessel_embolism_analysis(
                    interaction_results=interaction_results,
                    metrics_results=metrics_results,
                    statistics=vessel_embolism_stats,
                    output_dir=output_dir,
                    patient_name=patient_name,
                )
                results["output_files"].update(vessel_files)

                # Save enriched graph
                enriched_graph_path = os.path.join(
                    output_dir, f"{patient_name}_enriched_vessel_graph.json"
                )
                save_graph(enriched_graph, enriched_graph_path)
                results["output_files"]["enriched_vessel_graph"] = enriched_graph_path
                # Save comprehensive summary report
                summary_report_path = os.path.join(
                    output_dir, f"{patient_name}_analysis_report.txt"
                )
                save_analysis_report(results, summary_report_path)
                results["output_files"]["report"] = summary_report_path

                # Save level analysis if available
                if level_analysis and "error" not in level_analysis:
                    level_report_path = os.path.join(
                        output_dir, f"{patient_name}_level_analysis.csv"
                    )
                    save_level_analysis_results(level_analysis, level_report_path)
                    results["output_files"]["level_analysis"] = level_report_path

            except Exception as e:
                logger.error("Error saving comprehensive results: %s", e)
                results["save_error"] = str(e)

        # Final logging
        logger.info("Comprehensive embolism analysis completed successfully")
        logger.info("Summary: %d embolisms found", results["num_embolisms"])
        logger.info(
            "         %d vessel interactions detected",
            results["summary"]["total_vessel_interactions"],
        )
        logger.info(
            "         %d/%d vessels affected (%.1f%%)",
            results["summary"]["vessels_with_embolisms"],
            results["summary"]["vessels_in_graph"],
            results["summary"]["percentage_vessels_affected"],
        )

        return results

    except Exception as e:
        logger.error("Error in comprehensive embolism analysis: %s", e)
        raise RuntimeError("Failed to complete comprehensive embolism analysis") from e


def process_patient_complete_pipeline(
    artery_mask_path: str,
    lung_masks_dir: str,
    embolism_mask_path: str | None = None,
    output_dir: str | None = None,
    radius_ratio_threshold: float = 0.3,
    coordinate_format: str = "both",
    save_skeletons: bool = False,
    save_colored_volumes: list[str] | None = None,
    save_all_step_volumes: bool = False,  # NEW PARAMETER
    embolism_separation_method: str = "connected_components",
    embolism_fusion_threshold: float = 10.0,
    skip_steps: list[str] | None = None,
    stop_after: str | None = None,
    diameter_threshold: float = 0.5,
    length_threshold: float = 1.5,
) -> dict[str, Any]:
    """
    Complete end-to-end pipeline processing a patient from artery mask to final analysis.

    Parameters:
    - artery_mask_path: Path to artery mask file
    - lung_masks_dir: Directory containing lung mask files
    - embolism_mask_path: Path to embolism mask file (optional)
    - output_dir: Output directory (defaults to results/patient_name)
    - radius_ratio_threshold: Threshold for cleaning thin branches
    - coordinate_format: "voxel", "real", or "both"
    - save_skeletons: Whether to save skeleton representations
    - save_colored_volumes: List of volume types to save (only applied to labeling step by default)
    - save_all_step_volumes: If True, save colored volumes for all steps (clean, orient, label)
    - vessel_expansion_method: Method for vessel volume expansion
    - embolism_separation_method: Method for separating embolisms
    - embolism_fusion_threshold: Distance threshold for embolism fusion
    - skip_steps: List of steps to skip ["preprocess", "build", "clean", "orient", "label", "embolism"]
    - stop_after: Stop processing after specified step
    - diameter_threshold: Minimum diameter ratio for trunk vessels (default: 0.5)
    - length_threshold: Maximum length/diameter ratio for trunk vessels (default: 1.5)

    Returns:
    - Dictionary with all pipeline results
    """
    # Extract patient name and setup
    patient_name = os.path.basename(artery_mask_path).replace("_arteries.nii.gz", "")

    if output_dir is None:
        output_dir = os.path.join("results", patient_name)

    os.makedirs(output_dir, exist_ok=True)

    if skip_steps is None:
        skip_steps = []

    logger.info("=" * 60)
    logger.info("COMPLETE PIPELINE PROCESSING")
    logger.info("=" * 60)
    logger.info("Patient: %s", patient_name)
    logger.info("Output directory: %s", output_dir)
    logger.info("Artery mask: %s", artery_mask_path)
    if embolism_mask_path:
        logger.info("Embolism mask: %s", embolism_mask_path)

    # Log volume saving strategy
    if save_colored_volumes:
        if save_all_step_volumes:
            logger.info(
                "Colored volumes will be saved for ALL steps: %s", save_colored_volumes
            )
        else:
            logger.info(
                "Colored volumes will be saved ONLY for labeling step: %s",
                save_colored_volumes,
            )

    pipeline_results = {
        "patient_name": patient_name,
        "output_directory": output_dir,
        "input_files": {
            "artery_mask": artery_mask_path,
            "lung_masks_dir": lung_masks_dir,
            "embolism_mask": embolism_mask_path,
        },
    }

    try:
        # Step 1: Preprocessing
        if "preprocess" not in skip_steps:
            logger.info("STEP 1: Preprocessing artery mask")
            skeleton_path, distance_map_path = preprocess_artery_mask(
                artery_mask_path, output_dir
            )
            pipeline_results["preprocessing"] = {
                "skeleton_path": skeleton_path,
                "distance_map_path": distance_map_path,
            }

            if stop_after == "preprocess":
                return pipeline_results
        else:
            # Define paths for skipped step
            skeleton_path = os.path.join(output_dir, "skeleton.nii.gz")
            distance_map_path = os.path.join(output_dir, "distance_map.nii.gz")

        # Step 2: Build graph
        if "build" not in skip_steps:
            logger.info("STEP 2: Building vessel graph")
            build_result = build_graph_from_skeleton(
                skeleton_path,
                distance_map_path,
                output_dir,
                coordinate_format=coordinate_format,
                save_skeleton=save_skeletons,
            )
            pipeline_results["graph_building"] = build_result

            if stop_after == "build":
                return pipeline_results
        else:
            # Define paths for skipped step
            if coordinate_format == "both":
                build_result = {
                    "graph_paths": {
                        "voxel": os.path.join(output_dir, "branch_graph.json"),
                        "real": os.path.join(
                            output_dir, "branch_graph_real_coords.json"
                        ),
                    }
                }
            else:
                build_result = {
                    "graph_paths": {
                        coordinate_format: os.path.join(
                            output_dir,
                            f"branch_graph{'_real_coords' if coordinate_format == 'real' else ''}.json",
                        )
                    }
                }

        # Step 3: Clean graph
        if "clean" not in skip_steps:
            logger.info("STEP 3: Cleaning vessel graph")
            input_graph = (
                build_result["graph_paths"].get("voxel")
                or list(build_result["graph_paths"].values())[0]
            )

            # Only save colored volumes if save_all_step_volumes is True
            clean_volumes = save_colored_volumes if save_all_step_volumes else None

            clean_result = clean_vessel_graph(
                graph_path=input_graph,
                output_dir=output_dir,
                original_skeleton_path=skeleton_path,
                arteries_mask_path=artery_mask_path,
                radius_ratio_threshold=radius_ratio_threshold,
                coordinate_format=coordinate_format,
                save_skeleton=save_skeletons,
                save_colored_volumes=clean_volumes,
            )
            pipeline_results["graph_cleaning"] = clean_result

            if stop_after == "clean":
                return pipeline_results
        else:
            # Define paths for skipped step
            clean_result = {"graph_paths": build_result["graph_paths"]}

        # Step 4: Orient graph
        if "orient" not in skip_steps:
            logger.info("STEP 4: Orienting vessel graph")
            input_graph = (
                clean_result["graph_paths"].get("voxel")
                or list(clean_result["graph_paths"].values())[0]
            )

            # Only save colored volumes if save_all_step_volumes is True
            orient_volumes = save_colored_volumes if save_all_step_volumes else None

            orient_result = orient_vessel_graph(
                graph_path=input_graph,
                output_dir=output_dir,
                original_skeleton_path=skeleton_path,
                arteries_mask_path=artery_mask_path,
                coordinate_format=coordinate_format,
                save_skeleton=save_skeletons,
                save_colored_volumes=orient_volumes,
            )
            pipeline_results["graph_orienting"] = orient_result

            if stop_after == "orient":
                return pipeline_results
        else:
            # Define paths for skipped step
            orient_result = {"graph_paths": clean_result["graph_paths"]}

        # Step 5: Label graph
        if "label" not in skip_steps:
            logger.info("STEP 5: Labeling vessel graph")

            input_graph = None

            if "orient" not in skip_steps and "orient_result" in locals():
                # Use oriented graph from orient step
                input_graph = (
                    orient_result["graph_paths"].get("voxel")
                    or list(orient_result["graph_paths"].values())[0]
                )
                logger.info("Using oriented graph from orient step: %s", input_graph)

            else:
                # Orient step was skipped - look for existing oriented graph file
                oriented_graph_path = os.path.join(output_dir, "oriented_graph.json")
                if os.path.exists(oriented_graph_path):
                    input_graph = oriented_graph_path
                    logger.info("Found existing oriented graph: %s", input_graph)
                else:
                    # Also check for real coordinates version
                    oriented_graph_real_path = os.path.join(
                        output_dir, "oriented_graph_real_coords.json"
                    )
                    if os.path.exists(oriented_graph_real_path):
                        input_graph = oriented_graph_real_path
                        logger.info(
                            "Found existing oriented graph (real coords): %s",
                            input_graph,
                        )

            # Ensure we have an oriented graph
            if not input_graph:
                raise ValueError(
                    "Cannot label graph: No oriented graph found. Please run orient step first or ensure oriented_graph.json exists."
                )

            if not os.path.exists(input_graph):
                raise ValueError(f"Oriented graph file not found: {input_graph}")

            # Always save colored volumes for labeling step (this is the default behavior)
            label_result = label_vessel_graph(
                graph_path=input_graph,
                lung_masks_dir=lung_masks_dir,
                output_dir=output_dir,
                original_skeleton_path=skeleton_path,
                arteries_mask_path=artery_mask_path,
                coordinate_format=coordinate_format,
                save_skeleton=save_skeletons,
                save_colored_volumes=save_colored_volumes,  # Always use the provided volumes for labeling
                diameter_threshold=diameter_threshold,
                length_threshold=length_threshold,
            )
            pipeline_results["graph_labeling"] = label_result

            if stop_after == "label":
                return pipeline_results
        else:
            # Define paths for skipped step
            label_result = {"graph_paths": orient_result["graph_paths"]}

        # Step 6: Embolism analysis (if requested)
        if embolism_mask_path and "embolism" not in skip_steps:
            logger.info("STEP 6: Analyzing embolisms")

            # Use the most processed labeled graph available
            vessel_graph_path = None

            # First try to find existing graphs (prefer labeled > cleaned > branch)
            for graph_name in [
                "labeled_graph.json",
                "cleaned_graph.json",
                "branch_graph.json",
            ]:
                potential_path = os.path.join(output_dir, graph_name)
                if os.path.exists(potential_path):
                    vessel_graph_path = potential_path
                    logger.info(f"Found existing graph: {vessel_graph_path}")
                    break

            # Look for existing vessel volume files directly in the output directory
            vessel_mask_path = None
            level_mask_path = None

            # Search for existing volume files (prefer labeled > cleaned > branch)
            volume_patterns = [
                # Labeled volumes (most processed)
                (
                    "labeled_graph_vessel_volume_edge_id.nii.gz",
                    "labeled_graph_vessel_volume_level.nii.gz",
                ),
                # Cleaned volumes
                (
                    "cleaned_graph_vessel_volume_edge_id.nii.gz",
                    "cleaned_graph_vessel_volume_level.nii.gz",
                ),
                # Branch volumes
                (
                    "branch_graph_vessel_volume_edge_id.nii.gz",
                    "branch_graph_vessel_volume_level.nii.gz",
                ),
            ]

            for edge_pattern, level_pattern in volume_patterns:
                edge_path = os.path.join(output_dir, edge_pattern)
                level_path = os.path.join(output_dir, level_pattern)

                if os.path.exists(edge_path):
                    vessel_mask_path = edge_path
                    logger.info(
                        f"Found existing vessel volume (edge_id): {vessel_mask_path}"
                    )

                if os.path.exists(level_path):
                    level_mask_path = level_path
                    logger.info(
                        f"Found existing vessel volume (level): {level_mask_path}"
                    )

                if vessel_mask_path:  # Use the first set found
                    break

            if not vessel_graph_path:
                logger.warning(
                    "No vessel graph found for embolism analysis - running basic analysis only"
                )

            embolism_result = analyze_embolism_regions_with_graph_enrichment(
                mask_path=embolism_mask_path,
                patient_name=patient_name,
                output_dir=output_dir,
                vessel_graph_path=vessel_graph_path,
                vessel_mask_path=vessel_mask_path,
                level_mask_path=level_mask_path,
                separation_method=embolism_separation_method,
                fusion_threshold=embolism_fusion_threshold,
                include_colored_masks=True,
            )
            pipeline_results["embolism_analysis"] = embolism_result

            if stop_after == "embolism":
                return pipeline_results

        # Pipeline completed successfully
        logger.info("=" * 60)
        logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Patient: %s", patient_name)

        # Summary
        logger.info("Pipeline Summary:")
        logger.info("  ✓ Artery mask preprocessed")
        logger.info("  ✓ Vessel graph constructed and processed")
        logger.info("  ✓ Anatomical labels assigned")

        if embolism_mask_path and "embolism_analysis" in pipeline_results:
            emb_result = pipeline_results["embolism_analysis"]
            logger.info("  ✓ %d embolisms analyzed", emb_result["num_embolisms"])
            logger.info(
                "  ✓ Total embolic volume: %.2f mm³", emb_result["total_volume_mm3"]
            )

        pipeline_results["status"] = "completed"
        return pipeline_results

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        pipeline_results["status"] = "failed"
        pipeline_results["error"] = str(e)
        raise
