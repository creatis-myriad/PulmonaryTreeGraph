"""
Mastora Score Implementation

This module implements the Mastora score for assessing pulmonary embolism severity
based on vessel obstruction patterns in different arterial regions.

The Mastora score evaluates obstruction in three different modes:
- Central: Mediastinal and lobar arteries
- Peripheral: Segmental arteries only
- Global: All arterial levels combined

References:
- Mastora I, et al. Severity of acute pulmonary embolism: evaluation of a new spiral CT angiographic score in correlation with echocardiographic data. Eur Radiol 2003.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# Artery levels mapping (similar to genesis.analysis.config.ArteryLevel)
class ArteryLevel:
    """Artery level constants to match the original implementation."""

    MEDIASTINAL_1 = 1  # Main pulmonary artery
    MEDIASTINAL_2 = 2  # Right/Left pulmonary arteries
    LOBAR = 3
    SEGMENTAL = 4

    # Helper sets for grouping
    MEDIASTINAL = {1, 2}  # Both mediastinal levels
    ALL_CENTRAL = {1, 2, 3}  # Mediastinal + Lobar
    ALL_LEVELS = {1, 2, 3, 4}  # All levels


# Thresholds to discretize obstruction values into 5 levels (1-5) + 0 (no obstruction)
# Left edges are included in intervals, but 0 should be excluded from bin 1, thus the use of 1e-6 for left-most edge
DEGREE_THRESHOLDS = [1e-6, 0.25, 0.5, 0.75, 1]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _find_root_node(graph: nx.DiGraph) -> int:
    """Find the root node of a directed graph (node with no predecessors)."""
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            return node
    raise ValueError("No root node found in graph")


def _extract_patient_id_from_path_or_string(patient_input: str) -> str:
    """
    Extract 4-digit patient ID from various input formats.

    Args:
        patient_input: Patient identifier in various formats
                      (e.g., "0001", "patient_0001", "/path/to/0001_arteries.nii.gz")

    Returns:
        str: 4-digit patient ID (e.g., "0001")
    """
    # Try to find a 4-digit pattern
    match = re.search(r"\d{4}", patient_input)
    if match:
        return match.group()

    # Fallback: return original input
    return patient_input


# =============================================================================
# CSV STATISTICS FUNCTIONS
# =============================================================================


def _initialize_mastora_stats_csv(output_path: Path) -> None:
    """Create CSV file with header for Mastora statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "patient_id",
                "mastora_central_score",
                "mastora_peripheral_score",
                "mastora_global_score",
                "obstruction_attr",
                "central_segments_analyzed",
                "peripheral_segments_analyzed",
                "global_segments_analyzed",
                "central_mean_obstruction",
                "peripheral_mean_obstruction",
                "global_mean_obstruction",
            ]
        )


def _append_mastora_stats_to_csv(
    output_path: Path,
    patient_id: str,
    scores: dict[str, float],
    stats: dict[str, Any],
    obstruction_attr: str = "volumetric_obstruction",
) -> None:
    """Append Mastora statistics row to CSV file."""
    # Extract 4-digit patient ID
    patient_id_clean = _extract_patient_id_from_path_or_string(patient_id)

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                patient_id_clean,
                f"{scores['central']:.6f}",
                f"{scores['peripheral']:.6f}",
                f"{scores['global']:.6f}",
                obstruction_attr,
                stats["central_segments"],
                stats["peripheral_segments"],
                stats["global_segments"],
                f"{stats['central_mean_obstruction']:.6f}",
                f"{stats['peripheral_mean_obstruction']:.6f}",
                f"{stats['global_mean_obstruction']:.6f}",
            ]
        )


# =============================================================================
# MAIN MASTORA SCORE FUNCTION
# =============================================================================


def mastora_score(
    graph: nx.DiGraph,
    mode: Literal["central", "peripheral", "global"],
    obstruction_attr: str = "volumetric_obstruction",
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
    debug: bool = False,
) -> float | tuple[float, dict]:
    """
    Compute the Mastora score for a directed graph with embolism information.

    Args:
        graph: Directed graph representing the arterial tree
        mode: Variant of the Mastora score to compute:
            - 'central': Considers obstructions in the mediastinal (levels 1,2) and lobar (level 3) arteries
            - 'peripheral': Considers obstructions in the segmental arteries (level 4)
            - 'global': Considers obstructions in all arteries (levels 1,2,3,4)
        obstruction_attr: The name of the edge attribute to use for obstruction values
        patient_id: Patient identifier for statistics tracking
        patient_stats_path: Path for individual patient statistics CSV
        global_stats_path: Path for global statistics CSV
        debug: If True, return debug information for visualization

    Returns:
        float or tuple: Mastora score (0-1) or (score, debug_info) if debug=True

    Algorithm:
        1. Traverse graph from root using depth-first search
        2. Collect obstruction values for arteries at specified levels
        3. Discretize obstruction values into 5 levels (1-5) + 0 (no obstruction)
        4. Compute average normalized score: mean(degrees / max_degrees)
    """

    # Determine numerical levels for which to look for obstructions, based on mode
    match mode:
        case "global":
            levels_to_search = ArteryLevel.ALL_LEVELS  # {1, 2, 3, 4}
        case "central":
            levels_to_search = ArteryLevel.ALL_CENTRAL  # {1, 2, 3}
        case "peripheral":
            levels_to_search = {ArteryLevel.SEGMENTAL}  # {4}
        case _:
            raise ValueError(
                f"Invalid mode '{mode}'. Allowed values are 'central', 'peripheral' or 'global'."
            )

    # Data structures to save data for each selected edge
    obstructions: dict[tuple[int, int], float] = {}
    debug_info: dict[tuple[int, int], str] = {}

    def _depth_first_search(node: int) -> None:
        """Recursive DFS traversal."""
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs.get(obstruction_attr, 0.0)
            artery_level = edge_attrs.get("level")

            if artery_level is None:
                logger.warning(
                    f"Edge ({node}, {child}) has no level attribute, skipping"
                )
                continue

            if artery_level in levels_to_search:
                obstructions[(node, child)] = artery_obstruction

                if debug:
                    # Updated level name mapping
                    level_name_map = {
                        1: "MEDIASTINAL_1",  # Main PA
                        2: "MEDIASTINAL_2",  # RPA/LPA
                        3: "LOBAR",
                        4: "SEGMENTAL",
                    }
                    level_name = level_name_map.get(artery_level, f"L{artery_level}")
                    debug_info[
                        (node, child)
                    ] = f"{level_name[0]}: {artery_obstruction:.3f}"

            # Recursively visit children
            _depth_first_search(child)

    # Find root and start DFS
    try:
        root_node = _find_root_node(graph)
        _depth_first_search(root_node)
    except ValueError as e:
        logger.error(f"Cannot compute Mastora score: {e}")
        return 0.0 if not debug else (0.0, {})

    # Compute score from collected data
    if not obstructions:
        logger.warning(
            f"No segments found for Mastora score computation (mode: {mode})"
        )
        return 0.0 if not debug else (0.0, {})

    obstructions_vals = list(obstructions.values())

    # Discretize obstruction values between {0...5}, so that normalization will be in [0, 1]
    degrees = np.digitize(obstructions_vals, DEGREE_THRESHOLDS)

    # Original paper summed the degrees across the landmark arteries, but here we normalize and average degrees
    # to get a score between 0 and 1, invariant to the exact number of arteries of the patient
    score = np.mean(degrees / len(DEGREE_THRESHOLDS)).item()

    logger.info(f"Mastora score ({mode}): {score:.4f}")
    logger.info(f"  Segments analyzed: {len(obstructions_vals)}")
    logger.info(f"  Mean obstruction: {np.mean(obstructions_vals):.4f}")
    logger.info(f"  Mean degree: {np.mean(degrees):.2f}")

    if debug:
        return score, debug_info
    return score


def compute_mastora_scores(
    graph: nx.DiGraph,
    obstruction_attr: str = "volumetric_obstruction",
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
) -> dict[str, float]:
    """
    Compute all Mastora score variants (central, peripheral, global) for a graph.

    Args:
        graph: Directed graph representing the arterial tree
        obstruction_attr: The name of the edge attribute to use for obstruction values
        patient_id: Patient identifier for statistics tracking
        patient_stats_path: Path for individual patient statistics CSV
        global_stats_path: Path for global statistics CSV

    Returns:
        Dictionary with all three Mastora scores
    """

    scores = {}
    stats = {}

    # Compute all three variants
    for mode in ["central", "peripheral", "global"]:
        try:
            score = mastora_score(
                graph,
                mode=mode,
                obstruction_attr=obstruction_attr,
                debug=False,
            )
            scores[mode] = score

            # Collect additional statistics for this mode
            levels_to_search = {
                "central": ArteryLevel.ALL_CENTRAL,  # {1, 2, 3}
                "peripheral": {ArteryLevel.SEGMENTAL},  # {4}
                "global": ArteryLevel.ALL_LEVELS,  # {1, 2, 3, 4}
            }[mode]

            # Count segments and compute mean obstruction for this mode
            obstructions = []
            for _, _, data in graph.edges(data=True):
                level = data.get("level")
                if level in levels_to_search:
                    obstructions.append(data.get(obstruction_attr, 0.0))

            stats[f"{mode}_segments"] = len(obstructions)
            stats[f"{mode}_mean_obstruction"] = (
                np.mean(obstructions) if obstructions else 0.0
            )

        except Exception as e:
            logger.error(f"Error computing Mastora score ({mode}): {e}")
            scores[mode] = 0.0
            stats[f"{mode}_segments"] = 0
            stats[f"{mode}_mean_obstruction"] = 0.0

    # Save statistics if paths provided
    if patient_id and (patient_stats_path or global_stats_path):
        if patient_stats_path:
            if not patient_stats_path.exists():
                _initialize_mastora_stats_csv(patient_stats_path)
            _append_mastora_stats_to_csv(
                patient_stats_path, patient_id, scores, stats, obstruction_attr
            )

        if global_stats_path:
            if not global_stats_path.exists():
                _initialize_mastora_stats_csv(global_stats_path)
            _append_mastora_stats_to_csv(
                global_stats_path, patient_id, scores, stats, obstruction_attr
            )

    return scores
