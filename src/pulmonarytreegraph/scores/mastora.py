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
                "obstruction_attr",
                # Central scores (mediastinal + lobar)
                "mastora_central_literature_score",
                "mastora_central_literature_score_pct",
                "mastora_central_patient_score",
                "mastora_central_patient_score_pct",
                "central_segments_literature",  # Fixed: 5 mediastinal + 6 lobar = 11
                "central_segments_patient",  # Actual count
                "central_total_weighted_score",
                "central_max_possible_score_literature",  # 11 × 5 = 55
                "central_max_possible_score_patient",  # actual_count × 5
                "central_mean_obstruction",
                # Peripheral scores (segmental)
                "mastora_peripheral_literature_score",
                "mastora_peripheral_literature_score_pct",
                "mastora_peripheral_patient_score",
                "mastora_peripheral_patient_score_pct",
                "peripheral_segments_literature",  # Fixed: 20 segmental
                "peripheral_segments_patient",  # Actual count
                "peripheral_total_weighted_score",
                "peripheral_max_possible_score_literature",  # 20 × 5 = 100
                "peripheral_max_possible_score_patient",  # actual_count × 5
                "peripheral_mean_obstruction",
                # Global scores (all levels)
                "mastora_global_literature_score",
                "mastora_global_literature_score_pct",
                "mastora_global_patient_score",
                "mastora_global_patient_score_pct",
                "global_segments_literature",  # Fixed: 5+6+20 = 31 total
                "global_segments_patient",  # Actual count
                "global_total_weighted_score",
                "global_max_possible_score_literature",  # 31 × 5 = 155
                "global_max_possible_score_patient",  # actual_count × 5
                "global_mean_obstruction",
            ]
        )


def _append_mastora_stats_to_csv(
    output_path: Path,
    patient_id: str,
    literature_scores: dict[str, float],
    patient_scores: dict[str, float],
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
                obstruction_attr,
                # Central scores
                f"{literature_scores['central']:.6f}",
                f"{literature_scores['central'] * 100:.2f}",
                f"{patient_scores['central']:.6f}",
                f"{patient_scores['central'] * 100:.2f}",
                11,  # Literature: 5 mediastinal + 6 lobar
                stats["central_segments_patient"],
                stats["central_total_weighted_score"],
                55,  # Literature: 11 × 5 = 55
                stats["central_max_possible_score_patient"],
                f"{stats['central_mean_obstruction']:.6f}",
                # Peripheral scores
                f"{literature_scores['peripheral']:.6f}",
                f"{literature_scores['peripheral'] * 100:.2f}",
                f"{patient_scores['peripheral']:.6f}",
                f"{patient_scores['peripheral'] * 100:.2f}",
                20,  # Literature: 20 segmental
                stats["peripheral_segments_patient"],
                stats["peripheral_total_weighted_score"],
                100,  # Literature: 20 × 5 = 100
                stats["peripheral_max_possible_score_patient"],
                f"{stats['peripheral_mean_obstruction']:.6f}",
                # Global scores
                f"{literature_scores['global']:.6f}",
                f"{literature_scores['global'] * 100:.2f}",
                f"{patient_scores['global']:.6f}",
                f"{patient_scores['global'] * 100:.2f}",
                31,  # Literature: 5+6+20 = 31 total
                stats["global_segments_patient"],
                stats["global_total_weighted_score"],
                155,  # Literature: 31 × 5 = 155
                stats["global_max_possible_score_patient"],
                f"{stats['global_mean_obstruction']:.6f}",
            ]
        )


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
    Compute both literature and patient-specific Mastora scores for a directed graph.

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
        float or tuple: Literature Mastora score (0-1) or (score, debug_info) if debug=True

    Literature Model (Mastora 2003):
        - 5 mediastinal arteries (levels 1-2): pulmonary trunk, RPA, LPA, right and left interlobar
        - 6 lobar arteries (level 3): RUL, RML, RLL, LUL, LLL, lingula
        - 20 segmental arteries (level 4): standard anatomical segments
        - Five-level grading: 0%, 1-24%, 25-49%, 50-74%, 75-99%, 100%

    Algorithm:
        1. Traverse graph from root using depth-first search
        2. Collect obstruction values for arteries at specified levels
        3. Discretize obstruction values into 6 levels (0-5)
        4. Compute two scores:
           - Literature: sum(degrees) / fixed_max_score (e.g., 155 for global)
           - Patient-specific: sum(degrees) / (actual_segments × 5)
    """

    # Literature vessel counts and max scores based on Mastora 2003 paper
    LITERATURE_MODEL = {
        "central": {
            "vessel_count": 11,  # 5 mediastinal + 6 lobar
            "max_score": 55,  # 11 × 5
        },
        "peripheral": {
            "vessel_count": 20,  # 20 segmental
            "max_score": 100,  # 20 × 5
        },
        "global": {
            "vessel_count": 31,  # 5 + 6 + 20 = 31 total
            "max_score": 155,  # 31 × 5
        },
    }

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
    actual_vessel_count = len(obstructions_vals)

    literature_model = LITERATURE_MODEL[mode]
    literature_vessel_count = literature_model["vessel_count"]
    literature_max_score = literature_model["max_score"]

    # Discretize obstruction values into 6 levels (0-5) using Mastora thresholds
    degrees = np.digitize(obstructions_vals, DEGREE_THRESHOLDS)

    # Compute total weighted score (sum of degrees) - THIS IS THE KEY DIFFERENCE
    total_weighted_score = np.sum(degrees)

    # Compute both normalization methods
    # Literature: normalize using fixed vessel count and max score from anatomical model
    literature_max_score_possible = literature_max_score  # e.g., 155 for global
    literature_score = (
        total_weighted_score / literature_max_score_possible
        if literature_max_score_possible > 0
        else 0.0
    )

    # Patient-specific: normalize using actual vessel count
    patient_max_score_possible = actual_vessel_count * 5  # 5 is max degree
    patient_score = (
        total_weighted_score / patient_max_score_possible
        if patient_max_score_possible > 0
        else 0.0
    )

    logger.info(
        f"Mastora {mode} Literature Score: {literature_score:.4f} ({literature_score * 100:.1f}%)"
    )
    logger.info(
        f"Mastora {mode} Patient Score: {patient_score:.4f} ({patient_score * 100:.1f}%)"
    )
    logger.info(
        f"  Segments analyzed: {actual_vessel_count} (literature expects {literature_vessel_count})"
    )
    logger.info(f"  Total weighted score: {total_weighted_score}")
    logger.info(f"  Literature max possible: {literature_max_score_possible}")
    logger.info(f"  Patient max possible: {patient_max_score_possible}")
    logger.info(f"  Mean obstruction: {np.mean(obstructions_vals):.4f}")

    if debug:
        return literature_score, debug_info

    # Return literature score as main score (for backward compatibility)
    return literature_score


def compute_mastora_scores(
    graph: nx.DiGraph,
    obstruction_attr: str = "volumetric_obstruction",
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
) -> dict[str, float]:
    """
    Compute all Mastora score variants (central, peripheral, global) with both literature and patient-specific scoring.

    Args:
        graph: Directed graph representing the arterial tree
        obstruction_attr: The name of the edge attribute to use for obstruction values
        patient_id: Patient identifier for statistics tracking
        patient_stats_path: Path for individual patient statistics CSV
        global_stats_path: Path for global statistics CSV

    Returns:
        Dictionary with literature Mastora scores (for backward compatibility)
    """

    literature_scores = {}
    patient_scores = {}
    stats = {}

    # Compute all three variants
    for mode in ["central", "peripheral", "global"]:
        try:
            # Get the literature score as main return value
            literature_score = mastora_score(
                graph,
                mode=mode,
                obstruction_attr=obstruction_attr,
                debug=False,
            )

            literature_scores[mode] = literature_score

            # Recalculate for patient score (we need both calculations)
            levels_to_search = {
                "central": ArteryLevel.ALL_CENTRAL,  # {1, 2, 3}
                "peripheral": ArteryLevel.SEGMENTAL,  # {4}
                "global": ArteryLevel.ALL_LEVELS,  # {1, 2, 3, 4}
            }[mode]

            # Collect data for this mode
            obstructions = []
            for _, _, data in graph.edges(data=True):
                level = data.get("level")
                if level in levels_to_search:
                    obstructions.append(data.get(obstruction_attr, 0.0))

            if obstructions:
                degrees = np.digitize(obstructions, DEGREE_THRESHOLDS)
                total_weighted_score = np.sum(degrees)
                actual_vessel_count = len(obstructions)
                patient_max_score_possible = actual_vessel_count * 5
                patient_scores[mode] = total_weighted_score / patient_max_score_possible

                # Store additional stats
                stats[f"{mode}_segments_patient"] = actual_vessel_count
                stats[f"{mode}_total_weighted_score"] = int(total_weighted_score)
                stats[f"{mode}_max_possible_score_patient"] = int(
                    patient_max_score_possible
                )
                stats[f"{mode}_mean_obstruction"] = np.mean(obstructions)
            else:
                patient_scores[mode] = 0.0
                stats[f"{mode}_segments_patient"] = 0
                stats[f"{mode}_total_weighted_score"] = 0
                stats[f"{mode}_max_possible_score_patient"] = 0
                stats[f"{mode}_mean_obstruction"] = 0.0

        except Exception as e:
            logger.error(f"Error computing Mastora score ({mode}): {e}")
            literature_scores[mode] = 0.0
            patient_scores[mode] = 0.0
            stats[f"{mode}_segments_patient"] = 0
            stats[f"{mode}_total_weighted_score"] = 0
            stats[f"{mode}_max_possible_score_patient"] = 0
            stats[f"{mode}_mean_obstruction"] = 0.0

    # Save statistics if paths provided
    if patient_id and (patient_stats_path or global_stats_path):
        if patient_stats_path:
            if not patient_stats_path.exists():
                _initialize_mastora_stats_csv(patient_stats_path)
            _append_mastora_stats_to_csv(
                patient_stats_path,
                patient_id,
                literature_scores,
                patient_scores,
                stats,
                obstruction_attr,
            )

        if global_stats_path:
            if not global_stats_path.exists():
                _initialize_mastora_stats_csv(global_stats_path)
            _append_mastora_stats_to_csv(
                global_stats_path,
                patient_id,
                literature_scores,
                patient_scores,
                stats,
                obstruction_attr,
            )

    # Return literature scores for backward compatibility
    return literature_scores
