"""
Qanadli Score Implementation

This module implements the Qanadli score for assessing pulmonary embolism severity
based on vessel obstruction and anatomical distribution.

The Qanadli score evaluates:
- Degree of obstruction (partial vs total)
- Number of affected segments
- Hierarchical weighting of vessel levels

References:
- Qanadli SD, et al. New CT index to quantify arterial obstruction in pulmonary embolism. AJR 2001.
"""

import csv
import logging
import re
from pathlib import Path

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


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


def _is_terminal(graph: nx.DiGraph, edge: tuple[int, int], terminal_level: int) -> bool:
    """
    Check if an edge is terminal, i.e. of defined level at most with no successors of that level at most.

    Args:
        graph: Directed graph representing the arterial tree
        edge: IDs of the source and destination nodes of the edge to check
        terminal_level: Level at the end of which descendants are considered terminal

    Returns:
        True if the edge is terminal, i.e. of defined level at most with no successors of that level at most
    """
    parent, node = edge
    edge_level = graph.edges[parent, node].get("level")
    if edge_level is None or edge_level > terminal_level:
        return False  # Edge above the terminal level cannot be terminal

    # Check if all children have level > terminal_level
    for child in graph.successors(node):
        child_level = graph.edges[node, child].get("level")
        if child_level is not None and child_level <= terminal_level:
            return False
    return True


def _count_terminal_descendants(
    graph: nx.DiGraph, edge: tuple[int, int], terminal_level: int
) -> int:
    """
    Count the number of terminal descendants of an edge.

    Args:
        graph: Directed graph representing the arterial tree
        edge: IDs of the source and destination nodes of the edge from which to start counting
        terminal_level: Artery level at the end of which descendants are considered terminal

    Returns:
        Number of terminal descendants of an edge
    """

    def _depth_first_search(_edge: tuple[int, int]) -> int:
        if _is_terminal(graph, _edge, terminal_level):
            return 1
        _parent, node = _edge
        return sum(
            _depth_first_search((node, child)) for child in graph.successors(node)
        )

    return _depth_first_search(edge)


# =============================================================================
# CSV STATISTICS FUNCTIONS
# =============================================================================


def _initialize_qanadli_stats_csv(output_path: Path) -> None:
    """Create CSV file with header for Qanadli statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "patient_id",
                "obstruction_attr",
                "total_segments",
                "obstructed_segments",
                "total_obstruction_segments",
                "partial_obstruction_segments",
                "total_weighted_score",
                # Literature Qanadli (fixed 40 points max)
                "qanadli_literature_score",
                "qanadli_literature_score_pct",
                "max_possible_score_literature",
                # Patient-specific Qanadli (dynamic max based on actual segments)
                "qanadli_patient_score",
                "qanadli_patient_score_pct",
                "max_possible_score_patient",
            ]
        )


def _append_qanadli_stats_to_csv(
    output_path: Path,
    patient_id: str,
    qanadli_literature_score: float,
    qanadli_patient_score: float,
    stats: dict,
    obstruction_attr: str = "volumetric_obstruction",
) -> None:
    """Append Qanadli statistics row to CSV file."""
    # Extract 4-digit patient ID
    patient_id_clean = _extract_patient_id_from_path_or_string(patient_id)

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                patient_id_clean,
                obstruction_attr,
                stats["total_segments"],
                stats["obstructed_segments"],
                stats["total_obstruction_segments"],
                stats["partial_obstruction_segments"],
                stats["total_weighted_score"],
                # Literature scores
                f"{qanadli_literature_score:.6f}",
                f"{qanadli_literature_score * 100:.2f}",
                stats["max_possible_score_literature"],
                # Patient-specific scores
                f"{qanadli_patient_score:.6f}",
                f"{qanadli_patient_score * 100:.2f}",
                stats["max_possible_score_patient"],
            ]
        )


# =============================================================================
# MAIN QANADLI SCORE FUNCTION
# =============================================================================


def compute_qanadli_score(
    graph: nx.DiGraph,
    partial_obstruction_thresh: float = 0.25,
    total_obstruction_thresh: float = 0.75,
    obstruction_attr: str = "volumetric_obstruction",
    terminal_level: int = 4,  # Segmental level
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
    debug: bool = False,
) -> float | tuple[float, dict]:
    """
    Compute both literature and patient-specific Qanadli scores for a directed graph.

    Args:
        graph: Directed graph representing the arterial tree
        partial_obstruction_thresh: Threshold to consider a segment partially obstructed (default: 0.25)
        total_obstruction_thresh: Threshold to consider a segment totally obstructed (default: 0.75)
        obstruction_attr: The name of the edge attribute to use for obstruction values
        terminal_level: Artery level considered terminal (default: 4 for segmental)
        patient_id: Patient identifier for statistics tracking
        patient_stats_path: Path for individual patient statistics CSV
        global_stats_path: Path for global statistics CSV
        debug: If True, return debug information for visualization

    Returns:
        float or tuple: Literature Qanadli score (0-1) or (score, debug_info) if debug=True

    Algorithm:
        1. Traverse graph from root using depth-first search
        2. For each artery at segmental level or above:
           - If obstruction > partial_thresh: count all descendant segments as obstructed
           - If terminal segment: count as 1 segment
           - Otherwise: continue recursion
        3. Compute two versions:
           - Literature: weighted_score / 40 (fixed 20 segments × 2 max degree)
           - Patient-specific: weighted_score / (2 × actual_segments)
    """

    # Data structures to save data for each selected edge
    descendant_segments_counts: dict[tuple[int, int], int] = {}
    obstructions: dict[tuple[int, int], float] = {}
    debug_info: dict[tuple[int, int], str] = {}

    def _store_edge_data(
        edge: tuple[int, int],
        obstruction: float,
        descendant_segments_count: int,
        level: int,
    ) -> None:
        """Store data for a given edge."""
        obstructions[edge] = obstruction
        descendant_segments_counts[edge] = descendant_segments_count

        if debug:
            degree = np.digitize(
                obstruction, [partial_obstruction_thresh, total_obstruction_thresh]
            )
            level_name = f"L{level}"
            debug_info[
                edge
            ] = f"{level_name}: {obstruction:.3f} (n_seg:{descendant_segments_count}, deg:{degree})"

    def _depth_first_search(node: int) -> None:
        """Recursive DFS traversal with subsegmental handling."""
        for child in graph.successors(node):
            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs.get(obstruction_attr, 0.0)
            artery_level = edge_attrs.get("level")

            if artery_level is None:
                logger.warning(
                    f"Edge ({node}, {child}) has no level attribute, skipping"
                )
                continue

            # Handle subsegmental PEs (level > terminal_level) with fixed 1 point
            if (
                artery_level == terminal_level + 1
                and artery_obstruction > partial_obstruction_thresh
            ):
                # Isolated subsegmental PE gets exactly 1 point regardless of degree
                logger.debug(f"Subsegmental PE at level {artery_level}: fixed 1 point")
                _store_edge_data((node, child), 1.0, 1, artery_level)  # Force degree=1
                # Continue recursion to check for more subsegmental PEs
                _depth_first_search(child)

            elif (
                artery_level <= terminal_level
            ):  # Original logic for segmental and above
                if artery_obstruction > partial_obstruction_thresh:
                    # If artery is obstructed enough:
                    # - Count the whole subtree as obstructed to the same degree
                    # - Stop recursion
                    descendant_segments_count = _count_terminal_descendants(
                        graph, (node, child), terminal_level
                    )
                    _store_edge_data(
                        (node, child),
                        artery_obstruction,
                        descendant_segments_count,
                        artery_level,
                    )

                elif _is_terminal(graph, (node, child), terminal_level):
                    # If artery has no descendants of terminal level or below:
                    # - Count it as one non-obstructed artery
                    # - Stop recursion
                    _store_edge_data((node, child), artery_obstruction, 1, artery_level)

                else:
                    # Recursively visit children
                    _depth_first_search(child)
            else:
                # Non-obstructed subsegmental: continue recursion
                _depth_first_search(child)

    # Find root and start DFS
    try:
        root_node = _find_root_node(graph)
        _depth_first_search(root_node)
    except ValueError as e:
        logger.error(f"Cannot compute Qanadli score: {e}")
        return 0.0 if not debug else (0.0, {})

    # Compute score with both normalization methods
    if not obstructions:
        logger.warning("No segments found for Qanadli score computation")
        return 0.0 if not debug else (0.0, {})

    obstructions_vals = list(obstructions.values())
    descendant_segments_counts_vals = list(descendant_segments_counts.values())

    # Handle subsegmental PEs: force degree=1 for obstruction values that were set to 1.0
    degrees = []
    for obs in obstructions_vals:
        if obs == 1.0:  # Subsegmental PE marker
            degrees.append(1)  # Fixed degree for subsegmental
        else:
            # Normal discretization for segmental and above
            degree = np.digitize(
                [obs], [partial_obstruction_thresh, total_obstruction_thresh]
            )[0]
            degrees.append(degree)

    degrees = np.array(degrees)

    # Compute weighted score
    weighted_degrees = np.array(descendant_segments_counts_vals) * degrees
    total_weighted_score = np.sum(weighted_degrees)
    total_segments = np.sum(descendant_segments_counts_vals)

    # Compute both normalization methods
    max_possible_score_literature = 40  # Fixed 20 segments × 2 max degree
    max_possible_score_patient = 2 * total_segments  # Dynamic based on actual segments

    # Qanadli scores (normalized to 0-1)
    qanadli_literature_score = (
        total_weighted_score / max_possible_score_literature
        if max_possible_score_literature > 0
        else 0.0
    )
    qanadli_patient_score = (
        total_weighted_score / max_possible_score_patient
        if max_possible_score_patient > 0
        else 0.0
    )

    # Statistics for CSV output
    stats = {
        "total_segments": int(total_segments),
        "obstructed_segments": int(np.sum(degrees > 0)),
        "partial_obstruction_segments": int(np.sum(degrees == 1)),
        "total_obstruction_segments": int(np.sum(degrees == 2)),
        "total_weighted_score": int(total_weighted_score),
        "max_possible_score_literature": int(max_possible_score_literature),
        "max_possible_score_patient": int(max_possible_score_patient),
    }

    logger.info(
        f"Qanadli Literature Score: {qanadli_literature_score:.4f} ({qanadli_literature_score * 100:.1f}%)"
    )
    logger.info(
        f"Qanadli Patient Score: {qanadli_patient_score:.4f} ({qanadli_patient_score * 100:.1f}%)"
    )
    logger.info(f"  Total segments: {stats['total_segments']} (literature assumes 20)")
    logger.info(f"  Obstructed segments: {stats['obstructed_segments']}")
    logger.info(f"  Partial obstructions: {stats['partial_obstruction_segments']}")
    logger.info(f"  Total obstructions: {stats['total_obstruction_segments']}")
    logger.info(f"  Weighted score: {stats['total_weighted_score']}")

    # Save statistics if paths provided
    if patient_id and patient_stats_path:
        if not patient_stats_path.exists():
            _initialize_qanadli_stats_csv(patient_stats_path)
        _append_qanadli_stats_to_csv(
            patient_stats_path,
            patient_id,
            qanadli_literature_score,
            qanadli_patient_score,
            stats,
            obstruction_attr,
        )

    if patient_id and global_stats_path:
        if not global_stats_path.exists():
            _initialize_qanadli_stats_csv(global_stats_path)
        _append_qanadli_stats_to_csv(
            global_stats_path,
            patient_id,
            qanadli_literature_score,
            qanadli_patient_score,
            stats,
            obstruction_attr,
        )

    if debug:
        return qanadli_literature_score, debug_info

    # Return the literature score as the main score (for backward compatibility)
    return qanadli_literature_score
