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
    """Create CSV file with header for comprehensive Qanadli statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "patient_id",
                "obstruction_attr",
                # Segment counts
                "total_segments_patient",
                "total_segments_literature",
                "obstructed_pathways",
                # Obstruction breakdown
                "partial_obstruction_pathways",
                "total_obstruction_pathways",
                # Literature scoring (fixed anatomical model)
                "literature_weighted_score",
                "literature_max_possible_score",
                "qanadli_literature_score",
                "qanadli_literature_score_pct",
                # Patient-specific scoring (actual anatomy)
                "patient_weighted_score",
                "patient_max_possible_score",
                "qanadli_patient_score",
                "qanadli_patient_score_pct",
                # Detailed breakdown by obstruction type
                "level_2_obstructions",
                "level_3_obstructions",
                "level_4_obstructions",
                "level_5_obstructions",
                # Literature contribution breakdown
                "literature_level_2_contribution",
                "literature_level_3_contribution",
                "literature_level_4_contribution",
                "literature_level_5_contribution",
                # Patient contribution breakdown
                "patient_level_2_contribution",
                "patient_level_3_contribution",
                "patient_level_4_contribution",
                "patient_level_5_contribution",
                # Severity indicators
                "mean_obstruction_value",
                "max_obstruction_value",
                "proximal_vs_distal_ratio",  # (Level 2+3) / (Level 4+5) contributions
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
    """Append comprehensive Qanadli statistics row to CSV file."""
    # Extract 4-digit patient ID
    patient_id_clean = _extract_patient_id_from_path_or_string(patient_id)

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                patient_id_clean,
                obstruction_attr,
                # Segment counts
                stats["total_segments_patient"],
                20,  # Literature always assumes 20 segments
                stats["obstructed_pathways"],
                # Obstruction breakdown
                stats["partial_obstruction_pathways"],
                stats["total_obstruction_pathways"],
                # Literature scoring
                stats["literature_weighted_score"],
                stats["literature_max_possible_score"],
                f"{qanadli_literature_score:.6f}",
                f"{qanadli_literature_score * 100:.2f}",
                # Patient-specific scoring
                stats["patient_weighted_score"],
                stats["patient_max_possible_score"],
                f"{qanadli_patient_score:.6f}",
                f"{qanadli_patient_score * 100:.2f}",
                # Detailed breakdown by obstruction type
                stats["level_2_obstructions"],
                stats["level_3_obstructions"],
                stats["level_4_obstructions"],
                stats["level_5_obstructions"],
                # Literature contribution breakdown
                stats["literature_level_2_contribution"],
                stats["literature_level_3_contribution"],
                stats["literature_level_4_contribution"],
                stats["literature_level_5_contribution"],
                # Patient contribution breakdown
                stats["patient_level_2_contribution"],
                stats["patient_level_3_contribution"],
                stats["patient_level_4_contribution"],
                stats["patient_level_5_contribution"],
                # Severity indicators
                f"{stats['mean_obstruction_value']:.6f}",
                f"{stats['max_obstruction_value']:.6f}",
                f"{stats['proximal_vs_distal_ratio']:.4f}",
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
) -> tuple[float, float]:
    """Compute both literature and patient-specific Qanadli scores for a directed graph."""

    # Literature segment counts by lobe (Qanadli 2001)
    LITERATURE_LOBE_SEGMENTS = {
        "RUL": 3,
        "RML": 2,
        "RLL": 5,
        "LUL": 4,
        "LLL": 6,
        "interlobar": 3,  # Default for interlobar vessels
    }

    # Data structures to save data for each selected edge
    literature_weighted_scores: list[int] = []
    patient_weighted_scores: list[int] = []
    debug_info: dict[tuple[int, int], str] = {}
    detailed_breakdown: list[dict] = []

    def _get_literature_segment_count(level: int, lateral: str, lobe: str) -> int:
        """Get expected segment count based on literature anatomical model."""
        if level == 2:
            return 10
        elif level == 3:
            # Handle lobe as list (from your graph structure)
            if isinstance(lobe, list) and len(lobe) > 0:
                lobe_str = lobe[0]  # Take first element
            else:
                lobe_str = str(lobe) if lobe else "None"

            if lobe_str in LITERATURE_LOBE_SEGMENTS:
                return LITERATURE_LOBE_SEGMENTS[lobe_str]
            elif lobe_str == "None" or lobe_str is None:
                return 10  # Default to half the lung
            else:
                return 3  # Default fallback
        elif level >= 4:
            return 1
        else:
            return 1

    def _get_obstruction_degree(obstruction_value: float, level: int) -> int:
        """Get obstruction degree with special handling for subsegmental PEs."""
        if level == 5:  # Subsegmental PE
            return 1  # Always 1 point regardless of degree
        else:
            # Normal discretization for levels 1-4
            return np.digitize(
                [obstruction_value],
                [partial_obstruction_thresh, total_obstruction_thresh],
            )[0]

    def _depth_first_search(node: int, visited_pathway: set[tuple[int, int]]) -> None:
        """Recursive DFS traversal with proper stopping rules."""
        for child in graph.successors(node):
            edge_key = (node, child)

            if edge_key in visited_pathway:
                continue

            edge_attrs = graph.edges[node, child]
            artery_obstruction = edge_attrs.get(obstruction_attr, 0.0)
            artery_level = edge_attrs.get("level")
            artery_lateral = edge_attrs.get("lateral", "unknown")
            artery_lobe = edge_attrs.get("lobe", "None")

            if artery_level is None:
                logger.warning(
                    f"Edge ({node}, {child}) has no level attribute, skipping"
                )
                continue

            new_visited_pathway = visited_pathway.copy()
            new_visited_pathway.add(edge_key)

            # Check if this edge has significant obstruction
            if artery_obstruction > partial_obstruction_thresh:
                # OBSTRUCTION FOUND - Apply stopping rule

                # Get obstruction degree
                degree = _get_obstruction_degree(artery_obstruction, artery_level)

                # Literature score: use fixed anatomical model
                literature_segments = _get_literature_segment_count(
                    artery_level, artery_lateral, artery_lobe
                )
                literature_score_contribution = literature_segments * degree
                literature_weighted_scores.append(literature_score_contribution)

                # Patient-specific score: use actual segments_below attribute
                patient_segments = edge_attrs.get("segments_below", 1)
                patient_score_contribution = patient_segments * degree
                patient_weighted_scores.append(patient_score_contribution)

                # Store detailed breakdown for analysis
                detailed_breakdown.append(
                    {
                        "edge": edge_key,
                        "level": artery_level,
                        "lobe": artery_lobe,
                        "lateral": artery_lateral,
                        "obstruction": artery_obstruction,
                        "degree": degree,
                        "lit_segments": literature_segments,
                        "patient_segments": patient_segments,
                        "lit_contribution": literature_score_contribution,
                        "patient_contribution": patient_score_contribution,
                    }
                )

                # Debug info
                if debug:
                    debug_info[edge_key] = (
                        f"L{artery_level}_{artery_lobe}: obstruction={artery_obstruction:.3f}, "
                        f"degree={degree}, lit_seg={literature_segments}, "
                        f"patient_seg={patient_segments}"
                    )

                logger.debug(
                    f"Obstruction at L{artery_level} {artery_lobe}: "
                    f"lit_score={literature_score_contribution}, "
                    f"patient_score={patient_score_contribution}"
                )

                # STOP TRAVERSAL here - don't explore children (Qanadli stopping rule)

            else:
                # No significant obstruction - continue traversal
                if artery_level <= terminal_level:
                    _depth_first_search(child, new_visited_pathway)

    # Find root and start DFS
    try:
        root_node = _find_root_node(graph)
        _depth_first_search(root_node, set())
    except ValueError as e:
        logger.error(f"Cannot compute Qanadli score: {e}")
        return 0.0, 0.0

    # Compute final scores
    if not literature_weighted_scores:
        logger.warning("No obstructions found for Qanadli score computation")
        return 0.0, 0.0

    # Literature score: sum of contributions / fixed maximum (20 segments × 2 max degree)
    total_literature_score = sum(literature_weighted_scores)
    max_literature_score = 20 * 2  # Fixed 40 points maximum
    qanadli_literature_score = total_literature_score / max_literature_score

    # Patient-specific score: sum of contributions / actual maximum
    total_patient_score = sum(patient_weighted_scores)

    # Get actual total segments from root edge segments_below attribute
    root_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) if d.get("level") == 1
    ]
    if root_edges:
        actual_total_segments = root_edges[0][2].get("segments_below", 20)
    else:
        actual_total_segments = (
            max(1, total_patient_score // 2) if patient_weighted_scores else 20
        )

    max_patient_score = actual_total_segments * 2
    qanadli_patient_score = (
        total_patient_score / max_patient_score if max_patient_score > 0 else 0.0
    )

    # Detailed logging for understanding the calculation
    logger.info("=" * 60)
    logger.info("DETAILED QANADLI CALCULATION BREAKDOWN")
    logger.info("=" * 60)

    for i, breakdown in enumerate(detailed_breakdown, 1):
        lobe_str = (
            breakdown["lobe"][0]
            if isinstance(breakdown["lobe"], list)
            else breakdown["lobe"]
        )
        logger.info(
            f"Obstruction {i}: Level {breakdown['level']} {lobe_str} "
            f"({breakdown['lateral']})"
        )
        logger.info(f"  Raw obstruction: {breakdown['obstruction']:.3f}")
        logger.info(f"  Degree: {breakdown['degree']} (partial=1, total=2)")
        logger.info(f"  Literature segments: {breakdown['lit_segments']}")
        logger.info(f"  Patient segments: {breakdown['patient_segments']}")
        logger.info(
            f"  Literature contribution: {breakdown['lit_segments']} × {breakdown['degree']} = {breakdown['lit_contribution']}"
        )
        logger.info(
            f"  Patient contribution: {breakdown['patient_segments']} × {breakdown['degree']} = {breakdown['patient_contribution']}"
        )
        logger.info("")

    logger.info(
        f"LITERATURE TOTAL: {' + '.join(str(b['lit_contribution']) for b in detailed_breakdown)} = {total_literature_score}"
    )
    logger.info(
        f"PATIENT TOTAL: {' + '.join(str(b['patient_contribution']) for b in detailed_breakdown)} = {total_patient_score}"
    )
    logger.info("")
    logger.info(
        f"Literature Score: {total_literature_score} / {max_literature_score} = {qanadli_literature_score:.4f} ({qanadli_literature_score*100:.1f}%)"
    )
    logger.info(
        f"Patient Score: {total_patient_score} / {max_patient_score} = {qanadli_patient_score:.4f} ({qanadli_patient_score*100:.1f}%)"
    )
    logger.info("=" * 60)

    # Calculate detailed statistics for CSV
    obstruction_values = [b["obstruction"] for b in detailed_breakdown]

    # Count obstructions by level
    level_counts = {2: 0, 3: 0, 4: 0, 5: 0}
    literature_level_contributions = {2: 0, 3: 0, 4: 0, 5: 0}
    patient_level_contributions = {2: 0, 3: 0, 4: 0, 5: 0}

    for breakdown in detailed_breakdown:
        level = breakdown["level"]
        if level in level_counts:
            level_counts[level] += 1
            literature_level_contributions[level] += breakdown["lit_contribution"]
            patient_level_contributions[level] += breakdown["patient_contribution"]

    # Calculate proximal vs distal ratio
    proximal_score = (
        literature_level_contributions[2] + literature_level_contributions[3]
    )
    distal_score = literature_level_contributions[4] + literature_level_contributions[5]
    proximal_vs_distal_ratio = (
        proximal_score / max(1, distal_score) if distal_score > 0 else proximal_score
    )

    # Enhanced statistics for CSV output
    stats = {
        "total_segments_patient": actual_total_segments,
        "obstructed_pathways": len(literature_weighted_scores),
        "partial_obstruction_pathways": len(
            [s for s in literature_weighted_scores if s % 2 != 0]
        ),
        "total_obstruction_pathways": len(
            [s for s in literature_weighted_scores if s % 2 == 0 and s > 0]
        ),
        "literature_weighted_score": total_literature_score,
        "literature_max_possible_score": max_literature_score,
        "patient_weighted_score": total_patient_score,
        "patient_max_possible_score": max_patient_score,
        # Level breakdown
        "level_2_obstructions": level_counts[2],
        "level_3_obstructions": level_counts[3],
        "level_4_obstructions": level_counts[4],
        "level_5_obstructions": level_counts[5],
        # Literature contributions by level
        "literature_level_2_contribution": literature_level_contributions[2],
        "literature_level_3_contribution": literature_level_contributions[3],
        "literature_level_4_contribution": literature_level_contributions[4],
        "literature_level_5_contribution": literature_level_contributions[5],
        # Patient contributions by level
        "patient_level_2_contribution": patient_level_contributions[2],
        "patient_level_3_contribution": patient_level_contributions[3],
        "patient_level_4_contribution": patient_level_contributions[4],
        "patient_level_5_contribution": patient_level_contributions[5],
        # Severity indicators
        "mean_obstruction_value": np.mean(obstruction_values)
        if obstruction_values
        else 0.0,
        "max_obstruction_value": max(obstruction_values) if obstruction_values else 0.0,
        "proximal_vs_distal_ratio": proximal_vs_distal_ratio,
    }

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

    return qanadli_literature_score, qanadli_patient_score
