"""
Module for cleaning and optimizing pulmonary vascular tree graphs.
This module provides functionality to clean and optimize graph representations
of pulmonary vascular trees by removing suspicious cycles, short branches, and fusing segments.
"""

import csv
import logging
from pathlib import Path

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# STATISTICS TRACKING FUNCTIONS
# =============================================================================


def _get_detailed_stats(graph: nx.Graph) -> dict:
    """Calculate detailed graph statistics for comprehensive monitoring."""
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "terminals": 0,  # degree 1
        "bifurcation": 0,  # degree 3+
        "linear": 0,  # degree 2
        "isolated": 0,  # degree 0
        "cycles": len(nx.cycle_basis(graph)) if graph.number_of_edges() > 0 else 0,
        "components": nx.number_connected_components(graph),
    }

    for node in graph.nodes:
        degree = graph.degree(node)
        if degree == 0:
            stats["isolated"] += 1
        elif degree == 1:
            stats["terminals"] += 1
        elif degree == 2:
            stats["linear"] += 1
        else:  # degree >= 3
            stats["bifurcation"] += 1

    return stats


def _initialize_stats_csv(output_path: Path) -> None:
    """Create CSV file with header for cleaning statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "patient_id",
                "step",
                "nodes",
                "edges",
                "terminals",
                "bifurcation",
                "linear",
                "isolated",
                "cycles",
                "nodes_removed",
                "edges_removed",
            ]
        )


def _append_stats_to_csv(
    output_path: Path,
    patient_id: str,
    step: str,
    stats: dict,
    previous_stats: dict | None = None,
) -> None:
    """Append statistics row to CSV file."""
    nodes_removed = 0
    edges_removed = 0
    if previous_stats:
        nodes_removed = previous_stats["nodes"] - stats["nodes"]
        edges_removed = previous_stats["edges"] - stats["edges"]

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                patient_id,
                step,
                stats["nodes"],
                stats["edges"],
                stats["terminals"],
                stats["bifurcation"],
                stats["linear"],
                stats["isolated"],
                stats["cycles"],
                nodes_removed,
                edges_removed,
            ]
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def remove_isolated_nodes(graph: nx.Graph) -> None:
    """
    Remove nodes that have no connections in the graph.
    in-place.

    Args:
        graph: NetworkX graph to clean (modified in-place)
    """
    graph.remove_nodes_from(list(nx.isolates(graph)))


def get_cycle_edges(graph: nx.Graph) -> list[list[tuple[int, int]]]:
    """
    Detect and return all cycles in the graph as lists of edges.

    This function finds all fundamental cycles in the graph using NetworkX's
    cycle_basis algorithm and converts each cycle from a list of nodes to
    a list of edges (node pairs). Each cycle is represented as a closed
    path of edges.

    Args:
        graph: NetworkX graph to analyze

    Returns:
        List of cycles, where each cycle is a list of edges (tuples of node pairs).
        Empty list if no cycles are found.
    """
    cycles = nx.cycle_basis(graph)
    return [
        [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        for cycle in cycles
    ]


def detect_radius_pattern(
    radius_list: list[float], threshold_factor: float = 0.5
) -> bool:
    """
    Detect suspicious radius patterns indicating false vascular connections.

    This function identifies vessel segments showing a "constriction pattern":
    a significant decrease followed by an increase in radius. Such patterns
    are characteristic of false connections between closely spaced vessels,
    where the skeleton algorithm incorrectly bridges two separate structures.

    Args:
        radius_list: Sequence of radius values along the vessel segment.
                    Can be list, tuple, or numpy array (handles nested structures)
        threshold_factor: Minimum ratio for both decrease and increase phases
                         to be considered significant (default: 0.5)

    Returns:
        bool: True if a constriction pattern is detected, False otherwise

    Algorithm:
        1. Find minimum radius and its position
        2. If minimum is at endpoints, return False (normal tapering)
        3. Calculate decrease ratio: (max_before - min) / max_before
        4. Calculate increase ratio: (max_after - min) / min
        5. Return True if both ratios exceed threshold

    Note:
        Robust to various input formats (numpy arrays, nested lists, scalars).
        Returns False for invalid/insufficient data or processing errors.
    """
    try:
        # Debug: Print the input type and structure
        logger.debug("Input radius_list type: %s", type(radius_list))
        logger.debug("Input radius_list: %s", radius_list)

        # Fix: Check for empty array properly
        if radius_list is None:
            return False

        # Handle numpy arrays specifically
        if isinstance(radius_list, np.ndarray):
            if radius_list.size == 0:
                return False
            radius_array = radius_list.copy()
        # Handle lists and tuples
        elif isinstance(radius_list, list | tuple):
            if len(radius_list) == 0:
                return False
            if len(radius_list) < 3:
                return False
            # Check if it's a nested structure
            if len(radius_list) > 0 and isinstance(
                radius_list[0], list | tuple | np.ndarray
            ):
                # Flatten nested structure
                flattened = []
                for item in radius_list:
                    if isinstance(item, list | tuple):
                        flattened.extend(item)
                    elif isinstance(item, np.ndarray):
                        flattened.extend(item.flatten())
                    else:
                        flattened.append(item)
                radius_array = np.asarray(flattened)
            else:
                radius_array = np.asarray(radius_list)
        else:
            try:
                radius_array = np.asarray(radius_list)
                if radius_array.size == 0:
                    return False
            except (ValueError, TypeError):
                logger.debug("Could not convert radius_list to numpy array")
                return False

        # Ensure it's 1D and has enough elements
        if radius_array.ndim != 1:
            radius_array = radius_array.flatten()

        if radius_array.size < 3:
            logger.debug("Radius array too small: %d elements", radius_array.size)
            return False

        # Check if we have valid numeric data
        if not np.issubdtype(radius_array.dtype, np.number):
            logger.debug("Radius array is not numeric: %s", radius_array.dtype)
            return False

        # Check for NaN or infinite values
        if not np.isfinite(radius_array).all():
            logger.debug("Radius array contains NaN or infinite values")
            return False

        # Calculate min and its index
        min_value = np.min(radius_array)
        min_idx = np.argmin(radius_array)

        # Ensure min_idx is a scalar integer
        if hasattr(min_idx, "item"):
            min_idx = min_idx.item()
        min_idx = int(min_idx)

        # Check if minimum is at the endpoints (normal case)
        if min_idx == 0 or min_idx == len(radius_array) - 1:
            logger.debug(
                "Minimum radius at endpoint (index %d of %d) - not a constriction pattern",
                min_idx,
                len(radius_array),
            )
            return False

        # Calculate ratios
        max_before = np.max(radius_array[:min_idx])
        max_after = np.max(radius_array[min_idx:])

        # Ensure these are scalars
        if hasattr(max_before, "item"):
            max_before = max_before.item()
        if hasattr(max_after, "item"):
            max_after = max_after.item()
        if hasattr(min_value, "item"):
            min_value = min_value.item()

        # Avoid division by zero
        if max_before <= 0 or min_value <= 0:
            logger.debug(
                "Invalid radius values: max_before=%.3f, min_value=%.3f",
                max_before,
                min_value,
            )
            return False

        decrease_ratio = (max_before - min_value) / max_before
        increase_ratio = (max_after - min_value) / min_value

        # FIXED: Use >= instead of > to include boundary cases
        result = (
            decrease_ratio >= threshold_factor and increase_ratio >= threshold_factor
        )

        logger.info(
            "Pattern analysis: min_idx=%d/%d, pattern: %.3f → %.3f → %.3f",
            min_idx,
            len(radius_array),
            max_before,
            min_value,
            max_after,
        )
        logger.info(
            "Ratios: decrease=%.3f (threshold=%.3f), increase=%.3f (threshold=%.3f) → %s",
            decrease_ratio,
            threshold_factor,
            increase_ratio,
            threshold_factor,
            "DETECTED" if result else "NOT DETECTED",
        )

        return result

    except Exception as e:
        logger.error("Error in detect_radius_pattern: %s", e)
        logger.error("Input was: %s (type: %s)", radius_list, type(radius_list))
        import traceback

        logger.error("Full traceback: %s", traceback.format_exc())
        return False


def get_edge_data(graph: nx.Graph, node1: int, node2: int) -> dict | None:
    """
    Safely retrieve edge data regardless of edge orientation.

    NetworkX edges can be accessed in either direction (node1->node2 or
    node2->node1). This function attempts both orientations and returns
    the edge data if the edge exists, or None if not found.

    Args:
        graph: NetworkX graph containing the edge
        node1: First node of the edge
        node2: Second node of the edge

    Returns:
        dict: Edge attributes dictionary if edge exists, None otherwise
    """
    if graph.has_edge(node1, node2):
        return graph.edges[node1, node2]
    elif graph.has_edge(node2, node1):
        return graph.edges[node2, node1]
    return None


# =============================================================================
# FUSION HELPER FUNCTIONS
# =============================================================================


def _is_fusion_candidate(
    graph: nx.Graph, node: int, processed_nodes: set
) -> tuple[bool, list[int]]:
    """
    Determine if a node is eligible for linear segment fusion.

    A node is eligible for fusion if it:
    - Has not been processed yet
    - Has exactly 2 neighbors (is on a linear path)
    - Both edges to neighbors exist

    Args:
        graph: NetworkX graph being processed
        node: Node to evaluate for fusion eligibility
        processed_nodes: Set of already processed nodes to skip

    Returns:
        tuple: (is_eligible: bool, neighbors: list[int])
               neighbors list is empty if not eligible
    """
    if node in processed_nodes or not graph.has_node(node):
        return False, []

    if graph.degree(node) != 2:
        return False, []

    neighbors = list(graph.neighbors(node))
    if len(neighbors) != 2:
        return False, []

    # Verify edges exist
    if not (graph.has_edge(node, neighbors[0]) and graph.has_edge(node, neighbors[1])):
        return False, []

    return True, neighbors


def _combine_edge_segments(
    edge1_centerline: np.ndarray,
    edge1_radius: np.ndarray,
    edge2_centerline: np.ndarray,
    edge2_radius: np.ndarray,
) -> tuple[list, list]:
    """
    Combine two oriented edge segments into a single continuous segment.

    This function merges two consecutive edge segments by concatenating their
    geometric properties (centerlines and radii). It removes duplicate points
    at the connection to ensure continuity. Assumes input segments are properly
    oriented (edge1 ends where edge2 starts).

    Args:
        edge1_centerline: 3D coordinates of first segment
        edge1_radius: Radius values along first segment
        edge2_centerline: 3D coordinates of second segment
        edge2_radius: Radius values along second segment

    Returns:
        tuple: (combined_centerline: list, combined_radius: list)
               Empty centerline list if input data is insufficient

    Algorithm:
        1. Convert inputs to numpy arrays for processing
        2. If both centerlines have data:
           - Remove duplicate middle point (edge1[:-1] + edge2)
           - Concatenate radius arrays
        3. If no centerline data: concatenate only radius arrays
    """

    # Ensure inputs are numpy arrays
    edge1_centerline = np.asarray(edge1_centerline)
    edge2_centerline = np.asarray(edge2_centerline)
    edge1_radius = np.asarray(edge1_radius)
    edge2_radius = np.asarray(edge2_radius)

    # Check if both centerlines have data using size comparison (not direct array comparison)
    edge1_has_data = edge1_centerline.size > 0
    edge2_has_data = edge2_centerline.size > 0

    if edge1_has_data and edge2_has_data:
        # Remove duplicate middle point and concatenate
        try:
            combined_centerline = np.concatenate(
                [edge1_centerline[:-1], edge2_centerline]
            )
            combined_radius = np.concatenate([edge1_radius, edge2_radius])
            return combined_centerline.tolist(), combined_radius.tolist()
        except (ValueError, IndexError) as e:
            logger.warning("Error combining centerlines: %s", e)
            # Fallback: just concatenate radius arrays
            radius1 = edge1_radius.tolist() if edge1_radius.size > 0 else []
            radius2 = edge2_radius.tolist() if edge2_radius.size > 0 else []
            return [], radius1 + radius2
    else:
        # Handle case with no centerline data
        radius1 = edge1_radius.tolist() if edge1_radius.size > 0 else []
        radius2 = edge2_radius.tolist() if edge2_radius.size > 0 else []
        return [], radius1 + radius2


def _create_fused_edge_data(
    edge1_data: dict,
    edge2_data: dict,
    combined_centerline: list,
    combined_radius: list,
    edge_id: int,
) -> dict:
    """
    Create comprehensive edge attributes for a fused segment.

    This function combines properties from two original edges into a new
    edge dictionary with updated geometric properties and statistics.

    Args:
        edge1_data: Attributes dictionary from first edge
        edge2_data: Attributes dictionary from second edge
        combined_centerline: Merged 3D coordinate sequence
        combined_radius: Merged radius sequence
        edge_id: Unique identifier for the new fused edge

    Returns:
        dict: Complete edge attributes dictionary with:
              - length: Sum of original lengths
              - mean_radius: Average of combined radius values
              - min/max_radius: Extrema of combined radii
              - centerline, radius: Combined geometric data
              - id, name: New identifiers
              - Other attributes copied from first edge
    """
    combined_length = edge1_data.get("length", 0) + edge2_data.get("length", 0)
    combined_mean_radius = np.mean(combined_radius) if combined_radius else 0
    min_radius = min(combined_radius) if combined_radius else 0
    max_radius = max(combined_radius) if combined_radius else 0

    new_edge_data = {
        "length": combined_length,
        "mean_radius": combined_mean_radius,
        "radius": combined_radius,
        "centerline": combined_centerline,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "id": edge_id,
        "name": f"edge_{edge_id}",
    }

    # Copy other attributes from first edge
    for key, value in edge1_data.items():
        if key not in new_edge_data:
            new_edge_data[key] = value

    return new_edge_data


def _fuse_two_segments(
    graph: nx.Graph, middle_node: int, neighbor1: int, neighbor2: int, edge_id: int
) -> tuple[dict, bool]:
    """
    Execute the fusion of two segments connected by a middle node.

    This function handles the complex process of merging two edges that
    share a common middle node. It properly orients the segments, combines
    their geometric properties, and creates new edge data.

    Args:
        graph: NetworkX graph containing the segments
        middle_node: Intermediate node to be removed during fusion
        neighbor1: First endpoint of the fused segment
        neighbor2: Second endpoint of the fused segment
        edge_id: Unique identifier for the new fused edge

    Returns:
        tuple: (new_edge_data: dict, success: bool)
               success is always True if function completes

    Algorithm:
        1. Extract edge data and node positions
        2. Determine proper orientation for each segment:
           - Segment 1: neighbor1 → middle_node
           - Segment 2: middle_node → neighbor2
        3. Orient segments by reversing if necessary
        4. Combine oriented segments using _combine_edge_segments
        5. Create comprehensive new edge data
    """
    # Get edge data and positions
    edge1_data = graph.edges[middle_node, neighbor1]
    edge2_data = graph.edges[middle_node, neighbor2]

    # Get positions to determine direction
    middle_pos = graph.nodes[middle_node]["pos"]
    neighbor1_pos = graph.nodes[neighbor1]["pos"]
    neighbor2_pos = graph.nodes[neighbor2]["pos"]

    # Get centerlines and radius - ensure they are lists/arrays
    edge1_centerline = edge1_data.get("centerline", [])
    edge2_centerline = edge2_data.get("centerline", [])
    edge1_radius = edge1_data.get("radius", [])
    edge2_radius = edge2_data.get("radius", [])

    # Convert to numpy arrays and check if they contain data
    edge1_centerline = np.asarray(edge1_centerline)
    edge2_centerline = np.asarray(edge2_centerline)
    edge1_radius = np.asarray(edge1_radius)
    edge2_radius = np.asarray(edge2_radius)

    if edge1_centerline.size > 0 and edge2_centerline.size > 0:
        # Convert positions to numpy arrays for manipulation
        middle_pos = np.array(middle_pos)
        neighbor1_pos = np.array(neighbor1_pos)
        neighbor2_pos = np.array(neighbor2_pos)

        # Orient edge1: should go neighbor1 -> middle
        # Check each condition separately to avoid ambiguous truth values
        start_matches = np.array_equal(edge1_centerline[0], neighbor1_pos)
        end_matches = np.array_equal(edge1_centerline[-1], middle_pos)

        if start_matches and end_matches:
            oriented_edge1_centerline = edge1_centerline
            oriented_edge1_radius = edge1_radius
        else:
            # Reverse it
            oriented_edge1_centerline = edge1_centerline[::-1]
            oriented_edge1_radius = edge1_radius[::-1]

        # Orient edge2: should go middle -> neighbor2
        # Check each condition separately to avoid ambiguous truth values
        start_matches = np.array_equal(edge2_centerline[0], middle_pos)
        end_matches = np.array_equal(edge2_centerline[-1], neighbor2_pos)

        if start_matches and end_matches:
            oriented_edge2_centerline = edge2_centerline
            oriented_edge2_radius = edge2_radius
        else:
            # Reverse it
            oriented_edge2_centerline = edge2_centerline[::-1]
            oriented_edge2_radius = edge2_radius[::-1]

        # Combine the oriented segments
        combined_centerline, combined_radius = _combine_edge_segments(
            oriented_edge1_centerline,
            oriented_edge1_radius,
            oriented_edge2_centerline,
            oriented_edge2_radius,
        )
    else:
        # Handle case with no centerline data
        combined_centerline = []
        combined_radius = edge1_radius.tolist() + edge2_radius.tolist()

    # Create new edge data
    new_edge_data = _create_fused_edge_data(
        edge1_data, edge2_data, combined_centerline, combined_radius, edge_id
    )

    return new_edge_data, True


def fuse_linear_segments(graph: nx.Graph) -> nx.Graph:
    """
    Simplify graph by fusing consecutive linear segments into single edges.

    This function identifies nodes with exactly 2 neighbors (intermediate points
    on linear paths) and merges the adjacent edges into single edges. This
    reduces graph complexity while preserving all geometric information.

    Args:
        graph: NetworkX graph to simplify (not modified)

    Returns:
        nx.Graph: New graph with linear segments fused

    Algorithm:
        1. Identify all degree-2 nodes (fusion candidates)
        2. For each candidate:
           - Verify eligibility and get 2 neighbors
           - Fuse the two connecting edges
           - Remove middle node, add new edge between neighbors
        3. Track processed nodes to avoid conflicts
        4. Assign sequential IDs to new fused edges
    """
    graph = graph.copy()
    fused_count = 0
    next_id = graph.number_of_edges() + 1
    processed_nodes = set()

    for node in list(graph.nodes):
        # Check if this node is a valid fusion candidate
        is_valid, neighbors = _is_fusion_candidate(graph, node, processed_nodes)
        if not is_valid or len(neighbors) != 2:
            processed_nodes.add(node)
            continue

        processed_nodes.add(node)
        # Safely unpack neighbors after confirming it has exactly 2 elements
        neighbor1, neighbor2 = neighbors[0], neighbors[1]

        try:
            # Fuse the two segments
            new_edge_data, success = _fuse_two_segments(
                graph, node, neighbor1, neighbor2, next_id
            )

            if success:
                # Apply the fusion: remove middle node and add new edge
                graph.remove_node(node)

                if not graph.has_edge(neighbor1, neighbor2):
                    graph.add_edge(neighbor1, neighbor2, **new_edge_data)
                    fused_count += 1
                    next_id += 1

        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to fuse segments at node %s: %s", node, e)
            continue

    if fused_count > 0:
        logger.info("Fused %d linear segments", fused_count)

    return graph


# =============================================================================
# BRANCH REMOVAL HELPER FUNCTIONS
# =============================================================================


def _should_remove_branch(
    _node: int,
    _parent: int,
    edge_data: dict,
    parent_radius: float,
    end_radius: float,
    radius_ratio_threshold: float,
    length_ratio_threshold: float,
) -> bool:
    """
    Apply morphological criteria to determine if a terminal branch is artifactual.

    This function evaluates terminal branches against anatomical plausibility
    criteria. Branches that are very short or both thin and relatively short
    are considered artifacts from segmentation or skeletonization.

    Args:
        _node: Terminal node (unused, kept for interface consistency)
        _parent: Parent node (unused, kept for interface consistency)
        edge_data: Dictionary containing edge properties (length, etc.)
        parent_radius: Radius at the parent node
        end_radius: Radius at the terminal node
        radius_ratio_threshold: Minimum ratio end_radius/parent_radius to keep. default 0.3
        length_ratio_threshold: Length threshold relative to parent diameter. default 1.5

    Returns:
        bool: True if branch should be removed, False if it should be kept

    Criteria:
        1. Very short: length < parent_diameter → remove
        2. Thin and relatively short:
           length < parent_diameter * length_ratio AND
           radius_ratio < radius_ratio_threshold → remove
    """
    length = edge_data.get("length", 0)

    # Ensure parent_radius is a scalar
    if hasattr(parent_radius, "item"):
        parent_radius = parent_radius.item()
    if hasattr(end_radius, "item"):
        end_radius = end_radius.item()

    parent_diameter = 2 * float(parent_radius)
    radius_ratio = float(end_radius) / float(parent_radius) if parent_radius > 0 else 0

    # Case 1: Very short branch (much shorter than parent diameter)
    if length < parent_diameter:
        logger.debug(
            "Removing very short branch (ID %s): length=%.2f < diameter=%.2f",
            edge_data.get("id", "unknown"),
            length,
            parent_diameter,
        )
        return True

    # Case 2: Slightly longer branch but with very thin end
    elif (
        length < parent_diameter * length_ratio_threshold
        and radius_ratio < radius_ratio_threshold
    ):
        logger.debug(
            "Removing thin long branch (ID %s): length=%.2f, ratio=%.3f",
            edge_data.get("id", "unknown"),
            length,
            radius_ratio,
        )
        return True

    return False


def _find_branches_to_remove(
    graph: nx.Graph, radius_ratio_threshold: float, length_ratio_threshold: float
) -> list[tuple[int, int, str]]:
    """
    Identify all terminal branches that meet removal criteria.

    This function efficiently scans all terminal nodes (degree 1) and
    evaluates their connecting edges against morphological criteria for
    artifact removal.

    Args:
        graph: NetworkX graph to analyze
        radius_ratio_threshold: Minimum end/parent radius ratio to preserve
        length_ratio_threshold: Length threshold factor relative to parent diameter

    Returns:
        list: Tuples of (terminal_node, parent_node, edge_id) for removal
              Empty list if no branches meet removal criteria

    Algorithm:
        1. Pre-filter nodes to terminal nodes only (degree == 1)
        2. For each terminal node:
           - Get parent node and edge data
           - Extract radius values (ensure scalar conversion)
           - Apply _should_remove_branch criteria
           - Add to removal list if criteria met
        3. Return complete removal list for batch processing

    Optimization:
        Pre-filtering terminal nodes improves performance on large graphs
        compared to checking all nodes.
    """
    to_remove = []

    # Pre-filter terminal nodes for better performance
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    for node in terminal_nodes:
        parent = next(graph.neighbors(node))
        edge_data = graph.edges[node, parent]

        # Safely extract radius values
        parent_radius = graph.nodes[parent]["radius"]
        end_radius = graph.nodes[node]["radius"]

        # Ensure radius values are scalars
        if hasattr(parent_radius, "item"):
            parent_radius = parent_radius.item()
        if hasattr(end_radius, "item"):
            end_radius = end_radius.item()

        if _should_remove_branch(
            node,
            parent,
            edge_data,
            parent_radius,
            end_radius,
            radius_ratio_threshold,
            length_ratio_threshold,
        ):
            to_remove.append((node, parent, edge_data.get("id", "unknown")))

    return to_remove


def remove_single_neighbor_nodes(graph: nx.Graph) -> nx.Graph:
    """
    Iteratively remove terminal nodes connected by very short edges.

    This function removes terminal artifacts: nodes with single neighbors
    connected by edges shorter than 1.0 mm. These typically represent
    noise or pixel-level artifacts that survived earlier cleaning steps.

    Args:
        graph: NetworkX graph to clean (not modified)

    Returns:
        nx.Graph: New graph with short terminal segments removed

    Algorithm:
        1. While changes occur:
           - Find all degree-1 nodes with edge length < 1.0 mm
           - Remove all such nodes in batch
           - Update removal count
        2. Continue until no more nodes meet criteria
        3. Log total removals for monitoring

    Iterative Design:
        Removal of nodes may create new terminal nodes, so iteration
        continues until convergence (no new nodes to remove).
    """
    graph = graph.copy()
    removed_count = 0

    while True:
        nodes_to_remove = []
        for node in graph.nodes:
            if graph.degree(node) == 1:
                neighbor = next(graph.neighbors(node))
                edge_length = graph.edges[node, neighbor].get("length", 0)
                if edge_length < 1.0:
                    nodes_to_remove.append(node)

        if not nodes_to_remove:
            break

        graph.remove_nodes_from(nodes_to_remove)
        removed_count += len(nodes_to_remove)

    if removed_count > 0:
        logger.info("Removed %d single-neighbor nodes", removed_count)

    return graph


# =============================================================================
# MAIN CLEANING FUNCTIONS
# =============================================================================


def _get_radius_pattern_ratios(
    radius_list: list[float], threshold_factor: float = 0.5
) -> tuple[float, float] | None:
    """
    Calculate decrease and increase ratios for radius pattern analysis.

    Returns the actual ratios if suspicious pattern detected, None otherwise.
    This is a helper function for _remove_suspicious_cycle_edges.

    Args:
        radius_list: Sequence of radius values along the vessel segment
        threshold_factor: Base threshold for ratio evaluation

    Returns:
        tuple: (decrease_ratio, increase_ratio) if suspicious, None otherwise

    Simple two-threshold approach:
        - PASS 1 (threshold = 0.5): Both ratios >= 0.5
        - PASS 2 (threshold = 0.4): Both ratios >= 0.4
    """
    try:
        # Reuse existing logic from detect_radius_pattern but return ratios
        if radius_list is None:
            logger.debug("Radius list is None")
            return None

        # Handle numpy arrays specifically
        if isinstance(radius_list, np.ndarray):
            if radius_list.size == 0:
                return None
            radius_array = radius_list.copy()
        # Handle lists and tuples
        elif isinstance(radius_list, list | tuple):
            if len(radius_list) == 0:
                return None
            if len(radius_list) < 3:
                return None
            # Check if it's a nested structure
            if len(radius_list) > 0 and isinstance(
                radius_list[0], list | tuple | np.ndarray
            ):
                # Flatten nested structure
                flattened = []
                for item in radius_list:
                    if isinstance(item, list | tuple):
                        flattened.extend(item)
                    elif isinstance(item, np.ndarray):
                        flattened.extend(item.flatten())
                    else:
                        flattened.append(item)
                radius_array = np.asarray(flattened)
            else:
                radius_array = np.asarray(radius_list)
        else:
            try:
                radius_array = np.asarray(radius_list)
                if radius_array.size == 0:
                    return None
            except (ValueError, TypeError):
                return None

        # Ensure it's 1D and has enough elements
        if radius_array.ndim != 1:
            radius_array = radius_array.flatten()

        if radius_array.size < 3:
            logger.debug("Insufficient data points (found %d)", radius_array.size)
            return None

        # Check if we have valid numeric data
        if not np.issubdtype(radius_array.dtype, np.number):
            return None

        # Check for NaN or infinite values
        if not np.isfinite(radius_array).all():
            logger.debug("Found NaN or infinite values")
            return None

        # Calculate min and its index
        min_value = np.min(radius_array)
        min_idx = np.argmin(radius_array)

        # Ensure min_idx is a scalar integer
        if hasattr(min_idx, "item"):
            min_idx = min_idx.item()
        min_idx = int(min_idx)

        # Check if minimum is at the endpoints (normal case)
        if min_idx == 0 or min_idx == len(radius_array) - 1:
            logger.debug(
                "Minimum at endpoint (index %d of %d) - skipping (normal tapering)",
                min_idx,
                len(radius_array),
            )
            return None

        # Calculate ratios
        max_before = np.max(radius_array[:min_idx])
        max_after = np.max(radius_array[min_idx:])

        # Ensure these are scalars
        if hasattr(max_before, "item"):
            max_before = max_before.item()
        if hasattr(max_after, "item"):
            max_after = max_after.item()
        if hasattr(min_value, "item"):
            min_value = min_value.item()

        # Avoid division by zero
        if max_before <= 0 or min_value <= 0:
            logger.debug(
                "Invalid radius values: max_before=%.3f, min_value=%.3f",
                max_before,
                min_value,
            )
            return None

        decrease_ratio = (max_before - min_value) / max_before
        increase_ratio = (max_after - min_value) / min_value

        logger.debug(
            "Calculated ratios: decrease=%.3f, increase=%.3f",
            decrease_ratio,
            increase_ratio,
        )

        # SIMPLE TWO-PASS LOGIC
        if decrease_ratio >= threshold_factor and increase_ratio >= threshold_factor:
            logger.debug("Detection: both ratios >= %.3f", threshold_factor)
            return decrease_ratio, increase_ratio
        else:
            logger.debug(
                "NOT detected: decrease %.3f, increase %.3f (threshold %.3f)",
                decrease_ratio,
                increase_ratio,
                threshold_factor,
            )

        return None

    except Exception as e:
        logger.error("Error in _get_radius_pattern_ratios: %s", e)
        return None


def _detect_fusion_artifacts(edge_data: dict) -> tuple[float, str] | None:
    """
    Detect fusion artifacts using geometric criteria.

    Returns inverted ratio (for sorting) and reason if edge is suspicious, None otherwise.
    For fusion detection, we want the SMALLEST length-to-diameter ratio to win.
    """
    try:
        length = edge_data.get("length", 0)
        mean_radius = edge_data.get("mean_radius", 0)

        # Ensure scalar values
        if hasattr(length, "item"):
            length = length.item()
        if hasattr(mean_radius, "item"):
            mean_radius = mean_radius.item()

        if float(length) <= 0 or float(mean_radius) <= 0:
            return None

        # Very short branch (much shorter than parent diameter)
        diameter = 2 * float(mean_radius)

        logger.debug(
            "length=%.2f, diameter=%.2f, length/diameter=%.2f",
            length,
            diameter,
            length / diameter,
        )

        if length < diameter:
            logger.debug(
                "Possible fusion artifact detected (ID %s): length=%.2f < diameter=%.2f",
                edge_data.get("id", "unknown"),
                length,
                diameter,
            )
            length_to_diameter_ratio = length / diameter
            inverted_priority = (
                1.0 / length_to_diameter_ratio
            )  # Small ratio = high priority
            reason = f"short_fusion_ratio_{length_to_diameter_ratio:.3f}"

            return length_to_diameter_ratio, inverted_priority, reason
        else:
            return None

    except Exception as e:
        logger.error("Error in _detect_fusion_artifacts: %s", e)
        return None


def _remove_suspicious_cycle_edges(
    graph: nx.Graph,
) -> tuple[int, list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]]:
    """
    Detect and remove edges in cycles that show suspicious radius patterns.

    Uses a two-pass approach:
    1. First pass: Normal threshold (0.5), re-detects cycles after each modification
    2. Second pass: Relaxed threshold (0.4) for remaining cycles without removals

    Returns:
        tuple: (num_removed: int, cycle_tracking: list)
               cycle_tracking contains tuples of (original_cycle, remaining_edges)
               for each cycle that had edges removed

    Algorithm per pass:
        1. Detect cycles
        2. Inspect first cycle, if edge removed → restart iteration
        3. If no edge removed from first cycle → inspect second cycle
        4. Continue until all cycles inspected OR an edge is removed
        5. If edge removed → restart iteration (re-detect cycles)
        6. If no edges removed in complete iteration → pass complete
    """
    removed_edges = []
    cycle_tracking = []

    try:
        # STEP 0: Remove self-loops first (always artifacts)
        logger.info("=== STEP 0: Removing self-loops ===")
        self_loops = []

        for node in list(graph.nodes):
            if graph.has_edge(node, node):
                edge_data = graph.edges[node, node]
                edge_id = edge_data.get("id", "unknown")
                self_loops.append((node, edge_id))

                logger.info("Found self-loop: Node %d [ID: %s] → itself", node, edge_id)

                # Remove the self-loop
                graph.remove_edge(node, node)
                removed_edges.append(((node, node), edge_id))

                logger.info("  → Removed self-loop: Node %d [ID: %s]", node, edge_id)

        if self_loops:
            logger.info(
                "Removed %d self-loops with IDs: %s",
                len(self_loops),
                [edge_id for _, edge_id in self_loops],
            )
        else:
            logger.info("No self-loops found")

        # Two-pass approach: radius patterns (0.5), then relaxed radius (0.4)
        pass_configs = [
            {"type": "radius", "threshold": 0.5, "name": "radius pattern (strict)"},
            {"type": "radius", "threshold": 0.4, "name": "radius pattern (moderate)"},
        ]

        for pass_number, config in enumerate(pass_configs, 1):
            pass_threshold = config["threshold"]
            pass_name = config["name"]

            logger.info("=== PASS %d: Using %s ===", pass_number, pass_name)

            pass_removed_count = 0
            iteration = 1
            max_iterations = 50  # Safety limit per pass

            while iteration <= max_iterations:
                try:
                    logger.info(
                        "Pass %d, Iteration %d: Getting cycle edges",
                        pass_number,
                        iteration,
                    )
                    cycles = get_cycle_edges(graph)

                    if not cycles:
                        logger.info("No cycles found in iteration %d", iteration)
                        break

                    logger.info(
                        "Found %d cycles in iteration %d", len(cycles), iteration
                    )

                    # Process cycles one by one until all inspected OR edge removed
                    edge_removed_this_iteration = False

                    for cycle_index, cycle in enumerate(cycles):
                        logger.info(
                            "Processing cycle %d/%d with %d edges",
                            cycle_index + 1,
                            len(cycles),
                            len(cycle),
                        )

                        # Extract cycle information with positions
                        cycle_nodes = set()
                        for edge in cycle:
                            cycle_nodes.add(edge[0])
                            cycle_nodes.add(edge[1])

                        logger.info("Cycle %d detected (undirected)", cycle_index + 1)
                        logger.info(
                            "Cycle %d has %d edges involving %d nodes",
                            cycle_index + 1,
                            len(cycle),
                            len(cycle_nodes),
                        )

                        # SKIP cycles with self-loops (should be eliminated in Step 0)
                        has_self_loop = any(edge[0] == edge[1] for edge in cycle)
                        if has_self_loop:
                            logger.warning(
                                "Cycle %d contains self-loop - skipping",
                                cycle_index + 1,
                            )
                            continue

                        # Log cycle edges with positions
                        for u, v in cycle:
                            pos_u = graph.nodes[u].get("pos", "unknown")
                            pos_v = graph.nodes[v].get("pos", "unknown")
                            edge_data = get_edge_data(graph, u, v)
                            edge_id = (
                                edge_data.get("id", "unknown")
                                if edge_data
                                else "unknown"
                            )

                            edge_idx = cycle.index((u, v)) + 1
                            logger.info(
                                "  Cycle %d edge %d [ID: %s]: Node %d @%s ↔ Node %d @%s",
                                cycle_index + 1,
                                edge_idx,
                                edge_id,
                                u,
                                pos_u,
                                v,
                                pos_v,
                            )

                        # Summary of all nodes involved
                        logger.info(
                            "All nodes in cycle %d: %s",
                            cycle_index + 1,
                            sorted(cycle_nodes),
                        )

                        # Track original cycle for potential tracking
                        original_cycle = cycle.copy()

                        # Collect all suspicious edges with their ratios for THIS cycle
                        suspicious_edges_with_ratios = []

                        # Check each edge in this cycle for suspicion and collect ratios
                        for edge in cycle:
                            node1, node2 = edge

                            # Skip self-loops (should not exist at this point)
                            if node1 == node2:
                                logger.warning(
                                    "Self-loop found in cycle analysis: (%d, %d) - skipping",
                                    node1,
                                    node2,
                                )
                                continue

                            pos1 = graph.nodes[node1].get("pos", "unknown")
                            pos2 = graph.nodes[node2].get("pos", "unknown")

                            try:
                                edge_data = get_edge_data(graph, *edge)
                                if edge_data:
                                    edge_id = edge_data.get("id", "unknown")
                                    logger.info(
                                        "Checking edge (%d, %d) [ID: %s]: positions %s ↔ %s",
                                        node1,
                                        node2,
                                        edge_id,
                                        pos1,
                                        pos2,
                                    )

                                    # Radius pattern detection only
                                    radius_data = edge_data.get("radius", [])
                                    ratios = _get_radius_pattern_ratios(
                                        radius_data, threshold_factor=pass_threshold
                                    )

                                    if ratios is not None:  # Edge is suspicious
                                        decrease_ratio, increase_ratio = ratios
                                        combined_ratio = decrease_ratio + increase_ratio

                                        detection_info = {
                                            "decrease_ratio": decrease_ratio,
                                            "increase_ratio": increase_ratio,
                                            "combined_ratio": combined_ratio,
                                            "detection_type": f"radius_pattern_th_{pass_threshold}",
                                        }

                                        logger.info(
                                            "  Suspicious edge (%d, %d) [ID: %s]: decrease=%.3f, increase=%.3f, combined=%.3f",
                                            node1,
                                            node2,
                                            edge_id,
                                            decrease_ratio,
                                            increase_ratio,
                                            combined_ratio,
                                        )

                                        suspicious_edges_with_ratios.append(
                                            {
                                                "edge": edge,
                                                "edge_data": edge_data,
                                                "decrease_ratio": detection_info[
                                                    "decrease_ratio"
                                                ],
                                                "increase_ratio": detection_info[
                                                    "increase_ratio"
                                                ],
                                                "combined_ratio": detection_info[
                                                    "combined_ratio"
                                                ],
                                                "detection_type": detection_info[
                                                    "detection_type"
                                                ],
                                            }
                                        )

                            except Exception as e:
                                logger.error(
                                    "Error processing edge (%d, %d) in cycle %d: %s",
                                    node1,
                                    node2,
                                    cycle_index + 1,
                                    e,
                                )
                                continue

                        # Remove only the most suspicious edge (highest combined ratio) from THIS cycle
                        if suspicious_edges_with_ratios:
                            # Sort by combined ratio (highest first)
                            suspicious_edges_with_ratios.sort(
                                key=lambda x: x["combined_ratio"], reverse=True
                            )

                            # Remove only the most suspicious edge
                            most_suspicious = suspicious_edges_with_ratios[0]
                            edge = most_suspicious["edge"]
                            edge_data = most_suspicious["edge_data"]
                            detection_type = most_suspicious["detection_type"]

                            logger.info(
                                "Cycle %d: Found %d suspicious edges, removing most suspicious: (%d, %d) [ID: %s] with score %.3f (%s)",
                                cycle_index + 1,
                                len(suspicious_edges_with_ratios),
                                edge[0],
                                edge[1],
                                edge_data.get("id", "unknown"),
                                most_suspicious["combined_ratio"],
                                detection_type,
                            )

                            # Log all suspicious edges for comparison
                            for idx, sus_edge in enumerate(
                                suspicious_edges_with_ratios
                            ):
                                rank_text = (
                                    "SELECTED" if idx == 0 else f"rank {idx + 1}"
                                )
                                logger.info(
                                    "  %s: (%d, %d) [ID: %s] - score=%.3f (%s)",
                                    rank_text,
                                    sus_edge["edge"][0],
                                    sus_edge["edge"][1],
                                    sus_edge["edge_data"].get("id", "unknown"),
                                    sus_edge["combined_ratio"],
                                    sus_edge["detection_type"],
                                )

                            if graph.has_edge(*edge):
                                # Remove the edge
                                removed_edges.append(
                                    (edge, edge_data.get("id", "unknown"))
                                )
                                graph.remove_edge(*edge)
                                pass_removed_count += 1
                                edge_removed_this_iteration = True

                                # Track this modified cycle
                                remaining_edges = [
                                    e for e in original_cycle if e != edge
                                ]
                                cycle_tracking.append((original_cycle, remaining_edges))

                                logger.info(
                                    "  → Removed most suspicious edge from cycle %d: (%d, %d) [ID: %s] (pass=%d, score=%.3f, %s)",
                                    cycle_index + 1,
                                    edge[0],
                                    edge[1],
                                    edge_data.get("id", "unknown"),
                                    pass_number,
                                    most_suspicious["combined_ratio"],
                                    detection_type,
                                )

                                # STOP processing cycles - restart iteration
                                logger.info(
                                    "Edge removed - stopping cycle inspection and restarting iteration"
                                )
                                break
                        else:
                            logger.info(
                                "Cycle %d: No edges removed (no suspicious patterns found with %s)",
                                cycle_index + 1,
                                pass_name,
                            )

                    # Summary of this iteration
                    if edge_removed_this_iteration:
                        logger.info(
                            "Iteration %d: Edge removed - restarting with fresh cycle detection",
                            iteration,
                        )
                    else:
                        logger.info(
                            "Iteration %d: No edges removed from any cycle - pass %d complete",
                            iteration,
                            pass_number,
                        )
                        break  # No more changes possible in this pass

                    iteration += 1

                except Exception as e:
                    logger.error("Error in iteration %d: %s", iteration, e)
                    import traceback

                    logger.error("Full traceback: %s", traceback.format_exc())
                    break

            # End of iterations for this pass
            if iteration > max_iterations:
                logger.warning(
                    "Reached maximum iterations (%d) for pass %d",
                    max_iterations,
                    pass_number,
                )

            logger.info(
                "Pass %d complete: removed %d edges total",
                pass_number,
                pass_removed_count,
            )

            # Decide whether to proceed to next pass
            if pass_number < len(pass_configs):
                # Check if cycles remain for next pass
                remaining_cycles = get_cycle_edges(graph)
                if remaining_cycles:
                    logger.info(
                        "Pass %d complete but %d cycles remain. Proceeding to pass %d.",
                        pass_number,
                        len(remaining_cycles),
                        pass_number + 1,
                    )
                else:
                    logger.info(
                        "Pass %d eliminated all cycles. Skipping remaining passes.",
                        pass_number,
                    )
                    break

    except Exception as e:
        logger.error("Error in _remove_suspicious_cycle_edges: %s", e)
        import traceback

        logger.error("Full traceback: %s", traceback.format_exc())

    # Final summary
    if removed_edges:
        edge_ids = [str(edge_id) for _, edge_id in removed_edges]
        logger.info(
            "=== FINAL SUMMARY: Removed %d suspicious cycle edges with IDs: %s ===",
            len(removed_edges),
            edge_ids,
        )
    else:
        logger.info("=== FINAL SUMMARY: No suspicious cycle edges were removed ===")

    # Check for remaining cycles
    final_cycles = get_cycle_edges(graph)
    if final_cycles:
        logger.warning(
            "=== %d cycles still remain after cleaning ===", len(final_cycles)
        )
        for i, cycle in enumerate(final_cycles):
            logger.warning(
                "  Remaining cycle %d: %d edges involving nodes %s",
                i + 1,
                len(cycle),
                sorted(set(node for edge in cycle for node in edge)),
            )
    else:
        logger.info("=== All cycles successfully removed ===")

    return len(removed_edges), cycle_tracking


def _count_large_terminal_vessels(
    graph: nx.Graph,
    min_diameter: float = 10.0,
    central_only: bool = True,
    centrality_threshold: float = 0.3,
) -> int:
    """
    Count terminal vessels with diameter above threshold for safety validation.

    This function counts major terminal vessels to ensure cleaning operations
    don't accidentally remove important anatomical structures. Used in safety
    checks to prevent over-aggressive cleaning.

    Args:
        graph: NetworkX graph to analyze
        min_diameter: Minimum mean diameter (mm) to consider vessel "large"
        central_only: If True, only count vessels within centrality threshold
        centrality_threshold: Maximum normalized distance from center (0-1 scale)

    Returns:
        int: Number of terminal vessels meeting diameter criteria

    Algorithm:
        1. Filter nodes to terminal nodes (degree == 1)
        2. If central_only: Calculate image center and filter by distance
        3. For each terminal node:
           - Get connecting edge data
           - Calculate mean diameter from edge mean_radius
           - Count if diameter >= threshold
        3. Log detected large terminals for verification

    Use Cases:
        - Pre-cleaning validation: ensure sufficient large terminals exist
        - Post-cleaning validation: verify large terminals preserved
        - Safety thresholds: prevent removal if would leave < 1 large terminal

    Note:
        Threshold of 10.0 mm chosen to identify major pulmonary vessels
        that should be preserved regardless of other criteria.
        When central_only=True, only anatomically central vessels count.
    """
    count = 0
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    # Calculate image center if central filtering is enabled
    image_center = None
    max_distance = None
    if central_only and len(graph.nodes) > 0:
        all_positions = [graph.nodes[n]["pos"] for n in graph.nodes]
        center_x = np.mean([pos[0] for pos in all_positions])
        center_y = np.mean([pos[1] for pos in all_positions])
        image_center = (center_x, center_y)

        # Calculate max distance among terminals for normalization
        terminal_distances = []
        for node in terminal_nodes:
            distance = _calculate_axial_centrality(node, graph, image_center)
            terminal_distances.append(distance)
        max_distance = max(terminal_distances) if terminal_distances else 1.0

    for node in terminal_nodes:
        if graph.neighbors(node):  # Ensure it has a neighbor
            neighbor = next(graph.neighbors(node))
            edge_data = graph.edges[node, neighbor]
            mean_radius = edge_data.get("mean_radius", 0)
            mean_diameter = 2 * mean_radius

            if mean_diameter >= min_diameter:
                # Apply centrality filter if enabled
                if central_only and image_center is not None:
                    distance = _calculate_axial_centrality(node, graph, image_center)
                    normalized_distance = (
                        distance / max_distance if max_distance > 0 else 0
                    )

                    if normalized_distance > centrality_threshold:
                        logger.debug(
                            f"Large terminal vessel excluded (too peripheral): node {node}, "
                            f"diameter={mean_diameter:.2f}, normalized_distance={normalized_distance:.3f}"
                        )
                        continue  # Skip peripheral vessels

                count += 1
                logger.debug(
                    f"Large {'central ' if central_only else ''}terminal vessel found: "
                    f"node {node}, diameter={mean_diameter:.2f}"
                )

    return count


def _get_large_terminal_vessels(
    graph: nx.Graph, min_diameter: float = 10.0
) -> list[tuple[int, float, float]]:
    """
    Retrieve terminal vessels exceeding diameter threshold with their measurements.

    Similar to _count_large_terminal_vessels but returns detailed information
    about each large terminal vessel including node ID, diameter, and length.
    Used for protective logic during aggressive cleaning operations.

    Args:
        graph: NetworkX graph to analyze
        min_diameter: Minimum mean diameter (mm) to consider vessel "large"

    Returns:
        list: Tuples of (node_id, diameter, length) for each large terminal vessel
              Sorted by combined score (diameter × length) when used in protection logic

    Note:
        Enables targeted protection of specific large terminals during
        cleaning operations that might otherwise remove them.
        Now includes length to help identify main trunk vessels.
    """
    large_terminals = []
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    for node in terminal_nodes:
        if graph.neighbors(node):  # Ensure it has a neighbor
            neighbor = next(graph.neighbors(node))
            edge_data = graph.edges[node, neighbor]
            mean_radius = edge_data.get("mean_radius", 0)
            mean_diameter = 2 * mean_radius
            length = edge_data.get("length", 0)

            if mean_diameter >= min_diameter:
                large_terminals.append((node, mean_diameter, length))

    return large_terminals


def _calculate_axial_centrality(
    node: int, graph: nx.Graph, image_center: tuple[float, float]
) -> float:
    """
    Calculate distance from node to image center in axial plane (x, y).

    Args:
        node: Node ID to evaluate
        graph: Graph containing node position data
        image_center: (x, y) coordinates of image center

    Returns:
        float: Euclidean distance from node to center in axial plane
    """
    pos = graph.nodes[node]["pos"]
    # Distance in axial plane (x, y), ignoring z
    dx = pos[0] - image_center[0]
    dy = pos[1] - image_center[1]
    return np.sqrt(dx * dx + dy * dy)


def _filter_removal_to_protect_largest(
    to_remove: list[tuple[int, int, str]],
    large_terminals: list[tuple[int, float, float]],
    graph: nx.Graph,
    prefer_central: bool = True,
    centrality_threshold: float = 0.3,
) -> tuple[list[tuple[int, int, str]], int]:
    """
    Filter branch removal list to protect the largest and longest central terminal vessel.

    This safety mechanism prevents over-aggressive cleaning by ensuring
    the best terminal vessel is never removed, using a combined score of
    diameter and length, with preference for vessels near the image center
    (anatomically more likely to be main trunk).

    Args:
        to_remove: List of (node, parent, edge_id) tuples marked for removal
        large_terminals: List of (node, diameter, length) tuples for large terminals
        graph: Graph containing node position data
        prefer_central: If True, prefer vessels near image center
        centrality_threshold: Maximum distance ratio from center (0-1 scale)

    Returns:
        tuple: (filtered_removal_list, protected_count)
               filtered_removal_list excludes protected terminals
               protected_count is number of vessels protected

    Algorithm:
        1. Calculate image center from all node positions
        2. If prefer_central:
           - Filter large terminals within centrality threshold
           - Calculate combined score = diameter × length for each
           - Select vessel with highest combined score
           - Fall back to absolute best if no central vessels
        3. Otherwise: Select vessel with highest combined score
        4. Filter removal list to exclude selected vessel
        5. Log protection action for verification

    Selection Criteria:
        Combined score = diameter (mm) × length (mm)
        This favors vessels that are both thick AND long (main trunk characteristics)

    Safety Logic:
        Activated when cleaning would result in < 1 large terminal vessel
        remaining, ensuring anatomically important structures survive.
        Centrality preference helps select main trunk over peripheral branches.
    """
    if not large_terminals:
        return to_remove, 0

    # Calculate combined score for each terminal: diameter × length
    # This favors vessels that are both large in diameter and long
    terminals_with_score = []
    for node, diameter, length in large_terminals:
        combined_score = diameter * length
        terminals_with_score.append((node, diameter, length, combined_score))

    # Sort by combined score (highest first)
    terminals_with_score.sort(key=lambda x: x[3], reverse=True)

    if prefer_central and len(graph.nodes) > 0:
        # Calculate image center in axial plane (x, y)
        all_positions = [graph.nodes[n]["pos"] for n in graph.nodes]
        center_x = np.mean([pos[0] for pos in all_positions])
        center_y = np.mean([pos[1] for pos in all_positions])
        image_center = (center_x, center_y)

        # Calculate max distance for normalization
        max_distance = max(
            _calculate_axial_centrality(node, graph, image_center)
            for node, _, _, _ in terminals_with_score
        )

        # Filter terminals within centrality threshold
        central_terminals = []
        for node, diameter, length, combined_score in terminals_with_score:
            distance = _calculate_axial_centrality(node, graph, image_center)
            normalized_distance = distance / max_distance if max_distance > 0 else 0

            if normalized_distance <= centrality_threshold:
                central_terminals.append(
                    (node, diameter, length, combined_score, distance)
                )
                logger.debug(
                    f"Central terminal candidate: node {node}, diameter={diameter:.2f}mm, "
                    f"length={length:.2f}mm, score={combined_score:.1f}, "
                    f"distance={distance:.1f}mm, normalized={normalized_distance:.3f}"
                )

        # Select best among central vessels (highest combined score), or absolute best if none central
        if central_terminals:
            # Sort by combined score (highest first)
            central_terminals.sort(key=lambda x: x[3], reverse=True)
            largest_terminal_node = central_terminals[0][0]
            largest_diameter = central_terminals[0][1]
            largest_length = central_terminals[0][2]
            largest_score = central_terminals[0][3]
            distance_from_center = central_terminals[0][4]
            logger.info(
                f"Protecting best CENTRAL terminal vessel: node {largest_terminal_node} "
                f"(diameter={largest_diameter:.2f}mm, length={largest_length:.2f}mm, "
                f"score={largest_score:.1f}, axial_distance={distance_from_center:.1f}mm, "
                f"{len(central_terminals)} central candidates)"
            )
        else:
            # No central vessels, fall back to absolute best by combined score
            largest_terminal_node = terminals_with_score[0][0]
            largest_diameter = terminals_with_score[0][1]
            largest_length = terminals_with_score[0][2]
            largest_score = terminals_with_score[0][3]
            logger.warning(
                f"No central vessels found within threshold. "
                f"Protecting absolute best: node {largest_terminal_node} "
                f"(diameter={largest_diameter:.2f}mm, length={largest_length:.2f}mm, score={largest_score:.1f})"
            )
    else:
        # Default behavior: protect vessel with highest combined score
        largest_terminal_node = terminals_with_score[0][0]
        largest_diameter = terminals_with_score[0][1]
        largest_length = terminals_with_score[0][2]
        largest_score = terminals_with_score[0][3]
        logger.info(
            f"Protecting best terminal vessel: node {largest_terminal_node} "
            f"(diameter={largest_diameter:.2f}mm, length={largest_length:.2f}mm, score={largest_score:.1f})"
        )

    # Filter out the selected terminal from removal list
    filtered_to_remove = []
    protected_count = 0

    for node, parent, edge_id in to_remove:
        if node != largest_terminal_node:
            filtered_to_remove.append((node, parent, edge_id))
        else:
            logger.info(
                f"Protected large terminal vessel from removal: node {node} (ID: {edge_id})"
            )
            protected_count += 1

    return filtered_to_remove, protected_count


def _remove_artefact_branches(
    graph: nx.Graph,
    radius_ratio_threshold: float,
    length_ratio_threshold: float,
    protect_largest_central: bool,
    min_diameter_vessel_protected: float = 10.0,
) -> int:
    """
    Remove terminal branches identified as artifacts with safety protections.

    This is the main branch cleaning function that removes short, thin, or
    otherwise artifactual terminal branches while implementing safety checks
    to preserve important anatomical structures.

    Args:
        graph: NetworkX graph to clean (modified in-place)
        radius_ratio_threshold: Minimum end/parent radius ratio to preserve branches
        length_ratio_threshold: Length factor relative to parent diameter
        protect_largest_central: Whether to protect the largest central terminal vessel
        min_diameter_vessel_protected: Minimum diameter to consider a vessel "large" and protect it

    Returns:
        int: Number of branches removed

    Algorithm:
        1. Count initial large terminal vessels for baseline
        2. Identify all branches meeting removal criteria
        3. Test removal impact on large terminal vessel count
        4. If < 1 large terminal would remain:
           - Apply protection logic to preserve largest terminal
           - Filter removal list accordingly
        5. Apply filtered removals to graph
        6. Log detailed removal statistics

    Safety Features:
        - Pre-removal simulation to assess impact
        - Automatic protection of largest terminal if needed
        - Comprehensive logging for verification
        - Error handling for individual branch failures

    Note:
        Central function for morphological cleaning. Balances aggressive
        artifact removal with anatomical structure preservation.
    """
    try:
        if protect_largest_central:
            logger.info(
                "Starting artefact branch removal with protection for largest CENTRAL terminal vessel (diameter >= %.1f mm)",
                min_diameter_vessel_protected,
            )
        else:
            logger.info(
                "Starting artefact branch removal with protection for largest terminal vessel (diameter >= %.1f mm)",
                min_diameter_vessel_protected,
            )
        # Count large CENTRAL terminal vessels before cleaning
        initial_large_terminals = _count_large_terminal_vessels(
            graph,
            min_diameter=min_diameter_vessel_protected,
            central_only=protect_largest_central,
            centrality_threshold=0.3,
        )
        if protect_largest_central:
            logger.info(
                f"Initial large CENTRAL terminal vessels (diameter >= 10mm, central 30%): {initial_large_terminals}"
            )
        else:
            logger.info(
                f"Initial large terminal vessels (diameter >= 10mm): {initial_large_terminals}"
            )

        logger.debug("Finding branches to remove")
        to_remove = _find_branches_to_remove(
            graph, radius_ratio_threshold, length_ratio_threshold
        )
        logger.debug("Found %d branches to remove", len(to_remove))

        # Create a test copy to check what would happen after removal
        test_graph = graph.copy()

        # Test remove the marked branches
        for node, parent, _ in to_remove:
            if test_graph.has_edge(node, parent):
                test_graph.remove_edge(node, parent)

        # Remove any isolated nodes that result
        test_graph.remove_nodes_from(list(nx.isolates(test_graph)))

        # Count large CENTRAL terminals after test removal
        final_large_terminals = _count_large_terminal_vessels(
            test_graph,
            min_diameter=min_diameter_vessel_protected,
            central_only=protect_largest_central,
            centrality_threshold=0.3,
        )
        if protect_largest_central:
            logger.info(
                f"Large CENTRAL terminal vessels after test removal: {final_large_terminals}"
            )
        else:
            logger.info(
                f"Large terminal vessels after test removal: {final_large_terminals}"
            )

        # Only apply protection logic if we would end up with 0 large CENTRAL terminals
        if final_large_terminals < 1:
            if protect_largest_central:
                logger.warning(
                    f"Safety check triggered: Only {final_large_terminals} large CENTRAL terminals would remain. "
                    "Applying protection logic to preserve largest central terminal vessel."
                )
            else:
                logger.warning(
                    f"Safety check triggered: Only {final_large_terminals} large terminals would remain. "
                    "Applying protection logic to preserve largest terminal vessel."
                )

            # Get large terminals and filter removal list
            large_terminals = _get_large_terminal_vessels(graph, min_diameter=10.0)
            filtered_to_remove, protected_count = _filter_removal_to_protect_largest(
                to_remove,
                large_terminals,
                graph,
                prefer_central=protect_largest_central,
                centrality_threshold=0.3,
            )

            if protected_count > 0:
                logger.info(
                    f"Protected {protected_count} large terminal vessels from removal"
                )
                to_remove = filtered_to_remove
        else:
            if protect_largest_central:
                logger.info(
                    f"Safety check passed: {final_large_terminals} large CENTRAL terminals will remain. Proceeding with normal cleaning."
                )
            else:
                logger.info(
                    f"Safety check passed: {final_large_terminals} large terminals will remain. Proceeding with normal cleaning."
                )
        # Apply the removals to the actual graph
        removed_ids = []
        for node, parent, edge_id in to_remove:
            try:
                if graph.has_edge(node, parent):
                    removed_ids.append(str(edge_id))
                    graph.remove_edge(node, parent)
            except Exception as e:
                logger.error("Error removing branch edge %s-%s: %s", node, parent, e)
                continue

        if to_remove:
            logger.info(
                "Removed %d short/thin terminal branches with IDs: %s",
                len(to_remove),
                removed_ids,
            )

        return len(to_remove)

    except Exception as e:
        logger.error("Error in _remove_artefact_branches: %s", e)
        import traceback

        logger.error("Full traceback: %s", traceback.format_exc())
        return 0


def _validate_graph_structure(graph: nx.Graph) -> bool:
    """
    Verify that graph contains required node and edge attributes.

    This
        - Edge attributes: "length", "radius", "centerline", "id" (optional)
        - Empty graph handling (considered valid)

    Note:
        Called at start of cleaning pipeline to catch structural issues early.
        Missing optional attributes logged as debug messages only.
    """
    if graph.number_of_nodes() == 0:
        return True

    # Check node attributes
    sample_node = next(iter(graph.nodes))
    required_node_attrs = ["radius", "pos"]

    for attr in required_node_attrs:
        if attr not in graph.nodes[sample_node]:
            logger.error("Missing required node attribute: %s", attr)
            return False

    # Check edge attributes (optional but log if missing)
    if graph.number_of_edges() > 0:
        sample_edge = next(iter(graph.edges))
        edge_data = graph.edges[sample_edge]
        optional_edge_attrs = ["length", "radius", "centerline", "id"]

        for attr in optional_edge_attrs:
            if attr not in edge_data:
                logger.debug("Missing optional edge attribute: %s", attr)

    return True


def get_graph_stats(graph: nx.Graph) -> dict[str, int]:
    """
    Calculate graph statistics for monitoring and logging.

    Args:
        graph: NetworkX graph to analyze

    Returns:
        dict: Statistics dictionary containing:
              - nodes: Total node count
              - edges: Total edge count
              - terminals: Nodes with degree 1 (vessel endpoints)
              - branches: Nodes with degree 3+ (bifurcations/junctions)
              - linear: Nodes with degree 2 (intermediate points)

    Use Cases:
        - Before/after cleaning comparisons
        - Pipeline monitoring and logging
        - Quality assessment of graph structure
        - Debugging unusual graph topologies
    """
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "terminals": 0, "branches": 0}

    degree_counts = {}
    for node in graph.nodes:
        degree = graph.degree(node)
        degree_counts[degree] = degree_counts.get(degree, 0) + 1

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "terminals": degree_counts.get(1, 0),  # degree 1
        "bifurcation": degree_counts.get(3, 0) + degree_counts.get(4, 0),  # degree 3+
        "linear": degree_counts.get(2, 0),  # degree 2
    }


def clean_graph(
    graph: nx.Graph,
    radius_ratio_threshold: float = 0.3,
    length_ratio_threshold: float = 1.5,
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
    save_cycle_skeletons: bool = False,
    cycle_skeletons_dir: Path | None = None,
    reference_skeleton_path: Path | None = None,
    protect_largest_central: bool = True,
    min_diameter_vessel_protected: float = 10.0,
) -> nx.Graph:
    """
    Execute comprehensive graph cleaning pipeline with statistics tracking.

    This is the main entry point for graph cleaning operations. It orchestrates
    the complete cleaning pipeline while providing validation, error handling,
    and progress monitoring. Statistics are always generated.

    Args:
        graph: NetworkX graph to clean (not modified, copy returned)
        radius_ratio_threshold: Minimum end/parent radius ratio for branch preservation
                               Range: (0, 1), default 0.3
        length_ratio_threshold: Length factor relative to parent diameter
                               Range: > 0, default 1.5
        patient_id: Optional patient identifier for statistics tracking
        patient_stats_path: Path to patient-specific CSV file (required)
        global_stats_path: Path to global CSV file (required)
        save_cycle_skeletons: If True, save skeletons of cycles before/after cleaning
        cycle_skeletons_dir: Directory to save cycle skeletons (required if save_cycle_skeletons=True)
        reference_skeleton_path: Path to reference skeleton for header/affine (required if save_cycle_skeletons=True)
        protect_largest_central: Whether to protect the largest central terminal vessel during branch removal
        min_diameter_vessel_protected: Minimum diameter (in mm) of vessels to protect during cleaning

    Returns:
        nx.Graph: Cleaned graph with artifacts removed and segments fused
                  Returns copy/empty graph on error conditions

    Pipeline Steps:
        1. Remove suspicious cycle edges
        2. Remove artifact branches (creates degree-2 nodes)
        3. Remove short terminal nodes (creates degree-2 nodes)
        4. Remove isolated nodes (cleanup)
        5. Iteratively fuse linear segments until none remain
    """
    # Enhanced input validation
    if graph is None:
        logger.error("None graph provided to clean_graph")
        return nx.Graph()

    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph provided to clean_graph")
        return graph.copy()

    # Validate required node/edge attributes
    if not _validate_graph_structure(graph):
        return graph.copy()

    # Validate thresholds
    if not (0 < radius_ratio_threshold < 1):
        logger.warning(
            "Invalid radius_ratio_threshold %.3f, using default 0.3",
            radius_ratio_threshold,
        )
        radius_ratio_threshold = 0.3

    if length_ratio_threshold <= 0:
        logger.warning(
            "Invalid length_ratio_threshold %.3f, using default 1.5",
            length_ratio_threshold,
        )
        length_ratio_threshold = 1.5

    # Validate and initialize statistics paths
    if patient_stats_path is None or global_stats_path is None:
        logger.error("Both patient_stats_path and global_stats_path are required")
        return graph.copy()

    patient_stats_path = Path(patient_stats_path)
    global_stats_path = Path(global_stats_path)

    # Initialize CSV files
    _initialize_stats_csv(patient_stats_path)
    if not global_stats_path.exists():
        _initialize_stats_csv(global_stats_path)

    if patient_id is None:
        patient_id = "unknown"

    # Validate cycle skeleton saving parameters
    if save_cycle_skeletons:
        if cycle_skeletons_dir is None:
            logger.warning(
                "save_cycle_skeletons=True but no cycle_skeletons_dir provided. Skipping cycle skeleton saving."
            )
            save_cycle_skeletons = False
        elif reference_skeleton_path is None:
            logger.warning(
                "save_cycle_skeletons=True but no reference_skeleton_path provided. Skipping cycle skeleton saving."
            )
            save_cycle_skeletons = False
        else:
            cycle_skeletons_dir = Path(cycle_skeletons_dir)
            cycle_skeletons_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cycle skeletons will be saved to: %s", cycle_skeletons_dir)

    # Load reference skeleton data if needed for cycle saving
    skeleton_shape = None
    skeleton_header = None
    skeleton_affine = None
    if save_cycle_skeletons:
        from ..utils.io import load_niftii

        skeleton_data, skeleton_header, _, skeleton_affine = load_niftii(
            str(reference_skeleton_path), "skeleton_original"
        )
        skeleton_shape = skeleton_data.shape
        logger.info("Loaded reference skeleton with shape: %s", skeleton_shape)

    # Make copy and track changes
    graph = graph.copy()

    # Record initial state
    previous_stats = _get_detailed_stats(graph)
    _append_stats_to_csv(
        patient_stats_path, patient_id, "0_initial", previous_stats, None
    )
    _append_stats_to_csv(
        global_stats_path, patient_id, "0_initial", previous_stats, None
    )

    logger.info(
        "Initial graph state: %d nodes, %d edges",
        previous_stats["nodes"],
        previous_stats["edges"],
    )

    try:
        logger.info("Starting graph cleaning pipeline")

        # Step 1: Remove suspicious cycle edges (with cycle skeleton saving)
        logger.info("Step 1: Removing suspicious cycle edges")

        # Detect cycles BEFORE cleaning
        cycles_before = get_cycle_edges(graph)
        logger.info("Found %d cycles before cleaning", len(cycles_before))

        # Save cycles BEFORE cleaning
        if save_cycle_skeletons and cycles_before:
            from ..utils.io import save_all_cycles_skeletons

            before_dir = cycle_skeletons_dir / "before_cleaning"
            saved_before = save_all_cycles_skeletons(
                graph=graph,
                cycles=cycles_before,
                reference_shape=skeleton_shape,
                header=skeleton_header,
                affine=skeleton_affine,
                output_dir=str(before_dir),
                prefix=f"{patient_id}_cycle_before",
            )
            logger.info("Saved %d cycle skeletons (before cleaning)", len(saved_before))

        # Perform cycle edge removal and get tracking info
        num_removed, cycle_tracking = _remove_suspicious_cycle_edges(graph)

        logger.info("Removed %d suspicious edges from cycles", num_removed)
        logger.info("Modified %d cycles (had edges removed)", len(cycle_tracking))

        # Save AFTER versions: only the cycles that were modified (had edges removed)
        if save_cycle_skeletons and cycle_tracking:
            from ..utils.io import save_all_cycles_skeletons

            after_dir = cycle_skeletons_dir / "after_cleaning"

            # Extract just the remaining edges from modified cycles
            remaining_cycles = [remaining for _, remaining in cycle_tracking]

            saved_after = save_all_cycles_skeletons(
                graph=graph,
                cycles=remaining_cycles,
                reference_shape=skeleton_shape,
                header=skeleton_header,
                affine=skeleton_affine,
                output_dir=str(after_dir),
                prefix=f"{patient_id}_cycle_after",
            )
            logger.info(
                "Saved %d cycle skeletons (after cleaning - modified cycles only)",
                len(saved_after),
            )

            # Log details about each modified cycle
            for idx, (original, remaining) in enumerate(cycle_tracking):
                logger.info(
                    "Cycle %d: %d edges before → %d edges after (removed %d)",
                    idx + 1,
                    len(original),
                    len(remaining),
                    len(original) - len(remaining),
                )
        elif save_cycle_skeletons and not cycle_tracking:
            logger.info("No cycles were modified - no 'after' skeletons to save")

        current_stats = _get_detailed_stats(graph)
        _append_stats_to_csv(
            patient_stats_path,
            patient_id,
            "1_remove_cycles",
            current_stats,
            previous_stats,
        )
        _append_stats_to_csv(
            global_stats_path,
            patient_id,
            "1_remove_cycles",
            current_stats,
            previous_stats,
        )
        previous_stats = current_stats

        # Step 2: Remove artifact branches
        logger.info("Step 2: Removing artefact branches")
        _remove_artefact_branches(
            graph,
            radius_ratio_threshold,
            length_ratio_threshold,
            protect_largest_central,
            min_diameter_vessel_protected,
        )
        current_stats = _get_detailed_stats(graph)
        _append_stats_to_csv(
            patient_stats_path,
            patient_id,
            "2_remove_branches",
            current_stats,
            previous_stats,
        )
        _append_stats_to_csv(
            global_stats_path,
            patient_id,
            "2_remove_branches",
            current_stats,
            previous_stats,
        )
        previous_stats = current_stats

        # Step 3: Remove short terminal nodes (DISABLED - redundant with Step 2 and breaks protection)
        # This step removed terminal nodes with edge length < 1.0mm, but:
        # - Step 2 already handles artifact removal with proper morphological criteria
        # - This step ignores vessel diameter and can remove protected large terminals
        # logger.info("Step 3: Removing single neighbor nodes")
        # nodes_before = graph.number_of_nodes()
        # graph = remove_single_neighbor_nodes(graph)
        # nodes_after = graph.number_of_nodes()
        # logger.info("Step 3 removed %d nodes", nodes_before - nodes_after)
        # current_stats = _get_detailed_stats(graph)
        # _append_stats_to_csv(
        #     patient_stats_path,
        #     patient_id,
        #     "3_remove_terminals",
        #     current_stats,
        #     previous_stats,
        # )
        # _append_stats_to_csv(
        #     global_stats_path,
        #     patient_id,
        #     "3_remove_terminals",
        #     current_stats,
        #     previous_stats,
        # )
        # previous_stats = current_stats

        # Step 3: Remove isolated nodes (previously Step 4)
        logger.info("Step 3: Removing isolated nodes")
        remove_isolated_nodes(graph)
        current_stats = _get_detailed_stats(graph)
        _append_stats_to_csv(
            patient_stats_path,
            patient_id,
            "3_remove_isolated",
            current_stats,
            previous_stats,
        )
        _append_stats_to_csv(
            global_stats_path,
            patient_id,
            "3_remove_isolated",
            current_stats,
            previous_stats,
        )
        previous_stats = current_stats

        # Step 4: Iteratively fuse linear segments until none remain (previously Step 5)
        fusion_pass = 1
        max_fusion_passes = 10  # Safety limit to prevent infinite loops

        while current_stats["linear"] > 0 and fusion_pass <= max_fusion_passes:
            logger.info(
                "Step 4.%d: Fusing linear segments (pass %d, %d linear nodes remaining)",
                fusion_pass,
                fusion_pass,
                current_stats["linear"],
            )

            # Store state before fusion
            # nodes_before = current_stats["nodes"]
            linear_before = current_stats["linear"]

            # Perform fusion
            graph = fuse_linear_segments(graph)
            current_stats = _get_detailed_stats(graph)

            # Log this fusion pass
            step_name = f"4_fuse_linear_pass_{fusion_pass}"
            _append_stats_to_csv(
                patient_stats_path, patient_id, step_name, current_stats, previous_stats
            )
            _append_stats_to_csv(
                global_stats_path, patient_id, step_name, current_stats, previous_stats
            )
            previous_stats = current_stats

            # Check if we made progress
            if current_stats["linear"] >= linear_before:
                logger.warning(
                    "Fusion pass %d did not reduce linear nodes (%d -> %d). Stopping iteration.",
                    fusion_pass,
                    linear_before,
                    current_stats["linear"],
                )
                break

            fusion_pass += 1

        # Final status check
        if current_stats["linear"] > 0:
            if fusion_pass > max_fusion_passes:
                logger.error(
                    "Reached maximum fusion passes (%d) with %d linear nodes remaining. "
                    "This may indicate a bug in the fusion logic.",
                    max_fusion_passes,
                    current_stats["linear"],
                )
            else:
                logger.warning(
                    "Completed fusion after %d passes, but %d linear nodes still remain. "
                    "These may be part of complex cycles or edge cases.",
                    fusion_pass - 1,
                    current_stats["linear"],
                )
        else:
            logger.info(
                "Successfully fused all linear segments after %d passes",
                fusion_pass - 1,
            )

        logger.info(
            "Final graph state: %d nodes, %d edges",
            current_stats["nodes"],
            current_stats["edges"],
        )
        logger.info("Patient statistics saved to: %s", patient_stats_path)
        logger.info("Global statistics updated: %s", global_stats_path)

        # Step 5: Final statistics (previously Step 6)
        current_stats = _get_detailed_stats(graph)
        _append_stats_to_csv(
            patient_stats_path, patient_id, "5_final", current_stats, previous_stats
        )
        _append_stats_to_csv(
            global_stats_path, patient_id, "5_final", current_stats, previous_stats
        )

    except Exception as e:
        logger.error("Error during graph cleaning: %s", e)
        import traceback

        logger.error("Full traceback: %s", traceback.format_exc())
        return graph

    return graph
