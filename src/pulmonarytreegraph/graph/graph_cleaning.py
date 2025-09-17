"""
Module for cleaning and optimizing pulmonary vascular tree graphs.
This module provides functionality to clean and optimize graph representations
of pulmonary vascular trees by removing suspicious cycles, short branches, and fusing segments.
"""

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

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
            return False

        decrease_ratio = (max_before - min_value) / max_before
        increase_ratio = (max_after - min_value) / min_value

        result = decrease_ratio > threshold_factor and increase_ratio > threshold_factor
        logger.debug(
            "Pattern detection result: %s (decrease: %.3f, increase: %.3f)",
            result,
            decrease_ratio,
            increase_ratio,
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
        # More efficient node removal logic
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


def _remove_suspicious_cycle_edges(graph: nx.Graph) -> int:
    """
    Detect and remove edges in cycles that show suspicious radius patterns.

    Vascular networks are typically tree-like; cycles usually indicate false
    connections created during segmentation due to close proximity of vessels.
    This function identifies cycles and removes edges showing constriction
    patterns characteristic of artifacts.

    Args:
        graph: NetworkX graph to clean (modified in-place)

    Returns:
        int: Number of edges removed

    Algorithm:
        1. Detect all cycles using get_cycle_edges()
        2. For each edge in each cycle:
           - Extract radius profile along the edge
           - Test for constriction pattern using detect_radius_pattern()
           - Remove edge if pattern detected
        3. Log removed edge IDs for traceability
    """
    removed_edges = []

    try:
        logger.debug("Getting cycle edges")
        cycles = get_cycle_edges(graph)
        logger.debug("Found %d cycles", len(cycles))

        for i, cycle in enumerate(cycles):
            logger.debug("Processing cycle %d with %d edges", i, len(cycle))

            for j, edge in enumerate(cycle):
                logger.debug("Processing edge %d: %s", j, edge)

                try:
                    edge_data = get_edge_data(graph, *edge)
                    if edge_data:
                        radius_data = edge_data.get("radius", [])
                        logger.debug(
                            "Radius data type: %s, length: %d",
                            type(radius_data),
                            len(radius_data) if hasattr(radius_data, "__len__") else 0,
                        )

                        if detect_radius_pattern(radius_data):
                            if graph.has_edge(*edge):
                                removed_edges.append(
                                    (edge, edge_data.get("id", "unknown"))
                                )
                                graph.remove_edge(*edge)
                                logger.debug("Removed suspicious edge: %s", edge)
                except Exception as e:
                    logger.error("Error processing edge %s in cycle %d: %s", edge, i, e)
                    continue

    except Exception as e:
        logger.error("Error in _remove_suspicious_cycle_edges: %s", e)
        import traceback

        logger.error("Full traceback: %s", traceback.format_exc())

    if removed_edges:
        edge_ids = [str(edge_id) for _, edge_id in removed_edges]
        logger.info(
            "Removed %d suspicious cycle edges with IDs: %s",
            len(removed_edges),
            edge_ids,
        )

    return len(removed_edges)


def _count_large_terminal_vessels(graph: nx.Graph, min_diameter: float = 10.0) -> int:
    """
    Count terminal vessels with diameter above threshold for safety validation.

    This function counts major terminal vessels to ensure cleaning operations
    don't accidentally remove important anatomical structures. Used in safety
    checks to prevent over-aggressive cleaning.

    Args:
        graph: NetworkX graph to analyze
        min_diameter: Minimum mean diameter (mm) to consider vessel "large"

    Returns:
        int: Number of terminal vessels meeting diameter criteria

    Algorithm:
        1. Filter nodes to terminal nodes (degree == 1)
        2. For each terminal node:
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
    """
    count = 0
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    for node in terminal_nodes:
        if graph.neighbors(node):  # Ensure it has a neighbor
            neighbor = next(graph.neighbors(node))
            edge_data = graph.edges[node, neighbor]
            mean_radius = edge_data.get("mean_radius", 0)
            mean_diameter = 2 * mean_radius

            if mean_diameter >= min_diameter:
                count += 1
                logger.debug(
                    f"Large terminal vessel found: node {node}, diameter={mean_diameter:.2f}"
                )

    return count


def _get_large_terminal_vessels(
    graph: nx.Graph, min_diameter: float = 10.0
) -> list[tuple[int, float]]:
    """
    Retrieve terminal vessels exceeding diameter threshold with their measurements.

    Similar to _count_large_terminal_vessels but returns detailed information
    about each large terminal vessel including node ID and diameter. Used
    for protective logic during aggressive cleaning operations.

    Args:
        graph: NetworkX graph to analyze
        min_diameter: Minimum mean diameter (mm) to consider vessel "large"

    Returns:
        list: Tuples of (node_id, diameter) for each large terminal vessel
              Sorted by diameter (largest first) when used in protection logic

    Note:
        Enables targeted protection of specific large terminals during
        cleaning operations that might otherwise remove them.
    """
    large_terminals = []
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    for node in terminal_nodes:
        if graph.neighbors(node):  # Ensure it has a neighbor
            neighbor = next(graph.neighbors(node))
            edge_data = graph.edges[node, neighbor]
            mean_radius = edge_data.get("mean_radius", 0)
            mean_diameter = 2 * mean_radius

            if mean_diameter >= min_diameter:
                large_terminals.append((node, mean_diameter))

    return large_terminals


def _filter_removal_to_protect_largest(
    to_remove: list[tuple[int, int, str]], large_terminals: list[tuple[int, float]]
) -> tuple[list[tuple[int, int, str]], int]:
    """
    Filter branch removal list to protect the largest terminal vessel.

    This safety mechanism prevents over-aggressive cleaning by ensuring
    the largest terminal vessel is never removed, even if it meets other
    removal criteria. Critical for preserving anatomical integrity.

    Args:
        to_remove: List of (node, parent, edge_id) tuples marked for removal
        large_terminals: List of (node, diameter) tuples for large terminals

    Returns:
        tuple: (filtered_removal_list, protected_count)
               filtered_removal_list excludes protected terminals
               protected_count is number of vessels protected

    Algorithm:
        1. Sort large terminals by diameter (largest first)
        2. Identify largest terminal vessel
        3. Filter removal list to exclude largest terminal
        4. Log protection action for verification

    Safety Logic:
        Activated when cleaning would result in < 1 large terminal vessel
        remaining, ensuring anatomically important structures survive.
    """
    if not large_terminals:
        return to_remove, 0

    # Sort by diameter (largest first) and get the largest
    large_terminals.sort(key=lambda x: x[1], reverse=True)
    largest_terminal_node = large_terminals[0][0]
    largest_diameter = large_terminals[0][1]

    logger.info(
        f"Protecting largest terminal vessel: node {largest_terminal_node} (diameter={largest_diameter:.2f})"
    )

    # Filter out the largest terminal from removal list
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
    graph: nx.Graph, radius_ratio_threshold: float, length_ratio_threshold: float
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
        # Count large terminal vessels before cleaning
        initial_large_terminals = _count_large_terminal_vessels(
            graph, min_diameter=10.0
        )
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

        # Count large terminals after test removal
        final_large_terminals = _count_large_terminal_vessels(
            test_graph, min_diameter=10.0
        )
        logger.info(
            f"Large terminal vessels after test removal: {final_large_terminals}"
        )

        # Only apply protection logic if we would end up with 0 large terminals
        if final_large_terminals < 1:
            logger.warning(
                f"Safety check triggered: Only {final_large_terminals} large terminals would remain. "
                "Applying protection logic to preserve largest terminal vessel."
            )

            # Get large terminals and filter removal list
            large_terminals = _get_large_terminal_vessels(graph, min_diameter=10.0)
            filtered_to_remove, protected_count = _filter_removal_to_protect_largest(
                to_remove, large_terminals
            )

            if protected_count > 0:
                logger.info(
                    f"Protected {protected_count} large terminal vessels from removal"
                )
                to_remove = filtered_to_remove
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

    This validation function ensures the input graph has the necessary
    attributes for cleaning operations. Prevents runtime errors and
    provides clear feedback about missing data.

    Args:
        graph: NetworkX graph to validate

    Returns:
        bool: True if graph structure is valid, False otherwise

    Validation Checks:
        - Node attributes: "radius", "pos" (required)
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
        "branches": degree_counts.get(3, 0) + degree_counts.get(4, 0),  # degree 3+
        "linear": degree_counts.get(2, 0),  # degree 2
    }


def clean_graph(
    graph: nx.Graph,
    radius_ratio_threshold: float = 0.3,
    length_ratio_threshold: float = 1.5,
) -> nx.Graph:
    """
    Execute comprehensive graph cleaning pipeline with full error handling.

    This is the main entry point for graph cleaning operations. It orchestrates
    the complete cleaning pipeline while providing validation, error handling,
    and progress monitoring.

    Args:
        graph: NetworkX graph to clean (not modified, copy returned)
        radius_ratio_threshold: Minimum end/parent radius ratio for branch preservation
                               Range: (0, 1), default 0.3
        length_ratio_threshold: Length factor relative to parent diameter
                               Range: > 0, default 1.5

    Returns:
        nx.Graph: Cleaned graph with artifacts removed and segments fused
                  Returns copy/empty graph on error conditions

    Pipeline Steps:
        1. Input validation and parameter verification
        2. Structure validation (required attributes)
        3. Suspicious cycle edge removal
        4. Artifact branch removal (with safety checks)
        5. Linear segment fusion
        6. Short terminal node removal
        7. Isolated node cleanup
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

    # Make copy and track changes
    graph = graph.copy()
    initial_stats = get_graph_stats(graph)

    try:
        # Apply cleaning pipeline with progress tracking
        logger.info("Starting graph cleaning pipeline")

        logger.info("Step 1: Removing suspicious cycle edges")
        _remove_suspicious_cycle_edges(graph)

        logger.info("Step 2: Removing artefact branches")
        _remove_artefact_branches(graph, radius_ratio_threshold, length_ratio_threshold)

        logger.info("Step 3: Fusing linear segments")
        graph = fuse_linear_segments(graph)

        logger.info("Step 4: Removing single neighbor nodes")
        graph = remove_single_neighbor_nodes(graph)

        logger.info("Step 5: Removing isolated nodes")
        remove_isolated_nodes(graph)

        final_stats = get_graph_stats(graph)
        nodes_removed = initial_stats["nodes"] - final_stats["nodes"]
        edges_removed = initial_stats["edges"] - final_stats["edges"]

        if nodes_removed > 0 or edges_removed > 0:
            logger.info(
                "Cleaning complete: removed %d nodes, %d edges",
                nodes_removed,
                edges_removed,
            )

    except Exception as e:
        logger.error("Error during graph cleaning: %s", e)
        import traceback

        logger.error("Full traceback: %s", traceback.format_exc())
        return graph

    return graph
