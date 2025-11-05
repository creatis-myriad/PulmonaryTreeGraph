"""
Graph construction module for building pulmonary vascular tree graphs.

This module provides functions to convert binary masks into skeletons and then into
NetworkX graphs representing the vascular tree structure.
"""

import csv
import logging
from pathlib import Path

import networkx as nx
import numpy as np
from networkx import from_scipy_sparse_array
from skan.csr import skeleton_to_csgraph

logger = logging.getLogger(__name__)

_initialized_global_csvs = (
    set()
)  # Track which global CSVs have been initialized this run

# =============================================================================
# STATISTICS TRACKING FOR CONSTRUCTION
# =============================================================================


def _initialize_construction_stats_csv(output_path: Path, force: bool = False) -> None:
    """Create CSV file with header for construction statistics.

    Args:
        output_path: Path to CSV file
        force: If True, always recreate the file. If False, only create if not exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing file if force=True
    if force and output_path.exists():
        output_path.unlink()
        logger.debug("Deleted existing CSV file: %s", output_path)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "patient_id",
                "step",
                "nodes",
                "edges",
                "components",
                "terminals",
                "bifurcations",
                "linear",
                "isolated",
                "cycles",
                "voxel_coverage",
            ]
        )
    logger.debug("Created CSV file: %s", output_path)


def _get_construction_stats(
    graph: nx.Graph, original_skeleton: np.ndarray | None = None
) -> dict:
    """Calculate detailed graph statistics during construction."""
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "terminals": 0,
        "bifurcations": 0,
        "linear": 0,
        "isolated": 0,
        "cycles": len(nx.cycle_basis(graph)) if graph.number_of_edges() > 0 else 0,
        "voxel_coverage": 0.0,
    }

    # Count node types
    for node in graph.nodes:
        degree = graph.degree(node)
        if degree == 0:
            stats["isolated"] += 1
        elif degree == 1:
            stats["terminals"] += 1
        elif degree == 2:
            stats["linear"] += 1
        else:
            stats["bifurcations"] += 1

    # Calculate voxel coverage if original skeleton provided
    if original_skeleton is not None:
        original_voxel_count = np.sum(original_skeleton > 0)
        if original_voxel_count > 0:
            covered_voxels = set()

            for _, _, data in graph.edges(data=True):
                centerline = data.get("centerline", [])
                for pos in centerline:
                    voxel_pos = tuple(int(round(coord)) for coord in pos)
                    if (
                        0 <= voxel_pos[0] < original_skeleton.shape[0]
                        and 0 <= voxel_pos[1] < original_skeleton.shape[1]
                        and 0 <= voxel_pos[2] < original_skeleton.shape[2]
                    ):
                        covered_voxels.add(voxel_pos)

            stats["voxel_coverage"] = (len(covered_voxels) / original_voxel_count) * 100

    return stats


def _append_construction_stats_to_csv(
    output_path: Path,
    patient_id: str,
    step: str,
    stats: dict,
) -> None:
    """Append construction statistics row to CSV file."""
    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                patient_id,
                step,
                stats["nodes"],
                stats["edges"],
                stats["components"],
                stats["terminals"],
                stats["bifurcations"],
                stats["linear"],
                stats["isolated"],
                stats["cycles"],
                f"{stats['voxel_coverage']:.2f}",
            ]
        )


# =============================================================================
# PIXEL-LEVEL GRAPH CONSTRUCTION
# =============================================================================


def skeleton_to_graph(
    skeleton: np.ndarray,
    distance_map: np.ndarray,
    spacing: tuple[float, float, float],
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
) -> nx.Graph:
    """
    Convert a skeleton (binary 3D array) to a pixel-level NetworkX graph.

    This function transforms a binary skeleton into a graph where each voxel
    becomes a node. Uses the skan library for robust skeleton-to-graph conversion
    with proper 3D connectivity analysis.

    Args:
        skeleton: Binary 3D array representing vessel skeleton
        distance_map: 3D array representing the distance map for radius extraction
        spacing: Voxel spacing in (x, y, z) dimensions for accurate measurements

    Returns:
        NetworkX graph with nodes at each skeleton pixel, edges between connected pixels.
        Each node has 'pos' (3D coordinates) and 'radius' (from distance map) attributes.

    Algorithm:
        1. Count input skeleton voxels for logging
        2. Convert skeleton to sparse graph format using skan.skeleton_to_csgraph
        3. Create NetworkX graph from sparse adjacency matrix
        4. Add node attributes: position and radius from distance map
        5. Log conversion statistics and any voxel losses
    """
    # Count input skeleton voxels
    input_voxel_count = np.sum(skeleton > 0)
    logger.info("Input skeleton contains %d voxels", input_voxel_count)

    # Convert skeleton to sparse graph format
    adjacency_matrix, pixel_coordinates = skeleton_to_csgraph(skeleton, spacing=spacing)
    graph = from_scipy_sparse_array(adjacency_matrix)

    # Count output nodes and track conversion losses
    output_node_count = graph.number_of_nodes()
    logger.info("skeleton_to_csgraph produced %d nodes", output_node_count)
    logger.info(
        "Lost %d voxels in skeleton_to_csgraph conversion",
        input_voxel_count - output_node_count,
    )

    # Add node attributes: position and radius
    pixel_coordinates = np.asarray(pixel_coordinates).T

    for node_id in graph.nodes:
        pos = pixel_coordinates[node_id].tolist()
        graph.nodes[node_id]["pos"] = pos
        # Store radius directly from distance_map, converting to Python float to preserve precision
        radius_val = distance_map[pos[0], pos[1], pos[2]]
        graph.nodes[node_id]["radius"] = float(radius_val) if radius_val > 0 else 1e-6

    # Track statistics with voxel coverage calculation for pixel graph
    if patient_stats_path and patient_id:
        # Calculate voxel coverage for pixel graph
        # Since pixel graph nodes ARE the skeleton voxels, coverage should be ~100%
        covered_voxels = set()
        for node_id in graph.nodes:
            pos = graph.nodes[node_id]["pos"]
            voxel_pos = tuple(int(round(coord)) for coord in pos)
            covered_voxels.add(voxel_pos)

        original_voxel_count = np.sum(skeleton > 0)
        voxel_coverage = (
            (len(covered_voxels) / original_voxel_count * 100)
            if original_voxel_count > 0
            else 0.0
        )

        stats = _get_construction_stats(graph)
        stats["voxel_coverage"] = voxel_coverage

        _append_construction_stats_to_csv(
            patient_stats_path, patient_id, "1_pixel_graph", stats
        )
        if global_stats_path:
            _append_construction_stats_to_csv(
                global_stats_path, patient_id, "1_pixel_graph", stats
            )

    return graph


# =============================================================================
# PATH ANALYSIS UTILITIES
# =============================================================================


def is_path_endpoint(graph: nx.Graph, node: int) -> bool:
    """
    Check if a node is an endpoint or branch point in the graph.

    A node is considered an endpoint if it has any number of neighbors
    other than exactly 2. This identifies anatomically significant points:
    - Degree 1: Terminal points (vessel ends)
    - Degree 3+: Branch points (bifurcations, trifurcations, etc)
    - Degree 2: Intermediate points on linear segments (NOT endpoints)

    Args:
        graph: NetworkX graph to analyze
        node: Node ID to check

    Returns:
        bool: True if node is endpoint/branch point, False if it's on a linear path

    Algorithm:
        1. Get the degree (number of neighbors) of the node
        2. Return True if degree != 2, else False
    """
    return len(list(graph.neighbors(node))) != 2


def get_edge_data(graph: nx.Graph, start: int, end: int) -> tuple[list, list, float]:
    """
    Extract geometric properties from a pixel-level edge.

    Retrieves centerline coordinates, radius values, and length from edge attributes.
    Provides fallback defaults if attributes are missing to ensure robust processing.

    Args:
        graph: NetworkX graph containing edge data
        start: Start node ID
        end: End node ID

    Returns:
        tuple: (centerline, radius_list, length)
               - centerline: List of 3D coordinates along the edge
               - radius_list: List of radius values at each point
               - length: Edge length (from 'weight' attribute or default 1.0)

    Algorithm:
        1. Access edge data between start and end nodes
        2. Extract 'centerline', 'radius', and 'weight' attributes with defaults
        3. Return extracted properties
    """
    edge = graph.edges[(start, end)]
    pos_start = graph.nodes[start]["pos"]
    pos_end = graph.nodes[end]["pos"]

    # Extract or compute default values
    centerline = edge.get("centerline", [pos_start, pos_end])
    radius = edge.get(
        "radius",
        [graph.nodes[start].get("radius", 0.0), graph.nodes[end].get("radius", 0.0)],
    )
    length = edge.get("weight", 1.0)
    return centerline, radius, length


# =============================================================================
# PATH FOLLOWING ALGORITHM
# =============================================================================


def follow_path(
    graph: nx.Graph, start: int, neighbor: int, visited: set
) -> tuple[int, list, list, float]:
    """
    Follow a linear path through the graph until reaching a significant node.

    This function traces a path from a starting point through intermediate nodes
    (degree 2) until reaching a terminal point (degree 1) or branch point (degree 3+).
    Accumulates geometric properties along the way.

    Args:
        graph: Input graph
        start: Starting node
        neighbor: First neighbor to follow
        visited: Set of already visited (source, target) edges (modified in-place)

    Returns:
        tuple: (end_node, centerline, radius, total_length)
               - end_node: Final significant node reached
               - centerline: Combined 3D coordinates along entire path
               - radius: Combined radius values along entire path
               - total_length: Sum of all edge lengths in the path

    Algorithm:
        1. Initialize with first edge data
        2. Follow path through intermediate nodes (degree 2)
        3. Accumulate geometric properties at each step
        4. Stop at next significant node (degree ≠ 2)
        5. Mark all traversed edges as visited
    """
    # Initialize with first segment
    centerline, radius, total_length = get_edge_data(graph, start, neighbor)
    current, prev = neighbor, start
    visited.add((start, neighbor))  # Mark the first segment as visited immediately

    # Follow path until reaching next endpoint/branch point
    while not is_path_endpoint(graph, current):
        # Get next node in the linear path
        next_nodes = [n for n in graph.neighbors(current) if n != prev]
        if not next_nodes:
            break  # Dead end encountered

        next_node = next_nodes[0]

        # Get data for next segment
        next_centerline, next_radius, length = get_edge_data(graph, current, next_node)

        # Accumulate geometric properties (skip first point to avoid duplication)
        centerline.extend(next_centerline[1:])
        radius.extend(next_radius[1:])
        total_length += length

        # Mark edge as visited and advance
        visited.add((current, next_node))
        prev, current = current, next_node

    # The loop terminates when 'current' is the endpoint.
    # The centerline should naturally end at this point.
    final_pos = graph.nodes[current]["pos"]
    if not np.array_equal(centerline[-1], final_pos):
        # This indicates a discrepancy, but for robustness, we append the correct final position.
        logger.debug(
            "Correcting centerline end: appending final node %d position", current
        )
        centerline.append(final_pos)
        radius.append(graph.nodes[current].get("radius", 0.0))

    return current, centerline, radius, total_length


# =============================================================================
# BRANCH GRAPH CONSTRUCTION
# =============================================================================


def create_branch_graph(
    graph: nx.Graph,
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
    original_skeleton: np.ndarray | None = None,
) -> nx.Graph:
    """
    Transform pixel-level graph into branch-level graph representation with statistics tracking.

    Consolidates linear path segments into single edges connecting anatomically
    significant nodes (terminals and branch points). Preserves all geometric
    properties while dramatically simplifying graph structure.

    Args:
        graph: Pixel-level NetworkX graph with individual skeleton pixels as nodes

    Returns:
        Branch-level NetworkX graph with vessels as edges between branch points.
        Nodes represent terminals/bifurcations, edges represent vessel segments.

    Algorithm:
        1. Identify all endpoint and branch point nodes
        2. For each edge in the original graph:
           - If not yet visited, follow the linear path to the next significant node
           - Accumulate geometric properties along the path
           - Create single edge representing the entire vessel segment
        3. Track visited edges to avoid duplication
        4. Assign unique IDs to all created edges

    Edge Attributes:
        - id: Unique numeric identifier
        - name: Human-readable identifier ("edge_N")
        - centerline: 3D coordinates along vessel centerline
        - radius: Radius values along centerline
        - length: Total vessel segment length
        - min/max/mean_radius: Statistical measures
    """
    branches_graph = nx.Graph()
    edge_counter = 1
    visited = set()

    # Identify all anatomically significant nodes (terminals and branch points)
    endpoints = [n for n in graph.nodes() if is_path_endpoint(graph, n)]
    logger.info("Found %d endpoints/branch points", len(endpoints))

    # Process all edges in the graph to ensure complete coverage
    total_edges = graph.number_of_edges()
    processed_edges = 0

    for path_start_node, path_start_neighbor in graph.edges():
        # Skip if this edge pair already processed
        if (path_start_node, path_start_neighbor) in visited or (
            path_start_neighbor,
            path_start_node,
        ) in visited:
            continue

        processed_edges += 1
        if processed_edges % 1000 == 0:
            logger.debug("Processed %d/%d edges", processed_edges, total_edges)

        # These are the final start/end nodes of the new branch edge
        final_start_node = path_start_node

        # Determine direction and follow the path
        if is_path_endpoint(graph, path_start_node) and not is_path_endpoint(
            graph, path_start_neighbor
        ):
            final_end_node, centerline, radius, length = follow_path(
                graph, path_start_node, path_start_neighbor, visited
            )
        elif is_path_endpoint(graph, path_start_neighbor) and not is_path_endpoint(
            graph, path_start_node
        ):
            # The path starts from the neighbor, so we swap
            final_start_node = path_start_neighbor
            final_end_node, centerline, radius, length = follow_path(
                graph, path_start_neighbor, path_start_node, visited
            )
        else:
            # For intermediate-to-intermediate paths, follow from the first node
            final_end_node, centerline, radius, length = follow_path(
                graph, path_start_node, path_start_neighbor, visited
            )

        # Add both endpoint nodes to branch graph
        for node in (final_start_node, final_end_node):
            if node not in branches_graph:
                branches_graph.add_node(
                    node,
                    pos=graph.nodes[node]["pos"],
                    radius=graph.nodes[node].get("radius", 0.0),
                )

        # Orient centerline to flow from final_start_node to final_end_node
        if len(centerline) > 0:
            centerline_array = np.asarray(centerline)
            start_pos = np.array(graph.nodes[final_start_node]["pos"])
            end_pos = np.array(graph.nodes[final_end_node]["pos"])

            first_point = centerline_array[0]
            last_point = centerline_array[-1]

            # Check if centerline flows from start to end
            first_matches_start = np.array_equal(first_point, start_pos)
            last_matches_end = np.array_equal(last_point, end_pos)

            if not (first_matches_start and last_matches_end):
                # Check if it's reversed
                first_matches_end = np.array_equal(first_point, end_pos)
                last_matches_start = np.array_equal(last_point, start_pos)

                if first_matches_end and last_matches_start:
                    # Reverse centerline and radius to match edge direction
                    logger.debug(
                        "Reversing centerline for edge %d->%d (ID %d)",
                        final_start_node,
                        final_end_node,
                        edge_counter,
                    )
                    centerline = list(reversed(centerline))
                    radius = list(reversed(radius))
                else:
                    # This warning should now only trigger for genuine issues
                    logger.warning(
                        "Ambiguous centerline orientation for edge %d->%d (ID %d)",
                        final_start_node,
                        final_end_node,
                        edge_counter,
                    )
                    logger.debug("  Start node %d pos: %s", final_start_node, start_pos)
                    logger.debug("  End node %d pos: %s", final_end_node, end_pos)
                    logger.debug("  Centerline first point: %s", first_point)
                    logger.debug("  Centerline last point: %s", last_point)
                    logger.debug("  Centerline length: %d points", len(centerline))
                    # Check comparison results
                    logger.debug("  first_matches_start: %s", first_matches_start)
                    logger.debug("  last_matches_end: %s", last_matches_end)
                    logger.debug("  first_matches_end: %s", first_matches_end)
                    logger.debug("  last_matches_start: %s", last_matches_start)
                    # Show a few centerline points for context
                    if len(centerline) > 2:
                        logger.debug("  First 3 centerline points: %s", centerline[:3])
                        logger.debug("  Last 3 centerline points: %s", centerline[-3:])

        # Create edge with all geometric properties
        branches_graph.add_edge(
            final_start_node,
            final_end_node,
            id=edge_counter,
            name=f"edge_{edge_counter}",
            centerline=centerline,
            radius=radius,
            length=length,
            min_radius=min(radius) if radius else 0.0,
            max_radius=max(radius) if radius else 0.0,
            mean_radius=np.mean(radius).item() if radius else 0.0,
        )
        edge_counter += 1

    logger.info(
        "Branch graph created with %d nodes and %d edges",
        branches_graph.number_of_nodes(),
        branches_graph.number_of_edges(),
    )

    # Verify that all edges were processed
    logger.info("Processed %d out of %d original edges", processed_edges, total_edges)

    # Track statistics
    if patient_stats_path and patient_id:
        stats = _get_construction_stats(branches_graph, original_skeleton)
        _append_construction_stats_to_csv(
            patient_stats_path, patient_id, "2_branch_graph", stats
        )
        if global_stats_path:
            _append_construction_stats_to_csv(
                global_stats_path, patient_id, "2_branch_graph", stats
            )

    return branches_graph


# =============================================================================
# VALIDATION AND QUALITY CONTROL
# =============================================================================


def verify_voxel_coverage(
    branch_graph: nx.Graph, original_skeleton: np.ndarray
) -> None:
    """
    Verify that all skeleton voxels are represented in the branch graph.

    Quality control function that validates the fidelity of the graph
    representation by checking coverage of original skeleton voxels.
    Critical for ensuring no anatomical information is lost during conversion.

    Args:
        branch_graph: Branch-level NetworkX graph to validate
        original_skeleton: Original binary skeleton array for comparison

    Returns:
        None (logs comprehensive coverage statistics)

    Algorithm:
        1. Count total voxels in original skeleton
        2. Extract all centerline points from branch graph edges
        3. Convert continuous coordinates to discrete voxel positions
        4. Count unique voxel positions covered
        5. Calculate and report coverage statistics

    Coverage Metrics:
        - Total original voxels
        - Voxels covered by branch graph
        - Missing voxels (original - covered)
        - Coverage percentage
    """
    original_voxel_count = np.sum(original_skeleton > 0)

    # Collect all voxel positions covered by branch graph centerlines
    covered_voxels = set()

    # Iterate through all edges and their centerlines
    for _, _, data in branch_graph.edges(data=True):
        centerline = data.get("centerline", [])
        for pos in centerline:
            # Convert continuous coordinates to discrete voxel coordinates
            voxel_pos = tuple(int(round(coord)) for coord in pos)

            # Ensure coordinates are within skeleton bounds
            if (
                0 <= voxel_pos[0] < original_skeleton.shape[0]
                and 0 <= voxel_pos[1] < original_skeleton.shape[1]
                and 0 <= voxel_pos[2] < original_skeleton.shape[2]
            ):
                covered_voxels.add(voxel_pos)

    # Calculate coverage statistics
    covered_count = len(covered_voxels)
    missing_count = original_voxel_count - covered_count
    coverage_percentage = (
        (covered_count / original_voxel_count) * 100 if original_voxel_count > 0 else 0
    )

    # Log coverage results
    logger.info("Voxel coverage analysis:")
    logger.info("  Original skeleton voxels: %d", original_voxel_count)
    logger.info("  Covered by branch graph: %d", covered_count)
    logger.info("  Missing voxels: %d", missing_count)
    logger.info("  Coverage: %.2f%%", coverage_percentage)

    # Report missing voxels
    if missing_count > 0:
        logger.info(
            "%d voxels from original skeleton missing in graph representation",
            missing_count,
        )
    else:
        logger.info("Perfect skeleton coverage achieved!")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def mask_to_branch_graph(
    skeleton: np.ndarray,
    spacing,
    distance_map: np.ndarray,
    patient_id: str | None = None,
    patient_stats_path: Path | None = None,
    global_stats_path: Path | None = None,
) -> nx.Graph:
    """
    Main entry point that orchestrates the complete conversion process
    from binary skeleton to mathematical graph representation.

    Args:
        skeleton: Binary 3D array representing vessel skeleton
        spacing: Voxel spacing in (x, y, z) dimensions
        distance_map: 3D array with distance values

    Returns:
        Branch-level NetworkX graph representing vascular tree structure

    Algorithm:
        1. Log original skeleton voxel count
        2. Convert skeleton to pixel-level graph
        3. Create branch-level graph from pixel graph
        4. Verify voxel coverage against original skeleton
        5. Return final branch graph
    """
    # Initialize stats CSVs if provided
    if patient_id and patient_stats_path:
        _initialize_construction_stats_csv(patient_stats_path, force=True)

    # For global CSV: only initialize once per Python session
    if patient_id and global_stats_path:
        global_path_str = str(global_stats_path.resolve())

        if global_path_str not in _initialized_global_csvs:
            # First time seeing this path - initialize it
            _initialize_construction_stats_csv(global_stats_path, force=True)
            _initialized_global_csvs.add(global_path_str)
            logger.info(
                "Initialized global construction stats CSV: %s", global_stats_path
            )

    # Log original skeleton voxel count
    skeleton_voxel_count = np.sum(skeleton > 0)
    logger.info("Original skeleton contains %d voxels", skeleton_voxel_count)

    # Convert skeleton to pixel-level graph (will track its own stats with voxel coverage)
    pixel_graph = skeleton_to_graph(
        skeleton,
        distance_map,
        spacing,
        patient_id,
        patient_stats_path,
        global_stats_path,
    )
    logger.info(
        "Pixel graph contains %d nodes, %d edges",
        pixel_graph.number_of_nodes(),
        pixel_graph.number_of_edges(),
    )

    # Create branch-level graph from pixel graph (will track its own stats with voxel coverage)
    branch_graph = create_branch_graph(
        pixel_graph, patient_id, patient_stats_path, global_stats_path, skeleton
    )

    # Verify voxel coverage against original skeleton (for logging only)
    verify_voxel_coverage(branch_graph, skeleton)

    return branch_graph
