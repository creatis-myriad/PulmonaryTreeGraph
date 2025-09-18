"""
Module for orienting undirected graphs based on vessel properties.

This module provides functionality to convert undirected vascular tree graphs
into directed trees with proper parent-child orientation, starting from the
largest terminal vessel as the root.
"""

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ROOT SELECTION UTILITIES
# =============================================================================


def find_main_end_branch_by_mean_radius(graph: nx.Graph) -> tuple[int, int, float, str]:
    """
    Identify the terminal branch with the largest mean radius to serve as tree root.

    This function finds the most significant terminal vessel (connected to a degree-1
    node) based on mean radius, which typically represents the main trunk or
    primary vessel that should serve as the root of the oriented tree.

    Args:
        graph: Undirected NetworkX graph representing the vascular tree

    Returns:
        tuple: (start_node, end_node, mean_radius, edge_name)

    Algorithm:
        1. Scan all edges to identify terminal branches (connected to degree-1 nodes)
        2. Compare mean_radius values across terminal branches
        3. Select branch with maximum mean_radius as root candidate
        4. Return branch endpoints and metadata
    """
    if graph.number_of_edges() == 0:
        raise ValueError("Graph has no edges.")

    max_branch = None
    max_mean_radius = -np.inf
    terminal_edges_found = 0

    # Scan all edges for terminal branches
    for u, v, data in graph.edges(data=True):
        # Check if either u or v is an endpoint (degree 1)
        if graph.degree(u) == 1 or graph.degree(v) == 1:
            terminal_edges_found += 1
            mean_radius = data.get("mean_radius", 0.0)

            # Track branch with maximum mean radius
            if mean_radius > max_mean_radius:
                max_mean_radius = mean_radius
                edge_name = data.get("name", f"edge_{u}_{v}")
                max_branch = (u, v, mean_radius, edge_name)

    # Validate results
    if max_branch is None:
        if terminal_edges_found == 0:
            raise ValueError(
                "No terminal branches found in the graph (all nodes have degree > 1)."
            )
        else:
            raise ValueError("Terminal branches found but no valid mean_radius data.")

    logger.debug(
        "Found %d terminal edges, selected edge with mean_radius %.3f",
        terminal_edges_found,
        max_mean_radius,
    )
    return max_branch


# =============================================================================
# GRAPH CONSTRUCTION UTILITIES
# =============================================================================


def get_or_create_node_id(
    node, node_mapping: dict[int, int], node_counter: dict[str, int]
) -> int:
    """
    Get existing node ID or create new sequential ID for graph construction.

    This function maintains a mapping from original node IDs to new sequential
    IDs (starting from 0) for the oriented graph. Ensures consistent ID
    assignment across the orientation process.

    Args:
        node: Original node ID from source graph
        node_mapping: Dictionary mapping original IDs to new IDs
        node_counter: Counter dictionary with 'count' key for next available ID

    Returns:
        int: New sequential node ID (0-based)
    """
    if node not in node_mapping:
        node_mapping[node] = node_counter["count"]
        node_counter["count"] += 1
    return node_mapping[node]


def add_node(new_graph: nx.DiGraph, new_node_id: int, old_node: int, graph: nx.Graph):
    """
    Add node with all attributes to the new directed graph.

    Copies a node from the original undirected graph to the new directed graph
    with a new ID, preserving all node attributes (position, radius, etc.).

    Args:
        new_graph: Target directed graph being constructed
        new_node_id: New sequential ID for the node
        old_node: Original node ID from source graph
        graph: Source undirected graph containing original node data
    """
    if new_node_id not in new_graph:
        new_graph.add_node(new_node_id, **graph.nodes[old_node])


def prepare_edge_attrs(
    graph: nx.Graph, source: int, target: int, edge_counter: dict[str, int]
) -> dict:
    """
    Prepare comprehensive edge attributes with proper centerline orientation.

    This function copies edge attributes from the original graph, reassigns
    sequential IDs starting from 1, and ensures centerline geometry flows
    correctly from source to target node.

    Args:
        graph: Source undirected graph containing original edge
        source: Source node ID in the directed edge
        target: Target node ID in the directed edge
        edge_counter: Counter dictionary for sequential edge ID assignment

    Returns:
        dict: Complete edge attributes dictionary with:
              - id: Sequential ID starting from 1
              - name: Updated edge name
              - centerline: Properly oriented 3D coordinate sequence
              - radius: Corresponding radius values (oriented with centerline)
              - Other attributes: Copied from original edge

    Centerline Orientation Logic:
        1. Check if first point matches source position and last matches target
        2. If correctly oriented: no changes needed
        3. If reversed: reverse both centerline and radius arrays
        4. If ambiguous: log warning but proceed
        5. Return updated edge attributes
    """
    edge_attrs = graph.edges[source, target].copy()

    # Reassign sequential ID starting from 1
    edge_attrs["id"] = edge_counter["count"]
    edge_attrs["name"] = f"edge_{edge_counter['count']}"
    edge_counter["count"] += 1

    # Ensure centerline flows from source to target
    centerline = edge_attrs.get("centerline", [])
    if len(centerline) > 0:
        try:
            # Convert positions to numpy arrays for comparison
            centerline_first = np.array(edge_attrs["centerline"][0])
            centerline_last = np.array(edge_attrs["centerline"][-1])
            source_pos = np.array(graph.nodes[source]["pos"])
            target_pos = np.array(graph.nodes[target]["pos"])

            # Check if centerline is correctly oriented (source -> target)
            first_matches_source = np.array_equal(centerline_first, source_pos)
            last_matches_target = np.array_equal(centerline_last, target_pos)

            if first_matches_source and last_matches_target:
                # Already correctly oriented
                logger.debug(f"Centerline already correct for edge {source}->{target}")

            elif np.array_equal(centerline_first, target_pos) and np.array_equal(
                centerline_last, source_pos
            ):
                # Needs reversal
                logger.debug(f"Reversing centerline for edge {source}->{target}")
                edge_attrs["centerline"] = list(reversed(edge_attrs["centerline"]))
                if "radius" in edge_attrs:
                    edge_attrs["radius"] = list(reversed(edge_attrs["radius"]))
            else:
                # Ambiguous case - log warning
                logger.warning(
                    f"Ambiguous centerline orientation for edge {source}->{target}"
                )

        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Failed to check centerline orientation: %s", e)

    return edge_attrs


# =============================================================================
# TREE ORIENTATION ALGORITHM
# =============================================================================


def ensure_proper_tree_orientation(graph: nx.Graph, root: int) -> nx.DiGraph:
    """
    Create properly oriented directed tree using breadth-first traversal.

    This function converts an undirected graph into a directed tree with
    consistent parent→child orientation. Uses BFS to establish parent-child
    relationships and ensures proper edge direction throughout the tree.

    Args:
        graph: Source undirected graph to orient
        root: Node to use as tree root (receives ID 0 in new graph)

    Returns:
        nx.DiGraph: Directed tree with proper orientation and sequential IDs

    Algorithm:
        1. Initialize root node with ID 0
        2. Use BFS to establish parent-child relationships:
           - Visit neighbors in breadth-first order
           - Mark parent for each visited node
           - Build complete parent map
        3. Create directed edges from parent to child
        4. Orient centerlines to match edge direction
        5. Assign sequential IDs to nodes (0-based) and edges (1-based)
    """
    node_mapping: dict[int, int] = {}
    node_counter = {"count": 0}
    edge_counter = {"count": 1}

    new_graph = nx.DiGraph()

    # Root always gets ID 0
    root_id = get_or_create_node_id(root, node_mapping, node_counter)
    add_node(new_graph, root_id, root, graph)

    # BFS to ensure proper parent-child relationships
    queue = [root]
    visited = set([root])
    parent_map = {root: None}

    # Build parent-child relationships
    while queue:
        current = queue.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current
                queue.append(neighbor)

    # Add all nodes (except root, already added)
    for node in visited:
        if node != root:  # Skip root, already added
            new_node_id = get_or_create_node_id(node, node_mapping, node_counter)
            add_node(new_graph, new_node_id, node, graph)

    # Add directed edges from parent to child
    for child, parent in parent_map.items():
        if parent is not None:  # Skip root
            parent_id = node_mapping[parent]
            child_id = node_mapping[child]
            edge_attrs = prepare_edge_attrs(graph, parent, child, edge_counter)
            new_graph.add_edge(parent_id, child_id, **edge_attrs)

    return new_graph


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_centerline_orientations(graph: nx.DiGraph) -> list[str]:
    """
    Validate that all centerlines flow correctly from source to target.

    Quality control function that verifies geometric consistency of centerlines
    with edge directions.

    Args:
        graph: Directed graph to validate

    Returns:
        list: Issues found (empty list if all centerlines properly oriented)

    Validation Checks:
        - First centerline point matches source node position
        - Last centerline point matches target node position
        - Both conditions must be satisfied for each directed edge
    """
    issues = []

    for source, target, data in graph.edges(data=True):
        centerline = data.get("centerline", [])
        if len(centerline) == 0:
            continue

        try:
            centerline_array = np.asarray(centerline)
            source_pos = np.array(graph.nodes[source]["pos"])
            target_pos = np.array(graph.nodes[target]["pos"])

            first_point = np.array(centerline_array[0])
            last_point = np.array(centerline_array[-1])

            # Check if centerline flows from source to target
            first_matches_source = np.array_equal(first_point, source_pos)
            last_matches_target = np.array_equal(last_point, target_pos)

            if not (first_matches_source and last_matches_target):
                issues.append(
                    f"Edge {source}->{target}: centerline doesn't flow from source to target"
                )

        except Exception as e:
            issues.append(
                f"Edge {source}->{target}: failed to validate centerline - {e}"
            )

    return issues


def validate_parent_child_centerline_connectivity(graph: nx.DiGraph) -> list[str]:
    """
    Validate geometric continuity between connected vessel segments.

    Ensures that vessel segments connect properly at branch points by
    verifying that parent centerlines end where child centerlines begin.

    Args:
        graph: Directed graph to validate

    Returns:
        list: Connectivity issues found (empty if all connections valid)

    Validation Logic:
        For each parent-child edge pair:
        1. Child's first centerline point should match parent node position
        2. Child's last centerline point should match child node position
        3. This ensures geometric flow from parent through child
    """
    issues = []

    for node in graph.nodes():
        # Get all outgoing edges (children) from this node
        for child in graph.successors(node):
            try:
                # Get the edge data from parent to child
                edge_data = graph.edges[node, child]
                centerline = edge_data.get("centerline", [])

                if len(centerline) == 0:
                    continue

                # First point should match parent node position
                child_first_point = np.array(centerline[0])
                node_pos = np.array(graph.nodes[node]["pos"])

                if not np.array_equal(child_first_point, node_pos):
                    issues.append(
                        f"Child edge {node}->{child}: first centerline point {child_first_point} != parent node pos {node_pos}"
                    )

                # Last point should match child node position
                child_last_point = np.array(centerline[-1])
                child_node_pos = np.array(graph.nodes[child]["pos"])

                if not np.array_equal(child_last_point, child_node_pos):
                    issues.append(
                        f"Child edge {node}->{child}: last centerline point {child_last_point} != child node pos {child_node_pos}"
                    )

            except Exception as e:
                issues.append(
                    f"Failed to validate connectivity for edge {node}->{child}: {e}"
                )

    return issues


def validate_tree_orientation(
    graph: nx.DiGraph, root_id: int
) -> tuple[bool, list[str]]:
    """
    Comprehensive validation of directed tree structure and geometric consistency.

    Performs complete validation of the oriented tree including topological
    structure, orientation correctness, and geometric continuity. Essential
    quality control before using graph for analysis.

    Args:
        graph: Directed graph to validate
        root_id: Expected root node ID (should be 0)

    Returns:
        tuple: (is_valid: bool, issues: list[str])
               is_valid is True only if all validations pass
               issues contains detailed descriptions of any problems found

    Validation Categories:
        1. **Topological Structure**:
           - Graph is weakly connected
           - Has n-1 edges for n nodes (tree property)
           - Root has no incoming edges
           - All non-root nodes have exactly 1 incoming edge
           - At least one leaf node exists

        2. **Geometric Consistency**:
           - All centerlines flow from source to target
           - Parent-child centerlines connect properly at branch points
    """
    issues = []

    # ---Topological validation ---
    # Check connectivity
    if not nx.is_weakly_connected(graph):
        issues.append("Graph is not connected")

    # Check tree property: edges = nodes - 1
    if graph.number_of_edges() != graph.number_of_nodes() - 1:
        issues.append(
            f"Not a tree: {graph.number_of_edges()} edges for {graph.number_of_nodes()} nodes"
        )

    # Check root has no incoming edges
    if graph.in_degree(root_id) != 0:
        issues.append(f"Root node {root_id} has incoming edges")

    # Check all non-root nodes have exactly one incoming edge
    for node in graph.nodes():
        if node != root_id and graph.in_degree(node) != 1:
            issues.append(
                f"Non-root node {node} has {graph.in_degree(node)} incoming edges (should be 1)"
            )

    # Check leaves have no outgoing edges
    leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    if len(leaves) == 0:
        issues.append("No leaf nodes found")

    # --- Geometric validation ---
    # Validate centerline orientations flow from parent to child
    centerline_issues = validate_centerline_orientations(graph)
    issues.extend(centerline_issues)

    # Validate parent-child centerline connectivity
    connectivity_issues = validate_parent_child_centerline_connectivity(graph)
    issues.extend(connectivity_issues)

    return len(issues) == 0, issues


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def orient_graph(graph: nx.Graph) -> nx.DiGraph:
    """
    Convert undirected vascular graph to properly oriented directed tree.

    This is the main entry point for graph orientation. It identifies the
    optimal root node, performs the orientation transformation, and validates
    the result. Creates a directed tree suitable for flow analysis and
    hierarchical vascular structure analysis.

    Args:
        graph: Undirected NetworkX graph representing vascular tree

    Returns:
        nx.DiGraph: Properly oriented directed tree with:
                   - Root at ID 0 (largest terminal vessel)
                   - Sequential node IDs starting from 0
                   - Sequential edge IDs starting from 1
                   - Consistent parent→child orientation
                   - Properly oriented centerlines and radii

    Algorithm:
        1. **Root Selection**: Find terminal branch with maximum mean radius
        2. **Root Identification**: Choose degree-1 endpoint as tree root
        3. **Orientation**: Use BFS to create consistent parent-child structure
        4. **Validation**: Comprehensive quality control of result
    """
    # Input validation
    if graph is None:
        logger.error("None graph provided to orient_graph")
        return nx.DiGraph()

    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph provided for orientation")
        return nx.DiGraph()

    if graph.number_of_edges() == 0:
        logger.warning("Graph has no edges to orient")
        return nx.DiGraph()

    # Find optimal root branch
    try:
        (
            start_node,
            end_node,
            mean_radius,
            edge_name,
        ) = find_main_end_branch_by_mean_radius(graph)
    except ValueError as e:
        logger.error("Failed to find starting branch for orientation: %s", e)
        return nx.DiGraph()

    # Select root node (degree-1 endpoint)
    root = start_node if graph.degree(start_node) == 1 else end_node

    logger.info(
        "Starting orientation from edge %s (mean_radius=%.3f) with root node %d",
        edge_name,
        mean_radius,
        root,
    )

    # Perform orientation
    new_graph = ensure_proper_tree_orientation(graph, root)

    # Validation (root always gets ID 0)
    root_id = 0
    logger.info(f"Validating tree with root_id={root_id} (original root was {root})")

    is_valid, issues = validate_tree_orientation(new_graph, root_id)

    if not is_valid:
        logger.error("Tree orientation validation failed:")
        for issue in issues:
            logger.error("  - %s", issue)

        # Debug information
        logger.error("Debug info:")
        logger.error(
            f"  - Graph has {new_graph.number_of_nodes()} nodes, {new_graph.number_of_edges()} edges"
        )
        logger.error(
            f"  - Root node {root_id} in_degree: {new_graph.in_degree(root_id)}"
        )
        logger.error(
            f"  - Root node {root_id} out_degree: {new_graph.out_degree(root_id)}"
        )

        # Identify nodes with incorrect in-degree
        zero_in_degree = [n for n in new_graph.nodes() if new_graph.in_degree(n) == 0]
        logger.error(f"  - Nodes with 0 incoming edges: {zero_in_degree}")

    else:
        logger.info("Tree orientation validation passed")

    # Final statistics
    edges_processed = new_graph.number_of_edges()
    logger.info(
        "Graph orientation complete: %d nodes, %d edges",
        new_graph.number_of_nodes(),
        edges_processed,
    )
    return new_graph
