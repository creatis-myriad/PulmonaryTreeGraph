"""
Module for orienting undirected graphs based on vessel properties.
Orients the graph by starting from the end branch with the maximum mean radius.
"""

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def find_main_end_branch_by_mean_radius(graph: nx.Graph) -> tuple[int, int, float, str]:
    """
    Find the end branch (edge connected to a node of degree 1) with the largest mean_radius.
    Returns (start_node, end_node, mean_radius, edge_name).
    """
    if graph.number_of_edges() == 0:
        raise ValueError("Graph has no edges.")

    max_branch = None
    max_mean_radius = -np.inf
    terminal_edges_found = 0

    for u, v, data in graph.edges(data=True):
        # Check if either u or v is an endpoint (degree 1)
        if graph.degree(u) == 1 or graph.degree(v) == 1:
            terminal_edges_found += 1
            mean_radius = data.get("mean_radius", 0.0)
            if mean_radius > max_mean_radius:
                max_mean_radius = mean_radius
                edge_name = data.get("name", f"edge_{u}_{v}")
                max_branch = (u, v, mean_radius, edge_name)

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


def get_or_create_node_id(
    node, node_mapping: dict[int, int], node_counter: dict[str, int]
) -> int:
    """Get existing or create new node ID."""
    if node not in node_mapping:
        node_mapping[node] = node_counter["count"]
        node_counter["count"] += 1
    return node_mapping[node]


def add_node(new_graph: nx.DiGraph, new_node_id: int, old_node: int, graph: nx.Graph):
    """Add node with attributes to new graph."""
    if new_node_id not in new_graph:
        new_graph.add_node(new_node_id, **graph.nodes[old_node])


def prepare_edge_attrs(
    graph: nx.Graph, source: int, target: int, edge_counter: dict[str, int]
) -> dict:
    """Prepare edge attributes including centerline orientation and reassign ID."""
    edge_attrs = graph.edges[source, target].copy()

    # Always reassign new ID starting from 1 during orientation
    edge_attrs["id"] = edge_counter["count"]
    edge_attrs["name"] = f"edge_{edge_counter['count']}"
    edge_counter["count"] += 1

    # Fix centerline orientation
    centerline = edge_attrs.get("centerline", [])
    if len(centerline) > 0:
        try:
            # Convert to numpy arrays for consistent comparison
            centerline_first = np.array(edge_attrs["centerline"][0])
            centerline_last = np.array(edge_attrs["centerline"][-1])
            source_pos = np.array(graph.nodes[source]["pos"])
            target_pos = np.array(graph.nodes[target]["pos"])

            # Check if centerline is correctly oriented (source -> target)
            first_matches_source = np.array_equal(centerline_first, source_pos)
            last_matches_target = np.array_equal(centerline_last, target_pos)

            # If both match, centerline is correctly oriented
            if first_matches_source and last_matches_target:
                logger.debug(f"Centerline already correct for edge {source}->{target}")

            # If both are reversed, fix the orientation
            elif np.array_equal(centerline_first, target_pos) and np.array_equal(
                centerline_last, source_pos
            ):
                logger.debug(f"Reversing centerline for edge {source}->{target}")
                edge_attrs["centerline"] = list(reversed(edge_attrs["centerline"]))
                if "radius" in edge_attrs:
                    edge_attrs["radius"] = list(reversed(edge_attrs["radius"]))

        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Failed to check centerline orientation: %s", e)

    return edge_attrs


def validate_centerline_orientations(graph: nx.DiGraph) -> list[str]:
    """
    Validate that centerlines flow from source to target in directed edges.
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
            first_matches_source = np.allclose(first_point, source_pos, atol=1e-6)
            last_matches_target = np.allclose(last_point, target_pos, atol=1e-6)

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
    Validate that parent's last centerline point equals child's first centerline point.
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

                # The first point of this edge should match the node's position
                child_first_point = np.array(centerline[0])
                node_pos = np.array(graph.nodes[node]["pos"])

                if not np.array_equal(child_first_point, node_pos):
                    issues.append(
                        f"Child edge {node}->{child}: first centerline point {child_first_point} != parent node pos {node_pos}"
                    )

                # The last point of this edge should match the child node's position
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
    Validate that the directed tree is correctly oriented from root to leaves.

    Returns:
        tuple: (is_valid, list_of_issues)
    """
    issues = []

    # Check if it's a tree (connected and n-1 edges)
    if not nx.is_weakly_connected(graph):
        issues.append("Graph is not connected")

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

    # Validate centerline orientations flow from parent to child
    centerline_issues = validate_centerline_orientations(graph)
    issues.extend(centerline_issues)

    # Validate parent-child centerline connectivity
    connectivity_issues = validate_parent_child_centerline_connectivity(graph)
    issues.extend(connectivity_issues)

    return len(issues) == 0, issues


def ensure_proper_tree_orientation(graph: nx.Graph, root: int) -> nx.DiGraph:
    """
    Create a properly oriented tree using BFS to ensure parent->child direction.
    """
    node_mapping: dict[int, int] = {}
    node_counter = {"count": 0}
    edge_counter = {"count": 1}

    new_graph = nx.DiGraph()

    # IMPORTANT: Ensure root gets ID 0
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

    # Create directed graph with proper orientation (skip root since already added)
    for node in visited:
        if node != root:  # Skip root, already added
            new_node_id = get_or_create_node_id(node, node_mapping, node_counter)
            add_node(new_graph, new_node_id, node, graph)

    # Add edges from parent to child
    for child, parent in parent_map.items():
        if parent is not None:  # Skip root
            parent_id = node_mapping[parent]
            child_id = node_mapping[child]
            edge_attrs = prepare_edge_attrs(graph, parent, child, edge_counter)
            new_graph.add_edge(parent_id, child_id, **edge_attrs)

    return new_graph


def orient_graph(graph: nx.Graph) -> nx.DiGraph:
    """
    Orient an undirected vessel graph into a directed graph,
    starting from the end branch with the maximum mean radius.
    Reassigns edge IDs starting from 1.

    Returns:
    - oriented_graph: The directed graph with new IDs starting from 1
    """
    if graph is None:
        logger.error("None graph provided to orient_graph")
        return nx.DiGraph()

    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph provided for orientation")
        return nx.DiGraph()

    if graph.number_of_edges() == 0:
        logger.warning("Graph has no edges to orient")
        return nx.DiGraph()

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

    # Choose the endpoint node (degree 1) as the root for orientation
    if graph.degree(start_node) == 1:
        root = start_node
    else:
        root = end_node

    logger.info("Starting orientation from edge %s with root node %d", edge_name, root)

    # Use improved orientation method
    new_graph = ensure_proper_tree_orientation(graph, root)

    # Validate the orientation - root should always be ID 0
    root_id = 0  # Root always gets ID 0 in our mapping
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

        # Show nodes with 0 incoming edges
        zero_in_degree = [n for n in new_graph.nodes() if new_graph.in_degree(n) == 0]
        logger.error(f"  - Nodes with 0 incoming edges: {zero_in_degree}")

    else:
        logger.info("Tree orientation validation passed")

    edges_processed = new_graph.number_of_edges()
    logger.info(
        "Graph orientation complete: %d nodes, %d edges",
        new_graph.number_of_nodes(),
        edges_processed,
    )
    return new_graph
