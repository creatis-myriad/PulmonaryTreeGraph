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
            if not np.array_equal(
                edge_attrs["centerline"][0], graph.nodes[source]["pos"]
            ):
                edge_attrs["centerline"] = list(reversed(edge_attrs["centerline"]))
                edge_attrs["radius"] = list(reversed(edge_attrs["radius"]))
        except (ValueError, TypeError) as e:
            logger.warning("Failed to check centerline orientation: %s", e)

    return edge_attrs


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

    # Get positions
    start_position = np.array(graph.nodes[start_node]["pos"])
    end_position = np.array(graph.nodes[end_node]["pos"])

    logger.info("Starting orientation from edge %s", edge_name)
    logger.info(
        "Start node: %d at %s, End node: %d at %s, Mean radius: %.3f",
        start_node,
        start_position,
        end_node,
        end_position,
        mean_radius,
    )

    # Choose the endpoint node (degree 1) as the root for orientation
    if graph.degree(start_node) == 1:
        root = start_node
    else:
        root = end_node

    node_mapping: dict[int, int] = {}
    node_counter = {"count": 0}
    edge_counter = {"count": 1}  # Start edge IDs from 1

    new_graph = nx.DiGraph()
    stack = [root]
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        new_node_id = get_or_create_node_id(node, node_mapping, node_counter)
        add_node(new_graph, new_node_id, node, graph)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                new_neighbor_id = get_or_create_node_id(
                    neighbor, node_mapping, node_counter
                )
                add_node(new_graph, new_neighbor_id, neighbor, graph)
                edge_attrs = prepare_edge_attrs(graph, node, neighbor, edge_counter)
                new_graph.add_edge(new_node_id, new_neighbor_id, **edge_attrs)
                stack.append(neighbor)

    edges_processed = edge_counter["count"] - 1
    logger.info(
        "Graph orientation complete: %d nodes, %d edges (reassigned IDs 1-%d)",
        new_graph.number_of_nodes(),
        new_graph.number_of_edges(),
        edges_processed,
    )
    return new_graph
