"""
Graph construction module for building pulmonary vascular tree graphs.

This module provides functions to convert binary masks into skeletons and then into
NetworkX graphs representing the vascular tree structure.
"""

import logging

import networkx as nx
import numpy as np
from networkx import from_scipy_sparse_array
from skan.csr import skeleton_to_csgraph

logger = logging.getLogger(__name__)


def skeleton_to_graph(
    skeleton: np.ndarray, distance_map: np.ndarray, spacing: tuple[float, float, float]
) -> nx.Graph:
    """
    Convert a skeleton (binary 3D array) to a pixel-level NetworkX graph.
    Adds 'radius' property to each node from the distance map.
    """
    adjacency_matrix, pixel_coordinates = skeleton_to_csgraph(skeleton, spacing=spacing)
    graph = from_scipy_sparse_array(adjacency_matrix)
    pixel_coordinates = np.asarray(pixel_coordinates).T
    for node_id in graph.nodes:
        pos = pixel_coordinates[node_id].tolist()
        graph.nodes[node_id]["pos"] = pos
        graph.nodes[node_id]["radius"] = distance_map[pos[0], pos[1], pos[2]]
    return graph


def is_path_endpoint(graph: nx.Graph, node: int) -> bool:
    """
    Check if a node is an endpoint or branch point in the graph.

    A node is considered an endpoint if it has any number of neighbors
    other than exactly 2 (meaning it's either a terminal point with 1 neighbor
    or a branch point with 3+ neighbors).
    """
    return len(list(graph.neighbors(node))) != 2


def get_edge_data(graph: nx.Graph, start: int, end: int) -> tuple[list, list, float]:
    """Get centerline and radius data for edge."""
    edge = graph.edges[(start, end)]
    pos_start = graph.nodes[start]["pos"]
    pos_end = graph.nodes[end]["pos"]
    centerline = edge.get("centerline", [pos_start, pos_end])
    radius = edge.get(
        "radius",
        [graph.nodes[start].get("radius", 0.0), graph.nodes[end].get("radius", 0.0)],
    )
    length = edge.get("weight", 1.0)
    return centerline, radius, length


def follow_path(
    graph: nx.Graph, start: int, neighbor: int, visited: set
) -> tuple[int, list, list, float]:
    """
    Follow path until next endpoint, collecting data.

    Args:
        graph: Input graph
        start: Starting node
        neighbor: First neighbor to follow
        visited: Set of already visited (source, target) edges (modified in-place)

    Returns:
        Tuple of (end_node, centerline, radius, total_length)
    """
    centerline, radius, total_length = get_edge_data(graph, start, neighbor)
    current, prev = neighbor, start

    while not is_path_endpoint(graph, current):
        next_nodes = [n for n in graph.neighbors(current) if n != prev]
        if not next_nodes:
            break
        next_node = next_nodes[0]
        next_centerline, next_radius, length = get_edge_data(graph, current, next_node)
        centerline.extend(next_centerline[1:])
        radius.extend(next_radius[1:])
        total_length += length
        visited.add((prev, current))
        prev, current = current, next_node
    final_pos = graph.nodes[current]["pos"]
    if not np.array_equal(centerline[-1], final_pos):
        centerline.append(final_pos)
        radius.append(graph.nodes[current].get("radius", 0.0))
    visited.add((prev, current))
    return current, centerline, radius, total_length


def create_branch_graph(graph: nx.Graph) -> nx.Graph:
    """Merge pixel-level graph into branch-level graph."""
    branches_graph = nx.Graph()
    edge_counter = 1
    visited = set()

    logger.info("Branch graph created with %d nodes", branches_graph.number_of_nodes())

    endpoints = [n for n in graph.nodes() if is_path_endpoint(graph, n)]
    logger.info("Found %d endpoints/branch points", len(endpoints))

    for i, start_node in enumerate(endpoints):
        if i > 0 and i % 100 == 0:
            logger.debug("Processed %d/%d endpoints", i, len(endpoints))
        for neighbor in graph.neighbors(start_node):
            if (start_node, neighbor) in visited or (neighbor, start_node) in visited:
                continue
            end_node, centerline, radius, length = follow_path(
                graph, start_node, neighbor, visited
            )
            for node in (start_node, end_node):
                branches_graph.add_node(
                    node,
                    pos=graph.nodes[node]["pos"],
                    radius=graph.nodes[node].get("radius", 0.0),
                )
            branches_graph.add_edge(
                start_node,
                end_node,
                id=edge_counter,  # Assign ID starting from 1
                name=f"edge_{edge_counter}",
                centerline=centerline,
                radius=radius,
                length=length,
                min_radius=min(radius),
                max_radius=max(radius),
                mean_radius=float(np.mean(radius)) if radius else 0.0,
            )
            edge_counter += 1

    logger.info(
        "Branch graph created with %d nodes and %d edges",
        branches_graph.number_of_nodes(),
        branches_graph.number_of_edges(),
    )

    return branches_graph


def mask_to_branch_graph(
    skeleton: np.ndarray, spacing, distance_map: np.ndarray
) -> nx.Graph:
    """
    Complete pipeline: skeleton + distance map -> branch-level graph.

    This function performs the full conversion process:
    1. Converts skeleton to pixel-level graph with radius information
    2. Merges pixel connections into branch-level graph
    3. Adds metrics like radius, length and centerline to each branch

    Parameters:
    - skeleton: Binary 3D array representing vessel centerlines
    - spacing: Voxel spacing in mm (z,y,x)
    - distance_map: 3D array with distance values representing vessel radius

    Returns:
    - NetworkX graph with nodes at branch points and edges representing vessel segments
    """
    pixel_graph = skeleton_to_graph(skeleton, distance_map, spacing)
    branch_graph = create_branch_graph(pixel_graph)
    return branch_graph
