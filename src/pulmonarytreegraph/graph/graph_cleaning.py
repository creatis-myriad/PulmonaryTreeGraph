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
    """Remove nodes that have no connections in the graph."""
    graph.remove_nodes_from(list(nx.isolates(graph)))


def get_cycle_edges(graph: nx.Graph) -> list[list[tuple[int, int]]]:
    """Get all cycles as lists of edges."""
    cycles = nx.cycle_basis(graph)
    return [
        [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        for cycle in cycles
    ]


def detect_radius_pattern(
    radius_list: list[float], threshold_factor: float = 0.5
) -> bool:
    """Detect if a vessel segment shows a significant decrease then increase in radius."""
    if not radius_list or len(radius_list) < 3:
        return False

    try:
        min_value = min(radius_list)
        min_idx = np.argmin(radius_list)
    except (ValueError, TypeError) as e:
        logger.warning("Error analyzing radius list: %s", e)
        return False

    if min_idx == 0 or min_idx == len(radius_list) - 1:
        return False  # Min at endpoints is normal

    max_before = max(radius_list[:min_idx])
    max_after = max(radius_list[min_idx:])
    decrease_ratio = (max_before - min_value) / max_before if max_before > 0 else 0
    increase_ratio = (max_after - min_value) / min_value if min_value > 0 else 0

    return decrease_ratio > threshold_factor and increase_ratio > threshold_factor


def get_edge_data(graph: nx.Graph, node1: int, node2: int) -> dict | None:
    """Safely get edge data regardless of edge orientation."""
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
    """Check if a node is a valid candidate for fusion and return its neighbors."""
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
    """Combine two oriented edge segments, removing duplicate middle point."""
    if edge1_centerline.size > 0 and edge2_centerline.size > 0:
        # Remove duplicate middle point and concatenate
        combined_centerline = np.concatenate([edge1_centerline[:-1], edge2_centerline])
        combined_radius = np.concatenate([edge1_radius, edge2_radius])
        return combined_centerline.tolist(), combined_radius.tolist()
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
    """Create new edge data for the fused segment."""
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
    """Fuse two segments connected by a middle node."""
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

    if len(edge1_centerline) > 0 and len(edge2_centerline) > 0:
        # Convert to numpy arrays for manipulation
        edge1_centerline = np.array(edge1_centerline)
        edge2_centerline = np.array(edge2_centerline)
        edge1_radius = np.array(edge1_radius)
        edge2_radius = np.array(edge2_radius)
        middle_pos = np.array(middle_pos)
        neighbor1_pos = np.array(neighbor1_pos)
        neighbor2_pos = np.array(neighbor2_pos)

        # Orient edge1: should go neighbor1 -> middle
        if np.array_equal(edge1_centerline[0], neighbor1_pos) and np.array_equal(
            edge1_centerline[-1], middle_pos
        ):
            oriented_edge1_centerline = edge1_centerline
            oriented_edge1_radius = edge1_radius
        else:
            # Reverse it
            oriented_edge1_centerline = edge1_centerline[::-1]
            oriented_edge1_radius = edge1_radius[::-1]

        # Orient edge2: should go middle -> neighbor2
        if np.array_equal(edge2_centerline[0], middle_pos) and np.array_equal(
            edge2_centerline[-1], neighbor2_pos
        ):
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
        combined_radius = list(edge1_radius) + list(edge2_radius)

    # Create new edge data
    new_edge_data = _create_fused_edge_data(
        edge1_data, edge2_data, combined_centerline, combined_radius, edge_id
    )

    return new_edge_data, True


def fuse_linear_segments(graph: nx.Graph) -> nx.Graph:
    """
    Fuse linear segments where nodes have exactly 2 neighbors.
    Connects neighbor1 -> middle_node -> neighbor2 into neighbor1 -> neighbor2.
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
    """Determine if a terminal branch should be removed based on criteria."""
    length = edge_data.get("length", 0)
    parent_diameter = 2 * parent_radius
    radius_ratio = end_radius / parent_radius if parent_radius > 0 else 0

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
    """Find all terminal branches that should be removed (optimized)."""
    to_remove = []

    # Pre-filter terminal nodes for better performance
    terminal_nodes = [node for node in graph.nodes if graph.degree(node) == 1]

    for node in terminal_nodes:
        parent = next(graph.neighbors(node))
        edge_data = graph.edges[node, parent]
        parent_radius = graph.nodes[parent]["radius"]
        end_radius = graph.nodes[node]["radius"]

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
    """Remove nodes with single neighbors (optimized version)."""
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
    """Remove suspicious edges in cycles and return count."""
    removed_edges = []

    for cycle in get_cycle_edges(graph):
        for edge in cycle:
            edge_data = get_edge_data(graph, *edge)
            if edge_data and detect_radius_pattern(edge_data.get("radius", [])):
                if graph.has_edge(*edge):
                    removed_edges.append((edge, edge_data.get("id", "unknown")))
                    graph.remove_edge(*edge)

    if removed_edges:
        edge_ids = [str(edge_id) for _, edge_id in removed_edges]
        logger.info(
            "Removed %d suspicious cycle edges with IDs: %s",
            len(removed_edges),
            edge_ids,
        )

    return len(removed_edges)


def _remove_artefact_branches(
    graph: nx.Graph, radius_ratio_threshold: float, length_ratio_threshold: float
) -> int:
    """Remove terminal branches based on criteria and return count."""
    to_remove = _find_branches_to_remove(
        graph, radius_ratio_threshold, length_ratio_threshold
    )

    # Remove the marked branches
    removed_ids = []
    for node, parent, edge_id in to_remove:
        if graph.has_edge(node, parent):
            removed_ids.append(str(edge_id))
            graph.remove_edge(node, parent)

    if to_remove:
        logger.info(
            "Removed %d short/thin terminal branches with IDs: %s",
            len(to_remove),
            removed_ids,
        )

    return len(to_remove)


def _validate_graph_structure(graph: nx.Graph) -> bool:
    """Validate that graph has required attributes."""
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
    """Get basic graph statistics."""
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
    """Clean graph with standardized pipeline and comprehensive error handling."""
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
        _remove_suspicious_cycle_edges(graph)
        _remove_artefact_branches(graph, radius_ratio_threshold, length_ratio_threshold)
        graph = fuse_linear_segments(graph)
        graph = remove_single_neighbor_nodes(graph)
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

    except (nx.NetworkXError, AttributeError, TypeError, ValueError, KeyError) as e:
        logger.error("Error during graph cleaning: %s", e)
        return graph

    return graph
