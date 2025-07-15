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
    """Find all terminal branches that should be removed (optimized)."""
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
    """Count terminal vessels (degree 1) with large mean diameter."""
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
    """Get terminal vessels (degree 1) with large mean diameter and their diameters."""
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
    """Filter removal list to protect the largest terminal vessel."""
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
    """Remove terminal branches based on criteria and return count, with safety check for large terminals."""
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
