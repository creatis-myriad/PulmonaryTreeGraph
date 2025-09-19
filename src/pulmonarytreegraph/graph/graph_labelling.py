"""
Module for labeling pulmonary vascular tree graphs with anatomical information.

This module provides functions to label the main pulmonary artery (MPA), right pulmonary artery (RPA),
left pulmonary artery (LPA), laterality, and lobe information on a directed graph representation.
It uses breadth-first search (BFS) to traverse the graph and assign labels based on spatial relationships
with provided lung masks. The hierarchy of vessel levels is also established, with MPA as level 1, RPA and LPA as level 2,
lobar vessels as level 3, segmental vessels as level 4, and subsequent branches assigned increasing levels.
Vessels connecting multiple lobes are marked as interlobar.
"""

import logging
from collections import defaultdict, deque

import networkx as nx

logger = logging.getLogger(__name__)


# =============================================================================
# BASIC UTILITIES
# =============================================================================


def _is_in_mask(pos: list[int], mask) -> bool:
    """
    Check if a 3D position is within a binary mask.

    Args:
        pos: 3D coordinate [x, y, z] in voxel space
        mask: Binary 3D array representing anatomical region

    Returns:
        bool: True if position is within mask (value == 1), False otherwise
    """
    return mask[pos[0], pos[1], pos[2]] == 1


def _normalize_lobe_to_list(lobe_data) -> list:
    """
    Convert lobe data to consistent list format for processing.

    Args:
        lobe_data: Lobe information in various formats (string, list, etc.)

    Returns:
        list: Normalized lobe data as list
    """
    if isinstance(lobe_data, list):
        return lobe_data
    else:
        return [lobe_data]


def _is_unknown_lobe(lobe_data) -> bool:
    """
    Check if lobe data represents unknown/unlabeled state.

    Args:
        lobe_data: Lobe information to check (list or string format)

    Returns:
        bool: True if lobe is unknown/unlabeled, False otherwise
    """
    if isinstance(lobe_data, list):
        return len(lobe_data) == 0 or (
            len(lobe_data) == 1 and lobe_data[0] == "unknown"
        )
    else:
        return lobe_data == "unknown"


# =============================================================================
# ANATOMICAL CLASSIFICATION
# =============================================================================


def _determine_laterality(graph: nx.DiGraph, start_node: int, lungs: dict) -> str:
    """
    Determine if a vessel branch goes to right or left lung using BFS traversal.

    Args:
        graph: Directed graph representing vascular tree
        start_node: Starting node for laterality determination
        lungs: Dictionary containing 'right' and 'left' lung masks

    Returns:
        str: 'right', 'left', or 'unknown' laterality

    Algorithm:
        Uses breadth-first search to traverse descendants until finding
        a node within either lung mask, limiting search for performance.
    """
    queue = deque([start_node])
    visited = set([start_node])
    nodes_checked = 0

    while queue and nodes_checked < 50:  # Limit search for performance
        node = queue.popleft()
        pos = graph.nodes[node]["pos"]
        nodes_checked += 1

        if _is_in_mask(pos, lungs["right"]):
            return "right"
        elif _is_in_mask(pos, lungs["left"]):
            return "left"

        # Add unvisited successors to queue
        for successor in graph.successors(node):
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)

    return "unknown"


def _get_endpoint_lobe(graph: nx.DiGraph, node: int, lobe_masks: dict) -> str:
    """
    Determine which anatomical lobe contains a terminal vessel endpoint.

    Foundation for anatomical labeling - endpoint classification
    propagates up through vessel tree.

    Args:
        graph: Directed graph representing vascular tree
        node: Terminal node to classify
        lobe_masks: Dictionary of lobe name -> binary mask mappings

    Returns:
        str: Lobe name ('RUL', 'RML', 'RLL', 'LUL', 'LLL') or 'unknown'

    Algorithm:
        1. Check endpoint position directly in lobe masks
        2. If not found, check beginning of terminal branch
        3. Use priority order for common lobes

    """
    # Check most common lobes first for better average performance
    priority_order = ["RUL", "LUL", "RLL", "LLL", "RML"]

    # First, check if the endpoint itself is in any lobe
    endpoint_pos = graph.nodes[node]["pos"]
    for lobe_name in priority_order:
        if lobe_name in lobe_masks and _is_in_mask(endpoint_pos, lobe_masks[lobe_name]):
            logger.debug(f"Endpoint {node} found in lobe {lobe_name}")
            return lobe_name

    # If endpoint is not in any lobe mask, check the beginning node of the end branch
    predecessors = list(graph.predecessors(node))
    if predecessors:
        # Get the beginning node (source of the end branch)
        beginning_node = predecessors[0]
        beginning_pos = graph.nodes[beginning_node]["pos"]

        # Check if beginning node is in any lobe
        for lobe_name in priority_order:
            if lobe_name in lobe_masks and _is_in_mask(
                beginning_pos, lobe_masks[lobe_name]
            ):
                logger.debug(
                    f"Endpoint {node} not in lobe, but beginning node {beginning_node} is in {lobe_name}"
                )
                return lobe_name

    logger.debug(f"Neither endpoint {node} nor its beginning node are in any lobe mask")
    return "unknown"


# =============================================================================
# HIERARCHY ESTABLISHMENT
# =============================================================================


def _label_main_artery(graph: nx.DiGraph) -> None:
    """
    Sets first edge from root as level 1 with name 'MPA',
    lateral='None', and lobe=['None'].
    All other levels derive from this primary vessel classification.

    Args:
        graph: Directed graph to label (modified in-place)
    """
    root_node = 0
    successors = list(graph.successors(root_node))
    if not successors:
        logger.warning("No successors found for root node")
        return

    first_bifurcation = successors[0]

    graph.edges[root_node, first_bifurcation]["level"] = 1
    graph.edges[root_node, first_bifurcation]["name"] = "MPA"
    graph.edges[root_node, first_bifurcation]["lateral"] = "None"
    graph.edges[root_node, first_bifurcation]["lobe"] = ["None"]

    logger.info("Main pulmonary artery labeled")


def _propagate_laterality(graph: nx.DiGraph, start_node: int, lateral: str) -> None:
    """
    Propagate laterality information to all descendant vessels using BFS traversal.

    Critical for establishing consistent left/right classification
    throughout vessel tree after RPA/LPA identification. Ensures
    all downstream vessels inherit proper laterality from their
    primary branch (RPA or LPA).

    Args:
        graph: Directed graph to label (modified in-place)
        start_node: Node to begin laterality propagation from
        lateral: Laterality value ('right' or 'left') to propagate

    Algorithm:
        Uses breadth-first search to traverse all descendants and assign
        laterality to edges that don't already have this attribute set.

    """
    queue = deque([start_node])
    visited = set([start_node])

    while queue:
        node = queue.popleft()

        for successor in graph.successors(node):
            if "lateral" not in graph.edges[node, successor]:
                graph.edges[node, successor]["lateral"] = lateral

            if successor not in visited:
                visited.add(successor)
                queue.append(successor)


def _label_laterality_and_bifurcation(graph: nx.DiGraph, lungs: dict) -> None:
    """
    Label right and left pulmonary arteries (RPA/LPA) as level 2 vessels.

    Args:
        graph: Directed graph to label (modified in-place)
        lungs: Dictionary containing right and left lung masks

    Algorithm:
        1. Find MPA termination (first bifurcation node)
        2. For each branch, determine laterality using lung masks
        3. Label as RPA or LPA with level 2
        4. Propagate laterality to all descendants using _propagate_laterality
    """
    # Find the first bifurcation node
    mpa_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("level") == 1]
    if not mpa_edges:
        logger.warning("No MPA edge found")
        return

    _, bifurcation_node = mpa_edges[0]

    # Label RPA and LPA
    for successor in graph.successors(bifurcation_node):
        lateral = _determine_laterality(graph, successor, lungs)
        vessel_name = "RPA" if lateral == "right" else "LPA"

        graph.edges[bifurcation_node, successor]["level"] = 2
        graph.edges[bifurcation_node, successor]["name"] = vessel_name
        graph.edges[bifurcation_node, successor]["lateral"] = lateral
        graph.edges[bifurcation_node, successor]["lobe"] = ["None"]

        # Propagate laterality to all descendants
        _propagate_laterality(graph, successor, lateral)

    logger.info("Laterality and first bifurcation labeled")


def _label_remaining_as_level3(graph: nx.DiGraph) -> None:
    """
    Label all unlabeled vessels as level 3 in batch operation.
    Level 3 represents initial classification before hierarchical rules.

    Args:
        graph: Directed graph to label (modified in-place)
    """
    unlabeled_edges = []

    for u, v, data in graph.edges(data=True):
        if "level" not in data:
            unlabeled_edges.append((u, v))

    # Batch update
    for u, v in unlabeled_edges:
        graph.edges[u, v]["level"] = 3
        graph.edges[u, v]["name"] = f"Level3_{u}_{v}"
        graph.edges[u, v]["lobe"] = []

    logger.info("Labeled %d remaining vessels as level 3", len(unlabeled_edges))


# =============================================================================
# LOBE CLASSIFICATION
# =============================================================================


def _trace_back_and_label(graph: nx.DiGraph, start_node: int, lobe: str) -> None:
    """
    Trace from endpoint back to level 2, adding lobe information to vessel path.

    Args:
        graph: Directed graph to label (modified in-place)
        start_node: Terminal node to trace backward from
        lobe: Lobe name to propagate along the path

    Algorithm:
        1. Start at terminal node and follow predecessors
        2. Add lobe to edge lobe list (maintaining list format)
        3. Stop when reaching level 2 (RPA/LPA) or root
        4. Skip edges already containing the lobe

    Note:
        Core mechanism for propagating anatomical information from
        endpoints through vessel tree to establish lobe territories.
    """
    current = start_node
    visited = set()

    while current is not None and current not in visited:
        visited.add(current)

        # Get predecessors (should be exactly one in a tree, except for root)
        predecessors = list(graph.predecessors(current))
        if not predecessors:
            break  # Reached root

        parent = predecessors[0]

        # Get edge data
        if graph.has_edge(parent, current):
            edge_data = graph.edges[parent, current]
            current_level = edge_data.get("level", 1)

            # Only add lobe to edges at level 3 and above (not level 1 MPA or level 2 RPA/LPA)
            if current_level > 2:
                # Initialize lobe list if it doesn't exist or is not a list
                current_lobe = edge_data.get("lobe", "unknown")
                if not isinstance(current_lobe, list):
                    if current_lobe in ["unknown", "None"]:
                        edge_data["lobe"] = []
                    else:
                        edge_data["lobe"] = [current_lobe]

                # Add the lobe if it's not already present
                if lobe not in edge_data["lobe"]:
                    edge_data["lobe"].append(lobe)
                    logger.debug(
                        f"Added lobe {lobe} to edge {parent}->{current} (level {current_level})"
                    )

            # Stop if we reach level 2
            if current_level <= 2:
                break

        current = parent


def _find_sibling_lobe(graph: nx.DiGraph, u: int, v: int) -> tuple[str, list[str]]:
    """
    Determine vessel lobe by analyzing parent and sibling vessel information.

    Args:
        graph: Directed graph with partial lobe labeling
        u: Source node of edge to classify
        v: Target node of edge to classify

    Returns:
        tuple: (primary_lobe_label, all_contributing_lobes_list)

    Algorithm:
        1. Get parent edge lobe information
        2. Collect sibling edge lobe data at same level
        3. Handle interlobar vs single-lobe consensus
        4. Return primary classification and contributing lobe list
    """
    # Get the parent of edge u->v
    parents = list(graph.predecessors(u))
    if not parents:
        return "unknown", []

    parent = parents[0]  # Should only have one parent in a tree

    # Get all children (siblings) of the parent
    siblings = list(graph.successors(parent))
    all_lobes = []
    interlobar_lobes = []  # Track lobes from interlobar vessels

    current_level = graph.edges[u, v].get("level", 3)

    # 1. First, check the parent edge's lobe information
    parent_is_interlobar = False
    grandparents = list(graph.predecessors(parent))
    if grandparents:
        grandparent = grandparents[0]
        if graph.has_edge(grandparent, parent):
            parent_edge_data = graph.edges[grandparent, parent]
            parent_lobe = _normalize_lobe_to_list(parent_edge_data.get("lobe", []))

            if len(parent_lobe) > 1:
                parent_is_interlobar = True
                interlobar_lobes.extend(parent_lobe)
            elif len(parent_lobe) == 1 and parent_lobe[0] not in ["unknown", "None"]:
                all_lobes.append(parent_lobe[0])

    # 2. Then check sibling information
    sibling_single_lobes = []  # Track single-lobe siblings separately
    for sibling in siblings:
        if sibling != u:  # Don't include the current edge's target
            if graph.has_edge(parent, sibling):
                sibling_edge_data = graph.edges[parent, sibling]
                sibling_lobe = _normalize_lobe_to_list(
                    sibling_edge_data.get("lobe", [])
                )
                sibling_level = sibling_edge_data.get("level", 3)

                # Only consider siblings at the same level
                if sibling_level == current_level:
                    if len(sibling_lobe) > 1:
                        interlobar_lobes.extend(sibling_lobe)
                    elif len(sibling_lobe) == 1 and sibling_lobe[0] not in [
                        "unknown",
                        "None",
                    ]:
                        all_lobes.append(sibling_lobe[0])
                        sibling_single_lobes.append(sibling_lobe[0])

    # 3. Special case: Parent is interlobar but all siblings have the same single lobe
    if parent_is_interlobar and sibling_single_lobes:
        unique_sibling_lobes = list(set(sibling_single_lobes))
        if len(unique_sibling_lobes) == 1:
            # All siblings agree on one lobe - use that instead of interlobar
            consensus_lobe = unique_sibling_lobes[0]
            logger.debug(
                f"Parent is interlobar but all siblings agree on {consensus_lobe} for edge {u}->{v}"
            )
            return consensus_lobe, [consensus_lobe]

    # 4. Combine all lobe information
    all_contributing_lobes = all_lobes + interlobar_lobes

    if not all_contributing_lobes:
        logger.debug(f"No valid lobe information found for edge {u}->{v}")
        return "unknown", []

    # Remove duplicates while preserving order
    unique_lobes = list(dict.fromkeys(all_contributing_lobes))

    # 5. Determine final lobe assignment
    if len(set(unique_lobes)) == 1:
        # All sources point to the same lobe
        final_lobe = unique_lobes[0]
        logger.debug(f"All sources point to same lobe: {final_lobe} for edge {u}->{v}")
        return final_lobe, unique_lobes
    else:
        # Multiple different lobes found - mark as interlobar
        logger.debug(
            f"Multiple lobes found from parent/siblings: {unique_lobes}, marking as interlobar for edge {u}->{v}"
        )
        return "interlobar", unique_lobes


def _label_lobes_from_endpoints(graph: nx.DiGraph, lobe_masks: dict) -> None:
    """
    Primary lobe labeling using endpoint classification and iterative propagation.

    Args:
        graph: Directed graph to label (modified in-place)
        lobe_masks: Dictionary mapping lobe names to binary masks

    Algorithm:
        1. Find all terminal nodes (endpoints)
        2. Classify each endpoint using lobe masks
        3. Trace back from classified endpoints
        4. Apply iterative sibling-based labeling for remaining unknowns
        5. Continue until convergence or maximum iterations

    Quality Control:
        - Tracks labeling progress through iterations
        - Logs remaining unknown vessels for debugging
        - Implements convergence detection to avoid infinite loops

    Note:
        Main orchestration function for anatomical lobe assignment.
        Combines direct mask classification with inference algorithms.
    """
    # Find all endpoints efficiently
    endpoints = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    logger.debug("Processing %d endpoints for lobe labeling", len(endpoints))

    successful_traces = 0
    for endpoint in endpoints:
        lobe = _get_endpoint_lobe(graph, endpoint, lobe_masks)
        if lobe != "unknown":
            _trace_back_and_label(graph, endpoint, lobe)
            successful_traces += 1

    logger.info("Lobe labeling from endpoints complete")

    # Check initial state after endpoint tracing
    unknown_after_endpoints = len(
        [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if _is_unknown_lobe(data.get("lobe", []))
        ]
    )

    if unknown_after_endpoints == 0:
        logger.info("All lobes labeled successfully from endpoints")
        return
    else:
        logger.info(
            f"After endpoint tracing: {unknown_after_endpoints} edges still unknown"
        )

        max_iterations = 10
        for iteration in range(max_iterations):
            initial_count = len(
                [
                    (u, v)
                    for u, v, data in graph.edges(data=True)
                    if _is_unknown_lobe(data.get("lobe", []))
                ]
            )

            logger.info(
                f"Sibling labeling iteration {iteration + 1}: {initial_count} unknown edges"
            )

            labeled_count = _label_unknown_vessels_with_sibling(graph)

            final_count = len(
                [
                    (u, v)
                    for u, v, data in graph.edges(data=True)
                    if _is_unknown_lobe(data.get("lobe", []))
                ]
            )

            actual_improvement = initial_count - final_count

            logger.info(
                f"After iteration {iteration + 1}: {final_count} unknown edges remain"
            )
            logger.info(
                f"Function reported labeling {labeled_count}, actual improvement: {actual_improvement}"
            )

            # Show remaining unknown edges
            if final_count > 0:
                remaining_unknown = [
                    (u, v)
                    for u, v, data in graph.edges(data=True)
                    if _is_unknown_lobe(data.get("lobe", []))
                ]
                for u, v in remaining_unknown:
                    logger.warning(f"Edge {u}->{v} has unknown lobe")

            # Stop if no actual improvement
            if actual_improvement == 0:
                logger.info(
                    f"No actual improvement in iteration {iteration + 1}, stopping"
                )
                break


def _label_unknown_vessels_with_sibling(graph: nx.DiGraph) -> int:
    """
    Label vessels with unknown lobes using sibling and parent vessel information.
    Designed for iterative application - each pass may enable
    classification of additional vessels as more context becomes available.

    Args:
        graph: Directed graph with partial lobe labeling (modified in-place)

    Returns:
        int: Number of vessels successfully labeled in this iteration

    Algorithm:
        1. Identify all vessels with unknown/empty lobe classification
        2. For each unknown vessel, analyze parent and sibling lobe data
        3. Apply consensus rules for single vs multiple lobe territories
        4. Update vessel lobe information based on inference


    """
    logger.debug("Starting sibling-based labeling for unknown vessels")

    # Get all edges with unknown lobe
    unknown_edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if _is_unknown_lobe(data.get("lobe", []))
    ]

    logger.info(
        f"Found {len(unknown_edges)} unknown edges to process with sibling labeling"
    )

    actually_labeled_count = 0
    for u, v, edge_data in unknown_edges:
        current_lobe = edge_data.get("lobe", [])
        if _is_unknown_lobe(current_lobe):
            logger.debug(f"Processing edge {u}->{v}")

            sibling_lobe_label, all_lobes_list = _find_sibling_lobe(graph, u, v)

            logger.debug(
                f"  _find_sibling_lobe returned: lobe_label='{sibling_lobe_label}', all_lobes={all_lobes_list}"
            )

            if sibling_lobe_label != "unknown" and all_lobes_list:
                # Set the lobe field as a list containing all lobes
                edge_data["lobe"] = all_lobes_list.copy()

                logger.info(
                    f"Assigning lobe {sibling_lobe_label} to edge {u}->{v} based on sibling/parent info"
                )
                logger.debug(f"Edge {u}->{v} now has lobe={all_lobes_list}")
                actually_labeled_count += 1
            else:
                logger.debug(f"Could not find sibling/parent lobe for edge {u}->{v}")

    logger.info(
        f"Labeled {actually_labeled_count} unknown vessels using sibling/parent information"
    )
    return actually_labeled_count


def _mark_interlobar_vessels(graph: nx.DiGraph) -> None:
    """
    Identify and mark vessels that serve multiple anatomical lobes.
    Sets name patterns like "Interlobar_RUL_RML_u_v" for multi-lobe vessels
    and "RUL_u_v" for single-lobe vessels.

    Args:
        graph: Directed graph with lobe classifications (modified in-place)

    Algorithm:
        1. Examine lobe list for each vessel
        2. Vessels with multiple lobes marked as interlobar
        3. Update vessel names to reflect interlobar status
        4. Maintain list format for consistency
    """
    edges_to_update = []

    for u, v, data in graph.edges(data=True):
        # Skip MPA, RPA, LPA - they keep their ["None"] format
        if data.get("level") <= 2 and data.get("name") in ["MPA", "RPA", "LPA"]:
            continue

        current_lobe = _normalize_lobe_to_list(data.get("lobe", []))

        # Update name based on lobe content, but keep lobe as list
        if len(current_lobe) > 1:
            # Multiple lobes - this is interlobar
            lobe_str = "_".join(sorted(current_lobe))
            edges_to_update.append(
                (u, v, current_lobe, f"Interlobar_{lobe_str}_{u}_{v}")
            )
        elif len(current_lobe) == 1 and current_lobe[0] not in ["unknown", "None"]:
            # Single lobe
            lobe = current_lobe[0]
            edges_to_update.append((u, v, current_lobe, f"{lobe}_{u}_{v}"))
        else:
            # Empty or unknown
            edges_to_update.append((u, v, ["unknown"], f"Unknown_{u}_{v}"))

    # Apply updates - keep as lists for now
    for u, v, lobe_list, name in edges_to_update:
        graph.edges[u, v]["lobe"] = lobe_list
        graph.edges[u, v]["name"] = name

    logger.info("Processed %d edges for interlobar marking", len(edges_to_update))


def _convert_lobes_to_strings(graph: nx.DiGraph) -> None:
    """
    Convert lobe data from list format to final string format.

    Args:
        graph: Directed graph with list-format lobe data (modified in-place)

    Conversion Rules:
        - Multiple lobes → "interlobar"
        - Single lobe → lobe name string
        - Empty list → "unknown"
        - Already strings → unchanged
    """
    for _, _, data in graph.edges(data=True):
        lobe = data.get("lobe", [])

        if isinstance(lobe, list):
            if len(lobe) > 1:
                # Multiple lobes - mark as interlobar
                data["lobe"] = "interlobar"
            elif len(lobe) == 1:
                # Single lobe
                data["lobe"] = lobe[0]
            else:
                # Empty list - mark as unknown
                data["lobe"] = "unknown"
        # If already string, leave as is

    logger.info("Converted lobe data from lists to strings")


# =============================================================================
# HIERARCHICAL CLASSIFICATION
# =============================================================================


def _create_hierarchical_levels(
    graph: nx.DiGraph, diameter_threshold: float = 0.7, length_threshold: float = 2
) -> None:
    """
    Apply iterative bifurcation rules to establish vessel hierarchy beyond level 2.

    Args:
        graph: Directed graph with basic level classification (modified in-place)
        diameter_threshold: Minimum diameter ratio for trunk vessel classification
        length_threshold: Maximum length/diameter ratio for trunk vessels

    Algorithm:
        1. Start from level 2 vessels (RPA/LPA)
        2. Use breadth-first traversal to process descendants
        3. Apply vessel classification rules at each bifurcation
        4. Assign appropriate hierarchy levels based on morphology
    """

    # Find all level 2 edges to start from
    level2_edges = [
        (u, v) for u, v, data in graph.edges(data=True) if data.get("level") == 2
    ]

    queue = deque()

    # Initialize queue with children of level 2 edges
    for u, v in level2_edges:
        parent_diameter = graph.edges[u, v].get("mean_radius", 0) * 2
        for successor in graph.successors(v):
            queue.append((v, successor, 2, parent_diameter))

    processed_edges = set()

    while queue:
        parent_node, current_node, parent_level, parent_diameter = queue.popleft()

        edge_key = (parent_node, current_node)
        if edge_key in processed_edges:
            continue
        processed_edges.add(edge_key)

        edge_data = graph.edges[parent_node, current_node]

        # Process this edge
        new_level = _apply_vessel_rules(
            graph,
            parent_node,
            current_node,
            parent_level,
            parent_diameter,
            diameter_threshold,
            length_threshold,
        )

        # Use cached data for diameter calculation
        current_diameter = edge_data.get("mean_radius", 0) * 2
        for successor in graph.successors(current_node):
            queue.append((current_node, successor, new_level, current_diameter))

    logger.info("Hierarchical levels created (iterative)")


def _apply_vessel_rules(
    graph: nx.DiGraph,
    parent_node: int,
    current_node: int,
    parent_level: int,
    parent_diameter: float,
    diameter_threshold: float,
    length_threshold: float,
) -> int:
    """
    Apply morphological classification rules to determine vessel hierarchy level.

    Args:
        graph: Directed graph being classified (modified in-place)
        parent_node: Source node of vessel being classified
        current_node: Target node of vessel being classified
        parent_level: Hierarchy level of parent vessel
        parent_diameter: Diameter of parent vessel for ratio calculations
        diameter_threshold: Minimum diameter ratio for trunk classification
        length_threshold: Maximum length/diameter ratio for trunk vessels

    Returns:
        int: New hierarchy level assigned to vessel

    Classification Logic:
        1. Level 3 interlobar vessels remain level 3
        2. Level 3 with level 2 parents: trunk vs level 4 based on criteria
        3. Higher levels: trunk (maintain level) vs branch (increment level)
    """
    edge_data = graph.edges[parent_node, current_node]
    current_level = edge_data.get("level", 3)

    if current_level < 3:
        return current_level

    vessel_radius = edge_data.get("mean_radius", 0)
    vessel_diameter = vessel_radius * 2
    vessel_length = edge_data.get("length", 0)
    vessel_lobe = edge_data.get("lobe", "unknown")

    if isinstance(vessel_lobe, list):
        if len(vessel_lobe) > 1:
            vessel_lobe = "interlobar"
        elif len(vessel_lobe) == 0:
            vessel_lobe = "unknown"
        else:
            vessel_lobe = vessel_lobe[0]

    if current_level == 3 and vessel_lobe == "interlobar":
        new_level = 3
        edge_data["level"] = new_level
        edge_data["name"] = f"Interlobar_L3_{parent_node}_{current_node}"

    elif current_level == 3 and parent_level == 2:
        logger.debug(
            "Processing level 3 edge with parent level 2: %s -> %s",
            parent_node,
            current_node,
        )
        # Pass parent_level to trunk detection
        is_trunk = _is_trunk_vessel(
            vessel_diameter,
            vessel_length,
            parent_diameter,
            vessel_lobe,
            diameter_threshold,
            length_threshold,
            parent_level,
        )

        if is_trunk:
            new_level = 3
            edge_data["name"] = f"Trunk_L3_{vessel_lobe}_{parent_node}_{current_node}"
        else:
            new_level = 4
            edge_data["name"] = f"Level4_{vessel_lobe}_{parent_node}_{current_node}"

        edge_data["level"] = new_level

    else:
        # Pass parent_level to trunk detection
        is_trunk = _is_trunk_vessel(
            vessel_diameter,
            vessel_length,
            parent_diameter,
            vessel_lobe,
            diameter_threshold,
            length_threshold,
            parent_level,
        )

        if is_trunk:
            new_level = parent_level
            edge_data[
                "name"
            ] = f"Trunk{new_level}_{vessel_lobe}_{parent_node}_{current_node}"
        else:
            new_level = parent_level + 1
            edge_data[
                "name"
            ] = f"Level{new_level}_{vessel_lobe}_{parent_node}_{current_node}"

        edge_data["level"] = new_level

    return new_level


def _is_trunk_vessel(
    vessel_diameter: float,
    vessel_length: float,
    parent_diameter: float,
    vessel_lobe: str,
    diameter_threshold: float,
    length_threshold: float,
    parent_level: int = None,
) -> bool:
    """
    Determine if vessel qualifies as trunk vessel using morphological criteria.

    Args:
        vessel_diameter: Diameter of vessel being classified
        vessel_length: Length of vessel being classified
        parent_diameter: Diameter of parent vessel
        vessel_lobe: Anatomical lobe served by vessel
        diameter_threshold: Minimum diameter ratio threshold
        length_threshold: Maximum length/diameter ratio threshold
        parent_level: Hierarchy level of parent vessel

    Returns:
        bool: True if vessel qualifies as trunk, False for branch classification

    Priority Rules:
        1. Parent level ≤ 2: Diameter-only criteria (early branching)
        2. Lower lobes (RLL/LLL): Diameter-only criteria (accommodation)
        3. Upper lobes: Combined diameter AND length criteria
    """
    if parent_diameter == 0:
        return False

    if vessel_diameter == 0:
        return False

    diameter_ratio = vessel_diameter / parent_diameter
    length_ratio = vessel_length / vessel_diameter

    #  Fixed threshold for upper lobes
    if vessel_lobe in ["RUL", "LUL", "RML"]:
        # after initial bifurcation from level 2, use stricter criteria
        if parent_level is not None and parent_level <= 2:
            is_trunk_diameter = diameter_ratio >= 0.5
            is_trunk_length = length_ratio <= 2.5
            return is_trunk_diameter or is_trunk_length
        # for higher levels in upper lobes, use combined criteria
        elif parent_level is not None and parent_level >= 3:
            is_trunk_diameter = diameter_ratio >= 0.7
            is_trunk_length = length_ratio <= 2.5
            return is_trunk_diameter and is_trunk_length

    else:  # Lower lobes (RLL/LLL) or interlobar vessels
        # Use standard thresholds for lower lobes
        is_trunk_diameter = diameter_ratio >= 0.5
        is_trunk_length = length_ratio <= 2.5
        return is_trunk_diameter and is_trunk_length


# =============================================================================
# SEGMENTAL ANALYSIS
# =============================================================================


def _count_segments_by_linear_chains(
    graph: nx.DiGraph,
    current_node: int,
    segments_found: set,
    segment_to_lobe: dict,
    visited_nodes: set = None,
) -> None:
    """
    Count vascular segments as linear chains rather than individual edges.

    Args:
        graph: Directed graph representing vascular tree
        current_node: Node to start traversal from
        segments_found: Set collecting unique segment chain identifiers
        segment_to_lobe: Dictionary mapping segment chains to anatomical lobes
        visited_nodes: Set tracking visited nodes to prevent double-counting

    Algorithm:
        1. Traverse tree looking for level 4+ vessel chains
        2. Identify linear sequences of connected segment-level vessels
        3. Count each chain once using chain start node as identifier
        4. Map each chain to its primary anatomical lobe

    Chain Definition:
        Linear sequence of level 4+ vessels with single predecessor/successor
        relationships, representing functional vascular segments.

    """
    if visited_nodes is None:
        visited_nodes = set()

    # Process all children
    for child in graph.successors(current_node):
        if child in visited_nodes:
            continue

        # Get edge data
        child_edge = graph.edges[current_node, child]
        child_level = child_edge.get("level", 3)

        if child_level == 4:
            # Found start of a segment chain - trace to the end
            segment_start = child
            segment_end = _find_segment_chain_end(graph, child)

            # Only count this chain once using the start node
            if segment_start not in segments_found:
                segments_found.add(segment_start)

                # Map this segment chain to its lobe - handle list format
                lobe = child_edge.get("lobe", "unknown")
                if isinstance(lobe, list):
                    if len(lobe) > 1:
                        segment_to_lobe[segment_start] = "interlobar"
                    elif len(lobe) == 1:
                        segment_to_lobe[segment_start] = lobe[0]
                    else:
                        segment_to_lobe[segment_start] = "unknown"
                else:
                    segment_to_lobe[segment_start] = lobe

            # Mark all nodes in this chain as visited to avoid double counting
            _mark_chain_as_visited(graph, segment_start, segment_end, visited_nodes)

        elif child_level > 4:
            # Skip paths that are already past segments
            continue
        else:
            # For non-segment edges, continue traversal
            _count_segments_by_linear_chains(
                graph, child, segments_found, segment_to_lobe, visited_nodes
            )


def _find_segment_chain_end(graph: nx.DiGraph, start_node: int) -> int:
    """
    Locate the terminal node of a linear segment chain.

    Args:
        graph: Directed graph with hierarchical levels
        start_node: Beginning node of segment chain

    Returns:
        int: Final node in the linear chain of level 4+ vessels

    Algorithm:
        Follow single-successor path until reaching branching point,
        terminal node, or level transition boundary.
    """
    current = start_node

    while True:
        successors = list(graph.successors(current))

        # If no successors or multiple successors, this is the end
        if len(successors) != 1:
            break

        next_node = successors[0]
        next_edge = graph.edges[current, next_node]
        next_level = next_edge.get("level", 3)

        # If next edge is not level 4+, this is the end of segment chain
        if next_level < 4:
            break

        # Continue along the chain
        current = next_node

    return current


def _mark_chain_as_visited(
    graph: nx.DiGraph, start_node: int, end_node: int, visited_nodes: set
) -> None:
    """
    Mark all nodes in a segment chain as processed to prevent double-counting.

    Args:
        graph: Directed graph being analyzed
        start_node: First node in segment chain
        end_node: Last node in segment chain
        visited_nodes: Set of nodes to mark as processed
    """
    current = start_node
    visited_nodes.add(current)

    while current != end_node:
        successors = list(graph.successors(current))
        if len(successors) != 1:
            break

        next_node = successors[0]
        next_edge = graph.edges[current, next_node]

        if next_edge.get("level", 3) < 4:
            break

        visited_nodes.add(next_node)
        current = next_node


def _add_segment_counts(graph: nx.DiGraph) -> dict[str, int]:
    """
    Calculate and add segment counts to each vessel edge as metadata.
    This adds the 'segments_below' attribute to each edge with level 1-3,
    representing the number of unique downstream segment chains (not
    individual level 4 edges).

    Args:
        graph: Directed graph with hierarchical classification (modified in-place)

    Returns:
        dict: Segment counts organized by anatomical lobe

    Algorithm:
        1. For each level 1-3 vessel, count downstream segment chains
        2. Add 'segments_below' attribute to vessel edges
        3. Verify count consistency across hierarchy levels
        4. Generate summary statistics by anatomical region
    """
    logger.info("Counting downstream segment chains for each vessel")

    # We'll build segment_to_lobe during traversal
    segment_to_lobe = {}

    # Track segment chains by vessel
    all_segments = set()
    segments_by_level = {1: set(), 2: set(), 3: set()}
    segments_by_lobe = {
        lobe: set()
        for lobe in ["None", "interlobar", "RUL", "RML", "RLL", "LUL", "LLL", "unknown"]
    }

    # Count segment chains downstream of each vessel
    for _, v, edge_data in graph.edges(data=True):
        if edge_data.get("level") in [1, 2, 3]:
            segments_found = set()  # Track unique segment chain starts

            # Find segment chains downstream from this edge
            _count_segments_by_linear_chains(graph, v, segments_found, segment_to_lobe)

            # Add the count to edge attributes
            segment_count = len(segments_found)
            if segment_count > 0:
                edge_data["segments_below"] = segment_count

                # Add these segment chains to the correct collections
                level = edge_data.get("level")
                if level in segments_by_level:
                    segments_by_level[level].update(segments_found)
                all_segments.update(segments_found)

                # For each segment chain found, add it to its primary lobe collection
                for segment_node in segments_found:
                    if segment_node in segment_to_lobe:
                        primary_lobe = segment_to_lobe[segment_node]
                        if primary_lobe in segments_by_lobe:
                            segments_by_lobe[primary_lobe].add(segment_node)

                logger.debug(
                    "Edge (%s): %s segment chains below",
                    edge_data.get("name", "unnamed"),
                    segment_count,
                )

    # Convert sets to counts for reporting
    segment_counts_by_lobe = {
        lobe: len(segments)
        for lobe, segments in segments_by_lobe.items()
        if len(segments) > 0
    }
    total_segments = len(all_segments)

    # Log summary statistics
    logger.info("Total unique segment chains in tree: %s", total_segments)

    # Check that MPA has all segments
    mpa_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) if d.get("name") == "MPA"
    ]
    if mpa_edges:
        mpa_segment_count = mpa_edges[0][2].get("segments_below", 0)
        if mpa_segment_count != total_segments:
            logger.warning(
                "Segment count mismatch: MPA shows %d segment chains but found %d in tree",
                mpa_segment_count,
                total_segments,
            )

    # Log segment chains by lobe
    for lobe, count in segment_counts_by_lobe.items():
        logger.info("Segment chains in %s: %d", lobe, count)

    # Verify our count adds up correctly
    lobe_totals = sum(
        len(segments)
        for lobe, segments in segments_by_lobe.items()
        if lobe not in ["None", "interlobar", "unknown"]
    )
    if total_segments != lobe_totals:
        logger.warning(
            "Segment count inconsistency: Sum of lobe segment chains (%d) doesn't match total (%d)",
            lobe_totals,
            total_segments,
        )

    return segment_counts_by_lobe


# =============================================================================
# VALIDATION AND QUALITY CONTROL
# =============================================================================


def _check_lobe_masks(lobe_masks: dict) -> None:
    """
    Validate that anatomical lobe masks contain data and warn of issues.

    Args:
        lobe_masks: Dictionary of lobe name -> binary mask mappings

    Validation Checks:
        - Detects None/empty masks
        - Counts non-zero voxels in each mask
        - Warns about missing anatomical regions
    """
    empty_masks = []

    for lobe_name, mask in lobe_masks.items():
        if mask is None:
            empty_masks.append(lobe_name)
            logger.warning("Lobe mask for %s is None", lobe_name)
        elif hasattr(mask, "sum"):
            # For numpy arrays or similar
            if mask.sum() == 0:
                empty_masks.append(lobe_name)
                logger.warning("Lobe mask for %s is empty (no voxels set)", lobe_name)
        elif hasattr(mask, "__len__"):
            # For lists or other sequences
            if len(mask) == 0:
                empty_masks.append(lobe_name)
                logger.warning("Lobe mask for %s has zero length", lobe_name)

    if empty_masks:
        logger.warning("Found %d empty lobe masks: %s", len(empty_masks), empty_masks)
        logger.warning("This may cause incomplete lobe labeling")
    else:
        logger.info("All lobe masks contain data")


def _check_for_missing_lobes(graph: nx.DiGraph) -> None:
    """
    Verify that all expected anatomical lobes are represented in labeled graph.

    Args:
        graph: Completely labeled directed graph

    Expected Lobes:
        RUL, RML, RLL, LUL, LLL (standard pulmonary anatomy)

    Quality Assessment:
        Warns if any expected lobe completely absent from vessel tree,
        which may indicate segmentation issues or anatomical variants.
    """
    expected_lobes = {"RUL", "RML", "RLL", "LUL", "LLL"}
    found_lobes = set()

    # Collect all lobes found in the graph
    for _, _, data in graph.edges(data=True):
        lobe = data.get("lobe", "unknown")

        # Handle both list and string formats
        if isinstance(lobe, list):
            for lobe_name in lobe:
                if lobe_name in expected_lobes:
                    found_lobes.add(lobe_name)
        elif lobe in expected_lobes:
            found_lobes.add(lobe)

    missing_lobes = expected_lobes - found_lobes

    if missing_lobes:
        logger.warning("Missing lobes detected: %s", sorted(missing_lobes))
    else:
        logger.info("All expected lobes found in vessel tree: %s", sorted(found_lobes))


def _print_level_summary(graph: nx.DiGraph) -> None:
    """
    Generate summary statistics for labeled vascular tree.

    Args:
        graph: Completely labeled and analyzed directed graph

    Statistics Generated:
        - Vessel count by hierarchy level
        - Vessel count by anatomical lobe
        - Segment chain distribution
        - Maximum hierarchy level reached
        - Cross-validation of counting methods
    """
    level_counts = defaultdict(int)
    lobe_counts = defaultdict(int)

    # Collect statistics in one pass
    for _, _, data in graph.edges(data=True):
        # Level statistics
        level = data.get("level", "unknown")
        level_counts[level] += 1

        # Lobe statistics - handle both list and string formats
        lobe = data.get("lobe", "unknown")
        if isinstance(lobe, list):
            if len(lobe) > 1:
                # Multiple lobes - count as interlobar
                lobe_counts["interlobar"] += 1
            elif len(lobe) == 1:
                # Single lobe in list
                lobe_counts[lobe[0]] += 1
            else:
                # Empty list
                lobe_counts["unknown"] += 1
        else:
            # String format (backward compatibility)
            lobe_counts[lobe] += 1

    # Print labeling statistics
    logger.info("Labeling statistics:")
    logger.info(
        "  Final graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    logger.info("  Edges by lobe: %s", dict(lobe_counts))
    logger.info("  Edges by level: %s", dict(level_counts))

    # Add segment chain counting and statistics
    segment_counts_by_lobe = _add_segment_counts(graph)

    # Print detailed level summary
    logger.info("LEVEL SUMMARY:")
    logger.info("=" * 50)
    for level in sorted(level_counts.keys()):
        logger.info("Level %s: %d edges", level, level_counts[level])

    # Find max level
    numeric_levels = [k for k in level_counts.keys() if isinstance(k, int)]
    if numeric_levels:
        max_level = max(numeric_levels)
        logger.info("Maximum level reached: %d", max_level)

    # Print segment chain summary
    if segment_counts_by_lobe:
        logger.info("SEGMENT CHAIN SUMMARY:")
        logger.info("-" * 30)
        total_segment_chains = sum(segment_counts_by_lobe.values())
        logger.info("Total segment chains in tree: %d", total_segment_chains)

        # Verify this matches level 4 edges that start segment chains
        level4_edges = [d for u, v, d in graph.edges(data=True) if d.get("level") == 4]
        level4_count = len(level4_edges)
        logger.info("Level 4 edges (individual): %d", level4_count)

        for lobe, count in sorted(segment_counts_by_lobe.items()):
            if count > 0:
                logger.info("  %s: %d segment chains", lobe, count)

    logger.info("=" * 50)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def label_graph(
    graph: nx.DiGraph,
    lungs_dict: dict[str, any],
    diameter_threshold: float = 0.5,
    length_threshold: float = 1.5,
) -> nx.DiGraph:
    """
    Complete anatomical labeling pipeline for pulmonary vascular tree graphs.

    This is the main orchestration function that coordinates all labeling steps
    to transform a basic directed tree into a comprehensive anatomical model
    with hierarchy levels, lobe assignments, and segment classifications.

    Args:
        graph: Oriented directed graph representing vascular tree
        lungs_dict: Dictionary containing all anatomical masks:
                   - lung_right, lung_left: laterality classification
                   - lung_upper_lobe_right, lung_middle_lobe_right, etc.: lobe masks
        diameter_threshold: Minimum diameter ratio for trunk vessel classification
        length_threshold: Maximum length/diameter ratio for trunk vessels

    Returns:
        nx.DiGraph: Comprehensively labeled graph with:
                   - level: Hierarchy classification (1=MPA, 2=RPA/LPA, 3+=branches)
                   - lateral: Left/right laterality
                   - lobe: Anatomical lobe assignment
                   - name: Human-readable vessel identifier
                   - segments_below: Count of downstream segment chains

    Pipeline Steps:
        1. Hierarchy Foundation: Label MPA (level 1) and RPA/LPA (level 2)
        2. Basic Classification: Assign level 3 to remaining vessels
        3. Anatomical Mapping: Classify vessels by anatomical lobe
        4. Interlobar Detection: Identify vessels serving multiple lobes
        5. Format Standardization: Convert to consistent string formats
        6. Advanced Hierarchy: Apply morphological rules for levels 3+
        7. Quality Control: Validate completeness and consistency
        8. Statistical Analysis: Generate segment counts and summaries

    Morphological Criteria:
        - Trunk vessels: Large diameter relative to parent, short relative to diameter
        - Branch vessels: Smaller diameter, longer relative to diameter
        - Lobe-specific rules: Account for anatomical differences between regions

    Quality Assurance:
        - Pre-validation of input masks
        - Iterative unknown vessel classification
        - Post-validation of expected anatomical coverage
        - Statistical reporting
    """
    logger.info("Starting anatomical labeling")

    # Prepare lung masks
    lungs = {"right": lungs_dict["lung_right"], "left": lungs_dict["lung_left"]}

    lobe_masks = {
        "RUL": lungs_dict["lung_upper_lobe_right"],
        "RML": lungs_dict["lung_middle_lobe_right"],
        "RLL": lungs_dict["lung_lower_lobe_right"],
        "LUL": lungs_dict["lung_upper_lobe_left"],
        "LLL": lungs_dict["lung_lower_lobe_left"],
    }

    # Check lobe masks before processing
    _check_lobe_masks(lobe_masks)

    # Step 1: Label main artery (level 1)
    _label_main_artery(graph)

    # Step 2: Label laterality and first bifurcation (level 2)
    _label_laterality_and_bifurcation(graph, lungs)

    # Step 3: Label all remaining vessels as level 3
    _label_remaining_as_level3(graph)

    # Step 4: Label lobes based on endpoints
    _label_lobes_from_endpoints(graph, lobe_masks)

    # Step 5: Mark interlobar vessels
    _mark_interlobar_vessels(graph)

    # Step 6 : Convert lobes to strings
    _convert_lobes_to_strings(graph)

    # Step 7: Create hierarchical levels based on bifurcation rules with configurable thresholds
    # logger.debug("Starting hierarchical level creation...")
    _create_hierarchical_levels(graph, diameter_threshold, length_threshold)

    # Step 8: Check for missing lobes
    _check_for_missing_lobes(graph)

    # Add level summary
    _print_level_summary(graph)

    logger.info("Labeling complete")
    return graph
