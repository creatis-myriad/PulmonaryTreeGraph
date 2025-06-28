"""
Vessel-Graph and Embolism Integration Module

This module provides functions for analyzing interactions between pulmonary embolisms
and vessel graphs, including enrichment, metrics computation, and region analysis.

Main Functions:
- analyze_embolism_vessel_interactions(): Identify which vessels are affected by embolisms
- compute_embolism_vessel_metrics(): Calculate detailed obstruction and overlap metrics
- enrich_vessel_graph_with_embolisms(): Annotate vessel graph with embolism data
- analyze_embolism_regions_with_graph_enrichment(): Complete pipeline for vessel-embolism analysis
- get_embolism_vessel_statistics(): Generate summary statistics
- visualize_vessel_embolism_overlap(): Create visualization data
"""

import logging
import os
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pulmonarytreegraph.utils.metrics import (
    calculate_volume_from_voxels,
)

logger = logging.getLogger(__name__)


def analyze_embolism_vessel_interactions(
    embolism_objects: list[dict[str, Any]],
    vessel_mask: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Analyze which vessels are affected by each embolism.

    Parameters:
    - embolism_objects: List of embolism dictionaries with coordinates
    - vessel_mask: 3D numpy array with vessel IDs as values (0 = background)
    - spacing: Voxel spacing (x, y, z) in mm, optional for volume calculations

    Returns:
    - List of dictionaries with embolism-vessel interaction data

    Raises:
    - ValueError: If input data is invalid
    - RuntimeError: If analysis fails
    """
    logger.info(
        "Analyzing embolism-vessel interactions for %d embolisms", len(embolism_objects)
    )

    if not embolism_objects:
        logger.warning("No embolism objects provided")
        return []

    if vessel_mask is None or vessel_mask.size == 0:
        logger.error("Invalid vessel mask provided")
        raise ValueError("Vessel mask cannot be empty")

    try:
        results = []

        for embolism in embolism_objects:
            try:
                embolism_id = embolism.get("id", "unknown")
                coords = embolism.get("coordinates", [])

                if len(coords) == 0:
                    logger.warning("Empty coordinates for embolism %s", embolism_id)
                    continue

                vessel_ids = set()
                vessel_voxel_counts = {}
                total_embolism_voxels = len(coords)
                voxels_in_vessels = 0

                # Analyze each voxel of the embolism
                for coord in coords:
                    try:
                        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])

                        # Check bounds
                        if (
                            0 <= x < vessel_mask.shape[0]
                            and 0 <= y < vessel_mask.shape[1]
                            and 0 <= z < vessel_mask.shape[2]
                        ):
                            vessel_id = vessel_mask[x, y, z]
                            if vessel_id > 0:
                                vessel_ids.add(int(vessel_id))
                                vessel_voxel_counts[int(vessel_id)] = (
                                    vessel_voxel_counts.get(int(vessel_id), 0) + 1
                                )
                                voxels_in_vessels += 1

                    except (ValueError, IndexError):
                        logger.warning(
                            "Invalid coordinate for embolism %s: %s", embolism_id, coord
                        )
                        continue

                # Calculate vessel overlap percentage
                vessel_overlap_percentage = (
                    (voxels_in_vessels / total_embolism_voxels * 100)
                    if total_embolism_voxels > 0
                    else 0
                )

                interaction_data = {
                    "embolism_id": embolism_id,
                    "affected_vessel_ids": sorted(list(vessel_ids)),
                    "num_affected_vessels": len(vessel_ids),
                    "vessel_voxel_counts": vessel_voxel_counts,
                    "total_embolism_voxels": total_embolism_voxels,
                    "voxels_in_vessels": voxels_in_vessels,
                    "vessel_overlap_percentage": vessel_overlap_percentage,
                }

                # Add volume information if spacing is available
                if spacing and len(spacing) == 3:
                    embolism_spacing = embolism.get("spacing", spacing)
                    volume_info = calculate_volume_from_voxels(
                        voxels_in_vessels, embolism_spacing
                    )
                    interaction_data.update(
                        {
                            "vessel_overlap_volume_mm3": volume_info["volume_mm3"],
                            "vessel_overlap_volume_cm3": volume_info["volume_cm3"],
                        }
                    )

                results.append(interaction_data)

                logger.debug(
                    "Embolism %s affects %d vessels: %s (%.1f%% overlap)",
                    embolism_id,
                    len(vessel_ids),
                    list(vessel_ids),
                    vessel_overlap_percentage,
                )

            except (KeyError, TypeError) as e:
                logger.error(
                    "Error processing embolism %s: %s", embolism.get("id", "unknown"), e
                )
                continue

        logger.info("Successfully analyzed interactions for %d embolisms", len(results))
        return results

    except Exception as e:
        logger.error("Error during embolism-vessel interaction analysis: %s", e)
        raise RuntimeError("Failed to analyze embolism-vessel interactions") from e


def compute_embolism_vessel_metrics(
    embolism_objects: list[dict[str, Any]],
    vessel_mask: np.ndarray,
    include_vessel_analysis: bool = True,
) -> list[dict[str, Any]]:
    """
    Compute detailed metrics for embolism-vessel overlap, including obstruction ratios.

    Parameters:
    - embolism_objects: List of embolism dictionaries
    - vessel_mask: 3D numpy array with vessel IDs as values
    - include_vessel_analysis: Whether to compute per-vessel statistics

    Returns:
    - List of dictionaries with detailed vessel metrics per embolism

    Raises:
    - ValueError: If input data is invalid
    - RuntimeError: If metrics computation fails
    """
    logger.info(
        "Computing detailed embolism-vessel metrics for %d embolisms",
        len(embolism_objects),
    )

    if not embolism_objects:
        logger.warning("No embolism objects provided")
        return []

    if vessel_mask is None or vessel_mask.size == 0:
        logger.error("Invalid vessel mask provided")
        raise ValueError("Vessel mask cannot be empty")

    try:
        metrics = []

        # Pre-compute vessel sizes if needed
        vessel_sizes = {}
        if include_vessel_analysis:
            logger.debug("Pre-computing vessel sizes")
            unique_vessels = np.unique(vessel_mask)
            unique_vessels = unique_vessels[unique_vessels > 0]  # Remove background

            for vessel_id in unique_vessels:
                vessel_sizes[int(vessel_id)] = np.sum(vessel_mask == vessel_id)

        for embolism in embolism_objects:
            try:
                embolism_id = embolism.get("id", "unknown")
                coords = embolism.get("coordinates", [])
                spacing = embolism.get("spacing", (1.0, 1.0, 1.0))

                if len(coords) == 0:
                    logger.warning("Empty coordinates for embolism %s", embolism_id)
                    continue

                # Count embolism voxels per vessel
                vessel_counts = {}
                for coord in coords:
                    try:
                        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])

                        if (
                            0 <= x < vessel_mask.shape[0]
                            and 0 <= y < vessel_mask.shape[1]
                            and 0 <= z < vessel_mask.shape[2]
                        ):
                            vessel_id = vessel_mask[x, y, z]
                            if vessel_id > 0:
                                vessel_counts[int(vessel_id)] = (
                                    vessel_counts.get(int(vessel_id), 0) + 1
                                )

                    except (ValueError, IndexError):
                        continue

                # Compute metrics for each affected vessel
                vessel_metrics = []
                for vessel_id, embolism_voxels in vessel_counts.items():
                    try:
                        # Volume calculations
                        embolism_volume = calculate_volume_from_voxels(
                            embolism_voxels, spacing
                        )

                        vessel_metric = {
                            "vessel_id": vessel_id,
                            "embolism_voxels": embolism_voxels,
                            "embolism_volume_mm3": embolism_volume["volume_mm3"],
                            "embolism_volume_cm3": embolism_volume["volume_cm3"],
                        }

                        # Add vessel-specific metrics if available
                        if include_vessel_analysis and vessel_id in vessel_sizes:
                            vessel_voxels = vessel_sizes[vessel_id]
                            vessel_volume = calculate_volume_from_voxels(
                                vessel_voxels, spacing
                            )

                            obstruction_percent = (
                                (embolism_voxels / vessel_voxels * 100)
                                if vessel_voxels > 0
                                else 0
                            )

                            vessel_metric.update(
                                {
                                    "vessel_voxels": vessel_voxels,
                                    "vessel_volume_mm3": vessel_volume["volume_mm3"],
                                    "vessel_volume_cm3": vessel_volume["volume_cm3"],
                                    "obstruction_percent": obstruction_percent,
                                }
                            )

                        vessel_metrics.append(vessel_metric)

                    except (KeyError, TypeError) as e:
                        logger.warning(
                            "Error computing metrics for vessel %d in embolism %s: %s",
                            vessel_id,
                            embolism_id,
                            e,
                        )
                        continue

                # Sort by obstruction percentage (highest first)
                if include_vessel_analysis:
                    vessel_metrics.sort(
                        key=lambda x: x.get("obstruction_percent", 0), reverse=True
                    )

                embolism_metrics = {
                    "embolism_id": embolism_id,
                    "vessel_metrics": vessel_metrics,
                    "num_affected_vessels": len(vessel_metrics),
                    "total_embolism_voxels": len(coords),
                }

                # Add summary statistics
                if vessel_metrics and include_vessel_analysis:
                    obstructions = [
                        vm.get("obstruction_percent", 0) for vm in vessel_metrics
                    ]
                    embolism_metrics.update(
                        {
                            "max_obstruction_percent": max(obstructions),
                            "mean_obstruction_percent": np.mean(obstructions),
                            "total_embolism_in_vessels": sum(
                                vm["embolism_voxels"] for vm in vessel_metrics
                            ),
                        }
                    )

                metrics.append(embolism_metrics)

                logger.debug(
                    "Computed metrics for embolism %s: %d vessels affected",
                    embolism_id,
                    len(vessel_metrics),
                )

            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    "Error processing embolism %s: %s", embolism.get("id", "unknown"), e
                )
                continue

        logger.info("Successfully computed metrics for %d embolisms", len(metrics))
        return metrics

    except Exception as e:
        logger.error("Error during embolism-vessel metrics computation: %s", e)
        raise RuntimeError("Failed to compute embolism-vessel metrics") from e


def enrich_vessel_graph_with_embolisms(
    vessel_graph: nx.DiGraph,
    embolism_objects: list[dict[str, Any]],
    vessel_mask: np.ndarray,
    interaction_results: list[dict[str, Any]] | None = None,
) -> nx.DiGraph:
    """
    Annotate/enrich the vessel graph with embolism information using NetworkX graph attributes.

    Parameters:
    - vessel_graph: NetworkX DiGraph representing vessel structure
    - embolism_objects: List of embolism dictionaries
    - vessel_mask: 3D numpy array with vessel IDs
    - interaction_results: Pre-computed interaction results (optional)

    Returns:
    - Enriched NetworkX DiGraph with embolism annotations

    Raises:
    - ValueError: If input data is invalid
    - RuntimeError: If enrichment fails
    """
    logger.info(
        "Enriching vessel graph with embolism data from %d embolisms",
        len(embolism_objects),
    )

    if vessel_graph is None:
        logger.error("Vessel graph cannot be None")
        raise ValueError("Vessel graph is required")

    if not isinstance(vessel_graph, nx.DiGraph):
        logger.error("Vessel graph must be a NetworkX DiGraph")
        raise ValueError("Vessel graph must be a NetworkX DiGraph")

    if not embolism_objects:
        logger.warning("No embolism objects provided for enrichment")
        return vessel_graph

    try:
        # Use provided interaction results or compute them
        if interaction_results is None:
            logger.debug("Computing interaction results for graph enrichment")
            interaction_results = analyze_embolism_vessel_interactions(
                embolism_objects, vessel_mask
            )

        # Initialize graph-level embolism data
        vessel_graph.graph["embolism_annotations"] = {}
        vessel_graph.graph["vessel_embolism_summary"] = {}

        # Process each embolism's vessel interactions
        for interaction in interaction_results:
            embolism_id = interaction["embolism_id"]
            affected_vessels = interaction["affected_vessel_ids"]
            vessel_voxel_counts = interaction.get("vessel_voxel_counts", {})

            for vessel_id in affected_vessels:
                # Add embolism data to node attributes if vessel exists in graph
                if vessel_id in vessel_graph.nodes:
                    # Initialize embolism list for this vessel if not exists
                    if "embolisms" not in vessel_graph.nodes[vessel_id]:
                        vessel_graph.nodes[vessel_id]["embolisms"] = []

                    # Add embolism info to vessel node
                    vessel_graph.nodes[vessel_id]["embolisms"].append(
                        {
                            "embolism_id": embolism_id,
                            "voxels_affected": vessel_voxel_counts.get(vessel_id, 0),
                            "overlap_percentage": interaction.get(
                                "vessel_overlap_percentage", 0
                            ),
                        }
                    )

                    # Update vessel-level embolism counts
                    vessel_graph.nodes[vessel_id]["num_embolisms"] = len(
                        vessel_graph.nodes[vessel_id]["embolisms"]
                    )
                    vessel_graph.nodes[vessel_id]["total_embolism_voxels"] = sum(
                        emb["voxels_affected"]
                        for emb in vessel_graph.nodes[vessel_id]["embolisms"]
                    )
                    vessel_graph.nodes[vessel_id]["has_embolism"] = True
                else:
                    logger.warning(
                        "Vessel ID %d found in mask but not in graph", vessel_id
                    )

                # Add to graph-level annotations
                if vessel_id not in vessel_graph.graph["embolism_annotations"]:
                    vessel_graph.graph["embolism_annotations"][vessel_id] = []

                vessel_graph.graph["embolism_annotations"][vessel_id].append(
                    {
                        "embolism_id": embolism_id,
                        "voxels_affected": vessel_voxel_counts.get(vessel_id, 0),
                        "overlap_percentage": interaction.get(
                            "vessel_overlap_percentage", 0
                        ),
                    }
                )

                # Update graph-level vessel summary
                if vessel_id not in vessel_graph.graph["vessel_embolism_summary"]:
                    vessel_graph.graph["vessel_embolism_summary"][vessel_id] = {
                        "num_embolisms": 0,
                        "total_embolism_voxels": 0,
                        "embolism_ids": [],
                    }

                summary = vessel_graph.graph["vessel_embolism_summary"][vessel_id]
                summary["num_embolisms"] += 1
                summary["total_embolism_voxels"] += vessel_voxel_counts.get(
                    vessel_id, 0
                )
                summary["embolism_ids"].append(embolism_id)

            logger.debug(
                "Annotated vessels %s for embolism %s", affected_vessels, embolism_id
            )

        # Mark vessels without embolisms
        for node_id in vessel_graph.nodes:
            if "has_embolism" not in vessel_graph.nodes[node_id]:
                vessel_graph.nodes[node_id]["has_embolism"] = False
                vessel_graph.nodes[node_id]["num_embolisms"] = 0
                vessel_graph.nodes[node_id]["embolisms"] = []

        # Add global embolism statistics to graph
        vessel_graph.graph["embolism_global_stats"] = {
            "total_embolisms": len(embolism_objects),
            "affected_vessels": len(vessel_graph.graph["embolism_annotations"]),
            "total_interactions": sum(
                len(interaction["affected_vessel_ids"])
                for interaction in interaction_results
            ),
            "vessels_in_graph": vessel_graph.number_of_nodes(),
            "vessels_with_embolisms": sum(
                1
                for node in vessel_graph.nodes
                if vessel_graph.nodes[node].get("has_embolism", False)
            ),
        }

        logger.info(
            "Successfully enriched vessel graph with %d embolism interactions",
            vessel_graph.graph["embolism_global_stats"]["total_interactions"],
        )
        logger.info(
            "Affected %d out of %d vessels in graph",
            vessel_graph.graph["embolism_global_stats"]["vessels_with_embolisms"],
            vessel_graph.graph["embolism_global_stats"]["vessels_in_graph"],
        )

        return vessel_graph

    except Exception as e:
        logger.error("Error enriching vessel graph with embolisms: %s", e)
        raise RuntimeError("Failed to enrich vessel graph") from e


def get_embolism_vessel_statistics(
    interaction_results: list[dict[str, Any]],
    metrics_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate  statistics from embolism-vessel analysis.

    Parameters:
    - interaction_results: Results from analyze_embolism_vessel_interactions
    - metrics_results: Results from compute_embolism_vessel_metrics

    Returns:
    - Dictionary with summary statistics
    """
    logger.info("Generating embolism-vessel statistics")

    try:
        stats = {
            "embolism_stats": {},
            "vessel_stats": {},
            "interaction_stats": {},
            "obstruction_stats": {},
        }

        if not interaction_results and not metrics_results:
            logger.warning("No results provided for statistics generation")
            return stats

        # Embolism statistics
        total_embolisms = len(interaction_results)
        embolisms_affecting_vessels = sum(
            1 for r in interaction_results if r["num_affected_vessels"] > 0
        )

        stats["embolism_stats"] = {
            "total_embolisms": total_embolisms,
            "embolisms_affecting_vessels": embolisms_affecting_vessels,
            "percentage_affecting_vessels": (
                embolisms_affecting_vessels / total_embolisms * 100
            )
            if total_embolisms > 0
            else 0,
        }

        # Vessel statistics
        all_affected_vessels = set()
        vessel_embolism_counts = {}

        for interaction in interaction_results:
            for vessel_id in interaction["affected_vessel_ids"]:
                all_affected_vessels.add(vessel_id)
                vessel_embolism_counts[vessel_id] = (
                    vessel_embolism_counts.get(vessel_id, 0) + 1
                )

        stats["vessel_stats"] = {
            "total_affected_vessels": len(all_affected_vessels),
            "max_embolisms_per_vessel": max(vessel_embolism_counts.values())
            if vessel_embolism_counts
            else 0,
            "mean_embolisms_per_vessel": np.mean(list(vessel_embolism_counts.values()))
            if vessel_embolism_counts
            else 0,
        }

        # Interaction statistics
        total_interactions = sum(r["num_affected_vessels"] for r in interaction_results)
        overlap_percentages = [
            r["vessel_overlap_percentage"]
            for r in interaction_results
            if r["vessel_overlap_percentage"] > 0
        ]

        stats["interaction_stats"] = {
            "total_interactions": total_interactions,
            "mean_vessels_per_embolism": (total_interactions / total_embolisms)
            if total_embolisms > 0
            else 0,
        }

        if overlap_percentages:
            stats["interaction_stats"].update(
                {
                    "mean_overlap_percentage": np.mean(overlap_percentages),
                    "max_overlap_percentage": max(overlap_percentages),
                    "min_overlap_percentage": min(overlap_percentages),
                }
            )

        # Obstruction statistics (from metrics if available)
        if metrics_results:
            all_obstructions = []

            for metric in metrics_results:
                for vessel_metric in metric.get("vessel_metrics", []):
                    obstruction = vessel_metric.get("obstruction_percent", 0)
                    if obstruction > 0:
                        all_obstructions.append(obstruction)

            if all_obstructions:
                stats["obstruction_stats"] = {
                    "mean_obstruction_percent": np.mean(all_obstructions),
                    "max_obstruction_percent": max(all_obstructions),
                    "vessels_with_obstruction": len(all_obstructions),
                }

        logger.info("Generated  embolism-vessel statistics")
        return stats

    except (ValueError, KeyError, TypeError) as e:
        logger.error("Error generating embolism-vessel statistics: %s", e)
        return {"error": str(e)}


def save_vessel_embolism_analysis(
    interaction_results: list[dict[str, Any]],
    metrics_results: list[dict[str, Any]],
    statistics: dict[str, Any],
    output_dir: str,
    patient_name: str,
) -> dict[str, str]:
    """
    Save vessel-embolism analysis results to files.

    Parameters:
    - interaction_results: Vessel interaction results
    - metrics_results: Detailed metrics results
    - statistics: Summary statistics
    - output_dir: Output directory
    - patient_name: Patient identifier

    Returns:
    - Dictionary with paths to saved files
    """
    logger.info("Saving vessel-embolism analysis results to %s", output_dir)

    try:
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        # Save interaction results
        if interaction_results:
            interactions_path = os.path.join(
                output_dir, f"{patient_name}_vessel_interactions.csv"
            )
            interaction_df = pd.json_normalize(interaction_results)
            interaction_df.to_csv(interactions_path, index=False)
            saved_files["interactions"] = interactions_path

        # Save detailed metrics
        if metrics_results:
            metrics_path = os.path.join(
                output_dir, f"{patient_name}_vessel_metrics.csv"
            )
            # Flatten metrics for CSV
            flattened_metrics = []
            for metric in metrics_results:
                embolism_id = metric["embolism_id"]
                for vessel_metric in metric.get("vessel_metrics", []):
                    row = {"embolism_id": embolism_id}
                    row.update(vessel_metric)
                    flattened_metrics.append(row)

            if flattened_metrics:
                metrics_df = pd.DataFrame(flattened_metrics)
                metrics_df.to_csv(metrics_path, index=False)
                saved_files["metrics"] = metrics_path

        # Save statistics report
        stats_path = os.path.join(
            output_dir, f"{patient_name}_vessel_embolism_stats.txt"
        )
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("VESSEL-EMBOLISM INTERACTION ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Patient: {patient_name}\n\n")

            for category, data in statistics.items():
                if isinstance(data, dict) and data:
                    f.write(f"{category.upper().replace('_', ' ')}:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in data.items():
                        if isinstance(value, dict):
                            f.write(f"{key}: {value}\n")
                        elif isinstance(value, float):
                            f.write(f"{key}: {value:.2f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")

        saved_files["statistics"] = stats_path

        logger.info("Successfully saved vessel-embolism analysis results")
        return saved_files

    except OSError as e:
        logger.error("Error saving vessel-embolism analysis: %s", e)
        raise RuntimeError("Failed to save analysis results") from e


def get_vessel_embolism_info_from_graph(
    vessel_graph: nx.DiGraph, vessel_id: int
) -> dict[str, Any]:
    """
    Extract embolism information for a specific vessel from the enriched graph.

    Parameters:
    - vessel_graph: Enriched NetworkX DiGraph
    - vessel_id: ID of the vessel to query

    Returns:
    - Dictionary with vessel embolism information
    """
    if vessel_id not in vessel_graph.nodes:
        return {"error": f"Vessel {vessel_id} not found in graph"}

    node_data = vessel_graph.nodes[vessel_id]
    return {
        "vessel_id": vessel_id,
        "has_embolism": node_data.get("has_embolism", False),
        "num_embolisms": node_data.get("num_embolisms", 0),
        "total_embolism_voxels": node_data.get("total_embolism_voxels", 0),
        "embolisms": node_data.get("embolisms", []),
    }


def get_vessels_with_embolisms(vessel_graph: nx.DiGraph) -> list[int]:
    """
    Get list of vessel IDs that have embolisms.

    Parameters:
    - vessel_graph: Enriched NetworkX DiGraph

    Returns:
    - List of vessel IDs with embolisms
    """
    return [
        node_id
        for node_id, data in vessel_graph.nodes(data=True)
        if data.get("has_embolism", False)
    ]


def analyze_embolism_regions_with_graph_enrichment(
    embolism_objects: list[dict[str, Any]],
    vessel_graph: nx.DiGraph,
    vessel_mask: np.ndarray,
    output_dir: str | None = None,
    patient_name: str | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Complete pipeline for embolism-vessel graph enrichment and analysis.

    Parameters:
    - embolism_objects: List of embolism dictionaries
    - vessel_graph: NetworkX DiGraph representing vessel structure
    - vessel_mask: 3D numpy array with vessel IDs
    - output_dir: Output directory for saving results
    - patient_name: Patient identifier
    - save_results: Whether to save results to files

    Returns:
    - Dictionary with all analysis results

    Raises:
    - ValueError: If required parameters are missing
    - RuntimeError: If pipeline fails
    """
    logger.info("Running complete embolism-vessel graph enrichment pipeline")

    try:
        # Validate inputs
        if not embolism_objects:
            logger.warning("No embolism objects provided")
            return {"error": "No embolism objects provided"}

        if vessel_graph is None:
            logger.error("Vessel graph is required")
            raise ValueError("Vessel graph cannot be None")

        if not isinstance(vessel_graph, nx.DiGraph):
            logger.error("Vessel graph must be a NetworkX DiGraph")
            raise ValueError("Vessel graph must be a NetworkX DiGraph")

        if vessel_mask is None or vessel_mask.size == 0:
            logger.error("Vessel mask is required")
            raise ValueError("Vessel mask cannot be empty")

        # Step 1: Analyze vessel interactions
        logger.info("Step 1: Analyzing embolism-vessel interactions")
        interaction_results = analyze_embolism_vessel_interactions(
            embolism_objects, vessel_mask
        )

        # Step 2: Compute detailed metrics
        logger.info("Step 2: Computing detailed vessel metrics")
        metrics_results = compute_embolism_vessel_metrics(embolism_objects, vessel_mask)

        # Step 3: Enrich vessel graph
        logger.info("Step 3: Enriching vessel graph")
        enriched_graph = enrich_vessel_graph_with_embolisms(
            vessel_graph, embolism_objects, vessel_mask, interaction_results
        )

        # Step 4: Generate statistics
        logger.info("Step 4: Generating  statistics")
        statistics = get_embolism_vessel_statistics(
            interaction_results, metrics_results
        )

        # Prepare results
        results = {
            "interaction_results": interaction_results,
            "metrics_results": metrics_results,
            "enriched_graph": enriched_graph,
            "statistics": statistics,
            "summary": {
                "total_embolisms": len(embolism_objects),
                "total_interactions": sum(
                    r["num_affected_vessels"] for r in interaction_results
                ),
                "affected_vessels": len(
                    set().union(
                        *[r["affected_vessel_ids"] for r in interaction_results]
                    )
                ),
                "vessels_in_graph": enriched_graph.number_of_nodes(),
                "vessels_with_embolisms": len(
                    get_vessels_with_embolisms(enriched_graph)
                ),
            },
        }

        # Step 5: Save results if requested
        if save_results and output_dir and patient_name:
            logger.info("Step 5: Saving analysis results")
            try:
                saved_files = save_vessel_embolism_analysis(
                    interaction_results,
                    metrics_results,
                    statistics,
                    output_dir,
                    patient_name,
                )
                results["saved_files"] = saved_files
            except (OSError, ValueError) as e:
                logger.warning("Failed to save results: %s", e)
                results["save_error"] = str(e)

        logger.info("Pipeline completed successfully")
        logger.info(
            "Summary: %d embolisms, %d interactions, %d affected vessels (%d/%d vessels in graph)",
            results["summary"]["total_embolisms"],
            results["summary"]["total_interactions"],
            results["summary"]["affected_vessels"],
            results["summary"]["vessels_with_embolisms"],
            results["summary"]["vessels_in_graph"],
        )

        return results

    except Exception as e:
        logger.error("Error in embolism-vessel enrichment pipeline: %s", e)
        raise RuntimeError(
            "Failed to complete vessel-embolism analysis pipeline"
        ) from e


def analyze_embolism_by_levels(
    embolism_objects: list[dict[str, Any]],
    level_mask: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze embolism distribution across anatomical levels.

    Parameters:
    - embolism_objects: List of embolism dictionaries
    - level_mask: 3D numpy array with level IDs
    - vessel_mask: 3D numpy array with vessel IDs (optional)

    Returns:
    - Dictionary with level-based analysis results
    """
    logger.info("Analyzing embolism distribution across anatomical levels")

    try:
        level_stats = {}
        level_embolisms = {}

        for embolism in embolism_objects:
            embolism_id = embolism["id"]
            coordinates = embolism["coordinates"]

            # Determine which levels this embolism occupies
            embolism_levels = set()
            for coord in coordinates:
                x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                if (
                    0 <= x < level_mask.shape[0]
                    and 0 <= y < level_mask.shape[1]
                    and 0 <= z < level_mask.shape[2]
                ):
                    level_id = level_mask[x, y, z]
                    if level_id > 0:
                        embolism_levels.add(int(level_id))

            # Record embolism for each level it occupies
            for level_id in embolism_levels:
                if level_id not in level_embolisms:
                    level_embolisms[level_id] = []
                level_embolisms[level_id].append(embolism_id)

        # Calculate statistics for each level
        for level_id, embolism_ids in level_embolisms.items():
            level_stats[level_id] = {
                "level_id": level_id,
                "num_embolisms": len(embolism_ids),
                "embolism_ids": embolism_ids,
                "total_volume_mm3": sum(
                    emb["volume_mm3"]
                    for emb in embolism_objects
                    if emb["id"] in embolism_ids
                ),
            }

        return {
            "level_stats": level_stats,
            "total_levels_affected": len(level_stats),
            "embolisms_by_level": level_embolisms,
        }

    except Exception as e:
        logger.error("Error in level-based analysis: %s", e)
        raise RuntimeError("Failed to analyze embolisms by levels") from e


def save_analysis_report(results: dict[str, Any], output_path: str) -> None:
    """Save analysis report to text file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("COMPREHENSIVE PULMONARY EMBOLISM ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Patient: {results['patient_name']}\n")
            f.write(
                f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Basic embolism statistics
            f.write("EMBOLISM OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total embolisms: {results['num_embolisms']}\n")
            f.write(
                f"Total volume: {results['total_volume_mm3']:.2f} mm³ ({results['total_volume_cm3']:.4f} cm³)\n\n"
            )

            # Vessel interaction summary
            summary = results["summary"]
            f.write("VESSEL INTERACTION SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Embolisms affecting vessels: {summary['embolisms_affecting_vessels']}/{summary['total_embolisms']}\n"
            )
            f.write(
                f"Total vessel interactions: {summary['total_vessel_interactions']}\n"
            )
            f.write(f"Unique affected vessels: {summary['unique_affected_vessels']}\n")
            f.write(f"Vessels in graph: {summary['vessels_in_graph']}\n")
            f.write(
                f"Vessels with embolisms: {summary['vessels_with_embolisms']} ({summary['percentage_vessels_affected']:.1f}%)\n\n"
            )

            # Level analysis if available
            if (
                results.get("level_analysis")
                and "level_stats" in results["level_analysis"]
            ):
                f.write("LEVEL DISTRIBUTION:\n")
                f.write("-" * 18 + "\n")
                level_stats = results["level_analysis"]["level_stats"]
                for level_id, stats in level_stats.items():
                    f.write(
                        f"Level {level_id}: {stats['num_embolisms']} embolisms, {stats['total_volume_mm3']:.2f} mm³\n"
                    )
                f.write("\n")

            # Vessel-embolism statistics
            if "vessel_embolism_statistics" in results:
                vessel_stats = results["vessel_embolism_statistics"]
                if "embolism_stats" in vessel_stats:
                    f.write("DETAILED VESSEL STATISTICS:\n")
                    f.write("-" * 28 + "\n")
                    emb_stats = vessel_stats["embolism_stats"]
                    f.write(
                        f"Embolisms affecting vessels: {emb_stats.get('embolisms_affecting_vessels', 0)}\n"
                    )
                    f.write(
                        f"Percentage affecting vessels: {emb_stats.get('percentage_affecting_vessels', 0):.1f}%\n"
                    )

                    if "vessel_stats" in vessel_stats:
                        v_stats = vessel_stats["vessel_stats"]
                        f.write(
                            f"Total affected vessels: {v_stats.get('total_affected_vessels', 0)}\n"
                        )
                        f.write(
                            f"Max embolisms per vessel: {v_stats.get('max_embolisms_per_vessel', 0)}\n"
                        )
                        f.write(
                            f"Mean embolisms per vessel: {v_stats.get('mean_embolisms_per_vessel', 0):.2f}\n"
                        )

        logger.info("Comprehensive analysis report saved: %s", output_path)
    except Exception as e:
        logger.error("Error saving  report: %s", e)
        raise


def save_level_analysis_results(
    level_analysis: dict[str, Any], output_path: str
) -> None:
    """Save level analysis results to CSV file."""
    try:
        if "level_stats" not in level_analysis:
            return

        level_data = []
        for level_id, stats in level_analysis["level_stats"].items():
            level_data.append(
                {
                    "level_id": level_id,
                    "num_embolisms": stats["num_embolisms"],
                    "total_volume_mm3": stats["total_volume_mm3"],
                    "embolism_ids": ",".join(map(str, stats["embolism_ids"])),
                }
            )

        if level_data:
            df = pd.DataFrame(level_data)
            df.to_csv(output_path, index=False)
            logger.info("Level analysis results saved: %s", output_path)
    except Exception as e:
        logger.error("Error saving level analysis: %s", e)
        raise
