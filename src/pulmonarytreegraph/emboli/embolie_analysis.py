"""
Pulmonary Embolism Analysis Module

This module provides analysis functions for embolism
evaluation and workflow orchestration.

Main Functions:
- analyze_embolism_regions(): Complete basic embolism analysis pipeline
- analyze_embolisms_with_visualization(): Complete pipeline with visualization output
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulmonarytreegraph.emboli.embolie_construction import (
    create_embolism_objects,
)
from pulmonarytreegraph.utils.io import (
    create_colored_embolism_masks,
    load_graph,
    load_niftii,
    save_embolism_masks,
)
from pulmonarytreegraph.utils.metrics import (
    calculate_total_volume,
)

logger = logging.getLogger(__name__)


# =============================================================================
# REPORTING AND DOCUMENTATION UTILITIES
# =============================================================================


def generate_embolism_report(
    embolism_objects: list[dict[str, Any]],
    patient_name: str,
    output_path: str,
) -> None:
    """
    Generate a detailed embolism analysis report.

    Args:
        embolism_objects: List of structured embolism data objects with metadata
        patient_name: Patient identifier for report header and traceability
        output_path: Full file path for generated report (typically .txt format)

    Returns:
        None: Writes report to specified file path

    Algorithm:
        1. Calculate volume statistics (total, mean, median, std)
        2. Write formatted report with overview, statistics, and individual details
    """
    logger.info("Generating embolism report for %s at %s", patient_name, output_path)

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Calculate statistics
        if not embolism_objects:
            total_volume_mm3 = 0.0
            total_volume_cm3 = 0.0
            volumes = []
        else:
            total_volume_mm3, total_volume_cm3 = calculate_total_volume(
                embolism_objects, "volume_mm3"
            )
            volumes = [emb["volume_mm3"] for emb in embolism_objects]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("PULMONARY EMBOLISM ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n\n")

            # Basic embolism statistics
            f.write("EMBOLISM OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Patient: {patient_name}\n")
            f.write(f"Total embolisms detected: {len(embolism_objects)}\n")
            f.write(
                f"Total embolic volume: {total_volume_mm3:.2f} mm³ ({total_volume_cm3:.4f} cm³)\n\n"
            )

            if embolism_objects:
                # Volume distribution statistics
                f.write("VOLUME STATISTICS:\n")
                f.write("-" * 18 + "\n")
                f.write(f"Average volume: {np.mean(volumes):.2f} mm³\n")
                f.write(f"Median volume: {np.median(volumes):.2f} mm³\n")
                f.write(f"Largest embolism: {np.max(volumes):.2f} mm³\n")
                f.write(f"Smallest embolism: {np.min(volumes):.2f} mm³\n")
                f.write(f"Standard deviation: {np.std(volumes):.2f} mm³\n\n")

                # Individual embolism details
                f.write("INDIVIDUAL EMBOLISM DETAILS:\n")
                f.write("-" * 30 + "\n")

                for embolism in embolism_objects:
                    f.write(f"\nEmbolism {embolism['id']}:\n")
                    f.write(
                        f"  Volume: {embolism['volume_mm3']:.2f} mm³ ({embolism['volume_cm3']:.4f} cm³)\n"
                    )
                    f.write(f"  Voxel count: {embolism['voxel_count']}\n")
                    if total_volume_mm3 > 0:
                        percentage = (embolism["volume_mm3"] / total_volume_mm3) * 100
                        f.write(f"  Percentage of total: {percentage:.1f}%\n")

        logger.info("Embolism report generated successfully")

    except (OSError, ValueError) as e:
        logger.error("Error generating embolism report: %s", e)
        raise


# =============================================================================
# CSV EXPORT AND GLOBAL STATISTICS
# =============================================================================


def _initialize_embolism_stats_csv(output_path: Path) -> None:
    """Create CSV file with header for embolism statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "patient_id",
                "embolism_id",
                "separation_method",
                "fusion_threshold_mm",
                "voxel_count",
                "volume_mm3",
                "volume_cm3",
                "percentage_of_total_volume",
                "is_largest_embolism",
                "is_total_summary",
                "num_total_embolisms",
            ]
        )


def _append_embolism_stats_to_csv(
    output_path: Path,
    patient_id: str,
    embolism_objects: list[dict[str, Any]],
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
) -> None:
    """Append embolism statistics to CSV file, including individual embolisms and total summary."""
    import csv

    # Extract 4-digit patient ID
    patient_id_clean = (
        patient_id.split("_")[0][-4:] if "_" in patient_id else patient_id[-4:]
    )

    # Calculate totals
    if embolism_objects:
        total_volume_mm3, total_volume_cm3 = calculate_total_volume(
            embolism_objects, "volume_mm3"
        )
        total_voxel_count = sum(emb["voxel_count"] for emb in embolism_objects)

        # Find largest embolism
        volumes = [emb["volume_mm3"] for emb in embolism_objects]
        max_volume = max(volumes) if volumes else 0
    else:
        total_volume_mm3 = 0.0
        total_volume_cm3 = 0.0
        total_voxel_count = 0
        max_volume = 0.0

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Individual embolism rows
        for embolism in embolism_objects:
            percentage = (
                (embolism["volume_mm3"] / total_volume_mm3 * 100)
                if total_volume_mm3 > 0
                else 0.0
            )
            is_largest = (
                (embolism["volume_mm3"] == max_volume) if max_volume > 0 else False
            )

            writer.writerow(
                [
                    patient_id_clean,
                    embolism["id"],
                    separation_method,
                    fusion_threshold,
                    embolism["voxel_count"],
                    f"{embolism['volume_mm3']:.6f}",
                    f"{embolism['volume_cm3']:.6f}",
                    f"{percentage:.2f}",
                    is_largest,
                    False,  # is_total_summary
                    len(embolism_objects),
                ]
            )

        # Total summary row
        writer.writerow(
            [
                patient_id_clean,
                "total",  # Special embolism_id for totals
                separation_method,
                fusion_threshold,
                total_voxel_count,
                f"{total_volume_mm3:.6f}",
                f"{total_volume_cm3:.6f}",
                100.00,  # percentage_of_total_volume
                False,  # is_largest_embolism
                True,  # is_total_summary
                len(embolism_objects),
            ]
        )


def save_embolism_metrics(
    embolism_objects: list[dict[str, Any]],
    output_path: str,
    patient_id: str | None = None,
    global_stats_path: Path | None = None,
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
) -> None:
    """
    Export structured embolism metrics to CSV format with optional global statistics.

    Args:
        embolism_objects: List of embolism data dictionaries with complete metadata
        output_path: Full file path for individual patient CSV export
        patient_id: Patient identifier for global statistics
        global_stats_path: Path for global statistics CSV
        separation_method: Method used for embolism separation
        fusion_threshold: Fusion threshold used in analysis

    Returns:
        None: Writes CSV files to specified paths
    """
    logger.debug("Saving embolism metrics to %s", output_path)

    try:
        # Save individual patient metrics (existing functionality)
        csv_data = []

        # Calculate totals for percentage calculations
        if embolism_objects:
            total_volume_mm3, total_volume_cm3 = calculate_total_volume(
                embolism_objects, "volume_mm3"
            )
            total_voxel_count = sum(emb["voxel_count"] for emb in embolism_objects)
        else:
            total_volume_mm3 = 0.0
            total_volume_cm3 = 0.0
            total_voxel_count = 0

        # Individual embolism rows
        for embolism in embolism_objects:
            percentage = (
                (embolism["volume_mm3"] / total_volume_mm3 * 100)
                if total_volume_mm3 > 0
                else 0.0
            )

            row = {
                "embolism_id": embolism["id"],
                "patient_name": embolism["patient_name"],
                "voxel_count": embolism["voxel_count"],
                "volume_mm3": embolism["volume_mm3"],
                "volume_cm3": embolism["volume_cm3"],
                "percentage_of_total": percentage,
                "separation_method": separation_method,
                "fusion_threshold_mm": fusion_threshold,
            }
            csv_data.append(row)

        # Add TOTAL summary row
        if embolism_objects:
            total_row = {
                "embolism_id": "total",
                "patient_name": embolism_objects[0]["patient_name"],
                "voxel_count": total_voxel_count,
                "volume_mm3": total_volume_mm3,
                "volume_cm3": total_volume_cm3,
                "percentage_of_total": 100.0,
                "separation_method": separation_method,
                "fusion_threshold_mm": fusion_threshold,
            }
            csv_data.append(total_row)

        # Save individual patient CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            logger.debug(
                "Saved metrics for %d embolisms + total summary", len(embolism_objects)
            )
        else:
            # Create empty CSV with headers
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# No embolism data available\n")
                f.write(
                    "embolism_id,patient_name,voxel_count,volume_mm3,volume_cm3,"
                    "percentage_of_total,separation_method,fusion_threshold_mm\n"
                )

        # Save to global statistics CSV if requested
        if patient_id and global_stats_path:
            if not global_stats_path.exists():
                _initialize_embolism_stats_csv(global_stats_path)
            _append_embolism_stats_to_csv(
                global_stats_path,
                patient_id,
                embolism_objects,
                separation_method,
                fusion_threshold,
            )
            logger.debug("Appended embolism statistics to global CSV")

    except (OSError, ValueError) as e:
        logger.error("Error saving embolism metrics: %s", e)
        raise RuntimeError(f"Error saving embolism metrics: {e}") from e


def _initialize_enriched_embolism_stats_csv(output_path: Path) -> None:
    """Create CSV file with header for enriched embolism statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Create header with all anatomical columns
        header = [
            "patient_id",
            "embolism_id",
            "separation_method",
            "fusion_threshold_mm",
            "voxel_count",
            "volume_mm3",
            "volume_cm3",
            "percentage_of_total_volume",
            "is_largest_embolism",
            "is_total_summary",
            "num_total_embolisms",
            # Level percentages
            *[f"level_{i}_pct" for i in range(1, 11)],
            # Laterality percentages
            "right_lung_pct",
            "left_lung_pct",
            "unknown_laterality_pct",
            # Lobe percentages
            "RUL_pct",
            "RML_pct",
            "RLL_pct",
            "LUL_pct",
            "LLL_pct",
            "interlobar_pct",
            "unknown_lobe_pct",
            # Dominant features
            "dominant_level",
            "dominant_laterality",
            "dominant_lobe",
            # Clinical indicators
            "is_proximal",
            "is_bilateral",
            # number of vessels with emboli
            "vessels_with_emboli",
        ]
        writer.writerow(header)


def _append_enriched_embolism_stats_to_csv(
    output_path: Path,
    enriched_df: pd.DataFrame,
    patient_id: str,
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
) -> None:
    """Append enriched embolism statistics to CSV file."""
    # Extract 4-digit patient ID
    patient_id_clean = (
        patient_id.split("_")[0][-4:] if "_" in patient_id else patient_id[-4:]
    )

    # Find the largest embolism volume (exclude total row)
    non_total_rows = enriched_df[enriched_df["embolism_id"] != "total"]
    largest_embolism_id = None

    if len(non_total_rows) > 0:
        max_idx = non_total_rows["volume_mm3"].idxmax()
        largest_embolism_id = enriched_df.at[max_idx, "embolism_id"]

    with open(output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for _, row in enriched_df.iterrows():
            # Determine if this is the largest embolism
            is_largest = row.get("embolism_id") == largest_embolism_id
            is_total = row.get("embolism_id") == "total"

            # Prepare row data with all columns
            csv_row = [
                patient_id_clean,
                row.get("embolism_id", ""),
                separation_method,
                fusion_threshold,
                row.get("voxel_count", 0),
                f"{row.get('volume_mm3', 0):.6f}",
                f"{row.get('volume_cm3', 0):.6f}",
                f"{row.get('percentage_of_total', 0):.2f}",
                is_largest,  # is_largest_embolism - only True for actual largest
                is_total,  # is_total_summary
                len(enriched_df) - 1
                if "total" in enriched_df["embolism_id"].values
                else len(enriched_df),
                # Level percentages
                *[row.get(f"level_{i}_pct", 0.0) for i in range(1, 11)],
                # Laterality percentages
                row.get("right_lung_pct", 0.0),
                row.get("left_lung_pct", 0.0),
                row.get("unknown_laterality_pct", 0.0),
                # Lobe percentages
                row.get("RUL_pct", 0.0),
                row.get("RML_pct", 0.0),
                row.get("RLL_pct", 0.0),
                row.get("LUL_pct", 0.0),
                row.get("LLL_pct", 0.0),
                row.get("interlobar_pct", 0.0),
                row.get("unknown_lobe_pct", 0.0),
                # Dominant features
                row.get("dominant_level", "unknown"),
                row.get("dominant_laterality", "unknown"),
                row.get("dominant_lobe", "unknown"),
                # Clinical indicators
                row.get("is_proximal", False),
                row.get("is_bilateral", False),
                # Number of vessels involved
                row.get("vessels_with_emboli", 0),
            ]
            writer.writerow(csv_row)


# =============================================================================
# CORE ANALYSIS PIPELINES
# =============================================================================


def analyze_embolism_regions(
    mask_path: str,
    patient_name: str,
    output_dir: str,
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
    global_stats_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute complete pipeline for embolism analysis with optional global statistics.

    Args:
        mask_path: Absolute path to binary embolism mask file (NIfTI format)
        patient_name: Unique patient identifier for traceability and reporting
        output_dir: Output directory path for analysis results and generated files
        separation_method: Algorithm for embolism separation
        fusion_threshold: Distance threshold in millimeters for component fusion
        global_stats_path: Optional path for global statistics CSV

    Returns:
        dict[str, Any]: Analysis results with comprehensive statistics
    """
    logger.info("Starting embolism analysis for patient %s", patient_name)

    try:
        # Load mask
        logger.info("Loading embolism mask from %s", mask_path)
        mask_data, _, spacing, _ = load_niftii(mask_path, "embolism")

        # Create embolism objects (includes separation and volume calculation)
        embolism_objects = create_embolism_objects(
            mask_data=mask_data,
            patient_name=patient_name,
            spacing=spacing,
            separation_method=separation_method,
            fusion_threshold_mm=fusion_threshold,
        )

        if not embolism_objects:
            logger.warning("No embolisms found in mask")
            # Still save empty entry to global CSV for completeness
            if global_stats_path:
                if not global_stats_path.exists():
                    _initialize_embolism_stats_csv(global_stats_path)
                _append_embolism_stats_to_csv(
                    global_stats_path,
                    patient_name,
                    [],  # Empty list
                    separation_method,
                    fusion_threshold,
                )

            return {
                "patient_name": patient_name,
                "num_embolisms": 0,
                "total_volume_mm3": 0.0,
                "total_volume_cm3": 0.0,
                "embolism_objects": [],
                "output_files": {},
            }

        # Calculate totals using metrics utility
        total_mm3, total_cm3 = calculate_total_volume(embolism_objects, "volume_mm3")

        # Create output directory and nested step folders
        logger.info("Creating output directory: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        reports_dir = os.path.join(output_dir, "reports")
        volumes_dir = os.path.join(output_dir, "volumes")
        stats_dir = os.path.join(output_dir, "stats")
        individual_emboli_dir = os.path.join(volumes_dir, "individual_emboli")
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(volumes_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(individual_emboli_dir, exist_ok=True)

        # Save individual masks
        mask_paths = []
        try:
            logger.info("Saving individual embolism masks")
            mask_paths = save_embolism_masks(
                embolism_objects, mask_path, individual_emboli_dir
            )
            logger.info(
                "Successfully saved %d individual embolism masks", len(mask_paths)
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error saving individual masks: %s", e)

        # Generate report
        report_path = os.path.join(reports_dir, f"{patient_name}_embolism_report.txt")
        try:
            logger.info("Generating embolism report at %s", report_path)
            generate_embolism_report(embolism_objects, patient_name, report_path)
            logger.info("Embolism report generated successfully")
        except (OSError, ValueError) as e:
            logger.error("Error generating embolism report: %s", e)

        # Save metrics (enhanced with global stats)
        metrics_path = os.path.join(stats_dir, f"{patient_name}_embolism_metrics.csv")
        try:
            logger.info("Saving embolism metrics to %s", metrics_path)
            save_embolism_metrics(
                embolism_objects,
                metrics_path,
                patient_id=patient_name,
                global_stats_path=global_stats_path,
                separation_method=separation_method,
                fusion_threshold=fusion_threshold,
            )
            logger.info("Saved metrics for %d embolisms", len(embolism_objects))
        except (OSError, ValueError) as e:
            logger.error("Error saving embolism metrics: %s", e)

        # Prepare results
        results = {
            "patient_name": patient_name,
            "num_embolisms": len(embolism_objects),
            "total_volume_mm3": total_mm3,
            "total_volume_cm3": total_cm3,
            "embolism_objects": embolism_objects,
            "output_files": {
                "report": report_path,
                "metrics": metrics_path,
                "individual_masks": mask_paths,
            },
        }

        logger.info("Embolism analysis completed successfully")
        logger.info(
            "Found %d embolisms with total volume %.1f mm³ (%.4f cm³)",
            len(embolism_objects),
            total_mm3,
            total_cm3,
        )

        return results

    except Exception as e:
        logger.error("Error during embolism analysis: %s", e)
        raise


def analyze_embolisms_with_visualization(
    mask_path: str,
    patient_name: str,
    output_dir: str,
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
    include_colored_masks: bool = True,
    global_stats_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute complete embolism analysis pipeline including visualization with optional global statistics.

    Args:
        mask_path: Path to binary embolism mask file
        patient_name: Patient identifier for output organization
        output_dir: Directory for analysis results and visualizations
        separation_method: Embolism separation algorithm selection
        fusion_threshold: Distance threshold for component fusion (mm)
        include_colored_masks: Enable generation of color-coded visualization masks
        global_stats_path: Optional path for global statistics CSV

    Returns:
        dict[str, Any]: Extended analysis results including visualization file paths
    """
    logger.info(
        "Starting embolism analysis with visualization for patient %s", patient_name
    )

    # Ensure consistent step-level organization for embolism analysis
    analysis_output_dir = os.path.normpath(output_dir)
    if os.path.basename(analysis_output_dir) != "6_embolism_analysis":
        analysis_output_dir = os.path.join(analysis_output_dir, "6_embolism_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)
    logger.info("Embolism analysis output directory: %s", analysis_output_dir)

    try:
        # Basic embolism analysis (now with global stats support)
        results = analyze_embolism_regions(
            mask_path=mask_path,
            patient_name=patient_name,
            output_dir=analysis_output_dir,
            separation_method=separation_method,
            fusion_threshold=fusion_threshold,
            global_stats_path=global_stats_path,
        )

        if results["num_embolisms"] == 0:
            logger.warning("No embolisms found - skipping visualization")
            return results

        # Colored mask creation
        if include_colored_masks:
            logger.info("Creating colored embolism mask")
            try:
                colored_path = create_colored_embolism_masks(
                    embolism_objects=results["embolism_objects"],
                    original_mask_path=mask_path,
                    output_dir=os.path.join(analysis_output_dir, "volumes"),
                )
                results["output_files"]["colored_mask"] = colored_path

            except (OSError, ValueError) as e:
                logger.warning("Colored mask creation failed: %s", e)

        logger.info("Embolism analysis with visualization completed successfully")
        return results

    except Exception as e:
        logger.error("Error during embolism analysis with visualization: %s", e)
        raise


# =============================================================================
# SIMPLIFIED ANATOMICAL REDISTRIBUTION FUNCTIONS
# =============================================================================


def redistribute_embolism_anatomical_data(
    enriched_graph_path: str,
    embolism_metrics_path: str,
    patient_name: str,
    output_dir: str,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """
    Redistribute anatomical information from existing enriched graph and embolism metrics.
    This is a simplified version that reads existing data and redistributes it.

    Args:
        enriched_graph_path: Path to existing enriched graph JSON
        embolism_metrics_path: Path to existing embolism metrics CSV
        patient_name: Patient identifier
        output_dir: Output directory for results
        force_recompute: If True, recompute even if enriched files exist

    Returns:
        dict: Combined results with TEV statistics and enriched metrics
    """
    logger.info("Redistributing embolism anatomical data...")
    logger.info(f"  Enriched graph: {enriched_graph_path}")
    logger.info(f"  Embolism metrics: {embolism_metrics_path}")

    try:
        # Create stats subdirectory if it doesn't exist
        stats_dir = os.path.join(output_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        # Check if already processed
        enriched_metrics_path = os.path.join(
            stats_dir, f"{patient_name}_embolism_metrics_enriched.csv"
        )

        if not force_recompute and os.path.exists(enriched_metrics_path):
            logger.info("Found existing enriched files - loading from disk")
            enriched_df = pd.read_csv(enriched_metrics_path)

            # Extract TEV stats from the total row
            total_row = enriched_df[enriched_df["embolism_id"] == "total"]
            if len(total_row) > 0:
                total_data = total_row.iloc[0]
                tev_stats = {
                    "patient_name": patient_name,
                    "total_tev_mm3": total_data.get("volume_mm3", 0),
                    "vessels_with_emboli": total_data.get("vessels_with_emboli", 0),
                    "max_level_with_pe": max(
                        [
                            i
                            for i in range(1, 11)
                            if total_data.get(f"level_{i}_pct", 0) > 0
                        ],
                        default=0,
                    ),
                    "proximal_tev_ratio": (
                        total_data.get("level_1_pct", 0)
                        + total_data.get("level_2_pct", 0)
                        + total_data.get("level_3_pct", 0)
                    )
                    / 100.0,
                    "laterality_dominance": total_data.get(
                        "dominant_laterality", "unknown"
                    ),
                }
            else:
                tev_stats = {}

            return {
                "enriched_metrics_path": enriched_metrics_path,
                "enriched_dataframe": enriched_df,
                "tev_statistics": tev_stats,
                "loaded_from_existing": True,
            }

        # Load enriched graph
        logger.info("Loading enriched graph...")
        enriched_graph = load_graph(enriched_graph_path)

        # Load embolism metrics
        logger.info("Loading embolism metrics...")
        embolism_df = pd.read_csv(embolism_metrics_path)

        # Step 1: Aggregate anatomical data from enriched graph
        logger.info("Step 1: Aggregating anatomical data from enriched graph...")
        anatomical_aggregation = aggregate_anatomical_data_from_graph(enriched_graph)

        # Step 2: Redistribute to individual embolisms
        logger.info("Step 2: Redistributing anatomical data to individual embolisms...")
        enriched_df = redistribute_to_individual_embolisms(
            embolism_df, enriched_graph, anatomical_aggregation
        )

        # Step 3: Extract TEV statistics from the enriched DataFrame
        logger.info("Step 3: Extracting TEV statistics from enriched data...")
        total_row = enriched_df[enriched_df["embolism_id"] == "total"]
        if len(total_row) > 0:
            total_data = total_row.iloc[0]
            tev_stats = {
                "patient_name": patient_name,
                "total_tev_mm3": total_data.get("volume_mm3", 0),
                "vessels_with_emboli": total_data.get("vessels_with_emboli", 0),
                "max_level_with_pe": max(
                    [
                        i
                        for i in range(1, 11)
                        if total_data.get(f"level_{i}_pct", 0) > 0
                    ],
                    default=0,
                ),
                "proximal_tev_ratio": (
                    total_data.get("level_1_pct", 0)
                    + total_data.get("level_2_pct", 0)
                    + total_data.get("level_3_pct", 0)
                )
                / 100.0,
                "laterality_dominance": total_data.get(
                    "dominant_laterality", "unknown"
                ),
            }
        else:
            tev_stats = {}

        # Save results
        logger.info("Saving redistributed results...")
        enriched_df.to_csv(enriched_metrics_path, index=False)

        logger.info("Anatomical redistribution completed successfully")

        return {
            "enriched_metrics_path": enriched_metrics_path,
            "enriched_dataframe": enriched_df,
            "tev_statistics": tev_stats,
            "anatomical_aggregation": anatomical_aggregation,
        }

    except Exception as e:
        logger.error(f"Error redistributing anatomical data: {e}")
        raise RuntimeError(f"Failed to redistribute anatomical data: {e}") from e


def aggregate_anatomical_data_from_graph(enriched_graph: Any) -> dict[str, Any]:
    """
    Aggregate all anatomical data from the enriched graph.
    This creates a comprehensive summary of all anatomical information.
    """
    logger.debug("Aggregating anatomical data from enriched graph...")

    # Initialize aggregators
    total_by_level = {}
    total_by_laterality = {"right": 0.0, "left": 0.0, "unknown": 0.0}
    total_by_lobe = {
        "RUL": 0.0,
        "RML": 0.0,
        "RLL": 0.0,  # Right lung
        "LUL": 0.0,
        "LLL": 0.0,  # Left lung
        "interlobar": 0.0,
        "unknown": 0.0,  # Special categories
    }

    # Individual embolism data
    embolism_locations = {}  # embolism_id -> {levels: {}, lateralities: {}, lobes: {}}

    total_embolic_volume = 0.0
    vessels_with_emboli = 0

    edge_count = 0
    for _, _, edge_data in enriched_graph.edges(data=True):
        edge_count += 1

        # Get vessel anatomical information
        vessel_level = edge_data.get("level", None)
        vessel_laterality = edge_data.get("lateral", "unknown")
        vessel_lobe = edge_data.get("lobe", ["unknown"])

        # Handle lobe as list or None
        if isinstance(vessel_lobe, list) and len(vessel_lobe) > 0:
            lobe_str = vessel_lobe[0]
        elif vessel_lobe is None:
            lobe_str = "unknown"
        else:
            lobe_str = str(vessel_lobe) if vessel_lobe else "unknown"

        # Handle lateral = None
        if vessel_laterality is None:
            vessel_laterality = "unknown"

        # Get total embolic volume for this vessel
        total_vessel_volume = edge_data.get("total_embolism_volume_mm3", 0.0)

        if edge_count <= 5:  # Debug first 5 edges
            logger.debug(
                f"Edge {edge_count}: total_embolism_volume_mm3 = {total_vessel_volume}"
            )
            logger.debug(
                f"Edge {edge_count}: vessel_level = {vessel_level}, lateral = {vessel_laterality}, lobe = {lobe_str}"
            )

        if total_vessel_volume > 0:
            vessels_with_emboli += 1
            total_embolic_volume += total_vessel_volume

            # Aggregate by anatomical features
            if vessel_level is not None:
                if vessel_level not in total_by_level:
                    total_by_level[vessel_level] = 0.0
                total_by_level[vessel_level] += total_vessel_volume

            if vessel_laterality in total_by_laterality:
                total_by_laterality[vessel_laterality] += total_vessel_volume
            else:
                total_by_laterality["unknown"] += total_vessel_volume

            if lobe_str in total_by_lobe:
                total_by_lobe[lobe_str] += total_vessel_volume
            else:
                total_by_lobe["unknown"] += total_vessel_volume

            # Process individual embolisms in this vessel
            # Get embolism volumes dictionary - the keys ARE the embolism IDs!
            embolisms_volumes = edge_data.get("embolisms_volumes_mm3", {})

            if edge_count <= 5:
                logger.debug(
                    f"Edge {edge_count}: embolisms_volumes_mm3 = {embolisms_volumes}"
                )

            # Extract embolism IDs from the dictionary keys
            for embolism_id, volume in embolisms_volumes.items():
                if volume > 0:
                    try:
                        emb_id = int(embolism_id)
                    except (ValueError, TypeError):
                        continue

                    if emb_id not in embolism_locations:
                        embolism_locations[emb_id] = {
                            "levels": {},
                            "lateralities": {},
                            "lobes": {},
                            "total_volume": 0.0,
                        }

                    # Add to individual embolism
                    embolism_locations[emb_id]["total_volume"] += volume

                    if vessel_level is not None:
                        if vessel_level not in embolism_locations[emb_id]["levels"]:
                            embolism_locations[emb_id]["levels"][vessel_level] = 0.0
                        embolism_locations[emb_id]["levels"][vessel_level] += volume

                    if (
                        vessel_laterality
                        not in embolism_locations[emb_id]["lateralities"]
                    ):
                        embolism_locations[emb_id]["lateralities"][
                            vessel_laterality
                        ] = 0.0
                    embolism_locations[emb_id]["lateralities"][
                        vessel_laterality
                    ] += volume

                    if lobe_str not in embolism_locations[emb_id]["lobes"]:
                        embolism_locations[emb_id]["lobes"][lobe_str] = 0.0
                    embolism_locations[emb_id]["lobes"][lobe_str] += volume

                    if edge_count <= 5:
                        logger.debug(
                            f"Edge {edge_count}: Added {volume:.2f} mm³ to embolism {emb_id} (level {vessel_level}, {vessel_laterality}, {lobe_str})"
                        )

    # Calculate percentages
    def calc_percentages(data_dict, total):
        return {
            k: (v / total * 100) if total > 0 else 0.0 for k, v in data_dict.items()
        }

    total_by_level_pct = calc_percentages(total_by_level, total_embolic_volume)
    total_by_laterality_pct = calc_percentages(
        total_by_laterality, total_embolic_volume
    )
    total_by_lobe_pct = calc_percentages(total_by_lobe, total_embolic_volume)

    # Calculate individual percentages
    for _, data in embolism_locations.items():
        total_vol = data["total_volume"]
        data["levels_pct"] = calc_percentages(data["levels"], total_vol)
        data["lateralities_pct"] = calc_percentages(data["lateralities"], total_vol)
        data["lobes_pct"] = calc_percentages(data["lobes"], total_vol)

    logger.debug(f"Aggregated data for {len(embolism_locations)} embolisms")
    logger.debug(f"Total embolic volume: {total_embolic_volume:.2f} mm³")
    logger.debug(f"Vessels with emboli: {vessels_with_emboli}")
    logger.debug(f"Total edges processed: {edge_count}")

    # Log individual embolism summaries
    for emb_id, data in embolism_locations.items():
        logger.debug(
            f"Embolism {emb_id}: {data['total_volume']:.2f} mm³, levels: {list(data['levels'].keys())}, lateralities: {list(data['lateralities'].keys())}, lobes: {list(data['lobes'].keys())}"
        )

    return {
        "total_embolic_volume": total_embolic_volume,
        "vessels_with_emboli": vessels_with_emboli,
        "total_by_level": total_by_level,
        "total_by_laterality": total_by_laterality,
        "total_by_lobe": total_by_lobe,
        "total_by_level_pct": total_by_level_pct,
        "total_by_laterality_pct": total_by_laterality_pct,
        "total_by_lobe_pct": total_by_lobe_pct,
        "embolism_locations": embolism_locations,
        "max_level_with_pe": max(total_by_level.keys()) if total_by_level else 0,
    }


def redistribute_to_individual_embolisms(
    embolism_df: pd.DataFrame, enriched_graph: Any, anatomical_aggregation: dict
) -> pd.DataFrame:
    """
    Add anatomical percentage columns to the embolism DataFrame using precomputed aggregation.
    """
    logger.debug("Redistributing anatomical data to individual embolisms...")

    # Make a copy to avoid modifying original
    enriched_df = embolism_df.copy()

    # Add anatomical columns including vessels_with_emboli
    anatomical_columns = {
        # Level percentages (1-10)
        **{f"level_{i}_pct": 0.0 for i in range(1, 11)},
        # Laterality percentages
        "right_lung_pct": 0.0,
        "left_lung_pct": 0.0,
        "unknown_laterality_pct": 0.0,
        # Lobe percentages
        "RUL_pct": 0.0,
        "RML_pct": 0.0,
        "RLL_pct": 0.0,
        "LUL_pct": 0.0,
        "LLL_pct": 0.0,
        "interlobar_pct": 0.0,
        "unknown_lobe_pct": 0.0,
        # Dominant features
        "dominant_level": "unknown",
        "dominant_laterality": "unknown",
        "dominant_lobe": "unknown",
        # Clinical indicators
        "is_proximal": False,
        "is_bilateral": False,
        # Vessels with emboli (specific to each embolism or total)
        "vessels_with_emboli": 0,
    }

    for col, default_val in anatomical_columns.items():
        enriched_df[col] = default_val

    embolism_locations = anatomical_aggregation["embolism_locations"]

    # Calculate vessels_with_emboli for each individual embolism
    individual_vessel_counts = {}
    total_vessels_with_any_emboli = set()  # Use set to avoid double counting

    # Go through enriched graph to count vessels per embolism
    for u, v, edge_data in enriched_graph.edges(data=True):
        total_vessel_volume = edge_data.get("total_embolism_volume_mm3", 0.0)

        if total_vessel_volume > 0:
            # This vessel has emboli - add to total set
            vessel_id = (u, v)  # Use edge as vessel identifier
            total_vessels_with_any_emboli.add(vessel_id)

            # Count per individual embolism
            embolisms_volumes = edge_data.get("embolisms_volumes_mm3", {})
            for embolism_id, volume in embolisms_volumes.items():
                if volume > 0:
                    try:
                        emb_id = int(embolism_id)
                        if emb_id not in individual_vessel_counts:
                            individual_vessel_counts[emb_id] = set()
                        individual_vessel_counts[emb_id].add(vessel_id)
                    except (ValueError, TypeError):
                        continue

    # Convert sets to counts
    individual_vessel_counts = {
        emb_id: len(vessels) for emb_id, vessels in individual_vessel_counts.items()
    }
    total_unique_vessels_with_emboli = len(total_vessels_with_any_emboli)

    logger.debug(f"Individual vessel counts: {individual_vessel_counts}")
    logger.debug(
        f"Total unique vessels with emboli: {total_unique_vessels_with_emboli}"
    )

    # Fill data for each embolism
    for idx, row in enriched_df.iterrows():
        embolism_id = row["embolism_id"]

        # Handle total row separately
        if embolism_id == "total":
            # Use aggregated totals
            for level in range(1, 11):
                enriched_df.at[idx, f"level_{level}_pct"] = anatomical_aggregation[
                    "total_by_level_pct"
                ].get(level, 0.0)

            enriched_df.at[idx, "right_lung_pct"] = anatomical_aggregation[
                "total_by_laterality_pct"
            ].get("right", 0.0)
            enriched_df.at[idx, "left_lung_pct"] = anatomical_aggregation[
                "total_by_laterality_pct"
            ].get("left", 0.0)
            enriched_df.at[idx, "unknown_laterality_pct"] = anatomical_aggregation[
                "total_by_laterality_pct"
            ].get("unknown", 0.0)

            for lobe in ["RUL", "RML", "RLL", "LUL", "LLL", "interlobar", "unknown"]:
                enriched_df.at[idx, f"{lobe}_pct"] = anatomical_aggregation[
                    "total_by_lobe_pct"
                ].get(lobe, 0.0)

            # Dominant features for total
            if anatomical_aggregation["total_by_level_pct"]:
                dominant_level = max(
                    anatomical_aggregation["total_by_level_pct"].items(),
                    key=lambda x: x[1],
                )[0]
                enriched_df.at[idx, "dominant_level"] = str(dominant_level)

            if anatomical_aggregation["total_by_laterality_pct"]:
                dominant_laterality = max(
                    anatomical_aggregation["total_by_laterality_pct"].items(),
                    key=lambda x: x[1],
                )[0]
                enriched_df.at[idx, "dominant_laterality"] = dominant_laterality

            if anatomical_aggregation["total_by_lobe_pct"]:
                dominant_lobe = max(
                    anatomical_aggregation["total_by_lobe_pct"].items(),
                    key=lambda x: x[1],
                )[0]
                enriched_df.at[idx, "dominant_lobe"] = dominant_lobe

            # Clinical indicators for total
            proximal_levels = [1, 2, 3]
            proximal_pct = sum(
                anatomical_aggregation["total_by_level_pct"].get(level, 0)
                for level in proximal_levels
            )
            enriched_df.at[idx, "is_proximal"] = proximal_pct > 50.0

            right_pct = anatomical_aggregation["total_by_laterality_pct"].get(
                "right", 0
            )
            left_pct = anatomical_aggregation["total_by_laterality_pct"].get("left", 0)
            enriched_df.at[idx, "is_bilateral"] = right_pct > 0 and left_pct > 0

            # Total unique vessels with emboli (calculated independently)
            enriched_df.at[
                idx, "vessels_with_emboli"
            ] = total_unique_vessels_with_emboli

            continue

        # Individual embolism
        try:
            emb_id = int(embolism_id)
        except (ValueError, TypeError):
            continue

        if emb_id in embolism_locations:
            data = embolism_locations[emb_id]

            # Level percentages
            for level in range(1, 11):
                enriched_df.at[idx, f"level_{level}_pct"] = data["levels_pct"].get(
                    level, 0.0
                )

            # Laterality percentages
            enriched_df.at[idx, "right_lung_pct"] = data["lateralities_pct"].get(
                "right", 0.0
            )
            enriched_df.at[idx, "left_lung_pct"] = data["lateralities_pct"].get(
                "left", 0.0
            )
            enriched_df.at[idx, "unknown_laterality_pct"] = data[
                "lateralities_pct"
            ].get("unknown", 0.0)

            # Lobe percentages
            for lobe in ["RUL", "RML", "RLL", "LUL", "LLL", "interlobar", "unknown"]:
                enriched_df.at[idx, f"{lobe}_pct"] = data["lobes_pct"].get(lobe, 0.0)

            # Dominant features
            if data["levels_pct"]:
                dominant_level = max(data["levels_pct"].items(), key=lambda x: x[1])[0]
                enriched_df.at[idx, "dominant_level"] = str(dominant_level)

            if data["lateralities_pct"]:
                dominant_laterality = max(
                    data["lateralities_pct"].items(), key=lambda x: x[1]
                )[0]
                enriched_df.at[idx, "dominant_laterality"] = dominant_laterality

            if data["lobes_pct"]:
                dominant_lobe = max(data["lobes_pct"].items(), key=lambda x: x[1])[0]
                enriched_df.at[idx, "dominant_lobe"] = dominant_lobe

            # Clinical indicators
            proximal_levels = [1, 2, 3]
            proximal_pct = sum(
                data["levels_pct"].get(level, 0) for level in proximal_levels
            )
            enriched_df.at[idx, "is_proximal"] = proximal_pct > 50.0

            right_pct = data["lateralities_pct"].get("right", 0)
            left_pct = data["lateralities_pct"].get("left", 0)
            enriched_df.at[idx, "is_bilateral"] = right_pct > 0 and left_pct > 0

            # Number of vessels affected by this specific embolism
            enriched_df.at[idx, "vessels_with_emboli"] = individual_vessel_counts.get(
                emb_id, 0
            )

    logger.debug("Successfully redistributed anatomical data to all embolisms")
    return enriched_df


def load_existing_anatomical_data(
    enriched_metrics_path: str, tev_stats_path: str
) -> dict[str, Any]:
    """Load existing anatomical data from files."""
    logger.debug("Loading existing anatomical data from files...")

    enriched_df = pd.read_csv(enriched_metrics_path)
    tev_df = pd.read_csv(tev_stats_path)
    tev_stats = tev_df.iloc[0].to_dict() if len(tev_df) > 0 else {}

    return {
        "enriched_metrics_path": enriched_metrics_path,
        "tev_statistics_path": tev_stats_path,
        "enriched_dataframe": enriched_df,
        "tev_statistics": tev_stats,
        "loaded_from_existing": True,
    }
