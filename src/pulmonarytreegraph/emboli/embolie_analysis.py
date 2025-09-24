"""
Pulmonary Embolism Analysis Module

This module provides analysis functions for embolism
evaluation and workflow orchestration.

Main Functions:
- analyze_embolism_regions(): Complete basic embolism analysis pipeline
- analyze_embolisms_with_visualization(): Complete pipeline with visualization output
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from pulmonarytreegraph.emboli.embolie_construction import (
    create_embolism_objects,
)
from pulmonarytreegraph.utils.io import (
    create_colored_embolism_masks,
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
# DATA EXPORT AND PERSISTENCE
# =============================================================================


def save_embolism_metrics(
    embolism_objects: list[dict[str, Any]], output_path: str
) -> None:
    """
    Export structured embolism metrics to CSV format

    Args:
        embolism_objects: List of embolism data dictionaries with complete metadata
        output_path: Full file path for CSV export (including .csv extension)

    Returns:
        None: Writes CSV file to specified path

    Algorithm:
        1. Convert embolism objects to tabular format
        2. Export to CSV with headers for database compatibility
        3. Handle empty cases with appropriate placeholder content
    """
    logger.debug("Saving embolism metrics to %s", output_path)

    try:
        # Prepare data for CSV
        csv_data = []

        for embolism in embolism_objects:
            row = {
                "embolism_id": embolism["id"],
                "patient_name": embolism["patient_name"],
                "voxel_count": embolism["voxel_count"],
                "volume_mm3": embolism["volume_mm3"],
                "volume_cm3": embolism["volume_cm3"],
            }
            csv_data.append(row)

        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            logger.debug("Saved metrics for %d embolisms", len(csv_data))
        else:
            # Create empty CSV with headers
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# No embolism data available\n")
                f.write(
                    "embolism_id,patient_name,voxel_count,volume_mm3,volume_cm3,spacing_x,spacing_y,spacing_z\n"
                )

    except (OSError, ValueError) as e:
        logger.error("Error saving embolism metrics: %s", e)
        raise RuntimeError(f"Error saving embolism metrics: {e}") from e


# =============================================================================
# CORE ANALYSIS PIPELINES
# =============================================================================


def analyze_embolism_regions(
    mask_path: str,
    patient_name: str,
    output_dir: str,
    separation_method: str = "connected_components",
    fusion_threshold: float = 10.0,
) -> dict[str, Any]:
    """
    Execute complete pipeline for embolism analysis.

    Main orchestrating function that implements the full workflow for pulmonary
    embolism detection, quantification, and basic reporting. Processes medical
    imaging masks to identify individual embolisms, calculate volumes, generate
    reports, and export structured data for clinical analysis.

    Args:
        mask_path: Absolute path to binary embolism mask file (NIfTI format)
        patient_name: Unique patient identifier for traceability and reporting
        output_dir: Output directory path for analysis results and generated files
        separation_method: Algorithm for embolism separation:
                          - "connected_components": Standard connected components
                          - "connected_components_with_threshold": With distance fusion
        fusion_threshold: Distance threshold in millimeters for component fusion

    Returns:
        dict[str, Any]: Analysis results containing:
            - patient_name: Patient identifier
            - num_embolisms: Total number of detected embolisms
            - total_volume_mm3: Total embolic volume in cubic millimeters
            - total_volume_cm3: Total embolic volume in cubic centimeters
            - embolism_objects: List of individual embolism data objects
            - output_files: Dictionary of generated file paths

    Algorithm :
        1. Load embolism mask and validate input
        2. Identify and separate individual embolisms using specified method
        3. Calculate volumes and create structured embolism objects
        4. Generate detailed text report summarizing findings
        5. Export embolism metrics to CSV for database integration
        6. Save individual embolism masks for visualization and review
        7. Return results dictionary with summary and file paths

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

        # Create output directory
        logger.info("Creating output directory: %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save individual masks
        mask_paths = []
        try:
            logger.info("Saving individual embolism masks")
            mask_paths = save_embolism_masks(embolism_objects, mask_path, output_dir)
            logger.info(
                "Successfully saved %d individual embolism masks", len(mask_paths)
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error saving individual masks: %s", e)

        # Generate report
        report_path = os.path.join(output_dir, f"{patient_name}_embolism_report.txt")
        try:
            logger.info("Generating embolism report at %s", report_path)
            generate_embolism_report(embolism_objects, patient_name, report_path)
            logger.info("Embolism report generated successfully")
        except (OSError, ValueError) as e:
            logger.error("Error generating embolism report: %s", e)

        # Save metrics
        metrics_path = os.path.join(output_dir, f"{patient_name}_embolism_metrics.csv")
        try:
            logger.info("Saving embolism metrics to %s", metrics_path)
            save_embolism_metrics(embolism_objects, metrics_path)
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
) -> dict[str, Any]:
    """
    Execute complete embolism analysis pipeline including visualization,
    meaning the generation of color-coded masks with unique IDs per embolism.

    Args:
        mask_path: Path to binary embolism mask file
        patient_name: Patient identifier for output organization
        output_dir: Directory for analysis results and visualizations
        separation_method: Embolism separation algorithm selection
        fusion_threshold: Distance threshold for component fusion (mm)
        include_colored_masks: Enable generation of color-coded visualization masks

    Returns:
        dict[str, Any]: Extended analysis results including visualization file paths

    Algorithm:
        1. Execute complete basic analysis pipeline
        2. Generate color-coded masks with unique IDs per embolism
        3. Combine analysis results with visualization file paths
    """
    logger.info(
        "Starting embolism analysis with visualization for patient %s", patient_name
    )

    try:
        # Basic embolism analysis
        results = analyze_embolism_regions(
            mask_path=mask_path,
            patient_name=patient_name,
            output_dir=output_dir,
            separation_method=separation_method,
            fusion_threshold=fusion_threshold,
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
                    output_dir=output_dir,
                )
                results["output_files"]["colored_mask"] = colored_path

            except (OSError, ValueError) as e:
                logger.warning("Colored mask creation failed: %s", e)

        logger.info("Embolism analysis with visualization completed successfully")
        return results

    except Exception as e:
        logger.error("Error during embolism analysis with visualization: %s", e)
        raise
