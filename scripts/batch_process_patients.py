#!/usr/bin/env python3
"""
Batch processing script for multiple patients.
Processes folders containing artery masks and lung masks through the complete pipeline.
Adapted for the specific directory structure with separate folders for different mask types.
"""

import argparse
import glob
import logging
import os
import shutil
import sys
import time
from typing import Any

from pulmonarytreegraph.pipeline.core import process_patient_complete_pipeline
from pulmonarytreegraph.scores.compute_scores import compute_literature_scores


def setup_logging(output_dir: str = None, log_level: str = "INFO") -> str:
    """Setup logging for batch processing."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Generate a consistent batch run identifier
    log_id = str(int(time.time()))

    # Create logs directory if output_dir is provided
    if output_dir:
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"batch_processing_{log_id}.log")

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        print(f"Logging to file: {log_file}")
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    return log_id


def find_patients(
    data_dir: str = None,
    arteries_dir: str = None,
    embolies_dir: str = None,
    images_dir: str = None,
    lungs_dir: str = None,
    log_id: str = None,
) -> list[dict[str, str]]:
    """
    Find all patient files in the specific directory structure.

    Expected structure:
    data_dir/
    ├── arteries_mask/
    │   ├── 0028.nii.gz
    │   └── XXXX.nii.gz
    ├── embolies_mask/
    │   ├── 0028.nii.gz
    │   └── XXXX.nii.gz
    ├── images/
    │   ├── 0028.nii.gz
    │   └── XXXX.nii.gz
    └── lungs_mask/
        ├── 0028/
        │   ├── lung_left.nii.gz
        │   ├── lung_right.nii.gz
        │   └── ...
        └── XXXX/
            └── ...

    Args:
        data_dir: Base data directory containing the structured folders.
        arteries_dir: Path to the arteries mask folder.
        embolies_dir: Path to the embolies mask folder.
        images_dir: Path to the images folder.
        lungs_dir: Path to the lungs mask folder.

    Returns:
        List of patient dictionaries with file paths.
    """
    patients = []

    # If data_dir is provided, construct folder paths
    if data_dir:
        arteries_dir = os.path.join(data_dir, "arteries_mask")
        embolies_dir = os.path.join(data_dir, "embolies_mask")
        images_dir = os.path.join(data_dir, "images")
        lungs_dir = os.path.join(data_dir, "lungs_mask")

    # Check if required directories exist
    if not arteries_dir or not os.path.exists(arteries_dir):
        raise FileNotFoundError(f"Arteries mask directory not found: {arteries_dir}")
    if not lungs_dir or not os.path.exists(lungs_dir):
        raise FileNotFoundError(f"Lungs mask directory not found: {lungs_dir}")

    # Find all artery files
    artery_pattern = os.path.join(arteries_dir, "*.nii.gz")
    artery_files = glob.glob(artery_pattern)

    if not artery_files:
        raise FileNotFoundError(f"No artery files found with pattern: {artery_pattern}")

    for artery_file in artery_files:
        # Extract patient ID from filename
        filename = os.path.basename(artery_file)
        patient_id = filename.replace(".nii.gz", "")

        patient_info = {
            "id": patient_id,
            "artery_mask": artery_file,
            "lung_masks_dir": None,
            "embolism_mask": None,
            "image": None,
            "log_id": log_id,
        }

        # Look for lung masks directory
        lung_masks_dir = os.path.join(lungs_dir, patient_id)
        if os.path.exists(lung_masks_dir) and os.path.isdir(lung_masks_dir):
            lung_files = glob.glob(os.path.join(lung_masks_dir, "*.nii.gz"))
            if lung_files:
                patient_info["lung_masks_dir"] = lung_masks_dir
            else:
                print(
                    f"Warning: Lung masks directory exists but contains no .nii.gz files: {lung_masks_dir}"
                )
        else:
            print(
                f"Warning: No lung masks directory found for patient {patient_id}: {lung_masks_dir}"
            )

        # Look for embolism mask
        if embolies_dir:
            embolism_file = os.path.join(embolies_dir, f"{patient_id}.nii.gz")
            if os.path.exists(embolism_file):
                patient_info["embolism_mask"] = embolism_file

        # Look for original image
        if images_dir:
            image_file = os.path.join(images_dir, f"{patient_id}.nii.gz")
            if os.path.exists(image_file):
                patient_info["image"] = image_file

        patients.append(patient_info)

    return sorted(patients, key=lambda x: x["id"])


def validate_patient_files(patient_info: dict[str, str]) -> dict[str, Any]:
    """
    Validate that all required files exist for a patient.

    Args:
        patient_info: Patient information dictionary
        logger: Logger instance

    Returns:
        Dictionary with validation results
    """
    validation = {"valid": True, "warnings": [], "errors": []}

    patient_id = patient_info["id"]

    # Check required files
    if not os.path.exists(patient_info["artery_mask"]):
        validation["errors"].append(
            f"Artery mask not found: {patient_info['artery_mask']}"
        )
        validation["valid"] = False

    if not patient_info["lung_masks_dir"]:
        validation["errors"].append(
            f"No lung masks directory found for patient {patient_id}"
        )
        validation["valid"] = False
    elif not os.path.exists(patient_info["lung_masks_dir"]):
        validation["errors"].append(
            f"Lung masks directory does not exist: {patient_info['lung_masks_dir']}"
        )
        validation["valid"] = False
    else:
        # Check lung mask files
        lung_files = glob.glob(os.path.join(patient_info["lung_masks_dir"], "*.nii.gz"))
        if not lung_files:
            validation["errors"].append(
                f"No lung mask files found in: {patient_info['lung_masks_dir']}"
            )
            validation["valid"] = False
        else:
            # Check for expected lung files
            expected_files = [
                "lung_left.nii.gz",
                "lung_right.nii.gz",
                "lung_upper_lobe_left.nii.gz",
                "lung_upper_lobe_right.nii.gz",
                "lung_lower_lobe_left.nii.gz",
                "lung_lower_lobe_right.nii.gz",
                "lung_middle_lobe_right.nii.gz",
                "lungs.nii.gz",
            ]

            found_files = [os.path.basename(f) for f in lung_files]
            missing_files = [f for f in expected_files if f not in found_files]

            if missing_files:
                validation["warnings"].append(
                    f"Some expected lung files missing: {missing_files}"
                )

    # Check optional files
    if not patient_info["embolism_mask"]:
        validation["warnings"].append(
            f"No embolism mask found for patient {patient_id}"
        )
    elif not os.path.exists(patient_info["embolism_mask"]):
        validation["warnings"].append(
            f"Embolism mask path exists in config but file not found: {patient_info['embolism_mask']}"
        )

    if not patient_info["image"]:
        validation["warnings"].append(
            f"No original image found for patient {patient_id}"
        )
    elif not os.path.exists(patient_info["image"]):
        validation["warnings"].append(
            f"Original image path exists in config but file not found: {patient_info['image']}"
        )

    return validation


def process_single_patient(
    patient_info: dict[str, str],
    output_base_dir: str,
    processing_options: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Process a single patient through the pipeline.

    Args:
        patient_info: Dictionary with patient file paths
        output_base_dir: Base output directory
        processing_options: Processing configuration
        logger: Logger instance

    Returns:
        Processing results dictionary
    """
    patient_id = patient_info["id"]
    start_time = time.time()

    logger.info("=" * 80)
    logger.info(f"PROCESSING PATIENT: {patient_id}")
    logger.info("=" * 80)

    # Create patient output directory
    patient_output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    # Validate patient files
    validation = validate_patient_files(patient_info)

    # Log validation results
    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)

    if validation["errors"]:
        for error in validation["errors"]:
            logger.error(error)

    if not validation["valid"]:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "error": "File validation failed",
            "processing_time": time.time() - start_time,
            "validation": validation,
        }

    # Log patient files
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"Artery mask: {patient_info['artery_mask']}")
    logger.info(f"Lung masks dir: {patient_info['lung_masks_dir']}")
    logger.info(f"Embolism mask: {patient_info.get('embolism_mask', 'Not available')}")
    logger.info(f"Original image: {patient_info.get('image', 'Not available')}")
    logger.info(f"Output directory: {patient_output_dir}")

    # List lung mask files for verification
    if patient_info["lung_masks_dir"]:
        lung_files = glob.glob(os.path.join(patient_info["lung_masks_dir"], "*.nii.gz"))
        logger.info(f"Found {len(lung_files)} lung mask files:")
        for lung_file in sorted(lung_files):
            logger.info(f"  - {os.path.basename(lung_file)}")

    try:
        # Run the complete pipeline
        results = process_patient_complete_pipeline(
            artery_mask_path=patient_info["artery_mask"],
            lung_masks_dir=patient_info["lung_masks_dir"],
            log_id=patient_info["log_id"],
            embolism_mask_path=patient_info.get("embolism_mask"),
            output_dir=patient_output_dir,
            radius_ratio_threshold=processing_options.get(
                "radius_ratio_threshold", 0.3
            ),
            coordinate_format=processing_options.get("coordinate_format", "both"),
            save_skeletons=processing_options.get("save_skeletons", False),
            save_colored_volumes=processing_options.get("save_colored_volumes"),
            save_all_step_volumes=processing_options.get(
                "save_all_step_volumes", False
            ),
            save_cycle_skeletons=processing_options.get("save_cycle_skeletons", False),
            embolism_separation_method=processing_options.get(
                "embolism_separation_method", "connected_components"
            ),
            embolism_fusion_threshold=processing_options.get(
                "embolism_fusion_threshold", 10.0
            ),
            skip_steps=processing_options.get("skip_steps", []),
            stop_after=processing_options.get("stop_after"),
            diameter_threshold=processing_options.get("diameter_threshold", 0.7),
            length_threshold=processing_options.get("length_threshold", 1.5),
            compute_transversal_obstruction=processing_options.get(
                "compute_transversal_obstruction", False
            ),
            transversal_obstruction_kwargs=processing_options.get(
                "transversal_obstruction_kwargs", {}
            ),
            compute_qanadli=processing_options.get("compute_qanadli", False),
            compute_mastora=processing_options.get("compute_mastora", False),
            obstruction_attr=processing_options.get(
                "obstruction_attr", "volumetric_obstruction"
            ),
        )

        # Copy enriched graphs to central directory if requested
        if processing_options.get("save_enriched_graphs") and processing_options.get(
            "enriched_graphs_dir"
        ):
            copy_enriched_graphs(
                patient_output_dir,
                processing_options["enriched_graphs_dir"],
                patient_id,
                logger,
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Patient {patient_id} completed successfully in {processing_time:.1f} seconds"
        )

        # Add timing and patient info to results
        results.update(
            {
                "patient_id": patient_id,
                "processing_time": processing_time,
                "status": "completed",
                "validation": validation,
            }
        )

        return results

    except (FileNotFoundError, ValueError) as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Patient {patient_id} failed after {processing_time:.1f} seconds: {str(e)}"
        )
        logger.exception("Full traceback:")

        return {
            "patient_id": patient_id,
            "processing_time": processing_time,
            "status": "failed",
            "error": str(e),
            "validation": validation,
        }
    except RuntimeError as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Patient {patient_id} encountered a runtime error after {processing_time:.1f} seconds: {str(e)}"
        )
        logger.exception("Runtime error traceback:")
        return {
            "patient_id": patient_id,
            "processing_time": processing_time,
            "status": "failed",
            "error": f"Runtime error: {str(e)}",
            "validation": validation,
        }


def copy_enriched_graphs(
    patient_output_dir: str,
    enriched_graphs_dir: str,
    patient_id: str,
    logger: logging.Logger,
):
    """
    Copy enriched graph files from patient output directory to central enriched_graphs directory.

    Args:
        patient_output_dir: Patient's individual output directory
        enriched_graphs_dir: Central directory for all enriched graphs
        patient_id: Patient identifier
        logger: Logger instance
    """

    # Based on the core.py logic, enriched graphs are saved with these patterns:
    enriched_graph_files = [
        "enriched_graph.json",
        "enriched_graph_real_coords.json",
    ]

    copied_files = []

    for filename in enriched_graph_files:
        source_path = os.path.join(patient_output_dir, filename)

        if os.path.exists(source_path):
            # Add patient ID prefix to avoid naming conflicts
            new_filename = f"{patient_id}_{filename}"
            dest_path = os.path.join(enriched_graphs_dir, new_filename)

            try:
                shutil.copy2(source_path, dest_path)
                copied_files.append(new_filename)
                logger.debug(f"Copied enriched graph: {filename} -> {new_filename}")
            except Exception as e:
                logger.warning(f"Failed to copy enriched graph {filename}: {e}")

    if copied_files:
        logger.info(
            f"Copied {len(copied_files)} enriched graph files for patient {patient_id}"
        )
    else:
        logger.warning(f"No enriched graph files found for patient {patient_id}")


def generate_batch_report(
    results: list[dict[str, Any]],
    report_dir: str,
    logger: logging.Logger,
    log_id: str | None = None,
):
    """Generate a comprehensive batch processing report."""
    os.makedirs(report_dir, exist_ok=True)
    report_filename = (
        f"batch_processing_report_{log_id}.txt"
        if log_id
        else "batch_processing_report.txt"
    )
    report_path = os.path.join(report_dir, report_filename)

    successful = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    total_time = sum(r["processing_time"] for r in results)
    avg_time = total_time / len(results) if results else 0

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BATCH PROCESSING REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total patients processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Success rate: {len(successful)/len(results)*100:.1f}%\n\n")

        f.write(
            f"Total processing time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)\n"
        )
        f.write(f"Average time per patient: {avg_time:.1f} seconds\n\n")

        if successful:
            f.write("SUCCESSFUL PATIENTS:\n")
            f.write("-" * 30 + "\n")
            for result in successful:
                f.write(f"  {result['patient_id']}: {result['processing_time']:.1f}s\n")
                # Add embolism summary if available
                if "embolism_analysis" in result:
                    emb = result["embolism_analysis"]
                    f.write(
                        f"    Embolisms: {emb.get('num_embolisms', 0)}, Volume: {emb.get('total_volume_mm3', 0):.2f} mm³\n"
                    )
            f.write("\n")

        if failed:
            f.write("FAILED PATIENTS:\n")
            f.write("-" * 30 + "\n")
            for result in failed:
                f.write(
                    f"  {result['patient_id']}: {result.get('error', 'Unknown error')}\n"
                )
            f.write("\n")

        # Validation summary
        f.write("VALIDATION SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            validation = result.get("validation", {})
            warnings = validation.get("warnings", [])
            errors = validation.get("errors", [])

            f.write(f"\n{result['patient_id']}:\n")
            if errors:
                f.write("  Errors:\n")
                for error in errors:
                    f.write(f"    - {error}\n")
            if warnings:
                f.write("  Warnings:\n")
                for warning in warnings:
                    f.write(f"    - {warning}\n")
            if not errors and not warnings:
                f.write("  No issues detected\n")

    logger.info(f"Batch report saved to: {report_path}")


def run_scores_only_mode(args, logger):
    """Run scoring on existing enriched graphs."""

    logger.info("Running scores-only mode")
    logger.info("=" * 50)

    # Handle the --compute-all-scores flag
    if args.compute_all_scores:
        args.compute_qanadli = True
        args.compute_mastora = True

    if not (args.compute_qanadli or args.compute_mastora):
        logger.error(
            "Must specify at least one score to compute (use --compute-qanadli, --compute-mastora, or --compute-all-scores)"
        )
        return False

    # Find enriched graphs in output directory or specific enriched_graphs folder
    if args.save_enriched_graphs:
        enriched_graphs_dir = os.path.join(args.output_dir, "enriched_graphs")
    else:
        enriched_graphs_dir = args.output_dir

    logger.info(f"Looking for enriched graphs in: {enriched_graphs_dir}")
    logger.info(f"Obstruction attribute: {args.obstruction_attr}")
    logger.info(f"Qanadli score: {args.compute_qanadli}")
    logger.info(f"Mastora scores: {args.compute_mastora}")

    try:
        # Import the helper functions from scores.py
        sys.path.append(os.path.dirname(__file__))

        # Import directly or copy the functions we need
        graph_files = find_enriched_graphs_for_scoring(enriched_graphs_dir)
        logger.info(f"Found {len(graph_files)} enriched graph files")

        if not graph_files:
            logger.error("No enriched graph files found")
            return False

        # Filter by patient list if provided
        if args.patient_list and os.path.exists(args.patient_list):
            with open(args.patient_list) as f:
                allowed_patients = set(line.strip() for line in f if line.strip())
            graph_files = [
                g for g in graph_files if g["patient_id"] in allowed_patients
            ]
            logger.info(f"Filtered to {len(graph_files)} patients from patient list")

        # Exclude patients if specified
        if args.exclude_patients:
            exclude_set = set(args.exclude_patients)
            graph_files = [g for g in graph_files if g["patient_id"] not in exclude_set]
            logger.info(f"Excluded patients, {len(graph_files)} remaining")

        if not graph_files:
            logger.error("No graphs to process after filtering")
            return False

        # Process graphs
        results = []
        start_time = time.time()

        for i, graph_file in enumerate(graph_files, 1):
            logger.info(
                f"\n>>> Processing {i}/{len(graph_files)}: {graph_file['patient_id']}"
            )

            result = compute_scores_for_single_graph(
                graph_file=graph_file,
                compute_qanadli=args.compute_qanadli,
                compute_mastora=args.compute_mastora,
                obstruction_attr=args.obstruction_attr,
                global_stats_dir=getattr(args, "global_stats_dir", None),
                logger=logger,
            )
            results.append(result)

        # Summary
        total_time = time.time() - start_time
        successful = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] == "failed"]
        skipped = [r for r in results if r["status"] == "skipped"]

        logger.info("\n" + "=" * 50)
        logger.info("SCORES-ONLY MODE COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Skipped: {len(skipped)}")
        logger.info(f"Total time: {total_time:.1f} seconds")

        if failed:
            logger.info("\nFailed patients:")
            for r in failed:
                logger.info(f"  {r['patient_id']}: {r['error']}")

        if skipped:
            logger.info("\nSkipped patients:")
            for r in skipped:
                logger.info(f"  {r['patient_id']}: {r['reason']}")

        return len(failed) == 0

    except Exception as e:
        logger.error(f"Error in scores-only mode: {e}")
        logger.exception("Full traceback:")
        return False


def find_enriched_graphs_for_scoring(input_path: str) -> list[dict]:
    """
    Find enriched graph files in directory structure.
    Adapted from scores.py for batch processing.

    Args:
        input_path: Path to directory containing enriched graphs

    Returns:
        List of dictionaries with graph file info
    """
    graph_files = []

    if os.path.isfile(input_path):
        # Single file
        if input_path.endswith(".json"):
            patient_id = extract_patient_id_from_graph_path(input_path)
            graph_files.append(
                {
                    "path": input_path,
                    "patient_id": patient_id,
                    "output_dir": os.path.dirname(input_path),
                }
            )
    else:
        # Directory - search for enriched graphs
        search_patterns = [
            "*/enriched_graph*.json",
            "enriched_graph*.json",
            "*_enriched_graph*.json",
            "*/enriched_graph_real_coords.json",  # Include real coordinate versions
            "*_enriched_graph_real_coords.json",
        ]

        for pattern in search_patterns:
            search_path = os.path.join(input_path, pattern)
            files = glob.glob(search_path, recursive=True)

            for file_path in files:
                patient_id = extract_patient_id_from_graph_path(file_path)
                output_dir = os.path.dirname(file_path)

                # Avoid duplicates - prefer the standard enriched_graph.json over real_coords version
                if any(
                    g["patient_id"] == patient_id and "real_coords" not in g["path"]
                    for g in graph_files
                ):
                    continue

                graph_files.append(
                    {
                        "path": file_path,
                        "patient_id": patient_id,
                        "output_dir": output_dir,
                    }
                )

    # Remove duplicates, preferring non-real-coords versions
    unique_graphs = {}
    for graph in graph_files:
        patient_id = graph["patient_id"]
        if patient_id not in unique_graphs or "real_coords" not in graph["path"]:
            unique_graphs[patient_id] = graph

    return list(unique_graphs.values())


def extract_patient_id_from_graph_path(file_path: str) -> str:
    """Extract patient ID from enriched graph file path."""
    import re

    basename = os.path.basename(file_path)
    dirname = os.path.basename(os.path.dirname(file_path))

    # Pattern 1: patient_id in directory name (e.g., /results/0001/enriched_graph.json)
    if re.match(r"\d{4}", dirname):
        return dirname

    # Pattern 2: patient_id in filename (e.g., 0001_enriched_graph.json)
    match = re.search(r"(\d{4})_", basename)
    if match:
        return match.group(1)

    # Pattern 3: just numbers in directory name
    if dirname.isdigit():
        return dirname.zfill(4)  # Pad to 4 digits

    # Fallback: use directory name
    return dirname


def compute_scores_for_single_graph(
    graph_file: dict,
    compute_qanadli: bool,
    compute_mastora: bool,
    obstruction_attr: str,
    global_stats_dir: str | None,
    logger: logging.Logger,
) -> dict:
    """Compute scores for a single enriched graph."""

    graph_path = graph_file["path"]
    patient_id = graph_file["patient_id"]
    output_dir = graph_file["output_dir"]

    logger.info(f"Loading graph: {os.path.basename(graph_path)}")

    start_time = time.time()

    try:
        # Import load_graph function
        from pulmonarytreegraph.utils.io import load_graph

        # Load the enriched graph
        enriched_graph = load_graph(graph_path)

        # Check if graph has the required obstruction attribute
        has_obstruction_data = False
        edge_count = 0
        for _, _, data in enriched_graph.edges(data=True):
            edge_count += 1
            if obstruction_attr in data:
                has_obstruction_data = True
                break

        logger.info(
            f"Graph has {len(enriched_graph.nodes())} nodes, {edge_count} edges"
        )

        if not has_obstruction_data:
            logger.warning(
                f"Graph {patient_id} does not contain '{obstruction_attr}' data"
            )
            return {
                "patient_id": patient_id,
                "status": "skipped",
                "reason": f"missing_{obstruction_attr}",
                "processing_time": time.time() - start_time,
            }

        # Compute literature scores
        scores = compute_literature_scores(
            enriched_graph=enriched_graph,
            output_dir=output_dir,
            patient_id=patient_id,
            compute_qanadli=compute_qanadli,
            compute_mastora=compute_mastora,
            obstruction_attr=obstruction_attr,
            global_stats_dir=global_stats_dir,
        )

        processing_time = time.time() - start_time

        logger.info(f"✓ {patient_id} completed in {processing_time:.1f}s")

        # Log scores
        for score_name, score_value in scores.items():
            logger.info(f"  {score_name}: {score_value:.4f}")

        return {
            "patient_id": patient_id,
            "status": "completed",
            "scores": scores,
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"✗ {patient_id} failed: {e}")
        logger.exception("Error details:")
        return {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e),
            "processing_time": processing_time,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple patients through the pulmonary tree pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected directory structure:
  data_dir/
  ├── arteries_mask/
  │   ├── 0028.nii.gz
  │   └── XXXX.nii.gz
  ├── embolies_mask/
  │   ├── 0028.nii.gz
  │   └── XXXX.nii.gz (optional)
  ├── images/
  │   ├── 0028.nii.gz
  │   └── XXXX.nii.gz (optional)
  └── lungs_mask/
      ├── 0028/
      │   ├── lung_left.nii.gz
      │   ├── lung_right.nii.gz
      │   ├── lung_upper_lobe_left.nii.gz
      │   ├── lung_upper_lobe_right.nii.gz
      │   ├── lung_lower_lobe_left.nii.gz
      │   ├── lung_lower_lobe_right.nii.gz
      │   ├── lung_middle_lobe_right.nii.gz
      │   └── lungs.nii.gz
      └── XXXX/
          └── ...
        """,
    )

    # Input/Output
    parser.add_argument(
        "--data-dir",
        help="Base data directory containing arteries_mask/, lungs_mask/, etc.",
    )
    parser.add_argument(
        "--arteries-dir",
        help="Path to the arteries mask folder.",
    )
    parser.add_argument(
        "--embolies-dir",
        help="Path to the embolies mask folder.",
    )
    parser.add_argument(
        "--images-dir",
        help="Path to the images folder.",
    )
    parser.add_argument(
        "--lungs-dir",
        help="Path to the lungs mask folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="batch_results",
        help="Base output directory for all patients (default: batch_results)",
    )

    # Patient selection
    parser.add_argument(
        "--patient-list", help="Text file with patient IDs to process (one per line)"
    )
    parser.add_argument(
        "--exclude-patients", nargs="+", help="Patient IDs to exclude from processing"
    )
    parser.add_argument(
        "--max-patients", type=int, help="Maximum number of patients to process"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate patient files, don't run processing",
    )

    # Processing options
    parser.add_argument(
        "--radius-ratio-threshold",
        type=float,
        default=0.3,
        help="Threshold for radius ratio to remove thin branches (default: 0.3)",
    )
    parser.add_argument(
        "--diameter-threshold",
        type=float,
        default=0.7,
        help="Minimum diameter ratio for trunk vessels (default: 0.7)",
    )
    parser.add_argument(
        "--length-threshold",
        type=float,
        default=1.5,
        help="Maximum length/diameter ratio for trunk vessels (default: 1.5)",
    )
    parser.add_argument(
        "--coordinate-format",
        choices=["voxel", "real", "both"],
        default="both",
        help="Coordinate format for saved graphs (default: both)",
    )
    parser.add_argument(
        "--save-skeletons",
        action="store_true",
        help="Save binary and edge_id skeletons at each step",
    )
    parser.add_argument(
        "--save-colored-volumes",
        nargs="+",
        choices=["edge_id", "level", "lobe"],
        help="Save vessel volumes with specified coloring",
    )
    parser.add_argument(
        "--save-all-step-volumes",
        action="store_true",
        help="Save vessel volumes at each processing step",
    )
    parser.add_argument(
        "--save-cycle-skeletons",
        action="store_true",
        help="Save skeletons of detected cycles before and after cleaning",
    )

    parser.add_argument(
        "--embolism-separation-method",
        choices=["connected_components", "connected_components_with_threshold"],
        default="connected_components",
        help="Method for separating embolisms",
    )
    parser.add_argument(
        "--embolism-fusion-threshold",
        type=float,
        default=5.0,
        help="Distance threshold for fusing nearby embolism components",
    )
    parser.add_argument(
        "--save-enriched-graphs",
        action="store_true",
        help="Save all enriched graphs into a specific folder in the output directory.",
    )

    # Literature scores
    parser.add_argument(
        "--compute-qanadli",
        action="store_true",
        help="Compute Qanadli obstruction score",
    )
    parser.add_argument(
        "--compute-mastora",
        action="store_true",
        help="Compute Mastora obstruction scores (central, peripheral, global)",
    )
    parser.add_argument(
        "--compute-all-scores",
        action="store_true",
        help="Compute all available literature scores (equivalent to --compute-qanadli --compute-mastora)",
    )
    parser.add_argument(
        "--obstruction-attr",
        choices=["volumetric_obstruction", "transversal_obstruction_max"],
        default="volumetric_obstruction",
        help="Obstruction attribute to use for literature scores (default: volumetric_obstruction)",
    )

    # Processing control
    parser.add_argument(
        "--skip-steps",
        nargs="*",
        choices=["preprocess", "build", "clean", "orient", "label", "embolism"],
        default=[],
        help="Skip specific processing steps",
    )
    parser.add_argument(
        "--stop-after",
        choices=["preprocess", "build", "clean", "orient", "label", "embolism"],
        help="Stop processing after specified step",
    )
    parser.add_argument(
        "--compute-transversal-obstruction",
        action="store_true",
        help="Compute transversal obstruction metrics for each vessel segment",
    )
    parser.add_argument(
        "--cpr-padding",
        type=float,
        default=1.0,
        help="Padding for CPR in transversal obstruction (default: 1.0)",
    )

    # Execution control
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing other patients if one fails (default: True)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Heatmap generation
    parser.add_argument(
        "--generate-heatmaps",
        action="store_true",
        default=False,
        help="Generate segment heatmaps after processing (default: False)",
    )
    parser.add_argument(
        "--batch-name",
        help="Name for the batch (used in heatmap titles, defaults to output directory name)",
    )
    parser.add_argument(
        "--comparison-batches",
        nargs="+",
        help="List of other batch directories for comparison heatmaps",
    )
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Only compute literature scores from existing enriched graphs (skip pipeline)",
    )

    args = parser.parse_args()

    # Create output directory
    if not args.output_dir:
        args.output_dir = "batch_results"
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging and batch identifiers
    log_id = setup_logging(args.output_dir, args.log_level)
    args.log_id = log_id
    args.reports_dir = os.path.join(args.output_dir, "reports", log_id)
    args.global_stats_dir = os.path.join(args.output_dir, "global_stats", log_id)

    logger = logging.getLogger(__name__)

    logger.info("Starting batch processing")
    logger.info("Data directory: %s", args.data_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Batch report directory: %s", args.reports_dir)
    logger.info("Batch global stats directory: %s", args.global_stats_dir)

    # If scores-only mode is requested, run the scoring logic and exit
    if args.scores_only:
        # Run scores-only mode
        success = run_scores_only_mode(args, logger)
        return success

    try:
        # Find all patients
        all_patients = find_patients(
            data_dir=args.data_dir,
            arteries_dir=args.arteries_dir,
            embolies_dir=args.embolies_dir,
            images_dir=args.images_dir,
            lungs_dir=args.lungs_dir,
            log_id=args.log_id,
        )
        logger.info("Found %d patients", len(all_patients))
    except FileNotFoundError as e:
        logger.error("Directory structure error: %s", e)
        return False
    except (OSError, ValueError) as e:
        logger.error("Error finding patients: %s", e)
        return False
    if not all_patients:
        logger.error("No patients found!")
        return False

    # Filter patients if requested
    patients_to_process = all_patients

    # Filter by patient list
    if args.patient_list and os.path.exists(args.patient_list):
        with open(args.patient_list, encoding="utf-8") as f:
            allowed_patients = set(line.strip() for line in f if line.strip())
        patients_to_process = [
            p for p in patients_to_process if p["id"] in allowed_patients
        ]
        logger.info(
            "Filtered to %d patients from patient list", len(patients_to_process)
        )

    # Exclude patients
    if args.exclude_patients:
        exclude_set = set(args.exclude_patients)
        patients_to_process = [
            p for p in patients_to_process if p["id"] not in exclude_set
        ]
        logger.info(
            "Excluded %d patients, %d remaining",
            len(exclude_set),
            len(patients_to_process),
        )

    # Limit number of patients
    if args.max_patients:
        patients_to_process = patients_to_process[: args.max_patients]
        logger.info("Limited to first %d patients", len(patients_to_process))

    if not patients_to_process:
        logger.error("No patients found to process after filtering!")
        return False

    # Show processing plan
    logger.info("Patients to process:")
    for i, patient in enumerate(patients_to_process, 1):
        lung_status = "✓" if patient["lung_masks_dir"] else "✗"
        embolism_status = "✓" if patient["embolism_mask"] else "✗"
        image_status = "✓" if patient["image"] else "✗"
        logger.info(
            "  %3d. %s - Artery: ✓, Lungs: %s, Embolism: %s, Image: %s",
            i,
            patient["id"],
            lung_status,
            embolism_status,
            image_status,
        )

    # If validation only, run validation and exit
    if args.validate_only:
        logger.info("\nRunning validation only...")
        validation_results = []

        for patient in patients_to_process:
            validation = validate_patient_files(patient)
            validation_results.append(
                {
                    "patient_id": patient["id"],
                    "validation": validation,
                    "status": "validated"
                    if validation["valid"]
                    else "validation_failed",
                    "processing_time": 0,
                }
            )

            logger.info("\nPatient %s:", patient["id"])
            if validation["valid"]:
                logger.info("  ✓ Validation passed")
            else:
                logger.error("  ✗ Validation failed")

            for warning in validation.get("warnings", []):
                logger.warning("  Warning: %s", warning)
            for error in validation.get("errors", []):
                logger.error("  Error: %s", error)

        generate_batch_report(validation_results, args.reports_dir, logger, args.log_id)

        valid_patients = [r for r in validation_results if r["validation"]["valid"]]
        logger.info(
            "\nValidation summary: %d/%d patients valid",
            len(valid_patients),
            len(validation_results),
        )

        return len(valid_patients) == len(validation_results)

    # Create enriched graphs directory if requested
    if args.save_enriched_graphs:
        enriched_graphs_dir = os.path.join(args.output_dir, "enriched_graphs")
        os.makedirs(enriched_graphs_dir, exist_ok=True)
        logger.info("Enriched graphs will be saved to: %s", enriched_graphs_dir)

    # Prepare processing options
    processing_options = {
        "radius_ratio_threshold": args.radius_ratio_threshold,
        "diameter_threshold": args.diameter_threshold,
        "length_threshold": args.length_threshold,
        "coordinate_format": args.coordinate_format,
        "save_skeletons": args.save_skeletons,
        "save_colored_volumes": args.save_colored_volumes,
        "save_all_step_volumes": args.save_all_step_volumes,
        "save_cycle_skeletons": args.save_cycle_skeletons,
        "embolism_separation_method": args.embolism_separation_method,
        "embolism_fusion_threshold": args.embolism_fusion_threshold,
        "skip_steps": args.skip_steps,
        "stop_after": args.stop_after,
        "compute_transversal_obstruction": args.compute_transversal_obstruction,
        "transversal_obstruction_kwargs": {"cpr_padding": args.cpr_padding},
        "save_enriched_graphs": args.save_enriched_graphs,
        "enriched_graphs_dir": enriched_graphs_dir
        if args.save_enriched_graphs
        else None,
        "compute_qanadli": args.compute_qanadli,
        "compute_mastora": args.compute_mastora,
        "obstruction_attr": args.obstruction_attr,
    }

    # Handle the --compute-all-scores flag
    if args.compute_all_scores:
        processing_options["compute_qanadli"] = True
        processing_options["compute_mastora"] = True

    # Process patients
    results = []
    start_time = time.time()

    logger.info("Processing %d patients sequentially", len(patients_to_process))

    for i, patient in enumerate(patients_to_process, 1):
        logger.info(
            "\n>>> Processing patient %d/%d: %s",
            i,
            len(patients_to_process),
            patient["id"],
        )

        try:
            result = process_single_patient(
                patient, args.output_dir, processing_options, logger
            )
            results.append(result)

            # Log completion for this patient
            if result["status"] == "completed":
                logger.info("✓ Patient %s completed successfully", patient["id"])
                if "embolism_analysis" in result:
                    emb = result["embolism_analysis"]
                    logger.info(
                        "  Embolisms: %d, Volume: %.2f mm³",
                        emb.get("num_embolisms", 0),
                        emb.get("total_volume_mm3", 0),
                    )
            else:
                logger.error(
                    "✗ Patient %s failed: %s",
                    patient["id"],
                    result.get("error", "Unknown error"),
                )
            if result["status"] == "failed" and not args.continue_on_error:
                logger.error("Stopping batch processing due to failure")
                break

        except KeyboardInterrupt:
            logger.warning("Batch processing interrupted by user")
            break
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error("Unexpected error processing patient %s: %s", patient["id"], e)
            if not args.continue_on_error:
                break
            results.append(
                {
                    "patient_id": patient["id"],
                    "status": "failed",
                    "error": str(e),
                    "processing_time": 0,
                }
            )

    # Generate final report
    total_time = time.time() - start_time
    successful = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    logger.info("\n%s", "=" * 80)
    logger.info("BATCH PROCESSING COMPLETED")
    logger.info("=" * 80)
    logger.info("Total patients: %d", len(results))
    logger.info(
        "Successful: %d (%.1f%%)",
        len(successful),
        len(successful) / len(results) * 100 if results else 0,
    )
    logger.info(
        "Failed: %d (%.1f%%)",
        len(failed),
        len(failed) / len(results) * 100 if results else 0,
    )
    logger.info("Total time: %.1f seconds (%.1f hours)", total_time, total_time / 3600)
    if results:
        logger.info("Average time per patient: %.1f seconds", total_time / len(results))

    # Summary of embolism analysis results
    patients_with_embolisms = [r for r in successful if "embolism_analysis" in r]
    if patients_with_embolisms:
        total_embolisms = sum(
            r["embolism_analysis"].get("num_embolisms", 0)
            for r in patients_with_embolisms
        )
        total_volume = sum(
            r["embolism_analysis"].get("total_volume_mm3", 0)
            for r in patients_with_embolisms
        )
        logger.info(
            "Embolism analysis: %d patients, %d embolisms, %.2f mm³ total volume",
            len(patients_with_embolisms),
            total_embolisms,
            total_volume,
        )

    generate_batch_report(results, args.reports_dir, logger, args.log_id)

    # ADD THE QUICK ANALYSIS HERE - AFTER BATCH REPORT GENERATION
    # Quick segment analysis from global_labelling_stats.csv
    global_stats_path = os.path.join(
        args.global_stats_dir, "global_labelling_stats.csv"
    )
    if os.path.exists(global_stats_path) and successful:
        logger.info("\n" + "=" * 80)
        logger.info("SEGMENT ANALYSIS SUMMARY")
        logger.info("=" * 80)

        try:
            from pulmonarytreegraph.utils.segment_heatmap_generator import (
                quick_analysis,
            )

            metrics = quick_analysis(str(global_stats_path), verbose=True)

            # Log key summary metrics
            if metrics:
                success_rate = metrics.get("percentage_lobes_meeting_expected", 0)
                logger.info(
                    f"\nOverall Success Rate: {success_rate:.1f}% of lobes meet expected segment counts"
                )

                # Show per-lobe summary
                lobe_summary = []
                for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
                    mean_key = f"{lobe}_mean"
                    meets_key = f"{lobe}_meets_expected"
                    if mean_key in metrics and meets_key in metrics:
                        status = "✓" if metrics[meets_key] else "✗"
                        lobe_summary.append(f"{lobe}: {metrics[mean_key]:.1f} {status}")

                if lobe_summary:
                    logger.info(f"Per-lobe averages: {' | '.join(lobe_summary)}")

        except ImportError as e:
            logger.warning(f"Could not import segment analysis: {e}")
        except Exception as e:
            logger.error(f"Error during segment analysis: {e}")
    elif not os.path.exists(global_stats_path):
        logger.warning(
            "Global labelling stats file not found - skipping segment analysis"
        )
    elif not successful:
        logger.warning("No successful patients - skipping segment analysis")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
