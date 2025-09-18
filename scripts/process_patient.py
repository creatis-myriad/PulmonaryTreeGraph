"""
Complete pipeline to process a patient from artery mask to labeled graph.
Uses new core pipeline functions.
"""

import argparse
import logging
import os
import sys
import time

from pulmonarytreegraph.pipeline.core import process_patient_complete_pipeline


def setup_logging(patient_output_path: str = None, log_level: str = "INFO"):
    """
    Setup logging for patient processing with both console and file output.

    Args:
        patient_output_path: Patient output directory for log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if output path is provided
    if patient_output_path:
        os.makedirs(patient_output_path, exist_ok=True)
        # Create global timestamp for this session
        global SESSION_TIMESTAMP
        SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

        log_file = os.path.join(
            patient_output_path, f"patient_processing_{SESSION_TIMESTAMP}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
        print(f"Patient processing log: {log_file}")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def generate_patient_report(
    patient_name: str, results: dict, patient_output_path: str, processing_time: float
):
    """
    Generate a detailed patient processing report.

    Args:
        patient_name: Patient identifier
        results: Processing results dictionary
        patient_output_path: Patient output directory
        processing_time: Total processing time in seconds
    """
    # Use the same timestamp as the log file
    global SESSION_TIMESTAMP
    report_filename = f"patient_processing_report_{SESSION_TIMESTAMP}.txt"
    report_path = os.path.join(patient_output_path, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PATIENT PROCESSING REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Patient ID: {patient_name}\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {patient_output_path}\n")
        f.write(
            f"Total Processing Time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)\n\n"
        )

        # Pipeline steps completed
        f.write("PIPELINE STEPS:\n")
        f.write("-" * 30 + "\n")

        step_names = {
            "preprocessing": "Preprocessing",
            "skeleton": "Skeleton Generation",
            "distance_map": "Distance Map Generation",
            "built_graph": "Graph Construction",
            "cleaned_graph": "Graph Cleaning",
            "oriented_graph": "Graph Orientation",
            "labeled_graph": "Graph Labeling",
            "embolism_analysis": "Embolism Analysis",
        }

        for step_key, step_name in step_names.items():
            if step_key in results and results[step_key] is not None:
                f.write(f"  ✓ {step_name}\n")

                # Add specific details based on result type
                result = results[step_key]
                if hasattr(result, "number_of_nodes"):  # NetworkX graph
                    f.write(f"    - Nodes: {result.number_of_nodes()}\n")
                    f.write(f"    - Edges: {result.number_of_edges()}\n")
                elif isinstance(result, dict):  # Dictionary result
                    if "num_embolisms" in result:
                        f.write(
                            f"    - Embolisms detected: {result.get('num_embolisms', 0)}\n"
                        )
                        f.write(
                            f"    - Total embolic volume: {result.get('total_volume_mm3', 0):.2f} mm³\n"
                        )
                elif hasattr(result, "shape"):  # Numpy array
                    f.write(f"    - Array shape: {result.shape}\n")
                else:
                    f.write(f"    - Result type: {type(result)}\n")
            else:
                f.write(f"  - {step_name} (skipped or not completed)\n")

        f.write("\n")

        # File outputs
        f.write("OUTPUT FILES:\n")
        f.write("-" * 30 + "\n")

        # List all files in the output directory
        try:
            output_files = []
            for root, _, files in os.walk(patient_output_path):
                for file in files:
                    rel_path = os.path.relpath(
                        os.path.join(root, file), patient_output_path
                    )
                    output_files.append(rel_path)

            # Group files by type
            graphs = [file for file in output_files if file.endswith(".json")]

            # Separate skeleton files from other volume files
            volumes = [file for file in output_files if file.endswith(".nii.gz")]

            # Include both log and report files in reports section
            reports = []
            for file in output_files:
                if (file.endswith(".txt") and "patient_processing_report" in file) or (
                    file.endswith(".log") and "patient_processing" in file
                ):
                    reports.append(file)

            others = [
                file for file in output_files if file not in graphs + volumes + reports
            ]

            if graphs:
                f.write("  Graph files:\n")
                for graph_file in sorted(graphs):
                    f.write(f"    - {graph_file}\n")

            if volumes:
                f.write("  Volume files:\n")
                for vol_file in sorted(volumes):
                    f.write(f"    - {vol_file}\n")

            if reports:
                f.write("  Reports and logs:\n")
                for report_file in sorted(reports):
                    f.write(f"    - {report_file}\n")

            if others:
                f.write("  Other files:\n")
                for other_file in sorted(others):
                    f.write(f"    - {other_file}\n")

        except Exception as e:
            f.write(f"  Error listing output files: {e}\n")

        f.write("\n")

        # Processing summary
        f.write("PROCESSING SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write("✓ Patient processing completed successfully\n")
        f.write(f"✓ Results saved to: {patient_output_path}\n")

        if "embolism_analysis" in results:
            emb_result = results["embolism_analysis"]
            f.write(
                f"✓ Embolism analysis completed: {emb_result['num_embolisms']} embolisms found\n"
            )

        f.write("\n")
        f.write("Report generated: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    logging.info(f"Patient processing report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process pulmonary artery data through the complete pipeline."
    )

    # Input files
    parser.add_argument(
        "--artery",
        required=True,
        help="Artery mask file path (e.g. data/0001_arteries.nii.gz)",
    )
    parser.add_argument(
        "--lung-masks-dir", required=True, help="Directory containing lung mask files"
    )
    parser.add_argument(
        "--embolism-mask",
        help="Embolism mask file path (e.g. data/0001_embolisms.nii.gz)",
    )

    # Embolism analysis options
    parser.add_argument(
        "--embolism-separation-method",
        choices=["connected_components", "connected_components_with_threshold"],
        default="connected_components",
        help="Method for separating embolisms (default: connected_components)",
    )
    parser.add_argument(
        "--embolism-fusion-threshold",
        type=float,
        default=5.0,
        help="Distance threshold for fusing nearby embolism components (default: 5.0)",
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
        default=0.5,
        help="Minimum diameter ratio for trunk vessels (default: 0.5)",
    )
    parser.add_argument(
        "--length-threshold",
        type=float,
        default=1.5,
        help="Maximum length/diameter ratio for trunk vessels (default: 1.5)",
    )

    # Output options
    parser.add_argument(
        "--output-dir", help="Output directory (defaults to results/patient_name)"
    )
    parser.add_argument(
        "--coordinate-format",
        choices=["voxel", "real", "both"],
        default="both",
        help="Coordinate format for saved graphs: voxel, real-world, or both (default)",
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

    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable patient-specific log file generation",
    )

    args = parser.parse_args()

    # Extract patient name and create output directory
    patient_name = os.path.basename(args.artery).replace("_arteries.nii.gz", "")

    if args.output_dir:
        patient_output_path = args.output_dir
    else:
        patient_output_path = os.path.join("results", patient_name)

    os.makedirs(patient_output_path, exist_ok=True)

    # Setup logging with file output unless disabled
    log_output_path = None if args.no_log_file else patient_output_path
    setup_logging(log_output_path, args.log_level)

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("PULMONARY TREE GRAPH PROCESSING")
    logger.info("=" * 80)
    logger.info("Patient: %s", patient_name)
    logger.info("Output directory: %s", patient_output_path)
    logger.info("Processing started: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Log input files
    logger.info("Input files:")
    logger.info("  Artery mask: %s", args.artery)
    logger.info("  Lung masks directory: %s", args.lung_masks_dir)
    logger.info(
        "  Embolism mask: %s",
        args.embolism_mask if args.embolism_mask else "Not provided",
    )

    # Log processing parameters
    logger.info("Processing parameters:")
    logger.info("  Radius ratio threshold: %.2f", args.radius_ratio_threshold)
    logger.info("  Diameter threshold: %.2f", args.diameter_threshold)
    logger.info("  Length threshold: %.2f", args.length_threshold)
    logger.info("  Coordinate format: %s", args.coordinate_format)
    logger.info("  Skip steps: %s", args.skip_steps if args.skip_steps else "None")
    logger.info("  Stop after: %s", args.stop_after if args.stop_after else "All steps")

    start_time = time.time()

    try:
        # Call the function with the correct parameters that match the function signature
        results = process_patient_complete_pipeline(
            artery_mask_path=args.artery,
            lung_masks_dir=args.lung_masks_dir,
            embolism_mask_path=args.embolism_mask,
            output_dir=patient_output_path,
            radius_ratio_threshold=args.radius_ratio_threshold,
            coordinate_format=args.coordinate_format,
            save_skeletons=args.save_skeletons,
            save_colored_volumes=args.save_colored_volumes,
            save_all_step_volumes=args.save_all_step_volumes,
            embolism_separation_method=args.embolism_separation_method,
            embolism_fusion_threshold=args.embolism_fusion_threshold,
            skip_steps=args.skip_steps,
            stop_after=args.stop_after,
            diameter_threshold=args.diameter_threshold,
            length_threshold=args.length_threshold,
            compute_transversal_obstruction=args.compute_transversal_obstruction,
            transversal_obstruction_kwargs={"cpr_padding": args.cpr_padding},
        )

        processing_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("Patient: %s", patient_name)
        logger.info(
            "Total processing time: %.1f seconds (%.1f minutes)",
            processing_time,
            processing_time / 60,
        )
        logger.info("Output directory: %s", patient_output_path)

        if "embolism_analysis" in results:
            emb_result = results["embolism_analysis"]
            logger.info("  ✓ %d embolisms analyzed", emb_result["num_embolisms"])
            logger.info(
                "  ✓ Total embolic volume: %.2f mm³",
                emb_result["total_volume_mm3"],
            )

        # Generate patient report
        generate_patient_report(
            patient_name, results, patient_output_path, processing_time
        )

        logger.info("Processing completed: %s", time.strftime("%Y-%m-%d %H:%M:%S"))

        return True

    except (OSError, ValueError, ImportError) as e:
        processing_time = time.time() - start_time
        logger.error(
            "Pipeline failed after %.1f seconds with error: %s", processing_time, str(e)
        )
        logger.exception("Full traceback:")

        # Generate error report
        try:
            generate_patient_report(
                patient_name, {}, patient_output_path, processing_time
            )
        except Exception as report_error:
            logger.error("Failed to generate error report: %s", report_error)

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# Add at the top of the file, after imports
SESSION_TIMESTAMP = None
