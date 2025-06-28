"""
Complete pipeline to process a patient from artery mask to labeled graph.
Uses new core pipeline functions.
"""

import argparse
import logging
import os
import sys

from pulmonarytreegraph.pipeline.core import process_patient_complete_pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


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

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Extract patient name and create output directory
    patient_name = os.path.basename(args.artery).replace("_arteries.nii.gz", "")

    if args.output_dir:
        patient_output_path = args.output_dir
    else:
        patient_output_path = os.path.join("results", patient_name)

    os.makedirs(patient_output_path, exist_ok=True)

    logger.info("Processing patient: %s", patient_name)
    logger.info("Output directory: %s", patient_output_path)

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
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Patient: %s", patient_name)
        logger.info("Output directory: %s", patient_output_path)

        if "embolism_analysis" in results:
            emb_result = results["embolism_analysis"]
            logger.info("  ✓ %d embolisms analyzed", emb_result["num_embolisms"])
            logger.info(
                "  ✓ Total embolic volume: %.2f mm³",
                emb_result["total_volume_mm3"],
            )

        return True

    except (OSError, ValueError, ImportError) as e:
        logger.error("Pipeline failed with error: %s", str(e))
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
