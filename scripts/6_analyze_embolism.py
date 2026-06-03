"""
Analyze pulmonary embolisms from medical imaging masks.
"""

import argparse
import logging
import sys
from pathlib import Path

from pulmonarytreegraph.pipeline.core import (
    analyze_embolisms_complete,
    analyze_embolisms_with_visualization,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pulmonary embolisms from medical imaging masks."
    )

    # Input files
    parser.add_argument(
        "--embolism-mask",
        required=True,
        help="Path to embolism mask file (e.g. data/0001_embolisms.nii.gz)",
    )
    parser.add_argument(
        "--patient-name",
        help="Patient identifier (if not provided, extracted from filename)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: results/patient_name/)",
    )

    # Analysis options
    parser.add_argument(
        "--separation-method",
        "--embolism-separation-method",
        choices=[
            "connected_components",
            "watershed",
            "connected_components_with_threshold",
        ],
        default="connected_components",
        help="Method for separating embolisms (default: connected_components)",
    )
    parser.add_argument(
        "--fusion-threshold",
        "--embolism-fusion-threshold",
        type=float,
        default=10.0,
        help="Distance threshold for fusing nearby components in mm (default: 10.0)",
    )
    parser.add_argument(
        "--compute-transversal-obstruction",
        action="store_true",
        help="Compute transversal obstruction metrics for each vessel segment",
    )
    parser.add_argument(
        "--include-colored-masks",
        action="store_true",
        default=True,
        help="Create colored visualization masks (default: True)",
    )

    # Literature scores options
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
        "--obstruction-attr",
        choices=["volumetric_obstruction", "transversal_obstruction_max"],
        default="volumetric_obstruction",
        help="Obstruction attribute to use for literature scores (default: volumetric_obstruction)",
    )

    # Vessel interaction analysis
    parser.add_argument(
        "--vessel-mask", help="Path to labeled vessel mask for interaction analysis"
    )
    parser.add_argument(
        "--vessel-graph", help="Path to vessel graph file for advanced analysis"
    )
    parser.add_argument("--level-mask", help="Path to labeled anatomical level mask")
    parser.add_argument(
        "--disable-vessel-analysis",
        action="store_true",
        help="Disable vessel interaction analysis even if vessel data is provided",
    )
    parser.add_argument(
        "--coordinate-format",
        choices=["voxel", "real", "both"],
        default="both",
        help="Coordinate format for saved graphs (default: both)",
    )

    # Verbose output
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()

    logger = logging.getLogger(__name__)

    # Extract patient name if not provided
    if args.patient_name:
        patient_name = args.patient_name
    else:
        filename = Path(args.embolism_mask).stem
        if filename.endswith(".nii"):
            filename = filename[:-4]
        for sep in ["_embolism", "_embol", "_ep"]:
            if sep in filename:
                patient_name = filename.split(sep)[0]
                break
        else:
            patient_name = filename

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results/{patient_name}"

    logger.info("Starting embolism analysis")
    logger.info("Patient: %s", patient_name)
    logger.info("Input mask: %s", args.embolism_mask)
    logger.info("Output directory: %s", output_dir)
    logger.info("Separation method: %s", args.separation_method)
    logger.info("Fusion threshold: %.1f mm", args.fusion_threshold)

    try:
        # Determine whether to use vessel analysis
        compute_vessel_analysis = (
            args.vessel_mask or args.vessel_graph
        ) and not args.disable_vessel_analysis

        if compute_vessel_analysis:
            logger.info("Running complete pipeline with vessel analysis")

            # Prepare transversal obstruction kwargs if needed
            transversal_obstruction_kwargs = None
            if args.compute_transversal_obstruction:
                transversal_obstruction_kwargs = {}  # Add any specific parameters here

            results = analyze_embolisms_complete(
                embolism_mask_path=args.embolism_mask,
                patient_name=patient_name,
                output_dir=output_dir,
                vessel_graph_path=args.vessel_graph,
                vessel_mask_path=args.vessel_mask,
                level_mask_path=args.level_mask,
                separation_method=args.separation_method,
                fusion_threshold=args.fusion_threshold,
                include_colored_masks=args.include_colored_masks,
                compute_vessel_analysis=True,
                compute_transversal_obstruction=args.compute_transversal_obstruction,
                transversal_obstruction_kwargs=transversal_obstruction_kwargs,
                coordinate_format=args.coordinate_format,
                compute_qanadli=args.compute_qanadli,
                compute_mastora=args.compute_mastora,
                obstruction_attr=args.obstruction_attr,
            )
        else:
            logger.info("Running basic pipeline analysis")
            results = analyze_embolisms_with_visualization(
                mask_path=args.embolism_mask,
                patient_name=patient_name,
                output_dir=output_dir,
                separation_method=args.separation_method,
                fusion_threshold=args.fusion_threshold,
                include_colored_masks=args.include_colored_masks,
            )

        # Output summary
        logger.info("=" * 60)
        logger.info("EMBOLISM ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Patient: %s", results["patient_name"])
        logger.info("Number of embolisms found: %d", results["num_embolisms"])
        logger.info(
            "Total embolic volume: %.2f mm³ (%.4f cm³)",
            results["total_volume_mm3"],
            results["total_volume_cm3"],
        )

        if results["num_embolisms"] > 0:
            logger.info("Individual embolism volumes:")
            for emb in results["embolism_objects"]:
                logger.info(
                    "  Embolism %s: %.2f mm³ (%.4f cm³)",
                    emb["id"],
                    emb["volume_mm3"],
                    emb["volume_cm3"],
                )

        if "output_files" in results:
            logger.info("Generated files:")
            for file_type, file_path in results["output_files"].items():
                if file_type == "individual_masks":
                    logger.info(
                        "  Individual masks: %d files in %s", len(file_path), output_dir
                    )
                else:
                    logger.info(
                        "  %s: %s", file_type.replace("_", " ").title(), file_path
                    )

        # Additional information for complete analysis
        if compute_vessel_analysis and results.get("vessel_analysis_performed", False):
            logger.info("Vessel analysis results:")
            logger.info("  Analysis type: %s", results.get("analysis_type", "unknown"))
            if "enriched_vessel_graph" in results:
                logger.info("  Enriched vessel graph generated")
            if args.compute_transversal_obstruction:
                logger.info("  Transversal obstruction analysis completed")

        logger.info("Analysis completed successfully!")
        return True

    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error("Embolism analysis failed: %s", str(e))
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
