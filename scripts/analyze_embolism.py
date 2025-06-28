"""
Analyze pulmonary embolisms from medical imaging masks.
"""

import argparse
import logging
import sys
from pathlib import Path

from pulmonarytreegraph.emboli.embolie_analysis import (
    analyze_embolisms_with_visualization,
)
from pulmonarytreegraph.pipeline.core import (
    analyze_embolism_regions_with_graph_enrichment,
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
        help="Output directory (default: results/patient_name/embolisms/)",
    )

    # Analysis options
    parser.add_argument(
        "--separation-method",
        choices=["connected_components", "connected_components_with_threshold"],
        default="connected_components_with_threshold",
        help="Method for separating embolisms (default: connected_components)",
    )
    parser.add_argument(
        "--fusion-threshold",
        type=float,
        default=10.0,
        help="Distance threshold for fusing nearby components in voxels (default: 10.0)",
    )

    # Vessel interaction analysis
    parser.add_argument(
        "--vessel-mask", help="Path to labeled vessel mask for interaction analysis"
    )
    parser.add_argument(
        "--vessel-graph", help="Path to vessel graph file for advanced analysis"
    )
    parser.add_argument("--level-mask", help="Path to labeled anatomical level mask")

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
        output_dir = f"results/{patient_name}/embolisms"

    logger.info("Starting embolism analysis")
    logger.info("Patient: %s", patient_name)
    logger.info("Input mask: %s", args.embolism_mask)
    logger.info("Output directory: %s", output_dir)
    logger.info("Separation method: %s", args.separation_method)
    if args.separation_method == "connected_components_with_threshold":
        logger.info("Fusion threshold: %.1f voxels", args.fusion_threshold)

    try:
        if args.vessel_mask or args.vessel_graph:
            logger.info("Running pipeline with vessel graph/mask enrichment")
            results = analyze_embolism_regions_with_graph_enrichment(
                mask_path=args.embolism_mask,
                patient_name=patient_name,
                output_dir=output_dir,
                vessel_graph_path=args.vessel_graph,
                vessel_mask_path=args.vessel_mask,
                level_mask_path=args.level_mask,
                separation_method=args.separation_method,
                fusion_threshold=args.fusion_threshold,
                include_colored_masks=True,
            )
        else:
            logger.info("Running basic pipeline analysis")
            results = analyze_embolisms_with_visualization(
                mask_path=args.embolism_mask,
                patient_name=patient_name,
                output_dir=output_dir,
                separation_method=args.separation_method,
                fusion_threshold=args.fusion_threshold,
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

        logger.info("Analysis completed successfully!")
        return True

    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error("Embolism analysis failed: %s", str(e))
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
