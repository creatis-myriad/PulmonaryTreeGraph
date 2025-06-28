"""
Preprocess an artery mask: create skeleton and distance map.
"""

import argparse
import logging
import os
import sys

from pulmonarytreegraph.pipeline.core import preprocess_artery_mask


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess artery mask: create skeleton and distance map."
    )
    parser.add_argument(
        "--artery",
        required=True,
        help="Path to artery mask file (e.g. 0001_arteries.nii.gz)",
    )
    parser.add_argument(
        "--output-dir", help="Output directory (defaults to results/patient_name)"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Extract patient name for default output directory
        if not args.output_dir:
            patient_name = os.path.basename(args.artery).replace("_arteries.nii.gz", "")
            output_dir = os.path.join("results", patient_name)
        else:
            output_dir = args.output_dir

        logger.info("Starting artery mask preprocessing")
        logger.info("Input artery mask: %s", args.artery)
        logger.info("Output directory: %s", output_dir)

        # Call the pipeline function
        skeleton_path, distance_map_path = preprocess_artery_mask(
            args.artery, output_dir
        )

        logger.info("Preprocessing completed successfully!")
        logger.info("Skeleton saved: %s", skeleton_path)
        logger.info("Distance map saved: %s", distance_map_path)

        return True

    except (OSError, ValueError, ImportError) as e:
        logger.error("Preprocessing failed: %s", str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
