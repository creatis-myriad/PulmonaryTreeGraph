"""
Build a graph from skeleton and distance map.
"""

import argparse
import logging
import os
import sys

from pulmonarytreegraph.pipeline.core import build_graph_from_skeleton


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build graph from skeleton and distance map."
    )
    parser.add_argument(
        "--skeleton",
        required=True,
        help="Skeleton file path (e.g. results/0001/skeleton.nii.gz)",
    )
    parser.add_argument(
        "--distance-map",
        required=True,
        help="Distance map file path (e.g. results/0001/distance_map.nii.gz)",
    )
    parser.add_argument(
        "--output-dir", help="Output directory (defaults to results/patient_name)"
    )
    parser.add_argument(
        "--coordinate-format",
        choices=["voxel", "real", "both"],
        default="both",
        help="Coordinate format for saved graph: voxel, real-world, or both (default)",
    )
    parser.add_argument(
        "--save-skeletons",
        action="store_true",
        help="Save both binary and edge_id colored skeleton from graph",
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

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Extract patient name for default output directory
        if not args.output_dir:
            # Try to extract patient name from skeleton path
            skeleton_basename = os.path.basename(args.skeleton)
            if skeleton_basename == "skeleton.nii.gz":
                # Path like: results/0001/skeleton.nii.gz
                patient_name = os.path.basename(os.path.dirname(args.skeleton))
            else:
                # Path like: 0001_skeleton.nii.gz
                patient_name = skeleton_basename.replace(
                    "_skeleton.nii.gz", ""
                ).replace("skeleton.nii.gz", "")

            output_dir = os.path.join("results", patient_name)
        else:
            output_dir = args.output_dir

        logger.info("Starting graph building from skeleton and distance map")
        logger.info("Skeleton: %s", args.skeleton)
        logger.info("Distance map: %s", args.distance_map)
        logger.info("Output directory: %s", output_dir)
        logger.info("Coordinate format: %s", args.coordinate_format)
        logger.info("Save skeletons: %s", args.save_skeletons)
        logger.info("Save colored volumes: %s", args.save_colored_volumes)
        logger.info("Save all step volumes: %s", args.save_all_step_volumes)
        logger.info("Save cycle skeletons: %s", args.save_cycle_skeletons)

        # Call the pipeline function
        result = build_graph_from_skeleton(
            skeleton_path=args.skeleton,
            distance_map_path=args.distance_map,
            output_dir=output_dir,
            coordinate_format=args.coordinate_format,
            save_skeleton=args.save_skeletons,
            save_colored_volumes=args.save_colored_volumes,
            save_all_step_volumes=args.save_all_step_volumes,
            save_cycle_skeletons=args.save_cycle_skeletons,
        )

        logger.info("Graph building completed successfully!")

        # Log output files
        if "graph_paths" in result:
            logger.info("Graph files saved:")
            for coord_type, path in result["graph_paths"].items():
                logger.info("  %s coordinates: %s", coord_type, path)

        if "skeleton_paths" in result and result["skeleton_paths"]:
            logger.info("Skeleton files saved:")
            for skeleton_type, path in result["skeleton_paths"].items():
                logger.info("  %s skeleton: %s", skeleton_type, path)

        # Log validation results
        if "validation" in result:
            validation = result["validation"]
            logger.info("Graph validation:")
            logger.info("  Perfect match: %s", validation.get("perfect_match", "N/A"))
            logger.info(
                "  Dice coefficient: %.4f", validation.get("dice_coefficient", 0)
            )

            if args.detailed_validation:
                logger.info(
                    "  Original voxels: %d", validation.get("original_voxels", 0)
                )
                logger.info(
                    "  Recreated voxels: %d", validation.get("recreated_voxels", 0)
                )
                logger.info(
                    "  Coverage: %.2f%%", validation.get("coverage_original", 0) * 100
                )
                logger.info("  Missing voxels: %d", validation.get("missing_voxels", 0))
                logger.info("  Extra voxels: %d", validation.get("extra_voxels", 0))

        return True

    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error("Graph building failed: %s", str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
