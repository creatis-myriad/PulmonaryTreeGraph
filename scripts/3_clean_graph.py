"""
Clean a graph by removing suspicious cycle edges and short/thin end branches.
"""

import argparse
import logging
import os
import sys

from pulmonarytreegraph.pipeline.core import clean_vessel_graph


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean graph by removing suspicious edges and short/thin branches."
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Input graph file path (e.g. results/0001/branch_graph.json)",
    )
    parser.add_argument(
        "--original-skeleton",
        help="Original skeleton file (required for skeleton saving or coordinate conversion)",
    )
    parser.add_argument(
        "--arteries-mask", help="Arteries mask file (required for vessel volumes)"
    )
    parser.add_argument(
        "--distance-map", help="Distance map file (optional, for better performance)"
    )
    parser.add_argument(
        "--output-dir", help="Output directory (defaults to results/patient_name)"
    )
    parser.add_argument(
        "--radius-ratio-threshold",
        type=float,
        default=0.3,
        help="Threshold for radius ratio to remove thin branches (default: 0.3)",
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
            # Try to extract patient name from graph path
            graph_basename = os.path.basename(args.graph)
            if "branch_graph" in graph_basename:
                # Path like: results/0001/branch_graph.json
                patient_name = os.path.basename(os.path.dirname(args.graph))
            else:
                # Path like: 0001_branch_graph.json
                patient_name = graph_basename.split("_")[0]

            output_dir = os.path.join("results", patient_name)
        else:
            output_dir = args.output_dir

        # Log only essential user-facing information
        logger.info("Starting graph cleaning")
        logger.info("Input graph: %s", args.graph)
        logger.info("Output directory: %s", output_dir)
        logger.info("Radius ratio threshold: %.3f", args.radius_ratio_threshold)

        # Optional parameters only if specified
        if args.coordinate_format != "both":
            logger.info("Coordinate format: %s", args.coordinate_format)
        if args.save_skeletons:
            logger.info("Save skeletons: %s", args.save_skeletons)
        if args.save_colored_volumes:
            logger.info("Save colored volumes: %s", args.save_colored_volumes)
        if args.save_all_step_volumes:
            logger.info(
                "Save colored volumes for all steps: %s", args.save_all_step_volumes
            )
        if args.save_cycle_skeletons:
            logger.info("Save cycle skeletons: %s", args.save_cycle_skeletons)

        # Call the pipeline function - let it handle detailed logging
        clean_vessel_graph(
            graph_path=args.graph,
            output_dir=output_dir,
            original_skeleton_path=args.original_skeleton,
            arteries_mask_path=args.arteries_mask,
            radius_ratio_threshold=args.radius_ratio_threshold,
            coordinate_format=args.coordinate_format,
            save_skeleton=args.save_skeletons,
            save_colored_volumes=args.save_colored_volumes,
            save_cycle_skeletons=args.save_cycle_skeletons,
        )

        # Only log completion - remove all the detailed file listings
        logger.info("Graph cleaning completed successfully!")

        return True

    except (OSError, ValueError, argparse.ArgumentError) as e:
        logger.error("Graph cleaning failed: %s", str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
