"""
Orient a graph by starting from the end branch with the maximum mean radius.
"""

import argparse
import logging
import os
import sys

from pulmonarytreegraph.pipeline.core import orient_vessel_graph


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Orient graph by starting from the end branch with maximum mean radius."
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Input graph file path (e.g. results/0001/cleaned_graph.json)",
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
        "--coordinate-format",
        choices=["voxel", "real", "both"],
        default="both",
        help="Coordinate format for saved graph: voxel, real-world, or both (default)",
    )
    parser.add_argument(
        "--save-skeleton",
        action="store_true",
        help="Save both binary and edge_id colored skeleton from graph",
    )
    parser.add_argument(
        "--save-colored-volumes",
        nargs="+",
        choices=["edge_id", "level", "lobe"],
        help="Save vessel volumes with specified coloring",
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Extract patient name for default output directory
        if not args.output_dir:
            graph_basename = os.path.basename(args.graph)
            if "cleaned_graph" in graph_basename or "branch_graph" in graph_basename:
                patient_name = os.path.basename(os.path.dirname(args.graph))
            else:
                patient_name = graph_basename.split("_")[0]
            output_dir = os.path.join("results", patient_name)
        else:
            output_dir = args.output_dir

        # Minimal user-facing logging
        logger.info("Orienting graph: %s -> %s", args.graph, output_dir)

        # Only log non-default parameters
        if args.coordinate_format != "both":
            logger.info("Coordinate format: %s", args.coordinate_format)
        if args.save_skeleton:
            logger.info("Saving skeleton files")
        if args.save_colored_volumes:
            logger.info("Saving colored volumes: %s", args.save_colored_volumes)

        # Call the pipeline function - let it handle detailed logging
        orient_vessel_graph(
            graph_path=args.graph,
            output_dir=output_dir,
            original_skeleton_path=args.original_skeleton,
            arteries_mask_path=args.arteries_mask,
            coordinate_format=args.coordinate_format,
            save_skeleton=args.save_skeleton,
            save_colored_volumes=args.save_colored_volumes,
        )

        # Simple completion message
        logger.info("Graph orientation completed successfully!")
        return True

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error("Graph orientation failed: %s", str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
