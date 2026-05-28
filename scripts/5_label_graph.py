"""
Label a graph with anatomical information (lobes, levels, laterality).
"""

import argparse
import logging
import os
import sys

from pulmonarytreegraph.pipeline.core import label_vessel_graph


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Label graph with anatomical information."
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Input graph file path (e.g. results/0001/oriented_graph.json)",
    )
    parser.add_argument(
        "--lung-masks-dir", required=True, help="Directory containing lung mask files"
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

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging output"
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Extract patient name for default output directory
        if not args.output_dir:
            # Try to extract patient name from graph path
            graph_basename = os.path.basename(args.graph)
            if (
                "oriented_graph" in graph_basename
                or "cleaned_graph" in graph_basename
                or "branch_graph" in graph_basename
            ):
                # Path like: results/0001/oriented_graph.json
                patient_name = os.path.basename(os.path.dirname(args.graph))
            else:
                # Path like: 0001_oriented_graph.json
                patient_name = graph_basename.split("_")[0]

            output_dir = os.path.join("results", patient_name)
        else:
            output_dir = args.output_dir

        logger.info("Starting graph labeling")
        logger.info("Input graph: %s", args.graph)
        logger.info("Lung masks directory: %s", args.lung_masks_dir)
        logger.info("Output directory: %s", output_dir)
        logger.info("Coordinate format: %s", args.coordinate_format)
        logger.info("Save skeleton: %s", args.save_skeleton)

        if args.save_colored_volumes:
            logger.info("Save colored volumes: %s", args.save_colored_volumes)
        if args.vessel_expansion_method:
            logger.info("Vessel expansion method: %s", args.vessel_expansion_method)

        # Call the pipeline function
        label_vessel_graph(
            graph_path=args.graph,
            lung_masks_dir=args.lung_masks_dir,
            output_dir=output_dir,
            original_skeleton_path=args.original_skeleton,
            arteries_mask_path=args.arteries_mask,
            coordinate_format=args.coordinate_format,
            save_skeleton=args.save_skeleton,
            save_colored_volumes=args.save_colored_volumes,
        )

        logger.info("Graph labeling completed successfully!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Graph labeling failed: %s", str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
