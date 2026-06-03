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
        help="Input graph file path (e.g. results/0001/3_clean_graph/graph/cleaned_graph.json)",
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
        "--save-skeletons",
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
            path = os.path.normpath(args.graph)
            parts = path.split(os.sep)
            if "results" in parts:
                results_index = parts.index("results")
                if results_index + 1 < len(parts):
                    patient_name = parts[results_index + 1]
                else:
                    patient_name = os.path.basename(os.path.dirname(path))
            else:
                patient_name = os.path.basename(os.path.dirname(path))
            output_dir = os.path.join("results", patient_name)
        else:
            output_dir = args.output_dir

        logger.info("Orienting graph: %s -> %s", args.graph, output_dir)

        logger.info("Coordinate format: %s", args.coordinate_format)
        logger.info("Save skeletons: %s", args.save_skeletons)
        logger.info("Save colored volumes: %s", args.save_colored_volumes)
        logger.info("Save all step volumes: %s", args.save_all_step_volumes)
        logger.info("Save cycle skeletons: %s", args.save_cycle_skeletons)

        orient_vessel_graph(
            graph_path=args.graph,
            output_dir=output_dir,
            original_skeleton_path=args.original_skeleton,
            arteries_mask_path=args.arteries_mask,
            coordinate_format=args.coordinate_format,
            save_skeleton=args.save_skeletons,
            save_colored_volumes=args.save_colored_volumes,
        )

        logger.info("Graph orientation completed successfully!")
        return True

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error("Graph orientation failed: %s", str(e))
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
