#!/usr/bin/env python3
"""
Standalone script to compute literature scores from existing enriched graphs.
"""

import argparse
import glob
import logging
import os
import sys
import time

from pulmonarytreegraph.scores.compute_scores import compute_literature_scores
from pulmonarytreegraph.utils.io import load_graph


def setup_logging(log_level: str = "INFO"):
    """Setup logging for scoring."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def find_enriched_graphs(input_path: str) -> list[dict]:
    """
    Find enriched graph files in directory structure.

    Args:
        input_path: Path to directory containing enriched graphs or single graph file

    Returns:
        List of dictionaries with graph file info
    """
    graph_files = []

    if os.path.isfile(input_path):
        # Single file
        if input_path.endswith(".json"):
            patient_id = extract_patient_id_from_path(input_path)
            graph_files.append(
                {
                    "path": input_path,
                    "patient_id": patient_id,
                    "output_dir": os.path.dirname(input_path),
                }
            )
    else:
        # Directory - search for enriched graphs
        search_patterns = [
            "*/enriched_graph.json",  # Prefer standard version
            "*/enriched_graph_real_coords.json",
            "enriched_graph.json",
            "enriched_graph_real_coords.json",
            "*_enriched_graph.json",
            "*_enriched_graph_real_coords.json",
        ]

        for pattern in search_patterns:
            search_path = os.path.join(input_path, pattern)
            files = glob.glob(search_path, recursive=True)

            for file_path in files:
                patient_id = extract_patient_id_from_path(file_path)
                output_dir = os.path.dirname(file_path)

                graph_files.append(
                    {
                        "path": file_path,
                        "patient_id": patient_id,
                        "output_dir": output_dir,
                    }
                )

    unique_graphs = {}
    for graph in graph_files:
        patient_id = graph["patient_id"]
        file_path = graph["path"]

        if patient_id not in unique_graphs:
            unique_graphs[patient_id] = graph
        else:
            # Prefer standard version over real_coords version
            existing_path = unique_graphs[patient_id]["path"]

            if "real_coords" not in file_path and "real_coords" in existing_path:
                # Current is standard, existing is real_coords -> replace
                unique_graphs[patient_id] = graph
            # Otherwise keep the existing one

    return list(unique_graphs.values())


def extract_patient_id_from_path(file_path: str) -> str:
    """Extract patient ID from file path."""
    # Try different patterns
    basename = os.path.basename(file_path)
    dirname = os.path.basename(os.path.dirname(file_path))

    # Pattern 1: patient_id in directory name (e.g., /results/0001/enriched_graph.json)
    if dirname.isdigit() or len(dirname) == 4:
        return dirname

    # Pattern 2: patient_id in filename (e.g., 0001_enriched_graph.json)
    parts = basename.split("_")
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return part

    # Fallback: use directory name
    return dirname


def compute_scores_for_graph(
    graph_file: dict,
    compute_qanadli: bool,
    compute_mastora: bool,
    obstruction_attr: str,
    logger: logging.Logger,
) -> dict:
    """Compute scores for a single enriched graph."""

    graph_path = graph_file["path"]
    patient_id = graph_file["patient_id"]
    output_dir = graph_file["output_dir"]

    logger.info(f"Processing {patient_id}: {graph_path}")

    start_time = time.time()

    try:
        # Load the enriched graph
        enriched_graph = load_graph(graph_path)

        # Check if graph has the required obstruction attribute
        has_obstruction_data = False
        for _, _, data in enriched_graph.edges(data=True):
            if obstruction_attr in data:
                has_obstruction_data = True
                break

        if not has_obstruction_data:
            logger.warning(
                f"Graph {patient_id} does not contain '{obstruction_attr}' data"
            )
            return {
                "patient_id": patient_id,
                "status": "skipped",
                "reason": f"missing_{obstruction_attr}",
                "processing_time": time.time() - start_time,
            }

        # Compute literature scores
        scores = compute_literature_scores(
            enriched_graph=enriched_graph,
            output_dir=output_dir,
            patient_id=patient_id,
            compute_qanadli=compute_qanadli,
            compute_mastora=compute_mastora,
            obstruction_attr=obstruction_attr,
        )

        processing_time = time.time() - start_time

        logger.info(f"✓ {patient_id} completed in {processing_time:.1f}s")

        # Log scores
        for score_name, score_value in scores.items():
            logger.info(f"  {score_name}: {score_value:.4f}")

        return {
            "patient_id": patient_id,
            "status": "completed",
            "scores": scores,
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"✗ {patient_id} failed: {e}")
        return {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e),
            "processing_time": processing_time,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compute literature scores from existing enriched graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single graph file
  python scores.py --input results/0001/enriched_graph.json --compute-all-scores

  # Directory of patient results
  python scores.py --input results/ --compute-qanadli --obstruction-attr transversal_obstruction_max

  # Batch directory with enriched graphs
  python scores.py --input batch_results/enriched_graphs/ --compute-all-scores
        """,
    )

    # Input
    parser.add_argument(
        "--input",
        required=True,
        help="Path to enriched graph file or directory containing enriched graphs",
    )

    # Score selection
    parser.add_argument(
        "--compute-qanadli",
        action="store_true",
        help="Compute Qanadli obstruction score",
    )
    parser.add_argument(
        "--compute-mastora",
        action="store_true",
        help="Compute Mastora obstruction scores",
    )
    parser.add_argument(
        "--compute-all-scores",
        action="store_true",
        help="Compute all available literature scores",
    )

    # Parameters
    parser.add_argument(
        "--obstruction-attr",
        choices=["volumetric_obstruction", "transversal_obstruction_max"],
        default="volumetric_obstruction",
        help="Obstruction attribute to use for scoring (default: volumetric_obstruction)",
    )

    # Options
    parser.add_argument(
        "--patient-list", help="Text file with patient IDs to process (one per line)"
    )
    parser.add_argument(
        "--exclude-patients", nargs="+", help="Patient IDs to exclude from processing"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Handle --compute-all-scores
    if args.compute_all_scores:
        args.compute_qanadli = True
        args.compute_mastora = True

    if not (args.compute_qanadli or args.compute_mastora):
        logger.error("Must specify at least one score to compute")
        return False

    logger.info("Literature Scores Computation")
    logger.info("=" * 50)
    logger.info(f"Input: {args.input}")
    logger.info(f"Obstruction attribute: {args.obstruction_attr}")
    logger.info(f"Qanadli score: {args.compute_qanadli}")
    logger.info(f"Mastora scores: {args.compute_mastora}")

    try:
        # Find enriched graph files
        graph_files = find_enriched_graphs(args.input)
        logger.info(f"Found {len(graph_files)} enriched graph files")

        if not graph_files:
            logger.error("No enriched graph files found")
            return False

        # Filter by patient list if provided
        if args.patient_list and os.path.exists(args.patient_list):
            with open(args.patient_list) as f:
                allowed_patients = set(line.strip() for line in f if line.strip())
            graph_files = [
                g for g in graph_files if g["patient_id"] in allowed_patients
            ]
            logger.info(f"Filtered to {len(graph_files)} patients from patient list")

        # Exclude patients if specified
        if args.exclude_patients:
            exclude_set = set(args.exclude_patients)
            graph_files = [g for g in graph_files if g["patient_id"] not in exclude_set]
            logger.info(f"Excluded patients, {len(graph_files)} remaining")

        if not graph_files:
            logger.error("No graphs to process after filtering")
            return False

        # Process graphs
        results = []
        start_time = time.time()

        for i, graph_file in enumerate(graph_files, 1):
            logger.info(f"\n>>> Processing {i}/{len(graph_files)}")

            result = compute_scores_for_graph(
                graph_file=graph_file,
                compute_qanadli=args.compute_qanadli,
                compute_mastora=args.compute_mastora,
                obstruction_attr=args.obstruction_attr,
                logger=logger,
            )
            results.append(result)

        # Summary
        total_time = time.time() - start_time
        successful = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] == "failed"]
        skipped = [r for r in results if r["status"] == "skipped"]

        logger.info("\n" + "=" * 50)
        logger.info("SCORING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Skipped: {len(skipped)}")
        logger.info(f"Total time: {total_time:.1f} seconds")

        if failed:
            logger.info("\nFailed patients:")
            for r in failed:
                logger.info(f"  {r['patient_id']}: {r['error']}")

        if skipped:
            logger.info("\nSkipped patients:")
            for r in skipped:
                logger.info(f"  {r['patient_id']}: {r['reason']}")

        return len(failed) == 0

    except Exception as e:
        logger.error(f"Error in scoring pipeline: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
