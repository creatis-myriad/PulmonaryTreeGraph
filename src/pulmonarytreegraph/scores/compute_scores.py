"""
Main entry point for computing literature-based obstruction scores.
"""

import logging
from pathlib import Path

import networkx as nx

from .mastora import compute_mastora_scores
from .qanadli import compute_qanadli_score

logger = logging.getLogger(__name__)


def compute_literature_scores(
    enriched_graph: nx.DiGraph,
    output_dir: str,
    patient_id: str | None = None,
    compute_qanadli: bool = False,
    compute_mastora: bool = False,
    obstruction_attr: str = "volumetric_obstruction",
) -> dict[str, float]:
    """
    Compute literature-based obstruction scores for an enriched vessel graph.

    Args:
        enriched_graph: NetworkX graph with embolism information
        output_dir: Output directory for statistics files
        patient_id: Patient identifier
        compute_qanadli: Whether to compute Qanadli score
        compute_mastora: Whether to compute Mastora scores (all variants)

    Returns:
        Dictionary with computed scores
    """

    scores = {}

    # Early return if nothing to compute
    if not (compute_qanadli or compute_mastora):
        logger.info("No literature scores requested")
        return scores

    # Setup paths for statistics
    if patient_id:
        results_base_dir = Path(output_dir).parent

        # Individual patient paths
        patient_qanadli_path = (
            Path(output_dir) / "qanadli_scores.csv" if compute_qanadli else None
        )
        patient_mastora_path = (
            Path(output_dir) / "mastora_scores.csv" if compute_mastora else None
        )

        # Global paths
        global_qanadli_path = (
            results_base_dir / "global_qanadli_scores.csv" if compute_qanadli else None
        )
        global_mastora_path = (
            results_base_dir / "global_mastora_scores.csv" if compute_mastora else None
        )
    else:
        patient_qanadli_path = None
        patient_mastora_path = None
        global_qanadli_path = None
        global_mastora_path = None

    logger.info("Computing literature-based obstruction scores...")
    logger.info(f"Using obstruction attribute: {obstruction_attr}")

    # Qanadli Score
    if compute_qanadli:
        try:
            qanadli_literature_score, qanadli_patient_score = compute_qanadli_score(
                enriched_graph,
                patient_id=patient_id,
                patient_stats_path=patient_qanadli_path,
                global_stats_path=global_qanadli_path,
                obstruction_attr=obstruction_attr,
            )
            scores["qanadli_literature"] = qanadli_literature_score
            scores["qanadli_patient"] = qanadli_patient_score
            logger.info(f"Qanadli literature score: {qanadli_literature_score:.4f}")
            logger.info(f"Qanadli patient score: {qanadli_patient_score:.4f}")

        except Exception as e:
            logger.error(f"Error computing Qanadli score: {e}")
            scores["qanadli"] = 0.0

    # Mastora Scores (all variants)
    if compute_mastora:
        try:
            mastora_literature_scores, mastora_patient_scores = compute_mastora_scores(
                enriched_graph,
                patient_id=patient_id,
                patient_stats_path=patient_mastora_path,
                global_stats_path=global_mastora_path,
                obstruction_attr=obstruction_attr,
            )

            # Add all Mastora variants to scores
            scores["mastora_central_literature"] = mastora_literature_scores["central"]
            scores["mastora_peripheral_literature"] = mastora_literature_scores[
                "peripheral"
            ]
            scores["mastora_global_literature"] = mastora_literature_scores["global"]
            scores["mastora_central_patient"] = mastora_patient_scores["central"]
            scores["mastora_peripheral_patient"] = mastora_patient_scores["peripheral"]
            scores["mastora_global_patient"] = mastora_patient_scores["global"]

            logger.info(
                f"Mastora central score: {mastora_literature_scores['central']:.4f}"
            )
            logger.info(
                f"Mastora peripheral score: {mastora_literature_scores['peripheral']:.4f}"
            )
            logger.info(
                f"Mastora global score: {mastora_literature_scores['global']:.4f}"
            )
            logger.info(
                f"Mastora central patient score: {mastora_patient_scores['central']:.4f}"
            )
            logger.info(
                f"Mastora peripheral patient score: {mastora_patient_scores['peripheral']:.4f}"
            )
            logger.info(
                f"Mastora global patient score: {mastora_patient_scores['global']:.4f}"
            )

        except Exception as e:
            logger.error(f"Error computing Mastora scores: {e}")
            scores["mastora_central_literature"] = 0.0
            scores["mastora_peripheral_literature"] = 0.0
            scores["mastora_global_literature"] = 0.0

    return scores
