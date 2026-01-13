"""
Literature-based scoring functions for pulmonary embolism assessment.

This package provides implementations of established scoring systems
for quantifying pulmonary embolism severity based on vessel obstruction.
"""

from .compute_scores import compute_literature_scores
from .mastora import compute_mastora_scores
from .qanadli import compute_qanadli_score

__all__ = [
    "compute_qanadli_score",
    "compute_mastora_scores",
    "compute_literature_scores",
]
