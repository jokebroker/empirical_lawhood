"""Board-specific linear RC-ladder numerical capability for physical scale morphism."""

from .contracts import ResistorCapacitorLadderActionSegment, ResistorCapacitorLadderModelConfig, ResistorCapacitorLadderNumericalSummary
from .matrix_exponential import ResistorCapacitorLadderTrajectory, solve_matrix_exponential
from .mna import ResistorCapacitorLadderOperator, build_operator, slowest_decay_rate

__all__ = [
    'ResistorCapacitorLadderActionSegment',
    'ResistorCapacitorLadderModelConfig',
    'ResistorCapacitorLadderNumericalSummary',
    'ResistorCapacitorLadderOperator',
    'ResistorCapacitorLadderTrajectory',
    "build_operator",
    "slowest_decay_rate",
    "solve_matrix_exponential",
]
