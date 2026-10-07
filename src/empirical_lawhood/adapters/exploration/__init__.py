"""Registered substrate-neutral automated exploration capabilities."""

from .catalog import exploration_catalog_snapshot
from .detectors import DetectorRegistry, bind_evidence_view, default_detector_registry
from .execution import (
    AnalysisCoordinateObservation,
    AutomaticSkeptic,
    ExplorationWaveInput,
    ExplorationWaveResult,
    HypothesisSynthesizer,
    RegisteredAnalysisExecutor,
    execute_wave,
)
from .portfolio import ExplorationPortfolioPlanner, default_portfolio_policy
from .registry import exploration_capability_keys, exploration_capability_registry
from .templates import AnalysisProposalFactory, default_template_library

__all__ = [
    "AnalysisProposalFactory",
    "AnalysisCoordinateObservation",
    "AutomaticSkeptic",
    "DetectorRegistry",
    "ExplorationPortfolioPlanner",
    "ExplorationWaveInput",
    "ExplorationWaveResult",
    "HypothesisSynthesizer",
    "RegisteredAnalysisExecutor",
    "bind_evidence_view",
    "default_detector_registry",
    "default_portfolio_policy",
    "default_template_library",
    "exploration_catalog_snapshot",
    "execute_wave",
    'exploration_capability_keys',
    'exploration_capability_registry',
]
