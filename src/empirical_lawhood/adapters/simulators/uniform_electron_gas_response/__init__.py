"""uniform electron gas transverse receiver screen adapter."""

from .analysis import analyze_branch, evaluate_closure_only
from .contracts import load_config
from .reference_world import build_truth_case, run_truth_known_conformance

__all__ = [
    "analyze_branch",
    "build_truth_case",
    "evaluate_closure_only",
    "load_config",
    "run_truth_known_conformance",
]
