"""Cross-program effective-law metatheory comparison."""

from .analysis import PosthocDocumentSet, execute_analysis, execute_integrated_skeptic, execute_skeptic, synthesize
from .authoring import build_authorized_package, build_freeze
from .execution import EffectiveLawPosthocExecutionPorts, execute_plan, verify_complete_run
from .sources import qualify_parent_sources

__all__ = [
    "PosthocDocumentSet",
    "EffectiveLawPosthocExecutionPorts",
    "build_authorized_package",
    "build_freeze",
    "execute_analysis",
    "execute_plan",
    "execute_integrated_skeptic",
    "execute_skeptic",
    "qualify_parent_sources",
    "synthesize",
    "verify_complete_run",
]
