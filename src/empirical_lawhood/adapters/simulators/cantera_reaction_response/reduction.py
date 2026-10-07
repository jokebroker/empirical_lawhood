"""Cantera entry points for the canonical target-neutral selective dependence response reducer."""

from __future__ import annotations

from empirical_lawhood.adapters.methods.selective_dependence_response.analysis import SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature, SelectiveDependenceResponseUnitContrast
from empirical_lawhood.adapters.methods.selective_dependence_response.analysis_design import SelectiveDependenceResponseTargetAnalysisFreeze, analyze_development_panel, reduce_exchange_contrasts, reduce_selective_signature
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseTargetPanel
from empirical_lawhood.kernel.evidence import OutcomeAccess


def _validate(analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze) -> None:
    if analysis_freeze.target_id != "target.cantera-selective-dependence-response":
        raise ValueError("Cantera reduction received another target analysis freeze")


def cantera_exchange_contrasts(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> dict[str, tuple[SelectiveDependenceResponseUnitContrast, ...]]:
    _validate(analysis_freeze)
    return reduce_exchange_contrasts(panel, analysis_freeze)


def cantera_selective_signature(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
    *,
    outcome_access: OutcomeAccess,
) -> SelectiveDependenceResponseSelectiveDependenceSignature:
    _validate(analysis_freeze)
    return reduce_selective_signature(
        panel,
        analysis_freeze,
        outcome_access=outcome_access,
    )


def analyze_cantera_development(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> tuple[SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature]:
    _validate(analysis_freeze)
    return analyze_development_panel(panel, analysis_freeze)


__all__ = [
    "analyze_cantera_development",
    "cantera_exchange_contrasts",
    "cantera_selective_signature",
]
