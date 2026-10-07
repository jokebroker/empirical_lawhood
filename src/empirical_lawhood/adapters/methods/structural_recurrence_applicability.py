"Non-mutating structural recurrence applicability overlay for independent substrate grounding.\n\nThe overlay constrains target representation, power and interpretation around\nthe frozen margin structural recurrence forecast predictor.  It imports no generated-target coefficient,\nthreshold, map, likelihood or native coordinate from generated-target continuations.\n"

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import ClassVar

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceApplicabilityCloseoutBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-applicability-closeout-binding'

    binding_id: str
    tranche_id: str
    original_closeout_commit: str
    original_closeout_tree: str
    terminal: str
    current_scientific_record_sha256: str
    source_export: IndependentSourceExport

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.tranche_id, field_name="tranche_id")
        for name in ("original_closeout_commit", "original_closeout_tree"):
            if re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase Git SHA-1")
        validate_nonempty(self.terminal, field_name="terminal")
        validate_sha256(
            self.current_scientific_record_sha256,
            field_name="current_scientific_record_sha256",
        )
        if type(self.source_export) is not IndependentSourceExport:
            raise ValueError("applicability closeout requires its authenticated external source export mapping")
        if (
            self.original_closeout_commit != self.source_export.original_source_commit
            or self.current_scientific_record_sha256 != self.source_export.target_artifact.sha256
            or self.tranche_id != self.source_export.target_source.object_id
        ):
            raise ValueError("applicability closeout substitutes original or current source custody")


_TERMINAL_BY_TRANCHE = {
    'structural-recurrence-power-region-identification': "POWER_REGION_IDENTIFIED",
    'structural-recurrence-scale-confound-detection': "SCALE_CONFOUND_DETECTED",
    'structural-recurrence-predictive-scale-covariance-study': "PREDICTIVE_SCALE_COVARIANCE_NOT_IDENTIFIED",
}


@dataclass(frozen=True, slots=True)
class IndependentSubstrateApplicabilityOverlay(CanonicalRecord):
    'Typed power, scale-confound and covariance constraints around the frozen structural predictor.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-applicability-overlay'

    overlay_id: str
    frozen_margin_forecast_method_port: ObjectIdentity
    closeouts: tuple[StructuralRecurrenceApplicabilityCloseoutBinding, ...]
    predictive_scale_covariance_handoff_sha256: str
    joint_resolution_operands: tuple[str, ...]
    distinct_scale_operands: tuple[str, ...]
    target_binding_requirements: tuple[str, ...]
    scale_covariance_disposition: str
    generated_target_parameters_imported: bool
    margin_forecast_operators_changed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.overlay_id, field_name="overlay_id")
        require_sorted_unique_ids(
            self.closeouts,
            attribute="binding_id",
            field_name="closeouts",
        )
        if {row.tranche_id: row.terminal for row in self.closeouts} != _TERMINAL_BY_TRANCHE:
            raise ValueError('applicability overlay requires the power, scale-confound and covariance dispositions')
        validate_sha256(
            self.predictive_scale_covariance_handoff_sha256,
            field_name='predictive_scale_covariance_handoff_sha256',
        )
        for name in (
            "joint_resolution_operands",
            "distinct_scale_operands",
            "target_binding_requirements",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if self.joint_resolution_operands != tuple(
            sorted(
                (
                    "complete-unit-panel-size",
                    "locator-or-observation-grid",
                    "native-receiver-coordinate-width",
                    "simultaneous-inference-family",
                )
            )
        ):
            raise ValueError('Power-region joint-resolution contract is incomplete')
        if self.distinct_scale_operands != tuple(
            sorted(
                (
                    "dependence-construction",
                    "numerical-or-finite-system-view",
                    "observation-grid",
                    "observation-operator",
                    "receiver-gauge",
                )
            )
        ):
            raise ValueError('Scale-confound typed scale operands are incomplete')
        if self.scale_covariance_disposition != 'EXCLUDED_WITHOUT_IDENTIFIED_PREDICTIVE_SCALE_COVARIANCE':
            raise ValueError("Predictive scale-covariance negative must exclude an independent substrate grounding scale-covariance claim")
        if self.generated_target_parameters_imported:
            raise ValueError("generated-target parameters cannot enter independent substrate grounding")
        if self.margin_forecast_operators_changed:
            raise ValueError('applicability overlay cannot mutate margin structural recurrence forecast')


def applicability_overlay(
    *,
    frozen_margin_forecast_method_port: ObjectIdentity,
    closeouts: tuple[StructuralRecurrenceApplicabilityCloseoutBinding, ...],
    predictive_scale_covariance_handoff_sha256: str,
) -> IndependentSubstrateApplicabilityOverlay:
    return IndependentSubstrateApplicabilityOverlay(
        overlay_id='independent-substrate.structural-recurrence-applicability-overlay',
        frozen_margin_forecast_method_port=frozen_margin_forecast_method_port,
        closeouts=closeouts,
        predictive_scale_covariance_handoff_sha256=predictive_scale_covariance_handoff_sha256,
        joint_resolution_operands=tuple(
            sorted(
                (
                    "complete-unit-panel-size",
                    "locator-or-observation-grid",
                    "native-receiver-coordinate-width",
                    "simultaneous-inference-family",
                )
            )
        ),
        distinct_scale_operands=tuple(
            sorted(
                (
                    "dependence-construction",
                    "numerical-or-finite-system-view",
                    "observation-grid",
                    "observation-operator",
                    "receiver-gauge",
                )
            )
        ),
        target_binding_requirements=tuple(
            sorted(
                (
                    "dependence-construction",
                    "exact-receiver-map-and-measurement-backaction",
                    "native-coordinate-units-and-receiver-tolerance",
                    "panel-grid-simultaneous-inference-binding",
                    "scale-flow-status-untested-unsupported-or-outside-question",
                    "solver-backend-finite-view-and-observation-operator",
                )
            )
        ),
        scale_covariance_disposition='EXCLUDED_WITHOUT_IDENTIFIED_PREDICTIVE_SCALE_COVARIANCE',
        generated_target_parameters_imported=False,
        margin_forecast_operators_changed=False,
    )


__all__ = [
    'IndependentSubstrateApplicabilityOverlay',
    'StructuralRecurrenceApplicabilityCloseoutBinding',
    "applicability_overlay",
]
