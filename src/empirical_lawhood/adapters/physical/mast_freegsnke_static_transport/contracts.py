"""Canonical records for the bounded public MAST static-transport experiment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


class StaticTransportDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class StaticTransportGateReceipt(CanonicalRecord):
    """Compact operational/scientific gate receipt with no dense values."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-freegsnke-static-transport/static-transport-gate-receipt'

    receipt_id: str
    gate_id: str
    disposition: StaticTransportDisposition
    reason_codes: tuple[str, ...]
    facts: dict[str, object]
    artifacts: tuple[ArtifactIdentity, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.gate_id, field_name="gate_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_ids(
            self.artifacts,
            attribute="artifact_id",
            field_name="artifacts",
        )


@dataclass(frozen=True, slots=True)
class StaticTransportFacePrediction(CanonicalRecord):
    """One predeclared face and its development-only prediction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-freegsnke-static-transport/static-transport-face-prediction'

    face_id: str
    predictive_level: str
    applicable: bool
    metric_unit: str
    development_maximum: str | None
    development_view_disagreement_maximum: str | None
    categorical_prediction: str | None
    source_uncertainty_qualified: bool
    decisive_falsifier: str
    limitation_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.face_id, field_name="face_id")
        validate_nonempty(self.predictive_level, field_name="predictive_level")
        validate_nonempty(self.metric_unit, field_name="metric_unit")
        validate_nonempty(self.decisive_falsifier, field_name="decisive_falsifier")
        require_sorted_unique_strings(
            self.limitation_codes,
            field_name="limitation_codes",
        )
        if self.predictive_level == "METRIC" and self.categorical_prediction is not None:
            raise ValueError("metric prediction cannot carry a category")
        if self.predictive_level == "CATEGORICAL" and self.development_maximum is not None:
            raise ValueError("categorical prediction cannot carry a metric maximum")


@dataclass(frozen=True, slots=True)
class StaticTransportPredictionPackage(CanonicalRecord):
    """Outcome-blind freeze before protected preparation or receiver contact."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-freegsnke-static-transport/static-transport-prediction-package'

    prediction_id: str
    experiment_id: str
    selected_route: str
    source_shot_id: str
    target_shot_id: str
    source_member_sha256: str
    target_member_sha256: str
    machine_archive_sha256: str
    slice_times_s: tuple[str, ...]
    coordinate_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    predictions: tuple[StaticTransportFacePrediction, ...]
    target_preparation_contact_count: int
    target_receiver_contact_count: int
    target_execution_count: int
    evaluator_implementation_sha256: str
    claim_ceiling: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.prediction_id, field_name="prediction_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        if self.selected_route != "ROUTE_U_26M5_JMPP":
            raise ValueError("prediction package binds another source route")
        if (self.source_shot_id, self.target_shot_id) != ("45292", "45425"):
            raise ValueError("prediction package differs from the frozen split")
        for name, value in (
            ("source_member_sha256", self.source_member_sha256),
            ("target_member_sha256", self.target_member_sha256),
            ("machine_archive_sha256", self.machine_archive_sha256),
            ("evaluator_implementation_sha256", self.evaluator_implementation_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_strings(self.coordinate_ids, field_name="coordinate_ids")
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
        )
        require_sorted_unique_ids(
            self.predictions,
            attribute="face_id",
            field_name="predictions",
        )
        if any(
            value != 0
            for value in (
                self.target_preparation_contact_count,
                self.target_receiver_contact_count,
                self.target_execution_count,
            )
        ):
            raise ValueError("prediction must freeze before every target contact")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("prediction package must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class StaticTransportFaceResult(CanonicalRecord):
    """Facewise, noncompensating protected-target adjudication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-freegsnke-static-transport/static-transport-face-result'

    face_id: str
    disposition: StaticTransportDisposition
    predictive_level: str
    complete_unit_numerator: int
    complete_unit_denominator: int
    target_value: str | None
    issued_limit: str | None
    point_estimate_inside_development_envelope: bool | None
    falsifier_triggered: bool | None
    reason_codes: tuple[str, ...]
    evidence_ceiling: str

    def __post_init__(self) -> None:
        validate_stable_id(self.face_id, field_name="face_id")
        if self.complete_unit_numerator < 0 or self.complete_unit_denominator < 0:
            raise ValueError("complete-unit counts must be nonnegative")
        if self.complete_unit_numerator > self.complete_unit_denominator:
            raise ValueError("complete numerator exceeds denominator")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class StaticTransportCloseout(CanonicalRecord):
    """Immutable set-valued terminal; deliberately not a ResponseLaw."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-freegsnke-static-transport/static-transport-closeout'

    closeout_id: str
    experiment_id: str
    prediction_package_fingerprint: str
    route_receipt: ArtifactIdentity
    development_artifacts: tuple[ArtifactIdentity, ...]
    target_artifacts: tuple[ArtifactIdentity, ...]
    face_results: tuple[StaticTransportFaceResult, ...]
    coordinate_disposition: StaticTransportDisposition
    coordinate_reason_codes: tuple[str, ...]
    achieved_dependence_class: str
    shared_lineage: bool
    operational_status: str
    claim_ceiling: str
    response_law_constructed: bool
    admission_or_controller_evaluation_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        validate_sha256(
            self.prediction_package_fingerprint,
            field_name="prediction_package_fingerprint",
        )
        require_sorted_unique_ids(
            self.development_artifacts,
            attribute="artifact_id",
            field_name="development_artifacts",
        )
        require_sorted_unique_ids(
            self.target_artifacts,
            attribute="artifact_id",
            field_name="target_artifacts",
        )
        require_sorted_unique_ids(
            self.face_results,
            attribute="face_id",
            field_name="face_results",
        )
        require_sorted_unique_strings(
            self.coordinate_reason_codes,
            field_name="coordinate_reason_codes",
        )
        if self.response_law_constructed or self.admission_or_controller_evaluation_constructed:
            raise ValueError("static transport closeout cannot promote into measurement through controller use")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("terminal closeout must retain revealed visibility")


__all__ = [
    'StaticTransportCloseout',
    'StaticTransportDisposition',
    'StaticTransportFacePrediction',
    'StaticTransportFaceResult',
    'StaticTransportGateReceipt',
    'StaticTransportPredictionPackage',
]
