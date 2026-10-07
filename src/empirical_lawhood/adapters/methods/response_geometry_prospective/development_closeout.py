"""development evidence joins and endpoint-specific eligibility; no second law finalizer."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.adapters.methods.contracts import ComponentUncertaintyFamilyAssessment
from .development_acquisition import ResponseGeometryDevelopmentOfflineAcquisitionReport
from .development_analysis import ResponseGeometryDevelopmentAnalysisReport
from .development_geometry_assessment import ResponseGeometryDevelopmentGeometryReport
from .development_models import REPRESENTATIONS
from .development_terminal import ResponseGeometryDevelopmentQualificationConfig, ResponseGeometryDevelopmentValidationReport


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentCloseoutConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-closeout-config'
    config_id: str
    qualification: ResponseGeometryDevelopmentQualificationConfig

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentContextResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-context-result'
    config: ObjectIdentity
    validation: ResponseGeometryDevelopmentValidationReport
    analysis: ResponseGeometryDevelopmentAnalysisReport
    scope: QualificationScopeSpec
    claim_units: ClaimUnitBinding
    family: ComponentUncertaintyFamilyAssessment
    qualification: LawQualificationResult
    geometry_report: ObjectIdentity
    acquisition: ResponseGeometryDevelopmentOfflineAcquisitionReport

    def __post_init__(self) -> None:
        projection = ObjectIdentity.from_record(self.validation.report_id, self.validation)
        if (
            self.config.object_schema != ResponseGeometryDevelopmentQualificationConfig.SCHEMA
            or self.acquisition.config != self.config
            or any(c != self.context for c in (self.analysis.context, self.acquisition.context))
            or self.analysis.validation_report != projection
            or self.qualification.dataset_or_projection != projection
            or self.qualification.candidate_family_assessment
            != ObjectIdentity.from_record(self.family.assessment_id, self.family)
            or self.claim_units.scope != ObjectIdentity.from_record(self.scope.scope_id, self.scope)
            or self.qualification.claim_unit_binding
            != ObjectIdentity.from_record(self.claim_units.binding_id, self.claim_units)
            or self.acquisition.fit_result != self.validation.fit_result
            or self.acquisition.calibration_result != self.validation.calibration_result
            or self.analysis.support_result != self.acquisition.support_result
            or self.geometry_report.object_schema != ResponseGeometryDevelopmentGeometryReport.SCHEMA
            or self.acquisition.geometry_report != self.geometry_report
        ):
            raise ValueError("development context result changes its actual scientific-owner lineage")

    @property
    def context(self) -> str:
        return self.validation.context

    @property
    def result_id(self) -> str:
        return f"response-geometry-development.assess.{self.context}"


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentContextEligibility(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-context-eligibility'
    context: str
    context_result: ObjectIdentity
    law_status: ScientificStatus
    selected_representation: str | None
    finite_screen_representations: tuple[str, ...]
    available_representation_contrasts: tuple[str, ...]
    selected_model_contact_roots: tuple[int, ...]
    acquisition_contact_available: bool
    failure_detection_charts: tuple[str, ...]
    operational_action_qualified: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in ("assembling", "prepared")
            or self.context_result.object_schema != ResponseGeometryDevelopmentContextResult.SCHEMA
            or (
                self.selected_representation is not None
                and self.selected_representation not in REPRESENTATIONS
            )
        ):
            raise ValueError("development eligibility changes its context/selected-model role")
        if self.selected_model_contact_roots != tuple(
            sorted(set(self.selected_model_contact_roots))
        ) or any(i not in range(32, 64) for i in self.selected_model_contact_roots):
            raise ValueError("development contact eligibility inflates its root denominator")
        if self.acquisition_contact_available != (
            self.law_status is ScientificStatus.SUPPORTED
            and len(self.selected_model_contact_roots) >= 8
        ):
            raise ValueError(
                "development acquisition eligibility changes the qualified-law/contact conjunction"
            )
        if self.operational_action_qualified:
            raise ValueError("development fixed-prefix analysis cannot qualify the unexecuted stopped policy")


def response_geometry_development_context_eligibility(result: ResponseGeometryDevelopmentContextResult) -> ResponseGeometryDevelopmentContextEligibility:
    selected_id = result.qualification.selected_candidate_id
    selected = (
        None if selected_id is None else selected_id.removeprefix(f"development.affine.{result.context}.")
    )
    contact = tuple(
        r.root_index
        for r in result.acquisition.rows
        if r.representation == selected and r.material_stable_contact
    )
    contrasts = tuple(
        c.comparator
        for c in result.analysis.contrasts
        if (
            sum(v is not None for v in c.typed_minus_comparator_loss.root_values) >= 16
            and len(c.informative_roots) >= 8
        )
    )
    failure = tuple(
        sorted(
            f"{s.representation}.{s.parent}"
            for s in result.analysis.support_evaluations
            if s.failure_detection_evaluable
        )
    )
    reasons = {
        "ACT_STOPPED_STATE_AND_WINDOW_NOT_QUALIFIED",
        "PROTECTED_FEASIBILITY_AND_AUTHORITY_NOT_GRANTED",
    }
    if result.qualification.scientific_status is not ScientificStatus.SUPPORTED:
        reasons.add("LOCAL_RESPONSE_LAW_NOT_QUALIFIED")
    if len(contact) < 8:
        reasons.add("SELECTED_MODEL_ACQUISITION_CONTACT_UNAVAILABLE")
    if not failure:
        reasons.add("NATURAL_FAILURE_DETECTION_CONTRAST_UNAVAILABLE")
    return ResponseGeometryDevelopmentContextEligibility(
        result.context,
        ObjectIdentity.from_record(result.result_id, result),
        result.qualification.scientific_status,
        selected,
        tuple(s.representation for s in result.validation.screens if s.finite_screen_passes),
        contrasts,
        contact,
        result.qualification.scientific_status is ScientificStatus.SUPPORTED and len(contact) >= 8,
        failure,
        False,
        tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentDevelopmentResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-development-result'
    config: ObjectIdentity
    contexts: tuple[ResponseGeometryDevelopmentContextEligibility, ...]
    paired_representation_contrasts: tuple[str, ...]
    scientific_status: ScientificStatus
    native_root_count: int = 128
    validation_independent_root_count: int = 64
    reasons: tuple[str, ...] = (
        "DEVELOPMENT_ONLY_SEPARATE_REP_ACQ_ACT_DISPOSITIONS",
        "SCIENTIFIC_STATUS_AGGREGATES_LOCAL_LAW_ONLY",
    )

    def __post_init__(self) -> None:
        if (
            self.config.object_schema != ResponseGeometryDevelopmentQualificationConfig.SCHEMA
            or tuple(c.context for c in self.contexts) != ("assembling", "prepared")
            or self.native_root_count != 128
            or self.validation_independent_root_count != 64
        ):
            raise ValueError("development closeout changes its frozen two-context denominator")
        expected = tuple(
            r
            for r in REPRESENTATIONS[1:]
            if all(r in c.available_representation_contrasts for c in self.contexts)
        )
        if (
            self.paired_representation_contrasts != expected
            or self.scientific_status is not _status(self.contexts)
        ):
            raise ValueError("development closeout changes its endpoint or existing-law status reduction")

    @property
    def result_id(self) -> str:
        return "response-geometry-development.evaluate"


def _status(contexts: tuple[ResponseGeometryDevelopmentContextEligibility, ...]) -> ScientificStatus:
    statuses = {c.law_status for c in contexts}
    if len(statuses) == 1:
        return next(iter(statuses))
    if ScientificStatus.SUPPORTED in statuses:
        return ScientificStatus.MIXED
    return (
        ScientificStatus.UNEVALUABLE
        if ScientificStatus.UNEVALUABLE in statuses
        else ScientificStatus.NOT_SUPPORTED
    )


def close_response_geometry_development_development(
    config: ResponseGeometryDevelopmentQualificationConfig, contexts: tuple[ResponseGeometryDevelopmentContextResult, ...]
) -> ResponseGeometryDevelopmentDevelopmentResult:
    config_id = ObjectIdentity.from_record(config.config_id, config)
    if tuple(c.context for c in contexts) != ("assembling", "prepared") or any(
        c.config != config_id for c in contexts
    ):
        raise ValueError("development closeout requires its two authenticated context results")
    eligible = tuple(response_geometry_development_context_eligibility(c) for c in contexts)
    contrasts = tuple(
        r
        for r in REPRESENTATIONS[1:]
        if all(r in c.available_representation_contrasts for c in eligible)
    )
    return ResponseGeometryDevelopmentDevelopmentResult(config_id, eligible, contrasts, _status(eligible))
