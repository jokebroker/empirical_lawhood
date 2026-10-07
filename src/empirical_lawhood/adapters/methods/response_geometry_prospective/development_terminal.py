"""development validation publication and composition of the existing law owners."""

from dataclasses import dataclass
from hashlib import sha256
import io
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import CandidateEvaluatorImplementation, ComponentUncertaintyFamilyAssessment
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadDecoderRegistry,
    LawAssessmentAssembler,
    QualificationProfileEvaluatorRegistry,
    ResponseLawQualificationService,
)
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationRequest, LawEvaluationResult
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_writer, response_hdf5_group, response_hdf5_text
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane

from .development_assessment import ResponseGeometryDevelopmentModelScreen, ResponseGeometryDevelopmentValidationOperands, response_geometry_development_qualification_metrics, validate_response_geometry_development_context
from .development_family import response_geometry_development_candidate_evidence, response_geometry_development_family
from .development_law import ResponseGeometryDevelopmentAffineLawDecoder, DEVELOPMENT_LAW_KEY, DEVELOPMENT_LAW_MAXIMUM_BYTES, build_response_geometry_development_affine_payload
from .development_models import ResponseGeometryDevelopmentMeasuredView, REPRESENTATIONS
from .development_projection import ResponseGeometryDevelopmentProjectionConfig
from .development_qualification import ResponseGeometryDevelopmentAffineQualificationProfile
from .development_records import ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentMethodConfig, _inputs, read_response_geometry_development_fit, report_identities


DEVELOPMENT_VALIDATION_SCHEMA = 'empirical-lawhood/methods/response-geometry-prospective/development-validation-arrays-hdf5'

DEVELOPMENT_VALIDATION_METADATA = {
    "empirical_lawhood_payload_schema": DEVELOPMENT_VALIDATION_SCHEMA,
    "empirical_lawhood_units": '{"prediction,error,halfwidth":"hilbert-schmidt-native"}',
    "empirical_lawhood_frames": '{"receiver":"frozen-preparent-mode"}',
    "empirical_lawhood_clocks": '{"endpoint":"320-reference-ticks-after-observed-invocation"}',
    "empirical_lawhood_keys": '["representation","root32-through63","parent","primary,half-step","NEG,HOLD,POS"]',
}
DEVELOPMENT_VALIDATION_MAXIMUM_BYTES = 4 * 1024**2


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentQualificationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-qualification-config'
    config_id: str
    system: SystemSpec
    source: ResponseGeometryDevelopmentNativeConfig
    projection: ResponseGeometryDevelopmentProjectionConfig
    method: ResponseGeometryDevelopmentMethodConfig

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.projection.source_config
            != ObjectIdentity.from_record(self.source.config_id, self.source)
            or self.method.projection_config
            != ObjectIdentity.from_record(self.projection.config_id, self.projection)
            or self.method.design_packet_sha256 != self.source.design_packet_sha256
        ):
            raise ValueError("development qualification changes source/projection/method scientific lineage")


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentValidationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-validation-report'
    context: str
    config: ObjectIdentity
    fit_result: ObjectIdentity
    calibration_result: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    screens: tuple[ResponseGeometryDevelopmentModelScreen, ...]
    qualification_metrics: tuple[tuple[str, tuple[NamedDecimal, ...]], ...]
    data_sha256: str

    def __post_init__(self) -> None:
        _inputs(self.context, self.config, self.input_reports, range(32, 64))
        if (
            tuple(s.representation for s in self.screens) != REPRESENTATIONS
            or any(s.context != self.context for s in self.screens)
            or tuple(r for r, _ in self.qualification_metrics) != REPRESENTATIONS
            or self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
            or self.calibration_result.object_schema != ResponseGeometryDevelopmentCalibrationResult.SCHEMA
        ):
            raise ValueError("development validation report changes its complete context/model roster")
        validate_sha256(self.data_sha256, field_name="data_sha256")

    @property
    def report_id(self) -> str:
        return f"response-geometry-development.validation.{self.context}"


def _validation_bytes(
    config: ResponseGeometryDevelopmentMethodConfig, operands: tuple[ResponseGeometryDevelopmentValidationOperands, ...]
) -> bytes:
    output = io.BytesIO()
    with response_hdf5_writer(output) as artifact:
        for name, value in {
            "schema": DEVELOPMENT_VALIDATION_SCHEMA,
            "method_config_sha256": config.fingerprint(),
            **DEVELOPMENT_VALIDATION_METADATA,
        }.items():
            response_hdf5_text(artifact, name, value)
        for operand in operands:
            group = response_hdf5_group(artifact, operand.screen.representation)
            for name, array in (
                ("prediction", operand.predictions),
                ("error", operand.errors),
                ("halfwidth", operand.halfwidths),
            ):
                group.create_dataset(name, data=array, track_times=False)
    payload = output.getvalue()
    if len(payload) > DEVELOPMENT_VALIDATION_MAXIMUM_BYTES:
        raise ValueError("development validation arrays exceed their declared bound")
    return payload


def qualify_response_geometry_development_context(
    *,
    config: ResponseGeometryDevelopmentQualificationConfig,
    fit: ResponseGeometryDevelopmentFitResult,
    fit_payload: bytes,
    fit_artifact: ArtifactIdentity,
    calibration: ResponseGeometryDevelopmentCalibrationResult,
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...],
    validation_artifact_id: str,
    payload_plane: CandidatePayloadPlane,
    profile_owner: QualificationProofOwner,
    evaluator_implementation: CandidateEvaluatorImplementation,
    selector_capability: ObjectIdentity,
    selector_implementation: ObjectIdentity,
) -> tuple[
    ResponseGeometryDevelopmentValidationReport,
    bytes,
    QualificationScopeSpec,
    ClaimUnitBinding,
    ComponentUncertaintyFamilyAssessment,
    LawQualificationResult,
]:
    """Read frozen fit/calibration, assess every candidate, and call the sole finalizer.

    The runtime provider supplies authorized input bytes and a production payload
    plane. This scientific composition neither acquires source nor fits a model.
    """
    method_identity = ObjectIdentity.from_record(config.method.config_id, config.method)
    fit_identity = ObjectIdentity.from_record(fit.result_id, fit)
    if (
        fit.config != method_identity
        or calibration.config != method_identity
        or calibration.fit_result != fit_identity
        or calibration.calibration.context != fit.context
        or fit_artifact.sha256 != fit.data_sha256
        or fit_artifact.size_bytes != len(fit_payload)
        or any(v.report.projection_config != config.method.projection_config for v in views)
        or evaluator_implementation.evaluator_key != DEVELOPMENT_LAW_KEY
    ):
        raise ValueError("development qualification changes authenticated fit/calibration/projection lineage")
    groups = read_response_geometry_development_fit(fit, fit_payload)
    operands = validate_response_geometry_development_context(groups, calibration.calibration, views, context=fit.context)
    data = _validation_bytes(config.method, operands)
    report = ResponseGeometryDevelopmentValidationReport(
        fit.context,
        method_identity,
        fit_identity,
        ObjectIdentity.from_record(calibration.result_id, calibration),
        report_identities(views),
        tuple(v.screen for v in operands),
        tuple((v.screen.representation, response_geometry_development_qualification_metrics(v, views)) for v in operands),
        sha256(data).hexdigest(),
    )
    report_identity = ObjectIdentity.from_record(report.report_id, report)
    scope, binding, ledger = response_geometry_development_family(
        system=config.system,
        source=config.source,
        config=config.method,
        projection=report_identity,
        groups=groups,
        selector_capability=selector_capability,
        selector_implementation=selector_implementation,
    )
    evidence = EvidenceLink(
        f"evidence-link.{report.report_id}",
        EvidenceRelation.DERIVED_FROM,
        report_identity,
        ObjectIdentity.from_record(config.system.relation.relation_id, config.system.relation),
        tuple(sorted((fit_artifact.artifact_id, validation_artifact_id))),
        config.system.world.world_id,
        ledger.obligation_template.information_cutoff_id,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        (VisibilityCeiling.DEVELOPMENT_ONLY,),
        "Authenticated fit and independent internal-validation operands for the finite native relation; development-only.",
    )
    implementation = ObjectIdentity.from_record(
        evaluator_implementation.implementation_id, evaluator_implementation
    )
    decoder, profile = ResponseGeometryDevelopmentAffineLawDecoder(), ResponseGeometryDevelopmentAffineQualificationProfile(profile_owner)
    decoders = CandidatePayloadDecoderRegistry((decoder,))
    assembler = LawAssessmentAssembler(
        payload_plane, decoders, QualificationProfileEvaluatorRegistry((profile,))
    )
    metrics_by_representation = dict(report.qualification_metrics)
    assessments = []
    for group in groups:
        payload = build_response_geometry_development_affine_payload(group, fit, fit_artifact, calibration)
        artifact = ArtifactIdentity(
            f"payload.{payload.payload_id}",
            "development-complete-affine-law",
            payload.SCHEMA,
            payload.fingerprint(),
            "application/json",
            len(payload.canonical_bytes()),
        )
        evaluator = ExecutableReference(
            f"evaluator.{payload.payload_id}",
            evaluator_implementation.capability_key,
            evaluator_implementation.capability_version,
            DEVELOPMENT_LAW_KEY,
            artifact,
            SafePayloadFormat.CANONICAL_JSON,
            LawEvaluationRequest.SCHEMA,
            LawEvaluationResult.SCHEMA,
            True,
        )
        publication = payload_plane.publish_candidate_payload(
            payload=payload.canonical_bytes(),
            evaluator=evaluator,
            implementation=implementation,
            decoder_schema=decoder.decoder_schema,
            decoder_version=decoder.decoder_version,
            maximum_decode_bytes=DEVELOPMENT_LAW_MAXIMUM_BYTES,
        )
        metrics = metrics_by_representation[group.representation]
        candidate = response_geometry_development_candidate_evidence(
            ledger, binding, payload, publication, metrics, (evidence,)
        )
        assessments.append(
            assembler.assemble(
                system=config.system,
                candidate=candidate,
                qualification_profile=ObjectIdentity.from_record(
                    profile.profile.profile_id, profile.profile
                ),
                metrics=tuple(
                    NamedDecimal(f"{candidate.candidate_id}.{v.value_id}", v.value, v.unit)
                    for v in metrics
                ),
            )
        )
    family = CandidateFamilyAssembler().assemble(ledger, tuple(assessments))
    qualification = ResponseLawQualificationService(payload_plane, decoders).qualify(
        config.system, report_identity, family
    )
    return report, data, scope, binding, family, qualification
