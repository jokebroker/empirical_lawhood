"""Conditional sealed-evaluation DAG for one eligible selective dependence response target.

The provider is constructed only after development has closed and an exact
outcome-reveal authority exists.  Configuration carries fingerprints and
static keys only; target physics and evaluator callables remain composition
owned.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import Callable, ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputKind,
    WorkerInputPort,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificInputRole, ScientificStage
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .analysis import build_panel
from .analysis_design import SelectiveDependenceResponseTargetAnalysisFreeze
from .contracts import SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponsePhase, SelectiveDependenceResponseTargetHandoff, SelectiveDependenceResponseTargetPanel
from .evaluation import SelectiveDependenceResponseEvaluationPackage, SelectiveDependenceResponseSealedShardManifest, SelectiveDependenceResponseTargetEvaluationBundle, build_sealed_index
from .development_completion import SelectiveDependenceResponseDevelopmentCompletionEnvelope
from .forecast import SelectiveDependenceResponseDevelopmentBundle, SelectiveDependenceResponseStudyForecastQualification
from .study_completion import SelectiveDependenceResponseStudyForecastCompletionEnvelope


SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION = "1.0.0"


class SelectiveDependenceResponseEvaluationOperation(StrEnum):
    FREEZE_EVALUATION_PACKAGE = "FREEZE_EVALUATION_PACKAGE"
    GENERATE_EVALUATION = "GENERATE_EVALUATION"
    PUBLISH_TARGET_HANDOFF = "PUBLISH_TARGET_HANDOFF"
    REVEAL_AND_ADJUDICATE = "REVEAL_AND_ADJUDICATE"


def evaluation_capability_key(target_slug: str, operation: SelectiveDependenceResponseEvaluationOperation) -> str:
    return f"simulator.selective-dependence-response.{target_slug}.{operation.value.lower().replace('_', '-')}"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseEvaluationRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-evaluation-runtime-config'

    config_id: str
    target_id: str
    target_slug: str
    operation: SelectiveDependenceResponseEvaluationOperation
    capability_key: str
    capability_version: str
    evaluation_package_sha256: str
    development_completion_sha256: str
    development_bundle_sha256: str
    study_completion_sha256: str
    design_sha256: str
    analysis_freeze_sha256: str
    construct_review_sha256: str
    source_qualification_sha256: str
    execution_authority_sha256: str
    reveal_authority_sha256: str
    evaluation_complete_unit_ids_sha256: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        for name in ("config_id", "target_id", "target_slug", "capability_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        for name in (
            "evaluation_package_sha256",
            "development_completion_sha256",
            "development_bundle_sha256",
            'study_completion_sha256',
            "design_sha256",
            "analysis_freeze_sha256",
            "construct_review_sha256",
            "source_qualification_sha256",
            "execution_authority_sha256",
            "reveal_authority_sha256",
            "evaluation_complete_unit_ids_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.capability_key != evaluation_capability_key(self.target_slug, self.operation):
            raise ValueError("evaluation operation and capability key differ")
        if self.capability_version != SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION:
            raise ValueError("evaluation capability version differs")
        if not 0 < self.maximum_input_bytes <= 256 * 1024**2:
            raise ValueError("evaluation input byte ceiling differs")


def decode_evaluation_runtime_config(payload: bytes) -> SelectiveDependenceResponseEvaluationRuntimeConfig:
    return decode_canonical_bytes(payload, SelectiveDependenceResponseEvaluationRuntimeConfig, maximum_bytes=256 * 1024)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseEvaluationBinding:
    """Composition-owned exact target, parent and authority binding."""

    target_id: str
    target_slug: str
    provider_key: str
    design: CanonicalRecord
    design_type: type[CanonicalRecord]
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze
    development_completion: SelectiveDependenceResponseDevelopmentCompletionEnvelope
    development: SelectiveDependenceResponseDevelopmentBundle
    construct_review: SelectiveDependenceResponseConstructReviewAttestation
    source_qualification: CanonicalRecord
    source_qualification_type: type[CanonicalRecord]
    evaluation_package: SelectiveDependenceResponseEvaluationPackage
    study_completion: SelectiveDependenceResponseStudyForecastCompletionEnvelope
    execution_authority: StudyOperationAuthority
    reveal_authority: StudyOperationAuthority
    verify_source_binding: Callable[[CanonicalRecord, CanonicalRecord], None]
    execute_unit: Callable[[CanonicalRecord, str, SelectiveDependenceResponsePhase], SelectiveDependenceResponseCompleteUnitResult]
    evaluate_target: Callable[..., SelectiveDependenceResponseTargetEvaluationBundle]

    def __post_init__(self) -> None:
        for name in ("target_id", "target_slug", "provider_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not isinstance(self.design, self.design_type):
            raise ValueError("evaluation design type differs")
        if not isinstance(self.source_qualification, self.source_qualification_type):
            raise ValueError("evaluation source-qualification type differs")
        if any(
            value != self.target_id
            for value in (
                self.development.target_id,
                self.development_completion.target_id,
                self.construct_review.target_id,
                self.evaluation_package.target_id,
            )
        ):
            raise ValueError("evaluation binding crosses targets")
        if self.development_completion.development_bundle != self.development:
            raise ValueError("evaluation binding development completion differs")
        if self.evaluation_package.development_completion != ObjectIdentity.from_record(
            self.development_completion.envelope_id,
            self.development_completion,
        ):
            raise ValueError("evaluation package binds another development completion")
        if self.evaluation_package.development_bundle != ObjectIdentity.from_record(
            self.development.bundle_id, self.development
        ):
            raise ValueError("evaluation package binds another development bundle")
        if self.evaluation_package.design != ObjectIdentity.from_record(
            getattr(self.design, "design_id"), self.design
        ):
            raise ValueError("evaluation package binds another design")
        if self.evaluation_package.analysis_freeze != ObjectIdentity.from_record(
            self.analysis_freeze.freeze_id, self.analysis_freeze
        ):
            raise ValueError("evaluation package binds another analysis freeze")
        if self.analysis_freeze.design != self.evaluation_package.design:
            raise ValueError("evaluation analysis freeze binds another design")
        if self.evaluation_package.construct_review != ObjectIdentity.from_record(
            self.construct_review.attestation_id, self.construct_review
        ):
            raise ValueError("evaluation package binds another construct review")
        qualification_id = getattr(self.source_qualification, "qualification_id")
        if self.evaluation_package.source_qualification != ObjectIdentity.from_record(
            qualification_id, self.source_qualification
        ):
            raise ValueError("evaluation package binds another source qualification")
        if self.evaluation_package.study_forecast_completion != (
            ObjectIdentity.from_record(
                self.study_completion.envelope_id,
                self.study_completion,
            )
        ):
            raise ValueError("evaluation package binds another programme completion")
        if self.evaluation_package.study_forecast_qualification != (
            ObjectIdentity.from_record(
                self.study_completion.study_qualification.qualification_id,
                self.study_completion.study_qualification,
            )
        ):
            raise ValueError("evaluation package binds another programme forecast gate")
        package_identity = ObjectIdentity.from_record(
            self.evaluation_package.package_id, self.evaluation_package
        )
        programme_identity = ObjectIdentity.from_record(
            self.study_completion.envelope_id,
            self.study_completion,
        )
        if (
            self.execution_authority.kind is not StudyAuthorityKind.EXPERIMENT_EXECUTION
            or not self.execution_authority.allows_execution
            or self.execution_authority.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.execution_authority.subject != package_identity
            or self.execution_authority.prerequisite_authority != programme_identity
        ):
            raise ValueError("evaluation binding lacks exact execution authority")
        execution_identity = ObjectIdentity.from_record(
            self.execution_authority.authority_id, self.execution_authority
        )
        if (
            self.reveal_authority.kind is not StudyAuthorityKind.OUTCOME_REVEAL
            or not self.reveal_authority.allows_reveal
            or self.reveal_authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.reveal_authority.subject != package_identity
            or self.reveal_authority.prerequisite_authority != execution_identity
        ):
            raise ValueError("evaluation binding lacks exact reveal authority")

    @property
    def study_forecast_qualification(
        self,
    ) -> SelectiveDependenceResponseStudyForecastQualification:
        return self.study_completion.study_qualification


def evaluation_runtime_config(
    binding: SelectiveDependenceResponseEvaluationBinding,
    operation: SelectiveDependenceResponseEvaluationOperation,
) -> SelectiveDependenceResponseEvaluationRuntimeConfig:
    return SelectiveDependenceResponseEvaluationRuntimeConfig(
        config_id=(
            f"selective-dependence-response.{binding.target_slug}.{operation.value.lower().replace('_', '-')}.config"
        ),
        target_id=binding.target_id,
        target_slug=binding.target_slug,
        operation=operation,
        capability_key=evaluation_capability_key(binding.target_slug, operation),
        capability_version=SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION,
        evaluation_package_sha256=binding.evaluation_package.fingerprint(),
        development_completion_sha256=binding.development_completion.fingerprint(),
        development_bundle_sha256=binding.development.fingerprint(),
        study_completion_sha256=binding.study_completion.fingerprint(),
        design_sha256=binding.design.fingerprint(),
        analysis_freeze_sha256=binding.analysis_freeze.fingerprint(),
        construct_review_sha256=binding.construct_review.fingerprint(),
        source_qualification_sha256=binding.source_qualification.fingerprint(),
        execution_authority_sha256=binding.execution_authority.fingerprint(),
        reveal_authority_sha256=binding.reveal_authority.fingerprint(),
        evaluation_complete_unit_ids_sha256=(
            binding.evaluation_package.evaluation_complete_unit_ids_sha256
        ),
        maximum_input_bytes=256 * 1024**2,
    )


def _resources(operation: SelectiveDependenceResponseEvaluationOperation) -> ResourceBudget:
    values = {
        SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE: (
            10,
            64 * 1024**2,
            2 * 1024**2,
        ),
        SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION: (
            1800,
            1536 * 1024**2,
            128 * 1024**2,
        ),
        SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE: (
            300,
            1536 * 1024**2,
            64 * 1024**2,
        ),
        SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF: (
            30,
            128 * 1024**2,
            2 * 1024**2,
        ),
    }[operation]
    wall, memory, output = values
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=memory,
        gpu_devices=0,
        wall_time_seconds=wall,
        source_scan_bytes=256 * 1024**2,
        output_bytes=output,
    )


def build_evaluation_registry(
    binding: SelectiveDependenceResponseEvaluationBinding, *, implementation_sha256: str
) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    schemas = {
        SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE: (
            (
                SelectiveDependenceResponseEvaluationPackage.SCHEMA,
                SelectiveDependenceResponseStudyForecastCompletionEnvelope.SCHEMA,
                SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA,
                SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
            ),
            (SelectiveDependenceResponseEvaluationPackage.SCHEMA,),
        ),
        SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION: (
            (
                SelectiveDependenceResponseEvaluationPackage.SCHEMA,
                binding.design.SCHEMA,
                binding.source_qualification_type.SCHEMA,
                StudyOperationAuthority.SCHEMA,
            ),
            (SelectiveDependenceResponseSealedShardManifest.SCHEMA, SelectiveDependenceResponseTargetPanel.SCHEMA),
        ),
        SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE: (
            (
                SelectiveDependenceResponseEvaluationPackage.SCHEMA,
                SelectiveDependenceResponseSealedShardManifest.SCHEMA,
                SelectiveDependenceResponseTargetPanel.SCHEMA,
                SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA,
                SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,
                SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
                StudyOperationAuthority.SCHEMA,
            ),
            (SelectiveDependenceResponseTargetEvaluationBundle.SCHEMA,),
        ),
        SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF: (
            (SelectiveDependenceResponseTargetEvaluationBundle.SCHEMA,),
            (SelectiveDependenceResponseTargetHandoff.SCHEMA,),
        ),
    }
    kinds = {
        SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE: CapabilityKind.REPORTER,
        SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION: CapabilityKind.SIMULATOR,
        SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE: CapabilityKind.EVALUATOR,
        SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF: CapabilityKind.REPORTER,
    }
    manifests = []
    for operation in SelectiveDependenceResponseEvaluationOperation:
        permissions: tuple[CapabilityPermission, ...]
        if operation is SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE:
            permissions = (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            )
            maximum_access = OutcomeAccess.EVALUATOR_REVEAL
        elif operation is SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF:
            permissions = (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
            )
            maximum_access = OutcomeAccess.EVALUATION_REVEALED
        elif operation is SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION:
            permissions = (
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            )
            maximum_access = OutcomeAccess.EVALUATION_SEALED
        else:
            permissions = (
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            )
            maximum_access = OutcomeAccess.DEVELOPMENT_VISIBLE
        input_schemas, output_schemas = schemas[operation]
        manifests.append(
            CapabilityManifest(
                capability_key=evaluation_capability_key(binding.target_slug, operation),
                capability_version=SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION,
                kind=kinds[operation],
                config_schema=SelectiveDependenceResponseEvaluationRuntimeConfig.SCHEMA,
                config_schema_sha256=sha256(
                    SelectiveDependenceResponseEvaluationRuntimeConfig.SCHEMA.encode("ascii")
                ).hexdigest(),
                input_schema_ids=tuple(sorted(input_schemas)),
                output_schema_ids=tuple(sorted(output_schemas)),
                permissions=tuple(sorted(permissions)),
                maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                maximum_outcome_access=maximum_access,
                resource_ceiling=_resources(operation),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id=f"cpython-3.11-selective-dependence-response-{binding.target_slug}-evaluation",
                requires_clean_commit=True,
                requires_active_mount=True,
                requires_network=False,
                conformance_check_ids=tuple(
                    sorted(
                        (
                            "complete-unit-nonreplication",
                            "exact-sealed-full-fan-in",
                            f"{binding.target_slug}-{operation.value.lower().replace('_', '-')}",
                            "single-authorized-reveal",
                        )
                    )
                ),
                implementation_sha256=implementation_sha256,
            )
        )
    return CapabilityRegistry(
        registry_id=f"selective-dependence-response-{binding.target_slug}-evaluation-runtime",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def evaluation_candidate_registrations(
    binding: SelectiveDependenceResponseEvaluationBinding, *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    registry = build_evaluation_registry(binding, implementation_sha256=implementation_sha256)
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=binding.provider_key,
            provider_version=SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=256 * 1024,
        )
        for manifest in registry.capabilities
    )


def _step_id(operation: SelectiveDependenceResponseEvaluationOperation) -> str:
    return {
        SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE: "freeze-evaluation-package",
        SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION: "generate-sealed-evaluation-panel",
        SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE: "reveal-and-adjudicate-target",
        SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF: "publish-target-handoff",
    }[operation]


def build_evaluation_protocol(
    binding: SelectiveDependenceResponseEvaluationBinding,
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    specs = {
        SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE: (
            ScientificStage.FREEZE,
            (),
            (SelectiveDependenceResponseEvaluationPackage.SCHEMA,),
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            BarrierKind.FREEZE,
        ),
        SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION: (
            ScientificStage.ACQUIRE,
            ("freeze-evaluation-package",),
            (SelectiveDependenceResponseSealedShardManifest.SCHEMA, SelectiveDependenceResponseTargetPanel.SCHEMA),
            OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
            BarrierKind.AUTHORITY,
        ),
        SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE: (
            ScientificStage.EVALUATE,
            (
                "freeze-evaluation-package",
                "generate-sealed-evaluation-panel",
            ),
            (SelectiveDependenceResponseTargetEvaluationBundle.SCHEMA,),
            OutcomeAccess.EVALUATOR_REVEAL,
            VisibilityCeiling.OUTCOME_VISIBLE,
            BarrierKind.REVEAL,
        ),
        SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF: (
            ScientificStage.REPORT,
            ("reveal-and-adjudicate-target",),
            (SelectiveDependenceResponseTargetHandoff.SCHEMA,),
            OutcomeAccess.EVALUATION_REVEALED,
            VisibilityCeiling.OUTCOME_VISIBLE,
            BarrierKind.NONE,
        ),
    }
    expected_steps = {_step_id(value) for value in SelectiveDependenceResponseEvaluationOperation}
    if set(config_by_step_id) != expected_steps:
        raise ValueError("evaluation protocol config roster differs")
    steps = []
    for operation in SelectiveDependenceResponseEvaluationOperation:
        step_id = _step_id(operation)
        stage, dependencies, output_schemas, access, visibility, barrier = specs[operation]
        manifest = registry.resolve(
            evaluation_capability_key(binding.target_slug, operation),
            SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION,
        )
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("evaluation protocol config schema differs")
        outputs = tuple(
            sorted(
                (
                    OutputTemplate(
                        output_id=f"{step_id}.{schema.rsplit('/', 2)[-2]}",
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    )
                    for schema in output_schemas
                ),
                key=lambda value: value.output_id,
            )
        )
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stage,
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=config,
                dependency_step_ids=tuple(sorted(dependencies)),
                outputs=outputs,
                required_permissions=manifest.permissions,
                requested_outcome_access=access,
                visibility_ceiling=visibility,
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(f"selective-dependence-response-{binding.target_slug}-{step_id}",),
                barrier=barrier,
                maximum_attempts=1,
                obligation_ids=(f"selective-dependence-response-{binding.target_slug}-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id=f"selective-dependence-response-{binding.target_slug}-evaluation-protocol",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _output_for_schema(step: ProtocolStepTemplate, schema: str) -> OutputTemplate:
    matches = tuple(value for value in step.outputs if value.payload_schema == schema)
    if len(matches) != 1:
        raise ValueError(f"evaluation step lacks one output for {schema}")
    return matches[0]


def _external_record_ceiling(record: CanonicalRecord) -> int:
    """Use the exact frozen control-record size, with a small bounded margin."""

    return max(256 * 1024, len(record.canonical_bytes()))


def evaluation_scientific_graph(
    binding: SelectiveDependenceResponseEvaluationBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    by_step = {value.step_id: value for value in protocol.steps}
    nodes = tuple(
        sorted(
            (
                CandidateGraphNode(
                    node_id=step.step_id,
                    stage=step.stage,
                    capability_key=step.capability_key,
                    capability_version=step.capability_version,
                    implementation_sha256=registry.resolve(
                        step.capability_key, step.capability_version
                    ).implementation_sha256,
                    protocol_step_sha256=step.fingerprint(),
                    obligation_ids=step.obligation_ids,
                    outcome_access=step.requested_outcome_access,
                    visibility_ceiling=step.visibility_ceiling,
                    resource_budget=step.resource_budget,
                )
                for step in protocol.steps
            ),
            key=lambda value: value.node_id,
        )
    )
    external_specs = (
        (
            "input.evaluation-package",
            binding.evaluation_package.package_id,
            binding.evaluation_package,
            ScientificInputRole.PARENT_RECEIPT,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            ("freeze-evaluation-package",),
        ),
        (
            "input.programme-forecast-completion",
            binding.study_completion.envelope_id,
            binding.study_completion,
            ScientificInputRole.PARENT_RECEIPT,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            ("freeze-evaluation-package",),
        ),
        (
            "input.development-completion",
            binding.development_completion.envelope_id,
            binding.development_completion,
            ScientificInputRole.PARENT_RECEIPT,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            ("freeze-evaluation-package", "reveal-and-adjudicate-target"),
        ),
        (
            "input.target-design",
            getattr(binding.design, "design_id"),
            binding.design,
            ScientificInputRole.MODEL,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            ("generate-sealed-evaluation-panel",),
        ),
        (
            "input.target-analysis-freeze",
            binding.analysis_freeze.freeze_id,
            binding.analysis_freeze,
            ScientificInputRole.DENOMINATOR,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            ("freeze-evaluation-package", "reveal-and-adjudicate-target"),
        ),
        (
            "input.source-qualification",
            getattr(binding.source_qualification, "qualification_id"),
            binding.source_qualification,
            ScientificInputRole.PARENT_RECEIPT,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
            ("generate-sealed-evaluation-panel",),
        ),
        (
            "input.execution-authority",
            binding.execution_authority.authority_id,
            binding.execution_authority,
            ScientificInputRole.AUTHORITY,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            ("generate-sealed-evaluation-panel",),
        ),
        (
            "input.construct-review",
            binding.construct_review.attestation_id,
            binding.construct_review,
            ScientificInputRole.PARENT_RECEIPT,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            ("reveal-and-adjudicate-target",),
        ),
        (
            "input.reveal-authority",
            binding.reveal_authority.authority_id,
            binding.reveal_authority,
            ScientificInputRole.AUTHORITY,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            ("reveal-and-adjudicate-target",),
        ),
    )
    external_inputs = tuple(
        sorted(
            (
                CandidateGraphExternalInput(
                    input_id=input_id,
                    scientific_role=role,
                    logical_artifact_id=artifact_id,
                    content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
                    expected_content_sha256=record.fingerprint(),
                    payload_schema=record.SCHEMA,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    maximum_size_bytes=_external_record_ceiling(record),
                    outcome_access=access,
                    visibility_ceiling=visibility,
                )
                for input_id, artifact_id, record, role, access, visibility, _ in external_specs
            ),
            key=lambda value: value.input_id,
        )
    )
    edge_specs = (
        (
            "freeze-evaluation-package",
            SelectiveDependenceResponseEvaluationPackage.SCHEMA,
            "generate-sealed-evaluation-panel",
            ScientificInputRole.MODEL,
        ),
        (
            "freeze-evaluation-package",
            SelectiveDependenceResponseEvaluationPackage.SCHEMA,
            "reveal-and-adjudicate-target",
            ScientificInputRole.MODEL,
        ),
        (
            "generate-sealed-evaluation-panel",
            SelectiveDependenceResponseTargetPanel.SCHEMA,
            "reveal-and-adjudicate-target",
            ScientificInputRole.OUTCOME,
        ),
        (
            "generate-sealed-evaluation-panel",
            SelectiveDependenceResponseSealedShardManifest.SCHEMA,
            "reveal-and-adjudicate-target",
            ScientificInputRole.QUALIFICATION,
        ),
        (
            "reveal-and-adjudicate-target",
            SelectiveDependenceResponseTargetEvaluationBundle.SCHEMA,
            "publish-target-handoff",
            ScientificInputRole.OUTCOME,
        ),
    )
    edges = [
        CandidateGraphEdge(
            edge_id=f"edge.{input_id}.{consumer_id}",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=input_id,
            consumer_node_id=consumer_id,
            consumer_input_id=f"{input_id}.{consumer_id}",
            scientific_role=role,
            logical_artifact_id=artifact_id,
            payload_schema=record.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=_external_record_ceiling(record),
            outcome_access=access,
            visibility_ceiling=visibility,
            barrier=(
                BarrierKind.REVEAL
                if by_step[consumer_id].barrier is BarrierKind.REVEAL
                else by_step[consumer_id].barrier
            ),
        )
        for input_id, artifact_id, record, role, access, visibility, consumers in external_specs
        for consumer_id in consumers
    ]
    for producer_id, schema, consumer_id, role in edge_specs:
        producer = by_step[producer_id]
        output = _output_for_schema(producer, schema)
        consumer = by_step[consumer_id]
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{producer_id}.{output.output_id}.{consumer_id}",
                producer_node_id=producer_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=consumer_id,
                consumer_input_id=f"{producer_id}.{schema.rsplit('/', 2)[-2]}",
                scientific_role=role,
                logical_artifact_id=(
                    f"artifact.selective-dependence-response.{binding.target_slug}.{producer_id}."
                    f"{schema.rsplit('/', 2)[-2]}"
                ),
                payload_schema=schema,
                media_type=output.media_type,
                maximum_size_bytes=producer.resource_budget.output_bytes,
                outcome_access=producer.requested_outcome_access,
                visibility_ceiling=producer.visibility_ceiling,
                barrier=(
                    BarrierKind.REVEAL
                    if consumer.barrier is BarrierKind.REVEAL
                    else BarrierKind.NONE
                ),
            )
        )
    return CandidateScientificGraph(
        graph_id=f"graph.selective-dependence-response.{binding.target_slug}.evaluation",
        external_inputs=external_inputs,
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def evaluation_candidate_catalog(
    binding: SelectiveDependenceResponseEvaluationBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateCapabilityCatalog:
    graph = evaluation_scientific_graph(binding, protocol=protocol, registry=registry)
    incoming = {
        node.node_id: tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id
        )
        for node in graph.nodes
    }
    coverage = ObligationCoverage(
        coverage_id=f"coverage.selective-dependence-response.{binding.target_slug}.evaluation",
        bindings=tuple(
            sorted(
                (
                    ObligationCoverageBinding(
                        obligation_id=obligation_id,
                        proof_owner_node_id=step.step_id,
                        required_output_id=step.outputs[0].output_id,
                        contributor_edge_ids=tuple(sorted(incoming[step.step_id])),
                    )
                    for step in protocol.steps
                    for obligation_id in step.obligation_ids
                ),
                key=lambda value: value.obligation_id,
            )
        ),
    )
    implementation_ids = {value.implementation_sha256 for value in registry.capabilities}
    if len(implementation_ids) != 1:
        raise ValueError("evaluation implementation identities differ")
    return CandidateCapabilityCatalog(
        catalog_id=f"selective-dependence-response-{binding.target_slug}-evaluation-candidate-catalog",
        registrations=evaluation_candidate_registrations(
            binding, implementation_sha256=next(iter(implementation_ids))
        ),
        templates=(
            StudyTemplate(
                template_key=f"selective-dependence-response.{binding.target_slug}.evaluation",
                template_version=SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION,
                protocol=protocol,
                graph=graph,
                coverage=coverage,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseEvaluationConfigAdapter:
    provider: str

    @property
    def provider_key(self) -> str:
        return self.provider

    @property
    def provider_version(self) -> str:
        return SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != SelectiveDependenceResponseEvaluationRuntimeConfig.SCHEMA:
            raise ValueError("evaluation config adapter schema differs")
        decode_evaluation_runtime_config(payload)


def evaluation_config_decoders(
    binding: SelectiveDependenceResponseEvaluationBinding,
    catalog: CandidateCapabilityCatalog,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    providers = {
        f"{value.provider_key}@{value.provider_version}" for value in catalog.registrations
    }
    if providers != {f"{binding.provider_key}@{SELECTIVE_DEPENDENCE_RESPONSE_EVALUATION_PROTOCOL_VERSION}"}:
        raise ValueError("evaluation candidate provider roster differs")
    return (SelectiveDependenceResponseEvaluationConfigAdapter(binding.provider_key),)


def _input_port(
    context: TaskContext,
    schema: str,
    kind: WorkerInputKind,
) -> WorkerInputPort:
    matches = tuple(
        value
        for value in context.input_ports
        if value.kind is kind and value.payload_schema == schema
    )
    if len(matches) != 1:
        raise ValueError(
            f"evaluation runner requires one {kind.value.lower()} input with schema {schema}"
        )
    return matches[0]


def _dependency_port(context: TaskContext, schema: str) -> WorkerInputPort:
    return _input_port(context, schema, WorkerInputKind.DEPENDENCY)


def _external_port(context: TaskContext, schema: str) -> WorkerInputPort:
    return _input_port(context, schema, WorkerInputKind.EXTERNAL)


def _decode_dependency(
    context: TaskContext,
    schema: str,
    record_type: type[CanonicalRecord],
) -> CanonicalRecord:
    return decode_canonical_bytes(
        _dependency_port(context, schema).read(256 * 1024**2),
        record_type,
        maximum_bytes=256 * 1024**2,
    )


def _decode_external(
    context: TaskContext,
    schema: str,
    record_type: type[CanonicalRecord],
) -> CanonicalRecord:
    return decode_canonical_bytes(
        _external_port(context, schema).read(256 * 1024**2),
        record_type,
        maximum_bytes=256 * 1024**2,
    )


class _PackageRunner:
    def __init__(self, manifest: CapabilityManifest, binding: SelectiveDependenceResponseEvaluationBinding) -> None:
        self.manifest = manifest
        self.binding = binding
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        package = _decode_external(
            context, SelectiveDependenceResponseEvaluationPackage.SCHEMA, SelectiveDependenceResponseEvaluationPackage
        )
        programme_completion = _decode_external(
            context,
            SelectiveDependenceResponseStudyForecastCompletionEnvelope.SCHEMA,
            SelectiveDependenceResponseStudyForecastCompletionEnvelope,
        )
        development_completion = _decode_external(
            context,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        )
        analysis_freeze = _decode_external(
            context,
            SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
            SelectiveDependenceResponseTargetAnalysisFreeze,
        )
        if not isinstance(
            development_completion,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        ):
            raise TypeError("evaluation development completion decoded to another type")
        if (
            package != self.binding.evaluation_package
            or programme_completion != self.binding.study_completion
            or development_completion != self.binding.development_completion
            or analysis_freeze != self.binding.analysis_freeze
        ):
            raise ValueError("evaluation package freeze inputs differ from binding")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=package.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("selective-dependence-response-evaluation-package-byte-frozen", True, ()),),
        )


class _GeneratorRunner:
    def __init__(self, manifest: CapabilityManifest, binding: SelectiveDependenceResponseEvaluationBinding) -> None:
        self.manifest = manifest
        self.binding = binding
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if context.outcome_access is not OutcomeAccess.EVALUATION_SEALED or {
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        } & set(context.permissions):
            raise PermissionError("evaluation generator crossed the reveal boundary")
        package = _decode_dependency(
            context, SelectiveDependenceResponseEvaluationPackage.SCHEMA, SelectiveDependenceResponseEvaluationPackage
        )
        if package != self.binding.evaluation_package:
            raise ValueError("evaluation generator package differs")
        design = _decode_external(context, self.binding.design.SCHEMA, self.binding.design_type)
        source = _decode_external(
            context,
            self.binding.source_qualification_type.SCHEMA,
            self.binding.source_qualification_type,
        )
        execution_authority = _decode_external(
            context,
            StudyOperationAuthority.SCHEMA,
            StudyOperationAuthority,
        )
        if (
            design != self.binding.design
            or source != self.binding.source_qualification
            or execution_authority != self.binding.execution_authority
        ):
            raise ValueError("evaluation generator external inputs differ from binding")
        self.binding.verify_source_binding(design, source)
        units = tuple(
            self.binding.execute_unit(design, unit_id, SelectiveDependenceResponsePhase.EVALUATION)
            for unit_id in package.evaluation_complete_unit_ids
        )
        panel = build_panel(
            panel_id=f"panel.{self.binding.target_slug}.evaluation",
            target_id=self.binding.target_id,
            complete_units=units,
        )
        shard = SelectiveDependenceResponseSealedShardManifest(
            manifest_id=f"manifest.{self.binding.target_slug}.evaluation-shard",
            target_id=self.binding.target_id,
            evaluation_package=ObjectIdentity.from_record(package.package_id, package),
            sealed_panel=ObjectIdentity.from_record(panel.panel_id, panel),
            complete_unit_ids=panel.expected_complete_unit_ids,
            complete_unit_ids_sha256=panel.expected_complete_unit_ids_sha256,
            native_receiver_value_count=0,
            prediction_correctness_count=0,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        payloads = {
            SelectiveDependenceResponseTargetPanel.SCHEMA: panel.canonical_bytes(),
            SelectiveDependenceResponseSealedShardManifest.SCHEMA: shard.canonical_bytes(),
        }
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=payloads[port.payload_schema],
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("selective-dependence-response-evaluation-full-fan-in-generated", True, ()),
                ReceiptCheck("selective-dependence-response-evaluation-progress-value-free", True, ()),
            ),
        )


class _EvaluatorRunner:
    def __init__(self, manifest: CapabilityManifest, binding: SelectiveDependenceResponseEvaluationBinding) -> None:
        self.manifest = manifest
        self.binding = binding
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        required = {
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        }
        if context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL or not required.issubset(
            context.permissions
        ):
            raise PermissionError("target evaluator lacks exact reveal authority")
        package = _decode_dependency(
            context, SelectiveDependenceResponseEvaluationPackage.SCHEMA, SelectiveDependenceResponseEvaluationPackage
        )
        panel = _decode_dependency(context, SelectiveDependenceResponseTargetPanel.SCHEMA, SelectiveDependenceResponseTargetPanel)
        shard = _decode_dependency(
            context,
            SelectiveDependenceResponseSealedShardManifest.SCHEMA,
            SelectiveDependenceResponseSealedShardManifest,
        )
        if (
            not isinstance(package, SelectiveDependenceResponseEvaluationPackage)
            or not isinstance(panel, SelectiveDependenceResponseTargetPanel)
            or not isinstance(shard, SelectiveDependenceResponseSealedShardManifest)
        ):
            raise TypeError("evaluator dependencies decoded to another type")
        panel_port = _dependency_port(context, SelectiveDependenceResponseTargetPanel.SCHEMA)
        shard_port = _dependency_port(context, SelectiveDependenceResponseSealedShardManifest.SCHEMA)
        generator_receipts = tuple(
            value
            for value in context.dependency_receipts
            if {
                panel_port.materialization_id,
                shard_port.materialization_id,
            }.issubset(value.output_materialization_ids)
        )
        if len(generator_receipts) != 1:
            raise ValueError("sealed panel and manifest lack one exact generator receipt")
        generator_receipt = generator_receipts[0]
        index = build_sealed_index(
            package=package,
            shard_manifest=shard,
            generator_receipt_ids=(generator_receipt.receipt_id,),
            generator_materialization_ids=generator_receipt.output_materialization_ids,
        )
        development_completion = _decode_external(
            context,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        )
        review = _decode_external(
            context,
            SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,
            SelectiveDependenceResponseConstructReviewAttestation,
        )
        reveal_authority = _decode_external(
            context,
            StudyOperationAuthority.SCHEMA,
            StudyOperationAuthority,
        )
        analysis_freeze = _decode_external(
            context,
            SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
            SelectiveDependenceResponseTargetAnalysisFreeze,
        )
        if not isinstance(
            development_completion,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        ):
            raise TypeError("evaluation development completion decoded to another type")
        if (
            development_completion != self.binding.development_completion
            or review != self.binding.construct_review
            or reveal_authority != self.binding.reveal_authority
            or analysis_freeze != self.binding.analysis_freeze
        ):
            raise ValueError("evaluation reveal inputs differ from binding")
        result = self.binding.evaluate_target(
            package=package,
            development=development_completion.development_bundle,
            construct_review=review,
            sealed_index=index,
            panel=panel,
            reveal_authority=ObjectIdentity.from_record(
                reveal_authority.authority_id,
                reveal_authority,
            ),
            evaluator_task_id=context.task_id,
            evaluator_attempt_id=context.attempt_id,
            input_receipt_ids=context.dependency_receipt_ids,
            input_materialization_ids=context.dependency_input_materialization_ids,
            analysis_freeze=analysis_freeze,
        )
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("selective-dependence-response-single-atomic-reveal", True, ()),
                ReceiptCheck("selective-dependence-response-target-axes-terminal", True, ()),
            ),
        )


class _HandoffRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        result = _decode_dependency(
            context,
            SelectiveDependenceResponseTargetEvaluationBundle.SCHEMA,
            SelectiveDependenceResponseTargetEvaluationBundle,
        )
        if not isinstance(result, SelectiveDependenceResponseTargetEvaluationBundle):
            raise TypeError("handoff dependency decoded to another type")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.handoff.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck(
                    "selective-dependence-response-handoff-has-no-native-values",
                    result.handoff.native_numeric_value_count == 0,
                    (),
                ),
            ),
        )


class SelectiveDependenceResponseEvaluationRuntimeProvider(CampaignRuntimeProvider):
    def __init__(self, *, registry: CapabilityRegistry, binding: SelectiveDependenceResponseEvaluationBinding) -> None:
        expected = build_evaluation_registry(
            binding,
            implementation_sha256=registry.capabilities[0].implementation_sha256,
        )
        if registry != expected:
            raise ValueError("evaluation provider registry differs")
        if (
            binding.evaluation_package.evaluation_implementation_sha256
            != registry.capabilities[0].implementation_sha256
        ):
            raise ValueError("evaluation package and provider implementation differ")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.binding = binding
        runners: list[TaskRunner] = []
        for manifest in registry.capabilities:
            operation_text = manifest.capability_key.rsplit(".", 1)[-1].replace("-", "_").upper()
            operation = SelectiveDependenceResponseEvaluationOperation(operation_text)
            if operation is SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE:
                runner: TaskRunner = _PackageRunner(manifest, binding)
            elif operation is SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION:
                runner = _GeneratorRunner(manifest, binding)
            elif operation is SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE:
                runner = _EvaluatorRunner(manifest, binding)
            else:
                runner = _HandoffRunner(manifest)
            runners.append(runner)
        self._runners = tuple(runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("evaluation provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("evaluation provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            self.binding.evaluation_package.package_id: self.binding.evaluation_package,
            self.binding.study_completion.envelope_id: (self.binding.study_completion),
            self.binding.development_completion.envelope_id: (self.binding.development_completion),
            getattr(self.binding.design, "design_id"): self.binding.design,
            self.binding.analysis_freeze.freeze_id: self.binding.analysis_freeze,
            getattr(self.binding.source_qualification, "qualification_id"): (
                self.binding.source_qualification
            ),
            self.binding.execution_authority.authority_id: self.binding.execution_authority,
            self.binding.construct_review.attestation_id: self.binding.construct_review,
            self.binding.reveal_authority.authority_id: self.binding.reveal_authority,
        }
        configs = {
            value.capability_key: evaluation_runtime_config(
                self.binding,
                SelectiveDependenceResponseEvaluationOperation(
                    value.capability_key.rsplit(".", 1)[-1].replace("-", "_").upper()
                ),
            )
            for value in self.registry.capabilities
        }
        for task in plan.tasks:
            records[task.capability.config.artifact_id] = configs[task.capability.capability_key]
        if set(records) != set(specs):
            raise ValueError("evaluation external input roster differs")

        def record_identity(
            record: CanonicalRecord,
        ) -> tuple[str, VisibilityCeiling, OutcomeAccess]:
            if isinstance(record, SelectiveDependenceResponseEvaluationRuntimeConfig):
                return record.config_id, VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND
            if isinstance(record, SelectiveDependenceResponseEvaluationPackage):
                return (
                    record.package_id,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
            if isinstance(record, SelectiveDependenceResponseStudyForecastCompletionEnvelope):
                return (
                    record.envelope_id,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
            if isinstance(record, SelectiveDependenceResponseDevelopmentCompletionEnvelope):
                return (
                    record.envelope_id,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
            if isinstance(record, SelectiveDependenceResponseConstructReviewAttestation):
                return (
                    record.attestation_id,
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
            if isinstance(record, SelectiveDependenceResponseTargetAnalysisFreeze):
                return (
                    record.freeze_id,
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
            if isinstance(record, StudyOperationAuthority):
                return (
                    record.authority_id,
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
            if isinstance(record, self.binding.design_type):
                return (
                    getattr(record, "design_id"),
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
            if isinstance(record, self.binding.source_qualification_type):
                return (
                    getattr(record, "qualification_id"),
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
            raise TypeError("evaluation external input type differs")

        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            record_id, visibility, access = record_identity(record)
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=spec.expected_visibility_ceiling or visibility,
                    outcome_access=spec.expected_outcome_access or access,
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=spec.expected_content_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("evaluation provider semantic registry differs")
        types: dict[str, type[CanonicalRecord]] = {
            SelectiveDependenceResponseEvaluationPackage.SCHEMA: SelectiveDependenceResponseEvaluationPackage,
            SelectiveDependenceResponseSealedShardManifest.SCHEMA: SelectiveDependenceResponseSealedShardManifest,
            SelectiveDependenceResponseTargetPanel.SCHEMA: SelectiveDependenceResponseTargetPanel,
            SelectiveDependenceResponseTargetEvaluationBundle.SCHEMA: SelectiveDependenceResponseTargetEvaluationBundle,
            SelectiveDependenceResponseTargetHandoff.SCHEMA: SelectiveDependenceResponseTargetHandoff,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(
                                field.name
                                for field in fields(record_type)  # type: ignore[arg-type]
                            )
                        ),
                    )
                )
        return tuple(sorted(values, key=lambda value: (value.capability_key, value.payload_schema)))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("evaluation provider adjudication registry differs")
        return None


__all__ = [
    'SelectiveDependenceResponseEvaluationBinding',
    'SelectiveDependenceResponseEvaluationOperation',
    'SelectiveDependenceResponseEvaluationRuntimeConfig',
    'SelectiveDependenceResponseEvaluationRuntimeProvider',
    "build_evaluation_protocol",
    "build_evaluation_registry",
    "decode_evaluation_runtime_config",
    "evaluation_candidate_catalog",
    "evaluation_config_decoders",
    "evaluation_runtime_config",
    "evaluation_scientific_graph",
]
