"""Reusable public development DAG for the two selective dependence response simulator targets.

Configuration contains only static capability identities and content digests.
The concrete target binding is installed by target-owned registration code; it
is never loaded from configuration or an import string.
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
from .contracts import SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponseContaminationLedger, SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponseMethodQuestionFreeze, SelectiveDependenceResponsePhase, SelectiveDependenceResponsePreparationDistributionFreeze, SelectiveDependenceResponseTargetPanel
from .forecast import SelectiveDependenceResponseDevelopmentBundle, SelectiveDependenceResponseDevelopmentLineage
from .method_completion import SelectiveDependenceResponseMethodCompletionEnvelope
from .source_completion import SelectiveDependenceResponseSourceCanaryCompletionEnvelope


SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION = "1.0.0"


class SelectiveDependenceResponseTargetOperation(StrEnum):
    ANALYZE_DEVELOPMENT = "ANALYZE_DEVELOPMENT"
    BIND_CONSTRUCT_REVIEW = "BIND_CONSTRUCT_REVIEW"
    FREEZE_ANALYSIS = "FREEZE_ANALYSIS"
    FREEZE_DESIGN = "FREEZE_DESIGN"
    GENERATE_DEVELOPMENT = "GENERATE_DEVELOPMENT"
    QUALIFY_SOURCE = "QUALIFY_SOURCE"


class SelectiveDependenceResponseTargetStage(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    SOURCE_CANARY = "SOURCE_CANARY"


SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS = (
    SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW,
    SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS,
    SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN,
    SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE,
)
SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS = (
    SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT,
    SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW,
    SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS,
    SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN,
    SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT,
)


def target_capability_key(target_slug: str, operation: SelectiveDependenceResponseTargetOperation) -> str:
    return f"simulator.selective-dependence-response.{target_slug}.{operation.value.lower().replace('_', '-')}"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-runtime-config'

    config_id: str
    target_id: str
    target_slug: str
    operation: SelectiveDependenceResponseTargetOperation
    capability_key: str
    capability_version: str
    design_sha256: str
    preparation_sha256: str
    analysis_freeze_sha256: str
    construct_review_sha256: str
    development_unit_ids_sha256: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        for name in ("config_id", "target_id", "target_slug", "capability_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        for name in (
            "design_sha256",
            "preparation_sha256",
            "analysis_freeze_sha256",
            "construct_review_sha256",
            "development_unit_ids_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.capability_key != target_capability_key(self.target_slug, self.operation):
            raise ValueError("target operation and capability key differ")
        if self.capability_version != SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION:
            raise ValueError("target capability version differs")
        if not 0 < self.maximum_input_bytes <= 256 * 1024**2:
            raise ValueError("target input byte ceiling differs")


def decode_target_runtime_config(payload: bytes) -> SelectiveDependenceResponseTargetRuntimeConfig:
    return decode_canonical_bytes(payload, SelectiveDependenceResponseTargetRuntimeConfig, maximum_bytes=128 * 1024)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetBinding:
    """Composition-owned, non-configurable binding to one static target adapter."""

    target_id: str
    target_slug: str
    provider_key: str
    design: CanonicalRecord
    design_id: str
    design_type: type[CanonicalRecord]
    preparation: SelectiveDependenceResponsePreparationDistributionFreeze
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze
    method_question: SelectiveDependenceResponseMethodQuestionFreeze
    contamination_ledger: SelectiveDependenceResponseContaminationLedger
    construct_review: SelectiveDependenceResponseConstructReviewAttestation
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope | None
    source_qualification_type: type[CanonicalRecord]
    qualify_source: Callable[[CanonicalRecord], tuple[CanonicalRecord, SelectiveDependenceResponseTargetPanel]]
    execute_unit: Callable[[CanonicalRecord, str, SelectiveDependenceResponsePhase], SelectiveDependenceResponseCompleteUnitResult]
    analyze_development: Callable[
        [SelectiveDependenceResponseTargetPanel, SelectiveDependenceResponseDevelopmentLineage], SelectiveDependenceResponseDevelopmentBundle
    ]
    source_qualification: CanonicalRecord | None = None
    source_completion: SelectiveDependenceResponseSourceCanaryCompletionEnvelope | None = None

    def __post_init__(self) -> None:
        for name in ("target_id", "target_slug", "provider_key", "design_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.preparation.target_id != self.target_id:
            raise ValueError("target binding preparation differs")
        if self.analysis_freeze.target_id != self.target_id:
            raise ValueError("target binding analysis freeze differs")
        if self.analysis_freeze.design != ObjectIdentity.from_record(
            self.design_id, self.design
        ) or self.analysis_freeze.preparation_freeze != ObjectIdentity.from_record(
            self.preparation.freeze_id, self.preparation
        ):
            raise ValueError("target binding analysis identities differ")
        if self.construct_review.target_id != self.target_id:
            raise ValueError("target binding construct review differs")
        if not isinstance(self.design, self.design_type):
            raise ValueError("target binding design type differs")
        if self.method_completion is not None and (
            self.method_completion.method_question
            != ObjectIdentity.from_record(
                self.method_question.freeze_id,
                self.method_question,
            )
            or not self.method_completion.all_passed
            or self.method_completion.target_response_count
            or self.method_completion.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("target binding method completion differs")
        if (self.source_qualification is None) != (self.source_completion is None):
            raise ValueError("target development source records must be supplied together")
        if self.source_completion is not None:
            source_qualification = self.source_qualification
            if source_qualification is None:
                raise ValueError("target development source qualification is absent")
            if (
                self.source_completion.target_id != self.target_id
                or self.source_completion.design
                != ObjectIdentity.from_record(self.design_id, self.design)
                or self.source_completion.preparation_freeze
                != ObjectIdentity.from_record(self.preparation.freeze_id, self.preparation)
                or self.source_completion.analysis_freeze
                != ObjectIdentity.from_record(self.analysis_freeze.freeze_id, self.analysis_freeze)
                or self.source_completion.construct_review
                != ObjectIdentity.from_record(
                    self.construct_review.attestation_id, self.construct_review
                )
                or self.source_completion.source_qualification
                != ObjectIdentity.from_record(
                    str(getattr(source_qualification, "qualification_id")),
                    source_qualification,
                )
            ):
                raise ValueError("target development source completion differs")


def target_runtime_config(
    binding: SelectiveDependenceResponseTargetBinding, operation: SelectiveDependenceResponseTargetOperation
) -> SelectiveDependenceResponseTargetRuntimeConfig:
    from .contracts import digest_ids

    return SelectiveDependenceResponseTargetRuntimeConfig(
        config_id=(
            f"selective-dependence-response.{binding.target_slug}.{operation.value.lower().replace('_', '-')}.config"
        ),
        target_id=binding.target_id,
        target_slug=binding.target_slug,
        operation=operation,
        capability_key=target_capability_key(binding.target_slug, operation),
        capability_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
        design_sha256=binding.design.fingerprint(),
        preparation_sha256=binding.preparation.fingerprint(),
        analysis_freeze_sha256=binding.analysis_freeze.fingerprint(),
        construct_review_sha256=binding.construct_review.fingerprint(),
        development_unit_ids_sha256=digest_ids(binding.preparation.development_unit_ids),
        maximum_input_bytes=256 * 1024**2,
    )


def _manifest_resources(operation: SelectiveDependenceResponseTargetOperation) -> ResourceBudget:
    values = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: (5, 32 * 1024**2, 256 * 1024),
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: (5, 32 * 1024**2, 512 * 1024),
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: (5, 32 * 1024**2, 256 * 1024),
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: (180, 1024 * 1024**2, 2 * 1024**2),
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: (
            1800,
            1024 * 1024**2,
            64 * 1024**2,
        ),
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: (
            120,
            1024 * 1024**2,
            32 * 1024**2,
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


def build_target_registry(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    implementation_sha256: str,
    operations: tuple[SelectiveDependenceResponseTargetOperation, ...] = tuple(SelectiveDependenceResponseTargetOperation),
    stage: SelectiveDependenceResponseTargetStage | None = None,
) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if (
        tuple(sorted(set(operations), key=lambda value: value.value))
        != tuple(sorted(operations, key=lambda value: value.value))
        or not operations
    ):
        raise ValueError("target registry operation roster must be unique and nonempty")
    schemas = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: (
            (binding.design.SCHEMA,),
            (binding.design.SCHEMA,),
        ),
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: (
            (SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,),
            (SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,),
        ),
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: (
            (SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,),
            (SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,),
        ),
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: (
            (
                binding.design.SCHEMA,
                SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,
                SelectiveDependenceResponsePreparationDistributionFreeze.SCHEMA,
                SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
                SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
                SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA,
                SelectiveDependenceResponseContaminationLedger.SCHEMA,
            ),
            (binding.source_qualification_type.SCHEMA, SelectiveDependenceResponseTargetPanel.SCHEMA),
        ),
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: (
            (
                binding.design.SCHEMA,
                SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,
                SelectiveDependenceResponsePreparationDistributionFreeze.SCHEMA,
                binding.source_qualification_type.SCHEMA,
                SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
                SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA,
                SelectiveDependenceResponseSourceCanaryCompletionEnvelope.SCHEMA,
            ),
            (SelectiveDependenceResponseTargetPanel.SCHEMA,),
        ),
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: (
            (
                binding.design.SCHEMA,
                SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,
                SelectiveDependenceResponsePreparationDistributionFreeze.SCHEMA,
                SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
                SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA,
                SelectiveDependenceResponseContaminationLedger.SCHEMA,
                binding.source_qualification_type.SCHEMA,
                SelectiveDependenceResponseTargetPanel.SCHEMA,
                SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,
                SelectiveDependenceResponseSourceCanaryCompletionEnvelope.SCHEMA,
            ),
            (SelectiveDependenceResponseDevelopmentBundle.SCHEMA,),
        ),
    }
    kinds = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: CapabilityKind.TRANSFORM,
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: CapabilityKind.TRANSFORM,
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: CapabilityKind.REPORTER,
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: CapabilityKind.NUMERICAL_QUALIFIER,
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: CapabilityKind.SIMULATOR,
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: CapabilityKind.LAW_IDENTIFIER,
    }
    manifests = []
    for operation in operations:
        development_visible = operation in {
            SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE,
            SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT,
            SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT,
        }
        permissions = [CapabilityPermission.READ_EXTERNAL_ARTIFACTS]
        if operation in {
            SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT,
            SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT,
        }:
            permissions.append(CapabilityPermission.READ_DEVELOPMENT)
        input_schemas, output_schemas = schemas[operation]
        manifests.append(
            CapabilityManifest(
                capability_key=target_capability_key(binding.target_slug, operation),
                capability_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
                kind=kinds[operation],
                config_schema=SelectiveDependenceResponseTargetRuntimeConfig.SCHEMA,
                config_schema_sha256=sha256(
                    SelectiveDependenceResponseTargetRuntimeConfig.SCHEMA.encode("ascii")
                ).hexdigest(),
                input_schema_ids=tuple(sorted(input_schemas)),
                output_schema_ids=output_schemas,
                permissions=tuple(sorted(permissions)),
                maximum_evidence_ceiling=(
                    EvidenceCeiling.LOCAL_LAW
                    if development_visible
                    else EvidenceCeiling.NON_PROMOTABLE
                ),
                maximum_outcome_access=(
                    OutcomeAccess.DEVELOPMENT_VISIBLE
                    if development_visible
                    else OutcomeAccess.OUTCOME_BLIND
                ),
                resource_ceiling=_manifest_resources(operation),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id=f"cpython-3.11-selective-dependence-response-{binding.target_slug}",
                requires_clean_commit=True,
                requires_active_mount=development_visible,
                requires_network=False,
                conformance_check_ids=tuple(
                    sorted(
                        (
                            "complete-unit-nonreplication",
                            f"{binding.target_slug}-{operation.value.lower().replace('_', '-')}",
                            "stage-resolved-action-chain",
                        )
                    )
                ),
                implementation_sha256=implementation_sha256,
            )
        )
    return CapabilityRegistry(
        registry_id=(
            f"selective-dependence-response-{binding.target_slug}-"
            f"{(stage.value.lower().replace('_', '-') if stage is not None else 'combined')}-runtime"
        ),
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def target_candidate_registrations(
    binding: SelectiveDependenceResponseTargetBinding, *, registry: CapabilityRegistry
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=binding.provider_key,
            provider_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=128 * 1024,
        )
        for manifest in registry.capabilities
    )


def _step_id(operation: SelectiveDependenceResponseTargetOperation) -> str:
    return {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: "freeze-target-design",
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: "freeze-target-analysis",
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: "bind-construct-review",
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: "qualify-target-source",
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: "generate-development-panel",
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: "analyze-development",
    }[operation]


def build_target_protocol(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    raise ValueError(
        "combined source-canary/development protocols are prohibited; use the split builders"
    )
    specs = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: (
            ScientificStage.FREEZE,
            (),
            (binding.design.SCHEMA,),
            BarrierKind.FREEZE,
        ),
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: (
            ScientificStage.FREEZE,
            (),
            (SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,),
            BarrierKind.FREEZE,
        ),
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: (
            ScientificStage.FREEZE,
            (),
            (SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,),
            BarrierKind.FREEZE,
        ),
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: (
            ScientificStage.QUALIFY,
            ("bind-construct-review", "freeze-target-design"),
            (binding.source_qualification_type.SCHEMA, SelectiveDependenceResponseTargetPanel.SCHEMA),
            BarrierKind.NONE,
        ),
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: (
            ScientificStage.DEVELOP,
            (
                "bind-construct-review",
                "freeze-target-analysis",
                "freeze-target-design",
                "qualify-target-source",
            ),
            (SelectiveDependenceResponseTargetPanel.SCHEMA,),
            BarrierKind.NONE,
        ),
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: (
            ScientificStage.DEVELOP,
            (
                "bind-construct-review",
                "freeze-target-analysis",
                "freeze-target-design",
                "generate-development-panel",
                "qualify-target-source",
            ),
            (SelectiveDependenceResponseDevelopmentBundle.SCHEMA,),
            BarrierKind.FREEZE,
        ),
    }
    expected_steps = {_step_id(value) for value in SelectiveDependenceResponseTargetOperation}
    if set(config_by_step_id) != expected_steps:
        raise ValueError("target protocol config roster differs")
    steps = []
    for operation in SelectiveDependenceResponseTargetOperation:
        step_id = _step_id(operation)
        stage, dependencies, output_schemas, barrier = specs[operation]
        manifest = registry.resolve(
            target_capability_key(binding.target_slug, operation),
            SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
        )
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("target protocol config schema differs")
        access = (
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if operation
            in {
                SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE,
                SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT,
                SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT,
            }
            else OutcomeAccess.OUTCOME_BLIND
        )
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stage,
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=config,
                dependency_step_ids=dependencies,
                outputs=tuple(
                    OutputTemplate(
                        output_id=(
                            f"{step_id}.record"
                            if len(output_schemas) == 1
                            else (
                                f"{step_id}.qualification"
                                if index == 0
                                else f"{step_id}.source-canary-panel"
                            )
                        ),
                        payload_schema=output_schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    )
                    for index, output_schema in enumerate(output_schemas)
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=access,
                visibility_ceiling=(
                    VisibilityCeiling.DEVELOPMENT_ONLY
                    if access is OutcomeAccess.DEVELOPMENT_VISIBLE
                    else VisibilityCeiling.PROSPECTIVE
                ),
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(f"selective-dependence-response-{binding.target_slug}-{step_id}",),
                barrier=barrier,
                maximum_attempts=2,
                obligation_ids=(f"selective-dependence-response-{binding.target_slug}-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id=f"selective-dependence-response-{binding.target_slug}-development-protocol",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _build_split_target_protocol(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    stage: SelectiveDependenceResponseTargetStage,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    if binding.method_completion is None:
        raise ValueError("target protocol requires terminal method completion")
    operations = (
        SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS
        if stage is SelectiveDependenceResponseTargetStage.SOURCE_CANARY
        else SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS
    )
    if stage is SelectiveDependenceResponseTargetStage.DEVELOPMENT and (
        binding.source_qualification is None or binding.source_completion is None
    ):
        raise ValueError("development protocol requires terminal source completion")
    output_schemas = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: (binding.design.SCHEMA,),
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: (SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA,),
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: (SelectiveDependenceResponseConstructReviewAttestation.SCHEMA,),
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: (
            binding.source_qualification_type.SCHEMA,
            SelectiveDependenceResponseTargetPanel.SCHEMA,
        ),
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: (SelectiveDependenceResponseTargetPanel.SCHEMA,),
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: (SelectiveDependenceResponseDevelopmentBundle.SCHEMA,),
    }
    dependencies = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: (),
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: (),
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: (),
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: (
            "bind-construct-review",
            "freeze-target-analysis",
            "freeze-target-design",
        ),
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: (
            "bind-construct-review",
            "freeze-target-analysis",
            "freeze-target-design",
        ),
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: (
            "bind-construct-review",
            "freeze-target-analysis",
            "freeze-target-design",
            "generate-development-panel",
        ),
    }
    stages = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: ScientificStage.FREEZE,
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: ScientificStage.FREEZE,
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: ScientificStage.FREEZE,
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: ScientificStage.QUALIFY,
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: ScientificStage.ACQUIRE,
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: ScientificStage.DEVELOP,
    }
    barriers = {
        operation: (
            BarrierKind.FREEZE
            if operation
            in {
                SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN,
                SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS,
                SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW,
                SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT,
            }
            else BarrierKind.NONE
        )
        for operation in operations
    }
    expected_steps = {_step_id(value) for value in operations}
    if set(config_by_step_id) != expected_steps:
        raise ValueError("split target protocol config roster differs")
    expected_keys = {target_capability_key(binding.target_slug, value) for value in operations}
    if {value.capability_key for value in registry.capabilities} != expected_keys:
        raise ValueError("split target registry operation roster differs")
    steps = []
    for operation in operations:
        step_id = _step_id(operation)
        manifest = registry.resolve(
            target_capability_key(binding.target_slug, operation),
            SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
        )
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("split target protocol config schema differs")
        access = (
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if operation
            in {
                SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE,
                SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT,
                SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT,
            }
            else OutcomeAccess.OUTCOME_BLIND
        )
        schemas = output_schemas[operation]
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stages[operation],
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=config,
                dependency_step_ids=dependencies[operation],
                outputs=tuple(
                    OutputTemplate(
                        output_id=(
                            f"{step_id}.record"
                            if len(schemas) == 1
                            else (
                                f"{step_id}.qualification"
                                if index == 0
                                else f"{step_id}.source-canary-panel"
                            )
                        ),
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    )
                    for index, schema in enumerate(schemas)
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=access,
                visibility_ceiling=(
                    VisibilityCeiling.DEVELOPMENT_ONLY
                    if access is OutcomeAccess.DEVELOPMENT_VISIBLE
                    else VisibilityCeiling.PROSPECTIVE
                ),
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(f"selective-dependence-response-{binding.target_slug}-{step_id}",),
                barrier=barriers[operation],
                maximum_attempts=2,
                obligation_ids=(f"selective-dependence-response-{binding.target_slug}-{step_id}-contract",),
            )
        )
    stage_slug = stage.value.lower().replace("_", "-")
    return ProtocolTemplate(
        template_id=f"selective-dependence-response-{binding.target_slug}-{stage_slug}-protocol",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def build_target_source_canary_protocol(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    return _build_split_target_protocol(
        binding,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        registry=registry,
        config_by_step_id=config_by_step_id,
    )


def build_target_development_protocol(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    return _build_split_target_protocol(
        binding,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        registry=registry,
        config_by_step_id=config_by_step_id,
    )


def target_scientific_graph(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    raise ValueError(
        "combined source-canary/development graphs are prohibited; use the split builders"
    )
    by_step = {value.step_id: value for value in protocol.steps}
    if binding.method_completion is None:
        raise ValueError("target graph requires terminal method completion")
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
            "input.target-design",
            binding.design_id,
            binding.design,
            ScientificInputRole.MODEL,
            ("freeze-target-design",),
        ),
        (
            "input.construct-review",
            binding.construct_review.attestation_id,
            binding.construct_review,
            ScientificInputRole.PARENT_RECEIPT,
            ("bind-construct-review",),
        ),
        (
            "input.preparation-freeze",
            binding.preparation.freeze_id,
            binding.preparation,
            ScientificInputRole.DENOMINATOR,
            (
                "analyze-development",
                "generate-development-panel",
                "qualify-target-source",
            ),
        ),
        (
            "input.target-analysis-freeze",
            binding.analysis_freeze.freeze_id,
            binding.analysis_freeze,
            ScientificInputRole.DENOMINATOR,
            ("freeze-target-analysis",),
        ),
        (
            "input.method-question",
            binding.method_question.freeze_id,
            binding.method_question,
            ScientificInputRole.PARENT_RECEIPT,
            ("analyze-development",),
        ),
        (
            "input.contamination-ledger",
            binding.contamination_ledger.ledger_id,
            binding.contamination_ledger,
            ScientificInputRole.PARENT_RECEIPT,
            ("analyze-development",),
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
                    maximum_size_bytes=2 * 1024**2,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                )
                for input_id, artifact_id, record, role, _consumers in external_specs
            ),
            key=lambda value: value.input_id,
        )
    )
    roles = {
        ("bind-construct-review", "qualify-target-source"): ScientificInputRole.QUALIFICATION,
        ("freeze-target-design", "qualify-target-source"): ScientificInputRole.DENOMINATOR,
        ("bind-construct-review", "generate-development-panel"): ScientificInputRole.QUALIFICATION,
        ("freeze-target-design", "generate-development-panel"): ScientificInputRole.DENOMINATOR,
        (
            "freeze-target-analysis",
            "generate-development-panel",
        ): ScientificInputRole.DENOMINATOR,
        ("qualify-target-source", "generate-development-panel"): ScientificInputRole.SOURCE,
        ("bind-construct-review", "analyze-development"): ScientificInputRole.QUALIFICATION,
        ("freeze-target-design", "analyze-development"): ScientificInputRole.DENOMINATOR,
        ("freeze-target-analysis", "analyze-development"): ScientificInputRole.DENOMINATOR,
        ("generate-development-panel", "analyze-development"): ScientificInputRole.OUTCOME,
        ("qualify-target-source", "analyze-development"): ScientificInputRole.SOURCE,
    }
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
            maximum_size_bytes=2 * 1024**2,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=by_step[consumer_id].barrier,
        )
        for input_id, artifact_id, record, role, consumers in external_specs
        for consumer_id in consumers
    ]
    for (producer_id, consumer_id), role in roles.items():
        producer = by_step[producer_id]
        output = producer.outputs[0]
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{producer_id}.{consumer_id}",
                producer_node_id=producer_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=consumer_id,
                consumer_input_id=f"{producer_id}-record",
                scientific_role=role,
                logical_artifact_id=(f"artifact.selective-dependence-response.{binding.target_slug}.{producer_id}"),
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=producer.resource_budget.output_bytes,
                outcome_access=producer.requested_outcome_access,
                visibility_ceiling=producer.visibility_ceiling,
                barrier=(
                    BarrierKind.FREEZE
                    if producer.barrier is BarrierKind.FREEZE
                    else BarrierKind.NONE
                ),
            )
        )
    return CandidateScientificGraph(
        graph_id=f"graph.selective-dependence-response.{binding.target_slug}.development",
        external_inputs=external_inputs,
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _split_target_scientific_graph(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    stage: SelectiveDependenceResponseTargetStage,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    if binding.method_completion is None:
        raise ValueError("target graph requires terminal method completion")
    ExternalSpec = tuple[
        str,
        str,
        CanonicalRecord,
        ScientificInputRole,
        tuple[str, ...],
        OutcomeAccess,
        VisibilityCeiling,
    ]
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
    common: tuple[ExternalSpec, ...] = (
        (
            "input.target-design",
            binding.design_id,
            binding.design,
            ScientificInputRole.MODEL,
            ("freeze-target-design",),
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
        (
            "input.construct-review",
            binding.construct_review.attestation_id,
            binding.construct_review,
            ScientificInputRole.QUALIFICATION,
            ("bind-construct-review",),
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
        (
            "input.target-analysis-freeze",
            binding.analysis_freeze.freeze_id,
            binding.analysis_freeze,
            ScientificInputRole.MODEL,
            ("freeze-target-analysis",),
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
    )
    if stage is SelectiveDependenceResponseTargetStage.SOURCE_CANARY:
        external_specs: tuple[ExternalSpec, ...] = (
            *common,
            (
                "input.preparation-freeze",
                binding.preparation.freeze_id,
                binding.preparation,
                ScientificInputRole.PREPARED_MEDIUM,
                ("qualify-target-source",),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.method-question",
                binding.method_question.freeze_id,
                binding.method_question,
                ScientificInputRole.MODEL,
                ("qualify-target-source",),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.contamination-ledger",
                binding.contamination_ledger.ledger_id,
                binding.contamination_ledger,
                ScientificInputRole.QUALIFICATION,
                ("qualify-target-source",),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.method-completion",
                binding.method_completion.envelope_id,
                binding.method_completion,
                ScientificInputRole.PARENT_RECEIPT,
                ("qualify-target-source",),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
        )
        internal_roles = {
            ("bind-construct-review", "qualify-target-source"): (ScientificInputRole.QUALIFICATION),
            ("freeze-target-analysis", "qualify-target-source"): (ScientificInputRole.DENOMINATOR),
            ("freeze-target-design", "qualify-target-source"): (ScientificInputRole.DENOMINATOR),
        }
    else:
        if binding.source_qualification is None or binding.source_completion is None:
            raise ValueError("development graph requires terminal source completion")
        qualification_id = str(getattr(binding.source_qualification, "qualification_id"))
        external_specs = (
            *common,
            (
                "input.preparation-freeze",
                binding.preparation.freeze_id,
                binding.preparation,
                ScientificInputRole.PREPARED_MEDIUM,
                ("analyze-development", "generate-development-panel"),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.method-question",
                binding.method_question.freeze_id,
                binding.method_question,
                ScientificInputRole.MODEL,
                ("analyze-development",),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.contamination-ledger",
                binding.contamination_ledger.ledger_id,
                binding.contamination_ledger,
                ScientificInputRole.QUALIFICATION,
                ("analyze-development",),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.method-completion",
                binding.method_completion.envelope_id,
                binding.method_completion,
                ScientificInputRole.PARENT_RECEIPT,
                ("analyze-development", "generate-development-panel"),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            ),
            (
                "input.source-qualification",
                qualification_id,
                binding.source_qualification,
                ScientificInputRole.SOURCE,
                ("analyze-development", "generate-development-panel"),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
            ),
            (
                "input.source-completion",
                binding.source_completion.envelope_id,
                binding.source_completion,
                ScientificInputRole.PARENT_RECEIPT,
                ("analyze-development", "generate-development-panel"),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
            ),
        )
        internal_roles = {
            ("bind-construct-review", "generate-development-panel"): (
                ScientificInputRole.QUALIFICATION
            ),
            ("freeze-target-analysis", "generate-development-panel"): (
                ScientificInputRole.DENOMINATOR
            ),
            ("freeze-target-design", "generate-development-panel"): (
                ScientificInputRole.DENOMINATOR
            ),
            ("bind-construct-review", "analyze-development"): (ScientificInputRole.QUALIFICATION),
            ("freeze-target-analysis", "analyze-development"): (ScientificInputRole.DENOMINATOR),
            ("freeze-target-design", "analyze-development"): (ScientificInputRole.DENOMINATOR),
            ("generate-development-panel", "analyze-development"): (ScientificInputRole.OUTCOME),
        }
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
                    maximum_size_bytes=2 * 1024**2,
                    outcome_access=access,
                    visibility_ceiling=visibility,
                )
                for (
                    input_id,
                    artifact_id,
                    record,
                    role,
                    _consumers,
                    access,
                    visibility,
                ) in external_specs
            ),
            key=lambda value: value.input_id,
        )
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
            maximum_size_bytes=2 * 1024**2,
            outcome_access=access,
            visibility_ceiling=visibility,
            barrier=by_step[consumer_id].barrier,
        )
        for (
            input_id,
            artifact_id,
            record,
            role,
            consumers,
            access,
            visibility,
        ) in external_specs
        for consumer_id in consumers
    ]
    for (producer_id, consumer_id), role in internal_roles.items():
        producer = by_step[producer_id]
        output = producer.outputs[0]
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{producer_id}.{consumer_id}",
                producer_node_id=producer_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=consumer_id,
                consumer_input_id=f"{producer_id}-record",
                scientific_role=role,
                logical_artifact_id=(f"artifact.selective-dependence-response.{binding.target_slug}.{producer_id}"),
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=producer.resource_budget.output_bytes,
                outcome_access=producer.requested_outcome_access,
                visibility_ceiling=producer.visibility_ceiling,
                barrier=(
                    BarrierKind.FREEZE
                    if producer.barrier is BarrierKind.FREEZE
                    else BarrierKind.NONE
                ),
            )
        )
    stage_slug = stage.value.lower().replace("_", "-")
    return CandidateScientificGraph(
        graph_id=f"graph.selective-dependence-response.{binding.target_slug}.{stage_slug}",
        external_inputs=external_inputs,
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def target_source_canary_scientific_graph(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    return _split_target_scientific_graph(
        binding,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        protocol=protocol,
        registry=registry,
    )


def target_development_scientific_graph(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    return _split_target_scientific_graph(
        binding,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        protocol=protocol,
        registry=registry,
    )


def target_candidate_catalog(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateCapabilityCatalog:
    raise ValueError(
        "combined source-canary/development catalogs are prohibited; use the split builders"
    )
    graph = target_scientific_graph(binding, protocol=protocol, registry=registry)
    incoming = {
        node.node_id: tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id
        )
        for node in graph.nodes
    }
    coverage = ObligationCoverage(
        coverage_id=f"coverage.selective-dependence-response.{binding.target_slug}.development",
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
    implementations = {value.implementation_sha256 for value in registry.capabilities}
    if len(implementations) != 1:
        raise ValueError("target implementation identities differ")
    return CandidateCapabilityCatalog(
        catalog_id=f"selective-dependence-response-{binding.target_slug}-candidate-catalog",
        registrations=target_candidate_registrations(binding, registry=registry),
        templates=(
            StudyTemplate(
                template_key=f"selective-dependence-response.{binding.target_slug}.development",
                template_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
                protocol=protocol,
                graph=graph,
                coverage=coverage,
            ),
        ),
    )


def _split_target_candidate_catalog(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    stage: SelectiveDependenceResponseTargetStage,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateCapabilityCatalog:
    graph = _split_target_scientific_graph(
        binding,
        stage=stage,
        protocol=protocol,
        registry=registry,
    )
    incoming = {
        node.node_id: tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id
        )
        for node in graph.nodes
    }
    stage_slug = stage.value.lower().replace("_", "-")
    coverage = ObligationCoverage(
        coverage_id=f"coverage.selective-dependence-response.{binding.target_slug}.{stage_slug}",
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
    return CandidateCapabilityCatalog(
        catalog_id=f"selective-dependence-response-{binding.target_slug}-{stage_slug}-candidate-catalog",
        registrations=target_candidate_registrations(binding, registry=registry),
        templates=(
            StudyTemplate(
                template_key=f"selective-dependence-response.{binding.target_slug}.{stage.value.lower()}",
                template_version=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION,
                protocol=protocol,
                graph=graph,
                coverage=coverage,
            ),
        ),
    )


def target_source_canary_candidate_catalog(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateCapabilityCatalog:
    return _split_target_candidate_catalog(
        binding,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        protocol=protocol,
        registry=registry,
    )


def target_development_candidate_catalog(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateCapabilityCatalog:
    return _split_target_candidate_catalog(
        binding,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        protocol=protocol,
        registry=registry,
    )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetConfigAdapter:
    provider: str

    @property
    def provider_key(self) -> str:
        return self.provider

    @property
    def provider_version(self) -> str:
        return SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != SelectiveDependenceResponseTargetRuntimeConfig.SCHEMA:
            raise ValueError("target config adapter schema differs")
        decode_target_runtime_config(payload)


def target_config_decoders(
    binding: SelectiveDependenceResponseTargetBinding, catalog: CandidateCapabilityCatalog
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    providers = {
        f"{value.provider_key}@{value.provider_version}" for value in catalog.registrations
    }
    if providers != {f"{binding.provider_key}@{SELECTIVE_DEPENDENCE_RESPONSE_TARGET_PROTOCOL_VERSION}"}:
        raise ValueError("target candidate provider roster differs")
    return (SelectiveDependenceResponseTargetConfigAdapter(binding.provider_key),)


def _dependency_by_schema(context: TaskContext, schema: str) -> bytes:
    matches = tuple(
        value
        for value in context.input_ports
        if value.kind is WorkerInputKind.DEPENDENCY and value.payload_schema == schema
    )
    if len(matches) != 1:
        raise ValueError(f"target runner requires one dependency with schema {schema}")
    return matches[0].read(256 * 1024**2)


def _external_by_schema(context: TaskContext, schema: str) -> bytes:
    matches = tuple(
        value
        for value in context.input_ports
        if value.kind is WorkerInputKind.EXTERNAL and value.payload_schema == schema
    )
    if len(matches) != 1:
        raise ValueError(f"target runner requires one external input with schema {schema}")
    return matches[0].read(256 * 1024**2)


def _input_by_schema(context: TaskContext, schema: str) -> bytes:
    matches = tuple(value for value in context.input_ports if value.payload_schema == schema)
    if len(matches) != 1:
        raise ValueError(f"target runner requires one input with schema {schema}")
    return matches[0].read(256 * 1024**2)


class _RecordRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        record: CanonicalRecord,
        record_type: type[CanonicalRecord],
    ) -> None:
        self.manifest = manifest
        self.record = record
        self.record_type = record_type
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if len(context.output_ports) != 1:
            raise ValueError("target record runner output roster differs")
        record = decode_canonical_bytes(
            _external_by_schema(context, self.record.SCHEMA),
            self.record_type,
            maximum_bytes=2 * 1024**2,
        )
        if record != self.record:
            raise ValueError("target frozen record differs from its exact graph input")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("selective-dependence-response-static-target-record", True, ()),),
        )


class _SourceRunner:
    def __init__(self, manifest: CapabilityManifest, binding: SelectiveDependenceResponseTargetBinding) -> None:
        self.manifest = manifest
        self.binding = binding
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        design = decode_canonical_bytes(
            _dependency_by_schema(context, self.binding.design.SCHEMA),
            self.binding.design_type,
            maximum_bytes=2 * 1024**2,
        )
        review = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseConstructReviewAttestation.SCHEMA),
            SelectiveDependenceResponseConstructReviewAttestation,
            maximum_bytes=2 * 1024**2,
        )
        analysis_freeze = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA),
            SelectiveDependenceResponseTargetAnalysisFreeze,
            maximum_bytes=2 * 1024**2,
        )
        preparation = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponsePreparationDistributionFreeze.SCHEMA),
            SelectiveDependenceResponsePreparationDistributionFreeze,
            maximum_bytes=2 * 1024**2,
        )
        question = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA),
            SelectiveDependenceResponseMethodQuestionFreeze,
            maximum_bytes=2 * 1024**2,
        )
        contamination = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseContaminationLedger.SCHEMA),
            SelectiveDependenceResponseContaminationLedger,
            maximum_bytes=2 * 1024**2,
        )
        method_completion = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA),
            SelectiveDependenceResponseMethodCompletionEnvelope,
            maximum_bytes=2 * 1024**2,
        )
        if (
            design != self.binding.design
            or review != self.binding.construct_review
            or preparation != self.binding.preparation
            or analysis_freeze != self.binding.analysis_freeze
            or question != self.binding.method_question
            or contamination != self.binding.contamination_ledger
            or method_completion != self.binding.method_completion
        ):
            raise ValueError("target source dependencies differ from the static binding")
        result, canary_panel = self.binding.qualify_source(design)
        source_ready = bool(getattr(result, "source_ready", False))
        result_panel = getattr(result, "canary_panel", None)
        expected_result_identities = tuple(
            ObjectIdentity.from_record(value.result_id, value)
            for value in canary_panel.complete_units
        )
        if (
            not isinstance(result, self.binding.source_qualification_type)
            or canary_panel.target_id != self.binding.target_id
            or canary_panel.phase is not SelectiveDependenceResponsePhase.CANARY
            or canary_panel.expected_complete_unit_ids != self.binding.preparation.canary_unit_ids
            or result_panel != ObjectIdentity.from_record(canary_panel.panel_id, canary_panel)
            or getattr(result, "canary_result_identities", None) != expected_result_identities
        ):
            raise ValueError("target source qualification evidence does not close")
        payloads = {
            self.binding.source_qualification_type.SCHEMA: result.canonical_bytes(),
            SelectiveDependenceResponseTargetPanel.SCHEMA: canary_panel.canonical_bytes(),
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
                ReceiptCheck(
                    "selective-dependence-response-source-qualified",
                    source_ready,
                    () if source_ready else ("source-unavailable",),
                ),
            ),
        )


class _DevelopmentRunner:
    def __init__(self, manifest: CapabilityManifest, binding: SelectiveDependenceResponseTargetBinding) -> None:
        self.manifest = manifest
        self.binding = binding
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        design = decode_canonical_bytes(
            _dependency_by_schema(context, self.binding.design.SCHEMA),
            self.binding.design_type,
            maximum_bytes=2 * 1024**2,
        )
        review = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseConstructReviewAttestation.SCHEMA),
            SelectiveDependenceResponseConstructReviewAttestation,
            maximum_bytes=2 * 1024**2,
        )
        qualification = decode_canonical_bytes(
            _input_by_schema(context, self.binding.source_qualification_type.SCHEMA),
            self.binding.source_qualification_type,
            maximum_bytes=2 * 1024**2,
        )
        source_completion = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseSourceCanaryCompletionEnvelope.SCHEMA),
            SelectiveDependenceResponseSourceCanaryCompletionEnvelope,
            maximum_bytes=2 * 1024**2,
        )
        method_completion = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA),
            SelectiveDependenceResponseMethodCompletionEnvelope,
            maximum_bytes=2 * 1024**2,
        )
        analysis_freeze = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA),
            SelectiveDependenceResponseTargetAnalysisFreeze,
            maximum_bytes=2 * 1024**2,
        )
        preparation = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponsePreparationDistributionFreeze.SCHEMA),
            SelectiveDependenceResponsePreparationDistributionFreeze,
            maximum_bytes=2 * 1024**2,
        )
        if (
            design != self.binding.design
            or review != self.binding.construct_review
            or preparation != self.binding.preparation
            or analysis_freeze != self.binding.analysis_freeze
            or qualification != self.binding.source_qualification
            or source_completion != self.binding.source_completion
            or method_completion != self.binding.method_completion
            or source_completion.source_qualification
            != ObjectIdentity.from_record(
                str(getattr(qualification, "qualification_id")), qualification
            )
        ):
            raise ValueError("development dependencies differ from the static binding")
        if not getattr(qualification, "source_ready", False):
            raise ValueError("source-unqualified")
        units = tuple(
            self.binding.execute_unit(design, unit_id, SelectiveDependenceResponsePhase.DEVELOPMENT)
            for unit_id in preparation.development_unit_ids
        )
        panel = build_panel(
            panel_id=f"panel.{self.binding.target_slug}.development",
            target_id=self.binding.target_id,
            complete_units=units,
        )
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=panel.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("selective-dependence-response-development-full-fan-in", True, ()),),
        )


class _AnalysisRunner:
    def __init__(self, manifest: CapabilityManifest, binding: SelectiveDependenceResponseTargetBinding) -> None:
        self.manifest = manifest
        self.binding = binding
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        panel = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseTargetPanel.SCHEMA),
            SelectiveDependenceResponseTargetPanel,
            maximum_bytes=256 * 1024**2,
        )
        design = decode_canonical_bytes(
            _dependency_by_schema(context, self.binding.design.SCHEMA),
            self.binding.design_type,
            maximum_bytes=2 * 1024**2,
        )
        review = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseConstructReviewAttestation.SCHEMA),
            SelectiveDependenceResponseConstructReviewAttestation,
            maximum_bytes=2 * 1024**2,
        )
        qualification = decode_canonical_bytes(
            _input_by_schema(context, self.binding.source_qualification_type.SCHEMA),
            self.binding.source_qualification_type,
            maximum_bytes=2 * 1024**2,
        )
        source_completion = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseSourceCanaryCompletionEnvelope.SCHEMA),
            SelectiveDependenceResponseSourceCanaryCompletionEnvelope,
            maximum_bytes=2 * 1024**2,
        )
        method_completion = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA),
            SelectiveDependenceResponseMethodCompletionEnvelope,
            maximum_bytes=2 * 1024**2,
        )
        analysis_freeze = decode_canonical_bytes(
            _dependency_by_schema(context, SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA),
            SelectiveDependenceResponseTargetAnalysisFreeze,
            maximum_bytes=2 * 1024**2,
        )
        preparation = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponsePreparationDistributionFreeze.SCHEMA),
            SelectiveDependenceResponsePreparationDistributionFreeze,
            maximum_bytes=2 * 1024**2,
        )
        question = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA),
            SelectiveDependenceResponseMethodQuestionFreeze,
            maximum_bytes=2 * 1024**2,
        )
        contamination = decode_canonical_bytes(
            _external_by_schema(context, SelectiveDependenceResponseContaminationLedger.SCHEMA),
            SelectiveDependenceResponseContaminationLedger,
            maximum_bytes=2 * 1024**2,
        )
        if (
            design != self.binding.design
            or review != self.binding.construct_review
            or preparation != self.binding.preparation
            or question != self.binding.method_question
            or contamination != self.binding.contamination_ledger
            or analysis_freeze != self.binding.analysis_freeze
            or qualification != self.binding.source_qualification
            or source_completion != self.binding.source_completion
            or method_completion != self.binding.method_completion
        ):
            raise ValueError("analysis dependencies differ from the static binding")
        if not getattr(qualification, "source_ready", False):
            raise ValueError("analysis source is unqualified")
        qualification_id = str(getattr(qualification, "qualification_id"))
        lineage = SelectiveDependenceResponseDevelopmentLineage(
            lineage_id=f"lineage.{self.binding.target_slug}.development",
            target_id=self.binding.target_id,
            source_qualification=ObjectIdentity.from_record(qualification_id, qualification),
            source_completion=ObjectIdentity.from_record(
                source_completion.envelope_id,
                source_completion,
            ),
            method_completion=ObjectIdentity.from_record(
                method_completion.envelope_id,
                method_completion,
            ),
            implementation_sha256=self.manifest.implementation_sha256,
            dependency_receipt_ids=context.dependency_receipt_ids,
            dependency_materialization_ids=context.dependency_input_materialization_ids,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        result = self.binding.analyze_development(panel, lineage)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck(
                    "selective-dependence-response-development-terminal",
                    result.evaluation_eligible or bool(result.parent.stop_codes),
                    result.parent.stop_codes,
                ),
            ),
        )


class SelectiveDependenceResponseTargetRuntimeProvider(CampaignRuntimeProvider):
    def __init__(self, *, registry: CapabilityRegistry, binding: SelectiveDependenceResponseTargetBinding) -> None:
        operations = tuple(
            sorted(
                (
                    SelectiveDependenceResponseTargetOperation(
                        value.capability_key.rsplit(".", 1)[-1].replace("-", "_").upper()
                    )
                    for value in registry.capabilities
                ),
                key=lambda value: value.value,
            )
        )
        operation_set = set(operations)
        if operation_set == set(SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS):
            stage: SelectiveDependenceResponseTargetStage | None = SelectiveDependenceResponseTargetStage.SOURCE_CANARY
        elif operation_set == set(SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS):
            stage = SelectiveDependenceResponseTargetStage.DEVELOPMENT
        elif operation_set == set(SelectiveDependenceResponseTargetOperation):
            raise ValueError("combined target providers are prohibited")
        else:
            raise ValueError("target provider operation stage differs")
        expected = build_target_registry(
            binding,
            implementation_sha256=registry.capabilities[0].implementation_sha256,
            operations=operations,
            stage=stage,
        )
        if registry != expected:
            raise ValueError("target provider registry differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.binding = binding
        runners: list[TaskRunner] = []
        for manifest in registry.capabilities:
            operation_text = manifest.capability_key.rsplit(".", 1)[-1].replace("-", "_").upper()
            operation = SelectiveDependenceResponseTargetOperation(operation_text)
            if operation is SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN:
                runner: TaskRunner = _RecordRunner(manifest, binding.design, binding.design_type)
            elif operation is SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS:
                runner = _RecordRunner(
                    manifest,
                    binding.analysis_freeze,
                    SelectiveDependenceResponseTargetAnalysisFreeze,
                )
            elif operation is SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW:
                runner = _RecordRunner(
                    manifest,
                    binding.construct_review,
                    SelectiveDependenceResponseConstructReviewAttestation,
                )
            elif operation is SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE:
                runner = _SourceRunner(manifest, binding)
            elif operation is SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT:
                runner = _DevelopmentRunner(manifest, binding)
            else:
                runner = _AnalysisRunner(manifest, binding)
            runners.append(runner)
        self._runners = tuple(runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("target provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("target provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            self.binding.design_id: self.binding.design,
            self.binding.construct_review.attestation_id: self.binding.construct_review,
            self.binding.preparation.freeze_id: self.binding.preparation,
            self.binding.analysis_freeze.freeze_id: self.binding.analysis_freeze,
            self.binding.method_question.freeze_id: self.binding.method_question,
            self.binding.contamination_ledger.ledger_id: self.binding.contamination_ledger,
        }
        if self.binding.method_completion is not None:
            records[self.binding.method_completion.envelope_id] = self.binding.method_completion
        if self.binding.source_qualification is not None:
            records[str(getattr(self.binding.source_qualification, "qualification_id"))] = (
                self.binding.source_qualification
            )
        if self.binding.source_completion is not None:
            records[self.binding.source_completion.envelope_id] = self.binding.source_completion
        records = {
            artifact_id: record for artifact_id, record in records.items() if artifact_id in specs
        }
        configs = {
            value.capability_key: target_runtime_config(
                self.binding,
                SelectiveDependenceResponseTargetOperation(
                    value.capability_key.rsplit(".", 1)[-1].replace("-", "_").upper()
                ),
            )
            for value in self.registry.capabilities
        }
        for task in plan.tasks:
            records[task.capability.config.artifact_id] = configs[task.capability.capability_key]
        if set(records) != set(specs):
            raise ValueError("target external input roster differs")

        def record_id(record: CanonicalRecord) -> str:
            if isinstance(record, SelectiveDependenceResponseTargetRuntimeConfig):
                return record.config_id
            if isinstance(record, SelectiveDependenceResponsePreparationDistributionFreeze):
                return record.freeze_id
            if isinstance(record, SelectiveDependenceResponseTargetAnalysisFreeze):
                return record.freeze_id
            if isinstance(record, SelectiveDependenceResponseMethodQuestionFreeze):
                return record.freeze_id
            if isinstance(record, SelectiveDependenceResponseContaminationLedger):
                return record.ledger_id
            if isinstance(record, SelectiveDependenceResponseConstructReviewAttestation):
                return record.attestation_id
            if isinstance(record, SelectiveDependenceResponseMethodCompletionEnvelope):
                return record.envelope_id
            if isinstance(record, SelectiveDependenceResponseSourceCanaryCompletionEnvelope):
                return record.envelope_id
            if isinstance(record, self.binding.source_qualification_type):
                return str(getattr(record, "qualification_id"))
            if isinstance(record, self.binding.design_type):
                return self.binding.design_id
            raise TypeError("target external input type differs")

        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id(record), record),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=(spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND),
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                    ),
                    outcome_access=(spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND),
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
            raise ValueError("target provider semantic registry differs")
        types: dict[str, type[CanonicalRecord]] = {
            self.binding.design.SCHEMA: self.binding.design_type,
            SelectiveDependenceResponseTargetAnalysisFreeze.SCHEMA: SelectiveDependenceResponseTargetAnalysisFreeze,
            SelectiveDependenceResponseConstructReviewAttestation.SCHEMA: SelectiveDependenceResponseConstructReviewAttestation,
            self.binding.source_qualification_type.SCHEMA: (self.binding.source_qualification_type),
            SelectiveDependenceResponseTargetPanel.SCHEMA: SelectiveDependenceResponseTargetPanel,
            SelectiveDependenceResponseDevelopmentBundle.SCHEMA: SelectiveDependenceResponseDevelopmentBundle,
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
            raise ValueError("target provider adjudication registry differs")
        return None


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS",
    "SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS",
    'SelectiveDependenceResponseTargetBinding',
    'SelectiveDependenceResponseTargetOperation',
    'SelectiveDependenceResponseTargetRuntimeConfig',
    'SelectiveDependenceResponseTargetRuntimeProvider',
    'SelectiveDependenceResponseTargetStage',
    "build_target_development_protocol",
    "build_target_registry",
    "build_target_source_canary_protocol",
    "decode_target_runtime_config",
    "target_config_decoders",
    "target_development_candidate_catalog",
    "target_development_scientific_graph",
    "target_runtime_config",
    "target_source_canary_candidate_catalog",
    "target_source_canary_scientific_graph",
]
