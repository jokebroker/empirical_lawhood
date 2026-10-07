"""Closed Prospective reference, controller-use and recovery public contract composition used by the Prospective reference, controller-use and recovery release route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
from typing import Any, cast

from empirical_lawhood.adapters.composition.acquisition_recurrence_margin_contracts import ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS, ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES, AcquisitionRecurrenceMarginContractRegistry, build_acquisition_recurrence_margin_contract_registry
from empirical_lawhood.adapters.control.study_bridge import BoundActionAwareControllerRevealReceipt
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.planning.finite_chart_reference import FiniteChartReferenceDesign, ReferenceComparisonDimensionSpec
from empirical_lawhood.planning.nested_controller_evaluation import NestedEvaluationCellLocator, NestedPreparationOccurrence, PreparationMedianReducerRegistration, ActionAwareControllerEvaluationPlan, NestedTaskUnitSpec, ProspectiveEvaluationPrecommitment
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.controller_evaluation_nested import ActionAwareControllerCohortAdjudication, ActionAwareControllerUnitEvaluation, PreparationMedianMemberEffect, SealedActionAwareControllerBundle, ProspectiveEvaluationBindingReceipt, ProspectiveExecutionEventPrefix, ProspectiveExecutionEvent, RevealedReferenceClassMembership, RevealedNestedPhysicalCell, RevealedActionAwareControllerBundle, SealedNestedPhysicalCell
from empirical_lawhood.runtime.candidate_compiler import ExecutableStudyCompilationReport
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ChildResourceTokenState, ExecutionResourceCellState, ExecutionResourceEnvelopeEvent, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, JitCellSignatureProjection, JitExpressionFieldValue, JitGraphSignatureManifest, JitGraphSignatureObservation, NonTimeResourceBudget, PredevelopmentJitSignatureCensus, ProgressHeartbeat, ProgressLivenessContract, RosterCapacityBranchSpec, RosterCapacityDecision, RunExecutionResourceEnvelope
from empirical_lawhood.runtime.finite_chart_reference import ReferenceActionAssessment, ReferenceCellAssessment, ReferenceClassMembershipReceipt, ReferenceDispositionClass, ReferenceDispositionMember, ReferenceGateAssessment
from empirical_lawhood.runtime.synthetic_controller_comparison_conformance import SyntheticControllerCoordinateConformanceReceipt, SyntheticControllerRouteConformanceReceipt, SyntheticControllerComparisonTaskSpec
from empirical_lawhood.runtime.multi_world_readiness_contracts import ReadinessProofOwnerBinding, MultiWorldReadinessProofOwner
from empirical_lawhood.runtime.plans import ExecutionPlan, RunPlan
from empirical_lawhood.runtime.study_extensions import StudyExtensionMaterializationReceipt
from empirical_lawhood.runtime.study_issue import ExtensionPublicationReceipt, StudyExtensionDecoderRegistration, IssuedExecutableStudyManifest
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract
from empirical_lawhood.runtime.recovery import RunRecoveryIndex, RunRecoveryTerminalEvent, CandidateTaskRecoveryEvent
from empirical_lawhood.runtime.response_law_release import MultiWorldReadinessConsumerPin, MultiWorldReadinessConformanceEvidence, MultiWorldReadinessReleaseAttestation, MultiWorldReadinessReleaseGenerator, MultiWorldReadinessRelease
from empirical_lawhood.runtime.scientific_graph_preservation import ResourceGraphPreservationReceipt
from empirical_lawhood.runtime.static_codecs import (
    CanonicalRecordCodecRegistry,
    build_canonical_record_codec_registry,
)


REFERENCE_CLASS_CAPABILITY_KEY = "reference.finite-chart-class"
REFERENCE_MEMBERSHIP_CAPABILITY_KEY = "reference.commitment-membership"
PROSPECTIVE_BINDING_CAPABILITY_KEY = "controller.prospective-binding"
NESTED_CONTROLLER_USE_CAPABILITY_KEY = "controller.action-aware-nested-controller-use"
DEADLINE_FREE_RECOVERY_CAPABILITY_KEY = "runtime.deadline-free-resource-recovery"

PROSPECTIVE_EXECUTION_ADDITIVE_CAPABILITY_KEYS = tuple(
    sorted(
        (
            REFERENCE_CLASS_CAPABILITY_KEY,
            REFERENCE_MEMBERSHIP_CAPABILITY_KEY,
            PROSPECTIVE_BINDING_CAPABILITY_KEY,
            NESTED_CONTROLLER_USE_CAPABILITY_KEY,
            DEADLINE_FREE_RECOVERY_CAPABILITY_KEY,
        )
    )
)
PROSPECTIVE_EXECUTION_CAPABILITY_KEYS = tuple(sorted((*ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS, *PROSPECTIVE_EXECUTION_ADDITIVE_CAPABILITY_KEYS)))

PROSPECTIVE_EXECUTION_ADDITIVE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    ChildResourceTokenLimit,
    ChildResourceTokenState,
    BoundActionAwareControllerRevealReceipt,
    ActionAwareControllerCohortAdjudication,
    ActionAwareControllerUnitEvaluation,
    ExecutionPlan,
    ExecutionResourceCellState,
    ExecutionResourceEnvelopeEvent,
    ExecutionResourceEnvelopeSpec,
    ExecutionResourceTaskCellSpec,
    MultiWorldReadinessConsumerPin,
    MultiWorldReadinessConformanceEvidence,
    ExtensionPublicationReceipt,
    FiniteChartReferenceDesign,
    SyntheticControllerCoordinateConformanceReceipt,
    SyntheticControllerRouteConformanceReceipt,
    SyntheticControllerComparisonTaskSpec,
    ReadinessProofOwnerBinding,
    MultiWorldReadinessProofOwner,
    JitCellSignatureProjection,
    JitExpressionFieldValue,
    JitGraphSignatureManifest,
    JitGraphSignatureObservation,
    NestedEvaluationCellLocator,
    PreparationMedianMemberEffect,
    NestedPreparationOccurrence,
    PreparationMedianReducerRegistration,
    ActionAwareControllerEvaluationPlan,
    NestedTaskUnitSpec,
    NonTimeResourceBudget,
    PredevelopmentJitSignatureCensus,
    StudyExtensionDecoderRegistration,
    StudyExtensionMaterializationReceipt,
    ProgressHeartbeat,
    ProgressLivenessContract,
    SealedActionAwareControllerBundle,
    ProspectiveEvaluationBindingReceipt,
    ProspectiveEvaluationPrecommitment,
    ProspectiveExecutionEventPrefix,
    ProspectiveExecutionEvent,
    ReferenceActionAssessment,
    ReferenceCellAssessment,
    ReferenceClassMembershipReceipt,
    ReferenceComparisonDimensionSpec,
    ReferenceDispositionClass,
    ReferenceDispositionMember,
    ReferenceGateAssessment,
    MultiWorldReadinessReleaseAttestation,
    MultiWorldReadinessReleaseGenerator,
    MultiWorldReadinessRelease,
    RevealedReferenceClassMembership,
    RevealedNestedPhysicalCell,
    RevealedActionAwareControllerBundle,
    RosterCapacityBranchSpec,
    RosterCapacityDecision,
    RunExecutionResourceEnvelope,
    RunPlan,
    RunRecoveryIndex,
    RunRecoveryTerminalEvent,
    ResourceGraphPreservationReceipt,
    SealedNestedPhysicalCell,
    IssuedExecutableStudyManifest,
    ExecutableStudyCompilationReport,
    CandidateTaskRecoveryEvent,
)
PROSPECTIVE_EXECUTION_RECORD_TYPES = tuple(
    {value.SCHEMA: value for value in (*ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES, *PROSPECTIVE_EXECUTION_ADDITIVE_RECORD_TYPES)}.values()
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=4,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=8 * 1024**3,
        output_bytes=512 * 1024**2,
    )


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    evidence: EvidenceCeiling,
    access: OutcomeAccess,
    permissions: tuple[CapabilityPermission, ...],
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version="1.0.0",
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=_digest(config_schema),
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions, key=lambda value: value.value)),
        maximum_evidence_ceiling=evidence,
        maximum_outcome_access=access,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-prospective-execution",
        requires_clean_commit=True,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def prospective_execution_additive_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    if set(implementation_sha256_by_key) != set(PROSPECTIVE_EXECUTION_ADDITIVE_CAPABILITY_KEYS):
        raise ValueError("Prospective reference, controller-use and recovery implementation identities must cover the exact additive family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = (
        (
            REFERENCE_CLASS_CAPABILITY_KEY,
            CapabilityKind.EVALUATOR,
            FiniteChartReferenceDesign.SCHEMA,
            (FiniteChartReferenceDesign.SCHEMA, ReferenceCellAssessment.SCHEMA),
            (ReferenceActionAssessment.SCHEMA, ReferenceDispositionClass.SCHEMA),
            EvidenceCeiling.CONTROLLER_USE,
            OutcomeAccess.EVALUATOR_REVEAL,
            (CapabilityPermission.READ_SEALED_OUTCOMES, CapabilityPermission.REVEAL_OUTCOMES),
            ("prospective-execution-complete-finite-product", "prospective-execution-set-valued-equivalence", "prospective-execution-member-seed-unit"),
        ),
        (
            REFERENCE_MEMBERSHIP_CAPABILITY_KEY,
            CapabilityKind.EVALUATOR,
            ReferenceDispositionClass.SCHEMA,
            (ReferenceDispositionClass.SCHEMA,),
            (ReferenceClassMembershipReceipt.SCHEMA,),
            EvidenceCeiling.CONTROLLER_USE,
            OutcomeAccess.EVALUATOR_REVEAL,
            (),
            ("prospective-execution-exact-action-membership", "prospective-execution-hold-nonattempt-distinct"),
        ),
        (
            PROSPECTIVE_BINDING_CAPABILITY_KEY,
            CapabilityKind.CONTROLLER_SYNTHESIZER,
            ProspectiveEvaluationPrecommitment.SCHEMA,
            (
                ProspectiveEvaluationPrecommitment.SCHEMA,
                ProspectiveExecutionEventPrefix.SCHEMA,
            ),
            (ProspectiveEvaluationBindingReceipt.SCHEMA, ProspectiveExecutionEvent.SCHEMA),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.OUTCOME_BLIND,
            (),
            ("prospective-execution-atomic-empty-prefix", "prospective-execution-admission-only-compiled-product", "prospective-execution-no-late-binding"),
        ),
        (
            NESTED_CONTROLLER_USE_CAPABILITY_KEY,
            CapabilityKind.EVALUATOR,
            ActionAwareControllerEvaluationPlan.SCHEMA,
            (
                SealedActionAwareControllerBundle.SCHEMA,
                RevealedActionAwareControllerBundle.SCHEMA,
            ),
            (ActionAwareControllerUnitEvaluation.SCHEMA, ActionAwareControllerCohortAdjudication.SCHEMA),
            EvidenceCeiling.CONTROLLER_USE,
            OutcomeAccess.EVALUATOR_REVEAL,
            (CapabilityPermission.READ_SEALED_OUTCOMES, CapabilityPermission.REVEAL_OUTCOMES),
            ("prospective-execution-action-aware-physical-union", "prospective-execution-median-then-member-min", "prospective-execution-itt-task-unit"),
        ),
        (
            DEADLINE_FREE_RECOVERY_CAPABILITY_KEY,
            CapabilityKind.REPORTER,
            ExecutionResourceEnvelopeSpec.SCHEMA,
            (ExecutionPlan.SCHEMA, ExecutionResourceEnvelopeSpec.SCHEMA),
            (RunExecutionResourceEnvelope.SCHEMA, RunRecoveryTerminalEvent.SCHEMA),
            EvidenceCeiling.MEASUREMENT,
            OutcomeAccess.OUTCOME_BLIND,
            (),
            ("prospective-execution-no-deadline-field", "prospective-execution-progress-liveness", "prospective-execution-first-valid-recovery"),
        ),
    )
    return tuple(
        sorted(
            (
                _manifest(
                    key=key,
                    kind=kind,
                    config_schema=config_schema,
                    inputs=inputs,
                    outputs=outputs,
                    evidence=evidence,
                    access=access,
                    permissions=permissions,
                    checks=checks,
                    implementation_sha256=implementation_sha256_by_key[key],
                )
                for key, kind, config_schema, inputs, outputs, evidence, access, permissions, checks in specifications
            ),
            key=lambda value: value.registry_id,
        )
    )


@dataclass(frozen=True, slots=True)
class ProspectiveExecutionContractRegistry:
    capability_registry: CapabilityRegistry
    proof_owners: tuple[MultiWorldReadinessProofOwner, ...]
    proof_owner_bindings: tuple[ReadinessProofOwnerBinding, ...]
    codec_registry: CanonicalRecordCodecRegistry
    semantic_output_contracts: tuple[CapabilityOutputSemanticContract, ...]
    acquisition_recurrence_margin: AcquisitionRecurrenceMarginContractRegistry

    def __post_init__(self) -> None:
        manifests = {value.capability_key: value for value in self.capability_registry.capabilities}
        if set(manifests) != set(PROSPECTIVE_EXECUTION_CAPABILITY_KEYS):
            raise ValueError("Prospective reference, controller-use and recovery registry changes the exact Prospective reference, controller-use and recovery capability family")
        owners = {value.obligation_id: value for value in self.proof_owners}
        if len(owners) != len(self.proof_owners):
            raise ValueError("Prospective reference, controller-use and recovery contract obligations have multiple proof owners")
        owner_ids = {
            ObjectIdentity.from_record(value.owner_id, value): value for value in self.proof_owners
        }
        for binding in self.proof_owner_bindings:
            owner = owner_ids.get(binding.proof_owner)
            manifest = manifests.get(binding.capability_manifest.object_id)
            if (
                owner is None
                or owner.obligation_id != binding.obligation_id
                or manifest is None
                or ObjectIdentity.from_record(manifest.capability_key, manifest)
                != binding.capability_manifest
                or binding.output_schema not in manifest.output_schema_ids
                or binding.maximum_evidence_ceiling is not manifest.maximum_evidence_ceiling
                or binding.maximum_outcome_access is not manifest.maximum_outcome_access
            ):
                raise ValueError("Prospective reference, controller-use and recovery proof binding differs from owner/capability/output semantics")
        expected_semantics = {
            (manifest.capability_key, schema)
            for manifest in manifests.values()
            for schema in manifest.output_schema_ids
        }
        if {
            (value.capability_key, value.payload_schema) for value in self.semantic_output_contracts
        } != expected_semantics:
            raise ValueError("Prospective reference, controller-use and recovery semantic output contracts are incomplete")
        if not {schema for _, schema in expected_semantics} <= set(
            self.codec_registry.record_types
        ):
            raise ValueError("Prospective reference, controller-use and recovery output semantics lack closed static codecs")


def build_prospective_execution_contract_registry(
    *,
    implementation_sha256_by_key: Mapping[str, str],
    decoder_implementation_sha256: str,
) -> ProspectiveExecutionContractRegistry:
    if set(implementation_sha256_by_key) != set(PROSPECTIVE_EXECUTION_CAPABILITY_KEYS):
        raise ValueError("Prospective reference, controller-use and recovery implementation identities must cover the exact Prospective reference, controller-use and recovery family")
    acquisition_recurrence_margin = build_acquisition_recurrence_margin_contract_registry(
        implementation_sha256_by_key={
            key: implementation_sha256_by_key[key] for key in ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS
        },
        decoder_implementation_sha256=decoder_implementation_sha256,
    )
    additive = prospective_execution_additive_capability_manifests(
        {key: implementation_sha256_by_key[key] for key in PROSPECTIVE_EXECUTION_ADDITIVE_CAPABILITY_KEYS}
    )
    manifests = tuple(
        sorted(
            (*acquisition_recurrence_margin.capability_registry.capabilities, *additive), key=lambda value: value.registry_id
        )
    )
    capability_registry = CapabilityRegistry(
        registry_id="prospective-execution-capabilities",
        capabilities=manifests,
    )
    obligation_by_key = {
        REFERENCE_CLASS_CAPABILITY_KEY: "law-qualification.set-valued-finite-chart-reference",
        REFERENCE_MEMBERSHIP_CAPABILITY_KEY: "law-qualification.set-valued-finite-chart-reference",
        PROSPECTIVE_BINDING_CAPABILITY_KEY: "prospective.evaluation-prefix-binding",
        NESTED_CONTROLLER_USE_CAPABILITY_KEY: "controller-use.action-aware-nested-evaluation",
        DEADLINE_FREE_RECOVERY_CAPABILITY_KEY: "runtime.deadline-free-resource-recovery",
    }
    owner_specs = {
        obligation: (
            key,
            {
                "law-qualification.set-valued-finite-chart-reference": (
                    "empirical_lawhood.runtime.finite_chart_reference",
                    'evaluate_finite_reference_class',
                    "Evaluate the complete action/member/seed product and preserve every equivalent disposition.",
                ),
                "prospective.evaluation-prefix-binding": (
                    "empirical_lawhood.runtime.controller_evaluation_nested",
                    "ProspectiveEvaluationBindingCoordinator",
                    "Atomically bind the admission-only programme before any protected tick or outcome event.",
                ),
                "controller-use.action-aware-nested-evaluation": (
                    "empirical_lawhood.runtime.controller_evaluation_nested",
                    "ActionAwareNestedControllerUseEvaluator",
                    "Reduce action-aware physical cells by repeat median and worst member at task-unit level.",
                ),
                "runtime.deadline-free-resource-recovery": (
                    "empirical_lawhood.runtime.execution_envelope",
                    'RunExecutionResourceEnvelopeMachine',
                    "Enforce finite non-time resources, progress liveness and first-valid receipt recovery without elapsed-time control.",
                ),
            }[obligation],
        )
        for key, obligation in obligation_by_key.items()
    }
    proof_owners = tuple(
        sorted(
            (
                MultiWorldReadinessProofOwner(
                    owner_id=f"proof-owner.{obligation}",
                    obligation_id=obligation,
                    capability_manifest=ObjectIdentity.from_record(
                        key, next(value for value in manifests if value.capability_key == key)
                    ),
                    owner_module=details[0],
                    owner_symbol=details[1],
                    rule_semantics=details[2],
                )
                for obligation, (key, details) in owner_specs.items()
            ),
            key=lambda value: value.owner_id,
        )
    )
    owner_by_obligation = {value.obligation_id: value for value in proof_owners}
    bindings = tuple(
        sorted(
            (
                ReadinessProofOwnerBinding(
                    binding_id=f"proof-binding.{manifest.capability_key}.{schema.rsplit('/', 2)[-2]}",
                    obligation_id=obligation_by_key[manifest.capability_key],
                    output_schema=schema,
                    proof_owner=ObjectIdentity.from_record(
                        owner_by_obligation[obligation_by_key[manifest.capability_key]].owner_id,
                        owner_by_obligation[obligation_by_key[manifest.capability_key]],
                    ),
                    capability_manifest=ObjectIdentity.from_record(
                        manifest.capability_key,
                        manifest,
                    ),
                    maximum_evidence_ceiling=manifest.maximum_evidence_ceiling,
                    maximum_outcome_access=manifest.maximum_outcome_access,
                )
                for manifest in additive
                for schema in manifest.output_schema_ids
            ),
            key=lambda value: value.binding_id,
        )
    )
    maximum_bytes = {
        value.SCHEMA: (
            64 * 1024 * 1024 if value is RunExecutionResourceEnvelope else 16 * 1024 * 1024
        )
        for value in PROSPECTIVE_EXECUTION_RECORD_TYPES
    }
    codecs = build_canonical_record_codec_registry(
        registry_id="prospective-execution-canonical-codecs",
        record_types=PROSPECTIVE_EXECUTION_RECORD_TYPES,
        maximum_bytes_by_schema=maximum_bytes,
        decoder_implementation_sha256=decoder_implementation_sha256,
    )
    record_by_schema = {value.SCHEMA: value for value in PROSPECTIVE_EXECUTION_RECORD_TYPES}
    semantics = tuple(
        sorted(
            (
                CapabilityOutputSemanticContract.from_manifest(
                    manifest,
                    payload_schema=schema,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    top_level_keys=("schema", "value", "version"),
                    value_keys=tuple(
                        sorted(field.name for field in fields(cast(Any, record_by_schema[schema])))
                    ),
                )
                for manifest in manifests
                for schema in manifest.output_schema_ids
            ),
            key=lambda value: value.key,
        )
    )
    return ProspectiveExecutionContractRegistry(
        capability_registry=capability_registry,
        proof_owners=proof_owners,
        proof_owner_bindings=bindings,
        codec_registry=codecs,
        semantic_output_contracts=semantics,
        acquisition_recurrence_margin=acquisition_recurrence_margin,
    )


__all__ = [
    "DEADLINE_FREE_RECOVERY_CAPABILITY_KEY",
    'ProspectiveExecutionContractRegistry',
    "NESTED_CONTROLLER_USE_CAPABILITY_KEY",
    "PROSPECTIVE_BINDING_CAPABILITY_KEY",
    "REFERENCE_CLASS_CAPABILITY_KEY",
    "REFERENCE_MEMBERSHIP_CAPABILITY_KEY",
    "PROSPECTIVE_EXECUTION_ADDITIVE_CAPABILITY_KEYS",
    "PROSPECTIVE_EXECUTION_CAPABILITY_KEYS",
    "PROSPECTIVE_EXECUTION_RECORD_TYPES",
    'build_prospective_execution_contract_registry',
    'prospective_execution_additive_capability_manifests',
]
