"""Closed Multi-world issue, overlap morphism and nonpooling adjudication multi-world capability, proof-owner and codec composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
from typing import Any, cast

from empirical_lawhood.adapters.composition.prospective_execution_contracts import PROSPECTIVE_EXECUTION_CAPABILITY_KEYS, PROSPECTIVE_EXECUTION_RECORD_TYPES, ProspectiveExecutionContractRegistry, build_prospective_execution_contract_registry
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.planning.multi_world_study import ArchiveOutcomeProtectionPlan, ArchiveOverlapQualification, ArchiveToSimulatorPartialMorphismSpec, MultiWorldJointAdjudicationPlan, MorphismFieldBinding, MorphismNegativeControlPlan, MultiWorldOutcomeBarrierPlan, MultiWorldOutcomeBarrierSpec, MultiWorldStudyChildScientificBinding, PropertyTransportExpectation
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.multi_world_readiness_contracts import ReadinessProofOwnerBinding, MultiWorldReadinessProofOwner
from empirical_lawhood.runtime.multi_world_study_runbook import MultiWorldStudyOperationHelpBinding, MultiWorldStudyOperationRequestContract, MultiWorldStudyOperationRunbookStep, MultiWorldStudyOperationRunbook
from empirical_lawhood.runtime.multi_world_route_conformance import MultiWorldConformanceCommandReceipt, MultiWorldPublicRouteConformanceReceipt, MultiWorldExecutedConformanceSuite
from empirical_lawhood.runtime.multi_world_study import ArchiveOverlapObservation, ArchiveOverlapQualificationResult, ArchiveTwoActionClassPair, MultiWorldJointAdjudicationResult, MorphismControlContrastReceipt, MorphismControlStateObservation, MorphismControlContrast, PropertyMorphismMemberVerdict, PropertyMorphismStateInput, PropertyMorphismVerdict, ToraxMemberTwoActionClassPair, WorldLocalStudyResult
from empirical_lawhood.runtime.study_bundle_compiler import StudyBundleCandidate
from empirical_lawhood.runtime.multi_world_study_issue import IssuedMultiWorldStudyChild, IssuedMultiWorldJointDescendant, IssuedMultiWorldStudy, MultiWorldAuthorityRequired, MultiWorldOutcomeBarrierEvent, MultiWorldOutcomeBarrierPrefix, MultiWorldStudyChildExecutionBinding, MultiWorldStudyChildRecoveryBinding, MultiWorldStudyExecutionPlan, MultiWorldStudyIssueManifest, MultiWorldStudyPublicationReceipt, MultiWorldStudyRecoveryIndex, MultiWorldStudyBarrierStatus
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract
from empirical_lawhood.runtime.static_codecs import (
    CanonicalRecordCodecRegistry,
    build_canonical_record_codec_registry,
)


MULTI_WORLD_BUNDLE_CAPABILITY_KEY = "bundle.multi-world-issue-recovery"
MORPHISM_CONTROLS_CAPABILITY_KEY = "morphism.archive-overlap-controls"
NONPOOLING_JOINT_CAPABILITY_KEY = "adjudication.nonpooling-joint"

MULTI_WORLD_EXECUTION_ADDITIVE_CAPABILITY_KEYS = tuple(
    sorted(
        (
            MULTI_WORLD_BUNDLE_CAPABILITY_KEY,
            MORPHISM_CONTROLS_CAPABILITY_KEY,
            NONPOOLING_JOINT_CAPABILITY_KEY,
        )
    )
)
MULTI_WORLD_EXECUTION_CAPABILITY_KEYS = tuple(sorted((*PROSPECTIVE_EXECUTION_CAPABILITY_KEYS, *MULTI_WORLD_EXECUTION_ADDITIVE_CAPABILITY_KEYS)))

MULTI_WORLD_EXECUTION_ADDITIVE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    ArchiveOutcomeProtectionPlan,
    ArchiveOverlapObservation,
    ArchiveOverlapQualificationResult,
    ArchiveOverlapQualification,
    ArchiveToSimulatorPartialMorphismSpec,
    ArchiveTwoActionClassPair,
    MultiWorldStudyOperationHelpBinding,
    MultiWorldStudyOperationRequestContract,
    MultiWorldStudyOperationRunbookStep,
    MultiWorldStudyOperationRunbook,
    MultiWorldConformanceCommandReceipt,
    MultiWorldPublicRouteConformanceReceipt,
    MultiWorldExecutedConformanceSuite,
    IssuedMultiWorldStudyChild,
    IssuedMultiWorldJointDescendant,
    IssuedMultiWorldStudy,
    MultiWorldJointAdjudicationPlan,
    MultiWorldJointAdjudicationResult,
    MorphismControlContrastReceipt,
    MorphismControlStateObservation,
    MorphismFieldBinding,
    MorphismControlContrast,
    MorphismNegativeControlPlan,
    MultiWorldAuthorityRequired,
    MultiWorldOutcomeBarrierEvent,
    MultiWorldOutcomeBarrierPlan,
    MultiWorldOutcomeBarrierPrefix,
    MultiWorldOutcomeBarrierSpec,
    StudyBundleCandidate,
    MultiWorldStudyBarrierStatus,
    MultiWorldStudyChildExecutionBinding,
    MultiWorldStudyChildRecoveryBinding,
    MultiWorldStudyChildScientificBinding,
    MultiWorldStudyExecutionPlan,
    MultiWorldStudyIssueManifest,
    MultiWorldStudyPublicationReceipt,
    MultiWorldStudyRecoveryIndex,
    PropertyMorphismMemberVerdict,
    PropertyMorphismStateInput,
    PropertyMorphismVerdict,
    PropertyTransportExpectation,
    ToraxMemberTwoActionClassPair,
    WorldLocalStudyResult,
)
MULTI_WORLD_EXECUTION_RECORD_TYPES = tuple(
    {value.SCHEMA: value for value in (*PROSPECTIVE_EXECUTION_RECORD_TYPES, *MULTI_WORLD_EXECUTION_ADDITIVE_RECORD_TYPES)}.values()
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
        runtime_id="cpython-3.11-multi-world-execution",
        requires_clean_commit=True,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def multi_world_execution_additive_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    if set(implementation_sha256_by_key) != set(MULTI_WORLD_EXECUTION_ADDITIVE_CAPABILITY_KEYS):
        raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication implementation identities must cover the exact additive family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = (
        (
            MULTI_WORLD_BUNDLE_CAPABILITY_KEY,
            CapabilityKind.EVALUATOR,
            MultiWorldStudyIssueManifest.SCHEMA,
            (
                StudyBundleCandidate.SCHEMA,
                MultiWorldStudyIssueManifest.SCHEMA,
                MultiWorldOutcomeBarrierPlan.SCHEMA,
            ),
            (
                IssuedMultiWorldStudy.SCHEMA,
                MultiWorldStudyOperationRunbook.SCHEMA,
                MultiWorldPublicRouteConformanceReceipt.SCHEMA,
                MultiWorldOutcomeBarrierPrefix.SCHEMA,
                MultiWorldStudyRecoveryIndex.SCHEMA,
            ),
            EvidenceCeiling.MEASUREMENT,
            OutcomeAccess.EVALUATOR_REVEAL,
            (
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            (
                "multi-world-execution-distinct-child-science",
                "multi-world-execution-simulator-before-archive-reveal",
                "multi-world-execution-immutable-prefix-and-result-recovery",
            ),
        ),
        (
            MORPHISM_CONTROLS_CAPABILITY_KEY,
            CapabilityKind.TRANSPORT_TESTER,
            ArchiveToSimulatorPartialMorphismSpec.SCHEMA,
            (
                ArchiveOverlapObservation.SCHEMA,
                ArchiveToSimulatorPartialMorphismSpec.SCHEMA,
                MorphismControlStateObservation.SCHEMA,
                PropertyMorphismStateInput.SCHEMA,
            ),
            (
                ArchiveOverlapQualificationResult.SCHEMA,
                MorphismControlContrastReceipt.SCHEMA,
                PropertyMorphismVerdict.SCHEMA,
            ),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.EVALUATOR_REVEAL,
            (CapabilityPermission.READ_SEALED_OUTCOMES,),
            (
                "multi-world-execution-overlap-local-recurrence",
                "multi-world-execution-common-negative-control-denominator",
                "multi-world-execution-four-morphism-dispositions",
            ),
        ),
        (
            NONPOOLING_JOINT_CAPABILITY_KEY,
            CapabilityKind.HYPOTHESIS_ADJUDICATOR,
            MultiWorldJointAdjudicationPlan.SCHEMA,
            (
                ArchiveOverlapQualificationResult.SCHEMA,
                MultiWorldJointAdjudicationPlan.SCHEMA,
                MorphismControlContrastReceipt.SCHEMA,
                PropertyMorphismVerdict.SCHEMA,
                WorldLocalStudyResult.SCHEMA,
            ),
            (MultiWorldJointAdjudicationResult.SCHEMA,),
            EvidenceCeiling.CONTROLLER_USE,
            OutcomeAccess.EVALUATOR_REVEAL,
            (CapabilityPermission.READ_SEALED_OUTCOMES,),
            (
                "multi-world-execution-world-local-nonpooling",
                "multi-world-execution-archive-lawhood-no-simulator-rescue",
                "multi-world-execution-deterministic-joint-closeout",
            ),
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
class MultiWorldExecutionContractRegistry:
    capability_registry: CapabilityRegistry
    proof_owners: tuple[MultiWorldReadinessProofOwner, ...]
    proof_owner_bindings: tuple[ReadinessProofOwnerBinding, ...]
    codec_registry: CanonicalRecordCodecRegistry
    semantic_output_contracts: tuple[CapabilityOutputSemanticContract, ...]
    prospective_execution: ProspectiveExecutionContractRegistry

    def __post_init__(self) -> None:
        manifests = {value.capability_key: value for value in self.capability_registry.capabilities}
        if set(manifests) != set(MULTI_WORLD_EXECUTION_CAPABILITY_KEYS):
            raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication registry changes the exact Multi-world issue, overlap morphism and nonpooling adjudication capability family")
        owners = {value.obligation_id: value for value in self.proof_owners}
        if len(owners) != len(self.proof_owners):
            raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication contract obligations have multiple proof owners")
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
                raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication proof binding differs from owner/capability/output semantics")
        expected_semantics = {
            (manifest.capability_key, schema)
            for manifest in manifests.values()
            for schema in manifest.output_schema_ids
        }
        if {
            (value.capability_key, value.payload_schema) for value in self.semantic_output_contracts
        } != expected_semantics:
            raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication semantic output contracts are incomplete")
        if not {schema for _, schema in expected_semantics} <= set(
            self.codec_registry.record_types
        ):
            raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication output semantics lack closed static codecs")


def build_multi_world_execution_contract_registry(
    *,
    implementation_sha256_by_key: Mapping[str, str],
    decoder_implementation_sha256: str,
) -> MultiWorldExecutionContractRegistry:
    if set(implementation_sha256_by_key) != set(MULTI_WORLD_EXECUTION_CAPABILITY_KEYS):
        raise ValueError("Multi-world issue, overlap morphism and nonpooling adjudication implementation identities must cover the exact Multi-world issue, overlap morphism and nonpooling adjudication family")
    prospective_execution = build_prospective_execution_contract_registry(
        implementation_sha256_by_key={
            key: implementation_sha256_by_key[key] for key in PROSPECTIVE_EXECUTION_CAPABILITY_KEYS
        },
        decoder_implementation_sha256=decoder_implementation_sha256,
    )
    additive = multi_world_execution_additive_capability_manifests(
        {key: implementation_sha256_by_key[key] for key in MULTI_WORLD_EXECUTION_ADDITIVE_CAPABILITY_KEYS}
    )
    manifests = tuple(
        sorted(
            (*prospective_execution.capability_registry.capabilities, *additive),
            key=lambda value: value.registry_id,
        )
    )
    capability_registry = CapabilityRegistry(
        registry_id="multi-world-execution-capabilities",
        capabilities=manifests,
    )
    owner_details = {
        MULTI_WORLD_BUNDLE_CAPABILITY_KEY: (
            'empirical_lawhood.runtime.multi_world_study_issue',
            'MultiWorldStudyRecoveryIndex',
            "Issue distinct world-local children, enforce ordered reveal and recover exact immutable results.",
        ),
        MORPHISM_CONTROLS_CAPABILITY_KEY: (
            'empirical_lawhood.runtime.multi_world_study',
            "evaluate_property_morphism",
            "Adjudicate the overlap-local partial morphism and all negative maps without transporting local magnitudes or evidence rungs.",
        ),
        NONPOOLING_JOINT_CAPABILITY_KEY: (
            'empirical_lawhood.runtime.multi_world_study',
            'adjudicate_joint_multi_world_study',
            "Combine terminal world-local propositions without pooling or simulator rescue of archive lawhood.",
        ),
    }
    new_owners = tuple(
        sorted(
            (
                MultiWorldReadinessProofOwner(
                    owner_id=f"proof-owner.{key}",
                    obligation_id=key,
                    capability_manifest=ObjectIdentity.from_record(
                        key,
                        next(value for value in additive if value.capability_key == key),
                    ),
                    owner_module=details[0],
                    owner_symbol=details[1],
                    rule_semantics=details[2],
                )
                for key, details in owner_details.items()
            ),
            key=lambda value: value.owner_id,
        )
    )
    proof_owners = tuple(sorted((*prospective_execution.proof_owners, *new_owners), key=lambda value: value.owner_id))
    owner_by_obligation = {value.obligation_id: value for value in new_owners}
    new_bindings = tuple(
        sorted(
            (
                ReadinessProofOwnerBinding(
                    binding_id=f"proof-binding.{manifest.capability_key}.{schema.rsplit('/', 2)[-2]}",
                    obligation_id=manifest.capability_key,
                    output_schema=schema,
                    proof_owner=ObjectIdentity.from_record(
                        owner_by_obligation[manifest.capability_key].owner_id,
                        owner_by_obligation[manifest.capability_key],
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
    proof_bindings = tuple(
        sorted((*prospective_execution.proof_owner_bindings, *new_bindings), key=lambda value: value.binding_id)
    )
    codecs = build_canonical_record_codec_registry(
        registry_id="multi-world-execution-canonical-codecs",
        record_types=MULTI_WORLD_EXECUTION_RECORD_TYPES,
        maximum_bytes_by_schema={value.SCHEMA: 64 * 1024 * 1024 for value in MULTI_WORLD_EXECUTION_RECORD_TYPES},
        decoder_implementation_sha256=decoder_implementation_sha256,
    )
    record_by_schema = {value.SCHEMA: value for value in MULTI_WORLD_EXECUTION_RECORD_TYPES}
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
    return MultiWorldExecutionContractRegistry(
        capability_registry=capability_registry,
        proof_owners=proof_owners,
        proof_owner_bindings=proof_bindings,
        codec_registry=codecs,
        semantic_output_contracts=semantics,
        prospective_execution=prospective_execution,
    )


__all__ = [
    'MultiWorldExecutionContractRegistry',
    "MORPHISM_CONTROLS_CAPABILITY_KEY",
    "MULTI_WORLD_BUNDLE_CAPABILITY_KEY",
    "NONPOOLING_JOINT_CAPABILITY_KEY",
    "MULTI_WORLD_EXECUTION_ADDITIVE_CAPABILITY_KEYS",
    "MULTI_WORLD_EXECUTION_CAPABILITY_KEYS",
    "MULTI_WORLD_EXECUTION_RECORD_TYPES",
    'build_multi_world_execution_contract_registry',
    'multi_world_execution_additive_capability_manifests',
]
