"""Closed Acquisition, recurrence and certified admission-margin capability, proof-owner, codec and output-semantics registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
from typing import Any, cast

from empirical_lawhood.adapters.geometry.admission_receipts import CertifiedAdmissionMarginReceiptProducer
from empirical_lawhood.adapters.methods.receiver_conditioned_io.farthest_point_maximin import FarthestPointContinuationConfig, FarthestPointContinuationResult
from empirical_lawhood.adapters.methods.recurrence_qualification import (
    RecurrenceGuardedResponseLawQualificationService,
)
from empirical_lawhood.adapters.simulators.mast_torax_state_transport import MappedActionSupportProjectionReceipt, MappedActionSupportProjection, MappedDevelopmentDonorState, MappedDevelopmentPanelCell, MappedDevelopmentSupportAtlas, MappedDonorDistanceReceipt, MappedStateCoordinate, MappedStateSupportQuery, MappedSupportMetricAxis
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.planning.adaptive_acquisition import AcquisitionQueryCoordinate, ActionPreparationRecurrenceQualificationRequirement, BoundedQueryAcquisitionPlan
from empirical_lawhood.planning.gate_margin import GateMarginProofOwner, GateMarginReceipt
from empirical_lawhood.planning.evidence_geometry import ControlledMapAdmissionReceiptCorpus
from empirical_lawhood.runtime.adaptive_acquisition_validation import AcquisitionBundleAssessment, AcquisitionCheckpointAssessment, AcquisitionCheckpointCell, AcquisitionDecision, AcquisitionPriorityComponent, AcquisitionRoundState, AcquisitionTraceReceipt, ActionPreparationEvidenceCell, ActionPreparationRecurrenceEvidenceSource, ActionPreparationRecurrenceProofOwner, ActionPreparationRecurrenceQualificationBinding, ActionPreparationRecurrenceReceipt, QueryBundleReceipt, QueryCoordinateOutput
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.multi_world_readiness_contracts import ReadinessRuleProofOwnerBinding
from empirical_lawhood.runtime.gate_margin_projection import CertifiedAdmissionMarginCorpusUse, CertifiedAdmissionMarginProjection
from empirical_lawhood.runtime.providers import CapabilityOutputSemanticContract
from empirical_lawhood.runtime.recurrence_controller_study_guard import ActionPreparationRecurrenceAdmissionUse, ActionPreparationRecurrenceControllerStudyUse, RecurrenceBoundCompiledController
from empirical_lawhood.runtime.static_codecs import (
    CanonicalRecordCodecRegistry,
    build_canonical_record_codec_registry,
)


BOUNDED_ACQUISITION_CAPABILITY_KEY = "acquisition.bounded-query-selector"
MAXIMIN_CONTINUATION_CAPABILITY_KEY = "acquisition.farthest-point-continuation"
RECURRENCE_CAPABILITY_KEY = "qualification.action-preparation-recurrence"
GATE_MARGIN_CAPABILITY_KEY = "certification.gate-margin"
CERTIFIED_MARGIN_CORPUS_BINDER_CAPABILITY_KEY = "certification.certified-admission-margin-corpus-binder"

ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS = tuple(
    sorted(
        (
            BOUNDED_ACQUISITION_CAPABILITY_KEY,
            MAXIMIN_CONTINUATION_CAPABILITY_KEY,
            RECURRENCE_CAPABILITY_KEY,
            RecurrenceGuardedResponseLawQualificationService.capability_key,
            GATE_MARGIN_CAPABILITY_KEY,
            CertifiedAdmissionMarginReceiptProducer.capability_key,
            CERTIFIED_MARGIN_CORPUS_BINDER_CAPABILITY_KEY,
        )
    )
)

ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    AcquisitionBundleAssessment,
    AcquisitionCheckpointAssessment,
    AcquisitionCheckpointCell,
    AcquisitionDecision,
    AcquisitionPriorityComponent,
    AcquisitionQueryCoordinate,
    AcquisitionRoundState,
    AcquisitionTraceReceipt,
    ActionPreparationEvidenceCell,
    ActionPreparationRecurrenceEvidenceSource,
    ActionPreparationRecurrenceAdmissionUse,
    ActionPreparationRecurrenceControllerStudyUse,
    ActionPreparationRecurrenceProofOwner,
    ActionPreparationRecurrenceQualificationBinding,
    ActionPreparationRecurrenceQualificationRequirement,
    ActionPreparationRecurrenceReceipt,
    BoundedQueryAcquisitionPlan,
    CertifiedAdmissionMarginCorpusUse,
    CertifiedAdmissionMarginProjection,
    FarthestPointContinuationConfig,
    FarthestPointContinuationResult,
    ReadinessRuleProofOwnerBinding,
    GateMarginProofOwner,
    GateMarginReceipt,
    MappedActionSupportProjectionReceipt,
    MappedActionSupportProjection,
    MappedDevelopmentDonorState,
    MappedDevelopmentPanelCell,
    MappedDevelopmentSupportAtlas,
    MappedDonorDistanceReceipt,
    MappedStateCoordinate,
    MappedStateSupportQuery,
    MappedSupportMetricAxis,
    QueryBundleReceipt,
    QueryCoordinateOutput,
    RecurrenceBoundCompiledController,
)


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode()).hexdigest()


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
    input_schemas: tuple[str, ...],
    output_schemas: tuple[str, ...],
    maximum_evidence_ceiling: EvidenceCeiling,
    maximum_outcome_access: OutcomeAccess,
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version="1.0.0",
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=_schema_sha256(config_schema),
        input_schema_ids=tuple(sorted(input_schemas)),
        output_schema_ids=tuple(sorted(output_schemas)),
        permissions=(),
        maximum_evidence_ceiling=maximum_evidence_ceiling,
        maximum_outcome_access=maximum_outcome_access,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-acquisition-recurrence-margin",
        requires_clean_commit=True,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def acquisition_recurrence_margin_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    if set(implementation_sha256_by_key) != set(ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS):
        raise ValueError("Acquisition, recurrence and certified admission-margin implementation identities must cover the exact capability family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = (
        (
            BOUNDED_ACQUISITION_CAPABILITY_KEY,
            CapabilityKind.EXPERIMENT_DESIGNER,
            BoundedQueryAcquisitionPlan.SCHEMA,
            (
                AcquisitionCheckpointAssessment.SCHEMA,
                BoundedQueryAcquisitionPlan.SCHEMA,
                AcquisitionRoundState.SCHEMA,
            ),
            (
                AcquisitionDecision.SCHEMA,
                QueryBundleReceipt.SCHEMA,
                AcquisitionTraceReceipt.SCHEMA,
            ),
            EvidenceCeiling.ORDER_RELATION,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            (
                "acquisition-recurrence-margin-exact-issued-query-product",
                "acquisition-recurrence-margin-three-way-decision-and-zero-call-skip",
                "acquisition-recurrence-margin-two-round-receipt-dependency",
            ),
        ),
        (
            MAXIMIN_CONTINUATION_CAPABILITY_KEY,
            CapabilityKind.EXPERIMENT_DESIGNER,
            FarthestPointContinuationConfig.SCHEMA,
            (FarthestPointContinuationConfig.SCHEMA,),
            (FarthestPointContinuationResult.SCHEMA,),
            EvidenceCeiling.ORDER_RELATION,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            (
                "acquisition-recurrence-margin-already-queried-roster-excluded",
                "acquisition-recurrence-margin-frozen-native-scale-distance",
                "acquisition-recurrence-margin-lexical-tie-break",
            ),
        ),
        (
            RECURRENCE_CAPABILITY_KEY,
            CapabilityKind.NUMERICAL_QUALIFIER,
            ActionPreparationRecurrenceQualificationRequirement.SCHEMA,
            (
                AcquisitionTraceReceipt.SCHEMA,
                ActionPreparationEvidenceCell.SCHEMA,
                ActionPreparationRecurrenceEvidenceSource.SCHEMA,
                ActionPreparationRecurrenceQualificationRequirement.SCHEMA,
            ),
            (ActionPreparationRecurrenceReceipt.SCHEMA,),
            EvidenceCeiling.ORDER_RELATION,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            (
                "acquisition-recurrence-margin-two-distinct-preparations",
                "acquisition-recurrence-margin-active-hold-member-product",
                "acquisition-recurrence-margin-no-post-reference-repair",
            ),
        ),
        (
            RecurrenceGuardedResponseLawQualificationService.capability_key,
            CapabilityKind.LAW_IDENTIFIER,
            ActionPreparationRecurrenceQualificationRequirement.SCHEMA,
            (
                ActionPreparationRecurrenceQualificationRequirement.SCHEMA,
                ActionPreparationRecurrenceReceipt.SCHEMA,
            ),
            (ActionPreparationRecurrenceQualificationBinding.SCHEMA,),
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            (
                "acquisition-recurrence-margin-recurrence-consumed-before-sole-finalizer",
                "acquisition-recurrence-margin-no-second-law-finalizer",
                "acquisition-recurrence-margin-receipt-digest-continuity",
            ),
        ),
        (
            GATE_MARGIN_CAPABILITY_KEY,
            CapabilityKind.ADMISSION_EVALUATOR,
            GateMarginReceipt.SCHEMA,
            (GateMarginReceipt.SCHEMA,),
            (GateMarginReceipt.SCHEMA,),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.OUTCOME_BLIND,
            (
                "acquisition-recurrence-margin-native-margin-and-uncertainty-bounds",
                "acquisition-recurrence-margin-numeric-versus-uncertainty-indeterminacy",
                "acquisition-recurrence-margin-padding-never-authors-pass",
            ),
        ),
        (
            CertifiedAdmissionMarginReceiptProducer.capability_key,
            CapabilityKind.ADMISSION_EVALUATOR,
            GateMarginReceipt.SCHEMA,
            (GateMarginReceipt.SCHEMA,),
            (CertifiedAdmissionMarginProjection.SCHEMA,),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.OUTCOME_BLIND,
            (
                "acquisition-recurrence-margin-certified-operand-only",
                "acquisition-recurrence-margin-indeterminate-remains-unevaluable",
                "acquisition-recurrence-margin-unchanged-raw-admission-producer",
            ),
        ),
        (
            CERTIFIED_MARGIN_CORPUS_BINDER_CAPABILITY_KEY,
            CapabilityKind.ADMISSION_EVALUATOR,
            CertifiedAdmissionMarginCorpusUse.SCHEMA,
            (ControlledMapAdmissionReceiptCorpus.SCHEMA, CertifiedAdmissionMarginProjection.SCHEMA),
            (CertifiedAdmissionMarginCorpusUse.SCHEMA,),
            EvidenceCeiling.ADMISSION,
            OutcomeAccess.OUTCOME_BLIND,
            (
                "acquisition-recurrence-margin-exact-complete-certified-gate-corpus",
                "acquisition-recurrence-margin-no-uncertified-gate-substitution",
                "acquisition-recurrence-margin-margin-digest-programme-continuity",
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
                    input_schemas=input_schemas,
                    output_schemas=output_schemas,
                    maximum_evidence_ceiling=evidence_ceiling,
                    maximum_outcome_access=outcome_access,
                    checks=checks,
                    implementation_sha256=implementation_sha256_by_key[key],
                )
                for (
                    key,
                    kind,
                    config_schema,
                    input_schemas,
                    output_schemas,
                    evidence_ceiling,
                    outcome_access,
                    checks,
                ) in specifications
            ),
            key=lambda value: value.registry_id,
        )
    )


@dataclass(frozen=True, slots=True)
class AcquisitionRecurrenceMarginContractRegistry:
    capability_registry: CapabilityRegistry
    recurrence_proof_owner: ActionPreparationRecurrenceProofOwner
    gate_margin_proof_owner: GateMarginProofOwner
    proof_owner_bindings: tuple[ReadinessRuleProofOwnerBinding, ...]
    codec_registry: CanonicalRecordCodecRegistry
    semantic_output_contracts: tuple[CapabilityOutputSemanticContract, ...]

    def __post_init__(self) -> None:
        manifests = {value.capability_key: value for value in self.capability_registry.capabilities}
        if set(manifests) != set(ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS):
            raise ValueError("Acquisition, recurrence and certified admission-margin contract registry changes the capability family")
        semantics = {
            (value.capability_key, value.payload_schema) for value in self.semantic_output_contracts
        }
        expected_semantics = {
            (manifest.capability_key, schema)
            for manifest in manifests.values()
            for schema in manifest.output_schema_ids
        }
        if semantics != expected_semantics:
            raise ValueError("Acquisition, recurrence and certified admission-margin semantic output contracts are incomplete")
        codec_schemas = set(self.codec_registry.record_types)
        if not {schema for _, schema in expected_semantics} <= codec_schemas:
            raise ValueError("Acquisition, recurrence and certified admission-margin output semantic contract lacks a static codec")
        expected_proof_outputs = {
            ActionPreparationRecurrenceReceipt.SCHEMA,
            ActionPreparationRecurrenceQualificationBinding.SCHEMA,
            GateMarginReceipt.SCHEMA,
            CertifiedAdmissionMarginCorpusUse.SCHEMA,
            CertifiedAdmissionMarginProjection.SCHEMA,
        }
        if {value.output_schema for value in self.proof_owner_bindings} != (expected_proof_outputs):
            raise ValueError("Acquisition, recurrence and certified admission-margin proof-owner bindings are incomplete")
        recurrence_manifest = manifests[RECURRENCE_CAPABILITY_KEY]
        margin_manifest = manifests[GATE_MARGIN_CAPABILITY_KEY]
        if (
            self.recurrence_proof_owner.capability_key != recurrence_manifest.capability_key
            or self.recurrence_proof_owner.capability_version
            != recurrence_manifest.capability_version
            or self.recurrence_proof_owner.implementation_sha256
            != recurrence_manifest.implementation_sha256
            or self.gate_margin_proof_owner.capability_key != margin_manifest.capability_key
            or self.gate_margin_proof_owner.capability_version != margin_manifest.capability_version
            or self.gate_margin_proof_owner.implementation_sha256
            != margin_manifest.implementation_sha256
        ):
            raise ValueError("Acquisition, recurrence and certified admission-margin proof owner differs from its registered capability")


def build_acquisition_recurrence_margin_contract_registry(
    *,
    implementation_sha256_by_key: Mapping[str, str],
    decoder_implementation_sha256: str,
) -> AcquisitionRecurrenceMarginContractRegistry:
    manifests = acquisition_recurrence_margin_capability_manifests(implementation_sha256_by_key)
    capability_registry = CapabilityRegistry(
        registry_id="acquisition-recurrence-margin-capabilities",
        capabilities=manifests,
    )
    by_key = {value.capability_key: value for value in manifests}
    recurrence_manifest = by_key[RECURRENCE_CAPABILITY_KEY]
    margin_manifest = by_key[GATE_MARGIN_CAPABILITY_KEY]
    recurrence_owner = ActionPreparationRecurrenceProofOwner(
        owner_id="proof-owner.action-preparation-recurrence",
        obligation_id="order-relation.action-preparation-recurrence",
        capability_key=recurrence_manifest.capability_key,
        capability_version=recurrence_manifest.capability_version,
        config_sha256=recurrence_manifest.config_schema_sha256,
        implementation_sha256=recurrence_manifest.implementation_sha256,
    )
    margin_owner = GateMarginProofOwner(
        owner_id="proof-owner.gate-margin-certification",
        obligation_id="admission.gate-margin-certification",
        capability_key=margin_manifest.capability_key,
        capability_version=margin_manifest.capability_version,
        config_sha256=margin_manifest.config_schema_sha256,
        implementation_sha256=margin_manifest.implementation_sha256,
    )
    recurrence_owner_identity = ObjectIdentity.from_record(
        recurrence_owner.owner_id,
        recurrence_owner,
    )
    margin_owner_identity = ObjectIdentity.from_record(
        margin_owner.owner_id,
        margin_owner,
    )
    proof_specs = (
        (
            "proof-binding.action-preparation-recurrence-receipt",
            "order-relation.action-preparation-recurrence",
            ActionPreparationRecurrenceReceipt.SCHEMA,
            recurrence_owner_identity,
            RECURRENCE_CAPABILITY_KEY,
            "empirical_lawhood.runtime.adaptive_acquisition_validation",
            "evaluate_action_preparation_recurrence",
            "Require two distinct preparations with matched active/HOLD cells for every member.",
        ),
        (
            "proof-binding.recurrence-qualification-binding",
            "order-relation.action-preparation-recurrence",
            ActionPreparationRecurrenceQualificationBinding.SCHEMA,
            recurrence_owner_identity,
            RecurrenceGuardedResponseLawQualificationService.capability_key,
            "empirical_lawhood.adapters.methods.recurrence_qualification",
            "RecurrenceGuardedResponseLawQualificationService",
            "Consume exact supported order-relation receipts before delegating to the sole law finalizer.",
        ),
        (
            "proof-binding.gate-margin-receipt",
            "admission.gate-margin-certification",
            GateMarginReceipt.SCHEMA,
            margin_owner_identity,
            GATE_MARGIN_CAPABILITY_KEY,
            "empirical_lawhood.planning.gate_margin",
            "certify_gate_margin",
            "Derive signed native margin, uncertainty interval and numeric-floor disposition.",
        ),
        (
            "proof-binding.certified-admission-margin-projection",
            "admission.gate-margin-certification",
            CertifiedAdmissionMarginProjection.SCHEMA,
            margin_owner_identity,
            CertifiedAdmissionMarginReceiptProducer.capability_key,
            'empirical_lawhood.adapters.geometry.admission_receipts',
            "CertifiedAdmissionMarginReceiptProducer",
            "Pass only a certified signed operand or UNEVALUABLE into the unchanged raw admission producer.",
        ),
        (
            "proof-binding.certified-admission-margin-corpus-use",
            "admission.gate-margin-certification",
            CertifiedAdmissionMarginCorpusUse.SCHEMA,
            margin_owner_identity,
            CERTIFIED_MARGIN_CORPUS_BINDER_CAPABILITY_KEY,
            "empirical_lawhood.runtime.gate_margin_projection",
            'bind_certified_admission_margin_corpus_use',
            "Require one exact certified-margin projection for every gate receipt before programme use.",
        ),
    )
    proof_bindings = tuple(
        sorted(
            (
                ReadinessRuleProofOwnerBinding(
                    binding_id=binding_id,
                    obligation_id=obligation_id,
                    output_schema=output_schema,
                    proof_owner=owner,
                    capability_manifest=ObjectIdentity.from_record(
                        by_key[capability_key].capability_key,
                        by_key[capability_key],
                    ),
                    owner_module=owner_module,
                    owner_symbol=owner_symbol,
                    maximum_evidence_ceiling=by_key[capability_key].maximum_evidence_ceiling,
                    maximum_outcome_access=by_key[capability_key].maximum_outcome_access,
                    rule_semantics=rule_semantics,
                )
                for (
                    binding_id,
                    obligation_id,
                    output_schema,
                    owner,
                    capability_key,
                    owner_module,
                    owner_symbol,
                    rule_semantics,
                ) in proof_specs
            ),
            key=lambda value: value.binding_id,
        )
    )
    maximum_bytes = {
        record_type.SCHEMA: (
            16 * 1024 * 1024
            if record_type
            in {
                AcquisitionTraceReceipt,
                MappedDevelopmentSupportAtlas,
                MappedActionSupportProjectionReceipt,
                RecurrenceBoundCompiledController,
            }
            else 4 * 1024 * 1024
        )
        for record_type in ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES
    }
    codec_registry = build_canonical_record_codec_registry(
        registry_id="acquisition-recurrence-margin-canonical-codecs",
        record_types=ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES,
        maximum_bytes_by_schema=maximum_bytes,
        decoder_implementation_sha256=decoder_implementation_sha256,
    )
    record_by_schema = {value.SCHEMA: value for value in ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES}
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
    return AcquisitionRecurrenceMarginContractRegistry(
        capability_registry=capability_registry,
        recurrence_proof_owner=recurrence_owner,
        gate_margin_proof_owner=margin_owner,
        proof_owner_bindings=proof_bindings,
        codec_registry=codec_registry,
        semantic_output_contracts=semantics,
    )


__all__ = [
    "BOUNDED_ACQUISITION_CAPABILITY_KEY",
    "CERTIFIED_MARGIN_CORPUS_BINDER_CAPABILITY_KEY",
    'AcquisitionRecurrenceMarginContractRegistry',
    "GATE_MARGIN_CAPABILITY_KEY",
    "MAXIMIN_CONTINUATION_CAPABILITY_KEY",
    "RECURRENCE_CAPABILITY_KEY",
    "ACQUISITION_RECURRENCE_MARGIN_CAPABILITY_KEYS",
    "ACQUISITION_RECURRENCE_MARGIN_RECORD_TYPES",
    'build_acquisition_recurrence_margin_contract_registry',
    'acquisition_recurrence_margin_capability_manifests',
]
