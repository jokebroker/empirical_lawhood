"""Static registration for fixed-panel confirmatory finite-action inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .confirmatory_finite_action import CONFIRMATORY_FINITE_ACTION_KEY, CONFIRMATORY_FINITE_ACTION_VERSION, ConfirmatoryFiniteActionConfig, ConfirmatoryFiniteActionProducer, ConfirmatoryFiniteActionResult
from .contracts import LawCandidateEvidence
from .finite_action_identification import FiniteActionCandidateScaffold
from .law_assessment import (
    CandidatePayloadDecoderRegistry,
    CandidatePayloadPublisher,
    CanonicalFiniteActionCompatibilitySetDecoder,
    QualificationProfileEvaluatorRegistry,
)
from .qualification_profiles import (
    MethodEquivalentProfileKind,
    MethodEquivalentQualificationProfileEvaluator,
    QualificationProofOwner,
)
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceProjection
from empirical_lawhood.planning.identification_evidence_extensions import IdentificationEvidenceProjectionExtension


CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY = "finite-action.confirmatory-provider"
CONFIRMATORY_FINITE_ACTION_PROFILE_KEY = "qualification-profile.confirmatory-finite-action"
CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT_PROFILE_KEY = (
    "qualification-profile.confirmatory-finite-action-local-support"
)
CONFIRMATORY_FINITE_ACTION_CONFIG_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES = 512 * 1024


def _record_identity(record: CanonicalRecord) -> ObjectIdentity:
    if isinstance(record, ConfirmatoryFiniteActionConfig):
        object_id = record.config_id
    elif isinstance(record, IdentificationEvidenceProjection):
        object_id = record.projection_id
    elif isinstance(record, IdentificationEvidenceProjectionExtension):
        object_id = record.extension_id
    elif isinstance(record, FiniteActionCandidateScaffold):
        object_id = record.scaffold_id
    else:  # pragma: no cover - provider record roster is closed above
        raise TypeError("confirmatory provider received another record type")
    return ObjectIdentity.from_record(object_id, record)


def confirmatory_finite_action_manifest(*, implementation_sha256: str) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=CONFIRMATORY_FINITE_ACTION_KEY,
        capability_version=CONFIRMATORY_FINITE_ACTION_VERSION,
        kind=CapabilityKind.LAW_IDENTIFIER,
        config_schema=ConfirmatoryFiniteActionConfig.SCHEMA,
        config_schema_sha256=hashlib.sha256(
            ConfirmatoryFiniteActionConfig.SCHEMA.encode("utf-8")
        ).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    FiniteActionCandidateScaffold.SCHEMA,
                    IdentificationEvidenceProjection.SCHEMA,
                    IdentificationEvidenceProjectionExtension.SCHEMA,
                )
            )
        ),
        output_schema_ids=tuple(
            sorted((ConfirmatoryFiniteActionResult.SCHEMA, LawCandidateEvidence.SCHEMA))
        ),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=8 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=3_600,
            source_scan_bytes=4 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-confirmatory-finite-action",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "bonferroni-one-sided-student-bound",
            "complete-independent-unit-word-grid",
            "exact-multi-occurrence-action-word",
            "fresh-confirmatory-panel",
            "member-structural-not-magnitude-agreement",
            "prefix-closure-and-tier-count-median",
            "role-specific-causal-controls",
            "sole-terminal-law-finalizer",
        ),
        implementation_sha256=implementation_sha256,
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionConfigDecoder:
    config: ConfirmatoryFiniteActionConfig
    provider_key: str = CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY
    provider_version: str = CONFIRMATORY_FINITE_ACTION_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != ConfirmatoryFiniteActionConfig.SCHEMA:
            raise ValueError("confirmatory finite-action config schema differs")
        decoded = decode_canonical_bytes(
            payload,
            ConfirmatoryFiniteActionConfig,
            maximum_bytes=CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
        )
        if decoded != self.config:
            raise ValueError("confirmatory finite-action config differs from registration")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionComponents:
    capability_registry: CapabilityRegistry
    candidate_registration: CandidateCapabilityRegistration
    config_decoder: ConfirmatoryFiniteActionConfigDecoder
    producer: ConfirmatoryFiniteActionProducer
    decoder_registry: CandidatePayloadDecoderRegistry
    profile_registry: QualificationProfileEvaluatorRegistry


class _ConfirmatoryFiniteActionRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: ConfirmatoryFiniteActionConfig,
        producer: ConfirmatoryFiniteActionProducer,
    ) -> None:
        self.manifest = manifest
        self.config = config
        self.producer = producer

    def execute(self, context: TaskContext) -> RunnerResult:
        decoded: dict[str, CanonicalRecord] = {}
        types: tuple[type[CanonicalRecord], ...] = (
            ConfirmatoryFiniteActionConfig,
            IdentificationEvidenceProjection,
            IdentificationEvidenceProjectionExtension,
            FiniteActionCandidateScaffold,
        )
        by_schema = {value.SCHEMA: value for value in types}
        for port in context.input_ports:
            record_type = by_schema.get(port.payload_schema)
            if record_type is None or port.payload_schema in decoded:
                raise ValueError("confirmatory runner input roster differs")
            decoded[port.payload_schema] = decode_canonical_bytes(
                port.read(), record_type, maximum_bytes=port.size_bytes
            )
        if set(decoded) != set(by_schema):
            raise ValueError("confirmatory runner requires config/projection/companion/scaffold")
        config = decoded[ConfirmatoryFiniteActionConfig.SCHEMA]
        projection = decoded[IdentificationEvidenceProjection.SCHEMA]
        extension = decoded[IdentificationEvidenceProjectionExtension.SCHEMA]
        scaffold = decoded[FiniteActionCandidateScaffold.SCHEMA]
        if not isinstance(config, ConfirmatoryFiniteActionConfig):  # pragma: no cover
            raise TypeError("confirmatory runner decoded another config")
        if not isinstance(projection, IdentificationEvidenceProjection):  # pragma: no cover
            raise TypeError("confirmatory runner decoded another projection")
        if not isinstance(
            extension, IdentificationEvidenceProjectionExtension
        ):  # pragma: no cover
            raise TypeError("confirmatory runner decoded another companion")
        if not isinstance(scaffold, FiniteActionCandidateScaffold):  # pragma: no cover
            raise TypeError("confirmatory runner decoded another scaffold")
        if (
            config != self.config
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("confirmatory runner identity/authority differs")
        result = self.producer.identify(
            projection=projection,
            extension=extension,
            config=config,
            scaffold=scaffold,
        )
        records: dict[str, CanonicalRecord] = {
            ConfirmatoryFiniteActionResult.SCHEMA: result,
            LawCandidateEvidence.SCHEMA: result.candidate_evidence,
        }
        if {value.payload_schema for value in context.output_ports} != set(records):
            raise ValueError("confirmatory runner output roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=records[port.payload_schema].canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("confirmatory-finite-action-exact-roster", True, ()),
                ReceiptCheck("confirmatory-finite-action-no-terminal-verdict", True, ()),
            ),
        )


class ConfirmatoryFiniteActionCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Closed runtime provider for one exact confirmatory method instance."""

    def __init__(
        self,
        *,
        components: ConfirmatoryFiniteActionComponents,
        config: ConfirmatoryFiniteActionConfig,
        projection: IdentificationEvidenceProjection,
        extension: IdentificationEvidenceProjectionExtension,
        scaffold: FiniteActionCandidateScaffold,
    ) -> None:
        self.registry = components.capability_registry
        self.registry_sha256 = self.registry.fingerprint()
        self.capability_count = 1
        self.config = config
        named = tuple(
            value
            for value in self.registry.capabilities
            if value.capability_key == CONFIRMATORY_FINITE_ACTION_KEY
            and value.capability_version == CONFIRMATORY_FINITE_ACTION_VERSION
            and value.implementation_sha256 == config.implementation_sha256
        )
        if len(named) != 1:
            raise ValueError(
                "confirmatory provider requires one exact manifest in its complete registry"
            )
        self.manifest = named[0]
        self.records: tuple[CanonicalRecord, ...] = (
            config,
            projection,
            extension,
            scaffold,
        )
        self._runner = _ConfirmatoryFiniteActionRunner(
            manifest=self.manifest,
            config=config,
            producer=components.producer,
        )
        if components.config_decoder.config != config:
            raise ValueError("confirmatory provider config decoder differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("confirmatory provider registry/source differs")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("confirmatory provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
            and task.capability_implementation_sha256 == self.manifest.implementation_sha256
            for value in task.external_inputs
        }
        by_schema = {value.SCHEMA: value for value in self.records}
        if len(by_schema) != len(self.records):
            raise ValueError("confirmatory provider external record schemas collide")
        records: dict[str, CanonicalRecord] = {}
        for task in plan.tasks:
            if not (
                task.capability.capability_key == self.manifest.capability_key
                and task.capability.capability_version == self.manifest.capability_version
                and task.capability_implementation_sha256 == self.manifest.implementation_sha256
            ):
                continue
            records[task.capability.config.artifact_id] = self.config
            for spec in task.external_inputs:
                if spec.logical_artifact_id == task.capability.config.artifact_id:
                    continue
                expected_schema = spec.expected_payload_schema
                if expected_schema is None:
                    raise ValueError("confirmatory external input schema is unbound")
                record = by_schema.get(expected_schema)
                if record is None:
                    raise ValueError("confirmatory external input schema is unbound")
                records[spec.logical_artifact_id] = record
        if set(records) != set(specs):
            raise ValueError("confirmatory provider external input roster differs")
        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            parent = ArtifactLineageParent(
                identity=_record_identity(record),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=CONFIRMATORY_FINITE_ACTION_CONFIG_MEDIA_TYPE,
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                    ),
                    outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
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
            raise ValueError("confirmatory provider semantic registry differs")
        del execution_plan
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=record_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(record_type))),
            )
            for record_type in (
                ConfirmatoryFiniteActionResult,
                LawCandidateEvidence,
            )
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("confirmatory provider adjudication registry differs")
        del execution_plan
        return None


def compose_confirmatory_finite_action(
    *,
    config: ConfirmatoryFiniteActionConfig,
    payload_publisher: CandidatePayloadPublisher,
) -> ConfirmatoryFiniteActionComponents:
    if (
        config.method_key != CONFIRMATORY_FINITE_ACTION_KEY
        or config.method_version != CONFIRMATORY_FINITE_ACTION_VERSION
    ):
        raise ValueError("confirmatory finite-action config selects another implementation")
    manifest = confirmatory_finite_action_manifest(
        implementation_sha256=config.implementation_sha256
    )
    profile_evaluator = MethodEquivalentQualificationProfileEvaluator(
        owner=QualificationProofOwner(
            owner_id="proof-owner.confirmatory-finite-action.method-equivalent",
            capability_key=CONFIRMATORY_FINITE_ACTION_PROFILE_KEY,
            capability_version=CONFIRMATORY_FINITE_ACTION_VERSION,
            implementation_sha256=config.implementation_sha256,
        ),
        kind=MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION,
    )
    decoder = CanonicalFiniteActionCompatibilitySetDecoder()
    return ConfirmatoryFiniteActionComponents(
        capability_registry=CapabilityRegistry(
            registry_id="confirmatory-finite-action",
            capabilities=(manifest,),
        ),
        candidate_registration=CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY,
            provider_version=CONFIRMATORY_FINITE_ACTION_VERSION,
            config_media_type=CONFIRMATORY_FINITE_ACTION_CONFIG_MEDIA_TYPE,
            maximum_config_bytes=CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
        ),
        config_decoder=ConfirmatoryFiniteActionConfigDecoder(config),
        producer=ConfirmatoryFiniteActionProducer(payload_publisher),
        decoder_registry=CandidatePayloadDecoderRegistry(decoders=(decoder,)),
        profile_registry=QualificationProfileEvaluatorRegistry(evaluators=(profile_evaluator,)),
    )


def confirmatory_finite_action_local_support_profile_evaluator(
    *,
    implementation_sha256: str,
) -> MethodEquivalentQualificationProfileEvaluator:
    """Register the additive prepared-D/local-support proof owner."""

    return MethodEquivalentQualificationProfileEvaluator(
        owner=QualificationProofOwner(
            owner_id="proof-owner.confirmatory-finite-action.local-support",
            capability_key=CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT_PROFILE_KEY,
            capability_version=CONFIRMATORY_FINITE_ACTION_VERSION,
            implementation_sha256=implementation_sha256,
        ),
        kind=MethodEquivalentProfileKind.CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT,
    )


__all__ = [
    "CONFIRMATORY_FINITE_ACTION_CONFIG_MEDIA_TYPE",
    "CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES",
    "CONFIRMATORY_FINITE_ACTION_LOCAL_SUPPORT_PROFILE_KEY",
    "CONFIRMATORY_FINITE_ACTION_PROFILE_KEY",
    "CONFIRMATORY_FINITE_ACTION_PROVIDER_KEY",
    "ConfirmatoryFiniteActionComponents",
    "ConfirmatoryFiniteActionCampaignRuntimeProvider",
    "ConfirmatoryFiniteActionConfigDecoder",
    "compose_confirmatory_finite_action",
    "confirmatory_finite_action_local_support_profile_evaluator",
    "confirmatory_finite_action_manifest",
]
