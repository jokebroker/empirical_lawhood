"""Static composition for the production finite-action identification method."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceProjection
from empirical_lawhood.planning.identification_evidence_extensions import IdentificationEvidenceProjectionExtension
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import LawCandidateEvidence
from .finite_action_identification import FiniteActionCandidateScaffold, FiniteActionIdentificationConfig, FiniteActionIdentificationProducer, FiniteActionIdentificationResult
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


FINITE_ACTION_IDENTIFICATION_KEY = "finite-action.compatibility-set"
FINITE_ACTION_IDENTIFICATION_VERSION = "1.0.0"
FINITE_ACTION_PROVIDER_KEY = "finite-action.identification-provider"
FINITE_ACTION_PROFILE_KEY = "qualification-profile.finite-action"
FINITE_ACTION_CONFIG_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
FINITE_ACTION_MAXIMUM_CONFIG_BYTES = 512 * 1024


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def finite_action_identification_manifest(*, implementation_sha256: str) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=FINITE_ACTION_IDENTIFICATION_KEY,
        capability_version=FINITE_ACTION_IDENTIFICATION_VERSION,
        kind=CapabilityKind.LAW_IDENTIFIER,
        config_schema=FiniteActionIdentificationConfig.SCHEMA,
        config_schema_sha256=_schema_sha256(FiniteActionIdentificationConfig.SCHEMA),
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
            sorted(
                (
                    FiniteActionIdentificationResult.SCHEMA,
                    LawCandidateEvidence.SCHEMA,
                )
            )
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
        runtime_id="cpython-3.11-finite-action",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "complete-cell-roster",
            "complete-physical-unit-resampling",
            "exact-action-comparator-order",
            "exact-projection-companion",
            "native-unit-frame-clock-preservation",
            "partial-invalid-unevaluable-preservation",
            "sole-terminal-law-finalizer",
        ),
        implementation_sha256=implementation_sha256,
    )


@dataclass(frozen=True, slots=True)
class FiniteActionIdentificationRegistration(CanonicalRecord):
    """Exact config, implementation, decoder and proof-profile binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-action-identification-registration'

    registration_id: str
    capability_manifest: ObjectIdentity
    config: ObjectIdentity
    qualification_profile: ObjectIdentity
    decoder_schema: str
    decoder_version: str
    payload_schema: str
    evaluator_key: str

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_semantic_version(self.decoder_version)
        validate_stable_id(self.evaluator_key, field_name="evaluator_key")
        if self.capability_manifest.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("finite-action registration requires a capability manifest")
        if self.config.object_schema != FiniteActionIdentificationConfig.SCHEMA:
            raise ValueError("finite-action registration requires the exact method config")
        if self.qualification_profile.object_schema != 'empirical-lawhood/methods/component-qualification-profile':
            raise ValueError("finite-action registration requires a qualification profile")
        decoder = CanonicalFiniteActionCompatibilitySetDecoder()
        if (
            self.decoder_schema != decoder.decoder_schema
            or self.decoder_version != decoder.decoder_version
            or self.payload_schema != decoder.payload_schema
        ):
            raise ValueError("finite-action registration changes its canonical decoder")
        if self.evaluator_key != "canonical-finite-action-law":
            raise ValueError("finite-action registration changes its support-limited evaluator")


@dataclass(frozen=True, slots=True)
class FiniteActionIdentificationConfigDecoder:
    """Adapter-owned, exact-config decoder for candidate composition."""

    config: FiniteActionIdentificationConfig
    provider_key: str = FINITE_ACTION_PROVIDER_KEY
    provider_version: str = FINITE_ACTION_IDENTIFICATION_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != FiniteActionIdentificationConfig.SCHEMA:
            raise ValueError("finite-action config schema differs")
        decoded = decode_canonical_bytes(
            payload,
            FiniteActionIdentificationConfig,
            maximum_bytes=FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
        )
        if decoded != self.config:
            raise ValueError("finite-action config differs from the exact registration")


@dataclass(frozen=True, slots=True)
class FiniteActionIdentificationComponents:
    """Executable objects statically composed around one exact method config."""

    registration: FiniteActionIdentificationRegistration
    capability_registry: CapabilityRegistry
    candidate_registration: CandidateCapabilityRegistration
    config_decoder: FiniteActionIdentificationConfigDecoder
    producer: FiniteActionIdentificationProducer
    decoder_registry: CandidatePayloadDecoderRegistry
    profile_registry: QualificationProfileEvaluatorRegistry


def compose_finite_action_identification(
    *,
    config: FiniteActionIdentificationConfig,
    payload_publisher: CandidatePayloadPublisher,
) -> FiniteActionIdentificationComponents:
    """Bind the production method without config-selected Python objects."""

    if (
        config.method_key != FINITE_ACTION_IDENTIFICATION_KEY
        or config.method_version != FINITE_ACTION_IDENTIFICATION_VERSION
    ):
        raise ValueError("finite-action config selects another registered implementation")
    manifest = finite_action_identification_manifest(
        implementation_sha256=config.implementation_sha256
    )
    owner = QualificationProofOwner(
        owner_id="proof-owner.finite-action.method-equivalent",
        capability_key=FINITE_ACTION_PROFILE_KEY,
        capability_version=FINITE_ACTION_IDENTIFICATION_VERSION,
        implementation_sha256=config.implementation_sha256,
    )
    profile_evaluator = MethodEquivalentQualificationProfileEvaluator(
        owner=owner,
        kind=MethodEquivalentProfileKind.FINITE_ACTION,
    )
    decoder = CanonicalFiniteActionCompatibilitySetDecoder()
    registration = FiniteActionIdentificationRegistration(
        registration_id="registration.finite-action-identification",
        capability_manifest=ObjectIdentity.from_record(manifest.capability_key, manifest),
        config=ObjectIdentity.from_record(config.config_id, config),
        qualification_profile=ObjectIdentity.from_record(
            profile_evaluator.profile.profile_id,
            profile_evaluator.profile,
        ),
        decoder_schema=decoder.decoder_schema,
        decoder_version=decoder.decoder_version,
        payload_schema=decoder.payload_schema,
        evaluator_key="canonical-finite-action-law",
    )
    candidate_registration = CandidateCapabilityRegistration(
        manifest=manifest,
        provider_key=FINITE_ACTION_PROVIDER_KEY,
        provider_version=FINITE_ACTION_IDENTIFICATION_VERSION,
        config_media_type=FINITE_ACTION_CONFIG_MEDIA_TYPE,
        maximum_config_bytes=FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
    )
    return FiniteActionIdentificationComponents(
        registration=registration,
        capability_registry=CapabilityRegistry(
            registry_id="finite-action-identification",
            capabilities=(manifest,),
        ),
        candidate_registration=candidate_registration,
        config_decoder=FiniteActionIdentificationConfigDecoder(config),
        producer=FiniteActionIdentificationProducer(payload_publisher),
        decoder_registry=CandidatePayloadDecoderRegistry(decoders=(decoder,)),
        profile_registry=QualificationProfileEvaluatorRegistry(evaluators=(profile_evaluator,)),
    )


__all__ = [
    "FINITE_ACTION_CONFIG_MEDIA_TYPE",
    "FINITE_ACTION_IDENTIFICATION_KEY",
    "FINITE_ACTION_IDENTIFICATION_VERSION",
    "FINITE_ACTION_MAXIMUM_CONFIG_BYTES",
    "FINITE_ACTION_PROFILE_KEY",
    "FINITE_ACTION_PROVIDER_KEY",
    "FiniteActionIdentificationComponents",
    "FiniteActionIdentificationConfigDecoder",
    'FiniteActionIdentificationRegistration',
    "compose_finite_action_identification",
    "finite_action_identification_manifest",
]
