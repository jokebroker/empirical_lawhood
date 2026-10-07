"""Authenticated replay of issued extension bytes for executable factories.

This module separates three facts that must not be conflated:

* an extension was proposed and issued under exact immutable identities;
* its outcome-blind bytes can be decoded by installed, code-owned codecs; and
* an independently authorized operation may pass those decoded records to a
  provider factory.

Decoding never grants source, issue, execution, actuation, or reveal authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar, Protocol, runtime_checkable

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.experiment_entry import ProposedStudyExtension

from .capabilities import CapabilityRegistry
from .executable_bindings import CampaignRuntimeProviderResolver, ExecutableCapabilityBinding, ExecutablePlatformPort, GeneratedExecutableBindingAggregate
from .study_issue import RetrospectiveIssuedStudy, IssuedStudyMember, StudyExtensionDecoderRegistration, IssuedExecutableStudyManifest, decode_issued_extension_payload
from .providers import CampaignRuntimeProvider
from .static_codecs import CanonicalRecordCodecRegistration, CanonicalRecordCodecRegistry


MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES = 64 * 1024 * 1024


class IssuedExtensionPayloadSource(Protocol):
    """Read one already-publication-verified issued member under a byte cap."""

    def read_member(
        self,
        *,
        issue_id: str,
        member: IssuedStudyMember,
        maximum_bytes: int,
    ) -> bytes: ...


@runtime_checkable
class BatchedIssuedExtensionPayloadSource(Protocol):
    """Authenticate a bounded member set within one publication-read operation."""

    def read_members(
        self,
        *,
        issue_id: str,
        requests: tuple[tuple[IssuedStudyMember, int], ...],
    ) -> tuple[bytes, ...]: ...


@dataclass(frozen=True, slots=True)
class _PreparedPayload:
    proposed: ProposedStudyExtension
    member: IssuedStudyMember
    owner: ExecutableCapabilityBinding
    decoder: StudyExtensionDecoderRegistration
    codec: CanonicalRecordCodecRegistration
    maximum_bytes: int


def _read_payloads(
    source: IssuedExtensionPayloadSource,
    issue_id: str,
    prepared: list[_PreparedPayload],
) -> tuple[bytes, ...]:
    if isinstance(source, BatchedIssuedExtensionPayloadSource):
        payloads = source.read_members(
            issue_id=issue_id,
            requests=tuple((item.member, item.maximum_bytes) for item in prepared),
        )
    else:
        payloads = tuple(
            source.read_member(
                issue_id=issue_id, member=item.member, maximum_bytes=item.maximum_bytes
            )
            for item in prepared
        )
    if len(payloads) != len(prepared):
        raise ValueError("issued payload source returned another member count")
    return payloads


@dataclass(frozen=True, slots=True)
class IssuedExtensionPayloadReplay(CanonicalRecord):
    """Exact immutable identities and bytes observed for one issued payload."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-extension-payload-replay'

    replay_id: str
    extension_id: str
    namespace_id: str
    proposed_extension: ObjectIdentity
    issued_member: ObjectIdentity
    executable_binding: ObjectIdentity
    decoder_registration: ObjectIdentity
    canonical_codec_registration: ObjectIdentity
    decoded_record: ObjectIdentity
    byte_count: int
    physical_sha256: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    consumed: bool
    grants_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("replay_id", "extension_id", "namespace_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.proposed_extension.object_schema != ProposedStudyExtension.SCHEMA:
            raise ValueError("payload replay binds another proposed-extension schema")
        if self.issued_member.object_schema != IssuedStudyMember.SCHEMA:
            raise ValueError("payload replay binds another issued-member schema")
        if self.executable_binding.object_schema != ExecutableCapabilityBinding.SCHEMA:
            raise ValueError("payload replay binds another executable-binding schema")
        if self.decoder_registration.object_schema != StudyExtensionDecoderRegistration.SCHEMA:
            raise ValueError("payload replay binds another issued decoder schema")
        if (
            self.canonical_codec_registration.object_schema
            != CanonicalRecordCodecRegistration.SCHEMA
        ):
            raise ValueError("payload replay binds another canonical codec schema")
        if self.byte_count < 1 or self.byte_count > MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES:
            raise ValueError("payload replay byte count is outside its bound")
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.physical_sha256 != self.decoded_record.object_fingerprint:
            raise ValueError("payload replay physical and decoded identities differ")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("issued executable payload replay exceeds prospective visibility")
        if not self.consumed or self.grants_authority:
            raise ValueError("payload replay must be consumed and non-authorizing")


@dataclass(frozen=True, slots=True)
class IssuedExtensionPayloadReplaySet(CanonicalRecord):
    "Complete consumed roster for one exact issued extension set."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-extension-payload-replay-set'

    replay_set_id: str
    issued_study: ObjectIdentity
    issued_extension_set: ObjectIdentity
    executable_binding_aggregate: ObjectIdentity
    selected_binding_ids: tuple[str, ...]
    payloads: tuple[IssuedExtensionPayloadReplay, ...]
    decoded_records: tuple[ObjectIdentity, ...]
    total_bytes_read: int
    complete: bool
    all_payloads_consumed: bool
    source_read: bool
    grants_authority: bool = False
    executed: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.replay_set_id, field_name="replay_set_id")
        expected_schema = (
            RetrospectiveIssuedStudy.SCHEMA
            if isinstance(self, RetrospectivePayloadReplaySet)
            else IssuedExecutableStudyManifest.SCHEMA
        )
        if self.issued_study.object_schema != expected_schema:
            raise ValueError("replay set binds another issued programme schema")
        if (
            self.issued_extension_set.object_schema
            != 'empirical-lawhood/runtime/issued-extension-set'
        ):
            raise ValueError("replay set binds another issued extension-set schema")
        if (
            self.executable_binding_aggregate.object_schema
            != GeneratedExecutableBindingAggregate.SCHEMA
        ):
            raise ValueError("replay set binds another executable aggregate schema")
        require_sorted_unique_strings(
            self.selected_binding_ids,
            field_name="selected_binding_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.payloads,
            attribute="replay_id",
            field_name="payloads",
        )
        require_sorted_unique_ids(
            self.decoded_records,
            attribute="object_id",
            field_name="decoded_records",
        )
        if not self.payloads or len(self.payloads) != len(self.decoded_records):
            raise ValueError("replay set payload/decoded rosters differ")
        if self.total_bytes_read != sum(value.byte_count for value in self.payloads):
            raise ValueError("replay set total bytes differ from its payloads")
        if self.total_bytes_read > MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES:
            raise ValueError("replay set exceeds its total byte bound")
        if not (self.complete and self.all_payloads_consumed and self.source_read):
            raise ValueError("replay set must attest a complete consumed external read")
        if self.grants_authority or self.executed:
            raise ValueError("payload replay cannot grant authority or execute")


@dataclass(frozen=True, slots=True)
class ExecutableProviderReconstructionReceipt(CanonicalRecord):
    """Non-scientific receipt for an authorized factory reconstruction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-provider-reconstruction-receipt'

    receipt_id: str
    payload_replay_set: ObjectIdentity
    capability_registry: ObjectIdentity
    provider_registry_sha256: str
    provider_resolver_sha256: str
    reconstruction_state_sha256: str
    authorization_replayed: bool
    scientific_recomputation_count: int
    scientific_retuning_count: int
    executed: bool
    authenticated_records: tuple[ObjectIdentity, ...] = ()
    authentication_receipts: tuple[ObjectIdentity, ...] = ()
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        expected_schema = (
            RetrospectivePayloadReplaySet.SCHEMA
            if isinstance(self, RetrospectiveProviderReconstruction)
            else IssuedExtensionPayloadReplaySet.SCHEMA
        )
        if self.payload_replay_set.object_schema != expected_schema:
            raise ValueError("provider reconstruction binds another payload replay schema")
        for name in (
            "provider_registry_sha256",
            "provider_resolver_sha256",
            "reconstruction_state_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if not self.authorization_replayed:
            raise ValueError("provider reconstruction requires prior authorization replay")
        for name in ("authenticated_records", "authentication_receipts"):
            require_sorted_unique_ids(
                getattr(self, name),
                attribute="object_id",
                field_name=name,
            )
        if bool(self.authenticated_records) != bool(self.authentication_receipts):
            raise ValueError(
                "provider reconstruction authenticated records/evidence are inconsistent"
            )
        if self.scientific_recomputation_count or self.scientific_retuning_count:
            raise ValueError("provider reconstruction recomputed or retuned science")
        if self.executed or self.grants_authority:
            raise ValueError("provider reconstruction cannot execute or grant authority")


@dataclass(frozen=True, slots=True)
class ExecutableProviderRecoveryReceipt(CanonicalRecord):
    """Byte/identity equality across a fresh provider reconstruction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-provider-recovery-receipt'

    receipt_id: str
    source_reconstruction: ObjectIdentity
    recovered_reconstruction: ObjectIdentity
    source_state_sha256: str
    recovered_state_sha256: str
    provider_registry_sha256: str
    scientific_recomputation_count: int
    scientific_retuning_count: int
    recovered_without_repository_scientific_bytes: bool
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        expected_schema = (
            RetrospectiveProviderReconstruction.SCHEMA
            if isinstance(self, RetrospectiveProviderRecovery)
            else ExecutableProviderReconstructionReceipt.SCHEMA
        )
        for value in (self.source_reconstruction, self.recovered_reconstruction):
            if value.object_schema != expected_schema:
                raise ValueError("provider recovery binds another reconstruction schema")
        for name in (
            "source_state_sha256",
            "recovered_state_sha256",
            "provider_registry_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.source_state_sha256 != self.recovered_state_sha256:
            raise ValueError("provider recovery changed reconstruction-state bytes")
        if self.scientific_recomputation_count or self.scientific_retuning_count:
            raise ValueError("provider recovery recomputed or retuned science")
        if not self.recovered_without_repository_scientific_bytes or self.grants_authority:
            raise ValueError("provider recovery overclaims its custody/authority boundary")


@dataclass(frozen=True, slots=True)
class ResolvedIssuedExtensionPayloads:
    records: tuple[CanonicalRecord, ...]
    replay_set: IssuedExtensionPayloadReplaySet

    def __post_init__(self) -> None:
        identities = tuple(
            ObjectIdentity.from_record(identity.object_id, record)
            for identity, record in zip(
                self.replay_set.decoded_records,
                self.records,
                strict=True,
            )
        )
        if identities != self.replay_set.decoded_records:
            raise ValueError("resolved issued records differ from their replay identities")


@dataclass(frozen=True, slots=True)
class ResolvedIssuedContractEvidenceInputs:
    """Bounded exact issued records selected for preexecution evidence only."""

    records: tuple[CanonicalRecord, ...]
    payloads: tuple[IssuedExtensionPayloadReplay, ...]
    total_bytes_read: int

    def __post_init__(self) -> None:
        if not self.records or len(self.records) != len(self.payloads):
            raise ValueError("contract evidence input records/replays differ")
        if self.total_bytes_read != sum(value.byte_count for value in self.payloads):
            raise ValueError("contract evidence input byte total differs")
        if tuple(value.SCHEMA for value in self.records) != tuple(
            sorted(value.SCHEMA for value in self.records)
        ):
            raise ValueError("contract evidence input records are not schema-sorted")


@dataclass(frozen=True, slots=True)
class AuthenticatedExecutableRecord:
    """One separately authenticated source/input record passed to a factory.

    The caller must replay the named qualification receipts through their
    owning service before constructing this wrapper.  Reconstruction records
    only their exact identities; it never embeds the scientific payload.
    """

    identity: ObjectIdentity
    record: CanonicalRecord
    authentication_receipts: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        if ObjectIdentity.from_record(self.identity.object_id, self.record) != self.identity:
            raise ValueError("authenticated executable record identity differs from its bytes")
        require_sorted_unique_ids(
            self.authentication_receipts,
            attribute="object_id",
            field_name="authentication_receipts",
        )
        if not self.authentication_receipts:
            raise ValueError("authenticated executable record lacks qualification evidence")


@dataclass(frozen=True, slots=True)
class ReconstructedExecutableProvider:
    provider: CampaignRuntimeProvider
    payloads: ResolvedIssuedExtensionPayloads
    receipt: ExecutableProviderReconstructionReceipt


class IssuedExtensionPayloadResolver:
    """Replay one complete issued set against selected installed bindings."""

    def __init__(
        self,
        *,
        source: IssuedExtensionPayloadSource,
        codecs: CanonicalRecordCodecRegistry,
        maximum_total_bytes: int = MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES,
    ) -> None:
        if not 0 < maximum_total_bytes <= MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES:
            raise ValueError("issued extension total byte limit is invalid")
        self.source = source
        self.codecs = codecs
        self.maximum_total_bytes = maximum_total_bytes

    def resolve(
        self,
        *,
        manifest: IssuedExecutableStudyManifest,
        aggregate: GeneratedExecutableBindingAggregate,
        selected_binding_ids: tuple[str, ...],
    ) -> ResolvedIssuedExtensionPayloads:
        require_sorted_unique_strings(
            selected_binding_ids,
            field_name="selected_binding_ids",
            allow_empty=False,
        )
        selected = tuple(
            next(
                (value for value in aggregate.bindings if value.binding_id == binding_id),
                None,
            )
            for binding_id in selected_binding_ids
        )
        if any(value is None for value in selected):
            raise ValueError("selected executable binding is not in the generated aggregate")
        bindings = tuple(value for value in selected if value is not None)
        schema_owners: dict[str, ExecutableCapabilityBinding] = {}
        required_schemas: set[str] = set()
        for binding in bindings:
            required_schemas.update(binding.required_issued_payload_schemas)
            for decoder in binding.issued_decoder_registrations:
                schema = decoder.payload_schema
                if schema in schema_owners:
                    raise ValueError(
                        "selected executable bindings overlap issued payload ownership"
                    )
                schema_owners[schema] = binding
        if not schema_owners:
            raise ValueError("selected executable bindings require no issued payloads")

        proposed_set = manifest.candidate.authoring_package.extension_set
        issued = manifest.issued_extensions
        if issued.proposed_extension_set != ObjectIdentity.from_record(
            proposed_set.extension_set_id,
            proposed_set,
        ):
            raise ValueError("issued extension set differs from the candidate proposal")
        proposed_by_id = {value.extension_id: value for value in proposed_set.extensions}
        members_by_identity = {
            ObjectIdentity.from_record(value.member_id, value): value for value in issued.members
        }
        if len(proposed_by_id) != len(proposed_set.extensions) or (
            len(members_by_identity) != len(issued.members)
        ):
            raise ValueError("issued extension roster is duplicated")

        records: list[CanonicalRecord] = []
        replays: list[IssuedExtensionPayloadReplay] = []
        consumed_schemas: set[str] = set()
        consumed_members: set[ObjectIdentity] = set()
        total_bytes = 0
        prepared: list[_PreparedPayload] = []
        for issued_binding in issued.bindings:
            proposed = proposed_by_id.get(issued_binding.extension_id)
            member = members_by_identity.get(issued_binding.issued_member)
            if (
                proposed is None
                or member is None
                or issued_binding.proposed_extension
                != ObjectIdentity.from_record(proposed.extension_id, proposed)
            ):
                raise ValueError("issued payload binding differs from its proposal/member")
            owner = schema_owners.get(proposed.payload.object_schema)
            if owner is None:
                raise ValueError("issued payload is unconsumed by selected executable bindings")
            if proposed.payload.object_schema in consumed_schemas:
                raise ValueError("issued payload schema is duplicated")
            consumed_schemas.add(proposed.payload.object_schema)
            if issued_binding.issued_member in consumed_members:
                raise ValueError("issued member is consumed more than once")
            consumed_members.add(issued_binding.issued_member)

            matched_decoder = next(
                (
                    value
                    for value in owner.issued_decoder_registrations
                    if ObjectIdentity.from_record(value.registration_id, value)
                    == issued_binding.decoder_registration
                ),
                None,
            )
            if matched_decoder is None or not owner.accepts_issued_decoder_registration(
                matched_decoder
            ):
                raise ValueError("issued decoder differs from the executable binding")
            codec = self.codecs.registration(proposed.payload.object_schema)
            if (
                codec.record_version != proposed.payload.object_version
                or codec.maximum_bytes > matched_decoder.maximum_payload_bytes
                or codec.decoder_implementation_sha256 != matched_decoder.implementation_sha256
            ):
                raise ValueError("canonical codec differs from the issued decoder binding")
            maximum_bytes = min(
                self.maximum_total_bytes - total_bytes,
                matched_decoder.maximum_payload_bytes,
                codec.maximum_bytes,
            )
            if maximum_bytes < member.size_bytes:
                raise ValueError("issued payload exceeds its aggregate or decoder byte bound")
            prepared.append(
                _PreparedPayload(proposed, member, owner, matched_decoder, codec, maximum_bytes)
            )
            total_bytes += member.size_bytes

        if consumed_members != set(members_by_identity):
            raise ValueError("issued extension member roster has missing or extra payloads")
        if not required_schemas.issubset(consumed_schemas):
            raise ValueError("selected executable binding payload roster is incomplete")

        payloads = _read_payloads(self.source, manifest.issue_id, prepared)
        for item, payload in zip(prepared, payloads, strict=True):
            proposed, member, owner = item.proposed, item.member, item.owner
            matched_decoder, codec = item.decoder, item.codec
            physical_sha256 = hashlib.sha256(payload).hexdigest()
            if (
                len(payload) != member.size_bytes
                or len(payload) != proposed.payload_size_bytes
                or physical_sha256 != member.physical_sha256
                or physical_sha256 != member.logical_content_sha256
                or physical_sha256 != proposed.payload.object_fingerprint
            ):
                raise ValueError("issued extension physical bytes differ from immutable identity")
            record_type = self.codecs.record_types[proposed.payload.object_schema]
            record = decode_issued_extension_payload(
                payload=payload,
                proposed=proposed,
                registration=matched_decoder,
                record_type=record_type,
            )
            decoded_identity = ObjectIdentity.from_record(proposed.payload.object_id, record)
            if decoded_identity != proposed.payload or record.canonical_bytes() != payload:
                raise ValueError("decoded extension record differs from its proposed identity")
            records.append(record)
            replays.append(
                IssuedExtensionPayloadReplay(
                    replay_id=f"issued-payload-replay.{proposed.extension_id}",
                    extension_id=proposed.extension_id,
                    namespace_id=proposed.namespace_id,
                    proposed_extension=ObjectIdentity.from_record(
                        proposed.extension_id,
                        proposed,
                    ),
                    issued_member=ObjectIdentity.from_record(member.member_id, member),
                    executable_binding=ObjectIdentity.from_record(owner.binding_id, owner),
                    decoder_registration=ObjectIdentity.from_record(
                        matched_decoder.registration_id,
                        matched_decoder,
                    ),
                    canonical_codec_registration=ObjectIdentity.from_record(
                        codec.codec_id,
                        codec,
                    ),
                    decoded_record=decoded_identity,
                    byte_count=len(payload),
                    physical_sha256=physical_sha256,
                    outcome_access=issued.outcome_access,
                    visibility_ceiling=issued.visibility_ceiling,
                    consumed=True,
                )
            )

        ordered_replays = tuple(sorted(replays, key=lambda value: value.replay_id))
        ordered_record_pairs = tuple(
            sorted(
                (
                    (ObjectIdentity.from_record(replay.decoded_record.object_id, record), record)
                    for replay, record in zip(replays, records, strict=True)
                ),
                key=lambda value: value[0].object_id,
            )
        )
        ordered_records = tuple(value[1] for value in ordered_record_pairs)
        ordered_record_identities = tuple(value[0] for value in ordered_record_pairs)
        replay_type = (
            RetrospectivePayloadReplaySet
            if isinstance(manifest, RetrospectiveIssuedStudy)
            else IssuedExtensionPayloadReplaySet
        )
        replay_set = replay_type(
            replay_set_id=f"issued-payload-replay-set.{manifest.issue_id}",
            issued_study=ObjectIdentity.from_record(manifest.issue_id, manifest),
            issued_extension_set=ObjectIdentity.from_record(
                issued.issued_extension_set_id,
                issued,
            ),
            executable_binding_aggregate=ObjectIdentity.from_record(
                aggregate.aggregate_id,
                aggregate,
            ),
            selected_binding_ids=selected_binding_ids,
            payloads=ordered_replays,
            decoded_records=ordered_record_identities,
            total_bytes_read=total_bytes,
            complete=True,
            all_payloads_consumed=True,
            source_read=True,
        )
        return ResolvedIssuedExtensionPayloads(
            records=ordered_records,
            replay_set=replay_set,
        )

    def resolve_declared_records(
        self,
        *,
        manifest: IssuedExecutableStudyManifest,
        aggregate: GeneratedExecutableBindingAggregate,
        required_schema_ids: tuple[str, ...],
    ) -> ResolvedIssuedContractEvidenceInputs:
        """Replay selected declared payloads through sole generated decoder owners.

        Unlike provider reconstruction, this route does not imply that every
        selected binding is being activated.  It authenticates only the typed
        evidence inputs and constructs no provider.
        """

        require_sorted_unique_strings(
            required_schema_ids,
            field_name="required_schema_ids",
            allow_empty=False,
        )
        proposed_set = manifest.candidate.authoring_package.extension_set
        issued = manifest.issued_extensions
        proposed_by_schema = {
            value.payload.object_schema: value for value in proposed_set.extensions
        }
        if len(proposed_by_schema) != len(proposed_set.extensions):
            raise ValueError("issued contract-evidence payload schemas are duplicated")
        issued_bindings = {value.proposed_extension: value for value in issued.bindings}
        members = {
            ObjectIdentity.from_record(value.member_id, value): value for value in issued.members
        }
        records: list[CanonicalRecord] = []
        replays: list[IssuedExtensionPayloadReplay] = []
        total_bytes = 0
        prepared: list[_PreparedPayload] = []
        for schema in required_schema_ids:
            proposed = proposed_by_schema.get(schema)
            if proposed is None:
                raise ValueError("issued contract-evidence payload is missing")
            proposed_identity = ObjectIdentity.from_record(proposed.extension_id, proposed)
            issued_binding = issued_bindings.get(proposed_identity)
            if issued_binding is None:
                raise ValueError("issued contract-evidence proposal is unbound")
            member = members.get(issued_binding.issued_member)
            if member is None:
                raise ValueError("issued contract-evidence member is missing")
            owners = tuple(
                binding
                for binding in aggregate.bindings
                if any(
                    decoder.payload_schema == schema
                    and ObjectIdentity.from_record(decoder.registration_id, decoder)
                    == issued_binding.decoder_registration
                    for decoder in binding.issued_decoder_registrations
                )
            )
            if not owners:
                raise ValueError(
                    "issued contract-evidence payload lacks one generated decoder owner"
                )
            # Every match names the same full issued decoder identity. A
            # compiler/runtime pair can consume that one canonical codec;
            # selected runtime ownership is checked separately by resolve().
            owner = owners[0]
            decoder = next(
                value
                for value in owner.issued_decoder_registrations
                if value.payload_schema == schema
            )
            codec = self.codecs.registration(schema)
            if (
                codec.record_version != proposed.payload.object_version
                or codec.maximum_bytes > decoder.maximum_payload_bytes
                or codec.decoder_implementation_sha256 != decoder.implementation_sha256
                or not owner.accepts_issued_decoder_registration(decoder)
            ):
                raise ValueError("contract-evidence codec differs from its issued decoder")
            maximum_bytes = min(
                self.maximum_total_bytes - total_bytes,
                decoder.maximum_payload_bytes,
                codec.maximum_bytes,
            )
            if maximum_bytes < member.size_bytes:
                raise ValueError("issued contract-evidence payload exceeds its byte bound")
            prepared.append(
                _PreparedPayload(proposed, member, owner, decoder, codec, maximum_bytes)
            )
            total_bytes += member.size_bytes

        payloads = _read_payloads(self.source, manifest.issue_id, prepared)
        for item, payload in zip(prepared, payloads, strict=True):
            proposed, member, owner = item.proposed, item.member, item.owner
            decoder, codec = item.decoder, item.codec
            proposed_identity = ObjectIdentity.from_record(proposed.extension_id, proposed)
            physical_sha256 = hashlib.sha256(payload).hexdigest()
            if (
                len(payload) != member.size_bytes
                or len(payload) != proposed.payload_size_bytes
                or physical_sha256 != member.physical_sha256
                or physical_sha256 != member.logical_content_sha256
                or physical_sha256 != proposed.payload.object_fingerprint
            ):
                raise ValueError("issued contract-evidence bytes differ from custody")
            record = decode_issued_extension_payload(
                payload=payload,
                proposed=proposed,
                registration=decoder,
                record_type=self.codecs.record_types[proposed.payload.object_schema],
            )
            decoded_identity = ObjectIdentity.from_record(proposed.payload.object_id, record)
            if decoded_identity != proposed.payload or record.canonical_bytes() != payload:
                raise ValueError("issued contract-evidence record differs from its identity")
            records.append(record)
            replays.append(
                IssuedExtensionPayloadReplay(
                    replay_id=f"contract-evidence-replay.{proposed.extension_id}",
                    extension_id=proposed.extension_id,
                    namespace_id=proposed.namespace_id,
                    proposed_extension=proposed_identity,
                    issued_member=ObjectIdentity.from_record(member.member_id, member),
                    executable_binding=ObjectIdentity.from_record(owner.binding_id, owner),
                    decoder_registration=ObjectIdentity.from_record(
                        decoder.registration_id,
                        decoder,
                    ),
                    canonical_codec_registration=ObjectIdentity.from_record(
                        codec.codec_id,
                        codec,
                    ),
                    decoded_record=decoded_identity,
                    byte_count=len(payload),
                    physical_sha256=physical_sha256,
                    outcome_access=issued.outcome_access,
                    visibility_ceiling=issued.visibility_ceiling,
                    consumed=True,
                )
            )
        pairs = tuple(sorted(zip(records, replays, strict=True), key=lambda value: value[0].SCHEMA))
        return ResolvedIssuedContractEvidenceInputs(
            records=tuple(value[0] for value in pairs),
            payloads=tuple(value[1] for value in pairs),
            total_bytes_read=total_bytes,
        )


def reconstruct_executable_provider(
    *,
    registry: CapabilityRegistry,
    manifest: IssuedExecutableStudyManifest,
    aggregate: GeneratedExecutableBindingAggregate,
    selected_binding_ids: tuple[str, ...],
    payload_resolver: IssuedExtensionPayloadResolver,
    provider_resolver: CampaignRuntimeProviderResolver,
    authenticated_records: tuple[AuthenticatedExecutableRecord, ...] = (),
    platform_ports: tuple[ExecutablePlatformPort, ...] = (),
    authorization_replayed: bool,
) -> ReconstructedExecutableProvider:
    """Construct only after the caller independently replayed operation authority."""

    # This helper is also a security boundary in its own right.  Do not rely on
    # a facade/execution caller to have checked the flag before the externally
    # guarded payload source is touched: a direct caller must observe the same
    # zero-read authority stop.
    if not authorization_replayed:
        raise PermissionError("provider construction requires prior authorization replay")
    resolved = payload_resolver.resolve(
        manifest=manifest,
        aggregate=aggregate,
        selected_binding_ids=selected_binding_ids,
    )
    selected = tuple(
        next(value for value in aggregate.bindings if value.binding_id == binding_id)
        for binding_id in selected_binding_ids
    )
    expected_authenticated_schemas = {
        schema for binding in selected for schema in binding.required_authenticated_record_schemas
    }
    authenticated_schemas = tuple(value.record.SCHEMA for value in authenticated_records)
    if tuple(sorted(set(authenticated_schemas))) != authenticated_schemas:
        raise ValueError("authenticated executable records must have sorted unique schemas")
    if set(authenticated_schemas) != expected_authenticated_schemas:
        raise ValueError("authenticated executable record roster differs from its bindings")
    if set(authenticated_schemas) & {value.SCHEMA for value in resolved.records}:
        raise ValueError("issued and separately authenticated executable records overlap")
    construction_records = tuple(
        sorted(
            (
                *resolved.records,
                *(value.record for value in authenticated_records),
            ),
            key=lambda value: value.SCHEMA,
        )
    )
    provider = provider_resolver.resolve(
        registry,
        decoded_records=construction_records,
        platform_ports=platform_ports,
    )
    if provider.registry_sha256 != registry.fingerprint():
        raise ValueError("reconstructed provider registry identity differs")
    state_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "payload_replay_set": ObjectIdentity.from_record(
                    resolved.replay_set.replay_set_id,
                    resolved.replay_set,
                ),
                "provider_registry_sha256": provider.registry_sha256,
                "provider_resolver_sha256": provider_resolver.resolver_fingerprint,
                "authenticated_records": tuple(
                    sorted(
                        (value.identity for value in authenticated_records),
                        key=lambda value: value.object_id,
                    )
                ),
                "authentication_receipts": tuple(
                    sorted(
                        {
                            receipt
                            for value in authenticated_records
                            for receipt in value.authentication_receipts
                        },
                        key=lambda value: value.object_id,
                    )
                ),
            }
        )
    ).hexdigest()
    authenticated_identities = tuple(
        sorted(
            (value.identity for value in authenticated_records),
            key=lambda value: value.object_id,
        )
    )
    authentication_receipts = tuple(
        sorted(
            {
                receipt
                for value in authenticated_records
                for receipt in value.authentication_receipts
            },
            key=lambda value: value.object_id,
        )
    )
    receipt_type = (
        RetrospectiveProviderReconstruction
        if isinstance(resolved.replay_set, RetrospectivePayloadReplaySet)
        else ExecutableProviderReconstructionReceipt
    )
    receipt = receipt_type(
        receipt_id=f"provider-reconstruction.{manifest.issue_id}",
        payload_replay_set=ObjectIdentity.from_record(
            resolved.replay_set.replay_set_id,
            resolved.replay_set,
        ),
        capability_registry=ObjectIdentity.from_record(registry.registry_id, registry),
        provider_registry_sha256=provider.registry_sha256,
        provider_resolver_sha256=provider_resolver.resolver_fingerprint,
        reconstruction_state_sha256=state_sha256,
        authorization_replayed=True,
        scientific_recomputation_count=0,
        scientific_retuning_count=0,
        executed=False,
        authenticated_records=authenticated_identities,
        authentication_receipts=authentication_receipts,
    )
    return ReconstructedExecutableProvider(
        provider=provider,
        payloads=resolved,
        receipt=receipt,
    )


def verify_executable_provider_recovery(
    source: ExecutableProviderReconstructionReceipt,
    recovered: ExecutableProviderReconstructionReceipt,
) -> ExecutableProviderRecoveryReceipt:
    """Prove a fresh reconstruction retained exact payload/provider state."""

    if (
        source.payload_replay_set != recovered.payload_replay_set
        or source.capability_registry != recovered.capability_registry
        or source.provider_registry_sha256 != recovered.provider_registry_sha256
        or source.provider_resolver_sha256 != recovered.provider_resolver_sha256
        or source.authenticated_records != recovered.authenticated_records
        or source.authentication_receipts != recovered.authentication_receipts
        or source.reconstruction_state_sha256 != recovered.reconstruction_state_sha256
    ):
        raise ValueError("fresh provider reconstruction differs from its source state")
    receipt_type = (
        RetrospectiveProviderRecovery
        if isinstance(source, RetrospectiveProviderReconstruction)
        else ExecutableProviderRecoveryReceipt
    )
    return receipt_type(
        receipt_id=f"provider-recovery.{source.receipt_id}",
        source_reconstruction=ObjectIdentity.from_record(source.receipt_id, source),
        recovered_reconstruction=ObjectIdentity.from_record(
            recovered.receipt_id,
            recovered,
        ),
        source_state_sha256=source.reconstruction_state_sha256,
        recovered_state_sha256=recovered.reconstruction_state_sha256,
        provider_registry_sha256=source.provider_registry_sha256,
        scientific_recomputation_count=0,
        scientific_retuning_count=0,
        recovered_without_repository_scientific_bytes=True,
    )


@dataclass(frozen=True, slots=True)
class RetrospectivePayloadReplaySet(IssuedExtensionPayloadReplaySet):
    """Exact historical issue identity; replay grants no scientific authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-payload-replay-set'


@dataclass(frozen=True, slots=True)
class RetrospectiveProviderReconstruction(ExecutableProviderReconstructionReceipt):
    """Reconstruct the historical control records without promoting exposure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-provider-reconstruction'


@dataclass(frozen=True, slots=True)
class RetrospectiveProviderRecovery(ExecutableProviderRecoveryReceipt):
    """Same historical reconstruction identity, with zero scientific reruns."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-provider-recovery'


__all__ = [
    "AuthenticatedExecutableRecord",
    'ExecutableProviderReconstructionReceipt',
    'ExecutableProviderRecoveryReceipt',
    'IssuedExtensionPayloadReplaySet',
    'IssuedExtensionPayloadReplay',
    'IssuedExtensionPayloadResolver',
    "IssuedExtensionPayloadSource",
    "MAX_EXECUTABLE_ISSUED_EXTENSION_BYTES",
    "ReconstructedExecutableProvider",
    "ResolvedIssuedContractEvidenceInputs",
    "ResolvedIssuedExtensionPayloads",
    'reconstruct_executable_provider',
    'verify_executable_provider_recovery',
]
