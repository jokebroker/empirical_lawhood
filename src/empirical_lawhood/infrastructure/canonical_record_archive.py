"""Bounded immutable archives over the existing guarded prepared-record store."""

from dataclasses import dataclass
from hashlib import sha256
from typing import TypeVar, Protocol
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive, MAX_ARCHIVED_RECORD_BYTES
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerTickReceipt
from empirical_lawhood.runtime.controller_evaluation_trajectory import TrajectoryCallbackLink
from .prepared_execution_events import ExternalPreparedExecutionEventStore

RecordT = TypeVar("RecordT", bound=CanonicalRecord)


class CanonicalArchiveBudget(Protocol):
    def reserve(self, relative_path: str, payload_bytes: int) -> None: ...


@dataclass(frozen=True)
class ExternalCanonicalRecordArchive:
    store: ExternalPreparedExecutionEventStore
    prefix_id: str
    publication_budget: CanonicalArchiveBudget | None = None

    def _prefix(self, object_id: str) -> str:
        # Bound directory entries on the declared VFAT external medium.
        return f"{self.prefix_id}.{sha256(object_id.encode()).hexdigest()[:2]}"

    def publish_record(self, object_id: str, record: CanonicalRecord) -> ArtifactIdentity:
        physical = (
            record
            if record.SCHEMA in self.store.record_schemas
            and record.SCHEMA != CanonicalRecordArchive.SCHEMA
            else CanonicalRecordArchive.pack(object_id, record)
        )
        if self.publication_budget is not None:
            if len(object_id) > 192 or len(physical.SCHEMA) > 192:
                raise ValueError("bounded archive identity exceeds its quota manifest proof")
            self.publication_budget.reserve(
                self.store._record_path(self._prefix(object_id), object_id),
                len(physical.canonical_bytes()),
            )
        # The returned artifact identifies physical archive bytes, while its
        # subject carries the original scientific identity.
        return self.store.publish_prepared_record(
            prefix_id=self._prefix(object_id), object_id=object_id, record=physical
        )

    def read_record(self, subject: ObjectIdentity, record_type: type[RecordT]) -> RecordT:
        payload, object_id, schema = self.store._read(
            self.store._record_path(self._prefix(subject.object_id), subject.object_id),
            maximum_bytes=MAX_ARCHIVED_RECORD_BYTES,
        )
        if schema in self.store.record_schemas and schema != CanonicalRecordArchive.SCHEMA:
            if record_type.SCHEMA != schema or object_id != subject.object_id:
                raise ValueError("direct record locator/schema differs")
            result = decode_canonical_bytes(
                payload, record_type, maximum_bytes=MAX_ARCHIVED_RECORD_BYTES
            )
            if ObjectIdentity.from_record(subject.object_id, result) != subject:
                raise ValueError("direct record identity differs")
            return result
        archive = decode_canonical_bytes(
            payload, CanonicalRecordArchive, maximum_bytes=MAX_ARCHIVED_RECORD_BYTES
        )
        if (
            object_id != subject.object_id
            or schema != archive.SCHEMA
            or archive.subject != subject
            or subject.object_schema != record_type.SCHEMA
        ):
            raise ValueError("archived record locator/subject/schema differs")
        result = decode_canonical_bytes(
            archive.unpack(), record_type, maximum_bytes=MAX_ARCHIVED_RECORD_BYTES
        )
        if ObjectIdentity.from_record(subject.object_id, result) != subject:
            raise ValueError("archived record logical identity differs")
        return result

    def read_compiled(self, identity: ObjectIdentity) -> CompiledDeliveryControllerStudy:
        return self.read_record(identity, CompiledDeliveryControllerStudy)

    def read_tick(self, identity: ObjectIdentity) -> DeliveryControllerTickReceipt:
        return self.read_record(identity, DeliveryControllerTickReceipt)

    def read_link(self, identity: ObjectIdentity) -> TrajectoryCallbackLink:
        return self.read_record(identity, TrajectoryCallbackLink)
