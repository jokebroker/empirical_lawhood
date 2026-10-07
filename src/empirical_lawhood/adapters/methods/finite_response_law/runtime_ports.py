"Finite response-law evaluation worker configuration over existing guarded authority/receipt/control stores.\n\nNo scientific execution, authority creation or filesystem access occurs on import\nor construction. Each operation reconstructs the existing store in its worker.\n"

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from empirical_lawhood.adapters.methods.finite_response_law.consumer import FiniteResponseLawNativeUnitReadoutMap
from empirical_lawhood.adapters.methods.finite_response_law.control_services import FiniteResponseLawControllerBindingConfig
from empirical_lawhood.adapters.methods.finite_response_law.evaluation_native_records import FiniteResponseLawEvaluationInterface
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.prepared_execution_events import (
    ExternalPreparedExecutionEventStore,
)
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.planning.nested_controller_evaluation import PreparedFutureSlot
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_compiler import CompiledLeastMagnitudeControllerStudy
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedCommonStartBinding, PreparedDesignBindingReceipt, PreparedExecutionCensus, PreparedExecutionEventPrefix, PreparedForecastParentCommitment, PreparedFutureCompletion, PreparedInstanceBindingReceipt, PreparedParentReturn
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt
from empirical_lawhood.runtime.execution import DependencyReceiptBinding

PREPARED_SCHEMAS = tuple(
    sorted(
        cls.SCHEMA
        for cls in (
            CompiledLeastMagnitudeControllerStudy,
            DeliveryControllerDecisionCommitment,
            DeliveryControllerTickReceipt,
            PreparedCommonStartBinding,
            PreparedDesignBindingReceipt,
            PreparedForecastParentCommitment,
            PreparedInstanceBindingReceipt,
            PreparedParentReturn,
            PreparedFutureSlot,
            PreparedFutureCompletion,
            PreparedExecutionCensus,
            FiniteResponseLawEvaluationInterface,
            FiniteResponseLawNativeUnitReadoutMap,
            FiniteResponseLawControllerBindingConfig,
            StudyOperationAuthority,
        )
    )
)


@dataclass(frozen=True)
class FiniteResponseLawPreparedStorePort:
    root: GuardedExternalRoot
    relative_root: str
    minimum_free_bytes: int

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_root)
        if self.minimum_free_bytes < 0:
            raise ValueError("Finite response-law evaluation prepared storage has a negative free-space floor")

    def _store(self) -> ExternalPreparedExecutionEventStore:
        return ExternalPreparedExecutionEventStore(
            ExternalArtifactPlane(self.root),
            state_root_relative_path=self.relative_root,
            record_schemas=PREPARED_SCHEMAS,
            minimum_free_bytes=self.minimum_free_bytes,
            maximum_record_bytes=16 * 1024**2,
        )

    def publish_prepared_record(
        self, *, prefix_id: str, object_id: str, record: CanonicalRecord
    ) -> ArtifactIdentity:
        return self._store().publish_prepared_record(
            prefix_id=prefix_id, object_id=object_id, record=record
        )

    def read_prepared_record(
        self, *, prefix_id: str, subject: ObjectIdentity, artifact: ArtifactIdentity
    ) -> bytes:
        return self._store().read_prepared_record(
            prefix_id=prefix_id, subject=subject, artifact=artifact
        )

    def verify_prepared_subject(
        self, *, prefix_id: str, subject: ObjectIdentity, artifact: ArtifactIdentity
    ) -> None:
        self._store().verify_prepared_subject(
            prefix_id=prefix_id, subject=subject, artifact=artifact
        )

    def load_prepared(self, prefix_id: str) -> PreparedExecutionEventPrefix:
        return self._store().load_prepared(prefix_id)

    def create_prepared(self, prefix: PreparedExecutionEventPrefix) -> None:
        self._store().create_prepared(prefix)

    def compare_and_append_prepared(
        self, *, expected_prefix_sha256: str, updated: PreparedExecutionEventPrefix
    ) -> None:
        self._store().compare_and_append_prepared(
            expected_prefix_sha256=expected_prefix_sha256, updated=updated
        )


@dataclass(frozen=True)
class FiniteResponseLawControlRuntimeBinding:
    root: GuardedExternalRoot
    run_id: str
    prepared_store: FiniteResponseLawPreparedStorePort
    issued_study: ObjectIdentity
    prerequisite_authority: ObjectIdentity
    authority: ObjectIdentity
    grantee_id: str
    compiler_release_id: str
    reveal_authority_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.run_id, self.grantee_id, self.compiler_release_id):
            validate_stable_id(value, field_name="prospective evaluation runtime identity")
        if self.prepared_store.root != self.root:
            raise ValueError("Finite response-law evaluation runtime substitutes its prepared-store root")
        if self.authority.object_schema != StudyOperationAuthority.SCHEMA:
            raise ValueError(
                "Finite response-law evaluation runtime requires an actual operation authority identity"
            )

    def execution_authority(self) -> StudyOperationAuthority:
        record = ExternalStudyOperationAuthorityStore(
            ExternalArtifactPlane(self.root)
        ).load(self.authority.object_id)
        if ObjectIdentity.from_record(record.authority_id, record) != self.authority:
            raise ValueError(
                "Finite response-law evaluation execution authority differs from its frozen runtime identity"
            )
        return record

    def outcome_authority(self) -> StudyOperationAuthority:
        from empirical_lawhood.infrastructure.study_issue import PROGRAMME_REVEAL_GRANTEE_ID

        if self.reveal_authority_id is None:
            raise PermissionError(
                "Finite response-law evaluation outcomes remain sealed without a reveal authority binding"
            )
        record = ExternalStudyOperationAuthorityStore(
            ExternalArtifactPlane(self.root)
        ).load(self.reveal_authority_id)
        require_study_authority(
            record,
            kind=StudyAuthorityKind.OUTCOME_REVEAL,
            subject=self.issued_study,
            prerequisite_authority=self.authority,
            grantee_id=PROGRAMME_REVEAL_GRANTEE_ID,
            at_utc=self.now(),
        )
        return record

    def dependency_receipt(
        self, run_id: str, binding: DependencyReceiptBinding
    ) -> CanonicalTaskReceipt:
        if run_id != self.run_id:
            raise ValueError("Finite response-law evaluation control cannot read receipts from a different run")
        receipt = ExternalTaskReceiptStore(
            ExternalArtifactPlane(self.root)
        ).read_by_receipt_id(run_id, binding.task_id, binding.receipt_id)
        if (
            receipt is None
            or receipt.receipt_id != binding.receipt_id
            or receipt.run_id != run_id
            or receipt.task_id != binding.task_id
            or not set(binding.output_materialization_ids).issubset(
                {m.materialization_id for m in receipt.output_materializations}
            )
        ):
            raise ValueError(
                "Finite response-law evaluation dependency differs from its exact committed task receipt"
            )
        return receipt

    def now(self) -> str:
        return (
            datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )


def prospective_evaluation_platform_ports(
    root: GuardedExternalRoot,
    payload_reader: Any,
    *,
    runtime: FiniteResponseLawControlRuntimeBinding,
) -> tuple[Any, ...]:
    """Distinct provider port keys, one existing authoritative store configuration."""
    from empirical_lawhood.adapters.methods.finite_response_law.control_ports import CONTROL_RUNTIME_PORT, REVEAL_CONTROL_PORT, SEALED_CONTROL_PORT, SOURCE_CONTROL_PORT
    from empirical_lawhood.adapters.methods.finite_response_law.method_provider import CANDIDATE_PAYLOAD_PORT, INPUT_RESOLVER_PORT
    from empirical_lawhood.infrastructure.candidate_sources import (
        ExternalContentAddressedInputResolver,
    )
    from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort

    values: dict[str, Any] = {
        k: runtime
        for k in (
            CONTROL_RUNTIME_PORT,
            SOURCE_CONTROL_PORT,
            SEALED_CONTROL_PORT,
            REVEAL_CONTROL_PORT,
        )
    }
    values[CANDIDATE_PAYLOAD_PORT] = payload_reader
    values[INPUT_RESOLVER_PORT] = ExternalContentAddressedInputResolver(root)
    return tuple(ExecutablePlatformPort(k, values[k]) for k in sorted(values))
