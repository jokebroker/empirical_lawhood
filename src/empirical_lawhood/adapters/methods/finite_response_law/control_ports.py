"Bounded finite response-law evaluation dependencies on existing authority, receipt and prepared stores."

from typing import Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore
from empirical_lawhood.runtime.execution import DependencyReceiptBinding

CONTROL_RUNTIME_PORT = "finite-response-law.prospective-control-control-runtime"
SOURCE_CONTROL_PORT = "finite-response-law.prospective-control-source-control-store"
SEALED_CONTROL_PORT = "finite-response-law.prospective-control-sealed-control-store"
REVEAL_CONTROL_PORT = "finite-response-law.prospective-control-reveal-control-store"


class FiniteResponseLawControlRuntimePort(Protocol):
    """Worker-reconstructible binding supplied by ordinary runtime composition.

    Implementations must load the authenticated existing stores. The adapter
    does not open filesystem paths, instantiate infrastructure or grant authority.
    """

    @property
    def prepared_store(self) -> DurablePreparedExecutionEventStore: ...

    @property
    def issued_study(self) -> ObjectIdentity: ...

    @property
    def prerequisite_authority(self) -> ObjectIdentity: ...

    @property
    def grantee_id(self) -> str: ...

    @property
    def compiler_release_id(self) -> str: ...

    def execution_authority(self) -> StudyOperationAuthority: ...

    def outcome_authority(self) -> StudyOperationAuthority: ...

    def dependency_receipt(
        self, run_id: str, binding: DependencyReceiptBinding
    ) -> CanonicalTaskReceipt: ...

    def now(self) -> str: ...
