"Receipt-bound import of all original finite response-law evaluation preparations, with no native work."

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.source_qualification import ProspectiveRetainedPredecessor
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import validate_retained_qualification_receipt

from .assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from .contracts import native_invocations
from .evaluation_contracts import FiniteResponseLawEvaluationConfig
from .native_artifact import NATIVE_PAIR_SCHEMA
from .source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawEvaluationTaskResult


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationRetention(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-retention'
    )
    source: FiniteResponseLawEvaluationConfig | FiniteResponseLawAssignedEvaluationConfig
    donor_run_id: str
    donor_closeout: ArtifactIdentity
    prefixes: tuple[ProspectiveRetainedPredecessor, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.donor_run_id, field_name="donor_run_id")
        assigned = type(self.source) is FiniteResponseLawAssignedEvaluationConfig
        if type(self.source) not in (
            FiniteResponseLawEvaluationConfig,
            FiniteResponseLawAssignedEvaluationConfig,
        ) or (
            self.donor_closeout.payload_schema != 'empirical-lawhood/document/json'
            or self.donor_closeout.media_type != "application/json"
            or self.donor_closeout.size_bytes <= 0
        ):
            raise ValueError("Finite response-law evaluation continuation requires the exact failed donor closeout")
        native = tuple(
            t for t in native_invocations(self.source) if t.phase == "prefix"
        )
        if len(self.prefixes) != 64 or tuple(
            p.segment_id for p in self.prefixes
        ) != tuple(t.task_id for t in native):
            raise ValueError("Finite response-law evaluation continuation must retain all 64 original preparations")
        for prior, invocation in zip(self.prefixes, native, strict=True):
            root = invocation.root
            if (
                type(prior) is not ProspectiveRetainedPredecessor
                or prior.physical_independent_unit_id != root.physical_unit_id
                or prior.native_clock_id != "finite-response-law.reference-clock"
                or prior.end != Decimal(invocation.clocks[1])
                or prior.view_ids
                != tuple(f"{root.root_id}.flh-project.r{v}" for v in (1, 2))
                or len(prior.artifacts) != 3
                or {a.payload_schema for a in prior.artifacts}
                != {
                    (
                        FiniteResponseLawAssignedEvaluationTaskResult
                        if assigned
                        else FiniteResponseLawEvaluationTaskResult
                    ).SCHEMA,
                    NATIVE_PAIR_SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                }
            ):
                raise ValueError(
                    "Finite response-law evaluation retained prefix changes its unit, clock, views or outputs"
                )

    @property
    def config_id(self) -> str:
        return f"{self.source.spec_id}.retained-prefixes"

    def authenticate(
        self,
        prior: ProspectiveRetainedPredecessor,
        receipt: CanonicalTaskReceipt,
        record: FiniteResponseLawEvaluationTaskResult | FiniteResponseLawAssignedEvaluationTaskResult,
    ) -> None:
        """The worker has already authenticated each complete byte payload."""
        validate_retained_qualification_receipt(prior, receipt)
        expected = next(
            t for t in native_invocations(self.source) if t.task_id == prior.segment_id
        )
        if (
            prior not in self.prefixes
            or type(record)
            is not (
                FiniteResponseLawAssignedEvaluationTaskResult
                if type(self.source) is FiniteResponseLawAssignedEvaluationConfig
                else FiniteResponseLawEvaluationTaskResult
            )
            or receipt.run_id != self.donor_run_id
            or record.invocation != expected
            or record.invocation.source
            != ObjectIdentity.from_record(self.source.spec_id, self.source)
            or record.native_pair is None
            or not any(
                a.payload_schema == NATIVE_PAIR_SCHEMA
                and (a.sha256, a.size_bytes)
                == (record.native_pair.data_sha256, record.native_pair.data_bytes)
                for a in prior.artifacts
            )
            or record.predecessors
        ):
            raise ValueError("Finite response-law evaluation prefix import changes its successful native origin")
