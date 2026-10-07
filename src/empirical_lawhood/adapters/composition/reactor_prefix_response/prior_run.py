"Operational exposure ledger for the publication-only corrective prefix-response study."

from dataclasses import dataclass
from typing import ClassVar
import re

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256


@dataclass(frozen=True, slots=True)
class ReactorPriorRun(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/reactor-prior-run'
    run_id: str
    source_commit: str
    candidate_sha256: str
    execution_plan_sha256: str
    resource_journal_sha256: str
    native_launches: int
    completed_rk4_updates: int
    scientific_output_receipts: int
    evaluator_executed: bool
    response_values_inspected_for_current_design: bool
    reason_code: str

    def __post_init__(self) -> None:
        for value in (
            self.candidate_sha256,
            self.execution_plan_sha256,
            self.resource_journal_sha256,
        ):
            validate_sha256(value, field_name="prior_run_sha256")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None:
            raise ValueError("prior run requires an explicit source commit")
        expected = {
            "reactor-prefix-response.output-record-declaration-failure": (
                0,
                False,
                "OUTPUT_RECORD_VERSION_DECLARATION_MISSING",
            ),
            "reactor-prefix-response.evaluator-custody-serialization-failure": (
                20,
                False,
                "EVALUATOR_CUSTODY_PORT_NOT_SPAWN_SERIALIZABLE",
            ),
            "reactor-prefix-response.terminal-evidence-class-mixing-failure": (
                20,
                True,
                "TERMINAL_OUTPUT_BATCH_MIXES_EVIDENCE_CLASSES",
            ),
        }
        if (
            expected.get(self.run_id)
            != (
                self.scientific_output_receipts,
                self.evaluator_executed,
                self.reason_code,
            )
            or self.native_launches != 20
            or self.completed_rk4_updates != 600
            or self.response_values_inspected_for_current_design
        ):
            raise ValueError("reactor corrective ledger differs from its bounded predecessor")
