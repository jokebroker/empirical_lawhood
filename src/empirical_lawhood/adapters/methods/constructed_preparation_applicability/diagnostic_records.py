"""Closed current input/result contracts for outcome-visible P12 analysis.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.composition.constructed_preparation_applicability.selection import ConstructedPreparationApplicabilitySelection
from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PreparationApplicabilityResultAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id, validate_sha256, validate_relative_locator


@dataclass(frozen=True, slots=True)
class PreparationDiagnosticRootReceipts(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/diagnostic-root-receipts"
    root_id: str
    # Exact tasks: prefix, select, parents, lower, assay, measure.
    receipt_ids: tuple[str, ...]

    def __post_init__(self):
        validate_stable_id(self.root_id)
        if len(self.receipt_ids) != 6 or len(set(self.receipt_ids)) != 6:
            raise ValueError("diagnostic root requires six distinct exact task receipts")
        for value in self.receipt_ids:
            validate_stable_id(value)


@dataclass(frozen=True, slots=True)
class PreparationDiagnosticPhaseInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/diagnostic-phase-input"
    selection: ConstructedPreparationApplicabilitySelection
    result_access: PreparationApplicabilityResultAccess
    roots: tuple[PreparationDiagnosticRootReceipts, ...]
    report_receipt_id: str
    adjudication_receipt_id: str
    recovery_index_relative_path: str

    def __post_init__(self):
        from empirical_lawhood.adapters.methods.preparation_applicability.config import preparation_run_id
        if (tuple(value.root_id for value in self.roots) != self.selection.stage.root_ids
            or self.result_access.run_id != preparation_run_id(self.selection.stage)):
            raise ValueError("diagnostic input changes the complete assigned phase/run census")
        for value in (self.report_receipt_id, self.adjudication_receipt_id):
            validate_stable_id(value)
        validate_relative_locator(self.recovery_index_relative_path)
        ids = tuple(value for root in self.roots for value in root.receipt_ids)
        if len(set((*ids, self.report_receipt_id, self.adjudication_receipt_id))) != len(ids) + 2:
            raise ValueError("diagnostic input duplicates a phase receipt")


@dataclass(frozen=True, slots=True)
class PreparationDiagnosticInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/diagnostic-input"
    analysis_id: str
    qualification: PreparationDiagnosticPhaseInput
    evaluation: PreparationDiagnosticPhaseInput
    analysis_source_sha256: str
    # One explicitly assigned bootstrap stream per independent phase.
    bootstrap_seed_q: int
    bootstrap_seed_e: int

    def __post_init__(self):
        validate_stable_id(self.analysis_id)
        validate_sha256(self.analysis_source_sha256)
        if (self.qualification.selection.stage.phase != "Q"
            or self.evaluation.selection.stage.phase != "E"
            or self.qualification.result_access.run_id == self.evaluation.result_access.run_id):
            raise ValueError("diagnostic requires separate complete Q8 and E32 inputs")
        q, e = self.qualification.selection.stage, self.evaluation.selection.stage
        if (set(q.root_ids) & set(e.root_ids) or q.design != e.design
            or self.qualification.selection.source != self.evaluation.selection.source
            or q.upstream[0].artifact != e.upstream[0].artifact):
            raise ValueError("diagnostic phases change fixed source/lower or independent roles")
        for seed in (self.bootstrap_seed_q, self.bootstrap_seed_e):
            if type(seed) is not int or not 0 <= seed < 2**128:
                raise ValueError("diagnostic requires explicit128-bit bootstrap allocations")
        if self.bootstrap_seed_q == self.bootstrap_seed_e:
            raise ValueError("diagnostic Q/E bootstrap streams must be distinct")
        consumed = {seed.effective_seed for stage in (q, e) for root in stage.allocation.roots
                    for seed in root.scientific_seeds}
        if self.bootstrap_seed_q in consumed or self.bootstrap_seed_e in consumed:
            raise ValueError("diagnostic bootstrap streams collide with native/request purposes")


def diagnostic_json_payload(value) -> str:
    """Stable finite JSON for opaque post-hoc tables, not canonical numeric inputs.

    The original floating numerical reports retain their JSON numbers. Typed
    scientific inputs and original measurement records keep their Decimal
    canonical transport; this string cannot enter a prospective calculation.
    """
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"


@dataclass(frozen=True, slots=True)
class PreparationDiagnosticReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/constructed-preparation-applicability/diagnostic-report"
    analysis_id: str
    input_identity: ObjectIdentity
    input_artifacts: tuple[ObjectIdentity, ...]
    data_payload: str
    ceiling: str = "OUTCOME_VISIBLE_DIAGNOSTICS_NOT_NEW_CONFIRMATION_OR_ADMISSION"
    native_calls: int = 0

    def __post_init__(self):
        import json
        validate_stable_id(self.analysis_id)
        if (self.input_identity.object_schema != PreparationDiagnosticInput.SCHEMA
            or self.ceiling != "OUTCOME_VISIBLE_DIAGNOSTICS_NOT_NEW_CONFIRMATION_OR_ADMISSION"
            or type(self.native_calls) is not int or self.native_calls != 0
            or not self.input_artifacts or len(self.data_payload.encode()) > 1024**2):
            raise ValueError("diagnostic report changes its custody/analysis ceiling")
        data = json.loads(self.data_payload)
        if (set(data) != {"Q", "E"} or diagnostic_json_payload(data) != self.data_payload
            or data["Q"]["n_independent_roots"] != 8
            or data["E"]["n_independent_roots"] != 32):
            raise ValueError("diagnostic report changes its complete phase denominators")
