"""Code-owned portable authoring selection; operational paths stay in the API."""

from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import ConstructedPreparationStage, ConstructedPreparationSource
from empirical_lawhood.adapters.methods.preparation_applicability.issued_inputs import PreparationApplicabilityPublishedInput
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.preparation_applicability.conformance import reject_conformance_reuse


@dataclass(frozen=True, slots=True)
class ConstructedPreparationApplicabilitySelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/composition/constructed-preparation-applicability/selection"
    selection_id: str
    stage: ConstructedPreparationStage
    source: ConstructedPreparationSource
    upstream_records: tuple[PreparationApplicabilityPublishedInput, ...]

    def __post_init__(self):
        validate_stable_id(self.selection_id)
        expected = tuple(sorted(("conformance", "exposure", *(value.key for value in self.stage.upstream))))
        if (tuple(value.key for value in self.upstream_records) != expected
            or self.source.design != self.stage.design
            or self.stage.source.object_fingerprint != self.source.fingerprint()
            or self.stage.source.object_schema != self.source.SCHEMA):
            raise ValueError("authoring selection changes its current source/input roster")
        for value in self.upstream_records:
            value.require_stage(self.stage)
        conformance=next(value for value in self.upstream_records if value.key=="conformance")
        reject_conformance_reuse(self.stage,conformance.record)
        if (self.source.native_implementation!=conformance.record.implementation
            or self.source.conformance_receipt!=ObjectIdentity.from_record(conformance.receipt.receipt_id,conformance.receipt)
            or self.source.python_version!=conformance.record.python_version
            or self.source.numpy_version!=conformance.record.numpy_version):
            raise ValueError("source substitutes its actual native conformance publication")

    @property
    def qualified_report(self):
        return next((value.record for value in self.upstream_records if value.key == "qualification"), None)
