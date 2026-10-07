"""Explicit numerical denominator codes with original and current unit custody."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateCompleteTargetUnit
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_strings, validate_stable_id


@dataclass(frozen=True, slots=True)
class Grid2OpForecastDenominatorInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulators/grid2op-response/forecast-denominator-inputs"

    input_id: str
    current_native_panel: ObjectIdentity
    target_design: ObjectIdentity
    forecast_alphabet: ObjectIdentity
    phase: str
    unit_ids: tuple[str, ...]
    denominator_codes: tuple[int, ...]
    unit_exports: tuple[IndependentSourceExport, ...]
    verified_numerical_inputs: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        require_sorted_unique_strings(self.unit_ids, field_name="unit_ids", allow_empty=False)
        if (
            any(type(getattr(self, name)) is not ObjectIdentity for name in (
                "current_native_panel", "target_design", "forecast_alphabet", "verified_numerical_inputs"
            ))
            or self.phase not in ("DEVELOPMENT", "EVALUATION")
            or len(self.denominator_codes) != len(self.unit_ids)
            or len(self.unit_exports) != len(self.unit_ids)
            or any(type(code) is not int or not 0 <= code < 2**32 for code in self.denominator_codes)
            or any(type(export) is not IndependentSourceExport for export in self.unit_exports)
            or self.verified_numerical_inputs.object_schema != "empirical-lawhood/migration/verified-numerical-inputs"
            or self.verified_numerical_inputs.object_version != "1.0.0"
        ):
            raise ValueError("Grid2Op forecast denominator requires complete ordered original numerical codes and separately verified current unit exports")
        for unit_id, export in zip(self.unit_ids, self.unit_exports, strict=True):
            if (
                export.target_source.object_id != unit_id
                or export.target_source.object_schema != IndependentSubstrateCompleteTargetUnit.SCHEMA
                or export.target_physical_unit_ids != (unit_id,)
            ):
                raise ValueError("Grid2Op forecast denominator export is detached from its exact current unit")

    def validate_panel(self, *, panel: ObjectIdentity, phase: str, unit_ids: tuple[str, ...]) -> None:
        if self.current_native_panel != panel or self.phase != phase or self.unit_ids != unit_ids:
            raise ValueError("Grid2Op forecast denominator input differs from its exact current panel, phase or acquisition order")

    def validate_evidence(
        self, *, target_design: ObjectIdentity, forecast_alphabet: ObjectIdentity,
        phase: str, unit_ids: tuple[str, ...], current_unit_fingerprints: tuple[str, ...],
    ) -> None:
        if (
            self.target_design != target_design
            or self.forecast_alphabet != forecast_alphabet
            or self.phase != phase
            or self.unit_ids != unit_ids
            or tuple(export.target_source.object_fingerprint for export in self.unit_exports) != current_unit_fingerprints
        ):
            raise ValueError("Grid2Op numerical denominator input is detached from the exact current design, alphabet or lowered native units")

    def code_for(self, unit_id: str) -> int:
        try:
            position = self.unit_ids.index(unit_id)
        except ValueError as exc:
            raise ValueError("Grid2Op denominator code requires its exact declared original unit input") from exc
        return self.denominator_codes[position]
