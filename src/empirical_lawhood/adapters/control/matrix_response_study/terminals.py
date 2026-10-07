"""Executable negative-path rehearsal for the Six-matrix response compatibility phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)


class MatrixResponseCompatibilityStop(StrEnum):
    SOURCE_FAILURE = "SOURCE_FAILURE"
    ZERO_LAW = "ZERO_LAW"
    MISSING_ADMISSION_CELL = "MISSING_ADMISSION_CELL"
    OUTER_CONTROLLER_USE_NOT_APPLICABLE = "OUTER_CONTROLLER_USE_NOT_APPLICABLE"
    INNER_CONTROLLER_USE_NOT_APPLICABLE = "INNER_CONTROLLER_USE_NOT_APPLICABLE"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    RESOURCE_CEILING = "RESOURCE_CEILING"


_TERMINAL_IDS = {
    MatrixResponseCompatibilityStop.SOURCE_FAILURE: "matrix-response-study.terminal.source-failure",
    MatrixResponseCompatibilityStop.ZERO_LAW: "matrix-response-study.terminal.zero-supported-law",
    MatrixResponseCompatibilityStop.MISSING_ADMISSION_CELL: "matrix-response-study.terminal.missing-admission-cell",
    MatrixResponseCompatibilityStop.OUTER_CONTROLLER_USE_NOT_APPLICABLE: ("matrix-response-study.terminal.outer-controller-use-nonattempt"),
    MatrixResponseCompatibilityStop.INNER_CONTROLLER_USE_NOT_APPLICABLE: ("matrix-response-study.terminal.inner-controller-use-nonattempt.00"),
    MatrixResponseCompatibilityStop.AUTHORITY_REQUIRED: "matrix-response-study.terminal.authority-required",
    MatrixResponseCompatibilityStop.RESOURCE_CEILING: "matrix-response-study.terminal.resource-ceiling-nonadmitted",
}


@dataclass(frozen=True, slots=True)
class MatrixResponseStopProbe(CanonicalRecord):
    """Independent observable facts used to exercise one fail-closed path."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-stop-probe'
    VERSION: ClassVar[str] = "1.0.0"

    probe_id: str
    source_ready: bool
    supported_law_count: int
    required_admission_cell_count: int
    available_admission_cell_count: int
    outer_controller_use_applicable: bool
    inner_controller_use_applicable: bool
    operation_authority_present: bool
    forecast_resource_work: int
    resource_work_ceiling: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.probe_id, field_name="probe_id")
        if (
            min(
                self.supported_law_count,
                self.required_admission_cell_count,
                self.available_admission_cell_count,
                self.forecast_resource_work,
                self.resource_work_ceiling,
            )
            < 0
        ):
            raise ValueError("Six-matrix response terminal probe counts cannot be negative")
        if self.required_admission_cell_count < 1 or self.resource_work_ceiling < 1:
            raise ValueError("Six-matrix response terminal probe requires positive programme admission/resource requirements")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("Six-matrix response terminal rehearsal cannot inspect protected outcomes")


@dataclass(frozen=True, slots=True)
class MatrixResponseCompatibilityStopReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-compatibility-stop-receipt'
    VERSION: ClassVar[str] = "1.0.0"

    receipt_id: str
    probe_id: str
    stop: MatrixResponseCompatibilityStop
    terminal_id: str
    reason_codes: tuple[str, ...]
    scientific_execution_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("receipt_id", "probe_id", "terminal_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.terminal_id != _TERMINAL_IDS[self.stop]:
            raise ValueError("Six-matrix response terminal receipt uses another closed terminal identity")
        if self.scientific_execution_count or self.grants_authority:
            raise ValueError("Six-matrix response terminal rehearsal cannot execute or grant authority")


def rehearse_matrix_response_study_stop(probe: MatrixResponseStopProbe) -> MatrixResponseCompatibilityStopReceipt:
    triggered = tuple(
        stop
        for stop, condition in (
            (MatrixResponseCompatibilityStop.SOURCE_FAILURE, not probe.source_ready),
            (MatrixResponseCompatibilityStop.ZERO_LAW, probe.supported_law_count == 0),
            (
                MatrixResponseCompatibilityStop.MISSING_ADMISSION_CELL,
                probe.available_admission_cell_count < probe.required_admission_cell_count,
            ),
            (
                MatrixResponseCompatibilityStop.OUTER_CONTROLLER_USE_NOT_APPLICABLE,
                not probe.outer_controller_use_applicable,
            ),
            (
                MatrixResponseCompatibilityStop.INNER_CONTROLLER_USE_NOT_APPLICABLE,
                not probe.inner_controller_use_applicable,
            ),
            (
                MatrixResponseCompatibilityStop.AUTHORITY_REQUIRED,
                not probe.operation_authority_present,
            ),
            (
                MatrixResponseCompatibilityStop.RESOURCE_CEILING,
                probe.forecast_resource_work > probe.resource_work_ceiling,
            ),
        )
        if condition
    )
    if len(triggered) != 1:
        raise ValueError("Six-matrix response terminal rehearsal requires exactly one isolated stop")
    stop = triggered[0]
    return MatrixResponseCompatibilityStopReceipt(
        receipt_id=f"matrix-response-study-stop-receipt.{probe.probe_id}",
        probe_id=probe.probe_id,
        stop=stop,
        terminal_id=_TERMINAL_IDS[stop],
        reason_codes=(f"SIX_MATRIX_RESPONSE_{stop.value}",),
        scientific_execution_count=0,
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseCompatibilityStopReceipt',
    'MatrixResponseCompatibilityStop',
    'MatrixResponseStopProbe',
    'rehearse_matrix_response_study_stop',
]
