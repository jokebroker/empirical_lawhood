"""Scientific operand surfaces consumed after current record authentication.

These protocols describe the reducer boundary, not decoders or custody. Public
providers must decode the eventual concrete current records and bind the exact
OriginalFiniteResponseLaw artifact before calling these functions.
"""

from decimal import Decimal
from typing import Protocol


class NativePhaseOperand(Protocol):
    root_id: str
    phase: str
    schedule_index: int | None
    refinement: int
    future_index: int | None
    word_index: int | None
    incoming_sha256: str | None
    disposition: str
    parent_work: Decimal
    signed_work: Decimal
    absolute_work: Decimal
    ticks: tuple[int, ...]
    positions_base64: str
    transfer_base64: str
    transfer_known: tuple[bool, ...]
    history_ticks: tuple[int, ...]
    history_positions_base64: str
    history_momenta_base64: str
    streams: tuple[object, ...]
    innovation_sha256: str

    def fingerprint(self) -> str: ...


class PrefixOperand(Protocol):
    root_id: str
    phases: tuple[NativePhaseOperand, ...]
    frame_base64: str | None

    def fingerprint(self) -> str: ...


class PanelOperand(Protocol):
    root_id: str
    prefix_sha256: str
    phases: tuple[NativePhaseOperand, ...]


class ConstructorMeasurement(Protocol):
    root_id: str
    complete: bool
    handoff: tuple[Decimal, ...]
    maxima: tuple[Decimal, ...]
