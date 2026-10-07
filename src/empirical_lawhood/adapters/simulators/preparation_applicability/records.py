"""Native provenance and reusable checkpoints for a fresh fixed census."""

from typing import Any


from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.matrix_inputs import MatrixRootAllocation
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationSchedule
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import (
    ResponseGeometryNativeProbeCheckpoint,
)
from empirical_lawhood.adapters.methods.preparation_applicability.records import finite


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityStream(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/stream"
    scientific_seed: int

    def __post_init__(self) -> None:
        if type(self.scientific_seed) is not int or not 0 <= self.scientific_seed < 2**128:
            raise ValueError("native stream requires its consumed 128-bit PCG64 seed")

    def generator(self) -> Any:
        import numpy as np

        return np.random.Generator(np.random.PCG64(self.scientific_seed))


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityNativePhase(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/native-phase"
    root_id: str
    phase: str
    schedule_index: int | None
    future_index: int | None
    word_index: int | None
    refinement: int
    source_sha256: str
    incoming_sha256: str | None
    start_tick: int
    end_tick: int
    completed_intervals: int
    nonzero_intervals: int
    impulse: tuple[Decimal, ...]
    signed_work: Decimal
    absolute_work: Decimal
    parent_work: Decimal
    disposition: str
    reason: str | None
    innovation_sha256: str
    streams: tuple[PreparationApplicabilityStream, ...]
    trace_base64: str
    ticks: tuple[int, ...]
    positions_base64: str
    momenta_base64: str
    transfer_base64: str
    transfer_known: tuple[bool, ...]
    passive: ResponseGeometryNativeProbeCheckpoint | None
    history_ticks: tuple[int, ...]
    history_positions_base64: str
    history_momenta_base64: str
    allocation: MatrixRootAllocation = field(kw_only=True)
    preparation_schedule: FiniteResponseLawPreparationSchedule | None = field(default=None, kw_only=True)
    accepted: bool = field(default=True, kw_only=True)
    applied_force_kicks: int = field(default=0, kw_only=True)
    squared_input: Decimal = field(default=Decimal(0), kw_only=True)

    def __post_init__(self) -> None:
        from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import (
            _decode,
        )

        if self.allocation.root_id != self.root_id:
            raise ValueError("native phase changes its declared physical allocation")
        purpose = f"future-{self.future_index + 1}" if self.phase == "future" and self.future_index is not None else self.phase
        if tuple(stream.scientific_seed for stream in self.streams) != tuple(self.allocation.seed_for(name) for name in (purpose, purpose + "-bridge")):
            raise ValueError("native phase substitutes its actual numeric allocation")
        for v in (self.source_sha256, self.innovation_sha256):
            validate_sha256(v, field_name="native_sha256")
        if self.incoming_sha256 is not None:
            validate_sha256(self.incoming_sha256, field_name="incoming_sha256")
        clocks = {"prefix": (0, 4096), "parent": (4096, 4496), "future": (4496, 4688)}
        if (
            self.phase not in clocks
            or (self.start_tick, self.end_tick) != clocks[self.phase]
            or self.refinement not in (1, 2)
        ):
            raise ValueError("native phase changes its declared clocks/view")
        if not 0 <= self.completed_intervals <= (self.end_tick - self.start_tick) * self.refinement:
            raise ValueError("native phase changes its completed interval count")
        if self.disposition not in ("COMPLETE", "NUMERICAL_FAILURE", "OBSERVATION_FAILURE") or (
            self.reason is None
        ) != (self.disposition == "COMPLETE"):
            raise ValueError("native phase loses its adverse outcome")
        if (
            self.disposition == "COMPLETE"
            and self.completed_intervals != (self.end_tick - self.start_tick) * self.refinement
        ):
            raise ValueError("complete phase omits native intervals")
        finite(self.impulse, 2)
        finite((self.signed_work, self.absolute_work, self.parent_work, self.squared_input), 4)
        if (
            min(self.absolute_work, self.parent_work, self.squared_input) < 0
            or not 0 <= self.nonzero_intervals <= self.completed_intervals
        ):
            raise ValueError("native effort is negative or changes its interval census")
        if self.phase == "prefix":
            if any(
                v is not None
                for v in (
                    self.incoming_sha256,
                    self.schedule_index,
                    self.future_index,
                    self.word_index,
                )
            ):
                raise ValueError("fresh prefix imports a predecessor")
        elif self.schedule_index not in range(9) or self.incoming_sha256 is None:
            raise ValueError("continuation lacks its exact preparation predecessor")
        if self.phase == "future" and (
            self.future_index not in (0, 1) or self.word_index not in range(9)
        ):
            raise ValueError("future changes signed-word/future census")
        count = len(self.ticks)
        if len(self.transfer_known) != count or tuple(sorted(set(self.ticks))) != self.ticks:
            raise ValueError("native samples lose their observation clock")
        for v in (self.positions_base64, self.momenta_base64):
            _decode(v, (count, 2, 3, 4, 4))
        _decode(self.transfer_base64, (count, 15, 15), real=True)
        trace = _decode(self.trace_base64, (self.completed_intervals, 9), real=True)
        import numpy as np

        if self.accepted is not True or self.applied_force_kicks != (
            2 * self.completed_intervals if self.phase == "future" else 0
        ):
            raise ValueError("native receipt changes accepted/applied action accounting")
        if not np.array_equal(
            trace[:, 0], np.arange(self.completed_intervals) + self.start_tick * self.refinement
        ):
            raise ValueError("realized trace shifts the native clock")
        if self.phase == "prefix":
            ramp = np.minimum(
                1.0, (np.arange(self.completed_intervals) + 1) / (256 * self.refinement)
            )
            initial_coupling = 0.0 if self.allocation.cohort == "cir1" else 8.0
            coupling = initial_coupling + ramp[:, None] * (np.asarray((2 / 3, 22 / 3)) - initial_coupling)
        elif self.phase == "parent":
            from .contracts import SCHEDULES
            from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_instruments import (
                preparation_policy_preparation_schedule,
            )

            assert self.schedule_index is not None
            coupling = preparation_policy_preparation_schedule(
                self.preparation_schedule if self.preparation_schedule is not None else SCHEDULES[self.schedule_index], refinement=self.refinement
            )[: self.completed_intervals]
        else:
            coupling = np.tile((2 / 3, 22 / 3), (self.completed_intervals, 1))
        if not np.allclose(trace[:, 3:5], coupling, atol=1e-12, rtol=0):
            raise ValueError("requested coupling schedule differs from realized preparation")
        if self.phase == "future":
            from .contracts import WORDS

            assert self.word_index is not None
            word = WORDS[self.word_index]
            expected = np.zeros((self.completed_intervals, 2))
            expected[
                : min(self.completed_intervals, 64 * self.refinement), word.direction_index
            ] = word.sign * float(word.magnitude)
            if not np.array_equal(trace[:, 5:7], expected) or not np.array_equal(
                trace[:, 7:9], expected
            ):
                raise ValueError(
                    "requested/applied force kicks differ from the realized native word"
                )
        elif np.any(trace[:, 5:]):
            raise ValueError("preparation/prefix received an undeclared child force")
        if self.disposition == "COMPLETE" and not np.allclose(
            trace[-1, 3:5], (2 / 3, 22 / 3), atol=1e-12, rtol=0
        ):
            raise ValueError("completed phase did not return to the reference denominator")
        if (self.passive is not None) != (self.disposition == "COMPLETE"):
            raise ValueError("only complete phases retain reusable checkpoints")
        if self.passive is not None:
            if (
                self.history_ticks != tuple(range(self.end_tick - 480, self.end_tick + 1, 16))
                or self.passive.native_step != self.end_tick * self.refinement
            ):
                raise ValueError("checkpoint changes its causal history or view clock")
            for v in (self.history_positions_base64, self.history_momenta_base64):
                _decode(v, (31, 2, 3, 4, 4))

    def restore(self) -> Any:
        import numpy as np
        from empirical_lawhood.adapters.simulators.prepared_response.source import _Branch
        from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import (
            ResponseGeometryNativePassiveObserver,
            _decode,
        )
        from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState

        if self.passive is None:
            raise ValueError("failed native phase has no continuation")
        p = _decode(self.history_positions_base64, (31, 2, 3, 4, 4))
        m = _decode(self.history_momenta_base64, (31, 2, 3, 4, 4))
        state = SixMatrixState(2, p[-1], m[-1], self.passive.native_step, 2 / 3, 22 / 3)
        return _Branch(
            state,
            ResponseGeometryNativePassiveObserver.restore(self.passive),
            list(self.history_ticks),
            list(np.asarray(p)),
            list(np.asarray(m)),
        )


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityPrefix(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/prefix"
    root_id: str
    phases: tuple[PreparationApplicabilityNativePhase, ...]
    frame_base64: str | None
    features: tuple[tuple[Decimal, ...], ...]

    def __post_init__(self) -> None:
        if tuple(p.refinement for p in self.phases) != (1, 2) or any(
            p.root_id != self.root_id or p.phase != "prefix" for p in self.phases
        ):
            raise ValueError("prefix changes its common root or view census")
        if self.frame_base64 is not None:
            if len(self.features) != 2:
                raise ValueError("resolved prefix omits a view")
            for v in self.features:
                finite(v, 24)


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/preparation-applicability/panel"
    root_id: str
    prefix_sha256: str
    selection_sha256: str
    phases: tuple[PreparationApplicabilityNativePhase, ...]
    lower_seal_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.lower_seal_sha256, field_name="lower_seal_sha256")
        validate_sha256(self.prefix_sha256, field_name="prefix_sha256")
        validate_sha256(self.selection_sha256, field_name="selection_sha256")
        keys = tuple(
            (p.schedule_index, p.refinement, p.phase, p.future_index, p.word_index)
            for p in self.phases
        )
        if len(keys) != len(set(keys)) or any(p.root_id != self.root_id for p in self.phases):
            raise ValueError("panel duplicates acquisition cells or changes root")
