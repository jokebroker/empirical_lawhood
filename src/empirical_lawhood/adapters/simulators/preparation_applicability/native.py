"""Fresh wrappers around owned mechanics; no historical source execution."""

from decimal import Decimal
from empirical_lawhood.kernel.matrix_inputs import MatrixRootAllocation
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationSchedule
from .contracts import WORDS, SCHEDULES
from typing import Callable, Any
import numpy as np

from empirical_lawhood.adapters.methods.preparation_applicability.records import numbers
from empirical_lawhood.adapters.simulators.prepared_response.contracts import (
    prepared_native_member,
    prepared_numerical_view,
)
from empirical_lawhood.adapters.simulators.prepared_response.source import (
    _Branch,
    march_native_intervals,
)
from empirical_lawhood.adapters.simulators.prepared_response.instruments import (
    PreparedPortFrame,
    select_prepared_ports,
)
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import compact_interface
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_instruments import (
    preparation_policy_preparation_schedule,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import (
    derive_probe_roster,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import (
    ResponseGeometryNativePassiveObserver,
    _decode,
    _encode,
)
from .records import PreparationApplicabilityNativePhase, PreparationApplicabilityPrefix, PreparationApplicabilityStream



def acquire_phase(
    *,
    phase_name: str,
    allocation: MatrixRootAllocation,
    refinement: int,
    source_sha256: str,
    incoming: PreparationApplicabilityNativePhase | None = None,
    schedule_index: int | None = None,
    future_index: int | None = None,
    word_index: int | None = None,
    frame: PreparedPortFrame | None = None,
    progress: Callable[[int], None] | None = None,
    preparation_schedule: FiniteResponseLawPreparationSchedule | None = None,
    constructed: bool = False,
) -> PreparationApplicabilityNativePhase:
    schedule: Any
    start, end = {"prefix": (0, 4096), "parent": (4096, 4496), "future": (4496, 4688)}[phase_name]
    root_id = allocation.root_id
    if allocation.cohort not in (("constructed",) if constructed else ("q2", "cir1")) or allocation.source_prefix is not None:
        raise ValueError("native wrapper changes its assigned prefix source kind")
    if phase_name == "prefix":
        if incoming is not None:
            raise ValueError("fresh prefix imports a state")
        initial_coupling = 0.0 if allocation.cohort == "cir1" else 8.0
        state = ideal_state(q=2, alpha_tilde_x=initial_coupling, alpha_tilde_y=initial_coupling, constitution="00" if allocation.cohort == "cir1" else "11")
        roster = derive_probe_roster(
            config_fingerprint=allocation.fingerprint(),
            scientific_seed=allocation.seed_for("passive-probes"),
            rule_id="cc1-applicability-v1.passive-probes",
        )
        branch = _Branch(
            state,
            ResponseGeometryNativePassiveObserver(
                roster=roster,
                y=state.positions[1],
                y_velocity=state.momenta[1],
                refinement=refinement,
            ),
            [0],
            [state.positions],
            [state.momenta],
        )
        ramp = np.minimum(1.0, (np.arange(4096 * refinement) + 1) / (256 * refinement))
        schedule = initial_coupling + ramp[:, None] * (np.asarray((2 / 3, 22 / 3)) - initial_coupling)
        purpose = "prefix"
    else:
        if (
            incoming is None
            or incoming.allocation != allocation
            or incoming.root_id != root_id
            or incoming.refinement != refinement
            or incoming.source_sha256 != source_sha256
            or incoming.phase != ("prefix" if phase_name == "parent" else "parent")
            or incoming.end_tick != start
            or frame is None
            or frame.cutoff_tick != 4096
        ):
            raise ValueError("native continuation changes its authenticated predecessor")
        if phase_name == "future" and incoming.schedule_index != schedule_index:
            raise ValueError("future changes its actual parent")
        if schedule_index is None or (phase_name == "future" and future_index is None):
            raise ValueError("continuation omits its assigned schedule/future")
        branch = incoming.restore()
        schedule = (
            preparation_policy_preparation_schedule(preparation_schedule or SCHEDULES[schedule_index], refinement=refinement)
            if phase_name == "parent"
            else None
        )
        if phase_name == "parent":
            purpose = "parent"
        else:
            assert future_index is not None
            purpose = f"future-{future_index + 1}"
    if constructed:
        from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationNativePhase, ConstructedPreparationStream
        from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.native import nominal_digest
        def allocated_stream(name: str):
            return ConstructedPreparationStream(allocation.seed_for(name), None if phase_name == "future" else nominal_digest(name), Decimal(1) if phase_name == "future" else Decimal(".002"))
        record_type = ConstructedPreparationNativePhase
    else:
        def allocated_stream(name: str):
            return PreparationApplicabilityStream(allocation.seed_for(name))
        record_type = PreparationApplicabilityNativePhase
    stream, bridge = (allocated_stream(name) for name in (purpose, purpose + "-bridge"))
    raw = march_native_intervals(
        branch=branch,
        start=start,
        end=end,
        refinement=refinement,
        member=prepared_native_member(),
        view=prepared_numerical_view(refinement),
        schedule=schedule,
        word=None if word_index is None else WORDS[word_index],
        frame=frame,
        rng=stream.generator(),
        stream=stream,
        bridge=bridge.generator(),
        bridge_stream=bridge,
        accumulate_parent_work=phase_name == "parent",
        force_parent_ticks=400,
        progress=progress,
    )
    return record_type(
        root_id,
        phase_name,
        schedule_index,
        future_index,
        word_index,
        refinement,
        source_sha256,
        None if incoming is None else incoming.fingerprint(),
        start,
        end,
        raw.completed,
        raw.nonzero,
        numbers(raw.impulse),
        Decimal(str(raw.signed_work)),
        Decimal(str(raw.absolute_work)),
        Decimal(str(raw.parent_work)),
        raw.disposition,
        raw.reason,
        raw.innovation_sha256,
        raw.streams,
        _encode(raw.realized_trace),
        tuple(map(int, raw.ticks)),
        _encode(raw.positions),
        _encode(raw.momenta),
        _encode(np.where(raw.transfer_known[:, None, None], raw.transfer, 0.0)),
        tuple(map(bool, raw.transfer_known)),
        raw.branch.observer.checkpoint() if raw.disposition == "COMPLETE" else None,
        tuple(raw.branch.ticks) if raw.disposition == "COMPLETE" else (),
        _encode(np.asarray(raw.branch.positions)) if raw.disposition == "COMPLETE" else "",
        _encode(np.asarray(raw.branch.momenta)) if raw.disposition == "COMPLETE" else "",
        allocation=allocation,
        preparation_schedule=preparation_schedule or (SCHEDULES[schedule_index] if phase_name == "parent" else None),
        applied_force_kicks=2 * raw.completed if phase_name == "future" else 0,
        squared_input=Decimal(str(raw.squared_input)),
    )


def acquire_prefix(
    allocation: MatrixRootAllocation, source_sha256: str, progress: Callable[[int], None] | None = None, *, constructed: bool = False
) -> PreparationApplicabilityPrefix:
    root_id = allocation.root_id
    if constructed:
        from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationPrefix
        prefix_type = ConstructedPreparationPrefix
    else:
        prefix_type = PreparationApplicabilityPrefix
    phases: list[PreparationApplicabilityNativePhase] = []
    for refinement in (1, 2):
        offset = sum(p.completed_intervals for p in phases)
        advance = None if progress is None else lambda n: progress(offset + n)
        phases.append(
            acquire_phase(
                phase_name="prefix",
                allocation=allocation,
                constructed=constructed,
                refinement=refinement,
                source_sha256=source_sha256,
                progress=advance,
            )
        )
    if any(p.disposition != "COMPLETE" for p in phases):
        return prefix_type(root_id, tuple(phases), None, ())
    primary = phases[0]
    positions = _decode(primary.history_positions_base64, (31, 2, 3, 4, 4))
    frame = select_prepared_ports(
        ticks=primary.history_ticks[-16:], observations=positions[-16:, 0], cutoff_tick=4096
    )
    if frame is None:
        return prefix_type(root_id, tuple(phases), None, ())
    features = tuple(
        numbers(
            compact_interface(
                frame=frame,
                ticks=p.history_ticks,
                positions=_decode(p.history_positions_base64, (31, 2, 3, 4, 4)),
                momenta=_decode(p.history_momenta_base64, (31, 2, 3, 4, 4)),
                cutoff_tick=4096,
            ).values
        )
        for p in phases
    )
    return prefix_type(root_id, tuple(phases), _encode(frame.modes), features)
