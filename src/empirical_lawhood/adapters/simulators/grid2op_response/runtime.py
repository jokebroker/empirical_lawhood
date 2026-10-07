"""Injected Grid2Op episode execution boundary and exact branch accounting.

The core boundary remains import-safe when Grid2Op is not installed.  The
concrete held-runtime adapter below imports the optional simulator only when it
is constructed by outer composition.  One environment is retained per worker
thread so a parallel scientific DAG pays initialization once per worker, not
once per branch.
"""

from __future__ import annotations

from datetime import UTC
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import math
import re
from threading import local
from typing import Callable, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateTargetPhase

from .contracts import Grid2OpActionBranchRequest, Grid2OpActionKind, Grid2OpBranchTrace, Grid2OpChronicBinding, Grid2OpEpisodeRequest, Grid2OpEpisodeResult, Grid2OpSourceBinding, Grid2OpNativeStep


class Grid2OpEnvironmentPort(Protocol):
    """Trusted composition-owned bridge to one exact held Grid2Op runtime."""

    def source_binding(self) -> Grid2OpSourceBinding: ...

    def execute_branch(
        self,
        *,
        request: Grid2OpEpisodeRequest,
        branch: Grid2OpActionBranchRequest,
        chronic: Grid2OpChronicBinding,
    ) -> Grid2OpBranchTrace: ...


Grid2OpProgress = Callable[[str, str, int], None]


def _noop_progress(_branch_id: str, _stage: str, _ordinal: int) -> None:
    return


def _topology_sha256(observation: object) -> str:
    values = tuple(int(value) for value in observation.topo_vect)  # type: ignore[attr-defined]
    payload = (",".join(str(value) for value in values) + "\n").encode("ascii")
    return sha256(payload).hexdigest()


def _connected_substation_components(observation: object) -> int:
    """Count components on the native active-line substation graph.

    The busbar-resolved state remains separately bound by ``topo_vect``.  This
    receiver is intentionally a coarser declared gauge, not a replacement for
    the native topology observation.
    """

    n_sub = int(observation.n_sub)  # type: ignore[attr-defined]
    parents = list(range(n_sub))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for active, origin, extremity in zip(
        observation.line_status,  # type: ignore[attr-defined]
        observation.line_or_to_subid,  # type: ignore[attr-defined]
        observation.line_ex_to_subid,  # type: ignore[attr-defined]
        strict=True,
    ):
        if bool(active):
            union(int(origin), int(extremity))
    return len({find(value) for value in range(n_sub)})


def _line_index(target_id: str) -> int:
    matched = re.fullmatch(r"line-(\d{3})", target_id)
    if matched is None:
        raise ValueError("Grid2Op line target id is not canonical")
    return int(matched.group(1))


class Grid2OpOfflineEnvironmentAdapter:
    """Concrete bridge to one held offline Grid2Op/LightSim environment.

    ``dataset_path`` is an execution locator owned by composition; it never
    enters a canonical scientific record.  The source record binds the bytes.
    """

    def __init__(
        self,
        *,
        source: Grid2OpSourceBinding,
        dataset_path: Path,
        progress: Grid2OpProgress = _noop_progress,
    ) -> None:
        if not dataset_path.is_absolute() or not dataset_path.is_dir():
            raise ValueError("held Grid2Op dataset path is absent or relative")
        self._source = source
        self._dataset_path = dataset_path
        self._progress = progress
        self._thread = local()

    def source_binding(self) -> Grid2OpSourceBinding:
        return self._source

    def close(self) -> None:
        environment = getattr(self._thread, "environment", None)
        if environment is not None:
            environment.close()
            self._thread.environment = None

    def _environment(self) -> object:
        value = getattr(self._thread, "environment", None)
        if value is not None:
            return value
        try:
            import grid2op  # type: ignore[import-not-found]
            from lightsim2grid import LightSimBackend  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - external integration gate
            raise RuntimeError("held Grid2Op runtime is not importable") from error
        value = grid2op.make(str(self._dataset_path), backend=LightSimBackend())
        backend = type(value.backend)
        observed_backend = f"{backend.__module__}.{backend.__qualname__}"
        if (
            not observed_backend.endswith("LightSimBackend")
            and "LightSimBackend_" not in observed_backend
        ):
            value.close()
            raise ValueError("held Grid2Op backend differs from source binding")
        self._thread.environment = value
        return value

    @staticmethod
    def _native_action(environment: object, branch: Grid2OpActionBranchRequest) -> object:
        action = branch.action
        if action.kind is Grid2OpActionKind.HOLD:
            return environment.action_space({})  # type: ignore[attr-defined]
        if action.kind is not Grid2OpActionKind.SET_LINE_STATUS:
            raise ValueError("issued Grid2Op independent substrate grounding chart supports only hold/line-status fibres")
        indexes = tuple(_line_index(value) for value in action.target_native_ids)
        if any(index < 0 or index >= int(environment.n_line) for index in indexes):  # type: ignore[attr-defined]
            raise ValueError("Grid2Op line-status action lies outside the held grid")
        return environment.action_space(  # type: ignore[attr-defined]
            {"set_line_status": list(zip(indexes, action.integer_values, strict=True))}
        )

    @staticmethod
    def _realized(
        observation: object,
        branch: Grid2OpActionBranchRequest,
        *,
        replaced: bool,
    ) -> tuple[bool, str | None]:
        if replaced or branch.action.kind is Grid2OpActionKind.HOLD:
            return True, "HOLD"
        indexes = tuple(_line_index(value) for value in branch.action.target_native_ids)
        requested = branch.action.integer_values
        matched = all(
            bool(observation.line_status[index]) == (value > 0)  # type: ignore[attr-defined]
            for index, value in zip(indexes, requested, strict=True)
        )
        return (matched, branch.action.action_code if matched else None)

    def execute_branch(
        self,
        *,
        request: Grid2OpEpisodeRequest,
        branch: Grid2OpActionBranchRequest,
        chronic: Grid2OpChronicBinding,
    ) -> Grid2OpBranchTrace:
        environment = self._environment()
        options: dict[str, object] = {"time serie id": chronic.native_chronic_id}
        if request.initial_step:
            options["init ts"] = request.initial_step
        self._progress(branch.branch_id, "reset-start", 0)
        observation = environment.reset(  # type: ignore[attr-defined]
            seed=request.environment_seed,
            options=options,
        )
        reset_sha256 = sha256(
            (
                _topology_sha256(observation)
                + "\n"
                + ",".join(format(float(value), ".17g") for value in observation.rho)  # type: ignore[attr-defined]
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        self._progress(branch.branch_id, "reset-complete", 0)
        requested_action = self._native_action(environment, branch)
        hold = environment.action_space({})  # type: ignore[attr-defined]
        horizons = set(branch.receiver_horizon_steps)
        steps: list[Grid2OpNativeStep] = []
        for ordinal in range(1, max(horizons) + 1):
            self._progress(branch.branch_id, "step-start", ordinal)
            observation, _reward, done, info = environment.step(  # type: ignore[attr-defined]
                requested_action if ordinal == 1 else hold
            )
            illegal = bool(info.get("is_illegal", False))
            ambiguous = bool(info.get("is_ambiguous", False))
            replaced = illegal or ambiguous
            realized, realized_code = self._realized(
                observation,
                branch,
                replaced=replaced,
            )
            exceptions = tuple(value for value in info.get("exception", ()) if value)
            terminal = bool(done) or bool(exceptions)
            rho_values = tuple(float(value) for value in observation.rho)  # type: ignore[attr-defined]
            finite_rho = bool(rho_values) and all(math.isfinite(value) for value in rho_values)
            reasons: set[str] = set()
            if exceptions:
                reasons.add("GRID2OP_BACKEND_EXCEPTION")
            if not finite_rho:
                reasons.add("GRID2OP_RHO_NONFINITE_OR_ABSENT")
            if not realized:
                reasons.add("GRID2OP_REALIZED_ACTION_UNOBSERVED")
            valid = finite_rho and not exceptions and realized
            if ordinal in horizons or terminal:
                topology_sha256 = _topology_sha256(observation)
                observed_time = observation.get_time_stamp()  # type: ignore[attr-defined]
                if observed_time.tzinfo is None:
                    observed_time = observed_time.replace(tzinfo=UTC)
                timestamp = observed_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                effective = "HOLD" if replaced else branch.action.action_code
                steps.append(
                    Grid2OpNativeStep(
                        step_id=f"step.{request.unit_id}.{branch.branch_id}.{ordinal:06d}",
                        branch_id=branch.branch_id,
                        horizon_step=ordinal,
                        environment_tick=request.initial_step + ordinal,
                        timestamp_utc=timestamp,
                        requested_action_code=branch.action.action_code,
                        accepted_action_code=effective,
                        applied_action_code=effective,
                        realized_action_code=realized_code,
                        action_illegal=illegal,
                        action_ambiguous=ambiguous,
                        realization_observed=realized,
                        topology_state_id=f"topology.grid2op.{topology_sha256[:24]}",
                        topology_vector_sha256=topology_sha256,
                        connected_component_count=_connected_substation_components(observation),
                        disconnected_line_count=sum(
                            not bool(value)
                            for value in observation.line_status  # type: ignore[attr-defined]
                        ),
                        maximum_rho=(
                            Decimal(format(max(rho_values), ".17g")) if finite_rho else None
                        ),
                        done=terminal,
                        has_error=bool(exceptions),
                        observation_valid=valid,
                        reason_codes=tuple(sorted(reasons)),
                    )
                )
            self._progress(branch.branch_id, "step-complete", ordinal)
            if terminal:
                break
        if not steps:
            raise RuntimeError("Grid2Op branch terminated before its first receiver horizon")
        return Grid2OpBranchTrace(
            trace_id=f"trace.{request.unit_id}.{branch.branch_id}",
            request=ObjectIdentity.from_record(request.request_id, request),
            branch=ObjectIdentity.from_record(branch.branch_id, branch),
            chronic=request.chronic,
            reset_state_sha256=reset_sha256,
            reset_exact=True,
            steps=tuple(steps),
        )


def execute_grid2op_episode(
    *,
    port: Grid2OpEnvironmentPort,
    source: Grid2OpSourceBinding,
    request: Grid2OpEpisodeRequest,
) -> Grid2OpEpisodeResult:
    """Execute every frozen branch and retain them as one chronic unit."""

    if port.source_binding() != source:
        raise ValueError("Grid2Op execution port source binding differs")
    if (
        request.phase is not IndependentSubstrateTargetPhase.DEVELOPMENT
        and not source.fresh_target_identity_disjoint
    ):
        raise ValueError("Grid2Op sealed execution requires a disjoint held source")
    source_identity = ObjectIdentity.from_record(source.binding_id, source)
    if request.source_binding != source_identity:
        raise ValueError("Grid2Op request source binding differs")
    chronics = {value.chronic_id: value for value in source.chronic_bindings}
    chronic = chronics.get(request.chronic.object_id)
    if chronic is None or request.chronic != ObjectIdentity.from_record(
        chronic.chronic_id,
        chronic,
    ):
        raise ValueError("Grid2Op request chronic is absent or differs")
    request_identity = ObjectIdentity.from_record(request.request_id, request)
    traces = []
    for branch in request.branches:
        trace = port.execute_branch(
            request=request,
            branch=branch,
            chronic=chronic,
        )
        observed_horizons = tuple(value.horizon_step for value in trace.steps)
        requested_horizons = branch.receiver_horizon_steps
        requested_prefix = requested_horizons[: len(observed_horizons)]
        terminal_between_horizons = (
            trace.steps[-1].done
            and observed_horizons[:-1] == requested_horizons[: len(observed_horizons) - 1]
            and observed_horizons[-1] not in requested_horizons
            and observed_horizons[-1] <= requested_horizons[-1]
        )
        if (
            trace.request != request_identity
            or trace.branch != ObjectIdentity.from_record(branch.branch_id, branch)
            or trace.chronic != request.chronic
            or (observed_horizons != requested_prefix and not terminal_between_horizons)
            or (len(observed_horizons) != len(requested_horizons) and not trace.steps[-1].done)
        ):
            raise ValueError("Grid2Op branch trace differs from its frozen request")
        traces.append(trace)
    ordered = tuple(sorted(traces, key=lambda value: value.trace_id))
    if len({value.reset_state_sha256 for value in ordered}) != 1:
        raise ValueError("Grid2Op action branches did not begin from one exact reset state")
    return Grid2OpEpisodeResult(
        result_id=f"result.{request.request_id}",
        request=request_identity,
        source_binding=source_identity,
        chronic=request.chronic,
        traces=ordered,
        scientific_unit_count=1,
        reward_used_for_science=False,
        outcome_access=request.outcome_access,
    )


__all__ = [
    "Grid2OpEnvironmentPort",
    "Grid2OpProgress",
    'Grid2OpOfflineEnvironmentAdapter',
    "execute_grid2op_episode",
]
