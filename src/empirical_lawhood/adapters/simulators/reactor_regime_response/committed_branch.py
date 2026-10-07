"""Replay a frozen nominal tape with one actual finite-controller delivery.

The caller must persist all four request commitments before opening any of
their native branches.  This owner never selects an action or changes a tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal as D
from typing import Any, Callable, Protocol, cast

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeController, NativeEpisode, TapeController, acquire_episode
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorAssignedScenario, ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand, ReactorDelivery, ReactorFinalDelivery
from empirical_lawhood.runtime.controller_runtime import TickDisposition


class PreparedRequest(Protocol):
    @property
    def delivery(self) -> Any: ...

    def deliver(self, publisher: Any) -> Any: ...


class WatchedPreparedTape(TapeController):
    """Remember the actual delayed callback seen by the fixed tape."""

    def __init__(self, tape: np.ndarray) -> None:
        super().__init__(tape)
        self.latest: tuple[float, float, float, float] | None = None

    def reset(self, spec: dict[str, Any]) -> None:
        self.latest = None
        super().reset(spec)

    def step(self, t_s: float, y: dict[str, float], dt_s: float) -> tuple[float, float]:
        self.latest = (
            t_s, y["t_reactor_k"], y["t_jacket_k"], y["dosed_kg"]
        )
        return TapeController.step(self, t_s, y, dt_s)


@dataclass
class CommittedNominalBranchOwner:
    """The exact committed word is delivered once at its selected callback."""

    root: str
    branch_id: str
    callback: int
    expected_observations: np.ndarray
    expected_requests: np.ndarray
    expected_stages: np.ndarray
    expected_tape: np.ndarray
    prepared: PreparedRequest
    publisher: Any
    failed: bool = False
    delivered: bool = False
    checked_callbacks: int = 0
    tick: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if (
            not 0 < self.callback < 2880
            or self.branch_id not in {
                f"{self.root}.request-{index}" for index in range(4)
            }
            or self.expected_observations.shape[1:] != (4,)
            or len(self.expected_observations) <= self.callback
            or self.expected_requests.shape != (2880, 2)
            or self.expected_stages.shape[1:] != (4,)
            or len(self.expected_stages) < self.callback
            or self.expected_tape.shape != (2880, 2)
            or not all(
                np.isfinite(value).all()
                for value in (
                    self.expected_observations[:self.callback + 1],
                    self.expected_requests,
                    self.expected_stages[:self.callback],
                    self.expected_tape,
                )
            )
            or not np.array_equal(
                self.expected_tape[: self.callback],
                self.expected_requests[: self.callback],
            )
        ):
            raise ValueError("nominal owner lacks the complete frozen preparation tape")
        command = self.prepared.delivery.mapping.full_native.command
        if (
            command.time_s != D(self.callback * 10)
            or tuple(self.expected_tape[self.callback])
            != (float(command.feed_kg_s), float(command.jacket_k))
        ):
            raise ValueError("nominal branch tape differs from its actual admission command")

    def advance(
        self, controller: NativeController, previous: tuple[float, float], session: Any
    ) -> ReactorDelivery | ReactorFinalDelivery | None:
        k = self.checked_callbacks
        if k >= 2880 or not isinstance(controller, WatchedPreparedTape) or controller.latest is None:
            raise ValueError("native callback differs from the saved causal preparation")
        if k <= self.callback and controller.latest != tuple(self.expected_observations[k]):
            raise ValueError("native callback differs from the saved causal preparation")
        if k <= self.callback:
            expected_previous = (
                (0.0, 316.0)
                if k == 0
                else (float(self.expected_stages[k - 1, 2]), float(self.expected_stages[k - 1, 3]))
            )
            if previous != expected_previous:
                raise ValueError("native applied prefix differs from the sealed preparation")
        self.checked_callbacks += 1
        if k == self.callback:
            if self.delivered:
                raise ValueError("committed reactor request delivered twice")
            self.prepared.delivery.session = session
            self.tick = self.prepared.deliver(self.publisher)
            self.delivered = True
            self.failed = (
                self.tick.disposition is not TickDisposition.ACTION_DELIVERED
                or not self.tick.delivery_trace.exact
            )
            return cast(ReactorDelivery | ReactorFinalDelivery | None, self.prepared.delivery.last_delivery)
        request = self.expected_tape[k]
        return cast(ReactorDelivery | ReactorFinalDelivery, session.advance(
            ReactorCommand(
                f"{self.branch_id}.prepared-replay.{k:04d}",
                D(k * 10),
                D(repr(float(request[0]))),
                D(repr(float(request[1]))),
            )
        ))


def acquire_committed_nominal_branch(
    *,
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    base: NativeEpisode,
    callback: int,
    request_index: int,
    prepared: PreparedRequest,
    publisher: Any,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> tuple[NativeEpisode, CommittedNominalBranchOwner]:
    """One owner-delivered full continuation from the saved selected prefix."""
    from .panel import assay_tape

    command = prepared.delivery.mapping.full_native.command
    if (
        base.root != scenario.unit_id
        or base.dt != 1
        or command.time_s != D(callback * 10)
        or request_index not in range(4)
    ):
        raise ValueError("owner branch changed its selected native preparation")
    tape = assay_tape(base, callback, float(command.feed_kg_s))
    owner = CommittedNominalBranchOwner(
        scenario.unit_id,
        f"{scenario.unit_id}.request-{request_index}",
        callback,
        base.observations,
        base.requests,
        base.stages,
        tape,
        prepared,
        publisher,
    )
    kwargs: dict[str, Any] = {"dt": 1.0, "owner": owner}
    if progress is not None:
        kwargs["progress"] = progress
    episode = acquire(
        source, scenario, f"{scenario.unit_id}.request-{request_index}.committed-task",
        WatchedPreparedTape(tape), **kwargs,
    )
    if episode.complete and (not owner.delivered or owner.checked_callbacks != 2880):
        raise ValueError("complete nominal branch omitted its actual owner delivery")
    return episode, owner


def acquire_committed_refined_replay(
    *,
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    base: NativeEpisode,
    refined_preparation: NativeEpisode | None,
    callback: int,
    feed_kg_s: float,
    request_id: str,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> NativeEpisode:
    """Replay the same nominal tape at dt=.5 after the owner commitment."""
    from .panel import assay_tape

    if (
        base.root != scenario.unit_id
        or refined_preparation is not None and refined_preparation.root != scenario.unit_id
        or base.dt != 1
        or refined_preparation is not None and refined_preparation.dt != .5
        or request_id not in {
            f"{scenario.unit_id}.request-{index}" for index in range(4)
        }
    ):
        raise ValueError("refined branch lacks its exact selected preparation")
    tape = assay_tape(base, callback, feed_kg_s)
    kwargs: dict[str, Any] = {"dt": .5}
    if progress is not None:
        kwargs["progress"] = progress
    episode = acquire(
        source, scenario, f"{request_id}.refined-task", TapeController(tape), **kwargs,
    )
    if refined_preparation is not None and len(refined_preparation.observations) > callback and len(episode.observations) > callback and (
        not np.array_equal(episode.observations[:callback + 1], refined_preparation.observations[:callback + 1])
        or not np.array_equal(episode.requests[:callback], refined_preparation.requests[:callback])
        or not np.array_equal(episode.stages[:callback], refined_preparation.stages[:callback])
        or not np.array_equal(episode.requests, tape[:len(episode.requests)])
    ):
        raise ValueError("refined branch changed its frozen same-tape causal prefix")
    return episode


def acquire_prepared_evaluator_chart(
    *,
    source: ReactorBatchSource,
    scenario: ReactorAssignedScenario,
    nominal_preparation: NativeEpisode,
    refined_preparation: NativeEpisode | None,
    callback: int,
    acquire: Callable[..., NativeEpisode] = acquire_episode,
    progress: Callable[[int], None] | None = None,
) -> tuple[NativeEpisode, ...]:
    """Six separate evaluator branches, with one zero reference per view."""
    from .panel import FEED_WORDS, assay_tape

    if (
        nominal_preparation.root != scenario.unit_id
        or refined_preparation is not None and refined_preparation.root != scenario.unit_id
        or nominal_preparation.dt != 1
        or refined_preparation is not None and refined_preparation.dt != .5
        or not 0 < callback < 2880
    ):
        raise ValueError("D evaluator chart lacks the selected paired preparation")
    branches: list[NativeEpisode] = []
    for word, feed in enumerate(FEED_WORDS):
        tape = assay_tape(nominal_preparation, callback, feed)
        for view, preparation in enumerate((nominal_preparation, refined_preparation)):
            kwargs: dict[str, Any] = {"dt": .5 if view else 1.0}
            if progress is not None:
                kwargs["progress"] = lambda value, index=len(branches): progress(index * 2880 + value)
            branch = acquire(
                source,
                scenario,
                f"{scenario.unit_id}.evaluator-word-{word}.view-{view}",
                TapeController(tape), **kwargs,
            )
            if preparation is not None and len(preparation.observations) > callback and len(branch.observations) > callback and (
                branch.root != scenario.unit_id
                or not np.array_equal(
                    branch.observations[:callback + 1],
                    preparation.observations[:callback + 1],
                )
                or not np.array_equal(branch.stages[:callback], preparation.stages[:callback])
                or not np.array_equal(branch.requests, tape[:len(branch.requests)])
            ):
                raise ValueError("D evaluator changed the shared native prefix or tape")
            branches.append(branch)
    return tuple(branches)
