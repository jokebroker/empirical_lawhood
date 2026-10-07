"""Replay actual cached deliveries of a bounded, already committed native word."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import Any, cast
import numpy as np
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeController
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession
from empirical_lawhood.adapters.simulators.reactor_regime_response.committed_branch import PreparedRequest, WatchedPreparedTape
from empirical_lawhood.runtime.controller_runtime import TickDisposition
from .contracts import ReactorCommand, ReactorDelivery, ReactorFinalDelivery


@dataclass
class FiniteCommittedBranchOwner:
    """Adapt the existing per-callback acquisition loop to a macro delivery.

    At the cutoff the generic delivery owner advances the real bridge the declared number of
    times. Its actual delivery records then supply the remaining callbacks
    of the existing recorder. These are neither new bridge calls nor invented
    observations. At the declared endpoint the recorder and bridge clocks meet again.
    """

    root: str
    branch_id: str
    callback: int
    expected_observations: np.ndarray
    expected_stages: np.ndarray
    tape: np.ndarray
    prepared: PreparedRequest
    publisher: Any
    failed: bool = False
    delivered: bool = False
    checked_callbacks: int = 0
    tick: Any = None

    def __post_init__(self) -> None:
        mapping = self.prepared.delivery.mapping
        if (
            not 60 <= self.callback <= 2508
            or (
                self.expected_observations.shape[1:] != (4,)
                or len(self.expected_observations) <= self.callback
            )
            or (self.expected_stages.shape[1:] != (4,) or len(self.expected_stages) < self.callback)
            or self.tape.shape != (2880, 2)
            or any(
                m.command.time_s != D((self.callback + i) * 10)
                or tuple(self.tape[self.callback + i])
                != (float(m.command.feed_kg_s), float(m.command.jacket_k))
                for i, m in enumerate(mapping.mappings)
            )
        ):
            raise ValueError("macro owner lacks its exact frozen prefix and bounded-command tape")

    def advance(
        self,
        controller: NativeController,
        previous: tuple[float, float],
        session: BatchDeliverySession,
    ) -> ReactorDelivery | ReactorFinalDelivery | None:
        k = self.checked_callbacks
        if (
            self.failed
            or k >= 2880
            or not isinstance(controller, WatchedPreparedTape)
            or controller.latest is None
        ):
            raise ValueError("macro recorder callback is missing, repeated or after failure")
        if k <= self.callback:
            prior = (0.0, 316.0) if not k else tuple(map(float, self.expected_stages[k - 1, 2:]))
            if controller.latest != tuple(self.expected_observations[k]) or previous != prior:
                raise ValueError("macro owner received another causal preparation")
        self.checked_callbacks += 1
        if k == self.callback:
            if self.delivered:
                raise ValueError("macro was already delivered")
            self.prepared.delivery.session = session
            self.tick = self.prepared.deliver(self.publisher)
            self.delivered = True
            self.failed = (
                self.tick.disposition is not TickDisposition.ACTION_DELIVERED
                or not self.tick.delivery_trace.exact
                or len(self.prepared.delivery.last_deliveries)
                != len(self.prepared.delivery.mapping.mappings)
            )
            return (
                self.prepared.delivery.last_deliveries[0]
                if self.prepared.delivery.last_deliveries
                else None
            )
        if self.callback < k < self.callback + len(self.prepared.delivery.mapping.mappings):
            raw = self.prepared.delivery.last_deliveries[k - self.callback]
            earlier = self.prepared.delivery.last_deliveries[k - self.callback - 1]
            m = earlier.next_measurement
            if controller.latest != tuple(
                float(v) for v in (m.time_s, m.t_reactor_k, m.t_jacket_k, m.dosed_kg)
            ) or previous != (float(earlier.applied_feed_kg_s), float(earlier.applied_jacket_k)):
                raise ValueError("macro recorder changed an actual cached native delivery")
            return cast(ReactorDelivery, raw)
        request = self.tape[k]
        return session.advance(
            ReactorCommand(
                f"{self.branch_id}.replay-{k:04d}",
                D(k * 10),
                D(repr(float(request[0]))),
                D(repr(float(request[1]))),
            )
        )
