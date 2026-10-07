"""Actual callback prefix capture before a conditional native command is delivered."""

from dataclasses import dataclass, field
from decimal import Decimal as D
from typing import Callable

import numpy as np

from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalContext
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeController
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import BatchDeliverySession
from empirical_lawhood.adapters.simulators.reactor_regime_response.committed_branch import WatchedPreparedTape
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand, ReactorDelivery, ReactorFinalDelivery
from empirical_lawhood.kernel.provenance import ObjectIdentity


@dataclass
class CausalTapeOwner:
    root: str
    first_callback: int
    seal: Callable[[ClassicalContext], ObjectIdentity]
    failed: bool = False
    context: ClassicalContext | None = None
    published: ObjectIdentity | None = None
    observations: list[tuple[float, float, float, float]] = field(default_factory=list)
    requests: list[tuple[float, float]] = field(default_factory=list)
    stages: list[tuple[float, float, float, float]] = field(default_factory=list)
    exposures: list[tuple[tuple[float, float, float, float], ...]] = field(default_factory=list)

    def advance(
        self,
        controller: NativeController,
        previous: tuple[float, float],
        session: BatchDeliverySession,
    ) -> ReactorDelivery | ReactorFinalDelivery:
        if not isinstance(controller, WatchedPreparedTape) or controller.latest is None:
            raise ValueError("conditional assay requires the actual observed native callback")
        k = len(self.requests)
        if controller.latest[0] != k * 10 or previous != (
            (0.0, 316.0) if not k else self.stages[-1][2:]
        ):
            raise ValueError("causal recorder changed native callback order or applied history")
        self.observations.append(controller.latest)
        if k == self.first_callback + 12:
            if self.published is not None:
                raise ValueError("conditional prediction was already published")
            self.context = ClassicalContext(
                self.root,
                "induced",
                k,
                LocalArrayPayload.pack(
                    {
                        "observations": np.asarray(self.observations),
                        "requests": np.asarray(self.requests),
                        "stages": np.asarray(self.stages),
                        "exposure": np.asarray(self.exposures),
                    }
                ),
                (),
                self.first_callback,
            )
            # The installed publisher returns only after immutable persistence.
            # Neither a native grid nor a later response is available here.
            self.published = self.seal(self.context)
        requested = tuple(map(float, controller.tape[k]))
        command = ReactorCommand(
            f"{self.root}.conditional-assay-{k:04d}", D(k * 10), *(D(repr(v)) for v in requested)
        )
        delivered = session.advance(command)
        if delivered.command != command:
            raise ValueError("conditional assay received another requested command")
        self.requests.append((requested[0], requested[1]))
        self.stages.append(
            (
                float(delivered.accepted_feed_kg_s),
                float(delivered.accepted_jacket_k),
                float(delivered.applied_feed_kg_s),
                float(delivered.applied_jacket_k),
            )
        )
        self.exposures.append(
            tuple(
                (float(e.time_s), float(e.duration_s), float(e.feed_kg_s), float(e.jacket_k))
                for e in delivered.exposures
            )
        )
        return delivered
