"""Native callback history and numerical refusal; no forged owner commitments."""

from __future__ import annotations
from dataclasses import asdict
from typing import Any
from .numerical import CausalFeatures, Decision, FrozenFit, Observation, digest, select


class EmpiricalNonattempt(RuntimeError):
    """The native API has no refusal tuple; the unchanged grader sees failure."""


class NumericalController:
    def __init__(self, model: FrozenFit, q: float, actuator: Any) -> None:
        self.model, self.q, self.actuator = model, q, actuator
        self.policy_identity = digest({"model": asdict(model), "q": q})
        self._frozen_model, self._frozen_q = model, q
        self._causal = CausalFeatures()
        self.history: tuple[Observation, ...] = ()
        self.previous = (0.0, 316.0)
        self.decisions: list[Decision] = []
        self.requests: list[tuple[float, float]] = []
        self.chain: list[str] = []
        self.refused = False
        self._committed_history: tuple[Observation, ...] = self.history

    def reset(self, spec: dict[str, Any]) -> None:
        if spec["timing"]["sample_dt_s"] != 10 or spec["limits"]["horizon_s"] != 28800:
            raise ValueError("native spec changes the frozen clock")
        self.history = ()
        self._causal = CausalFeatures()
        self._committed_history = self.history
        self.previous = (0.0, 316.0)
        self.decisions = []
        self.requests = []
        self.chain = []
        self.refused = False

    def step(self, t_s: float, y: dict[str, float], dt_s: float) -> tuple[float, float]:
        if self.refused or dt_s != 10 or t_s != len(self.history) * 10:
            raise ValueError("stale/reordered callback or resumed refusal")
        if self.model is not self._frozen_model or self.q != self._frozen_q:
            raise ValueError("frozen policy was changed after construction")
        if self.history is not self._committed_history:
            raise ValueError("observation history changed after decision")
        observation = Observation(
            float(t_s), float(y["t_reactor_k"]), float(y["t_jacket_k"]), float(y["dosed_kg"])
        )
        self.history += (observation,)
        self._causal.append(observation)
        projections = self.actuator.project(observation, self.previous)
        decision = self.make_decision(projections, t_s, y, dt_s)
        self.decisions.append(decision)
        self._committed_history = self.history
        # A numerical audit chain is explicitly not an owner commitment receipt.
        self.chain.append(
            digest(
                {
                    "previous": self.chain[-1] if self.chain else None,
                    "policy": self.policy_identity,
                    "callback": len(self.history) - 1,
                    "decision": asdict(decision),
                }
            )
        )
        if decision.selected is None:
            self.refused = True
            raise EmpiricalNonattempt(f"NONATTEMPT callback={len(self.history) - 1}")
        projection = projections[decision.selected]
        self.previous = projection.applied
        self.requests.append(projection.requested)
        return (float(projection.requested[0]), float(projection.requested[1]))

    def make_decision(
        self, projections: Any, t_s: float, y: dict[str, float], dt_s: float
    ) -> Decision:
        return select(
            self.history,
            self.previous[1],
            projections,
            self.model,
            self.q,
            causal_state=self._causal,
        )
