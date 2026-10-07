"""Fixed diagnostic policy substitutions and pinned published-source comparators."""

from dataclasses import replace
from hashlib import sha256
from typing import Any
from .numerical import Decision, FrozenFit, select, digest
from .controller import NumericalController
from .config import COMPARATOR_SOURCES


class DiagnosticController(NumericalController):
    def __init__(
        self, model: FrozenFit, q: float, actuator: Any, arm: str, proposal: Any = None
    ) -> None:
        if arm not in ("F0", "F1", "MARGIN", "FIXED") or (arm == "FIXED") != (proposal is not None):
            raise ValueError("undeclared diagnostic treatment")
        super().__init__(model, q, actuator)
        self.arm, self.proposal = arm, proposal
        self.policy_identity = digest({"numerical_policy": self.policy_identity, "arm": arm})

    def reset(self, spec: dict[str, Any]) -> None:
        super().reset(spec)
        if self.proposal is not None:
            self.proposal.reset(spec)

    def make_decision(
        self, projections: Any, t_s: float, y: dict[str, float], dt_s: float
    ) -> Decision:
        decision = select(
            self.history,
            self.previous[1],
            projections,
            self.model,
            self.q,
            causal_state=self._causal,
            temperature_halfwidth=0.25 if self.arm == "MARGIN" else None,
        )
        if self.arm != "FIXED":
            return decision
        feed, jacket = self.proposal.step(t_s, y, dt_s)
        eligible = [
            c
            for c in decision.candidates
            if not c.reasons and c.projection.action == min(c.aliases)
        ]
        if not eligible:
            return decision
        selected = min(
            eligible,
            key=lambda c: (
                abs(c.projection.requested[0] - feed) / 0.032
                + abs(c.projection.requested[1] - jacket) / 68.3,
                c.projection.action,
            ),
        ).projection.action
        return replace(decision, selected=selected)


def published_controller(arm: str, source: bytes) -> Any:
    expected = next((digest for name, _, digest in COMPARATOR_SOURCES if name == arm), None)
    if expected is None or sha256(source).hexdigest() != expected:
        raise ValueError("published comparator source/configuration differs")
    # The only executable values are these exact four source-pinned published
    # implementations. Configuration cannot name another module or source.
    namespace: dict[str, Any] = {"__name__": f"reactor_published_{arm.lower()}"}
    exec(compile(source, f"<pinned-reactor-{arm.lower()}>", "exec"), namespace)
    cls = namespace["Controller"]
    return cls(back_off=0.0 if arm == "SCHEDULED_BACKOFF_ZERO" else 0.5) if arm in ("SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF") else cls()
