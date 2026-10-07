"Full-batch binding for the pinned native callback/delivery bridge."

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .bridge import ReactorBridge, _Boundary, _decimal
from .batch_design import ReactorAssignedScenario
from .contracts import ReactorCommand, ReactorDelivery, ReactorExposure, ReactorFinalDelivery


class ReactorBatchBridge(ReactorBridge):
    """2,880 distinct native decisions, with an explicit final interval.

    Public observations still cross the callback barrier one at a time. Native
    truth is available only after the complete result returns, for the sealed
    evaluator output. Closing early never creates a completed result.
    """

    _maximum_decisions = 2880
    _result: Any = None

    def __init__(
        self,
        *,
        plant_bytes: bytes,
        params_bytes: bytes,
        public_scenarios_bytes: bytes,
        plant_dt_s: Decimal,
        scenario_id: str | None = None,
        assigned_scenario: ReactorAssignedScenario | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if (scenario_id is None) == (assigned_scenario is None):
            raise ValueError("batch bridge requires exactly one frozen scenario selection")
        super().__init__(
            plant_bytes=plant_bytes,
            params_bytes=params_bytes,
            public_scenarios_bytes=public_scenarios_bytes,
            scenario_id="nominal" if assigned_scenario is not None else str(scenario_id),
            plant_dt_s=plant_dt_s,
            timeout_s=timeout_s,
        )
        if assigned_scenario is not None:
            self._scenario = assigned_scenario.native_dict()

    def advance(self, command: ReactorCommand) -> ReactorDelivery | ReactorFinalDelivery:
        delivery = self._deliver_boundary(command).delivery
        if delivery is None:
            raise ValueError("native batch advance lacks observed delivery")
        return delivery

    def __enter__(self) -> ReactorBatchBridge:
        return self

    @property
    def completed_result(self) -> Any:
        if self._decisions != self._maximum_decisions or self._result is None:
            raise ValueError("stopped or incomplete batch has no completed native result")
        return self._result

    def _finish_native(
        self,
        result: Any,
        params: Any,
        pending: ReactorCommand | None,
        accepted: tuple[Decimal, Decimal, Decimal, Decimal] | None,
        exposures: tuple[ReactorExposure, ...],
    ) -> None:
        if (
            pending is None
            or accepted is None
            or result is None
            or _decimal(params.horizon_s) != Decimal(28800)
            or _decimal(result.time_s[-1]) != Decimal(28800)
            or len(result.time_s) != int(Decimal(28800) / self._plant_dt_s) + 1
        ):
            raise ValueError("native batch result does not cover its full declared grid")
        delivery = ReactorFinalDelivery(pending, *accepted, exposures, Decimal(28800))
        self._result = result
        self._responses.put_nowait(_Boundary(None, delivery))
