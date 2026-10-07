"""Synchronous two-decision bridge around the unmodified native simulate loop.

The worker waits at each native callback. A delivery returns only after the
plant has consumed the requested sample and reached the following callback.
Instrumentation reads native locals; it never calculates actuator behavior or
advances an independent model. This is a bounded prefix, not a completed batch.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
import json
from queue import Empty, Queue
import sys
from threading import Event, Thread
from types import FrameType, ModuleType, SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from .contracts import PARAMS_SHA256, PLANT_SHA256, PUBLIC_SCENARIOS_SHA256, ReactorCommand, ReactorDelivery, ReactorExposure, ReactorFinalDelivery, ReactorMeasurement


def _decimal(value: object) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("native reactor emitted a nonfinite value")
    return result


class _PrefixClosed(Exception):
    """Stop at a callback without authoring another request or a batch result."""


class ReactorWorkerStopError(RuntimeError):
    """Abort the task if native work cannot be joined; do not start another branch."""


@dataclass(frozen=True, slots=True)
class _Boundary:
    measurement: ReactorMeasurement | None
    delivery: ReactorDelivery | ReactorFinalDelivery | None


class ReactorBridge:
    """Task-local native session; construction alone does not start simulation.

    The caller must already hold source/execution authority. Bytes are supplied
    through authenticated task inputs, never discovered from caller paths.
    Each session owns a separate module and worker; no shared NumPy or upstream
    module globals are patched. Close is mandatory, including on refusal.
    """

    _maximum_decisions = 2

    def __init__(
        self,
        *,
        plant_bytes: bytes,
        params_bytes: bytes,
        public_scenarios_bytes: bytes,
        scenario_id: str,
        plant_dt_s: Decimal,
        timeout_s: float = 10.0,
    ) -> None:
        for data, digest in (
            (plant_bytes, PLANT_SHA256),
            (params_bytes, PARAMS_SHA256),
            (public_scenarios_bytes, PUBLIC_SCENARIOS_SHA256),
        ):
            if sha256(data).hexdigest() != digest:
                raise ValueError("reactor source bytes differ from the installed pin")
        if plant_dt_s not in (Decimal("1"), Decimal("0.5")):
            raise ValueError("reactor binding admits only the native and half-step views")
        if not 0 < timeout_s <= 60:
            raise ValueError("reactor bridge requires a finite bounded wait")
        scenarios = json.loads(public_scenarios_bytes)["scenarios"]
        selected = [s for s in scenarios if s["scenario_id"] == scenario_id]
        if len(selected) != 1:
            raise ValueError("reactor bridge requires one pinned public scenario")
        self._source = plant_bytes
        self._params = params_bytes
        self._scenario = selected[0]
        self._plant_dt_s = plant_dt_s
        self._timeout_s = timeout_s
        self._requests: Queue[ReactorCommand] = Queue(maxsize=1)
        self._responses: Queue[_Boundary | BaseException] = Queue(maxsize=1)
        self._stop = Event()
        self._thread: Thread | None = None
        self._measurement: ReactorMeasurement | None = None
        self._decisions = 0
        self._decision_ids: set[str] = set()
        self._closed = False

    def start(self) -> ReactorMeasurement:
        if self._thread is not None or self._closed:
            raise ValueError("reactor bridge cannot restart a native identity")
        self._thread = Thread(target=self._worker, name="tbs-reactor-prefix", daemon=True)
        self._thread.start()
        boundary = self._receive()
        if (
            boundary.delivery is not None
            or boundary.measurement is None
            or boundary.measurement.time_s != 0
        ):
            self.close()
            raise ValueError("reactor bridge initial callback differs")
        self._measurement = boundary.measurement
        return boundary.measurement

    def deliver(self, command: ReactorCommand) -> ReactorDelivery:
        delivery = self._deliver_boundary(command).delivery
        if not isinstance(delivery, ReactorDelivery):
            raise ValueError("prefix delivery requires its following native observation")
        return delivery

    def _deliver_boundary(self, command: ReactorCommand) -> _Boundary:
        if self._closed or self._measurement is None or self._decisions >= self._maximum_decisions:
            raise ValueError("reactor bridge is not at an available decision boundary")
        if command.time_s != self._measurement.time_s:
            raise ValueError("reactor request uses a stale or future observation clock")
        if command.decision_id in self._decision_ids:
            raise ValueError("reactor decision identity cannot be reused")
        self._decision_ids.add(command.decision_id)
        self._requests.put_nowait(command)
        boundary = self._receive()
        if boundary.delivery is None or boundary.delivery.command != command:
            self.close()
            raise ValueError("reactor delivery does not belong to the pending decision")
        self._decisions += 1
        self._measurement = boundary.measurement
        return boundary

    def close(self) -> None:
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._timeout_s)
            if self._thread.is_alive():
                raise ReactorWorkerStopError("reactor worker did not stop; delivery is unevaluable")

    def __enter__(self) -> ReactorBridge:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _receive(self) -> _Boundary:
        try:
            value = self._responses.get(timeout=self._timeout_s)
        except Empty as exc:
            self.close()
            raise RuntimeError("reactor delivery timed out; no completion evidence") from exc
        if isinstance(value, BaseException):
            self.close()
            raise RuntimeError("native reactor failed; no completion evidence") from value
        return value

    def _worker(self) -> None:
        module_name = f"_reactor_native_{uuid4().hex}"
        module = ModuleType(module_name)
        sys.modules[module_name] = module
        try:
            # Executable bytes have the code-owned exact pin checked in __init__.
            exec(compile(self._source, "<pinned-reactor-native>", "exec"), module.__dict__)
            self._run_native(module)
        except _PrefixClosed:
            pass
        except BaseException as exc:
            if not self._stop.is_set():
                self._responses.put_nowait(exc)
        finally:
            sys.settrace(None)
            sys.modules.pop(module_name, None)

    def _run_native(self, module: ModuleType) -> None:
        # Adapt only from_json's read port; retain the upstream JSON-to-params map.
        native: Any = module
        native.pathlib = SimpleNamespace(
            Path=lambda _path: SimpleNamespace(read_text=lambda: self._params.decode("utf-8"))
        )
        params = replace(
            native.PlantParams.from_json("authenticated-task-input"),
            plant_dt_s=float(self._plant_dt_s),
        )
        scenario = native.Scenario.from_dict(self._scenario)
        self._drive(native.simulate, native.rk4_step, params, scenario)

    def _drive(
        self, simulate: Callable[..., Any], rk4_step: Callable[..., Any], params: Any, scenario: Any
    ) -> None:
        tree = ast.parse(self._source)
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "simulate"
        )
        # The pinned native loop has one inner `for k in range(sub)` after both
        # actuator updates. Read its locals rather than reproducing their rules.
        sites = [
            n.lineno
            for n in ast.walk(function)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "k"
        ]
        if len(sites) != 1:
            raise ValueError("native actuator instrumentation site differs")
        accepted: tuple[Decimal, Decimal, Decimal, Decimal] | None = None
        exposures: list[ReactorExposure] = []
        pending: ReactorCommand | None = None

        def trace(frame: FrameType, event: str, _arg: object) -> Any:
            nonlocal accepted
            if frame.f_code not in (simulate.__code__, rk4_step.__code__):
                return None
            if self._stop.is_set():
                raise _PrefixClosed
            if frame.f_code is simulate.__code__:
                if event == "line" and frame.f_lineno == sites[0] and accepted is None:
                    values = frame.f_locals
                    accepted = tuple(
                        _decimal(values[k])
                        for k in ("feed_target", "tj_target", "feed_applied", "tj_applied")
                    )  # type: ignore[assignment]
                return trace
            if frame.f_code is rk4_step.__code__:
                if event == "return":
                    values = frame.f_locals
                    exposures.append(
                        ReactorExposure(
                            *(_decimal(values[k]) for k in ("t", "dt", "feed_kg_s", "tj_cmd_k"))
                        )
                    )
                return trace
            return None

        def callback(
            time_s: float, measurement: dict[str, float], dt_s: float
        ) -> tuple[float, float]:
            nonlocal pending, accepted
            self._retain_boundary(sys._getframe(1), time_s)
            visible = ReactorMeasurement(
                _decimal(time_s),
                _decimal(dt_s),
                *(_decimal(measurement[k]) for k in ("t_reactor_k", "t_jacket_k", "dosed_kg")),
            )
            delivery = None
            if pending is not None:
                if accepted is None:
                    raise ValueError("native actuator stages were not observed")
                delivery = ReactorDelivery(pending, *accepted, tuple(exposures), visible)
            self._responses.put_nowait(_Boundary(visible, delivery))
            while not self._stop.is_set():
                try:
                    pending = self._requests.get(timeout=0.05)
                    accepted = None
                    exposures.clear()
                    return float(pending.feed_kg_s), float(pending.jacket_k)
                except Empty:
                    continue
            raise _PrefixClosed

        sys.settrace(trace)
        result = simulate(params, scenario, callback)
        self._finish_native(result, params, pending, accepted, tuple(exposures))

    def _retain_boundary(self, frame: FrameType, time_s: float) -> None:
        """Optional evaluator instrumentation; the ordinary bridge retains no truth."""

    def _finish_native(
        self,
        result: Any,
        params: Any,
        pending: ReactorCommand | None,
        accepted: tuple[Decimal, Decimal, Decimal, Decimal] | None,
        exposures: tuple[ReactorExposure, ...],
    ) -> None:
        raise ValueError("native full batch ended unexpectedly inside a two-decision prefix")
