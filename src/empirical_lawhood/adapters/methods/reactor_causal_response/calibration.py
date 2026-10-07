"Raw whole-root prediction operands; no local scientific finalizer or controller use."

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
import numpy as np
from .numerical import FrozenFit, digest


@dataclass(frozen=True)
class RootOperands:
    root: str
    clocks: np.ndarray
    actions: np.ndarray
    features: np.ndarray
    labels: np.ndarray  # callback, view, (Tpeak,Cend,Dend)
    # Every native step retained separately in each view; columns time,dt,feed,jacket.
    projected_nominal: np.ndarray
    delivered_nominal: np.ndarray
    projected_refined: np.ndarray
    delivered_refined: np.ndarray
    requests: np.ndarray  # callback,view,(feed,jacket)
    observations: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    consumer_q: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    model_sha256: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    stages: np.ndarray = field(default_factory=lambda: np.empty((0, 2, 4)))


def causal_reasons(
    raw: RootOperands, model: FrozenFit, expected_q: float | None
) -> tuple[str, ...]:
    """Reconstruct chosen-path features/actions before any response-label access."""
    from .controller import NumericalController, EmpiricalNonattempt
    from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator

    if (
        raw.observations.shape != (2880, 4)
        or raw.consumer_q.shape != (1,)
        or raw.model_sha256.shape != (32,)
    ):
        return ("MISSING_CAUSAL_POLICY_OPERANDS",)
    if (
        not np.isfinite(raw.observations).all()
        or not np.isfinite(raw.consumer_q).all()
        or raw.consumer_q[0] < 0
    ):
        return ("NONFINITE_CAUSAL_POLICY_OPERANDS",)
    if not np.array_equal(raw.model_sha256, list(bytes.fromhex(digest(asdict(model))))):
        return ("FROZEN_MODEL_IDENTITY_DIFFERS",)
    if expected_q is not None and raw.consumer_q[0] != expected_q:
        return ("FROZEN_CONSUMER_Q_DIFFERS",)
    if raw.actions.dtype.kind not in "iu" or not np.isin(raw.actions, range(9)).all():
        return ("INVALID_NOMINAL_ACTION_ID",)
    policy = NumericalController(model, float(raw.consumer_q[0]), Actuator())
    for k, (t, temperature, jacket, dose) in enumerate(raw.observations):
        if t != k * 10 or (k == 0 and dose != 0):
            return ("CAUSAL_CLOCK_OR_INITIAL_DOSE_DIFFERS",)
        if k and abs(dose - raw.labels[k - 1, 0, 2]) > 1e-6:
            return ("OBSERVED_DOSE_DELIVERY_DIFFERS",)
        try:
            request = policy.step(
                float(t),
                {
                    "t_reactor_k": float(temperature),
                    "t_jacket_k": float(jacket),
                    "dosed_kg": float(dose),
                },
                10.0,
            )
        except (ValueError, EmpiricalNonattempt):
            return ("CHOSEN_CONSUMER_REFUSED_OR_INVALID",)
        chosen = policy.decisions[-1].selected
        assert chosen is not None
        candidate = policy.decisions[-1].candidates[chosen]
        if (
            raw.actions[k] != chosen
            or not np.array_equal(raw.requests[k, 0], request)
            or not np.array_equal(raw.features[k], candidate.features)
        ):
            return ("CAUSAL_FEATURE_ACTION_OR_POLICY_DIFFERS",)
        # No need to retain all nine candidate objects after validation.
        policy.decisions.clear()
    return ()


@dataclass(frozen=True)
class RootScore:
    root: str
    value: float | None
    reasons: tuple[str, ...]
    callbacks: int
    receiver_views: int


@dataclass(frozen=True)
class Calibration:
    scores: tuple[RootScore, ...]
    rank: int
    q: float | None
    halfwidth: tuple[float, float] | None
    disposition: str


def root_score(
    raw: RootOperands, model: FrozenFit, *, consumer_q: float | None = None
) -> RootScore:
    reasons: list[str] = []
    if (
        raw.clocks.shape != (2880,)
        or not np.array_equal(raw.clocks, np.arange(2880) * 10)
        or raw.actions.shape != (2880,)
        or raw.features.shape != (2880, 23)
        or raw.labels.shape != (2880, 2, 3)
        or raw.requests.shape != (2880, 2, 2)
    ):
        return RootScore(raw.root, None, ("INCOMPLETE_ASSIGNED_CENSUS",), len(raw.clocks), 0)
    reasons.extend(causal_reasons(raw, model, consumer_q))
    if not np.isfinite(raw.labels).all():
        reasons.append("MISSING_OR_NONFINITE_LABEL")
    if not np.all(np.abs(raw.labels[:, 0] - raw.labels[:, 1]) <= (0.01, 0.0002, 0.000001)):
        reasons.append("NUMERICAL_INVALID")
    if not np.array_equal(raw.requests[:, 0], raw.requests[:, 1]):
        reasons.append("REFINEMENT_TAPE_DIFFERS")
    from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import project_tape_stages

    for projected, delivered, steps, dt in (
        (raw.projected_nominal, raw.delivered_nominal, 10, 1.0),
        (raw.projected_refined, raw.delivered_refined, 20, 0.5),
    ):
        try:
            view = 0 if dt == 1 else 1
            expected_stages, reconstructed = project_tape_stages(raw.requests[:, view], dt)
            if raw.stages.shape != (2880, 2, 4) or not np.array_equal(
                raw.stages[:, view], expected_stages
            ):
                reasons.append("ACCEPTED_APPLIED_STAGE_MISMATCH")
            if not np.array_equal(projected, reconstructed):
                reasons.append("REQUEST_PROJECTION_MISMATCH")
        except ValueError:
            reasons.append("REQUEST_PROJECTION_MISMATCH")
        if projected.shape != (2880, steps, 4) or delivered.shape != projected.shape:
            reasons.append("INCOMPLETE_EXPOSURE")
            continue
        if (
            not np.isfinite(delivered).all()
            or not np.array_equal(projected, delivered)
            or not np.array_equal(delivered[:, :, 0], raw.clocks[:, None] + np.arange(steps) * dt)
            or not np.all(delivered[:, :, 1] == dt)
        ):
            reasons.append("DELIVERY_OR_CLOCK_INVALID")
        else:
            # Actual dose is checked against each entire profile, including the final delivery.
            end_dose = np.cumsum(np.sum(delivered[:, :, 1] * delivered[:, :, 2], axis=1))
            view = 0 if dt == 1 else 1
            if not np.all(np.abs(end_dose - raw.labels[:, view, 2]) <= 0.000001):
                reasons.append("DOSE_EXPOSURE_MISMATCH")
    # NUMERICAL-01 refines the entire pinned actuator/integrator pipeline.
    # The exact within-view projections above retain dose-cap exposure changes.
    if not model.support_mask(raw.features, raw.clocks, raw.actions).all():
        reasons.append("OUTSIDE_SUPPORT")
    predicted = model.predict(raw.features)
    if not np.isfinite(predicted).all():
        reasons.append("NONFINITE_PREDICTION")
    maximum = (
        None
        if reasons
        else float(np.max(np.abs(raw.labels[:, :, :2] - predicted[:, None, :]) / (0.25, 0.01)))
    )
    return RootScore(raw.root, maximum, tuple(sorted(set(reasons))), 2880, 2880 * 2 * 2)


def calibrate(
    roots: tuple[RootOperands, ...], model: FrozenFit, *, consumer_q: float | None = None
) -> Calibration:
    expected = tuple(f"reactor-empirical-calibration-{i:03d}" for i in range(32))
    if tuple(r.root for r in roots) != expected:
        raise ValueError("complete assigned 32-root calibration census required")
    scores = tuple(root_score(r, model, consumer_q=consumer_q) for r in roots)
    rank = math.ceil(33 * 0.95)
    values = sorted(float("inf") if s.value is None else s.value for s in scores)
    q = values[rank - 1] if rank <= len(values) else float("inf")
    return Calibration(
        scores,
        rank,
        q if math.isfinite(q) else None,
        (q * 0.25 + 0.01, q * 0.01 + 0.0002) if math.isfinite(q) else None,
        "CALIBRATED" if q <= 1 else "CALIBRATION_UNUSABLE",
    )


@dataclass(frozen=True)
class Adequacy:
    root: str
    adequate: bool
    reasons: tuple[str, ...]


def adequacy(raw: RootOperands, model: FrozenFit, q: float | None) -> Adequacy:
    score = root_score(raw, model, consumer_q=q)
    reasons = list(score.reasons)
    if q is None or not math.isfinite(q) or q < 0:
        reasons.append("CALIBRATION_UNUSABLE")
    else:
        width = np.array((q * 0.25 + 0.01, q * 0.01 + 0.0002))
        if np.any(width > (0.26, 0.0102)):
            reasons.append("WIDTH")
        if raw.labels.shape == (2880, 2, 3) and raw.features.shape == (2880, 23):
            errors = np.abs(raw.labels[:, :, :2] - model.predict(raw.features)[:, None, :])
            if not np.all(errors <= width):
                reasons.append("COVERAGE")
    return Adequacy(raw.root, not reasons, tuple(sorted(set(reasons))))


def qualification_operands(
    roots: tuple[RootOperands, ...], model: FrozenFit, q: float | None
) -> tuple[tuple[Adequacy, ...], float]:
    from empirical_lawhood.adapters.methods.prepared_response.statistics import cp_bounds

    if tuple(r.root for r in roots) != tuple(
        f"reactor-empirical-qualification-{i:03d}" for i in range(32)
    ):
        raise ValueError("complete assigned qualification census required")
    values = tuple(adequacy(r, model, q) for r in roots)
    bound = cp_bounds(sum(v.adequate for v in values), 32, alpha=0.05)[0]
    return values, float(bound)


@dataclass(frozen=True)
class TaskOperands:
    completed: bool
    temperature: tuple[float, ...]
    final_dose: float | None
    final_conversion: float | None
    completion_time: float | None
    reference_time: float | None


def task_predicates(a: bool, task: TaskOperands) -> tuple[bool, bool, bool]:
    "Descriptive raw A/J/C predicates ONLY; never an existing-owner controller-use verdict."
    finite = all(
        v is not None and math.isfinite(v)
        for v in (task.final_dose, task.final_conversion, task.completion_time, task.reference_time)
    )
    j = bool(
        task.completed
        and finite
        and task.temperature
        and all(math.isfinite(t) and t <= 356.2 for t in task.temperature)
        and task.final_dose is not None
        and task.final_dose >= 0.999 * 287.3
        and task.final_conversion is not None
        and task.final_conversion >= 0.98
        and task.reference_time is not None
        and task.reference_time > 0
        and task.completion_time is not None
        and task.completion_time / task.reference_time <= 1.06
    )
    return a, j, a and j
