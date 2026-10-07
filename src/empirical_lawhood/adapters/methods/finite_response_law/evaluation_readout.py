"Independent scalar finite response-law evaluation events and all-root inference; never selects effects.\n\nUses frozen predictions/choices and authenticated measurements. It does not use\nthe generic controller-use verdict to construct scalar success, loss or missingness.\n"

from dataclasses import dataclass
from decimal import Decimal as D
from math import comb, isfinite
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.prepared_response.statistics import cp_bounds, paired_difference_bounds
from empirical_lawhood.adapters.methods.effective_law_comparison.statistics import complete_unit_bootstrap_mean
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationDisposition
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import _assigned_stage_unit
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition

from .control_closeout import FiniteResponseLawRootSealedControl
from .control_delivery import _force
from .control_reveal import FiniteResponseLawProspectiveRootEvaluation
from .evaluation_native_records import FiniteResponseLawEvaluationViewObservation
from .law_binding import NATIVE_WORDS, output_quantities
from .science import FiniteResponseLawScienceSpec, seed_for

SPEC = FiniteResponseLawScienceSpec()
BOUNDARIES = ("cached", "composed", "direct")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawScalarConsumerEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-scalar-consumer-event'
    policy_id: str
    selected_word: int
    independently_expected_word: int
    success: bool
    false_admission: bool
    numerical_agreement: bool
    measured_available: bool

    def __post_init__(self) -> None:
        if (
            self.policy_id not in {f"{b}.consumer-{c}" for b in BOUNDARIES for c in (0, 1)}
            or any(
                type(v) is not int or not -1 <= v < 8
                for v in (self.selected_word, self.independently_expected_word)
            )
            or self.selected_word != self.independently_expected_word
            or self.false_admission != (self.selected_word >= 0 and not self.success)
            or self.success
            and (
                self.selected_word < 0
                or not self.numerical_agreement
                or not self.measured_available
            )
        ):
            raise ValueError("Finite response-law evaluation scalar event disagrees with the frozen finite native decision")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawRootInferenceOperands(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-root-inference-operands'
    root_id: str
    events: tuple[FiniteResponseLawScalarConsumerEvent, ...]
    response_losses: tuple[tuple[str, D | None], ...]
    response_numerical_agreement: bool
    prediction_halfwidths: tuple[tuple[str, tuple[D | None, ...]], ...]
    bootstrap_seed: int | None = None

    def __post_init__(self) -> None:
        seed_for("bootstrap", self.root_id.rsplit(".r", 1)[0], committed_seed=self.bootstrap_seed)
        if (
            self.root_id not in {f"prospective-evaluation.r{i:03d}" for i in range(64)}
            and not _assigned_stage_unit(self.root_id, "prospective-evaluation", 64)
            or tuple(e.policy_id for e in self.events)
            != tuple(sorted({e.policy_id for e in self.events}))
            or tuple(b for b, _ in self.response_losses) != BOUNDARIES
            or tuple(b for b, _ in self.prediction_halfwidths) != BOUNDARIES
            or any(
                len(values) != 32
                or any(v is not None and (not v.is_finite() or v < 0) for v in values)
                for _, values in self.prediction_halfwidths
            )
            or any(v is not None and (not v.is_finite() or v < 0) for _, v in self.response_losses)
        ):
            raise ValueError("Finite response-law evaluation root readout changes its assigned census or finite losses")


def root_operands(
    sealed: FiniteResponseLawRootSealedControl,
    views: tuple[FiniteResponseLawEvaluationViewObservation, ...],
    prospective_root_evaluation: FiniteResponseLawProspectiveRootEvaluation,
) -> FiniteResponseLawRootInferenceOperands:
    root = sealed.join.lock.forecast.root
    if tuple(v.root for v in views) != (root, root) or tuple(v.refinement for v in views) != (1, 2):
        raise ValueError("Finite response-law evaluation scalar readout requires both complete assigned views")
    # Raw endpoint arithmetic: +/− signed mates, preserving their independent
    # purpose/view axes. Missing HOLD affects use, never signed response loss.
    y = np.full((4, 8, 2, 2), np.nan)
    response_valid = np.zeros((4, 2, 2, 2), dtype=bool)
    use_valid = np.zeros((4, 2, 2), dtype=bool)
    for v, view in enumerate(views):
        words = {(w.invocation.word, w.invocation.purpose): w for w in view.words}
        for k, positive in enumerate(NATIVE_WORDS[1::2]):
            negative = NATIVE_WORDS[2 * k]
            for f, purpose in enumerate(("future-1", "future-2")):
                p, n = words[positive, purpose], words[negative, purpose]
                hold = next(
                    w
                    for w in view.words
                    if w.invocation.purpose == purpose
                    and w.invocation.word is not None
                    and w.invocation.word.sign == 0
                )
                for j in (0, 1):
                    positive_value, negative_value = p.outputs[j], n.outputs[j]
                    if positive_value is not None and negative_value is not None:
                        y[k, j, f, v] = (float(positive_value) - float(negative_value)) / 2
                        response_valid[k, j, f, v] = all(
                            w.complete and w.maximum_force_error == 0 for w in (p, n)
                        )
                for j, value in enumerate((*p.outputs[2:5], *n.outputs[2:5]), start=2):
                    if value is not None:
                        y[k, j, f, v] = float(value)
                use_valid[k, f, v] = all(
                    w.complete
                    and w.maximum_force_error == 0
                    and all(x is not None for x in w.outputs)
                    for w in (p, n, hold)
                )
    nu = np.asarray([float(d / 8) for d in SPEC.delta])
    delta = np.asarray([float(d) for d in SPEC.delta])
    tables = {t.boundary: t for t in sealed.join.lock.forecast.tables}
    events = []
    for early in sealed.join.lock.parents:
        boundary, consumer_text = early.policy_id.split(".consumer-")
        consumer = int(consumer_text)
        request = sealed.join.lock.requests.requests[consumer]
        table = tables[boundary]
        expected = -1
        for word_index, force in enumerate(NATIVE_WORDS):
            pairs = [
                (r, value)
                for r, value in zip(table.requests, table.results, strict=True)
                if r.action_word is not None and _force(r.action_word) == force
            ]
            if len(pairs) != 2:
                raise ValueError("Finite response-law evaluation independent selection loses a word/view prediction")
            admissible = sealed.join.lock.forecast.primary_interface is not None
            for _, result in pairs:
                if result.disposition is not LawEvaluationDisposition.SUPPORTED:
                    admissible = False
                    continue
                values = {v.quantity_id: float(v.value) for v in result.response_values}
                widths = {
                    v.quantity_id: float(v.value)
                    for v in result.uncertainty_values
                    if v.value_id.startswith("halfwidth.")
                }
                mean = np.asarray([values[q.quantity_id] for q in output_quantities()])
                h = np.asarray([widths[q.quantity_id] for q in output_quantities()]) + nu
                axis, sense = request.direction // 2, 1 if request.direction % 2 == 0 else -1
                admissible = admissible and bool(
                    (h <= delta).all()
                    and sense * mean[axis] - h[axis] >= float(request.lower)
                    and sense * mean[axis] + h[axis] <= float(SPEC.upper[consumer])
                    and abs(mean[1 - axis]) + h[1 - axis] <= float(SPEC.transverse[consumer])
                    and (
                        np.maximum(mean[2:] + h[2:], 0)
                        <= np.asarray([float(v) for v in SPEC.preservation] * 2)
                    ).all()
                )
            if admissible:
                expected = word_index
                break
        decision = early.decision
        selected = -1
        if decision.disposition is not CommitmentDisposition.NONATTEMPT:
            assert decision.action_binding is not None
            selected = NATIVE_WORDS.index(_force(decision.action_binding.action_word))
        available, numerical, success = False, False, False
        if selected >= 0:
            k, sign = selected // 2, -1 if selected % 2 == 0 else 1
            actual = y[k].copy()
            actual[:2] *= sign
            if sign < 0:
                actual[2:] = actual[[5, 6, 7, 2, 3, 4]]
            available = bool(use_valid[k].all() and np.isfinite(actual).all())
            numerical = bool((np.abs(actual[..., 0] - actual[..., 1]) <= nu[:, None]).all())
            axis, sense = request.direction // 2, 1 if request.direction % 2 == 0 else -1
            longitudinal = sense * actual[axis]
            success = bool(
                available
                and numerical
                and sealed.join.use_allowed
                and (longitudinal >= float(request.lower)).all()
                and (longitudinal <= float(SPEC.upper[consumer])).all()
                and (np.abs(actual[1 - axis]) <= float(SPEC.transverse[consumer])).all()
                and (
                    actual[2:]
                    <= np.asarray([float(v) for v in SPEC.preservation] * 2)[:, None, None]
                ).all()
            )
        events.append(
            FiniteResponseLawScalarConsumerEvent(
                early.policy_id,
                selected,
                expected,
                success,
                selected >= 0 and not success,
                numerical,
                available,
            )
        )
    if [(e.policy_id, e.selected_word >= 0, e.success, e.false_admission) for e in events] != [
        (u.policy_id, u.admitted, u.task_success, u.admitted_failure) for u in prospective_root_evaluation.units
    ]:
        raise ValueError("Generic controller use disagrees with independent all-assigned scalar events")
    losses = []
    for boundary in BOUNDARIES:
        loss_table = tables.get(boundary)
        loss = None
        if loss_table is not None and loss_table.point_mean is not None and response_valid.all():
            mean = np.asarray([float(v) for v in loss_table.point_mean]).reshape(4, 8)
            loss_value = float(np.mean((y[:, :2] - mean[:, :2, None, None]) ** 2) / 0.01**2)
            if isfinite(loss_value):
                loss = D(format(loss_value, ".17g"))
        losses.append((boundary, loss))
    widths_by_boundary = []
    for boundary in BOUNDARIES:
        diagnostic_table = tables.get(boundary)
        diagnostic_widths: list[D | None] = []
        for word in NATIVE_WORDS[1::2]:
            diagnostic_result = (
                None
                if diagnostic_table is None
                else next(
                    value
                    for request, value in zip(
                        diagnostic_table.requests, diagnostic_table.results, strict=True
                    )
                    if request.action_word is not None and _force(request.action_word) == word
                )
            )
            statistical = (
                {}
                if diagnostic_result is None
                else {
                    v.quantity_id: v.value
                    for v in diagnostic_result.uncertainty_values
                    if v.value_id.startswith("halfwidth.")
                }
            )
            diagnostic_widths.extend(
                None
                if q.quantity_id not in statistical
                else statistical[q.quantity_id] + SPEC.delta[j] / 8
                for j, q in enumerate(output_quantities())
            )
        widths_by_boundary.append((boundary, tuple(diagnostic_widths)))
    return FiniteResponseLawRootInferenceOperands(
        root.stage_unit,
        tuple(events),
        tuple(losses),
        bool((np.abs(y[:, :2, :, 0] - y[:, :2, :, 1]) <= nu[None, :2, None]).all()),
        tuple(widths_by_boundary),
        dict(getattr(root, "scientific_seeds", ())).get("bootstrap"),
    )


def cohort_inference(
    rows: tuple[FiniteResponseLawRootInferenceOperands, ...], *, lower_qualified: bool, cached_qualified: bool
) -> dict[str, object]:
    if len(rows) != 64:
        raise ValueError("Finite response-law evaluation inference requires every assigned root in its frozen order")
    namespace = rows[0].root_id.rsplit(".r", 1)[0]
    if tuple(r.root_id for r in rows) != tuple(
        f"{namespace}.r{i:03d}" for i in range(64)
    ):
        raise ValueError("Finite response-law evaluation inference requires every assigned root in its frozen order")
    bootstrap_seed = seed_for("bootstrap", namespace, committed_seed=rows[0].bootstrap_seed)
    if any(seed_for("bootstrap", namespace, committed_seed=r.bootstrap_seed) != bootstrap_seed for r in rows):
        raise ValueError("Finite response-law evaluation inference requires one committed cohort bootstrap allocation")
    successes, false = {}, {}
    for b in BOUNDARIES:
        local = [tuple(e for e in r.events if e.policy_id.startswith(b + ".")) for r in rows]
        successes[b] = tuple(len(e) == 2 and all(v.success for v in e) for e in local)
        false[b] = tuple(any(v.false_admission for v in e) for e in local)
    successes_n = sum(successes["composed"])
    false_n = sum(false["composed"])
    lo, hi = cp_bounds(successes_n, 64)[0], cp_bounds(false_n, 64)[1]
    use = lo >= 0.70 and hi <= 0.10
    operands = {b: tuple(dict(r.response_losses)[b] for r in rows) for b in BOUNDARIES}
    information = None
    bootstrap = None
    bootstrap_stability = None
    reduction = None
    if all(v is not None for b in ("composed", "cached") for v in operands[b]):
        composed = tuple(float(v) for v in operands["composed"] if v is not None)
        cached = tuple(float(v) for v in operands["cached"] if v is not None)
        differences = tuple(a - b for a, b in zip(composed, cached, strict=True))
        bootstrap = complete_unit_bootstrap_mean(
            differences, seed=bootstrap_seed, draws=20000, confidence=0.90
        )
        bootstrap_stability = complete_unit_bootstrap_mean(
            differences, seed=bootstrap_seed, draws=10000, confidence=0.90
        )
        cached_mean, composed_mean = sum(cached) / 64, sum(composed) / 64
        reduction = None if cached_mean == 0 else 1 - composed_mean / cached_mean
        information = (
            cached_mean > 0
            and composed_mean <= 0.90 * cached_mean
            and float(str(bootstrap["upper"])) < 0
        )
    improved = sum(
        a and not c for a, c in zip(successes["composed"], successes["cached"], strict=True)
    )
    deteriorated = sum(
        c and not a for a, c in zip(successes["composed"], successes["cached"], strict=True)
    )
    discordant = improved + deteriorated
    p = (
        1.0
        if discordant == 0
        else sum(comb(discordant, k) for k in range(improved, discordant + 1)) / 2**discordant
    )
    tested = use and information is True and cached_qualified
    return {
        "independent_roots": 64,
        "joint_successes": successes_n,
        "false_admission_episodes": false_n,
        "use_cp_lower": lo,
        "false_cp_upper": hi,
        "tier1_use_supported": use,
        "tier1_information_supported": information,
        "information_relative_reduction": reduction,
        "paired_root_bootstrap": bootstrap,
        "bootstrap_stability_first_10000_same_seed_diagnostic_only": bootstrap_stability,
        "added_use_tested": tested,
        "added_use_one_sided_exact_p": p if tested else None,
        "added_use_discordances": {"composed_only": improved, "cached_only": deteriorated},
        "added_use_paired_effect_interval": paired_difference_bounds(improved, deteriorated, 64),
        "tier1_added_use_supported": tested and improved > deteriorated and p <= 0.05,
        "preparation_policy_eligible": lower_qualified and use,
        "method_joint_success_counts": {b: sum(v) for b, v in successes.items()},
        "method_false_admission_counts": {b: sum(v) for b, v in false.items()},
        "response_losses": {
            b: [None if v is None else str(v) for v in values] for b, values in operands.items()
        },
        "response_numerical_discrepancy_roots": [
            r.root_id for r in rows if not r.response_numerical_agreement
        ],
        "individual_consumer_counts": {
            f"{b}.consumer-{c}": {
                "success": sum(
                    e.success for r in rows for e in r.events if e.policy_id == f"{b}.consumer-{c}"
                ),
                "false_admission": sum(
                    e.false_admission
                    for r in rows
                    for e in r.events
                    if e.policy_id == f"{b}.consumer-{c}"
                ),
                "nonattempt": sum(
                    e.selected_word < 0
                    for r in rows
                    for e in r.events
                    if e.policy_id == f"{b}.consumer-{c}"
                ),
                "successful_magnitude_counts": {
                    str(magnitude): sum(
                        e.success
                        and e.selected_word >= 0
                        and NATIVE_WORDS[e.selected_word].magnitude == magnitude
                        for r in rows
                        for e in r.events
                        if e.policy_id == f"{b}.consumer-{c}"
                    )
                    for magnitude in (8, 16)
                },
            }
            for b in BOUNDARIES
            for c in (0, 1)
        },
        "action_change_vs_cached_counts": {
            b: sum(
                next(e.selected_word for e in r.events if e.policy_id == f"{b}.consumer-{c}")
                != next(e.selected_word for e in r.events if e.policy_id == f"cached.consumer-{c}")
                for r in rows
                for c in (0, 1)
            )
            for b in ("composed", "direct")
        },
        "prediction_halfwidths_native": {
            b: [
                [None if v is None else str(v) for v in dict(r.prediction_halfwidths)[b]]
                for r in rows
            ]
            for b in BOUNDARIES
        },
    }
