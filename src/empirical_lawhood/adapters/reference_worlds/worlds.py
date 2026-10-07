"""Definitions of the sixteen deterministic R3 truth-known worlds."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from functools import lru_cache

from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.kernel.references import ControllerDecisionKind, NamedDecimal
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus

from .builders import build_reference_system
from .contracts import (
    REQUIRED_CONTROL_IDS,
    ReferenceCase,
    ReferenceControlSuite,
    ReferenceOracle,
    ReferenceWorldKind,
    ReferenceWorldSpec,
)


def _named(values: tuple[tuple[str, str, str], ...]) -> tuple[NamedDecimal, ...]:
    return tuple(
        sorted(
            (
                NamedDecimal(value_id=value_id, value=Decimal(value), unit=unit)
                for value_id, value, unit in values
            ),
            key=lambda item: item.value_id,
        )
    )


def _case(
    kind: ReferenceWorldKind,
    index: int,
    values: tuple[tuple[str, str, str], ...],
    *,
    denominator: str = "reference-cell",
    model: str = "reference-model",
    view: str = "fine-view",
    tags: tuple[str, ...] = (),
) -> ReferenceCase:
    return ReferenceCase(
        case_id=f"{kind.value}-case-{index:03d}",
        independent_unit_id="reference-preparation",
        denominator_cell_id=denominator,
        action_id="action",
        gauge_id="receiver",
        horizon_id="reference-horizon",
        model_id=model,
        numerical_view_id=view,
        values=_named(values),
        tags=tuple(sorted(tags)),
    )


def _controls(kind: ReferenceWorldKind) -> ReferenceControlSuite:
    registered = tuple(f"{kind.value}-analysis-{index}" for index in range(1, 4))
    return ReferenceControlSuite(
        control_ids=REQUIRED_CONTROL_IDS,
        supported_action_lower=Decimal("-2"),
        supported_action_upper=Decimal("2"),
        wrong_action_value=Decimal("1"),
        wrong_action_effect=Decimal("0"),
        minimum_response_effect=Decimal("0.5"),
        unsupported_action_value=Decimal("3"),
        registered_analysis_ids=registered,
        executed_analysis_ids=registered,
        raw_signal_p=Decimal("0.04"),
        multiplicity_cutoff=Decimal("0.01"),
        nominal_model_safe=True,
        discrepant_model_safe=False,
        outside_admission_gate_passes=(True, False, True),
        parent_visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        fresh_visibility=VisibilityCeiling.PROSPECTIVE,
    )


def _oracle(
    specific_check_ids: tuple[str, ...],
    metrics: tuple[tuple[str, str, str], ...],
    recovered: tuple[str, ...],
    rejected: tuple[str, ...],
    *,
    admission: AdmissionStatus = AdmissionStatus.ADMITTED,
    controller: ControllerDecisionKind = ControllerDecisionKind.ACTION,
    nominated: tuple[str, ...] = (),
    supported: tuple[str, ...] = (),
    status: ScientificStatus = ScientificStatus.SUPPORTED,
) -> ReferenceOracle:
    return ReferenceOracle(
        expected_check_ids=tuple(
            sorted((*specific_check_ids, *(f"control-{item}" for item in REQUIRED_CONTROL_IDS)))
        ),
        expected_metrics=_named(metrics),
        recovered_structure_ids=tuple(sorted(recovered)),
        rejected_law_ids=tuple(sorted(rejected)),
        admission_status=admission,
        controller_decision=controller,
        nominated_hypothesis_ids=tuple(sorted(nominated)),
        supported_hypothesis_ids=tuple(sorted(supported)),
        scientific_status=status,
        parent_claim_preserved=True,
    )


def _world(
    kind: ReferenceWorldKind,
    description: str,
    cases: tuple[ReferenceCase, ...],
    oracle: ReferenceOracle,
    **system_options: object,
) -> ReferenceWorldSpec:
    return ReferenceWorldSpec(
        reference_id=kind.value,
        kind=kind,
        description=description,
        system=build_reference_system(kind.value, **system_options),  # type: ignore[arg-type]
        cases=tuple(sorted(cases, key=lambda item: item.case_id)),
        controls=_controls(kind),
        oracle=oracle,
    )


def _stable_linear() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.STABLE_LINEAR
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", str(action), "1"),
                ("receiver", str(Decimal(2) * action + Decimal(1)), "1"),
                ("sink", str(abs(action) / Decimal(10)), "1"),
            ),
        )
        for index, action in enumerate(
            (Decimal(-2), Decimal(-1), Decimal(0), Decimal(1), Decimal(2)),
            start=1,
        )
    )
    return _world(
        kind,
        "Stable affine response with an exact ten-second finite-action slope.",
        cases,
        _oracle(
            ("linear-intercept", "linear-residual", "linear-slope"),
            (
                ("intercept", "1", "1"),
                ("max-residual", "0", "1"),
                ("slope", "2", "1"),
            ),
            ("finite-horizon-response", "rank-one-response"),
            ("zero-response-law",),
        ),
    )


def _nonlinear_charted() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.NONLINEAR_CHARTED
    actions = (
        (Decimal("-1.5"), "left-chart"),
        (Decimal("-1"), "left-chart"),
        (Decimal("-0.5"), "left-chart"),
        (Decimal("0.5"), "right-chart"),
        (Decimal("1"), "right-chart"),
        (Decimal("1.5"), "right-chart"),
    )
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", str(action), "1"),
                ("receiver", str(action + action * action), "1"),
                ("sink", "0", "1"),
            ),
            denominator=chart,
            tags=(chart,),
        )
        for index, (action, chart) in enumerate(actions, start=1)
    )
    return _world(
        kind,
        "Quadratic response on two supported charts separated by a real atlas gap.",
        cases,
        _oracle(
            ("atlas-gap", "chart-count", "local-curvature"),
            (
                ("chart-count", "2", "1"),
                ("curvature", "2", "1"),
                ("gap-width", "1", "1"),
            ),
            ("atlas-boundary", "charted-curvature"),
            ("global-smooth-law",),
            admission=AdmissionStatus.PARTIAL,
        ),
    )


def _hysteretic_memory() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.HYSTERETIC_MEMORY
    histories = (Decimal(-1), Decimal(0), Decimal(1))
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("history", str(history), "1"),
                ("receiver", str(Decimal(2) + Decimal("1.5") * history), "1"),
                ("sink", "0", "1"),
            ),
        )
        for index, history in enumerate(histories, start=1)
    )
    return _world(
        kind,
        "Equal current actions diverge under retained history, defeating memoryless closure.",
        cases,
        _oracle(
            ("history-coefficient", "memoryless-recurrence"),
            (
                ("history-coefficient", "1.5", "1"),
                ("memoryless-range", "3", "1"),
            ),
            ("history-dependent-response",),
            ("memoryless-law",),
        ),
        history_dependent=True,
    )


def _multirate_delayed() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.MULTIRATE_DELAYED
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("applied-time", str(start + Decimal(2)), "s"),
                ("receiver", "2", "1"),
                ("receiver-time", str(start + Decimal(5)), "s"),
                ("requested-time", str(start), "s"),
                ("sink", "0", "1"),
            ),
        )
        for index, start in enumerate((Decimal(0), Decimal(10)), start=1)
    )
    return _world(
        kind,
        "Requested, applied and receiver events live on distinct delayed clocks.",
        cases,
        _oracle(
            ("applied-delay", "receiver-delay", "same-label-rejection"),
            (
                ("applied-delay", "2", "s"),
                ("receiver-delay", "3", "s"),
            ),
            ("applied-clock-delay", "receiver-clock-delay"),
            ("nominal-label-clock-law",),
        ),
        multirate=True,
    )


def _hybrid_partial() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.HYBRID_PARTIAL
    rows = (
        ("0.2", "0", "0.2"),
        ("0.8", "0", "0.8"),
        ("1.2", "1", "0.2"),
        ("1.8", "1", "0.8"),
    )
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("mode", mode, "1"),
                ("receiver", output, "1"),
                ("sink", "0", "1"),
                ("state", state, "1"),
            ),
            tags=(f"mode-{mode}",),
        )
        for index, (state, mode, output) in enumerate(rows, start=1)
    )
    return _world(
        kind,
        "A switching reset map produces an event and an ambiguous partial observation.",
        cases,
        _oracle(
            ("event-threshold", "partial-observation"),
            (
                ("ambiguous-output-count", "2", "1"),
                ("event-threshold", "1", "1"),
            ),
            ("hybrid-event", "partial-observation"),
            ("continuous-fully-observed-law",),
            admission=AdmissionStatus.PARTIAL,
            controller=ControllerDecisionKind.HOLD,
        ),
    )


def _strong_empty_admission() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.STRONG_EMPTY_ADMISSION
    cases = (
        _case(
            kind,
            1,
            (
                ("action", "0", "1"),
                ("receiver", "0", "1"),
                ("sink", "0", "1"),
                ("sink-limit", "5", "1"),
            ),
        ),
        _case(
            kind,
            2,
            (
                ("action", "1", "1"),
                ("receiver", "10", "1"),
                ("sink", "20", "1"),
                ("sink-limit", "5", "1"),
            ),
        ),
    )
    return _world(
        kind,
        "A large target response is unusable because the decisive sink always fails.",
        cases,
        _oracle(
            ("admission-intersection", "strong-response"),
            (("response-gain", "10", "1"), ("worst-sink", "20", "1")),
            ("strong-response",),
            ("scalar-reward-admission",),
            admission=AdmissionStatus.EMPTY,
            controller=ControllerDecisionKind.HOLD,
        ),
    )


def _coupled_interfaces() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.COUPLED_INTERFACES
    cases = (
        _case(
            kind,
            1,
            (
                ("action", "1", "1"),
                ("coupling-input", "1", "1"),
                ("coupling-output", "1", "1"),
                ("receiver", "1", "1"),
                ("sink", "0", "1"),
            ),
        ),
    )
    return _world(
        kind,
        "Two components expose one unit/clock-valid and one deliberately invalid coupling.",
        cases,
        _oracle(
            ("invalid-interface", "valid-interface"),
            (
                ("invalid-interface-errors", "1", "1"),
                ("valid-interface-count", "1", "1"),
            ),
            ("composable-interface",),
            ("unit-mismatched-interface",),
        ),
        coupled=True,
    )


def _drifting_degrading() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.DRIFTING_DEGRADING
    rows = (
        ("0", "0", "0"),
        ("0", "1", "2"),
        ("1", "0", "0"),
        ("1", "1", "0.5"),
    )
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", action, "1"),
                ("epoch", epoch, "1"),
                ("receiver", receiver, "1"),
                ("sink", "0", "1"),
            ),
            tags=(f"epoch-{epoch}",),
        )
        for index, (epoch, action, receiver) in enumerate(rows, start=1)
    )
    return _world(
        kind,
        "The local response slope degrades and must invalidate the active chart.",
        cases,
        _oracle(
            ("chart-invalidation", "drift-detection"),
            (
                ("post-drift-slope", "0.5", "1"),
                ("pre-drift-slope", "2", "1"),
                ("slope-change", "1.5", "1"),
            ),
            ("chart-invalidation", "response-drift"),
            ("stationary-law",),
            admission=AdmissionStatus.EMPTY,
            controller=ControllerDecisionKind.HOLD,
            status=ScientificStatus.MIXED,
        ),
    )


def _observational_equivalence() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.OBSERVATIONAL_EQUIVALENCE
    rows = (
        ("model-a", "0", "0", "observational"),
        ("model-b", "0", "0", "observational"),
        ("model-a", "1", "1", "intervention"),
        ("model-b", "1", "2", "intervention"),
    )
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", action, "1"),
                ("receiver", receiver, "1"),
                ("sink", "0", "1"),
            ),
            model=model,
            tags=(phase,),
        )
        for index, (model, action, receiver, phase) in enumerate(rows, start=1)
    )
    return _world(
        kind,
        "Two models are observationally identical but separate under supported intervention.",
        cases,
        _oracle(
            ("intervention-divergence", "observational-equivalence"),
            (
                ("intervention-difference", "1", "1"),
                ("observational-difference", "0", "1"),
            ),
            ("intervention-discrimination",),
            ("observational-identification",),
            admission=AdmissionStatus.NOT_EVALUATED,
            controller=ControllerDecisionKind.HOLD,
        ),
    )


def _numerical_false_structure() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.NUMERICAL_FALSE_STRUCTURE
    matrices = {
        "coarse-view": (("1", "0"), ("0", "1")),
        "fine-view": (("1", "1"), ("2", "2")),
    }
    cases: list[ReferenceCase] = []
    index = 1
    for view, matrix in matrices.items():
        for row_index, row in enumerate(matrix):
            for column_index, response in enumerate(row):
                cases.append(
                    _case(
                        kind,
                        index,
                        (
                            ("action", str(column_index), "1"),
                            ("column", str(column_index), "1"),
                            ("receiver", response, "1"),
                            ("row", str(row_index), "1"),
                            ("sink", "0", "1"),
                        ),
                        view=view,
                    )
                )
                index += 1
    return _world(
        kind,
        "A coarse discretization invents rank and an admission basin absent when qualified.",
        tuple(cases),
        _oracle(
            ("rank-instability", "refined-basin-rejection"),
            (
                ("coarse-rank", "2", "1"),
                ("fine-rank", "1", "1"),
                ("rank-change", "1", "1"),
            ),
            ("numerical-structural-instability",),
            ("coarse-admission-basin", "coarse-rank-law"),
            admission=AdmissionStatus.EMPTY,
            controller=ControllerDecisionKind.HOLD,
            status=ScientificStatus.NOT_SUPPORTED,
        ),
        numerical=True,
    )


def _rare_decisive_sink() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.RARE_DECISIVE_SINK
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("receiver", "1", "1"),
                ("sink", "100" if index == 100 else "0", "1"),
                ("sink-limit", "2", "1"),
            ),
            tags=("rare-sink",) if index == 100 else (),
        )
        for index in range(1, 101)
    )
    return _world(
        kind,
        "A one-percent catastrophic sink is erased by favourable average performance.",
        cases,
        _oracle(
            ("average-performance-decoy", "rare-sink-falsifier"),
            (
                ("mean-receiver", "1", "1"),
                ("mean-sink", "1", "1"),
                ("worst-sink", "100", "1"),
            ),
            ("rare-decisive-sink",),
            ("mean-score-admission",),
            admission=AdmissionStatus.EMPTY,
            controller=ControllerDecisionKind.HOLD,
        ),
    )


def _discrepancy_exploitation() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.DISCREPANCY_EXPLOITATION
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("receiver", "2", "1"),
                ("sink", sink, "1"),
                ("sink-limit", "5", "1"),
            ),
            model=model,
        )
        for index, (model, sink) in enumerate(
            (("nominal-model", "0"), ("discrepant-model", "10")), start=1
        )
    )
    return _world(
        kind,
        "A nominal controller exploits simulator error and fails the frozen model set.",
        cases,
        _oracle(
            ("nominal-exploitation", "robust-model-set"),
            (("model-set-worst-sink", "10", "1"), ("nominal-sink", "0", "1")),
            ("discrepancy-exploitation",),
            ("nominal-controller",),
            admission=AdmissionStatus.EMPTY,
            controller=ControllerDecisionKind.HOLD,
        ),
        numerical=True,
    )


def _latency_boundary() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.LATENCY_BOUNDARY
    cases = (
        _case(
            kind,
            1,
            (
                ("action", "1", "1"),
                ("cadence", "0.1", "s"),
                ("latency", "0.2", "s"),
                ("prediction-error", "0", "1"),
                ("receiver", "1", "1"),
                ("sink", "0", "1"),
            ),
        ),
    )
    return _world(
        kind,
        "An exact offline model misses the declared observation-to-action deadline.",
        cases,
        _oracle(
            ("accuracy-is-insufficient", "latency-deadline"),
            (
                ("latency", "0.2", "s"),
                ("latency-overrun", "0.1", "s"),
                ("prediction-error", "0", "1"),
            ),
            ("compute-envelope-boundary",),
            ("offline-accuracy-controller",),
            admission=AdmissionStatus.EMPTY,
            controller=ControllerDecisionKind.HOLD,
        ),
        numerical=True,
        latency_seconds=Decimal("0.2"),
        deadline_seconds=Decimal("0.1"),
    )


def _planted_relational_anomaly() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY
    rows = (
        ("planted-relation", "0.8", "1", "1", "1", "1"),
        ("attractive-decoy", "1.2", "0", "0", "1", "1"),
        ("null-candidate", "0.1", "1", "1", "1", "1"),
    )
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("denominator-match", denominator, "1"),
                ("effect", effect, "1"),
                ("gauge-match", gauge, "1"),
                ("horizon-match", horizon, "1"),
                ("receiver", effect, "1"),
                ("recurrence", recurrence, "1"),
                ("sink", "0", "1"),
            ),
            model=candidate,
            tags=(candidate,),
        )
        for index, (
            candidate,
            effect,
            recurrence,
            denominator,
            gauge,
            horizon,
        ) in enumerate(rows, start=1)
    )
    return _world(
        kind,
        "The correct denominator/gauge/horizon anomaly beats a larger but invalid decoy.",
        cases,
        _oracle(
            ("decoy-rejection", "planted-anomaly-recovery"),
            (("largest-raw-effect", "1.2", "1"), ("selected-effect", "0.8", "1")),
            ("planted-relational-anomaly",),
            ("attractive-decoy",),
            admission=AdmissionStatus.NOT_EVALUATED,
            controller=ControllerDecisionKind.HOLD,
            nominated=("planted-relation",),
            status=ScientificStatus.PARTIAL,
        ),
    )


def _null_search_family() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.NULL_SEARCH_FAMILY
    probabilities = (
        "0.02",
        "0.06",
        "0.08",
        "0.11",
        "0.14",
        "0.18",
        "0.22",
        "0.31",
        "0.44",
        "0.57",
        "0.71",
        "0.89",
    )
    cases = tuple(
        _case(
            kind,
            index,
            (
                ("action", "1", "1"),
                ("heterogeneity", "1" if index % 2 else "-1", "1"),
                ("p-value", probability, "1"),
                ("receiver", "0", "1"),
                ("sink", "0", "1"),
            ),
            model=f"candidate-{index:02d}",
        )
        for index, probability in enumerate(probabilities, start=1)
    )
    return _world(
        kind,
        "A broad null family contains a chance raw signal but no corrected preference.",
        cases,
        _oracle(
            ("complete-null-family", "multiplicity-null"),
            (("corrected-selection-count", "0", "1"), ("minimum-p", "0.02", "1")),
            ("honest-null-search",),
            ("preferred-hypothesis",),
            admission=AdmissionStatus.NOT_EVALUATED,
            controller=ControllerDecisionKind.HOLD,
            status=ScientificStatus.NOT_SUPPORTED,
        ),
    )


def _retrospective_defeated() -> ReferenceWorldSpec:
    kind = ReferenceWorldKind.RETROSPECTIVE_DEFEATED
    cases = (
        _case(
            kind,
            1,
            (
                ("action", "1", "1"),
                ("effect", "1", "1"),
                ("p-value", "0.01", "1"),
                ("receiver", "1", "1"),
                ("sink", "0", "1"),
            ),
            model="retrospective-mechanism",
            tags=("outcome-visible",),
        ),
        _case(
            kind,
            2,
            (
                ("action", "1", "1"),
                ("effect", "0", "1"),
                ("p-value", "0.8", "1"),
                ("receiver", "0", "1"),
                ("sink", "0", "1"),
            ),
            model="fresh-prospective-test",
            tags=("fresh-evidence",),
        ),
    )
    return _world(
        kind,
        "A plausible retrospective mechanism is defeated by separately generated evidence.",
        cases,
        _oracle(
            ("fresh-defeat", "parent-claim-immutability"),
            (("fresh-effect", "0", "1"), ("retrospective-effect", "1", "1")),
            ("prospective-falsification",),
            ("retrospective-mechanism",),
            admission=AdmissionStatus.NOT_EVALUATED,
            controller=ControllerDecisionKind.HOLD,
            nominated=("retrospective-mechanism",),
            status=ScientificStatus.NOT_SUPPORTED,
        ),
    )


_BUILDERS: tuple[Callable[[], ReferenceWorldSpec], ...] = (
    _stable_linear,
    _nonlinear_charted,
    _hysteretic_memory,
    _multirate_delayed,
    _hybrid_partial,
    _strong_empty_admission,
    _coupled_interfaces,
    _drifting_degrading,
    _observational_equivalence,
    _numerical_false_structure,
    _rare_decisive_sink,
    _discrepancy_exploitation,
    _latency_boundary,
    _planted_relational_anomaly,
    _null_search_family,
    _retrospective_defeated,
)


@lru_cache(maxsize=1)
def reference_worlds() -> tuple[ReferenceWorldSpec, ...]:
    worlds = tuple(builder() for builder in _BUILDERS)
    return tuple(sorted(worlds, key=lambda world: world.reference_id))


def get_reference_world(kind: ReferenceWorldKind | str) -> ReferenceWorldSpec:
    requested = ReferenceWorldKind(kind)
    for world in reference_worlds():
        if world.kind is requested:
            return world
    raise KeyError(requested.value)
