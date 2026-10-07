"""Receipt predicates for bounded response relations, independent of substrate units."""

from decimal import Decimal as D
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import (
    PredicateDirection,
    ReceiverInterval,
    TemporalPredicateSemantics,
)
from empirical_lawhood.kernel.references import NamedDecimal, ExecutableReference
from empirical_lawhood.planning.evidence_geometry import GatePredicateSpec, GatePredicateKind


def bounded_response_gate(
    *,
    stem: str,
    kind: AdmissionGateKind,
    response_receiver: str,
    response_minimum: NamedDecimal,
    safety_receiver: str,
    safety_maximum: NamedDecimal,
    protected_interval: ReceiverInterval,
    evaluator: ExecutableReference,
) -> GatePredicateSpec:
    """A lower response target with an upper safety guard and boolean evidence gates.

    TARGET measures containment margin against zero; REACHABILITY uses the
    requested response. Native units, receivers and clocks are supplied intact.
    Other receiver geometries should construct GatePredicateSpec directly.
    """
    target = kind in (AdmissionGateKind.TARGET, AdmissionGateKind.REACHABILITY)
    upper_guard = kind in (AdmissionGateKind.PHYSICAL_SINK, AdmissionGateKind.BASELINE_PRESERVATION)
    receiver = response_receiver if target else safety_receiver
    if target:
        predicate = GatePredicateKind.SCALAR_AT_LEAST
        direction = PredicateDirection.AT_LEAST
        # The finite-set owner derives TARGET as a containment *margin*.
        # The request threshold belongs in its frozen target box; the raw
        # TARGET observation must therefore be tested against zero.
        threshold = D(0) if kind is AdmissionGateKind.TARGET else response_minimum.value
        lower, upper, expected = (
            NamedDecimal(f"{stem}.{kind.value.lower()}.lower", threshold, response_minimum.unit),
            None,
            None,
        )
    elif upper_guard:
        predicate = GatePredicateKind.SCALAR_AT_MOST
        direction = PredicateDirection.AT_MOST
        lower, upper, expected = (
            None,
            NamedDecimal(
                f"{stem}.{kind.value.lower()}.upper", safety_maximum.value, safety_maximum.unit
            ),
            None,
        )
    else:
        predicate = GatePredicateKind.BOOLEAN_EQUALS
        direction = PredicateDirection.AT_LEAST
        lower, upper, expected = None, None, True
    return GatePredicateSpec(
        f"{stem}.gate.{kind.value.lower()}",
        kind,
        receiver,
        direction,
        f"{stem}.operand.{kind.value.lower()}",
        protected_interval,
        predicate,
        lower,
        upper,
        expected,
        None,
        TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH
        if kind is AdmissionGateKind.BASELINE_PRESERVATION
        else None,
        (f"{stem}.constraint.{kind.value.lower()}",),
        evaluator,
    )
