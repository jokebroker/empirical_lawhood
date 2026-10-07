"""Truth-known receiver-geometry method calibration for RICQ.

The method is deliberately finite-domain and disposition-oriented.  It tests
local column rank, exact or numerical fiber enumeration, and path continuation
without promoting finite sheets to an empirical ontology.  Static fixture keys
select only implementations registered in this module; no config contains a
callable or expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import permutations
from math import cos, pi, sin
from typing import Any, ClassVar, Final

import numpy as np

from empirical_lawhood.adapters.reference_worlds.receiver_geometry import (
    KELLER_WITNESS_PREIMAGES,
    KELLER_WITNESS_TARGET,
    verify_keller_exact_fixture,
)
from empirical_lawhood.kernel.receiver_geometry_control import (
    BranchTopologyDisposition,
    LocalRegularityDisposition,
    ReceiverFiberDisposition,
)
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


RECEIVER_GEOMETRY_METHOD_VERSION: Final = "1.0.0"
RECEIVER_GEOMETRY_FIXTURE_IDS: Final = (
    "fixture.continuous-projection",
    "fixture.history-resolved-alias",
    "fixture.injective-linear",
    "fixture.keller-nonproper",
    "fixture.missing-branch-counterfeit",
    "fixture.ordinary-fold",
    "fixture.stochastic-mixture",
    "fixture.threefold-cover",
)


class ReceiverGeometryMethodStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ReceiverGeometryMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-geometry-method-config'

    config_id: str
    fixture_ids: tuple[str, ...]
    singular_value_floor: Decimal
    root_match_tolerance: Decimal
    observation_tolerance: Decimal
    continuation_steps: int
    refinement_steps: int
    method_version: str
    prohibited_adaptations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_strings(
            self.fixture_ids,
            field_name="fixture_ids",
            allow_empty=False,
        )
        if self.fixture_ids != RECEIVER_GEOMETRY_FIXTURE_IDS:
            raise ValueError("receiver-geometry config changes the frozen fixture roster")
        for name, value in (
            ("singular_value_floor", self.singular_value_floor),
            ("root_match_tolerance", self.root_match_tolerance),
            ("observation_tolerance", self.observation_tolerance),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.continuation_steps < 24 or self.refinement_steps <= self.continuation_steps:
            raise ValueError("receiver continuation grid is below its frozen resolution")
        if self.method_version != RECEIVER_GEOMETRY_METHOD_VERSION:
            raise ValueError("receiver-geometry method version is unsupported")
        require_sorted_unique_strings(
            self.prohibited_adaptations,
            field_name="prohibited_adaptations",
            allow_empty=False,
        )


def default_receiver_geometry_method_config() -> ReceiverGeometryMethodConfig:
    return ReceiverGeometryMethodConfig(
        config_id="config.receiver-conditioned-io.receiver-geometry-method",
        fixture_ids=RECEIVER_GEOMETRY_FIXTURE_IDS,
        singular_value_floor=Decimal("0.0000000001"),
        root_match_tolerance=Decimal("0.000001"),
        observation_tolerance=Decimal("0.000000001"),
        continuation_steps=72,
        refinement_steps=144,
        method_version=RECEIVER_GEOMETRY_METHOD_VERSION,
        prohibited_adaptations=(
            "no-fixture-label-as-prediction",
            "no-outcome-adaptive-floor",
            "no-preparation-as-sheet",
            "no-rank-only-finite-fiber-promotion",
        ),
    )


@dataclass(frozen=True, slots=True)
class ReceiverGeometryFixtureResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-geometry-fixture-result'

    result_id: str
    fixture_id: str
    state_dimension: int
    observation_dimension: int
    local_rank: int | None
    local_regularity: LocalRegularityDisposition
    fiber_disposition: ReceiverFiberDisposition
    branch_disposition: BranchTopologyDisposition | None
    effective_count_lower: int | None
    effective_count_upper: int | None
    forward_permutation: tuple[int, ...] | None
    reverse_permutation: tuple[int, ...] | None
    metrics: tuple[NamedDecimal, ...]
    status: ReceiverGeometryMethodStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        if self.fixture_id not in RECEIVER_GEOMETRY_FIXTURE_IDS:
            raise ValueError("receiver-geometry result names an unregistered fixture")
        if self.state_dimension < 1 or self.observation_dimension < 1:
            raise ValueError("receiver-geometry dimensions must be positive")
        if self.local_rank is not None and not (
            0 <= self.local_rank <= min(self.state_dimension, self.observation_dimension)
        ):
            raise ValueError("receiver-geometry local rank is outside matrix dimensions")
        if self.local_regularity is LocalRegularityDisposition.UNEVALUABLE:
            if self.local_rank is not None:
                raise ValueError("unevaluable local rank cannot impute a rank")
        elif self.local_rank is None:
            raise ValueError("evaluated local regularity requires a rank")
        require_sorted_unique_ids(
            self.metrics,
            attribute="value_id",
            field_name="metrics",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ReceiverGeometryMethodStatus.SUPPORTED and self.reason_codes:
            raise ValueError("supported geometry fixture cannot carry failure reasons")
        if self.status is not ReceiverGeometryMethodStatus.SUPPORTED and not self.reason_codes:
            raise ValueError("non-supported geometry fixture requires reasons")
        if (self.forward_permutation is None) != (self.reverse_permutation is None):
            raise ValueError("geometry fixture requires paired path permutations")
        if self.forward_permutation is not None:
            expected = tuple(range(len(self.forward_permutation)))
            if (
                tuple(sorted(self.forward_permutation)) != expected
                or tuple(sorted(self.reverse_permutation or ())) != expected
            ):
                raise ValueError("geometry fixture path result is not a permutation")


@dataclass(frozen=True, slots=True)
class ReceiverGeometryMethodSuiteResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-geometry-method-suite-result'

    suite_id: str
    config_sha256: str
    fixture_results: tuple[ReceiverGeometryFixtureResult, ...]
    false_finite_promotion_count: int
    missed_primary_count: int
    status: ReceiverGeometryMethodStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        if len(self.config_sha256) != 64:
            raise ValueError("receiver-geometry suite lacks exact config identity")
        require_sorted_unique_ids(
            self.fixture_results,
            attribute="fixture_id",
            field_name="fixture_results",
        )
        if tuple(value.fixture_id for value in self.fixture_results) != (
            RECEIVER_GEOMETRY_FIXTURE_IDS
        ):
            raise ValueError("receiver-geometry suite result is incomplete")
        observed_false = sum(
            value.fixture_id
            in {
                "fixture.continuous-projection",
                "fixture.missing-branch-counterfeit",
                "fixture.stochastic-mixture",
            }
            and value.fiber_disposition is ReceiverFiberDisposition.FINITE_MULTIPLE
            for value in self.fixture_results
        )
        observed_missed = sum(
            value.fixture_id
            in {
                "fixture.injective-linear",
                "fixture.keller-nonproper",
                "fixture.ordinary-fold",
                "fixture.threefold-cover",
            }
            and value.status is not ReceiverGeometryMethodStatus.SUPPORTED
            for value in self.fixture_results
        )
        if (
            self.false_finite_promotion_count != observed_false
            or self.missed_primary_count != observed_missed
        ):
            raise ValueError("receiver-geometry suite error counts are not derived")
        expected_status = (
            ReceiverGeometryMethodStatus.SUPPORTED
            if observed_false == 0 and observed_missed == 0
            else ReceiverGeometryMethodStatus.OPPOSED
        )
        if self.status is not expected_status:
            raise ValueError("receiver-geometry suite status differs from decisive controls")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_reasons = (
            ()
            if expected_status is ReceiverGeometryMethodStatus.SUPPORTED
            else ("RECEIVER_GEOMETRY_DECISIVE_CONTROL_FAILED",)
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("receiver-geometry suite reasons differ from its status")


def numerical_rank(matrix: np.ndarray[Any, np.dtype[np.float64]], floor: float) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular_values > floor))


def _match_roots(
    previous: tuple[complex, ...],
    current: tuple[complex, ...],
) -> tuple[complex, ...]:
    if len(previous) != len(current):
        raise ValueError("continuation lost or created a root")
    order = min(
        permutations(range(len(current))),
        key=lambda permutation: sum(
            abs(previous[index] - current[permutation[index]]) for index in range(len(previous))
        ),
    )
    return tuple(current[index] for index in order)


def _cube_roots(value: complex) -> tuple[complex, ...]:
    roots = np.roots(np.asarray((1, 0, 0, -value), dtype=np.complex128))
    return tuple(complex(value) for value in roots)


def _permutation_against_initial(
    initial: tuple[complex, ...],
    final: tuple[complex, ...],
) -> tuple[int, ...]:
    return min(
        permutations(range(len(initial))),
        key=lambda permutation: sum(
            abs(final[index] - initial[permutation[index]]) for index in range(len(final))
        ),
    )


def track_cube_cover(
    *,
    winding: int,
    steps: int,
    radius: float = 1.0,
    center: complex = 0j,
    inject_missing_branch: bool = False,
) -> tuple[int, ...] | None:
    """Continue all roots of ``z**3 = target`` along one frozen loop."""

    if steps < 24 or radius <= 0:
        raise ValueError("cube-cover continuation path is below its method bounds")
    targets = tuple(
        center
        + radius
        * complex(
            cos(2 * pi * winding * index / steps),
            sin(2 * pi * winding * index / steps),
        )
        for index in range(steps + 1)
    )
    tracked = _cube_roots(targets[0])
    initial = tracked
    for index, target in enumerate(targets[1:], start=1):
        roots = _cube_roots(target)
        if inject_missing_branch and index == steps // 2:
            roots = roots[:-1]
        try:
            tracked = _match_roots(tracked, roots)
        except ValueError:
            return None
    return _permutation_against_initial(initial, tracked)


def _metric(value_id: str, value: int | float) -> NamedDecimal:
    return NamedDecimal(
        value_id=value_id,
        value=Decimal(str(value)),
        unit="1",
    )


def _rank_disposition(rank: int | None, state_dimension: int) -> LocalRegularityDisposition:
    if rank is None:
        return LocalRegularityDisposition.UNEVALUABLE
    return (
        LocalRegularityDisposition.FULL_RANK
        if rank == state_dimension
        else LocalRegularityDisposition.RANK_DEFICIENT
    )


def _result(
    fixture_id: str,
    *,
    state_dimension: int,
    observation_dimension: int,
    rank: int | None,
    fiber: ReceiverFiberDisposition,
    branch: BranchTopologyDisposition | None,
    count: tuple[int, int] | None,
    forward: tuple[int, ...] | None = None,
    reverse: tuple[int, ...] | None = None,
    metrics: tuple[NamedDecimal, ...] = (),
    status: ReceiverGeometryMethodStatus = ReceiverGeometryMethodStatus.SUPPORTED,
    reasons: tuple[str, ...] = (),
) -> ReceiverGeometryFixtureResult:
    return ReceiverGeometryFixtureResult(
        result_id=f"result.{fixture_id}",
        fixture_id=fixture_id,
        state_dimension=state_dimension,
        observation_dimension=observation_dimension,
        local_rank=rank,
        local_regularity=_rank_disposition(rank, state_dimension),
        fiber_disposition=fiber,
        branch_disposition=branch,
        effective_count_lower=count[0] if count is not None else None,
        effective_count_upper=count[1] if count is not None else None,
        forward_permutation=forward,
        reverse_permutation=reverse,
        metrics=metrics,
        status=status,
        reason_codes=reasons,
    )


def run_receiver_geometry_truth_known_suite(
    config: ReceiverGeometryMethodConfig,
) -> ReceiverGeometryMethodSuiteResult:
    """Run the frozen exact/numerical fixture roster without label-derived verdicts."""

    floor = float(config.singular_value_floor)
    linear_rank = numerical_rank(np.eye(2, dtype=np.float64), floor)
    projection_rank = numerical_rank(
        np.asarray(((1.0, 0.0),), dtype=np.float64),
        floor,
    )
    fold_rank = numerical_rank(np.asarray(((2.0,),), dtype=np.float64), floor)
    cover_rank = numerical_rank(np.asarray(((3.0, 0.0), (0.0, 3.0))), floor)

    forward = track_cube_cover(winding=1, steps=config.continuation_steps)
    reverse = track_cube_cover(winding=-1, steps=config.continuation_steps)
    refined = track_cube_cover(winding=1, steps=config.refinement_steps)
    contractible = track_cube_cover(
        winding=1,
        steps=config.continuation_steps,
        radius=0.2,
        center=1 + 0j,
    )
    missing = track_cube_cover(
        winding=1,
        steps=config.continuation_steps,
        inject_missing_branch=True,
    )
    cover_supported = (
        forward is not None
        and reverse is not None
        and refined == forward
        and contractible == (0, 1, 2)
        and missing is None
    )

    exact_facts = verify_keller_exact_fixture()
    results = (
        _result(
            "fixture.continuous-projection",
            state_dimension=2,
            observation_dimension=1,
            rank=projection_rank,
            fiber=ReceiverFiberDisposition.NONFINITE_OR_CONTINUOUS,
            branch=None,
            count=None,
            metrics=(_metric("projection-column-rank", projection_rank),),
        ),
        _result(
            "fixture.history-resolved-alias",
            state_dimension=2,
            observation_dimension=2,
            rank=2,
            fiber=ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN,
            branch=None,
            count=(1, 1),
            metrics=(_metric("history-augmented-candidate-count", 1),),
        ),
        _result(
            "fixture.injective-linear",
            state_dimension=2,
            observation_dimension=2,
            rank=linear_rank,
            fiber=ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN,
            branch=BranchTopologyDisposition.STABLE_FINITE_SHEETS,
            count=(1, 1),
            metrics=(_metric("linear-rank", linear_rank),),
        ),
        _result(
            "fixture.keller-nonproper",
            state_dimension=3,
            observation_dimension=3,
            rank=3,
            fiber=ReceiverFiberDisposition.FINITE_MULTIPLE,
            branch=BranchTopologyDisposition.NONCRITICAL_SHEET_LOSS,
            count=(3, 3),
            metrics=(
                _metric("keller-exact-fact-count", len(exact_facts)),
                _metric("keller-witness-preimage-count", len(KELLER_WITNESS_PREIMAGES)),
                _metric("keller-witness-target-dimension", len(KELLER_WITNESS_TARGET)),
            ),
        ),
        _result(
            "fixture.missing-branch-counterfeit",
            state_dimension=2,
            observation_dimension=2,
            rank=2,
            fiber=ReceiverFiberDisposition.UNEVALUABLE,
            branch=BranchTopologyDisposition.UNEVALUABLE,
            count=None,
            status=ReceiverGeometryMethodStatus.UNEVALUABLE,
            reasons=("RECEIVER_CONTINUATION_BRANCH_MISSING",),
        ),
        _result(
            "fixture.ordinary-fold",
            state_dimension=1,
            observation_dimension=1,
            rank=fold_rank,
            fiber=ReceiverFiberDisposition.FINITE_MULTIPLE,
            branch=BranchTopologyDisposition.ORDINARY_FOLD,
            count=(2, 2),
            metrics=(_metric("fold-regular-point-rank", fold_rank),),
        ),
        _result(
            "fixture.stochastic-mixture",
            state_dimension=1,
            observation_dimension=1,
            rank=None,
            fiber=ReceiverFiberDisposition.MIXTURE_OR_UNRESOLVED,
            branch=BranchTopologyDisposition.STOCHASTIC_SWITCHING,
            count=None,
            metrics=(_metric("mixture-component-count-known-to-evaluator", 2),),
        ),
        _result(
            "fixture.threefold-cover",
            state_dimension=2,
            observation_dimension=2,
            rank=cover_rank,
            fiber=ReceiverFiberDisposition.FINITE_MULTIPLE,
            branch=BranchTopologyDisposition.MONODROMY,
            count=(3, 3),
            forward=forward,
            reverse=reverse,
            metrics=(
                _metric("continuation-refinement-agreement", int(refined == forward)),
                _metric("contractible-loop-identity", int(contractible == (0, 1, 2))),
            ),
            status=(
                ReceiverGeometryMethodStatus.SUPPORTED
                if cover_supported
                else ReceiverGeometryMethodStatus.OPPOSED
            ),
            reasons=() if cover_supported else ("RECEIVER_CONTINUATION_CONTROL_FAILED",),
        ),
    )
    false_promotions = sum(
        value.fixture_id
        in {
            "fixture.continuous-projection",
            "fixture.missing-branch-counterfeit",
            "fixture.stochastic-mixture",
        }
        and value.fiber_disposition is ReceiverFiberDisposition.FINITE_MULTIPLE
        for value in results
    )
    missed = sum(
        value.fixture_id
        in {
            "fixture.injective-linear",
            "fixture.keller-nonproper",
            "fixture.ordinary-fold",
            "fixture.threefold-cover",
        }
        and value.status is not ReceiverGeometryMethodStatus.SUPPORTED
        for value in results
    )
    return ReceiverGeometryMethodSuiteResult(
        suite_id="suite.receiver-conditioned-io.receiver-geometry-truth-known",
        config_sha256=config.fingerprint(),
        fixture_results=results,
        false_finite_promotion_count=false_promotions,
        missed_primary_count=missed,
        status=(
            ReceiverGeometryMethodStatus.SUPPORTED
            if false_promotions == 0 and missed == 0
            else ReceiverGeometryMethodStatus.OPPOSED
        ),
        reason_codes=()
        if false_promotions == 0 and missed == 0
        else ("RECEIVER_GEOMETRY_DECISIVE_CONTROL_FAILED",),
    )


__all__ = [
    "RECEIVER_GEOMETRY_FIXTURE_IDS",
    "RECEIVER_GEOMETRY_METHOD_VERSION",
    "ReceiverGeometryFixtureResult",
    "ReceiverGeometryMethodConfig",
    "ReceiverGeometryMethodStatus",
    "ReceiverGeometryMethodSuiteResult",
    "default_receiver_geometry_method_config",
    "numerical_rank",
    "run_receiver_geometry_truth_known_suite",
    "track_cube_cover",
]
