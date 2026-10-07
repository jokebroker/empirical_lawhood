"""Exact synthetic controller comparison public-route conformance for flagship readiness."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)

from .route_conformance import PUBLIC_ROUTE_STAGE_ORDER, PublicRouteConformanceStage, RunEnvelopeConformanceSuiteReceipt


class SyntheticControllerTaskArchetype(StrEnum):
    RECEIVER_SINK = 'RECEIVER_SINK'
    HIDDEN_HISTORY = 'HIDDEN_HISTORY'
    ACTION_PORT_ORDER = 'ACTION_PORT_ORDER'
    NUMERICAL_REGIME = 'NUMERICAL_REGIME'


class SyntheticControllerComparisonArm(StrEnum):
    WITNESS_GATED_IO = "witness-gated-io"
    IO_ONLY = "io-only"
    CONSTRAINED_MPC = "constrained-mpc"
    ROBUST_MAXIMIN = "robust-maximin"


@dataclass(frozen=True, slots=True)
class SyntheticControllerComparisonTaskSpec(CanonicalRecord):
    """One outcome-blind fake task in the exact balanced synthetic controller comparison roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/synthetic-controller-comparison-task-spec'

    task_id: str
    archetype: SyntheticControllerTaskArchetype
    generator_seed: int

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        if self.generator_seed < 0:
            raise ValueError("synthetic controller comparison generator seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class SyntheticControllerCoordinateConformanceReceipt(CanonicalRecord):
    """One task/arm coordinate joined to the complete truth-known public route."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/synthetic-controller-coordinate-conformance-receipt'

    receipt_id: str
    task: ObjectIdentity
    arm: SyntheticControllerComparisonArm
    public_route_suite: ObjectIdentity
    covered_stages: tuple[PublicRouteConformanceStage, ...]
    native_simulator_launches: int
    terminal: bool
    architecture_conformance_only: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.task.object_schema != SyntheticControllerComparisonTaskSpec.SCHEMA:
            raise ValueError("synthetic controller comparison coordinate binds another task schema")
        if self.public_route_suite.object_schema != RunEnvelopeConformanceSuiteReceipt.SCHEMA:
            raise ValueError("synthetic controller comparison coordinate binds another public-route suite schema")
        if self.covered_stages != PUBLIC_ROUTE_STAGE_ORDER:
            raise ValueError("synthetic controller comparison coordinate does not cover the exact public route")
        if self.native_simulator_launches != 0:
            raise ValueError("synthetic controller comparison conformance cannot launch an external simulator")
        if (
            not self.terminal
            or not self.architecture_conformance_only
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("synthetic controller comparison coordinate overclaims architecture conformance")


@dataclass(frozen=True, slots=True)
class SyntheticControllerRouteConformanceReceipt(CanonicalRecord):
    """Terminal exact 8-task by 4-arm fake-source campaign receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/synthetic-controller-route-conformance-receipt'

    receipt_id: str
    tasks: tuple[SyntheticControllerComparisonTaskSpec, ...]
    coordinates: tuple[SyntheticControllerCoordinateConformanceReceipt, ...]
    public_route_suite: ObjectIdentity
    external_artifact_plane_required: bool
    native_simulator_launches: int
    terminal: bool
    architecture_conformance_only: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(self.tasks, attribute="task_id", field_name="tasks")
        require_sorted_unique_ids(
            self.coordinates,
            attribute="receipt_id",
            field_name="coordinates",
        )
        if len(self.tasks) != 8:
            raise ValueError("synthetic controller comparison conformance requires exactly eight tasks")
        if len({value.generator_seed for value in self.tasks}) != len(self.tasks):
            raise ValueError("synthetic controller comparison task generator seeds must be distinct")
        if Counter(value.archetype for value in self.tasks) != Counter(
            {value: 2 for value in SyntheticControllerTaskArchetype}
        ):
            raise ValueError("synthetic controller comparison conformance requires two tasks per archetype")
        task_by_identity = {
            ObjectIdentity.from_record(value.task_id, value): value for value in self.tasks
        }
        expected_product = set(product(task_by_identity, SyntheticControllerComparisonArm))
        actual_product = {(value.task, value.arm) for value in self.coordinates}
        if len(self.coordinates) != 32 or actual_product != expected_product:
            raise ValueError("synthetic controller comparison conformance is not the exact task-by-arm product")
        if any(
            value.task not in task_by_identity
            or value.public_route_suite != self.public_route_suite
            or value.native_simulator_launches != 0
            for value in self.coordinates
        ):
            raise ValueError("synthetic controller comparison coordinate changes task, route or launch semantics")
        if self.public_route_suite.object_schema != RunEnvelopeConformanceSuiteReceipt.SCHEMA:
            raise ValueError("synthetic controller comparison campaign binds another public-route suite schema")
        if (
            not self.external_artifact_plane_required
            or self.native_simulator_launches != 0
            or not self.terminal
            or not self.architecture_conformance_only
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("synthetic controller comparison campaign crosses its architecture-only boundary")


def build_synthetic_controller_public_route_conformance_receipt(
    *,
    receipt_id: str,
    tasks: tuple[SyntheticControllerComparisonTaskSpec, ...],
    public_route_suite: RunEnvelopeConformanceSuiteReceipt,
) -> SyntheticControllerRouteConformanceReceipt:
    """Build the exact Cartesian synthetic controller comparison receipt over an executed truth-known route suite."""

    route_identity = ObjectIdentity.from_record(public_route_suite.suite_id, public_route_suite)
    ordered_tasks = tuple(sorted(tasks, key=lambda value: value.task_id))
    coordinates = tuple(
        sorted(
            (
                SyntheticControllerCoordinateConformanceReceipt(
                    receipt_id=f"{receipt_id}.{task.task_id}.{arm.value.lower()}",
                    task=ObjectIdentity.from_record(task.task_id, task),
                    arm=arm,
                    public_route_suite=route_identity,
                    covered_stages=PUBLIC_ROUTE_STAGE_ORDER,
                    native_simulator_launches=0,
                    terminal=True,
                    architecture_conformance_only=True,
                    created_empirical_evidence=False,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                )
                for task, arm in product(ordered_tasks, SyntheticControllerComparisonArm)
            ),
            key=lambda value: value.receipt_id,
        )
    )
    return SyntheticControllerRouteConformanceReceipt(
        receipt_id=receipt_id,
        tasks=ordered_tasks,
        coordinates=coordinates,
        public_route_suite=route_identity,
        external_artifact_plane_required=True,
        native_simulator_launches=0,
        terminal=True,
        architecture_conformance_only=True,
        created_empirical_evidence=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


__all__ = [
    'SyntheticControllerTaskArchetype',
    'SyntheticControllerComparisonArm',
    'SyntheticControllerCoordinateConformanceReceipt',
    'SyntheticControllerRouteConformanceReceipt',
    'SyntheticControllerComparisonTaskSpec',
    'build_synthetic_controller_public_route_conformance_receipt',
]
