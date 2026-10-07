"""Immutable D root output with all assigned branch slots and full continuations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import ROOTS
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalDecision
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_lock import PreparedForecastLock
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt

from .prospective_acquisition import ClassicalRootAcquisition


def _matches_lock(
    root: str,
    lock: PreparedForecastLock,
    compiled: CompiledDeliveryControllerStudy | None,
    commitment: DeliveryControllerDecisionCommitment | None,
) -> bool:
    return bool(
        compiled is not None
        and commitment is not None
        and lock.design.root_id == root
        and lock.parent.decision == commitment
        and lock.parent.decision.compiled_study
        == ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
    )


@dataclass(frozen=True, slots=True)
class ClassicalNativeBranch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-selected-action-response/classical-native-branch'

    root: str
    role: str
    index: int
    view: int
    status: str
    native_name: str | None
    failure: str | None
    arrays: LocalArrayPayload | None

    def __post_init__(self) -> None:
        expected_name = (
            f"{self.root}.request-{self.index}.{'committed' if self.view == 0 else 'refined'}-task"
            if self.role == "PRIMARY"
            else f"{self.root}.evaluator-word-{self.index}.view-{self.view}"
        )
        if (
            self.root not in {name for name, role, _, _ in ROOTS if role == "prospective"}
            or self.role not in ("PRIMARY", "EVALUATOR")
            or self.index not in (range(1) if self.role == "PRIMARY" else range(2))
            or self.view not in (0, 1)
            or self.status not in ("NONATTEMPT", "COMPLETED", "INCOMPLETE")
            or (self.status == "NONATTEMPT") != (self.arrays is None)
            or (self.arrays is None) != (self.native_name is None)
            or (self.native_name is not None and self.native_name != expected_name)
            or (self.status == "NONATTEMPT" and self.failure is not None)
            or (self.status == "COMPLETED" and self.failure is not None)
            or (self.status == "INCOMPLETE" and self.failure is None)
        ):
            raise ValueError("D native branch changed its assigned request/chart slot")

    @classmethod
    def from_episode(
        cls,
        root: str,
        role: str,
        index: int,
        view: int,
        episode: NativeEpisode | None,
    ) -> ClassicalNativeBranch:
        if episode is None:
            return cls(root, role, index, view, "NONATTEMPT", None, None, None)
        if episode.root != root or episode.dt != (1.0 if view == 0 else 0.5):
            raise ValueError("D branch substitutes its root or numerical view")
        arrays = LocalArrayPayload.pack(
            {
                "observations": episode.observations,
                "requests": episode.requests,
                "stages": episode.stages,
                "exposure": episode.exposure,
                "grid": episode.grid,
                "callback_cpu": episode.callback_cpu,
            }
        )
        return cls(
            root,
            role,
            index,
            view,
            "COMPLETED" if episode.complete else "INCOMPLETE",
            episode.episode,
            None if episode.complete else episode.failure or "INCOMPLETE_NATIVE_BATCH",
            arrays,
        )

    def episode(self) -> NativeEpisode | None:
        if self.arrays is None:
            return None
        arrays = self.arrays.unpack()
        expected = {"observations", "requests", "stages", "exposure", "grid", "callback_cpu"}
        if set(arrays) != expected or self.native_name is None:
            raise ValueError("D branch lost a required full native continuation array")
        episode = NativeEpisode(
            self.root,
            self.native_name,
            float(D(1) if self.view == 0 else D(".5")),
            arrays["observations"],
            arrays["requests"],
            arrays["stages"],
            arrays["exposure"],
            arrays["grid"],
            arrays["callback_cpu"],
            self.failure,
        )
        if episode.complete != (self.status == "COMPLETED"):
            raise ValueError("D branch full continuation disagrees with its status")
        return episode


@dataclass(frozen=True, slots=True)
class ClassicalNativeRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-selected-action-response/classical-native-root'

    root: str
    assignment: ObjectIdentity
    controller_evaluation_plan: ObjectIdentity | None
    compiled: tuple[CompiledDeliveryControllerStudy | None, ...]
    commitments: tuple[DeliveryControllerDecisionCommitment | None, ...]
    ticks: tuple[DeliveryControllerTickReceipt | None, ...]
    locks: tuple[PreparedForecastLock, ...]
    branches: tuple[ClassicalNativeBranch, ...]
    native_calls: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (role, index, view)
            for role, count in (("PRIMARY", 1), ("EVALUATOR", 2))
            for index in range(count)
            for view in (0, 1)
        )
        if (
            self.root not in {name for name, role, _, _ in ROOTS if role == "prospective"}
            or self.assignment.object_schema != ClassicalDecision.SCHEMA
            or self.assignment.object_id != f"{self.root}.classical-decision"
            or self.controller_evaluation_plan is not None
            and self.controller_evaluation_plan.object_schema
            != 'empirical-lawhood/planning/coupled-realization-controller-evaluation-plan'
            or len(self.compiled) != 1
            or len(self.commitments) != 1
            or len(self.ticks) != 1
            or len(self.locks) != (1 if self.controller_evaluation_plan is not None else 0)
            or tuple(lock.design.policy_id for lock in self.locks)
            != tuple(f"reactor-classical-request-{i}" for i in range(len(self.locks)))
            or any(
                not _matches_lock(self.root, lock, self.compiled[index], self.commitments[index])
                for index, lock in enumerate(self.locks)
            )
            or tuple((row.role, row.index, row.view) for row in self.branches) != expected
            or any(row.root != self.root for row in self.branches)
            or self.native_calls != sum(row.status != "NONATTEMPT" for row in self.branches)
            or not 0 <= self.native_calls <= 6
            or (
                self.controller_evaluation_plan is None
                and (self.native_calls != 0 or any(self.compiled) or any(self.commitments))
            )
        ):
            raise ValueError(
                "D root lost its one request and its reference/audit branches or controller-use identity"
            )

    @classmethod
    def from_acquisition(
        cls,
        assignment: ClassicalDecision,
        controller_evaluation_plan: ObjectIdentity | None,
        result: ClassicalRootAcquisition,
    ) -> ClassicalNativeRoot:
        if result.root != assignment.root:
            raise ValueError("D native root differs from sealed assignment")
        branches = tuple(
            ClassicalNativeBranch.from_episode(
                result.root,
                "PRIMARY",
                index,
                view,
                (result.nominal if view == 0 else result.refined)[index],
            )
            for index in range(1)
            for view in (0, 1)
        ) + tuple(
            ClassicalNativeBranch.from_episode(
                result.root,
                "EVALUATOR",
                index,
                view,
                result.evaluator_chart[2 * index + view]
                if len(result.evaluator_chart) == 4
                else None,
            )
            for index in range(2)
            for view in (0, 1)
        )
        return cls(
            result.root,
            ObjectIdentity.from_record(assignment.decision_id, assignment),
            controller_evaluation_plan,
            tuple(None if value is None else value.compiled for value in result.prepared),
            tuple(None if value is None else value.commitment for value in result.prepared),
            result.ticks,
            result.locks,
            branches,
            result.native_calls,
            result.reasons,
        )
