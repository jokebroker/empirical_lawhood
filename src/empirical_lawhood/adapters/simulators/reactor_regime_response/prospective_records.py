"""Immutable D root output with all assigned branch slots and full continuations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ROOTS
from empirical_lawhood.adapters.methods.reactor_regime_response.prospective_decision import CausalValidityRegimeAssignment
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_lock import RegimeDPreparedRequestLock
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt

from .prospective_acquisition import ProspectiveRootAcquisition


def _matches_lock(
    root: str,
    lock: RegimeDPreparedRequestLock,
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
class RegimeDNativeBranch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-regime-response/regime-d-native-branch'

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
            f"{self.root}.request-{self.index}."
            f"{'committed' if self.view == 0 else 'refined'}-task"
            if self.role == "PRIMARY" else
            f"{self.root}.evaluator-word-{self.index}.view-{self.view}"
        )
        if (
            self.root not in {name for name, role, _, _ in ROOTS if role == "prospective"}
            or self.role not in ("PRIMARY", "EVALUATOR")
            or self.index not in (range(4) if self.role == "PRIMARY" else range(3))
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
        cls, root: str, role: str, index: int, view: int,
        episode: NativeEpisode | None,
    ) -> RegimeDNativeBranch:
        if episode is None:
            return cls(root, role, index, view, "NONATTEMPT", None, None, None)
        if episode.root != root or episode.dt != (1.0 if view == 0 else .5):
            raise ValueError("D branch substitutes its root or numerical view")
        arrays = LocalArrayPayload.pack({
            "observations": episode.observations,
            "requests": episode.requests,
            "stages": episode.stages,
            "exposure": episode.exposure,
            "grid": episode.grid,
            "callback_cpu": episode.callback_cpu,
        })
        return cls(
            root, role, index, view,
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
            self.root, self.native_name,
            float(D(1) if self.view == 0 else D(".5")),
            arrays["observations"], arrays["requests"], arrays["stages"],
            arrays["exposure"], arrays["grid"], arrays["callback_cpu"],
            self.failure,
        )
        if episode.complete != (self.status == "COMPLETED"):
            raise ValueError("D branch full continuation disagrees with its status")
        return episode


@dataclass(frozen=True, slots=True)
class RegimeDNativeRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-regime-response/regime-d-native-root'

    root: str
    assignment: ObjectIdentity
    controller_evaluation_plan: ObjectIdentity | None
    compiled: tuple[CompiledDeliveryControllerStudy | None, ...]
    commitments: tuple[DeliveryControllerDecisionCommitment | None, ...]
    ticks: tuple[DeliveryControllerTickReceipt | None, ...]
    locks: tuple[RegimeDPreparedRequestLock, ...]
    branches: tuple[RegimeDNativeBranch, ...]
    native_calls: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (role, index, view)
            for role, count in (("PRIMARY", 4), ("EVALUATOR", 3))
            for index in range(count) for view in (0, 1)
        )
        if (
            self.root not in {name for name, role, _, _ in ROOTS if role == "prospective"}
            or self.assignment.object_schema != CausalValidityRegimeAssignment.SCHEMA
            or self.assignment.object_id != f"{self.root}.prospective-assignment"
            or self.controller_evaluation_plan is not None and self.controller_evaluation_plan.object_schema
            != 'empirical-lawhood/planning/coupled-realization-controller-evaluation-plan'
            or len(self.compiled) != 4
            or len(self.commitments) != 4
            or len(self.ticks) != 4
            or len(self.locks) != (4 if self.controller_evaluation_plan is not None else 0)
            or tuple(lock.request_index for lock in self.locks)
            != tuple(range(len(self.locks)))
            or any(
                not _matches_lock(
                    self.root, lock, self.compiled[index], self.commitments[index]
                )
                for index, lock in enumerate(self.locks)
            )
            or tuple((row.role, row.index, row.view) for row in self.branches) != expected
            or any(row.root != self.root for row in self.branches)
            or self.native_calls != sum(row.status != "NONATTEMPT" for row in self.branches)
            or not 0 <= self.native_calls <= 14
            or (self.controller_evaluation_plan is None and (
                self.native_calls != 0 or any(self.compiled) or any(self.commitments)
            ))
        ):
            raise ValueError("D root lost its four requests, six chart branches or controller-use identity")

    @classmethod
    def from_acquisition(
        cls,
        assignment: CausalValidityRegimeAssignment,
        controller_evaluation_plan: ObjectIdentity | None,
        result: ProspectiveRootAcquisition,
    ) -> RegimeDNativeRoot:
        if result.root != assignment.root:
            raise ValueError("D native root differs from sealed assignment")
        branches = tuple(
            RegimeDNativeBranch.from_episode(
                result.root, "PRIMARY", index, view,
                (result.nominal if view == 0 else result.refined)[index],
            )
            for index in range(4) for view in (0, 1)
        ) + tuple(
            RegimeDNativeBranch.from_episode(
                result.root, "EVALUATOR", index, view,
                result.evaluator_chart[2 * index + view]
                if len(result.evaluator_chart) == 6 else None,
            )
            for index in range(3) for view in (0, 1)
        )
        return cls(
            result.root,
            ObjectIdentity.from_record(assignment.assignment_id, assignment),
            controller_evaluation_plan,
            tuple(None if value is None else value.compiled for value in result.prepared),
            tuple(None if value is None else value.commitment for value in result.prepared),
            result.ticks,
            result.locks,
            branches,
            result.native_calls,
            result.reasons,
        )
