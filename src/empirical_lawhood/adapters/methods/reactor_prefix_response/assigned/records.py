# SPDX-License-Identifier: MPL-2.0

"Native-unit assignment and custody records."

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorAssignedPrefixConfig, ReactorAssignedPrefixPanel, ReactorPrefixAssignment, assigned_prefix_config
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import native_task_id
from empirical_lawhood.kernel.status import OperationalStatus

from ..design import ReactorScienceDesign, reactor_science_design
from ..projection import ReactorNativeCustody, ReactorProjection, ReactorStageEvidence
from ..result import ReactorScienceResult


@dataclass(frozen=True, slots=True)
class ReactorAssignedScienceDesign(ReactorScienceDesign):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/assigned/reactor-assigned-science-design'
    VERSION: ClassVar[str] = '1.0.0'
    native: ReactorAssignedPrefixConfig

    def __post_init__(self) -> None:
        ReactorScienceDesign.__post_init__(self)
        if type(self.native) is not ReactorAssignedPrefixConfig:
            raise ValueError("REACTOR_ASSIGNED_SCIENCE_NATIVE_SCHEMA_MISMATCH")


def assigned_science_design(
    assignment: ReactorPrefixAssignment,
) -> ReactorAssignedScienceDesign:
    original = reactor_science_design()
    return ReactorAssignedScienceDesign(
        f"{assignment.assignment_id}.science-design",
        assigned_prefix_config(assignment),
        original.maximum_heldout_absolute_error_k,
        original.maximum_refinement_absolute_delta_k,
        original.minimum_complete_units,
        original.bootstrap_replicates,
        original.bootstrap_seed,
        original.simultaneous_confidence_level,
    )


@dataclass(frozen=True, slots=True)
class ReactorAssignedStageEvidence(ReactorStageEvidence):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/assigned/reactor-assigned-stage-evidence'
    VERSION: ClassVar[str] = '1.0.0'
    PANEL_SCHEMA: ClassVar[str] = ReactorAssignedPrefixPanel.SCHEMA


@dataclass(frozen=True, slots=True)
class ReactorAssignedNativeCustody(ReactorNativeCustody):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/assigned/reactor-assigned-native-custody'
    VERSION: ClassVar[str] = '1.0.0'
    config: ReactorAssignedPrefixConfig

    def __post_init__(self) -> None:
        expected = tuple(
            native_task_id(*branch, config=self.config)
            for branch in self.config.branches
        )
        if (
            tuple(sorted(r.task_id for r in self.receipts)) != expected
            or len({r.receipt_id for r in self.receipts}) != 20
            or len({r.run_id for r in self.receipts}) != 1
            or any(
                r.operational_status is not OperationalStatus.SUCCEEDED
                for r in self.receipts
            )
        ):
            raise ValueError("REACTOR_ASSIGNED_NATIVE_CUSTODY_INCOMPLETE")


@dataclass(frozen=True, slots=True)
class ReactorAssignedProjection(ReactorProjection):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/assigned/reactor-assigned-projection'
    VERSION: ClassVar[str] = '1.0.0'
    design: ReactorAssignedScienceDesign
    stage_evidence: tuple[ReactorAssignedStageEvidence, ...]
    native_custody: ReactorAssignedNativeCustody | None = None


@dataclass(frozen=True, slots=True)
class ReactorAssignedScienceResult(ReactorScienceResult):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/assigned/reactor-assigned-science-result'
    VERSION: ClassVar[str] = '1.0.0'
    projection: ReactorAssignedProjection
