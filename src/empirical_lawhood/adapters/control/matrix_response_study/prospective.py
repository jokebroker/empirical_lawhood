"""Plan-scoped coordination for Six-matrix response outer and nested prospective evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy
from empirical_lawhood.planning.matched_action_hold_controller_evaluation import MatchedActionHoldControllerEvaluationPlan

from .contracts import MatrixResponseOuterArmDisposition, MatrixResponseOuterArm, MatrixResponseProspectiveEvaluationTopology


@dataclass(frozen=True, slots=True)
class MatrixResponsePhysicalPanel(CanonicalRecord):
    """The sole outer/inner prospective evaluation inference unit, distinct from its arm views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-physical-panel'
    VERSION: ClassVar[str] = "1.0.0"

    panel_id: str
    target_constitution_id: str
    preparation_instance_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    fresh_for_prospective_evaluation: bool

    def __post_init__(self) -> None:
        for name in (
            "panel_id",
            "preparation_instance_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.target_constitution_id not in {"00", "01", "10", "11"}:
            raise ValueError("Six-matrix response panel names an unknown target constitution")
        if not self.fresh_for_prospective_evaluation:
            raise ValueError("Six-matrix response prospective evaluation panels must be fresh prospective units")


@dataclass(frozen=True, slots=True)
class MatrixResponseOuterArmPlan(CanonicalRecord):
    """One arm allocation; panel IDs, not branches or rows, determine ``n``."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-outer-arm-plan'
    VERSION: ClassVar[str] = "1.0.0"

    plan_id: str
    arm: MatrixResponseOuterArm
    panel_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]
    law: ObjectIdentity | None
    study: ObjectIdentity | None
    matched_evaluation_plan: ObjectIdentity | None
    matched_branch_count: int
    panel_is_inference_unit: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        require_sorted_unique_strings(self.panel_ids, field_name="panel_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        if len(self.panel_ids) != len(self.physical_independent_unit_ids):
            raise ValueError("Six-matrix response arm plan changes panel/inference-unit cardinality")
        if self.arm.disposition is MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME:
            if (
                self.law is None
                or self.law.object_schema != ResponseLaw.SCHEMA
                or self.study is None
                or self.study.object_schema != AdmissionControllerStudy.SCHEMA
                or self.matched_evaluation_plan is None
                or self.matched_evaluation_plan.object_schema
                != MatchedActionHoldControllerEvaluationPlan.SCHEMA
                or self.matched_branch_count < len(self.panel_ids)
            ):
                raise ValueError("typed Six-matrix response arm lacks its exact law/programme compiler/matched-prospective evaluation plan")
        elif (
            any(
                value is not None
                for value in (self.law, self.study, self.matched_evaluation_plan)
            )
            or self.matched_branch_count
        ):
            raise ValueError("comparator/HOLD arm cannot claim a qualified law programme")
        if not self.panel_is_inference_unit or self.grants_authority:
            raise ValueError("Six-matrix response arm plan cannot inflate n or grant authority")


@dataclass(frozen=True, slots=True)
class MatrixResponseOuterProspectiveUmbrella(CanonicalRecord):
    """Five allocations over exactly 80 physical panels and four strata."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-outer-prospective-umbrella'
    VERSION: ClassVar[str] = "1.0.0"

    umbrella_id: str
    topology: ObjectIdentity
    panels: tuple[MatrixResponsePhysicalPanel, ...]
    arm_plans: tuple[MatrixResponseOuterArmPlan, ...]
    development_independent_unit_ids: tuple[str, ...]
    inference_unit_count: int
    arm_allocation_count: int
    matched_branch_count: int
    panel_is_sole_inference_unit: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.umbrella_id, field_name="umbrella_id")
        if self.topology.object_schema != MatrixResponseProspectiveEvaluationTopology.SCHEMA:
            raise ValueError("Six-matrix response prospective evaluation umbrella binds another topology schema")
        require_sorted_unique_ids(self.panels, attribute="panel_id", field_name="panels")
        require_sorted_unique_ids(self.arm_plans, attribute="plan_id", field_name="arm_plans")
        require_sorted_unique_strings(
            self.development_independent_unit_ids,
            field_name="development_independent_unit_ids",
        )
        panel_ids = tuple(value.panel_id for value in self.panels)
        physical_ids = tuple(value.physical_independent_unit_id for value in self.panels)
        if len(set(physical_ids)) != len(physical_ids):
            raise ValueError("Six-matrix response panels repeat a physical independent unit")
        allocated = tuple(panel_id for plan in self.arm_plans for panel_id in plan.panel_ids)
        if tuple(sorted(allocated)) != panel_ids or len(allocated) != len(set(allocated)):
            raise ValueError("Six-matrix response outer arms do not partition the exact panel roster")
        if set(physical_ids) & set(self.development_independent_unit_ids):
            raise ValueError("Six-matrix response prospective evaluation panel reuses a development/programme admission independent unit")
        if (
            self.inference_unit_count != len(self.panels)
            or self.inference_unit_count != 80
            or self.arm_allocation_count != len(self.panels)
            or len(self.arm_plans) != 5
        ):
            raise ValueError("Six-matrix response umbrella reclassifies arms/branches as independent n")
        if self.matched_branch_count != sum(value.matched_branch_count for value in self.arm_plans):
            raise ValueError("Six-matrix response matched branch count differs from its arm plans")
        if not self.panel_is_sole_inference_unit:
            raise ValueError("Six-matrix response panel must remain the sole prospective evaluation inference unit")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not self.visibility_ceiling.is_promotable
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response prospective evaluation umbrella crossed the prospective/authority boundary")


def build_matrix_response_study_outer_arm_plan(
    *,
    plan_id: str,
    arm: MatrixResponseOuterArm,
    panels: tuple[MatrixResponsePhysicalPanel, ...],
    law: ResponseLaw | None = None,
    study: AdmissionControllerStudy | None = None,
    matched_plan: MatchedActionHoldControllerEvaluationPlan | None = None,
) -> MatrixResponseOuterArmPlan:
    """Bind an actual generic matched plan only for the qualified typed arm."""

    panel_ids = tuple(sorted(value.panel_id for value in panels))
    physical_ids = tuple(sorted(value.physical_independent_unit_id for value in panels))
    if matched_plan is not None:
        planned_physical_ids = tuple(
            sorted(value.physical_independent_unit_id for value in matched_plan.units)
        )
        if planned_physical_ids != physical_ids:
            raise ValueError("generic matched-prospective evaluation units differ from the Six-matrix response panel allocation")
        matched_branch_count = len(matched_plan.execution_cells)
    else:
        matched_branch_count = 0
    return MatrixResponseOuterArmPlan(
        plan_id=plan_id,
        arm=arm,
        panel_ids=panel_ids,
        physical_independent_unit_ids=physical_ids,
        law=(None if law is None else ObjectIdentity.from_record(law.law_id, law)),
        study=(
            None
            if study is None
            else ObjectIdentity.from_record(study.study_id, study)
        ),
        matched_evaluation_plan=(
            None
            if matched_plan is None
            else ObjectIdentity.from_record(matched_plan.evaluation_plan_id, matched_plan)
        ),
        matched_branch_count=matched_branch_count,
        panel_is_inference_unit=True,
        grants_authority=False,
    )


def coordinate_matrix_response_study_outer_prospective(
    *,
    umbrella_id: str,
    topology: MatrixResponseProspectiveEvaluationTopology,
    panels: tuple[MatrixResponsePhysicalPanel, ...],
    arm_plans: tuple[MatrixResponseOuterArmPlan, ...],
    development_independent_unit_ids: tuple[str, ...],
) -> MatrixResponseOuterProspectiveUmbrella:
    """Apply the frozen 80-panel/five-arm topology without changing ``n``."""

    panels = tuple(sorted(panels, key=lambda value: value.panel_id))
    arm_plans = tuple(sorted(arm_plans, key=lambda value: value.plan_id))
    expected_arms = {value.arm_id: value for value in topology.outer_arms}
    if {value.arm.arm_id for value in arm_plans} != set(expected_arms) or any(
        value.arm != expected_arms[value.arm.arm_id] for value in arm_plans
    ):
        raise ValueError("Six-matrix response umbrella arm plans differ from the frozen five-arm roster")
    constitution_counts = {
        value: sum(panel.target_constitution_id == value for panel in panels)
        for value in topology.target_constitution_ids
    }
    if any(
        count != topology.panels_per_target_constitution for count in constitution_counts.values()
    ):
        raise ValueError("Six-matrix response panel roster differs from the four frozen target strata")
    by_panel = {value.panel_id: value for value in panels}
    for plan in arm_plans:
        counts = {
            value: sum(
                by_panel[panel_id].target_constitution_id == value for panel_id in plan.panel_ids
            )
            for value in topology.target_constitution_ids
        }
        if len(set(counts.values())) != 1 or next(iter(counts.values())) < 1:
            raise ValueError("each Six-matrix response arm must retain representation in every target stratum")
    return MatrixResponseOuterProspectiveUmbrella(
        umbrella_id=umbrella_id,
        topology=ObjectIdentity.from_record(topology.topology_id, topology),
        panels=panels,
        arm_plans=arm_plans,
        development_independent_unit_ids=tuple(sorted(development_independent_unit_ids)),
        inference_unit_count=len(panels),
        arm_allocation_count=sum(len(value.panel_ids) for value in arm_plans),
        matched_branch_count=sum(value.matched_branch_count for value in arm_plans),
        panel_is_sole_inference_unit=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        grants_authority=False,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponsePostOuterCheckpoint(CanonicalRecord):
    """Sealed state reached by one exact outer law/programme on one panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-post-outer-checkpoint'
    VERSION: ClassVar[str] = "1.0.0"

    checkpoint_id: str
    panel: ObjectIdentity
    outer_arm_plan: ObjectIdentity
    outer_law: ObjectIdentity
    outer_study: ObjectIdentity
    realized_state_sha256: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        validate_sha256(self.realized_state_sha256, field_name="realized_state_sha256")
        if (
            self.panel.object_schema != MatrixResponsePhysicalPanel.SCHEMA
            or self.outer_arm_plan.object_schema != MatrixResponseOuterArmPlan.SCHEMA
            or self.outer_law.object_schema != ResponseLaw.SCHEMA
            or self.outer_study.object_schema != AdmissionControllerStudy.SCHEMA
        ):
            raise ValueError("Six-matrix response checkpoint binds another panel/arm/law/programme schema")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or not self.visibility_ceiling.is_promotable
            or self.grants_authority
        ):
            raise ValueError("Six-matrix response checkpoint crosses outcome visibility or authority")


def seal_matrix_response_study_post_outer_checkpoint(
    *,
    checkpoint_id: str,
    panel: MatrixResponsePhysicalPanel,
    outer_arm_plan: MatrixResponseOuterArmPlan,
    outer_law: ResponseLaw,
    outer_study: AdmissionControllerStudy,
    realized_state: bytes,
) -> MatrixResponsePostOuterCheckpoint:
    """Seal only the state produced by the exact typed arm inputs."""

    if not realized_state:
        raise ValueError("Six-matrix response post-outer checkpoint cannot hash an empty state")
    law_identity = ObjectIdentity.from_record(outer_law.law_id, outer_law)
    programme_identity = ObjectIdentity.from_record(
        outer_study.study_id,
        outer_study,
    )
    if (
        outer_arm_plan.arm.disposition is not MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME
        or panel.panel_id not in outer_arm_plan.panel_ids
        or panel.physical_independent_unit_id not in outer_arm_plan.physical_independent_unit_ids
        or outer_arm_plan.law != law_identity
        or outer_arm_plan.study != programme_identity
    ):
        raise ValueError("Six-matrix response post-outer checkpoint substitutes its typed arm inputs")
    return MatrixResponsePostOuterCheckpoint(
        checkpoint_id=checkpoint_id,
        panel=ObjectIdentity.from_record(panel.panel_id, panel),
        outer_arm_plan=ObjectIdentity.from_record(outer_arm_plan.plan_id, outer_arm_plan),
        outer_law=law_identity,
        outer_study=programme_identity,
        realized_state_sha256=sha256(realized_state).hexdigest(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        grants_authority=False,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseInnerChildIssue(CanonicalRecord):
    """Frozen nested child inputs issued before any lower-world outcome access."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-inner-child-issue'
    VERSION: ClassVar[str] = "1.0.0"

    issue_id: str
    panel: ObjectIdentity
    post_outer_checkpoint: ObjectIdentity
    lower_law: ObjectIdentity
    lower_study: ObjectIdentity
    target_constitution_id: str
    anonymous_mode_index: int
    outcome_access: OutcomeAccess
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        if (
            self.panel.object_schema != MatrixResponsePhysicalPanel.SCHEMA
            or self.post_outer_checkpoint.object_schema != MatrixResponsePostOuterCheckpoint.SCHEMA
            or self.lower_law.object_schema != ResponseLaw.SCHEMA
            or self.lower_study.object_schema != AdmissionControllerStudy.SCHEMA
        ):
            raise ValueError("Six-matrix response inner issue binds another checkpoint/law/programme schema")
        if self.target_constitution_id not in {"01", "10", "11"}:
            raise ValueError("Six-matrix response inner prospective evaluation is not applicable to the zero constitution")
        if self.anonymous_mode_index not in range(4):
            raise ValueError("Six-matrix response inner issue names an unknown anonymous mode")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND or self.grants_authority:
            raise ValueError("Six-matrix response inner issue crossed the prospective/authority boundary")


@dataclass(frozen=True, slots=True)
class MatrixResponseInnerActionHoldChild(CanonicalRecord):
    """ACTION/HOLD views sharing the exact parent panel and checkpoint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/matrix-response-study/matrix-response-inner-action-hold-child'
    VERSION: ClassVar[str] = "1.0.0"

    child_id: str
    issue: ObjectIdentity
    panel: ObjectIdentity
    post_outer_checkpoint: ObjectIdentity
    lower_law: ObjectIdentity
    lower_study: ObjectIdentity
    action_view_id: str
    hold_view_id: str
    inference_unit_count: int
    branch_view_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("child_id", "action_view_id", "hold_view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.issue.object_schema != MatrixResponseInnerChildIssue.SCHEMA:
            raise ValueError("Six-matrix response inner child binds another issue schema")
        if self.action_view_id == self.hold_view_id:
            raise ValueError("Six-matrix response inner ACTION and HOLD must remain distinct views")
        if self.inference_unit_count != 1 or self.branch_view_count != 2:
            raise ValueError("Six-matrix response inner ACTION/HOLD views cannot inflate panel n")
        if self.grants_authority:
            raise ValueError("Six-matrix response inner child cannot grant authority")


def issue_matrix_response_study_inner_child(
    *,
    issue_id: str,
    panel: MatrixResponsePhysicalPanel,
    checkpoint: MatrixResponsePostOuterCheckpoint,
    lower_law: ResponseLaw,
    lower_study: AdmissionControllerStudy,
    target_constitution_id: str,
    anonymous_mode_index: int,
) -> MatrixResponseInnerChildIssue:
    if checkpoint.panel != ObjectIdentity.from_record(panel.panel_id, panel):
        raise ValueError("Six-matrix response inner issue substitutes the outer panel checkpoint")
    if panel.target_constitution_id != target_constitution_id:
        raise ValueError("Six-matrix response inner issue changes its parent target constitution")
    return MatrixResponseInnerChildIssue(
        issue_id=issue_id,
        panel=ObjectIdentity.from_record(panel.panel_id, panel),
        post_outer_checkpoint=ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint),
        lower_law=ObjectIdentity.from_record(lower_law.law_id, lower_law),
        lower_study=ObjectIdentity.from_record(lower_study.study_id, lower_study),
        target_constitution_id=target_constitution_id,
        anonymous_mode_index=anonymous_mode_index,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        grants_authority=False,
    )


def activate_matrix_response_study_inner_child(
    *,
    child_id: str,
    issue: MatrixResponseInnerChildIssue,
    panel: MatrixResponsePhysicalPanel,
    checkpoint: MatrixResponsePostOuterCheckpoint,
    lower_law: ResponseLaw,
    lower_study: AdmissionControllerStudy,
) -> MatrixResponseInnerActionHoldChild:
    """Reject every post-issue law/programme/checkpoint/panel substitution."""

    exact = (
        (issue.panel, ObjectIdentity.from_record(panel.panel_id, panel), "panel"),
        (
            issue.post_outer_checkpoint,
            ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint),
            "checkpoint",
        ),
        (issue.lower_law, ObjectIdentity.from_record(lower_law.law_id, lower_law), "law"),
        (
            issue.lower_study,
            ObjectIdentity.from_record(lower_study.study_id, lower_study),
            'study',
        ),
    )
    for expected, observed, label in exact:
        if expected != observed:
            raise ValueError(f"Six-matrix response inner child substituted its {label}")
    if (
        checkpoint.panel != issue.panel
        or panel.target_constitution_id != issue.target_constitution_id
    ):
        raise ValueError("Six-matrix response inner child loses outer panel/constitution identity")
    return MatrixResponseInnerActionHoldChild(
        child_id=child_id,
        issue=ObjectIdentity.from_record(issue.issue_id, issue),
        panel=issue.panel,
        post_outer_checkpoint=issue.post_outer_checkpoint,
        lower_law=issue.lower_law,
        lower_study=issue.lower_study,
        action_view_id=f"{child_id}.action",
        hold_view_id=f"{child_id}.hold",
        inference_unit_count=1,
        branch_view_count=2,
        grants_authority=False,
    )


class MatrixResponseProspectiveEvaluationCoordinator:
    """Bound topology façade over the plan-scoped pure coordination functions."""

    def __init__(self, topology: MatrixResponseProspectiveEvaluationTopology) -> None:
        self.topology = topology

    def coordinate_outer(
        self,
        *,
        umbrella_id: str,
        panels: tuple[MatrixResponsePhysicalPanel, ...],
        arm_plans: tuple[MatrixResponseOuterArmPlan, ...],
        development_independent_unit_ids: tuple[str, ...],
    ) -> MatrixResponseOuterProspectiveUmbrella:
        return coordinate_matrix_response_study_outer_prospective(
            umbrella_id=umbrella_id,
            topology=self.topology,
            panels=panels,
            arm_plans=arm_plans,
            development_independent_unit_ids=development_independent_unit_ids,
        )

    def issue_inner(
        self,
        *,
        issue_id: str,
        panel: MatrixResponsePhysicalPanel,
        checkpoint: MatrixResponsePostOuterCheckpoint,
        lower_law: ResponseLaw,
        lower_study: AdmissionControllerStudy,
        target_constitution_id: str,
        anonymous_mode_index: int,
    ) -> MatrixResponseInnerChildIssue:
        if target_constitution_id not in self.topology.inner_constitution_ids:
            raise ValueError("Six-matrix response inner child is not applicable to this constitution")
        if anonymous_mode_index not in range(self.topology.inner_modes_per_constitution):
            raise ValueError("Six-matrix response inner child is not applicable to this anonymous mode")
        return issue_matrix_response_study_inner_child(
            issue_id=issue_id,
            panel=panel,
            checkpoint=checkpoint,
            lower_law=lower_law,
            lower_study=lower_study,
            target_constitution_id=target_constitution_id,
            anonymous_mode_index=anonymous_mode_index,
        )

    def activate_inner(
        self,
        *,
        child_id: str,
        issue: MatrixResponseInnerChildIssue,
        panel: MatrixResponsePhysicalPanel,
        checkpoint: MatrixResponsePostOuterCheckpoint,
        lower_law: ResponseLaw,
        lower_study: AdmissionControllerStudy,
    ) -> MatrixResponseInnerActionHoldChild:
        return activate_matrix_response_study_inner_child(
            child_id=child_id,
            issue=issue,
            panel=panel,
            checkpoint=checkpoint,
            lower_law=lower_law,
            lower_study=lower_study,
        )


__all__ = [
    'MatrixResponseInnerActionHoldChild',
    'MatrixResponseInnerChildIssue',
    'MatrixResponseOuterArmPlan',
    'MatrixResponseOuterProspectiveUmbrella',
    'MatrixResponsePhysicalPanel',
    'MatrixResponsePostOuterCheckpoint',
    'MatrixResponseProspectiveEvaluationCoordinator',
    'activate_matrix_response_study_inner_child',
    'build_matrix_response_study_outer_arm_plan',
    'coordinate_matrix_response_study_outer_prospective',
    'issue_matrix_response_study_inner_child',
    'seal_matrix_response_study_post_outer_checkpoint',
]
