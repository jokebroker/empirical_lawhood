"""Inward execution records and closure checks for exploration waves."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.authority import ApprovalRequest
from empirical_lawhood.planning.discovery import (
    HypothesisSynthesis,
    SkepticReport,
    TemplateInstantiation,
)
from empirical_lawhood.planning.exploration import (
    AnalysisAttempt,
    AnalysisAttemptStatus,
    ExplorationPlan,
    ExploratoryFinding,
    ExploratoryFindingStatus,
    HypothesisDisposition,
    ProposalDisposition,
)
from empirical_lawhood.planning.prospective import ProspectiveDesignContext

from .capabilities import CapabilityPermission, CapabilityRegistry
from .plans import SnapshotVerification


@dataclass(frozen=True, slots=True)
class AnalysisCoordinateObservation(CanonicalRecord):
    """Bounded summary input for one pre-registered search-family coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/analysis-coordinate-observation'

    observation_id: str
    analysis_id: str
    family_member_id: str
    effect_quantity_id: str
    native_unit: str
    point: Decimal
    lower: Decimal
    upper: Decimal
    minimum_absolute_pattern: Decimal
    physical_independent_unit_count: int
    evaluable: bool
    operational_failure_reason_codes: tuple[str, ...]
    matched_null_passed: bool
    scaling_control_passed: bool
    aggregation_control_passed: bool
    leakage_control_passed: bool
    selection_control_passed: bool
    leave_one_unit_max_change: Decimal
    maximum_influence: Decimal

    def __post_init__(self) -> None:
        for name, identifier in (
            ("observation_id", self.observation_id),
            ("analysis_id", self.analysis_id),
            ("family_member_id", self.family_member_id),
            ("effect_quantity_id", self.effect_quantity_id),
        ):
            validate_stable_id(identifier, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, value in (
            ("point", self.point),
            ("lower", self.lower),
            ("upper", self.upper),
            ("minimum_absolute_pattern", self.minimum_absolute_pattern),
            ("leave_one_unit_max_change", self.leave_one_unit_max_change),
            ("maximum_influence", self.maximum_influence),
        ):
            validate_decimal(value, field_name=name)
        if self.lower > self.point or self.point > self.upper:
            raise ValueError("coordinate effect point must lie inside its interval")
        if self.minimum_absolute_pattern < 0 or self.maximum_influence < 0:
            raise ValueError("pattern and influence bounds must be nonnegative")
        if self.leave_one_unit_max_change < 0:
            raise ValueError("leave-one-unit influence must be nonnegative")
        if self.physical_independent_unit_count <= 0:
            raise ValueError("analysis coordinate requires physical independent units")
        require_sorted_unique_strings(
            self.operational_failure_reason_codes,
            field_name="operational_failure_reason_codes",
        )
        if self.operational_failure_reason_codes and self.evaluable:
            raise ValueError("operationally failed coordinate cannot be evaluable")

    @property
    def is_pattern(self) -> bool:
        return (
            self.evaluable
            and not self.operational_failure_reason_codes
            and abs(self.point) >= self.minimum_absolute_pattern
            and not (self.lower <= 0 <= self.upper)
        )


@dataclass(frozen=True, slots=True)
class ExplorationWaveInput(CanonicalRecord):
    """Frozen, outcome-visible inputs for one executable exploration wave."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exploration-wave-input'

    wave_input_id: str
    snapshot: EvidenceSnapshot
    snapshot_verification: SnapshotVerification
    plan: ExplorationPlan
    instantiations: tuple[TemplateInstantiation, ...]
    observations: tuple[AnalysisCoordinateObservation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.wave_input_id, field_name="wave_input_id")
        snapshot_identity = ObjectIdentity.from_record(
            self.snapshot.snapshot_id,
            self.snapshot,
        )
        if self.snapshot_verification.snapshot != snapshot_identity:
            raise ValueError("wave input verification binds another evidence snapshot")
        if self.plan.snapshot != snapshot_identity:
            raise ValueError("wave input plan binds another evidence snapshot")
        require_sorted_unique_ids(
            self.instantiations,
            attribute="instantiation_id",
            field_name="instantiations",
        )
        require_sorted_unique_ids(
            self.observations,
            attribute="observation_id",
            field_name="observations",
        )
        selected = {
            selection.proposal_id
            for selection in self.plan.selections
            if selection.disposition is ProposalDisposition.SELECTED
        }
        instantiated = {value.proposal.proposal_id for value in self.instantiations}
        if instantiated != selected or len(instantiated) != len(self.instantiations):
            raise ValueError("wave input instantiations must cover exactly the selected proposals")
        proposals = {value.proposal_id: value for value in self.plan.proposals}
        for instantiation in self.instantiations:
            proposal_id = instantiation.proposal.proposal_id
            if instantiation.proposal != proposals[proposal_id]:
                raise ValueError("wave input instantiation changes its frozen proposal")
            if not instantiation.ready:
                raise ValueError("selected wave input instantiation must be ready")
        selected_analysis_ids = {
            proposals[proposal_id].analysis.analysis_id for proposal_id in selected
        }
        observation_keys = tuple(
            (value.analysis_id, value.family_member_id) for value in self.observations
        )
        if len(set(observation_keys)) != len(observation_keys):
            raise ValueError("wave input observations duplicate an analysis coordinate")
        if any(value.analysis_id not in selected_analysis_ids for value in self.observations):
            raise ValueError("wave input observes an undeclared or unselected analysis")
        declared_coordinates = {
            (
                instantiation.proposal.analysis.analysis_id,
                member_id,
            )
            for instantiation in self.instantiations
            for member_id in instantiation.obligations.registered_family_member_ids
        }
        if not set(observation_keys).issubset(declared_coordinates):
            raise ValueError("wave input observes an undeclared family coordinate")


@dataclass(frozen=True, slots=True)
class ExplorationExecutionPackage(CanonicalRecord):
    """Executable exploration root containing inputs but no expected outputs."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exploration-execution-package'

    package_id: str
    execution_plan_id: str
    implementation_commit: str
    system: SystemSpec
    snapshot: EvidenceSnapshot
    snapshot_verification: SnapshotVerification
    exploration_plan: ExplorationPlan
    registry: CapabilityRegistry
    wave_input: ExplorationWaveInput
    design_context: ProspectiveDesignContext
    approval_request: ApprovalRequest
    skeptic_capability_key: str
    skeptic_capability_version: str
    synthesis_capability_key: str
    synthesis_capability_version: str

    def __post_init__(self) -> None:
        for name, value in (
            ("package_id", self.package_id),
            ("execution_plan_id", self.execution_plan_id),
            ("skeptic_capability_key", self.skeptic_capability_key),
            ("synthesis_capability_key", self.synthesis_capability_key),
        ):
            validate_stable_id(value, field_name=name)
        for value in (
            self.skeptic_capability_version,
            self.synthesis_capability_version,
        ):
            validate_semantic_version(value)
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        if self.execution_plan_id != f"execution.{self.exploration_plan.plan_id}":
            raise ValueError("execution_plan_id must derive from the exploration plan")
        snapshot_identity = ObjectIdentity.from_record(
            self.snapshot.snapshot_id,
            self.snapshot,
        )
        if self.snapshot_verification.snapshot != snapshot_identity:
            raise ValueError("snapshot verification binds another evidence snapshot")
        if self.exploration_plan.snapshot != snapshot_identity:
            raise ValueError("exploration plan binds another evidence snapshot")
        if self.snapshot.world_id != self.system.world.world_id:
            raise ValueError("exploration snapshot binds another evidence world")
        if self.exploration_plan.outcome_access is not self.snapshot.outcome_access:
            raise ValueError("exploration plan changes snapshot outcome access")
        if any(
            proposal.analysis.system_id != self.system.system_id
            or proposal.analysis.relation != self.system.relation
            or proposal.analysis.independent_unit_id != self.system.independent_unit.unit_id
            for proposal in self.exploration_plan.proposals
        ):
            raise ValueError("exploration analysis changes the bound system relation")
        if (
            self.wave_input.snapshot != self.snapshot
            or self.wave_input.snapshot_verification != self.snapshot_verification
            or self.wave_input.plan != self.exploration_plan
        ):
            raise ValueError("runtime wave input differs from the package freeze")
        if self.design_context.source_snapshot != snapshot_identity:
            raise ValueError("prospective context binds another evidence snapshot")
        if self.design_context.system != ObjectIdentity.from_record(
            self.system.system_id,
            self.system,
        ):
            raise ValueError("prospective context binds another system")
        if self.approval_request.action != self.design_context.authority_action:
            raise ValueError("approval request changes the proposed authority action")
        if (
            self.snapshot.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.exploration_plan.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.design_context.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("exploration execution inputs must remain outcome-visible")
        if any(
            capability.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            for capability in self.registry.capabilities
        ):
            raise ValueError("exploration registry cannot contain promotion authority")
        forbidden = {
            CapabilityPermission.APPROVE_NONACTUATING,
            CapabilityPermission.COMMAND_ACTUATOR,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
            CapabilityPermission.WRITE_CATALOG,
        }
        if any(
            forbidden.intersection(capability.permissions)
            for capability in self.registry.capabilities
        ):
            raise ValueError("exploration registry contains forbidden authority")
        self.registry.resolve(
            self.skeptic_capability_key,
            self.skeptic_capability_version,
        )
        self.registry.resolve(
            self.synthesis_capability_key,
            self.synthesis_capability_version,
        )


def compose_finding_status(
    attempts: tuple[AnalysisAttempt, ...],
) -> ExploratoryFindingStatus:
    """Compose a finding without treating a heterogeneous family as a pattern."""

    statuses = {attempt.status for attempt in attempts}
    if statuses == {AnalysisAttemptStatus.SUCCEEDED}:
        return ExploratoryFindingStatus.PATTERN
    if statuses == {AnalysisAttemptStatus.NULL}:
        return ExploratoryFindingStatus.NULL
    if statuses == {AnalysisAttemptStatus.FAILED}:
        return ExploratoryFindingStatus.FAILED
    if statuses == {AnalysisAttemptStatus.BLOCKED}:
        return ExploratoryFindingStatus.BLOCKED
    if statuses == {AnalysisAttemptStatus.UNEVALUABLE}:
        return ExploratoryFindingStatus.UNEVALUABLE
    return ExploratoryFindingStatus.MIXED


@dataclass(frozen=True, slots=True)
class ExplorationWaveResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exploration-wave-result'

    result_id: str
    plan: ObjectIdentity
    attempts: tuple[AnalysisAttempt, ...]
    findings: tuple[ExploratoryFinding, ...]
    skeptic_report: SkepticReport
    hypothesis_synthesis: HypothesisSynthesis

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.attempts,
            attribute="attempt_id",
            field_name="attempts",
        )
        require_sorted_unique_ids(
            self.findings,
            attribute="finding_id",
            field_name="findings",
        )
        if not self.attempts or not self.findings:
            raise ValueError("wave result requires complete attempts and findings")
        flattened_attempts = tuple(
            sorted(
                (attempt for finding in self.findings for attempt in finding.attempts),
                key=lambda value: value.attempt_id,
            )
        )
        if flattened_attempts != self.attempts:
            raise ValueError("wave attempts differ from the finding-bound attempts")
        if any(
            finding.plan_id != self.plan.object_id
            or any(attempt.proposal_id != finding.proposal_id for attempt in finding.attempts)
            or finding.status is not compose_finding_status(finding.attempts)
            for finding in self.findings
        ):
            raise ValueError("wave finding identity or status composition is invalid")
        finding_ids = tuple(finding.finding_id for finding in self.findings)
        if self.skeptic_report.plan != self.plan:
            raise ValueError("wave skeptic report binds another exploration plan")
        if self.skeptic_report.finding_ids != finding_ids:
            raise ValueError("wave skeptic report binds another finding family")
        if self.hypothesis_synthesis.skeptic_report != ObjectIdentity.from_record(
            self.skeptic_report.report_id,
            self.skeptic_report,
        ):
            raise ValueError("wave hypothesis synthesis binds another skeptic report")
        hypothesis_set = self.hypothesis_synthesis.hypothesis_set
        if hypothesis_set.finding_ids != finding_ids or any(
            hypothesis.supporting_finding_ids != finding_ids
            for hypothesis in hypothesis_set.hypotheses
        ):
            raise ValueError("wave hypotheses do not retain the exact finding family")
        has_pattern = any(
            finding.status is ExploratoryFindingStatus.PATTERN for finding in self.findings
        )
        pattern_hypothesis_id = f"hypothesis.{self.plan.object_id}.declared-response"
        if not has_pattern and any(
            hypothesis.hypothesis_id == pattern_hypothesis_id
            and hypothesis.disposition is HypothesisDisposition.PREFERRED
            for hypothesis in hypothesis_set.hypotheses
        ):
            raise ValueError("a preferred wave hypothesis requires an explicit pattern finding")


def validate_exploration_wave_result(
    result: ExplorationWaveResult,
    wave_input: ExplorationWaveInput,
) -> None:
    """Validate complete input-to-attempt-to-hypothesis identity closure."""

    plan_identity = ObjectIdentity.from_record(wave_input.plan.plan_id, wave_input.plan)
    if result.result_id != f"wave-result.{wave_input.plan.plan_id}" or result.plan != plan_identity:
        raise ValueError("wave result binds another frozen exploration plan")
    if (
        result.skeptic_report.report_id != f"skeptic-report.{wave_input.plan.plan_id}"
        or result.hypothesis_synthesis.synthesis_id
        != f"hypothesis-synthesis.{wave_input.plan.plan_id}"
        or result.hypothesis_synthesis.hypothesis_set.hypothesis_set_id
        != f"hypothesis-set.{wave_input.plan.plan_id}"
    ):
        raise ValueError("wave synthesis identities differ from the frozen plan")
    if any(
        value != wave_input.plan.outcome_access
        for value in (
            *(finding.outcome_access for finding in result.findings),
            result.skeptic_report.outcome_access,
            result.hypothesis_synthesis.hypothesis_set.outcome_access,
        )
    ) or any(
        value != wave_input.plan.visibility_ceiling
        for value in (
            *(finding.visibility_ceiling for finding in result.findings),
            result.skeptic_report.visibility_ceiling,
            result.hypothesis_synthesis.hypothesis_set.visibility_ceiling,
        )
    ):
        raise ValueError("wave result changes outcome access or visibility")
    instantiations = {value.proposal.proposal_id: value for value in wave_input.instantiations}
    if {finding.proposal_id for finding in result.findings} != set(instantiations):
        raise ValueError("wave findings do not cover exactly the selected proposals")
    for finding in result.findings:
        instantiation = instantiations[finding.proposal_id]
        analysis_id = instantiation.proposal.analysis.analysis_id
        if (
            finding.finding_id != f"finding.{analysis_id}"
            or {attempt.analysis_id for attempt in finding.attempts} != {analysis_id}
            or len(finding.evidence_links) != 1
            or finding.evidence_links[0].source
            != ObjectIdentity.from_record(
                wave_input.snapshot.snapshot_id,
                wave_input.snapshot,
            )
            or finding.evidence_links[0].target
            != ObjectIdentity.from_record(
                analysis_id,
                instantiation.proposal.analysis,
            )
        ):
            raise ValueError("wave finding attempts or evidence bind another analysis")
        coordinates = tuple(
            coordinate
            for attempt in finding.attempts
            for coordinate in attempt.family_coordinate_ids
        )
        if tuple(sorted(coordinates)) != instantiation.obligations.registered_family_member_ids:
            raise ValueError("wave finding does not retain the complete registered family")
    expected_attempts = tuple(
        sorted(
            (attempt for finding in result.findings for attempt in finding.attempts),
            key=lambda value: value.attempt_id,
        )
    )
    if result.attempts != expected_attempts:
        raise ValueError("wave result attempt closure differs")
