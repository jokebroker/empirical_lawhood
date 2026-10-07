"""Complete-family exploratory execution, skeptic checks and hypothesis synthesis."""

from __future__ import annotations

from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    EvidenceSnapshot,
    ObjectIdentity,
)
from empirical_lawhood.planning.discovery import (
    HypothesisSynthesis,
    SkepticCheck,
    SkepticCheckKind,
    SkepticReport,
    TemplateInstantiation,
)
from empirical_lawhood.planning.exploration import (
    AnalysisAttempt,
    AnalysisAttemptStatus,
    AnalysisProposal,
    AnalysisSpec,
    EffectEstimate,
    ExplorationPlan,
    ExploratoryFinding,
    ExploratoryFindingStatus,
    Hypothesis,
    HypothesisDisposition,
    HypothesisSet,
    ProposalDisposition,
)
from empirical_lawhood.runtime.exploration import (
    AnalysisCoordinateObservation as AnalysisCoordinateObservation,
    ExplorationWaveInput as ExplorationWaveInput,
    ExplorationWaveResult as ExplorationWaveResult,
    compose_finding_status,
)


def _selected_proposal_ids(plan: ExplorationPlan) -> tuple[str, ...]:
    return tuple(
        selection.proposal_id
        for selection in plan.selections
        if selection.disposition is ProposalDisposition.SELECTED
    )


def _classify_attempt(observation: AnalysisCoordinateObservation) -> AnalysisAttemptStatus:
    if observation.operational_failure_reason_codes:
        return AnalysisAttemptStatus.FAILED
    if not observation.evaluable:
        return AnalysisAttemptStatus.UNEVALUABLE
    if observation.is_pattern:
        return AnalysisAttemptStatus.SUCCEEDED
    return AnalysisAttemptStatus.NULL


def _attempt_and_effect(
    proposal_id: str,
    analysis_id: str,
    member_id: str,
    observation: AnalysisCoordinateObservation | None,
) -> tuple[AnalysisAttempt, EffectEstimate | None]:
    if observation is None:
        return (
            AnalysisAttempt(
                attempt_id=f"attempt.{analysis_id}.{member_id}",
                proposal_id=proposal_id,
                analysis_id=analysis_id,
                family_coordinate_ids=(member_id,),
                status=AnalysisAttemptStatus.UNEVALUABLE,
                artifact_ids=(),
                reason_codes=("registered-family-coordinate-missing",),
            ),
            None,
        )
    status = _classify_attempt(observation)
    status_details = {
        AnalysisAttemptStatus.SUCCEEDED: (
            (f"analysis-result.{analysis_id}.{member_id}",),
            (),
        ),
        AnalysisAttemptStatus.NULL: ((), ("descriptive-null",)),
        AnalysisAttemptStatus.UNEVALUABLE: ((), ("coordinate-unevaluable",)),
        AnalysisAttemptStatus.FAILED: ((), observation.operational_failure_reason_codes),
    }
    artifact_ids, reason_codes = status_details[status]
    attempt = AnalysisAttempt(
        attempt_id=f"attempt.{analysis_id}.{member_id}",
        proposal_id=proposal_id,
        analysis_id=analysis_id,
        family_coordinate_ids=(member_id,),
        status=status,
        artifact_ids=artifact_ids,
        reason_codes=reason_codes,
    )
    if status not in {AnalysisAttemptStatus.SUCCEEDED, AnalysisAttemptStatus.NULL}:
        return attempt, None
    return (
        attempt,
        EffectEstimate(
            effect_id=f"effect.{analysis_id}.{member_id}",
            quantity_id=observation.effect_quantity_id,
            native_unit=observation.native_unit,
            point=observation.point,
            lower=observation.lower,
            upper=observation.upper,
            physical_independent_unit_count=observation.physical_independent_unit_count,
        ),
    )


def _finding_link(snapshot: EvidenceSnapshot, analysis: AnalysisSpec) -> EvidenceLink:
    return EvidenceLink(
        link_id=f"evidence.{analysis.analysis_id}",
        relation=EvidenceRelation.DERIVED_FROM,
        source=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        target=ObjectIdentity.from_record(analysis.analysis_id, analysis),
        artifact_ids=tuple(artifact.artifact_id for artifact in snapshot.artifacts),
        world_id=snapshot.world_id,
        information_cutoff_id=snapshot.information_cutoff.cutoff_id,
        outcome_access=snapshot.outcome_access,
        visibility_ceiling=snapshot.visibility_ceiling,
        parent_visibility_ceilings=(snapshot.visibility_ceiling,),
        reason="Read-only registered analysis of an immutable outcome-visible snapshot.",
    )


def _build_finding(
    snapshot: EvidenceSnapshot,
    plan: ExplorationPlan,
    proposal: AnalysisProposal,
    attempts: tuple[AnalysisAttempt, ...],
    effects: tuple[EffectEstimate, ...],
) -> ExploratoryFinding:
    status = compose_finding_status(attempts)
    limitations = {
        "outcome-visible-non-promotable",
        *(reason for attempt in attempts for reason in attempt.reason_codes),
    }
    return ExploratoryFinding(
        finding_id=f"finding.{proposal.analysis.analysis_id}",
        plan_id=plan.plan_id,
        proposal_id=proposal.proposal_id,
        status=status,
        attempts=attempts,
        effects=effects,
        interpretation=(
            "Complete registered family retained; interpretation is descriptive "
            f"and terminal status is {status.value}."
        ),
        limitation_codes=tuple(sorted(limitations)),
        evidence_links=(_finding_link(snapshot, proposal.analysis),),
        outcome_access=snapshot.outcome_access,
        parent_visibility_ceiling=snapshot.visibility_ceiling,
        visibility_ceiling=snapshot.visibility_ceiling,
    )


def _execute_proposal(
    snapshot: EvidenceSnapshot,
    plan: ExplorationPlan,
    proposal: AnalysisProposal,
    instantiation: TemplateInstantiation,
    observation_by_key: dict[tuple[str, str], AnalysisCoordinateObservation],
) -> tuple[tuple[AnalysisAttempt, ...], ExploratoryFinding, set[tuple[str, str]]]:
    if not instantiation.ready:
        raise ValueError("selected proposal lacks a ready template instantiation")
    attempts: list[AnalysisAttempt] = []
    effects: list[EffectEstimate] = []
    consumed: set[tuple[str, str]] = set()
    analysis_id = proposal.analysis.analysis_id
    for member_id in instantiation.obligations.registered_family_member_ids:
        key = (analysis_id, member_id)
        observation = observation_by_key.get(key)
        if observation is not None:
            consumed.add(key)
        attempt, effect = _attempt_and_effect(
            proposal.proposal_id,
            analysis_id,
            member_id,
            observation,
        )
        attempts.append(attempt)
        if effect is not None:
            effects.append(effect)
    sorted_attempts = tuple(sorted(attempts, key=lambda value: value.attempt_id))
    finding = _build_finding(
        snapshot,
        plan,
        proposal,
        sorted_attempts,
        tuple(sorted(effects, key=lambda value: value.effect_id)),
    )
    return sorted_attempts, finding, consumed


class RegisteredAnalysisExecutor:
    """Execute complete declared coordinate summaries; no imports or code strings."""

    executor_key = "exploration.registered-summary-executor"
    executor_version = "1.0.0"

    def execute_proposal(
        self,
        snapshot: EvidenceSnapshot,
        plan: ExplorationPlan,
        instantiation: TemplateInstantiation,
        observations: tuple[AnalysisCoordinateObservation, ...],
    ) -> tuple[tuple[AnalysisAttempt, ...], ExploratoryFinding]:
        """Execute one selected proposal without constructing a reduced plan."""

        if plan.snapshot != ObjectIdentity.from_record(snapshot.snapshot_id, snapshot):
            raise ValueError("exploration executor snapshot differs from frozen plan")
        selected = set(_selected_proposal_ids(plan))
        proposal = instantiation.proposal
        if proposal.proposal_id not in selected:
            raise ValueError("analysis task received an unselected proposal")
        frozen = {value.proposal_id: value for value in plan.proposals}.get(proposal.proposal_id)
        if frozen != proposal:
            raise ValueError("analysis task proposal differs from the frozen plan")
        observation_by_key = {
            (value.analysis_id, value.family_member_id): value for value in observations
        }
        if len(observation_by_key) != len(observations):
            raise ValueError("exploration coordinate observations are duplicated")
        attempts, finding, consumed = _execute_proposal(
            snapshot,
            plan,
            proposal,
            instantiation,
            observation_by_key,
        )
        if set(observation_by_key) != consumed:
            raise ValueError("analysis task received foreign family coordinates")
        return attempts, finding

    def execute(
        self,
        snapshot: EvidenceSnapshot,
        plan: ExplorationPlan,
        instantiations: tuple[TemplateInstantiation, ...],
        observations: tuple[AnalysisCoordinateObservation, ...],
    ) -> tuple[tuple[AnalysisAttempt, ...], tuple[ExploratoryFinding, ...]]:
        if plan.snapshot != ObjectIdentity.from_record(snapshot.snapshot_id, snapshot):
            raise ValueError("exploration executor snapshot differs from frozen plan")
        instantiation_by_proposal = {value.proposal.proposal_id: value for value in instantiations}
        observation_by_key = {
            (value.analysis_id, value.family_member_id): value for value in observations
        }
        if len(observation_by_key) != len(observations):
            raise ValueError("exploration coordinate observations are duplicated")
        selected = _selected_proposal_ids(plan)
        attempts: list[AnalysisAttempt] = []
        findings: list[ExploratoryFinding] = []
        consumed: set[tuple[str, str]] = set()
        proposals = {proposal.proposal_id: proposal for proposal in plan.proposals}
        for proposal_id in selected:
            proposal = proposals[proposal_id]
            instantiation = instantiation_by_proposal.get(proposal_id)
            if instantiation is None:
                raise ValueError("selected proposal lacks a template instantiation")
            proposal_attempts, finding, proposal_consumed = _execute_proposal(
                snapshot,
                plan,
                proposal,
                instantiation,
                observation_by_key,
            )
            attempts.extend(proposal_attempts)
            findings.append(finding)
            consumed.update(proposal_consumed)
        unused = set(observation_by_key) - consumed
        if unused:
            raise ValueError("exploration inputs add undeclared or unselected family coordinates")
        return (
            tuple(sorted(attempts, key=lambda value: value.attempt_id)),
            tuple(sorted(findings, key=lambda value: value.finding_id)),
        )


def _skeptic_check(
    kind: SkepticCheckKind,
    passed: bool,
    failure_code: str,
    observation: str,
) -> SkepticCheck:
    return SkepticCheck(
        check_id=f"skeptic.{kind.value.lower().replace('_', '-')}",
        kind=kind,
        passed=passed,
        reason_codes=() if passed else (failure_code,),
        observation=observation,
    )


class AutomaticSkeptic:
    skeptic_key = "exploration.automatic-skeptic"
    skeptic_version = "1.0.0"

    def evaluate(
        self,
        plan: ExplorationPlan,
        instantiations: tuple[TemplateInstantiation, ...],
        observations: tuple[AnalysisCoordinateObservation, ...],
        findings: tuple[ExploratoryFinding, ...],
    ) -> SkepticReport:
        selected = set(_selected_proposal_ids(plan))
        selected_instantiations = tuple(
            value for value in instantiations if value.proposal.proposal_id in selected
        )
        expected = {
            (value.proposal.analysis.analysis_id, member)
            for value in selected_instantiations
            for member in value.obligations.registered_family_member_ids
        }
        observed = {(value.analysis_id, value.family_member_id) for value in observations}
        relevant = tuple(
            value
            for value in observations
            if (value.analysis_id, value.family_member_id) in expected
        )
        complete = expected == observed
        mechanical = complete and all(
            not value.operational_failure_reason_codes for value in relevant
        )
        alternatives = all(
            len(value.obligations.competing_explanation_ids) >= 2
            for value in selected_instantiations
        )
        matched_nulls = bool(relevant) and all(value.matched_null_passed for value in relevant)
        influence = bool(relevant) and all(
            value.leave_one_unit_max_change <= value.maximum_influence for value in relevant
        )
        scaling = bool(relevant) and all(value.scaling_control_passed for value in relevant)
        aggregation = bool(relevant) and all(value.aggregation_control_passed for value in relevant)
        leakage = bool(relevant) and all(value.leakage_control_passed for value in relevant)
        selection = bool(relevant) and all(value.selection_control_passed for value in relevant)
        checks = (
            _skeptic_check(
                SkepticCheckKind.AGGREGATION_ARTIFACT,
                aggregation,
                "aggregation-artifact-not-excluded",
                "All registered aggregation sensitivities retain their declared interpretation.",
            ),
            _skeptic_check(
                SkepticCheckKind.ALTERNATIVE_EXPLANATIONS,
                alternatives,
                "competing-explanations-missing",
                "Every selected analysis retains at least two competing explanations.",
            ),
            _skeptic_check(
                SkepticCheckKind.FAMILY_COMPLETENESS,
                complete,
                "registered-family-incomplete",
                "Observed coordinates exactly match the frozen Cartesian search family.",
            ),
            _skeptic_check(
                SkepticCheckKind.LEAKAGE,
                leakage,
                "causal-leakage-not-excluded",
                "Causal-cutoff and outcome-access leakage controls pass for every coordinate.",
            ),
            _skeptic_check(
                SkepticCheckKind.LEAVE_ONE_UNIT_INFLUENCE,
                influence,
                "single-unit-influence-too-large",
                "Leave-one-physical-unit changes remain below their declared bounds.",
            ),
            _skeptic_check(
                SkepticCheckKind.MATCHED_NULLS,
                matched_nulls,
                "matched-null-failed",
                "Matched null and falsifier construction passes for every evaluable coordinate.",
            ),
            _skeptic_check(
                SkepticCheckKind.MECHANICAL_CONSTRUCTION,
                mechanical,
                "mechanical-construction-failed",
                "Registered inputs, methods and outputs are complete and mechanically valid.",
            ),
            _skeptic_check(
                SkepticCheckKind.SCALING_ARTIFACT,
                scaling,
                "scaling-artifact-not-excluded",
                "Native-unit and normalized secondary views do not manufacture the observation.",
            ),
            _skeptic_check(
                SkepticCheckKind.SELECTION_ARTIFACT,
                selection,
                "selection-artifact-not-excluded",
                "The complete frozen family, including nulls and failures, remains visible.",
            ),
        )
        return SkepticReport(
            report_id=f"skeptic-report.{plan.plan_id}",
            plan=ObjectIdentity.from_record(plan.plan_id, plan),
            finding_ids=tuple(finding.finding_id for finding in findings),
            checks=tuple(sorted(checks, key=lambda check: check.check_id)),
            outcome_access=plan.outcome_access,
            parent_visibility_ceiling=plan.visibility_ceiling,
            visibility_ceiling=plan.visibility_ceiling,
        )


class HypothesisSynthesizer:
    synthesizer_key = "exploration.hypothesis-synthesis"
    synthesizer_version = "1.0.0"

    def synthesize(
        self,
        plan: ExplorationPlan,
        findings: tuple[ExploratoryFinding, ...],
        skeptic_report: SkepticReport,
    ) -> HypothesisSynthesis:
        if skeptic_report.plan != ObjectIdentity.from_record(plan.plan_id, plan):
            raise ValueError("hypothesis synthesizer received a foreign skeptic report")
        finding_ids = tuple(finding.finding_id for finding in findings)
        statuses = {finding.status for finding in findings}
        # A heterogeneous family is explicitly MIXED.  It may motivate a fresh
        # discriminating test, but it cannot stand in for a receipt-bound
        # finding whose whole registered family composed to PATTERN.
        has_pattern = ExploratoryFindingStatus.PATTERN in statuses
        all_null = statuses == {ExploratoryFindingStatus.NULL}
        if skeptic_report.passed and has_pattern:
            primary_disposition = HypothesisDisposition.PREFERRED
        elif all_null:
            primary_disposition = HypothesisDisposition.OPPOSED
        else:
            primary_disposition = HypothesisDisposition.UNRESOLVED
        hypotheses = (
            Hypothesis(
                hypothesis_id=f"hypothesis.{plan.plan_id}.declared-response",
                statement=(
                    "A relation-specific response or closure structure explains the retained "
                    "outcome-visible pattern."
                ),
                disposition=primary_disposition,
                supporting_finding_ids=finding_ids,
                opposing_observation=(
                    "The pattern disappears under fresh independent units, matched nulls or the "
                    "predeclared decisive falsifier."
                ),
                missing_evidence=(
                    "A separately authorized fresh test with frozen coordinates and falsifiers."
                ),
            ),
            Hypothesis(
                hypothesis_id=f"hypothesis.{plan.plan_id}.measurement-observation",
                statement=(
                    "Measurement, observer or delivery construction creates the apparent pattern."
                ),
                disposition=HypothesisDisposition.UNRESOLVED,
                supporting_finding_ids=finding_ids,
                opposing_observation=(
                    "The pattern recurs under an independently calibrated observation operator."
                ),
                missing_evidence="Independent observation or delivery validation units.",
            ),
            Hypothesis(
                hypothesis_id=f"hypothesis.{plan.plan_id}.selection-scaling-aggregation",
                statement=(
                    "Selection, scaling, aggregation or a high-influence unit creates the pattern."
                ),
                disposition=HypothesisDisposition.UNRESOLVED,
                supporting_finding_ids=finding_ids,
                opposing_observation=(
                    "The complete predeclared family recurs under native units and leave-one-unit "
                    "influence bounds."
                ),
                missing_evidence="Fresh family-complete replication at the physical-unit level.",
            ),
        )
        hypothesis_set = HypothesisSet(
            hypothesis_set_id=f"hypothesis-set.{plan.plan_id}",
            finding_ids=finding_ids,
            hypotheses=tuple(sorted(hypotheses, key=lambda value: value.hypothesis_id)),
            outcome_access=plan.outcome_access,
            parent_visibility_ceilings=tuple(finding.visibility_ceiling for finding in findings),
            visibility_ceiling=plan.visibility_ceiling,
        )
        failed_skeptic_codes = {
            reason for check in skeptic_report.checks for reason in check.reason_codes
        }
        limitation_codes = {
            "outcome-visible-non-promotable",
            *failed_skeptic_codes,
            *(code for finding in findings for code in finding.limitation_codes),
        }
        return HypothesisSynthesis(
            synthesis_id=f"hypothesis-synthesis.{plan.plan_id}",
            hypothesis_set=hypothesis_set,
            skeptic_report=ObjectIdentity.from_record(
                skeptic_report.report_id,
                skeptic_report,
            ),
            influence_diagnostic_ids=tuple(
                sorted(
                    check.check_id
                    for check in skeptic_report.checks
                    if check.kind is SkepticCheckKind.LEAVE_ONE_UNIT_INFLUENCE
                )
            ),
            limitation_codes=tuple(sorted(limitation_codes)),
            prospective_evidence_need_ids=(
                "fresh-independent-units",
                "frozen-decisive-falsifiers",
                "separate-prospective-authorization",
            ),
        )


def execute_wave(
    *,
    snapshot: EvidenceSnapshot,
    plan: ExplorationPlan,
    instantiations: tuple[TemplateInstantiation, ...],
    observations: tuple[AnalysisCoordinateObservation, ...],
) -> ExplorationWaveResult:
    attempts, findings = RegisteredAnalysisExecutor().execute(
        snapshot,
        plan,
        instantiations,
        observations,
    )
    skeptic = AutomaticSkeptic().evaluate(plan, instantiations, observations, findings)
    synthesis = HypothesisSynthesizer().synthesize(plan, findings, skeptic)
    return ExplorationWaveResult(
        result_id=f"wave-result.{plan.plan_id}",
        plan=ObjectIdentity.from_record(plan.plan_id, plan),
        attempts=attempts,
        findings=findings,
        skeptic_report=skeptic,
        hypothesis_synthesis=synthesis,
    )
