"""Registered fresh-evidence experiment designers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import ClaimSpec, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec, RevealBarrierSpec
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierKind,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import ProspectiveNomination
from empirical_lawhood.planning.prospective import (
    ExperimentDesignAudit,
    NominationObligations,
    ProspectiveDesignKind,
    experiment_design_audit_extension,
)


@dataclass(frozen=True, slots=True)
class DesignedExperiment:
    proposal: ExperimentProposal
    audit: ExperimentDesignAudit


class ProspectiveExperimentDesigner(Protocol):
    capability_key: ClassVar[str]
    capability_version: ClassVar[str]
    design_kind: ClassVar[ProspectiveDesignKind]

    def design(
        self,
        *,
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        system: SystemSpec,
    ) -> DesignedExperiment: ...


class _BaseDesigner:
    capability_version: ClassVar[str] = "1.0.0"
    design_kind: ClassVar[ProspectiveDesignKind]
    capability_key: ClassVar[str]

    def design(
        self,
        *,
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        system: SystemSpec,
    ) -> DesignedExperiment:
        self._validate(nomination, obligations, system)
        slug = self.design_kind.value.lower().replace("_", "-")
        experiment_id = f"experiment.{nomination.nomination_id}.{slug}"
        proposal_id = f"experiment-proposal.{nomination.nomination_id}.{slug}"
        candidate = ExperimentSpec(
            experiment_id=experiment_id,
            system_id=system.system_id,
            world_id=system.world.world_id,
            relation=system.relation,
            independent_unit_id=system.independent_unit.unit_id,
            claims=(self._claim(nomination, obligations, system, slug),),
            assignment=nomination.assignment,
            measurement_quantity_ids=nomination.measurement_quantity_ids,
            controls=tuple(
                sorted(
                    (requirement.control for requirement in obligations.controls),
                    key=lambda control: control.control_id,
                )
            ),
            precision_goals=nomination.precision_goals,
            information_cutoffs=(obligations.information_cutoff,),
            reveal_barrier=RevealBarrierSpec(
                barrier_id=f"reveal-barrier.{nomination.nomination_id}.{slug}",
                development_unit_ids=obligations.evidence_contract.development_unit_ids,
                evaluation_cohort_id=obligations.evidence_contract.evaluation_cohort_id,
                evaluation_manifest_sha256=(
                    obligations.evidence_contract.evaluation_manifest_sha256
                ),
                sealed_outcome_artifact_ids=(
                    obligations.evidence_contract.sealed_outcome_artifact_ids
                ),
                evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
                fresh_evidence=True,
            ),
            obligations=self._scientific_obligations(nomination, obligations, system, slug),
            design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            authority_policy_id=system.authority_policy.policy_id,
            readiness=ReadinessStatus.AUTHORITY_REQUIRED,
        )
        audit = ExperimentDesignAudit(
            audit_id=f"experiment-design-audit.{nomination.nomination_id}.{slug}",
            proposal_id=proposal_id,
            nomination=ObjectIdentity.from_record(nomination.nomination_id, nomination),
            design_kind=self.design_kind,
            objective=obligations.objective,
            alternative_design_ids=obligations.alternative_design_ids,
            control_requirement_ids=tuple(
                requirement.requirement_id for requirement in obligations.controls
            ),
            safety_constraint_ids=obligations.safety_constraint_ids,
            expected_falsification_value=obligations.objective.falsification_value,
            acquisition_value=self._acquisition_value(),
            stop_or_fallback_rule=(
                "Stop without acquisition when authority, safety, support or precision gates "
                "cannot be satisfied; never substitute a cheaper estimand."
            ),
            decision_cutoff=obligations.information_cutoff,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        proposal = ExperimentProposal(
            proposal_id=proposal_id,
            nomination=ObjectIdentity.from_record(nomination.nomination_id, nomination),
            candidate_experiment=candidate,
            alternative_design_ids=obligations.alternative_design_ids,
            expected_discrimination=self._expected_discrimination(),
            risk_codes=obligations.safety_constraint_ids,
            budget=obligations.budget,
            decision_cutoff=obligations.information_cutoff,
            proposed_by="prospective-design-worker",
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            parent_visibility_ceiling=nomination.visibility_ceiling,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            extensions=(experiment_design_audit_extension(audit),),
        )
        return DesignedExperiment(proposal=proposal, audit=audit)

    def _validate(
        self,
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        system: SystemSpec,
    ) -> None:
        if obligations.design_kind is not self.design_kind:
            raise ValueError("designer received another design family's nomination")
        if obligations.nomination_id != nomination.nomination_id:
            raise ValueError("prospective obligations bind another nomination")
        binding = next(
            (
                item
                for item in nomination.extensions
                if item.namespace == "prospective-nomination-obligations"
            ),
            None,
        )
        if (
            binding is None
            or binding.schema != obligations.SCHEMA
            or binding.payload_sha256 != obligations.fingerprint()
        ):
            raise ValueError("nomination does not bind its complete prospective obligations")
        if nomination.relation != system.relation:
            raise ValueError("prospective designer received a different system relation")
        if nomination.assignment.independent_unit_id != system.independent_unit.unit_id:
            raise ValueError("prospective design changes the physical independent-unit type")
        if nomination.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("nomination source visibility must remain outcome-visible")

    @staticmethod
    def _claim(
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        system: SystemSpec,
        slug: str,
    ) -> ClaimSpec:
        claim = ClaimSpec(
            claim_id=f"fresh-claim.{nomination.nomination_id}.{slug}",
            world_id=system.world.world_id,
            relation_id=system.relation.relation_id,
            proposition=obligations.claim_proposition,
            estimand=obligations.claim_estimand,
            physical_independent_unit_id=system.independent_unit.unit_id,
            requested_rung=obligations.requested_rung,
            evidence_ceiling=obligations.evidence_ceiling,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            promotion_rule=obligations.claim_promotion_rule,
            assumption_ids=obligations.claim_assumption_ids,
            derivation_parent_ids=(),
            parent_visibility_ceilings=(),
            numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
        )
        system.validate_claim(claim)
        return claim

    @staticmethod
    def _scientific_obligations(
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        system: SystemSpec,
        slug: str,
    ) -> ScientificObligations:
        prefix = f"prospective.{nomination.nomination_id}.{slug}"
        falsifiers = tuple(
            FalsifierSpec(
                falsifier_id=f"falsifier.{falsifier_id}",
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key=f"prospective-falsifier.{falsifier_id}",
                description="Predeclared decisive fresh-evidence falsifier.",
                decisive_rule=(
                    "Fail the requested claim when this falsifier does not clear its frozen rule."
                ),
                status=ObligationStatus.REQUIRED,
            )
            for falsifier_id in nomination.falsifier_ids
        )
        return ScientificObligations(
            obligations_id=f"scientific-obligations.{prefix}",
            support=SupportSpec(
                support_id=f"support.{prefix}",
                relation_id=system.relation.relation_id,
                independent_unit_id=system.independent_unit.unit_id,
                physical_unit_count=len(obligations.evidence_contract.fresh_independent_unit_ids),
                nested_numerical_view_count=len(system.numerical_views),
                information_cutoff_id=obligations.information_cutoff.cutoff_id,
                chart_ids=(f"native-chart.{system.relation.relation_id}",),
                denominator_cell_ids=obligations.denominator_cell_ids,
                action_bounds=obligations.action_bounds,
                status=ObligationStatus.REQUIRED,
            ),
            validity=ValiditySpec(
                validity_id=f"validity.{prefix}",
                validity_domain_ids=(system.system_id,),
                assumption_ids=obligations.claim_assumption_ids,
                exclusion_reason_codes=(),
                status=ObligationStatus.REQUIRED,
            ),
            uncertainty=UncertaintySpec(
                uncertainty_id=f"uncertainty.{prefix}",
                method_key="independent-unit-interval",
                independent_unit_id=system.independent_unit.unit_id,
                confidence_level=Decimal("0.95"),
                interval_quantity_ids=system.relation.receiver_quantity_ids,
                limitation_codes=(),
                status=ObligationStatus.REQUIRED,
            ),
            falsifiers=falsifiers,
            closure=ClosureSpec(
                closure_id=f"closure.{prefix}",
                recurrence_cell_ids=obligations.denominator_cell_ids,
                exchange_factor_ids=system.relation.denominator_quantity_ids,
                retained_history_ids=system.relation.history_quantity_ids,
                status=ObligationStatus.REQUIRED,
            ),
            structural_convergence=StructuralConvergenceSpec(
                convergence_id=f"convergence.{prefix}",
                required_structure_ids=("response-direction", "response-rank"),
                numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
                tolerances=(),
                status=ObligationStatus.REQUIRED,
            ),
            computability=ComputabilityEvidence(
                computability_id=f"computability.{prefix}",
                envelope_id=(
                    system.computability_envelopes[0].envelope_id
                    if system.computability_envelopes
                    else "prospective-measurement-envelope"
                ),
                numerical_view_ids=(
                    tuple(view.view_id for view in system.numerical_views)
                    or ("physical-measurement-view",)
                ),
                readiness=ReadinessStatus.PREREQUISITE_NOT_MET,
                unresolved_reason_codes=("fresh-execution-not-yet-run",),
            ),
        )

    def _expected_discrimination(self) -> str:
        return (
            f"The {self.design_kind.value.lower()} design compares all frozen hypotheses "
            "through their distinct predeclared fresh-evidence predictions."
        )

    def _acquisition_value(self) -> str:
        return (
            f"Acquire only if the {self.design_kind.value.lower()} design can falsify a model, "
            "closure, support or transport alternative; disagreement is retained, not averaged."
        )


class SimpleFactorialDesigner(_BaseDesigner):
    capability_key = "prospective.designer.simple-factorial"
    design_kind = ProspectiveDesignKind.SIMPLE_FACTORIAL


class CoverageExpansionDesigner(_BaseDesigner):
    capability_key = "prospective.designer.coverage-expansion"
    design_kind = ProspectiveDesignKind.COVERAGE_EXPANSION


class InformationRankDesigner(_BaseDesigner):
    capability_key = "prospective.designer.information-rank"
    design_kind = ProspectiveDesignKind.INFORMATION_RANK


class ModelDiscriminationDesigner(_BaseDesigner):
    capability_key = "prospective.designer.model-discrimination"
    design_kind = ProspectiveDesignKind.MODEL_DISCRIMINATION


@dataclass(frozen=True, slots=True)
class ProspectiveDesignerRegistry:
    designers: tuple[ProspectiveExperimentDesigner, ...]

    def __post_init__(self) -> None:
        kinds = tuple(designer.design_kind for designer in self.designers)
        keys = tuple(designer.capability_key for designer in self.designers)
        if len(set(kinds)) != len(kinds) or len(set(keys)) != len(keys):
            raise ValueError("prospective designer registry contains duplicates")
        if set(kinds) != set(ProspectiveDesignKind):
            raise ValueError("prospective registry must contain the exact four design families")

    def resolve(self, kind: ProspectiveDesignKind) -> ProspectiveExperimentDesigner:
        for designer in self.designers:
            if designer.design_kind is kind:
                return designer
        raise KeyError(kind)


def default_designer_registry() -> ProspectiveDesignerRegistry:
    return ProspectiveDesignerRegistry(
        designers=(
            CoverageExpansionDesigner(),
            InformationRankDesigner(),
            ModelDiscriminationDesigner(),
            SimpleFactorialDesigner(),
        )
    )
