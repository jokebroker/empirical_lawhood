"""Point-method orchestration through the sole qualification authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.identification import AdequacyCheckKind, AdequacyCheckResult, LawIdentificationResult, LawQualificationResult
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec

from .contracts import CandidateClaimTemplate, CandidateEvaluatorImplementation, CandidateFamilyLedger, CandidateFamilyMember, CandidateFit, CandidateMethodEvidenceReceipt, CandidateRosterDisposition, FalsifierObligationTemplate, IdentificationDataset, ComponentUncertaintyCandidateAssessment, LawCandidateAxisBinding, LawCandidateAxisMap, LawCandidateEvidence, LawIdentificationConfig, LawIdentifier, LawObligationTemplate, ObservationRole
from .law_assessment import CandidateFamilyAssembler, CandidatePayloadDecoderRegistry, CandidatePayloadPublisher, CandidatePayloadReader, CanonicalParametricLawModelDecoder, ResponseLawQualificationService, project_response_method
from .qualification_profiles import QualificationProofOwner, ResponseQualificationProfileEvaluator


def _reference_fit(system: SystemSpec, fits: tuple[CandidateFit, ...]) -> CandidateFit:
    levels = {
        view.view_id: {
            coordinate.coordinate_id: coordinate.refinement_level for coordinate in view.coordinates
        }
        for view in system.numerical_views
    }
    if not levels or any(
        set(value) != set(next(iter(levels.values()))) for value in levels.values()
    ):
        raise ValueError("law identification requires comparable numerical-view coordinates")
    dominant = tuple(
        view_id
        for view_id, coordinates in levels.items()
        if all(
            all(coordinates[key] >= other[key] for key in coordinates) for other in levels.values()
        )
    )
    if len(dominant) != 1:
        raise ValueError("law identification requires one uniquely finest numerical view")
    return next(fit for fit in fits if fit.model.numerical_view_id == dominant[0])


def _candidate_evaluator(fit: CandidateFit) -> ExecutableReference:
    payload_bytes = fit.model.canonical_bytes()
    payload = ArtifactIdentity(
        artifact_id=f"artifact.{fit.model.model_id}",
        role="law-model",
        payload_schema=fit.model.SCHEMA,
        sha256=fit.model.fingerprint(),
        media_type="application/json",
        size_bytes=len(payload_bytes),
    )
    return ExecutableReference(
        reference_id=f"evaluator.{fit.model.model_id}",
        capability_key=fit.model.method_key,
        capability_version=fit.model.method_version,
        evaluator_key="canonical-parametric-law",
        payload=payload,
        payload_format=SafePayloadFormat.CANONICAL_JSON,
        input_schema='empirical-lawhood/evaluation/named-law-input',
        output_schema='empirical-lawhood/evaluation/named-law-output',
        deterministic=True,
    )


def _evidence_link(
    system: SystemSpec,
    dataset: IdentificationDataset,
    evaluator: ExecutableReference,
) -> EvidenceLink:
    artifact_ids = tuple(
        sorted(
            {
                evaluator.payload.artifact_id,
                *(artifact.artifact_id for artifact in dataset.evidence_artifacts),
            }
        )
    )
    return EvidenceLink(
        link_id=f"evidence.{dataset.dataset_id}",
        relation=EvidenceRelation.DERIVED_FROM,
        source=ObjectIdentity.from_record(dataset.dataset_id, dataset),
        target=ObjectIdentity.from_record(system.relation.relation_id, system.relation),
        artifact_ids=artifact_ids,
        world_id=system.world.world_id,
        information_cutoff_id=dataset.information_cutoff_id,
        outcome_access=dataset.outcome_access,
        visibility_ceiling=dataset.visibility_ceiling,
        parent_visibility_ceilings=dataset.parent_visibility_ceilings,
        reason=(
            "The candidate assessment derives from the bound independent-unit evidence, "
            "safe canonical model and frozen information cutoff."
        ),
    )


def _response_method_scope(
    system: SystemSpec,
    dataset: IdentificationDataset,
    config: LawIdentificationConfig,
) -> tuple[QualificationScopeSpec, ClaimUnitBinding]:
    claim_id = f"claim.{dataset.dataset_id}.{config.method_kind.value.lower()}"
    scope = QualificationScopeSpec(
        scope_id=f"scope.{dataset.dataset_id}.{config.method_kind.value.lower()}",
        claim_id=claim_id,
        population_id=f"population.{dataset.dataset_id}",
        physical_unit_type_id=system.independent_unit.unit_id,
        independent_unit_instance_ids=dataset.physical_unit_instance_ids,
        nested_coordinate_ids=tuple(view.view_id for view in system.numerical_views),
        aggregation_level_id=system.independent_unit.unit_id,
        uncertainty_unit_id=system.independent_unit.unit_id,
        locality_scope_id=system.system_id,
        maximum_evidence_ceiling=(
            EvidenceCeiling.LOCAL_LAW
            if dataset.visibility_ceiling.is_promotable
            else EvidenceCeiling.NON_PROMOTABLE
        ),
    )
    binding = ClaimUnitBinding(
        binding_id=f"claim-unit.{dataset.dataset_id}.{config.method_kind.value.lower()}",
        claim_id=claim_id,
        scope=ObjectIdentity.from_record(scope.scope_id, scope),
        independent_unit_instance_ids=dataset.physical_unit_instance_ids,
        aggregation_level_id=system.independent_unit.unit_id,
    )
    return scope, binding


def _claim_template(
    dataset: IdentificationDataset,
    config: LawIdentificationConfig,
) -> CandidateClaimTemplate:
    suffix = config.method_kind.value.lower()
    return CandidateClaimTemplate(
        template_id=f"claim-template.{dataset.dataset_id}.{suffix}",
        terminal_result_id=f"qualification.{dataset.dataset_id}.{suffix}",
        compatibility_result_id=f"identification.{dataset.dataset_id}.{suffix}",
        claim_id=f"claim.{dataset.dataset_id}.{suffix}",
        law_id=f"law.{dataset.dataset_id}.{suffix}",
        proposition="The prepared denominator supports the declared local relational response law.",
        estimand=(
            "Finite-horizon receiver displacement under the native action chart, retained "
            "history and frozen denominator."
        ),
        promotion_rule=(
            "Every adequacy, uncertainty, structural, computability and visibility gate must "
            "pass; predictive advantage alone is insufficient."
        ),
        causal_strength=config.causal_strength,
        requested_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        interface_input_quantity_ids=config.feature_quantity_ids,
        interface_output_quantity_ids=config.receiver_quantity_ids,
        mapping_assumption_ids=config.mapping_assumption_ids,
        joint_response_sink_effort_distribution_identified=False,
    )


def _obligation_template(
    system: SystemSpec,
    dataset: IdentificationDataset,
    config: LawIdentificationConfig,
    fits: tuple[CandidateFit, ...],
    evidence_link: EvidenceLink,
) -> LawObligationTemplate:
    suffix = config.method_kind.value.lower()
    denominator_cells = tuple(
        sorted(
            {
                observation.denominator_cell_id
                for observation in dataset.observations
                if observation.role is ObservationRole.PRIMARY
            }
        )
    )
    exchange_factors = tuple(
        sorted({exchange.exchanged_factor_id for exchange in dataset.one_factor_exchanges})
    )
    view_ids = tuple(sorted(value.model.numerical_view_id for value in fits))
    envelope_ids = {
        view.computability_envelope_id
        for view in system.numerical_views
        if view.view_id in set(view_ids)
    }
    if len(envelope_ids) != 1:
        raise ValueError("law identification requires one common computability envelope")
    falsifier_checks = (
        (
            "wrong-action",
            FalsifierKind.WRONG_ACTION,
            "Wrong-action displacement and the baseline comparator must clear frozen thresholds.",
        ),
        (
            "within-cell-recurrence",
            FalsifierKind.WITHIN_CELL_RECURRENCE,
            "Each split, action, history, denominator and view cell must recur across units.",
        ),
        (
            "one-factor-exchange",
            FalsifierKind.ONE_FACTOR_EXCHANGE,
            "Predeclared one-factor exchanges must preserve response slopes within native-unit tolerances.",
        ),
        (
            "structural-convergence",
            FalsifierKind.STRUCTURAL_CONVERGENCE,
            "Response direction, rank, curvature and retained memory must survive refinement.",
        ),
    )
    return LawObligationTemplate(
        template_id=f"obligation-template.{dataset.dataset_id}.{suffix}",
        obligations_id=f"obligations.{dataset.dataset_id}.{suffix}",
        support_id=f"support.{dataset.dataset_id}",
        validity_id=f"validity.{dataset.dataset_id}",
        uncertainty_id=f"uncertainty-spec.{dataset.dataset_id}",
        closure_id=f"closure.{dataset.dataset_id}",
        structural_convergence_id=f"convergence-spec.{dataset.dataset_id}",
        computability_id=f"computability.{dataset.dataset_id}",
        independent_unit_id=system.independent_unit.unit_id,
        physical_unit_count=len(dataset.physical_unit_instance_ids),
        nested_numerical_view_count=len(view_ids),
        information_cutoff_id=dataset.information_cutoff_id,
        chart_ids=(config.chart_id,),
        denominator_cell_ids=denominator_cells,
        action_bounds=config.action_bounds,
        validity_domain_ids=denominator_cells,
        assumption_ids=config.mapping_assumption_ids,
        uncertainty_method_key="independent-unit-heldout-envelope",
        uncertainty_confidence_level=config.uncertainty_confidence_level,
        interval_quantity_ids=config.receiver_quantity_ids,
        uncertainty_limitation_codes=("descriptive-heldout-bound-not-population-coverage",),
        falsifiers=tuple(
            sorted(
                (
                    FalsifierObligationTemplate(
                        falsifier_id=f"falsifier.{identifier}",
                        kind=kind,
                        capability_key=f"baseline.{identifier}",
                        description=description,
                        decisive_rule=description,
                    )
                    for identifier, kind, description in falsifier_checks
                ),
                key=lambda value: value.falsifier_id,
            )
        ),
        recurrence_cell_ids=denominator_cells,
        exchange_factor_ids=exchange_factors,
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=("curvature", "memory", "response-direction", "response-rank"),
        numerical_view_ids=view_ids,
        structural_tolerances=tuple(
            sorted(
                (
                    NamedDecimal(
                        value_id=f"maximum-refinement-difference.{value.tolerance_id}",
                        value=value.maximum_refinement_difference,
                        unit=value.native_unit,
                    )
                    for value in config.structural_tolerances
                ),
                key=lambda value: value.value_id,
            )
        ),
        computability_envelope_id=next(iter(envelope_ids)),
    )


def default_response_method_proof_owner() -> QualificationProofOwner:
    return QualificationProofOwner(
        owner_id="proof-owner.response-method-point-profile",
        capability_key="baseline.response-method-qualification-profile",
        capability_version="1.0.0",
        implementation_sha256=hashlib.sha256(b"response-method-point-profile:1.0.0").hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class LawIdentificationService:
    """Fit every numerical view, then delegate all generic/final decisions."""

    payload_publisher: CandidatePayloadPublisher
    qualification: ResponseLawQualificationService
    profile_evaluator: ResponseQualificationProfileEvaluator
    family_assembler: CandidateFamilyAssembler = CandidateFamilyAssembler()

    def qualify_law(
        self,
        system: SystemSpec,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        identifier: LawIdentifier,
    ) -> LawQualificationResult:
        expected_system = ObjectIdentity.from_record(system.system_id, system)
        if dataset.system != expected_system or dataset.relation != system.relation:
            raise ValueError("identification dataset is bound to another system or relation")
        if config.method_key != identifier.method_key:
            raise ValueError("identification configuration selects another law method")
        fits = tuple(
            sorted(
                (
                    identifier.fit(dataset, config, numerical_view_id=view.view_id)
                    for view in system.numerical_views
                ),
                key=lambda value: value.model.numerical_view_id,
            )
        )
        reference_fit = _reference_fit(system, fits)
        evaluator = _candidate_evaluator(reference_fit)
        evidence_link = _evidence_link(system, dataset, evaluator)
        implementation = CandidateEvaluatorImplementation(
            implementation_id=f"implementation.{config.method_key}.canonical-parametric-law",
            capability_key=config.method_key,
            capability_version=config.method_version,
            evaluator_key="canonical-parametric-law",
            source_sha256=hashlib.sha256(
                f"{config.method_key}:canonical-parametric-law:1.0.0".encode()
            ).hexdigest(),
        )
        publication = self.payload_publisher.publish_candidate_payload(
            payload=reference_fit.model.canonical_bytes(),
            evaluator=evaluator,
            implementation=ObjectIdentity.from_record(
                implementation.implementation_id,
                implementation,
            ),
            decoder_schema='empirical-lawhood/methods/parametric-law/model-decoder',
            decoder_version="1.0.0",
            maximum_decode_bytes=1024 * 1024,
        )
        suffix = config.method_kind.value.lower()
        candidate_id = f"candidate.{dataset.dataset_id}.{suffix}"
        axis_map = LawCandidateAxisMap(
            axis_map_id=f"axis-map.{dataset.dataset_id}.{suffix}",
            bindings=(
                LawCandidateAxisBinding(
                    binding_id=f"axis-binding.{dataset.dataset_id}.{suffix}",
                    candidate_version_member_id=f"candidate-version.{dataset.dataset_id}.{suffix}",
                    denominator_member_id=f"denominator-member.{system.system_id}",
                    qualification_view_ids=tuple(
                        sorted(view.view_id for view in system.numerical_views)
                    ),
                    claimed_property_ids=("response-method-structural-qualification",),
                    nontransported_property_ids=("coefficient-transport",),
                ),
            ),
        )
        _, claim_binding = _response_method_scope(system, dataset, config)
        claim_template = _claim_template(dataset, config)
        obligation_template = _obligation_template(
            system,
            dataset,
            config,
            fits,
            evidence_link,
        )
        method_receipts = tuple(
            CandidateMethodEvidenceReceipt(
                receipt_id=f"method-evidence.{value.fit_id}",
                evidence_kind_id="parametric-fit",
                metrics=value.metrics,
                artifact_ids=(evaluator.payload.artifact_id,),
                evidence_link_ids=(evidence_link.link_id,),
                method_reason_codes=(),
            )
            for value in fits
        )
        candidate_evidence = LawCandidateEvidence(
            evidence_id=f"candidate-evidence.{dataset.dataset_id}.{suffix}",
            candidate_id=candidate_id,
            system=expected_system,
            dataset_or_projection=ObjectIdentity.from_record(dataset.dataset_id, dataset),
            config=ObjectIdentity.from_record(config.config_id, config),
            method_key=config.method_key,
            method_version=config.method_version,
            method_kind=config.method_kind,
            representation_kind=config.representation_kind,
            candidate_evaluator=evaluator,
            axis_map=axis_map,
            claim_unit_binding=ObjectIdentity.from_record(claim_binding.binding_id, claim_binding),
            physical_independent_unit_ids=dataset.physical_unit_instance_ids,
            claim_template=claim_template,
            obligation_template=obligation_template,
            method_receipts=tuple(sorted(method_receipts, key=lambda value: value.receipt_id)),
            payload_publication=publication,
            evidence_links=(evidence_link,),
            outcome_access=dataset.outcome_access,
            parent_visibility_ceilings=dataset.parent_visibility_ceilings,
            visibility_ceiling=dataset.visibility_ceiling,
        )
        profile_assessment, convergence = self.profile_evaluator.evaluate(
            system=system,
            dataset=dataset,
            config=config,
            fits=fits,
            reference_fit=reference_fit,
            candidate=candidate_evidence,
            evidence_links=(evidence_link,),
        )
        metrics = tuple(
            sorted(
                (*reference_fit.metrics, *convergence.metrics),
                key=lambda value: value.value_id,
            )
        )
        assessment = ComponentUncertaintyCandidateAssessment(
            assessment_id=f"candidate-assessment.{dataset.dataset_id}.{suffix}",
            candidate_id=candidate_id,
            candidate_evidence=candidate_evidence,
            profile_assessment=profile_assessment,
            metrics=metrics,
            response_method_structural_convergence=convergence,
        )
        selector = self.profile_evaluator.owner
        ledger = CandidateFamilyLedger(
            family_id=f"candidate-family.{dataset.dataset_id}.{suffix}",
            system=expected_system,
            dataset_or_projection=ObjectIdentity.from_record(dataset.dataset_id, dataset),
            method_key=config.method_key,
            method_version=config.method_version,
            method_kind=config.method_kind,
            representation_kind=config.representation_kind,
            chart_id=config.chart_id,
            axis_map=axis_map,
            claim_unit_binding=ObjectIdentity.from_record(claim_binding.binding_id, claim_binding),
            claim_template=claim_template,
            obligation_template=obligation_template,
            members=(
                CandidateFamilyMember(
                    candidate_id=candidate_id,
                    config=ObjectIdentity.from_record(config.config_id, config),
                    complexity_rank=0,
                    disposition=CandidateRosterDisposition.ASSESS_REQUIRED,
                    predeclared_reason_codes=(),
                ),
            ),
            candidate_generation_rule_ids=(config.method_key,),
            selection_threshold_ids=(config.config_id,),
            development_input_ids=(dataset.dataset_id,),
            selector_capability=ObjectIdentity.from_record(selector.owner_id, selector),
            selector_config=ObjectIdentity.from_record(
                self.profile_evaluator.profile.profile_id,
                self.profile_evaluator.profile,
            ),
            selector_implementation=ObjectIdentity.from_record(selector.owner_id, selector),
            multiplicity_family_id=f"multiplicity.{dataset.dataset_id}.{suffix}",
            multiplicity_rule_id="singleton-no-multiplicity-adjustment",
            randomness_seed_ids=(),
            tie_break_rule="lowest-complexity-rank-then-candidate-id",
            outcome_access=dataset.outcome_access,
            parent_visibility_ceilings=dataset.parent_visibility_ceilings,
            visibility_ceiling=dataset.visibility_ceiling,
        )
        family = self.family_assembler.assemble(ledger, (assessment,))
        return self.qualification.qualify(
            system,
            ObjectIdentity.from_record(dataset.dataset_id, dataset),
            family,
        )

    def identify(
        self,
        system: SystemSpec,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        identifier: LawIdentifier,
    ) -> LawIdentificationResult:
        return project_response_method(self.qualify_law(system, dataset, config, identifier))


def build_response_method_identification_service(
    *,
    payload_publisher: CandidatePayloadPublisher,
    payload_reader: CandidatePayloadReader,
) -> LawIdentificationService:
    owner = default_response_method_proof_owner()
    qualification = ResponseLawQualificationService(
        payload_reader=payload_reader,
        decoder_registry=CandidatePayloadDecoderRegistry(
            decoders=(CanonicalParametricLawModelDecoder(),),
        ),
    )
    return LawIdentificationService(
        payload_publisher=payload_publisher,
        qualification=qualification,
        profile_evaluator=ResponseQualificationProfileEvaluator(owner=owner),
    )


def check(result: LawIdentificationResult, kind: AdequacyCheckKind) -> AdequacyCheckResult:
    """Retrieve one named gate without exposing estimator-specific report structure."""

    return next(value for value in result.checks if value.kind is kind)
