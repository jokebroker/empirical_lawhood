"""Deterministic source-free prediction operands for metatheory R1."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec, EvidenceLineageBundle
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass, MetatheoryClaimKind, MetatheoryEvidenceCeiling, MetatheoryMethodSelection, MetatheoryPredictiveLevel
from empirical_lawhood.planning.metatheory_prediction import CategoricalForecast, DynamicalForecast, MetatheoryAdjudicationResult, MetatheoryAdjudicationSpec, MetatheoryForecastCell, MetatheoryMissingTargetPolicy, MetatheoryPredictionPackage, MetatheoryScoringMethodBinding, MetatheoryTargetSpec, MetricForecast
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityRegistry,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _identity(
    object_id: str,
    schema: str = 'empirical-lawhood/methods/structural-transport/synthetic-input/object',
) -> ObjectIdentity:
    return ObjectIdentity(object_id, schema, "1.0.0", _digest(f"{object_id}:{schema}"))


def _method(level: MetatheoryPredictiveLevel) -> MetatheoryMethodSelection:
    label = level.value.lower()
    schema = f'empirical-lawhood/methods/structural-transport/synthetic-prediction-score/config/{label}'
    return MetatheoryMethodSelection(
        selection_id=f"selection.source-free-property-transport-score.{label}",
        capability_key=f"executable-source-free-property-transport.score.{label}",
        capability_version="1.0.0",
        config=_identity(f"config.source-free-property-transport-score.{label}", schema),
        implementation_sha256=_digest(f"source-free-property-transport-score-{label}-implementation"),
    )


def _methods() -> tuple[MetatheoryScoringMethodBinding, ...]:
    return tuple(
        sorted(
            (
                MetatheoryScoringMethodBinding(
                    binding_id=f"scoring-method.{level.value.lower()}",
                    predictive_level=level,
                    method=_method(level),
                )
                for level in MetatheoryPredictiveLevel
            ),
            key=lambda value: value.binding_id,
        )
    )


def _registry() -> CapabilityRegistry:
    manifests = []
    for binding in _methods():
        method = binding.method
        manifests.append(
            CapabilityManifest(
                capability_key=method.capability_key,
                capability_version=method.capability_version,
                kind=CapabilityKind.EVALUATOR,
                config_schema=method.config.object_schema,
                config_schema_sha256=_digest(method.config.object_schema),
                input_schema_ids=(MetatheoryPredictionPackage.SCHEMA,),
                output_schema_ids=(MetatheoryAdjudicationResult.SCHEMA,),
                permissions=(),
                maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                resource_ceiling=ResourceBudget(1, 1024, 0, 1, 0, 1024),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython",
                requires_clean_commit=False,
                requires_active_mount=False,
                requires_network=False,
                conformance_check_ids=(f"conformance.{method.capability_key}",),
                implementation_sha256=method.implementation_sha256,
            )
        )
    return CapabilityRegistry(
        "registry.source-free-property-transport-score.synthetic",
        tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def _target(policy: MetatheoryMissingTargetPolicy) -> MetatheoryTargetSpec:
    return MetatheoryTargetSpec(
        target_id="target.source-free-property-transport.synthetic",
        evidence_world_profile=_identity("world.source-free-property-transport.target"),
        source_qualification_requirement=_identity("source-qualification.target"),
        preparation_requirement=_identity("preparation.target"),
        physical_unit_ids=("unit.source-free-property-transport.target",),
        complete_unit_role_id="role.complete-physical-unit",
        missing_target_policy=policy,
        target_visibility=VisibilityCeiling.PROSPECTIVE,
    )


def _cell(level: MetatheoryPredictiveLevel) -> MetatheoryForecastCell:
    label = level.value.lower()
    categorical = None
    metric = None
    dynamical = None
    if level is MetatheoryPredictiveLevel.CATEGORICAL:
        categorical = CategoricalForecast(
            forecast_id="forecast.categorical.synthetic",
            category_ids=("category.safe", "category.unsafe"),
            predicted_category_id="category.safe",
            category_probabilities=(
                NamedDecimal("category.safe", Decimal("0.9"), "1"),
                NamedDecimal("category.unsafe", Decimal("0.1"), "1"),
            ),
            unsafe_observed_category_ids=("category.unsafe",),
        )
    elif level is MetatheoryPredictiveLevel.METRIC:
        metric = MetricForecast(
            forecast_id="forecast.metric.synthetic",
            quantity_id="quantity.margin",
            native_unit="1",
            interval_lower=NamedDecimal("interval.margin.lower", Decimal("0.4"), "1"),
            interval_upper=NamedDecimal("interval.margin.upper", Decimal("0.6"), "1"),
            calibration_method=_method(level),
            uncertainty_operation_id="uncertainty.metric.synthetic",
            acceptance_rule_id="acceptance.metric.synthetic",
        )
    else:
        dynamical = DynamicalForecast(
            forecast_id="forecast.dynamical.synthetic",
            state_ids=("state.entered", "state.origin"),
            transition_id="transition.origin-to-entered",
            time_origin=_identity("time-origin.synthetic"),
            horizon=NamedDecimal("horizon.synthetic", Decimal(10), "s"),
            first_passage_lower=NamedDecimal("first-passage.lower", Decimal(2), "s"),
            first_passage_upper=NamedDecimal("first-passage.upper", Decimal(5), "s"),
            recovery_or_reentry_rule_id="rule.recovery.synthetic",
            censoring_rule_id="rule.censoring.synthetic",
        )
    return MetatheoryForecastCell(
        cell_id=f"forecast-cell.{label}",
        claim_kind=MetatheoryClaimKind.STRUCTURAL_RECURRENCE,
        predictive_level=level,
        target_id="target.source-free-property-transport.synthetic",
        physical_unit_id="unit.source-free-property-transport.target",
        coordinate_id="coordinate.synthetic",
        property_id=f"property.{label}",
        action_fibre_id="action-fibre.synthetic",
        face_id=f"face.{label}",
        categorical=categorical,
        metric=metric,
        dynamical=dynamical,
        falsifier_ids=(f"falsifier.{label}",),
        unevaluable_condition_ids=(f"unevaluable.{label}",),
    )


def _package(policy: MetatheoryMissingTargetPolicy) -> MetatheoryPredictionPackage:
    return MetatheoryPredictionPackage(
        package_id='source-free-property-transport-prediction.synthetic',
        development_parents=(_identity("development.property-survival"),),
        targets=(_target(policy),),
        forecast_cells=tuple(
            sorted(
                (_cell(level) for level in MetatheoryPredictiveLevel),
                key=lambda value: value.cell_id,
            )
        ),
        requested_dependence_class=(
            EvidenceDependenceClass.INDEPENDENT_GENERATOR_SHARED_ONTOLOGY
        ),
        dependence_spec=_identity("dependence.synthetic", EvidenceDependenceSpec.SCHEMA),
        complete_unit_rule_id="rule.complete-unit-first",
        aggregation_rule_id="rule.no-cross-level-compensation",
        scoring_methods=_methods(),
        decisive_falsifier_ids=("falsifier.decisive-false-safe",),
        planned_issue_id="issue-clock.source-free-property-transport.synthetic",
        information_cutoff=_identity("cutoff.source-free-property-transport.synthetic"),
        target_outcome_access_count=0,
        maximum_structural_evidence_ceiling=(MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE),
        ordinary_parent_identities=(),
    )


def _adjudication_spec(package: MetatheoryPredictionPackage) -> MetatheoryAdjudicationSpec:
    return MetatheoryAdjudicationSpec(
        spec_id="adjudication-spec.source-free-property-transport.synthetic",
        prediction_package=ObjectIdentity.from_record(package.package_id, package),
        scoring_methods=package.scoring_methods,
        multiplicity_rule_id="rule.multiplicity.synthetic",
        complete_unit_aggregation_rule_id="rule.complete-unit-first",
        terminal_rule_id="rule.mixed-levels-remain-mixed",
    )


def _dependence(package: MetatheoryPredictionPackage) -> EvidenceDependenceAssessment:
    return EvidenceDependenceAssessment(
        assessment_id="dependence-assessment.source-free-property-transport.synthetic",
        dependence_spec=package.dependence_spec,
        predecessor_lineage=_identity("lineage.predecessor", EvidenceLineageBundle.SCHEMA),
        target_lineage=_identity("lineage.target", EvidenceLineageBundle.SCHEMA),
        requested_class=package.requested_dependence_class,
        achieved_class=package.requested_dependence_class,
        shared_components=(),
        refused_stronger_classes=(),
        evidence_links=(),
        reason_codes=(),
    )


__all__ = ["_adjudication_spec", "_dependence", "_package", "_registry"]
