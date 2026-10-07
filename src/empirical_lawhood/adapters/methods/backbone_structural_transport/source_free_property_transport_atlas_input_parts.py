from __future__ import annotations

from decimal import Decimal
import hashlib

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.atlas_qualification import AtlasLawInputKind, AtlasQualificationSpec, ChartTransitionEvidence, SetValuedChartRegion
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeResult
from empirical_lawhood.planning.metatheory import MetatheoryEvidenceCeiling, MetatheoryMethodSelection, MetatheoryPredictiveLevel
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityManifest, CapabilityRegistry


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(object_id, schema, "1.0.0", _digest(f"{object_id}:{schema}"))


def _method(role: str) -> MetatheoryMethodSelection:
    return MetatheoryMethodSelection(
        selection_id=f"selection.atlas.{role}",
        capability_key=f"executable-source-free-property-transport.atlas.{role}",
        capability_version="1.0.0",
        config=_identity(f"config.atlas.{role}", f'empirical-lawhood/methods/structural-transport/synthetic-atlas/config/{role}'),
        implementation_sha256=_digest(f"atlas-{role}-implementation"),
    )


def _registry() -> CapabilityRegistry:
    values = []
    for role in ("response", "transition", "uncertainty"):
        method = _method(role)
        values.append(
            CapabilityManifest(
                capability_key=method.capability_key,
                capability_version="1.0.0",
                kind=CapabilityKind.ANALYSIS,
                config_schema=method.config.object_schema,
                config_schema_sha256=_digest(method.config.object_schema),
                input_schema_ids=(AtlasQualificationSpec.SCHEMA,),
                output_schema_ids=(ChartTransitionEvidence.SCHEMA,),
                permissions=(),
                maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
                resource_ceiling=ResourceBudget(1, 1024, 0, 1, 0, 1024),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython",
                requires_clean_commit=False,
                requires_active_mount=False,
                requires_network=False,
                conformance_check_ids=(f"conformance.atlas.{role}",),
                implementation_sha256=method.implementation_sha256,
            )
        )
    return CapabilityRegistry("registry.atlas.synthetic", tuple(values))


def _region() -> SetValuedChartRegion:
    return SetValuedChartRegion(
        region_id="region.atlas.synthetic",
        coordinate_system_id="coordinate-system.synthetic",
        support_cell_ids=("cell.source", "cell.target"),
        boundary_labels=("boundary.active",),
        stratum_labels=("stratum.inside", "stratum.outside"),
        uncertainty_representation=_identity(
            "uncertainty.atlas.synthetic",
            'empirical-lawhood/methods/structural-transport/synthetic-input/uncertainty',
        ),
        resolution=NamedDecimal("resolution.atlas", Decimal("0.01"), "1"),
        artifact=_identity("artifact.atlas.region", 'empirical-lawhood/methods/structural-transport/synthetic-input/artifact'),
    )


def _spec(
    *,
    levels: tuple[MetatheoryPredictiveLevel, ...],
    law_input_kind: AtlasLawInputKind = AtlasLawInputKind.LAW_ATLAS,
) -> AtlasQualificationSpec:
    return AtlasQualificationSpec(
        spec_id='atlas-qualification.synthetic',
        system_id="system.synthetic",
        evidence_world_id="world.synthetic",
        relation_id="relation.synthetic",
        chart_id="chart.synthetic",
        law_input_kind=law_input_kind,
        law_or_obstruction=_identity(
            "law-input.synthetic",
            (
                'empirical-lawhood/methods/structural-transport/synthetic-input/law-obstruction'
                if law_input_kind is AtlasLawInputKind.LAW_OBSTRUCTION
                else 'empirical-lawhood/methods/structural-transport/synthetic-input/response-atlas'
            ),
        ),
        coordinate_challenge_result=_identity(
            "coordinate-result.synthetic",
            CoordinateChallengeResult.SCHEMA,
        ),
        coordinate_cell_ids=("cell.coordinate.001", "cell.coordinate.002"),
        gate_owner_ids=("gate-owner.synthetic",),
        physical_unit_ids=("unit.atlas.001", "unit.atlas.002"),
        n=2,
        alpha=NamedDecimal("alpha.atlas", Decimal("0.05"), "1"),
        delta=NamedDecimal("delta.atlas", Decimal("0.01"), "1"),
        response_span_method=_method("response"),
        transition_method=_method("transition"),
        uncertainty_method=_method("uncertainty"),
        required_predictive_levels=levels,
        boundary_containment_rule_id="rule.boundary.synthetic",
        ambiguity_width_rule_id="rule.ambiguity.synthetic",
        recurrence_rule_id="rule.recurrence.synthetic",
        time_ordering_rule_id=(
            "rule.time-order.synthetic" if MetatheoryPredictiveLevel.DYNAMICAL in levels else None
        ),
        censoring_rule_id=(
            "rule.censoring.synthetic" if MetatheoryPredictiveLevel.DYNAMICAL in levels else None
        ),
        falsifier_ids=("falsifier.boundary", "falsifier.nonrecurrence"),
        maximum_ordinary_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_structural_evidence_ceiling=(MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _evidence(
    spec: AtlasQualificationSpec,
    region: SetValuedChartRegion,
    *,
    metric_opposed: bool = False,
    metric_too_wide: bool = False,
) -> tuple[ChartTransitionEvidence, ...]:
    values = []
    for level in spec.required_predictive_levels:
        metric = level is MetatheoryPredictiveLevel.METRIC
        values.append(
            ChartTransitionEvidence(
                evidence_id=f"evidence.atlas.{level.value.lower()}",
                qualification_spec=ObjectIdentity.from_record(spec.spec_id, spec),
                predictive_level=level,
                source_cell_id="cell.source",
                target_cell_id="cell.target",
                physical_unit_ids=spec.physical_unit_ids,
                direction_id="direction.forward",
                gate_margin_vector=_identity(
                    f"margin.{level.value.lower()}",
                    'empirical-lawhood/methods/structural-transport/synthetic-input/margin-vector',
                ),
                chart_region=ObjectIdentity.from_record(region.region_id, region),
                response_span_supported=True,
                transition_recurrent=not (metric and metric_opposed),
                uncertainty_localized=not (metric and metric_too_wide),
                ambiguity_width_within_rule=not (metric and metric_too_wide),
                gate_labels_preserved=True,
                observations_time_ordered=True,
                first_passage_censored=False,
                ambiguity_width=NamedDecimal(
                    f"ambiguity.{level.value.lower()}",
                    Decimal("0.2") if metric_too_wide and metric else Decimal("0.01"),
                    "1",
                ),
                method=spec.transition_method,
                publication=_identity(
                    f"publication.atlas.{level.value.lower()}",
                    'empirical-lawhood/methods/structural-transport/synthetic-input/publication',
                ),
                recovery=_identity(
                    f"recovery.atlas.{level.value.lower()}",
                    'empirical-lawhood/methods/structural-transport/synthetic-input/recovery',
                ),
                evidence_links=(),
            )
        )
    return tuple(values)
