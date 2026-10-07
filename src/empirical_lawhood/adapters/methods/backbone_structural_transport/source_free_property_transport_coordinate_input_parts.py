from __future__ import annotations

from decimal import Decimal
import hashlib


from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.coordinate_challenges import CoordinateCandidate, CoordinateChallengeEvidence, CoordinateChallengeKind, CoordinateChallengeNomination, CoordinateChallengeSpec, CoordinateSamplingMode, CoordinateSemanticRole, SufficiencyPredicate
from empirical_lawhood.planning.metatheory import MetatheoryEvidenceCeiling, MetatheoryMethodSelection
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityManifest, CapabilityRegistry


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=_digest(f"{object_id}:{schema}"),
    )


def _method(role: str) -> MetatheoryMethodSelection:
    return MetatheoryMethodSelection(
        selection_id=f"selection.coordinate.{role}",
        capability_key=f"executable-source-free-property-transport.coordinate.{role}",
        capability_version="1.0.0",
        config=_identity(f"config.coordinate.{role}", f'empirical-lawhood/methods/structural-transport/synthetic-coordinate-challenge/config/{role}'),
        implementation_sha256=_digest(f"coordinate-{role}-implementation"),
    )


def _registry() -> CapabilityRegistry:
    manifests = []
    for role in ("adjudicator", "constructor"):
        method = _method(role)
        manifests.append(
            CapabilityManifest(
                capability_key=method.capability_key,
                capability_version=method.capability_version,
                kind=CapabilityKind.ANALYSIS,
                config_schema=method.config.object_schema,
                config_schema_sha256=_digest(method.config.object_schema),
                input_schema_ids=(CoordinateChallengeSpec.SCHEMA,),
                output_schema_ids=('empirical-lawhood/methods/structural-transport/synthetic-input/coordinate-output',),
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
                conformance_check_ids=(f"conformance.coordinate.{role}",),
                implementation_sha256=method.implementation_sha256,
            )
        )
    return CapabilityRegistry(
        registry_id="registry.coordinate.synthetic",
        capabilities=tuple(manifests),
    )


def _spec(sampling_mode: CoordinateSamplingMode) -> CoordinateChallengeSpec:
    candidate = CoordinateCandidate(
        candidate_id="candidate.coordinate.synthetic",
        semantic_roles=tuple(sorted(CoordinateSemanticRole, key=lambda value: value.value)),
        field_ids=("field.action", "field.history", "field.receiver"),
        endpoint_ids=("endpoint.response",),
        property_ids=("property.decision", "property.future-response"),
        receiver_id="receiver.synthetic",
        action_word_id="word.synthetic.active",
        horizon_id="horizon.synthetic",
        resolution=NamedDecimal("resolution.coordinate", Decimal("0.01"), "1"),
        normalization_id="normalization.coordinate.synthetic",
        equivalence_relation_id="equivalence.coordinate.synthetic",
    )
    return CoordinateChallengeSpec(
        spec_id=f"coordinate-challenge.synthetic.{sampling_mode.value.lower()}",
        source_qualification=None,
        source_qualification_not_applicable_reasons=("TRUTH_KNOWN_CONFORMANCE",),
        candidates=(candidate,),
        comparator_family_id="comparator.coordinate.synthetic",
        challenge_kind=CoordinateChallengeKind.INFORMATIVE_FIBRE,
        sufficiency_predicates=tuple(sorted(SufficiencyPredicate, key=lambda value: value.value)),
        sampling_mode=sampling_mode,
        physical_unit_ids=("unit.coordinate.001", "unit.coordinate.002"),
        causal_cutoff=_identity(
            "cutoff.coordinate.synthetic", 'empirical-lawhood/methods/structural-transport/synthetic-input/cutoff'
        ),
        collision_tolerance=NamedDecimal("tolerance.collision", Decimal("0.01"), "1"),
        equivalence_tolerance=NamedDecimal("tolerance.equivalence", Decimal("0.01"), "1"),
        history_depths=(1, 2),
        normalized_information_budgets=(NamedDecimal("budget.information.1", Decimal("1"), "1"),),
        future_action_panel_id="panel.future-action.synthetic",
        future_horizon_panel_id="panel.future-horizon.synthetic",
        decision_reference_id="decision-reference.synthetic",
        construction_method=_method("constructor"),
        adjudication_method=_method("adjudicator"),
        falsifier_ids=("falsifier.adverse-collision", "falsifier.endpoint-saturation"),
        targetability_rule_id="rule.coordinate-targetability.synthetic",
        power_rule_id="rule.coordinate-power.synthetic",
        maximum_ordinary_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_structural_evidence_ceiling=(MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _inputs(
    spec: CoordinateChallengeSpec,
    *,
    entered: bool = True,
    future_equal: bool | None = True,
    decision_equal: bool | None = True,
    saturated: bool = False,
    untargetable_unit: str | None = None,
) -> tuple[tuple[CoordinateChallengeNomination, ...], tuple[CoordinateChallengeEvidence, ...]]:
    spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
    candidate = spec.candidates[0]
    candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
    nominations = []
    evidence = []
    for unit in spec.physical_unit_ids:
        targetable = unit != untargetable_unit
        nomination = CoordinateChallengeNomination(
            nomination_id=f"nomination.{candidate.candidate_id}.{unit}",
            challenge_spec=spec_identity,
            candidate=candidate_identity,
            physical_unit_id=unit,
            pair_id=f"pair.{unit}" if targetable else None,
            fibre_id=f"fibre.{unit}" if targetable else None,
            targetable=targetable,
            selection_lineage=(
                _identity(
                    f"selection-lineage.{unit}",
                    'empirical-lawhood/methods/structural-transport/synthetic-input/selection-lineage',
                ),
            ),
            construction_method=spec.construction_method,
            target_outcomes_read=False,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        actual_entered = entered and targetable
        observed = CoordinateChallengeEvidence(
            evidence_id=f"evidence.{nomination.nomination_id}",
            challenge_spec=spec_identity,
            nomination=ObjectIdentity.from_record(nomination.nomination_id, nomination),
            candidate=candidate_identity,
            physical_unit_id=unit,
            sampling_mode=spec.sampling_mode,
            informative_collision_entered=actual_entered,
            present_equal=True if actual_entered else None,
            future_response_equal=future_equal if actual_entered else None,
            decision_equal=decision_equal if actual_entered else None,
            endpoint_saturated=saturated,
            action_occurrence=_identity(
                f"action-occurrence.{unit}", 'empirical-lawhood/methods/structural-transport/synthetic-input/action'
            ),
            exact_structural_rank=2,
            numerical_rank_lower=1,
            numerical_rank_upper=2,
            requested_row_count=3,
            usable_conditioned_rank=2,
            conditioning=NamedDecimal(f"conditioning.{unit}", Decimal("2"), "1"),
            adjudication_method=spec.adjudication_method,
            publication=_identity(
                f"publication.{unit}", 'empirical-lawhood/methods/structural-transport/synthetic-input/publication'
            ),
            recovery=_identity(f"recovery.{unit}", 'empirical-lawhood/methods/structural-transport/synthetic-input/recovery'),
            evidence_links=(
                _identity(f"witness.{unit}", 'empirical-lawhood/methods/structural-transport/synthetic-input/witness'),
            ),
        )
        nominations.append(nomination)
        evidence.append(observed)
    return tuple(nominations), tuple(evidence)
