from __future__ import annotations

from decimal import Decimal
import hashlib


from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceApplicability, DecisionAssuranceSpec, DecisionAssuranceTarget, DecisionComparisonEvidence, DecisionDisposition, DecisionErrorKind, DecisionErrorRule
from empirical_lawhood.planning.metatheory import MetatheoryEvidenceCeiling, MetatheoryMethodSelection
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityManifest, CapabilityRegistry


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(
    object_id: str, schema: str = 'empirical-lawhood/methods/structural-transport/synthetic-input/object'
) -> ObjectIdentity:
    return ObjectIdentity(object_id, schema, "1.0.0", _digest(f"{object_id}:{schema}"))


def _method(role: str) -> MetatheoryMethodSelection:
    schema = f'empirical-lawhood/methods/structural-transport/synthetic-decision-assurance/config/{role}'
    return MetatheoryMethodSelection(
        selection_id=f"selection.decision.{role}",
        capability_key=f"executable-source-free-property-transport.decision.{role}",
        capability_version="1.0.0",
        config=_identity(f"config.decision.{role}", schema),
        implementation_sha256=_digest(f"decision-{role}-implementation"),
    )


def _registry() -> CapabilityRegistry:
    manifests = []
    for role in ("candidate", "reference", "uncertainty"):
        method = _method(role)
        manifests.append(
            CapabilityManifest(
                capability_key=method.capability_key,
                capability_version="1.0.0",
                kind=CapabilityKind.ANALYSIS,
                config_schema=method.config.object_schema,
                config_schema_sha256=_digest(method.config.object_schema),
                input_schema_ids=(DecisionAssuranceSpec.SCHEMA,),
                output_schema_ids=(DecisionComparisonEvidence.SCHEMA,),
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
                conformance_check_ids=(f"conformance.decision.{role}",),
                implementation_sha256=method.implementation_sha256,
            )
        )
    return CapabilityRegistry("registry.decision.synthetic", tuple(manifests))


def _target(index: int, *, unit: int | None = None) -> DecisionAssuranceTarget:
    unit_index = index if unit is None else unit
    return DecisionAssuranceTarget(
        target_id=f"target.decision.{index:03d}",
        physical_unit_id=f"unit.decision.{unit_index:03d}",
        action_fibre_id=f"fibre.decision.{index:03d}",
        member_id=f"member.decision.{index:03d}",
        view_id=f"view.decision.{index:03d}",
    )


def _rules() -> tuple[DecisionErrorRule, ...]:
    return tuple(
        DecisionErrorRule(
            error_kind=kind,
            decisive_veto=kind
            in {
                DecisionErrorKind.FALSE_ADMISSION,
                DecisionErrorKind.FALSE_SAFE_HOLD,
            },
            maximum_complete_unit_rate=NamedDecimal(
                f"maximum-rate.{kind.value.lower()}",
                Decimal("1"),
                "1",
            ),
            interval_rule_id=f"interval-rule.{kind.value.lower()}",
        )
        for kind in sorted(DecisionErrorKind, key=lambda value: value.value)
    )


def _comparison_spec(
    targets: tuple[DecisionAssuranceTarget, ...] | None = None,
) -> DecisionAssuranceSpec:
    selected = (_target(1),) if targets is None else targets
    units = tuple(sorted({value.physical_unit_id for value in selected}))
    return DecisionAssuranceSpec(
        spec_id='decision-assurance.synthetic',
        applicability=DecisionAssuranceApplicability.REFERENCE_COMPARISON_REQUIRED,
        applicability_reason_codes=(),
        candidate_representation=_identity("representation.candidate"),
        reference_representation=_identity("representation.reference"),
        direct_native_proof=None,
        target_semantics_id="semantics.target",
        sink_semantics_id="semantics.sink",
        gate_semantics_id="semantics.gate",
        direction_ids=("direction.forward",),
        physical_unit_ids=units,
        targets=selected,
        candidate_evaluator=_method("candidate"),
        reference_evaluator=_method("reference"),
        uncertainty_method=_method("uncertainty"),
        comparison_reveal_route=_identity(
            "reveal-route.decision",
            'empirical-lawhood/methods/structural-transport/synthetic-input/sealed-reveal-route',
        ),
        error_rules=_rules(),
        multiplicity_rule_id="multiplicity.complete-unit-first",
        aggregation_rule_id="aggregation.complete-unit-first",
        measured_hold_semantics_id="semantics.measured-hold",
        nonattempt_semantics_id="semantics.nonattempt",
        falsifier_ids=("falsifier.decision-mismatch", "falsifier.false-safe"),
        maximum_ordinary_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_structural_evidence_ceiling=MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _evidence(
    spec: DecisionAssuranceSpec,
    target: DecisionAssuranceTarget,
    *,
    candidate: DecisionDisposition,
    reference: DecisionDisposition,
    reference_hold_viable: bool | None = None,
    substitute: bool = False,
) -> DecisionComparisonEvidence:
    shared_occurrence = _identity(f"occurrence.{target.target_id}")
    shared_delivery = _identity(f"delivery.{target.target_id}")
    return DecisionComparisonEvidence(
        evidence_id=f"evidence.{target.target_id}",
        assurance_spec=ObjectIdentity.from_record(spec.spec_id, spec),
        target=ObjectIdentity.from_record(target.target_id, target),
        physical_unit_id=target.physical_unit_id,
        action_fibre_id=target.action_fibre_id,
        candidate_decision=candidate,
        reference_decision=reference,
        candidate_action_occurrence=(
            shared_occurrence if candidate is DecisionDisposition.ACTIVE_ACTION else None
        ),
        reference_action_occurrence=(
            (
                _identity(f"occurrence.reference-other.{target.target_id}")
                if substitute
                else shared_occurrence
            )
            if reference is DecisionDisposition.ACTIVE_ACTION
            else None
        ),
        candidate_delivery=(
            shared_delivery if candidate is DecisionDisposition.ACTIVE_ACTION else None
        ),
        reference_delivery=(
            (
                _identity(f"delivery.reference-other.{target.target_id}")
                if substitute
                else shared_delivery
            )
            if reference is DecisionDisposition.ACTIVE_ACTION
            else None
        ),
        candidate_gate_operands=_identity(f"gate.candidate.{target.target_id}"),
        reference_gate_operands=_identity(f"gate.reference.{target.target_id}"),
        reference_hold_supported_and_viable=(
            reference_hold_viable if reference is DecisionDisposition.HOLD else None
        ),
        candidate_evaluator=spec.candidate_evaluator,  # type: ignore[arg-type]
        reference_evaluator=spec.reference_evaluator,  # type: ignore[arg-type]
        uncertainty_method=spec.uncertainty_method,  # type: ignore[arg-type]
        publication=_identity(f"publication.{target.target_id}"),
        recovery=_identity(f"recovery.{target.target_id}"),
        evidence_links=(),
    )
