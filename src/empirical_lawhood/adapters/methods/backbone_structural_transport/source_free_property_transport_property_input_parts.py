from __future__ import annotations

from decimal import Decimal
import hashlib


from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.contrasting_objectives import PartialMorphismSpec, PartialMorphismUncertaintyOperation
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec, EvidenceLineageBundle
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass, MetatheoryApplicability, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection
from empirical_lawhood.planning.property_survival import PARTIAL_MORPHISM_ASSESSMENT_SCHEMA, PropertyPathCorrespondence, PropertySurvivalCellEvidence, PropertySurvivalKind, PropertySurvivalMemberSpec, PropertySurvivalPath, PropertySurvivalSpec
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityManifest, CapabilityRegistry


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(
    object_id: str, schema: str = 'empirical-lawhood/methods/structural-transport/synthetic-input/object'
) -> ObjectIdentity:
    return ObjectIdentity(object_id, schema, "1.0.0", _digest(f"{object_id}:{schema}"))


def _method() -> MetatheoryMethodSelection:
    schema = 'empirical-lawhood/methods/structural-transport/synthetic-property-survival/config'
    return MetatheoryMethodSelection(
        selection_id="selection.property-survival.synthetic",
        capability_key="executable-source-free-property-transport.property-survival.synthetic",
        capability_version="1.0.0",
        config=_identity("config.property-survival.synthetic", schema),
        implementation_sha256=_digest('property-survival-synthetic-implementation'),
    )


def _registry() -> CapabilityRegistry:
    method = _method()
    manifest = CapabilityManifest(
        capability_key=method.capability_key,
        capability_version=method.capability_version,
        kind=CapabilityKind.ANALYSIS,
        config_schema=method.config.object_schema,
        config_schema_sha256=_digest(method.config.object_schema),
        input_schema_ids=(PropertySurvivalSpec.SCHEMA,),
        output_schema_ids=(PropertySurvivalCellEvidence.SCHEMA,),
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
        conformance_check_ids=("conformance.property-survival.synthetic",),
        implementation_sha256=method.implementation_sha256,
    )
    return CapabilityRegistry("registry.property-survival.synthetic", (manifest,))


def _label(kind: PropertySurvivalKind) -> str:
    return kind.value.lower().replace("_", "-")


def _member(
    kind: PropertySurvivalKind,
    *,
    required: bool,
    composed: bool = False,
) -> PropertySurvivalMemberSpec:
    label = _label(kind)
    return PropertySurvivalMemberSpec(
        member_id=f"member.property.{label}",
        survival_kind=kind,
        partial_morphism_spec=_identity(
            f"partial-morphism.{label}",
            PartialMorphismSpec.SCHEMA,
        ),
        property_id=f"property.{label}",
        property_metric_id=f"metric.{label}",
        resolution=NamedDecimal(f"resolution.{label}", Decimal("0.01"), "1"),
        uncertainty_operation=PartialMorphismUncertaintyOperation.PRESERVE,
        uncertainty_operation_id=f"uncertainty.{label}",
        support=_identity(f"support.{label}"),
        direct_face_ids=(f"face.{label}.direct",) if required else (),
        composed_face_ids=(f"face.{label}.composed",) if required and composed else (),
        falsifier_ids=(f"falsifier.{label}",) if required else (),
        applicability=(
            MetatheoryApplicability.REQUIRED
            if required
            else MetatheoryApplicability.NOT_APPLICABLE
        ),
        applicability_reason_codes=() if required else ("PROPERTY_NOT_ENTERED",),
    )


def _dependence_identity() -> ObjectIdentity:
    return _identity("dependence-spec.synthetic", EvidenceDependenceSpec.SCHEMA)


def _spec(
    required: set[PropertySurvivalKind],
    *,
    composed: set[PropertySurvivalKind] | None = None,
) -> PropertySurvivalSpec:
    composed = set() if composed is None else composed
    members = tuple(
        sorted(
            (
                _member(kind, required=kind in required, composed=kind in composed)
                for kind in PropertySurvivalKind
            ),
            key=lambda value: value.member_id,
        )
    )
    faces = tuple(
        sorted(
            {
                face
                for member in members
                for face in (*member.direct_face_ids, *member.composed_face_ids)
            }
        )
    )
    return PropertySurvivalSpec(
        spec_id='property-survival.synthetic',
        transformation=_identity("transformation.synthetic"),
        context=_identity("context.synthetic"),
        source_evidence_world_profile=_identity("world.source"),
        target_evidence_world_profile=_identity("world.target"),
        dependence_spec=_dependence_identity(),
        physical_unit_ids=("unit.property.001",),
        members=members,
        required_face_ids=faces,
        aggregation_rule_id="aggregation.complete-unit-first",
        maximum_ordinary_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_structural_evidence_ceiling=MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _dependence_assessment(spec: PropertySurvivalSpec) -> EvidenceDependenceAssessment:
    return EvidenceDependenceAssessment(
        assessment_id="dependence-assessment.synthetic",
        dependence_spec=spec.dependence_spec,
        predecessor_lineage=_identity("lineage.predecessor", EvidenceLineageBundle.SCHEMA),
        target_lineage=_identity("lineage.target", EvidenceLineageBundle.SCHEMA),
        requested_class=EvidenceDependenceClass.SAME_IMPLEMENTATION_RESAMPLE,
        achieved_class=EvidenceDependenceClass.SAME_IMPLEMENTATION_RESAMPLE,
        shared_components=(),
        refused_stronger_classes=(),
        evidence_links=(),
        reason_codes=(),
    )


def _evidence(
    spec: PropertySurvivalSpec,
    dispositions: dict[
        tuple[PropertySurvivalKind, PropertySurvivalPath], MetatheoryCellDisposition
    ],
) -> tuple[PropertySurvivalCellEvidence, ...]:
    values = []
    for member in spec.members:
        if member.applicability is MetatheoryApplicability.NOT_APPLICABLE:
            continue
        for path, faces in (
            (PropertySurvivalPath.DIRECT, member.direct_face_ids),
            (PropertySurvivalPath.COMPOSED, member.composed_face_ids),
        ):
            for face in faces:
                disposition = dispositions[(member.survival_kind, path)]
                face_label = face.replace(".", "-")
                label = f"{_label(member.survival_kind)}.{path.value.lower()}.{face_label}"
                values.append(
                    PropertySurvivalCellEvidence(
                        evidence_id=f"evidence.property.{label}",
                        survival_spec=ObjectIdentity.from_record(spec.spec_id, spec),
                        member_spec=ObjectIdentity.from_record(member.member_id, member),
                        physical_unit_id=spec.physical_unit_ids[0],
                        face_id=face,
                        domain_cell_id=f"domain-cell.{label}",
                        path=path,
                        partial_morphism_assessment=_identity(
                            f"morphism-assessment.{label}",
                            PARTIAL_MORPHISM_ASSESSMENT_SCHEMA,
                        ),
                        source_observation=_identity(f"observation.source.{label}"),
                        target_observation=_identity(f"observation.target.{label}"),
                        defect=NamedDecimal(f"defect.{label}", Decimal("0"), "1"),
                        margin=NamedDecimal(f"margin.{label}", Decimal("1"), "1"),
                        uncertainty=NamedDecimal(f"uncertainty.{label}", Decimal("0"), "1"),
                        property_preserved=(
                            True
                            if disposition is MetatheoryCellDisposition.SUPPORTED
                            else False
                            if disposition is MetatheoryCellDisposition.OPPOSED
                            else None
                        ),
                        morphism_cell_disposition=disposition,
                        decision_assurance=None,
                        method=_method(),
                        evidence_links=(),
                    )
                )
    return tuple(sorted(values, key=lambda value: value.evidence_id))


def _correspondences(
    spec: PropertySurvivalSpec,
    evidence: tuple[PropertySurvivalCellEvidence, ...],
) -> tuple[PropertyPathCorrespondence, ...]:
    values = []
    for member in spec.members:
        if member.applicability is MetatheoryApplicability.NOT_APPLICABLE:
            continue
        direct = sorted(
            (
                value
                for value in evidence
                if value.member_spec.object_id == member.member_id
                and value.path is PropertySurvivalPath.DIRECT
            ),
            key=lambda value: (value.physical_unit_id, value.face_id),
        )
        composed = sorted(
            (
                value
                for value in evidence
                if value.member_spec.object_id == member.member_id
                and value.path is PropertySurvivalPath.COMPOSED
            ),
            key=lambda value: (value.physical_unit_id, value.face_id),
        )
        if composed:
            assert len(direct) == len(composed)
        for index, direct_value in enumerate(direct):
            composed_value = composed[index] if composed else None
            values.append(
                PropertyPathCorrespondence(
                    correspondence_id=f"correspondence.{member.member_id}.{index:02d}",
                    survival_spec=ObjectIdentity.from_record(spec.spec_id, spec),
                    member_spec=ObjectIdentity.from_record(member.member_id, member),
                    physical_unit_id=direct_value.physical_unit_id,
                    receiver_action_face_id=f"receiver-action-face.{member.member_id}.{index:02d}",
                    direct_face_id=direct_value.face_id,
                    direct_domain_cell_id=direct_value.domain_cell_id,
                    composed_face_id=(None if composed_value is None else composed_value.face_id),
                    composed_domain_cell_id=(
                        None if composed_value is None else composed_value.domain_cell_id
                    ),
                    applicability=(
                        MetatheoryApplicability.NOT_APPLICABLE
                        if composed_value is None
                        else MetatheoryApplicability.REQUIRED
                    ),
                    reason_codes=(
                        ("COMPOSED_PATH_NOT_DECLARED",) if composed_value is None else ()
                    ),
                    target_outcomes_read=False,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                )
            )
    return tuple(sorted(values, key=lambda value: value.correspondence_id))
