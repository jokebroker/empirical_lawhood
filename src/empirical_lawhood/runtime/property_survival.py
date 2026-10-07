"""Facewise, noncompensating property-survival finalization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceResult
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryApplicability, MetatheoryCellDisposition
from empirical_lawhood.planning.property_survival import PropertyPathCorrespondence, PropertyPathFaceConsistencyCell, PropertyPathConsistencyCell, PropertyPathConsistency, PropertySurvivalAssessment, FacePairedPropertySurvivalAssessment, PropertySurvivalCellEvidence, PropertySurvivalCell, PropertySurvivalKind, PropertySurvivalPath, PropertySurvivalSignature, PropertySurvivalSpec
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class PropertySurvivalError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PropertySurvivalService:
    capability_registry: CapabilityRegistry

    def finalize(
        self,
        *,
        spec: PropertySurvivalSpec,
        dependence_assessment: EvidenceDependenceAssessment,
        evidence: tuple[PropertySurvivalCellEvidence, ...],
        decision_assurance_results: tuple[DecisionAssuranceResult, ...] = (),
    ) -> PropertySurvivalAssessment:
        if dependence_assessment.dependence_spec != spec.dependence_spec:
            raise PropertySurvivalError("PROPERTY_DEPENDENCE_ASSESSMENT_MISMATCH")
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        members = {value.member_id: value for value in spec.members}
        expected = {
            (member.member_id, unit, face, path)
            for member in spec.members
            if member.applicability is MetatheoryApplicability.REQUIRED
            for unit in spec.physical_unit_ids
            for path, faces in (
                (PropertySurvivalPath.DIRECT, member.direct_face_ids),
                (PropertySurvivalPath.COMPOSED, member.composed_face_ids),
            )
            for face in faces
        }
        observed = {
            (value.member_spec.object_id, value.physical_unit_id, value.face_id, value.path): value
            for value in evidence
        }
        if len(observed) != len(evidence) or set(observed) != expected:
            raise PropertySurvivalError("PROPERTY_CELL_ROSTER_MISMATCH")
        decisions = {
            ObjectIdentity.from_record(value.result_id, value): value
            for value in decision_assurance_results
        }
        cells = []
        groups: dict[str, list[PropertySurvivalCell]] = defaultdict(list)
        path_values: list[PropertyPathConsistencyCell] = []
        links: dict[str, ObjectIdentity] = {}
        for key in sorted(
            observed, key=lambda value: (value[0], value[1], value[2], value[3].value)
        ):
            value = observed[key]
            member = members[key[0]]
            member_identity = ObjectIdentity.from_record(member.member_id, member)
            if value.survival_spec != spec_identity or value.member_spec != member_identity:
                raise PropertySurvivalError("PROPERTY_EVIDENCE_IDENTITY_MISMATCH")
            self._require_method(value)
            disposition = value.morphism_cell_disposition
            reasons: list[str] = []
            falsifiers: tuple[str, ...] = ()
            if disposition is MetatheoryCellDisposition.OPPOSED:
                falsifiers = member.falsifier_ids
                reasons.append("PROPERTY_SURVIVAL_FALSIFIED")
            elif disposition is MetatheoryCellDisposition.UNEVALUABLE:
                reasons.append("PROPERTY_SURVIVAL_UNEVALUABLE")
            if member.survival_kind is PropertySurvivalKind.DECISION_OR_ADMISSION:
                if value.decision_assurance is None:
                    disposition = MetatheoryCellDisposition.UNEVALUABLE
                    reasons.append("DECISION_ASSURANCE_MISSING")
                else:
                    try:
                        decision = decisions[value.decision_assurance]
                    except KeyError as error:
                        raise PropertySurvivalError(
                            "PROPERTY_DECISION_ASSURANCE_IDENTITY_MISMATCH"
                        ) from error
                    if decision.disposition is MetatheoryAggregateDisposition.OPPOSED and (
                        decision.false_admission_count or decision.false_safe_hold_count
                    ):
                        disposition = MetatheoryCellDisposition.OPPOSED
                        falsifiers = member.falsifier_ids
                        reasons.append("DECISION_ASSURANCE_UNSAFE_PROMOTION_VETO")
            elif value.decision_assurance is not None:
                raise PropertySurvivalError("NONDECISION_PROPERTY_CARRIES_DECISION_ASSURANCE")
            cell = PropertySurvivalCell(
                cell_id=f"cell.{value.evidence_id}",
                member_id=member.member_id,
                physical_unit_id=value.physical_unit_id,
                face_id=value.face_id,
                domain_cell_id=value.domain_cell_id,
                path=value.path,
                evidence=ObjectIdentity.from_record(value.evidence_id, value),
                disposition=disposition,
                decisive_falsifier_ids=falsifiers,
                reason_codes=tuple(sorted(reasons)),
            )
            cells.append(cell)
            groups[member.member_id].append(cell)
            for link in value.evidence_links:
                links[link.object_id] = link

        supported: list[str] = []
        opposed: list[str] = []
        unevaluable: list[str] = []
        not_applicable: list[str] = []
        reasons = []
        for member in spec.members:
            if member.applicability is MetatheoryApplicability.NOT_APPLICABLE:
                not_applicable.append(member.member_id)
                path_values.append(
                    PropertyPathConsistencyCell(
                        member_id=member.member_id,
                        disposition=PropertyPathConsistency.NOT_APPLICABLE,
                        reason_codes=member.applicability_reason_codes,
                    )
                )
                continue
            member_cells = groups[member.member_id]
            dispositions = {value.disposition for value in member_cells}
            if MetatheoryCellDisposition.OPPOSED in dispositions:
                opposed.append(member.member_id)
            elif MetatheoryCellDisposition.UNEVALUABLE in dispositions:
                unevaluable.append(member.member_id)
            elif dispositions == {MetatheoryCellDisposition.SUPPORTED}:
                supported.append(member.member_id)
            else:
                raise PropertySurvivalError("PROPERTY_MEMBER_HAS_NO_TERMINAL")
            direct = {
                (value.physical_unit_id, value.face_id): value.disposition
                for value in member_cells
                if value.path is PropertySurvivalPath.DIRECT
            }
            composed = {
                (value.physical_unit_id, value.face_id): value.disposition
                for value in member_cells
                if value.path is PropertySurvivalPath.COMPOSED
            }
            if composed:
                direct_dispositions = set(direct.values())
                composed_dispositions = set(composed.values())
                unevaluable_paths = MetatheoryCellDisposition.UNEVALUABLE in (
                    direct_dispositions | composed_dispositions
                )
                consistent = direct_dispositions == composed_dispositions
                consistency_reasons: tuple[str, ...]
                if unevaluable_paths:
                    consistency = PropertyPathConsistency.UNEVALUABLE
                    consistency_reasons = ("DIRECT_OR_COMPOSED_PATH_UNEVALUABLE",)
                elif consistent:
                    consistency = PropertyPathConsistency.CONSISTENT
                    consistency_reasons = ()
                else:
                    consistency = PropertyPathConsistency.OPPOSED
                    consistency_reasons = ("DIRECT_COMPOSED_DISAGREEMENT",)
                path_values.append(
                    PropertyPathConsistencyCell(
                        member_id=member.member_id,
                        disposition=consistency,
                        reason_codes=consistency_reasons,
                    )
                )
                if consistency is PropertyPathConsistency.OPPOSED:
                    reasons.append(f"DIRECT_COMPOSED_DISAGREEMENT.{member.member_id}")
                    if member.member_id in supported:
                        supported.remove(member.member_id)
                        opposed.append(member.member_id)
            else:
                path_values.append(
                    PropertyPathConsistencyCell(
                        member_id=member.member_id,
                        disposition=PropertyPathConsistency.NOT_APPLICABLE,
                        reason_codes=("COMPOSED_PATH_NOT_DECLARED",),
                    )
                )

        signature = PropertySurvivalSignature(
            signature_id=f"signature.{spec.spec_id}",
            supported_member_ids=tuple(sorted(supported)),
            opposed_member_ids=tuple(sorted(opposed)),
            unevaluable_member_ids=tuple(sorted(unevaluable)),
            not_applicable_member_ids=tuple(sorted(not_applicable)),
        )
        v1_assessments = {
            value.partial_morphism_assessment.object_id: value.partial_morphism_assessment
            for value in evidence
        }
        return PropertySurvivalAssessment(
            assessment_id=f"assessment.{spec.spec_id}",
            survival_spec=spec_identity,
            morphism_assessments=tuple(v1_assessments[key] for key in sorted(v1_assessments)),
            cell_evidence=tuple(sorted(evidence, key=lambda value: value.evidence_id)),
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            signature=signature,
            path_consistency=tuple(sorted(path_values, key=lambda value: value.member_id)),
            dependence_assessment=dependence_assessment,
            maximum_ordinary_evidence_ceiling=spec.maximum_ordinary_evidence_ceiling,
            maximum_structural_evidence_ceiling=spec.maximum_structural_evidence_ceiling,
            reason_codes=tuple(sorted(reasons)),
            response_law_produced=False,
            coefficient_or_sample_pooling_performed=False,
        )

    def finalize_face_paired(
        self,
        *,
        spec: PropertySurvivalSpec,
        dependence_assessment: EvidenceDependenceAssessment,
        evidence: tuple[PropertySurvivalCellEvidence, ...],
        path_correspondences: tuple[PropertyPathCorrespondence, ...],
        decision_assurance_results: tuple[DecisionAssuranceResult, ...] = (),
    ) -> FacePairedPropertySurvivalAssessment:
        "Finalize a face-paired property-survival assessment."

        property_assessment = self.finalize(
            spec=spec,
            dependence_assessment=dependence_assessment,
            evidence=evidence,
            decision_assurance_results=decision_assurance_results,
        )
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        members = {value.member_id: value for value in spec.members}
        evidence_by_operand = {
            (
                value.member_spec.object_id,
                value.physical_unit_id,
                value.face_id,
                value.domain_cell_id,
                value.path,
            ): value
            for value in evidence
        }
        correspondence_keys = {
            (
                value.member_spec.object_id,
                value.physical_unit_id,
                value.receiver_action_face_id,
            ): value
            for value in path_correspondences
        }
        if len(correspondence_keys) != len(path_correspondences):
            raise PropertySurvivalError("PROPERTY_PATH_CORRESPONDENCE_DUPLICATE")

        used_direct: set[tuple[str, str, str, str, PropertySurvivalPath]] = set()
        used_composed: set[tuple[str, str, str, str, PropertySurvivalPath]] = set()
        face_cells: list[PropertyPathFaceConsistencyCell] = []
        for correspondence in path_correspondences:
            member = members.get(correspondence.member_spec.object_id)
            if (
                correspondence.survival_spec != spec_identity
                or member is None
                or correspondence.member_spec
                != ObjectIdentity.from_record(member.member_id, member)
                or correspondence.physical_unit_id not in set(spec.physical_unit_ids)
            ):
                raise PropertySurvivalError("PROPERTY_PATH_CORRESPONDENCE_IDENTITY_MISMATCH")
            direct_key = (
                member.member_id,
                correspondence.physical_unit_id,
                correspondence.direct_face_id,
                correspondence.direct_domain_cell_id,
                PropertySurvivalPath.DIRECT,
            )
            try:
                direct = evidence_by_operand[direct_key]
            except KeyError as error:
                raise PropertySurvivalError("PROPERTY_PATH_DIRECT_OPERAND_MISMATCH") from error
            if direct_key in used_direct:
                raise PropertySurvivalError("PROPERTY_PATH_DIRECT_OPERAND_REUSED")
            used_direct.add(direct_key)

            composed: PropertySurvivalCellEvidence | None = None
            if correspondence.applicability is MetatheoryApplicability.REQUIRED:
                assert correspondence.composed_face_id is not None
                assert correspondence.composed_domain_cell_id is not None
                composed_key = (
                    member.member_id,
                    correspondence.physical_unit_id,
                    correspondence.composed_face_id,
                    correspondence.composed_domain_cell_id,
                    PropertySurvivalPath.COMPOSED,
                )
                try:
                    composed = evidence_by_operand[composed_key]
                except KeyError as error:
                    raise PropertySurvivalError(
                        "PROPERTY_PATH_COMPOSED_OPERAND_MISMATCH"
                    ) from error
                if composed_key in used_composed:
                    raise PropertySurvivalError("PROPERTY_PATH_COMPOSED_OPERAND_REUSED")
                used_composed.add(composed_key)
                pair = (direct.morphism_cell_disposition, composed.morphism_cell_disposition)
                if MetatheoryCellDisposition.UNEVALUABLE in pair:
                    disposition = PropertyPathConsistency.UNEVALUABLE
                    face_reasons: tuple[str, ...] = ("DIRECT_OR_COMPOSED_PATH_UNEVALUABLE",)
                elif pair[0] is pair[1]:
                    disposition = PropertyPathConsistency.CONSISTENT
                    face_reasons = ()
                else:
                    disposition = PropertyPathConsistency.OPPOSED
                    face_reasons = ("DIRECT_COMPOSED_FACE_DISAGREEMENT",)
            else:
                disposition = PropertyPathConsistency.NOT_APPLICABLE
                face_reasons = correspondence.reason_codes
            face_cells.append(
                PropertyPathFaceConsistencyCell(
                    cell_id=f"path-cell.{correspondence.correspondence_id}",
                    correspondence=ObjectIdentity.from_record(
                        correspondence.correspondence_id,
                        correspondence,
                    ),
                    member_id=member.member_id,
                    physical_unit_id=correspondence.physical_unit_id,
                    receiver_action_face_id=correspondence.receiver_action_face_id,
                    direct_evidence=ObjectIdentity.from_record(direct.evidence_id, direct),
                    composed_evidence=(
                        None
                        if composed is None
                        else ObjectIdentity.from_record(composed.evidence_id, composed)
                    ),
                    disposition=disposition,
                    reason_codes=face_reasons,
                )
            )

        expected_direct = {
            key for key in evidence_by_operand if key[-1] is PropertySurvivalPath.DIRECT
        }
        expected_composed = {
            key for key in evidence_by_operand if key[-1] is PropertySurvivalPath.COMPOSED
        }
        if used_direct != expected_direct or used_composed != expected_composed:
            raise PropertySurvivalError("PROPERTY_PATH_CORRESPONDENCE_ROSTER_MISMATCH")

        by_member: dict[str, list[PropertyPathFaceConsistencyCell]] = defaultdict(list)
        for cell in face_cells:
            by_member[cell.member_id].append(cell)
        path_values: list[PropertyPathConsistencyCell] = []
        supported = set(property_assessment.signature.supported_member_ids)
        opposed = set(property_assessment.signature.opposed_member_ids)
        unevaluable = set(property_assessment.signature.unevaluable_member_ids)
        not_applicable = set(property_assessment.signature.not_applicable_member_ids)
        assessment_reasons = set(property_assessment.reason_codes)
        for member in spec.members:
            if member.applicability is MetatheoryApplicability.NOT_APPLICABLE:
                consistency = PropertyPathConsistency.NOT_APPLICABLE
                consistency_reasons = member.applicability_reason_codes
            else:
                member_faces = by_member.get(member.member_id, [])
                if not member_faces:
                    raise PropertySurvivalError("PROPERTY_PATH_MEMBER_UNCONSUMED")
                dispositions = {value.disposition for value in member_faces}
                if PropertyPathConsistency.OPPOSED in dispositions:
                    consistency = PropertyPathConsistency.OPPOSED
                    consistency_reasons = ("DIRECT_COMPOSED_FACE_DISAGREEMENT",)
                    supported.discard(member.member_id)
                    unevaluable.discard(member.member_id)
                    opposed.add(member.member_id)
                    assessment_reasons.add(f"DIRECT_COMPOSED_DISAGREEMENT.{member.member_id}")
                elif PropertyPathConsistency.UNEVALUABLE in dispositions:
                    consistency = PropertyPathConsistency.UNEVALUABLE
                    consistency_reasons = ("DIRECT_OR_COMPOSED_PATH_UNEVALUABLE",)
                    if member.member_id not in opposed:
                        supported.discard(member.member_id)
                        unevaluable.add(member.member_id)
                elif dispositions == {PropertyPathConsistency.CONSISTENT}:
                    consistency = PropertyPathConsistency.CONSISTENT
                    consistency_reasons = ()
                elif dispositions == {PropertyPathConsistency.NOT_APPLICABLE}:
                    consistency = PropertyPathConsistency.NOT_APPLICABLE
                    consistency_reasons = ("COMPOSED_PATH_NOT_DECLARED",)
                else:
                    raise PropertySurvivalError("PROPERTY_PATH_MEMBER_HAS_NO_TERMINAL")
            path_values.append(
                PropertyPathConsistencyCell(
                    member_id=member.member_id,
                    disposition=consistency,
                    reason_codes=consistency_reasons,
                )
            )

        signature = PropertySurvivalSignature(
            signature_id=f"face-paired-property-survival-signature.{spec.spec_id}",
            supported_member_ids=tuple(sorted(supported)),
            opposed_member_ids=tuple(sorted(opposed)),
            unevaluable_member_ids=tuple(sorted(unevaluable)),
            not_applicable_member_ids=tuple(sorted(not_applicable)),
        )
        return FacePairedPropertySurvivalAssessment(
            assessment_id=f"face-paired-property-survival-assessment.{spec.spec_id}",
            property_assessment=property_assessment,
            path_correspondences=tuple(
                sorted(path_correspondences, key=lambda value: value.correspondence_id)
            ),
            path_face_consistency=tuple(sorted(face_cells, key=lambda value: value.cell_id)),
            signature=signature,
            path_consistency=tuple(sorted(path_values, key=lambda value: value.member_id)),
            reason_codes=tuple(sorted(assessment_reasons)),
            response_law_produced=False,
            coefficient_or_sample_pooling_performed=False,
        )

    def _require_method(self, evidence: PropertySurvivalCellEvidence) -> None:
        try:
            manifest = self.capability_registry.resolve(
                evidence.method.capability_key,
                evidence.method.capability_version,
            )
        except KeyError as error:
            raise PropertySurvivalError("PROPERTY_METHOD_UNREGISTERED") from error
        if (
            manifest.implementation_sha256 != evidence.method.implementation_sha256
            or manifest.config_schema != evidence.method.config.object_schema
        ):
            raise PropertySurvivalError("PROPERTY_METHOD_BINDING_DRIFT")


__all__ = ["PropertySurvivalError", "PropertySurvivalService"]
