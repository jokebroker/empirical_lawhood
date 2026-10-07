"Finite-certificate inputs to the existing admission owner.\n\nEvery gate and utility retains its numerical-view binding. This avoids reducing\none favorable view into a purported all-view certificate. Views remain nested\ncomputations over the same acquisition, never independent evidence units.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionGateKind, AdmissionSet
from empirical_lawhood.kernel.models import ModelSetSpec
from empirical_lawhood.kernel.status import AdmissionStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_geometry import AdmissionCoordinateGateReceipt, GatePredicateKind, ReceiptAdmissionAdmissionCandidateCell, ReceiptAdmissionActionFibre, ReceiptAdmissionSupportCell, ReceiptAdmissionRawDisposition, ReceiptAdmissionReceiptProductionPlan, UtilityEvaluationReceipt, _derive_receipt_admission_set, _exact_evidence_map, admission_coordinate_outside_support, validate_admission_admission_inputs, validate_admission_gate_and_utility_receipts
from empirical_lawhood.planning.finite_response_geometry import FiniteReachabilityMethodReceipt, FiniteCertificateStatus, FiniteSetDisposition
from empirical_lawhood.planning.geometry import AdmissionComparison


@dataclass(frozen=True, slots=True)
class FiniteCertificateAdmissionReceiptCorpus(CanonicalRecord):
    "Complete finite admission corpus over member × version × action × support × view.\n\n    There is one finite certificate per planned coordinate and one receipt per\n    gate/view and utility/view. The certificate itself contains every view.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-certificate-admission-receipt-corpus'

    corpus_id: str
    plan: ReceiptAdmissionReceiptProductionPlan
    gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...]
    reachability_receipts: tuple[FiniteReachabilityMethodReceipt, ...]
    utility_receipts: tuple[UtilityEvaluationReceipt, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.corpus_id, field_name="corpus_id")
        for name, values in (
            ("gate_receipts", self.gate_receipts),
            ("reachability_receipts", self.reachability_receipts),
            ("utility_receipts", self.utility_receipts),
        ):
            require_sorted_unique_ids(values, attribute="receipt_id", field_name=name)
        coordinates = {c.coordinate_id: c for c in self.plan.coordinates}
        expected_views = {
            (c.coordinate_id, v) for c in self.plan.coordinates for v in c.qualification_view_ids
        }
        expected_gates = {(c, kind, v) for c, v in expected_views for kind in AdmissionGateKind}
        actual_gates = {
            (
                r.planned_coordinate.coordinate_id,
                r.predicate.gate_kind,
                r.law_evaluation_binding.qualification_view_id,
            )
            for r in self.gate_receipts
        }
        if actual_gates != expected_gates or len(self.gate_receipts) != len(expected_gates):
            raise ValueError("finite admission gate corpus must cover every planned coordinate/gate/view")
        actual_utility = {
            (r.planned_coordinate.coordinate_id, r.law_evaluation_binding.qualification_view_id)
            for r in self.utility_receipts
        }
        if actual_utility != expected_views or len(self.utility_receipts) != len(expected_views):
            raise ValueError("finite admission utility corpus must cover every planned coordinate/view")
        if len(self.reachability_receipts) != len(coordinates) or {
            r.product_key for r in self.reachability_receipts
        } != set(coordinates):
            raise ValueError("finite admission certificate corpus is incomplete or contains extras")
        if any(
            not isinstance(r, FiniteReachabilityMethodReceipt) for r in self.reachability_receipts
        ):
            raise ValueError("controlled-map receipts cannot substitute for finite certificates")
        finite_bindings = tuple(
            b
            for r in self.reachability_receipts
            for b in r.request.response_set.evaluation_bindings
        )
        validate_admission_gate_and_utility_receipts(
            self.plan,
            self.gate_receipts,
            self.utility_receipts,
            finite_evaluation_bindings=finite_bindings,
        )
        expected_words = tuple(
            sorted(
                (a.action_word_identity for a in self.plan.action_fibres),
                key=lambda word: word.object_id,
            )
        )
        calibration_by_slice: dict[tuple[str, str, ObjectIdentity], ObjectIdentity] = {}
        first_request = self.reachability_receipts[0].request
        for receipt in self.reachability_receipts:
            if not isinstance(receipt, FiniteReachabilityMethodReceipt):
                raise ValueError(
                    "controlled-map receipts cannot substitute for finite certificates"
                )
            coordinate = coordinates[receipt.product_key]
            if (
                receipt.planned_coordinate != coordinate
                or receipt.information_cutoff != self.plan.information_cutoff
                or receipt.outcome_access is not self.plan.outcome_access
                or receipt.horizon != self.plan.horizon
                or receipt.constraint_ids != self.plan.reachability_constraint_ids
                or receipt.evidence_domain != self.plan.evidence_domain
                or receipt.authority_boundary != self.plan.authority_boundary
                or receipt.producer != self.plan.reachability_receipt_producer
                or receipt.resource_envelope != self.plan.reachability_resource_envelope
                or receipt.request.action_word
                != self.plan.action_fibre(coordinate.action_fibre).action_word
            ):
                raise ValueError("finite admission certificate rewrites its frozen plan")
            source = receipt.request.response_set
            law = next(
                law
                for law in self.plan.atlas.laws
                if ObjectIdentity.from_record(law.law_id, law) == coordinate.response_law
            )
            if source.law_payload != law.evaluator.payload:
                raise ValueError("finite set rewrites its qualified law payload")
            if source.joint_calibration.action_words != expected_words:
                raise ValueError(
                    "finite calibration omits part of the planned native action vector"
                )
            if (
                receipt.request.task != first_request.task
                or source.public_handoff != first_request.response_set.public_handoff
                or source.frame_translation != first_request.response_set.frame_translation
            ):
                raise ValueError("finite admission coordinates disagree on the public handoff or task")
            calibration = ObjectIdentity.from_record(
                source.joint_calibration.calibration_id, source.joint_calibration
            )
            slice_key = (
                coordinate.denominator_member_id,
                coordinate.candidate_version_id,
                coordinate.support_cell,
            )
            if calibration_by_slice.setdefault(slice_key, calibration) != calibration:
                raise ValueError("finite action slices do not share the same joint calibration")
            outside = admission_coordinate_outside_support(
                self.plan, coordinate, finite_evaluation_bindings=source.evaluation_bindings
            )
            if outside != (source.disposition is FiniteSetDisposition.OUTSIDE_SUPPORT):
                raise ValueError(
                    "finite admission set outside-support refusal differs from exact law support"
                )
            bindings = {b.qualification_view_id: b for b in source.evaluation_bindings}
            gates = tuple(r for r in self.gate_receipts if r.planned_coordinate == coordinate)
            utilities = tuple(
                r for r in self.utility_receipts if r.planned_coordinate == coordinate
            )
            view_receipts: tuple[AdmissionCoordinateGateReceipt | UtilityEvaluationReceipt, ...] = (
                *gates,
                *utilities,
            )
            if any(
                r.law_evaluation_binding != bindings[r.law_evaluation_binding.qualification_view_id]
                for r in view_receipts
            ):
                raise ValueError(
                    "finite admission gate/utility uses another exact per-view law evaluation"
                )
            direction_ids = tuple(
                sorted(
                    r.receipt_id
                    for r in gates
                    if r.predicate.gate_kind is AdmissionGateKind.REACHABILITY
                )
            )
            if receipt.admission_direction_gate_receipt_ids != direction_ids:
                raise ValueError(
                    "finite certificate omits or rewrites a numerical-view direction gate"
                )
            for gate in gates:
                if gate.predicate.gate_kind is AdmissionGateKind.TARGET:
                    _validate_finite_target_gate(receipt, gate)
        self._validate_evidence()

    def _validate_evidence(self) -> None:
        require_sorted_unique_ids(
            self.input_artifacts, attribute="artifact_id", field_name="input_artifacts"
        )
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if any(link.world_id != self.plan.atlas.world_id for link in self.evidence_links):
            raise ValueError("finite admission evidence belongs to another evidence world")
        nested_artifacts: dict[str, ArtifactIdentity] = {}
        receipts: tuple[
            AdmissionCoordinateGateReceipt
            | FiniteReachabilityMethodReceipt
            | UtilityEvaluationReceipt,
            ...,
        ] = (*self.gate_receipts, *self.reachability_receipts, *self.utility_receipts)
        for receipt in receipts:
            for artifact in receipt.input_artifacts:
                if nested_artifacts.setdefault(artifact.artifact_id, artifact) != artifact:
                    raise ValueError("finite admission receipts reuse an artifact ID with different bytes")
        if {a.artifact_id: a for a in self.input_artifacts} != nested_artifacts:
            raise ValueError("finite admission corpus artifacts differ from its exact receipts")
        links = tuple(link for receipt in receipts for link in receipt.evidence_links)
        if _exact_evidence_map(self.evidence_links) != _exact_evidence_map(links):
            raise ValueError("finite admission corpus evidence differs from its exact receipts")


def _validate_finite_target_gate(
    receipt: FiniteReachabilityMethodReceipt,
    gate: AdmissionCoordinateGateReceipt,
) -> None:
    source = receipt.request.response_set
    if source.disposition is not FiniteSetDisposition.AVAILABLE:
        if gate.disposition is ReceiptAdmissionRawDisposition.EVALUATED:
            raise ValueError("unavailable finite set cannot supply an evaluated target gate")
        return
    view = gate.law_evaluation_binding.qualification_view_id
    margins = tuple(m for m in receipt.margins if m.qualification_view_id == view)
    units = {m.coordinate.unit for m in margins}
    if len(units) != 1:
        raise ValueError(
            "finite target gate requires a declared composite normalization for unlike units"
        )
    # A union is existential over components, each of which is conjunctive over
    # native faces. The admission owner subsequently intersects every view.
    margin = max(
        min(
            min(m.lower_face_margin, m.upper_face_margin) for m in margins if m.box_id == box.box_id
        )
        for box in receipt.request.task.boxes
    )
    predicate = gate.predicate
    if (
        predicate.predicate_kind is not GatePredicateKind.SCALAR_AT_LEAST
        or predicate.lower is None
        or predicate.lower.value != Decimal(0)
        or predicate.lower.unit not in units
        or gate.disposition is not ReceiptAdmissionRawDisposition.EVALUATED
        or gate.observed_scalar is None
        or gate.observed_scalar.value != margin
        or gate.observed_scalar.unit not in units
    ):
        raise ValueError("finite target gate is not the complete native-set containment margin")
    required = (
        *source.source_artifacts,
        source.law_payload,
        source.joint_calibration.calibration_artifact,
    )
    artifacts = {a.artifact_id: a for a in gate.input_artifacts}
    if any(artifacts.get(a.artifact_id) != a for a in required):
        raise ValueError("finite target gate loses its authenticated joint-set derivation")


@dataclass(frozen=True, slots=True)
class FiniteCertificateAdmissionSpec(CanonicalRecord):
    """Same noncompensating admission owner, with a complete finite-certificate corpus."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-certificate-admission-spec'

    evaluation_id: str
    corpus: FiniteCertificateAdmissionReceiptCorpus
    nominal_denominator_member_id: str
    receiver_quantity_ids: tuple[str, ...]
    candidate_cells: tuple[ReceiptAdmissionAdmissionCandidateCell, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_admission_admission_inputs(self)


def derive_finite_certificate_admission_comparison(spec: FiniteCertificateAdmissionSpec) -> AdmissionComparison:
    """Use the existing gate intersection unchanged, including every required view."""

    nominal = _derive_receipt_admission_set(
        spec,
        member_ids=(spec.nominal_denominator_member_id,),
        suffix="nominal",
    )
    robust = _derive_receipt_admission_set(
        spec,
        member_ids=tuple(m.denominator_member_id for m in spec.corpus.plan.model_set.members),
        suffix="robust",
    )
    disagreement = tuple(sorted(set(nominal.admitted_cell_ids) ^ set(robust.admitted_cell_ids)))
    return AdmissionComparison(
        comparison_id=f"finite-certificate-admission-comparison.{spec.evaluation_id}",
        evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
        nominal=nominal,
        robust=robust,
        structurally_stable=not disagreement,
        disagreement_cell_ids=disagreement,
    )


@dataclass(frozen=True, slots=True)
class FiniteCoordinateCertificate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-coordinate-certificate'

    coordinate_id: str
    receipt: ObjectIdentity
    status: FiniteCertificateStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if self.receipt.object_schema != FiniteReachabilityMethodReceipt.SCHEMA:
            raise ValueError("finite coordinate requires a finite receipt, not map diagnostics")
        if not isinstance(self.status, FiniteCertificateStatus):
            raise ValueError("finite coordinate requires a typed certificate status")


@dataclass(frozen=True, slots=True)
class FiniteAdmissionCellCertification(CanonicalRecord):
    """Admission plus every exact member/version certificate; no rank surrogate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-admission-cell-certification'

    candidate_cell_id: str
    action_fibre: ObjectIdentity
    support_cell: ObjectIdentity
    admitted: bool
    certificates: tuple[FiniteCoordinateCertificate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_cell_id, field_name="candidate_cell_id")
        if self.action_fibre.object_schema != ReceiptAdmissionActionFibre.SCHEMA:
            raise ValueError("finite reachability requires an exact action fibre")
        if self.support_cell.object_schema != ReceiptAdmissionSupportCell.SCHEMA:
            raise ValueError("finite reachability requires an exact support cell")
        if type(self.admitted) is not bool or not self.certificates:
            raise ValueError("finite reachability requires admission and nonempty certificates")
        require_sorted_unique_ids(
            self.certificates, attribute="coordinate_id", field_name="certificates"
        )

    @property
    def certified(self) -> bool:
        return self.admitted and all(
            c.status is FiniteCertificateStatus.CERTIFIED for c in self.certificates
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *(
                        f"FINITE_{c.status.value}"
                        for c in self.certificates
                        if c.status is not FiniteCertificateStatus.CERTIFIED
                    ),
                    *(("ADMISSION_NOT_ADMITTED",) if not self.admitted else ()),
                }
            )
        )


class FiniteReachabilitySetStatus(StrEnum):
    ALL_ADMITTED_CERTIFIED = "ALL_ADMITTED_CERTIFIED"
    PARTIAL = "PARTIAL"
    NO_CERTIFIED_CELL = "NO_CERTIFIED_CELL"
    NO_ADMITTED_CELL = "NO_ADMITTED_CELL"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class FiniteAdmissionCertification(CanonicalRecord):
    """Finite admission-set certification, explicitly distinct from map reachability."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-admission-certification'

    reachability_id: str
    admission: ObjectIdentity
    corpus: ObjectIdentity
    model_set: ObjectIdentity
    denominator_member_ids: tuple[str, ...]
    candidate_version_ids: tuple[str, ...]
    cells: tuple[FiniteAdmissionCellCertification, ...]
    admission_status: AdmissionStatus
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.reachability_id, field_name="reachability_id")
        if (
            self.admission.object_schema != AdmissionSet.SCHEMA
            or self.corpus.object_schema != FiniteCertificateAdmissionReceiptCorpus.SCHEMA
        ):
            raise ValueError("finite reachability requires its exact admission and finite corpus")
        if self.model_set.object_schema != ModelSetSpec.SCHEMA:
            raise ValueError("finite reachability requires an exact model set")
        for name, values in (
            ("denominator_member_ids", self.denominator_member_ids),
            ("candidate_version_ids", self.candidate_version_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(self.cells, attribute="candidate_cell_id", field_name="cells")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if not self.cells or self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("finite reachability requires nonempty admission cells")
        inherited = VisibilityCeiling.most_restrictive(
            *(link.visibility_ceiling for link in self.evidence_links)
        )
        if (
            not self.visibility_ceiling.is_promotable
            or not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited)
        ):
            raise ValueError("finite reachability cannot lower evidence visibility")
        if not isinstance(self.admission_status, AdmissionStatus):
            raise ValueError("finite reachability requires a typed admission status")

    @property
    def certified_cell_ids(self) -> tuple[str, ...]:
        return tuple(c.candidate_cell_id for c in self.cells if c.certified)

    @property
    def status(self) -> FiniteReachabilitySetStatus:
        admitted = tuple(c for c in self.cells if c.admitted)
        certified = self.certified_cell_ids
        if not admitted:
            return (
                FiniteReachabilitySetStatus.UNEVALUABLE
                if self.admission_status is AdmissionStatus.UNEVALUABLE
                else FiniteReachabilitySetStatus.NO_ADMITTED_CELL
            )
        if len(certified) == len(admitted):
            return FiniteReachabilitySetStatus.ALL_ADMITTED_CERTIFIED
        if certified:
            return FiniteReachabilitySetStatus.PARTIAL
        if any(
            c.status is FiniteCertificateStatus.UNEVALUABLE
            for cell in admitted
            for c in cell.certificates
        ):
            return FiniteReachabilitySetStatus.UNEVALUABLE
        return FiniteReachabilitySetStatus.NO_CERTIFIED_CELL


@dataclass(frozen=True, slots=True)
class FiniteCertificateReachabilityComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-certificate-reachability-comparison'

    comparison_id: str
    evaluation: ObjectIdentity
    nominal: FiniteAdmissionCertification
    robust: FiniteAdmissionCertification

    def __post_init__(self) -> None:
        validate_stable_id(self.comparison_id, field_name="comparison_id")
        if self.evaluation.object_schema != FiniteCertificateReachabilitySpec.SCHEMA:
            raise ValueError("finite reachability comparison requires its finite evaluation spec")
        if (
            self.nominal.corpus != self.robust.corpus
            or self.nominal.model_set != self.robust.model_set
        ):
            raise ValueError(
                "finite nominal and robust results must use one exact corpus/model set"
            )
        if not set(self.nominal.denominator_member_ids) <= set(self.robust.denominator_member_ids):
            raise ValueError("finite nominal denominator is outside its robust roster")
        if set(self.robust.certified_cell_ids) - set(self.nominal.certified_cell_ids):
            raise ValueError("robust intersection cannot add a finite word absent from nominal")

    @property
    def disagreement_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.nominal.certified_cell_ids) ^ set(self.robust.certified_cell_ids))
        )

    @property
    def finite_admissible_set_stable(self) -> bool:
        return not self.disagreement_cell_ids


@dataclass(frozen=True, slots=True)
class FiniteCertificateReachabilitySpec(CanonicalRecord):
    """Use finite certification and noncompensating admission from one exact corpus."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-certificate-reachability-spec'

    evaluation_id: str
    admission_spec: FiniteCertificateAdmissionSpec
    admission: AdmissionComparison
    initial_set_id: str
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.initial_set_id, field_name="initial_set_id")
        if self.admission != derive_finite_certificate_admission_comparison(self.admission_spec):
            raise ValueError("finite reachability admission is not derived from its exact corpus")
        if self.initial_set_id != self.admission_spec.corpus.plan.initial_set_id:
            raise ValueError("finite reachability rewrites its frozen initial set")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("finite reachability must retain its admission ceiling")
        inherited = VisibilityCeiling.most_restrictive(
            self.admission_spec.visibility_ceiling,
            self.admission.nominal.visibility_ceiling,
            self.admission.robust.visibility_ceiling,
        )
        if (
            not self.visibility_ceiling.is_promotable
            or not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited)
        ):
            raise ValueError("finite reachability cannot lower admission visibility")


def _derive_finite_reachability_set(
    spec: FiniteCertificateReachabilitySpec,
    *,
    admission: AdmissionSet,
    member_ids: tuple[str, ...],
    suffix: str,
) -> FiniteAdmissionCertification:
    corpus = spec.admission_spec.corpus
    coordinates = {c.coordinate_id: c for c in corpus.plan.coordinates}
    receipt_by_coordinate = {r.product_key: r for r in corpus.reachability_receipts}
    admission_cells = {c.cell_id: c for c in admission.cells}
    cells = []
    for candidate in spec.admission_spec.candidate_cells:
        selected = tuple(
            c
            for c in candidate.planned_coordinate_ids
            if coordinates[c].denominator_member_id in member_ids
        )
        cells.append(
            FiniteAdmissionCellCertification(
                candidate.candidate_cell_id,
                candidate.action_fibre,
                candidate.support_cell,
                admission_cells[candidate.candidate_cell_id].admitted,
                tuple(
                    FiniteCoordinateCertificate(
                        c,
                        ObjectIdentity.from_record(
                            receipt_by_coordinate[c].receipt_id, receipt_by_coordinate[c]
                        ),
                        receipt_by_coordinate[c].status,
                    )
                    for c in selected
                ),
            )
        )
    versions = tuple(
        sorted(
            {
                v
                for member in corpus.plan.model_set.members
                if member.denominator_member_id in member_ids
                for v in member.candidate_version_ids
            }
        )
    )
    return FiniteAdmissionCertification(
        reachability_id=f"finite-certificate-reachability.{spec.evaluation_id}.{suffix}",
        admission=ObjectIdentity.from_record(admission.admission_id, admission),
        corpus=ObjectIdentity.from_record(corpus.corpus_id, corpus),
        model_set=ObjectIdentity.from_record(
            corpus.plan.model_set.model_set_id, corpus.plan.model_set
        ),
        denominator_member_ids=tuple(sorted(set(member_ids))),
        candidate_version_ids=versions,
        cells=tuple(cells),
        admission_status=admission.status,
        evidence_links=corpus.evidence_links,
        evidence_ceiling=spec.evidence_ceiling,
        visibility_ceiling=spec.visibility_ceiling,
    )


def derive_finite_certificate_reachability_comparison(
    spec: FiniteCertificateReachabilitySpec,
) -> FiniteCertificateReachabilityComparison:
    return FiniteCertificateReachabilityComparison(
        comparison_id=f"finite-certificate-reachability-comparison.{spec.evaluation_id}",
        evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
        nominal=_derive_finite_reachability_set(
            spec,
            admission=spec.admission.nominal,
            member_ids=(spec.admission_spec.nominal_denominator_member_id,),
            suffix="nominal",
        ),
        robust=_derive_finite_reachability_set(
            spec,
            admission=spec.admission.robust,
            member_ids=tuple(
                m.denominator_member_id for m in spec.admission_spec.corpus.plan.model_set.members
            ),
            suffix="robust",
        ),
    )
