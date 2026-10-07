"""Evidence-derived authoring gate for the sole current controller programme."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

from empirical_lawhood.kernel.admission import ReachabilityStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy, DeliveryControllerStudy, LeastMagnitudeControllerStudy, RequestedWordMagnitude, ActHoldControllerSynthesisPlan, ControllerInstanceBinding, DeliveryEquivalenceSpec, StageAwareDeliveryEquivalenceSpec, AdmissionControllerSynthesisPlan, ImplementationBinding, AdmissionMeasuredHoldFibre, ControllerActionBinding
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionSpec, ControlledMapReachabilitySpec, derive_receipt_admission_comparison, derive_controlled_map_reachability_comparison


@dataclass(frozen=True, slots=True)
class AdmissionControllerAuthoringRequest(CanonicalRecord):
    """Status-free authored inputs whose evidence is rederived before construction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/admission-controller-authoring-request'

    request_id: str
    study_id: str
    compiler_release_id: str
    system: SystemSpec
    action_bindings: tuple[ControllerActionBinding, ...]
    admission: ReceiptAdmissionSpec
    reachability: ControlledMapReachabilitySpec
    synthesis: AdmissionControllerSynthesisPlan
    implementations: tuple[ImplementationBinding, ...]
    measured_hold_fibre: AdmissionMeasuredHoldFibre | None
    prospective_evaluation: ObjectIdentity | None

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ('study_id', self.study_id),
            ("compiler_release_id", self.compiler_release_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.action_bindings,
            attribute="action_binding_id",
            field_name="action_bindings",
        )
        require_sorted_unique_ids(
            self.implementations,
            attribute="binding_id",
            field_name="implementations",
        )


@dataclass(frozen=True, slots=True)
class AdmissionControllerAuthor:
    "Construct an admission controller study only after exact local-law, admission, reachability and HOLD checks pass."

    def author(self, request: AdmissionControllerAuthoringRequest) -> AdmissionControllerStudy:
        atlas = request.admission.corpus.plan.atlas
        if not atlas.laws:
            raise ValueError("evidence-derived programme requires a supported nonempty atlas")
        admission = derive_receipt_admission_comparison(request.admission)
        if (
            admission.nominal.status is not AdmissionStatus.ADMITTED
            or admission.robust.status is not AdmissionStatus.ADMITTED
            or not admission.structurally_stable
        ):
            raise ValueError("evidence-derived programme requires stable nominal/robust admission")
        if request.reachability.admission_spec != request.admission:
            raise ValueError("evidence-derived programme reachability uses another admission corpus")
        reachability = derive_controlled_map_reachability_comparison(request.reachability)
        if (
            reachability.nominal.status is not ReachabilityStatus.REACHABLE
            or reachability.robust.status is not ReachabilityStatus.REACHABLE
            or not reachability.structurally_stable
        ):
            raise ValueError("evidence-derived programme requires stable controlled reachability")
        hold = request.measured_hold_fibre
        if hold is None:
            raise ValueError("evidence-derived programme requires a measured supported HOLD")
        hold_cell_id = hold.admission_candidate_cell.object_id
        if (
            hold_cell_id not in admission.nominal.admitted_cell_ids
            or hold_cell_id not in admission.robust.admitted_cell_ids
            or hold_cell_id not in reachability.nominal.reachable_cell_ids
            or hold_cell_id not in reachability.robust.reachable_cell_ids
        ):
            raise ValueError("measured HOLD is not admitted and reachable on its own admission fibre")
        if not request.synthesis.candidate_chart.candidates:
            raise ValueError("evidence-derived programme requires a finite native action chart")
        return AdmissionControllerStudy(
            study_id=request.study_id,
            compiler_release_id=request.compiler_release_id,
            system=request.system,
            action_bindings=request.action_bindings,
            admission=request.admission,
            reachability=request.reachability,
            synthesis=request.synthesis,
            implementations=request.implementations,
            measured_hold_fibre=hold,
            prospective_evaluation=request.prospective_evaluation,
            compilation_ceiling=EvidenceCeiling.ADMISSION,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )


@dataclass(frozen=True, slots=True)
class DeliveryControllerAuthoringRequest(CanonicalRecord):
    """Outcome-blind finite inputs, including empty or disagreeing admissible sets."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/delivery-controller-authoring-request'

    request_id: str
    study_id: str
    compiler_release_id: str
    system: SystemSpec
    action_bindings: tuple[ControllerActionBinding, ...]
    admission: FiniteCertificateAdmissionSpec
    reachability: FiniteCertificateReachabilitySpec
    synthesis: ActHoldControllerSynthesisPlan
    implementations: tuple[ImplementationBinding, ...]
    measured_hold_fibre: AdmissionMeasuredHoldFibre | None
    delivery_equivalence: DeliveryEquivalenceSpec | StageAwareDeliveryEquivalenceSpec
    allow_fallback_hold: bool
    instance_binding: ControllerInstanceBinding
    prospective_evaluation: ObjectIdentity | None

    def __post_init__(self) -> None:
        for name in ("request_id", 'study_id', "compiler_release_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.action_bindings, attribute="action_binding_id", field_name="action_bindings"
        )
        require_sorted_unique_ids(
            self.implementations, attribute="binding_id", field_name="implementations"
        )


class DeliveryControllerAuthor:
    """Retain complete qualified lineage; let the sole compiler decide ACT/HOLD/NONATTEMPT."""

    capability_key = "control.evidence-derived-programme-author"
    capability_version = "1.0.0"

    @staticmethod
    def author(request: DeliveryControllerAuthoringRequest) -> DeliveryControllerStudy:
        return DeliveryControllerStudy(
            study_id=request.study_id,
            compiler_release_id=request.compiler_release_id,
            system=request.system,
            action_bindings=request.action_bindings,
            admission=request.admission,
            reachability=request.reachability,
            synthesis=request.synthesis,
            implementations=request.implementations,
            measured_hold_fibre=request.measured_hold_fibre,
            delivery_equivalence=request.delivery_equivalence,
            allow_fallback_hold=request.allow_fallback_hold,
            instance_binding=request.instance_binding,
            prospective_evaluation=request.prospective_evaluation,
            compilation_ceiling=EvidenceCeiling.ADMISSION,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.most_restrictive(
                request.admission.visibility_ceiling,
                request.reachability.visibility_ceiling,
                request.synthesis.visibility_ceiling,
            ),
        )

    @staticmethod
    def author_native_magnitude(
        request: DeliveryControllerAuthoringRequest,
        *,
        native_magnitudes: tuple[RequestedWordMagnitude, ...],
    ) -> LeastMagnitudeControllerStudy:
        "Explicit native-magnitude ordering authoring map, before any programme issue/commitment.\n\n        Reuse the complete existing evidence checks; the new record separately\n        validates every magnitude against its exact requested native word.\n        "
        checked = DeliveryControllerAuthor.author(request)
        return LeastMagnitudeControllerStudy(
            **{f.name: getattr(checked, f.name) for f in fields(DeliveryControllerStudy)},
            native_magnitudes=native_magnitudes,
        )


__all__ = [
    'AdmissionControllerAuthor',
    'AdmissionControllerAuthoringRequest',
]
