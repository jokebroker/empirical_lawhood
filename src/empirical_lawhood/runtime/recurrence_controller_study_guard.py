"Candidate-family recurrence continuity through admission, programme, compiler and tick."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy
from empirical_lawhood.planning.evidence_geometry import ControlledMapAdmissionReceiptCorpus, ReceiptAdmissionReceiptProductionPlan
from empirical_lawhood.runtime.adaptive_acquisition_validation import ActionPreparationRecurrenceQualificationBinding, RecurrenceQualificationDisposition
from empirical_lawhood.runtime.controller_compiler import CompiledAdmissionControllerStudy
from empirical_lawhood.runtime.controller_runtime import AdmissionControllerTickReceipt, RuntimeObservation
from empirical_lawhood.runtime.gate_margin_projection import CertifiedAdmissionMarginCorpusUse


def _binding_matches_coordinate(
    binding: ActionPreparationRecurrenceQualificationBinding,
    *,
    action_word: ObjectIdentity,
    denominator_member_id: str,
    qualification_result: ObjectIdentity,
    response_law: ObjectIdentity,
) -> bool:
    result = binding.qualification_result
    requirement = binding.requirement
    if (
        binding.disposition is not RecurrenceQualificationDisposition.FINALIZED_SUPPORTED_LAW
        or result is None
        or binding.response_law is None
    ):
        return False
    return (
        action_word in (*requirement.active_action_words, requirement.matched_hold_action_word)
        and denominator_member_id in requirement.model_member_ids
        and qualification_result == ObjectIdentity.from_record(result.result_id, result)
        and response_law == binding.response_law
    )


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceAdmissionUse(CanonicalRecord):
    "Complete admission coordinate roster authorized by exact recurrence-qualified laws."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-preparation-recurrence-admission-use'

    use_id: str
    qualification_bindings: tuple[ActionPreparationRecurrenceQualificationBinding, ...]
    admission_receipt_production_plan: ReceiptAdmissionReceiptProductionPlan
    covered_coordinate_ids: tuple[str, ...]
    recurrence_receipt_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.use_id, field_name="use_id")
        require_sorted_unique_ids(
            self.qualification_bindings,
            attribute="binding_id",
            field_name="qualification_bindings",
        )
        if not self.qualification_bindings or any(
            value.disposition is not RecurrenceQualificationDisposition.FINALIZED_SUPPORTED_LAW
            for value in self.qualification_bindings
        ):
            raise ValueError("admission use requires recurrence-qualified supported laws")
        require_sorted_unique_strings(
            self.covered_coordinate_ids,
            field_name="covered_coordinate_ids",
            allow_empty=False,
        )
        expected_coordinates = tuple(
            sorted(value.coordinate_id for value in self.admission_receipt_production_plan.coordinates)
        )
        if self.covered_coordinate_ids != expected_coordinates:
            raise ValueError("recurrence admission use omits or adds a planned coordinate")
        require_sorted_unique_strings(
            self.recurrence_receipt_fingerprints,
            field_name="recurrence_receipt_fingerprints",
            allow_empty=False,
        )
        expected_fingerprints = tuple(
            sorted(
                {
                    fingerprint
                    for binding in self.qualification_bindings
                    for fingerprint in binding.recurrence_receipt_fingerprints
                }
            )
        )
        if self.recurrence_receipt_fingerprints != expected_fingerprints:
            raise ValueError("admission use changes its recurrence receipt digests")
        for coordinate in self.admission_receipt_production_plan.coordinates:
            action_word = self.admission_receipt_production_plan.action_fibre(coordinate.action_fibre).action_word_identity
            matches = tuple(
                binding
                for binding in self.qualification_bindings
                if _binding_matches_coordinate(
                    binding,
                    action_word=action_word,
                    denominator_member_id=coordinate.denominator_member_id,
                    qualification_result=coordinate.qualification_result,
                    response_law=coordinate.response_law,
                )
            )
            if len(matches) != 1:
                raise ValueError("admission coordinate lacks one exact recurrence-qualified law binding")


def bind_action_preparation_recurrence_admission_use(
    *,
    use_id: str,
    qualification_bindings: tuple[ActionPreparationRecurrenceQualificationBinding, ...],
    admission_receipt_production_plan: ReceiptAdmissionReceiptProductionPlan,
) -> ActionPreparationRecurrenceAdmissionUse:
    bindings = tuple(sorted(qualification_bindings, key=lambda value: value.binding_id))
    return ActionPreparationRecurrenceAdmissionUse(
        use_id=use_id,
        qualification_bindings=bindings,
        admission_receipt_production_plan=admission_receipt_production_plan,
        covered_coordinate_ids=tuple(sorted(value.coordinate_id for value in admission_receipt_production_plan.coordinates)),
        recurrence_receipt_fingerprints=tuple(
            sorted(
                {
                    fingerprint
                    for binding in bindings
                    for fingerprint in binding.recurrence_receipt_fingerprints
                }
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceControllerStudyUse(CanonicalRecord):
    "Programme-level proof that every active candidate remains bound to candidate-family recurrence."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-preparation-recurrence-controller-study-use'

    use_id: str
    recurrence_admission_use: ActionPreparationRecurrenceAdmissionUse
    certified_margin_use: CertifiedAdmissionMarginCorpusUse
    admission_corpus: ControlledMapAdmissionReceiptCorpus
    study: AdmissionControllerStudy
    active_action_binding_ids: tuple[str, ...]
    recurrence_receipt_fingerprints: tuple[str, ...]
    margin_receipt_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.use_id, field_name="use_id")
        if (
            self.admission_corpus.plan != self.recurrence_admission_use.admission_receipt_production_plan
            or self.certified_margin_use.admission_corpus != self.admission_corpus
            or self.study.admission.corpus != self.admission_corpus
        ):
            raise ValueError(
                "guarded programme use changes its recurrence/margin admission plan or corpus"
            )
        require_sorted_unique_strings(
            self.active_action_binding_ids,
            field_name="active_action_binding_ids",
            allow_empty=False,
        )
        cells = {
            ObjectIdentity.from_record(value.candidate_cell_id, value): value
            for value in self.study.admission.candidate_cells
        }
        expected_active = tuple(
            sorted(
                cells[value.admission_candidate_cell].action_fibre.object_id
                for value in self.study.synthesis.candidate_chart.candidates
            )
        )
        if self.active_action_binding_ids != expected_active:
            raise ValueError("recurrence programme use changes the active candidate roster")
        active_words = {
            word
            for binding in self.recurrence_admission_use.qualification_bindings
            for word in binding.requirement.active_action_words
        }
        admission_action_fibres_by_id = {value.action_binding_id: value for value in self.recurrence_admission_use.admission_receipt_production_plan.action_fibres}
        if any(
            admission_action_fibres_by_id[action_id].action_word_identity not in active_words
            for action_id in self.active_action_binding_ids
        ):
            raise ValueError("active programme action lacks recurrence qualification")
        require_sorted_unique_strings(
            self.recurrence_receipt_fingerprints,
            field_name="recurrence_receipt_fingerprints",
            allow_empty=False,
        )
        if self.recurrence_receipt_fingerprints != self.recurrence_admission_use.recurrence_receipt_fingerprints:
            raise ValueError("programme use changes recurrence receipt digests")
        require_sorted_unique_strings(
            self.margin_receipt_fingerprints,
            field_name="margin_receipt_fingerprints",
            allow_empty=False,
        )
        if (
            self.margin_receipt_fingerprints
            != self.certified_margin_use.margin_receipt_fingerprints
        ):
            raise ValueError("programme use changes certified margin receipt digests")


def bind_action_preparation_recurrence_study_use(
    *,
    use_id: str,
    recurrence_admission_use: ActionPreparationRecurrenceAdmissionUse,
    certified_margin_use: CertifiedAdmissionMarginCorpusUse,
    admission_corpus: ControlledMapAdmissionReceiptCorpus,
    study: AdmissionControllerStudy,
) -> ActionPreparationRecurrenceControllerStudyUse:
    cells = {
        ObjectIdentity.from_record(value.candidate_cell_id, value): value
        for value in study.admission.candidate_cells
    }
    return ActionPreparationRecurrenceControllerStudyUse(
        use_id=use_id,
        recurrence_admission_use=recurrence_admission_use,
        certified_margin_use=certified_margin_use,
        admission_corpus=admission_corpus,
        study=study,
        active_action_binding_ids=tuple(
            sorted(
                cells[value.admission_candidate_cell].action_fibre.object_id
                for value in study.synthesis.candidate_chart.candidates
            )
        ),
        recurrence_receipt_fingerprints=recurrence_admission_use.recurrence_receipt_fingerprints,
        margin_receipt_fingerprints=(certified_margin_use.margin_receipt_fingerprints),
    )


@dataclass(frozen=True, slots=True)
class RecurrenceBoundCompiledController(CanonicalRecord):
    "Compiled admission controller result retaining its candidate-family recurrence study-use proof."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/recurrence-bound-compiled-controller'

    binding_id: str
    study_use: ActionPreparationRecurrenceControllerStudyUse
    compiled: CompiledAdmissionControllerStudy
    recurrence_receipt_fingerprints: tuple[str, ...]
    margin_receipt_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        require_sorted_unique_strings(
            self.recurrence_receipt_fingerprints,
            field_name="recurrence_receipt_fingerprints",
            allow_empty=False,
        )
        if (
            self.compiled.study != self.study_use.study
            or self.recurrence_receipt_fingerprints
            != self.study_use.recurrence_receipt_fingerprints
        ):
            raise ValueError("compiled controller breaks recurrence/programme continuity")
        require_sorted_unique_strings(
            self.margin_receipt_fingerprints,
            field_name="margin_receipt_fingerprints",
            allow_empty=False,
        )
        if self.margin_receipt_fingerprints != self.study_use.margin_receipt_fingerprints:
            raise ValueError("compiled controller breaks margin/programme continuity")


def compile_recurrence_guarded_study(
    *,
    compiler: ControllerStudyCompilePort,
    study_use: ActionPreparationRecurrenceControllerStudyUse,
) -> RecurrenceBoundCompiledController:
    compiled = compiler.compile(study_use.study)
    return RecurrenceBoundCompiledController(
        binding_id=f"recurrence-bound-compiled.{compiled.compiled_study_id}",
        study_use=study_use,
        compiled=compiled,
        recurrence_receipt_fingerprints=(study_use.recurrence_receipt_fingerprints),
        margin_receipt_fingerprints=study_use.margin_receipt_fingerprints,
    )


class ControllerStudyCompilePort(Protocol):
    "Closed sole-route compiler composition consumed by the candidate-family recurrence guard."

    def compile(self, study: AdmissionControllerStudy) -> CompiledAdmissionControllerStudy: ...


class ControllerStudyTickPort(Protocol):
    "Closed sole-route tick composition consumed by the candidate-family recurrence guard."

    def tick(
        self,
        compiled: CompiledAdmissionControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt: ...


class RecurrenceBoundControllerRuntime:
    """Thin guard that delegates every decision to the unchanged runtime tick."""

    def __init__(
        self,
        *,
        binding: RecurrenceBoundCompiledController,
        controller_route: ControllerStudyTickPort,
    ) -> None:
        self._binding = binding
        self._controller_route = controller_route

    @property
    def binding(self) -> RecurrenceBoundCompiledController:
        return self._binding

    def tick(
        self,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt:
        result = self._controller_route.tick(
            self._binding.compiled,
            observation,
            commitment_coordinate=commitment_coordinate,
        )
        if not isinstance(result, AdmissionControllerTickReceipt):  # pragma: no cover
            raise AssertionError("recurrence-bound admission controller runtime returned a replay receipt")
        return result


__all__ = [
    'ActionPreparationRecurrenceAdmissionUse',
    'ActionPreparationRecurrenceControllerStudyUse',
    'ControllerStudyCompilePort',
    'ControllerStudyTickPort',
    'RecurrenceBoundCompiledController',
    "RecurrenceBoundControllerRuntime",
    'bind_action_preparation_recurrence_admission_use',
    'bind_action_preparation_recurrence_study_use',
    'compile_recurrence_guarded_study',
]
