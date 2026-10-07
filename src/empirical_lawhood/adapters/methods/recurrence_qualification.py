"Action/preparation recurrence guard around the sole response-law finalizer."

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.adaptive_acquisition import ActionPreparationRecurrenceQualificationRequirement
from empirical_lawhood.runtime.adaptive_acquisition_validation import ActionPreparationRecurrenceQualificationBinding, ActionPreparationRecurrenceReceipt, bind_action_preparation_recurrence_qualification, recurrence_qualification_obstructions

from .contracts import ComponentUncertaintyFamilyAssessment
from .law_assessment import ResponseLawQualificationService


@dataclass(frozen=True, slots=True)
class RecurrenceGuardedResponseLawQualificationService:
    "Refuse finalization unless every claimed active action has exact action/preparation recurrence proof."

    finalizer: ResponseLawQualificationService

    capability_key = "qualification.recurrence-guarded-response-law-finalizer"
    capability_version = "1.0.0"

    def qualify(
        self,
        *,
        system: SystemSpec,
        dataset_or_projection: ObjectIdentity,
        family: ComponentUncertaintyFamilyAssessment,
        requirement: ActionPreparationRecurrenceQualificationRequirement,
        recurrence_receipts: tuple[ActionPreparationRecurrenceReceipt, ...],
    ) -> ActionPreparationRecurrenceQualificationBinding:
        obstructions = recurrence_qualification_obstructions(
            requirement,
            recurrence_receipts,
        )
        if obstructions:
            return bind_action_preparation_recurrence_qualification(
                requirement=requirement,
                receipts=recurrence_receipts,
                qualification_result=None,
            )
        selected = family.selected_assessment()
        if selected is not None:
            if selected.candidate_evidence.config != requirement.method_config:
                raise ValueError("recurrence requirement changes the selected method config")
        elif requirement.method_config not in {value.config for value in family.ledger.members}:
            raise ValueError("recurrence requirement config is absent from the family roster")
        result = self.finalizer.qualify(
            system,
            dataset_or_projection,
            family,
        )
        return bind_action_preparation_recurrence_qualification(
            requirement=requirement,
            receipts=recurrence_receipts,
            qualification_result=result,
        )


__all__ = ["RecurrenceGuardedResponseLawQualificationService"]
