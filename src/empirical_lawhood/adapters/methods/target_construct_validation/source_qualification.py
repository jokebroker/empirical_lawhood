"""Pre-response source qualification and frozen alternate activation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .nomination import TargetConstructValidationNominationResult
from .nomination_freeze import BRIAN2_SDIST_SHA256, CANTERA_SDIST_SHA256, FIPY_SDIST_SHA256, frozen_nomination


class TargetConstructValidationSourceDisposition(StrEnum):
    QUALIFIED = "QUALIFIED"
    STOPPED_BEFORE_RESPONSE = "STOPPED_BEFORE_RESPONSE"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-source-qualification'

    qualification_id: str
    candidate_id: str
    package_name: str
    package_version: str
    source_distribution_sha256: str
    installed_runtime_id: str
    disposition: TargetConstructValidationSourceDisposition
    check_ids: tuple[str, ...]
    stop_code: str | None
    reason: str
    generator_response_count: int
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "qualification_id",
            "candidate_id",
            "package_name",
            "installed_runtime_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.package_version, field_name="package_version")
        validate_sha256(
            self.source_distribution_sha256,
            field_name="source_distribution_sha256",
        )
        require_sorted_unique_strings(self.check_ids, field_name="check_ids", allow_empty=False)
        validate_nonempty(self.reason, field_name="reason")
        stopped = self.disposition is TargetConstructValidationSourceDisposition.STOPPED_BEFORE_RESPONSE
        if stopped != (self.stop_code is not None):
            raise ValueError("source stop disposition and code differ")
        if self.stop_code is not None:
            validate_stable_id(self.stop_code, field_name="stop_code")
        if self.generator_response_count:
            raise ValueError("source qualification must precede every generator response")
        if self.protected_outcome_access_count:
            raise ValueError("source qualification cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source qualification must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetActivation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-activation'

    activation_id: str
    target_slot_id: str
    nomination_result: ObjectIdentity
    primary_candidate_id: str
    frozen_alternate_candidate_id: str | None
    activated_candidate_id: str
    source_qualifications: tuple[ObjectIdentity, ...]
    alternate_activated: bool
    activation_reason_code: str
    generator_response_count_before_activation: int
    no_convenience_override: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "activation_id",
            "target_slot_id",
            "primary_candidate_id",
            "activated_candidate_id",
            "activation_reason_code",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.frozen_alternate_candidate_id is not None:
            validate_stable_id(
                self.frozen_alternate_candidate_id,
                field_name="frozen_alternate_candidate_id",
            )
        require_sorted_unique_ids(
            self.source_qualifications,
            attribute="object_id",
            field_name="source_qualifications",
        )
        expected_alternate = self.activated_candidate_id != self.primary_candidate_id
        if self.alternate_activated != expected_alternate:
            raise ValueError("target activation alternate flag differs")
        if expected_alternate and self.activated_candidate_id != self.frozen_alternate_candidate_id:
            raise ValueError("target activation names an unfrozen alternate")
        if self.generator_response_count_before_activation:
            raise ValueError("alternate activation occurred after a generator response")
        if not self.no_convenience_override:
            raise ValueError("target activation cannot encode a convenience override")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target activation must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationSourceQualificationFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-source-qualification-freeze'

    freeze_id: str
    nomination_result: TargetConstructValidationNominationResult
    qualifications: tuple[TargetConstructValidationSourceQualification, ...]
    activations: tuple[TargetConstructValidationTargetActivation, ...]
    active_candidate_ids: tuple[str, ...]
    active_target_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_ids(
            self.qualifications,
            attribute="qualification_id",
            field_name="qualifications",
        )
        require_sorted_unique_ids(
            self.activations,
            attribute="activation_id",
            field_name="activations",
        )
        require_sorted_unique_strings(
            self.active_candidate_ids,
            field_name="active_candidate_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.active_target_ids,
            field_name="active_target_ids",
            allow_empty=False,
        )
        if len(self.activations) != 2 or len(self.active_candidate_ids) != 2:
            raise ValueError("source freeze requires exactly two active targets")
        if tuple(sorted(value.activated_candidate_id for value in self.activations)) != (
            self.active_candidate_ids
        ):
            raise ValueError("source freeze activation roster differs")
        qualification_ids = {
            ObjectIdentity.from_record(value.qualification_id, value)
            for value in self.qualifications
        }
        if any(
            not set(value.source_qualifications).issubset(qualification_ids)
            for value in self.activations
        ):
            raise ValueError("target activation cites an unknown source qualification")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source qualification freeze must remain outcome-blind")


def frozen_source_qualification() -> TargetConstructValidationSourceQualificationFreeze:
    nomination = frozen_nomination().nomination_result
    brian = TargetConstructValidationSourceQualification(
        qualification_id="source-qualification.brian2-2p9p0",
        candidate_id="candidate.brian2-lif",
        package_name="brian2",
        package_version="2.9.0",
        source_distribution_sha256=BRIAN2_SDIST_SHA256,
        installed_runtime_id="cpython-3p11-numpy-2p4p3-linux-x86-64",
        disposition=TargetConstructValidationSourceDisposition.STOPPED_BEFORE_RESPONSE,
        check_ids=("exact-source-hash", "import-canary", "python-version"),
        stop_code="source-api-incompatible",
        reason=(
            "Import stopped before generator construction because Brian2 2.9.0 references "
            "the removed numpy.ndarray.ptp attribute under the repository NumPy 2.x runtime."
        ),
        generator_response_count=0,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    cantera = TargetConstructValidationSourceQualification(
        qualification_id="source-qualification.cantera-3p2p0",
        candidate_id="candidate.cantera-cstr",
        package_name="cantera",
        package_version="3.2.0",
        source_distribution_sha256=CANTERA_SDIST_SHA256,
        installed_runtime_id="cpython-3p11-numpy-2p4p3-cantera-3p2p0-linux-x86-64",
        disposition=TargetConstructValidationSourceDisposition.QUALIFIED,
        check_ids=("exact-source-hash", "import-canary", "license-bsd-3-clause"),
        stop_code=None,
        reason="Exact locked package imported without constructing or advancing a reactor network.",
        generator_response_count=0,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    fipy = TargetConstructValidationSourceQualification(
        qualification_id="source-qualification.fipy-4p0p3",
        candidate_id="candidate.fipy-diffusion",
        package_name="fipy",
        package_version="4.0.3",
        source_distribution_sha256=FIPY_SDIST_SHA256,
        installed_runtime_id="cpython-3p11-numpy-2p4p3-fipy-4p0p3-scipy-linux-x86-64",
        disposition=TargetConstructValidationSourceDisposition.QUALIFIED,
        check_ids=("exact-source-hash", "import-canary", "solver-suite-scipy"),
        stop_code=None,
        reason="Exact locked package and SciPy solver suite imported without solving a PDE.",
        generator_response_count=0,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    by_id = {value.candidate_id: value for value in (brian, cantera, fipy)}
    nomination_identity = ObjectIdentity.from_record(nomination.result_id, nomination)
    n1 = TargetConstructValidationTargetActivation(
        activation_id='activation.target-cantera.stirred-reactor',
        target_slot_id="target-slot-n1",
        nomination_result=nomination_identity,
        primary_candidate_id=nomination.first_independent_target_candidate_id,
        frozen_alternate_candidate_id=nomination.first_independent_target_alternate_candidate_id,
        activated_candidate_id="candidate.cantera-cstr",
        source_qualifications=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(brian.qualification_id, brian),
                    ObjectIdentity.from_record(cantera.qualification_id, cantera),
                ),
                key=lambda value: value.object_id,
            )
        ),
        alternate_activated=True,
        activation_reason_code="source-api-incompatible",
        generator_response_count_before_activation=0,
        no_convenience_override=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    n2 = TargetConstructValidationTargetActivation(
        activation_id='activation.target-fipy.source-diffusion',
        target_slot_id="target-slot-n2",
        nomination_result=nomination_identity,
        primary_candidate_id=nomination.second_independent_target_candidate_id,
        frozen_alternate_candidate_id=nomination.second_independent_target_alternate_candidate_id,
        activated_candidate_id="candidate.fipy-diffusion",
        source_qualifications=(ObjectIdentity.from_record(fipy.qualification_id, fipy),),
        alternate_activated=False,
        activation_reason_code="primary-source-qualified",
        generator_response_count_before_activation=0,
        no_convenience_override=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return TargetConstructValidationSourceQualificationFreeze(
        freeze_id="target-construct-validation.source-qualification-freeze",
        nomination_result=nomination,
        qualifications=tuple(sorted(by_id.values(), key=lambda value: value.qualification_id)),
        activations=tuple(sorted((n1, n2), key=lambda value: value.activation_id)),
        active_candidate_ids=("candidate.cantera-cstr", "candidate.fipy-diffusion"),
        active_target_ids=('target-cantera.stirred-reactor', 'target-fipy.source-diffusion'),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    'TargetConstructValidationSourceDisposition',
    'TargetConstructValidationSourceQualification',
    'TargetConstructValidationSourceQualificationFreeze',
    'TargetConstructValidationTargetActivation',
    "frozen_source_qualification",
]
