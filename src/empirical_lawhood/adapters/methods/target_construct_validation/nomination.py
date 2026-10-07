"""Metadata-only, outcome-blind candidate registry and deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

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

from .native_dossier import TargetConstructValidationIndependenceDossier, TargetConstructValidationTargetNativeDossier


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-candidate'

    candidate_id: str
    domain_id: str
    solver_family_id: str
    generator_family_id: str
    task_statement: str
    generator_provenance: str
    specification_provenance: str
    preparation_unit_definition: str
    native_intervention_definition: str
    action_realization_chain: str
    receiver_definitions: tuple[str, ...]
    horizon_definitions: tuple[str, ...]
    causal_cutoff_definition: str
    likely_denominator_factor_ids: tuple[str, ...]
    likely_history_factor_ids: tuple[str, ...]
    source_cost_class: int
    execution_cost_class: int
    prior_project_exposure_ids: tuple[str, ...]
    prior_investigator_exposure_ids: tuple[str, ...]
    construct_threat_ids: tuple[str, ...]
    maximum_claim: str
    construct_non_tautological: bool
    task_specification_independent: bool
    new_generator_family: bool
    complete_action_realization_observable: bool
    independent_units_defensible: bool
    nontrivial_contrasts_available: bool
    sealed_evaluation_feasible: bool
    joint_power_feasible: bool
    source_metadata_complete: bool
    recurrence_eligible: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("candidate_id", "domain_id", "solver_family_id", "generator_family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "task_statement",
            "generator_provenance",
            "specification_provenance",
            "preparation_unit_definition",
            "native_intervention_definition",
            "action_realization_chain",
            "causal_cutoff_definition",
            "maximum_claim",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        for name in (
            "receiver_definitions",
            "horizon_definitions",
            "likely_denominator_factor_ids",
            "likely_history_factor_ids",
            "prior_project_exposure_ids",
            "prior_investigator_exposure_ids",
            "construct_threat_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name
                in {"prior_project_exposure_ids", "prior_investigator_exposure_ids"},
            )
        for name in ("source_cost_class", "execution_cost_class"):
            if not 0 <= getattr(self, name) <= 4:
                raise ValueError(f"{name} must be a bounded 0--4 metadata score")
        required = (
            self.construct_non_tautological,
            self.task_specification_independent,
            self.new_generator_family,
            self.complete_action_realization_observable,
            self.independent_units_defensible,
            self.nontrivial_contrasts_available,
            self.sealed_evaluation_feasible,
            self.joint_power_feasible,
            self.source_metadata_complete,
        )
        if self.recurrence_eligible != all(required):
            raise ValueError("candidate eligibility is not metadata-derived")
        if self.protected_outcome_access_count != 0:
            raise ValueError("candidate nomination cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("candidate nomination must remain outcome-blind")

    @property
    def ranking_key(self) -> tuple[int | str, ...]:
        return (
            -int(self.construct_non_tautological),
            -int(self.task_specification_independent),
            -int(self.new_generator_family),
            -int(self.complete_action_realization_observable),
            -int(self.independent_units_defensible),
            -int(self.nontrivial_contrasts_available),
            -int(self.sealed_evaluation_feasible),
            -int(self.joint_power_feasible),
            self.source_cost_class + self.execution_cost_class,
            self.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCandidateRegistry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-candidate-registry'

    registry_id: str
    candidates: tuple[TargetConstructValidationCandidate, ...]
    frozen_before_candidate_outcomes: bool
    candidate_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        require_sorted_unique_ids(
            self.candidates,
            attribute="candidate_id",
            field_name="candidates",
        )
        if not 1 <= len(self.candidates) <= 5:
            raise ValueError("candidate registry must contain one to five candidates")
        if not self.frozen_before_candidate_outcomes or self.candidate_outcome_access_count:
            raise ValueError("candidate registry must freeze before target outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("candidate registry must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationContaminationEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-contamination-entry'

    entry_id: str
    candidate_id: str
    exposure_kind: str
    exposed_party_id: str
    exposed_object_ids: tuple[str, ...]
    disqualifies_recurrence: bool

    def __post_init__(self) -> None:
        for name in ("entry_id", "candidate_id", "exposure_kind", "exposed_party_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.exposed_object_ids,
            field_name="exposed_object_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationContaminationLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-contamination-ledger'

    ledger_id: str
    candidate_registry: ObjectIdentity
    entries: tuple[TargetConstructValidationContaminationEntry, ...]
    transitive_exposure_audited: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        require_sorted_unique_ids(self.entries, attribute="entry_id", field_name="entries")
        if not self.transitive_exposure_audited:
            raise ValueError("contamination ledger requires transitive exposure audit")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("contamination ledger must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNominationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-nomination-result'

    result_id: str
    candidate_registry: ObjectIdentity
    ranked_candidate_ids: tuple[str, ...]
    first_independent_target_candidate_id: str
    second_independent_target_candidate_id: str
    first_independent_target_alternate_candidate_id: str | None
    second_independent_target_alternate_candidate_id: str | None
    no_convenience_override: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("result_id", "first_independent_target_candidate_id", "second_independent_target_candidate_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            tuple(sorted(self.ranked_candidate_ids)),
            field_name="ranked_candidate_ids_set",
            allow_empty=False,
        )
        if len(self.ranked_candidate_ids) != len(set(self.ranked_candidate_ids)):
            raise ValueError("nomination ranking contains duplicate identities")
        for name in ("first_independent_target_alternate_candidate_id", "second_independent_target_alternate_candidate_id"):
            value = getattr(self, name)
            if value is not None:
                validate_stable_id(value, field_name=name)
        selected = {self.first_independent_target_candidate_id, self.second_independent_target_candidate_id}
        alternates = {
            value
            for value in (
                self.first_independent_target_alternate_candidate_id,
                self.second_independent_target_alternate_candidate_id,
            )
            if value is not None
        }
        if len(selected) != 2 or selected & alternates:
            raise ValueError("nomination primary/alternate identities overlap")
        if not self.no_convenience_override:
            raise ValueError("nomination result cannot encode a convenience override")
        if self.protected_outcome_access_count:
            raise ValueError("nomination result cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("nomination result must remain outcome-blind")


def select_nominees(
    registry: TargetConstructValidationCandidateRegistry,
    *,
    contamination_disqualified_ids: Iterable[str] = (),
) -> TargetConstructValidationNominationResult:
    disqualified = frozenset(contamination_disqualified_ids)
    eligible = tuple(
        sorted(
            (
                value
                for value in registry.candidates
                if value.recurrence_eligible and value.candidate_id not in disqualified
            ),
            key=lambda value: value.ranking_key,
        )
    )
    if not eligible:
        raise ValueError("INSUFFICIENT_INDEPENDENT_TARGETS: no eligible N1")
    n1 = eligible[0]
    n2_values = tuple(
        value
        for value in eligible[1:]
        if value.domain_id != n1.domain_id
        and value.solver_family_id != n1.solver_family_id
        and value.generator_family_id != n1.generator_family_id
    )
    if not n2_values:
        raise ValueError("INSUFFICIENT_INDEPENDENT_TARGETS: no independent N2")
    n2 = n2_values[0]

    n1_alternate = next(
        (
            value
            for value in eligible
            if value.candidate_id not in {n1.candidate_id, n2.candidate_id}
        ),
        None,
    )
    n2_alternate = next(
        (
            value
            for value in eligible
            if value.candidate_id
            not in {
                n1.candidate_id,
                n2.candidate_id,
                None if n1_alternate is None else n1_alternate.candidate_id,
            }
            and value.domain_id != n1.domain_id
            and value.solver_family_id != n1.solver_family_id
            and value.generator_family_id != n1.generator_family_id
        ),
        None,
    )
    return TargetConstructValidationNominationResult(
        result_id="target-construct-validation.nomination-result",
        candidate_registry=ObjectIdentity.from_record(registry.registry_id, registry),
        ranked_candidate_ids=tuple(value.candidate_id for value in eligible),
        first_independent_target_candidate_id=n1.candidate_id,
        second_independent_target_candidate_id=n2.candidate_id,
        first_independent_target_alternate_candidate_id=(
            None if n1_alternate is None else n1_alternate.candidate_id
        ),
        second_independent_target_alternate_candidate_id=(
            None if n2_alternate is None else n2_alternate.candidate_id
        ),
        no_convenience_override=True,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCandidateFeasibility(CanonicalRecord):
    """Bounded static source/API feasibility recorded without simulator execution."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-candidate-feasibility'

    feasibility_id: str
    candidate_id: str
    package_name: str
    package_version: str
    source_distribution_sha256: str
    documentation_identity: str
    license_identity: str
    python_compatible: bool
    action_api_static_readable: bool
    reset_semantics_static_readable: bool
    receiver_api_static_readable: bool
    complete_unit_static_resolvable: bool
    generator_contacted: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("feasibility_id", "candidate_id", "package_name"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("package_version", "documentation_identity", "license_identity"):
            validate_nonempty(getattr(self, name), field_name=name)
        validate_sha256(
            self.source_distribution_sha256,
            field_name="source_distribution_sha256",
        )
        if not all(
            (
                self.python_compatible,
                self.action_api_static_readable,
                self.reset_semantics_static_readable,
                self.receiver_api_static_readable,
                self.complete_unit_static_resolvable,
            )
        ):
            raise ValueError("candidate static feasibility is incomplete")
        if self.generator_contacted:
            raise ValueError("source-feasibility qualification cannot execute or contact a candidate generator")
        if self.protected_outcome_access_count:
            raise ValueError("source-feasibility qualification cannot access candidate outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source-feasibility qualification must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetEnvelope(CanonicalRecord):
    """candidate-activation qualification source/resource/authority facts; not an issue or execution grant."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-envelope'

    envelope_id: str
    target_slot_id: str
    candidate_id: str
    source_package_name: str
    source_package_version: str
    source_distribution_sha256: str
    maximum_parallel_units: int
    cpu_cores_per_unit: int
    memory_bytes_per_unit: int
    unit_timeout_seconds: int
    phase_wall_time_seconds: int
    external_storage_root_id: str
    source_acquisition_authority_id: str
    execution_authority_id: str
    reveal_authority_id: str
    evaluator_identity: str
    network_after_source_binding: bool
    live_or_physical_authority_granted: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "envelope_id",
            "target_slot_id",
            "candidate_id",
            "source_package_name",
            "external_storage_root_id",
            "source_acquisition_authority_id",
            "execution_authority_id",
            "reveal_authority_id",
            "evaluator_identity",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.source_package_version, field_name="source_package_version")
        validate_sha256(
            self.source_distribution_sha256,
            field_name="source_distribution_sha256",
        )
        for name in (
            "maximum_parallel_units",
            "cpu_cores_per_unit",
            "memory_bytes_per_unit",
            "unit_timeout_seconds",
            "phase_wall_time_seconds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.execution_authority_id == self.reveal_authority_id:
            raise ValueError("execution and reveal authorities must remain distinct")
        if self.network_after_source_binding:
            raise ValueError("selected simulators must execute offline after exact binding")
        if self.live_or_physical_authority_granted:
            raise ValueError("target construct validation cannot grant live or physical authority")
        if self.protected_outcome_access_count:
            raise ValueError("candidate-activation qualification cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("candidate-activation qualification must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNominationFreeze(CanonicalRecord):
    """Exact outcome-blind nomination freeze from which all target work must descend."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-nomination-freeze'

    freeze_id: str
    registry: TargetConstructValidationCandidateRegistry
    contamination_ledger: TargetConstructValidationContaminationLedger
    native_dossiers: tuple[TargetConstructValidationTargetNativeDossier, ...]
    independence_dossiers: tuple[TargetConstructValidationIndependenceDossier, ...]
    feasibility_records: tuple[TargetConstructValidationCandidateFeasibility, ...]
    nomination_result: TargetConstructValidationNominationResult
    target_envelopes: tuple[TargetConstructValidationTargetEnvelope, ...]
    metadata_source_ids: tuple[str, ...]
    frozen_before_candidate_outcomes: bool
    candidate_generator_contact_count: int
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for name, values, attribute in (
            ("native_dossiers", self.native_dossiers, "dossier_id"),
            ("independence_dossiers", self.independence_dossiers, "dossier_id"),
            ("feasibility_records", self.feasibility_records, "feasibility_id"),
            ("target_envelopes", self.target_envelopes, "envelope_id"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)
        require_sorted_unique_strings(
            self.metadata_source_ids,
            field_name="metadata_source_ids",
            allow_empty=False,
        )
        expected = select_nominees(
            self.registry,
            contamination_disqualified_ids=(
                entry.candidate_id
                for entry in self.contamination_ledger.entries
                if entry.disqualifies_recurrence
            ),
        )
        if self.nomination_result != expected:
            raise ValueError("nomination result is not deterministically registry-derived")
        selected = {
            self.nomination_result.first_independent_target_candidate_id,
            self.nomination_result.second_independent_target_candidate_id,
        }
        if {value.source_candidate.object_id for value in self.native_dossiers} != selected:
            raise ValueError("native dossier roster differs from selected targets")
        if {value.candidate_id for value in self.feasibility_records} != selected:
            raise ValueError("feasibility roster differs from selected targets")
        if {value.candidate_id for value in self.target_envelopes} != selected:
            raise ValueError("target envelope roster differs from selected targets")
        if {value.target_id for value in self.independence_dossiers} != {
            value.target_id for value in self.native_dossiers
        }:
            raise ValueError("independence and native dossier target rosters differ")
        if (
            not self.frozen_before_candidate_outcomes
            or self.candidate_generator_contact_count
            or self.protected_outcome_access_count
        ):
            raise ValueError("outcome-blind nomination must freeze before any candidate response")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("nomination freeze must remain outcome-blind")
