"""Pure, outcome-blind closure of a frozen campaign execution contract.

The diagnostic matrix in this module is deliberately ephemeral.  It is not a
canonical record, grants no authority, and is never accepted as execution
evidence.  Preview and the pre-provider guard call the same function so their
noncompensating rules cannot drift.
"""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.observation_order import ObservationOrderExperimentExtension
from empirical_lawhood.planning.retrospective_prediction import RetrospectivePredictionExperiment, RetrospectivePredictionRole
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.planning.native_source import NativeLawQualificationExperiment, NativeCrossfitExperiment
from empirical_lawhood.planning.study_issue import StudyAuthorityKind
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment, PredecessorBoundSourceQualificationExperiment, QualifiedSourceUseExperiment, ProspectiveRetainedSourceUse

from .capabilities import CapabilityRegistry
from .artifacts import MAX_ARTIFACT_PUBLICATION_MEMBERS, derived_outcome_access
from .response_experiment import CompiledResponseExperiment, ResponseExperimentStageRole
from .observation_order import CompiledObservationOrderExperiment, ObservationOrderStageRole
from .response_experiment_ports import NativeInteractionKind, ResponseSubstrateBinding
from .plans import ProtocolExecutionPlan, CandidateExecutionPlan, ExecutionTask, ProtocolRunPlan, RunPlanStep, CandidateRunPlan
from .source_qualification import CompiledFreshSourceQualification, derive_source_qualification_topology, retained_qualification_input_reasons
from .retrospective_prediction import CompiledRetrospectivePrediction, RetrospectivePredictionBinding, retained_prediction_contract_reasons


class ExperimentContractSection(StrEnum):
    SCIENTIFIC_INTERFACE_TOTALITY = "SCIENTIFIC_INTERFACE_TOTALITY"
    IDENTITY_AND_CHRONOLOGY = "IDENTITY_AND_CHRONOLOGY"
    CUSTODY_GRAPH = "CUSTODY_GRAPH"
    WORST_CASE_BOUNDEDNESS = "WORST_CASE_BOUNDEDNESS"
    NATIVE_SEMANTIC_ROUND_TRIP = "NATIVE_SEMANTIC_ROUND_TRIP"
    RECOVERY_AND_SALVAGE = "RECOVERY_AND_SALVAGE"


class ExperimentContractApplicability(StrEnum):
    """Whether an issued package carries one supported experiment carrier."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    APPLICABLE = "APPLICABLE"


def derive_experiment_contract_applicability(
    issued_payload_schema_ids: tuple[str, ...],
) -> ExperimentContractApplicability:
    """Classify by the closed extension schema, never an adapter/family name."""

    if len(set(issued_payload_schema_ids)) != len(issued_payload_schema_ids):
        raise ValueError("issued payload schema roster is duplicated")
    supported = {
        ResponseExperimentExtensionSet.SCHEMA,
        NativeLawQualificationExperiment.SCHEMA,
        NativeCrossfitExperiment.SCHEMA,
        ObservationOrderExperimentExtension.SCHEMA,
        FreshSourceQualificationExperiment.SCHEMA,
        PredecessorBoundSourceQualificationExperiment.SCHEMA,
        QualifiedSourceUseExperiment.SCHEMA,
        ProspectiveRetainedSourceUse.SCHEMA,
        RetrospectivePredictionExperiment.SCHEMA,
    }
    selected = supported.intersection(issued_payload_schema_ids)
    if len(selected) > 1:
        raise ValueError("issued package selects multiple experiment carriers")
    return (
        ExperimentContractApplicability.APPLICABLE
        if selected
        else ExperimentContractApplicability.NOT_APPLICABLE
    )


@dataclass(frozen=True, slots=True)
class ExperimentContractCheck:
    """One diagnostic row; intentionally not a persisted platform record."""

    section: ExperimentContractSection
    passed: bool
    reason_codes: tuple[str, ...]
    referenced_fingerprints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentContractClosureMatrix:
    """Bounded conjunction returned to preview and the execution guard."""

    checks: tuple[ExperimentContractCheck, ...]

    @property
    def passed(self) -> bool:
        return len(self.checks) == len(ExperimentContractSection) and all(
            value.passed for value in self.checks
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(sorted({reason for check in self.checks for reason in check.reason_codes}))

    def require_pass(self) -> None:
        if not self.passed:
            raise ValueError("experiment contract closure failed: " + ",".join(self.reason_codes))


class ExperimentContractChronologyKind(StrEnum):
    """Effect-relevant milestones used by both preview and the issued guard."""

    SPECIFICATION_FROZEN = "SPECIFICATION_FROZEN"
    AUTHORITY_ISSUED = "AUTHORITY_ISSUED"
    SOURCE_CONTACT = "SOURCE_CONTACT"
    SOURCE_OBJECT_CUSTODIED = "SOURCE_OBJECT_CUSTODIED"
    DONOR_TERMINAL_FROZEN = "DONOR_TERMINAL_FROZEN"
    DONOR_LAW_FROZEN = "DONOR_LAW_FROZEN"
    FORECASTS_FROZEN = "FORECASTS_FROZEN"
    TARGET_ISSUED = "TARGET_ISSUED"
    TARGET_CONTACT = "TARGET_CONTACT"
    OUTCOME_REVEALED = "OUTCOME_REVEALED"


class ExperimentContractNativeDisposition(StrEnum):
    """Native adapter outcomes which must survive translation without collapse."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    PROVIDER_EXCEPTION = "PROVIDER_EXCEPTION"


class ExperimentContractRecoveryCutKind(StrEnum):
    BEFORE_SOURCE_CONTACT = "BEFORE_SOURCE_CONTACT"
    AFTER_SOURCE_CUSTODY = "AFTER_SOURCE_CUSTODY"
    AFTER_DONOR_FINALIZATION = "AFTER_DONOR_FINALIZATION"
    AFTER_TARGET_ISSUE = "AFTER_TARGET_ISSUE"
    AFTER_TARGET_EFFECT = "AFTER_TARGET_EFFECT"


@dataclass(frozen=True, slots=True)
class ExperimentContractStageApplicability:
    """Exact supported carrier and its outcome-blind compiled topology."""

    extension_set: (
        ResponseExperimentExtensionSet
        | ObservationOrderExperimentExtension
        | FreshSourceQualificationExperiment
        | RetrospectivePredictionExperiment
    )
    compiled_plan: (
        CompiledResponseExperiment
        | CompiledObservationOrderExperiment
        | CompiledFreshSourceQualification
        | CompiledRetrospectivePrediction
    )


@dataclass(frozen=True, slots=True)
class ExperimentContractPublicationBatch:
    """Planned atomic publication and producer lineage, before payload hashes exist."""

    batch_id: str
    producer_task_id: str
    member_output_ids: tuple[str, ...]
    lineage_edge_ids: tuple[str, ...]
    maximum_resident_bytes: int
    custody_ordinal: int


@dataclass(frozen=True, slots=True)
class ExperimentContractChronologyEvent:
    """One predeclared or issued event; ordering is explicit, never name-derived."""

    event_id: str
    ordinal: int
    kind: ExperimentContractChronologyKind
    occurrence_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentContractAuthorityBinding:
    """Exact issued or predeclared authority identity for named future effects."""

    authority: ObjectIdentity
    kind: StudyAuthorityKind
    authority_event_id: str
    authorized_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentContractNativeConformance:
    """Code-owned adapter conformance evidence selected for this exact binding."""

    binding: ResponseSubstrateBinding
    requested_receiver_ids: tuple[str, ...]
    requested_clock_ids: tuple[str, ...]
    preserved_dispositions: tuple[ExperimentContractNativeDisposition, ...]
    preserved_action_link_ids: tuple[str, ...]
    conformance_check_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentContractRecoveryCut:
    """One interruption cut and its non-replay/custody obligations."""

    cut_id: str
    kind: ExperimentContractRecoveryCutKind
    catalog_independent: bool
    exact_receipts_required: bool
    source_reacquisition_allowed: bool
    target_effect_replay_allowed: bool


@dataclass(frozen=True, slots=True)
class ExperimentContractEvidence:
    "Ephemeral authoritative inputs missing from the plan records themselves."

    stage_applicability: tuple[ExperimentContractStageApplicability, ...]
    publication_batches: tuple[ExperimentContractPublicationBatch, ...]
    chronology: tuple[ExperimentContractChronologyEvent, ...]
    authority_bindings: tuple[ExperimentContractAuthorityBinding, ...]
    native_conformance: tuple[ExperimentContractNativeConformance, ...]
    recovery_cuts: tuple[ExperimentContractRecoveryCut, ...]
    maximum_resident_bytes: int
    historical_prediction_bindings: tuple[RetrospectivePredictionBinding, ...] = ()
    historical_experiments: tuple[RetrospectiveExperimentSpec, ...] = ()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _check(
    section: ExperimentContractSection,
    reasons: list[str],
    *fingerprints: str,
) -> ExperimentContractCheck:
    return ExperimentContractCheck(
        section=section,
        passed=not reasons,
        reason_codes=tuple(sorted(set(reasons))),
        referenced_fingerprints=tuple(sorted(set(fingerprints))),
    )


def validate_experiment_contract_closure(
    *,
    run_plan: ProtocolRunPlan,
    execution_plan: ProtocolExecutionPlan,
    capability_registry: CapabilityRegistry,
    contract_evidence: ExperimentContractEvidence | None = None,
    applicability: ExperimentContractApplicability | None = None,
) -> ExperimentContractClosureMatrix:
    """Validate all source-free execution-contract sections without effects.

    Canonical constructors already enforce their local invariants.  This
    function supplies the missing cross-record conjunction: exact lowering,
    producer/consumer equality, immutable chronology, custody identity,
    worst-case output bounds, installed native semantic contracts and bounded
    replay.  It performs no source probing, persistence, issue, authority or
    provider construction.
    """

    run_fingerprint = run_plan.fingerprint()
    execution_fingerprint = execution_plan.fingerprint()
    registry_fingerprint = capability_registry.fingerprint()
    steps = {value.step_id: value for value in run_plan.steps}
    tasks = {value.task_id: value for value in execution_plan.tasks}
    effective_applicability = applicability or (
        ExperimentContractApplicability.APPLICABLE
        if contract_evidence is not None and contract_evidence.stage_applicability
        else ExperimentContractApplicability.NOT_APPLICABLE
    )

    interface_reasons: list[str] = []
    if (
        effective_applicability is ExperimentContractApplicability.APPLICABLE
        and contract_evidence is None
    ):
        interface_reasons.append("PARAMETERISED_CONTRACT_EVIDENCE_MISSING")
    if (
        effective_applicability is ExperimentContractApplicability.NOT_APPLICABLE
        and contract_evidence is not None
    ):
        interface_reasons.append("PARAMETERISED_CONTRACT_EVIDENCE_NOT_APPLICABLE")
    if set(steps) != set(tasks):
        interface_reasons.append("LOWERED_TASK_ROSTER_DIFFERS")
    for task_id in sorted(set(steps) & set(tasks)):
        step = steps[task_id]
        task = tasks[task_id]
        if (
            task.stage is not step.stage
            or task.capability != step.capability
            or task.dependency_task_ids != step.dependency_step_ids
            or task.external_inputs != step.external_inputs
            or task.outputs != step.outputs
            or task.resource_lock_ids != step.resource_lock_ids
            or task.barrier is not step.barrier
            or task.maximum_attempts != step.maximum_attempts
            or task.obligation_ids != step.obligation_ids
        ):
            interface_reasons.append(f"LOWERED_TASK_DIFFERS:{task_id}")
        try:
            capability_registry.require(step.capability)
        except ValueError:
            interface_reasons.append(f"CAPABILITY_CONTRACT_DIFFERS:{task_id}")
    if isinstance(run_plan, CandidateRunPlan) != isinstance(execution_plan, CandidateExecutionPlan):
        interface_reasons.append("EXACT_EDGE_PLAN_VERSION_DIFFERS")
    if isinstance(run_plan, CandidateRunPlan) and isinstance(execution_plan, CandidateExecutionPlan):
        for task_id in sorted(set(steps) & set(tasks)):
            step = steps[task_id]
            task = tasks[task_id]
            assert isinstance(step, RunPlanStep)
            assert isinstance(task, ExecutionTask)
            if task.scientific_inputs != step.scientific_inputs:
                interface_reasons.append(f"SCIENTIFIC_EDGE_LOWERING_DIFFERS:{task_id}")
    if contract_evidence is not None:
        if (
            effective_applicability is ExperimentContractApplicability.APPLICABLE
            and not contract_evidence.stage_applicability
        ):
            interface_reasons.append("PARAMETERISED_STAGE_APPLICABILITY_MISSING")
        if (
            effective_applicability is ExperimentContractApplicability.APPLICABLE
            and not contract_evidence.publication_batches
        ):
            interface_reasons.append("PARAMETERISED_PUBLICATION_EVIDENCE_MISSING")
        admission_stage_roles = {
            ResponseExperimentStageRole.ADMISSION_APPLICABILITY,
            ResponseExperimentStageRole.ADMISSION_ACQUISITION_AND_ADJUDICATION,
            ResponseExperimentStageRole.CONTROLLER_COMPILATION_AND_COMMIT,
        }
        prospective_evaluation_stage_roles = {
            ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_APPLICABILITY,
            ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_ACQUISITION,
            ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_REVEAL_AND_ADJUDICATION,
        }
        for value in contract_evidence.stage_applicability:
            extension = value.extension_set
            compiled = value.compiled_plan
            expected_extension = ObjectIdentity.from_record(
                extension.extension_set_id,
                extension,
            )
            if compiled.extension_set != expected_extension:
                interface_reasons.append(
                    f"APPLICABILITY_EXTENSION_DIFFERS:{extension.extension_set_id}"
                )
            if isinstance(extension, RetrospectivePredictionExperiment) and isinstance(
                compiled, CompiledRetrospectivePrediction
            ):
                roots = contract_evidence.historical_experiments
                if len(roots) != 1:
                    interface_reasons.append("HISTORICAL_EXPERIMENT_ROOT_REQUIRED")
                else:
                    root = roots[0]
                    expected_units = tuple(sorted({
                        sample.physical_independent_unit_id for sample in extension.samples
                        if sample.physical_independent_unit_id is not None
                    }))
                    expected_labels = tuple(sorted(
                        item.artifact.artifact_id for item in extension.inputs
                        if item.role.value == "TARGET_LABELS"
                    ))
                    if (
                        run_plan.experiment != ObjectIdentity.from_record(root.experiment_id, root)
                        or root.experiment_id != extension.experiment_id
                        or root.reveal_barrier.evaluation_manifest_sha256 != extension.fingerprint()
                        or root.reveal_barrier.sealed_outcome_artifact_ids != expected_labels
                        or root.known_physical_independent_unit_ids != expected_units
                        or run_plan.physical_preparation_ids != expected_units
                    ):
                        interface_reasons.append("HISTORICAL_EXPERIMENT_ROOT_DIFFERS")
                interface_reasons.extend(
                    retained_prediction_contract_reasons(
                        extension,
                        compiled,
                        execution_plan,
                        capability_registry,
                    )
                )
                bindings = contract_evidence.historical_prediction_bindings
                owners = {o.role: o for o in extension.owners}
                projection = owners[RetrospectivePredictionRole.TRAINING_PROJECTION]
                if (
                    len(bindings) != 1
                    or bindings[0].prediction_carrier != expected_extension
                    or bindings[0].projection_owner != projection.owner
                    or bindings[0].projection_config != projection.config
                    or contract_evidence.native_conformance
                ):
                    interface_reasons.append("HISTORICAL_PREDICTION_BINDING_DIFFERS")
            elif isinstance(extension, ResponseExperimentExtensionSet) and isinstance(
                compiled,
                CompiledResponseExperiment,
            ):
                stages = {stage.role: stage for stage in compiled.stages}
                if any(
                    stages[role].applicable != (extension.admission_config is not None)
                    for role in admission_stage_roles
                ):
                    interface_reasons.append(
                        f"ADMISSION_APPLICABILITY_DIFFERS:{extension.extension_set_id}"
                    )
                if any(
                    stages[role].applicable != (extension.prospective_evaluation_config is not None)
                    for role in prospective_evaluation_stage_roles
                ):
                    interface_reasons.append(
                        f"CONTROLLER_USE_APPLICABILITY_DIFFERS:{extension.extension_set_id}"
                    )
            elif isinstance(extension, ObservationOrderExperimentExtension) and isinstance(
                compiled,
                CompiledObservationOrderExperiment,
            ):
                stage_ids = {stage.task_id for stage in compiled.stages}
                if stage_ids != set(tasks):
                    interface_reasons.append(
                        f"OBSERVATION_STAGE_ROSTER_DIFFERS:{extension.extension_set_id}"
                    )
                acquisition = tuple(
                    stage
                    for stage in compiled.stages
                    if stage.role is ObservationOrderStageRole.ACQUISITION
                )
                projections = tuple(
                    stage
                    for stage in compiled.stages
                    if stage.role is ObservationOrderStageRole.PROJECTION
                )
                if {stage.acquisition_group_id for stage in acquisition} != {
                    group.acquisition_group_id for group in extension.acquisition_groups
                } or {stage.scientific_view_id for stage in projections} != {
                    view.view_id for view in extension.nested_views
                }:
                    interface_reasons.append(
                        f"OBSERVATION_LINEAGE_COVERAGE_DIFFERS:{extension.extension_set_id}"
                    )
            elif isinstance(extension, FreshSourceQualificationExperiment) and isinstance(
                compiled, CompiledFreshSourceQualification
            ):
                qualification_topology = derive_source_qualification_topology(extension)
                if isinstance(extension, PredecessorBoundSourceQualificationExperiment):
                    interface_reasons.extend(
                        retained_qualification_input_reasons(extension, execution_plan)
                    )
                if compiled != qualification_topology or {
                    stage.task_id for stage in qualification_topology.stages
                } != set(tasks):
                    interface_reasons.append(
                        f"QUALIFICATION_STAGE_ROSTER_DIFFERS:{extension.extension_set_id}"
                    )
                for stage in qualification_topology.stages:
                    qualification_task = tasks.get(stage.task_id)
                    if qualification_task is None:
                        continue
                    manifest = capability_registry.resolve(
                        qualification_task.capability.capability_key,
                        qualification_task.capability.capability_version,
                    )
                    if (
                        qualification_task.stage is not stage.stage
                        or qualification_task.dependency_task_ids != stage.dependency_task_ids
                        or ObjectIdentity.from_record(manifest.capability_key, manifest)
                        != stage.owner
                        or qualification_task.capability.config.config_schema
                        != stage.config.object_schema
                        or qualification_task.capability.config.content_sha256
                        != stage.config.object_fingerprint
                        or qualification_task.maximum_attempts != 1
                    ):
                        interface_reasons.append(
                            f"QUALIFICATION_STAGE_BINDING_DIFFERS:{stage.task_id}"
                        )
            else:
                interface_reasons.append(
                    f"APPLICABILITY_CARRIER_TOPOLOGY_MISMATCH:{extension.extension_set_id}"
                )
        if contract_evidence.historical_prediction_bindings and not any(
            isinstance(v.extension_set, RetrospectivePredictionExperiment)
            for v in contract_evidence.stage_applicability
        ):
            interface_reasons.append("HISTORICAL_PREDICTION_BINDING_NOT_APPLICABLE")
        for native in contract_evidence.native_conformance:
            receiver_ids = {value.receiver_id for value in native.binding.receivers}
            clock_ids = {value.clock_id for value in native.binding.clocks}
            if not set(native.requested_receiver_ids) <= receiver_ids:
                interface_reasons.append(f"GHOST_RECEIVER:{native.binding.binding_id}")
            if not set(native.requested_clock_ids) <= clock_ids:
                interface_reasons.append(f"GHOST_CLOCK:{native.binding.binding_id}")
        covered_outputs: list[tuple[str, str]] = []
        for batch in contract_evidence.publication_batches:
            planned_task = tasks.get(batch.producer_task_id)
            if planned_task is None:
                interface_reasons.append(f"PUBLICATION_PRODUCER_MISSING:{batch.batch_id}")
                continue
            task_output_ids = {value.output_id for value in planned_task.outputs}
            if not set(batch.member_output_ids) <= task_output_ids:
                interface_reasons.append(f"PUBLICATION_OUTPUT_GHOST:{batch.batch_id}")
            covered_outputs.extend(
                (batch.producer_task_id, output_id) for output_id in batch.member_output_ids
            )
        expected_outputs = [
            (task.task_id, output.output_id)
            for task in execution_plan.tasks
            for output in task.outputs
        ]
        if sorted(covered_outputs) != sorted(expected_outputs):
            interface_reasons.append("PUBLICATION_OUTPUT_ROSTER_DIFFERS")

    chronology_reasons: list[str] = []
    if not run_plan.frozen or not execution_plan.frozen:
        chronology_reasons.append("PLAN_NOT_FROZEN")
    if execution_plan.source_plan.object_id != run_plan.run_plan_id:
        chronology_reasons.append("EXECUTION_SOURCE_PLAN_ID_DIFFERS")
    if execution_plan.source_plan.object_fingerprint != run_fingerprint:
        chronology_reasons.append("EXECUTION_SOURCE_PLAN_BYTES_DIFFER")
    if run_plan.topological_step_ids() != execution_plan.topological_task_ids():
        chronology_reasons.append("LOWERED_CHRONOLOGY_DIFFERS")
    if isinstance(run_plan, CandidateRunPlan) and isinstance(execution_plan, CandidateExecutionPlan):
        if run_plan.candidate != execution_plan.candidate:
            chronology_reasons.append("ISSUED_CANDIDATE_DIFFERS")
    if contract_evidence is not None:
        chronology = tuple(sorted(contract_evidence.chronology, key=lambda value: value.ordinal))
        if tuple(value.ordinal for value in chronology) != tuple(range(1, len(chronology) + 1)):
            chronology_reasons.append("EVENT_CHRONOLOGY_NOT_TOTAL")
        if len({value.event_id for value in chronology}) != len(chronology):
            chronology_reasons.append("EVENT_ID_REPEATS")
        first_by_kind = {
            kind: next((value.ordinal for value in chronology if value.kind is kind), None)
            for kind in ExperimentContractChronologyKind
        }
        frozen = first_by_kind[ExperimentContractChronologyKind.SPECIFICATION_FROZEN]
        first_contact = first_by_kind[ExperimentContractChronologyKind.SOURCE_CONTACT]
        if first_contact is not None and (frozen is None or frozen >= first_contact):
            chronology_reasons.append("SOURCE_CONTACT_PRECEDES_FREEZE")
        target_contact = first_by_kind[ExperimentContractChronologyKind.TARGET_CONTACT]
        target_issue = first_by_kind[ExperimentContractChronologyKind.TARGET_ISSUED]
        if target_contact is not None and (target_issue is None or target_issue >= target_contact):
            chronology_reasons.append("TARGET_CONTACT_PRECEDES_ISSUE")
        for kind in (
            ExperimentContractChronologyKind.DONOR_TERMINAL_FROZEN,
            ExperimentContractChronologyKind.DONOR_LAW_FROZEN,
            ExperimentContractChronologyKind.FORECASTS_FROZEN,
        ):
            ordinal = first_by_kind[kind]
            if target_issue is not None and (ordinal is None or ordinal >= target_issue):
                chronology_reasons.append(f"{kind.value}_NOT_BEFORE_TARGET_ISSUE")
        last_custody_by_occurrence: dict[str, int] = {}
        for event in chronology:
            if (
                event.kind is ExperimentContractChronologyKind.SOURCE_OBJECT_CUSTODIED
                and event.occurrence_id is not None
            ):
                last_custody_by_occurrence[event.occurrence_id] = event.ordinal
            if event.kind is ExperimentContractChronologyKind.SOURCE_CONTACT:
                earlier_contacts = [
                    prior
                    for prior in chronology
                    if prior.kind is ExperimentContractChronologyKind.SOURCE_CONTACT
                    and prior.ordinal < event.ordinal
                    and prior.occurrence_id is not None
                ]
                if any(
                    last_custody_by_occurrence.get(prior.occurrence_id or "", event.ordinal)
                    >= event.ordinal
                    for prior in earlier_contacts
                ):
                    chronology_reasons.append("NEXT_SOURCE_CONTACT_PRECEDES_CUSTODY")
        events = {value.event_id: value for value in chronology}
        if len({value.authority for value in contract_evidence.authority_bindings}) != len(
            contract_evidence.authority_bindings
        ):
            chronology_reasons.append("AUTHORITY_IDENTITY_REPEATS")
        authorized: dict[str, list[StudyAuthorityKind]] = {}
        for authority_binding in contract_evidence.authority_bindings:
            authority_event = events.get(authority_binding.authority_event_id)
            if (
                authority_event is None
                or authority_event.kind is not ExperimentContractChronologyKind.AUTHORITY_ISSUED
            ):
                chronology_reasons.append(
                    f"AUTHORITY_EVENT_MISSING:{authority_binding.authority.object_id}"
                )
                continue
            for event_id in authority_binding.authorized_event_ids:
                effect = events.get(event_id)
                if effect is None or authority_event.ordinal >= effect.ordinal:
                    chronology_reasons.append(f"AUTHORITY_NOT_BEFORE_EFFECT:{event_id}")
                authorized.setdefault(event_id, []).append(authority_binding.kind)
        required_authority = {
            ExperimentContractChronologyKind.SOURCE_CONTACT: (
                StudyAuthorityKind.SOURCE_ACQUISITION
            ),
            ExperimentContractChronologyKind.SOURCE_OBJECT_CUSTODIED: (
                StudyAuthorityKind.CUSTODY_PUBLICATION
            ),
            ExperimentContractChronologyKind.TARGET_CONTACT: (
                StudyAuthorityKind.EXPERIMENT_EXECUTION
            ),
            ExperimentContractChronologyKind.OUTCOME_REVEALED: (
                StudyAuthorityKind.OUTCOME_REVEAL
            ),
        }
        for effect in chronology:
            expected = required_authority.get(effect.kind)
            if expected is not None and authorized.get(effect.event_id) != [expected]:
                chronology_reasons.append(f"EFFECT_AUTHORITY_DIFFERS:{effect.event_id}")

    custody_reasons: list[str] = []
    outputs = [output for task in execution_plan.tasks for output in task.outputs]
    for attribute in ("output_id", "logical_artifact_id", "relative_path"):
        values = [getattr(output, attribute) for output in outputs]
        if len(values) != len(set(values)):
            custody_reasons.append(f"ARTIFACT_{attribute.upper()}_REPEATS")
    if isinstance(execution_plan, CandidateExecutionPlan):
        outputs_by_task = {
            task.task_id: {output.output_id: output for output in task.outputs}
            for task in execution_plan.tasks
        }
        for task in execution_plan.tasks:
            for edge in task.scientific_inputs:
                if edge.producer_task_id is None:
                    operational = next(
                        (
                            value
                            for value in task.external_inputs
                            if value.input_id == edge.consumer_input_id
                        ),
                        None,
                    )
                    if operational is None:
                        custody_reasons.append(f"EXTERNAL_EDGE_UNBOUND:{edge.edge_id}")
                    continue
                produced = outputs_by_task.get(edge.producer_task_id, {}).get(
                    edge.producer_output_id or ""
                )
                if (
                    produced is None
                    or produced.logical_artifact_id != edge.operational_logical_artifact_id
                    or produced.payload_schema != edge.payload_schema
                    or produced.media_type != edge.media_type
                ):
                    custody_reasons.append(f"LINEAGE_EDGE_UNBOUND:{edge.edge_id}")
                if produced is not None:
                    if not edge.visibility_ceiling.is_at_least_as_restrictive_as(
                        produced.visibility_ceiling
                    ):
                        custody_reasons.append(f"LINEAGE_VISIBILITY_LOWERED:{edge.edge_id}")
                    if (
                        derived_outcome_access(edge.outcome_access, produced.outcome_access)
                        is not edge.outcome_access
                    ):
                        custody_reasons.append(f"LINEAGE_OUTCOME_ACCESS_LOWERED:{edge.edge_id}")
                    for output in task.outputs:
                        if not output.visibility_ceiling.is_at_least_as_restrictive_as(
                            produced.visibility_ceiling
                        ):
                            custody_reasons.append(
                                f"OUTPUT_LINEAGE_VISIBILITY_LOWERED:{output.output_id}"
                            )
    if contract_evidence is not None:
        ordinals = [value.custody_ordinal for value in contract_evidence.publication_batches]
        if sorted(ordinals) != list(range(1, len(ordinals) + 1)):
            custody_reasons.append("PUBLICATION_CUSTODY_NOT_MONOTONE")
        if len({value.batch_id for value in contract_evidence.publication_batches}) != len(
            contract_evidence.publication_batches
        ):
            custody_reasons.append("PUBLICATION_BATCH_ID_REPEATS")
        if isinstance(execution_plan, CandidateExecutionPlan):
            edge_ids = {
                edge.edge_id for task in execution_plan.tasks for edge in task.scientific_inputs
            }
            for batch in contract_evidence.publication_batches:
                if not set(batch.lineage_edge_ids) <= edge_ids:
                    custody_reasons.append(f"PUBLICATION_LINEAGE_GHOST:{batch.batch_id}")

    bound_reasons: list[str] = []
    for task in execution_plan.tasks:
        if len(task.outputs) > 64:
            bound_reasons.append(f"PUBLICATION_MEMBER_BOUND_EXCEEDED:{task.task_id}")
        maximum_output_bytes = task.capability.requested_resources.output_bytes
        if maximum_output_bytes <= 0:
            bound_reasons.append(f"OUTPUT_BYTE_BOUND_INVALID:{task.task_id}")
        input_budget = task.capability.requested_resources.source_scan_bytes
        if (task.external_inputs or task.dependency_task_ids) and input_budget <= 0:
            bound_reasons.append(f"INPUT_SCAN_BUDGET_MISSING:{task.task_id}")
        known_input_bytes = sum(value.expected_size_bytes or 0 for value in task.external_inputs)
        if known_input_bytes > input_budget:
            bound_reasons.append(f"INPUT_SCAN_BUDGET_EXCEEDED:{task.task_id}")
        if task.maximum_attempts < 1 or task.maximum_attempts > 8:
            bound_reasons.append(f"ATTEMPT_BOUND_INVALID:{task.task_id}")
    if execution_plan.minimum_free_bytes != sum(
        task.capability.requested_resources.output_bytes for task in execution_plan.tasks
    ):
        bound_reasons.append("WORST_CASE_STORAGE_BOUND_DIFFERS")
    if contract_evidence is not None:
        if contract_evidence.maximum_resident_bytes < 1:
            bound_reasons.append("MAXIMUM_RESIDENCY_INVALID")
        for batch in contract_evidence.publication_batches:
            if len(batch.member_output_ids) > MAX_ARTIFACT_PUBLICATION_MEMBERS:
                bound_reasons.append(f"PUBLICATION_MEMBER_BOUND_EXCEEDED:{batch.batch_id}")
            planned_task = tasks.get(batch.producer_task_id)
            if planned_task is not None and batch.maximum_resident_bytes > (
                planned_task.capability.requested_resources.output_bytes
            ):
                bound_reasons.append(f"PUBLICATION_RESIDENCY_EXCEEDS_TASK:{batch.batch_id}")
            if batch.maximum_resident_bytes > contract_evidence.maximum_resident_bytes:
                bound_reasons.append(f"MAXIMUM_RESIDENCY_EXCEEDED:{batch.batch_id}")

    native_reasons: list[str] = []
    for task in execution_plan.tasks:
        try:
            manifest = capability_registry.require(task.capability)
        except ValueError:
            continue
        if manifest.implementation_sha256 != task.capability_implementation_sha256:
            native_reasons.append(f"NATIVE_IMPLEMENTATION_DIFFERS:{task.task_id}")
        output_schemas = tuple(sorted({value.payload_schema for value in task.outputs}))
        if output_schemas != task.capability.required_output_schema_ids:
            native_reasons.append(f"NATIVE_OUTPUT_CONTRACT_DIFFERS:{task.task_id}")
    if contract_evidence is not None:
        all_dispositions = set(ExperimentContractNativeDisposition)
        for native in contract_evidence.native_conformance:
            binding = native.binding
            if set(native.preserved_dispositions) != all_dispositions:
                native_reasons.append(f"NATIVE_PARTIAL_EXCEPTION_LOSS:{binding.binding_id}")
            try:
                manifest = capability_registry.resolve(
                    binding.provider_key, binding.provider_version
                )
            except KeyError:
                native_reasons.append(f"NATIVE_PROVIDER_MISSING:{binding.binding_id}")
                continue
            if manifest.implementation_sha256 != binding.provider_implementation_sha256:
                native_reasons.append(
                    f"NATIVE_PROVIDER_IMPLEMENTATION_DIFFERS:{binding.binding_id}"
                )
            if not set(native.conformance_check_ids) <= set(manifest.conformance_check_ids):
                native_reasons.append(f"NATIVE_CONFORMANCE_EVIDENCE_GHOST:{binding.binding_id}")
            expected_links: set[str] = set()
            if binding.interaction_kind is NativeInteractionKind.INTERACTIVE_EXECUTION:
                assert binding.action_contract is not None
                expected_links = {
                    name
                    for name, observed in (
                        ("requested", binding.action_contract.requested_observable),
                        ("accepted", binding.action_contract.accepted_observable),
                        ("applied", binding.action_contract.applied_observable),
                        ("realized", binding.action_contract.realized_observable),
                    )
                    if observed
                }
            if set(native.preserved_action_link_ids) != expected_links:
                native_reasons.append(f"NATIVE_ACTION_CHAIN_DIFFERS:{binding.binding_id}")

    recovery_reasons: list[str] = []
    if run_plan.registry_sha256 != execution_plan.registry_sha256:
        recovery_reasons.append("RECOVERY_REGISTRY_IDENTITY_DIFFERS")
    if run_plan.implementation_commit != execution_plan.implementation_commit:
        recovery_reasons.append("RECOVERY_IMPLEMENTATION_IDENTITY_DIFFERS")
    if run_plan.registry_sha256 != capability_registry.fingerprint():
        recovery_reasons.append("RECOVERY_REGISTRY_NOT_AUTHENTICATED")
    if any(not task.obligation_ids for task in execution_plan.tasks):
        recovery_reasons.append("RECOVERY_TERMINAL_OBLIGATION_MISSING")
    if contract_evidence is not None:
        if len({value.cut_id for value in contract_evidence.recovery_cuts}) != len(
            contract_evidence.recovery_cuts
        ):
            recovery_reasons.append("RECOVERY_CUT_ID_REPEATS")
        for cut in contract_evidence.recovery_cuts:
            if not cut.catalog_independent or not cut.exact_receipts_required:
                recovery_reasons.append(f"RECOVERY_CUT_NOT_AUTHENTICATED:{cut.cut_id}")
            if (
                cut.kind
                in {
                    ExperimentContractRecoveryCutKind.AFTER_SOURCE_CUSTODY,
                    ExperimentContractRecoveryCutKind.AFTER_DONOR_FINALIZATION,
                    ExperimentContractRecoveryCutKind.AFTER_TARGET_ISSUE,
                    ExperimentContractRecoveryCutKind.AFTER_TARGET_EFFECT,
                }
                and cut.source_reacquisition_allowed
            ):
                recovery_reasons.append(f"RECOVERY_REACQUIRES_SOURCE:{cut.cut_id}")
            if (
                cut.kind is ExperimentContractRecoveryCutKind.AFTER_TARGET_EFFECT
                and cut.target_effect_replay_allowed
            ):
                recovery_reasons.append(f"RECOVERY_REPLAYS_TARGET_EFFECT:{cut.cut_id}")
        kinds = {value.kind for value in contract_evidence.recovery_cuts}
        chronology_kinds = {value.kind for value in contract_evidence.chronology}
        required_cuts: set[ExperimentContractRecoveryCutKind] = set()
        if ExperimentContractChronologyKind.SOURCE_CONTACT in chronology_kinds:
            required_cuts.update(
                {
                    ExperimentContractRecoveryCutKind.BEFORE_SOURCE_CONTACT,
                    ExperimentContractRecoveryCutKind.AFTER_SOURCE_CUSTODY,
                }
            )
        if ExperimentContractChronologyKind.DONOR_LAW_FROZEN in chronology_kinds:
            required_cuts.add(ExperimentContractRecoveryCutKind.AFTER_DONOR_FINALIZATION)
        if ExperimentContractChronologyKind.TARGET_ISSUED in chronology_kinds:
            required_cuts.add(ExperimentContractRecoveryCutKind.AFTER_TARGET_ISSUE)
        if ExperimentContractChronologyKind.TARGET_CONTACT in chronology_kinds:
            required_cuts.add(ExperimentContractRecoveryCutKind.AFTER_TARGET_EFFECT)
        for missing in sorted(required_cuts - kinds, key=lambda value: value.value):
            recovery_reasons.append(f"RECOVERY_CUT_MISSING:{missing.value}")

    checks = (
        _check(
            ExperimentContractSection.SCIENTIFIC_INTERFACE_TOTALITY,
            interface_reasons,
            run_fingerprint,
            execution_fingerprint,
            registry_fingerprint,
        ),
        _check(
            ExperimentContractSection.IDENTITY_AND_CHRONOLOGY,
            chronology_reasons,
            run_fingerprint,
            execution_fingerprint,
        ),
        _check(
            ExperimentContractSection.CUSTODY_GRAPH,
            custody_reasons,
            execution_fingerprint,
            _digest(tuple(output.logical_artifact_id for output in outputs)),
        ),
        _check(
            ExperimentContractSection.WORST_CASE_BOUNDEDNESS,
            bound_reasons,
            _digest(
                tuple(
                    (
                        task.task_id,
                        len(task.outputs),
                        task.capability.requested_resources.output_bytes,
                    )
                    for task in execution_plan.tasks
                )
            ),
        ),
        _check(
            ExperimentContractSection.NATIVE_SEMANTIC_ROUND_TRIP,
            native_reasons,
            registry_fingerprint,
        ),
        _check(
            ExperimentContractSection.RECOVERY_AND_SALVAGE,
            recovery_reasons,
            run_fingerprint,
            execution_fingerprint,
            registry_fingerprint,
        ),
    )
    return ExperimentContractClosureMatrix(checks=checks)


__all__ = [
    'ExperimentContractApplicability',
    'ExperimentContractAuthorityBinding',
    'ExperimentContractCheck',
    'ExperimentContractChronologyEvent',
    'ExperimentContractChronologyKind',
    'ExperimentContractClosureMatrix',
    'ExperimentContractEvidence',
    'ExperimentContractNativeConformance',
    'ExperimentContractNativeDisposition',
    'ExperimentContractPublicationBatch',
    'ExperimentContractRecoveryCutKind',
    'ExperimentContractRecoveryCut',
    'ExperimentContractSection',
    'ExperimentContractStageApplicability',
    'derive_experiment_contract_applicability',
    'validate_experiment_contract_closure',
]
