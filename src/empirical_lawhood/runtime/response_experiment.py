"Strict compilation and bounded sequencing for parameterised response experiment plans.\n\nCompilation is an outcome-blind identity join over the existing strict roots.\nThe spooler advances owner results and immutable receipts; it contains no\nprojection, law, admission, controller, or controller use estimator.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import OperationalStatus, ScientificStatus
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignOwnerRole, LinkedCampaignProfile
from empirical_lawhood.planning.response_experiment import MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES, ResponseQualificationConfig, ResponseExperimentExtensionSet, AdmissionExperimentConfig, ProspectiveUseExperimentConfig
from empirical_lawhood.planning.prospective_config import ProspectiveAuthorityOperation
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.native_source import NativeSourceProfile, NativeLawQualificationExperiment
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.execution_envelope import validate_optional_jit_identities
from empirical_lawhood.runtime.plans import ExecutionPlan, RunPlan
from empirical_lawhood.runtime.recovery import RunRecoveryIndex, RunRecoveryTerminalEvent
from empirical_lawhood.runtime.static_codecs import (
    CanonicalRecordCodecRegistry,
    build_canonical_record_codec_registry,
)


class ResponseExperimentStageRole(StrEnum):
    SOURCE_READINESS = "source-readiness"
    IDENTIFICATION_ISSUE = "identification-issue"
    IDENTIFICATION_ACQUISITION = "identification-acquisition"
    IDENTIFICATION_REVEAL_AND_ADJUDICATION = "identification-reveal-and-adjudication"
    ADMISSION_APPLICABILITY = "admission-applicability"
    ADMISSION_ACQUISITION_AND_ADJUDICATION = "admission-acquisition-and-adjudication"
    CONTROLLER_COMPILATION_AND_COMMIT = "controller-compilation-and-commit"
    PROSPECTIVE_EVALUATION_APPLICABILITY = "prospective-evaluation-applicability"
    PROSPECTIVE_EVALUATION_ACQUISITION = "prospective-evaluation-acquisition"
    PROSPECTIVE_EVALUATION_REVEAL_AND_ADJUDICATION = "prospective-evaluation-reveal-and-adjudication"
    EXPERIMENT_CLOSEOUT = "experiment-closeout"


class ResponseExperimentObstruction(StrEnum):
    NONE = "NONE"
    SCIENTIFIC_OPPOSITION = "SCIENTIFIC_OPPOSITION"
    MISSING_OPERAND = "MISSING_OPERAND"
    SOURCE_FAILURE = "SOURCE_FAILURE"
    UNTARGETABLE = "UNTARGETABLE"
    ACTION_FAILURE = "ACTION_FAILURE"
    DECISION_FAILURE = "DECISION_FAILURE"
    INSUFFICIENT_POWER = "INSUFFICIENT_POWER"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    STAGE_NOT_APPLICABLE = "STAGE_NOT_APPLICABLE"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    UNRESOLVED = "UNRESOLVED"


class ResponseStageExecutionBasis(StrEnum):
    NONEXECUTING = "nonexecuting"
    INJECTED_CONFORMANCE = "injected-conformance"
    ISSUED_EXECUTION = "issued-execution"


_STAGE_ROLES = tuple(ResponseExperimentStageRole)
_ADMISSION_STAGE_ROLES = frozenset(
    {
        ResponseExperimentStageRole.ADMISSION_APPLICABILITY,
        ResponseExperimentStageRole.ADMISSION_ACQUISITION_AND_ADJUDICATION,
        ResponseExperimentStageRole.CONTROLLER_COMPILATION_AND_COMMIT,
    }
)
_PROSPECTIVE_EVALUATION_STAGE_ROLES = frozenset(
    {
        ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_APPLICABILITY,
        ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_ACQUISITION,
        ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_REVEAL_AND_ADJUDICATION,
    }
)
STAGE_NOT_APPLICABLE_REASON = "STAGE_NOT_APPLICABLE_BY_ISSUED_PLAN"
_EFFECT_OPERATIONS: dict[
    ResponseExperimentStageRole,
    ProspectiveAuthorityOperation | None,
] = {
    ResponseExperimentStageRole.SOURCE_READINESS: None,
    ResponseExperimentStageRole.IDENTIFICATION_ISSUE: ProspectiveAuthorityOperation.ISSUE,
    ResponseExperimentStageRole.IDENTIFICATION_ACQUISITION: ProspectiveAuthorityOperation.EXECUTE,
    ResponseExperimentStageRole.IDENTIFICATION_REVEAL_AND_ADJUDICATION: ProspectiveAuthorityOperation.REVEAL,
    ResponseExperimentStageRole.ADMISSION_APPLICABILITY: None,
    ResponseExperimentStageRole.ADMISSION_ACQUISITION_AND_ADJUDICATION: ProspectiveAuthorityOperation.EXECUTE,
    ResponseExperimentStageRole.CONTROLLER_COMPILATION_AND_COMMIT: ProspectiveAuthorityOperation.EXECUTE,
    ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_APPLICABILITY: None,
    ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_ACQUISITION: ProspectiveAuthorityOperation.EXECUTE,
    ResponseExperimentStageRole.PROSPECTIVE_EVALUATION_REVEAL_AND_ADJUDICATION: ProspectiveAuthorityOperation.REVEAL,
    ResponseExperimentStageRole.EXPERIMENT_CLOSEOUT: None,
}


@dataclass(frozen=True, slots=True)
class ResponseExperimentStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-experiment-stage'

    stage_id: str
    role: ResponseExperimentStageRole
    predecessor_stage_id: str | None
    condition_id: str
    applicable: bool
    authority_operation: ProspectiveAuthorityOperation | None
    maximum_attempts: int
    runs_after_condition_false: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.stage_id, field_name="stage_id")
        validate_stable_id(self.condition_id, field_name="condition_id")
        if self.predecessor_stage_id is not None:
            validate_stable_id(
                self.predecessor_stage_id,
                field_name="predecessor_stage_id",
            )
        expected_operation = _EFFECT_OPERATIONS[self.role] if self.applicable else None
        if self.authority_operation is not expected_operation:
            raise ValueError("response experiment stage changes its effect authority operation")
        if self.maximum_attempts != 1:
            raise ValueError("scientific stage identities permit exactly one attempt")
        if self.runs_after_condition_false != (self.role is ResponseExperimentStageRole.EXPERIMENT_CLOSEOUT):
            raise ValueError("only exact closeout may run after condition-false")


@dataclass(frozen=True, slots=True)
class CompiledResponseExperiment(CanonicalRecord):
    "Outcome-blind conditional topology, not a RunPlan or authority grant."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-response-experiment'

    plan_id: str
    extension_set: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    linked_campaign_profile: ObjectIdentity
    stages: tuple[ResponseExperimentStage, ...]
    topology_sha256: str
    maximum_claim_ceiling: str
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        if len({value.stage_id for value in self.stages}) != len(self.stages):
            raise ValueError("compiled response experiment stage IDs repeat")
        validate_sha256(self.topology_sha256, field_name="topology_sha256")
        validate_nonempty(self.maximum_claim_ceiling, field_name="maximum_claim_ceiling")
        if tuple(value.role for value in self.stages) != _STAGE_ROLES:
            raise ValueError("compiled response experiment topology omits or reorders a stage")
        for index, stage in enumerate(self.stages):
            expected = None if index == 0 else self.stages[index - 1].stage_id
            if stage.predecessor_stage_id != expected:
                raise ValueError("compiled response experiment stage predecessor chain differs")
        if self.topology_sha256 != _topology_sha256(self.stages):
            raise ValueError("compiled response experiment topology digest differs")
        if (
            self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("compiled response experiment topology exceeds prospective authority")


@dataclass(frozen=True, slots=True)
class ResponseExperimentCompilationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-experiment-compilation-receipt'

    receipt_id: str
    extension_set: ObjectIdentity
    compiled_plan: ObjectIdentity
    strict_root_identities: tuple[ObjectIdentity, ...]
    extension_config_identities: tuple[ObjectIdentity, ...]
    identification_admission_physical_independent_unit_count: int
    identification_acquisition_group_count: int
    identification_nested_view_count: int
    prospective_evaluation_physical_independent_unit_count: int
    topology_sha256: str
    issued: bool
    execution_ready: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(self.topology_sha256, field_name="topology_sha256")
        if len(self.strict_root_identities) != 3 or len(set(self.strict_root_identities)) != 3:
            raise ValueError("response experiment compilation receipt requires three strict roots")
        if not 1 <= len(self.extension_config_identities) <= 3 or len(
            set(self.extension_config_identities)
        ) != len(self.extension_config_identities):
            raise ValueError("response experiment compilation receipt has invalid extension configs")
        for name in (
            'identification_admission_physical_independent_unit_count',
            'identification_acquisition_group_count',
            'identification_nested_view_count',
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.prospective_evaluation_physical_independent_unit_count < 0:
            raise ValueError("prospective_evaluation_physical_independent_unit_count must be nonnegative")
        if self.issued or self.execution_ready or self.grants_authority:
            raise ValueError("response experiment compilation cannot issue, authorize or claim readiness")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("response experiment compilation receipt must remain prospective")


def _topology_sha256(stages: tuple[ResponseExperimentStage, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            tuple(
                {
                    "role": value.role.value,
                    "predecessor_role": (None if index == 0 else stages[index - 1].role.value),
                    "authority_operation": (
                        None
                        if value.authority_operation is None
                        else value.authority_operation.value
                    ),
                    "applicable": value.applicable,
                    "maximum_attempts": value.maximum_attempts,
                    "runs_after_condition_false": value.runs_after_condition_false,
                }
                for index, value in enumerate(stages)
            )
        )
    ).hexdigest()


def _build_stages(
    plan_id: str,
    *,
    admission_applicable: bool = True,
    prospective_evaluation_applicable: bool = True,
) -> tuple[ResponseExperimentStage, ...]:
    if prospective_evaluation_applicable and not admission_applicable:
        raise ValueError("controller use stage applicability requires admission")
    stages: list[ResponseExperimentStage] = []
    for role in _STAGE_ROLES:
        applicable = not (
            (role in _ADMISSION_STAGE_ROLES and not admission_applicable)
            or (role in _PROSPECTIVE_EVALUATION_STAGE_ROLES and not prospective_evaluation_applicable)
        )
        slug = role.value.lower().replace("_", "-")
        stage_id = f"{plan_id}.{slug}"
        stages.append(
            ResponseExperimentStage(
                stage_id=stage_id,
                role=role,
                predecessor_stage_id=stages[-1].stage_id if stages else None,
                condition_id=f"condition.{plan_id}.{slug}",
                applicable=applicable,
                authority_operation=_EFFECT_OPERATIONS[role] if applicable else None,
                maximum_attempts=1,
                runs_after_condition_false=role is ResponseExperimentStageRole.EXPERIMENT_CLOSEOUT,
            )
        )
    return tuple(stages)


def _identity(record_id: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(record_id, record)


def _require_owner(
    profile: LinkedCampaignProfile,
    role: LinkedCampaignOwnerRole,
    expected: ObjectIdentity | None,
) -> None:
    observed = next(value.owner for value in profile.owners if value.role is role)
    if expected is not None and observed != expected:
        raise ValueError(f"parameterised config changes linked owner {role.value}")


def derive_response_experiment_topology(
    extension_set: ResponseExperimentExtensionSet,
) -> CompiledResponseExperiment:
    """Derive the one topology already owned by an authenticated extension.

    This is the common topology constructor used by strict-root compilation and
    preexecution replay.  It validates no substitute strict roots, performs no
    source access and grants no issue or execution authority.
    """

    identification_config = extension_set.identification_config
    admission_config = extension_set.admission_config
    plan_id = f"response-experiment-plan.{extension_set.extension_set_id}"
    stages = _build_stages(
        plan_id,
        admission_applicable=admission_config is not None,
        prospective_evaluation_applicable=extension_set.prospective_evaluation_config is not None,
    )
    return CompiledResponseExperiment(
        plan_id=plan_id,
        extension_set=_identity(extension_set.extension_set_id, extension_set),
        evidence_profile_selection=extension_set.evidence_profile_selection,
        source_pipeline_profile=extension_set.source_pipeline_profile,
        linked_campaign_profile=extension_set.linked_campaign_profile,
        stages=stages,
        topology_sha256=_topology_sha256(stages),
        maximum_claim_ceiling=(
            extension_set.prospective_evaluation_config.evaluation_template.maximum_claim_ceiling
            if extension_set.prospective_evaluation_config is not None
            else admission_config.finite_intervention_claim_ceiling
            if admission_config is not None
            else identification_config.maximum_evidence_ceiling.value
        ),
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def compile_response_experiment(
    *,
    extension_set: ResponseExperimentExtensionSet,
    evidence_profile_selection: EvidenceProfileSelection,
    source_pipeline_profile: SourcePipelineProfile | NativeSourceProfile,
    linked_campaign_profile: LinkedCampaignProfile | CampaignSpec,
) -> tuple[CompiledResponseExperiment, ResponseExperimentCompilationReceipt]:
    """Compile exact strict roots and extensions without issue or outcome access."""

    native = isinstance(extension_set, NativeLawQualificationExperiment)
    if native != isinstance(linked_campaign_profile, CampaignSpec):
        raise ValueError(
            "native response qualification requires its actual CampaignSpec; linked response experiment its linked profile"
        )
    if native != isinstance(source_pipeline_profile, NativeSourceProfile):
        raise ValueError("parameterised compilation cannot mix static and native source roots")
    if isinstance(source_pipeline_profile, NativeSourceProfile) and (
        source_pipeline_profile.physical_independent_unit_ids
        != extension_set.identification_config.identification_admission_physical_independent_unit_ids
    ):
        raise ValueError("native source profile changes the exact independent-unit roster")

    evidence_identity = _identity(
        evidence_profile_selection.selection_id,
        evidence_profile_selection,
    )
    source_identity = _identity(source_pipeline_profile.profile_id, source_pipeline_profile)
    linked_identity = _identity(
        linked_campaign_profile.campaign_id
        if isinstance(linked_campaign_profile, CampaignSpec)
        else linked_campaign_profile.profile_id,
        linked_campaign_profile,
    )
    if (
        extension_set.evidence_profile_selection != evidence_identity
        or extension_set.source_pipeline_profile != source_identity
        or extension_set.linked_campaign_profile != linked_identity
    ):
        raise ValueError("parameterised extension set differs from decoded strict roots")
    identification_config = extension_set.identification_config
    if isinstance(
        linked_campaign_profile, CampaignSpec
    ) and not linked_campaign_profile.budget.contains(source_pipeline_profile.resource_ceiling):
        raise ValueError("native source resource ceiling exceeds its actual campaign")
    if isinstance(linked_campaign_profile, LinkedCampaignProfile):
        if (
            linked_campaign_profile.evidence_profile_selection != evidence_identity
            or linked_campaign_profile.source_profile != source_identity
        ):
            raise ValueError("linked campaign differs from its evidence/source strict roots")

        rosters = linked_campaign_profile.rosters
        if set(identification_config.identification_admission_physical_independent_unit_ids) != set(
            rosters.physical_independent_unit_ids
        ):
            raise ValueError("identification and admission physical units differ from the linked campaign")
        if set(identification_config.numerical_member_ids) != set(rosters.denominator_member_ids):
            raise ValueError("response qualification numerical members differ from the linked campaign")
        identification_action_word_ids = (
            set()
            if identification_config.action_chart is None
            else {value.word_id for value in identification_config.action_chart.action_words}
        )
        if identification_action_word_ids != set(rosters.action_word_ids):
            raise ValueError("response qualification action chart differs from the linked campaign")
        prospective_evaluation_independent_unit_ids = (
            set()
            if extension_set.prospective_evaluation_config is None
            else {
                value.physical_independent_unit_id
                for value in extension_set.prospective_evaluation_config.evaluation_template.units
            }
        )
        if prospective_evaluation_independent_unit_ids != set(rosters.controller_use_independent_unit_ids):
            raise ValueError("controller use physical units differ from the linked campaign")
        for expected, observed, label in (
            (identification_config.projection_config, linked_campaign_profile.projection_config, "projection"),
            (identification_config.method_config, linked_campaign_profile.method_config, "method"),
            (
                identification_config.qualification_profile,
                linked_campaign_profile.qualification_profile,
                "qualification",
            ),
        ):
            if expected != observed:
                raise ValueError(f'response qualification {label} config differs from linked campaign')
        _require_owner(
            linked_campaign_profile,
            LinkedCampaignOwnerRole.SOLE_LAW_FINALIZER,
            identification_config.law_finalizer_owner,
        )
        _require_owner(
            linked_campaign_profile,
            LinkedCampaignOwnerRole.BATCH_ATLAS,
            identification_config.batch_atlas_owner,
        )
        admission_config = extension_set.admission_config
        if admission_config is not None:
            _require_owner(
                linked_campaign_profile,
                LinkedCampaignOwnerRole.ADMISSION_EVIDENCE_PRODUCER,
                admission_config.raw_admission_receipt_owner,
            )
            _require_owner(
                linked_campaign_profile,
                LinkedCampaignOwnerRole.ADMISSION,
                admission_config.admission_owner,
            )
            _require_owner(
                linked_campaign_profile,
                LinkedCampaignOwnerRole.REACHABILITY,
                admission_config.reachability_owner,
            )
            _require_owner(
                linked_campaign_profile,
                LinkedCampaignOwnerRole.SOLE_CONTROLLER_COMPILER,
                admission_config.controller_compiler_owner,
            )
            _require_owner(
                linked_campaign_profile,
                LinkedCampaignOwnerRole.CONTROLLER_RUNTIME,
                admission_config.controller_runtime_owner,
            )

    extension_identity = _identity(extension_set.extension_set_id, extension_set)
    compiled = derive_response_experiment_topology(extension_set)
    topology_sha256 = compiled.topology_sha256
    prospective_validation_count = (
        0
        if extension_set.prospective_evaluation_config is None
        else len(extension_set.prospective_evaluation_config.evaluation_template.units)
    )
    receipt = ResponseExperimentCompilationReceipt(
        receipt_id=f"response-experiment-compilation.{extension_set.extension_set_id}",
        extension_set=extension_identity,
        compiled_plan=_identity(compiled.plan_id, compiled),
        strict_root_identities=(evidence_identity, source_identity, linked_identity),
        extension_config_identities=extension_set.config_identities,
        identification_admission_physical_independent_unit_count=len(identification_config.identification_admission_physical_independent_unit_ids),
        identification_acquisition_group_count=len(identification_config.acquisition_groups),
        identification_nested_view_count=len(identification_config.nested_views),
        prospective_evaluation_physical_independent_unit_count=prospective_validation_count,
        topology_sha256=topology_sha256,
        issued=False,
        execution_ready=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return compiled, receipt


_DECODER_TYPES: tuple[type[CanonicalRecord], ...] = (
    ResponseQualificationConfig,
    AdmissionExperimentConfig,
    ProspectiveUseExperimentConfig,
    ResponseExperimentExtensionSet,
)


def build_response_experiment_decoder_registrations(
    *,
    implementation_sha256: str,
    record_types: tuple[type[CanonicalRecord], ...] = _DECODER_TYPES,
) -> tuple[StudyExtensionDecoderRegistration, ...]:
    """Build an injected closed decoder roster; this does not activate a public route."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    registrations = []
    for record_type in record_types:
        slug = record_type.SCHEMA.removeprefix("empirical-lawhood/").replace("/", ".")
        decoder_key = f"parameterised-response.{slug}"
        config_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "decoder_key": decoder_key,
                    "decoder_version": "1.0.0",
                    "mode": "exact-canonical-record",
                    "payload_schema": record_type.SCHEMA,
                }
            )
        ).hexdigest()
        registrations.append(
            StudyExtensionDecoderRegistration(
                registration_id=f"decoder-registration.{slug}",
                decoder_key=decoder_key,
                decoder_version="1.0.0",
                payload_schema=record_type.SCHEMA,
                payload_version=record_type.VERSION,
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha256,
                maximum_payload_bytes=MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES,
            )
        )
    return tuple(sorted(registrations, key=lambda value: value.registration_id))


def build_response_experiment_codec_registry(
    *,
    implementation_sha256: str,
) -> CanonicalRecordCodecRegistry:
    """Closed direct-composition codec registry; not a generated public activation."""

    return build_canonical_record_codec_registry(
        registry_id="canonical-codecs.parameterised-response",
        record_types=_DECODER_TYPES,
        maximum_bytes_by_schema={
            value.SCHEMA: MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES for value in _DECODER_TYPES
        },
        decoder_implementation_sha256=implementation_sha256,
    )


@dataclass(frozen=True, slots=True)
class ResponseStageAuthorityReplayDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-stage-authority-replay-decision'

    decision_id: str
    plan: ObjectIdentity
    stage_id: str
    operation: ProspectiveAuthorityOperation
    authority: ObjectIdentity | None
    authorized: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_stable_id(self.stage_id, field_name="stage_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.authorized:
            if self.authority is None or self.reason_codes:
                raise ValueError("authorized stage requires exact authority and no reasons")
        elif self.authority is not None or not self.reason_codes:
            raise ValueError("unauthorized stage cannot fabricate authority")


@dataclass(frozen=True, slots=True)
class ResponseExperimentExecutionClosure(CanonicalRecord):
    "Exact existing execution plan/recovery join; it creates no execution semantics."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-experiment-execution-closure'

    closure_id: str
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    issued_extension_set: ObjectIdentity
    execution_resource_envelope_spec: ObjectIdentity
    predevelopment_jit_signature_census: ObjectIdentity | None
    jit_graph_signature_manifest: ObjectIdentity | None
    recovery_index: ObjectIdentity
    recovery_terminal: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.closure_id, field_name="closure_id")
        expected_schemas = (
            RunPlan.SCHEMA,
            ExecutionPlan.SCHEMA,
            'empirical-lawhood/runtime/issued-extension-set',
            'empirical-lawhood/runtime/execution-resource-envelope-spec',
            RunRecoveryIndex.SCHEMA,
            RunRecoveryTerminalEvent.SCHEMA,
        )
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )
        observed_schemas = tuple(
            value.object_schema
            for value in (
                self.run_plan,
                self.execution_plan,
                self.issued_extension_set,
                self.execution_resource_envelope_spec,
                self.recovery_index,
                self.recovery_terminal,
            )
        )
        if observed_schemas != expected_schemas:
            raise ValueError("response experiment execution closure contains another plan/recovery schema")


def build_response_experiment_execution_closure(
    *,
    closure_id: str,
    run_plan: RunPlan,
    execution_plan: ExecutionPlan,
    recovery_index: RunRecoveryIndex,
    recovery_terminal: RunRecoveryTerminalEvent,
) -> ResponseExperimentExecutionClosure:
    "Join records already emitted by the existing issued execution route."

    run_identity = _identity(run_plan.run_plan_id, run_plan)
    execution_identity = _identity(execution_plan.execution_plan_id, execution_plan)
    if execution_plan.source_plan != run_identity:
        raise ValueError("execution plan was lowered from another run plan")
    run_bindings = (
        run_plan.issued_extension_set,
        run_plan.execution_resource_envelope_spec,
        run_plan.predevelopment_jit_signature_census,
        run_plan.jit_graph_signature_manifest,
    )
    execution_bindings = (
        execution_plan.issued_extension_set,
        execution_plan.execution_resource_envelope_spec,
        execution_plan.predevelopment_jit_signature_census,
        execution_plan.jit_graph_signature_manifest,
    )
    if execution_bindings != run_bindings:
        raise ValueError("run/execution resource or issued-extension bindings differ")
    recovery_index.validate_plan(execution_plan)
    recovery_identity = _identity(recovery_index.recovery_index_id, recovery_index)
    if (
        recovery_terminal.recovery_index != recovery_identity
        or recovery_terminal.run_id != run_plan.run_plan_id
    ):
        raise ValueError("recovery terminal closes another run/index")
    return ResponseExperimentExecutionClosure(
        closure_id=closure_id,
        run_plan=run_identity,
        execution_plan=execution_identity,
        issued_extension_set=run_bindings[0],
        execution_resource_envelope_spec=run_bindings[1],
        predevelopment_jit_signature_census=run_bindings[2],
        jit_graph_signature_manifest=run_bindings[3],
        recovery_index=recovery_identity,
        recovery_terminal=_identity(recovery_terminal.terminal_event_id, recovery_terminal),
    )


@dataclass(frozen=True, slots=True)
class ResponseStageOwnerResult(CanonicalRecord):
    """Opaque result from one existing scientific or operational stage owner."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-stage-owner-result'

    result_id: str
    plan: ObjectIdentity
    stage_id: str
    role: ResponseExperimentStageRole
    operational_status: OperationalStatus
    scientific_status: ScientificStatus
    obstruction: ResponseExperimentObstruction
    execution_basis: ResponseStageExecutionBasis
    execution_closure: ObjectIdentity | None
    scientific_product: ObjectIdentity | None
    condition_satisfied: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.stage_id, field_name="stage_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.operational_status not in {
            OperationalStatus.SUCCEEDED,
            OperationalStatus.FAILED,
            OperationalStatus.BLOCKED,
            OperationalStatus.SKIPPED,
        }:
            raise ValueError("stage owner result is not terminal")
        if self.execution_basis is ResponseStageExecutionBasis.ISSUED_EXECUTION:
            if (
                self.execution_closure is None
                or self.execution_closure.object_schema != ResponseExperimentExecutionClosure.SCHEMA
            ):
                raise ValueError("issued-execution owner result lacks its exact execution closure")
        elif self.execution_closure is not None:
            raise ValueError("owner result outside issued execution fabricates an execution closure")
        if (
            self.scientific_status is not ScientificStatus.NOT_TESTED
            and self.execution_basis is ResponseStageExecutionBasis.NONEXECUTING
        ):
            raise ValueError("scientific owner result lacks an execution/conformance basis")
        if self.scientific_status is ScientificStatus.SUPPORTED:
            if (
                self.operational_status is not OperationalStatus.SUCCEEDED
                or self.scientific_product is None
                or self.obstruction is not ResponseExperimentObstruction.NONE
                or self.reason_codes
                or not self.condition_satisfied
            ):
                raise ValueError("supported owner result is inconsistent")
        elif self.scientific_status is ScientificStatus.NOT_TESTED:
            if self.scientific_product is not None:
                raise ValueError("not-tested owner result cannot contain scientific product")
            if self.operational_status is OperationalStatus.SUCCEEDED:
                if self.obstruction is not ResponseExperimentObstruction.NONE or self.reason_codes:
                    raise ValueError("successful non-scientific stage carries obstruction")
            elif self.obstruction is ResponseExperimentObstruction.NONE or not self.reason_codes:
                raise ValueError("non-success stage requires an exact obstruction")
        elif (
            self.operational_status is not OperationalStatus.SUCCEEDED
            or self.scientific_product is None
            or self.obstruction is ResponseExperimentObstruction.NONE
            or not self.reason_codes
        ):
            raise ValueError("adverse scientific result must retain product and obstruction")


@dataclass(frozen=True, slots=True)
class ResponseStageAttemptReservation(CanonicalRecord):
    """No-replace reservation written before an owner may incur an effect."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-stage-attempt-reservation'

    reservation_id: str
    plan: ObjectIdentity
    stage_id: str
    role: ResponseExperimentStageRole
    predecessor: ObjectIdentity | None
    attempt_ordinal: int
    owner_effect_may_be_incurred: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.reservation_id, field_name="reservation_id")
        validate_stable_id(self.stage_id, field_name="stage_id")
        if self.attempt_ordinal != 1:
            raise ValueError("response experiment stage permits exactly one reserved attempt")
        if self.owner_effect_may_be_incurred != (_EFFECT_OPERATIONS[self.role] is not None):
            raise ValueError("stage reservation changes its effect classification")


@dataclass(frozen=True, slots=True)
class ResponseStageTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-stage-terminal'

    terminal_id: str
    plan: ObjectIdentity
    stage: ResponseExperimentStage
    predecessor: ObjectIdentity | None
    attempt_reservation: ObjectIdentity | None
    authority_decision: ObjectIdentity | None
    owner_result: ObjectIdentity | None
    operational_status: OperationalStatus
    scientific_status: ScientificStatus
    obstruction: ResponseExperimentObstruction
    scientific_product: ObjectIdentity | None
    condition_satisfied: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (self.stage.predecessor_stage_id is None) != (self.predecessor is None):
            raise ValueError("stage terminal predecessor presence differs from topology")
        if self.stage.authority_operation is None and self.authority_decision is not None:
            raise ValueError("effect-free stage cannot retain an authority decision")
        if not self.stage.applicable:
            if (
                self.attempt_reservation is not None
                or self.authority_decision is not None
                or self.owner_result is not None
                or self.operational_status is not OperationalStatus.SKIPPED
                or self.scientific_status is not ScientificStatus.NOT_TESTED
                or self.obstruction is not ResponseExperimentObstruction.STAGE_NOT_APPLICABLE
                or self.scientific_product is not None
                or self.condition_satisfied
                or self.reason_codes != (STAGE_NOT_APPLICABLE_REASON,)
            ):
                raise ValueError("inapplicable stage terminal fabricates an attempt or result")
            return
        if self.obstruction is ResponseExperimentObstruction.STAGE_NOT_APPLICABLE:
            raise ValueError("applicable stage cannot use the not-applicable terminal")
        if (
            self.attempt_reservation is not None
            and self.attempt_reservation.object_schema != ResponseStageAttemptReservation.SCHEMA
        ):
            raise ValueError("stage terminal binds another attempt-reservation schema")
        if self.operational_status is OperationalStatus.SKIPPED:
            if self.attempt_reservation is not None:
                raise ValueError("nonattempt stage cannot claim an attempt reservation")
        elif self.attempt_reservation is None:
            raise ValueError("attempted stage terminal requires its no-replace reservation")
        if self.operational_status is OperationalStatus.SUCCEEDED and self.owner_result is None:
            raise ValueError("successful stage terminal requires an owner result")
        if self.operational_status in {OperationalStatus.SKIPPED, OperationalStatus.BLOCKED}:
            if self.owner_result is not None or self.condition_satisfied:
                raise ValueError("synthetic nonattempt/authority terminal is inconsistent")
        if self.scientific_status is ScientificStatus.NOT_TESTED and self.scientific_product:
            raise ValueError("not-tested terminal cannot contain a scientific product")


class ResponseStageExecutor(Protocol):
    def execute_stage(
        self,
        *,
        plan: CompiledResponseExperiment,
        stage: ResponseExperimentStage,
        predecessor: ResponseStageTerminal | None,
    ) -> ResponseStageOwnerResult: ...


class ResponseStageAuthorityReplayPort(Protocol):
    def replay_authority(
        self,
        *,
        plan: CompiledResponseExperiment,
        stage: ResponseExperimentStage,
    ) -> ResponseStageAuthorityReplayDecision: ...


class ResponseStageReceiptStore(Protocol):
    def load_stage_attempt_reservation(
        self,
        *,
        plan_id: str,
        stage_id: str,
    ) -> ResponseStageAttemptReservation | None: ...

    def reserve_stage_attempt(self, reservation: ResponseStageAttemptReservation) -> None: ...

    def load_stage_terminal(
        self,
        *,
        plan_id: str,
        stage_id: str,
    ) -> ResponseStageTerminal | None: ...

    def put_stage_terminal(self, terminal: ResponseStageTerminal) -> None: ...


@dataclass(frozen=True, slots=True)
class ResponseExperimentTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-experiment-terminal'

    terminal_id: str
    plan: ObjectIdentity
    stage_terminals: tuple[ObjectIdentity, ...]
    terminal_operational_status: OperationalStatus
    terminal_scientific_status: ScientificStatus
    claim_ceiling: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal_id")
        validate_nonempty(self.claim_ceiling, field_name="claim_ceiling")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.stage_terminals:
            raise ValueError("experiment terminal requires stage receipts")
        if self.terminal_operational_status not in {
            OperationalStatus.SUCCEEDED,
            OperationalStatus.FAILED,
            OperationalStatus.BLOCKED,
        }:
            raise ValueError("experiment terminal has a nonterminal operational status")


class ResponseExperimentSpooler:
    """Idempotent condition/authority sequencer; existing owners do all science."""

    def __init__(
        self,
        *,
        executor: ResponseStageExecutor,
        authority: ResponseStageAuthorityReplayPort,
        store: ResponseStageReceiptStore,
    ) -> None:
        self._executor = executor
        self._authority = authority
        self._store = store

    @staticmethod
    def _terminal_identity(value: ResponseStageTerminal) -> ObjectIdentity:
        return _identity(value.terminal_id, value)

    def _persist(self, terminal: ResponseStageTerminal) -> ResponseStageTerminal:
        self._store.put_stage_terminal(terminal)
        return terminal

    def _nonattempt(
        self,
        *,
        plan: CompiledResponseExperiment,
        stage: ResponseExperimentStage,
        predecessor: ResponseStageTerminal,
    ) -> ResponseStageTerminal:
        return self._persist(
            ResponseStageTerminal(
                terminal_id=f"stage-terminal.{stage.stage_id}",
                plan=_identity(plan.plan_id, plan),
                stage=stage,
                predecessor=self._terminal_identity(predecessor),
                attempt_reservation=None,
                authority_decision=None,
                owner_result=None,
                operational_status=OperationalStatus.SKIPPED,
                scientific_status=ScientificStatus.NOT_TESTED,
                obstruction=ResponseExperimentObstruction.PREREQUISITE_NONATTEMPT,
                scientific_product=None,
                condition_satisfied=False,
                reason_codes=("UPSTREAM_CONDITION_FALSE",),
            )
        )

    def _not_applicable(
        self,
        *,
        plan: CompiledResponseExperiment,
        stage: ResponseExperimentStage,
        predecessor: ResponseStageTerminal | None,
    ) -> ResponseStageTerminal:
        return self._persist(
            ResponseStageTerminal(
                terminal_id=f"stage-terminal.{stage.stage_id}",
                plan=_identity(plan.plan_id, plan),
                stage=stage,
                predecessor=(None if predecessor is None else self._terminal_identity(predecessor)),
                attempt_reservation=None,
                authority_decision=None,
                owner_result=None,
                operational_status=OperationalStatus.SKIPPED,
                scientific_status=ScientificStatus.NOT_TESTED,
                obstruction=ResponseExperimentObstruction.STAGE_NOT_APPLICABLE,
                scientific_product=None,
                condition_satisfied=False,
                reason_codes=(STAGE_NOT_APPLICABLE_REASON,),
            )
        )

    @staticmethod
    def _reservation_identity(
        value: ResponseStageAttemptReservation,
    ) -> ObjectIdentity:
        return _identity(value.reservation_id, value)

    def _reserve_attempt(
        self,
        *,
        plan: CompiledResponseExperiment,
        stage: ResponseExperimentStage,
        predecessor: ResponseStageTerminal | None,
    ) -> ResponseStageAttemptReservation:
        reservation = ResponseStageAttemptReservation(
            reservation_id=f"stage-attempt.{stage.stage_id}.attempt-001",
            plan=_identity(plan.plan_id, plan),
            stage_id=stage.stage_id,
            role=stage.role,
            predecessor=(None if predecessor is None else self._terminal_identity(predecessor)),
            attempt_ordinal=1,
            owner_effect_may_be_incurred=stage.authority_operation is not None,
        )
        self._store.reserve_stage_attempt(reservation)
        return reservation

    def _unresolved_reserved_attempt(
        self,
        *,
        plan: CompiledResponseExperiment,
        stage: ResponseExperimentStage,
        predecessor: ResponseStageTerminal | None,
        reservation: ResponseStageAttemptReservation,
    ) -> ResponseStageTerminal:
        expected_predecessor = None if predecessor is None else self._terminal_identity(predecessor)
        if (
            reservation.plan != _identity(plan.plan_id, plan)
            or reservation.stage_id != stage.stage_id
            or reservation.role is not stage.role
            or reservation.predecessor != expected_predecessor
        ):
            raise ValueError("persisted attempt reservation belongs to another stage")
        return self._persist(
            ResponseStageTerminal(
                terminal_id=f"stage-terminal.{stage.stage_id}",
                plan=_identity(plan.plan_id, plan),
                stage=stage,
                predecessor=expected_predecessor,
                attempt_reservation=self._reservation_identity(reservation),
                authority_decision=None,
                owner_result=None,
                operational_status=OperationalStatus.BLOCKED,
                scientific_status=ScientificStatus.NOT_TESTED,
                obstruction=ResponseExperimentObstruction.UNRESOLVED,
                scientific_product=None,
                condition_satisfied=False,
                reason_codes=("PRIOR_ATTEMPT_TERMINAL_ABSENT_NO_RERUN",),
            )
        )

    def run(
        self,
        *,
        plan: CompiledResponseExperiment,
    ) -> tuple[ResponseStageTerminal, ...]:
        plan_identity = _identity(plan.plan_id, plan)
        terminals: list[ResponseStageTerminal] = []
        for stage in plan.stages:
            existing = self._store.load_stage_terminal(
                plan_id=plan.plan_id,
                stage_id=stage.stage_id,
            )
            if existing is not None:
                if existing.plan != plan_identity or existing.stage != stage:
                    raise ValueError("persisted stage terminal belongs to another plan/stage")
                if stage.predecessor_stage_id is not None and (
                    not terminals or existing.predecessor != self._terminal_identity(terminals[-1])
                ):
                    raise ValueError("persisted stage terminal has another predecessor")
                terminals.append(existing)
                continue

            predecessor = terminals[-1] if terminals else None
            if not stage.applicable:
                terminals.append(
                    self._not_applicable(
                        plan=plan,
                        stage=stage,
                        predecessor=predecessor,
                    )
                )
                continue
            if (
                predecessor is not None
                and not predecessor.condition_satisfied
                and not stage.runs_after_condition_false
            ):
                terminals.append(self._nonattempt(plan=plan, stage=stage, predecessor=predecessor))
                continue

            prior_reservation = self._store.load_stage_attempt_reservation(
                plan_id=plan.plan_id,
                stage_id=stage.stage_id,
            )
            if prior_reservation is not None:
                terminals.append(
                    self._unresolved_reserved_attempt(
                        plan=plan,
                        stage=stage,
                        predecessor=predecessor,
                        reservation=prior_reservation,
                    )
                )
                continue

            reservation = self._reserve_attempt(
                plan=plan,
                stage=stage,
                predecessor=predecessor,
            )
            reservation_identity = self._reservation_identity(reservation)

            authority_identity: ObjectIdentity | None = None
            if stage.authority_operation is not None:
                try:
                    decision = self._authority.replay_authority(plan=plan, stage=stage)
                except Exception as error:
                    terminal = ResponseStageTerminal(
                        terminal_id=f"stage-terminal.{stage.stage_id}",
                        plan=plan_identity,
                        stage=stage,
                        predecessor=(
                            None if predecessor is None else self._terminal_identity(predecessor)
                        ),
                        attempt_reservation=reservation_identity,
                        authority_decision=None,
                        owner_result=None,
                        operational_status=OperationalStatus.FAILED,
                        scientific_status=ScientificStatus.NOT_TESTED,
                        obstruction=ResponseExperimentObstruction.TECHNICAL_FAILURE,
                        scientific_product=None,
                        condition_satisfied=False,
                        reason_codes=(
                            f"AUTHORITY_REPLAY_EXCEPTION_{type(error).__name__.upper()}",
                        ),
                    )
                    terminals.append(self._persist(terminal))
                    continue
                if (
                    decision.plan != plan_identity
                    or decision.stage_id != stage.stage_id
                    or decision.operation is not stage.authority_operation
                ):
                    raise ValueError("authority replay decision differs from exact stage")
                authority_identity = _identity(decision.decision_id, decision)
                if not decision.authorized:
                    terminal = ResponseStageTerminal(
                        terminal_id=f"stage-terminal.{stage.stage_id}",
                        plan=plan_identity,
                        stage=stage,
                        predecessor=(
                            None if predecessor is None else self._terminal_identity(predecessor)
                        ),
                        attempt_reservation=reservation_identity,
                        authority_decision=authority_identity,
                        owner_result=None,
                        operational_status=OperationalStatus.BLOCKED,
                        scientific_status=ScientificStatus.NOT_TESTED,
                        obstruction=ResponseExperimentObstruction.AUTHORITY_REQUIRED,
                        scientific_product=None,
                        condition_satisfied=False,
                        reason_codes=decision.reason_codes,
                    )
                    terminals.append(self._persist(terminal))
                    continue

            try:
                result = self._executor.execute_stage(
                    plan=plan,
                    stage=stage,
                    predecessor=predecessor,
                )
            except Exception as error:  # sequencer records, but never retries, an owner crash
                result = ResponseStageOwnerResult(
                    result_id=f"owner-result.{stage.stage_id}.technical-failure",
                    plan=plan_identity,
                    stage_id=stage.stage_id,
                    role=stage.role,
                    operational_status=OperationalStatus.FAILED,
                    scientific_status=ScientificStatus.NOT_TESTED,
                    obstruction=ResponseExperimentObstruction.TECHNICAL_FAILURE,
                    execution_basis=ResponseStageExecutionBasis.NONEXECUTING,
                    execution_closure=None,
                    scientific_product=None,
                    condition_satisfied=False,
                    reason_codes=(f"OWNER_EXCEPTION_{type(error).__name__.upper()}",),
                )
            if (
                result.plan != plan_identity
                or result.stage_id != stage.stage_id
                or result.role is not stage.role
            ):
                raise ValueError("stage owner returned a result for another stage")
            terminal = ResponseStageTerminal(
                terminal_id=f"stage-terminal.{stage.stage_id}",
                plan=plan_identity,
                stage=stage,
                predecessor=(None if predecessor is None else self._terminal_identity(predecessor)),
                attempt_reservation=reservation_identity,
                authority_decision=authority_identity,
                owner_result=_identity(result.result_id, result),
                operational_status=result.operational_status,
                scientific_status=result.scientific_status,
                obstruction=result.obstruction,
                scientific_product=result.scientific_product,
                condition_satisfied=result.condition_satisfied,
                reason_codes=result.reason_codes,
            )
            terminals.append(self._persist(terminal))
        return tuple(terminals)

    def run_to_terminal(
        self,
        *,
        plan: CompiledResponseExperiment,
    ) -> tuple[tuple[ResponseStageTerminal, ...], ResponseExperimentTerminal]:
        terminals = self.run(plan=plan)
        failed = any(value.operational_status is OperationalStatus.FAILED for value in terminals)
        blocked = any(value.operational_status is OperationalStatus.BLOCKED for value in terminals)
        operational_status = (
            OperationalStatus.FAILED
            if failed
            else OperationalStatus.BLOCKED
            if blocked
            else OperationalStatus.SUCCEEDED
        )
        scientific_status = next(
            (
                value.scientific_status
                for value in reversed(terminals)
                if value.scientific_status is not ScientificStatus.NOT_TESTED
            ),
            ScientificStatus.NOT_TESTED,
        )
        reason_codes = tuple(
            sorted({reason for value in terminals for reason in value.reason_codes})
        )
        experiment_terminal = ResponseExperimentTerminal(
            terminal_id=f"experiment-terminal.{plan.plan_id}",
            plan=_identity(plan.plan_id, plan),
            stage_terminals=tuple(self._terminal_identity(value) for value in terminals),
            terminal_operational_status=operational_status,
            terminal_scientific_status=scientific_status,
            claim_ceiling=plan.maximum_claim_ceiling,
            reason_codes=reason_codes,
        )
        return terminals, experiment_terminal


__all__ = [
    'CompiledResponseExperiment',
    'ResponseStageAuthorityReplayDecision',
    'ResponseStageAuthorityReplayPort',
    'ResponseExperimentTerminal',
    'ResponseExperimentObstruction',
    'ResponseStageExecutionBasis',
    'ResponseExperimentCompilationReceipt',
    'ResponseStageExecutor',
    'ResponseStageAttemptReservation',
    'ResponseStageOwnerResult',
    'ResponseStageReceiptStore',
    'ResponseExperimentStageRole',
    'ResponseExperimentStage',
    'ResponseStageTerminal',
    'ResponseExperimentExecutionClosure',
    "STAGE_NOT_APPLICABLE_REASON",
    'ResponseExperimentSpooler',
    'build_response_experiment_codec_registry',
    'build_response_experiment_decoder_registrations',
    'build_response_experiment_execution_closure',
    'compile_response_experiment',
    'derive_response_experiment_topology',
]
