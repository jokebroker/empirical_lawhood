"""Conditional FreeGSNKE prospective validation issue, sealed execution and reveal DAG.

The child topology is frozen before admission evaluation reveal.  admission evaluation selects only the exact
action fibre carried by the issue record; it cannot change the prospective validation preparation
roster, source, numerical view, action chart, analysis grid, authorities or
evaluator.  Each fresh preparation is one execution node and returns one
sealed batch containing its nested selected-action/hold views.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import ClassVar, Final

from empirical_lawhood.adapters.methods.structural_recurrence import PolicyBranch
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_authoring import ConditionalGateKind, ConditionalChildRequest, ConditionalTerminalDisposition
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationReport, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, StudyCompilationReport
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.conditional_children import ConditionalChildInstantiation, ConditionalChildResolution, FrozenParentInputBinding, bind_frozen_parent_input, instantiate_conditional_child
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .contracts import FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA, FreeGsnkeActionChart, FreeGsnkeNumericalView, FreeGsnkePhase, FreeGsnkePreparation, FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetProcessResponse, FreeGsnkeTargetWorkerInput, freegsnke_saved_preparation_artifact
from .design import FreeGsnkeActionDesign
from .generation_protocol import FreeGsnkeEpisodeExecutor
from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceBridgeFreeze, FreeGsnkeStructuralRecurrenceDesignCandidate
from .registry import FREEGSNKE_CAPABILITY_VERSION
from .runtime import decode_freegsnke_target_response
from .target_analysis import FreeGsnkeTargetReductionConfig
from .target_design import FreeGsnkePowerFreeze
from .target_evaluation import FreeGsnkeAdmissionEvaluationEvaluation
from .target_validation import FreeGsnkeProspectiveValidationReductionConfig, FreeGsnkeProspectiveValidationReduction, FreeGsnkeProspectiveValidationRequestRoster, FreeGsnkeTargetValidation, build_freegsnke_prospective_requests, evaluate_freegsnke_prospective, freegsnke_prospective_parent_eligible, freeze_freegsnke_prospective_reduction_config, issue_freegsnke_prospective_roster, reduce_freegsnke_prospective_panel, selected_freegsnke_prospective_target_branches


FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY: Final = 'freegsnke.issue-prospective-validation-roster'
FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY: Final = 'freegsnke.execute-prospective-validation-unit'
FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY: Final = 'freegsnke.evaluate-prospective-validation-panel'
FREEGSNKE_PROSPECTIVE_VALIDATION_PROVIDER_KEY: Final = "independent-substrate-grounding.freegsnke-prospective-validation-provider"
FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID: Final = "parent-receipt.freegsnke.admission-evaluation"
FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_ARTIFACT_ID: Final = "artifact.parent-receipt.freegsnke.admission-evaluation"
FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID: Final = 'issue-prospective-validation-roster'
FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID: Final = 'evaluate-prospective-validation-panel'


class FreeGsnkeProspectiveValidationProtocolOperation(StrEnum):
    ISSUE = "ISSUE"
    EXECUTE_UNIT = "EXECUTE_UNIT"
    EVALUATE = "EVALUATE"


_OPERATION_KEY = {
    FreeGsnkeProspectiveValidationProtocolOperation.ISSUE: FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY,
    FreeGsnkeProspectiveValidationProtocolOperation.EXECUTE_UNIT: FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY,
    FreeGsnkeProspectiveValidationProtocolOperation.EVALUATE: FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY,
}


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class FreeGsnkeProspectiveValidationProtocolConfig(CanonicalRecord):
    """Outcome-blind child topology and authority binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-prospective-validation-protocol-config'

    config_id: str
    operation: FreeGsnkeProspectiveValidationProtocolOperation
    capability_key: str
    capability_version: str
    unit_id: str | None
    roster_id: str
    reduction_config_id: str
    validation_id: str
    source_binding: ObjectIdentity
    numerical_view: ObjectIdentity
    action_charts: tuple[ObjectIdentity, ...]
    action_design: ObjectIdentity
    power_freeze: ObjectIdentity
    selected_design_candidate: ObjectIdentity
    structural_recurrence_bridge: ObjectIdentity
    parent_evaluation_config: ObjectIdentity
    preparation_identities: tuple[ObjectIdentity, ...]
    prospective_validation_unit_ids: tuple[str, ...]
    prospective_validation_issue_authority: ObjectIdentity
    prospective_validation_execution_authority: ObjectIdentity
    prospective_validation_reveal_authority: ObjectIdentity
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "capability_key",
            "roster_id",
            "reduction_config_id",
            "validation_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        if self.capability_key != _OPERATION_KEY[self.operation]:
            raise ValueError("FreeGSNKE prospective validation operation/capability differs")
        if self.capability_version != FREEGSNKE_CAPABILITY_VERSION:
            raise ValueError("FreeGSNKE prospective validation capability version differs")
        if self.operation is FreeGsnkeProspectiveValidationProtocolOperation.EXECUTE_UNIT:
            if self.unit_id is None:
                raise ValueError("FreeGSNKE prospective validation execution config lacks a unit")
            validate_stable_id(self.unit_id, field_name="unit_id")
            if self.unit_id not in self.prospective_validation_unit_ids:
                raise ValueError("FreeGSNKE prospective validation execution unit is outside the frozen roster")
        elif self.unit_id is not None:
            raise ValueError("only a FreeGSNKE prospective validation execution config may name a unit")
        expected_schemas = (
            (self.source_binding, FreeGsnkeSourceBinding.SCHEMA),
            (self.numerical_view, FreeGsnkeNumericalView.SCHEMA),
            (self.action_design, FreeGsnkeActionDesign.SCHEMA),
            (self.power_freeze, FreeGsnkePowerFreeze.SCHEMA),
            (
                self.selected_design_candidate,
                FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA,
            ),
            (self.structural_recurrence_bridge, FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA),
            (
                self.parent_evaluation_config,
                FreeGsnkeTargetReductionConfig.SCHEMA,
            ),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("FreeGSNKE prospective validation config operand schema differs")
        require_sorted_unique_ids(
            self.action_charts,
            attribute="object_id",
            field_name="action_charts",
        )
        if not self.action_charts or any(
            value.object_schema != FreeGsnkeActionChart.SCHEMA for value in self.action_charts
        ):
            raise ValueError("FreeGSNKE prospective validation action-chart roster differs")
        require_sorted_unique_ids(
            self.preparation_identities,
            attribute="object_id",
            field_name="preparation_identities",
        )
        if any(
            value.object_schema != FreeGsnkePreparation.SCHEMA
            for value in self.preparation_identities
        ):
            raise ValueError("FreeGSNKE prospective validation config preparation schema differs")
        require_sorted_unique_strings(
            self.prospective_validation_unit_ids,
            field_name="prospective_validation_unit_ids",
            allow_empty=False,
        )
        if not set(self.prospective_validation_unit_ids) <= {value.object_id for value in self.preparation_identities}:
            raise ValueError("FreeGSNKE prospective validation units are absent from the preparation freeze")
        if (
            len(
                {
                    self.prospective_validation_issue_authority,
                    self.prospective_validation_execution_authority,
                    self.prospective_validation_reveal_authority,
                }
            )
            != 3
        ):
            raise ValueError("FreeGSNKE prospective validation issue/execution/reveal authorities repeat")
        if not 0 < self.maximum_input_bytes <= 512 * 1024**2:
            raise ValueError("FreeGSNKE prospective validation input bound differs")


@dataclass(frozen=True, slots=True)
class FreeGsnkeProspectiveValidationUnitResponseBatch(CanonicalRecord):
    """One sealed prospective validation preparation with nested selected-action/hold responses."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-prospective-validation-unit-response-batch'

    batch_id: str
    prospective_validation_roster: ObjectIdentity
    unit_id: str
    preparation: ObjectIdentity
    requests: tuple[FreeGsnkeProcessRequest, ...]
    responses: tuple[FreeGsnkeTargetProcessResponse, ...]
    prospective_validation_execution_authority: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_stable_id(self.unit_id, field_name="unit_id")
        if self.prospective_validation_roster.object_schema != FreeGsnkeProspectiveValidationRequestRoster.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation batch roster schema differs")
        if (
            self.preparation.object_schema != FreeGsnkePreparation.SCHEMA
            or self.preparation.object_id != self.unit_id
        ):
            raise ValueError("FreeGSNKE prospective validation batch preparation differs")
        require_sorted_unique_ids(
            self.requests,
            attribute="request_id",
            field_name="requests",
        )
        require_sorted_unique_ids(
            self.responses,
            attribute="response_id",
            field_name="responses",
        )
        if not self.requests or len(self.requests) != len(self.responses):
            raise ValueError("FreeGSNKE prospective validation batch request/response counts differ")
        response_by_request = {value.request.object_id: value for value in self.responses}
        if set(response_by_request) != {value.request_id for value in self.requests}:
            raise ValueError("FreeGSNKE prospective validation batch response roster differs")
        if any(
            value.phase is not FreeGsnkePhase.PROSPECTIVE_VALIDATION
            or value.preparation.preparation_id != self.unit_id
            or response_by_request[value.request_id].request
            != ObjectIdentity.from_record(value.request_id, value)
            for value in self.requests
        ):
            raise ValueError("FreeGSNKE prospective validation batch request binding differs")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("FreeGSNKE prospective validation unit batch must remain sealed")


def freegsnke_prospective_protocol_configs(
    *,
    source_binding: FreeGsnkeSourceBinding,
    numerical_view: FreeGsnkeNumericalView,
    action_charts: tuple[FreeGsnkeActionChart, ...],
    action_design: FreeGsnkeActionDesign,
    power_freeze: FreeGsnkePowerFreeze,
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    parent_evaluation_config: FreeGsnkeTargetReductionConfig,
    preparations: tuple[FreeGsnkePreparation, ...],
    prospective_validation_issue_authority: ObjectIdentity,
    prospective_validation_execution_authority: ObjectIdentity,
    prospective_validation_reveal_authority: ObjectIdentity,
    namespace: str,
) -> tuple[FreeGsnkeProspectiveValidationProtocolConfig, ...]:
    """Freeze one issue, one node per prospective validation unit and one evaluator config."""

    validate_stable_id(namespace, field_name="namespace")
    ordered_preparations = tuple(sorted(preparations, key=lambda value: value.preparation_id))
    source_identity = ObjectIdentity.from_record(
        source_binding.binding_id,
        source_binding,
    )
    view_identity = ObjectIdentity.from_record(numerical_view.view_id, numerical_view)
    chart_identities = tuple(
        sorted(
            (ObjectIdentity.from_record(value.chart_id, value) for value in action_charts),
            key=lambda value: value.object_id,
        )
    )
    if not chart_identities or len(chart_identities) != len(action_charts):
        raise ValueError("FreeGSNKE prospective validation action-chart identities repeat or are absent")
    action_identity = ObjectIdentity.from_record(action_design.design_id, action_design)
    power_identity = ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze)
    selected_identity = ObjectIdentity.from_record(
        selected_design_candidate.candidate_id,
        selected_design_candidate,
    )
    bridge_identity = ObjectIdentity.from_record(bridge.freeze_id, bridge)
    evaluation_config_identity = ObjectIdentity.from_record(
        parent_evaluation_config.config_id,
        parent_evaluation_config,
    )
    preparation_identities = tuple(
        ObjectIdentity.from_record(value.preparation_id, value) for value in ordered_preparations
    )

    def make_config(
        *,
        config_id: str,
        operation: FreeGsnkeProspectiveValidationProtocolOperation,
        capability_key: str,
        unit_id: str | None,
    ) -> FreeGsnkeProspectiveValidationProtocolConfig:
        return FreeGsnkeProspectiveValidationProtocolConfig(
            config_id=config_id,
            operation=operation,
            capability_key=capability_key,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            unit_id=unit_id,
            roster_id=f"roster.{namespace}",
            reduction_config_id=f"reduction-config.{namespace}",
            validation_id=f"validation.{namespace}",
            source_binding=source_identity,
            numerical_view=view_identity,
            action_charts=chart_identities,
            action_design=action_identity,
            power_freeze=power_identity,
            selected_design_candidate=selected_identity,
            structural_recurrence_bridge=bridge_identity,
            parent_evaluation_config=evaluation_config_identity,
            preparation_identities=preparation_identities,
            prospective_validation_unit_ids=power_freeze.prospective_validation_unit_ids,
            prospective_validation_issue_authority=prospective_validation_issue_authority,
            prospective_validation_execution_authority=prospective_validation_execution_authority,
            prospective_validation_reveal_authority=prospective_validation_reveal_authority,
            maximum_input_bytes=512 * 1024**2,
        )

    values = [
        make_config(
            config_id=f"config.{namespace}.issue",
            operation=FreeGsnkeProspectiveValidationProtocolOperation.ISSUE,
            capability_key=FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY,
            unit_id=None,
        )
    ]
    values.extend(
        make_config(
            config_id=f"config.{namespace}.execute.{unit_id}",
            operation=FreeGsnkeProspectiveValidationProtocolOperation.EXECUTE_UNIT,
            capability_key=FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY,
            unit_id=unit_id,
        )
        for unit_id in power_freeze.prospective_validation_unit_ids
    )
    values.append(
        make_config(
            config_id=f"config.{namespace}.evaluate",
            operation=FreeGsnkeProspectiveValidationProtocolOperation.EVALUATE,
            capability_key=FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY,
            unit_id=None,
        )
    )
    return tuple(sorted(values, key=lambda value: value.config_id))


def _budget(*, seconds: int, memory_gib: int, output_gib: int = 1) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=memory_gib * 1024**3,
        gpu_devices=0,
        wall_time_seconds=seconds,
        source_scan_bytes=512 * 1024**2,
        output_bytes=output_gib * 1024**3,
    )


_READ_WRITE = tuple(
    sorted(
        (
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        key=lambda value: value.value,
    )
)
_READ_VISIBLE = tuple(sorted((*_READ_WRITE, CapabilityPermission.READ_OUTCOME_VISIBLE)))
_EVALUATE = tuple(
    sorted(
        (
            *_READ_WRITE,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        )
    )
)


def freegsnke_prospective_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    """Return the closed static registry for one conditional prospective validation child."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    schema_sha256 = sha256(FreeGsnkeProspectiveValidationProtocolConfig.SCHEMA.encode("ascii")).hexdigest()
    specs = (
        (
            FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY,
            CapabilityKind.TRANSFORM,
            (
                FreeGsnkeProspectiveValidationProtocolConfig.SCHEMA,
                FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA,
                FreeGsnkeSourceBinding.SCHEMA,
                FreeGsnkeNumericalView.SCHEMA,
                FreeGsnkeActionChart.SCHEMA,
                FreeGsnkeSavedPreparation.SCHEMA,
                FreeGsnkeActionDesign.SCHEMA,
                FreeGsnkePowerFreeze.SCHEMA,
                FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA,
                FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA,
                FreeGsnkeTargetReductionConfig.SCHEMA,
                FreeGsnkePreparation.SCHEMA,
            ),
            (FreeGsnkeProspectiveValidationRequestRoster.SCHEMA, FreeGsnkeProspectiveValidationReductionConfig.SCHEMA),
            _READ_VISIBLE,
            OutcomeAccess.EVALUATION_REVEALED,
            _budget(seconds=5 * 60, memory_gib=2),
        ),
        (
            FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY,
            CapabilityKind.SIMULATOR,
            (
                FreeGsnkeProspectiveValidationProtocolConfig.SCHEMA,
                FreeGsnkeProspectiveValidationRequestRoster.SCHEMA,
                FreeGsnkeSourceBinding.SCHEMA,
                FreeGsnkeSavedPreparation.SCHEMA,
                FreeGsnkeNumericalView.SCHEMA,
                FreeGsnkeActionChart.SCHEMA,
                FreeGsnkeActionDesign.SCHEMA,
                FreeGsnkePreparation.SCHEMA,
            ),
            (FreeGsnkeProspectiveValidationUnitResponseBatch.SCHEMA,),
            _READ_VISIBLE,
            OutcomeAccess.EVALUATION_REVEALED,
            _budget(seconds=12 * 60 * 60, memory_gib=4, output_gib=8),
        ),
        (
            FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY,
            CapabilityKind.EVALUATOR,
            (
                FreeGsnkeProspectiveValidationProtocolConfig.SCHEMA,
                FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA,
                FreeGsnkeProspectiveValidationRequestRoster.SCHEMA,
                FreeGsnkeProspectiveValidationReductionConfig.SCHEMA,
                FreeGsnkeProspectiveValidationUnitResponseBatch.SCHEMA,
                FreeGsnkeActionDesign.SCHEMA,
                FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA,
                FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA,
            ),
            (FreeGsnkeProspectiveValidationReduction.SCHEMA, FreeGsnkeTargetValidation.SCHEMA),
            _EVALUATE,
            OutcomeAccess.EVALUATOR_REVEAL,
            _budget(seconds=30 * 60, memory_gib=8, output_gib=2),
        ),
    )
    manifests = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            kind=kind,
            config_schema=FreeGsnkeProspectiveValidationProtocolConfig.SCHEMA,
            config_schema_sha256=schema_sha256,
            input_schema_ids=tuple(sorted(inputs)),
            output_schema_ids=tuple(sorted(outputs)),
            permissions=permissions,
            maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
            maximum_outcome_access=access,
            resource_ceiling=budget,
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="independent-substrate-grounding-freegsnke-conditional-prospective-validation",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "conditional-parent-exact",
                "fresh-complete-preparation-units",
                "mandatory-hold-preserved",
                'prospective-validation-outcomes-remain-sealed',
                "separate-issue-execution-reveal-authorities",
            ),
            implementation_sha256=implementation_sha256,
        )
        for key, kind, inputs, outputs, permissions, access, budget in specs
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-conditional-prospective-validation",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def freegsnke_prospective_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_PROSPECTIVE_VALIDATION_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_prospective_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def _output(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        filename_suffix=".json",
    )


def _execute_step_id(unit_id: str) -> str:
    return f"execute-prospective-validation.{unit_id}"


def build_freegsnke_prospective_protocol(
    *,
    registry: CapabilityRegistry,
    configs: tuple[FreeGsnkeProspectiveValidationProtocolConfig, ...],
    config_refs: tuple[CapabilityConfigRef, ...],
) -> ProtocolTemplate:
    """Build the fixed issue/fresh-unit/evaluator prospective validation child topology."""

    config_by_id = {value.config_id: value for value in configs}
    ref_by_id = {value.config_id: value for value in config_refs}
    if (
        len(config_by_id) != len(configs)
        or len(ref_by_id) != len(config_refs)
        or set(config_by_id) != set(ref_by_id)
    ):
        raise ValueError("FreeGSNKE prospective validation config/reference roster differs")
    operations = {
        operation: tuple(value for value in configs if value.operation is operation)
        for operation in FreeGsnkeProspectiveValidationProtocolOperation
    }
    if (
        len(operations[FreeGsnkeProspectiveValidationProtocolOperation.ISSUE]) != 1
        or len(operations[FreeGsnkeProspectiveValidationProtocolOperation.EVALUATE]) != 1
        or not operations[FreeGsnkeProspectiveValidationProtocolOperation.EXECUTE_UNIT]
    ):
        raise ValueError("FreeGSNKE prospective validation operation roster differs")
    issue = operations[FreeGsnkeProspectiveValidationProtocolOperation.ISSUE][0]
    evaluate = operations[FreeGsnkeProspectiveValidationProtocolOperation.EVALUATE][0]
    executions = operations[FreeGsnkeProspectiveValidationProtocolOperation.EXECUTE_UNIT]
    common_fingerprint = (
        issue.source_binding,
        issue.numerical_view,
        issue.action_charts,
        issue.action_design,
        issue.power_freeze,
        issue.selected_design_candidate,
        issue.structural_recurrence_bridge,
        issue.parent_evaluation_config,
        issue.preparation_identities,
        issue.prospective_validation_unit_ids,
        issue.prospective_validation_issue_authority,
        issue.prospective_validation_execution_authority,
        issue.prospective_validation_reveal_authority,
    )
    if any(
        (
            value.source_binding,
            value.numerical_view,
            value.action_charts,
            value.action_design,
            value.power_freeze,
            value.selected_design_candidate,
            value.structural_recurrence_bridge,
            value.parent_evaluation_config,
            value.preparation_identities,
            value.prospective_validation_unit_ids,
            value.prospective_validation_issue_authority,
            value.prospective_validation_execution_authority,
            value.prospective_validation_reveal_authority,
        )
        != common_fingerprint
        for value in configs
    ):
        raise ValueError("FreeGSNKE prospective validation configs changed the frozen child operands")
    if tuple(sorted(value.unit_id for value in executions if value.unit_id)) != (issue.prospective_validation_unit_ids):
        raise ValueError("FreeGSNKE prospective validation execution-node roster differs")

    def checked_ref(config: FreeGsnkeProspectiveValidationProtocolConfig) -> CapabilityConfigRef:
        manifest = registry.resolve(
            config.capability_key,
            FREEGSNKE_CAPABILITY_VERSION,
        )
        ref = ref_by_id[config.config_id]
        if (
            ref.config_schema != config.SCHEMA
            or ref.config_schema_sha256 != manifest.config_schema_sha256
            or ref.content_sha256 != config.fingerprint()
        ):
            raise ValueError("FreeGSNKE prospective validation config reference differs")
        return ref

    issue_manifest = registry.resolve(
        FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    issue_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID,
        stage=ScientificStage.FREEZE,
        capability_key=FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        config=checked_ref(issue),
        dependency_step_ids=(),
        outputs=(
            _output('prospective-validation-issued-roster', FreeGsnkeProspectiveValidationRequestRoster.SCHEMA),
            _output('prospective-validation-reduction-config', FreeGsnkeProspectiveValidationReductionConfig.SCHEMA),
        ),
        required_permissions=issue_manifest.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=issue_manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-prospective-validation-issue",),
        barrier=BarrierKind.FREEZE,
        maximum_attempts=1,
        obligation_ids=(
            "freegsnke-prospective-validation-derived-only-from-exact-parent",
            "freegsnke-prospective-validation-fresh-preparation-roster-frozen",
            "freegsnke-prospective-validation-mandatory-hold-frozen",
        ),
    )
    execute_manifest = registry.resolve(
        FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    execution_steps = tuple(
        ProtocolStepTemplate(
            step_id=_execute_step_id(config.unit_id or "missing"),
            stage=ScientificStage.ACQUIRE,
            capability_key=FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            config=checked_ref(config),
            dependency_step_ids=(FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID,),
            outputs=(
                _output(
                    f"prospective-validation-unit-batch.{config.unit_id}",
                    FreeGsnkeProspectiveValidationUnitResponseBatch.SCHEMA,
                ),
            ),
            required_permissions=execute_manifest.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=execute_manifest.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.AUTHORITY,
            maximum_attempts=1,
            obligation_ids=(
                f"freegsnke-prospective-validation-complete-preparation-unit.{config.unit_id}",
                f"freegsnke-prospective-validation-new-outcome-remains-sealed.{config.unit_id}",
                (f"freegsnke-prospective-validation-request-accept-apply-realize-clocks.{config.unit_id}"),
                f"freegsnke-task-local-bounded-scratch.{config.unit_id}",
            ),
        )
        for config in sorted(
            executions,
            key=lambda value: value.unit_id or "",
        )
    )
    evaluate_manifest = registry.resolve(
        FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    evaluate_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID,
        stage=ScientificStage.EVALUATE,
        capability_key=FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        config=checked_ref(evaluate),
        dependency_step_ids=tuple(
            sorted(
                (
                    FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID,
                    *(value.step_id for value in execution_steps),
                )
            )
        ),
        outputs=(
            _output('prospective-validation-reduction', FreeGsnkeProspectiveValidationReduction.SCHEMA),
            _output("target-validation", FreeGsnkeTargetValidation.SCHEMA),
        ),
        required_permissions=evaluate_manifest.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=evaluate_manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-prospective-validation-evaluator",),
        barrier=BarrierKind.REVEAL,
        maximum_attempts=1,
        obligation_ids=(
            "freegsnke-prospective-validation-all-issued-units-retained",
            "freegsnke-prospective-validation-postissue-adverse-evidence-retained",
            "freegsnke-prospective-validation-predictive-match-not-rescued",
            "freegsnke-prospective-validation-separate-reveal-authority",
        ),
    )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-conditional-prospective-validation-protocol",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=tuple(
            sorted(
                (issue_step, *execution_steps, evaluate_step),
                key=lambda value: value.step_id,
            )
        ),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _record_id(record: CanonicalRecord) -> str:
    for attribute in (
        "binding_id",
        "saved_state_id",
        "view_id",
        "chart_id",
        "design_id",
        "freeze_id",
        "candidate_id",
        "config_id",
        "preparation_id",
    ):
        if hasattr(record, attribute):
            value = getattr(record, attribute)
            if isinstance(value, str):
                return value
    raise TypeError(f"unsupported FreeGSNKE prospective validation record identity: {type(record)!r}")


def _static_artifact_id(record: CanonicalRecord) -> str:
    return f"artifact.independent-substrate-grounding.freegsnke.prospective-validation.{_record_id(record)}"


def _static_input_id(record: CanonicalRecord) -> str:
    return f"input.independent-substrate-grounding.freegsnke.prospective-validation.{_record_id(record)}"


def _role(record: CanonicalRecord) -> ScientificInputRole:
    if isinstance(record, FreeGsnkeSourceBinding):
        return ScientificInputRole.PREPARED_MEDIUM
    if isinstance(record, FreeGsnkeSavedPreparation):
        return ScientificInputRole.PREPARED_MEDIUM
    if isinstance(record, FreeGsnkePreparation):
        return ScientificInputRole.DENOMINATOR
    if isinstance(record, (FreeGsnkeActionChart, FreeGsnkeActionDesign)):
        return ScientificInputRole.ACTION
    if isinstance(record, FreeGsnkeNumericalView):
        return ScientificInputRole.RECEIVER
    if isinstance(record, (FreeGsnkePowerFreeze, FreeGsnkeTargetReductionConfig)):
        return ScientificInputRole.QUALIFICATION
    return ScientificInputRole.MODEL


def _external(record: CanonicalRecord) -> CandidateGraphExternalInput:
    return CandidateGraphExternalInput(
        input_id=_static_input_id(record),
        scientific_role=_role(record),
        logical_artifact_id=_static_artifact_id(record),
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=record.fingerprint(),
        payload_schema=record.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=256 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def freegsnke_prospective_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    static_records: tuple[CanonicalRecord, ...],
) -> CandidateScientificGraph:
    """Bind all frozen operands plus one later exact admission evaluation substitution slot."""

    by_identity = {_record_id(value): value for value in static_records}
    if len(by_identity) != len(static_records):
        raise ValueError("FreeGSNKE prospective validation static record identities repeat")
    required_types = (
        FreeGsnkeSourceBinding,
        FreeGsnkeNumericalView,
        FreeGsnkeActionDesign,
        FreeGsnkePowerFreeze,
        FreeGsnkeStructuralRecurrenceDesignCandidate,
        FreeGsnkeStructuralRecurrenceBridgeFreeze,
        FreeGsnkeTargetReductionConfig,
    )
    if any(
        sum(isinstance(value, kind) for value in static_records) != 1 for kind in required_types
    ):
        raise ValueError("FreeGSNKE prospective validation static singleton operand roster differs")
    action_charts = tuple(
        sorted(
            (value for value in static_records if isinstance(value, FreeGsnkeActionChart)),
            key=lambda value: value.chart_id,
        )
    )
    if not action_charts:
        raise ValueError("FreeGSNKE prospective validation graph lacks action charts")
    preparations = tuple(
        value for value in static_records if isinstance(value, FreeGsnkePreparation)
    )
    saved_preparations = tuple(
        value for value in static_records if isinstance(value, FreeGsnkeSavedPreparation)
    )
    saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
    expected_saved_ids = {
        value.preparation_id
        for value in preparations
        if value.saved_state is not None
        and value.saved_state.payload_schema == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
    }
    if (
        not preparations
        or len(saved_by_preparation) != len(saved_preparations)
        or set(saved_by_preparation) != expected_saved_ids
        or any(
            value.action_chart
            not in {ObjectIdentity.from_record(chart.chart_id, chart) for chart in action_charts}
            for value in saved_preparations
        )
    ):
        raise ValueError("FreeGSNKE prospective validation graph lacks preparations")
    parent = CandidateGraphExternalInput(
        input_id=FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID,
        scientific_role=ScientificInputRole.PARENT_RECEIPT,
        logical_artifact_id=FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION,
        expected_content_sha256=None,
        payload_schema=FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=256 * 1024**2,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    externals = tuple(_external(value) for value in static_records)
    external_by_input_id = {value.input_id: value for value in externals}
    external_by_schema = {
        value.payload_schema: tuple(
            item for item in externals if item.payload_schema == value.payload_schema
        )
        for value in externals
    }
    steps = {value.step_id: value for value in protocol.steps}
    issue = steps[FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID]
    evaluator = steps[FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID]
    executions = tuple(
        value for value in protocol.steps if value.capability_key == FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY
    )
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=registry.resolve(
                step.capability_key,
                step.capability_version,
            ).implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    edges: list[CandidateGraphEdge] = []

    def external_edge(
        external: CandidateGraphExternalInput,
        consumer: ProtocolStepTemplate,
        suffix: str,
    ) -> None:
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.external.{suffix}.{consumer.step_id}",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=external.input_id,
                consumer_node_id=consumer.step_id,
                consumer_input_id=f"input.{suffix}",
                scientific_role=external.scientific_role,
                logical_artifact_id=external.logical_artifact_id,
                payload_schema=external.payload_schema,
                media_type=external.media_type,
                maximum_size_bytes=external.maximum_size_bytes,
                outcome_access=external.outcome_access,
                visibility_ceiling=external.visibility_ceiling,
                barrier=consumer.barrier,
            )
        )

    external_edge(parent, issue, "parent-admission-evaluation")
    external_edge(parent, evaluator, "parent-admission-evaluation")
    for record in static_records:
        external_edge(
            external_by_input_id[_static_input_id(record)],
            issue,
            _record_id(record),
        )

    singleton_execution_schemas = (
        FreeGsnkeSourceBinding.SCHEMA,
        FreeGsnkeNumericalView.SCHEMA,
        FreeGsnkeActionDesign.SCHEMA,
    )
    preparation_external_by_id = {
        _record_id(record): next(
            value for value in externals if value.input_id == _static_input_id(record)
        )
        for record in preparations
    }
    saved_external_by_id = {
        record.preparation_id: next(
            value for value in externals if value.input_id == _static_input_id(record)
        )
        for record in saved_preparations
    }
    chart_external_by_identity = {
        ObjectIdentity.from_record(record.chart_id, record): next(
            value for value in externals if value.input_id == _static_input_id(record)
        )
        for record in action_charts
    }
    for step in executions:
        unit_id = step.step_id.removeprefix('execute-prospective-validation.')
        if unit_id not in preparation_external_by_id:
            raise ValueError("FreeGSNKE prospective validation execution config/unit graph binding differs")
        for schema in singleton_execution_schemas:
            external_edge(external_by_schema[schema][0], step, schema.rsplit("/", 2)[-2])
        saved_record = saved_by_preparation.get(unit_id)
        if saved_record is not None:
            chart_identity = saved_record.action_chart
        elif len(action_charts) == 1:
            chart_identity = ObjectIdentity.from_record(
                action_charts[0].chart_id,
                action_charts[0],
            )
        else:
            raise ValueError("FreeGSNKE prospective validation historical preparation cannot select among action charts")
        external_edge(
            chart_external_by_identity[chart_identity],
            step,
            f"chart-{unit_id}",
        )
        external_edge(
            preparation_external_by_id[unit_id],
            step,
            unit_id,
        )
        saved_external = saved_external_by_id.get(unit_id)
        if saved_external is not None:
            external_edge(
                saved_external,
                step,
                f"saved-{unit_id}",
            )
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{issue.step_id}.roster.{step.step_id}",
                producer_node_id=issue.step_id,
                producer_output_id='prospective-validation-issued-roster',
                external_input_id=None,
                consumer_node_id=step.step_id,
                consumer_input_id='input.prospective-validation-issued-roster',
                scientific_role=ScientificInputRole.ACTION,
                logical_artifact_id='artifact.freegsnke.prospective-validation-issued-roster',
                payload_schema=FreeGsnkeProspectiveValidationRequestRoster.SCHEMA,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                maximum_size_bytes=issue.resource_budget.output_bytes,
                outcome_access=issue.requested_outcome_access,
                visibility_ceiling=issue.visibility_ceiling,
                barrier=step.barrier,
            )
        )
    for schema in (
        FreeGsnkeActionDesign.SCHEMA,
        FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA,
        FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA,
    ):
        external_edge(external_by_schema[schema][0], evaluator, schema.rsplit("/", 2)[-2])
    for output_id, schema, role in (
        ('prospective-validation-issued-roster', FreeGsnkeProspectiveValidationRequestRoster.SCHEMA, ScientificInputRole.ACTION),
        (
            'prospective-validation-reduction-config',
            FreeGsnkeProspectiveValidationReductionConfig.SCHEMA,
            ScientificInputRole.QUALIFICATION,
        ),
    ):
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{issue.step_id}.{output_id}.{evaluator.step_id}",
                producer_node_id=issue.step_id,
                producer_output_id=output_id,
                external_input_id=None,
                consumer_node_id=evaluator.step_id,
                consumer_input_id=f"input.{output_id}",
                scientific_role=role,
                logical_artifact_id=f"artifact.freegsnke.{output_id}",
                payload_schema=schema,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                maximum_size_bytes=issue.resource_budget.output_bytes,
                outcome_access=issue.requested_outcome_access,
                visibility_ceiling=issue.visibility_ceiling,
                barrier=evaluator.barrier,
            )
        )
    for step in executions:
        output = step.outputs[0]
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{step.step_id}.batch.{evaluator.step_id}",
                producer_node_id=step.step_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=evaluator.step_id,
                consumer_input_id=f"input.{output.output_id}",
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id=f"artifact.{step.step_id}.{output.output_id}",
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=step.resource_budget.output_bytes,
                outcome_access=step.requested_outcome_access,
                visibility_ceiling=step.visibility_ceiling,
                barrier=evaluator.barrier,
            )
        )
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.freegsnke-conditional-prospective-validation",
        external_inputs=tuple(sorted((parent, *externals), key=lambda value: value.input_id)),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def freegsnke_prospective_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
    """Close the conditional child under exact graph/obligation parity."""

    bindings = tuple(
        sorted(
            (
                ObligationCoverageBinding(
                    obligation_id=obligation,
                    proof_owner_node_id=step.step_id,
                    required_output_id=step.outputs[0].output_id,
                    contributor_edge_ids=tuple(
                        sorted(
                            value.edge_id
                            for value in graph.edges
                            if value.consumer_node_id == step.step_id
                        )
                    ),
                )
                for step in protocol.steps
                for obligation in step.obligation_ids
            ),
            key=lambda value: value.obligation_id,
        )
    )
    return StudyTemplate(
        template_key="independent-substrate-grounding.freegsnke.conditional-prospective-validation",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.independent-substrate-grounding.freegsnke.conditional-prospective-validation",
            bindings=bindings,
        ),
    )


def freegsnke_prospective_conditional_request(
    *,
    request_id: str,
    parent_node_id: str,
    power_freeze: FreeGsnkePowerFreeze,
    child_template: StudyTemplate,
) -> ConditionalChildRequest:
    """Predeclare the exact fresh prospective validation child before any parent outcome exists."""

    validate_stable_id(request_id, field_name="request_id")
    validate_stable_id(parent_node_id, field_name="parent_node_id")
    parent_slots = tuple(
        value
        for value in child_template.graph.external_inputs
        if value.input_id == FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID
        and value.scientific_role is ScientificInputRole.PARENT_RECEIPT
        and value.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
        and value.payload_schema == FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA
    )
    if (
        child_template.template_key != "independent-substrate-grounding.freegsnke.conditional-prospective-validation"
        or child_template.template_version != FREEGSNKE_CAPABILITY_VERSION
        or len(parent_slots) != 1
        or not power_freeze.prospective_validation_unit_ids
    ):
        raise ValueError("FreeGSNKE conditional prospective validation template/roster differs")
    return ConditionalChildRequest(
        request_id=request_id,
        parent_node_id=parent_node_id,
        parent_receipt_input_id=FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID,
        gate_kind=ConditionalGateKind.ADMISSION_AND_REACHABILITY_PASSED,
        template_key=child_template.template_key,
        evaluation_unit_ids=power_freeze.prospective_validation_unit_ids,
        eligible_disposition=(ConditionalTerminalDisposition.EXECUTION_ELIGIBLE),
        ineligible_disposition=(ConditionalTerminalDisposition.NOT_ATTEMPTED_PREREQUISITE_NOT_MET),
    )


def _decode_records(
    context: TaskContext,
    record_type: type[CanonicalRecord],
) -> tuple[CanonicalRecord, ...]:
    return tuple(
        decode_canonical_bytes(
            port.read(),
            record_type,
            maximum_bytes=port.size_bytes,
        )
        for port in context.input_ports
        if port.payload_schema == record_type.SCHEMA
    )


def _one_record(
    context: TaskContext,
    record_type: type[CanonicalRecord],
    *,
    label: str,
) -> CanonicalRecord:
    values = _decode_records(context, record_type)
    if len(values) != 1:
        raise ValueError(f"FreeGSNKE prospective validation {label} requires one exact record")
    return values[0]


def _identity(record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(_record_id(record), record)


def _action_charts_by_preparation(
    *,
    preparations: tuple[FreeGsnkePreparation, ...],
    action_charts: tuple[FreeGsnkeActionChart, ...],
    saved_preparations: tuple[FreeGsnkeSavedPreparation, ...],
) -> dict[str, FreeGsnkeActionChart]:
    """Resolve each preparation-local chart from its qualified saved state."""

    charts_by_identity = {_identity(value): value for value in action_charts}
    saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
    values: dict[str, FreeGsnkeActionChart] = {}
    for preparation in preparations:
        saved = saved_by_preparation.get(preparation.preparation_id)
        if saved is None:
            raise ValueError("FreeGSNKE prospective validation requires current saved-preparation custody; original records require a separately verified export before issue or execution")
        if preparation.saved_state != freegsnke_saved_preparation_artifact(saved):
            raise ValueError("FreeGSNKE preparation differs from its exact current saved artifact")
        try:
            values[preparation.preparation_id] = charts_by_identity[saved.action_chart]
        except KeyError as error:
            raise ValueError(
                "FreeGSNKE saved preparation names an absent action chart"
            ) from error
    return values


class _FreeGsnkeProspectiveValidationRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        *,
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        self.manifest = manifest
        self.source_factory = source_factory
        self.executor = executor
        self.execution_count = 0

    def _config(self, context: TaskContext) -> FreeGsnkeProspectiveValidationProtocolConfig:
        value = _one_record(
            context,
            FreeGsnkeProspectiveValidationProtocolConfig,
            label="runner config",
        )
        assert isinstance(value, FreeGsnkeProspectiveValidationProtocolConfig)
        if value.capability_key != self.manifest.capability_key:
            raise ValueError("FreeGSNKE prospective validation runner/config capability differs")
        return value

    @staticmethod
    def _emit(
        context: TaskContext,
        records: tuple[CanonicalRecord, ...],
        checks: tuple[ReceiptCheck, ...],
    ) -> RunnerResult:
        by_schema = {value.SCHEMA: value for value in records}
        if len(by_schema) != len(records) or set(by_schema) != {
            value.payload_schema for value in context.output_ports
        }:
            raise ValueError("FreeGSNKE prospective validation output schema roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=by_schema[port.payload_schema].canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=tuple(sorted(checks, key=lambda value: value.check_id)),
        )

    def _static_operands(
        self,
        context: TaskContext,
        config: FreeGsnkeProspectiveValidationProtocolConfig,
    ) -> tuple[
        FreeGsnkeSourceBinding,
        FreeGsnkeNumericalView,
        tuple[FreeGsnkeActionChart, ...],
        FreeGsnkeActionDesign,
        FreeGsnkePowerFreeze,
        FreeGsnkeStructuralRecurrenceDesignCandidate,
        FreeGsnkeStructuralRecurrenceBridgeFreeze,
        FreeGsnkeTargetReductionConfig,
        tuple[FreeGsnkePreparation, ...],
        tuple[FreeGsnkeSavedPreparation, ...],
    ]:
        record_types: tuple[type[CanonicalRecord], ...] = (
            FreeGsnkeSourceBinding,
            FreeGsnkeNumericalView,
            FreeGsnkeActionDesign,
            FreeGsnkePowerFreeze,
            FreeGsnkeStructuralRecurrenceDesignCandidate,
            FreeGsnkeStructuralRecurrenceBridgeFreeze,
            FreeGsnkeTargetReductionConfig,
        )
        values = tuple(
            _one_record(context, record_type, label=record_type.SCHEMA)
            for record_type in record_types
        )
        (
            source,
            numerical_view,
            action_design,
            power,
            selected,
            bridge,
            evaluation_config,
        ) = values
        assert isinstance(source, FreeGsnkeSourceBinding)
        assert isinstance(numerical_view, FreeGsnkeNumericalView)
        assert isinstance(action_design, FreeGsnkeActionDesign)
        assert isinstance(power, FreeGsnkePowerFreeze)
        assert isinstance(selected, FreeGsnkeStructuralRecurrenceDesignCandidate)
        assert isinstance(bridge, FreeGsnkeStructuralRecurrenceBridgeFreeze)
        assert isinstance(evaluation_config, FreeGsnkeTargetReductionConfig)
        observed = (
            _identity(source),
            _identity(numerical_view),
            _identity(action_design),
            _identity(power),
            _identity(selected),
            _identity(bridge),
            _identity(evaluation_config),
        )
        expected = (
            config.source_binding,
            config.numerical_view,
            config.action_design,
            config.power_freeze,
            config.selected_design_candidate,
            config.structural_recurrence_bridge,
            config.parent_evaluation_config,
        )
        if observed != expected:
            raise ValueError("FreeGSNKE prospective validation static operand bytes differ from config")
        action_charts_raw = _decode_records(context, FreeGsnkeActionChart)
        action_charts = tuple(
            sorted(
                (value for value in action_charts_raw if isinstance(value, FreeGsnkeActionChart)),
                key=lambda value: value.chart_id,
            )
        )
        if tuple(_identity(value) for value in action_charts) != config.action_charts:
            raise ValueError("FreeGSNKE prospective validation action-chart bytes differ from config")
        preparations_raw = _decode_records(context, FreeGsnkePreparation)
        preparations = tuple(
            sorted(
                (value for value in preparations_raw if isinstance(value, FreeGsnkePreparation)),
                key=lambda value: value.preparation_id,
            )
        )
        if tuple(_identity(value) for value in preparations) != (config.preparation_identities):
            raise ValueError("FreeGSNKE prospective validation preparation bytes differ from config")
        saved_raw = _decode_records(context, FreeGsnkeSavedPreparation)
        saved_preparations = tuple(
            sorted(
                (value for value in saved_raw if isinstance(value, FreeGsnkeSavedPreparation)),
                key=lambda value: value.preparation_id,
            )
        )
        _action_charts_by_preparation(
            preparations=preparations,
            action_charts=action_charts,
            saved_preparations=saved_preparations,
        )
        return (
            source,
            numerical_view,
            action_charts,
            action_design,
            power,
            selected,
            bridge,
            evaluation_config,
            preparations,
            saved_preparations,
        )

    def _issue(
        self,
        context: TaskContext,
        config: FreeGsnkeProspectiveValidationProtocolConfig,
    ) -> RunnerResult:
        parent = _one_record(context, FreeGsnkeAdmissionEvaluationEvaluation, label="admission evaluation parent")
        assert isinstance(parent, FreeGsnkeAdmissionEvaluationEvaluation)
        (
            source,
            numerical_view,
            action_charts,
            action_design,
            power,
            selected,
            _bridge,
            evaluation_config,
            preparations,
            saved_preparations,
        ) = self._static_operands(context, config)
        if self.source_factory() != source:
            raise ValueError("FreeGSNKE prospective validation live source differs from frozen source")
        prospective_validation_preparations = tuple(value for value in preparations if value.phase is FreeGsnkePhase.PROSPECTIVE_VALIDATION)
        prior_preparations = tuple(
            value for value in preparations if value.phase is not FreeGsnkePhase.PROSPECTIVE_VALIDATION
        )
        branch_ids = selected_freegsnke_prospective_target_branches(
            parent_admission_evaluation=parent,
            selected_design_candidate=selected,
        )
        requests = build_freegsnke_prospective_requests(
            source_binding=source,
            numerical_view=numerical_view,
            action_charts_by_preparation=_action_charts_by_preparation(
                preparations=prospective_validation_preparations,
                action_charts=action_charts,
                saved_preparations=saved_preparations,
            ),
            action_design=action_design,
            preparations=prospective_validation_preparations,
            selected_target_branch_ids=branch_ids,
        )
        roster = issue_freegsnke_prospective_roster(
            roster_id=config.roster_id,
            parent_admission_evaluation=parent,
            selected_design_candidate=selected,
            action_design=action_design,
            power_freeze=power,
            prior_preparations=prior_preparations,
            prospective_validation_preparations=prospective_validation_preparations,
            action_charts_by_preparation=_action_charts_by_preparation(
                preparations=prospective_validation_preparations,
                action_charts=action_charts,
                saved_preparations=saved_preparations,
            ),
            requests=requests,
            prospective_validation_issue_authority=config.prospective_validation_issue_authority,
        )
        reduction_config = freeze_freegsnke_prospective_reduction_config(
            config_id=config.reduction_config_id,
            roster=roster,
            parent_evaluation_config=evaluation_config,
            prospective_validation_preparations=prospective_validation_preparations,
        )
        return self._emit(
            context,
            (roster, reduction_config),
            (
                ReceiptCheck("freegsnke-prospective-validation-parent-exact", True, ()),
                ReceiptCheck("freegsnke-prospective-validation-fresh-roster-frozen", True, ()),
                ReceiptCheck("freegsnke-prospective-validation-mandatory-hold-frozen", True, ()),
            ),
        )

    def _execute_unit(
        self,
        context: TaskContext,
        config: FreeGsnkeProspectiveValidationProtocolConfig,
    ) -> RunnerResult:
        roster = _one_record(
            context,
            FreeGsnkeProspectiveValidationRequestRoster,
            label="issued roster",
        )
        assert isinstance(roster, FreeGsnkeProspectiveValidationRequestRoster)
        source = _one_record(context, FreeGsnkeSourceBinding, label="source")
        numerical_view = _one_record(
            context,
            FreeGsnkeNumericalView,
            label="numerical view",
        )
        action_chart = _one_record(context, FreeGsnkeActionChart, label="action chart")
        action_design = _one_record(
            context,
            FreeGsnkeActionDesign,
            label="action design",
        )
        preparation = _one_record(
            context,
            FreeGsnkePreparation,
            label="unit preparation",
        )
        saved_values = _decode_records(context, FreeGsnkeSavedPreparation)
        assert isinstance(source, FreeGsnkeSourceBinding)
        assert isinstance(numerical_view, FreeGsnkeNumericalView)
        assert isinstance(action_chart, FreeGsnkeActionChart)
        assert isinstance(action_design, FreeGsnkeActionDesign)
        assert isinstance(preparation, FreeGsnkePreparation)
        target_saved = (
            preparation.saved_state is not None
            and preparation.saved_state.payload_schema == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
        )
        if len(saved_values) != int(target_saved):
            raise ValueError("FreeGSNKE prospective validation saved-preparation input differs")
        saved_preparation = saved_values[0] if saved_values else None
        if saved_preparation is not None:
            assert isinstance(saved_preparation, FreeGsnkeSavedPreparation)
        if (
            config.unit_id != preparation.preparation_id
            or _identity(source) != config.source_binding
            or _identity(numerical_view) != config.numerical_view
            or _identity(action_chart) not in config.action_charts
            or _identity(action_design) != config.action_design
            or _identity(preparation) not in config.preparation_identities
            or self.source_factory() != source
        ):
            raise ValueError("FreeGSNKE prospective validation execution operand differs from config")
        requests = build_freegsnke_prospective_requests(
            source_binding=source,
            numerical_view=numerical_view,
            action_charts_by_preparation={preparation.preparation_id: action_chart},
            action_design=action_design,
            preparations=(preparation,),
            selected_target_branch_ids=roster.selected_target_branch_ids,
        )
        roster_requests = set(roster.requests)
        if any(
            ObjectIdentity.from_record(value.request_id, value) not in roster_requests
            for value in requests
        ):
            raise ValueError("FreeGSNKE prospective validation unit request differs from issued roster")
        if saved_preparation is not None:
            for request in requests:
                FreeGsnkeTargetWorkerInput(
                    input_id=f"worker-input.{request.request_id}",
                    request=request,
                    saved_preparation=saved_preparation,
                )
        responses = tuple(
            decode_freegsnke_target_response(
                request=request,
                payload=self.executor.execute(
                    request,
                    saved_preparation,
                ).canonical_bytes(),
            )
            for request in requests
        )
        batch = FreeGsnkeProspectiveValidationUnitResponseBatch(
            batch_id=f"batch.{roster.roster_id}.{preparation.preparation_id}",
            prospective_validation_roster=ObjectIdentity.from_record(roster.roster_id, roster),
            unit_id=preparation.preparation_id,
            preparation=ObjectIdentity.from_record(
                preparation.preparation_id,
                preparation,
            ),
            requests=requests,
            responses=responses,
            prospective_validation_execution_authority=config.prospective_validation_execution_authority,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        return self._emit(
            context,
            (batch,),
            (
                ReceiptCheck("freegsnke-prospective-validation-unit-complete", True, ()),
                ReceiptCheck("freegsnke-prospective-validation-new-outcome-sealed", True, ()),
                ReceiptCheck("freegsnke-prospective-validation-action-clock-ledger-retained", True, ()),
            ),
        )

    def _evaluate(
        self,
        context: TaskContext,
        config: FreeGsnkeProspectiveValidationProtocolConfig,
    ) -> RunnerResult:
        parent = _one_record(context, FreeGsnkeAdmissionEvaluationEvaluation, label="admission evaluation parent")
        roster = _one_record(
            context,
            FreeGsnkeProspectiveValidationRequestRoster,
            label="issued roster",
        )
        reduction_config = _one_record(
            context,
            FreeGsnkeProspectiveValidationReductionConfig,
            label="reduction config",
        )
        action_design = _one_record(
            context,
            FreeGsnkeActionDesign,
            label="action design",
        )
        bridge = _one_record(
            context,
            FreeGsnkeStructuralRecurrenceBridgeFreeze,
            label='structural recurrence bridge',
        )
        selected = _one_record(
            context,
            FreeGsnkeStructuralRecurrenceDesignCandidate,
            label="selected design",
        )
        assert isinstance(parent, FreeGsnkeAdmissionEvaluationEvaluation)
        assert isinstance(roster, FreeGsnkeProspectiveValidationRequestRoster)
        assert isinstance(reduction_config, FreeGsnkeProspectiveValidationReductionConfig)
        assert isinstance(action_design, FreeGsnkeActionDesign)
        assert isinstance(bridge, FreeGsnkeStructuralRecurrenceBridgeFreeze)
        assert isinstance(selected, FreeGsnkeStructuralRecurrenceDesignCandidate)
        if (
            _identity(action_design) != config.action_design
            or _identity(bridge) != config.structural_recurrence_bridge
            or _identity(selected) != config.selected_design_candidate
            or reduction_config.config_id != config.reduction_config_id
            or roster.roster_id != config.roster_id
        ):
            raise ValueError("FreeGSNKE prospective validation evaluator operand differs from config")
        batches_raw = _decode_records(context, FreeGsnkeProspectiveValidationUnitResponseBatch)
        batches = tuple(
            sorted(
                (
                    value
                    for value in batches_raw
                    if isinstance(value, FreeGsnkeProspectiveValidationUnitResponseBatch)
                ),
                key=lambda value: value.unit_id,
            )
        )
        if tuple(value.unit_id for value in batches) != roster.prospective_validation_unit_ids or any(
            value.prospective_validation_roster != ObjectIdentity.from_record(roster.roster_id, roster)
            or value.prospective_validation_execution_authority != config.prospective_validation_execution_authority
            for value in batches
        ):
            raise ValueError("FreeGSNKE prospective validation evaluator batch roster differs")
        requests = tuple(
            sorted(
                (request for batch in batches for request in batch.requests),
                key=lambda value: value.request_id,
            )
        )
        responses = tuple(
            sorted(
                (response for batch in batches for response in batch.responses),
                key=lambda value: value.response_id,
            )
        )
        reduction = reduce_freegsnke_prospective_panel(
            config=reduction_config,
            roster=roster,
            action_design=action_design,
            requests=requests,
            responses=responses,
        )
        validation = evaluate_freegsnke_prospective(
            validation_id=config.validation_id,
            parent_admission_evaluation=parent,
            bridge=bridge,
            selected_design_candidate=selected,
            action_design=action_design,
            roster=roster,
            reduction_config=reduction_config,
            reduction=reduction,
            prospective_validation_execution_authority=config.prospective_validation_execution_authority,
            prospective_validation_reveal_authority=config.prospective_validation_reveal_authority,
        )
        return self._emit(
            context,
            (reduction, validation),
            (
                ReceiptCheck("freegsnke-prospective-validation-all-issued-units-retained", True, ()),
                ReceiptCheck("freegsnke-prospective-validation-adverse-outcomes-retained", True, ()),
                ReceiptCheck("freegsnke-prospective-validation-predictor-not-rescued", True, ()),
            ),
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        config = self._config(context)
        if config.operation is FreeGsnkeProspectiveValidationProtocolOperation.ISSUE:
            return self._issue(context, config)
        if config.operation is FreeGsnkeProspectiveValidationProtocolOperation.EXECUTE_UNIT:
            return self._execute_unit(context, config)
        return self._evaluate(context, config)


def freegsnke_prospective_runners(
    *,
    registry: CapabilityRegistry,
    source_factory: Callable[[], FreeGsnkeSourceBinding],
    executor: FreeGsnkeEpisodeExecutor,
) -> tuple[TaskRunner, ...]:
    return tuple(
        _FreeGsnkeProspectiveValidationRunner(
            manifest,
            source_factory=source_factory,
            executor=executor,
        )
        for manifest in registry.capabilities
    )


class FreeGsnkeProspectiveValidationRuntimeProvider(CampaignRuntimeProvider):
    """Path-free provider for one instantiated conditional prospective validation child."""

    issued_source_schema_ids = (FrozenParentInputBinding.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        configs: tuple[FreeGsnkeProspectiveValidationProtocolConfig, ...],
        static_records: tuple[CanonicalRecord, ...],
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        expected = freegsnke_prospective_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("FreeGSNKE prospective validation provider registry differs")
        if {value.operation for value in configs} != set(FreeGsnkeProspectiveValidationProtocolOperation):
            raise ValueError("FreeGSNKE prospective validation provider config operations differ")
        if len({value.config_id for value in configs}) != len(configs):
            raise ValueError("FreeGSNKE prospective validation provider config identities repeat")
        if len({_record_id(value) for value in static_records}) != len(static_records):
            raise ValueError("FreeGSNKE prospective validation provider static identities repeat")
        preparations = tuple(
            value for value in static_records if isinstance(value, FreeGsnkePreparation)
        )
        action_charts = tuple(
            sorted(
                (value for value in static_records if isinstance(value, FreeGsnkeActionChart)),
                key=lambda value: value.chart_id,
            )
        )
        saved_preparations = tuple(
            value for value in static_records if isinstance(value, FreeGsnkeSavedPreparation)
        )
        saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
        expected_saved_ids = {
            value.preparation_id
            for value in preparations
            if value.saved_state is not None
            and value.saved_state.payload_schema == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
        }
        if (
            not preparations
            or not action_charts
            or len(saved_by_preparation) != len(saved_preparations)
            or set(saved_by_preparation) != expected_saved_ids
            or any(
                value.action_charts != tuple(_identity(chart) for chart in action_charts)
                for value in configs
            )
        ):
            raise ValueError("FreeGSNKE prospective validation provider saved-preparation roster differs")
        _action_charts_by_preparation(
            preparations=preparations,
            action_charts=action_charts,
            saved_preparations=saved_preparations,
        )
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.configs = tuple(sorted(configs, key=lambda value: value.config_id))
        self.static_records = tuple(sorted(static_records, key=lambda value: _record_id(value)))
        self._runners = freegsnke_prospective_runners(
            registry=registry,
            source_factory=source_factory,
            executor=executor,
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        if registry != self.registry:
            raise ValueError("FreeGSNKE prospective validation provider registry identity differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError("FreeGSNKE prospective validation provider plan registry differs")
        bindings = tuple(
            value for value in source_records if isinstance(value, FrozenParentInputBinding)
        )
        if len(bindings) != 1 or (bindings[0].external_input_id != FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID):
            raise ValueError("FreeGSNKE prospective validation execution lacks one frozen admission evaluation binding")
        parent_admission_evaluation = bindings[0].decode_parent(
            FreeGsnkeAdmissionEvaluationEvaluation,
            maximum_bytes=256 * 1024**2,
        )
        config_refs = {
            task.capability.config.artifact_id: task.capability.config for task in plan.tasks
        }
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        config_by_hash = {value.fingerprint(): value for value in self.configs}
        static_by_artifact = {_static_artifact_id(value): value for value in self.static_records}
        values = []
        for artifact_id, spec in sorted(specs.items()):
            config_ref = config_refs.get(artifact_id)
            if config_ref is not None:
                try:
                    record: CanonicalRecord = config_by_hash[config_ref.content_sha256]
                except KeyError as error:
                    raise ValueError("FreeGSNKE prospective validation plan references an unknown config") from error
                access = OutcomeAccess.OUTCOME_BLIND
                visibility = VisibilityCeiling.PROSPECTIVE
                record_id = _record_id(record)
            elif artifact_id == FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_ARTIFACT_ID:
                record = parent_admission_evaluation
                access = OutcomeAccess.EVALUATION_REVEALED
                visibility = VisibilityCeiling.OUTCOME_VISIBLE
                record_id = parent_admission_evaluation.evaluation_id
            else:
                try:
                    record = static_by_artifact[artifact_id]
                except KeyError as error:
                    raise ValueError(
                        f"FreeGSNKE prospective validation plan references an unknown input: {artifact_id}"
                    ) from error
                access = OutcomeAccess.OUTCOME_BLIND
                visibility = VisibilityCeiling.PROSPECTIVE
                record_id = _record_id(record)
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(spec.expected_visibility_ceiling or visibility),
                    outcome_access=(spec.expected_outcome_access or access),
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=spec.expected_content_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("FreeGSNKE prospective validation semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types: dict[str, type[CanonicalRecord]] = {
            FreeGsnkeProspectiveValidationRequestRoster.SCHEMA: FreeGsnkeProspectiveValidationRequestRoster,
            FreeGsnkeProspectiveValidationReductionConfig.SCHEMA: FreeGsnkeProspectiveValidationReductionConfig,
            FreeGsnkeProspectiveValidationUnitResponseBatch.SCHEMA: FreeGsnkeProspectiveValidationUnitResponseBatch,
            FreeGsnkeProspectiveValidationReduction.SCHEMA: FreeGsnkeProspectiveValidationReduction,
            FreeGsnkeTargetValidation.SCHEMA: FreeGsnkeTargetValidation,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                key = (manifest.capability_key, schema, ArtifactProfile.CANONICAL_JSON)
                if planned is not None and key not in planned:
                    continue
                record_type = record_types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(
                                value.name
                                for value in fields(record_type)  # type: ignore[arg-type]
                            )
                        ),
                    )
                )
        return tuple(sorted(values, key=lambda value: (value.capability_key, value.payload_schema)))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("FreeGSNKE prospective validation adjudication registry differs")
        del execution_plan
        return None


def instantiate_freegsnke_prospective(
    *,
    parent_compilation: CandidateCompilationReport | StudyCompilationReport,
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation,
) -> tuple[
    ConditionalChildInstantiation,
    CandidateCompilationReport | StudyCompilationReport | None,
]:
    """Instantiate only the exact frozen child or a typed nonissue."""

    base = (
        parent_compilation.base_report
        if isinstance(parent_compilation, StudyCompilationReport)
        else parent_compilation
    )
    if base.candidate is None or base.candidate.conditional_successor is None:
        raise ValueError("FreeGSNKE admission evaluation parent lacks a frozen conditional successor")
    request = base.candidate.conditional_successor.request
    if (
        request.parent_receipt_input_id != FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID
        or request.evaluation_unit_ids != parent_admission_evaluation.target_power_freeze.prospective_validation_unit_ids
    ):
        raise ValueError("FreeGSNKE prospective validation parent/frozen child roster differs")
    eligible = freegsnke_prospective_parent_eligible(parent_admission_evaluation)
    reasons: tuple[str, ...] = ()
    if not eligible:
        reasons = (
            ("CONTROLLER_USE_PREREQUISITE_NONATTEMPT",)
            if parent_admission_evaluation.observed_policy_branch is PolicyBranch.NONATTEMPT
            else tuple(
                sorted(
                    {
                        *parent_admission_evaluation.reason_codes,
                        "CONTROLLER_USE_PARENT_INELIGIBLE_OR_UNSAFE",
                    }
                )
            )
        )
    instantiation, child = instantiate_conditional_child(
        parent_compilation=base,
        parent_record_id=parent_admission_evaluation.evaluation_id,
        parent_record=parent_admission_evaluation,
        eligible=eligible,
        reason_codes=reasons,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    if child is None or not isinstance(
        parent_compilation,
        StudyCompilationReport,
    ):
        return instantiation, child
    if parent_compilation.candidate is None or child.candidate is None:
        raise ValueError("FreeGSNKE prospective validation standard child lost its authoring root")
    parent_standard = parent_compilation.candidate
    candidate_seed = {
        "base_candidate": ObjectIdentity.from_record(
            child.candidate.candidate_id,
            child.candidate,
        ),
        "parent_standard_candidate": ObjectIdentity.from_record(
            parent_standard.candidate_id,
            parent_standard,
        ),
        "parent_admission_evaluation": ObjectIdentity.from_record(
            parent_admission_evaluation.evaluation_id,
            parent_admission_evaluation,
        ),
    }
    candidate_digest = sha256(canonical_json_bytes(candidate_seed)).hexdigest()
    standard_child = replace(
        parent_standard,
        candidate_id=f"standard-candidate.conditional.{candidate_digest[:24]}",
        base_candidate=child.candidate,
    )
    report_seed = {
        "base_report": child,
        "candidate": standard_child,
        "parent_report": parent_compilation.report_id,
    }
    report_digest = sha256(canonical_json_bytes(report_seed)).hexdigest()
    child_report = replace(
        parent_compilation,
        report_id=f"standard-report.conditional.{report_digest[:24]}",
        base_report=child,
        candidate=standard_child,
    )
    return instantiation, child_report


@dataclass(frozen=True, slots=True)
class FreeGsnkeConditionalChildResolver:
    """Composition adapter for exact admission evaluation-derived FreeGSNKE prospective validation instantiation."""

    @property
    def parent_record_schemas(self) -> Mapping[str, type[CanonicalRecord]]:
        return MappingProxyType({FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA: FreeGsnkeAdmissionEvaluationEvaluation})

    def resolve(
        self,
        *,
        parent_compilation: StudyCompilationReport,
        parent_record: CanonicalRecord,
    ) -> ConditionalChildResolution:
        if not isinstance(parent_record, FreeGsnkeAdmissionEvaluationEvaluation):
            raise TypeError("FreeGSNKE prospective validation successor requires an exact admission evaluation evaluation")
        base = parent_compilation.base_report.candidate
        if base is None or base.conditional_successor is None:
            raise ValueError("FreeGSNKE parent lacks its frozen prospective validation successor")
        external_input_id = base.conditional_successor.request.parent_receipt_input_id
        instantiation, child = instantiate_freegsnke_prospective(
            parent_compilation=parent_compilation,
            parent_admission_evaluation=parent_record,
        )
        if child is None:
            return ConditionalChildResolution(
                instantiation=instantiation,
                compilation=None,
                parent_input_binding=None,
            )
        if not isinstance(child, StudyCompilationReport):
            raise TypeError("FreeGSNKE prospective validation child lost its standard compilation root")
        if child.candidate is None:
            raise ValueError("FreeGSNKE prospective validation child lacks its exact candidate")
        binding = bind_frozen_parent_input(
            candidate=child.candidate.base_candidate,
            external_input_id=external_input_id,
            parent_record_id=parent_record.evaluation_id,
            parent_record=parent_record,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        )
        return ConditionalChildResolution(
            instantiation=instantiation,
            compilation=child,
            parent_input_binding=binding,
        )


__all__ = [
    "FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_KEY",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_EXECUTE_KEY",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_KEY",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_ARTIFACT_ID",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_PARENT_INPUT_ID",
    "FREEGSNKE_PROSPECTIVE_VALIDATION_PROVIDER_KEY",
    'FreeGsnkeConditionalChildResolver',
    'FreeGsnkeProspectiveValidationProtocolConfig',
    "FreeGsnkeProspectiveValidationProtocolOperation",
    "FreeGsnkeProspectiveValidationRuntimeProvider",
    'FreeGsnkeProspectiveValidationUnitResponseBatch',
    'build_freegsnke_prospective_protocol',
    'freegsnke_prospective_candidate_registrations',
    'freegsnke_prospective_conditional_request',
    'freegsnke_prospective_study_template',
    'freegsnke_prospective_protocol_configs',
    'freegsnke_prospective_registry',
    'freegsnke_prospective_runners',
    'freegsnke_prospective_scientific_graph',
    'instantiate_freegsnke_prospective',
]
