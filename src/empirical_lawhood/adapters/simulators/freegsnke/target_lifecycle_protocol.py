"""Full FreeGSNKE development-freeze-sealed-evaluation development-through-admission evaluation DAG.

Source qualification, target design and power are exact pre-issue inputs.  The
runtime executes development preparations, freezes target-local selection and
the unchanged registered margin structural recurrence prediction, then executes evaluation
preparations sealed.  A single evaluator-owned reveal reduces all issued
evaluation units and emits the target-native admission evaluation record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Final

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentImplementationDossier, IndependentSubstrateForecastAlphabetGrammar, IndependentSubstrateTargetPredictionContract
from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus
from empirical_lawhood.adapters.methods.structural_recurrence_runtime import MARGIN_FORECAST_CAPABILITY_VERSION, MarginStructuralRecurrenceForecastPredictionRequest, MarginStructuralRecurrenceForecastRuntimeConfig, MarginStructuralRecurrenceForecastRuntimeOperation, registered_predict
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastConformance, MarginStructuralRecurrenceForecastMethodFreeze, MarginStructuralRecurrenceForecastPredictionIssue
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
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
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
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

from .contracts import FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA, FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetProcessResponse, FreeGsnkeTargetWorkerInput
from .design import FreeGsnkeActionDesign, FreeGsnkePhaseRequestRoster
from .forecasting import FreeGsnkeForecastEncoderFreeze
from .generation_protocol import (
    FreeGsnkeEpisodeExecutor,
    build_freegsnke_generation_protocol,
    freegsnke_generation_registry,
    freegsnke_generation_runners,
)
from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceBridgeFreeze
from .registry import (
    FREEGSNKE_CAPABILITY_VERSION,
    FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
    FREEGSNKE_EVALUATION_CAPABILITY_KEY,
)
from .target_analysis import FreeGsnkePhaseReduction, FreeGsnkeTargetReductionConfig, reduce_freegsnke_phase_panel
from .target_design import FreeGsnkeForecastAlphabetFreeze, FreeGsnkeMetricTopologyDesign, FreeGsnkePowerFreeze, FreeGsnkeTargetDesignFreeze
from .target_development import FreeGsnkeDevelopmentForecastFreeze, build_freegsnke_structural_recurrence_prediction_request, build_freegsnke_target_prediction_contract, freeze_freegsnke_development_forecasts
from .metric_bootstrap_inputs import FreeGsnkeMetricBootstrapInputs
from .target_evaluation import FreeGsnkeAdmissionEvaluationEvaluation, evaluate_freegsnke_admission
from .target_protocol import FREEGSNKE_DEVELOPMENT_REDUCTION_KEY, freegsnke_phase_reduction_runners, freegsnke_phase_registry


FREEGSNKE_TARGET_FREEZE_KEY: Final = "freegsnke.freeze-target-prediction"
FREEGSNKE_ADMISSION_EVALUATION_KEY: Final = "freegsnke.evaluate-target-admission"
FREEGSNKE_TARGET_PROVIDER_KEY: Final = "independent-substrate-grounding.freegsnke-target-lifecycle-provider"
FREEGSNKE_DEVELOPMENT_REDUCTION_STEP_ID: Final = "reduce-development-panel"
FREEGSNKE_TARGET_FREEZE_STEP_ID: Final = "freeze-target-prediction"
FREEGSNKE_ADMISSION_EVALUATION_STEP_ID: Final = "evaluate-target-admission"


class FreeGsnkeTargetLifecycleOperation(StrEnum):
    FREEZE_PREDICTION = "FREEZE_PREDICTION"
    EVALUATE_ADMISSION = "EVALUATE_ADMISSION"


_OPERATION_KEY = {
    FreeGsnkeTargetLifecycleOperation.FREEZE_PREDICTION: (FREEGSNKE_TARGET_FREEZE_KEY),
    FreeGsnkeTargetLifecycleOperation.EVALUATE_ADMISSION: FREEGSNKE_ADMISSION_EVALUATION_KEY,
}


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetPredictionFreeze(CanonicalRecord):
    'One immutable development selection, contract and margin structural recurrence forecast issue.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-prediction-freeze'

    freeze_id: str
    development_forecasts: FreeGsnkeDevelopmentForecastFreeze
    target_contract: IndependentSubstrateTargetPredictionContract
    prediction_request: MarginStructuralRecurrenceForecastPredictionRequest
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue
    structural_recurrence_runtime_config: ObjectIdentity
    frozen_before_evaluator_access: bool
    evaluation_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if self.structural_recurrence_runtime_config.object_schema != MarginStructuralRecurrenceForecastRuntimeConfig.SCHEMA:
            raise ValueError('FreeGSNKE target freeze structural recurrence runtime config differs')
        selection = self.development_forecasts.selection
        if (
            self.prediction_issue.method_freeze
            != ObjectIdentity.from_record(
                self.prediction_request.method_freeze.freeze_id,
                self.prediction_request.method_freeze,
            )
            or self.prediction_issue.conformance
            != ObjectIdentity.from_record(
                self.prediction_request.conformance.conformance_id,
                self.prediction_request.conformance,
            )
            or self.target_contract.structural_recurrence_target_design
            != ObjectIdentity.from_record(
                selection.selected_design_candidate.design.design_id,
                selection.selected_design_candidate.design,
            )
            or self.target_contract.structural_recurrence_development_evidence
            != ObjectIdentity.from_record(
                selection.structural_recurrence_development_evidence.evidence_id,
                selection.structural_recurrence_development_evidence,
            )
            or self.prediction_request.design != selection.selected_design_candidate.design
            or self.prediction_request.development != selection.structural_recurrence_development_evidence
            or self.prediction_issue.issue_id != self.prediction_request.issue_id
            or self.prediction_issue.structural_prediction.target_design
            != self.target_contract.structural_recurrence_target_design
            or self.prediction_issue.structural_prediction.development_evidence
            != self.target_contract.structural_recurrence_development_evidence
            or not self.prediction_issue.published_before_evaluation
            or self.prediction_issue.protected_outcome_access_count
        ):
            raise ValueError("FreeGSNKE target prediction freeze operands differ")
        if (
            not self.frozen_before_evaluator_access
            or self.evaluation_outcome_access_count
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("FreeGSNKE target prediction crossed evaluation reveal")


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetLifecycleConfig(CanonicalRecord):
    """Exact IDs, operands and authority split for the two target-owned nodes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-lifecycle-config'

    config_id: str
    operation: FreeGsnkeTargetLifecycleOperation
    capability_key: str
    capability_version: str
    target_prediction_freeze_id: str
    development_forecast_freeze_id: str
    development_selection_id: str
    target_contract_id: str
    prediction_request_id: str
    prediction_issue_id: str
    admission_evaluation_evaluation_id: str
    method_freeze: ObjectIdentity
    conformance: ObjectIdentity
    target_design: ObjectIdentity
    structural_recurrence_bridge: ObjectIdentity
    action_design: ObjectIdentity
    metric_topology_design: ObjectIdentity
    forecast_encoder: ObjectIdentity
    forecast_grammar: ObjectIdentity
    forecast_alphabet: ObjectIdentity
    power_freeze: ObjectIdentity
    independence_dossier: ObjectIdentity
    structural_recurrence_runtime_config: ObjectIdentity
    development_reduction_config: ObjectIdentity
    evaluation_reduction_config: ObjectIdentity
    admission_evaluation_execution_authority: ObjectIdentity
    admission_evaluation_reveal_authority: ObjectIdentity
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "capability_key",
            "target_prediction_freeze_id",
            "development_forecast_freeze_id",
            "development_selection_id",
            "target_contract_id",
            "prediction_request_id",
            "prediction_issue_id",
            "admission_evaluation_evaluation_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        if (
            self.capability_key != _OPERATION_KEY[self.operation]
            or self.capability_version != FREEGSNKE_CAPABILITY_VERSION
        ):
            raise ValueError("FreeGSNKE target lifecycle operation/capability differs")
        expected_schemas = (
            (self.method_freeze, MarginStructuralRecurrenceForecastMethodFreeze.SCHEMA),
            (self.conformance, MarginStructuralRecurrenceForecastConformance.SCHEMA),
            (self.target_design, FreeGsnkeTargetDesignFreeze.SCHEMA),
            (self.structural_recurrence_bridge, FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA),
            (self.action_design, FreeGsnkeActionDesign.SCHEMA),
            (
                self.metric_topology_design,
                FreeGsnkeMetricTopologyDesign.SCHEMA,
            ),
            (self.forecast_encoder, FreeGsnkeForecastEncoderFreeze.SCHEMA),
            (self.forecast_grammar, IndependentSubstrateForecastAlphabetGrammar.SCHEMA),
            (
                self.forecast_alphabet,
                FreeGsnkeForecastAlphabetFreeze.SCHEMA,
            ),
            (self.power_freeze, FreeGsnkePowerFreeze.SCHEMA),
            (
                self.independence_dossier,
                IndependentImplementationDossier.SCHEMA,
            ),
            (self.structural_recurrence_runtime_config, MarginStructuralRecurrenceForecastRuntimeConfig.SCHEMA),
            (
                self.development_reduction_config,
                FreeGsnkeTargetReductionConfig.SCHEMA,
            ),
            (
                self.evaluation_reduction_config,
                FreeGsnkeTargetReductionConfig.SCHEMA,
            ),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("FreeGSNKE target lifecycle operand schema differs")
        if self.admission_evaluation_execution_authority == self.admission_evaluation_reveal_authority:
            raise ValueError("FreeGSNKE admission evaluation execution/reveal authorities repeat")
        if not 0 < self.maximum_input_bytes <= 512 * 1024**2:
            raise ValueError("FreeGSNKE target lifecycle input bound differs")


def freegsnke_target_lifecycle_configs(
    *,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    conformance: MarginStructuralRecurrenceForecastConformance,
    target_design: FreeGsnkeTargetDesignFreeze,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    action_design: FreeGsnkeActionDesign,
    metric_topology_design: FreeGsnkeMetricTopologyDesign,
    encoder: FreeGsnkeForecastEncoderFreeze,
    forecast_grammar: IndependentSubstrateForecastAlphabetGrammar,
    forecast_alphabet: FreeGsnkeForecastAlphabetFreeze,
    power_freeze: FreeGsnkePowerFreeze,
    independence_dossier: IndependentImplementationDossier,
    structural_recurrence_runtime_config: MarginStructuralRecurrenceForecastRuntimeConfig,
    development_reduction_config: FreeGsnkeTargetReductionConfig,
    evaluation_reduction_config: FreeGsnkeTargetReductionConfig,
    admission_evaluation_execution_authority: ObjectIdentity,
    admission_evaluation_reveal_authority: ObjectIdentity,
    namespace: str,
) -> tuple[FreeGsnkeTargetLifecycleConfig, ...]:
    """Freeze the two target-owned configs with one common operand closure."""

    validate_stable_id(namespace, field_name="namespace")
    method_identity = ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze)
    conformance_identity = ObjectIdentity.from_record(
        conformance.conformance_id,
        conformance,
    )
    target_design_identity = ObjectIdentity.from_record(
        target_design.freeze_id,
        target_design,
    )
    bridge_identity = ObjectIdentity.from_record(bridge.freeze_id, bridge)
    action_design_identity = ObjectIdentity.from_record(
        action_design.design_id,
        action_design,
    )
    metric_topology_identity = ObjectIdentity.from_record(
        metric_topology_design.design_id,
        metric_topology_design,
    )
    encoder_identity = ObjectIdentity.from_record(encoder.encoder_id, encoder)
    grammar_identity = ObjectIdentity.from_record(
        forecast_grammar.grammar_id,
        forecast_grammar,
    )
    alphabet_identity = ObjectIdentity.from_record(
        forecast_alphabet.freeze_id,
        forecast_alphabet,
    )
    power_identity = ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze)
    dossier_identity = ObjectIdentity.from_record(
        independence_dossier.dossier_id,
        independence_dossier,
    )
    runtime_identity = ObjectIdentity.from_record(
        structural_recurrence_runtime_config.config_id,
        structural_recurrence_runtime_config,
    )
    development_config_identity = ObjectIdentity.from_record(
        development_reduction_config.config_id,
        development_reduction_config,
    )
    evaluation_config_identity = ObjectIdentity.from_record(
        evaluation_reduction_config.config_id,
        evaluation_reduction_config,
    )

    def build(
        operation: FreeGsnkeTargetLifecycleOperation,
    ) -> FreeGsnkeTargetLifecycleConfig:
        return FreeGsnkeTargetLifecycleConfig(
            config_id=f"config.{namespace}.{operation.value.lower().replace('_', '-')}",
            operation=operation,
            capability_key=_OPERATION_KEY[operation],
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            target_prediction_freeze_id=f"prediction-freeze.{namespace}",
            development_forecast_freeze_id=f"development-forecasts.{namespace}",
            development_selection_id=f"development-selection.{namespace}",
            target_contract_id=f"target-contract.{namespace}",
            prediction_request_id=f"prediction-request.{namespace}",
            prediction_issue_id=f"prediction-issue.{namespace}",
            admission_evaluation_evaluation_id=f"admission-evaluation.{namespace}",
            method_freeze=method_identity,
            conformance=conformance_identity,
            target_design=target_design_identity,
            structural_recurrence_bridge=bridge_identity,
            action_design=action_design_identity,
            metric_topology_design=metric_topology_identity,
            forecast_encoder=encoder_identity,
            forecast_grammar=grammar_identity,
            forecast_alphabet=alphabet_identity,
            power_freeze=power_identity,
            independence_dossier=dossier_identity,
            structural_recurrence_runtime_config=runtime_identity,
            development_reduction_config=development_config_identity,
            evaluation_reduction_config=evaluation_config_identity,
            admission_evaluation_execution_authority=admission_evaluation_execution_authority,
            admission_evaluation_reveal_authority=admission_evaluation_reveal_authority,
            maximum_input_bytes=512 * 1024**2,
        )

    return tuple(build(value) for value in FreeGsnkeTargetLifecycleOperation)


def _budget(*, seconds: int, memory_gib: int, output_gib: int = 1) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=2,
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
        )
    )
)
_DEVELOP = tuple(sorted((*_READ_WRITE, CapabilityPermission.READ_DEVELOPMENT)))
_EVALUATE = tuple(
    sorted(
        (
            *_DEVELOP,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        )
    )
)


def freegsnke_target_lifecycle_registry(
    *,
    implementation_sha256: str,
) -> CapabilityRegistry:
    """Compose generation, development reduction and two target-owned roles."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    generation = {
        value.capability_key: value
        for value in freegsnke_generation_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    }
    evaluation_generation = generation[FREEGSNKE_EVALUATION_CAPABILITY_KEY]
    generation[FREEGSNKE_EVALUATION_CAPABILITY_KEY] = replace(
        evaluation_generation,
        input_schema_ids=tuple(
            sorted(
                (
                    *evaluation_generation.input_schema_ids,
                    FreeGsnkeTargetPredictionFreeze.SCHEMA,
                )
            )
        ),
        permissions=tuple(
            sorted(
                (
                    *evaluation_generation.permissions,
                    CapabilityPermission.READ_DEVELOPMENT,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    development_reduction = freegsnke_phase_registry(
        implementation_sha256=implementation_sha256
    ).resolve(
        FREEGSNKE_DEVELOPMENT_REDUCTION_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    schema_sha256 = sha256(FreeGsnkeTargetLifecycleConfig.SCHEMA.encode("ascii")).hexdigest()
    freeze_inputs = (
        StructuralBootstrapInputCensus.SCHEMA,
        FreeGsnkeTargetLifecycleConfig.SCHEMA,
        FreeGsnkePhaseReduction.SCHEMA,
        MarginStructuralRecurrenceForecastMethodFreeze.SCHEMA,
        MarginStructuralRecurrenceForecastConformance.SCHEMA,
        FreeGsnkeTargetDesignFreeze.SCHEMA,
        FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA,
        FreeGsnkeActionDesign.SCHEMA,
        FreeGsnkeMetricTopologyDesign.SCHEMA,
        FreeGsnkeForecastEncoderFreeze.SCHEMA,
        IndependentSubstrateForecastAlphabetGrammar.SCHEMA,
        FreeGsnkeForecastAlphabetFreeze.SCHEMA,
        FreeGsnkePowerFreeze.SCHEMA,
        IndependentImplementationDossier.SCHEMA,
        MarginStructuralRecurrenceForecastRuntimeConfig.SCHEMA,
        FreeGsnkeTargetReductionConfig.SCHEMA,
    )
    admission_evaluation_inputs = (
        FreeGsnkeMetricBootstrapInputs.SCHEMA,
        FreeGsnkeTargetLifecycleConfig.SCHEMA,
        FreeGsnkeTargetPredictionFreeze.SCHEMA,
        FreeGsnkeTargetProcessResponse.SCHEMA,
        FreeGsnkeProcessRequest.SCHEMA,
        FreeGsnkePhaseRequestRoster.SCHEMA,
        FreeGsnkeTargetReductionConfig.SCHEMA,
        MarginStructuralRecurrenceForecastMethodFreeze.SCHEMA,
        FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA,
        FreeGsnkeActionDesign.SCHEMA,
        FreeGsnkeMetricTopologyDesign.SCHEMA,
        FreeGsnkeForecastEncoderFreeze.SCHEMA,
        IndependentSubstrateForecastAlphabetGrammar.SCHEMA,
        FreeGsnkeForecastAlphabetFreeze.SCHEMA,
        FreeGsnkePowerFreeze.SCHEMA,
    )
    lifecycle = (
        CapabilityManifest(
            capability_key=FREEGSNKE_TARGET_FREEZE_KEY,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            kind=CapabilityKind.TRANSFORM,
            config_schema=FreeGsnkeTargetLifecycleConfig.SCHEMA,
            config_schema_sha256=schema_sha256,
            input_schema_ids=tuple(sorted(freeze_inputs)),
            output_schema_ids=(FreeGsnkeTargetPredictionFreeze.SCHEMA,),
            permissions=_DEVELOP,
            maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
            maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            resource_ceiling=_budget(seconds=30 * 60, memory_gib=8),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="independent-substrate-grounding-freegsnke-target-freeze",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "development-only-selection",
                "finite-mapping-denominator-roster",
                'registered-margin-structural-recurrence-forecast-predictor-unchanged',
                "target-contract-before-evaluator-access",
            ),
            implementation_sha256=implementation_sha256,
        ),
        CapabilityManifest(
            capability_key=FREEGSNKE_ADMISSION_EVALUATION_KEY,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            kind=CapabilityKind.EVALUATOR,
            config_schema=FreeGsnkeTargetLifecycleConfig.SCHEMA,
            config_schema_sha256=schema_sha256,
            input_schema_ids=tuple(sorted(admission_evaluation_inputs)),
            output_schema_ids=(FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA,),
            permissions=_EVALUATE,
            maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
            maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            resource_ceiling=_budget(seconds=60 * 60, memory_gib=16, output_gib=2),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="independent-substrate-grounding-freegsnke-target-admission-evaluator",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "all-issued-evaluation-units-retained",
                "comparator-and-topology-metric-joint-evaluation",
                "noncompensating-target-admission",
                "single-evaluator-owned-reveal",
            ),
            implementation_sha256=implementation_sha256,
        ),
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-target-lifecycle",
        capabilities=tuple(
            sorted(
                (
                    *generation.values(),
                    development_reduction,
                    *lifecycle,
                ),
                key=lambda value: value.registry_id,
            )
        ),
    )


def freegsnke_target_lifecycle_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_TARGET_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_target_lifecycle_registry(
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


def _generation_subregistry(registry: CapabilityRegistry) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-target-generation-subregistry",
        capabilities=tuple(
            registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION)
            for key in (
                FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
                FREEGSNKE_EVALUATION_CAPABILITY_KEY,
            )
        ),
    )


def build_freegsnke_target_lifecycle_protocol(
    *,
    registry: CapabilityRegistry,
    development_requests: tuple[FreeGsnkeProcessRequest, ...],
    development_request_refs: tuple[CapabilityConfigRef, ...],
    evaluation_requests: tuple[FreeGsnkeProcessRequest, ...],
    evaluation_request_refs: tuple[CapabilityConfigRef, ...],
    development_reduction_config: FreeGsnkeTargetReductionConfig,
    development_reduction_ref: CapabilityConfigRef,
    lifecycle_configs: tuple[FreeGsnkeTargetLifecycleConfig, ...],
    lifecycle_refs: tuple[CapabilityConfigRef, ...],
) -> ProtocolTemplate:
    """Compose development, actual prediction freeze and one admission evaluation reveal."""

    if (
        {value.phase for value in development_requests} != {FreeGsnkePhase.DEVELOPMENT}
        or {value.phase for value in evaluation_requests} != {FreeGsnkePhase.EVALUATION}
        or development_reduction_config.phase is not FreeGsnkePhase.DEVELOPMENT
    ):
        raise ValueError("FreeGSNKE target lifecycle phase partition differs")
    generation_registry = _generation_subregistry(registry)
    development_generation = build_freegsnke_generation_protocol(
        registry=generation_registry,
        requests=development_requests,
        config_refs=development_request_refs,
    )
    evaluation_generation = build_freegsnke_generation_protocol(
        registry=generation_registry,
        requests=evaluation_requests,
        config_refs=evaluation_request_refs,
    )
    development_steps = tuple(
        replace(
            value,
            step_id=f"development.{value.step_id}",
            outputs=tuple(
                replace(output, output_id=f"development.{output.output_id}")
                for output in value.outputs
            ),
            obligation_ids=tuple(
                f"{obligation}.development.{value.config.config_id}"
                for obligation in value.obligation_ids
            ),
        )
        for value in development_generation.steps
    )
    freeze_config = next(
        (
            value
            for value in lifecycle_configs
            if value.operation is FreeGsnkeTargetLifecycleOperation.FREEZE_PREDICTION
        ),
        None,
    )
    admission_evaluation_config = next(
        (
            value
            for value in lifecycle_configs
            if value.operation is FreeGsnkeTargetLifecycleOperation.EVALUATE_ADMISSION
        ),
        None,
    )
    refs_by_id = {value.config_id: value for value in lifecycle_refs}
    if (
        freeze_config is None
        or admission_evaluation_config is None
        or len(lifecycle_configs) != 2
        or len(refs_by_id) != 2
        or set(refs_by_id) != {freeze_config.config_id, admission_evaluation_config.config_id}
    ):
        raise ValueError("FreeGSNKE target lifecycle config roster differs")

    def lifecycle_ref(
        config: FreeGsnkeTargetLifecycleConfig,
    ) -> CapabilityConfigRef:
        manifest = registry.resolve(config.capability_key, config.capability_version)
        ref = refs_by_id[config.config_id]
        if (
            ref.config_schema != config.SCHEMA
            or ref.config_schema_sha256 != manifest.config_schema_sha256
            or ref.content_sha256 != config.fingerprint()
        ):
            raise ValueError("FreeGSNKE target lifecycle config reference differs")
        return ref

    reduction_manifest = registry.resolve(
        FREEGSNKE_DEVELOPMENT_REDUCTION_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    if (
        development_reduction_ref.config_id != development_reduction_config.config_id
        or development_reduction_ref.config_schema != development_reduction_config.SCHEMA
        or development_reduction_ref.config_schema_sha256 != reduction_manifest.config_schema_sha256
        or development_reduction_ref.content_sha256 != development_reduction_config.fingerprint()
    ):
        raise ValueError("FreeGSNKE development reduction config reference differs")
    development_reduction = ProtocolStepTemplate(
        step_id=FREEGSNKE_DEVELOPMENT_REDUCTION_STEP_ID,
        stage=ScientificStage.DEVELOP,
        capability_key=FREEGSNKE_DEVELOPMENT_REDUCTION_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        config=development_reduction_ref,
        dependency_step_ids=tuple(sorted(value.step_id for value in development_steps)),
        outputs=(_output("development-reduction", FreeGsnkePhaseReduction.SCHEMA),),
        required_permissions=reduction_manifest.permissions,
        requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        resource_budget=reduction_manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-development-reducer",),
        barrier=BarrierKind.NONE,
        maximum_attempts=2,
        obligation_ids=(
            "freegsnke-development-all-issued-units-retained",
            "freegsnke-development-complete-preparation-unit",
        ),
    )
    freeze_manifest = registry.resolve(
        FREEGSNKE_TARGET_FREEZE_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    freeze_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_TARGET_FREEZE_STEP_ID,
        stage=ScientificStage.FREEZE,
        capability_key=FREEGSNKE_TARGET_FREEZE_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        config=lifecycle_ref(freeze_config),
        dependency_step_ids=(FREEGSNKE_DEVELOPMENT_REDUCTION_STEP_ID,),
        outputs=(
            _output(
                "target-prediction-freeze",
                FreeGsnkeTargetPredictionFreeze.SCHEMA,
            ),
        ),
        required_permissions=freeze_manifest.permissions,
        requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        resource_budget=freeze_manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-target-prediction-freeze",),
        barrier=BarrierKind.FREEZE,
        maximum_attempts=1,
        obligation_ids=(
            "freegsnke-development-only-selection",
            'freegsnke-exact-margin-structural-recurrence-forecast-prediction-freeze',
            "freegsnke-target-contract-before-evaluation",
        ),
    )
    evaluation_steps = tuple(
        replace(
            value,
            step_id=f"evaluation.{value.step_id}",
            dependency_step_ids=(FREEGSNKE_TARGET_FREEZE_STEP_ID,),
            outputs=tuple(
                replace(output, output_id=f"evaluation.{output.output_id}")
                for output in value.outputs
            ),
            obligation_ids=tuple(
                f"{obligation}.evaluation.{value.config.config_id}"
                for obligation in value.obligation_ids
            ),
            required_permissions=registry.resolve(
                FREEGSNKE_EVALUATION_CAPABILITY_KEY,
                FREEGSNKE_CAPABILITY_VERSION,
            ).permissions,
        )
        for value in evaluation_generation.steps
    )
    admission_evaluation_manifest = registry.resolve(
        FREEGSNKE_ADMISSION_EVALUATION_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    admission_evaluation_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_ADMISSION_EVALUATION_STEP_ID,
        stage=ScientificStage.EVALUATE,
        capability_key=FREEGSNKE_ADMISSION_EVALUATION_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        config=lifecycle_ref(admission_evaluation_config),
        dependency_step_ids=tuple(
            sorted(
                (
                    FREEGSNKE_TARGET_FREEZE_STEP_ID,
                    *(value.step_id for value in evaluation_steps),
                )
            )
        ),
        outputs=(_output("target-admission-evaluation", FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA),),
        required_permissions=admission_evaluation_manifest.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=admission_evaluation_manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-target-admission-evaluator",),
        barrier=BarrierKind.REVEAL,
        maximum_attempts=1,
        obligation_ids=(
            "freegsnke-admission-all-issued-evaluation-units-retained",
            "freegsnke-admission-comparators-and-topology-metric-joint",
            "freegsnke-noncompensating-admission",
            "freegsnke-admission-single-reveal",
        ),
    )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-target-development-through-admission-evaluation-protocol",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=tuple(
            sorted(
                (
                    *development_steps,
                    development_reduction,
                    freeze_step,
                    *evaluation_steps,
                    admission_evaluation_step,
                ),
                key=lambda value: value.step_id,
            )
        ),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _record_id(record: CanonicalRecord) -> str:
    for attribute in (
        "input_id",
        "census_id",
        "binding_id",
        "saved_state_id",
        "design_id",
        "roster_id",
        "request_id",
        "config_id",
        "freeze_id",
        "conformance_id",
        "encoder_id",
        "grammar_id",
        "dossier_id",
    ):
        if hasattr(record, attribute):
            value = getattr(record, attribute)
            if isinstance(value, str):
                return value
    raise TypeError(f"unsupported FreeGSNKE target record identity: {type(record)!r}")


def _artifact_id(record: CanonicalRecord) -> str:
    return f"artifact.independent-substrate-grounding.freegsnke.target.{_record_id(record)}"


def _input_id(record: CanonicalRecord) -> str:
    return f"input.independent-substrate-grounding.freegsnke.target.{_record_id(record)}"


def _role(record: CanonicalRecord) -> ScientificInputRole:
    if isinstance(record, FreeGsnkeSourceBinding):
        return ScientificInputRole.PREPARED_MEDIUM
    if isinstance(record, FreeGsnkeSavedPreparation):
        return ScientificInputRole.PREPARED_MEDIUM
    if isinstance(record, (FreeGsnkeActionDesign, FreeGsnkeProcessRequest)):
        return ScientificInputRole.ACTION
    if isinstance(
        record,
        (
            FreeGsnkePhaseRequestRoster,
            FreeGsnkeTargetReductionConfig,
            FreeGsnkePowerFreeze,
        ),
    ):
        return ScientificInputRole.DENOMINATOR
    if isinstance(
        record,
        (
            MarginStructuralRecurrenceForecastMethodFreeze,
            MarginStructuralRecurrenceForecastConformance,
            MarginStructuralRecurrenceForecastRuntimeConfig,
            FreeGsnkeTargetDesignFreeze,
            IndependentImplementationDossier,
        ),
    ):
        return ScientificInputRole.QUALIFICATION
    return ScientificInputRole.MODEL


def _external(record: CanonicalRecord) -> CandidateGraphExternalInput:
    return CandidateGraphExternalInput(
        input_id=_input_id(record),
        scientific_role=_role(record),
        logical_artifact_id=_artifact_id(record),
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=record.fingerprint(),
        payload_schema=record.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=256 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def freegsnke_target_lifecycle_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    static_records: tuple[CanonicalRecord, ...],
) -> CandidateScientificGraph:
    """Bind exact pre-issue operands and every claim-bearing response edge."""

    if len({_record_id(value) for value in static_records}) != len(static_records):
        raise ValueError("FreeGSNKE target static identities repeat")
    externals = tuple(_external(value) for value in static_records)
    external_by_id = {value.input_id: value for value in externals}
    records_by_type: dict[type[CanonicalRecord], tuple[CanonicalRecord, ...]] = {}
    for record in static_records:
        records_by_type.setdefault(type(record), ())
        records_by_type[type(record)] = (*records_by_type[type(record)], record)
    singleton_types: tuple[type[CanonicalRecord], ...] = (
        FreeGsnkeMetricBootstrapInputs,
        StructuralBootstrapInputCensus,
        FreeGsnkeSourceBinding,
        FreeGsnkeActionDesign,
        MarginStructuralRecurrenceForecastMethodFreeze,
        MarginStructuralRecurrenceForecastConformance,
        FreeGsnkeTargetDesignFreeze,
        FreeGsnkeStructuralRecurrenceBridgeFreeze,
        FreeGsnkeMetricTopologyDesign,
        FreeGsnkeForecastEncoderFreeze,
        IndependentSubstrateForecastAlphabetGrammar,
        FreeGsnkeForecastAlphabetFreeze,
        FreeGsnkePowerFreeze,
        IndependentImplementationDossier,
        MarginStructuralRecurrenceForecastRuntimeConfig,
    )
    if any(len(records_by_type.get(value, ())) != 1 for value in singleton_types):
        raise ValueError("FreeGSNKE target static singleton roster differs")
    requests = tuple(
        value for value in static_records if isinstance(value, FreeGsnkeProcessRequest)
    )
    saved_preparations = tuple(
        value for value in static_records if isinstance(value, FreeGsnkeSavedPreparation)
    )
    saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
    rosters = tuple(
        value for value in static_records if isinstance(value, FreeGsnkePhaseRequestRoster)
    )
    reduction_configs = tuple(
        value for value in static_records if isinstance(value, FreeGsnkeTargetReductionConfig)
    )
    expected_saved_ids = {
        value.preparation.preparation_id
        for value in requests
        if value.preparation.saved_state is not None
        and value.preparation.saved_state.payload_schema
        == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
    }
    if (
        {value.phase for value in requests}
        != {FreeGsnkePhase.DEVELOPMENT, FreeGsnkePhase.EVALUATION}
        or {value.phase for value in rosters}
        != {FreeGsnkePhase.DEVELOPMENT, FreeGsnkePhase.EVALUATION}
        or {value.phase for value in reduction_configs}
        != {FreeGsnkePhase.DEVELOPMENT, FreeGsnkePhase.EVALUATION}
        or len(saved_by_preparation) != len(saved_preparations)
        or set(saved_by_preparation) != expected_saved_ids
    ):
        raise ValueError("FreeGSNKE target graph phase operands differ")
    for request in requests:
        if request.preparation.preparation_id not in expected_saved_ids:
            continue
        FreeGsnkeTargetWorkerInput(
            input_id=f"worker-input.{request.request_id}",
            request=request,
            saved_preparation=saved_by_preparation[request.preparation.preparation_id],
        )
    source = records_by_type[FreeGsnkeSourceBinding][0]
    action_design = records_by_type[FreeGsnkeActionDesign][0]
    assert isinstance(source, FreeGsnkeSourceBinding)
    assert isinstance(action_design, FreeGsnkeActionDesign)
    steps = {value.step_id: value for value in protocol.steps}
    development_steps = tuple(
        value
        for value in protocol.steps
        if value.capability_key == FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY
    )
    evaluation_steps = tuple(
        value
        for value in protocol.steps
        if value.capability_key == FREEGSNKE_EVALUATION_CAPABILITY_KEY
    )
    reduction = steps[FREEGSNKE_DEVELOPMENT_REDUCTION_STEP_ID]
    freeze = steps[FREEGSNKE_TARGET_FREEZE_STEP_ID]
    admission_evaluation = steps[FREEGSNKE_ADMISSION_EVALUATION_STEP_ID]
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

    def external_edge(record: CanonicalRecord, consumer: ProtocolStepTemplate) -> None:
        external = external_by_id[_input_id(record)]
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.external.{_record_id(record)}.{consumer.step_id}",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=external.input_id,
                consumer_node_id=consumer.step_id,
                consumer_input_id=f"input.{_record_id(record)}",
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

    def dependency_edge(
        producer: ProtocolStepTemplate,
        output: OutputTemplate,
        consumer: ProtocolStepTemplate,
        *,
        role: ScientificInputRole,
    ) -> None:
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{producer.step_id}.{output.output_id}.{consumer.step_id}",
                producer_node_id=producer.step_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=consumer.step_id,
                consumer_input_id=f"input.{producer.step_id}.{output.output_id}",
                scientific_role=role,
                logical_artifact_id=f"artifact.{producer.step_id}.{output.output_id}",
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=producer.resource_budget.output_bytes,
                outcome_access=producer.requested_outcome_access,
                visibility_ceiling=producer.visibility_ceiling,
                barrier=consumer.barrier,
            )
        )

    request_by_id = {value.request_id: value for value in requests}
    for step in (*development_steps, *evaluation_steps):
        external_edge(source, step)
        try:
            request = request_by_id[step.config.config_id]
        except KeyError as error:
            raise ValueError("FreeGSNKE target generation step lacks its request") from error
        saved = saved_by_preparation.get(request.preparation.preparation_id)
        if saved is not None:
            external_edge(saved, step)
    for step in development_steps:
        dependency_edge(
            step,
            step.outputs[0],
            reduction,
            role=ScientificInputRole.OUTCOME,
        )
    development_roster = next(
        value for value in rosters if value.phase is FreeGsnkePhase.DEVELOPMENT
    )
    development_config = next(
        value for value in reduction_configs if value.phase is FreeGsnkePhase.DEVELOPMENT
    )
    external_edge(action_design, reduction)
    external_edge(development_roster, reduction)
    external_edge(development_config, reduction)
    for request in requests:
        if request.phase is FreeGsnkePhase.DEVELOPMENT:
            external_edge(request, reduction)
    dependency_edge(
        reduction,
        reduction.outputs[0],
        freeze,
        role=ScientificInputRole.OUTCOME,
    )
    freeze_static_types = (
        StructuralBootstrapInputCensus,
        MarginStructuralRecurrenceForecastMethodFreeze,
        MarginStructuralRecurrenceForecastConformance,
        FreeGsnkeTargetDesignFreeze,
        FreeGsnkeStructuralRecurrenceBridgeFreeze,
        FreeGsnkeActionDesign,
        FreeGsnkeMetricTopologyDesign,
        FreeGsnkeForecastEncoderFreeze,
        IndependentSubstrateForecastAlphabetGrammar,
        FreeGsnkeForecastAlphabetFreeze,
        FreeGsnkePowerFreeze,
        IndependentImplementationDossier,
        MarginStructuralRecurrenceForecastRuntimeConfig,
    )
    for record_type in freeze_static_types:
        external_edge(records_by_type[record_type][0], freeze)
    for config in reduction_configs:
        external_edge(config, freeze)
    for step in evaluation_steps:
        dependency_edge(
            freeze,
            freeze.outputs[0],
            step,
            role=ScientificInputRole.QUALIFICATION,
        )
        dependency_edge(
            step,
            step.outputs[0],
            admission_evaluation,
            role=ScientificInputRole.OUTCOME,
        )
    dependency_edge(
        freeze,
        freeze.outputs[0],
        admission_evaluation,
        role=ScientificInputRole.QUALIFICATION,
    )
    evaluation_roster = next(value for value in rosters if value.phase is FreeGsnkePhase.EVALUATION)
    evaluation_config = next(
        value for value in reduction_configs if value.phase is FreeGsnkePhase.EVALUATION
    )
    for record in (
        records_by_type[FreeGsnkeMetricBootstrapInputs][0],
        records_by_type[MarginStructuralRecurrenceForecastMethodFreeze][0],
        records_by_type[FreeGsnkeStructuralRecurrenceBridgeFreeze][0],
        action_design,
        records_by_type[FreeGsnkeMetricTopologyDesign][0],
        records_by_type[FreeGsnkeForecastEncoderFreeze][0],
        records_by_type[IndependentSubstrateForecastAlphabetGrammar][0],
        records_by_type[FreeGsnkeForecastAlphabetFreeze][0],
        records_by_type[FreeGsnkePowerFreeze][0],
        evaluation_roster,
        evaluation_config,
    ):
        external_edge(record, admission_evaluation)
    for request in requests:
        if request.phase is FreeGsnkePhase.EVALUATION:
            external_edge(request, admission_evaluation)
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.freegsnke-target-development-through-admission-evaluation",
        external_inputs=tuple(sorted(externals, key=lambda value: value.input_id)),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def freegsnke_target_lifecycle_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
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
        template_key="independent-substrate-grounding.freegsnke.target-development-through-admission-evaluation",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.independent-substrate-grounding.freegsnke.target-development-through-admission-evaluation",
            bindings=bindings,
        ),
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
        raise ValueError(f"FreeGSNKE target lifecycle {label} requires one record")
    return values[0]


def _identity(record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(_record_id(record), record)


def _phase_reduction_configs(
    context: TaskContext,
) -> tuple[FreeGsnkeTargetReductionConfig, FreeGsnkeTargetReductionConfig]:
    raw = _decode_records(context, FreeGsnkeTargetReductionConfig)
    values = tuple(
        sorted(
            (value for value in raw if isinstance(value, FreeGsnkeTargetReductionConfig)),
            key=lambda value: value.phase.value,
        )
    )
    by_phase = {value.phase: value for value in values}
    if len(values) != 2 or set(by_phase) != {
        FreeGsnkePhase.DEVELOPMENT,
        FreeGsnkePhase.EVALUATION,
    }:
        raise ValueError("FreeGSNKE target lifecycle phase configs differ")
    return (
        by_phase[FreeGsnkePhase.DEVELOPMENT],
        by_phase[FreeGsnkePhase.EVALUATION],
    )


class _FreeGsnkeTargetLifecycleRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def _config(self, context: TaskContext) -> FreeGsnkeTargetLifecycleConfig:
        value = _one_record(
            context,
            FreeGsnkeTargetLifecycleConfig,
            label="runner config",
        )
        assert isinstance(value, FreeGsnkeTargetLifecycleConfig)
        if (
            value.capability_key != self.manifest.capability_key
            or value.capability_version != self.manifest.capability_version
            or context.config.config_schema != value.SCHEMA
            or context.config.content_sha256 != value.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not self.manifest.maximum_outcome_access
            or any(port.size_bytes > value.maximum_input_bytes for port in context.input_ports)
        ):
            raise ValueError("FreeGSNKE target runner/config authority differs")
        return value

    @staticmethod
    def _emit(
        context: TaskContext,
        record: CanonicalRecord,
        checks: tuple[ReceiptCheck, ...],
    ) -> RunnerResult:
        if len(context.output_ports) != 1 or context.output_ports[0].payload_schema != (
            record.SCHEMA
        ):
            raise ValueError("FreeGSNKE target lifecycle output schema differs")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=record.canonical_bytes(),
                ),
            ),
            checks=tuple(sorted(checks, key=lambda value: value.check_id)),
        )

    @staticmethod
    def _common_operands(
        context: TaskContext,
    ) -> tuple[
        MarginStructuralRecurrenceForecastMethodFreeze,
        FreeGsnkeStructuralRecurrenceBridgeFreeze,
        FreeGsnkeActionDesign,
        FreeGsnkeMetricTopologyDesign,
        FreeGsnkeForecastEncoderFreeze,
        IndependentSubstrateForecastAlphabetGrammar,
        FreeGsnkeForecastAlphabetFreeze,
        FreeGsnkePowerFreeze,
    ]:
        values = tuple(
            _one_record(context, record_type, label=record_type.SCHEMA)
            for record_type in (
                MarginStructuralRecurrenceForecastMethodFreeze,
                FreeGsnkeStructuralRecurrenceBridgeFreeze,
                FreeGsnkeActionDesign,
                FreeGsnkeMetricTopologyDesign,
                FreeGsnkeForecastEncoderFreeze,
                IndependentSubstrateForecastAlphabetGrammar,
                FreeGsnkeForecastAlphabetFreeze,
                FreeGsnkePowerFreeze,
            )
        )
        method, bridge, action, metric, encoder, grammar, alphabet, power = values
        assert isinstance(method, MarginStructuralRecurrenceForecastMethodFreeze)
        assert isinstance(bridge, FreeGsnkeStructuralRecurrenceBridgeFreeze)
        assert isinstance(action, FreeGsnkeActionDesign)
        assert isinstance(metric, FreeGsnkeMetricTopologyDesign)
        assert isinstance(encoder, FreeGsnkeForecastEncoderFreeze)
        assert isinstance(grammar, IndependentSubstrateForecastAlphabetGrammar)
        assert isinstance(alphabet, FreeGsnkeForecastAlphabetFreeze)
        assert isinstance(power, FreeGsnkePowerFreeze)
        return method, bridge, action, metric, encoder, grammar, alphabet, power

    def _freeze_prediction(
        self,
        context: TaskContext,
        config: FreeGsnkeTargetLifecycleConfig,
    ) -> RunnerResult:
        (
            method,
            bridge,
            action,
            metric,
            encoder,
            grammar,
            alphabet,
            power,
        ) = self._common_operands(context)
        scientific_bootstrap_inputs = _one_record(context, StructuralBootstrapInputCensus, label="complete original bootstrap census")
        assert isinstance(scientific_bootstrap_inputs, StructuralBootstrapInputCensus)
        conformance = _one_record(context, MarginStructuralRecurrenceForecastConformance, label="conformance")
        target_design = _one_record(
            context,
            FreeGsnkeTargetDesignFreeze,
            label="target design",
        )
        dossier = _one_record(
            context,
            IndependentImplementationDossier,
            label="independence dossier",
        )
        runtime = _one_record(context, MarginStructuralRecurrenceForecastRuntimeConfig, label='structural recurrence runtime')
        development_reduction = _one_record(
            context,
            FreeGsnkePhaseReduction,
            label="development reduction",
        )
        development_config, evaluation_config = _phase_reduction_configs(context)
        assert isinstance(conformance, MarginStructuralRecurrenceForecastConformance)
        assert isinstance(target_design, FreeGsnkeTargetDesignFreeze)
        assert isinstance(dossier, IndependentImplementationDossier)
        assert isinstance(runtime, MarginStructuralRecurrenceForecastRuntimeConfig)
        assert isinstance(development_reduction, FreeGsnkePhaseReduction)
        observed = (
            _identity(method),
            _identity(conformance),
            _identity(target_design),
            _identity(bridge),
            _identity(action),
            _identity(metric),
            _identity(encoder),
            _identity(grammar),
            _identity(alphabet),
            _identity(power),
            _identity(dossier),
            _identity(runtime),
            _identity(development_config),
            _identity(evaluation_config),
        )
        expected = (
            config.method_freeze,
            config.conformance,
            config.target_design,
            config.structural_recurrence_bridge,
            config.action_design,
            config.metric_topology_design,
            config.forecast_encoder,
            config.forecast_grammar,
            config.forecast_alphabet,
            config.power_freeze,
            config.independence_dossier,
            config.structural_recurrence_runtime_config,
            config.development_reduction_config,
            config.evaluation_reduction_config,
        )
        if (
            observed != expected
            or runtime.operation is not MarginStructuralRecurrenceForecastRuntimeOperation.PREDICT
            or runtime.capability_version != MARGIN_FORECAST_CAPABILITY_VERSION
            or development_reduction.phase is not FreeGsnkePhase.DEVELOPMENT
            or development_reduction.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("FreeGSNKE prediction-freeze operand closure differs")
        forecasts = freeze_freegsnke_development_forecasts(
            freeze_id=config.development_forecast_freeze_id,
            selection_id=config.development_selection_id,
            target_design=target_design,
            bridge=bridge,
            reduction_config=development_config,
            development_reduction=development_reduction,
            action_design=action,
            metric_topology_design=metric,
            encoder=encoder,
            forecast_grammar=grammar,
            forecast_alphabet=alphabet,
            execution_authority_verified=True,
        )
        contract = build_freegsnke_target_prediction_contract(
            contract_id=config.target_contract_id,
            target_design=target_design,
            forecast_alphabet=alphabet,
            power_freeze=power,
            metric_topology_design=metric,
            independence_dossier=dossier,
            development_forecast_freeze=forecasts,
        )
        request = build_freegsnke_structural_recurrence_prediction_request(
            request_id=config.prediction_request_id,
            issue_id=config.prediction_issue_id,
            method_freeze=method,
            conformance=conformance,
            power_freeze=power,
            development_forecast_freeze=forecasts,
            contract=contract,
            scientific_bootstrap_inputs=scientific_bootstrap_inputs,
        )
        issue = decode_canonical_bytes(
            registered_predict(runtime.canonical_bytes(), request.canonical_bytes()),
            MarginStructuralRecurrenceForecastPredictionIssue,
            maximum_bytes=runtime.maximum_input_bytes,
        )
        freeze = FreeGsnkeTargetPredictionFreeze(
            freeze_id=config.target_prediction_freeze_id,
            development_forecasts=forecasts,
            target_contract=contract,
            prediction_request=request,
            prediction_issue=issue,
            structural_recurrence_runtime_config=_identity(runtime),
            frozen_before_evaluator_access=True,
            evaluation_outcome_access_count=0,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        return self._emit(
            context,
            freeze,
            (
                ReceiptCheck("freegsnke-development-selection-frozen", True, ()),
                ReceiptCheck('freegsnke-margin-structural-recurrence-forecast-predictor-registered', True, ()),
                ReceiptCheck("freegsnke-target-contract-pre-reveal", True, ()),
            ),
        )

    def _evaluate_admission(
        self,
        context: TaskContext,
        config: FreeGsnkeTargetLifecycleConfig,
    ) -> RunnerResult:
        metric_bootstrap_inputs = _one_record(context, FreeGsnkeMetricBootstrapInputs, label="metric bootstrap inputs")
        assert isinstance(metric_bootstrap_inputs, FreeGsnkeMetricBootstrapInputs)
        (
            method,
            bridge,
            action,
            metric,
            encoder,
            grammar,
            alphabet,
            power,
        ) = self._common_operands(context)
        freeze = _one_record(
            context,
            FreeGsnkeTargetPredictionFreeze,
            label="prediction freeze",
        )
        roster = _one_record(
            context,
            FreeGsnkePhaseRequestRoster,
            label="evaluation roster",
        )
        configs = _decode_records(context, FreeGsnkeTargetReductionConfig)
        requests_raw = _decode_records(context, FreeGsnkeProcessRequest)
        responses_raw = _decode_records(context, FreeGsnkeTargetProcessResponse)
        assert isinstance(freeze, FreeGsnkeTargetPredictionFreeze)
        assert isinstance(roster, FreeGsnkePhaseRequestRoster)
        evaluation_configs = tuple(
            value
            for value in configs
            if isinstance(value, FreeGsnkeTargetReductionConfig)
            and value.phase is FreeGsnkePhase.EVALUATION
        )
        requests = tuple(
            sorted(
                (value for value in requests_raw if isinstance(value, FreeGsnkeProcessRequest)),
                key=lambda value: value.request_id,
            )
        )
        responses = tuple(
            sorted(
                (
                    value
                    for value in responses_raw
                    if isinstance(value, FreeGsnkeTargetProcessResponse)
                ),
                key=lambda value: value.response_id,
            )
        )
        if len(evaluation_configs) != 1:
            raise ValueError("FreeGSNKE admission evaluation requires one evaluation reduction config")
        evaluation_config = evaluation_configs[0]
        observed = (
            _identity(method),
            _identity(bridge),
            _identity(action),
            _identity(metric),
            _identity(encoder),
            _identity(grammar),
            _identity(alphabet),
            _identity(power),
            _identity(evaluation_config),
        )
        expected = (
            config.method_freeze,
            config.structural_recurrence_bridge,
            config.action_design,
            config.metric_topology_design,
            config.forecast_encoder,
            config.forecast_grammar,
            config.forecast_alphabet,
            config.power_freeze,
            config.evaluation_reduction_config,
        )
        if (
            observed != expected
            or freeze.freeze_id != config.target_prediction_freeze_id
            or freeze.structural_recurrence_runtime_config != config.structural_recurrence_runtime_config
            or roster.phase is not FreeGsnkePhase.EVALUATION
            or {value.phase for value in requests} != {FreeGsnkePhase.EVALUATION}
        ):
            raise ValueError("FreeGSNKE admission evaluation frozen/evaluation operands differ")
        reduction = reduce_freegsnke_phase_panel(
            config=evaluation_config,
            roster=roster,
            action_design=action,
            requests=requests,
            responses=responses,
        )
        admission_evaluation = evaluate_freegsnke_admission(
            evaluation_id=config.admission_evaluation_evaluation_id,
            method_freeze=method,
            target_contract=freeze.target_contract,
            development_forecast_freeze=freeze.development_forecasts,
            prediction_issue=freeze.prediction_issue,
            target_power_freeze=power,
            metric_topology_design=metric,
            bridge=bridge,
            evaluation_reduction_config=evaluation_config,
            evaluation_reduction=reduction,
            action_design=action,
            encoder=encoder,
            forecast_grammar=grammar,
            forecast_alphabet=alphabet,
            execution_authority=config.admission_evaluation_execution_authority,
            reveal_authority=config.admission_evaluation_reveal_authority,
            metric_bootstrap_inputs=metric_bootstrap_inputs,
        )
        return self._emit(
            context,
            admission_evaluation,
            (
                ReceiptCheck("freegsnke-admission-adverse-outcomes-retained", True, ()),
                ReceiptCheck("freegsnke-admission-comparators-joint", True, ()),
                ReceiptCheck("freegsnke-admission-single-evaluator-reveal", True, ()),
            ),
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        config = self._config(context)
        if config.operation is FreeGsnkeTargetLifecycleOperation.FREEZE_PREDICTION:
            return self._freeze_prediction(context, config)
        return self._evaluate_admission(context, config)


def freegsnke_target_lifecycle_runners(
    *,
    registry: CapabilityRegistry,
) -> tuple[TaskRunner, ...]:
    return tuple(
        _FreeGsnkeTargetLifecycleRunner(
            registry.resolve(_OPERATION_KEY[operation], FREEGSNKE_CAPABILITY_VERSION)
        )
        for operation in FreeGsnkeTargetLifecycleOperation
    )


class FreeGsnkeTargetLifecycleRuntimeProvider(CampaignRuntimeProvider):
    """Path-free provider for the complete target-native development-through-admission evaluation parent."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        lifecycle_configs: tuple[FreeGsnkeTargetLifecycleConfig, ...],
        static_records: tuple[CanonicalRecord, ...],
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        if not registry.capabilities:
            raise ValueError("FreeGSNKE target lifecycle registry is empty")
        expected = freegsnke_target_lifecycle_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("FreeGSNKE target lifecycle provider registry differs")
        if (
            {value.operation for value in lifecycle_configs}
            != set(FreeGsnkeTargetLifecycleOperation)
            or len({value.config_id for value in lifecycle_configs}) != len(lifecycle_configs)
            or len({_record_id(value) for value in static_records}) != len(static_records)
        ):
            raise ValueError("FreeGSNKE target lifecycle provider roster differs")
        sources = tuple(
            value for value in static_records if isinstance(value, FreeGsnkeSourceBinding)
        )
        requests = tuple(
            value for value in static_records if isinstance(value, FreeGsnkeProcessRequest)
        )
        saved_preparations = tuple(
            value for value in static_records if isinstance(value, FreeGsnkeSavedPreparation)
        )
        saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
        expected_saved_ids = {
            value.preparation.preparation_id
            for value in requests
            if value.preparation.saved_state is not None
            and value.preparation.saved_state.payload_schema
            == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
        }
        if (
            len(sources) != 1
            or not requests
            or {value.phase for value in requests}
            != {FreeGsnkePhase.DEVELOPMENT, FreeGsnkePhase.EVALUATION}
            or {value.source_binding for value in requests} != {sources[0]}
            or len(saved_by_preparation) != len(saved_preparations)
            or set(saved_by_preparation) != expected_saved_ids
        ):
            raise ValueError("FreeGSNKE target lifecycle source/request scope differs")
        for request in requests:
            if request.preparation.preparation_id not in expected_saved_ids:
                continue
            FreeGsnkeTargetWorkerInput(
                input_id=f"worker-input.{request.request_id}",
                request=request,
                saved_preparation=saved_by_preparation[request.preparation.preparation_id],
            )
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.lifecycle_configs = tuple(sorted(lifecycle_configs, key=lambda value: value.config_id))
        self.static_records = tuple(sorted(static_records, key=_record_id))
        generation_runners = freegsnke_generation_runners(
            registry=registry,
            source_factory=source_factory,
            executor=executor,
        )
        reduction_runners = freegsnke_phase_reduction_runners(
            registry=registry,
            phases=(FreeGsnkePhase.DEVELOPMENT,),
        )
        target_runners = freegsnke_target_lifecycle_runners(registry=registry)
        self._runners = (*generation_runners, *reduction_runners, *target_runners)
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE target lifecycle registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE target lifecycle plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        config_refs = {
            task.capability.config.artifact_id: task.capability.config for task in plan.tasks
        }
        static_by_artifact = {_artifact_id(value): value for value in self.static_records}
        configurable = (*self.static_records, *self.lifecycle_configs)
        config_by_hash = {value.fingerprint(): value for value in configurable}
        values = []
        for artifact_id, spec in sorted(specs.items()):
            config_ref = config_refs.get(artifact_id)
            if config_ref is not None:
                try:
                    record = config_by_hash[config_ref.content_sha256]
                except KeyError as error:
                    raise ValueError(
                        "FreeGSNKE target plan references an unknown config"
                    ) from error
            else:
                try:
                    record = static_by_artifact[artifact_id]
                except KeyError as error:
                    raise ValueError(
                        f"FreeGSNKE target plan references an unknown input: {artifact_id}"
                    ) from error
            parent = ArtifactLineageParent(
                identity=_identity(record),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                    ),
                    outcome_access=(spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND),
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
            raise ValueError("FreeGSNKE target lifecycle semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types: dict[str, type[CanonicalRecord]] = {
            FreeGsnkeTargetProcessResponse.SCHEMA: (FreeGsnkeTargetProcessResponse),
            FreeGsnkePhaseReduction.SCHEMA: FreeGsnkePhaseReduction,
            FreeGsnkeTargetPredictionFreeze.SCHEMA: (FreeGsnkeTargetPredictionFreeze),
            FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA: FreeGsnkeAdmissionEvaluationEvaluation,
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
            raise ValueError("FreeGSNKE target lifecycle adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "FREEGSNKE_DEVELOPMENT_REDUCTION_STEP_ID",
    "FREEGSNKE_TARGET_FREEZE_KEY",
    "FREEGSNKE_TARGET_FREEZE_STEP_ID",
    "FREEGSNKE_ADMISSION_EVALUATION_KEY",
    "FREEGSNKE_ADMISSION_EVALUATION_STEP_ID",
    "FREEGSNKE_TARGET_PROVIDER_KEY",
    'FreeGsnkeTargetLifecycleConfig',
    "FreeGsnkeTargetLifecycleOperation",
    "FreeGsnkeTargetLifecycleRuntimeProvider",
    'FreeGsnkeTargetPredictionFreeze',
    "build_freegsnke_target_lifecycle_protocol",
    "freegsnke_target_lifecycle_candidate_registrations",
    "freegsnke_target_lifecycle_configs",
    'freegsnke_target_lifecycle_study_template',
    "freegsnke_target_lifecycle_registry",
    "freegsnke_target_lifecycle_runners",
    "freegsnke_target_lifecycle_scientific_graph",
]
