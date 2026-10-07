"""Public receipt-producing preissue gate over two selective-response development forecasts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificInputRole, ScientificStage
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .contracts import SelectiveDependenceResponseMethodQuestionFreeze
from .development_completion import SelectiveDependenceResponseDevelopmentCompletionEnvelope
from .forecast import SelectiveDependenceResponseStudyForecastQualification, qualify_study_forecasts


SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION = "1.0.0"
SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY = (
    "method.selective-dependence-response.programme-forecast.qualify-evaluation-issue"
)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseStudyForecastRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-study-forecast-runtime-config'

    config_id: str
    capability_key: str
    capability_version: str
    development_completions: tuple[ObjectIdentity, ...]
    maximum_completion_bytes: int
    evaluation_outcome_count: int

    def __post_init__(self) -> None:
        for name in ("config_id", "capability_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        if self.capability_key != SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY:
            raise ValueError("programme forecast capability key differs")
        if self.capability_version != SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION:
            raise ValueError("programme forecast capability version differs")
        require_sorted_unique_ids(
            self.development_completions,
            attribute="object_id",
            field_name="development_completions",
        )
        if len(self.development_completions) != 2:
            raise ValueError("programme forecast config requires two completions")
        if not 0 < self.maximum_completion_bytes <= 64 * 1024**2:
            raise ValueError("programme forecast completion byte ceiling differs")
        if self.evaluation_outcome_count:
            raise ValueError("programme forecast config cannot inspect evaluation")


def study_forecast_runtime_config(
    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ],
) -> SelectiveDependenceResponseStudyForecastRuntimeConfig:
    ordered = _ordered_completions(completions)
    return SelectiveDependenceResponseStudyForecastRuntimeConfig(
        config_id="selective-dependence-response.programme-forecast.config",
        capability_key=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY,
        capability_version=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION,
        development_completions=tuple(
            sorted(
                (ObjectIdentity.from_record(value.envelope_id, value) for value in ordered),
                key=lambda value: value.object_id,
            )
        ),
        maximum_completion_bytes=64 * 1024**2,
        evaluation_outcome_count=0,
    )


def _ordered_completions(
    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ],
) -> tuple[SelectiveDependenceResponseDevelopmentCompletionEnvelope, SelectiveDependenceResponseDevelopmentCompletionEnvelope]:
    ordered = tuple(sorted(completions, key=lambda value: value.target_slug))
    if tuple(value.target_slug for value in ordered) != ("cantera", "fipy"):
        raise ValueError("programme forecast requires exact Cantera/FiPy completions")
    if len({value.target_id for value in ordered}) != 2:
        raise ValueError("programme forecast completion target identities repeat")
    if any(not value.evaluation_eligible for value in ordered):
        raise ValueError("programme forecast requires evaluation-eligible completions")
    if len({value.development_implementation_sha256 for value in ordered}) != 1:
        raise ValueError("programme forecast completion implementations differ")
    return (ordered[0], ordered[1])


def decode_study_forecast_runtime_config(
    payload: bytes,
) -> SelectiveDependenceResponseStudyForecastRuntimeConfig:
    return decode_canonical_bytes(
        payload,
        SelectiveDependenceResponseStudyForecastRuntimeConfig,
        maximum_bytes=256 * 1024,
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=256 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=64 * 1024**2,
        output_bytes=2 * 1024**2,
    )


def study_forecast_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    manifest = CapabilityManifest(
        capability_key=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY,
        capability_version=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION,
        kind=CapabilityKind.REPORTER,
        config_schema=SelectiveDependenceResponseStudyForecastRuntimeConfig.SCHEMA,
        config_schema_sha256=sha256(
            SelectiveDependenceResponseStudyForecastRuntimeConfig.SCHEMA.encode("ascii")
        ).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    SelectiveDependenceResponseStudyForecastRuntimeConfig.SCHEMA,
                    SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA,
                    SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
                )
            )
        ),
        output_schema_ids=(SelectiveDependenceResponseStudyForecastQualification.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_DEVELOPMENT,
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-selective-dependence-response-programme-forecast",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "exact-two-development-completions",
            "outcome-free-programme-policy-diversity",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="selective-dependence-response-programme-forecast-runtime",
        capabilities=(manifest,),
    )


def study_forecast_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    manifest = study_forecast_registry(
        implementation_sha256=implementation_sha256
    ).capabilities[0]
    return (
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key="selective-dependence-response.programme-forecast-provider",
            provider_version=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=256 * 1024,
        ),
    )


def build_study_forecast_protocol(
    *,
    registry: CapabilityRegistry,
    config: CapabilityConfigRef,
) -> ProtocolTemplate:
    manifest = registry.resolve(
        SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY,
        SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION,
    )
    if (
        config.config_schema != manifest.config_schema
        or config.config_schema_sha256 != manifest.config_schema_sha256
    ):
        raise ValueError("programme forecast config reference differs")
    step = ProtocolStepTemplate(
        step_id="qualify-programme-forecast",
        stage=ScientificStage.FREEZE,
        capability_key=manifest.capability_key,
        capability_version=manifest.capability_version,
        config=config,
        dependency_step_ids=(),
        outputs=(
            OutputTemplate(
                output_id="qualify-programme-forecast.record",
                payload_schema=SelectiveDependenceResponseStudyForecastQualification.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=manifest.permissions,
        requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        resource_budget=manifest.resource_ceiling,
        resource_lock_ids=("selective-dependence-response-programme-forecast",),
        barrier=BarrierKind.FREEZE,
        maximum_attempts=2,
        obligation_ids=("selective-dependence-response-programme-policy-diversity-before-issue",),
    )
    return ProtocolTemplate(
        template_id="selective-dependence-response-programme-forecast-protocol",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION,
        steps=(step,),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def study_forecast_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
) -> CandidateScientificGraph:
    step = protocol.steps[0]
    ordered = _ordered_completions(completions)
    completion_inputs = tuple(
        CandidateGraphExternalInput(
            input_id=f"input.selective-dependence-response.development-completion.{value.target_slug}",
            scientific_role=ScientificInputRole.PARENT_RECEIPT,
            logical_artifact_id=value.envelope_id,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=value.fingerprint(),
            payload_schema=SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=64 * 1024**2,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        )
        for value in ordered
    )
    method_input = CandidateGraphExternalInput(
        input_id="input.selective-dependence-response.programme-method-question",
        scientific_role=ScientificInputRole.MODEL,
        logical_artifact_id=method_question.freeze_id,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=method_question.fingerprint(),
        payload_schema=SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=2 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    inputs = tuple(sorted((*completion_inputs, method_input), key=lambda value: value.input_id))
    node = CandidateGraphNode(
        node_id=step.step_id,
        stage=step.stage,
        capability_key=step.capability_key,
        capability_version=step.capability_version,
        implementation_sha256=registry.capabilities[0].implementation_sha256,
        protocol_step_sha256=step.fingerprint(),
        obligation_ids=step.obligation_ids,
        outcome_access=step.requested_outcome_access,
        visibility_ceiling=step.visibility_ceiling,
        resource_budget=step.resource_budget,
    )
    edges = tuple(
        CandidateGraphEdge(
            edge_id=f"edge.{value.input_id}.qualify-programme-forecast",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=value.input_id,
            consumer_node_id=step.step_id,
            consumer_input_id=f"{value.input_id}.qualify",
            scientific_role=value.scientific_role,
            logical_artifact_id=value.logical_artifact_id,
            payload_schema=value.payload_schema,
            media_type=value.media_type,
            maximum_size_bytes=value.maximum_size_bytes,
            outcome_access=value.outcome_access,
            visibility_ceiling=value.visibility_ceiling,
            barrier=step.barrier,
        )
        for value in inputs
    )
    return CandidateScientificGraph(
        graph_id="graph.selective-dependence-response.programme-forecast",
        external_inputs=inputs,
        nodes=(node,),
        edges=edges,
    )


def study_forecast_candidate_catalog(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
) -> CandidateCapabilityCatalog:
    graph = study_forecast_scientific_graph(
        protocol=protocol,
        registry=registry,
        completions=completions,
        method_question=method_question,
    )
    step = protocol.steps[0]
    return CandidateCapabilityCatalog(
        catalog_id="selective-dependence-response-programme-forecast-candidate-catalog",
        registrations=study_forecast_candidate_registrations(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        ),
        templates=(
            StudyTemplate(
                template_key="selective-dependence-response.programme-forecast",
                template_version=SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION,
                protocol=protocol,
                graph=graph,
                coverage=ObligationCoverage(
                    coverage_id="coverage.selective-dependence-response.programme-forecast",
                    bindings=(
                        ObligationCoverageBinding(
                            obligation_id=step.obligation_ids[0],
                            proof_owner_node_id=step.step_id,
                            required_output_id=step.outputs[0].output_id,
                            contributor_edge_ids=tuple(
                                sorted(value.edge_id for value in graph.edges)
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseStudyForecastConfigAdapter:
    provider_key: str = "selective-dependence-response.programme-forecast-provider"
    provider_version: str = SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != SelectiveDependenceResponseStudyForecastRuntimeConfig.SCHEMA:
            raise ValueError("programme forecast config adapter schema differs")
        decode_study_forecast_runtime_config(payload)


def study_forecast_config_decoders() -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return (SelectiveDependenceResponseStudyForecastConfigAdapter(),)


class _IndependentSubstrateForecastRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = tuple(
            decode_study_forecast_runtime_config(port.read())
            for port in context.input_ports
            if port.payload_schema == SelectiveDependenceResponseStudyForecastRuntimeConfig.SCHEMA
        )
        completions = tuple(
            decode_canonical_bytes(
                port.read(),
                SelectiveDependenceResponseDevelopmentCompletionEnvelope,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == SelectiveDependenceResponseDevelopmentCompletionEnvelope.SCHEMA
        )
        questions = tuple(
            decode_canonical_bytes(
                port.read(), SelectiveDependenceResponseMethodQuestionFreeze, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA
        )
        if len(configs) != 1 or len(completions) != 2 or len(questions) != 1:
            raise ValueError("programme forecast task input roster differs")
        typed = _ordered_completions((completions[0], completions[1]))
        observed = tuple(
            sorted(
                (ObjectIdentity.from_record(value.envelope_id, value) for value in typed),
                key=lambda value: value.object_id,
            )
        )
        if observed != configs[0].development_completions:
            raise ValueError("programme forecast completions differ from frozen config")
        bundles = tuple(value.development_bundle for value in typed)
        result = qualify_study_forecasts((bundles[0], bundles[1]))
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("selective-dependence-response-method-question-bound", True, ()),
                ReceiptCheck(
                    "selective-dependence-response-programme-policy-roster-adjudicated",
                    True,
                    (),
                ),
                ReceiptCheck("selective-dependence-response-two-development-completions", True, ()),
            ),
        )


class SelectiveDependenceResponseStudyForecastRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: SelectiveDependenceResponseStudyForecastRuntimeConfig,
        completions: tuple[
            SelectiveDependenceResponseDevelopmentCompletionEnvelope,
            SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        ],
        method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    ) -> None:
        if config != study_forecast_runtime_config(completions):
            raise ValueError("programme forecast provider config differs")
        if registry != study_forecast_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        ):
            raise ValueError("programme forecast provider registry differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.config = config
        self.completions = _ordered_completions(completions)
        self.method_question = method_question
        self._runner = _IndependentSubstrateForecastRunner(registry.capabilities[0])

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("programme forecast provider registry/source differs")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("programme forecast provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            value.envelope_id: value for value in self.completions
        }
        records[self.method_question.freeze_id] = self.method_question
        for task in plan.tasks:
            records[task.capability.config.artifact_id] = self.config
        if set(records) != set(specs):
            raise ValueError("programme forecast external input roster differs")
        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            is_config = isinstance(record, SelectiveDependenceResponseStudyForecastRuntimeConfig)
            is_question = isinstance(record, SelectiveDependenceResponseMethodQuestionFreeze)
            visibility = (
                VisibilityCeiling.PROSPECTIVE
                if is_config or is_question
                else VisibilityCeiling.DEVELOPMENT_ONLY
            )
            access = (
                OutcomeAccess.OUTCOME_BLIND
                if is_config or is_question
                else OutcomeAccess.DEVELOPMENT_VISIBLE
            )
            if isinstance(record, SelectiveDependenceResponseStudyForecastRuntimeConfig):
                record_id = record.config_id
            elif isinstance(record, SelectiveDependenceResponseDevelopmentCompletionEnvelope):
                record_id = record.envelope_id
            elif isinstance(record, SelectiveDependenceResponseMethodQuestionFreeze):
                record_id = record.freeze_id
            else:
                raise TypeError("programme forecast external input type differs")
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
                    visibility_ceiling=spec.expected_visibility_ceiling or visibility,
                    outcome_access=spec.expected_outcome_access or access,
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
            raise ValueError("programme forecast semantic registry differs")
        return (
            CapabilityOutputSemanticContract.from_manifest(
                registry.capabilities[0],
                payload_schema=SelectiveDependenceResponseStudyForecastQualification.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(value.name for value in fields(SelectiveDependenceResponseStudyForecastQualification))
                ),
            ),
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("programme forecast adjudication registry differs")
        return None


__all__ = [
    'SelectiveDependenceResponseStudyForecastRuntimeConfig',
    'SelectiveDependenceResponseStudyForecastRuntimeProvider',
    'build_study_forecast_protocol',
    'decode_study_forecast_runtime_config',
    'study_forecast_candidate_catalog',
    'study_forecast_config_decoders',
    'study_forecast_registry',
    'study_forecast_runtime_config',
    'study_forecast_scientific_graph',
]
