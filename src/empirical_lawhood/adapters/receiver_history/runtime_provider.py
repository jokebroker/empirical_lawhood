"""Closed runtime-provider binding for all five receiver-history phase DAGs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any, Mapping, TypeVar, cast

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.receiver_history_closure.inference import (
    build_development_metatheory_gate,
    build_empirical_metatheory_dossier,
    integrate_untouched_recurrence,
    synthesize_recurrence,
    targeted_continuation_gate,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_sha256,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    StreamingOutputEmitter,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.plans import ScientificInputRole
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .authoring import (
    ARRAY_PAYLOAD_SCHEMA,
    CONFIG_MEDIA_TYPE,
    DEVELOPMENT_REPORTER_KEY,
    ReceiverHistoryExternalRecord,
    REPORTER_KEY,
    VERSION,
    INT_ARRAY_PAYLOAD_SCHEMA,
    receiver_history_phase_registry,
    protocol_template,
    scientific_graph,
)
from .conformance import (
    audit_source_firewall,
    merge_canary_reports,
    merge_component_canaries,
    merge_concurrent_resource_reports,
    run_certificate_canary,
    run_cross_implementation_conformance,
    run_factorized_action_canary,
    run_generator_canary,
    run_lexical_direction_canary,
    run_observer_canary,
    run_one_pass_history_canary,
    run_resource_canary,
    run_transition_canary,
)
from .contracts import (
    ReceiverHistoryConfig,
    ReceiverHistoryDisorderFamily,
    ReceiverHistoryDevelopmentMetatheoryGate,
    ReceiverHistoryEmpiricalMetatheoryDossier,
    ReceiverHistoryEndpoint,
    ReceiverHistoryInferenceDispositionMatrix,
    ReceiverHistoryMethodFreeze,
    ReceiverHistoryPhase,
    ReceiverHistoryRecurrenceResult,
    ReceiverHistoryScientificState,
    ReceiverHistorySeedRosterCommitment,
    ReceiverHistoryTerminal,
    ReceiverHistoryTerminalCloseout,
    ReceiverHistoryTargetedContinuationGateRecord,
)
from .descriptors import deterministic_development_seed
from .endpoint_power import build_endpoint_power_atlas
from .runtime_contracts import (
    ReceiverHistoryAdjudicationBundle,
    ReceiverHistoryArrayManifest,
    ReceiverHistoryBootstrapSummary,
    ReceiverHistoryCanaryReport,
    ReceiverHistoryDenominatorBundle,
    ReceiverHistoryDevelopmentGate,
    ReceiverHistoryDevelopmentLedger,
    ReceiverHistoryEndpointCoordinatePowerAtlas,
    ReceiverHistoryEvaluationDesignFreeze,
    ReceiverHistoryGeneratorBundle,
    ReceiverHistoryHistoryBundle,
    ReceiverHistoryNominationFreeze,
    ReceiverHistoryObserverBundle,
    ReceiverHistoryPhaseCloseout,
    ReceiverHistoryRequestedUnitLedger,
    ReceiverHistorySeedRoster,
    ReceiverHistoryUntouchedBundle,
    ReceiverHistoryUntouchedFreeze,
)
from .workflow import (
    adjudication_bundle,
    target_adjudication_bundle,
    untouched_adjudication_bundle,
    create_seed_roster,
    denominator_bundle,
    development_ledger,
    evaluate_development_gate,
    freeze_evaluation_design,
    freeze_method,
    freeze_nominations,
    freeze_untouched,
    generator_bundle,
    target_generator_bundle,
    untouched_generator_bundle,
    history_bundle,
    receipt_closure_sha256,
    requested_unit_ledger,
    targeter_bundle,
    untouched_bundle,
    validate_seed_roster_commitment,
)


RECEIVER_HISTORY_SCIENCE_SOURCE_CLOSURE_PATHS = (
    "src/empirical_lawhood/adapters/history_budget_scientific_inputs.py",
    "src/empirical_lawhood/adapters/receiver_history/array_io.py",
    "src/empirical_lawhood/adapters/receiver_history/authoring.py",
    "src/empirical_lawhood/adapters/receiver_history/conformance.py",
    "src/empirical_lawhood/adapters/receiver_history/contracts.py",
    "src/empirical_lawhood/adapters/receiver_history/descriptors.py",
    "src/empirical_lawhood/adapters/receiver_history/endpoint_power.py",
    'src/empirical_lawhood/adapters/receiver_history/study_authoring.py',
    "src/empirical_lawhood/adapters/receiver_history/runtime_contracts.py",
    "src/empirical_lawhood/adapters/receiver_history/runtime_provider.py",
    "src/empirical_lawhood/adapters/receiver_history/workflow.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/alignment.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/conditioning.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/discrete_rank.py",
    "src/empirical_lawhood/adapters/methods/_arb.py",
    "src/empirical_lawhood/adapters/methods/certified_rank.py",
    "src/empirical_lawhood/kernel/serialization.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/evaluator.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/history.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/inference.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/lexical_targeting.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/preparation_sampler.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/structural_rank.py",
    "src/empirical_lawhood/adapters/simulators/rc_ladder/generator.py",
)

RECEIVER_HISTORY_INTEGRATION_SOURCE_CLOSURE_PATHS = (
    "pyproject.toml",
    "src/empirical_lawhood/adapters/receiver_history/composition.py",
    "src/empirical_lawhood/api/codecs.py",
    "src/empirical_lawhood/api/composition.py",
    "src/empirical_lawhood/api/execution.py",
    "src/empirical_lawhood/api/facade.py",
    'src/empirical_lawhood/infrastructure/study_issue.py',
    "src/empirical_lawhood/runtime/operator_profile.py",
    "uv.lock",
)

RECEIVER_HISTORY_ALL_SOURCE_CLOSURE_PATHS = tuple(
    sorted(
        {
            *RECEIVER_HISTORY_SCIENCE_SOURCE_CLOSURE_PATHS,
            *RECEIVER_HISTORY_INTEGRATION_SOURCE_CLOSURE_PATHS,
        }
    )
)


def _closure_digest(source_files: Mapping[str, bytes], paths: tuple[str, ...]) -> str:
    if any(path not in source_files for path in paths):
        raise ValueError("receiver-history source closure lacks a required member")
    return sha256(
        canonical_json_bytes(
            tuple((path, sha256(source_files[path]).hexdigest()) for path in paths)
        )
    ).hexdigest()


def science_source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint only the exact claim-bearing science-worker roster."""

    if tuple(sorted(source_files)) != RECEIVER_HISTORY_ALL_SOURCE_CLOSURE_PATHS:
        raise ValueError("receiver-history implementation source roster differs")
    return _closure_digest(source_files, RECEIVER_HISTORY_SCIENCE_SOURCE_CLOSURE_PATHS)


def integration_source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint the exact outer composition and operator-tooling roster."""

    if tuple(sorted(source_files)) != RECEIVER_HISTORY_ALL_SOURCE_CLOSURE_PATHS:
        raise ValueError("receiver-history implementation source roster differs")
    return _closure_digest(source_files, RECEIVER_HISTORY_INTEGRATION_SOURCE_CLOSURE_PATHS)


@dataclass(frozen=True, slots=True)
class ReceiverHistoryImplementationClosures:
    complete_sha256: str
    science_sha256: str
    integration_sha256: str
    observer_sha256: str
    generator_sha256: str
    evaluator_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "complete_sha256",
            "science_sha256",
            "integration_sha256",
            "observer_sha256",
            "generator_sha256",
            "evaluator_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)


def implementation_closures(
    source_files: Mapping[str, bytes],
) -> ReceiverHistoryImplementationClosures:
    science = science_source_closure_sha256(source_files)
    integration = integration_source_closure_sha256(source_files)
    return ReceiverHistoryImplementationClosures(
        complete_sha256=sha256(
            canonical_json_bytes(
                {
                    "integration_source_closure_sha256": integration,
                    "science_source_closure_sha256": science,
                }
            )
        ).hexdigest(),
        science_sha256=science,
        integration_sha256=integration,
        observer_sha256=_closure_digest(
            source_files,
            (
                "src/empirical_lawhood/adapters/history_budget_scientific_inputs.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/conditioning.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/discrete_rank.py",
                "src/empirical_lawhood/adapters/methods/_arb.py",
                "src/empirical_lawhood/adapters/methods/certified_rank.py",
                "src/empirical_lawhood/kernel/serialization.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/history.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/preparation_sampler.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/structural_rank.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/lexical_targeting.py",
            ),
        ),
        generator_sha256=_closure_digest(
            source_files,
            ("src/empirical_lawhood/adapters/simulators/rc_ladder/generator.py",),
        ),
        evaluator_sha256=_closure_digest(
            source_files,
            (
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/evaluator.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/alignment.py",
                "src/empirical_lawhood/adapters/methods/receiver_history_closure/inference.py",
            ),
        ),
    )


_RECORD_TYPES: dict[str, type[CanonicalRecord]] = {
    ReceiverHistoryAdjudicationBundle.SCHEMA: ReceiverHistoryAdjudicationBundle,
    ReceiverHistoryArrayManifest.SCHEMA: ReceiverHistoryArrayManifest,
    ReceiverHistoryBootstrapSummary.SCHEMA: ReceiverHistoryBootstrapSummary,
    ReceiverHistoryCanaryReport.SCHEMA: ReceiverHistoryCanaryReport,
    ReceiverHistoryConfig.SCHEMA: ReceiverHistoryConfig,
    ReceiverHistoryDenominatorBundle.SCHEMA: ReceiverHistoryDenominatorBundle,
    ReceiverHistoryDevelopmentGate.SCHEMA: ReceiverHistoryDevelopmentGate,
    ReceiverHistoryDevelopmentMetatheoryGate.SCHEMA: ReceiverHistoryDevelopmentMetatheoryGate,
    ReceiverHistoryDevelopmentLedger.SCHEMA: ReceiverHistoryDevelopmentLedger,
    ReceiverHistoryEndpointCoordinatePowerAtlas.SCHEMA: ReceiverHistoryEndpointCoordinatePowerAtlas,
    ReceiverHistoryEmpiricalMetatheoryDossier.SCHEMA: ReceiverHistoryEmpiricalMetatheoryDossier,
    ReceiverHistoryEvaluationDesignFreeze.SCHEMA: ReceiverHistoryEvaluationDesignFreeze,
    ReceiverHistoryGeneratorBundle.SCHEMA: ReceiverHistoryGeneratorBundle,
    ReceiverHistoryHistoryBundle.SCHEMA: ReceiverHistoryHistoryBundle,
    ReceiverHistoryInferenceDispositionMatrix.SCHEMA: ReceiverHistoryInferenceDispositionMatrix,
    ReceiverHistoryMethodFreeze.SCHEMA: ReceiverHistoryMethodFreeze,
    ReceiverHistoryNominationFreeze.SCHEMA: ReceiverHistoryNominationFreeze,
    ReceiverHistoryObserverBundle.SCHEMA: ReceiverHistoryObserverBundle,
    ReceiverHistoryPhaseCloseout.SCHEMA: ReceiverHistoryPhaseCloseout,
    ReceiverHistoryRecurrenceResult.SCHEMA: ReceiverHistoryRecurrenceResult,
    ReceiverHistoryRequestedUnitLedger.SCHEMA: ReceiverHistoryRequestedUnitLedger,
    ReceiverHistorySeedRoster.SCHEMA: ReceiverHistorySeedRoster,
    ReceiverHistorySeedRosterCommitment.SCHEMA: ReceiverHistorySeedRosterCommitment,
    ReceiverHistoryTerminalCloseout.SCHEMA: ReceiverHistoryTerminalCloseout,
    ReceiverHistoryTargetedContinuationGateRecord.SCHEMA: ReceiverHistoryTargetedContinuationGateRecord,
    ReceiverHistoryUntouchedBundle.SCHEMA: ReceiverHistoryUntouchedBundle,
    ReceiverHistoryUntouchedFreeze.SCHEMA: ReceiverHistoryUntouchedFreeze,
    ScientificAdjudicationRecord.SCHEMA: ScientificAdjudicationRecord,
}

_OUTPUT_STREAM_CHUNK_BYTES = 1024 * 1024


def output_semantic_contracts_from_execution_plan(
    execution_plan: ProtocolExecutionPlan,
    *,
    registry: CapabilityRegistry,
) -> tuple[CapabilityOutputSemanticContract, ...]:
    """Derive current output fields only from an exactly bound current plan."""

    if execution_plan.registry_sha256 != registry.fingerprint():
        raise ValueError("receiver-history current plan registry differs; current custody is required")
    for task in execution_plan.tasks:
        manifest = registry.require(task.capability)
        if task.capability_implementation_sha256 != manifest.implementation_sha256:
            raise ValueError("receiver-history current plan implementation differs; original history cannot supply current semantics")

    values: dict[
        tuple[str, str, str, ArtifactProfile],
        CapabilityOutputSemanticContract,
    ] = {}
    for task in execution_plan.tasks:
        for output in task.outputs:
            if output.payload_schema not in task.capability.required_output_schema_ids:
                raise ValueError(
                    "receiver-history plan output is absent from its capability contract"
                )
            contract = CapabilityOutputSemanticContract(
                capability_key=task.capability.capability_key,
                capability_version=task.capability.capability_version,
                capability_implementation_sha256=task.capability_implementation_sha256,
                payload_schema=output.payload_schema,
                profile=output.profile,
                top_level_keys=(
                    ()
                    if output.payload_schema in {ARRAY_PAYLOAD_SCHEMA, INT_ARRAY_PAYLOAD_SCHEMA}
                    else ("schema", "value", "version")
                ),
                value_keys=(
                    ()
                    if output.payload_schema in {ARRAY_PAYLOAD_SCHEMA, INT_ARRAY_PAYLOAD_SCHEMA}
                    else tuple(
                        sorted(
                            value.name
                            for value in fields(cast(Any, _RECORD_TYPES[output.payload_schema]))
                        )
                    )
                ),
            )
            existing = values.setdefault(contract.key, contract)
            if existing != contract:
                raise ValueError("receiver-history historical output semantics conflict")
    return tuple(values[key] for key in sorted(values))


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], kind: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) != 1:
        raise ValueError(f"receiver-history task requires exactly one {kind.__name__}")
    return values[0]


def _reports(records: tuple[CanonicalRecord, ...]) -> dict[str, ReceiverHistoryCanaryReport]:
    values = {
        value.report_id: value
        for value in records
        if isinstance(value, ReceiverHistoryCanaryReport)
    }
    if len(values) != sum(isinstance(value, ReceiverHistoryCanaryReport) for value in records):
        raise ValueError("receiver-history canary report IDs repeat")
    return values


def _closeout(
    *,
    closeout_id: str,
    phase: ReceiverHistoryPhase,
    records: tuple[CanonicalRecord, ...],
    extra_sha256s: tuple[str, ...] = (),
    passed: bool = True,
    reason_codes: tuple[str, ...] = (),
) -> ReceiverHistoryPhaseCloseout:
    return ReceiverHistoryPhaseCloseout(
        closeout_id=closeout_id,
        phase=phase,
        input_sha256s=tuple(sorted({*(value.fingerprint() for value in records), *extra_sha256s})),
        passed=passed,
        reason_codes=reason_codes,
        scientific_claim_assigned=False,
    )


def _scientific_adjudication(
    *,
    context: TaskContext,
    evaluability: AdjudicationEvaluability,
    scientific_status: ScientificStatus,
    reason_codes: tuple[str, ...],
) -> ScientificAdjudicationRecord:
    adjudication = context.scientific_adjudication_context
    if adjudication is None:
        raise ValueError("receiver-history scientific adjudication context is absent")
    output_ids = tuple(
        sorted(
            value.logical_artifact_id
            for value in context.output_ports
            if value.logical_artifact_id is not None
        )
    )
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id,
        adjudication_task_id=context.task_id,
        execution_plan=adjudication.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=output_ids,
        required_receipt_ids=context.dependency_receipt_ids,
        evidence_world_id=adjudication.evidence_world_id,
        evidence_world_kind=adjudication.evidence_world_kind,
        relation=adjudication.relation,
        independent_unit_id=adjudication.independent_unit_id,
        information_cutoffs=adjudication.information_cutoffs,
        visibility_ceiling=adjudication.visibility_ceiling,
        outcome_access=adjudication.outcome_access,
        evaluability=evaluability,
        scientific_status=scientific_status,
        admission_status=(
            AdmissionStatus.UNEVALUABLE
            if evaluability is AdjudicationEvaluability.UNEVALUABLE
            else AdmissionStatus.NOT_EVALUATED
        ),
        reason_codes=reason_codes,
    )


def _recurrence_adjudication_state(
    result: ReceiverHistoryRecurrenceResult,
) -> tuple[AdjudicationEvaluability, ScientificStatus, tuple[str, ...]]:
    brackets = (
        result.fixed_depth_dynamical_bracket,
        result.fixed_depth_target_bracket,
        result.fixed_depth_sink_bracket,
        result.normalized_budget_dynamical_bracket,
        result.normalized_budget_target_bracket,
        result.normalized_budget_sink_bracket,
    )
    if all(value is ReceiverHistoryScientificState.SUPPORTED for value in brackets):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.SUPPORTED,
            ('RECEIVER_HISTORY_ALL_PREDECLARED_TARGETED_BRACKETS_SUPPORTED',),
        )
    if all(value is ReceiverHistoryScientificState.UNEVALUABLE for value in brackets):
        return (
            AdjudicationEvaluability.UNEVALUABLE,
            ScientificStatus.UNEVALUABLE,
            ('RECEIVER_HISTORY_ALL_PREDECLARED_TARGETED_BRACKETS_UNEVALUABLE',),
        )
    if any(
        value
        in {
            ReceiverHistoryScientificState.UNEVALUABLE,
            ReceiverHistoryScientificState.TARGETABILITY_LIMITED,
        }
        for value in brackets
    ):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.PARTIAL,
            ('RECEIVER_HISTORY_TARGETED_BRACKET_TARGETABILITY_OR_INTEGRITY_LIMITED',),
        )
    return (
        AdjudicationEvaluability.EVALUABLE,
        ScientificStatus.MIXED,
        ('RECEIVER_HISTORY_NONCOMPENSATING_TARGETED_BRACKET_CONJUNCTION_FAILED',),
    )


@dataclass(frozen=True, slots=True)
class _TaskInputs:
    records: tuple[CanonicalRecord, ...]
    array_payloads: tuple[tuple[str, bytes], ...]


def _manifest(
    records: tuple[CanonicalRecord, ...], payload_schema: str
) -> ReceiverHistoryArrayManifest:
    values = tuple(
        value
        for value in records
        if isinstance(value, ReceiverHistoryArrayManifest)
        and value.payload_schema == payload_schema
    )
    if len(values) != 1:
        raise ValueError(f"receiver-history task requires one manifest for {payload_schema}")
    return values[0]


def _array_payload(inputs: _TaskInputs, payload_schema: str) -> bytes:
    values = tuple(payload for schema, payload in inputs.array_payloads if schema == payload_schema)
    if len(values) != 1:
        raise ValueError(f"receiver-history task requires one payload for {payload_schema}")
    return values[0]


class ReceiverHistoryRunner:
    """One static capability runner; task identity selects a closed operation."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: ReceiverHistoryConfig,
        registry: CapabilityRegistry,
        source_files: Mapping[str, bytes],
        closures: ReceiverHistoryImplementationClosures,
        injected_nomination_seeds: tuple[bytes, ...] | None,
    ) -> None:
        self.manifest = manifest
        self.config = config
        self.registry = registry
        self.source_files = dict(source_files)
        self.closures = closures
        self.injected_nomination_seeds = injected_nomination_seeds
        self.execution_count = 0
        self._seed_cache: (
            tuple[ReceiverHistorySeedRoster, ReceiverHistorySeedRosterCommitment] | None
        ) = None

    def _read(self, context: TaskContext) -> _TaskInputs:
        records: list[CanonicalRecord] = []
        arrays: list[tuple[str, bytes]] = []
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("receiver-history input size differs")
            if port.payload_schema in {ARRAY_PAYLOAD_SCHEMA, INT_ARRAY_PAYLOAD_SCHEMA}:
                arrays.append((port.payload_schema, payload))
                continue
            try:
                kind = _RECORD_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError(
                    "receiver-history task received an unknown input schema"
                ) from error
            records.append(decode_canonical_bytes(payload, kind, maximum_bytes=port.size_bytes))
        phase_configs = tuple(
            value for value in records if isinstance(value, ReceiverHistoryConfig)
        )
        if (
            not any(value == self.config for value in phase_configs)
            or any(
                value != self.config
                and not (
                    self.config.phase is ReceiverHistoryPhase.DEVELOPMENT
                    and value.phase is ReceiverHistoryPhase.TARGETED_EVALUATION
                )
                for value in phase_configs
            )
            or context.config.content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("receiver-history task config differs from its static provider")
        return _TaskInputs(tuple(records), tuple(arrays))

    def _evaluation_config(self, records: tuple[CanonicalRecord, ...]) -> ReceiverHistoryConfig:
        values = tuple(
            value
            for value in records
            if isinstance(value, ReceiverHistoryConfig)
            and value.phase is ReceiverHistoryPhase.TARGETED_EVALUATION
        )
        if len(values) != 1:
            raise ValueError("receiver-history task requires one frozen evaluation config")
        return values[0]

    def _validate_design(self, design: ReceiverHistoryEvaluationDesignFreeze) -> None:
        method = design.method_freeze
        if (
            self.config.phase is not ReceiverHistoryPhase.TARGETED_EVALUATION
            or design.evaluation_config_sha256 != self.config.fingerprint()
            or design.implementation_source_closure_sha256 != self.closures.complete_sha256
            or method.observer_implementation_sha256 != self.closures.observer_sha256
            or method.generator_implementation_sha256 != self.closures.generator_sha256
            or method.evaluator_implementation_sha256 != self.closures.evaluator_sha256
            or design.power_atlas.invalid_cell_count
        ):
            raise ValueError("receiver-history runtime differs from the evaluation design freeze")

    def _descriptor_seed(
        self, task_id: str, records: tuple[CanonicalRecord, ...]
    ) -> tuple[str, bytes]:
        unit_id = task_id.split(".", 1)[1]
        if self.config.phase is ReceiverHistoryPhase.DEVELOPMENT:
            return unit_id, deterministic_development_seed(unit_id)
        design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
        roster = _one(records, ReceiverHistorySeedRoster)
        self._validate_design(design)
        validate_seed_roster_commitment(roster, design.seed_roster_commitment)
        entries = tuple(value for value in roster.entries if value.unit_id == unit_id)
        if len(entries) != 1:
            raise ValueError("receiver-history evaluation descriptor seed binding differs")
        return unit_id, entries[0].seed_bytes

    def _dispatch(
        self, context: TaskContext, inputs: _TaskInputs
    ) -> dict[str, CanonicalRecord | bytes]:
        task_id = context.task_id
        records = inputs.records
        phase = self.config.phase
        if task_id == 'nomination-validate-design':
            return {"requested-unit-ledger": requested_unit_ledger(self.config)}
        if task_id == 'nomination-generate-seed-roster':
            ledger = _one(records, ReceiverHistoryRequestedUnitLedger)
            if ledger != requested_unit_ledger(self.config):
                raise ValueError("receiver-history seed roster ledger differs")
            if self._seed_cache is None:
                self._seed_cache = create_seed_roster(
                    self.config,
                    injected_seeds=self.injected_nomination_seeds,
                )
            roster, commitment = self._seed_cache
            return {"roster-commitment": commitment, "seed-roster": roster}
        if task_id == 'nomination-seal-roster':
            roster = _one(records, ReceiverHistorySeedRoster)
            commitment = _one(records, ReceiverHistorySeedRosterCommitment)
            validate_seed_roster_commitment(roster, commitment)
            return {
                "roster-freeze-closeout": _closeout(
                    closeout_id="receiver-history.nomination-roster-freeze",
                    phase=phase,
                    records=records,
                )
            }
        if task_id == 'nomination-closeout':
            return {
                "nomination-closeout": _closeout(
                    closeout_id="receiver-history.nomination-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=('RECEIVER_HISTORY_NOMINATION_PHASE_PREREQUISITE_ONLY',),
                ),
            }
        if task_id == 'qualification-profile-conformance':
            return {
                "profile-conformance": _closeout(
                    closeout_id="receiver-history.profile-conformance",
                    phase=phase,
                    records=records,
                    extra_sha256s=(self.registry.fingerprint(),),
                )
            }
        if task_id == 'qualification-dense-equations':
            return {"dense-truth": run_observer_canary(self.config)}
        if task_id == 'qualification-independent-generator':
            return {"sparse-generator": run_generator_canary(self.config)}
        if task_id == 'qualification-endpoint-carriers':
            return {
                "endpoint-carriers": merge_component_canaries(
                    report_id="receiver-history.endpoint-carriers-canary",
                    reports=(
                        run_certificate_canary(self.config),
                        run_lexical_direction_canary(),
                    ),
                )
            }
        if task_id == 'qualification-history-actions':
            return {
                "history-actions": merge_component_canaries(
                    report_id="receiver-history.history-actions-canary",
                    reports=(run_one_pass_history_canary(),),
                )
            }
        if task_id == 'qualification-action-factorization':
            return {
                "factorization": merge_component_canaries(
                    report_id="receiver-history.factorization-canary",
                    reports=(run_factorized_action_canary(self.config),),
                )
            }
        if task_id == 'qualification-source-firewall':
            reports = _reports(records)
            if len(reports) != 5 or any(not value.passed for value in reports.values()):
                raise ValueError("receiver-history component canary roster did not pass")
            checks = audit_source_firewall(self.source_files)
            numerical = run_cross_implementation_conformance(
                implementation_sha256=self.closures.complete_sha256,
                generator_report=reports["receiver-history.generator-canary"],
                observer_report=reports["receiver-history.observer-canary"],
                independence_check_ids=checks,
                config=self.config,
            )
            return {
                "firewall": merge_component_canaries(
                    report_id="receiver-history.firewall-conformance",
                    reports=(
                        numerical,
                        reports["receiver-history.endpoint-carriers-canary"],
                        reports["receiver-history.history-actions-canary"],
                        reports["receiver-history.factorization-canary"],
                    ),
                )
            }
        if task_id == 'qualification-inference-controls':
            return {
                "inference-fixtures": merge_component_canaries(
                    report_id="receiver-history.inference-fixtures-canary",
                    reports=(run_transition_canary(),),
                )
            }
        resource_tasks = {
            'qualification-resource-dynamical-smooth': (
                0,
                ReceiverHistoryDisorderFamily.SMOOTH_PERIODIC_12,
                (ReceiverHistoryEndpoint.DYNAMICAL,),
            ),
            'qualification-resource-target-correlated': (
                1,
                ReceiverHistoryDisorderFamily.CORRELATED_FIELD_12,
                (ReceiverHistoryEndpoint.TARGET_DECISION,),
            ),
            'qualification-resource-sink-multiscale': (
                2,
                ReceiverHistoryDisorderFamily.MULTISCALE_WAVELET_12,
                (ReceiverHistoryEndpoint.SINK_DECISION,),
            ),
            'qualification-resource-all-smooth': (
                3,
                ReceiverHistoryDisorderFamily.SMOOTH_PERIODIC_12,
                tuple(sorted(ReceiverHistoryEndpoint, key=lambda value: value.value)),
            ),
            'qualification-resource-all-correlated': (
                4,
                ReceiverHistoryDisorderFamily.CORRELATED_FIELD_12,
                tuple(sorted(ReceiverHistoryEndpoint, key=lambda value: value.value)),
            ),
            'qualification-resource-all-multiscale': (
                5,
                ReceiverHistoryDisorderFamily.MULTISCALE_WAVELET_12,
                tuple(sorted(ReceiverHistoryEndpoint, key=lambda value: value.value)),
            ),
        }
        if task_id in resource_tasks:
            reports = _reports(records)
            firewall = reports.get("receiver-history.firewall-conformance")
            inference = reports.get("receiver-history.inference-fixtures-canary")
            if firewall is None or inference is None or not firewall.passed or not inference.passed:
                raise ValueError("receiver-history resource canary lacks component qualification")
            index, family, endpoint_mask = resource_tasks[task_id]
            return {
                f"resource-canary-{index:02d}": run_resource_canary(
                    canary_config=self.config,
                    implementation_sha256=self.closures.complete_sha256,
                    block_index=index,
                    disorder_family=family,
                    endpoint_mask=endpoint_mask,
                )
            }
        if task_id == 'qualification-resource-admission':
            resource_reports = tuple(
                sorted(
                    (
                        value
                        for value in records
                        if isinstance(value, ReceiverHistoryCanaryReport)
                        and value.report_id.startswith("receiver-history.resource-canary.block-")
                    ),
                    key=lambda value: value.report_id,
                )
            )
            return {
                "resource-admission": merge_concurrent_resource_reports(
                    resource_reports,
                    maximum_parallel_tasks=self.config.maximum_parallel_tasks,
                )
            }
        if task_id == 'qualification-closeout':
            reports = _reports(records)
            qualification = merge_canary_reports(
                source_check_ids=("exact-static-contract-conformance",),
                numerical=reports["receiver-history.firewall-conformance"],
                resource_report=reports["receiver-history.resource-admission"],
            )
            return {
                "canary-close": qualification,
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=('RECEIVER_HISTORY_CANARY_PHASE_PREREQUISITE_ONLY',),
                ),
            }
        if task_id.startswith(('development-denominator.', 'targeted-denominator.')):
            unit_id, seed = self._descriptor_seed(task_id, records)
            return {
                "denominator-bundle": denominator_bundle(
                    config=self.config,
                    unit_id=unit_id,
                    seed=seed,
                )
            }
        if task_id.startswith(('development-history.', 'targeted-history.')):
            if phase in {ReceiverHistoryPhase.TARGETED_EVALUATION, ReceiverHistoryPhase.UNTOUCHED_EVALUATION}:
                self._validate_design(_one(records, ReceiverHistoryEvaluationDesignFreeze))
            history_execution = history_bundle(
                config=self.config,
                denominators=_one(records, ReceiverHistoryDenominatorBundle),
            )
            return {
                "history-array-manifest": history_execution.packed_arrays.manifest,
                "history-arrays": history_execution.packed_arrays.payload,
                "history-bundle": history_execution.bundle,
            }
        if task_id.startswith(
            ('development-endpoint-geometry.', 'targeted-endpoint-geometry.')
        ):
            design = (
                _one(records, ReceiverHistoryEvaluationDesignFreeze)
                if phase is ReceiverHistoryPhase.TARGETED_EVALUATION
                else None
            )
            if design is not None:
                self._validate_design(design)
            targeter_execution = targeter_bundle(
                config=self.config,
                denominators=_one(records, ReceiverHistoryDenominatorBundle),
                history=_one(records, ReceiverHistoryHistoryBundle),
                implementation_sha256=self.closures.observer_sha256,
                power_atlas=None if design is None else design.power_atlas,
            )
            return {
                "targeter-array-manifest": targeter_execution.packed_arrays.manifest,
                "targeter-arrays": targeter_execution.packed_arrays.payload,
                "targeter-bundle": targeter_execution.bundle,
            }
        if task_id.startswith(('development-untouched-preparation.', 'untouched-preparation.')):
            if phase in {ReceiverHistoryPhase.TARGETED_EVALUATION, ReceiverHistoryPhase.UNTOUCHED_EVALUATION}:
                self._validate_design(_one(records, ReceiverHistoryEvaluationDesignFreeze))
            denominators = _one(records, ReceiverHistoryDenominatorBundle)
            if phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
                unit_id = task_id.split(".", 1)[1]
                design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
                roster = _one(records, ReceiverHistorySeedRoster)
                validate_seed_roster_commitment(roster, design.seed_roster_commitment)
                entries = tuple(value for value in roster.entries if value.unit_id == unit_id)
                if (
                    len(entries) != 1
                    or denominator_bundle(
                        config=self.config, unit_id=unit_id, seed=entries[0].seed_bytes
                    )
                    != denominators
                ):
                    raise ValueError('receiver-history untouched descriptor does not exactly reuse targeted denominator')
            execution = untouched_bundle(
                config=self.config,
                denominators=denominators,
            )
            outputs: dict[str, CanonicalRecord | bytes] = {
                "untouched-float-array-manifest": execution.float_arrays.manifest,
                "untouched-float-arrays": execution.float_arrays.payload,
                "untouched-int-array-manifest": execution.int_arrays.manifest,
                "untouched-int-arrays": execution.int_arrays.payload,
                "untouched-bundle": execution.bundle,
            }
            # untouched has no descriptor task and must republish the externally bound
            # targeted denominator denominator for its downstream generator and adjudicator.  development denominator
            # already has a dedicated descriptor output, so an extra output
            # here would violate the frozen development task port roster.
            if phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
                outputs["denominator-bundle"] = denominators
            return outputs
        if task_id.startswith(('development-challenge-freeze.', 'targeted-challenge-freeze.')):
            design = (
                _one(records, ReceiverHistoryEvaluationDesignFreeze)
                if phase is ReceiverHistoryPhase.TARGETED_EVALUATION
                else None
            )
            if design is not None:
                self._validate_design(design)
            return {
                "nomination-freeze": freeze_nominations(
                    history=_one(records, ReceiverHistoryHistoryBundle),
                    observer=_one(records, ReceiverHistoryObserverBundle),
                    untouched=(
                        None
                        if phase
                        in {ReceiverHistoryPhase.DEVELOPMENT, ReceiverHistoryPhase.TARGETED_EVALUATION}
                        else _one(records, ReceiverHistoryUntouchedBundle)
                    ),
                    method_freeze=None if design is None else design.method_freeze,
                )
            }
        if task_id.startswith('untouched-freeze.'):
            design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
            self._validate_design(design)
            return {
                "untouched-freeze": freeze_untouched(
                    untouched=_one(records, ReceiverHistoryUntouchedBundle),
                    method_freeze=design.method_freeze,
                )
            }
        if task_id.startswith(
            ('development-independent-generation-certificate.', 'targeted-independent-generation-certificate.', 'untouched-independent-generator.')
        ):
            denominators = _one(records, ReceiverHistoryDenominatorBundle)
            nomination_freeze = (
                None
                if phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION
                else _one(records, ReceiverHistoryNominationFreeze)
            )
            if phase is ReceiverHistoryPhase.TARGETED_EVALUATION:
                assert nomination_freeze is not None
                design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
                self._validate_design(design)
                if nomination_freeze.method_freeze_sha256 != design.method_freeze.fingerprint():
                    raise ValueError("receiver-history generator nomination freeze differs")
            if phase in {ReceiverHistoryPhase.DEVELOPMENT, ReceiverHistoryPhase.TARGETED_EVALUATION}:
                assert nomination_freeze is not None
                generator_execution = target_generator_bundle(
                    config=self.config,
                    denominators=denominators,
                    nomination_freeze=nomination_freeze,
                    implementation_sha256=self.closures.generator_sha256,
                )
            elif phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
                untouched_record = _one(records, ReceiverHistoryUntouchedBundle)
                untouched_freeze = _one(records, ReceiverHistoryUntouchedFreeze)
                if (
                    untouched_freeze.untouched_bundle_sha256 != untouched_record.fingerprint()
                    or untouched_freeze.method_freeze_sha256
                    != _one(
                        records, ReceiverHistoryEvaluationDesignFreeze
                    ).method_freeze.fingerprint()
                ):
                    raise ValueError("receiver-history untouched freeze identity differs")
                untouched_float_payload = _array_payload(inputs, ARRAY_PAYLOAD_SCHEMA)
                untouched_int_payload = _array_payload(inputs, INT_ARRAY_PAYLOAD_SCHEMA)
                generator_execution = untouched_generator_bundle(
                    config=self.config,
                    denominators=denominators,
                    untouched=untouched_record,
                    untouched_float_payload=untouched_float_payload,
                    untouched_float_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                    untouched_int_payload=untouched_int_payload,
                    untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                    implementation_sha256=self.closures.generator_sha256,
                    maximum_array_bytes=max(
                        len(untouched_float_payload), len(untouched_int_payload)
                    ),
                )
            else:
                assert nomination_freeze is not None
                untouched_record = _one(records, ReceiverHistoryUntouchedBundle)
                untouched_float_payload = _array_payload(inputs, ARRAY_PAYLOAD_SCHEMA)
                untouched_int_payload = _array_payload(inputs, INT_ARRAY_PAYLOAD_SCHEMA)
                generator_execution = generator_bundle(
                    config=self.config,
                    denominators=denominators,
                    nomination_freeze=nomination_freeze,
                    untouched=untouched_record,
                    untouched_float_payload=untouched_float_payload,
                    untouched_float_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                    untouched_int_payload=untouched_int_payload,
                    untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                    implementation_sha256=self.closures.generator_sha256,
                    maximum_array_bytes=max(
                        len(untouched_float_payload), len(untouched_int_payload)
                    ),
                )
            return {
                "generator-array-manifest": generator_execution.packed_arrays.manifest,
                "generator-arrays": generator_execution.packed_arrays.payload,
                "generator-bundle": generator_execution.bundle,
            }
        if task_id.startswith(('development-unit-adjudication.', 'targeted-unit-adjudication.', 'untouched-adjudication.')):
            generator_payload = _array_payload(inputs, ARRAY_PAYLOAD_SCHEMA)
            if phase in {ReceiverHistoryPhase.DEVELOPMENT, ReceiverHistoryPhase.TARGETED_EVALUATION}:
                design = (
                    _one(records, ReceiverHistoryEvaluationDesignFreeze)
                    if phase is ReceiverHistoryPhase.TARGETED_EVALUATION
                    else None
                )
                if design is not None:
                    self._validate_design(design)
                return {
                    "adjudication-bundle": target_adjudication_bundle(
                        config=self.config,
                        denominators=_one(records, ReceiverHistoryDenominatorBundle),
                        history=_one(records, ReceiverHistoryHistoryBundle),
                        observer=_one(records, ReceiverHistoryObserverBundle),
                        generator=_one(records, ReceiverHistoryGeneratorBundle),
                        generator_arrays_payload=generator_payload,
                        generator_arrays_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                        method_freeze=None if design is None else design.method_freeze,
                        power_atlas=None if design is None else design.power_atlas,
                        maximum_array_bytes=len(generator_payload),
                    )
                }
            untouched_int_payload = _array_payload(inputs, INT_ARRAY_PAYLOAD_SCHEMA)
            if phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
                design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
                self._validate_design(design)
                return {
                    "adjudication-bundle": untouched_adjudication_bundle(
                        config=self.config,
                        denominators=_one(records, ReceiverHistoryDenominatorBundle),
                        untouched=_one(records, ReceiverHistoryUntouchedBundle),
                        untouched_int_payload=untouched_int_payload,
                        untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                        generator=_one(records, ReceiverHistoryGeneratorBundle),
                        generator_arrays_payload=generator_payload,
                        generator_arrays_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                        method_freeze=design.method_freeze,
                        maximum_array_bytes=max(len(generator_payload), len(untouched_int_payload)),
                    )
                }
            observer = _one(records, ReceiverHistoryObserverBundle)
            return {
                "adjudication-bundle": adjudication_bundle(
                    config=self.config,
                    denominators=_one(records, ReceiverHistoryDenominatorBundle),
                    history=_one(records, ReceiverHistoryHistoryBundle),
                    observer=observer,
                    untouched=_one(records, ReceiverHistoryUntouchedBundle),
                    untouched_int_payload=untouched_int_payload,
                    untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                    generator=_one(records, ReceiverHistoryGeneratorBundle),
                    generator_arrays_payload=generator_payload,
                    generator_arrays_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                    method_freeze=None,
                    # The payload is an input produced under the upstream
                    # generator's output ceiling, not this adjudicator's much
                    # smaller output ceiling.  The runtime has already opened
                    # and size-checked the typed port; retain the decoder's
                    # exact bound without spuriously rejecting valid arrays.
                    maximum_array_bytes=max(len(generator_payload), len(untouched_int_payload)),
                )
            }
        if task_id == 'development-ledger':
            bundles = tuple(
                sorted(
                    (
                        value
                        for value in records
                        if isinstance(value, ReceiverHistoryAdjudicationBundle)
                    ),
                    key=lambda value: value.unit_id,
                )
            )
            return {"development-ledger": development_ledger(adjudication_bundles=bundles)}
        if task_id == 'development-integrity-resource-source-gate':
            reports = _reports(records)
            canaries = tuple(
                value
                for value in reports.values()
                if value.report_id == "receiver-history.source-canary-qualification"
            )
            if len(canaries) != 1:
                raise ValueError('receiver-history development requires exact numerical qualification qualification')
            return {
                "development-gate": evaluate_development_gate(
                    _one(records, ReceiverHistoryDevelopmentLedger),
                    canaries[0],
                ),
            }
        if task_id == 'development-power-atlas':
            return {
                "endpoint-coordinate-power-atlas": build_endpoint_power_atlas(
                    _one(records, ReceiverHistoryDevelopmentLedger)
                )
            }
        if task_id == 'development-method-freeze':
            return {
                "method-freeze": freeze_method(
                    development_config=self.config,
                    development_gate=_one(records, ReceiverHistoryDevelopmentGate),
                    observer_implementation_sha256=self.closures.observer_sha256,
                    generator_implementation_sha256=self.closures.generator_sha256,
                    evaluator_implementation_sha256=self.closures.evaluator_sha256,
                    development_receipt_closure_sha256=receipt_closure_sha256(
                        context.dependency_receipt_ids,
                        context.dependency_input_materialization_ids,
                    ),
                )
            }
        if task_id == 'development-design-freeze':
            evaluation_config = self._evaluation_config(records)
            return {
                "evaluation-design-freeze": freeze_evaluation_design(
                    method_freeze=_one(records, ReceiverHistoryMethodFreeze),
                    power_atlas=_one(records, ReceiverHistoryEndpointCoordinatePowerAtlas),
                    evaluation_config=evaluation_config,
                    seed_roster_commitment=_one(records, ReceiverHistorySeedRosterCommitment),
                    implementation_source_closure_sha256=self.closures.complete_sha256,
                )
            }
        if task_id == 'development-closeout':
            design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
            powered = bool(design.power_atlas.eligible_cell_ids)
            adjudication_context = context.scientific_adjudication_context
            if adjudication_context is None:
                raise ValueError("receiver-history development execution-plan identity is absent")
            return {
                "development-closeout": _closeout(
                    closeout_id="receiver-history.development-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.EVALUABLE,
                    scientific_status=(
                        ScientificStatus.PARTIAL if powered else ScientificStatus.NOT_SUPPORTED
                    ),
                    reason_codes=(
                        'RECEIVER_HISTORY_ENDPOINT_POWER_ATLAS_PARTIALLY_CAPABLE'
                        if powered
                        else 'RECEIVER_HISTORY_ENDPOINT_POWER_ATLAS_EMPTY',
                    ),
                ),
                "development-metatheory-gate": build_development_metatheory_gate(
                    power_atlas=design.power_atlas,
                    method_freeze_sha256=design.method_freeze.fingerprint(),
                    evaluation_design_sha256=design.fingerprint(),
                    source_closure_sha256=self.closures.complete_sha256,
                    capability_registry_sha256=self.registry.fingerprint(),
                    execution_plan_sha256=adjudication_context.execution_plan.object_fingerprint,
                ),
            }
        if task_id == 'phase-diagram-synthesis':
            design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
            self._validate_design(design)
            bundles = tuple(
                value for value in records if isinstance(value, ReceiverHistoryAdjudicationBundle)
            )
            if len(bundles) != len(self.config.unit_ids) or any(
                value.method_freeze_sha256 != design.method_freeze.fingerprint()
                for value in bundles
            ):
                raise ValueError("receiver-history recurrence input bundle roster differs")
            recurrence_execution = synthesize_recurrence(
                config=self.config,
                method_freeze=design.method_freeze,
                power_atlas=design.power_atlas,
                adjudication_bundles=tuple(sorted(bundles, key=lambda value: value.unit_id)),
                include_untouched=False,
            )
            return {
                "bootstrap-summary": recurrence_execution.bootstrap_summary,
                "inference-disposition-matrix": recurrence_execution.disposition_matrix,
                "recurrence-result": recurrence_execution.result,
            }
        if task_id == 'empirical-metatheory-closeout':
            result = _one(records, ReceiverHistoryRecurrenceResult)
            matrix = _one(records, ReceiverHistoryInferenceDispositionMatrix)
            bootstrap = _one(records, ReceiverHistoryBootstrapSummary)
            if (
                result.bootstrap_summary_sha256 != bootstrap.fingerprint()
                or matrix.recurrence_result_sha256 != result.fingerprint()
            ):
                raise ValueError("receiver-history terminal bootstrap binding differs")
            adjudication_context = context.scientific_adjudication_context
            if adjudication_context is None:
                raise ValueError("receiver-history terminal execution-plan identity is absent")
            dossier = build_empirical_metatheory_dossier(
                result=result,
                matrix=matrix,
                source_closure_sha256=self.closures.complete_sha256,
                capability_registry_sha256=self.registry.fingerprint(),
                execution_plan_sha256=adjudication_context.execution_plan.object_fingerprint,
            )
            evaluability, scientific_status, reason_codes = _recurrence_adjudication_state(result)
            return {
                "empirical-metatheory-dossier": dossier,
                "t-continuation-gate": targeted_continuation_gate(result),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=evaluability,
                    scientific_status=scientific_status,
                    reason_codes=reason_codes,
                ),
                "targeted-closeout": ReceiverHistoryTerminalCloseout(
                    closeout_id="receiver-history.targeted-closeout",
                    terminal=ReceiverHistoryTerminal.TARGETED_EVALUATION_COMPLETE,
                    recurrence_result_sha256=result.fingerprint(),
                    artifact_ids=tuple(
                        sorted(value.artifact_id for value in context.input_bindings)
                    ),
                    receipt_ids=context.dependency_receipt_ids,
                    limitation_ids=(
                        "finite-entered-simulator-population",
                        "no-continuum-or-physical-transport",
                        'no-admission-prospective-use-or-controller-claim',
                    ),
                    evidence_ceiling=result.evidence_ceiling,
                    external_device_count=0,
                    physical_action_count=0,
                ),
            }
        if task_id == "untouched-phase-diagram-synthesis":
            design = _one(records, ReceiverHistoryEvaluationDesignFreeze)
            self._validate_design(design)
            results = tuple(
                value for value in records if isinstance(value, ReceiverHistoryRecurrenceResult)
            )
            if len(results) != 1 or results[0].untouched_cells:
                raise ValueError('receiver-history untouched synthesis requires the bounded targeted result')
            bundles = tuple(
                sorted(
                    (
                        value
                        for value in records
                        if isinstance(value, ReceiverHistoryAdjudicationBundle)
                    ),
                    key=lambda value: value.unit_id,
                )
            )
            integrated = integrate_untouched_recurrence(
                config=self.config,
                targeted_result=results[0],
                adjudication_bundles=bundles,
            )
            return {"recurrence-result": integrated}
        if task_id == 'integrated-phase-diagram-closeout':
            result = _one(records, ReceiverHistoryRecurrenceResult)
            evaluability, scientific_status, reason_codes = _recurrence_adjudication_state(result)
            return {
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=evaluability,
                    scientific_status=scientific_status,
                    reason_codes=reason_codes,
                ),
                "terminal-closeout": ReceiverHistoryTerminalCloseout(
                    closeout_id="receiver-history.integrated-closeout",
                    terminal=ReceiverHistoryTerminal.INTEGRATED_EVALUATION_COMPLETE,
                    recurrence_result_sha256=result.fingerprint(),
                    artifact_ids=tuple(
                        sorted(value.artifact_id for value in context.input_bindings)
                    ),
                    receipt_ids=context.dependency_receipt_ids,
                    limitation_ids=(
                        "finite-entered-simulator-population",
                        "conditional-u-act",
                        "no-continuum-or-physical-transport",
                        'no-admission-prospective-use-or-controller-claim',
                    ),
                    evidence_ceiling=result.evidence_ceiling,
                    external_device_count=0,
                    physical_action_count=0,
                ),
            }
        raise ValueError("unknown receiver-history task identity")

    def _execute_payloads(
        self,
        context: TaskContext,
    ) -> tuple[tuple[TaskOutputPayload, ...], tuple[ReceiptCheck, ...]]:
        self.execution_count += 1
        inputs = self._read(context)
        with threadpool_limits(limits=context.resource_budget.cpu_cores):
            outputs = self._dispatch(context, inputs)
        expected_output_ids = {
            port.output_id.removeprefix(f"{context.task_id}.") for port in context.output_ports
        }
        if set(outputs) != expected_output_ids:
            raise ValueError("receiver-history runner output roster differs from the plan")
        payloads = tuple(
            TaskOutputPayload(
                output_id=port.output_id,
                payload=(
                    value
                    if isinstance(
                        value := outputs[port.output_id.removeprefix(f"{context.task_id}.")],
                        bytes,
                    )
                    else value.canonical_bytes()
                ),
            )
            for port in context.output_ports
        )
        checks = (
            ReceiptCheck("actual-computation-completed", True, ()),
            ReceiptCheck("exact-typed-inputs-consumed", True, ()),
            ReceiptCheck("no-physical-action-network-or-source-write", True, ()),
            ReceiptCheck("task-cpu-thread-ceiling-applied", True, ()),
        )
        return payloads, checks

    def execute(self, context: TaskContext) -> RunnerResult:
        payloads, checks = self._execute_payloads(context)
        return RunnerResult(outputs=payloads, checks=checks)

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        """Materialize bounded chunks so large NumPy outputs never cross whole."""

        payloads, checks = self._execute_payloads(context)
        for output in payloads:
            for offset in range(0, len(output.payload), _OUTPUT_STREAM_CHUNK_BYTES):
                emitter.write(
                    output.output_id,
                    output.payload[offset : offset + _OUTPUT_STREAM_CHUNK_BYTES],
                )
        return checks


class ReceiverHistoryCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Bind one exact phase config, graph inputs and implementation closure."""

    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: ReceiverHistoryConfig,
        external_records: tuple[ReceiverHistoryExternalRecord, ...],
        source_files: Mapping[str, bytes],
        injected_nomination_seeds: tuple[bytes, ...] | None = None,
    ) -> None:
        closures = implementation_closures(source_files)
        expected = receiver_history_phase_registry(
            implementation_sha256=closures.complete_sha256,
            config=config,
        )
        candidate_registry = CapabilityRegistry(
            registry_id=(
                "candidate-registry."
                f"{sha256(canonical_json_bytes(expected.capabilities)).hexdigest()[:24]}"
            ),
            capabilities=expected.capabilities,
        )
        if registry not in (expected, candidate_registry):
            raise ValueError("receiver-history runtime registry differs")
        scientific_graph(
            registry=registry,
            config=config,
            protocol=protocol_template(registry=registry, config=config),
            external_records=external_records,
        )
        if (
            injected_nomination_seeds is not None
            and config.phase is not ReceiverHistoryPhase.NOMINATION
        ):
            raise ValueError("only the nomination provider accepts injected test seeds")
        self.registry = registry
        self.config = config
        self.external_records = external_records
        self.source_files = dict(source_files)
        self.closures = closures
        self.injected_nomination_seeds = injected_nomination_seeds
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self._runners: tuple[ReceiverHistoryRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("receiver-history runner registry/source binding differs")
        self._runners = tuple(
            ReceiverHistoryRunner(
                manifest=manifest,
                config=self.config,
                registry=self.registry,
                source_files=self.source_files,
                closures=self.closures,
                injected_nomination_seeds=self.injected_nomination_seeds,
            )
            for manifest in registry.capabilities
        )
        return self._runners

    def _external_records_by_artifact(self) -> dict[str, ReceiverHistoryExternalRecord]:
        phase_record = ReceiverHistoryExternalRecord(
            input_id=f"input.receiver-history.{self.config.phase.value.lower()}.config",
            record=self.config,
            role=ScientificInputRole.MODEL,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
        )
        values = (phase_record, *self.external_records)
        records = {value.input_id: value for value in values}
        records[f"config-artifact.{self.config.config_id}"] = phase_record
        return records

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("receiver-history external-input plan/source binding differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records = self._external_records_by_artifact()
        if set(specs) != set(records):
            unknown = tuple(sorted(set(specs) - set(records)))
            unused = tuple(sorted(set(records) - set(specs)))
            raise ValueError(
                f"receiver-history external input roster differs; unknown={unknown}; unused={unused}"
            )
        values = []
        for artifact_id, wrapped in sorted(records.items()):
            spec = specs[artifact_id]
            identity = ObjectIdentity.from_record(wrapped.input_id, wrapped.record)
            parent = ArtifactLineageParent(
                identity=identity,
                visibility_ceiling=wrapped.visibility,
                outcome_access=wrapped.outcome_access,
            )
            parents = [parent]
            if spec.identity_scope_sha256 is not None:
                parents.append(
                    ArtifactLineageParent(
                        identity=ObjectIdentity(
                            object_id=spec.input_id,
                            object_schema='empirical-lawhood/runtime/external-input-scope',
                            object_version="1.0.0",
                            object_fingerprint=spec.identity_scope_sha256,
                        ),
                        visibility_ceiling=parent.visibility_ceiling,
                        outcome_access=parent.outcome_access,
                    )
                )
            lineage = tuple(sorted(parents, key=lineage_parent_sort_key))
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=wrapped.record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=CONFIG_MEDIA_TYPE,
                    payload=wrapped.record.canonical_bytes(),
                    visibility_ceiling=(spec.expected_visibility_ceiling or wrapped.visibility),
                    outcome_access=spec.expected_outcome_access or wrapped.outcome_access,
                    parent_visibility_ceilings=tuple(
                        sorted(
                            (value.visibility_ceiling for value in lineage),
                            key=lambda value: value.value,
                        )
                    ),
                    lineage_parents=lineage,
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
            raise ValueError("receiver-history semantic registry differs")
        if execution_plan is not None:
            return output_semantic_contracts_from_execution_plan(execution_plan, registry=registry)
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                if schema in {ARRAY_PAYLOAD_SCHEMA, INT_ARRAY_PAYLOAD_SCHEMA}:
                    values.append(
                        CapabilityOutputSemanticContract.from_manifest(
                            manifest,
                            payload_schema=schema,
                            profile=ArtifactProfile.NUMPY_NO_PICKLE,
                        )
                    )
                    continue
                record_type = _RECORD_TYPES[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(value.name for value in fields(cast(Any, record_type)))
                        ),
                    )
                )
        return tuple(sorted(values, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        del execution_plan
        if registry != self.registry:
            raise ValueError("receiver-history adjudication registry differs")
        capability_key, task_id = {
            ReceiverHistoryPhase.NOMINATION: (
                DEVELOPMENT_REPORTER_KEY,
                'nomination-closeout',
            ),
            ReceiverHistoryPhase.CANARY: (
                DEVELOPMENT_REPORTER_KEY,
                'qualification-closeout',
            ),
            ReceiverHistoryPhase.DEVELOPMENT: (
                DEVELOPMENT_REPORTER_KEY,
                'development-closeout',
            ),
            ReceiverHistoryPhase.TARGETED_EVALUATION: (
                REPORTER_KEY,
                'empirical-metatheory-closeout',
            ),
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: (
                REPORTER_KEY,
                'integrated-phase-diagram-closeout',
            ),
        }[self.config.phase]
        manifest = registry.resolve(capability_key, VERSION)
        return ScientificAdjudicationOutputContract(
            capability_key=manifest.capability_key,
            capability_version=manifest.capability_version,
            output_id=f"{task_id}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=128 * 1024,
        )


__all__ = [
    "RECEIVER_HISTORY_ALL_SOURCE_CLOSURE_PATHS",
    "RECEIVER_HISTORY_INTEGRATION_SOURCE_CLOSURE_PATHS",
    "RECEIVER_HISTORY_SCIENCE_SOURCE_CLOSURE_PATHS",
    "ReceiverHistoryCampaignRuntimeProvider",
    "ReceiverHistoryImplementationClosures",
    "ReceiverHistoryRunner",
    "implementation_closures",
    "output_semantic_contracts_from_execution_plan",
    "integration_source_closure_sha256",
    "science_source_closure_sha256",
]
