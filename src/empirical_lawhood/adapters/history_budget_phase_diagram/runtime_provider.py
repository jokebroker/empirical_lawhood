"""Closed runtime-provider binding for all four history budget phase diagram phase DAGs."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetUnitScientificInput, require_history_budget_unit_inputs

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any, Mapping, TypeVar, cast

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.history_budget_phase_diagram.inference import synthesize_recurrence
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.adapters._decimal_custody import require_decimal_operands_preserved
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

from .authoring import ARRAY_PAYLOAD_SCHEMA, CONFIG_MEDIA_TYPE, DEVELOPMENT_REPORTER_KEY, HistoryBudgetPhaseDiagramExternalRecord, REPORTER_KEY, VERSION, INT_ARRAY_PAYLOAD_SCHEMA, history_budget_phase_diagram_phase_registry, protocol_template, scientific_graph
from .conformance import audit_source_firewall, merge_canary_reports, run_cross_implementation_conformance, run_conditioning_canary, run_discrete_rank_canary, run_generator_canary, run_observer_canary, run_resource_canary, run_structural_rank_canary, run_untouched_sampler_canary
from .contracts import HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramMethodFreeze, HistoryBudgetPhaseDiagramPhase, HistoryBudgetPhaseDiagramRecurrenceResult, HistoryBudgetPhaseDiagramScientificState, HistoryBudgetPhaseDiagramSeedRosterCommitment, HistoryBudgetPhaseDiagramTerminal, HistoryBudgetPhaseDiagramTerminalCloseout
from .descriptors import deterministic_development_seed
from .runtime_contracts import HistoryBudgetPhaseDiagramAdjudicationBundle, HistoryBudgetPhaseDiagramArrayManifest, HistoryBudgetPhaseDiagramBootstrapSummary, HistoryBudgetPhaseDiagramCanaryReport, HistoryBudgetPhaseDiagramDenominatorBundle, HistoryBudgetPhaseDiagramDevelopmentGate, HistoryBudgetPhaseDiagramDevelopmentLedger, HistoryBudgetPhaseDiagramEvaluationDesignFreeze, HistoryBudgetPhaseDiagramGeneratorBundle, HistoryBudgetPhaseDiagramHistoryBundle, HistoryBudgetPhaseDiagramNominationFreeze, HistoryBudgetPhaseDiagramObserverBundle, HistoryBudgetPhaseDiagramPhaseCloseout, HistoryBudgetPhaseDiagramRequestedUnitLedger, HistoryBudgetPhaseDiagramSeedRoster, HistoryBudgetPhaseDiagramUntouchedBundle
from .workflow import adjudication_bundle, create_seed_roster, denominator_bundle, development_ledger, evaluate_development_gate, freeze_evaluation_design, freeze_method, freeze_nominations, generator_bundle, history_bundle, receipt_closure_sha256, requested_unit_ledger, targeter_bundle, untouched_bundle, validate_seed_roster_commitment


HISTORY_BUDGET_PHASE_DIAGRAM_SCIENCE_SOURCE_CLOSURE_PATHS = (
    'src/empirical_lawhood/adapters/_decimal_custody.py',
    'src/empirical_lawhood/adapters/history_budget_fixed_scientific_inputs.py',
    'src/empirical_lawhood/adapters/history_budget_scientific_inputs.py',
    'src/empirical_lawhood/adapters/history_budget_seed_constants.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/array_io.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/authoring.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/conformance.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/contracts.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/descriptors.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/study_authoring.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/runtime_contracts.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/runtime_provider.py',
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/workflow.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/alignment.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/conditioning.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/discrete_rank.py',
    "src/empirical_lawhood/adapters/methods/_arb.py",
    "src/empirical_lawhood/adapters/methods/certified_rank.py",
    "src/empirical_lawhood/kernel/serialization.py",
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/evaluator.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/history.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/inference.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/preparation_sampler.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/structural_rank.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/targeting.py',
    'src/empirical_lawhood/adapters/simulators/rc_ladder_history_budget/generator.py',
)

HISTORY_BUDGET_PHASE_DIAGRAM_INTEGRATION_SOURCE_CLOSURE_PATHS = (
    "pyproject.toml",
    'src/empirical_lawhood/adapters/history_budget_phase_diagram/composition.py',
    "src/empirical_lawhood/api/composition.py",
    "src/empirical_lawhood/cli/app.py",
    "src/empirical_lawhood/cli/platform.py",
    'src/empirical_lawhood/infrastructure/study_issue.py',
    "src/empirical_lawhood/infrastructure/source_origin.py",
    "uv.lock",
)

HISTORY_BUDGET_PHASE_DIAGRAM_ALL_SOURCE_CLOSURE_PATHS = tuple(
    sorted(
        {
            *HISTORY_BUDGET_PHASE_DIAGRAM_SCIENCE_SOURCE_CLOSURE_PATHS,
            *HISTORY_BUDGET_PHASE_DIAGRAM_INTEGRATION_SOURCE_CLOSURE_PATHS,
        }
    )
)


def _closure_digest(source_files: Mapping[str, bytes], paths: tuple[str, ...]) -> str:
    if any(path not in source_files for path in paths):
        raise ValueError("history budget phase diagram source closure lacks a required member")
    return sha256(
        canonical_json_bytes(
            tuple((path, sha256(source_files[path]).hexdigest()) for path in paths)
        )
    ).hexdigest()


def science_source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint only the exact claim-bearing science-worker roster."""

    if tuple(sorted(source_files)) != HISTORY_BUDGET_PHASE_DIAGRAM_ALL_SOURCE_CLOSURE_PATHS:
        raise ValueError("history budget phase diagram implementation source roster differs")
    return _closure_digest(source_files, HISTORY_BUDGET_PHASE_DIAGRAM_SCIENCE_SOURCE_CLOSURE_PATHS)


def integration_source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint the exact outer composition and operator-tooling roster."""

    if tuple(sorted(source_files)) != HISTORY_BUDGET_PHASE_DIAGRAM_ALL_SOURCE_CLOSURE_PATHS:
        raise ValueError("history budget phase diagram implementation source roster differs")
    return _closure_digest(source_files, HISTORY_BUDGET_PHASE_DIAGRAM_INTEGRATION_SOURCE_CLOSURE_PATHS)


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramImplementationClosures:
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
) -> HistoryBudgetPhaseDiagramImplementationClosures:
    science = science_source_closure_sha256(source_files)
    integration = integration_source_closure_sha256(source_files)
    return HistoryBudgetPhaseDiagramImplementationClosures(
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
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/conditioning.py',
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/discrete_rank.py',
                "src/empirical_lawhood/adapters/methods/_arb.py",
                "src/empirical_lawhood/adapters/methods/certified_rank.py",
                "src/empirical_lawhood/kernel/serialization.py",
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/history.py',
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/preparation_sampler.py',
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/structural_rank.py',
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/targeting.py',
            ),
        ),
        generator_sha256=_closure_digest(
            source_files,
            ('src/empirical_lawhood/adapters/simulators/rc_ladder_history_budget/generator.py',),
        ),
        evaluator_sha256=_closure_digest(
            source_files,
            (
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/evaluator.py',
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/alignment.py',
                'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/inference.py',
            ),
        ),
    )


_RECORD_TYPES: dict[str, type[CanonicalRecord]] = {
    HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA: HistoryBudgetPhaseDiagramAdjudicationBundle,
    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA: HistoryBudgetPhaseDiagramArrayManifest,
    HistoryBudgetPhaseDiagramBootstrapSummary.SCHEMA: HistoryBudgetPhaseDiagramBootstrapSummary,
    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA: HistoryBudgetPhaseDiagramCanaryReport,
    HistoryBudgetPhaseDiagramConfig.SCHEMA: HistoryBudgetPhaseDiagramConfig,
    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA: HistoryBudgetPhaseDiagramDenominatorBundle,
    HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA: HistoryBudgetPhaseDiagramDevelopmentGate,
    HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA: HistoryBudgetPhaseDiagramDevelopmentLedger,
    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA: HistoryBudgetPhaseDiagramEvaluationDesignFreeze,
    HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA: HistoryBudgetPhaseDiagramGeneratorBundle,
    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA: HistoryBudgetPhaseDiagramHistoryBundle,
    HistoryBudgetPhaseDiagramMethodFreeze.SCHEMA: HistoryBudgetPhaseDiagramMethodFreeze,
    HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA: HistoryBudgetPhaseDiagramNominationFreeze,
    HistoryBudgetPhaseDiagramObserverBundle.SCHEMA: HistoryBudgetPhaseDiagramObserverBundle,
    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA: HistoryBudgetPhaseDiagramPhaseCloseout,
    HistoryBudgetPhaseDiagramRecurrenceResult.SCHEMA: HistoryBudgetPhaseDiagramRecurrenceResult,
    HistoryBudgetPhaseDiagramRequestedUnitLedger.SCHEMA: HistoryBudgetPhaseDiagramRequestedUnitLedger,
    HistoryBudgetPhaseDiagramSeedRoster.SCHEMA: HistoryBudgetPhaseDiagramSeedRoster,
    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA: HistoryBudgetPhaseDiagramSeedRosterCommitment,
    HistoryBudgetPhaseDiagramTerminalCloseout.SCHEMA: HistoryBudgetPhaseDiagramTerminalCloseout,
    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA: HistoryBudgetPhaseDiagramUntouchedBundle,
    ScientificAdjudicationRecord.SCHEMA: ScientificAdjudicationRecord,
}

_OUTPUT_STREAM_CHUNK_BYTES = 1024 * 1024


def output_semantic_contracts_from_execution_plan(
    execution_plan: ProtocolExecutionPlan,
) -> tuple[CapabilityOutputSemanticContract, ...]:
    """Reconstruct exact output semantics from one immutable historical plan."""

    values: dict[
        tuple[str, str, str, ArtifactProfile],
        CapabilityOutputSemanticContract,
    ] = {}
    for task in execution_plan.tasks:
        for output in task.outputs:
            if output.payload_schema not in task.capability.required_output_schema_ids:
                raise ValueError("history budget phase diagram plan output is absent from its capability contract")
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
                raise ValueError("history budget phase diagram historical output semantics conflict")
    return tuple(values[key] for key in sorted(values))


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], kind: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) != 1:
        raise ValueError(f"history budget phase diagram task requires exactly one {kind.__name__}")
    return values[0]


def _reports(records: tuple[CanonicalRecord, ...]) -> dict[str, HistoryBudgetPhaseDiagramCanaryReport]:
    values = {value.report_id: value for value in records if isinstance(value, HistoryBudgetPhaseDiagramCanaryReport)}
    if len(values) != sum(isinstance(value, HistoryBudgetPhaseDiagramCanaryReport) for value in records):
        raise ValueError("history budget phase diagram canary report IDs repeat")
    return values


def _closeout(
    *,
    closeout_id: str,
    phase: HistoryBudgetPhaseDiagramPhase,
    records: tuple[CanonicalRecord, ...],
    extra_sha256s: tuple[str, ...] = (),
    passed: bool = True,
    reason_codes: tuple[str, ...] = (),
) -> HistoryBudgetPhaseDiagramPhaseCloseout:
    return HistoryBudgetPhaseDiagramPhaseCloseout(
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
        raise ValueError("history budget phase diagram scientific adjudication context is absent")
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
    result: HistoryBudgetPhaseDiagramRecurrenceResult,
) -> tuple[AdjudicationEvaluability, ScientificStatus, tuple[str, ...]]:
    brackets = (
        result.fixed_depth_dynamical_bracket,
        result.fixed_depth_decision_bracket,
        result.normalized_budget_dynamical_bracket,
        result.normalized_budget_decision_bracket,
    )
    if all(value is HistoryBudgetPhaseDiagramScientificState.SUPPORTED for value in brackets):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.SUPPORTED,
            ("HISTORY_BUDGET_PHASE_DIAGRAM_ALL_PREDECLARED_T_BRACKETS_SUPPORTED",),
        )
    if all(value is HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE for value in brackets):
        return (
            AdjudicationEvaluability.UNEVALUABLE,
            ScientificStatus.UNEVALUABLE,
            ("HISTORY_BUDGET_PHASE_DIAGRAM_ALL_PREDECLARED_T_BRACKETS_UNEVALUABLE",),
        )
    if any(
        value in {HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE, HistoryBudgetPhaseDiagramScientificState.TARGETABILITY_LIMITED}
        for value in brackets
    ):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.PARTIAL,
            ("HISTORY_BUDGET_PHASE_DIAGRAM_T_BRACKET_TARGETABILITY_OR_INTEGRITY_LIMITED",),
        )
    return (
        AdjudicationEvaluability.EVALUABLE,
        ScientificStatus.MIXED,
        ("HISTORY_BUDGET_PHASE_DIAGRAM_NONCOMPENSATING_T_BRACKET_CONJUNCTION_FAILED",),
    )


@dataclass(frozen=True, slots=True)
class _TaskInputs:
    records: tuple[CanonicalRecord, ...]
    array_payloads: tuple[tuple[str, bytes], ...]


def _manifest(records: tuple[CanonicalRecord, ...], payload_schema: str) -> HistoryBudgetPhaseDiagramArrayManifest:
    values = tuple(
        value
        for value in records
        if isinstance(value, HistoryBudgetPhaseDiagramArrayManifest) and value.payload_schema == payload_schema
    )
    if len(values) != 1:
        raise ValueError(f"history budget phase diagram task requires one manifest for {payload_schema}")
    return values[0]


def _array_payload(inputs: _TaskInputs, payload_schema: str) -> bytes:
    values = tuple(payload for schema, payload in inputs.array_payloads if schema == payload_schema)
    if len(values) != 1:
        raise ValueError(f"history budget phase diagram task requires one payload for {payload_schema}")
    return values[0]


class HistoryBudgetPhaseDiagramRunner:
    """One static capability runner; task identity selects a closed operation."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: HistoryBudgetPhaseDiagramConfig,
        registry: CapabilityRegistry,
        source_files: Mapping[str, bytes],
        closures: HistoryBudgetPhaseDiagramImplementationClosures,
        injected_nomination_seeds: tuple[bytes, ...] | None,
        scientific_inputs: tuple[HistoryBudgetUnitScientificInput, ...],
    ) -> None:
        scientific_inputs = require_history_budget_unit_inputs(scientific_inputs, programme_ordinal=1, unit_ids=() if config.phase in (HistoryBudgetPhaseDiagramPhase.NOMINATION, HistoryBudgetPhaseDiagramPhase.CANARY) else config.unit_ids, scale_cells=config.scale_cells)
        self.manifest = manifest
        self.config = config
        self.scientific_inputs = scientific_inputs
        self.registry = registry
        self.source_files = dict(source_files)
        self.closures = closures
        self.injected_nomination_seeds = injected_nomination_seeds
        self.execution_count = 0
        self._seed_cache: tuple[HistoryBudgetPhaseDiagramSeedRoster, HistoryBudgetPhaseDiagramSeedRosterCommitment] | None = None

    def _read(self, context: TaskContext) -> _TaskInputs:
        records: list[CanonicalRecord] = []
        arrays: list[tuple[str, bytes]] = []
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("history budget phase diagram input size differs")
            if port.payload_schema in {ARRAY_PAYLOAD_SCHEMA, INT_ARRAY_PAYLOAD_SCHEMA}:
                arrays.append((port.payload_schema, payload))
                continue
            try:
                kind = _RECORD_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError("history budget phase diagram task received an unknown input schema") from error
            records.append(decode_canonical_bytes(payload, kind, maximum_bytes=port.size_bytes))
        phase_configs = tuple(value for value in records if isinstance(value, HistoryBudgetPhaseDiagramConfig))
        if (
            not any(value == self.config for value in phase_configs)
            or any(
                value != self.config
                and not (
                    self.config.phase is HistoryBudgetPhaseDiagramPhase.DEVELOPMENT
                    and value.phase is HistoryBudgetPhaseDiagramPhase.EVALUATION
                )
                for value in phase_configs
            )
            or context.config.content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("history budget phase diagram task config differs from its static provider")
        return _TaskInputs(tuple(records), tuple(arrays))

    def _evaluation_config(self, records: tuple[CanonicalRecord, ...]) -> HistoryBudgetPhaseDiagramConfig:
        values = tuple(
            value
            for value in records
            if isinstance(value, HistoryBudgetPhaseDiagramConfig) and value.phase is HistoryBudgetPhaseDiagramPhase.EVALUATION
        )
        if len(values) != 1:
            raise ValueError("history budget phase diagram task requires one evaluation config")
        return values[0]

    def _validate_design(self, design: HistoryBudgetPhaseDiagramEvaluationDesignFreeze) -> None:
        method = design.method_freeze
        if (
            self.config.phase is not HistoryBudgetPhaseDiagramPhase.EVALUATION
            or design.evaluation_config_sha256 != self.config.fingerprint()
            or design.implementation_source_closure_sha256 != self.closures.complete_sha256
            or method.observer_implementation_sha256 != self.closures.observer_sha256
            or method.generator_implementation_sha256 != self.closures.generator_sha256
            or method.evaluator_implementation_sha256 != self.closures.evaluator_sha256
        ):
            raise ValueError("history budget phase diagram runtime differs from the evaluation design freeze")

    def _descriptor_seed(
        self, task_id: str, records: tuple[CanonicalRecord, ...]
    ) -> tuple[str, bytes]:
        unit_id = task_id.split(".", 1)[1]
        if self.config.phase is HistoryBudgetPhaseDiagramPhase.DEVELOPMENT:
            return unit_id, deterministic_development_seed(unit_id)
        design = _one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
        roster = _one(records, HistoryBudgetPhaseDiagramSeedRoster)
        self._validate_design(design)
        validate_seed_roster_commitment(roster, design.seed_roster_commitment)
        entries = tuple(value for value in roster.entries if value.unit_id == unit_id)
        if len(entries) != 1:
            raise ValueError("history budget phase diagram evaluation descriptor seed binding differs")
        return unit_id, entries[0].seed_bytes

    def _dispatch(
        self, context: TaskContext, inputs: _TaskInputs
    ) -> dict[str, CanonicalRecord | bytes]:
        task_id = context.task_id
        records = inputs.records
        phase = self.config.phase
        if task_id == "nomination-validate-design":
            return {"requested-unit-ledger": requested_unit_ledger(self.config)}
        if task_id == "nomination-generate-seed-roster":
            ledger = _one(records, HistoryBudgetPhaseDiagramRequestedUnitLedger)
            if ledger != requested_unit_ledger(self.config):
                raise ValueError("history budget phase diagram seed roster ledger differs")
            if self._seed_cache is None:
                self._seed_cache = create_seed_roster(
                    self.config,
                    injected_seeds=self.injected_nomination_seeds,
                )
            roster, commitment = self._seed_cache
            return {"roster-commitment": commitment, "seed-roster": roster}
        if task_id == "nomination-seal-roster":
            roster = _one(records, HistoryBudgetPhaseDiagramSeedRoster)
            commitment = _one(records, HistoryBudgetPhaseDiagramSeedRosterCommitment)
            validate_seed_roster_commitment(roster, commitment)
            return {
                "roster-freeze-closeout": _closeout(
                    closeout_id="history-budget-phase-diagram.nomination-roster-freeze",
                    phase=phase,
                    records=records,
                )
            }
        if task_id == "nomination-closeout":
            return {
                "nomination-closeout": _closeout(
                    closeout_id="history-budget-phase-diagram.nomination-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("HISTORY_BUDGET_PHASE_DIAGRAM_NOMINATION_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id == "canary-config-schema-conformance":
            return {
                "contract-report": _closeout(
                    closeout_id="history-budget-phase-diagram.contract-conformance",
                    phase=phase,
                    records=records,
                    extra_sha256s=(self.registry.fingerprint(),),
                )
            }
        if task_id == "canary-dense-observer":
            return {"dense-observer-canary": run_observer_canary()}
        if task_id == "canary-sparse-generator":
            return {"sparse-generator-canary": run_generator_canary()}
        if task_id == "canary-structural-rank":
            return {"structural-rank-canary": run_structural_rank_canary()}
        if task_id == "canary-discrete-rank":
            return {"discrete-rank-canary": run_discrete_rank_canary()}
        if task_id == "canary-conditioning":
            return {"conditioning-canary": run_conditioning_canary()}
        if task_id == "canary-untouched-sampler":
            return {"untouched-sampler-canary": run_untouched_sampler_canary()}
        if task_id == "canary-generator-observer-firewall":
            reports = _reports(records)
            if len(reports) != 6 or any(not value.passed for value in reports.values()):
                raise ValueError("history budget phase diagram component canary roster did not pass")
            checks = audit_source_firewall(self.source_files)
            return {
                "firewall-report": _closeout(
                    closeout_id="history-budget-phase-diagram.generator-observer-firewall",
                    phase=phase,
                    records=records,
                    extra_sha256s=(
                        self.closures.observer_sha256,
                        self.closures.generator_sha256,
                        self.closures.evaluator_sha256,
                    ),
                    reason_codes=(),
                    passed=bool(checks),
                )
            }
        if task_id == "canary-cross-implementation-conformance":
            reports = _reports(records)
            independence = _one(records, HistoryBudgetPhaseDiagramPhaseCloseout)
            if not independence.passed:
                raise ValueError("history budget phase diagram independence audit did not pass")
            return {
                "numerical-conformance": run_cross_implementation_conformance(
                    implementation_sha256=self.closures.complete_sha256,
                    generator_report=reports["history-budget-phase-diagram.generator-canary"],
                    observer_report=reports["history-budget-phase-diagram.observer-canary"],
                    independence_check_ids=(
                        "generator-does-not-import-observer-or-physical-scale-methods",
                        "observer-does-not-import-generator-or-physical-scale-methods",
                        "scientific-intersection-is-contracts-only",
                    ),
                )
            }
        if task_id == "canary-excluded-resource":
            contract = _one(records, HistoryBudgetPhaseDiagramPhaseCloseout)
            numerical = tuple(
                value
                for value in records
                if isinstance(value, HistoryBudgetPhaseDiagramCanaryReport)
                and value.report_id == "history-budget-phase-diagram.numerical-conformance"
            )
            if not contract.passed or len(numerical) != 1 or not numerical[0].passed:
                raise ValueError("history budget phase diagram contract conformance did not pass")
            return {
                "resource-canary": run_resource_canary(
                    implementation_sha256=self.closures.complete_sha256
                )
            }
        if task_id == "c10-canary-qualification":
            reports = _reports(records)
            qualification = merge_canary_reports(
                source_check_ids=("exact-static-contract-conformance",),
                numerical=reports["history-budget-phase-diagram.numerical-conformance"],
                resource_report=reports["history-budget-phase-diagram.excluded-resource-canary"],
            )
            return {
                "canary-qualification": qualification,
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id.startswith(("development-descriptor.", "evaluation-descriptor.")):
            unit_id, seed = self._descriptor_seed(task_id, records)
            return {
                "denominator-bundle": denominator_bundle(
                        scientific_input=next(row for row in self.scientific_inputs if row.unit_id == unit_id),
                    config=self.config,
                    unit_id=unit_id,
                    seed=seed,
                )
            }
        if task_id.startswith(("development-history.", "evaluation-history.")):
            if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
                self._validate_design(_one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze))
            history_execution = history_bundle(
                config=self.config,
                denominators=_one(records, HistoryBudgetPhaseDiagramDenominatorBundle),
            )
            return {
                "history-array-manifest": history_execution.packed_arrays.manifest,
                "history-arrays": history_execution.packed_arrays.payload,
                "history-bundle": history_execution.bundle,
            }
        if task_id.startswith(("development-targeter.", "evaluation-targeter.")):
            if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
                self._validate_design(_one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze))
            targeter_execution = targeter_bundle(
                config=self.config,
                denominators=_one(records, HistoryBudgetPhaseDiagramDenominatorBundle),
                history=_one(records, HistoryBudgetPhaseDiagramHistoryBundle),
                implementation_sha256=self.closures.observer_sha256,
            )
            return {
                "targeter-array-manifest": targeter_execution.packed_arrays.manifest,
                "targeter-arrays": targeter_execution.packed_arrays.payload,
                "targeter-bundle": targeter_execution.bundle,
            }
        if task_id.startswith(("development-untouched.", "untouched-evaluation.")):
            if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
                self._validate_design(_one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze))
            execution = untouched_bundle(
                config=self.config,
                denominators=_one(records, HistoryBudgetPhaseDiagramDenominatorBundle),
            )
            return {
                "untouched-float-array-manifest": execution.float_arrays.manifest,
                "untouched-float-arrays": execution.float_arrays.payload,
                "untouched-int-array-manifest": execution.int_arrays.manifest,
                "untouched-int-arrays": execution.int_arrays.payload,
                "untouched-bundle": execution.bundle,
            }
        if task_id.startswith(("development-challenge-freeze.", "evaluation-challenge-freeze.")):
            design = (
                _one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
                if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION
                else None
            )
            if design is not None:
                self._validate_design(design)
            return {
                "nomination-freeze": freeze_nominations(
                    history=_one(records, HistoryBudgetPhaseDiagramHistoryBundle),
                    observer=_one(records, HistoryBudgetPhaseDiagramObserverBundle),
                    untouched=_one(records, HistoryBudgetPhaseDiagramUntouchedBundle),
                    method_freeze=None if design is None else design.method_freeze,
                )
            }
        if task_id.startswith(("development-generator.", "evaluation-generator.")):
            denominators = _one(records, HistoryBudgetPhaseDiagramDenominatorBundle)
            nomination_freeze = _one(records, HistoryBudgetPhaseDiagramNominationFreeze)
            if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
                design = _one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
                self._validate_design(design)
                if nomination_freeze.method_freeze_sha256 != design.method_freeze.fingerprint():
                    raise ValueError("history budget phase diagram generator nomination freeze differs")
            observer = nomination_freeze.observer_bundle
            untouched_record = _one(records, HistoryBudgetPhaseDiagramUntouchedBundle)
            untouched_float_payload = _array_payload(inputs, ARRAY_PAYLOAD_SCHEMA)
            generator_execution = generator_bundle(
                config=self.config,
                denominators=denominators,
                observer=observer,
                untouched=untouched_record,
                untouched_float_payload=untouched_float_payload,
                untouched_float_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                implementation_sha256=self.closures.generator_sha256,
                maximum_array_bytes=len(untouched_float_payload),
            )
            return {
                "generator-array-manifest": generator_execution.packed_arrays.manifest,
                "generator-arrays": generator_execution.packed_arrays.payload,
                "generator-bundle": generator_execution.bundle,
            }
        if task_id.startswith(("development-adjudicate.", "evaluation-adjudicate.")):
            if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
                design = _one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
                self._validate_design(design)
                nomination_freeze = _one(records, HistoryBudgetPhaseDiagramNominationFreeze)
                method: HistoryBudgetPhaseDiagramMethodFreeze | None = design.method_freeze
            else:
                nomination_freeze = _one(records, HistoryBudgetPhaseDiagramNominationFreeze)
                method = None
            observer = nomination_freeze.observer_bundle
            generator_payload = _array_payload(inputs, ARRAY_PAYLOAD_SCHEMA)
            untouched_int_payload = _array_payload(inputs, INT_ARRAY_PAYLOAD_SCHEMA)
            return {
                "adjudication-bundle": adjudication_bundle(
                    config=self.config,
                    denominators=_one(records, HistoryBudgetPhaseDiagramDenominatorBundle),
                    history=_one(records, HistoryBudgetPhaseDiagramHistoryBundle),
                    observer=observer,
                    untouched=_one(records, HistoryBudgetPhaseDiagramUntouchedBundle),
                    untouched_int_payload=untouched_int_payload,
                    untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                    generator=_one(records, HistoryBudgetPhaseDiagramGeneratorBundle),
                    generator_arrays_payload=generator_payload,
                    generator_arrays_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                    method_freeze=method,
                    # The payload is an input produced under the upstream
                    # generator's output ceiling, not this adjudicator's much
                    # smaller output ceiling.  The runtime has already opened
                    # and size-checked the typed port; retain the decoder's
                    # exact bound without spuriously rejecting valid arrays.
                    maximum_array_bytes=max(len(generator_payload), len(untouched_int_payload)),
                )
            }
        if task_id == "development-complete-ledger":
            bundles = tuple(
                sorted(
                    (value for value in records if isinstance(value, HistoryBudgetPhaseDiagramAdjudicationBundle)),
                    key=lambda value: value.unit_id,
                )
            )
            return {"development-ledger": development_ledger(adjudication_bundles=bundles)}
        if task_id == "development-correctness-power-resource-gate":
            reports = _reports(records)
            canaries = tuple(
                value
                for value in reports.values()
                if value.report_id == "history-budget-phase-diagram.source-canary-qualification"
            )
            if len(canaries) != 1:
                raise ValueError("history budget phase diagram development requires exact C0 qualification")
            return {
                "development-gate": evaluate_development_gate(
                    _one(records, HistoryBudgetPhaseDiagramDevelopmentLedger),
                    canaries[0],
                ),
            }
        if task_id == "f0-freeze-method":
            return {
                "method-freeze": freeze_method(
                    development_config=self.config,
                    development_gate=_one(records, HistoryBudgetPhaseDiagramDevelopmentGate),
                    observer_implementation_sha256=self.closures.observer_sha256,
                    generator_implementation_sha256=self.closures.generator_sha256,
                    evaluator_implementation_sha256=self.closures.evaluator_sha256,
                    development_receipt_closure_sha256=receipt_closure_sha256(
                        context.dependency_receipt_ids,
                        context.dependency_input_materialization_ids,
                    ),
                )
            }
        if task_id == "f1-freeze-evaluation-design":
            return {
                "evaluation-design-freeze": freeze_evaluation_design(
                    method_freeze=_one(records, HistoryBudgetPhaseDiagramMethodFreeze),
                    evaluation_config=self._evaluation_config(records),
                    seed_roster_commitment=_one(records, HistoryBudgetPhaseDiagramSeedRosterCommitment),
                    implementation_source_closure_sha256=self.closures.complete_sha256,
                )
            }
        if task_id == "f2-development-closeout":
            _one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
            return {
                "development-closeout": _closeout(
                    closeout_id="history-budget-phase-diagram.development-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("HISTORY_BUDGET_PHASE_DIAGRAM_DEVELOPMENT_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id == "recurrence-synthesize":
            design = _one(records, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
            self._validate_design(design)
            bundles = tuple(
                value for value in records if isinstance(value, HistoryBudgetPhaseDiagramAdjudicationBundle)
            )
            if len(bundles) != 90 or any(
                value.method_freeze_sha256 != design.method_freeze.fingerprint()
                for value in bundles
            ):
                raise ValueError("history budget phase diagram recurrence input bundle roster differs")
            recurrence_execution = synthesize_recurrence(
                config=self.config,
                method_freeze=design.method_freeze,
                adjudication_bundles=tuple(sorted(bundles, key=lambda value: value.unit_id)),
            )
            return {
                "bootstrap-summary": recurrence_execution.bootstrap_summary,
                "recurrence-result": recurrence_execution.result,
            }
        if task_id == "terminal-report":
            result = _one(records, HistoryBudgetPhaseDiagramRecurrenceResult)
            bootstrap = _one(records, HistoryBudgetPhaseDiagramBootstrapSummary)
            if result.bootstrap_summary_sha256 != bootstrap.fingerprint():
                raise ValueError("history budget phase diagram terminal bootstrap binding differs")
            evaluability, scientific_status, reason_codes = _recurrence_adjudication_state(result)
            return {
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=evaluability,
                    scientific_status=scientific_status,
                    reason_codes=reason_codes,
                ),
                "terminal-closeout": HistoryBudgetPhaseDiagramTerminalCloseout(
                    closeout_id="history-budget-phase-diagram.terminal-closeout",
                    terminal=HistoryBudgetPhaseDiagramTerminal.HISTORY_BUDGET_PHASE_DIAGRAM_EVALUATED,
                    recurrence_result_sha256=result.fingerprint(),
                    artifact_ids=tuple(
                        sorted(value.artifact_id for value in context.input_bindings)
                    ),
                    receipt_ids=context.dependency_receipt_ids,
                    limitation_ids=(
                        "finite-entered-simulator-population",
                        "no-continuum-or-physical-transport",
                        "no-controller-admission-prospective-controller-evaluation-or-controller-claim",
                    ),
                    evidence_ceiling=result.evidence_ceiling,
                    external_device_count=0,
                    physical_action_count=0,
                ),
            }
        raise ValueError("unknown history budget phase diagram task identity")

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
            raise ValueError("history budget phase diagram runner output roster differs from the plan")
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
        if sum(len(output.payload) for output in payloads) > context.resource_budget.output_bytes:
            raise ValueError("history budget phase diagram outputs exceed the output byte budget")
        for port, output in zip(context.output_ports, payloads, strict=True):
            original = outputs[port.output_id.removeprefix(f"{context.task_id}.")]
            if isinstance(original, HistoryBudgetPhaseDiagramHistoryBundle):
                require_decimal_operands_preserved(
                    original, output.payload, maximum_bytes=context.resource_budget.output_bytes,
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


class HistoryBudgetPhaseDiagramCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Bind one exact phase config, graph inputs and implementation closure."""

    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: HistoryBudgetPhaseDiagramConfig,
        external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...],
        source_files: Mapping[str, bytes],
        injected_nomination_seeds: tuple[bytes, ...] | None = None,
        scientific_inputs: tuple[HistoryBudgetUnitScientificInput, ...] | None = None,
    ) -> None:
        scientific_inputs = require_history_budget_unit_inputs(scientific_inputs, programme_ordinal=1, unit_ids=() if config.phase in (HistoryBudgetPhaseDiagramPhase.NOMINATION, HistoryBudgetPhaseDiagramPhase.CANARY) else config.unit_ids, scale_cells=config.scale_cells)
        closures = implementation_closures(source_files)
        expected = history_budget_phase_diagram_phase_registry(
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
            raise ValueError("history budget phase diagram runtime registry differs")
        scientific_graph(
            registry=registry,
            config=config,
            protocol=protocol_template(registry=registry, config=config),
            external_records=external_records,
        )
        if injected_nomination_seeds is not None and config.phase is not HistoryBudgetPhaseDiagramPhase.NOMINATION:
            raise ValueError("only the nomination provider accepts injected test seeds")
        self.registry = registry
        self.config = config
        self.scientific_inputs = scientific_inputs
        self.external_records = external_records
        self.source_files = dict(source_files)
        self.closures = closures
        self.injected_nomination_seeds = injected_nomination_seeds
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self._runners: tuple[HistoryBudgetPhaseDiagramRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("history budget phase diagram runner registry/source binding differs")
        self._runners = tuple(
            HistoryBudgetPhaseDiagramRunner(
                manifest=manifest,
                config=self.config,
                registry=self.registry,
                source_files=self.source_files,
                closures=self.closures,
                injected_nomination_seeds=self.injected_nomination_seeds,
                scientific_inputs=self.scientific_inputs,
            )
            for manifest in registry.capabilities
        )
        return self._runners

    def _external_records_by_artifact(self) -> dict[str, HistoryBudgetPhaseDiagramExternalRecord]:
        phase_record = HistoryBudgetPhaseDiagramExternalRecord(
            input_id=f"input.history-budget-phase-diagram.{self.config.phase.value.lower()}.config",
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
            raise ValueError("history budget phase diagram external-input plan/source binding differs")
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
                f"history budget phase diagram external input roster differs; unknown={unknown}; unused={unused}"
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
            raise ValueError("history budget phase diagram semantic registry differs")
        if execution_plan is not None:
            return output_semantic_contracts_from_execution_plan(execution_plan)
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
            raise ValueError("history budget phase diagram adjudication registry differs")
        capability_key, task_id = {
            HistoryBudgetPhaseDiagramPhase.NOMINATION: (
                DEVELOPMENT_REPORTER_KEY,
                "nomination-closeout",
            ),
            HistoryBudgetPhaseDiagramPhase.CANARY: (
                DEVELOPMENT_REPORTER_KEY,
                "c10-canary-qualification",
            ),
            HistoryBudgetPhaseDiagramPhase.DEVELOPMENT: (
                DEVELOPMENT_REPORTER_KEY,
                "f2-development-closeout",
            ),
            HistoryBudgetPhaseDiagramPhase.EVALUATION: (
                REPORTER_KEY,
                "terminal-report",
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
    "HISTORY_BUDGET_PHASE_DIAGRAM_ALL_SOURCE_CLOSURE_PATHS",
    "HISTORY_BUDGET_PHASE_DIAGRAM_INTEGRATION_SOURCE_CLOSURE_PATHS",
    "HISTORY_BUDGET_PHASE_DIAGRAM_SCIENCE_SOURCE_CLOSURE_PATHS",
    'HistoryBudgetPhaseDiagramCampaignRuntimeProvider',
    'HistoryBudgetPhaseDiagramImplementationClosures',
    'HistoryBudgetPhaseDiagramRunner',
    "implementation_closures",
    "output_semantic_contracts_from_execution_plan",
    "integration_source_closure_sha256",
    "science_source_closure_sha256",
]
