"""Closed runtime-provider binding for all five split cohort history budget phase DAGs."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetUnitScientificInput, require_history_budget_unit_inputs
from empirical_lawhood.adapters._decimal_custody import require_decimal_operands_preserved

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any, Mapping, TypeVar, cast

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.split_cohort_history_budget.inference import integrate_untouched_recurrence, synthesize_recurrence, targeted_continuation_gate
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

from .authoring import ARRAY_PAYLOAD_SCHEMA, CONFIG_MEDIA_TYPE, DEVELOPMENT_REPORTER_KEY, SplitCohortHistoryBudgetExternalRecord, REPORTER_KEY, VERSION, INT_ARRAY_PAYLOAD_SCHEMA, split_cohort_history_budget_phase_registry, protocol_template, scientific_graph
from .conformance import audit_source_firewall, merge_canary_reports, merge_component_canaries, merge_concurrent_resource_reports, run_certificate_canary, run_cross_implementation_conformance, run_factorized_action_canary, run_generator_canary, run_lexical_direction_canary, run_observer_canary, run_one_pass_history_canary, run_rank_separation_canary, run_resource_canary, run_transition_canary, run_untouched_sampler_canary
from .contracts import SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetMethodFreeze, SplitCohortHistoryBudgetPhase, SplitCohortHistoryBudgetRecurrenceResult, SplitCohortHistoryBudgetScientificState, SplitCohortHistoryBudgetSeedRosterCommitment, SplitCohortHistoryBudgetTerminal, SplitCohortHistoryBudgetTerminalCloseout, SplitCohortHistoryBudgetTContinuationGateRecord
from .descriptors import deterministic_development_seed
from .runtime_contracts import SplitCohortHistoryBudgetAdjudicationBundle, SplitCohortHistoryBudgetArrayManifest, SplitCohortHistoryBudgetBootstrapSummary, SplitCohortHistoryBudgetCanaryReport, SplitCohortHistoryBudgetDenominatorBundle, SplitCohortHistoryBudgetDevelopmentGate, SplitCohortHistoryBudgetDevelopmentLedger, SplitCohortHistoryBudgetEvaluationDesignFreeze, SplitCohortHistoryBudgetGeneratorBundle, SplitCohortHistoryBudgetHistoryBundle, SplitCohortHistoryBudgetNominationFreeze, SplitCohortHistoryBudgetObserverBundle, SplitCohortHistoryBudgetPhaseCloseout, SplitCohortHistoryBudgetRequestedUnitLedger, SplitCohortHistoryBudgetSeedRoster, SplitCohortHistoryBudgetUntouchedBundle, SplitCohortHistoryBudgetUntouchedFreeze
from .workflow import adjudication_bundle, target_adjudication_bundle, untouched_adjudication_bundle, create_seed_roster, denominator_bundle, development_ledger, evaluate_development_gate, freeze_evaluation_design, freeze_method, freeze_nominations, freeze_untouched, generator_bundle, target_generator_bundle, untouched_generator_bundle, history_bundle, receipt_closure_sha256, requested_unit_ledger, targeter_bundle, untouched_bundle, validate_seed_roster_commitment


SPLIT_COHORT_HISTORY_BUDGET_SCIENCE_SOURCE_CLOSURE_PATHS = (
    'src/empirical_lawhood/adapters/_decimal_custody.py',
    'src/empirical_lawhood/adapters/history_budget_fixed_scientific_inputs.py',
    'src/empirical_lawhood/adapters/history_budget_scientific_inputs.py',
    'src/empirical_lawhood/adapters/history_budget_seed_constants.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/array_io.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/authoring.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/conformance.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/contracts.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/descriptors.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/study_authoring.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/runtime_contracts.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/runtime_provider.py',
    'src/empirical_lawhood/adapters/split_cohort_history_budget/workflow.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/alignment.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/conditioning.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/discrete_rank.py',
    "src/empirical_lawhood/adapters/methods/_arb.py",
    "src/empirical_lawhood/adapters/methods/certified_rank.py",
    "src/empirical_lawhood/kernel/serialization.py",
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/evaluator.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/history.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/inference.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/preparation_sampler.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/structural_rank.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/targeting.py',
    'src/empirical_lawhood/adapters/simulators/rc_ladder_split_cohort_history/generator.py',
)

SPLIT_COHORT_HISTORY_BUDGET_INTEGRATION_SOURCE_CLOSURE_PATHS = (
    "pyproject.toml",
    'src/empirical_lawhood/adapters/split_cohort_history_budget/composition.py',
    "src/empirical_lawhood/api/composition.py",
    "src/empirical_lawhood/cli/app.py",
    "src/empirical_lawhood/cli/platform.py",
    'src/empirical_lawhood/infrastructure/study_issue.py',
    "src/empirical_lawhood/infrastructure/source_origin.py",
    "uv.lock",
)

SPLIT_COHORT_HISTORY_BUDGET_ALL_SOURCE_CLOSURE_PATHS = tuple(
    sorted(
        {
            *SPLIT_COHORT_HISTORY_BUDGET_SCIENCE_SOURCE_CLOSURE_PATHS,
            *SPLIT_COHORT_HISTORY_BUDGET_INTEGRATION_SOURCE_CLOSURE_PATHS,
        }
    )
)


def _closure_digest(source_files: Mapping[str, bytes], paths: tuple[str, ...]) -> str:
    if any(path not in source_files for path in paths):
        raise ValueError("split cohort history budget source closure lacks a required member")
    return sha256(
        canonical_json_bytes(
            tuple((path, sha256(source_files[path]).hexdigest()) for path in paths)
        )
    ).hexdigest()


def science_source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint only the exact claim-bearing science-worker roster."""

    if tuple(sorted(source_files)) != SPLIT_COHORT_HISTORY_BUDGET_ALL_SOURCE_CLOSURE_PATHS:
        raise ValueError("split cohort history budget implementation source roster differs")
    return _closure_digest(source_files, SPLIT_COHORT_HISTORY_BUDGET_SCIENCE_SOURCE_CLOSURE_PATHS)


def integration_source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint the exact outer composition and operator-tooling roster."""

    if tuple(sorted(source_files)) != SPLIT_COHORT_HISTORY_BUDGET_ALL_SOURCE_CLOSURE_PATHS:
        raise ValueError("split cohort history budget implementation source roster differs")
    return _closure_digest(source_files, SPLIT_COHORT_HISTORY_BUDGET_INTEGRATION_SOURCE_CLOSURE_PATHS)


@dataclass(frozen=True, slots=True)
class SplitCohortHistoryBudgetImplementationClosures:
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
) -> SplitCohortHistoryBudgetImplementationClosures:
    science = science_source_closure_sha256(source_files)
    integration = integration_source_closure_sha256(source_files)
    return SplitCohortHistoryBudgetImplementationClosures(
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
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/conditioning.py',
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/discrete_rank.py',
                "src/empirical_lawhood/adapters/methods/_arb.py",
                "src/empirical_lawhood/adapters/methods/certified_rank.py",
                "src/empirical_lawhood/kernel/serialization.py",
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/history.py',
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/preparation_sampler.py',
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/structural_rank.py',
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/targeting.py',
            ),
        ),
        generator_sha256=_closure_digest(
            source_files,
            ('src/empirical_lawhood/adapters/simulators/rc_ladder_split_cohort_history/generator.py',),
        ),
        evaluator_sha256=_closure_digest(
            source_files,
            (
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/evaluator.py',
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/alignment.py',
                'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/inference.py',
            ),
        ),
    )


_RECORD_TYPES: dict[str, type[CanonicalRecord]] = {
    SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA: SplitCohortHistoryBudgetAdjudicationBundle,
    SplitCohortHistoryBudgetArrayManifest.SCHEMA: SplitCohortHistoryBudgetArrayManifest,
    SplitCohortHistoryBudgetBootstrapSummary.SCHEMA: SplitCohortHistoryBudgetBootstrapSummary,
    SplitCohortHistoryBudgetCanaryReport.SCHEMA: SplitCohortHistoryBudgetCanaryReport,
    SplitCohortHistoryBudgetConfig.SCHEMA: SplitCohortHistoryBudgetConfig,
    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA: SplitCohortHistoryBudgetDenominatorBundle,
    SplitCohortHistoryBudgetDevelopmentGate.SCHEMA: SplitCohortHistoryBudgetDevelopmentGate,
    SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA: SplitCohortHistoryBudgetDevelopmentLedger,
    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA: SplitCohortHistoryBudgetEvaluationDesignFreeze,
    SplitCohortHistoryBudgetGeneratorBundle.SCHEMA: SplitCohortHistoryBudgetGeneratorBundle,
    SplitCohortHistoryBudgetHistoryBundle.SCHEMA: SplitCohortHistoryBudgetHistoryBundle,
    SplitCohortHistoryBudgetMethodFreeze.SCHEMA: SplitCohortHistoryBudgetMethodFreeze,
    SplitCohortHistoryBudgetNominationFreeze.SCHEMA: SplitCohortHistoryBudgetNominationFreeze,
    SplitCohortHistoryBudgetObserverBundle.SCHEMA: SplitCohortHistoryBudgetObserverBundle,
    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA: SplitCohortHistoryBudgetPhaseCloseout,
    SplitCohortHistoryBudgetRecurrenceResult.SCHEMA: SplitCohortHistoryBudgetRecurrenceResult,
    SplitCohortHistoryBudgetRequestedUnitLedger.SCHEMA: SplitCohortHistoryBudgetRequestedUnitLedger,
    SplitCohortHistoryBudgetSeedRoster.SCHEMA: SplitCohortHistoryBudgetSeedRoster,
    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA: SplitCohortHistoryBudgetSeedRosterCommitment,
    SplitCohortHistoryBudgetTerminalCloseout.SCHEMA: SplitCohortHistoryBudgetTerminalCloseout,
    SplitCohortHistoryBudgetTContinuationGateRecord.SCHEMA: SplitCohortHistoryBudgetTContinuationGateRecord,
    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA: SplitCohortHistoryBudgetUntouchedBundle,
    SplitCohortHistoryBudgetUntouchedFreeze.SCHEMA: SplitCohortHistoryBudgetUntouchedFreeze,
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
                raise ValueError("split cohort history budget plan output is absent from its capability contract")
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
                raise ValueError("split cohort history budget historical output semantics conflict")
    return tuple(values[key] for key in sorted(values))


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], kind: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) != 1:
        raise ValueError(f"split cohort history budget task requires exactly one {kind.__name__}")
    return values[0]


def _reports(records: tuple[CanonicalRecord, ...]) -> dict[str, SplitCohortHistoryBudgetCanaryReport]:
    values = {value.report_id: value for value in records if isinstance(value, SplitCohortHistoryBudgetCanaryReport)}
    if len(values) != sum(isinstance(value, SplitCohortHistoryBudgetCanaryReport) for value in records):
        raise ValueError("split cohort history budget canary report IDs repeat")
    return values


def _closeout(
    *,
    closeout_id: str,
    phase: SplitCohortHistoryBudgetPhase,
    records: tuple[CanonicalRecord, ...],
    extra_sha256s: tuple[str, ...] = (),
    passed: bool = True,
    reason_codes: tuple[str, ...] = (),
) -> SplitCohortHistoryBudgetPhaseCloseout:
    return SplitCohortHistoryBudgetPhaseCloseout(
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
        raise ValueError("split cohort history budget scientific adjudication context is absent")
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
    result: SplitCohortHistoryBudgetRecurrenceResult,
) -> tuple[AdjudicationEvaluability, ScientificStatus, tuple[str, ...]]:
    brackets = (
        result.fixed_depth_dynamical_bracket,
        result.fixed_depth_decision_bracket,
        result.normalized_budget_dynamical_bracket,
        result.normalized_budget_decision_bracket,
    )
    if all(value is SplitCohortHistoryBudgetScientificState.SUPPORTED for value in brackets):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.SUPPORTED,
            ("SPLIT_COHORT_HISTORY_BUDGET_ALL_PREDECLARED_T_BRACKETS_SUPPORTED",),
        )
    if all(value is SplitCohortHistoryBudgetScientificState.UNEVALUABLE for value in brackets):
        return (
            AdjudicationEvaluability.UNEVALUABLE,
            ScientificStatus.UNEVALUABLE,
            ("SPLIT_COHORT_HISTORY_BUDGET_ALL_PREDECLARED_T_BRACKETS_UNEVALUABLE",),
        )
    if any(
        value in {SplitCohortHistoryBudgetScientificState.UNEVALUABLE, SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED}
        for value in brackets
    ):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.PARTIAL,
            ("SPLIT_COHORT_HISTORY_BUDGET_T_BRACKET_TARGETABILITY_OR_INTEGRITY_LIMITED",),
        )
    return (
        AdjudicationEvaluability.EVALUABLE,
        ScientificStatus.MIXED,
        ("SPLIT_COHORT_HISTORY_BUDGET_NONCOMPENSATING_T_BRACKET_CONJUNCTION_FAILED",),
    )


@dataclass(frozen=True, slots=True)
class _TaskInputs:
    records: tuple[CanonicalRecord, ...]
    array_payloads: tuple[tuple[str, bytes], ...]


def _manifest(records: tuple[CanonicalRecord, ...], payload_schema: str) -> SplitCohortHistoryBudgetArrayManifest:
    values = tuple(
        value
        for value in records
        if isinstance(value, SplitCohortHistoryBudgetArrayManifest) and value.payload_schema == payload_schema
    )
    if len(values) != 1:
        raise ValueError(f"split cohort history budget task requires one manifest for {payload_schema}")
    return values[0]


def _array_payload(inputs: _TaskInputs, payload_schema: str) -> bytes:
    values = tuple(payload for schema, payload in inputs.array_payloads if schema == payload_schema)
    if len(values) != 1:
        raise ValueError(f"split cohort history budget task requires one payload for {payload_schema}")
    return values[0]


class SplitCohortHistoryBudgetRunner:
    """One static capability runner; task identity selects a closed operation."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: SplitCohortHistoryBudgetConfig,
        registry: CapabilityRegistry,
        source_files: Mapping[str, bytes],
        closures: SplitCohortHistoryBudgetImplementationClosures,
        injected_nomination_seeds: tuple[bytes, ...] | None,
        scientific_inputs: tuple[HistoryBudgetUnitScientificInput, ...],
    ) -> None:
        scientific_inputs = require_history_budget_unit_inputs(scientific_inputs, programme_ordinal=2, unit_ids=() if config.phase in (SplitCohortHistoryBudgetPhase.NOMINATION, SplitCohortHistoryBudgetPhase.CANARY) else config.unit_ids, scale_cells=config.scale_cells)
        self.manifest = manifest
        self.config = config
        self.scientific_inputs = scientific_inputs
        self.registry = registry
        self.source_files = dict(source_files)
        self.closures = closures
        self.injected_nomination_seeds = injected_nomination_seeds
        self.execution_count = 0
        self._seed_cache: tuple[SplitCohortHistoryBudgetSeedRoster, SplitCohortHistoryBudgetSeedRosterCommitment] | None = None

    def _read(self, context: TaskContext) -> _TaskInputs:
        records: list[CanonicalRecord] = []
        arrays: list[tuple[str, bytes]] = []
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("split cohort history budget input size differs")
            if port.payload_schema in {ARRAY_PAYLOAD_SCHEMA, INT_ARRAY_PAYLOAD_SCHEMA}:
                arrays.append((port.payload_schema, payload))
                continue
            try:
                kind = _RECORD_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError("split cohort history budget task received an unknown input schema") from error
            records.append(decode_canonical_bytes(payload, kind, maximum_bytes=port.size_bytes))
        phase_configs = tuple(value for value in records if isinstance(value, SplitCohortHistoryBudgetConfig))
        if (
            not any(value == self.config for value in phase_configs)
            or any(
                value != self.config
                and not (
                    self.config.phase is SplitCohortHistoryBudgetPhase.DEVELOPMENT
                    and value.phase in {SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION, SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION}
                )
                for value in phase_configs
            )
            or context.config.content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("split cohort history budget task config differs from its static provider")
        return _TaskInputs(tuple(records), tuple(arrays))

    def _evaluation_configs(
        self, records: tuple[CanonicalRecord, ...]
    ) -> tuple[SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetConfig]:
        values = tuple(
            value
            for value in records
            if isinstance(value, SplitCohortHistoryBudgetConfig)
            and value.phase in {SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION, SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION}
        )
        by_phase = {value.phase: value for value in values}
        if len(values) != 2 or set(by_phase) != {
            SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION,
            SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION,
        }:
            raise ValueError("split cohort history budget task requires paired T/U evaluation configs")
        return by_phase[SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION], by_phase[SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION]

    def _validate_design(self, design: SplitCohortHistoryBudgetEvaluationDesignFreeze) -> None:
        method = design.method_freeze
        if (
            self.config.phase not in {SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION, SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION}
            or (
                design.targeted_evaluation_config_sha256
                if self.config.phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION
                else design.untouched_evaluation_config_sha256
            )
            != self.config.fingerprint()
            or design.implementation_source_closure_sha256 != self.closures.complete_sha256
            or method.observer_implementation_sha256 != self.closures.observer_sha256
            or method.generator_implementation_sha256 != self.closures.generator_sha256
            or method.evaluator_implementation_sha256 != self.closures.evaluator_sha256
        ):
            raise ValueError("split cohort history budget runtime differs from the evaluation design freeze")

    def _descriptor_seed(
        self, task_id: str, records: tuple[CanonicalRecord, ...]
    ) -> tuple[str, bytes]:
        unit_id = task_id.split(".", 1)[1]
        if self.config.phase is SplitCohortHistoryBudgetPhase.DEVELOPMENT:
            return unit_id, deterministic_development_seed(unit_id)
        design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
        roster = _one(records, SplitCohortHistoryBudgetSeedRoster)
        self._validate_design(design)
        validate_seed_roster_commitment(roster, design.seed_roster_commitment)
        entries = tuple(value for value in roster.entries if value.unit_id == unit_id)
        if len(entries) != 1:
            raise ValueError("split cohort history budget evaluation descriptor seed binding differs")
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
            ledger = _one(records, SplitCohortHistoryBudgetRequestedUnitLedger)
            if ledger != requested_unit_ledger(self.config):
                raise ValueError("split cohort history budget seed roster ledger differs")
            if self._seed_cache is None:
                self._seed_cache = create_seed_roster(
                    self.config,
                    injected_seeds=self.injected_nomination_seeds,
                )
            roster, commitment = self._seed_cache
            return {"roster-commitment": commitment, "seed-roster": roster}
        if task_id == "nomination-seal-roster":
            roster = _one(records, SplitCohortHistoryBudgetSeedRoster)
            commitment = _one(records, SplitCohortHistoryBudgetSeedRosterCommitment)
            validate_seed_roster_commitment(roster, commitment)
            return {
                "roster-freeze-closeout": _closeout(
                    closeout_id="split-cohort-history-budget.nomination-roster-freeze",
                    phase=phase,
                    records=records,
                )
            }
        if task_id == "nomination-closeout":
            return {
                "nomination-closeout": _closeout(
                    closeout_id="split-cohort-history-budget.nomination-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("SPLIT_COHORT_HISTORY_BUDGET_NOMINATION_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id == "canary-config-schema-conformance":
            return {
                "contract-report": _closeout(
                    closeout_id="split-cohort-history-budget.contract-conformance",
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
            return {"certificate-canary": run_certificate_canary()}
        if task_id == "canary-discrete-rank":
            return {
                "direction-preparation-canary": merge_component_canaries(
                    report_id="split-cohort-history-budget.direction-preparation-canary",
                    reports=(run_lexical_direction_canary(), run_untouched_sampler_canary()),
                )
            }
        if task_id == "canary-conditioning":
            return {
                "rank-transition-canary": merge_component_canaries(
                    report_id="split-cohort-history-budget.rank-transition-canary",
                    reports=(run_rank_separation_canary(), run_transition_canary()),
                )
            }
        if task_id == "canary-untouched-sampler":
            return {
                "efficiency-equivalence-canary": merge_component_canaries(
                    report_id="split-cohort-history-budget.efficiency-equivalence-canary",
                    reports=(run_factorized_action_canary(), run_one_pass_history_canary()),
                )
            }
        if task_id == "canary-generator-observer-firewall":
            reports = _reports(records)
            if len(reports) != 6 or any(not value.passed for value in reports.values()):
                raise ValueError("split cohort history budget component canary roster did not pass")
            checks = audit_source_firewall(self.source_files)
            numerical = run_cross_implementation_conformance(
                implementation_sha256=self.closures.complete_sha256,
                generator_report=reports["split-cohort-history-budget.generator-canary"],
                observer_report=reports["split-cohort-history-budget.observer-canary"],
                independence_check_ids=checks,
            )
            return {
                "firewall-conformance": merge_component_canaries(
                    report_id="split-cohort-history-budget.firewall-component-conformance",
                    reports=(
                        numerical,
                        reports["split-cohort-history-budget.certificate-canary"],
                        reports["split-cohort-history-budget.direction-preparation-canary"],
                        reports["split-cohort-history-budget.rank-transition-canary"],
                        reports["split-cohort-history-budget.efficiency-equivalence-canary"],
                    ),
                )
            }
        resource_tasks = {
            "canary-cross-implementation-conformance": 0,
            "canary-excluded-resource": 1,
            "c10-excluded-resource-canary": 2,
            "c11-excluded-resource-canary": 3,
        }
        if task_id in resource_tasks:
            reports = _reports(records)
            firewall = reports.get("split-cohort-history-budget.firewall-component-conformance")
            if firewall is None or not firewall.passed:
                raise ValueError("split cohort history budget resource canary lacks component qualification")
            index = resource_tasks[task_id]
            return {
                f"resource-canary-{index:02d}": run_resource_canary(
                    implementation_sha256=self.closures.complete_sha256,
                    block_index=index,
                )
            }
        if task_id == "c12-concurrent-resource-qualification":
            reports = tuple(
                sorted(
                    (
                        value
                        for value in records
                        if isinstance(value, SplitCohortHistoryBudgetCanaryReport)
                        and value.report_id.startswith("split-cohort-history-budget.excluded-resource-canary.block-")
                    ),
                    key=lambda value: value.report_id,
                )
            )
            return {"resource-qualification": merge_concurrent_resource_reports(reports)}
        if task_id == "c13-canary-qualification":
            reports = _reports(records)
            qualification = merge_canary_reports(
                source_check_ids=("exact-static-contract-conformance",),
                numerical=reports["split-cohort-history-budget.firewall-component-conformance"],
                resource_report=reports["split-cohort-history-budget.concurrent-resource-qualification"],
            )
            return {
                "canary-qualification": qualification,
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("SPLIT_COHORT_HISTORY_BUDGET_CANARY_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id.startswith(("development-descriptor.", "targeted-evaluation-descriptor.")):
            unit_id, seed = self._descriptor_seed(task_id, records)
            return {
                "denominator-bundle": denominator_bundle(
                        scientific_input=next(row for row in self.scientific_inputs if row.unit_id == unit_id),
                    config=self.config,
                    unit_id=unit_id,
                    seed=seed,
                )
            }
        if task_id.startswith(("development-history.", "targeted-evaluation-history.")):
            if phase in {SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION, SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION}:
                self._validate_design(_one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze))
            history_execution = history_bundle(
                config=self.config,
                denominators=_one(records, SplitCohortHistoryBudgetDenominatorBundle),
            )
            return {
                "history-array-manifest": history_execution.packed_arrays.manifest,
                "history-arrays": history_execution.packed_arrays.payload,
                "history-bundle": history_execution.bundle,
            }
        if task_id.startswith(("development-targeter.", "targeted-evaluation-observer.")):
            if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION:
                self._validate_design(_one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze))
            targeter_execution = targeter_bundle(
                config=self.config,
                denominators=_one(records, SplitCohortHistoryBudgetDenominatorBundle),
                history=_one(records, SplitCohortHistoryBudgetHistoryBundle),
                implementation_sha256=self.closures.observer_sha256,
            )
            return {
                "targeter-array-manifest": targeter_execution.packed_arrays.manifest,
                "targeter-arrays": targeter_execution.packed_arrays.payload,
                "targeter-bundle": targeter_execution.bundle,
            }
        if task_id.startswith(("development-untouched.", "untouched-evaluation-preparation.")):
            if phase in {SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION, SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION}:
                self._validate_design(_one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze))
            denominators = _one(records, SplitCohortHistoryBudgetDenominatorBundle)
            if phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
                unit_id = task_id.split(".", 1)[1]
                design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
                roster = _one(records, SplitCohortHistoryBudgetSeedRoster)
                validate_seed_roster_commitment(roster, design.seed_roster_commitment)
                entries = tuple(value for value in roster.entries if value.unit_id == unit_id)
                if (
                    len(entries) != 1
                    or denominator_bundle(
                        scientific_input=next(row for row in self.scientific_inputs if row.unit_id == unit_id),
                        config=self.config, unit_id=unit_id, seed=entries[0].seed_bytes
                    )
                    != denominators
                ):
                    raise ValueError("split cohort history budget untouched-evaluation descriptor does not exactly reuse the development denominator")
            execution = untouched_bundle(
                config=self.config,
                denominators=denominators,
            )
            outputs: dict[str, object] = {
                "untouched-float-array-manifest": execution.float_arrays.manifest,
                "untouched-float-arrays": execution.float_arrays.payload,
                "untouched-int-array-manifest": execution.int_arrays.manifest,
                "untouched-int-arrays": execution.int_arrays.payload,
                "untouched-bundle": execution.bundle,
            }
            # U has no descriptor task and must republish the externally bound
            # untouched-evaluation denominator for its downstream generator and adjudicator.  development
            # already has a dedicated descriptor output, so an extra output
            # here would violate the frozen development task port roster.
            if phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
                outputs["denominator-bundle"] = denominators
            return outputs
        if task_id.startswith(("development-challenge-freeze.", "targeted-evaluation-challenge-freeze.")):
            design = (
                _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
                if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION
                else None
            )
            if design is not None:
                self._validate_design(design)
            return {
                "nomination-freeze": freeze_nominations(
                    history=_one(records, SplitCohortHistoryBudgetHistoryBundle),
                    observer=_one(records, SplitCohortHistoryBudgetObserverBundle),
                    untouched=(
                        None
                        if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION
                        else _one(records, SplitCohortHistoryBudgetUntouchedBundle)
                    ),
                    method_freeze=None if design is None else design.method_freeze,
                )
            }
        if task_id.startswith("untouched-evaluation-freeze."):
            design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
            self._validate_design(design)
            return {
                "untouched-freeze": freeze_untouched(
                    untouched=_one(records, SplitCohortHistoryBudgetUntouchedBundle),
                    method_freeze=design.method_freeze,
                )
            }
        if task_id.startswith(("development-generator.", "targeted-evaluation-generator.", "untouched-evaluation-generator.")):
            denominators = _one(records, SplitCohortHistoryBudgetDenominatorBundle)
            nomination_freeze = (
                None
                if phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION
                else _one(records, SplitCohortHistoryBudgetNominationFreeze)
            )
            if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION:
                assert nomination_freeze is not None
                design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
                self._validate_design(design)
                if nomination_freeze.method_freeze_sha256 != design.method_freeze.fingerprint():
                    raise ValueError("split cohort history budget generator nomination freeze differs")
            if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION:
                assert nomination_freeze is not None
                generator_execution = target_generator_bundle(
                    config=self.config,
                    denominators=denominators,
                    nomination_freeze=nomination_freeze,
                    implementation_sha256=self.closures.generator_sha256,
                )
            elif phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
                untouched_record = _one(records, SplitCohortHistoryBudgetUntouchedBundle)
                untouched_freeze = _one(records, SplitCohortHistoryBudgetUntouchedFreeze)
                if (
                    untouched_freeze.untouched_bundle_sha256 != untouched_record.fingerprint()
                    or untouched_freeze.method_freeze_sha256
                    != _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze).method_freeze.fingerprint()
                ):
                    raise ValueError("split cohort history budget untouched freeze identity differs")
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
                untouched_record = _one(records, SplitCohortHistoryBudgetUntouchedBundle)
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
        if task_id.startswith(("development-adjudicate.", "targeted-evaluation-adjudicate.", "untouched-evaluation-adjudicate.")):
            generator_payload = _array_payload(inputs, ARRAY_PAYLOAD_SCHEMA)
            if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION:
                design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
                self._validate_design(design)
                return {
                    "adjudication-bundle": target_adjudication_bundle(
                        config=self.config,
                        denominators=_one(records, SplitCohortHistoryBudgetDenominatorBundle),
                        history=_one(records, SplitCohortHistoryBudgetHistoryBundle),
                        observer=_one(records, SplitCohortHistoryBudgetObserverBundle),
                        generator=_one(records, SplitCohortHistoryBudgetGeneratorBundle),
                        generator_arrays_payload=generator_payload,
                        generator_arrays_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                        method_freeze=design.method_freeze,
                        maximum_array_bytes=len(generator_payload),
                    )
                }
            untouched_int_payload = _array_payload(inputs, INT_ARRAY_PAYLOAD_SCHEMA)
            if phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
                design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
                self._validate_design(design)
                return {
                    "adjudication-bundle": untouched_adjudication_bundle(
                        config=self.config,
                        denominators=_one(records, SplitCohortHistoryBudgetDenominatorBundle),
                        untouched=_one(records, SplitCohortHistoryBudgetUntouchedBundle),
                        untouched_int_payload=untouched_int_payload,
                        untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                        generator=_one(records, SplitCohortHistoryBudgetGeneratorBundle),
                        generator_arrays_payload=generator_payload,
                        generator_arrays_manifest=_manifest(records, ARRAY_PAYLOAD_SCHEMA),
                        method_freeze=design.method_freeze,
                        maximum_array_bytes=max(len(generator_payload), len(untouched_int_payload)),
                    )
                }
            observer = _one(records, SplitCohortHistoryBudgetObserverBundle)
            return {
                "adjudication-bundle": adjudication_bundle(
                    config=self.config,
                    denominators=_one(records, SplitCohortHistoryBudgetDenominatorBundle),
                    history=_one(records, SplitCohortHistoryBudgetHistoryBundle),
                    observer=observer,
                    untouched=_one(records, SplitCohortHistoryBudgetUntouchedBundle),
                    untouched_int_payload=untouched_int_payload,
                    untouched_int_manifest=_manifest(records, INT_ARRAY_PAYLOAD_SCHEMA),
                    generator=_one(records, SplitCohortHistoryBudgetGeneratorBundle),
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
        if task_id == "development-complete-ledger":
            bundles = tuple(
                sorted(
                    (value for value in records if isinstance(value, SplitCohortHistoryBudgetAdjudicationBundle)),
                    key=lambda value: value.unit_id,
                )
            )
            return {"development-ledger": development_ledger(adjudication_bundles=bundles)}
        if task_id == "development-correctness-power-resource-gate":
            reports = _reports(records)
            canaries = tuple(
                value
                for value in reports.values()
                if value.report_id == "split-cohort-history-budget.source-canary-qualification"
            )
            if len(canaries) != 1:
                raise ValueError("split cohort history budget development requires exact C0 qualification")
            return {
                "development-gate": evaluate_development_gate(
                    _one(records, SplitCohortHistoryBudgetDevelopmentLedger),
                    canaries[0],
                ),
            }
        if task_id == "f0-freeze-method":
            return {
                "method-freeze": freeze_method(
                    development_config=self.config,
                    development_gate=_one(records, SplitCohortHistoryBudgetDevelopmentGate),
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
            targeted_evaluation_config, untouched_evaluation_config = self._evaluation_configs(records)
            return {
                "evaluation-design-freeze": freeze_evaluation_design(
                    method_freeze=_one(records, SplitCohortHistoryBudgetMethodFreeze),
                    targeted_evaluation_config=targeted_evaluation_config,
                    untouched_evaluation_config=untouched_evaluation_config,
                    seed_roster_commitment=_one(records, SplitCohortHistoryBudgetSeedRosterCommitment),
                    implementation_source_closure_sha256=self.closures.complete_sha256,
                )
            }
        if task_id == "f2-development-closeout":
            _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
            return {
                "development-closeout": _closeout(
                    closeout_id="split-cohort-history-budget.development-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("SPLIT_COHORT_HISTORY_BUDGET_DEVELOPMENT_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id == "xt0-targeted-synthesize":
            design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
            self._validate_design(design)
            bundles = tuple(
                value for value in records if isinstance(value, SplitCohortHistoryBudgetAdjudicationBundle)
            )
            if len(bundles) != 90 or any(
                value.method_freeze_sha256 != design.method_freeze.fingerprint()
                for value in bundles
            ):
                raise ValueError("split cohort history budget recurrence input bundle roster differs")
            recurrence_execution = synthesize_recurrence(
                config=self.config,
                method_freeze=design.method_freeze,
                adjudication_bundles=tuple(sorted(bundles, key=lambda value: value.unit_id)),
                include_untouched=False,
            )
            return {
                "bootstrap-summary": recurrence_execution.bootstrap_summary,
                "recurrence-result": recurrence_execution.result,
            }
        if task_id == "xt1-targeted-closeout":
            result = _one(records, SplitCohortHistoryBudgetRecurrenceResult)
            bootstrap = _one(records, SplitCohortHistoryBudgetBootstrapSummary)
            if result.bootstrap_summary_sha256 != bootstrap.fingerprint():
                raise ValueError("split cohort history budget terminal bootstrap binding differs")
            evaluability, scientific_status, reason_codes = _recurrence_adjudication_state(result)
            return {
                "targeted-evaluation-continuation-gate": targeted_continuation_gate(result),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=evaluability,
                    scientific_status=scientific_status,
                    reason_codes=reason_codes,
                ),
                "targeted-closeout": SplitCohortHistoryBudgetTerminalCloseout(
                    closeout_id="split-cohort-history-budget.targeted-closeout",
                    terminal=SplitCohortHistoryBudgetTerminal.TARGETED_EVALUATION_COMPLETE,
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
        if task_id == "xu0-untouched-synthesize":
            design = _one(records, SplitCohortHistoryBudgetEvaluationDesignFreeze)
            self._validate_design(design)
            results = tuple(
                value for value in records if isinstance(value, SplitCohortHistoryBudgetRecurrenceResult)
            )
            if len(results) != 1 or results[0].untouched_cells:
                raise ValueError("split cohort history budget U synthesis requires the bounded T result")
            bundles = tuple(
                sorted(
                    (value for value in records if isinstance(value, SplitCohortHistoryBudgetAdjudicationBundle)),
                    key=lambda value: value.unit_id,
                )
            )
            integrated = integrate_untouched_recurrence(
                config=self.config,
                targeted_result=results[0],
                adjudication_bundles=bundles,
            )
            return {"recurrence-result": integrated}
        if task_id == "integrated-closeout":
            result = _one(records, SplitCohortHistoryBudgetRecurrenceResult)
            evaluability, scientific_status, reason_codes = _recurrence_adjudication_state(result)
            return {
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=evaluability,
                    scientific_status=scientific_status,
                    reason_codes=reason_codes,
                ),
                "terminal-closeout": SplitCohortHistoryBudgetTerminalCloseout(
                    closeout_id="split-cohort-history-budget.integrated-closeout",
                    terminal=SplitCohortHistoryBudgetTerminal.INTEGRATED_EVALUATION_COMPLETE,
                    recurrence_result_sha256=result.fingerprint(),
                    artifact_ids=tuple(
                        sorted(value.artifact_id for value in context.input_bindings)
                    ),
                    receipt_ids=context.dependency_receipt_ids,
                    limitation_ids=(
                        "finite-entered-simulator-population",
                        "conditional-u-act",
                        "no-continuum-or-physical-transport",
                        "no-controller-admission-prospective-controller-evaluation-or-controller-claim",
                    ),
                    evidence_ceiling=result.evidence_ceiling,
                    external_device_count=0,
                    physical_action_count=0,
                ),
            }
        raise ValueError("unknown split cohort history budget task identity")

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
            raise ValueError("split cohort history budget runner output roster differs from the plan")
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
            raise ValueError("split cohort history budget outputs exceed the output byte budget")
        for port, output in zip(context.output_ports, payloads, strict=True):
            original = outputs[port.output_id.removeprefix(f"{context.task_id}.")]
            if isinstance(original, SplitCohortHistoryBudgetHistoryBundle):
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


class SplitCohortHistoryBudgetCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Bind one exact phase config, graph inputs and implementation closure."""

    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: SplitCohortHistoryBudgetConfig,
        external_records: tuple[SplitCohortHistoryBudgetExternalRecord, ...],
        source_files: Mapping[str, bytes],
        injected_nomination_seeds: tuple[bytes, ...] | None = None,
        scientific_inputs: tuple[HistoryBudgetUnitScientificInput, ...] | None = None,
    ) -> None:
        scientific_inputs = require_history_budget_unit_inputs(scientific_inputs, programme_ordinal=2, unit_ids=() if config.phase in (SplitCohortHistoryBudgetPhase.NOMINATION, SplitCohortHistoryBudgetPhase.CANARY) else config.unit_ids, scale_cells=config.scale_cells)
        closures = implementation_closures(source_files)
        expected = split_cohort_history_budget_phase_registry(
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
            raise ValueError("split cohort history budget runtime registry differs")
        scientific_graph(
            registry=registry,
            config=config,
            protocol=protocol_template(registry=registry, config=config),
            external_records=external_records,
        )
        if injected_nomination_seeds is not None and config.phase is not SplitCohortHistoryBudgetPhase.NOMINATION:
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
        self._runners: tuple[SplitCohortHistoryBudgetRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("split cohort history budget runner registry/source binding differs")
        self._runners = tuple(
            SplitCohortHistoryBudgetRunner(
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

    def _external_records_by_artifact(self) -> dict[str, SplitCohortHistoryBudgetExternalRecord]:
        phase_record = SplitCohortHistoryBudgetExternalRecord(
            input_id=f"input.split-cohort-history-budget.{self.config.phase.value.lower()}.config",
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
            raise ValueError("split cohort history budget external-input plan/source binding differs")
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
                f"split cohort history budget external input roster differs; unknown={unknown}; unused={unused}"
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
            raise ValueError("split cohort history budget semantic registry differs")
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
            raise ValueError("split cohort history budget adjudication registry differs")
        capability_key, task_id = {
            SplitCohortHistoryBudgetPhase.NOMINATION: (
                DEVELOPMENT_REPORTER_KEY,
                "nomination-closeout",
            ),
            SplitCohortHistoryBudgetPhase.CANARY: (
                DEVELOPMENT_REPORTER_KEY,
                "c13-canary-qualification",
            ),
            SplitCohortHistoryBudgetPhase.DEVELOPMENT: (
                DEVELOPMENT_REPORTER_KEY,
                "f2-development-closeout",
            ),
            SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION: (
                REPORTER_KEY,
                "xt1-targeted-closeout",
            ),
            SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION: (
                REPORTER_KEY,
                "integrated-closeout",
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
    "SPLIT_COHORT_HISTORY_BUDGET_ALL_SOURCE_CLOSURE_PATHS",
    "SPLIT_COHORT_HISTORY_BUDGET_INTEGRATION_SOURCE_CLOSURE_PATHS",
    "SPLIT_COHORT_HISTORY_BUDGET_SCIENCE_SOURCE_CLOSURE_PATHS",
    'SplitCohortHistoryBudgetCampaignRuntimeProvider',
    'SplitCohortHistoryBudgetImplementationClosures',
    'SplitCohortHistoryBudgetRunner',
    "implementation_closures",
    "output_semantic_contracts_from_execution_plan",
    "integration_source_closure_sha256",
    "science_source_closure_sha256",
]
