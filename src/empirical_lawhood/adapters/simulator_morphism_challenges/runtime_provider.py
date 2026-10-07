"""Closed runtime-provider binding for all four simulator morphism challenges phase DAGs."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetUnitScientificInput, require_history_budget_unit_inputs

from dataclasses import dataclass, fields
from decimal import Decimal
from hashlib import sha256
from typing import Any, Mapping, TypeVar, cast

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.simulator_morphism_challenges.inference import synthesize_recurrence
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
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    TaskProgressEmitter,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.plans import ScientificInputRole
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .authoring import ARRAY_PAYLOAD_SCHEMA, CONFIG_MEDIA_TYPE, DEVELOPMENT_REPORTER_KEY, SimulatorMorphismChallengeExternalRecord, REPORTER_KEY, VERSION, simulator_morphism_challenges_phase_registry, protocol_template, scientific_graph
from .conformance import audit_source_firewall, merge_canary_reports, run_cross_implementation_conformance, run_generator_canary, run_observer_canary, run_resource_canary
from .contracts import SimulatorMorphismChallengeConfig, SimulatorMorphismChallengeMethodFreeze, SimulatorMorphismChallengePhase, SimulatorMorphismChallengeRecurrenceResult, SimulatorMorphismChallengeSeedRosterCommitment, SimulatorMorphismChallengeTerminal, SimulatorMorphismChallengeTerminalCloseout
from .descriptors import deterministic_development_seed
from .runtime_contracts import SimulatorMorphismChallengeAdjudicationBundle, SimulatorMorphismChallengeArrayManifest, SimulatorMorphismChallengeBootstrapSummary, SimulatorMorphismChallengeCanaryReport, SimulatorMorphismChallengeDenominatorBundle, SimulatorMorphismChallengeDevelopmentGate, SimulatorMorphismChallengeDevelopmentLedger, SimulatorMorphismChallengeEvaluationDesignFreeze, SimulatorMorphismChallengeGeneratorBundle, SimulatorMorphismChallengeHistoryBundle, SimulatorMorphismChallengeNominationFreeze, SimulatorMorphismChallengeObserverBundle, SimulatorMorphismChallengePhaseCloseout, SimulatorMorphismChallengeRequestedUnitLedger, SimulatorMorphismChallengeSeedRoster
from .workflow import adjudication_bundle, create_seed_roster, denominator_bundle, development_ledger, evaluate_development_gate, freeze_evaluation_design, freeze_method, freeze_nominations, generator_bundle, history_bundle, receipt_closure_sha256, requested_unit_ledger, targeter_bundle, validate_seed_roster_commitment
from .capability_configs import CAPABILITY_CONFIG_TYPES, RCChallengeCapabilityConfig


SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS = ('src/empirical_lawhood/adapters/history_budget_fixed_scientific_inputs.py',
 'src/empirical_lawhood/adapters/history_budget_scientific_inputs.py',
 'src/empirical_lawhood/adapters/history_budget_seed_constants.py',
 'src/empirical_lawhood/adapters/methods/rc_challenges/executable_binding.py',
 'src/empirical_lawhood/adapters/methods/rc_challenges/extension_bundle.py',
 'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/evaluator.py',
 'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/inference.py',
 'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/observer.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/array_io.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/authoring.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/capability_configs.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/composition.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/conformance.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/contracts.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/descriptors.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/executable_binding.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/extension_bundle.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/issued_inputs.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/numeric_inputs.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/prerequisites.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/retained_results.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/runtime_contracts.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/runtime_provider.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/specification.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/study_authoring.py',
 'src/empirical_lawhood/adapters/simulator_morphism_challenges/workflow.py',
 'src/empirical_lawhood/adapters/simulators/rc_ladder_morphism_challenges/generator.py')


def _closure_digest(source_files: Mapping[str, bytes], paths: tuple[str, ...]) -> str:
    if any(path not in source_files for path in paths):
        raise ValueError("simulator morphism challenges source closure lacks a required member")
    return sha256(
        canonical_json_bytes(
            tuple((path, sha256(source_files[path]).hexdigest()) for path in paths)
        )
    ).hexdigest()


def source_closure_sha256(source_files: Mapping[str, bytes]) -> str:
    """Fingerprint the exact claim-bearing source roster, including this provider."""

    if tuple(sorted(source_files)) != SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS:
        raise ValueError("simulator morphism challenges implementation source roster differs")
    return _closure_digest(source_files, SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS)


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeImplementationClosures:
    complete_sha256: str
    observer_sha256: str
    generator_sha256: str
    evaluator_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "complete_sha256",
            "observer_sha256",
            "generator_sha256",
            "evaluator_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)


def implementation_closures(
    source_files: Mapping[str, bytes],
) -> SimulatorMorphismChallengeImplementationClosures:
    return SimulatorMorphismChallengeImplementationClosures(
        complete_sha256=source_closure_sha256(source_files),
        observer_sha256=_closure_digest(
            source_files,
            ('src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/observer.py', 'src/empirical_lawhood/adapters/history_budget_scientific_inputs.py',),
        ),
        generator_sha256=_closure_digest(
            source_files,
            ('src/empirical_lawhood/adapters/simulators/rc_ladder_morphism_challenges/generator.py',),
        ),
        evaluator_sha256=_closure_digest(
            source_files,
            (
                'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/evaluator.py',
                'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/inference.py',
            ),
        ),
    )


_RECORD_TYPES: dict[str, type[CanonicalRecord]] = {
    SimulatorMorphismChallengeAdjudicationBundle.SCHEMA: SimulatorMorphismChallengeAdjudicationBundle,
    SimulatorMorphismChallengeArrayManifest.SCHEMA: SimulatorMorphismChallengeArrayManifest,
    SimulatorMorphismChallengeBootstrapSummary.SCHEMA: SimulatorMorphismChallengeBootstrapSummary,
    SimulatorMorphismChallengeCanaryReport.SCHEMA: SimulatorMorphismChallengeCanaryReport,
    SimulatorMorphismChallengeConfig.SCHEMA: SimulatorMorphismChallengeConfig,
    SimulatorMorphismChallengeDenominatorBundle.SCHEMA: SimulatorMorphismChallengeDenominatorBundle,
    SimulatorMorphismChallengeDevelopmentGate.SCHEMA: SimulatorMorphismChallengeDevelopmentGate,
    SimulatorMorphismChallengeDevelopmentLedger.SCHEMA: SimulatorMorphismChallengeDevelopmentLedger,
    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA: SimulatorMorphismChallengeEvaluationDesignFreeze,
    SimulatorMorphismChallengeGeneratorBundle.SCHEMA: SimulatorMorphismChallengeGeneratorBundle,
    SimulatorMorphismChallengeHistoryBundle.SCHEMA: SimulatorMorphismChallengeHistoryBundle,
    SimulatorMorphismChallengeMethodFreeze.SCHEMA: SimulatorMorphismChallengeMethodFreeze,
    SimulatorMorphismChallengeNominationFreeze.SCHEMA: SimulatorMorphismChallengeNominationFreeze,
    SimulatorMorphismChallengeObserverBundle.SCHEMA: SimulatorMorphismChallengeObserverBundle,
    SimulatorMorphismChallengePhaseCloseout.SCHEMA: SimulatorMorphismChallengePhaseCloseout,
    SimulatorMorphismChallengeRecurrenceResult.SCHEMA: SimulatorMorphismChallengeRecurrenceResult,
    SimulatorMorphismChallengeRequestedUnitLedger.SCHEMA: SimulatorMorphismChallengeRequestedUnitLedger,
    SimulatorMorphismChallengeSeedRoster.SCHEMA: SimulatorMorphismChallengeSeedRoster,
    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA: SimulatorMorphismChallengeSeedRosterCommitment,
    SimulatorMorphismChallengeTerminalCloseout.SCHEMA: SimulatorMorphismChallengeTerminalCloseout,
    ScientificAdjudicationRecord.SCHEMA: ScientificAdjudicationRecord,
}
_RECORD_TYPES.update((kind.SCHEMA, kind) for kind in CAPABILITY_CONFIG_TYPES.values())


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
                raise ValueError("simulator morphism challenges plan output is absent from its capability contract")
            contract = CapabilityOutputSemanticContract(
                capability_key=task.capability.capability_key,
                capability_version=task.capability.capability_version,
                capability_implementation_sha256=task.capability_implementation_sha256,
                payload_schema=output.payload_schema,
                profile=output.profile,
                top_level_keys=(
                    ()
                    if output.payload_schema == ARRAY_PAYLOAD_SCHEMA
                    else ("schema", "value", "version")
                ),
                value_keys=(
                    ()
                    if output.payload_schema == ARRAY_PAYLOAD_SCHEMA
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
                raise ValueError("simulator morphism challenges historical output semantics conflict")
    return tuple(values[key] for key in sorted(values))


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], kind: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, kind))
    if len(values) != 1:
        raise ValueError(f"simulator morphism challenges task requires exactly one {kind.__name__}")
    return values[0]


def _reports(records: tuple[CanonicalRecord, ...]) -> dict[str, SimulatorMorphismChallengeCanaryReport]:
    values = {value.report_id: value for value in records if isinstance(value, SimulatorMorphismChallengeCanaryReport)}
    if len(values) != sum(isinstance(value, SimulatorMorphismChallengeCanaryReport) for value in records):
        raise ValueError("simulator morphism challenges canary report IDs repeat")
    return values


def _closeout(
    *,
    closeout_id: str,
    phase: SimulatorMorphismChallengePhase,
    records: tuple[CanonicalRecord, ...],
    extra_sha256s: tuple[str, ...] = (),
    passed: bool = True,
    reason_codes: tuple[str, ...] = (),
) -> SimulatorMorphismChallengePhaseCloseout:
    return SimulatorMorphismChallengePhaseCloseout(
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
        raise ValueError("simulator morphism challenges scientific adjudication context is absent")
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
    result: SimulatorMorphismChallengeRecurrenceResult,
) -> tuple[AdjudicationEvaluability, ScientificStatus, tuple[str, ...]]:
    if result.dynamical_existence_recurs and result.decision_existence_recurs:
        reasons = {"SIMULATOR_MORPHISM_CHALLENGE_RECURRENCE_EXISTENCE_SUPPORTED"}
        if (
            result.dynamical_majority_opposition_recurs
            and result.decision_majority_opposition_recurs
        ):
            reasons.add("SIMULATOR_MORPHISM_CHALLENGE_RECURRENCE_MAJORITY_OPPOSITION_SUPPORTED")
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.SUPPORTED,
            tuple(sorted(reasons)),
        )
    if all(value.evaluable_count == 0 for value in result.cells):
        return (
            AdjudicationEvaluability.UNEVALUABLE,
            ScientificStatus.UNEVALUABLE,
            ("SIMULATOR_MORPHISM_CHALLENGE_RECURRENCE_ALL_CELLS_UNEVALUABLE",),
        )
    if any(value.evaluable_count == 0 for value in result.cells):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.PARTIAL,
            ("SIMULATOR_MORPHISM_CHALLENGE_RECURRENCE_CELL_UNEVALUABLE",),
        )
    if all(value.opposed_count == 0 for value in result.cells):
        return (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.NOT_SUPPORTED,
            ("SIMULATOR_MORPHISM_CHALLENGE_RECURRENCE_EXISTENCE_NOT_SUPPORTED",),
        )
    return (
        AdjudicationEvaluability.EVALUABLE,
        ScientificStatus.MIXED,
        ("SIMULATOR_MORPHISM_CHALLENGE_RECURRENCE_NONCOMPENSATING_CONJUNCTION_FAILED",),
    )


@dataclass(frozen=True, slots=True)
class _TaskInputs:
    records: tuple[CanonicalRecord, ...]
    array_payloads: tuple[bytes, ...]


class SimulatorMorphismChallengeRunner:
    """One static capability runner; task identity selects a closed operation."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: SimulatorMorphismChallengeConfig,
        registry: CapabilityRegistry,
        source_files: Mapping[str, bytes],
        closures: SimulatorMorphismChallengeImplementationClosures,
        injected_nomination_seeds: tuple[bytes, ...] | None,
        scientific_inputs: tuple[HistoryBudgetUnitScientificInput, ...],
    ) -> None:
        scientific_inputs = require_history_budget_unit_inputs(scientific_inputs, programme_ordinal=0, unit_ids=() if config.phase in (SimulatorMorphismChallengePhase.NOMINATION, SimulatorMorphismChallengePhase.CANARY) else config.unit_ids, scale_cells=config.scale_cells)
        self.manifest = manifest
        self.config = config
        self.scientific_inputs = scientific_inputs
        self.registry = registry
        self.source_files = dict(source_files)
        self.closures = closures
        self.injected_nomination_seeds = injected_nomination_seeds
        self.execution_count = 0
        self._seed_cache: tuple[SimulatorMorphismChallengeSeedRoster, SimulatorMorphismChallengeSeedRosterCommitment] | None = None

    def _read(self, context: TaskContext) -> _TaskInputs:
        records: list[CanonicalRecord] = []
        arrays: list[bytes] = []
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("simulator morphism challenges input size differs")
            if port.payload_schema == ARRAY_PAYLOAD_SCHEMA:
                arrays.append(payload)
                continue
            try:
                kind = _RECORD_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError("simulator morphism challenges task received an unknown input schema") from error
            record = decode_canonical_bytes(payload, kind, maximum_bytes=port.size_bytes)
            if isinstance(record, RCChallengeCapabilityConfig):
                if record.CAPABILITY_KEY != self.manifest.capability_key or record.config != self.config:
                    raise ValueError("RC task capability config differs from its exact phase input")
            else:
                records.append(record)
        phase_configs = tuple(value for value in records if isinstance(value, SimulatorMorphismChallengeConfig))
        if (
            not any(value == self.config for value in phase_configs)
            or any(
                value != self.config
                and not (
                    self.config.phase is SimulatorMorphismChallengePhase.DEVELOPMENT
                    and value.phase is SimulatorMorphismChallengePhase.EVALUATION
                )
                for value in phase_configs
            )
            or context.config.content_sha256 != CAPABILITY_CONFIG_TYPES[self.manifest.capability_key](self.config).fingerprint()
        ):
            raise ValueError("simulator morphism challenges task config differs from its static provider")
        return _TaskInputs(tuple(records), tuple(arrays))

    def _evaluation_config(self, records: tuple[CanonicalRecord, ...]) -> SimulatorMorphismChallengeConfig:
        values = tuple(
            value
            for value in records
            if isinstance(value, SimulatorMorphismChallengeConfig) and value.phase is SimulatorMorphismChallengePhase.EVALUATION
        )
        if len(values) != 1:
            raise ValueError("simulator morphism challenges task requires one evaluation config")
        return values[0]

    def _validate_design(self, design: SimulatorMorphismChallengeEvaluationDesignFreeze) -> None:
        method = design.method_freeze
        if (
            self.config.phase is not SimulatorMorphismChallengePhase.EVALUATION
            or design.evaluation_config_sha256 != self.config.fingerprint()
            or design.implementation_source_closure_sha256 != self.closures.complete_sha256
            or method.observer_implementation_sha256 != self.closures.observer_sha256
            or method.generator_implementation_sha256 != self.closures.generator_sha256
            or method.evaluator_implementation_sha256 != self.closures.evaluator_sha256
        ):
            raise ValueError("simulator morphism challenges runtime differs from the evaluation design freeze")

    def _descriptor_seed(
        self, task_id: str, records: tuple[CanonicalRecord, ...]
    ) -> tuple[str, bytes]:
        unit_id = task_id.split(".", 1)[1]
        if self.config.phase is SimulatorMorphismChallengePhase.DEVELOPMENT:
            return unit_id, deterministic_development_seed(unit_id)
        design = _one(records, SimulatorMorphismChallengeEvaluationDesignFreeze)
        roster = _one(records, SimulatorMorphismChallengeSeedRoster)
        self._validate_design(design)
        validate_seed_roster_commitment(roster, design.seed_roster_commitment)
        entries = tuple(value for value in roster.entries if value.unit_id == unit_id)
        if len(entries) != 1:
            raise ValueError("simulator morphism challenges evaluation descriptor seed binding differs")
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
            ledger = _one(records, SimulatorMorphismChallengeRequestedUnitLedger)
            if ledger != requested_unit_ledger(self.config):
                raise ValueError("simulator morphism challenges seed roster ledger differs")
            if self._seed_cache is None:
                self._seed_cache = create_seed_roster(
                    self.config,
                    injected_seeds=self.injected_nomination_seeds,
                )
            roster, commitment = self._seed_cache
            return {"roster-commitment": commitment, "seed-roster": roster}
        if task_id == "nomination-seal-roster":
            roster = _one(records, SimulatorMorphismChallengeSeedRoster)
            commitment = _one(records, SimulatorMorphismChallengeSeedRosterCommitment)
            validate_seed_roster_commitment(roster, commitment)
            return {
                "roster-freeze-closeout": _closeout(
                    closeout_id="simulator-morphism-challenges.nomination-roster-freeze",
                    phase=phase,
                    records=records,
                )
            }
        if task_id == "nomination-closeout":
            return {
                "nomination-closeout": _closeout(
                    closeout_id="simulator-morphism-challenges.nomination-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("SIMULATOR_MORPHISM_CHALLENGE_NOMINATION_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id == "canary-contract-conformance":
            return {
                "contract-report": _closeout(
                    closeout_id="simulator-morphism-challenges.contract-conformance",
                    phase=phase,
                    records=records,
                    extra_sha256s=(self.registry.fingerprint(),),
                )
            }
        if task_id == "canary-generator":
            return {"generator-canary": run_generator_canary()}
        if task_id == "canary-observer":
            return {"observer-canary": run_observer_canary()}
        if task_id == "canary-independence-audit":
            checks = audit_source_firewall(self.source_files)
            return {
                "independence-report": _closeout(
                    closeout_id="simulator-morphism-challenges.independence-audit",
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
            independence = _one(records, SimulatorMorphismChallengePhaseCloseout)
            if not independence.passed:
                raise ValueError("simulator morphism challenges independence audit did not pass")
            return {
                "numerical-conformance": run_cross_implementation_conformance(
                    implementation_sha256=self.closures.complete_sha256,
                    generator_report=reports["simulator-morphism-challenges.generator-canary"],
                    observer_report=reports["simulator-morphism-challenges.observer-canary"],
                    independence_check_ids=(
                        "generator-does-not-import-observer-or-physical-scale-methods",
                        "observer-does-not-import-generator-or-physical-scale-methods",
                        "scientific-intersection-is-contracts-only",
                    ),
                )
            }
        if task_id == "canary-resource":
            contract = _one(records, SimulatorMorphismChallengePhaseCloseout)
            if not contract.passed:
                raise ValueError("simulator morphism challenges contract conformance did not pass")
            return {
                "resource-canary": run_resource_canary(
                    implementation_sha256=self.closures.complete_sha256
                )
            }
        if task_id == "canary-adjudicate":
            reports = _reports(records)
            qualification = merge_canary_reports(
                source_check_ids=("exact-static-contract-conformance",),
                numerical=reports["simulator-morphism-challenges.numerical-conformance"],
                resource_report=reports["simulator-morphism-challenges.excluded-resource-canary"],
            )
            return {
                "canary-qualification": qualification,
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("SIMULATOR_MORPHISM_CHALLENGE_CANARY_PHASE_PREREQUISITE_ONLY",),
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
            if phase is SimulatorMorphismChallengePhase.EVALUATION:
                self._validate_design(_one(records, SimulatorMorphismChallengeEvaluationDesignFreeze))
            history_execution = history_bundle(
                config=self.config,
                denominators=_one(records, SimulatorMorphismChallengeDenominatorBundle),
            )
            return {
                "history-array-manifest": history_execution.packed_arrays.manifest,
                "history-arrays": history_execution.packed_arrays.payload,
                "history-bundle": history_execution.bundle,
            }
        if task_id.startswith(("development-targeter.", "evaluation-targeter.")):
            if phase is SimulatorMorphismChallengePhase.EVALUATION:
                self._validate_design(_one(records, SimulatorMorphismChallengeEvaluationDesignFreeze))
            targeter_execution = targeter_bundle(
                config=self.config,
                denominators=_one(records, SimulatorMorphismChallengeDenominatorBundle),
                history=_one(records, SimulatorMorphismChallengeHistoryBundle),
                implementation_sha256=self.closures.observer_sha256,
            )
            return {
                "targeter-array-manifest": targeter_execution.packed_arrays.manifest,
                "targeter-arrays": targeter_execution.packed_arrays.payload,
                "targeter-bundle": targeter_execution.bundle,
            }
        if task_id.startswith("evaluation-freeze-nomination."):
            design = _one(records, SimulatorMorphismChallengeEvaluationDesignFreeze)
            self._validate_design(design)
            return {
                "nomination-freeze": freeze_nominations(
                    history=_one(records, SimulatorMorphismChallengeHistoryBundle),
                    observer=_one(records, SimulatorMorphismChallengeObserverBundle),
                    method_freeze=design.method_freeze,
                )
            }
        if task_id.startswith(("development-generator.", "evaluation-generator.")):
            denominators = _one(records, SimulatorMorphismChallengeDenominatorBundle)
            if phase is SimulatorMorphismChallengePhase.EVALUATION:
                design = _one(records, SimulatorMorphismChallengeEvaluationDesignFreeze)
                self._validate_design(design)
                nomination_freeze = _one(records, SimulatorMorphismChallengeNominationFreeze)
                if nomination_freeze.method_freeze_sha256 != design.method_freeze.fingerprint():
                    raise ValueError("simulator morphism challenges generator nomination freeze differs")
                observer = nomination_freeze.observer_bundle
            else:
                observer = _one(records, SimulatorMorphismChallengeObserverBundle)
            generator_execution = generator_bundle(
                config=self.config,
                denominators=denominators,
                observer=observer,
                implementation_sha256=self.closures.generator_sha256,
            )
            return {
                "generator-array-manifest": generator_execution.packed_arrays.manifest,
                "generator-arrays": generator_execution.packed_arrays.payload,
                "generator-bundle": generator_execution.bundle,
            }
        if task_id.startswith(("development-adjudicate.", "evaluation-adjudicate.")):
            if len(inputs.array_payloads) != 1:
                raise ValueError("simulator morphism challenges adjudicator requires one bounded array payload")
            if phase is SimulatorMorphismChallengePhase.EVALUATION:
                design = _one(records, SimulatorMorphismChallengeEvaluationDesignFreeze)
                self._validate_design(design)
                nomination_freeze = _one(records, SimulatorMorphismChallengeNominationFreeze)
                observer = nomination_freeze.observer_bundle
                method: SimulatorMorphismChallengeMethodFreeze | None = design.method_freeze
            else:
                observer = _one(records, SimulatorMorphismChallengeObserverBundle)
                method = None
            return {
                "adjudication-bundle": adjudication_bundle(
                    config=self.config,
                    denominators=_one(records, SimulatorMorphismChallengeDenominatorBundle),
                    observer=observer,
                    generator=_one(records, SimulatorMorphismChallengeGeneratorBundle),
                    generator_arrays_payload=inputs.array_payloads[0],
                    generator_arrays_manifest=_one(records, SimulatorMorphismChallengeArrayManifest),
                    method_freeze=method,
                    # The payload is an input produced under the upstream
                    # generator's output ceiling, not this adjudicator's much
                    # smaller output ceiling.  The runtime has already opened
                    # and size-checked the typed port; retain the decoder's
                    # exact bound without spuriously rejecting valid arrays.
                    maximum_array_bytes=len(inputs.array_payloads[0]),
                )
            }
        if task_id == "development-power-and-correctness-gate":
            reports = _reports(records)
            canaries = tuple(
                value
                for value in reports.values()
                if value.report_id == "simulator-morphism-challenges.source-canary-qualification"
            )
            if len(canaries) != 1:
                raise ValueError("simulator morphism challenges development requires exact C0 qualification")
            bundles = tuple(
                sorted(
                    (value for value in records if isinstance(value, SimulatorMorphismChallengeAdjudicationBundle)),
                    key=lambda value: value.unit_id,
                )
            )
            development_result = development_ledger(
                canary=canaries[0], adjudication_bundles=bundles
            )
            return {
                "development-ledger": development_result,
                "development-gate": evaluate_development_gate(development_result),
            }
        if task_id == "f0-freeze-method":
            evaluation_config = self._evaluation_config(records)
            return {
                "method-freeze": freeze_method(
                    development_config=self.config,
                    evaluation_config=evaluation_config,
                    seed_roster_commitment=_one(records, SimulatorMorphismChallengeSeedRosterCommitment),
                    development_gate=_one(records, SimulatorMorphismChallengeDevelopmentGate),
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
                    method_freeze=_one(records, SimulatorMorphismChallengeMethodFreeze),
                    evaluation_config=self._evaluation_config(records),
                    seed_roster_commitment=_one(records, SimulatorMorphismChallengeSeedRosterCommitment),
                    implementation_source_closure_sha256=self.closures.complete_sha256,
                )
            }
        if task_id == "f2-development-closeout":
            _one(records, SimulatorMorphismChallengeEvaluationDesignFreeze)
            return {
                "development-closeout": _closeout(
                    closeout_id="simulator-morphism-challenges.development-closeout",
                    phase=phase,
                    records=records,
                ),
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=AdjudicationEvaluability.UNEVALUABLE,
                    scientific_status=ScientificStatus.UNEVALUABLE,
                    reason_codes=("SIMULATOR_MORPHISM_CHALLENGE_DEVELOPMENT_PHASE_PREREQUISITE_ONLY",),
                ),
            }
        if task_id == "recurrence-synthesize":
            design = _one(records, SimulatorMorphismChallengeEvaluationDesignFreeze)
            self._validate_design(design)
            bundles = tuple(
                value for value in records if isinstance(value, SimulatorMorphismChallengeAdjudicationBundle)
            )
            if len(bundles) != 36 or any(
                value.method_freeze_sha256 != design.method_freeze.fingerprint()
                for value in bundles
            ):
                raise ValueError("simulator morphism challenges recurrence input bundle roster differs")
            adjudications = tuple(
                sorted(
                    (value for bundle in bundles for value in bundle.adjudications),
                    key=lambda value: (value.unit_id, value.scale_cells),
                )
            )
            recurrence_execution = synthesize_recurrence(
                config=self.config,
                method_freeze=design.method_freeze,
                adjudications=adjudications,
            )
            return {
                "bootstrap-summary": recurrence_execution.bootstrap_summary,
                "recurrence-result": recurrence_execution.result,
            }
        if task_id == "terminal-report":
            result = _one(records, SimulatorMorphismChallengeRecurrenceResult)
            bootstrap = _one(records, SimulatorMorphismChallengeBootstrapSummary)
            if result.bootstrap_summary_sha256 != bootstrap.fingerprint():
                raise ValueError("simulator morphism challenges terminal bootstrap binding differs")
            evaluability, scientific_status, reason_codes = _recurrence_adjudication_state(result)
            return {
                "scientific-adjudication": _scientific_adjudication(
                    context=context,
                    evaluability=evaluability,
                    scientific_status=scientific_status,
                    reason_codes=reason_codes,
                ),
                "terminal-closeout": SimulatorMorphismChallengeTerminalCloseout(
                    closeout_id="simulator-morphism-challenges.terminal-closeout",
                    terminal=SimulatorMorphismChallengeTerminal.BOUNDED_SIMULATOR_MORPHISM_CHALLENGES_EVALUATED,
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
        raise ValueError("unknown simulator morphism challenges task identity")

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        inputs = self._read(context)
        with threadpool_limits(limits=context.resource_budget.cpu_cores):
            outputs = self._dispatch(context, inputs)
        expected_output_ids = {
            port.output_id.removeprefix(f"{context.task_id}.") for port in context.output_ports
        }
        if set(outputs) != expected_output_ids:
            raise ValueError("simulator morphism challenges runner output roster differs from the plan")
        return RunnerResult(
            outputs=tuple(
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
            ),
            checks=(
                ReceiptCheck("actual-computation-completed", True, ()),
                ReceiptCheck("exact-typed-inputs-consumed", True, ()),
                ReceiptCheck("no-physical-action-network-or-source-write", True, ()),
                ReceiptCheck("task-cpu-thread-ceiling-applied", True, ()),
            ),
        )

    def execute_with_progress(self, context: TaskContext, emitter: TaskProgressEmitter) -> RunnerResult:
        """Report one completed bounded DAG task through the existing scheduler port."""
        result = self.execute(context)
        emitter.advance(Decimal(1))
        return result


class SimulatorMorphismChallengeCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Bind one exact phase config, graph inputs and implementation closure."""

    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: SimulatorMorphismChallengeConfig,
        external_records: tuple[SimulatorMorphismChallengeExternalRecord, ...],
        source_files: Mapping[str, bytes],
        injected_nomination_seeds: tuple[bytes, ...] | None = None,
        scientific_inputs: tuple[HistoryBudgetUnitScientificInput, ...] | None = None,
        manifest_implementation_sha256: str | None = None,
    ) -> None:
        scientific_inputs = require_history_budget_unit_inputs(scientific_inputs, programme_ordinal=0, unit_ids=() if config.phase in (SimulatorMorphismChallengePhase.NOMINATION, SimulatorMorphismChallengePhase.CANARY) else config.unit_ids, scale_cells=config.scale_cells)
        closures = implementation_closures(source_files)
        expected = simulator_morphism_challenges_phase_registry(
            implementation_sha256=closures.complete_sha256 if manifest_implementation_sha256 is None else manifest_implementation_sha256,
            config=config,
        )
        if registry != expected:
            raise ValueError("simulator morphism challenges runtime registry differs")
        scientific_graph(
            registry=registry,
            config=config,
            protocol=protocol_template(registry=registry, config=config),
            external_records=external_records,
        )
        if injected_nomination_seeds is not None and config.phase is not SimulatorMorphismChallengePhase.NOMINATION:
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
        self._runners: tuple[SimulatorMorphismChallengeRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("simulator morphism challenges runner registry/source binding differs")
        self._runners = tuple(
            SimulatorMorphismChallengeRunner(
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

    def _external_records_by_artifact(self) -> dict[str, SimulatorMorphismChallengeExternalRecord]:
        phase_record = SimulatorMorphismChallengeExternalRecord(
            input_id=f"input.simulator-morphism-challenges.{self.config.phase.value.lower()}.config",
            record=self.config,
            role=ScientificInputRole.MODEL,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
        )
        values = (phase_record, *self.external_records)
        records = {value.input_id: value for value in values}
        for manifest in self.registry.capabilities:
            capability_config = CAPABILITY_CONFIG_TYPES[manifest.capability_key](self.config)
            records[f"config-artifact.{capability_config.config_id}"] = SimulatorMorphismChallengeExternalRecord(
                capability_config.config_id, capability_config, ScientificInputRole.MODEL,
                OutcomeAccess.OUTCOME_BLIND, VisibilityCeiling.PROSPECTIVE)
        return records

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("simulator morphism challenges external-input plan/source binding differs")
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
                f"simulator morphism challenges external input roster differs; unknown={unknown}; unused={unused}"
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
            raise ValueError("simulator morphism challenges semantic registry differs")
        if execution_plan is not None:
            return output_semantic_contracts_from_execution_plan(execution_plan)
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                if schema == ARRAY_PAYLOAD_SCHEMA:
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
            raise ValueError("simulator morphism challenges adjudication registry differs")
        capability_key, task_id = {
            SimulatorMorphismChallengePhase.NOMINATION: (
                DEVELOPMENT_REPORTER_KEY,
                "nomination-closeout",
            ),
            SimulatorMorphismChallengePhase.CANARY: (
                DEVELOPMENT_REPORTER_KEY,
                "canary-adjudicate",
            ),
            SimulatorMorphismChallengePhase.DEVELOPMENT: (
                DEVELOPMENT_REPORTER_KEY,
                "f2-development-closeout",
            ),
            SimulatorMorphismChallengePhase.EVALUATION: (
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
    "SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS",
    'SimulatorMorphismChallengeCampaignRuntimeProvider',
    'SimulatorMorphismChallengeImplementationClosures',
    'SimulatorMorphismChallengeRunner',
    "implementation_closures",
    "output_semantic_contracts_from_execution_plan",
    "source_closure_sha256",
]
