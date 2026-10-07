# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
"""Deadline-free package and plans, recovery and parity continuity."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
import time

import pytest

from empirical_lawhood.api.models import EnvelopeExperimentPackage, ExperimentPackage, ElapsedExperimentPackage, assemble_experiment_package, bind_issued_execution_package_elapsed_budget
from empirical_lawhood.api.codecs import loads_campaign_package_authoring
from empirical_lawhood.infrastructure.execution_resource_envelopes import ExternalExecutionResourceEnvelopeStore
from empirical_lawhood.infrastructure.campaign_elapsed_budgets import (
    ExternalCampaignElapsedBudgetStore,
)
from empirical_lawhood.infrastructure.execution import DurableExecutionResourceEnvelopeCoordinator, LocalProcessExecutor, LocalScheduler
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    FilesystemState,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.recovery import ExternalRunRecoveryStore
from empirical_lawhood.infrastructure.sql import (
    SQLiteOperationalRepository,
    create_catalog_engine,
    upgrade_catalog,
)
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    ExternalRootContract,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.runtime.compiler import compile_run_plan, lower_run_plan
from empirical_lawhood.runtime.execution_envelope import ChildResourceTokenLimit, ExecutionResourceEnvelopeSpec, ExecutionResourceTaskCellSpec, JitCellSignatureProjection, JitExpressionFieldValue, JitGraphSignatureManifest, JitGraphSignatureObservation, NonTimeResourceBudget, PredevelopmentJitSignatureCensus, ProgressHeartbeat, ProgressLivenessContract, RunExecutionResourceEnvelope, jit_graph_signature_sha256
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedBudgetSpec, CampaignElapsedReservationPlan, CampaignElapsedTaskProjection, DurableCampaignElapsedBudgetCoordinator
from empirical_lawhood.runtime.plans import ExecutionPlan, RunPlan
from empirical_lawhood.runtime.execution import (
    ExternalInputResolver,
    RunnerRegistry,
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    VerifiedArtifactInput,
)
from empirical_lawhood.runtime.recovery import RunRecoveryIndex, RunRecoveryTerminalEvent, build_run_recovery_index
from empirical_lawhood.runtime.scientific_graph_preservation import ResourceGraphPreservationReceipt, prove_scientific_graph_parity
from tests.issued_study_support import IssuedStudyFixture, build_issued_study_fixture


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(object_id: str, schema: str = 'empirical-lawhood/testing/evidence') -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=_digest(object_id),
    )


def _execution_package_fixture(
    fixture: IssuedStudyFixture,
    *,
    allow_reveal_barrier_resume: bool = False,
    native_simulator: bool = True,
    jit: bool | None = None,
) -> ExperimentPackage:
    issued_package = fixture.package
    steps = issued_package.protocol.steps
    projection_step = steps[0]
    simulator_step = projection_step if native_simulator else None
    compiled = native_simulator if jit is None else jit
    fields = (
        JitExpressionFieldValue(
            field_id="dynamic.action",
            canonical_value_sha256=_digest("hold"),
            expression_changing=False,
        ),
        JitExpressionFieldValue(
            field_id="static.capability",
            canonical_value_sha256=_digest(projection_step.capability_key),
            expression_changing=True,
        ),
    )
    projection = JitCellSignatureProjection(
        cell_id=f"jit.{projection_step.step_id}",
        fields=fields,
        signature_sha256=jit_graph_signature_sha256(fields),
    )
    cache = _identity("jit-cache.deadline-free", 'empirical-lawhood/testing/jit-cache')
    census = PredevelopmentJitSignatureCensus(
        census_id="jit-census.deadline-free",
        expression_field_probe=_identity(
            "jit-probe.deadline-free",
            'empirical-lawhood/testing/jit-field-probe',
        ),
        persistent_cache=cache,
        expression_changing_field_ids=(("static.capability",) if compiled else ()),
        runtime_dynamic_field_ids=(("dynamic.action",) if compiled else ()),
        cell_projections=((projection,) if compiled else ()),
        maximum_distinct_signatures=(1 if compiled else 0),
    )
    observation = JitGraphSignatureObservation(
        signature_sha256=projection.signature_sha256,
        cold_trace_receipt=_identity("jit-cold-trace.deadline-free"),
        cold_compile_receipt=_identity("jit-cold-compile.deadline-free"),
        persistent_cache_receipt=_identity("jit-cache-receipt.deadline-free"),
        warm_execution_receipts=(
            _identity("jit-warm-a.deadline-free"),
            _identity("jit-warm-b.deadline-free"),
        ),
    )
    manifest = JitGraphSignatureManifest(
        manifest_id="jit-manifest.deadline-free",
        predevelopment_census=ObjectIdentity.from_record(census.census_id, census),
        persistent_cache=cache,
        cell_projections=((projection,) if compiled else ()),
        observations=((observation,) if compiled else ()),
        maximum_distinct_signatures=(1 if compiled else 0),
    )
    cells = tuple(
        sorted(
            (
                ExecutionResourceTaskCellSpec(
                    cell_id=f"cell.{step.step_id}",
                    task_id=step.step_id,
                    child_id="child.synthetic",
                    physical_independent_unit_id=(
                        f"{issued_package.system.independent_unit.unit_id}.{step.step_id}"
                    ),
                    preparation_unit_id=f"preparation.{step.step_id}",
                    maximum_attempts=step.maximum_attempts,
                    physical_execution_cost=int(step is simulator_step),
                    retry_token_cost=int(step is simulator_step),
                    native_simulator_launch=step is simulator_step,
                    resource_budget=NonTimeResourceBudget(
                        budget_id=f"budget.{step.step_id}",
                        cpu_cores=step.resource_budget.cpu_cores,
                        memory_bytes=step.resource_budget.memory_bytes,
                        output_bytes=step.resource_budget.output_bytes,
                        source_request_limit=0,
                        source_byte_limit=step.resource_budget.source_scan_bytes,
                    ),
                    progress_liveness=(
                        ProgressLivenessContract(
                            contract_id=f"liveness.{step.step_id}",
                            heartbeat_schema=ProgressHeartbeat.SCHEMA,
                            work_unit_id="native-iteration",
                            maximum_no_progress_gap_seconds=Decimal(10),
                            monotone_counter_required=True,
                            progressing_worker_has_no_elapsed_limit=True,
                        )
                        if step is simulator_step
                        else None
                    ),
                    jit_cell_id=(
                        projection.cell_id if compiled and step is projection_step else None
                    ),
                    jit_signature_sha256=(
                        projection.signature_sha256
                        if compiled and step is projection_step
                        else None
                    ),
                )
                for step in steps
            ),
            key=lambda value: value.cell_id,
        )
    )
    physical_token_limit = sum(
        value.physical_execution_cost * value.maximum_attempts for value in cells
    )
    retry_token_limit = sum(
        value.retry_token_cost * (value.maximum_attempts - 1) for value in cells
    )
    envelope = ExecutionResourceEnvelopeSpec(
        envelope_spec_id="execution-resource-envelope.deadline-free",
        issued_study_extensions=ObjectIdentity.from_record(
            issued_package.issued_study.issued_extensions.issued_extension_set_id,
            issued_package.issued_study.issued_extensions,
        ),
        task_cells=cells,
        child_token_limits=(
            ChildResourceTokenLimit(
                child_id="child.synthetic",
                physical_execution_token_limit=physical_token_limit,
                retry_token_limit=retry_token_limit,
            ),
        ),
        roster_capacity_branches=(),
        allowlisted_retry_reason_codes=(
            "TASK_EXECUTION_FAILED",
            "WORKER_PROGRESS_STALLED",
            *(("authority-required",) if allow_reveal_barrier_resume else ()),
        ),
        allowlisted_resource_stop_reason_codes=(
            "CAMPAIGN_ELAPSED_BUDGET_INSUFFICIENT_FOR_PROJECTED_TASK",
            "RESOURCE_COMPUTABILITY_UNAVAILABLE",
            "RETRY_BUDGET_EXHAUSTED",
        ),
        predevelopment_jit_signature_census=None
        if not compiled
        else ObjectIdentity.from_record(
            census.census_id,
            census,
        ),
        jit_graph_signature_manifest=None
        if not compiled
        else ObjectIdentity.from_record(
            manifest.manifest_id,
            manifest,
        ),
        maximum_parallel_tasks=1,
        aggregate_memory_ceiling_bytes=sum(value.resource_budget.memory_bytes for value in cells),
        resource_lock_ids=(),
        non_gating_forecast_records=(),
        resource_contract_sha256=_digest("resource-contract.deadline-free"),
        nonrefundable_reservations=True,
        first_valid_success_wins=True,
        unknown_completion_requires_new_disposition=True,
        elapsed_time_has_no_control_effect=True,
    )
    return assemble_experiment_package(
        base=fixture.base,
        issued_study=issued_package.issued_study,
        publication_receipt=issued_package.publication_receipt,
        frozen_proposal=issued_package.frozen_proposal,
        scientific_approval=issued_package.scientific_approval,
        execution_authority=issued_package.execution_authority,
        execution_resource_envelope_spec=envelope,
        predevelopment_jit_signature_census=census if compiled else None,
        jit_graph_signature_manifest=manifest if compiled else None,
        run_plan_id="run.deadline-free-extensions",
        grantee_id="operator.execution-service",
        at_utc="2026-08-22T20:44:00Z",
    )


def _compile(
    fixture: IssuedStudyFixture,
    package: ExperimentPackage,
) -> tuple[RunPlan, ExecutionPlan]:
    candidate = package.issued_study.candidate
    base_candidate = candidate.base_candidate.base_candidate
    run_plan = compile_run_plan(
        run_plan_id=package.run_plan_id,
        campaign=package.campaign,
        system=package.system,
        experiment=package.experiment,
        frozen_proposal=package.frozen_proposal,
        authorization=package.scientific_approval,
        template=package.protocol,
        registry=package.registry,
        implementation_commit=package.implementation_commit,
        approval_service=fixture.approval_service,
        model_set=package.model_set,
        scientific_graph=base_candidate.scientific_graph,
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        issued_extension_set=package.issued_study.issued_extensions,
        execution_resource_envelope_spec=package.execution_resource_envelope_spec,
        predevelopment_jit_signature_census=(package.predevelopment_jit_signature_census),
        jit_graph_signature_manifest=package.jit_graph_signature_manifest,
    )
    assert isinstance(run_plan, RunPlan)
    execution_plan = lower_run_plan(run_plan, package.registry)
    assert isinstance(execution_plan, ExecutionPlan)
    return run_plan, execution_plan


class _StaticInspector:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def inspect(self, _contract: ExternalRootContract) -> FilesystemState:
        return FilesystemState(
            canonical_root=str(self.path),
            active_mount=True,
            writable=True,
            free_bytes=100_000_000,
            path_is_symlink=False,
        )


def _plane(path: Path) -> ExternalArtifactPlane:
    contract = ExternalRootContract(
        storage_root_id="test-deadline-free-external",
        logical_name="Deadline-free synthetic external artifact root",
        canonical_path=str(path.resolve()),
        required_mount_path=str(path.resolve()),
        mount_contract_schema='empirical-lawhood/testing/fixtures/execution-package-mount-contract',
        minimum_free_bytes=1,
    )
    return ExternalArtifactPlane(
        GuardedExternalRoot(contract, _StaticInspector(path)),
    )


class _StaticInputResolver(ExternalInputResolver):
    def __init__(self, values: tuple[VerifiedArtifactInput, ...]) -> None:
        self.values = {value.artifact_id: value for value in values}

    def resolve(
        self,
        logical_artifact_ids: tuple[str, ...],
    ) -> tuple[VerifiedArtifactInput, ...]:
        return tuple(self.values[value] for value in logical_artifact_ids)


def _external_inputs(
    plane: ExternalArtifactPlane,
    execution_plan: ExecutionPlan,
) -> _StaticInputResolver:
    configs = {
        task.capability.config.artifact_id: task.capability.config for task in execution_plan.tasks
    }
    artifact_ids = tuple(
        sorted(
            {
                artifact_id
                for task in execution_plan.tasks
                for artifact_id in task.external_input_artifact_ids
            }
        )
    )
    resolved = []
    for artifact_id in artifact_ids:
        config = configs.get(artifact_id)
        if config is None:
            spec = next(
                value
                for task in execution_plan.tasks
                for value in task.external_inputs
                if value.logical_artifact_id == artifact_id
            )
            payload = b"{}"
            assert spec.expected_payload_schema is not None
            assert spec.expected_media_type is not None
            payload_schema = spec.expected_payload_schema
            media_type = spec.expected_media_type
            profile = ArtifactProfile.CANONICAL_JSON
            logical_content_sha256 = spec.expected_content_sha256
            outcome_access = spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND
            parent_identity = ObjectIdentity(
                object_id=spec.input_id,
                object_schema='empirical-lawhood/runtime/external-input-scope',
                object_version="1.0.0",
                object_fingerprint=spec.identity_scope_sha256 or _digest(spec.input_id),
            )
        else:
            task_key = config.artifact_id.removeprefix("config-artifact.")
            payload = f"config payload for {task_key}".encode()
            payload_schema = config.config_schema
            media_type = "text/plain"
            profile = ArtifactProfile.TEXT_PARAMETERS
            logical_content_sha256 = config.content_sha256
            outcome_access = OutcomeAccess.OUTCOME_BLIND
            parent_identity = ObjectIdentity.from_record(config.config_id, config)
        parent = ArtifactLineageParent(
            identity=parent_identity,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=outcome_access,
        )
        result = plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=artifact_id,
                relative_path=f"inputs/{artifact_id}.txt",
                payload_schema=payload_schema,
                profile=profile,
                media_type=media_type,
                publication_scope_id="deadline-free-scheduler-external-inputs",
                publication_scope_relative_root="inputs",
                payload=payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                outcome_access=outcome_access,
                logical_content_sha256=logical_content_sha256,
                lineage_parents=(parent,),
            )
        )
        resolved.append(VerifiedArtifactInput(result.logical, result.materialization))
    return _StaticInputResolver(tuple(resolved))


class _DeterministicRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=canonical_json_bytes(
                        {
                            "output_id": port.output_id,
                            "schema": port.payload_schema,
                            "task_id": context.task_id,
                        }
                    ),
                )
                for port in context.output_ports
            ),
            checks=(ReceiptCheck("worker-contract-passed", True, ()),),
        )


class _ResourceEnvelopeStore:
    def __init__(self) -> None:
        self.records: dict[str, RunExecutionResourceEnvelope] = {}

    def create(
        self,
        envelope: RunExecutionResourceEnvelope,
        *,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None = None,
    ) -> None:
        if envelope.envelope_id in self.records:
            raise ValueError("resource envelope already exists")
        self.records[envelope.envelope_id] = envelope

    def load(self, envelope_id: str) -> RunExecutionResourceEnvelope:
        return self.records[envelope_id]

    current = load

    def compare_and_append(
        self,
        *,
        expected: RunExecutionResourceEnvelope,
        updated: RunExecutionResourceEnvelope,
    ) -> None:
        current = self.records[updated.envelope_id]
        if current != expected:
            raise ValueError("resource-envelope compare-and-append conflict")
        self.records[updated.envelope_id] = updated


class _EagerProgressRunner:
    def __init__(
        self,
        delegate: _DeterministicRunner,
        delay_seconds: float = 0.0,
    ) -> None:
        self.manifest = delegate.manifest
        self.delegate = delegate
        self.delay_seconds = delay_seconds

    def execute(self, context: TaskContext) -> RunnerResult:
        return self.delegate.execute(context)

    def execute_with_progress(self, context: TaskContext, emitter) -> RunnerResult:  # type: ignore[no-untyped-def]
        for counter in range(1, 6):
            if self.delay_seconds:
                time.sleep(self.delay_seconds / 5)
            emitter.advance(Decimal(counter))
        return self.delegate.execute(context)


class _ProgressRunner(_EagerProgressRunner):
    def __init__(
        self,
        delegate: _DeterministicRunner,
        projection: JitCellSignatureProjection,
        delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(delegate, delay_seconds)
        self.projection = projection

    def jit_signature_projection(self, context: TaskContext) -> JitCellSignatureProjection:
        del context
        return self.projection


class _EventClock:
    def __init__(self) -> None:
        self.second = 0

    def now_utc(self) -> str:
        result = f"2026-08-22T21:00:{self.second:02d}Z"
        self.second += 1
        return result


class _NativeLaunchClock:
    """Keep slow fixture setup out of the deliberately tiny interruption budget."""

    def __init__(self, coordinator, envelope_id):
        self.coordinator = coordinator
        self.envelope_id = envelope_id
        self.started = None

    def now_utc(self) -> str:
        if self.started is None and any(
            cell.state.value == "LAUNCHED"
            for cell in self.coordinator.current(self.envelope_id).cells
        ):
            self.started = time.monotonic()
        elapsed = 0 if self.started is None else int(time.monotonic() - self.started)
        return (datetime(2026, 8, 22, tzinfo=timezone.utc) + timedelta(seconds=elapsed)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize("jit", (False, True))
def test_current_package_compiles_and_preserves_recovery_and_graph_parity(
    tmp_path: Path,
    jit: bool,
) -> None:
    fixture = build_issued_study_fixture(tmp_path)
    package = _execution_package_fixture(fixture, jit=jit)
    assert not isinstance(package, EnvelopeExperimentPackage)
    assert not hasattr(package.execution_resource_envelope_spec, "deadline_utc")
    run_plan, execution_plan = _compile(fixture, package)
    assert run_plan.execution_resource_envelope_spec == (
        execution_plan.execution_resource_envelope_spec
    )
    recovery = build_run_recovery_index(
        execution_plan,
        run_id=run_plan.run_plan_id,
        wave_id="wave.deadline-free",
        authority_identities=(
            ObjectIdentity.from_record(
                package.execution_authority.authority_id,
                package.execution_authority,
            ),
        ),
    )
    assert isinstance(recovery, RunRecoveryIndex)
    recovery.validate_plan(execution_plan)
    parity = prove_scientific_graph_parity(
        candidate=package.issued_study.candidate,
        issued_candidate=package.issued_study.candidate,
        authorized_experiment=package.experiment,
        scientific_approval=ObjectIdentity.from_record(
            package.scientific_approval.authorization_id,
            package.scientific_approval,
        ),
        issue_manifest=ObjectIdentity.from_record(
            package.issued_study.issue_id,
            package.issued_study,
        ),
        package=ObjectIdentity.from_record(package.package_id, package),
        execution_authority=ObjectIdentity.from_record(
            package.execution_authority.authority_id,
            package.execution_authority,
        ),
        run_plan=run_plan,
        execution_plan=execution_plan,
        registry=package.registry,
    )
    assert isinstance(parity, ResourceGraphPreservationReceipt)
    assert parity.jit_graph_signature_manifest == run_plan.jit_graph_signature_manifest


def test_package_binds_elapsed_budget_and_exact_native_task_census(tmp_path: Path) -> None:
    fixture = build_issued_study_fixture(tmp_path)
    execution_package = _execution_package_fixture(fixture, jit=False)
    budget = CampaignElapsedBudgetSpec(
        "campaign-elapsed-budget.synthetic",
        ObjectIdentity.from_record(execution_package.campaign.campaign_id, execution_package.campaign),
        100,
        Decimal("0.20"),
    )
    plan = CampaignElapsedReservationPlan(
        "campaign-elapsed-reservations.synthetic",
        ObjectIdentity.from_record(budget.budget_id, budget),
        "stage.elapsed-budget-binding",
        "elapsed-envelope-binding.synthetic",
        ObjectIdentity.from_record(
            execution_package.execution_resource_envelope_spec.envelope_spec_id,
            execution_package.execution_resource_envelope_spec,
        ),
        _identity("elapsed-canary.synthetic"),
        tuple(
            CampaignElapsedTaskProjection(cell.task_id, Decimal(1))
            for cell in execution_package.execution_resource_envelope_spec.task_cells
            if cell.native_simulator_launch
        ),
    )
    package = bind_issued_execution_package_elapsed_budget(
        execution_package=execution_package,
        campaign_elapsed_budget=budget,
        campaign_elapsed_reservation_plan=plan,
    )
    assert isinstance(package, ElapsedExperimentPackage)
    assert isinstance(package, ExperimentPackage)
    assert loads_campaign_package_authoring(
        package.canonical_bytes().decode("utf-8"), media_type="application/json"
    ) == package
    _compile(fixture, package)


@pytest.mark.parametrize(("jit", "hard_boundary"), ((False, False), (True, False), (False, True)))
def test_current_scheduler_is_deadline_free_and_reconstructs_external_resource_events(
    tmp_path: Path,
    jit: bool,
    hard_boundary: bool,
) -> None:
    issue_root = tmp_path / "issue"
    issue_root.mkdir()
    fixture = build_issued_study_fixture(issue_root)
    package = _execution_package_fixture(fixture, jit=jit)
    run_plan, execution_plan = _compile(fixture, package)
    # This scheduler conformance fixture exercises infrastructure only.  Drop
    # the test package's synthetic pre-existing medium, whose deliberately
    # fictitious digest has no external bytes, while preserving task topology.
    execution_plan = replace(
        execution_plan,
        tasks=tuple(
            replace(
                task,
                capability=(
                    replace(
                        task.capability,
                        requested_resources=replace(
                            task.capability.requested_resources,
                            wall_time_seconds=1,
                        ),
                    )
                    if task.task_id
                    == next(
                        value.task_id
                        for value in package.execution_resource_envelope_spec.task_cells
                        if value.native_simulator_launch
                    )
                    else task.capability
                ),
                external_inputs=tuple(
                    value
                    for value in task.external_inputs
                    if value.logical_artifact_id != "materialization.reference-medium"
                ),
                scientific_inputs=tuple(
                    value
                    for value in task.scientific_inputs
                    if value.operational_logical_artifact_id != "materialization.reference-medium"
                ),
            )
            for task in execution_plan.tasks
        ),
    )
    external_root = tmp_path / "external"
    external_root.mkdir()
    plane = _plane(external_root)
    engine = create_catalog_engine(f"sqlite+pysqlite:///{tmp_path / 'catalog.sqlite3'}")
    upgrade_catalog(engine)
    repository = SQLiteOperationalRepository(engine)
    recovery_index = build_run_recovery_index(
        execution_plan,
        run_id=run_plan.run_plan_id,
        wave_id="wave.deadline-free-scheduler",
        authority_identities=(
            ObjectIdentity.from_record(
                package.execution_authority.authority_id,
                package.execution_authority,
            ),
        ),
    )
    assert isinstance(recovery_index, RunRecoveryIndex)
    recovery_store = ExternalRunRecoveryStore(plane)
    resource_store = ExternalExecutionResourceEnvelopeStore(plane.root)
    coordinator = DurableExecutionResourceEnvelopeCoordinator(resource_store)
    envelope_id = f"run-resource-envelope.{run_plan.run_plan_id}"
    coordinator.initialize(
        envelope_id=envelope_id,
        spec=package.execution_resource_envelope_spec,
        execution_plan=ObjectIdentity.from_record(
            execution_plan.execution_plan_id,
            execution_plan,
        ),
        jit_graph_signature_manifest=package.jit_graph_signature_manifest,
    )
    elapsed_budget = CampaignElapsedBudgetSpec(
        budget_id="deadline-free-execution.scheduler-test-budget",
        campaign_anchor=ObjectIdentity.from_record(
            package.campaign.campaign_id,
            package.campaign,
        ),
        cumulative_elapsed_ceiling_seconds=2 if hard_boundary else 100,
        reserve_fraction=Decimal(0),
    )
    elapsed_coordinator = DurableCampaignElapsedBudgetCoordinator(
        ExternalCampaignElapsedBudgetStore(
            plane,
            state_root_relative_path="control/test-campaign-elapsed",
        ),
        budget=elapsed_budget,
        ledger_id="deadline-free-execution.scheduler-test-ledger",
    )
    elapsed_plan = CampaignElapsedReservationPlan(
        plan_id="deadline-free-execution.scheduler-test-reservations",
        budget=ObjectIdentity.from_record(elapsed_budget.budget_id, elapsed_budget),
        stage_id="q",
        envelope_binding_id="deadline-free-execution.scheduler-test-envelope-binding",
        governed_envelope=ObjectIdentity.from_record(
            package.execution_resource_envelope_spec.envelope_spec_id,
            package.execution_resource_envelope_spec,
        ),
        projection_basis=_identity("deadline-free-execution.scheduler-test-canary"),
        task_projections=tuple(
            CampaignElapsedTaskProjection(value.task_id, Decimal("0.1"))
            for value in package.execution_resource_envelope_spec.task_cells
            if value.native_simulator_launch
        ),
    )
    native_cell = next(
        value
        for value in package.execution_resource_envelope_spec.task_cells
        if value.native_simulator_launch
    )
    projection = (
        None
        if package.jit_graph_signature_manifest is None
        else next(
            value
            for value in package.jit_graph_signature_manifest.cell_projections
            if value.cell_id == native_cell.jit_cell_id
        )
    )
    runners = []
    for manifest in package.registry.capabilities:
        delegate = _DeterministicRunner(manifest)
        runners.append(
            (
                    _EagerProgressRunner(
                        delegate,
                        delay_seconds=3.0 if hard_boundary else 1.2,
                    )
                    if projection is None
                    else _ProgressRunner(
                        delegate,
                        projection,
                        delay_seconds=3.0 if hard_boundary else 1.2,
                    )
            )
            if manifest.capability_key
            == next(
                value.capability.capability_key
                for value in execution_plan.tasks
                if value.task_id == native_cell.task_id
            )
            else delegate
        )
    input_resolver = _external_inputs(plane, execution_plan)
    scheduler = LocalScheduler(
        operational_repository=repository,
        artifact_writer=plane,
        receipt_store=ExternalTaskReceiptStore(plane),
        recovery_index=recovery_index,
        recovery_store=recovery_store,
        runner_registry=RunnerRegistry(
            tuple(runners),
            registry_sha256=package.registry.fingerprint(),
        ),
        implementation_commit=execution_plan.implementation_commit,
        external_input_resolver=input_resolver,
        executor=LocalProcessExecutor(scratch_root=plane.root),
        input_port_factory=plane,
        execution_resource_envelope_coordinator=coordinator,
        execution_resource_envelope_id=envelope_id,
        campaign_elapsed_budget_coordinator=elapsed_coordinator,
        campaign_elapsed_reservation_plan=elapsed_plan,
        jit_graph_signature_manifest=package.jit_graph_signature_manifest,
        execution_envelope_event_clock=(
            _NativeLaunchClock(coordinator, envelope_id) if hard_boundary else _EventClock()
        ),
    )
    assert (
        next(
            value.capability.requested_resources.wall_time_seconds
            for value in execution_plan.tasks
            if value.task_id == native_cell.task_id
        )
        == 1
    )
    started = time.monotonic()
    result = scheduler.execute(run_plan.run_plan_id, execution_plan)
    assert time.monotonic() - started > (0.8 if hard_boundary else 1.1)
    assert result.status.value == ("BLOCKED" if hard_boundary else "SUCCEEDED")
    durable = coordinator.load(envelope_id)
    if hard_boundary:
        assert durable.cell(native_cell.cell_id).state.value == "UNKNOWN_COMPLETION"
        assert elapsed_coordinator.current().exhausted is True
    else:
        assert all(value.state.value == "TERMINAL" for value in durable.cells)
        assert elapsed_coordinator.current().accumulated_active_seconds > 0
        assert any(value.kind.value == "PROGRESS_HEARTBEAT" for value in durable.events)
    reconstructed = recovery_store.reconstruct_execution_resource_envelope(
        index=recovery_index,
        envelope_id=envelope_id,
        spec=package.execution_resource_envelope_spec,
        execution_plan=ObjectIdentity.from_record(
            execution_plan.execution_plan_id,
            execution_plan,
        ),
        jit_graph_signature_manifest=package.jit_graph_signature_manifest,
    )
    assert reconstructed == durable
    if hard_boundary:
        assert scheduler.execute(run_plan.run_plan_id, execution_plan).status.value == "BLOCKED"
    terminal = recovery_store.read_terminal(recovery_index)
    assert isinstance(terminal, RunRecoveryTerminalEvent)
    engine.dispose()


def test_compiler_rejects_duplicate_resource_envelope_contracts(tmp_path: Path) -> None:
    fixture = build_issued_study_fixture(tmp_path)
    package = _execution_package_fixture(fixture)
    candidate = package.issued_study.candidate
    base_candidate = candidate.base_candidate.base_candidate
    with pytest.raises(ValueError, match="mutually exclusive"):
        compile_run_plan(
            run_plan_id=package.run_plan_id,
            campaign=package.campaign,
            system=package.system,
            experiment=package.experiment,
            frozen_proposal=package.frozen_proposal,
            authorization=package.scientific_approval,
            template=package.protocol,
            registry=package.registry,
            implementation_commit=package.implementation_commit,
            approval_service=fixture.approval_service,
            scientific_graph=base_candidate.scientific_graph,
            candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
            issued_extension_set=package.issued_study.issued_extensions,
            execution_envelope_spec=fixture.package.execution_envelope_spec,
            execution_resource_envelope_spec=(package.execution_resource_envelope_spec),
            predevelopment_jit_signature_census=(package.predevelopment_jit_signature_census),
            jit_graph_signature_manifest=package.jit_graph_signature_manifest,
        )
