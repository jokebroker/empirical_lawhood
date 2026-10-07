"""Current exposed selected-parent investigations, not discovered event frequency.

One nominated mathematical context is reconstructed faithfully. Other current
parents are explicitly separate conditional instances. Future branches remain
nested under that one parent and never acquire event-level independence.
"""

import base64
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.composition.matrix_response_study.selected_event_inputs import MatrixSelectedEventConfig, selected_event_protocol
from empirical_lawhood.adapters.methods.matrix_response_study import anisotropic_feasibility as geometry
from empirical_lawhood.adapters.methods.matrix_response_study import shooting_committor as shooting
from empirical_lawhood.adapters.methods.matrix_response_study.geometry_inputs import validate_geometry_source
from empirical_lawhood.adapters.simulators.six_matrix_response import shooting as native
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.scientific_inputs import SixMatrixResponseScientificSeedInput
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import SixMatrixResponseRNGStreamReceipt
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactManifest

from .matrix_geometry import MatrixGeometryScanInput, authenticate_matrix_native_environment, matrix_native_runtime_observation, prove_matrix_geometry_scan, read_matrix_geometry_result
from .matrix_native_custody import MatrixNativeCustody, matrix_scientific_code_sha256, preflight_matrix_native_store, retain_matrix_native_runtime

SELECTED_EVENT_INPUT_MAXIMUM_BYTES = 16 * 1024**2
SELECTED_EVENT_RESULT_MAXIMUM_BYTES = 16 * 1024**2
PUBLIC_SELECTED_EVENT_MASTER_SEED = 706607
NOMINATED_PARENT_DRIVER = "a0d387ebd22b510ccf5816ba09c1724780cff7bec5e7bf99ae4a518d28242cb9"
NOMINATED_PARENT_FINAL_STATE = "d1dd5ed2501e10de345110f59c4e1ed7ebe23065f5947eea848248cf090dcad5"
ORIGINAL_NUMERICAL_QUALIFICATION = "7530d6361529c0b97eb2094c38ab8bfa9f4f039d12a840f30bc149e436db281b"
CHECKPOINT_STEPS = (816, 848, 880, 896, 928, 960, 1024)


@dataclass(frozen=True, slots=True)
class MatrixSelectedParentRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/selected-events/parent-request"
    config_id: str
    source: SixMatrixResponseSixMatrixSourceConfig
    scientific_code_sha256: str
    environment_lock_sha256: str
    minimum_free_bytes: int = 1024**3
    maximum_output_bytes: int = SELECTED_EVENT_RESULT_MAXIMUM_BYTES

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_geometry_source(self.source)
        for name in ("scientific_code_sha256", "environment_lock_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.minimum_free_bytes < 1 or self.maximum_output_bytes != SELECTED_EVENT_RESULT_MAXIMUM_BYTES:
            raise ValueError("Selected parent requires bounded explicit storage")

    @property
    def run_id(self):
        return f"selected-parent.run-{self.fingerprint()[:32]}"


@dataclass(frozen=True, slots=True)
class MatrixSelectedParentOperand(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/selected-events/parent-operand"
    operand_id: str
    request_sha256: str
    conditional_source_instance: str
    source_config: ObjectIdentity
    rollout: geometry.MatrixResponseAnisotropicFeasibilityRolloutSummary
    implementation_commit: str
    numerical_qualification_sha256: str
    source_report: ObjectIdentity | None
    source_geometry_config: MatrixGeometryScanInput | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        validate_sha256(self.request_sha256, field_name="request_sha256")
        validate_sha256(self.numerical_qualification_sha256, field_name="numerical_qualification_sha256")
        if self.conditional_source_instance not in ("paper-nominated", "current-alternate"):
            raise ValueError("Selected parent requires an explicit conditional source instance")
        row = self.rollout
        if (not row.valid or row.q != 2 or row.stage is not geometry.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY
            or row.history_id != "matrix-history.joint-increasing-coupling"
            or row.constitution.value != "01" or row.factor_y.final_phi is None
            or row.requested_steps != 1024):
            raise ValueError("Selected parent is absent, invalid or has no admitted geometric event")
        if self.conditional_source_instance == "paper-nominated" and (
            row.alpha_x_index, row.alpha_y_index, row.seed_index, row.rng_seed_sha256, row.final_state_sha256) != (
                1, 11, 1, NOMINATED_PARENT_DRIVER, NOMINATED_PARENT_FINAL_STATE):
            raise ValueError("Paper nominated mathematical parent cannot be replaced")
        if self.conditional_source_instance == "current-alternate" and (self.source_report is None or self.source_geometry_config is None):
            raise ValueError("Alternate selected parent lacks its exact current geometry report")

    @property
    def rollouts(self):
        return (self.rollout,)


@dataclass(frozen=True, slots=True)
class MatrixSelectedFutureSeed(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/selected-events/future-seed"
    checkpoint_step: int
    stream_index: int
    full_seed_sha256: str

    def __post_init__(self) -> None:
        if type(self.checkpoint_step) is not int or type(self.stream_index) is not int:
            raise ValueError("Selected-event allocation requires integer scientific coordinates")
        validate_sha256(self.full_seed_sha256, field_name="full_seed_sha256")


@dataclass(frozen=True, slots=True)
class MatrixSelectedEventInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/selected-events/input"
    config_id: str
    parent_request: MatrixSelectedParentRequest
    parent: ArtifactManifest
    null_reference: MatrixGeometryScanInput
    protocol: MatrixSelectedEventConfig
    branch_seeds: tuple[MatrixSelectedFutureSeed, ...]
    bridge_seeds: tuple[MatrixSelectedFutureSeed, ...]
    prior_exposed_seed_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if tuple((s.checkpoint_step, s.stream_index) for s in self.branch_seeds) != tuple((step, i) for step in CHECKPOINT_STEPS for i in range(64)):
            raise ValueError("Selected event requires all seven complete 64-future cohorts")
        if tuple((s.checkpoint_step, s.stream_index) for s in self.bridge_seeds) != tuple((0, i) for i in range(1, 1025)):
            raise ValueError("Selected event requires the complete same-driver bridge allocation")
        if self.parent_request.source != self.null_reference.source:
            raise ValueError("Selected-event null reference uses another numerical source preset")
        if self.parent.logical.payload_schema != MatrixSelectedParentOperand.SCHEMA:
            raise ValueError("Selected-event parent has another scientific role/schema")
        seeds = tuple(s.full_seed_sha256 for s in (*self.branch_seeds, *self.bridge_seeds))
        effective = tuple(s[:32] for s in seeds)
        if len(set(effective)) != len(effective):
            raise ValueError("Selected event repeats an effective future/bridge stream")
        if tuple(sorted(set(self.prior_exposed_seed_sha256s))) != self.prior_exposed_seed_sha256s:
            raise ValueError("Selected event prior exposure census must be sorted and unique")
        for seed in self.prior_exposed_seed_sha256s:
            validate_sha256(seed, field_name="prior_exposed_seed_sha256s")
        excluded = {NOMINATED_PARENT_DRIVER[:32], self.protocol.parent_rng_seed_sha256[:32], *(s[:32] for s in self.prior_exposed_seed_sha256s)}
        if set(effective) & excluded:
            raise ValueError("Selected event reuses its exposed parent or previously exposed effective driver")

    @property
    def source(self):
        return self.parent_request.source

    @property
    def scientific_code_sha256(self):
        return self.parent_request.scientific_code_sha256

    @property
    def environment_lock_sha256(self):
        return self.parent_request.environment_lock_sha256

    @property
    def minimum_free_bytes(self):
        return self.parent_request.minimum_free_bytes

    @property
    def maximum_output_bytes(self):
        return self.parent_request.maximum_output_bytes

    @property
    def run_id(self):
        return f"selected-event.run-{self.fingerprint()[:32]}"


def selected_parent_example_input(*, environment_lock_sha256: str):
    from empirical_lawhood.adapters.composition.matrix_response_study.design import _source_config
    return MatrixSelectedParentRequest("selected-parent.exposed-paper-context", _source_config(),
        matrix_scientific_code_sha256(), environment_lock_sha256)


def _custody(config, writer, implementation_commit, *, read_only=False):
    return MatrixNativeCustody(writer=writer, config=config, run_id=config.run_id,
        implementation_commit=implementation_commit, minimum_free_bytes=config.minimum_free_bytes,
        maximum_output_bytes=config.maximum_output_bytes, read_only=read_only)


def read_selected_parent(*, config: MatrixSelectedParentRequest, artifact_writer: ExternalArtifactPlane, implementation_commit: str):
    if config.scientific_code_sha256 != matrix_scientific_code_sha256():
        raise ValueError("Selected parent scientific source owners changed after preparation")
    custody = _custody(config, artifact_writer, implementation_commit, read_only=True)
    operand = custody.load("parent", MatrixSelectedParentOperand)
    if (operand is None or operand.request_sha256 != config.fingerprint()
        or operand.implementation_commit != implementation_commit
        or operand.source_config != ObjectIdentity.from_record(config.source.config_id, config.source)):
        raise ValueError("Selected event lacks its complete authenticated current parent")
    if operand.conditional_source_instance == "paper-nominated":
        if custody.load("nominated-rollout", geometry.MatrixResponseAnisotropicFeasibilityRolloutSummary) != operand.rollout:
            raise ValueError("Selected nominated operand differs from its actual source receipt")
    else:
        report = read_matrix_geometry_result(config=operand.source_geometry_config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)["report"]
        if (ObjectIdentity.from_record(report.report_id, report) != operand.source_report
            or next((row for row in report.rollouts if row.rollout_id == operand.rollout.rollout_id), None) != operand.rollout):
            raise ValueError("Alternate selected operand differs from its actual complete geometry source")
    return operand, custody._manifest(f"runs/{config.run_id}/outputs/parent.json")


def reconstruct_nominated_selected_parent(*, config: MatrixSelectedParentRequest, artifact_writer: ExternalArtifactPlane,
                                         project_root: Path, implementation_commit: str):
    """Reconstruct the fixed exposed parent; mismatch stops, with no substitute."""
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    authenticate_matrix_native_environment(config, project_root=project_root)
    custody = _custody(config, artifact_writer, implementation_commit)
    if custody.load("parent", MatrixSelectedParentOperand) is not None:
        return read_selected_parent(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    from decimal import Decimal
    member = next(m for m in config.source.anisotropic_model.family_members if (m.mass_x, m.mass_y, m.cross_coupling_gamma) == (Decimal("0.5"), Decimal("0.5"), Decimal(1)))
    grid = geometry._grid(Decimal(0), Decimal(8), 13)
    row = custody.load("nominated-rollout", geometry.MatrixResponseAnisotropicFeasibilityRolloutSummary)
    if row is None:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            retain_matrix_native_runtime(custody, matrix_native_runtime_observation())
            custody.begin_native_effect("nominated-rollout")
            row = geometry.run_anisotropic_feasibility_rollout(source=config.source, member=member,
                stage=geometry.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY, q=2,
                alpha_x_index=1, alpha_y_index=11, alpha_tilde_x=grid[1], alpha_tilde_y=grid[11],
                history_id="matrix-history.joint-increasing-coupling", seed_index=1,
                numerical_view=config.source.primary_view, schedule=config.source.feasibility_envelope.feasibility_schedule,
                scientific_seed_sha256=NOMINATED_PARENT_DRIVER)
        custody.retain("nominated-rollout", row)
    parent = MatrixSelectedParentOperand(f"{config.config_id}.operand", config.fingerprint(), "paper-nominated",
        ObjectIdentity.from_record(config.source.config_id, config.source), row, implementation_commit,
        ORIGINAL_NUMERICAL_QUALIFICATION, None)
    custody.retain("parent", parent)
    return read_selected_parent(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)


def select_alternate_current_parent(*, config: MatrixSelectedParentRequest, geometry_config: MatrixGeometryScanInput,
                                    rollout_id: str, artifact_writer: ExternalArtifactPlane, implementation_commit: str):
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    result = read_matrix_geometry_result(config=geometry_config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    report = result["report"]
    row = next((row for row in report.rollouts if row.rollout_id == rollout_id), None)
    if row is None or geometry_config.source != config.source:
        raise ValueError("Alternate selected parent is absent or uses another source")
    parent = MatrixSelectedParentOperand(f"{config.config_id}.operand", config.fingerprint(), "current-alternate",
        report.source_config, row, implementation_commit, report.numerical_qualification_report_sha256,
        ObjectIdentity.from_record(report.report_id, report), geometry_config)
    custody = _custody(config, artifact_writer, implementation_commit)
    prior = custody.load("parent", MatrixSelectedParentOperand)
    if prior is not None and prior != parent:
        raise ValueError("Alternate selected parent cannot replace an existing conditional instance")
    if prior is None:
        custody.retain("parent", parent)
    return read_selected_parent(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)


def prepare_selected_event(*, config_id: str, parent_request: MatrixSelectedParentRequest,
                           null_reference: MatrixGeometryScanInput, master_seed: int,
                           artifact_writer: ExternalArtifactPlane, implementation_commit: str,
                           prior_exposed_seed_sha256s: tuple[str, ...] = ()):
    if type(master_seed) is not int or not 0 <= master_seed < 2**256:
        raise ValueError("Selected event requires an unsigned 256-bit future master seed")
    parent, manifest = read_selected_parent(config=parent_request, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    protocol = selected_event_protocol(config_id=f"{config_id}.protocol", source=parent_request.source,
        parent=parent, parent_locator=manifest.materialization.relative_path)
    def seed(step, i, role):
        return MatrixSelectedFutureSeed(step, i, sha256(canonical_json_bytes(("selected-event.current-futures-v1", master_seed, parent.rollout.rng_seed_sha256, role, step, i))).hexdigest())
    return MatrixSelectedEventInput(config_id, parent_request, manifest, null_reference, protocol,
        tuple(seed(step, i, "branch") for step in CHECKPOINT_STEPS for i in range(64)),
        tuple(seed(0, i, "bridge") for i in range(1, 1025)), prior_exposed_seed_sha256s)


def prove_selected_event(*, config: MatrixSelectedEventInput, artifact_writer: ExternalArtifactPlane, implementation_commit: str):
    parent, manifest = read_selected_parent(config=config.parent_request, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    if manifest != config.parent or config.protocol != selected_event_protocol(config_id=config.protocol.config_id,
        source=config.source, parent=parent, parent_locator=manifest.materialization.relative_path):
        raise ValueError("Selected event changes its exact parent or locked mathematical protocol")
    geometry_proof = prove_matrix_geometry_scan(config.null_reference)
    return {"config_sha256": config.fingerprint(), "conditional_source_instance": parent.conditional_source_instance,
        "exposed_parent": True, "independent_selected_events": 1, "checkpoint_count": 7,
        "conditional_futures": 448, "forward_steps_per_future": 1024, "bridge_coarse_steps": 1024,
        "maximum_integration_steps": config.protocol.total_integration_steps, "memory_bytes": config.protocol.maximum_memory_bytes,
        "maximum_output_bytes": config.protocol.maximum_output_bytes, "checkpoint_steps": CHECKPOINT_STEPS,
        "registered_native_owner": geometry_proof["installed_source_capability"], "scientific_code_sha256": config.scientific_code_sha256,
        "environment_lock_sha256": config.environment_lock_sha256, "evidence_ceiling": "NON_PROMOTABLE",
        "scientific_execution_performed": False}


@dataclass(frozen=True, slots=True)
class MatrixSelectedReplayTransport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/selected-events/exact-replay-transport"
    replay_result: shooting.MatrixResponseShootingCommittorReplayResult
    coarse_noises_base64: tuple[str, ...]
    rng_stream: SixMatrixResponseRNGStreamReceipt

    def __post_init__(self):
        if len(self.coarse_noises_base64) != 1024:
            raise ValueError("Selected replay transport omits original coarse innovations")
        for value in self.coarse_noises_base64:
            raw = base64.b64decode(value, validate=True)
            if len(raw) != 1536 or base64.b64encode(raw).decode("ascii") != value:
                raise ValueError("Selected replay innovation has another exact complex128 geometry")

    def replay(self):
        import numpy as np
        checkpoints = self.replay_result.checkpoints
        states = tuple(native.state_from_shooting_checkpoint(c)[0] for c in checkpoints)
        noises = tuple(np.frombuffer(base64.b64decode(raw), dtype="<c16").reshape((2, 3, 4, 4)) for raw in self.coarse_noises_base64)
        if any(not np.isfinite(noise).all() for noise in noises):
            raise ValueError("Selected replay contains nonfinite original innovations")
        return native.SixMatrixResponseShootingCommittorReplayData(checkpoints, states, noises,
            states[-1], checkpoints[-1].rng_state_json, self.rng_stream,
            self.replay_result.observed_final_state_sha256)


def _restart_exact(replay, *, member, view, protocol):
    import json
    import numpy as np
    from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import baoab_step, fast_receiver
    from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import spectral_receiver
    state, rng = native.state_from_shooting_checkpoint(replay.checkpoints[3])
    while state.step_index < protocol.parent_total_steps:
        state = baoab_step(state, member=member, numerical_view=view,
            next_alpha_tilde_x=float(protocol.target_alpha_tilde_x), next_alpha_tilde_y=float(protocol.target_alpha_tilde_y), rng=rng)
        if state.step_index > 768 and state.step_index % 16 == 0:
            fast_receiver(state=state, member=member)
        if state.step_index > 768 and state.step_index % 256 == 0:
            spectral_receiver(receiver_prefix=f"spectrum.{protocol.parent_rollout_id}.restart.{state.step_index}", q=2, positions=state.positions)
    return bool(np.array_equal(state.positions, replay.final_state.positions)
        and np.array_equal(state.momenta, replay.final_state.momenta)
        and json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":")) == replay.final_rng_state_json)


def _typed_seed(seed, *, role, root_id, context, source, receipt):
    return SixMatrixResponseScientificSeedInput(role, root_id, context, seed.stream_index,
        seed.full_seed_sha256, source.object_fingerprint, source, receipt)


def run_selected_event(*, config: MatrixSelectedEventInput, artifact_writer: ExternalArtifactPlane,
                       project_root: Path, implementation_commit: str, progress=None):
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    prove_selected_event(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    authenticate_matrix_native_environment(config, project_root=project_root)
    parent, _ = read_selected_parent(config=config.parent_request, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    null_report = read_matrix_geometry_result(config=config.null_reference, artifact_writer=artifact_writer, implementation_commit=implementation_commit)["report"]
    custody = _custody(config, artifact_writer, implementation_commit)
    terminal = custody.load("terminal", shooting.MatrixResponseShootingCommittorTerminalReport)
    if terminal is not None:
        return read_selected_event_result(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    member = next(m for m in config.source.anisotropic_model.family_members if m.member_id == config.protocol.member_id)
    protocol = config.protocol
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        runtime = matrix_native_runtime_observation()
        retain_matrix_native_runtime(custody, runtime)
        transport = custody.load("replay", MatrixSelectedReplayTransport)
        if transport is None:
            custody.begin_native_effect("replay")
            replay = native.replay_selected_co_anneal_event(member=member, numerical_view=config.source.primary_view,
                parent_rollout_id=protocol.parent_rollout_id, parent_report_sha256=parent.fingerprint(),
                target_x=protocol.target_alpha_tilde_x, target_y=protocol.target_alpha_tilde_y,
                seed_index=protocol.seed_index, scientific_seed_sha256=protocol.parent_rng_seed_sha256,
                checkpoint_steps=CHECKPOINT_STEPS)
            replay_result = shooting.build_replay_result(replay=replay, config=protocol, implementation_commit=implementation_commit,
                parent_report=parent, checkpoint_restart_exact=_restart_exact(replay, member=member, view=config.source.primary_view, protocol=protocol))
            transport = MatrixSelectedReplayTransport(replay_result, tuple(base64.b64encode(noise.astype("<c16", copy=False).tobytes(order="C")).decode("ascii") for noise in replay.coarse_noises), replay.rng_stream)
            custody.retain("replay", transport)
        else:
            replay = transport.replay()
            replay_result = transport.replay_result
        if not replay_result.parent_replay_exact:
            raise ValueError("Selected parent replay/restart is numerically inadmissible; no alternate is selected")
        seed_export = custody.load("seed-export", MatrixSelectedEventInput)
        if seed_export is None:
            custody.retain("seed-export", config)
        seed_receipt = custody.store.read_by_receipt_id(config.run_id, "seed-export", f"receipt.{config.run_id}.seed-export.attempt-001")
        source = ObjectIdentity.from_record(config.config_id, config)
        receipt = ObjectIdentity.from_record(seed_receipt.receipt_id, seed_receipt)
        bridge = custody.load("bridge", shooting.MatrixResponseShootingCommittorBridgeResult)
        if bridge is None:
            custody.begin_native_effect("bridge")
            bridge = shooting.execute_brownian_bridge_replay(replay=replay, replay_result=replay_result,
                member=member, primary_view=config.source.primary_view, half_view=config.source.secondary_view, config=protocol,
                scientific_seed_inputs=tuple(_typed_seed(s, role="shooting-bridge", root_id=protocol.parent_rollout_id,
                    context=protocol.fingerprint(), source=source, receipt=receipt) for s in config.bridge_seeds))
            custody.retain("bridge", bridge)
        amplitude_null = custody.load("amplitude-null", shooting.MatrixResponseShootingCommittorAmplitudeNullResult)
        if amplitude_null is None:
            amplitude_null = shooting.execute_amplitude_conditioned_null(report=null_report, config=protocol, selected_parent=parent.rollout)
            custody.retain("amplitude-null", amplitude_null)
        cohorts = []
        linearization = None
        if bridge.disposition is shooting.MatrixResponseShootingCommittorNumericalDisposition.EVENT_PATH_NUMERICALLY_CONCORDANT:
            for checkpoint in replay.checkpoints:
                cohort_id = f"cohort-{checkpoint.parent_step}"
                cohort = custody.load(cohort_id, shooting.MatrixResponseShootingCommittorCohortResult)
                if cohort is None:
                    branches = []
                    for seed in (s for s in config.branch_seeds if s.checkpoint_step == checkpoint.parent_step):
                        task_id = f"branch-{checkpoint.parent_step}-{seed.stream_index:02d}"
                        branch = custody.load(task_id, shooting.MatrixResponseShootingCommittorBranchResult)
                        if branch is None:
                            preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
                            custody.begin_native_effect(task_id)
                            typed = _typed_seed(seed, role="shooting-branch", root_id=f"checkpoint-sha256.{checkpoint.combined_state_sha256}",
                                context=checkpoint.combined_state_sha256, source=source, receipt=receipt)
                            branch = shooting._run_branch_task(shooting._BranchTask(checkpoint, member, config.source.primary_view, protocol, seed.stream_index, typed))
                            custody.retain(task_id, branch)
                        _validate_branch(branch, checkpoint, seed)
                        branches.append(branch)
                        if progress is not None:
                            progress("conditional-futures", len(cohorts) * 64 + len(branches), 448)
                    cohort = shooting.reduce_shooting_cohort(checkpoint=checkpoint, branches=tuple(branches), config=protocol)
                    custody.retain(cohort_id, cohort)
                cohorts.append(cohort)
            linearization = custody.load("linearization", shooting.MatrixResponseShootingCommittorLinearizationResult)
            if linearization is None:
                custody.begin_native_effect("linearization")
                linearization = shooting.execute_local_linearization(checkpoint=replay.checkpoints[3], member=member, config=protocol)
                custody.retain("linearization", linearization)
        terminal = shooting.finalize_matrix_response_study_shooting_committor(implementation_commit=implementation_commit,
            config=protocol, replay=replay_result, bridge=bridge, cohorts=tuple(cohorts), linearization=linearization, amplitude_null=amplitude_null)
        custody.retain("terminal", terminal)
    return {"report": terminal, "summary": selected_event_summary(terminal, parent), "runtime_observation": runtime}


def _validate_branch(branch, checkpoint, seed):
    if (branch.checkpoint_id, branch.checkpoint_step, branch.branch_index, branch.seed_document_sha256) != (
        checkpoint.checkpoint_id, checkpoint.parent_step, seed.stream_index, seed.full_seed_sha256):
        raise ValueError("Selected branch changes its checkpoint or allocated effective driver")


def _validate_replay(transport, config, parent, implementation_commit):
    result = transport.replay_result
    if (result.config_fingerprint, result.implementation_commit, result.parent_implementation_commit,
        result.parent_report_sha256, result.parent_rollout_id, result.derived_rng_seed_sha256,
        result.expected_final_state_sha256) != (
        config.protocol.fingerprint(), implementation_commit, parent.implementation_commit,
        parent.fingerprint(), parent.rollout.rollout_id, parent.rollout.rng_seed_sha256,
        parent.rollout.final_state_sha256):
        raise ValueError("Selected replay changes its actual parent, code or protocol")
    if transport.rng_stream.derived_seed_sha256 != parent.rollout.rng_seed_sha256:
        raise ValueError("Selected replay transport changes its actual parent driver")


def selected_event_summary(report, parent):
    return {"report_sha256": report.fingerprint(), "conditional_source_instance": parent.conditional_source_instance,
        "independent_selected_events": 1, "requested_conditional_futures": 448,
        "numerical_axis": report.numerical_axis, "committor_morphology_axis": report.committor_morphology_axis,
        "residence_axis": report.residence_axis, "stability_axis": report.stability_axis,
        "amplitude_null_axis": report.amplitude_null_axis, "combined_interpretation": report.combined_interpretation,
        "scientific_terminal": report.scientific_terminal, "reason_codes": report.reason_codes,
        "condition_false_descendants": report.condition_false_descendants,
        "selection_exposed": True, "evidence_ceiling": "NON_PROMOTABLE"}


def read_selected_event_result(*, config: MatrixSelectedEventInput, artifact_writer: ExternalArtifactPlane, implementation_commit: str):
    prove_selected_event(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    parent, _ = read_selected_parent(config=config.parent_request, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    custody = _custody(config, artifact_writer, implementation_commit, read_only=True)
    terminal = custody.load("terminal", shooting.MatrixResponseShootingCommittorTerminalReport)
    if terminal is None:
        raise ValueError("Selected event has no complete terminal receipt; resume the same exact input")
    transport = custody.load("replay", MatrixSelectedReplayTransport)
    bridge = custody.load("bridge", shooting.MatrixResponseShootingCommittorBridgeResult)
    amplitude_null = custody.load("amplitude-null", shooting.MatrixResponseShootingCommittorAmplitudeNullResult)
    if transport is None or bridge is None or amplitude_null is None or custody.load("seed-export", MatrixSelectedEventInput) != config:
        raise ValueError("Selected-event closed output inventory is incomplete")
    _validate_replay(transport, config, parent, implementation_commit)
    if (bridge.config_fingerprint != config.protocol.fingerprint()
        or bridge.parent_rollout_id != parent.rollout.rollout_id
        or bridge.bridge_seed_aggregate_sha256 != sha256("".join(seed.full_seed_sha256 for seed in config.bridge_seeds).encode("ascii")).hexdigest()
        or amplitude_null.config_fingerprint != config.protocol.fingerprint()
        or terminal.implementation_commit != implementation_commit
        or terminal.parent_report_sha256 != parent.fingerprint()):
        raise ValueError("Selected-event results change their current parent/code/protocol")
    null_report = read_matrix_geometry_result(config=config.null_reference, artifact_writer=artifact_writer,
        implementation_commit=implementation_commit)["report"]
    if amplitude_null != shooting.execute_amplitude_conditioned_null(report=null_report, config=config.protocol, selected_parent=parent.rollout):
        raise ValueError("Selected-event amplitude null differs from its exact complete current reference")
    if (terminal.config_fingerprint, terminal.replay_fingerprint, terminal.bridge_fingerprint, terminal.amplitude_null_fingerprint) != (
        config.protocol.fingerprint(), transport.replay_result.fingerprint(), bridge.fingerprint(), amplitude_null.fingerprint()):
        raise ValueError("Selected-event terminal changes its exact scientific dependencies")
    cohorts = []
    linearization = None
    if bridge.disposition is shooting.MatrixResponseShootingCommittorNumericalDisposition.EVENT_PATH_NUMERICALLY_CONCORDANT:
        for checkpoint in transport.replay_result.checkpoints:
            cohort = custody.load(f"cohort-{checkpoint.parent_step}", shooting.MatrixResponseShootingCommittorCohortResult)
            if cohort is None or len(cohort.branches) != 64 or cohort.checkpoint != checkpoint or cohort.config_fingerprint != config.protocol.fingerprint():
                raise ValueError("Selected-event result omits a complete assigned checkpoint cohort")
            branches = []
            for seed in (s for s in config.branch_seeds if s.checkpoint_step == checkpoint.parent_step):
                branch = custody.load(f"branch-{checkpoint.parent_step}-{seed.stream_index:02d}", shooting.MatrixResponseShootingCommittorBranchResult)
                if branch is None:
                    raise ValueError("Selected-event result omits an assigned conditional future")
                _validate_branch(branch, checkpoint, seed)
                branches.append(branch)
            if cohort != shooting.reduce_shooting_cohort(checkpoint=checkpoint, branches=tuple(branches), config=config.protocol):
                raise ValueError("Selected-event cohort differs from its exact retained future census")
            cohorts.append(cohort)
        linearization = custody.load("linearization", shooting.MatrixResponseShootingCommittorLinearizationResult)
        if linearization is None:
            raise ValueError("Selected-event result omits its assigned local linearization")
        if (linearization.config_fingerprint, linearization.checkpoint_id, linearization.checkpoint_step) != (
            config.protocol.fingerprint(), transport.replay_result.checkpoints[3].checkpoint_id, 896):
            raise ValueError("Selected-event linearization changes its exact checkpoint/protocol")
    if terminal.cohort_fingerprints != tuple(c.fingerprint() for c in cohorts) or terminal.linearization_fingerprint != (None if linearization is None else linearization.fingerprint()):
        raise ValueError("Selected-event terminal changes its exact closed inventory")
    expected_terminal = shooting.finalize_matrix_response_study_shooting_committor(
        implementation_commit=implementation_commit, config=config.protocol, replay=transport.replay_result,
        bridge=bridge, cohorts=tuple(cohorts), linearization=linearization, amplitude_null=amplitude_null)
    if terminal != expected_terminal:
        raise ValueError("Selected-event terminal differs from its complete scientific reduction")
    return {"report": terminal, "summary": selected_event_summary(terminal, parent), "cohorts": tuple(cohorts), "amplitude_null": amplitude_null}


def run_selected_event_conformance(*, config: MatrixSelectedEventInput, checkpoint_step: int, branch_index: int,
                                   artifact_writer: ExternalArtifactPlane, project_root: Path, implementation_commit: str):
    """One exposed replay/future seam; it does not qualify the event or run its cohorts."""
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    prove_selected_event(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    authenticate_matrix_native_environment(config, project_root=project_root)
    if checkpoint_step not in CHECKPOINT_STEPS or type(branch_index) is not int or not 0 <= branch_index < 64:
        raise ValueError("Selected-event conformance changes its declared checkpoint/branch coordinates")
    parent, _ = read_selected_parent(config=config.parent_request, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    custody = _custody(config, artifact_writer, implementation_commit)
    member = next(m for m in config.source.anisotropic_model.family_members if m.member_id == config.protocol.member_id)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        retain_matrix_native_runtime(custody, matrix_native_runtime_observation())
        transport = custody.load("conformance-replay", MatrixSelectedReplayTransport)
        if transport is None:
            custody.begin_native_effect("conformance-replay")
            protocol = config.protocol
            replay = native.replay_selected_co_anneal_event(member=member, numerical_view=config.source.primary_view,
                parent_rollout_id=protocol.parent_rollout_id, parent_report_sha256=parent.fingerprint(),
                target_x=protocol.target_alpha_tilde_x, target_y=protocol.target_alpha_tilde_y,
                seed_index=protocol.seed_index, scientific_seed_sha256=protocol.parent_rng_seed_sha256, checkpoint_steps=CHECKPOINT_STEPS)
            result = shooting.build_replay_result(replay=replay, config=protocol, implementation_commit=implementation_commit,
                parent_report=parent, checkpoint_restart_exact=_restart_exact(replay, member=member, view=config.source.primary_view, protocol=protocol))
            transport = MatrixSelectedReplayTransport(result,
                tuple(base64.b64encode(noise.astype("<c16", copy=False).tobytes(order="C")).decode("ascii") for noise in replay.coarse_noises), replay.rng_stream)
            custody.retain("conformance-replay", transport)
        _validate_replay(transport, config, parent, implementation_commit)
        if not transport.replay_result.parent_replay_exact:
            raise ValueError("Selected-event conformance parent replay/restart is inadmissible")
        seed = next(s for s in config.branch_seeds if (s.checkpoint_step, s.stream_index) == (checkpoint_step, branch_index))
        task_id = f"conformance-branch-{checkpoint_step}-{branch_index:02d}"
        branch = custody.load(task_id, shooting.MatrixResponseShootingCommittorBranchResult)
        checkpoint = next(c for c in transport.replay_result.checkpoints if c.parent_step == checkpoint_step)
        if branch is None:
            if custody.load("conformance-seed-export", MatrixSelectedEventInput) is None:
                custody.retain("conformance-seed-export", config)
            export = custody.store.read_by_receipt_id(config.run_id, "conformance-seed-export", f"receipt.{config.run_id}.conformance-seed-export.attempt-001")
            typed = _typed_seed(seed, role="shooting-branch", root_id=f"checkpoint-sha256.{checkpoint.combined_state_sha256}",
                context=checkpoint.combined_state_sha256, source=ObjectIdentity.from_record(config.config_id, config),
                receipt=ObjectIdentity.from_record(export.receipt_id, export))
            custody.begin_native_effect(task_id)
            branch = shooting._run_branch_task(shooting._BranchTask(checkpoint, member, config.source.primary_view, config.protocol, branch_index, typed))
            custody.retain(task_id, branch)
        _validate_branch(branch, checkpoint, seed)
    return {"branch": branch, "replay": transport.replay_result,
        "evidence_ceiling": "NON_PROMOTABLE", "full_investigation_performed": False,
        "event_qualification_performed": False}
