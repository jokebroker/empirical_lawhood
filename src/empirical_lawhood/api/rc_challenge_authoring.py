"""Public application handoff for separately issued current V3 RC phases."""

from decimal import Decimal
from dataclasses import dataclass, replace
from pathlib import Path

from empirical_lawhood.adapters.composition.task_resources import progress_resource_envelope
from empirical_lawhood.adapters.simulator_morphism_challenges.authoring import SimulatorMorphismChallengeExternalRecord
from empirical_lawhood.adapters.simulator_morphism_challenges.composition import RCChallengeCandidateContextProvider, SimulatorMorphismChallengePhaseComposition, compose_simulator_morphism_challenges_phase, runtime_platform_ports
from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeConfig
from empirical_lawhood.adapters.simulator_morphism_challenges.numeric_inputs import RCChallengeNumericInput
from empirical_lawhood.adapters.simulator_morphism_challenges.issued_inputs import RCChallengeIssuedInputs
from empirical_lawhood.adapters.simulator_morphism_challenges.retained_results import RCChallengeRetainedResult
from empirical_lawhood.adapters.simulator_morphism_challenges.runtime_provider import SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS
from empirical_lawhood.adapters.simulator_morphism_challenges.specification import current_rc_challenge_formal_register, current_rc_challenge_specification
from empirical_lawhood.api.authoring_handoff import create_authoring_api, write_exclusive_record, resolve_authoring_directory, preflight_authoring_output_directory
from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.api.rc_challenge_results import authenticate_rc_challenge_result_authority
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes, MAX_RUNTIME_PLAN_JSON_BYTES
from empirical_lawhood.infrastructure.source_closure import capture_clean_target_closure
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.api.integration_artifact_profiles import INTEGRATION_ARTIFACT_PROFILE_VALIDATORS
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure
from empirical_lawhood.runtime.artifacts import ArtifactWriter
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.study_issue import MAX_COMPOSED_PROGRAMME_CONTROL_BYTES, project_standard_study_extensions
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort, LayeredCampaignRuntimeProviderResolver
from empirical_lawhood.runtime.providers import CampaignRuntimeProviderRegistry
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.plans import CandidateExecutionPlan


RC_CHALLENGE_ARTIFACT_PROFILE_VALIDATORS = INTEGRATION_ARTIFACT_PROFILE_VALIDATORS


def author_rc_challenge_phase(*, root: Path, config: SimulatorMorphismChallengeConfig,
                              external_records: tuple[SimulatorMorphismChallengeExternalRecord, ...],
                              output_dir: Path, numeric_input: RCChallengeNumericInput | None = None,
                              artifact_writer: ExternalArtifactPlane,
                              prerequisite_custody: tuple[RCChallengeRetainedResult, ...] = ()) -> dict[str, object]:
    """Compile one fixed phase with exact current source, input custody and public factories.

    This exports candidate/preissue records. It does not issue, execute, reveal,
    manufacture upstream qualifications, or select an implicit latest result.
    """
    output_dir = preflight_authoring_output_directory(directory=output_dir, repo_root=root,
        artifact_writer=artifact_writer)
    closure = capture_clean_target_closure(root, f"{config.config_id}.source-closure")
    source_files = {path: read_bounded_bytes(root / path, maximum_bytes=2 * 1024**2)
                    for path in SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS}
    if sum(len(payload) for payload in source_files.values()) > 32 * 1024**2:
        raise ValueError("RC implementation source closure exceeds its bounded scan")
    composition = compose_simulator_morphism_challenges_phase(
        config=config, external_records=external_records, source_files=source_files,
        register=current_rc_challenge_formal_register(), numeric_input=numeric_input,
        implementation_sha256=closure.implementation_sha256,
        artifact_writer=artifact_writer,
        prerequisite_custody=prerequisite_custody,
        prerequisite_access_verifier=lambda retained: authenticate_rc_challenge_result_authority(retained, writer=artifact_writer),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_authoring_output_directory(directory=output_dir, repo_root=root,
        artifact_writer=artifact_writer, allow_existing_empty=True)
    with report_incomplete_output(output_dir) as progress:
        save = lambda name, record: write_exclusive_record(output_dir, name, record)
        save("config.json", config)
        save("operative-specification.json", current_rc_challenge_specification())
        save("issued-inputs.json", composition.issued_inputs)
        authoring_path = save("authoring.json", composition.authoring)
        save("base-authoring.json", composition.authoring.base)
        # Source/config qualification payloads remain explicit public compiler inputs.
        for index, record in enumerate((config, *(v.record for v in external_records),
                                        *composition.bundle.base.source_configs, *composition.bundle.base.qualifications)):
            save(f"candidate-input-{index:02d}.json", record)
        payload_paths = tuple(save(f"payload-{index:02d}.json", record) for index, record in enumerate(composition.payloads))
        decoder_paths = tuple(save(f"decoder-{index:02d}.json", record) for index, record in enumerate(composition.decoder_registrations))
        progress.stage = "candidate compilation"
        api = create_authoring_api(repo_root=root, candidate_context_provider=RCChallengeCandidateContextProvider(composition),
                                   candidate_capability_catalog=composition.catalog)
        compiled = api.compile_candidate(CompileCandidateRequest(authoring_path, payload_paths, decoder_paths))
        if not compiled.succeeded or compiled.payload is None or compiled.payload.report.candidate is None:
            raise RuntimeError(f"RC challenge candidate refused: {compiled.reason_codes}; {compiled.errors}")
        report = compiled.payload.report
        progress.stage = "plan projection"
        candidate, resources, projected, lowered = _project_phase(composition, closure, report)
        for name, record in (("candidate-report.json", report), ("candidate.json", candidate),
                             ("base-candidate.json", candidate.base_candidate), ("source-closure.json", closure),
                             ("resources.json", resources), ("preissue-run-plan.json", projected),
                             ("preissue-execution-plan.json", lowered)):
            save(name, record)
        return {"phase": config.phase.value, "candidate_sha256": candidate.fingerprint(),
                "implementation_commit": closure.implementation_commit,
                "implementation_source_closure_sha256": composition.issued_inputs.implementation_source_closure_sha256,
                "task_count": len(lowered.tasks), "output_count": sum(len(task.outputs) for task in lowered.tasks),
                "requested_independent_units": len(config.unit_ids), "scale_cells": config.scale_cells,
                "scientific_execution_performed": False, "output_dir": str(output_dir)}


def _project_phase(composition, closure, report):
    config = composition.bundle.base.config
    candidate = report.candidate
    base = candidate.base_candidate.base_candidate
    extensions, _, _ = project_standard_study_extensions(
        candidate=candidate, extension_payload_bytes=tuple(record.canonical_bytes() for record in composition.payloads),
        decoder_registrations=composition.decoder_registrations, extension_materializations=report.extension_materializations)
    protocol = base.protocol
    # Existing task budgets provide the full derived graph census; no native
    # computation or campaign is run to manufacture resource estimates.
    resources = progress_resource_envelope(
        prefix=config.config_id, design=config, protocol=protocol,
        issued_extensions=ObjectIdentity.from_record(extensions.issued_extension_set_id, extensions),
        cpu_seconds=sum(step.resource_budget.wall_time_seconds for step in protocol.steps),
        native_preparations=tuple((step.step_id,
            next((unit for unit in config.unit_ids if step.step_id.endswith(unit)), config.config_id),
            f"{config.config_id}.bounded-task.{step.step_id}") for step in protocol.steps),
        progress_counter="rc-challenge-completed-task",
        heartbeat_seconds=Decimal(max(step.resource_budget.wall_time_seconds for step in protocol.steps)), maximum_concurrency=1,
        maximum_memory_bytes=max(step.resource_budget.memory_bytes for step in protocol.steps),
        terminal_failure_codes=("RC_CHALLENGE_NUMERICAL_FAILURE",),
    )
    attempts = {step.step_id: step.maximum_attempts for step in protocol.steps}
    cells = tuple(replace(cell, maximum_attempts=attempts[cell.task_id]) for cell in resources.task_cells)
    resources = replace(resources, task_cells=cells,
        child_token_limits=tuple(replace(limit,
            physical_execution_token_limit=sum(cell.physical_execution_cost * cell.maximum_attempts for cell in cells if cell.child_id == limit.child_id),
            retry_token_limit=sum(cell.retry_token_cost * (cell.maximum_attempts - 1) for cell in cells if cell.child_id == limit.child_id))
            for limit in resources.child_token_limits))
    projected = compile_preissue_run_plan(
        run_plan_id=f"{config.config_id}.run-{composition.issued_inputs.fingerprint()[:32]}", candidate_record=base,
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        registry=composition.bundle.base.registry, implementation_commit=closure.implementation_commit,
        issued_extension_set=extensions, resource_envelope=resources, jit_census=None, jit_manifest=None)
    lowered = lower_run_plan(projected, composition.bundle.base.registry)
    return candidate, resources, projected, lowered


@dataclass(frozen=True, slots=True)
class RCChallengeApplicationHandoff:
    directory: Path
    composition: SimulatorMorphismChallengePhaseComposition
    projection: CandidateExecutionPlan
    resources: ExecutionResourceEnvelopeSpec
    platform_ports: tuple[ExecutablePlatformPort, ...]


def load_rc_challenge_handoff(*, directory: Path, root: Path,
                             storage_profile: OperatorStorageProfile,
                             artifact_writer: ArtifactWriter) -> RCChallengeApplicationHandoff:
    """Load exactly one selected retained phase; no search for latest or source fallback."""
    selected, _ = resolve_authoring_directory(directory, repo_root=root, storage_profile=storage_profile,
                                               escape_message="RC challenge handoff is outside the selected guarded external root")
    issued = decode_canonical_bytes(read_bounded_bytes(selected / "issued-inputs.json", maximum_bytes=8 * 1024**2),
                                    RCChallengeIssuedInputs, maximum_bytes=8 * 1024**2)
    closure = decode_canonical_bytes(read_bounded_bytes(selected / "source-closure.json", maximum_bytes=65536),
                                     ImplementationSourceClosure, maximum_bytes=65536)
    if capture_clean_target_closure(root, closure.source_closure_id) != closure:
        raise ValueError("RC retained authoring actual source archive has changed")
    source_files = {path: read_bounded_bytes(root / path, maximum_bytes=2 * 1024**2)
                    for path in SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS}
    composition = compose_simulator_morphism_challenges_phase(
        config=issued.config, external_records=issued.external_records(), source_files=source_files,
        register=current_rc_challenge_formal_register(), numeric_input=issued.numeric_input,
        implementation_sha256=closure.implementation_sha256,
        artifact_writer=artifact_writer, prerequisite_custody=issued.prerequisite_custody,
        prerequisite_access_verifier=lambda retained: authenticate_rc_challenge_result_authority(retained, writer=artifact_writer))
    if composition.issued_inputs != issued:
        raise ValueError("retained RC phase source/input reconstruction differs from the exact current owners")
    def require_record(filename, record):
        limit = MAX_RUNTIME_PLAN_JSON_BYTES if filename in ("preissue-run-plan.json", "preissue-execution-plan.json") else MAX_COMPOSED_PROGRAMME_CONTROL_BYTES
        raw = read_bounded_bytes(selected / filename, maximum_bytes=limit)
        if decode_canonical_bytes(raw, type(record), maximum_bytes=limit) != record:
            raise ValueError("RC retained authoring or exact projection reconstruction differs")

    for filename, record in (("config.json", issued.config),
                             ("operative-specification.json", current_rc_challenge_specification()),
                             ("authoring.json", composition.authoring), ("base-authoring.json", composition.authoring.base),
                             *((f"payload-{index:02d}.json", record) for index, record in enumerate(composition.payloads)),
                             *((f"decoder-{index:02d}.json", record) for index, record in enumerate(composition.decoder_registrations))):
        require_record(filename, record)
    api = create_authoring_api(repo_root=root, candidate_context_provider=RCChallengeCandidateContextProvider(composition),
                               candidate_capability_catalog=composition.catalog)
    compiled = api.compile_candidate(CompileCandidateRequest(selected / "authoring.json",
        tuple(selected / f"payload-{index:02d}.json" for index in range(len(composition.payloads))),
        tuple(selected / f"decoder-{index:02d}.json" for index in range(len(composition.decoder_registrations)))))
    if not compiled.succeeded or compiled.payload is None or compiled.payload.report.candidate is None:
        raise ValueError("RC retained current candidate recompilation refused")
    report = compiled.payload.report
    candidate, resources, projected, projection = _project_phase(composition, closure, report)
    for filename, record in (("candidate-report.json", report), ("candidate.json", candidate),
                             ("base-candidate.json", candidate.base_candidate), ("resources.json", resources),
                             ("preissue-run-plan.json", projected), ("preissue-execution-plan.json", projection)):
        require_record(filename, record)
    return RCChallengeApplicationHandoff(selected, composition, projection, resources,
        runtime_platform_ports(config=issued.config, source_files=source_files, artifact_writer=artifact_writer,
            prerequisite_access_verifier=lambda retained: authenticate_rc_challenge_result_authority(retained, writer=artifact_writer)))


def prove_rc_challenge_handoff(handoff: RCChallengeApplicationHandoff) -> dict[str, object]:
    """Prove installed reconstruction/output/resource coverage without executing tasks."""
    from empirical_lawhood.adapters.composition.generated_executable_bindings import EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY

    registry = handoff.composition.bundle.base.registry
    resolver = LayeredCampaignRuntimeProviderResolver(precomposed=CampaignRuntimeProviderRegistry(()),
                                                       factories=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY)
    provider = resolver.resolve(registry, decoded_records=tuple(sorted(handoff.composition.payloads, key=lambda value: value.SCHEMA)),
                                 platform_ports=tuple(sorted(handoff.platform_ports, key=lambda value: value.port_key)))
    from empirical_lawhood.api.integration_proof import prove_provider_projection

    proof = prove_provider_projection(provider=provider, registry=registry, projection=handoff.projection,
        resources=handoff.resources, proof_id=f"{handoff.projection.source_plan.object_id}.rc-proof")
    return {"phase": handoff.composition.bundle.base.config.phase.value,
            **proof,
            "implementation_source_closure_sha256": handoff.composition.issued_inputs.implementation_source_closure_sha256}
