"""The shipped RC study selects native and evaluator providers through candidate materialization."""

from decimal import Decimal
from pathlib import Path

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.rc_ladder_response.authoring import ResistorCapacitorCandidateContextProvider, build_rc_authoring
from empirical_lawhood.adapters.composition.rc_ladder_response.design import BUDGET
from empirical_lawhood.adapters.composition.task_resources import (
    progress_resource_envelope,
)
from empirical_lawhood.adapters.methods.rc_ladder_numerical_comparison.executable_binding import BINDING as EVALUATOR_BINDING
from empirical_lawhood.adapters.simulators.rc_ladder_response.campaign import VIEWS
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderStudyConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.executable_binding import BINDING as NATIVE_BINDING
from empirical_lawhood.adapters.simulators.rc_ladder_response.runtime_provider import native_task_id
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/rc-ladder-response/study.json"


def _record_paths(
    directory: Path, stem: str, records: tuple[CanonicalRecord, ...]
) -> tuple[Path, ...]:
    paths = []
    for index, record in enumerate(records):
        path = directory / f"{stem}-{index}.json"
        path.write_bytes(record.canonical_bytes())
        paths.append(path)
    return tuple(paths)


def test_shipped_rc_study_compiles_and_selects_three_exact_runners(tmp_path: Path) -> None:
    study = decode_canonical_bytes(STUDY.read_bytes(), ResistorCapacitorLadderStudyConfig, maximum_bytes=65536)
    experiment_id = "rc-candidate-integration-check"
    bundle = build_rc_authoring(
        study=study, experiment_id=experiment_id, implementation_sha256="0" * 64
    )
    authoring = tmp_path / "authoring.json"
    authoring.write_bytes(bundle.authoring.canonical_bytes())
    payload_paths = _record_paths(tmp_path, "payload", bundle.payloads)
    decoder_paths = _record_paths(tmp_path, "decoder", bundle.decoder_registrations)
    factories = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    api = EmpiricalLawhoodApi(
        repo_root=ROOT,
        external_root=None,
        candidate_context_provider=ResistorCapacitorCandidateContextProvider(bundle),
        candidate_capability_catalog=bundle.catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=factories.aggregate,
        executable_factory_registry=factories,
        study_extension_codec_registry=EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    )
    result = api.compile_candidate(
        CompileCandidateRequest(authoring, payload_paths, decoder_paths)
    )
    assert result.succeeded and result.payload is not None
    report = result.payload.report
    assert report.candidate is not None
    candidate = report.candidate
    base = candidate.base_candidate.base_candidate
    issued, _, _ = project_standard_study_extensions(
        candidate=candidate,
        extension_payload_bytes=tuple(record.canonical_bytes() for record in bundle.payloads),
        decoder_registrations=bundle.decoder_registrations,
        extension_materializations=report.extension_materializations,
    )
    resources = progress_resource_envelope(
        prefix=experiment_id,
        design=study,
        protocol=base.protocol,
        issued_extensions=ObjectIdentity.from_record(issued.issued_extension_set_id, issued),
        cpu_seconds=BUDGET.wall_time_seconds,
        native_preparations=tuple(
            (native_task_id(study, view), study.unit_id, f"{study.unit_id}.frozen-model")
            for view in VIEWS
        ),
        progress_counter="rc-completed-solver-view",
        heartbeat_seconds=Decimal(60),
        maximum_concurrency=1,
        maximum_memory_bytes=BUDGET.memory_bytes,
        terminal_failure_codes=("RC_NATIVE_NUMERICAL_FAILURE",),
    )
    plan = compile_preissue_run_plan(
        run_plan_id=experiment_id,
        candidate_record=base,
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        registry=bundle.standard_context.base.registry,
        implementation_commit="0" * 40,
        issued_extension_set=issued,
        resource_envelope=resources,
        jit_census=None,
        jit_manifest=None,
    )
    execution = lower_run_plan(plan, bundle.standard_context.base.registry)
    assert len(execution.tasks) == 3
    assert sum(len(task.outputs) for task in execution.tasks) == 4
    registry = bundle.standard_context.base.registry
    native = factories.provider_factory(NATIVE_BINDING.binding_id).build_provider(
        registry=registry, records=(study,), platform_ports=()
    )
    evaluator = factories.provider_factory(EVALUATOR_BINDING.binding_id).build_provider(
        registry=registry, records=(bundle.payloads[0],), platform_ports=()
    )
    assert len(native.external_inputs(execution)) == 2
    assert len(evaluator.external_inputs(execution)) == 1
    assert native.runners(registry)[0].manifest.capability_key == NATIVE_BINDING.capability_key
    assert evaluator.runners(registry)[0].manifest.capability_key == EVALUATOR_BINDING.capability_key
    assert evaluator.scientific_adjudication_contract(registry, execution).output_id.endswith(
        ".scientific-adjudication"
    )
