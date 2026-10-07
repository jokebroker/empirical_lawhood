"""Shipped uniform electron gas analytic config selects native and evaluator providers through the extension authoring contract."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.task_resources import (
    progress_resource_envelope,
)
from empirical_lawhood.adapters.methods.uniform_electron_gas_analytic_reference.executable_binding import BINDING as EVALUATOR_BINDING
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_authoring import UniformElectronGasAnalyticCandidateContextProvider, build_uniform_electron_gas_analytic_authoring
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_design import BUDGET
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_science import check_analytic_panel, generate_analytic_panel
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.executable_binding import BINDING as NATIVE_BINDING
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.native_quickstart import UniformElectronGasAnalyticReferenceConfig
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.runtime_provider import native_task_id
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/electron-gas-response/config.json"


def _config() -> UniformElectronGasAnalyticReferenceConfig:
    return decode_canonical_bytes(
        CONFIG.read_bytes(), UniformElectronGasAnalyticReferenceConfig, maximum_bytes=32768
    )


def test_shipped_analytic_unit_compiles_and_selects_both_providers(
    tmp_path: Path,
) -> None:
    config = _config()
    experiment_id = "uniform-electron-gas-candidate-integration-check"
    bundle = build_uniform_electron_gas_analytic_authoring(
        config=config, experiment_id=experiment_id, implementation_sha256="0" * 64
    )
    authoring = tmp_path / "authoring.json"
    authoring.write_bytes(bundle.authoring.canonical_bytes())
    payloads = []
    decoders = []
    for index, record in enumerate(bundle.payloads):
        path = tmp_path / f"payload-{index}.json"
        path.write_bytes(record.canonical_bytes())
        payloads.append(path)
    for index, record in enumerate(bundle.decoder_registrations):
        path = tmp_path / f"decoder-{index}.json"
        path.write_bytes(record.canonical_bytes())
        decoders.append(path)
    factories = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    api = EmpiricalLawhoodApi(
        repo_root=ROOT,
        external_root=None,
        candidate_context_provider=UniformElectronGasAnalyticCandidateContextProvider(bundle),
        candidate_capability_catalog=bundle.catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=factories.aggregate,
        executable_factory_registry=factories,
        study_extension_codec_registry=EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    )
    result = api.compile_candidate(
        CompileCandidateRequest(authoring, tuple(payloads), tuple(decoders))
    )
    assert result.succeeded and result.payload is not None
    report = result.payload.report
    assert report.candidate is not None
    candidate = report.candidate
    base = candidate.base_candidate.base_candidate
    issued, _, _ = project_standard_study_extensions(
        candidate=candidate,
        extension_payload_bytes=tuple(
            record.canonical_bytes() for record in bundle.payloads
        ),
        decoder_registrations=bundle.decoder_registrations,
        extension_materializations=report.extension_materializations,
    )
    resources = progress_resource_envelope(
        prefix=experiment_id,
        design=config,
        protocol=base.protocol,
        issued_extensions=ObjectIdentity.from_record(
            issued.issued_extension_set_id, issued
        ),
        cpu_seconds=BUDGET.wall_time_seconds,
        native_preparations=(
            (
                native_task_id(config),
                config.independent_unit_id,
                f"{config.independent_unit_id}.analytic-source",
            ),
        ),
        progress_counter="uniform-electron-gas-completed-analytic-acquisition",
        heartbeat_seconds=Decimal(60),
        maximum_concurrency=1,
        maximum_memory_bytes=BUDGET.memory_bytes,
        terminal_failure_codes=("UNIFORM_ELECTRON_GAS_ANALYTIC_SOURCE_FAILURE",),
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
    assert len(execution.tasks) == 2
    assert sum(len(task.outputs) for task in execution.tasks) == 3
    registry = bundle.standard_context.base.registry
    native = factories.provider_factory(NATIVE_BINDING.binding_id).build_provider(
        registry=registry, records=(config,), platform_ports=()
    )
    evaluation_config = next(
        record
        for record in bundle.payloads
        if record.SCHEMA == EVALUATOR_BINDING.required_issued_payload_schemas[0]
    )
    evaluator = factories.provider_factory(EVALUATOR_BINDING.binding_id).build_provider(
        registry=registry, records=(evaluation_config,), platform_ports=()
    )
    assert len(native.external_inputs(execution)) == 2
    assert len(evaluator.external_inputs(execution)) == 1
    assert (
        native.runners(registry)[0].manifest.capability_key
        == NATIVE_BINDING.capability_key
    )
    assert (
        evaluator.runners(registry)[0].manifest.capability_key
        == EVALUATOR_BINDING.capability_key
    )
    assert evaluator.scientific_adjudication_contract(
        registry, execution
    ).output_id.endswith(".scientific-adjudication")


def test_analytic_falsifier_uses_realized_action_and_one_acquisition() -> None:
    config = _config()
    panel = generate_analytic_panel(config)
    assert len(panel.rows) == 20
    assert {panel.independent_unit_id} == {config.independent_unit_id}
    assert check_analytic_panel(config, panel).passed
    first = panel.rows[0]
    changed = replace(
        panel,
        rows=(
            replace(
                first, realized_A_T=tuple(value / 2 for value in first.realized_A_T)
            ),
            *panel.rows[1:],
        ),
    )
    result = check_analytic_panel(config, changed)
    assert not result.passed
    assert "REALIZED_ACTION_DRIFT" in result.reason_codes
    wrong_sign = replace(
        panel,
        rows=tuple(
            replace(row, current_density=tuple(-value for value in row.current_density))
            for row in panel.rows
        ),
    )
    result = check_analytic_panel(config, wrong_sign)
    assert not result.passed
    assert "NONPOSITIVE_KERNEL_LOWER_BOUND" in result.reason_codes
