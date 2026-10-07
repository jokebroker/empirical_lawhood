'Shipped ambient pressure superconductor gauge covariant response method chart reaches strict candidate materialization and rejects false response.'

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

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
from empirical_lawhood.adapters.methods.synthetic_material_method.executable_binding import (
    BINDING as EVALUATOR_BINDING,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_gauge_covariant_response import (
    run_gauge_covariant_response_conformance,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.executable_binding import (
    BINDING as PRODUCER_BINDING,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_authoring import (
    SyntheticMaterialResponseMethodCandidateContextProvider,
    build_synthetic_material_response_method_authoring,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_contracts import (
    SyntheticMaterialResponseMethodPanel,
    check_synthetic_material_response_method_conformance,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_design import (
    BUDGET,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_packet import (
    load_synthetic_material_response_config,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_provider import (
    native_task_id,
)
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/lattice-pairing-method/authoring.json"


def test_shipped_synthetic_material_response_method_compiles_and_selects_exact_provider_chain(
    tmp_path: Path,
) -> None:
    config = load_synthetic_material_response_config(CONFIG)
    experiment_id = 'ambient-pressure-superconductor-gauge-covariant-response-method-integration-001'
    bundle = build_synthetic_material_response_method_authoring(
        config=config, experiment_id=experiment_id, implementation_sha256="0" * 64
    )
    authoring = tmp_path / "authoring.json"
    authoring.write_bytes(bundle.authoring.canonical_bytes())
    payloads = []
    decoders = []
    for index, record in enumerate(bundle.payloads):
        path = tmp_path / f'payload-{index}.json'
        path.write_bytes(record.canonical_bytes())
        payloads.append(path)
    for index, record in enumerate(bundle.decoder_registrations):
        path = tmp_path / f'decoder-{index}.json'
        path.write_bytes(record.canonical_bytes())
        decoders.append(path)
    factories = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    api = EmpiricalLawhoodApi(
        repo_root=ROOT,
        external_root=None,
        candidate_context_provider=SyntheticMaterialResponseMethodCandidateContextProvider(
            bundle
        ),
        candidate_capability_catalog=bundle.catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=factories.aggregate,
        executable_factory_registry=factories,
        study_extension_codec_registry=EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    )
    compiled = api.compile_candidate(
        CompileCandidateRequest(authoring, tuple(payloads), tuple(decoders))
    )
    assert compiled.succeeded and compiled.payload is not None
    report = compiled.payload.report
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
                f'{config.independent_unit_id}.method-suite',
            ),
        ),
        progress_counter='ambient-pressure-superconductor-completed-gauge-covariant-response-method-suite',
        heartbeat_seconds=Decimal(60),
        maximum_concurrency=1,
        maximum_memory_bytes=BUDGET.memory_bytes,
        terminal_failure_codes=("SYNTHETIC_MATERIAL_RESPONSE_METHOD_FAILURE",),
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
    producer = factories.provider_factory(PRODUCER_BINDING.binding_id).build_provider(
        registry=registry, records=(config,), platform_ports=()
    )
    evaluator_config = next(
        record
        for record in bundle.payloads
        if record.SCHEMA == EVALUATOR_BINDING.required_issued_payload_schemas[0]
    )
    evaluator = factories.provider_factory(EVALUATOR_BINDING.binding_id).build_provider(
        registry=registry, records=(evaluator_config,), platform_ports=()
    )
    producer_inputs = producer.external_inputs(execution)
    assert len(producer_inputs) == 2
    assert tuple(value.logical_artifact_id for value in producer_inputs) == tuple(
        sorted(value.logical_artifact_id for value in producer_inputs)
    )
    assert len(evaluator.external_inputs(execution)) == 1
    assert (
        producer.runners(registry)[0].manifest.capability_key
        == PRODUCER_BINDING.capability_key
    )
    assert (
        evaluator.runners(registry)[0].manifest.capability_key
        == EVALUATOR_BINDING.capability_key
    )
    assert evaluator.scientific_adjudication_contract(
        registry, execution
    ).output_id.endswith(".scientific-adjudication")
    assert (
        bundle.authoring.base.draft.experiment.independent_unit_id
        == 'ambient-pressure-superconductor-complete-synthetic-method-suite'
    )
    assert len(bundle.authoring.base.draft.system.numerical_views) == 2


def test_synthetic_material_response_method_science_and_sealed_falsifier_use_shipped_chart() -> (
    None
):
    config = load_synthetic_material_response_config(CONFIG)
    assert (
        config.requested_clock,
        config.accepted_clock,
        config.applied_clock,
        config.receiver_clock,
    ) == (0, 1, 2, 3)
    assert config.formalism.operating_temperature_K == Decimal(300)
    assert config.operating_pressure_Pa == Decimal(101325)
    conformance, observations = run_gauge_covariant_response_conformance()
    panel = SyntheticMaterialResponseMethodPanel(
        f'{config.config_id}.panel',
        config.independent_unit_id,
        config.fingerprint(),
        conformance,
        observations,
        True,
    )
    assert check_synthetic_material_response_method_conformance(config, panel).passed
    assert conformance.material_result_count == 0
    assert conformance.maximum_evidence_scope_id == "evidence-scope.method-fixture-only"
    by_name = {row.fixture_id.rsplit("-", 1)[-1]: row for row in observations}
    positive = next(row for row in observations if row.fixture_id.endswith("-positive"))
    normal = next(row for row in observations if row.fixture_id.endswith("-normal"))
    assert positive.j_plus_eV_per_link * positive.probe_amplitude < 0
    assert positive.stiffness_interval_lower_eV_per_link > 0
    assert (
        abs(normal.normal_total_eV_per_link)
        < config.fixture_suite.normal_cancellation_absolute_limit
    )
    assert (
        positive.ward_absolute_residual_eV < config.fixture_suite.ward_absolute_limit_eV
    )
    assert len(by_name) >= 7  # Names are nested conditions, never independent units.

    # Preserve the claimed ACCEPT and all identities, but reverse the signed current
    # and repair its digest. The evaluator must recompute the physical falsifier.
    wrong_sign = replace(positive, j_plus_eV_per_link=abs(positive.j_plus_eV_per_link))
    changed_observations = tuple(
        wrong_sign if row == positive else row for row in observations
    )
    changed_results = tuple(
        replace(row, observation_sha256=wrong_sign.fingerprint())
        if row.fixture_id == positive.fixture_id
        else row
        for row in conformance.fixture_results
    )
    forged = replace(
        panel,
        conformance=replace(conformance, fixture_results=changed_results),
        observations=changed_observations,
    )
    with pytest.raises(ValueError, match="re-evaluation"):
        check_synthetic_material_response_method_conformance(config, forged)

    with pytest.raises(ValueError, match="ambient pressure"):
        replace(config, operating_pressure_Pa=Decimal(200000))
