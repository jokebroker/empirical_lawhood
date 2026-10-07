"""Exposed software fixtures for V3 custody/reconstruction and scientific controls."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from decimal import Decimal
from io import BytesIO
from time import perf_counter

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.simulator_morphism_challenges.observer import DenseObserverOperator, _rank_forecast
from empirical_lawhood.adapters.simulator_morphism_challenges.authoring import simulator_morphism_challenges_phase_registry
from empirical_lawhood.adapters.simulator_morphism_challenges.composition import compose_simulator_morphism_challenges_phase, RCChallengeCandidateContextProvider
from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeDisorderFamily as Family, SimulatorMorphismChallengePhase as Phase
from empirical_lawhood.adapters.simulator_morphism_challenges.descriptors import default_config, generate_descriptor
from empirical_lawhood.adapters.simulator_morphism_challenges.executable_binding import BINDINGS, EXECUTABLE_BINDING_FACTORIES
from empirical_lawhood.adapters.simulator_morphism_challenges.extension_bundle import INSTALLED_IMPLEMENTATION_SHA256
from empirical_lawhood.adapters.simulator_morphism_challenges.issued_inputs import CAPABILITY_INPUT_TYPES, RCChallengeIssuedInputs, RCChallengeRuntimeInputPort, capability_port_key
from empirical_lawhood.adapters.simulator_morphism_challenges.capability_configs import CAPABILITY_CONFIG_TYPES
from empirical_lawhood.adapters.simulator_morphism_challenges.numeric_inputs import RCChallengeExposureCensus, RCChallengeNumericSource, RCChallengeOriginalNumericRecipe, allocate_numeric_source, original_descriptor_bytes, original_numeric_definition
from empirical_lawhood.adapters.simulator_morphism_challenges.runtime_provider import SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS, source_closure_sha256
from empirical_lawhood.adapters.simulator_morphism_challenges.specification import current_rc_challenge_formal_register
from empirical_lawhood.adapters.simulator_morphism_challenges.retained_results import RCChallengeResultAuthorityContext
from empirical_lawhood.adapters.simulator_morphism_challenges.prerequisites import phase_external_records
from empirical_lawhood.adapters.simulator_morphism_challenges.runtime_contracts import SimulatorMorphismChallengeCanaryReport, SimulatorMorphismChallengeEvaluationDesignFreeze
from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeMethodFreeze
from empirical_lawhood.adapters.simulator_morphism_challenges.workflow import create_seed_roster
from empirical_lawhood.adapters.simulator_morphism_challenges.runtime_provider import implementation_closures
from empirical_lawhood.api import rc_challenge_authoring as application
from empirical_lawhood.api.rc_challenge_results import bind_rc_challenge_retained_result, bind_rc_challenge_selected_output, read_rc_challenge_result
from empirical_lawhood.api.rc_challenge_inputs import publish_rc_challenge_numeric_input, read_rc_challenge_numeric_input, authenticate_rc_exposure_census
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.artifacts import ArtifactWriteRequest, ArtifactProfile, CanonicalTaskReceipt, ReceiptCheck
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest, ExternalTaskReceiptStore
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.execution import TaskContext, WorkerInputBinding, WorkerInputKind, WorkerInputPort, WorkerOutputPort, WorkerIsolationProfile, DependencyReceiptBinding
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from tests.finite_response_rerun_fixtures import publish_current_issue, publish_current_step_output


ROOT = Path(__file__).resolve().parents[1]


def _sources():
    return {path: (ROOT / path).read_bytes() for path in SIMULATOR_MORPHISM_CHALLENGE_SOURCE_CLOSURE_PATHS}


def _census():
    return RCChallengeExposureCensus("synthetic.complete-empty-census", (), (), True)


@pytest.fixture
def plane(tmp_path):
    store = tmp_path / "synthetic-store"
    store.mkdir()
    return ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        "synthetic.rc-store", "synthetic RC software store", str(store), "/",
        OperatorStorageProfile.SCHEMA, 1, None, None, (),
    )))


@pytest.fixture
def development_input(plane):
    return publish_rc_challenge_numeric_input(
        writer=plane, config=default_config(Phase.DEVELOPMENT), exposure_census=_census(),
        prior_source_payloads={}, source_id="synthetic.dev-source", input_id="synthetic.dev-input",
        publication_scope_id="synthetic.rc-scope", publication_scope_relative_root="synthetic/rc",
        relative_root="synthetic/rc/dev",
    )


def test_numeric_export_retains_actual_current_custody_and_full_development_census(development_input, plane):
    packet = read_rc_challenge_numeric_input(development_input.canonical_bytes(), writer=plane)
    rows = packet.scientific_inputs(default_config(Phase.DEVELOPMENT))
    assert len(rows) == 6 and all(len(row.descriptor_inputs) == 3 for row in rows)
    assert all(len(d.history_fibre_seed_sha256s) == 36 for row in rows for d in row.descriptor_inputs)
    assert all(d.original_descriptor_sha256 != d.current_descriptor_sha256 for row in rows for d in row.descriptor_inputs)
    assert all(d.original_source.sha256 == packet.source.fingerprint() and d.export_receipt.sha256 == packet.receipt.fingerprint() for row in rows for d in row.descriptor_inputs)
    assert packet.source.exposed_example
    assert packet.receipt.source_publication.materialization.relative_path.endswith("source.json")


def test_original_random_fibre_recipe_is_independent_of_current_schema_namespace(development_input):
    descriptor = generate_descriptor(unit_id="eval.smooth-periodic-12.block-00", family=Family.SMOOTH_PERIODIC_12, scale_cells=64, seed=b"q" * 32)
    # Independently reconstruct the source recipe's envelope, rather than using
    # the exporter or current descriptor fingerprint as the RNG oracle.
    payload = descriptor.canonical_bytes()
    current_prefix = ('{"schema":"' + descriptor.SCHEMA + '"').encode("ascii")
    original = b'{"schema":"icf-yolo/ipsmc/v3-denominator-descriptor/v1"' + payload[len(current_prefix):]
    digest = sha256(original).hexdigest()
    recipe, computed, seeds = original_numeric_definition(descriptor)
    assert recipe is RCChallengeOriginalNumericRecipe.GENERATED_ORIGINAL_ENVELOPE
    assert original_descriptor_bytes(descriptor) == original
    assert computed == digest
    assert seeds[17] == sha256(f"{digest}:17:random-fibre".encode("ascii")).hexdigest()
    row = development_input.export.descriptors[0]
    with pytest.raises(ValueError, match="recipe"):
        replace(row, original_descriptor_sha256=row.descriptor.fingerprint())


def test_known_development_preserves_opaque_original_numeric_reference_without_historical_custody(development_input):
    from empirical_lawhood.adapters.history_budget_fixed_scientific_inputs import _DESCRIPTOR_ROWS

    rows = {(row[1], row[2]): row for row in _DESCRIPTOR_ROWS if row[0] == 0}
    for exported in development_input.export.descriptors:
        pinned = rows[exported.descriptor.unit_id, exported.descriptor.scale_cells]
        assert exported.original_numeric_recipe is RCChallengeOriginalNumericRecipe.FROZEN_EXPOSED_DEVELOPMENT_REFERENCE
        assert (exported.original_descriptor_sha256, exported.history_fibre_seed_sha256s) == (pinned[4], pinned[6])
        assert exported.descriptor.fingerprint() == pinned[5]
    assert development_input.receipt.source_publication.logical.logical_artifact_id.startswith("synthetic.")


def test_export_custody_tampering_and_missing_physical_bytes_refuse(development_input, plane):
    publication = development_input.receipt.source_publication
    with pytest.raises(ValueError, match="source/unit/scale"):
        replace(development_input, source=replace(development_input.source, source_id="renamed.source"))
    # Verify the real guarded plane; changing its exact materialization locator
    # must fail even when the claimed logical source digest is unchanged.
    forged = replace(publication, materialization=replace(publication.materialization, relative_path="synthetic/missing.json"))
    with pytest.raises((OSError, ValueError, RuntimeError)):
        plane.verify(forged.materialization)


def test_exposure_census_authenticates_complete_prior_numeric_seeds(development_input):
    source = development_input.source
    payload = source.canonical_bytes()
    artifact = ArtifactIdentity("prior.rc-source", "prior-numeric-source", source.SCHEMA, sha256(payload).hexdigest(), "application/json", len(payload))
    digests = tuple(sorted(sha256(seed).hexdigest() for seed in source.seeds))
    census = RCChallengeExposureCensus("synthetic.prior-census", (artifact,), digests, True)
    authenticate_rc_exposure_census(census, {artifact.artifact_id: payload})
    with pytest.raises(ValueError, match="missing"):
        authenticate_rc_exposure_census(replace(census, exposed_seed_sha256s=digests[:-1]), {artifact.artifact_id: payload})
    with pytest.raises(ValueError, match="authentication"):
        authenticate_rc_exposure_census(census, {artifact.artifact_id: payload + b" "})
    with pytest.raises(ValueError, match="complete"):
        RCChallengeExposureCensus("incomplete", (), (), False)


def test_fresh_allocations_exclude_relabelled_exposed_numeric_units():
    config = default_config(Phase.NOMINATION)
    seeds = tuple(index.to_bytes(32, "big") for index in range(1, 37))
    census = RCChallengeExposureCensus("prior.exposure", (), (sha256(seeds[0]).hexdigest(),), True)
    with pytest.raises(ValueError, match="previously exposed"):
        RCChallengeNumericSource("new.label", Phase.NOMINATION, config.unit_ids, tuple(seed.hex() for seed in seeds), census, False)
    fixture = allocate_numeric_source(source_id="synthetic.exposed", config=config, exposure_census=census, exposed_seed_hexes=tuple(seed.hex() for seed in seeds))
    assert fixture.exposed_example
    with pytest.raises(ValueError, match="repeats"):
        replace(fixture, seed_hexes=(fixture.seed_hexes[0],) * 36)
    with pytest.raises(ValueError, match="before evaluation"):
        allocate_numeric_source(source_id="late", config=default_config(Phase.EVALUATION, seed_roster_commitment_sha256="a" * 64), exposure_census=_census())


def test_export_is_complete_and_wrong_current_phase_or_denominator_refuses(development_input):
    export = development_input.export
    with pytest.raises(ValueError, match="full"):
        replace(export, descriptors=export.descriptors[:-1]).require_source(development_input.source)
    with pytest.raises(ValueError, match="phase"):
        development_input.scientific_inputs(default_config(Phase.NOMINATION))
    first = development_input.scientific_inputs(default_config(Phase.DEVELOPMENT))[0]
    with pytest.raises(ValueError, match="absent"):
        first.descriptor_input("f" * 64)


def test_canary_composition_and_installed_factory_reconstruct_exact_issued_inputs(plane):
    config, sources = default_config(Phase.CANARY), _sources()
    composition = compose_simulator_morphism_challenges_phase(config=config, external_records=(), source_files=sources,
                                                               register=current_rc_challenge_formal_register())
    context = RCChallengeCandidateContextProvider(composition)
    assert context.resolve_standard(composition.authoring.base).context == composition.bundle.context
    assert len(composition.payloads) == 10 and len(composition.bundle.base.registry.capabilities) == 5
    assert len(composition.bundle.base.protocol.steps) == 7
    factory_by_key = {factory.binding.capability_key: factory for factory in EXECUTABLE_BINDING_FACTORIES}
    for payload in (value for value in composition.payloads if hasattr(value, "inputs")):
        payload = decode_canonical_bytes(payload.canonical_bytes(), type(payload), maximum_bytes=8 * 1024**2)
        key = payload.CAPABILITY_KEY
        provider = factory_by_key[key].build_provider(registry=composition.bundle.base.registry, records=(CAPABILITY_CONFIG_TYPES[key](config), payload),
            platform_ports=(ExecutablePlatformPort(capability_port_key(key), RCChallengeRuntimeInputPort(sources, plane)),))
        runners = provider.runners(composition.bundle.base.registry)
        assert len(runners) == 1 and runners[0].manifest.capability_key == key
        assert runners[0].execution_count == 0
    with pytest.raises(ValueError, match="package"):
        context.resolve_standard(replace(composition.authoring.base, package_id="different.package"))


def test_factories_reject_missing_input_custody_and_changed_source_before_any_runner(plane):
    config, sources = default_config(Phase.CANARY), _sources()
    registry = simulator_morphism_challenges_phase_registry(config=config, implementation_sha256=INSTALLED_IMPLEMENTATION_SHA256)
    factory = next(value for value in EXECUTABLE_BINDING_FACTORIES if value.binding.capability_key == registry.capabilities[0].capability_key)
    issued = RCChallengeIssuedInputs("synthetic.canary", config, (), None, source_closure_sha256(sources))
    payload = CAPABILITY_INPUT_TYPES[factory.binding.capability_key](issued)
    with pytest.raises(ValueError, match="custody platform"):
        factory.build_provider(registry=registry, records=(CAPABILITY_CONFIG_TYPES[factory.binding.capability_key](config), payload), platform_ports=())
    changed = {**sources, next(iter(sources)): b"changed current source"}
    with pytest.raises(ValueError, match="source closure"):
        factory.build_provider(registry=registry, records=(CAPABILITY_CONFIG_TYPES[factory.binding.capability_key](config), payload), platform_ports=(ExecutablePlatformPort(
            capability_port_key(factory.binding.capability_key), RCChallengeRuntimeInputPort(changed, plane)),))
    assert len({schema for binding in BINDINGS for schema in binding.construction_record_schema_ids}) == 30


def _rank_control(*, shrinking=False, invalid_row=False):
    config = default_config(Phase.CANARY)
    descriptor = generate_descriptor(unit_id="canary.truth-known.n16", family=Family.SMOOTH_PERIODIC_12, scale_cells=16, seed=b"z" * 32)
    receiver = np.zeros((8, 16))
    receiver[:7, 0] = 1.0
    receiver[7, 1] = 1e-10 if shrinking else 1e-13
    if invalid_row:
        receiver[0] = 0
    rates = np.ones(16) * -1.0
    if shrinking:
        rates[0] = -10_000.0
    operator = DenseObserverOperator(np.diag(rates), np.zeros(16), np.ones(16), receiver)
    return _rank_forecast(config, descriptor, operator)


def test_scientific_ranks_retain_distinct_row_scale_estimands_and_both_spectra():
    forecast, arrays, _ = _rank_control()
    assert forecast.rank_steps[0].effective_rank == 1
    assert forecast.rank_steps[0].algebraic_comparator_rank == 2
    assert forecast.effective_rank_right_censored
    assert len(forecast.rank_steps) == 36
    assert len(arrays) >= 2


def test_effective_rank_nonmonotonicity_and_invalid_rows_are_retained():
    forecast, _, _ = _rank_control(shrinking=True)
    assert forecast.rank_steps[0].effective_rank == 2
    assert any(row.nonmonotone_effective_rank for row in forecast.rank_steps)
    assert forecast.effective_rank_right_censored
    with pytest.raises(ValueError, match="invalid row"):
        _rank_control(invalid_row=True)


def _retained(plane, record, name, *, status=OperationalStatus.SUCCEEDED, access=OutcomeAccess.DEVELOPMENT_VISIBLE):
    """Synthetic software output with actual publication and exact receipt bytes."""
    def publish(value, label):
        result = plane.write(ArtifactWriteRequest(
            logical_artifact_id=f"synthetic.{name}.{label}", relative_path=f"synthetic/results/{name}-{label}.json",
            payload_schema=value.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON, media_type="application/json",
            publication_scope_id="synthetic.rc-results", publication_scope_relative_root="synthetic/results",
            payload=value.canonical_bytes(), visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            parent_visibility_ceilings=(), outcome_access=access, minimum_free_bytes=1,
        ))
        manifest = decode_artifact_manifest(plane.root.resolve(result.manifest_materialization.relative_path, for_write=False).read_bytes())
        return result, manifest
    output, manifest = publish(record, "output")
    attempt = f"synthetic-run.{name}.attempt-001"
    receipt = CanonicalTaskReceipt(f"receipt.{attempt}", "synthetic-run", name, attempt, "0" * 40,
        (), (output.materialization,), (output.logical,), (ReceiptCheck("synthetic-software-check", True, ()),),
        status, () if status is OperationalStatus.SUCCEEDED else ("SYNTHETIC_CANCELLED",))
    receipt_publication = ExternalTaskReceiptStore(plane, minimum_free_bytes=1).commit(receipt,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY, outcome_access=access)
    receipt_manifest = decode_artifact_manifest(plane.root.resolve(receipt_publication.manifest_materialization.relative_path, for_write=False).read_bytes())
    return bind_rc_challenge_retained_result(result_id=f"synthetic.{name}.retained", record_payload=record.canonical_bytes(),
        output_manifest=manifest, task_receipt=receipt, receipt_manifest=receipt_manifest, writer=plane)


@pytest.fixture
def phase_inputs(plane, development_input):
    nomination = default_config(Phase.NOMINATION)
    packet = publish_rc_challenge_numeric_input(writer=plane, config=nomination, exposure_census=_census(),
        prior_source_payloads={}, source_id="synthetic.nomination-source", input_id="synthetic.nomination-input",
        publication_scope_id="synthetic.nomination", publication_scope_relative_root="synthetic/nomination",
        relative_root="synthetic/nomination/input")
    roster, commitment = create_seed_roster(nomination, injected_seeds=packet.source.seeds)
    evaluation = default_config(Phase.EVALUATION, seed_roster_commitment_sha256=commitment.fingerprint())
    development = default_config(Phase.DEVELOPMENT)
    canary = SimulatorMorphismChallengeCanaryReport("simulator-morphism-challenges.source-canary-qualification",
        ("synthetic-software-only",), True, (), Decimal(1), 1, Decimal(1), Decimal(1), 1, 0)
    closures = implementation_closures(_sources())
    method = SimulatorMorphismChallengeMethodFreeze("synthetic.method-freeze", development.fingerprint(), evaluation.fingerprint(),
        commitment.fingerprint(), closures.observer_sha256, closures.generator_sha256, closures.evaluator_sha256,
        "c" * 64, ("synthetic-software-only",), 0, EvidenceCeiling.LOCAL_LAW)
    design = SimulatorMorphismChallengeEvaluationDesignFreeze("synthetic.design-freeze", method, evaluation.fingerprint(),
        commitment, closures.complete_sha256, 0)
    dev_prereqs = (_retained(plane, canary, "canary"), _retained(plane, commitment, "commitment"))
    eval_prereqs = (_retained(plane, design, "design"), _retained(plane, roster, "roster"))
    return {
        Phase.NOMINATION: (nomination, packet, (), ()),
        Phase.CANARY: (default_config(Phase.CANARY), None, (), ()),
        Phase.DEVELOPMENT: (development, development_input, phase_external_records(config=development, prerequisite_custody=dev_prereqs, evaluation_config=evaluation), dev_prereqs),
        Phase.EVALUATION: (evaluation, packet, phase_external_records(config=evaluation, prerequisite_custody=eval_prereqs), eval_prereqs),
    }


@pytest.mark.parametrize("phase", tuple(Phase))
def test_full_phase_public_authoring_projection_and_installed_factory_proof(phase, phase_inputs, plane, tmp_path, monkeypatch):
    config, packet, external, prerequisites = phase_inputs[phase]
    # Only the clean-worktree and physical-location seams are synthetic. The
    # public compiler, full-size graphs, codecs, factories and custody are real.
    closure = ImplementationSourceClosure("synthetic.source", SourceClosureKind.CLEAN_GIT_COMMIT,
        "0" * 40, "d" * 64, "e" * 64, True)
    monkeypatch.setattr(application, "capture_clean_target_closure", lambda *args: closure)
    monkeypatch.setattr(application, "resolve_authoring_directory", lambda directory, **kwargs: (directory, plane.root.contract))
    output = Path(plane.root.contract.canonical_path) / "authoring" / phase.value
    started = perf_counter()
    summary = application.author_rc_challenge_phase(root=ROOT, config=config, external_records=external,
        output_dir=output, numeric_input=packet, artifact_writer=plane, prerequisite_custody=prerequisites)
    authored = perf_counter()
    handoff = application.load_rc_challenge_handoff(directory=output, root=ROOT, storage_profile=None, artifact_writer=plane)
    loaded = perf_counter()
    proof = application.prove_rc_challenge_handoff(handoff)
    proved = perf_counter()
    print('public_handoff_seconds', {'family': 'rc-challenges', 'phase': phase.value,
        'independent_units': len(config.unit_ids), 'tasks': len(handoff.projection.tasks),
        'outputs': sum(len(task.outputs) for task in handoff.projection.tasks),
        'author': authored-started, 'load': loaded-authored, 'prove': proved-loaded}, flush=True)
    assert not summary["scientific_execution_performed"] and not proof["scientific_execution_performed"]
    assert proof["task_count"] == summary["task_count"] == len(handoff.resources.task_cells)
    assert proof["runner_count"] == len(handoff.composition.bundle.base.registry.capabilities)
    assert proof["output_contract_count"] > 0 and proof["external_input_count"] > 0
    assert summary["requested_independent_units"] == len(config.unit_ids)
    assert handoff.composition.runtime_provider._runners == ()
    assert handoff.composition.implementation_sha256 == closure.implementation_sha256
    assert all(manifest.implementation_sha256 == INSTALLED_IMPLEMENTATION_SHA256
               for manifest in handoff.composition.bundle.base.registry.capabilities)
    issue = publish_current_issue(plane, handoff, label=f"current-rc-{phase.value.lower()}")
    assert issue.manifest.source_closure == closure
    assert issue.manifest.candidate.base_candidate.base_candidate.implementation_sha256 == closure.implementation_sha256
    if phase is Phase.NOMINATION:
        roster, _ = create_seed_roster(config, injected_seeds=packet.source.seeds)
        receipt, _, published = publish_current_step_output(plane, issue, label="current-rc-seed-roster", record=roster)
        arguments = dict(result_id="synthetic.current-rc-roster-result", run_id=receipt.run_id,
            task_id=receipt.task_id, receipt_id=receipt.receipt_id,
            output_artifact_id=published.logical.logical_artifact_id, writer=plane)
        selected = bind_rc_challenge_selected_output(**arguments)
        assert selected.decode() == roster
        prior_raw = packet.source.canonical_bytes()
        prior = ArtifactIdentity(packet.receipt.source_publication.logical.logical_artifact_id, "prior-numeric-source",
            packet.source.SCHEMA, packet.source.fingerprint(), "application/json", len(prior_raw))
        prior_census = RCChallengeExposureCensus("synthetic.after-first-nomination", (prior,),
            tuple(sorted(sha256(seed).hexdigest() for seed in packet.source.seeds)), True)
        second = publish_rc_challenge_numeric_input(writer=plane, config=config, exposure_census=prior_census,
            prior_source_payloads={prior.artifact_id: prior_raw}, source_id="synthetic.second-nomination-source",
            input_id="synthetic.second-nomination-input", publication_scope_id="synthetic.second-nomination",
            publication_scope_relative_root="synthetic/second-nomination", relative_root="synthetic/second-nomination/input")
        second_output = Path(plane.root.contract.canonical_path) / "authoring" / "second-nomination"
        application.author_rc_challenge_phase(root=ROOT, config=config, external_records=(), output_dir=second_output,
            numeric_input=second, artifact_writer=plane)
        second_handoff = application.load_rc_challenge_handoff(directory=second_output, root=ROOT,
            storage_profile=None, artifact_writer=plane)
        assert second_handoff.projection.source_plan.object_id != handoff.projection.source_plan.object_id
        assert set(second.source.seed_hexes).isdisjoint(packet.source.seed_hexes)
    if phase is Phase.CANARY:
        resource_path = output / "resources.json"
        original = resource_path.read_bytes()
        resource_path.write_bytes(replace(handoff.resources, envelope_spec_id="synthetic.changed-resources").canonical_bytes())
        try:
            with pytest.raises(ValueError, match="projection reconstruction"):
                application.load_rc_challenge_handoff(directory=output, root=ROOT, storage_profile=None, artifact_writer=plane)
        finally:
            resource_path.write_bytes(original)
    if phase is Phase.EVALUATION:
        assert len(config.unit_ids) == 36 and config.scale_cells == (64, 128, 256)
        assert len(handoff.composition.runtime_provider.scientific_inputs) == 36
        assert sum(len(row.descriptor_inputs) for row in handoff.composition.runtime_provider.scientific_inputs) == 108
        from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeCellRecurrence, SimulatorMorphismChallengeRecurrenceResult
        cells = tuple(sorted((SimulatorMorphismChallengeCellRecurrence(
            f"synthetic.{family.value.lower().replace('_', '-')}.{scale}.{endpoint}", family, scale, endpoint,
            12, 0, 0, 0, 0, 0, 0, 12, None, Decimal(0), False, False)
            for family in Family for scale in (64, 128, 256) for endpoint in ("dynamical-closure", "decision-closure")),
            key=lambda cell: cell.cell_id))
        # Every unit is explicitly unevaluable in this fictional result: no
        # numerical campaign, bootstrap or scientific support is manufactured.
        recurrence = SimulatorMorphismChallengeRecurrenceResult("synthetic.current-evaluation-recurrence",
            prerequisites[0].decode().method_freeze.fingerprint(), cells, False, False, False, False,
            "a" * 64, (), EvidenceCeiling.LOCAL_LAW, False, False)
        receipt, _, published = publish_current_step_output(plane, issue, label="current-rc-recurrence", record=recurrence)
        authority = RCChallengeResultAuthorityContext("synthetic.current-rc-result-access",
            issue.publication.issued_study, issue.publication.execution_authority, issue.publication.reveal_authority)
        arguments = dict(result_id="synthetic.current-rc-recurrence-result", run_id=receipt.run_id,
            task_id=receipt.task_id, receipt_id=receipt.receipt_id,
            output_artifact_id=published.logical.logical_artifact_id, writer=plane)
        with pytest.raises(PermissionError, match="ACTUAL_CURRENT_AUTHORITY_REQUIRED"):
            bind_rc_challenge_selected_output(**arguments)
        selected = bind_rc_challenge_selected_output(**arguments, authority_context=authority)
        assert selected.decode() == recurrence
        view = read_rc_challenge_result(selected.canonical_bytes(), writer=plane)
        assert view.summary["run_id"] == receipt.run_id
        assert len({(cell["family"], cell["scale_cells"], cell["endpoint_id"]) for cell in view.summary["cells"]}) == 18
        assert all(cell["requested_count"] == cell["unevaluable_count"] == 12 for cell in view.summary["cells"])
        from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
        store = ExternalStudyOperationAuthorityStore(plane)
        expired = replace(store.load(authority.reveal_authority.object_id),
            authority_id='synthetic.expired.rc-reveal', issued_at_utc='2026-09-01T00:00:00Z',
            expires_at_utc='2026-09-02T00:00:00Z')
        store.persist(expired)
        with monkeypatch.context() as refusal:
            verify = plane.verify_manifest
            def before_protected_bytes(manifest, *args, **kwargs):
                if manifest.materialization.relative_path == published.materialization.relative_path:
                    raise AssertionError('invalid current grant contacted protected scientific bytes')
                return verify(manifest, *args, **kwargs)
            refusal.setattr(plane, 'verify_manifest', before_protected_bytes)
            for identity in (authority.execution_authority, authority.reveal_authority):
                sidecar = plane.root.resolve(store._relative_path(identity.object_id) + '.manifest.json', for_write=False)
                preserved = sidecar.read_bytes()
                sidecar.unlink()
                try:
                    with pytest.raises(KeyError):
                        bind_rc_challenge_selected_output(**arguments, authority_context=authority)
                finally:
                    sidecar.write_bytes(preserved)
            with pytest.raises(PermissionError):
                bind_rc_challenge_selected_output(**arguments, authority_context=replace(authority,
                    reveal_authority=ObjectIdentity.from_record(expired.authority_id, expired)))
        # The same exact controls become readable only after current custody is
        # restored; no previous success caches grant validity across operations.
        assert read_rc_challenge_result(bind_rc_challenge_selected_output(**arguments,
            authority_context=authority).canonical_bytes(), writer=plane).record == recurrence
        from empirical_lawhood.api.rc_challenge_results import _authenticate_result_access
        with pytest.raises(PermissionError, match="CURRENT_ISSUED_SOURCE_MISMATCH"):
            _authenticate_result_access(context=authority, receipt=replace(receipt, implementation_commit="1" * 40),
                logical=published.logical, writer=plane)
        with pytest.raises(PermissionError, match="CURRENT_ISSUED_TASK_OUTPUT_MISMATCH"):
            _authenticate_result_access(context=authority, receipt=receipt,
                logical=replace(published.logical, logical_artifact_id="synthetic.foreign-output"), writer=plane)
        with pytest.raises((OSError, ValueError, PermissionError)):
            _authenticate_result_access(context=authority, receipt=replace(receipt, run_id="synthetic.foreign-run"),
                logical=published.logical, writer=plane)


def test_retained_negative_canary_and_cancelled_receipt_remain_visible_and_refuse_downstream(plane):
    negative = SimulatorMorphismChallengeCanaryReport("simulator-morphism-challenges.source-canary-qualification",
        ("synthetic-negative",), False, ("SYNTHETIC_CONFORMANCE_STOP",), Decimal(0), 0, Decimal(0), Decimal(0), 0, 0)
    retained = _retained(plane, negative, "negative-canary")
    view = read_rc_challenge_result(retained.canonical_bytes(), writer=plane)
    assert view.summary["passed"] is False and view.summary["reason_codes"] == ("SYNTHETIC_CONFORMANCE_STOP",)
    assert view.summary["scientific_execution_performed_by_reader"] is False
    cancelled = _retained(plane, negative, "cancelled-canary", status=OperationalStatus.BLOCKED)
    assert read_rc_challenge_result(cancelled.canonical_bytes(), writer=plane).summary["operational_status"] == "BLOCKED"
    with pytest.raises(ValueError, match="successful upstream"):
        cancelled.require_prerequisite(negative)
    with pytest.raises(ValueError, match="manifest"):
        replace(retained, canonical_payload_text=negative.canonical_bytes().decode() + " ")


def test_phase_prerequisites_require_receipts_and_current_method_source(phase_inputs, plane):
    config, packet, external, prerequisites = phase_inputs[Phase.EVALUATION]
    with pytest.raises(ValueError, match="complete authenticated"):
        compose_simulator_morphism_challenges_phase(config=config, external_records=external, source_files=_sources(),
            register=current_rc_challenge_formal_register(), numeric_input=packet, artifact_writer=plane)
    composition = compose_simulator_morphism_challenges_phase(config=config, external_records=external, source_files=_sources(),
        register=current_rc_challenge_formal_register(), numeric_input=packet, artifact_writer=plane, prerequisite_custody=prerequisites)
    changed = {**_sources(), "src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/observer.py": b"new observer source"}
    forged = replace(composition.issued_inputs, implementation_source_closure_sha256=source_closure_sha256(changed))
    with pytest.raises(ValueError, match="design/method freeze"):
        RCChallengeRuntimeInputPort(changed, plane).authenticate(forged)


def test_selected_output_uses_exact_stored_receipt_and_sealed_custody_does_not_grant_access(plane):
    report = SimulatorMorphismChallengeCanaryReport("simulator-morphism-challenges.source-canary-qualification",
        ("synthetic-selection",), True, (), Decimal(0), 0, Decimal(0), Decimal(0), 0, 0)
    retained = _retained(plane, report, "selected-canary")
    selected = bind_rc_challenge_selected_output(result_id="synthetic.selected-output", run_id="synthetic-run",
        task_id="selected-canary", receipt_id=retained.task_receipt.receipt_id,
        output_artifact_id=retained.output_manifest.logical.logical_artifact_id, writer=plane)
    assert selected.decode() == report and selected.task_receipt == retained.task_receipt
    with pytest.raises(ValueError, match="ABSENT_FROM_EXACT"):
        bind_rc_challenge_selected_output(result_id="synthetic.wrong-output", run_id="synthetic-run",
            task_id="selected-canary", receipt_id=retained.task_receipt.receipt_id,
            output_artifact_id="synthetic.unreceipted", writer=plane)
    with pytest.raises(PermissionError, match="ACTUAL_CURRENT_AUTHORITY_REQUIRED"):
        _retained(plane, report, "sealed-canary", access=OutcomeAccess.EVALUATION_SEALED)


@pytest.mark.parametrize("destination", ("outside", "traversal", "symlink", "occupied"))
def test_authoring_destination_refuses_before_source_or_prerequisite_reads(destination, plane, tmp_path, monkeypatch):
    guarded = Path(plane.root.contract.canonical_path)
    if destination == "outside":
        output = tmp_path / "outside-authoring"
    elif destination == "traversal":
        output = guarded / ".." / "traversal-authoring"
    elif destination == "symlink":
        output = guarded / "linked-authoring"
        output.symlink_to(tmp_path / "unselected-destination", target_is_directory=True)
    else:
        output = guarded / "occupied-authoring"
        output.mkdir()
        (output / "retained-user-file.txt").write_text("preserve me", encoding="utf-8")

    contacted = []

    def source_contact(*args, **kwargs):
        contacted.append("source closure")
        raise AssertionError("invalid output must refuse before upstream reads")

    monkeypatch.setattr(application, "capture_clean_target_closure", source_contact)
    with pytest.raises((ValueError, FileExistsError)):
        application.author_rc_challenge_phase(root=ROOT, config=default_config(Phase.CANARY),
            external_records=(), output_dir=output, artifact_writer=plane)
    assert not contacted
    if destination == "occupied":
        assert (output / "retained-user-file.txt").read_text(encoding="utf-8") == "preserve me"
    elif destination != "symlink":
        assert not output.exists()


def test_installed_codec_registry_remains_closed_at_declared_upper_bound():
    from empirical_lawhood.adapters.composition.generated_executable_bindings import EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY
    from empirical_lawhood.runtime.static_codecs import CanonicalRecordCodecRegistry, MAX_STATIC_CODEC_REGISTRATIONS

    installed = EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY
    assert len(installed.registrations) <= MAX_STATIC_CODEC_REGISTRATIONS
    oversized = tuple(replace(installed.registrations[0], record_schema=f"empirical-lawhood/tests/bounded-codec-{index:03d}")
        for index in range(MAX_STATIC_CODEC_REGISTRATIONS + 1))
    with pytest.raises(ValueError, match="invalid closed size"):
        CanonicalRecordCodecRegistry("synthetic.oversized-codec-registry", oversized, {})


@pytest.mark.parametrize("writer", (None, object()))
def test_authoring_requires_actual_guarded_plane_before_source_reads(writer, tmp_path, monkeypatch):
    contacted = []
    monkeypatch.setattr(application, "capture_clean_target_closure", lambda *args: contacted.append(args))
    with pytest.raises(TypeError, match="actual guarded external artifact plane"):
        application.author_rc_challenge_phase(root=ROOT, config=default_config(Phase.CANARY),
            external_records=(), output_dir=tmp_path / "unsupported-authoring", artifact_writer=writer)
    assert not contacted and not (tmp_path / "unsupported-authoring").exists()


def test_actual_numeric_export_through_registered_provider_measurement_and_retained_result(phase_inputs, plane, monkeypatch):
    """One exposed development unit, three fixed views; no phase or qualification run."""
    from empirical_lawhood.adapters.simulator_morphism_challenges.runtime_contracts import SimulatorMorphismChallengeDenominatorBundle, SimulatorMorphismChallengeHistoryBundle, SimulatorMorphismChallengeArrayManifest
    from empirical_lawhood.adapters.simulator_morphism_challenges import runtime_provider

    plane = ExternalArtifactPlane(plane.root, validators=application.RC_CHALLENGE_ARTIFACT_PROFILE_VALIDATORS)
    config, packet, external, prerequisites = phase_inputs[Phase.DEVELOPMENT]
    sources = _sources()
    composition = compose_simulator_morphism_challenges_phase(config=config, external_records=external,
        source_files=sources, register=current_rc_challenge_formal_register(), numeric_input=packet,
        artifact_writer=plane, prerequisite_custody=prerequisites)
    unit = config.unit_ids[0]
    steps = {step.step_id: step for step in composition.bundle.base.protocol.steps}
    factories = {factory.binding.capability_key: factory for factory in EXECUTABLE_BINDING_FACTORIES}

    class Reader:
        def __init__(self, payload):
            self.stream = BytesIO(payload)

        @property
        def bytes_read(self):
            return self.stream.tell()

        def read(self, size=-1):
            return self.stream.read(size)

        def close(self):
            self.stream.close()

    def execute(step, dependency=None):
        key = step.capability_key
        issued = next(payload for payload in composition.payloads if hasattr(payload, "inputs") and getattr(payload, "CAPABILITY_KEY", None) == key)
        wrapped = CAPABILITY_CONFIG_TYPES[key](config)
        provider = factories[key].build_provider(registry=composition.bundle.base.registry, records=(wrapped, issued),
            platform_ports=(ExecutablePlatformPort(capability_port_key(key), RCChallengeRuntimeInputPort(sources, plane)),))
        runner = provider.runners(composition.bundle.base.registry)[0]
        records = (("synthetic.phase-config", config), ("synthetic.wrapper-config", wrapped))
        ports = [WorkerInputPort(WorkerInputBinding(name, name + ".materialization", record.SCHEMA,
            "application/json", len(record.canonical_bytes()), VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.OUTCOME_BLIND, WorkerInputKind.EXTERNAL), Reader(record.canonical_bytes())) for name, record in records]
        receipts = ()
        if dependency is not None:
            dependency.authenticate(plane)
            logical, materialization = dependency.output_manifest.logical, dependency.output_manifest.materialization
            ports.append(WorkerInputPort(WorkerInputBinding(logical.logical_artifact_id, materialization.materialization_id,
                logical.payload_schema, logical.media_type, materialization.size_bytes, logical.visibility_ceiling,
                logical.outcome_access, WorkerInputKind.DEPENDENCY), Reader(dependency.canonical_payload_text.encode())))
            receipts = (DependencyReceiptBinding(dependency.task_receipt.receipt_id, dependency.task_receipt.task_id,
                (materialization.materialization_id,)),)
        ports.sort(key=lambda port: port.artifact_id)
        context = TaskContext("synthetic-run", step.step_id, f"synthetic-run.{step.step_id}.attempt-001", step.config,
            tuple(port.binding for port in ports), tuple(ports), tuple(WorkerOutputPort(f"{step.step_id}.{output.output_id}",
                output.payload_schema, output.profile, output.media_type) for output in step.outputs),
            step.required_permissions, step.requested_outcome_access, step.resource_budget,
            WorkerIsolationProfile.EXPLORATION_NO_NETWORK, receipts)
        try:
            result = runner.execute(context)
        finally:
            for port in ports:
                port.close()
        assert runner.execution_count == 1 and all(check.passed for check in result.checks)
        return result

    descriptor_step = steps[f"development-descriptor.{unit}"]
    result = execute(descriptor_step)
    denominators = decode_canonical_bytes(result.outputs[0].payload, SimulatorMorphismChallengeDenominatorBundle, maximum_bytes=2 * 1024**2)
    assert tuple(descriptor.scale_cells for descriptor in denominators.descriptors) == (16, 32, 64)
    retained_denominators = _retained(plane, denominators, descriptor_step.step_id)
    history_step = steps[f"development-history.{unit}"]
    result = execute(history_step, retained_denominators)
    outputs = {output.output_id.removeprefix(history_step.step_id + "."): output.payload for output in result.outputs}
    history = decode_canonical_bytes(outputs["history-bundle"], SimulatorMorphismChallengeHistoryBundle, maximum_bytes=2 * 1024**2)
    manifest = decode_canonical_bytes(outputs["history-array-manifest"], SimulatorMorphismChallengeArrayManifest, maximum_bytes=2 * 1024**2)
    assert manifest.payload_sha256 == sha256(outputs["history-arrays"]).hexdigest()
    assert history.denominator_bundle_sha256 == denominators.fingerprint() and history.arrays_manifest_sha256 == manifest.fingerprint()
    assert history.outcome_count == 0 and len(history.forecasts) == 3
    assert all(tuple(step.depth for step in forecast.rank_steps) == tuple(range(36)) for forecast in history.forecasts)
    assert all(any(entry.array_id == f"n{scale}.{suffix}" for entry in manifest.entries)
        for scale in (16, 32, 64) for suffix in ("singular-spectra", "row-equilibrated-singular-spectra"))
    publications = []
    for output in history_step.outputs:
        written = plane.write(ArtifactWriteRequest(
            logical_artifact_id=f"artifact.synthetic-run.{history_step.step_id}.{output.output_id}",
            relative_path=f"synthetic/provider-smoke/{history_step.step_id}/{output.output_id}{output.filename_suffix}",
            payload_schema=output.payload_schema, profile=output.profile, media_type=output.media_type,
            publication_scope_id="synthetic.bounded-provider-measurement", publication_scope_relative_root="synthetic/provider-smoke",
            payload=outputs[output.output_id], visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            parent_visibility_ceilings=(), outcome_access=OutcomeAccess.OUTCOME_BLIND, minimum_free_bytes=1))
        publication = decode_artifact_manifest(plane.root.resolve(written.manifest_materialization.relative_path, for_write=False).read_bytes())
        publications.append((written, publication))
    attempt = f"synthetic-run.{history_step.step_id}.attempt-001"
    receipt = CanonicalTaskReceipt(f"receipt.{attempt}", "synthetic-run", history_step.step_id, attempt, "0" * 40,
        (retained_denominators.output_manifest.materialization.materialization_id,),
        tuple(sorted((written.materialization for written, _ in publications), key=lambda value: value.materialization_id)),
        tuple(sorted((written.logical for written, _ in publications), key=lambda value: value.logical_artifact_id)),
        result.checks, OperationalStatus.SUCCEEDED, ())
    receipt_publication = ExternalTaskReceiptStore(plane, minimum_free_bytes=1).commit(receipt,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY, outcome_access=OutcomeAccess.OUTCOME_BLIND)
    receipt_manifest = decode_artifact_manifest(plane.root.resolve(receipt_publication.manifest_materialization.relative_path, for_write=False).read_bytes())
    history_manifest = next(publication for _, publication in publications if publication.logical.payload_schema == history.SCHEMA)
    retained_history = bind_rc_challenge_retained_result(result_id="synthetic.bounded-provider-history", record_payload=history.canonical_bytes(),
        output_manifest=history_manifest, task_receipt=receipt, receipt_manifest=receipt_manifest, writer=plane)
    assert read_rc_challenge_result(retained_history.canonical_bytes(), writer=plane).record == history
    from empirical_lawhood.api.integration_handoffs import integration_artifact_profile_validators

    fresh_public_plane = ExternalArtifactPlane(plane.root, validators=integration_artifact_profile_validators())
    assert read_rc_challenge_result(retained_history.canonical_bytes(), writer=fresh_public_plane).record == history

    # A corrupted custodied numeric source must refuse before either scientific
    # operation can be contacted by a new registered provider.
    source_path = plane.root.resolve(packet.receipt.source_publication.materialization.relative_path, for_write=False)
    original = source_path.read_bytes()
    contacted = []
    monkeypatch.setattr(runtime_provider, "denominator_bundle", lambda **kwargs: contacted.append("descriptor"))
    monkeypatch.setattr(runtime_provider, "history_bundle", lambda **kwargs: contacted.append("measurement"))
    source_path.write_bytes(original + b" ")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            execute(descriptor_step)
        assert not contacted
    finally:
        source_path.write_bytes(original)
