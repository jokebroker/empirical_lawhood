# SPDX-License-Identifier: MPL-2.0

"""Assigned noise reaches the real plant; exposed prefixes and production joins stay distinct."""

from dataclasses import replace
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.reactor_prefix_response.authoring import ReactorPrefixResponseCandidateContextProvider, build_reactor_authoring
from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import AssignedReactorAuthoringProfile, fresh_assignment_status
from empirical_lawhood.adapters.composition.reactor_prefix_response.resources import reactor_resource_envelope
from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorAssignedPrefixPanel, ReactorPrefixAssignedUnit, ReactorPrefixAssignment, ReactorPrefixPriorCensus, assigned_prefix_config
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import SCENARIOS, ReactorPrefixPanel, acquire_prefix_branch
from empirical_lawhood.api.facade import EmpiricalLawhoodApi
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions
from empirical_lawhood.runtime.providers import ExternalInputPayload

ROOT = Path(__file__).parents[1]


def prior_census():
    return ReactorPrefixPriorCensus(
        "synthetic.census",
        ("unit.public-exposed",),
        ("seed.native-reactor.100",),
        ObjectIdentity(
            "synthetic.inventory",
            'empirical-lawhood/synthetic/inventory',
            "1.0.0",
            "0" * 64,
        ),
    )


def assignment():
    stem = "synthetic-assigned-prefix.cohort"
    return ReactorPrefixAssignment(
        stem,
        tuple(
            ReactorPrefixAssignedUnit(
                f"{stem}.{scenario.replace('_', '-')}", scenario, 900_000_000 + i
            )
            for i, scenario in enumerate(SCENARIOS)
        ),
        ObjectIdentity.from_record(prior_census().census_id, prior_census()),
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    )


def test_assignment_cannot_relabel_or_reuse_exposed_units_or_seeds():
    selected = assignment()
    assert len(selected.independent_unit_ids) == len(selected.seed_ids) == 5
    for changed in (
        {"prior_seed_ids": selected.seed_ids},
        {"prior_unit_ids": selected.independent_unit_ids},
        {"prior_unit_ids": tuple(sorted(u.unit_id for u in selected.units))},
        {"prior_seed_ids": tuple(sorted(str(u.noise_seed) for u in selected.units))},
    ):
        census = replace(prior_census(), **changed)
        with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
            selected.check_prior_census(census)
        with pytest.raises(ValueError, match="COLLISION"):
            replace(
                selected,
                prior_census=ObjectIdentity.from_record(census.census_id, census),
            ).check_prior_census(census)
    with pytest.raises(ValueError, match="COLLISION"):
        replace(selected, units=(selected.units[0],) * 5)
    profile = AssignedReactorAuthoringProfile(
        "synthetic.profile",
        "synthetic-assigned-prefix",
        load_packaged_reactor_source().fingerprint(),
        selected,
        prior_census(),
    )
    assert fresh_assignment_status(profile)["prospective_issue_eligible"] is False


def test_issue_exclusions_omit_only_impossible_native_numeric_aliases():
    canonical = "seed.pcg64dxsm.80000000000000000000000000000000"
    impossible = f"seed.native-reactor.{2**127}"
    census = replace(
        prior_census(),
        prior_seed_ids=tuple(
            sorted((canonical, impossible, "seed.native-reactor.100"))
        ),
    )
    assert census.prior_seed_ids == tuple(
        sorted((canonical, impossible, "seed.native-reactor.100"))
    )
    assert census.authoring_seed_ids == tuple(
        sorted((canonical, "seed.native-reactor.100"))
    )
    selected = replace(
        assignment(), prior_census=ObjectIdentity.from_record(census.census_id, census)
    )
    selected.check_prior_census(census)
    # The surviving valid alias still rejects the actual native seed at authoring.
    with pytest.raises(ValueError, match="COLLISION"):
        replace(
            selected,
            units=(replace(selected.units[0], noise_seed=100), *selected.units[1:]),
        ).check_prior_census(census)


def test_actual_native_seed_changes_noise_and_retains_paired_nested_deliveries():
    source = load_packaged_reactor_source()
    config = assigned_prefix_config(assignment())
    branches = config.branches[:4]
    panels = tuple(acquire_prefix_branch(config, source, branch) for branch in branches)
    assert all(type(panel) is ReactorAssignedPrefixPanel for panel in panels)
    starts = {
        panel.episodes[0].deliveries[0].next_measurement.t_reactor_k for panel in panels
    }
    assert len(starts) == 1  # One physical seed shared by arms and refinements.
    for panel in panels:
        episode = panel.episodes[0]
        assert episode.failure_code is None
        assert episode.unit == config.assignment.units[0]
        assert len(episode.deliveries) == 2
        assert [row.command.time_s for row in episode.deliveries] == [0, 10]
        assert episode.deliveries[-1].next_measurement.time_s == 20
        assert (
            decode_canonical_bytes(
                panel.canonical_bytes(), type(panel), maximum_bytes=1024**2
            )
            == panel
        )
        with pytest.raises(ValueError):
            decode_canonical_bytes(
                panel.canonical_bytes(), ReactorPrefixPanel, maximum_bytes=1024**2
            )
    changed = replace(config.assignment.units[0], noise_seed=1_000_000_001)
    revised = assigned_prefix_config(
        replace(config.assignment, units=(changed, *config.assignment.units[1:]))
    )
    other = acquire_prefix_branch(revised, source, revised.branches[0])
    assert other.episodes[0].deliveries[0].next_measurement.t_reactor_k not in starts


def test_assigned_panel_reaches_retained_projection_and_law_owners(tmp_path):
    """Excluded development integration; these are no campaign receipts."""
    from empirical_lawhood.adapters.methods.reactor_prefix_response.assigned.records import ReactorAssignedScienceResult, assigned_science_design
    from empirical_lawhood.adapters.methods.reactor_prefix_response.projection import project_panel
    from empirical_lawhood.adapters.methods.reactor_prefix_response.binding import bind_finite_chain
    from empirical_lawhood.adapters.methods.reactor_prefix_response.chain import identify_and_qualify
    from empirical_lawhood.infrastructure.artifacts import (
        ExternalArtifactPlane,
        GuardedExternalRoot,
    )
    from empirical_lawhood.infrastructure.candidate_payloads import (
        ExternalCandidatePayloadPlane,
    )
    from empirical_lawhood.runtime.artifacts import (
        ArtifactManifest,
        ArtifactWriteRequest,
        ExternalRootContract,
    )
    from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile

    root = tmp_path / "synthetic"
    root.mkdir()
    plane = ExternalArtifactPlane(
        GuardedExternalRoot(
            ExternalRootContract(
                "synthetic.assigned",
                "Synthetic development fixture",
                str(root),
                "/",
                OperatorStorageProfile.SCHEMA,
                1,
                None,
                None,
                (),
            )
        )
    )
    design = assigned_science_design(assignment())
    source = load_packaged_reactor_source()
    panel = ReactorAssignedPrefixPanel(
        design.native,
        tuple(
            episode
            for branch in design.native.branches
            for episode in acquire_prefix_branch(design.native, source, branch).episodes
        ),
    )
    plane.write(
        ArtifactWriteRequest(
            logical_artifact_id="synthetic.panel",
            relative_path="synthetic/panel.json",
            payload_schema=panel.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            publication_scope_id="synthetic.panel",
            publication_scope_relative_root="synthetic",
            payload=panel.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            parent_visibility_ceilings=(),
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
    )
    manifest = decode_canonical_bytes(
        (root / "synthetic/panel.json.manifest.json").read_bytes(),
        ArtifactManifest,
        maximum_bytes=1024**2,
    )
    identity = ObjectIdentity.from_record("synthetic.input", panel)
    projection = project_panel(
        design=design,
        panel=panel,
        artifact=manifest,
        source_qualification=identity,
        runtime_qualification=identity,
        observer_qualification=identity,
    )
    assert projection.extension is not None
    models = ExternalCandidatePayloadPlane(
        plane,
        "synthetic/models",
        "synthetic.models",
        VisibilityCeiling.DEVELOPMENT_ONLY,
        OutcomeAccess.EVALUATOR_REVEAL,
        (),
        1,
    )
    config = bind_finite_chain(projection, models)
    identification, qualification = identify_and_qualify(
        config=config,
        projection=projection.projection,
        extension=projection.extension,
        publisher=models,
        reader=models,
    )
    result = ReactorAssignedScienceResult(
        projection, config, identification, qualification
    )
    assert (
        decode_canonical_bytes(
            result.canonical_bytes(), type(result), maximum_bytes=16 * 1024**2
        )
        == result
    )
    assert result.projection.native_custody is None  # Never a receipted campaign.
    assert len(config.method.physical_independent_unit_ids) == 5


def test_assigned_candidate_compiles_all_providers_inputs_resources_and_output_schemas(
    tmp_path,
):
    selected = assignment()
    source = load_packaged_reactor_source()
    bundle = build_reactor_authoring(
        source=source,
        implementation_sha256="0" * 64,
        fresh_experiment_id="synthetic-assigned-prefix",
        assignment=selected,
        prior_census=prior_census(),
    )

    def save(name, record):
        path = tmp_path / name
        path.write_bytes(record.canonical_bytes())
        return path

    payloads = tuple(
        save(f"payload-{i}.json", record) for i, record in enumerate(bundle.payloads)
    )
    decoders = tuple(
        save(f"decoder-{i}.json", record)
        for i, record in enumerate(bundle.decoder_registrations)
    )
    api = EmpiricalLawhoodApi(
        repo_root=ROOT,
        external_root=None,
        candidate_context_provider=ReactorPrefixResponseCandidateContextProvider(bundle),
        candidate_capability_catalog=bundle.catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.aggregate,
        executable_factory_registry=EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
        study_extension_codec_registry=EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY,
    )
    result = api.compile_candidate(
        CompileCandidateRequest(
            save("authoring.json", bundle.authoring), payloads, decoders
        )
    )
    assert result.succeeded, (result.reason_codes, result.errors)
    report = result.payload.report
    candidate = report.candidate
    issued, _, _ = project_standard_study_extensions(
        candidate=candidate,
        extension_payload_bytes=tuple(
            record.canonical_bytes() for record in bundle.payloads
        ),
        decoder_registrations=bundle.decoder_registrations,
        extension_materializations=report.extension_materializations,
    )
    base = candidate.base_candidate.base_candidate
    native = next(p for p in bundle.payloads if hasattr(p, "branches"))
    resources = reactor_resource_envelope(
        base.protocol,
        ObjectIdentity.from_record(issued.issued_extension_set_id, issued),
        run_id="synthetic-assigned-prefix",
        native_config=native,
    )
    registry = bundle.standard_context.base.registry
    projected = compile_preissue_run_plan(
        run_plan_id="synthetic-assigned-prefix",
        candidate_record=base,
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        registry=registry,
        implementation_commit="0" * 40,
        issued_extension_set=issued,
        resource_envelope=resources,
        jit_census=None,
        jit_manifest=None,
    )
    execution = lower_run_plan(projected, registry)
    assert len(execution.tasks) == 21
    assert (
        len(
            {
                cell.physical_independent_unit_id
                for cell in resources.task_cells
                if cell.physical_independent_unit_id
            }
        )
        == 5
    )
    source_port = ExternalInputPayload.from_bytes(
        logical_artifact_id="synthetic-assigned-prefix.source-bundle",
        payload_schema=source.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        payload=source.canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(),
        lineage_parents=(),
        logical_content_sha256=source.fingerprint(),
    )
    for capability in registry.capabilities:
        binding = next(
            binding
            for binding in EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.aggregate.bindings
            if binding.capability_key == capability.capability_key
        )
        factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.provider_factory(
            binding.binding_id
        )
        records = tuple(
            p
            for p in bundle.payloads
            if p.SCHEMA in {r.record_schema for r in binding.accepted_config_types}
        )
        ports = tuple(
            ExecutablePlatformPort(
                key, source_port if key == "tbs-reactor-source-input" else Guard()
            )
            for key in binding.required_platform_port_keys
        )
        provider = factory.build_provider(
            registry=registry, records=records, platform_ports=ports
        )
        runners = provider.runners(registry)
        assert len(runners) == 1
        ForkingPickler.dumps(runners[0])
        assert provider.external_inputs(execution)
        assert provider.output_semantic_contracts(registry)


class Guard:
    def publish_candidate_payload(self, **kwargs):
        raise AssertionError("No publication in preflight")

    def read_candidate_payload(self, *args):
        raise AssertionError("No candidate read in preflight")

    def read_dependency(self, *args):
        raise AssertionError("No outcome read in preflight")
