"""An upstream identity file cannot stand in for installed live reactor ports."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.simulators.reactor_prefix_response.prepared_input import ReactorPreparedInput, check_prepared_input
from empirical_lawhood.adapters.simulators.reactor_prefix_response.reactor_binding import ReactorResolvedPort, ReactorSourceBinding, EnvironmentBoundReactorSourceBinding, required_upstream_port_keys
from empirical_lawhood.api.codecs import load_registered_authoring
from empirical_lawhood.kernel.provenance import ObjectIdentity

ROOT = Path(__file__).parents[1]


def _identity(key: str) -> ObjectIdentity:
    return ObjectIdentity(
        f"empirical-lawhood.synthetic.{key}",
        'empirical-lawhood/test/synthetic-port',
        "1.0.0",
        "0" * 64,
    )


def _binding(
    native_sha: str, source_sha: str, route: str = "local"
) -> ReactorSourceBinding:
    from hashlib import sha256
    from empirical_lawhood.adapters.methods.reactor_causal_response.native_benchmark import EmpiricalNativeEnvironment

    source = os.environ.get("REACTOR_HELD_SOURCE_ROOT")
    if source is None:
        return ReactorSourceBinding(
            route,
            native_sha,
            source_sha,
            tuple((key, _identity(key)) for key in required_upstream_port_keys(route)),
            "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
        )
    root = Path(source)
    environment = EmpiricalNativeEnvironment(
        "synthetic.native-environment",
        "sha256:" + "0" * 64,
        "749bc764b667502d47d2d99de988ee6e089b6aa4",
        tuple(
            (p.relative_to(root).as_posix(), sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ),
    )
    return EnvironmentBoundReactorSourceBinding(
        route,
        native_sha,
        source_sha,
        tuple((key, _identity(key)) for key in required_upstream_port_keys(route)),
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
        environment,
    )


def test_upstream_binding_has_the_exact_closed_factory_roster(tmp_path: Path) -> None:
    assert [
        len(required_upstream_port_keys(route))
        for route in (
            "local",
            "feed",
            "regime",
            "finite-control-frontier",
            "selected-action-response",
            "staged-pulse-response",
            "causal-response-study",
        )
    ] == [2, 2, 6, 5, 4, 6, 4]
    binding = _binding("a" * 64, "b" * 64)
    path = tmp_path / "upstream.json"
    path.write_bytes(binding.canonical_bytes())
    assert (
        load_registered_authoring(
            path,
            root_schemas={
                ReactorSourceBinding.SCHEMA: ReactorSourceBinding,
                EnvironmentBoundReactorSourceBinding.SCHEMA: EnvironmentBoundReactorSourceBinding,
            },
            maximum_bytes=16 * 1024,
        )
        == binding
    )
    with pytest.raises(ValueError, match="REACTOR_UPSTREAM_BINDING_INVALID"):
        replace(binding, port_identities=binding.port_identities[:1])
    with pytest.raises(ValueError, match="REACTOR_UPSTREAM_BINDING_INVALID"):
        replace(binding, evidence_role="PROSPECTIVE")


@pytest.mark.held
def test_upstream_binding_requires_a_trusted_store_before_provider() -> None:
    source_name = os.environ.get("REACTOR_HELD_SOURCE_ROOT")
    discovery_name = os.environ.get("REACTOR_LOCAL_DISCOVERY")
    if not source_name or not discovery_name:
        pytest.skip("authentic held reactor source and discovery are not mounted")
    config = load_registered_authoring(
        ROOT / "experiments/reactor-response/local-input.json",
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    source_root, discovery = Path(source_name), Path(discovery_name)
    preflight = check_prepared_input(
        config, source_root=source_root, discovery_file=discovery
    )
    binding = _binding(preflight["native_config_sha256"], preflight["source_sha256"])
    with pytest.raises(ValueError, match="REACTOR_UPSTREAM_BINDING_MISMATCH"):
        check_prepared_input(
            config,
            source_root=source_root,
            discovery_file=discovery,
            upstream_binding=replace(binding, source_bundle_sha256="f" * 64),
        )
    with pytest.raises(ValueError, match="REACTOR_TARGET_PORT_STORE_REQUIRED"):
        check_prepared_input(
            config,
            source_root=source_root,
            discovery_file=discovery,
            upstream_binding=binding,
        )

    class SyntheticGuardedPort:
        def read_dependency(self, *args: object) -> None:
            pass

        def task(self, *args: object) -> None:
            pass

    class SyntheticStore:
        def __init__(self, wrong_identity: bool = False) -> None:
            self.wrong_identity = wrong_identity

        def resolve_port(self, identity: ObjectIdentity) -> ReactorResolvedPort:
            selected = (
                replace(identity, object_fingerprint="f" * 64)
                if self.wrong_identity
                else identity
            )
            return ReactorResolvedPort(selected, SyntheticGuardedPort())

    with pytest.raises(ValueError, match="REACTOR_TARGET_PORT_IDENTITY_MISMATCH"):
        check_prepared_input(
            config,
            source_root=source_root,
            discovery_file=discovery,
            upstream_binding=binding,
            port_store=SyntheticStore(wrong_identity=True),
        )
    report = check_prepared_input(
        config,
        source_root=source_root,
        discovery_file=discovery,
        upstream_binding=binding,
        port_store=SyntheticStore(),
    )
    assert report["provider_built"] is True
    assert report["runner_selected"] == 'LocalNativeRunner'
    assert report["source_input_sha256"] != report["source_sha256"]
    assert report["native_contact"] is False
    assert report["native_tasks_executed"] == 0


@pytest.mark.held
def test_frontier_input_tuple_must_contain_the_authenticated_study_source() -> None:
    source_name = os.environ.get("REACTOR_HELD_SOURCE_ROOT")
    discovery_name = os.environ.get("REACTOR_LOCAL_DISCOVERY")
    if not source_name or not discovery_name:
        pytest.skip("authentic held reactor source and discovery are not mounted")
    config = load_registered_authoring(
        ROOT / 'experiments/reactor-response/finite-control-frontier-input.json',
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    source_root, discovery = Path(source_name), Path(discovery_name)
    preflight = check_prepared_input(
        config, source_root=source_root, discovery_file=discovery
    )
    binding = _binding(
        preflight["native_config_sha256"], preflight["source_sha256"], "finite-control-frontier"
    )

    class WrongSourceStore:
        def resolve_port(self, identity: ObjectIdentity) -> ReactorResolvedPort:
            value: object = (
                ()
                if identity.object_id.endswith("frontier-native-inputs")
                else object()
            )
            return ReactorResolvedPort(identity, value)

    with pytest.raises(ValueError, match="REACTOR_UPSTREAM_SOURCE_IDENTITY_MISMATCH"):
        check_prepared_input(
            config,
            source_root=source_root,
            discovery_file=discovery,
            upstream_binding=binding,
            port_store=WrongSourceStore(),
        )


@pytest.mark.parametrize(
    "route",
    (
        "local",
        "feed",
        "regime",
        "finite-control-frontier",
        "selected-action-response",
        "staged-pulse-response",
        "causal-response-study",
    ),
)
@pytest.mark.held
def test_all_reactor_factories_with_synthetic_guarded_ports(route: str) -> None:
    """EXPOSED_DEVELOPMENT_NONPROMOTABLE: no task, publication or grant is real."""
    from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ClassicalDesign, ClassicalStage, ClassicalUpstream
    from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import FrontierDesign
    from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment
    from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.phase import FrontierPhase, FrontierUpstream
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_input import load_batch_source
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.reactor_binding import _study_source_payload
    from empirical_lawhood.kernel.references import ArtifactIdentity

    source_name, discovery_name = (
        os.environ.get("REACTOR_HELD_SOURCE_ROOT"),
        os.environ.get("REACTOR_LOCAL_DISCOVERY"),
    )
    if not source_name or not discovery_name:
        pytest.skip("authentic held source and discovery are not mounted")
    source_root = Path(source_name)
    discovery = None if route == "causal-response-study" else Path(discovery_name)
    config = load_registered_authoring(
        ROOT / f"experiments/reactor-response/{route}-input.json",
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    preflight = check_prepared_input(
        config, source_root=source_root, discovery_file=discovery
    )
    binding = _binding(
        preflight["native_config_sha256"], preflight["source_sha256"], route
    )
    receipt = ObjectIdentity(
        "synthetic.receipt",
        'empirical-lawhood/runtime/canonical-task-receipt',
        "1.0.0",
        "a" * 64,
    )
    artifact = ArtifactIdentity(
        "synthetic.parent",
        "synthetic-parent",
        'empirical-lawhood/test/parent',
        "b" * 64,
        "application/json",
        2,
    )

    class GuardedPort:
        authority = resources = issued_programme = _identity("synthetic-authority")
        continuation = None

        def forbidden(self, *args: object, **kwargs: object) -> None:
            raise AssertionError(
                "input preflight crossed a native or publication boundary"
            )

        read_dependency = task = read_candidate_payload = forbidden
        open_control_store = open_prepared_store = freeze_clock = create_source = (
            forbidden
        )

    class Store:
        def resolve_port(self, identity: ObjectIdentity) -> ReactorResolvedPort:
            key = identity.object_id.removeprefix("empirical-lawhood.synthetic.")
            if key == "frontier-phase":
                value = FrontierPhase(
                    FrontierDesign(),
                    "C",
                    (
                        FrontierUpstream(
                            "development",
                            replace(
                                artifact, payload_schema=FrontierDevelopment.SCHEMA
                            ),
                            receipt,
                            "synthetic-run",
                        ),
                    ),
                )
            elif key == "classical-stage":
                value = ClassicalStage(
                    ClassicalDesign(),
                    "base-menu-comparison-QUALIFICATION",
                    (
                        ClassicalUpstream(
                            "nomination", artifact, receipt, "synthetic-run"
                        ),
                    ),
                )
            elif key.endswith("-prior-artifacts"):
                value = ()
            elif key in ("frontier-native-inputs", "classical-native-inputs"):
                value = (
                    _study_source_payload(
                        source_root,
                        load_batch_source(source_root),
                        environment=binding.native_environment,
                    ),
                )
            else:
                value = GuardedPort()
            # Canonical phase/stage identities must bind their actual bytes even
            # in a fixture. Other ports are deliberately guarded synthetic ones.
            return ReactorResolvedPort(identity, value)

    store = Store()
    pairs = []
    for key, identity in binding.port_identities:
        port = store.resolve_port(identity).port
        if hasattr(port, "canonical_bytes"):
            identity = ObjectIdentity.from_record(identity.object_id, port)
        pairs.append((key, identity))
    binding = replace(binding, port_identities=tuple(pairs))
    report = check_prepared_input(
        config,
        source_root=source_root,
        discovery_file=discovery,
        upstream_binding=binding,
        port_store=store,
    )
    assert report["provider_built"] is True
    assert report["missing_port_keys"] == ()
    assert report["native_tasks_executed"] == 0
    assert report["native_contact"] is False
