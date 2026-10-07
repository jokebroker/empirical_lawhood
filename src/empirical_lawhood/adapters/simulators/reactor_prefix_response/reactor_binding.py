"""No-effect selection of the retained reactor native factories."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Protocol

from empirical_lawhood.adapters.composition.generated_executable_bindings import (
    EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY,
)
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.providers import ExternalInputPayload

from .batch_design import ReactorBatchSource
from .batch_input import _ReactorSourceSnapshot
from empirical_lawhood.adapters.methods.reactor_causal_response.native_benchmark import EmpiricalNativeEnvironment

_BINDINGS = {
    "batch": "binding.terminal-bench-science-batch.reactor-batch",
    "matched-replay-history": "binding.reactor-matched-replay-forecast.reactor-batch",
    "local": "binding.terminal-bench-science-local.native-acquisition",
    "feed": "binding.terminal-bench-science-feed.native-acquisition",
    "regime": "binding.terminal-bench-science-regime.native-assay",
    "finite-control-frontier": "binding.reactor-finite-control-frontier.native",
    "selected-action-response": "binding.reactor-selected-action-response.native",
    "staged-pulse-response": "binding.reactor-staged-pulse-response.native",
    "causal-response-study": "binding.reactor-causal-response.native-acquisition",
}

_SOURCE_PORTS = {
    "batch": "tbs-reactor-batch-source-input",
    "matched-replay-history": "reactor-matched-replay-source-input",
    "local": "reactor-local-native-source-input",
    "feed": "reactor-feed-native-source-input",
    "regime": "reactor-regime-native-source-input",
    "selected-action-response": "classical-source-input",
    "causal-response-study": "reactor-empirical-native-source-input",
}


def required_upstream_port_keys(route: str) -> tuple[str, ...]:
    """Use the installed factory roster, excluding only its source input port."""

    if route not in _BINDINGS or route in ("batch", "matched-replay-history"):
        raise ValueError("REACTOR_UPSTREAM_ROUTE_INVALID")
    binding = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.provider_factory(
        _BINDINGS[route]
    ).binding
    return tuple(
        key
        for key in binding.required_platform_port_keys
        if key != _SOURCE_PORTS.get(route)
    )


@dataclass(frozen=True, slots=True)
class ReactorSourceBinding(CanonicalRecord):
    """A selector of identities; it is never a store or a grant of authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/reactor-source-binding'
    route: str
    native_config_sha256: str
    source_bundle_sha256: str
    port_identities: tuple[tuple[str, ObjectIdentity], ...]
    evidence_role: str

    def __post_init__(self) -> None:
        validate_sha256(self.native_config_sha256, field_name="native_config_sha256")
        validate_sha256(self.source_bundle_sha256, field_name="source_bundle_sha256")
        if (
            tuple(key for key, _ in self.port_identities)
            != required_upstream_port_keys(self.route)
            or any(
                not isinstance(identity, ObjectIdentity)
                for _, identity in self.port_identities
            )
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError("REACTOR_UPSTREAM_BINDING_INVALID: exact ports or role")


@dataclass(frozen=True, slots=True)
class EnvironmentBoundReactorSourceBinding(ReactorSourceBinding):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/environment-bound-reactor-source-binding'
    VERSION: ClassVar[str] = '1.0.0'
    native_environment: EmpiricalNativeEnvironment

    def __post_init__(self) -> None:
        ReactorSourceBinding.__post_init__(self)
        if type(self.native_environment) is not EmpiricalNativeEnvironment:
            raise ValueError("REACTOR_NATIVE_ENVIRONMENT_REQUIRED")


@dataclass(frozen=True, slots=True)
class ReactorResolvedPort:
    identity: ObjectIdentity
    port: object


class ReactorTargetPortStore(Protocol):
    """Trusted installed context; a JSON file cannot implement this port."""

    def resolve_port(self, identity: ObjectIdentity) -> ReactorResolvedPort: ...


def _study_source_payload(
    source_root: Path,
    source: ReactorBatchSource,
    *,
    environment: EmpiricalNativeEnvironment | None = None,
    _snapshot: _ReactorSourceSnapshot | None = None,
) -> ExternalInputPayload:
    """Rebuild the declared source schema from bounded authenticated source bytes."""

    from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalComparatorSources, EmpiricalStudySource
    from empirical_lawhood.adapters.methods.reactor_causal_response.config import COMPARATOR_SOURCES

    if type(environment) is not EmpiricalNativeEnvironment:
        raise ValueError("REACTOR_NATIVE_ENVIRONMENT_REQUIRED")

    snapshot = _snapshot or _ReactorSourceSnapshot(source_root)
    if snapshot.root != source_root:
        raise ValueError("REACTOR_SOURCE_SNAPSHOT_ROOT_MISMATCH")
    census = snapshot.census()
    by_name = dict(census)
    comparators: list[tuple[str, str]] = []
    for arm, relative, expected in COMPARATOR_SOURCES:
        if by_name.get(relative) != expected:
            raise ValueError(f"REACTOR_COMPARATOR_PIN_MISMATCH: {arm} {relative}")
        comparators.append((arm, snapshot.read_member(relative, 4 * 1024**2).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")))
    if environment.source_files != tuple(census):
        raise ValueError("REACTOR_NATIVE_ENVIRONMENT_SOURCE_CENSUS_MISMATCH")
    study = EmpiricalStudySource(
        source,
        EmpiricalComparatorSources(tuple(sorted(comparators))),
        ObjectIdentity.from_record(environment.environment_id, environment),
    )
    return ExternalInputPayload.from_bytes(
        logical_artifact_id="reactor-authenticated-study-source",
        payload_schema=EmpiricalStudySource.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        payload=study.canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(),
        lineage_parents=(),
    )


def inspect_reactor_binding(
    route: str,
    native: CanonicalRecord,
    source: ReactorBatchSource,
    *,
    source_root: Path | None = None,
    upstream_binding: ReactorSourceBinding | None = None,
    port_store: ReactorTargetPortStore | None = None,
    _snapshot: _ReactorSourceSnapshot | None = None,
) -> dict[str, object]:
    """Construct a provider only when its actual closed port roster is present."""

    binding_id = _BINDINGS[route]
    factories = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    factory = factories.provider_factory(binding_id)
    binding = factory.binding
    if native.SCHEMA not in {
        item.record_schema for item in binding.accepted_config_types
    }:
        raise ValueError("REACTOR_NATIVE_CONFIG_BINDING_MISMATCH")
    if source.fingerprint() != getattr(
        native, "source_bundle_sha256", source.fingerprint()
    ):
        raise ValueError("REACTOR_SOURCE_CONFIG_MISMATCH")

    source_key = _SOURCE_PORTS.get(route)
    missing = tuple(
        key
        for key in binding.required_platform_port_keys
        if route not in ("batch", "matched-replay-history") or key != source_key
    )
    report: dict[str, object] = {
        "binding_selected": binding_id,
        "capability_key": binding.capability_key,
        "factory": type(factory).__name__,
        "required_port_keys": binding.required_platform_port_keys,
        "missing_port_keys": missing,
        "provider_built": False,
        "native_contact": False,
        "status": "UPSTREAM_PORTS_REQUIRED" if missing else "PROVIDER_BUILT",
    }
    if upstream_binding is None and missing:
        return report
    if upstream_binding is not None:
        if (
            upstream_binding.route != route
            or upstream_binding.native_config_sha256 != native.fingerprint()
            or upstream_binding.source_bundle_sha256 != source.fingerprint()
        ):
            raise ValueError("REACTOR_UPSTREAM_BINDING_MISMATCH")
        if port_store is None:
            raise ValueError("REACTOR_TARGET_PORT_STORE_REQUIRED")
        if not callable(getattr(port_store, "resolve_port", None)):
            raise ValueError("REACTOR_TARGET_PORT_STORE_INVALID")
        if source_root is None:
            raise ValueError("REACTOR_SOURCE_ROOT_REQUIRED")
        resolved: dict[str, object] = {}
        for key, identity in upstream_binding.port_identities:
            materialized_identity = None
            try:
                validate_binding = getattr(port_store, "validate_binding", None)
                if callable(validate_binding):
                    materialized_identity = validate_binding(
                        route=route,
                        native_config_sha256=native.fingerprint(),
                        key=key,
                        identity=identity,
                    )
                entry = port_store.resolve_port(identity)
            except LookupError as error:
                raise ValueError(f"REACTOR_TARGET_PORT_UNRESOLVED: {key}") from error
            if (
                not isinstance(entry, ReactorResolvedPort)
                or entry.identity != identity
                or entry.port is None
            ):
                raise ValueError(f"REACTOR_TARGET_PORT_IDENTITY_MISMATCH: {key}")
            if isinstance(entry.port, CanonicalRecord) and (
                ObjectIdentity.from_record(identity.object_id, entry.port)
                != (materialized_identity or identity)
            ):
                raise ValueError(f"REACTOR_TARGET_PORT_BYTES_MISMATCH: {key}")
            resolved[key] = entry.port
        payload = _study_source_payload(
            source_root,
            source,
            environment=getattr(upstream_binding, "native_environment", None),
            _snapshot=_snapshot,
        )
        if source_key is not None:
            resolved[source_key] = payload
        else:
            inputs_key = (
                "frontier-native-inputs"
                if route == "finite-control-frontier"
                else "classical-native-inputs"
            )
            inputs = resolved[inputs_key]
            matched = (
                ()
                if not isinstance(inputs, tuple)
                else tuple(
                    item
                    for item in inputs
                    if isinstance(item, ExternalInputPayload)
                    and item.payload_schema == payload.payload_schema
                    and item.source_sha256 == payload.source_sha256
                    and item.size_bytes == payload.size_bytes
                    and item.profile == payload.profile
                    and item.outcome_access is OutcomeAccess.OUTCOME_BLIND
                    and item.visibility_ceiling is VisibilityCeiling.DEVELOPMENT_ONLY
                )
            )
            if (
                len(matched) != 1
                or matched[0].maximum_bytes > 4 * 1024**2
                or sha256(b"".join(matched[0].chunks())).hexdigest()
                != payload.source_sha256
            ):
                raise ValueError("REACTOR_UPSTREAM_SOURCE_IDENTITY_MISMATCH")
        platform_ports = tuple(
            ExecutablePlatformPort(key, resolved[key])
            for key in binding.required_platform_port_keys
        )
        report["source_input_sha256"] = payload.source_sha256
    else:
        assert source_key is not None
        payload = ExternalInputPayload.from_bytes(
            logical_artifact_id=f"reactor-{route}-authenticated-source",
            payload_schema=ReactorBatchSource.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            payload=source.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            parent_visibility_ceilings=(),
            lineage_parents=(),
        )
        platform_ports = (ExecutablePlatformPort(source_key, payload),)
    registry = GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
    provider = factory.build_provider(
        registry=registry,
        records=(native,),
        platform_ports=platform_ports,
    )
    runners = provider.runners(registry)
    if (
        len(runners) != 1
        or runners[0].manifest.capability_key != binding.capability_key
    ):
        raise ValueError("REACTOR_PROVIDER_RUNNER_BINDING_MISMATCH")
    report["provider_built"] = True
    report["missing_port_keys"] = ()
    report["status"] = "PROVIDER_BUILT"
    report["runner_selected"] = type(runners[0]).__name__
    return report


__all__ = [
    'ReactorResolvedPort',
    "ReactorTargetPortStore",
    'ReactorSourceBinding',
    'EnvironmentBoundReactorSourceBinding',
    "inspect_reactor_binding",
    "required_upstream_port_keys",
]
