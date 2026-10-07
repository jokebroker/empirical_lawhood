"""Import-safe metadata constructors shared by this finite assay adapter binding."""

from empirical_lawhood.adapters.composition.discovery import (
    manifest as _manifest,
    component as _component,
    bundle as _bundle,
    executable as _executable,
)
from empirical_lawhood.adapters.composition.protocol_helpers import (
    CANONICAL_MEDIA_TYPE as CANONICAL_MEDIA_TYPE,
)

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityKind,
    CapabilityPermission,
)
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentRegistration, ExtensionComponentKind, ExtensionContributionKind
from empirical_lawhood.runtime.executable_bindings import ExecutableCapabilityBinding


def manifest(
    role: str,
    config: type[CanonicalRecord],
    kind: CapabilityKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    ceiling: EvidenceCeiling,
    access: OutcomeAccess,
    wall_seconds: int,
    *,
    namespace: str = "response-geometry-assay",
    resources: ResourceBudget | None = None,
    read_permission: CapabilityPermission | None = None,
    implementation_id: str = "frozen-frozen-common-assay-assay",
    conformance_ids: tuple[str, ...] = (
        "exact-frozen-assay-roster",
        "no-scientific-retry",
        "separate-root-view-action-clocks",
    ),
) -> CapabilityManifest:
    return _manifest(
        role,
        config,
        kind,
        inputs,
        outputs,
        ceiling,
        access,
        wall_seconds,
        namespace=namespace,
        resources=resources
        or ResourceBudget(
            1,
            3 * 1024**3,
            0,
            wall_seconds,
            {"source": 32 * 1024**2, "projection": 1024**3, "evaluation": 2 * 1024**3}[
                role
            ],
            17 * 1024**2,
        ),
        read_permission=read_permission
        if read_permission is not None
        else (
            CapabilityPermission.READ_OUTCOME_VISIBLE
            if role == "evaluation"
            else CapabilityPermission.READ_SEALED_OUTCOMES
        ),
        implementation_id=implementation_id,
        conformance_ids=conformance_ids,
    )


def component(
    key: str,
    kind: ExtensionComponentKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    implementation: str,
    namespace: str = "response-geometry-assay",
) -> ExtensionComponentRegistration:
    return _component(key, kind, inputs, outputs, implementation, namespace)


def bundle(
    role: str,
    kind: ExtensionContributionKind,
    capability: CapabilityManifest,
    records: tuple[type[CanonicalRecord], ...],
    hdf5_schema: str | None,
    *,
    namespace: str = "response-geometry-assay",
    maximum_config_bytes: int = 4 * 1024**2,
) -> tuple[ExtensionBundleContribution, tuple[ExtensionComponentRegistration, ...]]:
    return _bundle(
        role,
        kind,
        capability,
        records,
        hdf5_schema,
        namespace=namespace,
        maximum_config_bytes=maximum_config_bytes,
    )


def executable(
    capability: CapabilityManifest,
    components: tuple[ExtensionComponentRegistration, ...],
    records: tuple[type[CanonicalRecord], ...],
) -> ExecutableCapabilityBinding:
    return _executable(capability, components, records)
