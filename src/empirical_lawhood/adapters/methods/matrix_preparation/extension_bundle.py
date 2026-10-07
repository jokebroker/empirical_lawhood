"""Static preparation method discovery with separate scientific information ports."""

from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentKind, ExtensionContributionKind
from empirical_lawhood.adapters.composition.discovery import bundle, component, manifest
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT
from .continuation import PreparationProjectionContinuation
from .method_continuation import METHOD_CONTINUATION_ROLES, method_continuation_type
from .topology import DATA_OUTPUTS, INPUT_SCHEMAS, METHOD_CONFIG_TYPES, METHOD_ROLES, OUTPUT_MIB, STAGE_SCHEMA, WALL_SECONDS, task_output_schemas


METHOD_CAPABILITIES = tuple(
    manifest(
        role,
        METHOD_CONFIG_TYPES[role],
        CapabilityKind.LAW_IDENTIFIER
        if role == "description"
        else CapabilityKind.NUMERICAL_QUALIFIER
        if role == "numerical-semantics"
        else CapabilityKind.EVALUATOR
        if role == "evaluation"
        else CapabilityKind.TRANSFORM
        if role == "projection"
        else CapabilityKind.ANALYSIS,
        (
            METHOD_CONFIG_TYPES[role].SCHEMA,
            *INPUT_SCHEMAS[role],
            STAGE_SCHEMA,
            *((ObjectIdentity.SCHEMA,) if role == "description" else ()),
        ),
        tuple(task_output_schemas(role).values()),
        EvidenceCeiling.LOCAL_LAW
        if role in ("description", "evaluation")
        else EvidenceCeiling.RESPONSE,
        OutcomeAccess.EVALUATION_REVEALED
        if role == "evaluation"
        else OutcomeAccess.DEVELOPMENT_VISIBLE,
        WALL_SECONDS[role],
        namespace=DEVELOPMENT,
        resources=ResourceBudget(
            1, 2 * 1024**3, 0, WALL_SECONDS[role], 2 * 1024**3, OUTPUT_MIB[role] * 1024**2
        ),
        read_permission=CapabilityPermission.READ_DEVELOPMENT,
        implementation_id="matrix-preparation-projection-with-custodied-native-imports"
        if role == "projection"
        else "matrix-preparation-frozen-observable-and-independent-prospective-task-methods",
        conformance_ids=(
            "exact-root-role-before-artifact-read",
            "no-hidden-or-task-input-to-forecast",
            "separate-description-forecast-task-terminals",
            "shared-law-finalizer",
        ),
    )
    for role in METHOD_ROLES
)

_built = []
for role, capability in zip(METHOD_ROLES, METHOD_CAPABILITIES, strict=True):
    schemas = tuple(schema for _, schema in DATA_OUTPUTS[role])
    contribution, components = bundle(
        role,
        ExtensionContributionKind.METHOD,
        capability,
        (
            METHOD_CONFIG_TYPES[role],
            *((PreparationProjectionContinuation,) if role == "projection" else ()),
            *((method_continuation_type(role),) if role in METHOD_CONTINUATION_ROLES else ()),
        ),
        schemas[0] if schemas else None,
        namespace=DEVELOPMENT,
    )
    extra = tuple(
        component(
            f"{role}.hdf5-validator.{i}",
            ExtensionComponentKind.ARTIFACT_VALIDATOR,
            (schema,),
            ('empirical-lawhood/runtime/artifact-generic-validation',),
            capability.implementation_sha256,
            DEVELOPMENT,
        )
        for i, schema in enumerate(schemas[1:], start=1)
    )
    if extra:
        contribution = replace(
            contribution,
            artifact_validators=tuple(
                sorted(
                    (
                        *contribution.artifact_validators,
                        *(ObjectIdentity.from_record(c.registration_id, c) for c in extra),
                    ),
                    key=lambda r: r.object_id,
                )
            ),
        )
        components = (*components, *extra)
    _built.append((contribution, components))
METHOD_COMPONENTS = tuple(components for _, components in _built)
_contributions = tuple(c for c, _ in _built)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _contributions[0],
    contribution_id=f"extension-contribution.{DEVELOPMENT}.methods",
    capability_manifests=tuple(sorted(METHOD_CAPABILITIES, key=lambda c: c.registry_id)),
    candidate_capability_registrations=tuple(
        sorted(
            (v for c in _contributions for v in c.candidate_capability_registrations),
            key=lambda v: v.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (v for c in _contributions for v in c.profile_registrations),
            key=lambda v: v.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted({v for c in _contributions for v in c.config_decoders}, key=lambda v: v.object_id)
    ),
    artifact_validators=tuple(
        sorted(
            (v for c in _contributions for v in c.artifact_validators), key=lambda v: v.object_id
        )
    ),
    runtime_providers=tuple(
        sorted((v for c in _contributions for v in c.runtime_providers), key=lambda v: v.object_id)
    ),
)
