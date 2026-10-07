"""Static discovery of retained-measurement development analysis; no native capability."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.response_geometry_prospective.development_continuation import RETAINED_ANALYSIS_RUN, ResponseGeometryDevelopmentAnalysisContinuationConfig
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_records import DEVELOPMENT_FIT_SCHEMA
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_terminal import DEVELOPMENT_VALIDATION_SCHEMA
from empirical_lawhood.adapters.methods.response_geometry_development.extension_bundle import DEVELOPMENT_CAPABILITIES, DEVELOPMENT_CONFIG_TYPES, DEVELOPMENT_ROLES
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import bundle
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentKind, ExtensionContributionKind

RETAINED_ANALYSIS_ROLES = ("development", "assessment", "evaluation")
RETAINED_ANALYSIS_CONFIG_TYPES = tuple(
    DEVELOPMENT_CONFIG_TYPES[DEVELOPMENT_ROLES.index(role)] for role in RETAINED_ANALYSIS_ROLES
)
RETAINED_ANALYSIS_CAPABILITIES = tuple(
    replace(
        DEVELOPMENT_CAPABILITIES[DEVELOPMENT_ROLES.index(role)],
        capability_key=f"{RETAINED_ANALYSIS_RUN}.{role}",
        runtime_id=f"{RETAINED_ANALYSIS_RUN}.{role}",
        input_schema_ids=tuple(
            sorted(
                (
                    *DEVELOPMENT_CAPABILITIES[DEVELOPMENT_ROLES.index(role)].input_schema_ids,
                    *(
                        (ResponseGeometryDevelopmentAnalysisContinuationConfig.SCHEMA,)
                        if role in ("development", "assessment")
                        else ()
                    ),
                )
            )
        ),
        implementation_sha256=sha256(
            f"{RETAINED_ANALYSIS_RUN}.{role}.retained-measurement-bindings".encode()
        ).hexdigest(),
        conformance_check_ids=(
            "exact-retained-development-custody-and-splits",
            "no-native-capability-or-task",
            "separate-analysis-input-publication",
            "unchanged-development-scientific-owners",
        ),
    )
    for role in RETAINED_ANALYSIS_ROLES
)
_BUILT = tuple(
    bundle(
        role,
        ExtensionContributionKind.METHOD,
        capability,
        (ResponseGeometryDevelopmentAnalysisContinuationConfig,),
        hdf5,
        namespace=RETAINED_ANALYSIS_RUN,
    )
    for role, capability, kind, hdf5 in zip(
        RETAINED_ANALYSIS_ROLES,
        RETAINED_ANALYSIS_CAPABILITIES,
        RETAINED_ANALYSIS_CONFIG_TYPES,
        (DEVELOPMENT_FIT_SCHEMA, DEVELOPMENT_VALIDATION_SCHEMA, None),
        strict=True,
    )
)
_SHARED_DECODER = next(
    value
    for value in _BUILT[0][1]
    if value.kind is ExtensionComponentKind.CONFIG_DECODER
)
RETAINED_ANALYSIS_COMPONENTS = tuple(
    (
        _SHARED_DECODER,
        *(
            value
            for value in components
            if value.kind is not ExtensionComponentKind.CONFIG_DECODER
        ),
    )
    for _, components in _BUILT
)
_CONTRIBUTIONS = tuple(
    replace(
        contribution, config_decoders=contribution.config_decoders if i == 0 else ()
    )
    for i, (contribution, _) in enumerate(_BUILT)
)
EXTENSION_BUNDLE_CONTRIBUTION = replace(
    _CONTRIBUTIONS[0],
    contribution_id=f"extension-contribution.{RETAINED_ANALYSIS_RUN}.methods",
    capability_manifests=tuple(
        sorted(RETAINED_ANALYSIS_CAPABILITIES, key=lambda value: value.registry_id)
    ),
    candidate_capability_registrations=tuple(
        sorted(
            (
                value
                for c in _CONTRIBUTIONS
                for value in c.candidate_capability_registrations
            ),
            key=lambda value: value.manifest.registry_id,
        )
    ),
    profile_registrations=tuple(
        sorted(
            (value for c in _CONTRIBUTIONS for value in c.profile_registrations),
            key=lambda value: value.registration_id,
        )
    ),
    config_decoders=tuple(
        sorted(
            (value for c in _CONTRIBUTIONS for value in c.config_decoders),
            key=lambda value: value.object_id,
        )
    ),
    artifact_validators=tuple(
        sorted(
            (value for c in _CONTRIBUTIONS for value in c.artifact_validators),
            key=lambda value: value.object_id,
        )
    ),
    runtime_providers=tuple(
        sorted(
            (value for c in _CONTRIBUTIONS for value in c.runtime_providers),
            key=lambda value: value.object_id,
        )
    ),
)
