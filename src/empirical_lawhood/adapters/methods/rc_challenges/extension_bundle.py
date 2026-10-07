"""Static discovery leaf; scientific records and composition stay with their owner."""

from dataclasses import replace

from empirical_lawhood.adapters.simulator_morphism_challenges.extension_bundle import (
    EXTENSION_BUNDLE_CONTRIBUTIONS,
)


EXTENSION_BUNDLE_CONTRIBUTION = replace(
    EXTENSION_BUNDLE_CONTRIBUTIONS[0],
    contribution_id="extension-contribution.simulator-morphism-challenges",
    capability_manifests=tuple(sorted(
        (value for part in EXTENSION_BUNDLE_CONTRIBUTIONS for value in part.capability_manifests),
        key=lambda value: value.registry_id,
    )),
    candidate_capability_registrations=tuple(sorted(
        (value for part in EXTENSION_BUNDLE_CONTRIBUTIONS for value in part.candidate_capability_registrations),
        key=lambda value: value.registration_id,
    )),
    profile_registrations=tuple(sorted(
        (value for part in EXTENSION_BUNDLE_CONTRIBUTIONS for value in part.profile_registrations),
        key=lambda value: value.registration_id,
    )),
    config_decoders=tuple(sorted(
        (value for part in EXTENSION_BUNDLE_CONTRIBUTIONS for value in part.config_decoders),
        key=lambda value: value.object_id,
    )),
    runtime_providers=tuple(sorted(
        (value for part in EXTENSION_BUNDLE_CONTRIBUTIONS for value in part.runtime_providers),
        key=lambda value: value.object_id,
    )),
)
