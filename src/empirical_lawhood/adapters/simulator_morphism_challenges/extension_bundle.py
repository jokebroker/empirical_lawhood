"""Installed RC challenge descriptors, separate from actual producing-source custody."""

from hashlib import sha256

from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from .authoring import simulator_morphism_challenges_registry
from .issued_inputs import CAPABILITY_INPUT_TYPES
from .capability_configs import CAPABILITY_CONFIG_TYPES


# This hashes the declared installed implementation descriptor, not source bytes.
# Exact current producing-code closure and independent method hashes are retained
# and verified by issued_inputs/runtime_provider and the scientific phase freezes.
INSTALLED_IMPLEMENTATION_DESCRIPTOR = (
    "rc-v3-current-four-phase-binding-v1",
    "dense-observer-independent-sparse-generator",
    "original-canonical-v3-descriptor-depth-random-fibre-pcg64",
    "unscaled-relative-svd-strict-1e-12",
    "individual-row-equilibrated-relative-svd-strict-1e-12",
    "36-disorder-blocks-3-nested-scales-108-adjudications",
    "whole-block-bootstrap-10000-seed3140159-alpha0.05-over18",
    "authenticated-current-numeric-source-export-and-prior-exposure-census",
)
INSTALLED_IMPLEMENTATION_SHA256 = sha256(canonical_json_bytes(INSTALLED_IMPLEMENTATION_DESCRIPTOR)).hexdigest()
CAPABILITIES = simulator_morphism_challenges_registry(implementation_sha256=INSTALLED_IMPLEMENTATION_SHA256).capabilities
_ASSEMBLED = tuple(
    bundle(capability.capability_key.removeprefix("simulator-morphism-challenges."),
           ExtensionContributionKind.SOURCE, capability,
           (CAPABILITY_CONFIG_TYPES[capability.capability_key], CAPABILITY_INPUT_TYPES[capability.capability_key]), None,
           namespace="simulator-morphism-challenges", maximum_config_bytes=8 * 1024**2)
    for capability in CAPABILITIES
)
EXTENSION_BUNDLE_CONTRIBUTIONS = tuple(value[0] for value in _ASSEMBLED)
COMPONENTS_BY_CAPABILITY = {capability.capability_key: value[1] for capability, value in zip(CAPABILITIES, _ASSEMBLED, strict=True)}
