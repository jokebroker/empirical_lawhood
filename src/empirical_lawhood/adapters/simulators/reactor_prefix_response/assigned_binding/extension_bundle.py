# SPDX-License-Identifier: MPL-2.0

"""Noncontact discovery for an additive assigned native panel."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from ..assigned import ReactorAssignedPrefixConfig, ReactorAssignedPrefixPanel
from ..extension_bundle import CAPABILITY as PUBLIC_CAPABILITY
from ..panel import ReactorSourceBundle

CAPABILITY = replace(
    PUBLIC_CAPABILITY,
    capability_key="assigned-reactor-prefix-response",
    config_schema=ReactorAssignedPrefixConfig.SCHEMA,
    config_schema_sha256=sha256(
        ReactorAssignedPrefixConfig.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=tuple(
        sorted((ReactorAssignedPrefixConfig.SCHEMA, ReactorSourceBundle.SCHEMA))
    ),
    output_schema_ids=(ReactorAssignedPrefixPanel.SCHEMA,),
    implementation_sha256=sha256(
        b"assigned-reactor-prefix:actual-native-seed-nested-paired-views"
    ).hexdigest(),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "assigned-reactor-prefix",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ReactorAssignedPrefixConfig,),
    None,
    namespace="assigned-reactor-prefix-response",
)
