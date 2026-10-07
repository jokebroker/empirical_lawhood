# SPDX-License-Identifier: MPL-2.0

"""Assigned panel enters the same finite-action identification and finalizer."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorAssignedPrefixPanel
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from ..extension_bundle import CAPABILITY as PUBLIC_CAPABILITY
from .records import ReactorAssignedScienceDesign, ReactorAssignedScienceResult

INPUT_TYPES = (ReactorAssignedScienceDesign, ReactorAssignedPrefixPanel)
OUTPUT_TYPES = (ReactorAssignedScienceResult, ScientificAdjudicationRecord)
CAPABILITY = replace(
    PUBLIC_CAPABILITY,
    capability_key="assigned-reactor-finite-chain",
    config_schema=ReactorAssignedScienceDesign.SCHEMA,
    config_schema_sha256=sha256(
        ReactorAssignedScienceDesign.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=tuple(sorted(t.SCHEMA for t in INPUT_TYPES)),
    output_schema_ids=tuple(sorted(t.SCHEMA for t in OUTPUT_TYPES)),
    implementation_sha256=sha256(
        b"assigned-finite-chain:five-unit-custody-existing-finite-owner"
    ).hexdigest(),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "assigned-finite-chain",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ReactorAssignedScienceDesign,),
    None,
    namespace="assigned-reactor-prefix-response",
)
