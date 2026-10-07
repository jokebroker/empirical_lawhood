"""Static discovery for the pinned native bridge and complete pulse macros."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation, FrontierPrivate, FrontierAssay, FrontierRetainedRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.prospective_lock import FrontierFrozenRoot
from .config import FrontierNativeConfig
from .prospective import FrontierNativeRoot

OUTPUT_RECORDS = (FrontierPreparation, FrontierPrivate, FrontierAssay, FrontierNativeRoot)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-finite-control-frontier.native",
    kind=CapabilityKind.SIMULATOR,
    config_schema=FrontierNativeConfig.SCHEMA,
    config_schema_sha256=sha256(FrontierNativeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            {
                r.SCHEMA
                for r in (
                    *OUTPUT_RECORDS,
                    FrontierNativeConfig,
                    EmpiricalStudySource,
                    FrontierRetainedRoot,
                    FrontierPredictionSeal,
                    FrontierFrozenRoot,
                    ReactorFiniteControlFrontierProspectivePlan,
                )
            }
        )
    ),
    output_schema_ids=tuple(sorted(r.SCHEMA for r in OUTPUT_RECORDS)),
    permissions=tuple(
        sorted(
            (
                *FINITE_ACTION_MANIFEST.permissions,
                CapabilityPermission.READ_FROZEN_MODELS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
            )
        )
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 1800, 8 * 1024**3, 512 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-finite-control-frontier:pinned-native-unshifted-exploration-complete-twelve-command-pulse-paired-refinement"
    ).hexdigest(),
    runtime_id="cpython-reactor-finite-control-frontier",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (FrontierNativeConfig,),
    None,
    namespace="reactor-finite-control-frontier",
)
