"""Static pinned-source registration; no discovery-time source contact."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.discovery import ClassicalNomination
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.law_terminal import ClassicalLaws
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_lock import ClassicalFrozenRoot
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prospective_plan import ReactorStagedPulseResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.phase_records import ClassicalFirstScreen, ClassicalInducedBundle
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.prediction import ClassicalPredictionSeal
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalPreparation, ClassicalPrivate, ClassicalAssay
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation, FrontierPrivate, FrontierAssay
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .config import ClassicalNativeConfig
from .prospective import ClassicalNativeRoot

OUTPUT_RECORDS = (
    ClassicalPreparation,
    ClassicalPrivate,
    ClassicalAssay,
    ClassicalInducedBundle,
    ClassicalNativeRoot,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-staged-pulse-response.native",
    kind=CapabilityKind.SIMULATOR,
    config_schema=ClassicalNativeConfig.SCHEMA,
    config_schema_sha256=sha256(ClassicalNativeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            {
                r.SCHEMA
                for r in (
                    *OUTPUT_RECORDS,
                    ClassicalNativeConfig,
                    EmpiricalStudySource,
                    FrontierPreparation,
                    FrontierPrivate,
                    FrontierAssay,
                    ClassicalNomination,
                    ClassicalFirstScreen,
                    ClassicalPredictionSeal,
                    ClassicalFrozenRoot,
                    ReactorStagedPulseResponseProspectivePlan,
                    ClassicalLaws,
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
        b"reactor-staged-pulse-response:pinned-native-finite-words-nominal-refined-actual-two-cutoff-prefix"
    ).hexdigest(),
    runtime_id="cpython-reactor-staged-pulse-response",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ClassicalNativeConfig,),
    None,
    namespace="reactor-staged-pulse-response",
)
