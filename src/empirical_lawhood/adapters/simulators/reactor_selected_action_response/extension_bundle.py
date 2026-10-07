"Static discovery for the pinned reactor's selected-action preparation and delivery."

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalCausal, ClassicalPrivate, ClassicalAssay, ClassicalDecision, ClassicalQualification
from empirical_lawhood.adapters.methods.reactor_selected_action_response.law_terminal import ClassicalLaw
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .config import ClassicalNativeConfig
from .prospective_records import ClassicalNativeRoot

OUTPUT_RECORDS = (ClassicalCausal, ClassicalPrivate, ClassicalAssay, ClassicalNativeRoot)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-selected-action-response.native",
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
                    ClassicalDecision,
                    ClassicalQualification,
                    ClassicalLaw,
                    ReactorSelectedActionResponseProspectivePlan,
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
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 1800, 8 * 1024**3, 256 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-selected-action-response:native-pinned-medium-causal-preparation-sealed-delivery"
    ).hexdigest(),
    runtime_id="cpython-reactor-selected-action-response",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ClassicalNativeConfig,),
    None,
    namespace="reactor-selected-action-response",
)
