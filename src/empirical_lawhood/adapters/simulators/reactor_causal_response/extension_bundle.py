"""Import-safe empirical acquisition discovery, separate from its effectful runner."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource, TimedReactorConfirmationEnvelope
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalAcquisitionEnvelope, EmpiricalDiscoveryEnvelope, EmpiricalCalibrationEnvelope, EmpiricalQualificationTerminal
from empirical_lawhood.adapters.methods.reactor_causal_response.transport import CausalReactorRootEnvelope
from .config import EmpiricalNativeConfig, SOURCE_IMPLEMENTATION

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-causal-response.native-acquisition",
    kind=CapabilityKind.SIMULATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    permissions=tuple(
        sorted((*FINITE_ACTION_MANIFEST.permissions, CapabilityPermission.READ_FROZEN_MODELS))
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    config_schema=EmpiricalNativeConfig.SCHEMA,
    config_schema_sha256=sha256(EmpiricalNativeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                EmpiricalNativeConfig.SCHEMA,
                EmpiricalStudySource.SCHEMA,
                EmpiricalQualificationTerminal.SCHEMA,
                EmpiricalDiscoveryEnvelope.SCHEMA,
                EmpiricalCalibrationEnvelope.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted(
            (
                EmpiricalAcquisitionEnvelope.SCHEMA,
                CausalReactorRootEnvelope.SCHEMA,
                TimedReactorConfirmationEnvelope.SCHEMA,
            )
        )
    ),
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 14400, 256 * 1024**2, 4 * 1024**3),
    implementation_sha256=SOURCE_IMPLEMENTATION,
    runtime_id="cpython-3.11.14-numpy-2.4.6-reactor-causal-response",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native-acquisition",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (EmpiricalNativeConfig,),
    None,
    namespace="reactor-causal-response",
)
