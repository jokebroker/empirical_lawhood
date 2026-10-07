"Additive native capability for the regime-response study's sealed two-stage assay."

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from empirical_lawhood.adapters.methods.reactor_regime_response.nomination_records import RegimeNominationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.calibration_pipeline import RegimeCalibrationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.qualification_pipeline import RegimeQualificationPackage
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.adapters.methods.reactor_regime_response.law_terminal import RegimeJointLawResult
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from empirical_lawhood.adapters.methods.reactor_regime_response.prospective_decision import CausalValidityRegimeAssignment
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from .config import ReactorRegimeNativeConfig
from .prospective_records import RegimeDNativeRoot

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science-regime.native-assay",
    kind=CapabilityKind.SIMULATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    permissions=tuple(
        sorted((
            *FINITE_ACTION_MANIFEST.permissions,
            CapabilityPermission.READ_FROZEN_MODELS,
            CapabilityPermission.READ_SEALED_OUTCOMES,
        ))
    ),
    config_schema=ReactorRegimeNativeConfig.SCHEMA,
    config_schema_sha256=sha256(ReactorRegimeNativeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                ReactorRegimeNativeConfig.SCHEMA,
                EmpiricalStudySource.SCHEMA,
                RegimeCausalPreparation.SCHEMA,
                RegimePrivatePreparation.SCHEMA,
                RegimePredictionSeal.SCHEMA,
                RegimeNominationPackage.SCHEMA,
                RegimeCalibrationPackage.SCHEMA,
                RegimeQualificationPackage.SCHEMA,
                RegimeJointLawResult.SCHEMA,
                CausalValidityRegimeAssignment.SCHEMA,
                ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted(
            (
                RegimeCausalPreparation.SCHEMA,
                RegimePrivatePreparation.SCHEMA,
                RegimeAssayPanel.SCHEMA,
                RegimeDNativeRoot.SCHEMA,
            )
        )
    ),
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 7200, 2 * 1024**3, 256 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-regime-native:sealed-panel-and-selected-prospective-preparation"
    ).hexdigest(),
    runtime_id="cpython-reactor-regime",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native-assay",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ReactorRegimeNativeConfig,),
    None,
    namespace="reactor-regime-response",
)
