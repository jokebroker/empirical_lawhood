"""Additive public method capability for sealed B/C reductions."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from .law_terminal import RegimeJointLawResult
from .control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from .control_prospective_closeout import RegimeDRootSealResult
from .control_prospective_reveal import RegimeDRootRevealResult
from .control_prospective_cohort import ReactorRegimeResponseCohortProspective
from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot
from .prospective_decision import CausalValidityRegimeAssignment
from .discovery_records import RegimeDiscoveryTraining, RegimeDiscoveryDevelopment
from .discovery_diagnostics import RegimePhaseDiagnostics, RegimeDiscoveryConfirmation
from .preassay_readout import RegimePreassayReadout, RegimeOpportunityReadout
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from .config import ReactorRegimeResponseDesign
from .calibration_pipeline import RegimeCalibrationPackage
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .qualification_pipeline import RegimeQualificationPackage
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science-regime.fit-and-seal",
    kind=CapabilityKind.EVALUATOR,
    maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            )
        )
    ),
    config_schema=ReactorRegimeResponseDesign.SCHEMA,
    config_schema_sha256=sha256(ReactorRegimeResponseDesign.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                ReactorRegimeResponseDesign.SCHEMA,
                RegimeCausalPreparation.SCHEMA,
                RegimePredictionSeal.SCHEMA,
                RegimeAssayPanel.SCHEMA,
                RegimeFitPackage.SCHEMA,
                RegimeDiscoveryTraining.SCHEMA,
                RegimeDiscoveryDevelopment.SCHEMA,
                RegimePhaseDiagnostics.SCHEMA,
                RegimeDiscoveryConfirmation.SCHEMA,
                RegimePreassayReadout.SCHEMA,
                RegimeNominationPackage.SCHEMA,
                RegimeCalibrationPackage.SCHEMA,
                RegimeQualificationPackage.SCHEMA,
                RegimeJointLawResult.SCHEMA,
                RegimePrivatePreparation.SCHEMA,
                CausalValidityRegimeAssignment.SCHEMA,
                RegimeDNativeRoot.SCHEMA,
                ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA,
                RegimeDRootSealResult.SCHEMA,
                RegimeDRootRevealResult.SCHEMA,
                ReactorRegimeResponseCohortProspective.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(sorted((
        RegimePredictionSeal.SCHEMA,
        RegimeFitPackage.SCHEMA,
        RegimeDiscoveryTraining.SCHEMA,
        RegimeDiscoveryDevelopment.SCHEMA,
        RegimePhaseDiagnostics.SCHEMA,
        RegimeDiscoveryConfirmation.SCHEMA,
        RegimePreassayReadout.SCHEMA,
        RegimeOpportunityReadout.SCHEMA,
        RegimeNominationPackage.SCHEMA,
        RegimeCalibrationPackage.SCHEMA,
        RegimeQualificationPackage.SCHEMA,
        RegimeJointLawResult.SCHEMA,
        CausalValidityRegimeAssignment.SCHEMA,
        ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA,
        RegimeDRootSealResult.SCHEMA,
        RegimeDRootRevealResult.SCHEMA,
        ReactorRegimeResponseCohortProspective.SCHEMA,
        ScientificAdjudicationRecord.SCHEMA,
    ))),
    implementation_sha256=sha256(
        b"reactor-regime-method:fit-seals-nomination-calibration-sole-owner-qualification-adjudication"
    ).hexdigest(),
    runtime_id="cpython-reactor-regime",
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 7200, 8 * 1024**3, 256 * 1024**2),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "fit-and-seal",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ReactorRegimeResponseDesign,),
    None,
    namespace="reactor-regime-response",
)
