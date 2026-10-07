"Static discovery for the closed nine-stage classical staged-pulse reductions."

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation, FrontierAssay
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.prospective import ClassicalNativeRoot
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .calibration import ClassicalCalibration
from .comparison import ClassicalComparison
from .config import ClassicalStage
from .discovery import ClassicalNomination
from .law_terminal import ClassicalLaws
from .measurement import ClassicalMeasuredRoot
from .prospective_cohort import ReactorStagedPulseResponseProspectiveCohort
from .prospective_lock import ClassicalFrozenRoot
from .prospective_plan import ReactorStagedPulseResponseProspectivePlan
from .prospective_reveal import ClassicalRevealedRoot
from .prospective_seal import ClassicalSealedRoot
from .phase_records import ClassicalFirstScreen, ClassicalInducedBundle
from .prediction import ClassicalPredictionSeal
from .qualification import ClassicalQualification
from .records import ClassicalPreparation, ClassicalAssay

OUTPUT_RECORDS = (
    ClassicalCalibration,
    ClassicalNomination,
    ClassicalFirstScreen,
    ClassicalPredictionSeal,
    ClassicalMeasuredRoot,
    ClassicalQualification,
    ClassicalLaws,
    ReactorStagedPulseResponseProspectivePlan,
    ClassicalFrozenRoot,
    ClassicalSealedRoot,
    ClassicalRevealedRoot,
    ReactorStagedPulseResponseProspectiveCohort,
    ClassicalComparison,
    ScientificAdjudicationRecord,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-staged-pulse-response.method",
    kind=CapabilityKind.EVALUATOR,
    config_schema=ClassicalStage.SCHEMA,
    config_schema_sha256=sha256(ClassicalStage.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            {
                r.SCHEMA
                for r in (
                    *OUTPUT_RECORDS,
                    ClassicalStage,
                    ClassicalPreparation,
                    ClassicalAssay,
                    ClassicalInducedBundle,
                    ClassicalNativeRoot,
                    EmpiricalStudySource,
                    FrontierDevelopment,
                    FrontierMeasuredRoot,
                    FrontierPreparation,
                    FrontierAssay,
                    FrontierPredictionSeal,
                )
            }
        )
    ),
    output_schema_ids=tuple(sorted(r.SCHEMA for r in OUTPUT_RECORDS)),
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
    maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 10800, 8 * 1024**3, 512 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-staged-pulse-response:fixed-calibration-receiver-map-finite-law-owner-prepared-use-bounded-sequence-root-families"
    ).hexdigest(),
    runtime_id="cpython-reactor-staged-pulse-response",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "method",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ClassicalStage,),
    None,
    namespace="reactor-staged-pulse-response",
)
