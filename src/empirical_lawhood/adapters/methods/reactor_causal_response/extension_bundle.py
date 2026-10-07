"""Noncontact discovery of the empirical numerical operand producer."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, EvidenceCeiling
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from .config import EmpiricalRecipe
from .payload import DevelopmentBoundReactorResponsePayload
from .transport import CausalReactorRootEnvelope
from .terminal import EmpiricalQualificationResult
from .experiment_records import EmpiricalAcquisitionEnvelope, EmpiricalDiscoveryEnvelope, EmpiricalCalibrationEnvelope, EmpiricalQualificationTerminal

from .campaign_records import EmpiricalStudySource, TimedReactorConfirmationEnvelope, EmpiricalRootAnalysis, EmpiricalContributionResult, EmpiricalNativeBenchmarkResult

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-causal-response.numerical-method",
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
            ),
            key=lambda p: p.value,
        )
    ),
    config_schema=EmpiricalRecipe.SCHEMA,
    config_schema_sha256=sha256(EmpiricalRecipe.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                EmpiricalRecipe.SCHEMA,
                EmpiricalStudySource.SCHEMA,
                TimedReactorConfirmationEnvelope.SCHEMA,
                EmpiricalRootAnalysis.SCHEMA,
                EmpiricalContributionResult.SCHEMA,
                EmpiricalNativeBenchmarkResult.SCHEMA,
                DevelopmentBoundReactorResponsePayload.SCHEMA,
                CausalReactorRootEnvelope.SCHEMA,
                EmpiricalQualificationResult.SCHEMA,
                EmpiricalAcquisitionEnvelope.SCHEMA,
                EmpiricalDiscoveryEnvelope.SCHEMA,
                EmpiricalCalibrationEnvelope.SCHEMA,
                EmpiricalQualificationTerminal.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted(
            (
                EmpiricalDiscoveryEnvelope.SCHEMA,
                EmpiricalCalibrationEnvelope.SCHEMA,
                EmpiricalQualificationTerminal.SCHEMA,
                EmpiricalQualificationResult.SCHEMA,
                EmpiricalRootAnalysis.SCHEMA,
                EmpiricalContributionResult.SCHEMA,
                EmpiricalNativeBenchmarkResult.SCHEMA,
                ScientificAdjudicationRecord.SCHEMA,
            )
        )
    ),
    implementation_sha256=sha256(
        b"reactor-empirical-eight-root:causal-nine-ridge-rank32-descriptive-five-native"
    ).hexdigest(),
    runtime_id="cpython-reactor-causal-response",
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 12 * 3600, 256 * 1024**3, 256 * 1024**3),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "numerical-method",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (EmpiricalRecipe,),
    None,
    namespace="reactor-causal-response",
)
