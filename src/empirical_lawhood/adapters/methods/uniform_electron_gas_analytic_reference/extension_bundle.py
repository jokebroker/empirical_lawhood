"""Discover the attached uniform electron gas analytic falsifier and adjudication capability."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_contracts import UniformElectronGasAnalyticCheck, UniformElectronGasAnalyticEvaluationConfig, UniformElectronGasAnalyticPanel
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="uniform-electron-gas-analytic-reference.finite-q-check",
    kind=CapabilityKind.EVALUATOR,
    config_schema=UniformElectronGasAnalyticEvaluationConfig.SCHEMA,
    config_schema_sha256=sha256(
        UniformElectronGasAnalyticEvaluationConfig.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=(UniformElectronGasAnalyticPanel.SCHEMA,),
    output_schema_ids=tuple(
        sorted((UniformElectronGasAnalyticCheck.SCHEMA, ScientificAdjudicationRecord.SCHEMA))
    ),
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            ),
            key=lambda value: value.value,
        )
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 120, 1024**2, 1024**2),
    implementation_sha256=sha256(
        b"uniform-electron-gas-analytic-reference:realized-action-finite-q-slab-falsifier"
    ).hexdigest(),
    runtime_id="cpython-3.11-scipy-1.17.1-uniform-electron-gas-analytic-check",
    conformance_check_ids=(
        "finite-q-intercept-and-slab-falsifier",
        "realized-si-action-and-receiver-clock",
        "synthetic-reference-nonpromotable",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "finite-q-check",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (UniformElectronGasAnalyticEvaluationConfig,),
    None,
    namespace="uniform-electron-gas-analytic-reference",
)
