"""Static evaluator discovery for RC solver agreement and charge balance."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderNativePanel, ResistorCapacitorLadderNumericalCheck
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from .contracts import ResistorCapacitorLadderEvaluationConfig

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="rc-ladder-response.numerical-check",
    kind=CapabilityKind.EVALUATOR,
    config_schema=ResistorCapacitorLadderEvaluationConfig.SCHEMA,
    config_schema_sha256=sha256(ResistorCapacitorLadderEvaluationConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(ResistorCapacitorLadderNativePanel.SCHEMA,),
    output_schema_ids=tuple(
        sorted((ResistorCapacitorLadderNumericalCheck.SCHEMA, ScientificAdjudicationRecord.SCHEMA))
    ),
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            ),
            key=lambda permission: permission.value,
        )
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 120, 512 * 1024, 1024**2),
    implementation_sha256=sha256(
        b"rc-ladder-response:two-view-numerical-check-and-grid-charge-falsifier"
    ).hexdigest(),
    runtime_id="cpython-3.11-rc-ladder-response-numerical-check",
    conformance_check_ids=(
        "frozen-voltage-and-charge-limits",
        "one-model-unit-two-native-views",
        "synthetic-numerical-ceiling",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "numerical-check",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ResistorCapacitorLadderEvaluationConfig,),
    None,
    namespace="rc-ladder-response",
)
