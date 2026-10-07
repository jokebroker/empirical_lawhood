"""Static discovery for the selected-action qualification/control reductions."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.prospective_records import ClassicalNativeRoot
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .config import ClassicalDesign
from .records import ClassicalCausal, ClassicalPrivate, ClassicalAssay, ClassicalDecision, ClassicalQualification
from .law_terminal import ClassicalLaw
from .control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from .control_prospective_closeout import ClassicalSealResult
from .control_prospective_reveal import ClassicalRevealResult
from .control_prospective_cohort import ReactorSelectedActionResponseProspectiveCohort

OUTPUT_RECORDS = (
    ClassicalDecision,
    ClassicalQualification,
    ClassicalLaw,
    ReactorSelectedActionResponseProspectivePlan,
    ClassicalSealResult,
    ClassicalRevealResult,
    ReactorSelectedActionResponseProspectiveCohort,
    ScientificAdjudicationRecord,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-selected-action-response.method",
    kind=CapabilityKind.EVALUATOR,
    config_schema=ClassicalDesign.SCHEMA,
    config_schema_sha256=sha256(ClassicalDesign.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            {
                r.SCHEMA
                for r in (
                    *OUTPUT_RECORDS,
                    ClassicalDesign,
                    ClassicalCausal,
                    ClassicalPrivate,
                    ClassicalAssay,
                    ClassicalNativeRoot,
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
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 1800, 8 * 1024**3, 256 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-selected-action-response:selected-bound-sole-local-law-finite-admission-prepared-controller-use"
    ).hexdigest(),
    runtime_id="cpython-reactor-selected-action-response",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "method",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ClassicalDesign,),
    None,
    namespace="reactor-selected-action-response",
)
