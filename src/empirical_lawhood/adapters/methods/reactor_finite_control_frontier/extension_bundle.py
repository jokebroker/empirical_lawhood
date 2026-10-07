"""Static discovery of finite frontier reductions and their scientific owners."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.prospective import FrontierNativeRoot
from .phase import FrontierPhase
from .records import FrontierPreparation, FrontierAssay
from .selection import FrontierPredictionSeal
from .measurement import FrontierMeasuredRoot
from .discovery import FrontierDevelopment
from .qualification import FrontierQualification
from .law_terminal import FrontierLaws
from .prospective_plan import ReactorFiniteControlFrontierProspectivePlan
from .prospective_lock import FrontierFrozenRoot
from .prospective_seal import FrontierSealedRoot
from .prospective_reveal import FrontierRevealedRoot
from .prospective_cohort import ArchivedReactorFrontierCohort
from .comparison import FrontierComparison

OUTPUT_RECORDS = (
    FrontierPredictionSeal,
    FrontierMeasuredRoot,
    FrontierDevelopment,
    FrontierQualification,
    FrontierLaws,
    ReactorFiniteControlFrontierProspectivePlan,
    FrontierFrozenRoot,
    FrontierSealedRoot,
    FrontierRevealedRoot,
    ArchivedReactorFrontierCohort,
    FrontierComparison,
    ScientificAdjudicationRecord,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-finite-control-frontier.method",
    kind=CapabilityKind.EVALUATOR,
    config_schema=FrontierPhase.SCHEMA,
    config_schema_sha256=sha256(FrontierPhase.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            {
                r.SCHEMA
                for r in (
                    *OUTPUT_RECORDS,
                    FrontierPhase,
                    FrontierPreparation,
                    FrontierAssay,
                    FrontierNativeRoot,
                    EmpiricalStudySource,
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
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 7200, 8 * 1024**3, 512 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-finite-control-frontier:72-independent-laws-singleton-admission-ranking-prepared-controller-use-root-joins"
    ).hexdigest(),
    runtime_id="cpython-reactor-finite-control-frontier",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "method",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (FrontierPhase,),
    None,
    namespace="reactor-finite-control-frontier",
)
