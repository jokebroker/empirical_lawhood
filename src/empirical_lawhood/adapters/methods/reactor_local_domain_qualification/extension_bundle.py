"""Static local-law qualification binding to the existing scientific owners."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from .config import LocalQualificationDesign
from .records import LocalRootEvidence
from .scoring import LocalCalibration
from .terminal import LocalQualificationResult

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science-local.local-qualification",
    kind=CapabilityKind.EVALUATOR,
    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
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
    config_schema=LocalQualificationDesign.SCHEMA,
    config_schema_sha256=sha256(LocalQualificationDesign.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (
                LocalQualificationDesign.SCHEMA,
                LocalRootEvidence.SCHEMA,
                LocalCalibration.SCHEMA,
                LocalQualificationResult.SCHEMA,
            )
        )
    ),
    output_schema_ids=tuple(
        sorted(
            (
                LocalCalibration.SCHEMA,
                LocalQualificationResult.SCHEMA,
                ScientificAdjudicationRecord.SCHEMA,
            )
        )
    ),
    implementation_sha256=sha256(
        b"reactor-local-qualification:receiver-domain-conditional-root-maxima-existing-law-owner"
    ).hexdigest(),
    runtime_id="cpython-reactor-local",
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 7200, 8 * 1024**3, 128 * 1024**2),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "local-qualification",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (LocalQualificationDesign,),
    None,
    namespace="reactor-local-domain-qualification",
)
