"""Import-safe discovery of fresh reactor local action assays."""

from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalRootEvidence
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.scoring import LocalCalibration
from .config import LocalNativeConfig

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science-local.native-acquisition",
    kind=CapabilityKind.SIMULATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    permissions=tuple(
        sorted((*FINITE_ACTION_MANIFEST.permissions, CapabilityPermission.READ_FROZEN_MODELS))
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    config_schema=LocalNativeConfig.SCHEMA,
    config_schema_sha256=sha256(LocalNativeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            (LocalNativeConfig.SCHEMA, EmpiricalStudySource.SCHEMA, LocalCalibration.SCHEMA)
        )
    ),
    output_schema_ids=(LocalRootEvidence.SCHEMA,),
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 1800, 256 * 1024**2, 128 * 1024**2),
    implementation_sha256=sha256(
        b"reactor-local-native:causal-anchor-full-tape-nine-word-matched-views"
    ).hexdigest(),
    runtime_id="cpython-reactor-local",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native-acquisition",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (LocalNativeConfig,),
    None,
    namespace="reactor-local-domain-qualification",
)
