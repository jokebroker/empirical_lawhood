'Discover the attached ambient pressure superconductor gauge covariant response method falsifier and adjudication.'

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_contracts import SyntheticMaterialResponseMethodCheck, SyntheticMaterialResponseMethodEvaluationConfig, SyntheticMaterialResponseMethodPanel
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="synthetic-material-response-method.fixture-check",
    kind=CapabilityKind.EVALUATOR,
    config_schema=SyntheticMaterialResponseMethodEvaluationConfig.SCHEMA,
    config_schema_sha256=sha256(
        SyntheticMaterialResponseMethodEvaluationConfig.SCHEMA.encode()
    ).hexdigest(),
    input_schema_ids=(SyntheticMaterialResponseMethodPanel.SCHEMA,),
    output_schema_ids=tuple(
        sorted((SyntheticMaterialResponseMethodCheck.SCHEMA, ScientificAdjudicationRecord.SCHEMA))
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
        b"synthetic-material-response-method:strict-nine-fixture-adjudication"
    ).hexdigest(),
    runtime_id="cpython-3.11-numpy-synthetic-material-response-method-check",
    conformance_check_ids=tuple(
        sorted(
            (
                "nine-fixture-accept-reject-intersection",
                "normal-ward-sign-and-view-falsifiers",
                "method-fixture-only-nonpromotable",
            )
        )
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "fixture-check",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (SyntheticMaterialResponseMethodEvaluationConfig,),
    None,
    namespace="synthetic-material-response-method",
)
