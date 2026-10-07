'Discoverable synthetic gauge covariant response gauge covariant response fixture producer, without material promotion.'

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from .synthetic_material_method_contracts import SyntheticMaterialResponseMethodConfig, SyntheticMaterialResponseMethodPanel

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="synthetic-material-response-method.fixture-producer",
    kind=CapabilityKind.SIMULATOR,
    config_schema=SyntheticMaterialResponseMethodConfig.SCHEMA,
    config_schema_sha256=sha256(SyntheticMaterialResponseMethodConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(SyntheticMaterialResponseMethodConfig.SCHEMA,),
    output_schema_ids=(SyntheticMaterialResponseMethodPanel.SCHEMA,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 600, 128 * 1024, 1024**2),
    implementation_sha256=sha256(
        b"synthetic-material-response-method:one-synthetic-nine-fixture-gauge-covariant-suite"
    ).hexdigest(),
    runtime_id="cpython-3.11-numpy-synthetic-material-response-method",
    conformance_check_ids=tuple(
        sorted(
            (
                "one-synthetic-suite-nine-nested-fixtures",
                "requested-accepted-applied-lattice-probe",
                "signed-current-normal-ward-and-two-mesh-interval",
            )
        )
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "fixture-producer",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (SyntheticMaterialResponseMethodConfig,),
    None,
    namespace="synthetic-material-response-method",
)
