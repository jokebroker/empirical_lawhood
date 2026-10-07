from dataclasses import replace
from hashlib import sha256
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import (
    ConstructedPreparationStage,
    ConstructedPreparationNativeConfig,
    ConstructedPreparationSource,
)
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import (
    ConstructedPreparationSelection,
    ConstructedPreparationMeasuredRoot,
)
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationLowerSeal
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationParents
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationReport
from empirical_lawhood.adapters.methods.finite_response_law.original_f import OriginalFiniteResponseLaw
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import (
    ConstructedPreparationPrefix,
    ConstructedPreparationPanel,
)

OUTPUT_RECORDS = (
    ConstructedPreparationPrefix,
    ConstructedPreparationParents,
    ConstructedPreparationPanel,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="constructed-preparation-applicability.native",
    kind=CapabilityKind.SIMULATOR,
    config_schema=ConstructedPreparationNativeConfig.SCHEMA,
    config_schema_sha256=sha256(ConstructedPreparationNativeConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(
        sorted(
            {
                r.SCHEMA
                for r in (
                    ConstructedPreparationSource,
                    ConstructedPreparationStage,
                    ConstructedPreparationNativeConfig,
                    ConstructedPreparationSelection,
                    ConstructedPreparationLowerSeal,
                    ConstructedPreparationMeasuredRoot,
                    ConstructedPreparationReport,
                    OriginalFiniteResponseLaw,
                    ConstructedPreparationPrefix,
                    ConstructedPreparationPanel,
                    ConstructedPreparationParents,
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
                CapabilityPermission.READ_FROZEN_MODELS,
            )
        )
    ),
    maximum_evidence_ceiling=EvidenceCeiling.RESPONSE,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(1, 8 * 1024**3, 0, 3600, 4 * 1024**3, 128 * 1024**2),
    implementation_sha256=sha256(
        b"constructed-preparation-applicability.native:prescribed-mean-fresh-native-preparations-qualified-boundary-unchanged-F"
    ).hexdigest(),
    runtime_id="cpython-constructed-preparation-applicability",
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ConstructedPreparationNativeConfig,),
    None,
    namespace="constructed-preparation-applicability",
)
