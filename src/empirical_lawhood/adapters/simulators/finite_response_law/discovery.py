"Finite response-law native discovery metadata; registration alone grants no source readiness."

from dataclasses import replace

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.composition.discovery import manifest, bundle
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA as RETAINED_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from .contracts import FiniteResponseLawNativeConfig
from .native_artifact import NATIVE_PAIR_SCHEMA
from .source_outputs import FiniteResponseLawNativeTaskResult

SOURCE_RECORDS: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawNativeConfig,
    PredecessorBoundSourceQualificationExperiment,
    PredecessorBoundSourceQualificationSubstrateBinding,
)
SOURCE_CAPABILITY = manifest(
    "source",
    FiniteResponseLawNativeConfig,
    CapabilityKind.SIMULATOR,
    (
        FiniteResponseLawNativeConfig.SCHEMA,
        FiniteResponseLawNativeTaskResult.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        RETAINED_PAIR_SCHEMA,
        CanonicalTaskReceipt.SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (FiniteResponseLawNativeTaskResult.SCHEMA, NATIVE_PAIR_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
    namespace=PROGRAMME,
    resources=ResourceBudget(1, 768 * 1024**2, 0, 1800, 48 * 1024**2, 20 * 1024**2),
    implementation_id="explicit-native-clocks-pcg64-retained-q2-continuation",
    conformance_ids=(
        "coupled-views-separate-future-streams",
        "exact-flh-native-census",
        "retained-predecessor-custody-before-effect",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
SOURCE_CAPABILITY = replace(
    SOURCE_CAPABILITY,
    permissions=tuple(
        sorted((*SOURCE_CAPABILITY.permissions, CapabilityPermission.READ_OUTCOME_VISIBLE))
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    SOURCE_CAPABILITY,
    SOURCE_RECORDS,
    NATIVE_PAIR_SCHEMA,
    namespace=PROGRAMME,
    maximum_config_bytes=8 * 1024**2,
)
