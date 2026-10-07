"Import-safe information response prediction native acquisition discovery, without source contact."
from empirical_lawhood.runtime.capabilities import CapabilityPermission


from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.composition.discovery import manifest, bundle
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.methods.information_response.records import InformationResponseCommittedPrediction
from .contracts import InformationResponseNativeConfig

NAMESPACE = "information-response-prediction"
SOURCE_RECORDS = (
    InformationResponseNativeConfig,
    FreshSourceQualificationExperiment,
    FreshSourceQualificationSubstrateBinding,
)
SOURCE_CAPABILITY = manifest(
    "source",
    InformationResponseNativeConfig,
    CapabilityKind.SIMULATOR,
    (
        InformationResponseNativeConfig.SCHEMA,
        PreparedNativeTaskResult.SCHEMA,
        InformationResponseCommittedPrediction.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (
        PreparedNativeTaskResult.SCHEMA,
        InformationResponseCommittedPrediction.SCHEMA,
        NATIVE_PAIR_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
    namespace=NAMESPACE,
    resources=ResourceBudget(1, 768 * 1024**2, 0, 1800, 32 * 1024**2, 20 * 1024**2),
    implementation_id="prepared-mechanics-with-committed-information-handoff-predictions",
    conformance_ids=(
        "exact-information-response-prediction-native-census",
        "forecast-before-future-receipt",
        "paired-independent-future-innovations",
    ),
    read_permission=(CapabilityPermission.READ_OUTCOME_VISIBLE if 'source' == "evaluation" else CapabilityPermission.READ_SEALED_OUTCOMES),
)
EXTENSION_BUNDLE_CONTRIBUTION, SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    SOURCE_CAPABILITY,
    (InformationResponseNativeConfig,),
    NATIVE_PAIR_SCHEMA,
    namespace=NAMESPACE,
)
