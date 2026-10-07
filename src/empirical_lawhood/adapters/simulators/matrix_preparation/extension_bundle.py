"""Static discovery for the bounded native preparation continuation source."""

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.composition.discovery import bundle, manifest
from .contracts import DEVELOPMENT, NATIVE_SCHEMA, PreparationNativeResult, PreparationSourceConfig, PREFIX_BUNDLE_SCHEMA
from .continuation import PreparationDevelopmentContinuation


SOURCE_CAPABILITY = manifest(
    "source",
    PreparationSourceConfig,
    CapabilityKind.SIMULATOR,
    (PreparationSourceConfig.SCHEMA, PREFIX_BUNDLE_SCHEMA),
    (PreparationNativeResult.SCHEMA, NATIVE_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.DEVELOPMENT_VISIBLE,
    3600,
    namespace=DEVELOPMENT,
    resources=ResourceBudget(1, 2 * 1024**3, 0, 3600, 8 * 1024**2, 98 * 1024**2),
    read_permission=CapabilityPermission.READ_DEVELOPMENT,
    implementation_id="matrix-preparation-native-root-bundles-and-custodied-development-completion",
    conformance_ids=(
        "common-preparent-mode",
        "exact-parent-audit-task-view-roster",
        "exposed-root-prefix-custody",
        "independent-audit-task-purposes",
        "no-scientific-retry",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    SOURCE_CAPABILITY,
    (PreparationSourceConfig, PreparationDevelopmentContinuation),
    NATIVE_SCHEMA,
    namespace=DEVELOPMENT,
)
