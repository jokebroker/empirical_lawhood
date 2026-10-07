"""Static assay native discovery; construction never advances the medium."""

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeSegmentResult
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import ASSAY_HDF5_SCHEMA

from .discovery import bundle, manifest

SOURCE_RECORDS: tuple[type[CanonicalRecord], ...] = (
    ResponseGeometryAssayNativeConfig,
    FreshSourceQualificationExperiment,
    FreshSourceQualificationSubstrateBinding,
)
ASSAY_SOURCE_CAPABILITY = manifest(
    "source",
    ResponseGeometryAssayNativeConfig,
    CapabilityKind.SIMULATOR,
    (
        ResponseGeometryAssayNativeConfig.SCHEMA,
        ResponseGeometryAssayNativeSegmentResult.SCHEMA,
        ASSAY_HDF5_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (ResponseGeometryAssayNativeSegmentResult.SCHEMA, ASSAY_HDF5_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.EVALUATION_SEALED,
    1800,
)
EXTENSION_BUNDLE_CONTRIBUTION, ASSAY_SOURCE_COMPONENTS = bundle(
    "source", ExtensionContributionKind.SOURCE, ASSAY_SOURCE_CAPABILITY, SOURCE_RECORDS, ASSAY_HDF5_SCHEMA
)
