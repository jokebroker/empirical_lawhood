"""Stable public application API for scientific workflow callers."""

from typing import TYPE_CHECKING

from empirical_lawhood.planning.datasets import (
    AcquisitionState,
    CustodyState,
    DatasetBindingRole,
    ExternalIdentifier,
    ExternalIdentifierKind,
)
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.datasets import (
    MAX_DATASET_PAGE_RECORDS,
    DatasetCatalogRecordKind,
)
from .codecs import (
    MAX_AUTHORING_BYTES,
    MAX_CAMPAIGN_PACKAGE_BYTES,
    AuthoringCodecError,
    load_authoring,
    loads_authoring,
)
from .execution import (
    AttemptHistorySummary,
    RunExecutionSummary,
    RunStatusSummary,
)
from .facade import EmpiricalLawhoodApi
from .models import CampaignAuthorityRequired, ExperimentPackageAssemblySummary, AuthorizationSummary, CapabilityListSummary, CapabilityShowSummary, CatalogObjectSummary, CatalogQuerySummary, CatalogRebuildSummary, CatalogSummary, StudyCompilationSummary, ExecutableStudyCompilationSummary, StudyBundleCompilationSummary, CompilationSummary, PreissueReadinessSummary, LinkedCampaignCompilationSummary, DatabaseMutationSummary, DatasetCatalogRecordSummary, DatasetListSummary, DatasetManifestSummary, DatasetOperationPreviewSummary, PublicSourceAcquisitionSummary, PublicSourceCustodySummary, DatasetProjectionRebuildSummary, DatasetRegistrationSummary, DatasetShowSummary, DatasetStoragePreflightSummary, DoctorActionReadiness, DoctorAuthorityBoundary, DoctorBackendSummary, DoctorDependencySummary, DoctorSummary, DocumentSummary, SourceProfileCompilationSummary, ExplorationCompilationSummary, ExplorationPortfolioSummary, HypothesisSummary, NominationSummary, StudyIssueSummary, ExecutableStudyIssueSummary, MultiWorldStudyIssueSummary, ProposalSummary, ScientificInspectionSummary, SystemSummary, WriteEffectSummary
from .results import AdvanceStudyBundleRequest, AssembleExperimentPackageRequest, BindElapsedBudgetRequest, ApiError, ApiResult, CapabilityListRequest, CapabilityShowRequest, CampaignStatusRequest, CampaignMaintenanceStopRequest, CampaignMaintenanceInterruptionRequest, CatalogQueryRequest, CatalogRebuildRequest, CompileCampaignRequest, CompileStudyRequest, CompileCandidateRequest, CompileLinkedCampaignProfileRequest, CompileStudyBundleRequest, CompileSourceProfileRequest, CloseStudyBundleRequest, DatabaseMutationRequest, DatasetListRequest, DatasetManifestRequest, DatasetOperationPreviewRequest, PublicSourceAcquisitionRequest, DatasetProjectionRebuildRequest, DatasetRegistrationRequest, DatasetShowRequest, DocumentRequest, ErrorCategory, IssueStudyRequest, IssueExtensionsRequest, IssueStudyBundleRequest, OperationStatus, CheckReadinessRequest, StudyBundleStatusRequest, ResumeCampaignRequest, RunCampaignRequest

if TYPE_CHECKING:
    from .composition import create_cli_api


_COMPOSITION_EXPORTS = frozenset(
    {
        "create_api",
        "create_cli_api",
        "create_inspection_api",
    }
)


def __getattr__(name: str) -> object:
    """Load the large application composition only when a caller requests it."""

    if name not in _COMPOSITION_EXPORTS:
        raise AttributeError(name)
    from . import composition

    return getattr(composition, name)


__all__ = [
    'AdvanceStudyBundleRequest',
    'AssembleExperimentPackageRequest',
    'BindElapsedBudgetRequest',
    "ApiError",
    "ApiResult",
    "AcquisitionState",
    "AuthoringCodecError",
    "AuthorizationSummary",
    "AttemptHistorySummary",
    'CampaignAuthorityRequired',
    'ExperimentPackageAssemblySummary',
    "CampaignStatusRequest",
    "CampaignMaintenanceStopRequest",
    "CampaignMaintenanceInterruptionRequest",
    "CapabilityListRequest",
    "CapabilityListSummary",
    "CapabilityShowRequest",
    "CapabilityShowSummary",
    "CapabilityKind",
    "CatalogObjectSummary",
    "CatalogQueryRequest",
    "CatalogQuerySummary",
    "CatalogRebuildRequest",
    "CatalogRebuildSummary",
    "CatalogSummary",
    'StudyCompilationSummary',
    'ExecutableStudyCompilationSummary',
    'StudyBundleCompilationSummary',
    "CompilationSummary",
    "PreissueReadinessSummary",
    'LinkedCampaignCompilationSummary',
    "CompileCampaignRequest",
    'CompileStudyRequest',
    'CompileCandidateRequest',
    "CompileLinkedCampaignProfileRequest",
    'CompileStudyBundleRequest',
    "CompileSourceProfileRequest",
    'CloseStudyBundleRequest',
    "DatabaseMutationRequest",
    "DatabaseMutationSummary",
    "DatasetCatalogRecordSummary",
    "DatasetCatalogRecordKind",
    "DatasetBindingRole",
    "DatasetListRequest",
    "DatasetListSummary",
    "DatasetManifestRequest",
    "DatasetManifestSummary",
    "DatasetOperationPreviewRequest",
    "PublicSourceAcquisitionRequest",
    "PublicSourceAcquisitionSummary",
    "PublicSourceCustodySummary",
    "DatasetOperationPreviewSummary",
    "DatasetProjectionRebuildRequest",
    "DatasetProjectionRebuildSummary",
    "DatasetRegistrationRequest",
    "DatasetRegistrationSummary",
    "DatasetShowRequest",
    "DatasetShowSummary",
    "DatasetStoragePreflightSummary",
    "CustodyState",
    "DoctorActionReadiness",
    "DoctorAuthorityBoundary",
    "DoctorBackendSummary",
    "DoctorDependencySummary",
    "DoctorSummary",
    "DocumentRequest",
    "DocumentSummary",
    "ExternalIdentifier",
    "ExternalIdentifierKind",
    "ErrorCategory",
    "ExplorationCompilationSummary",
    "ExplorationPortfolioSummary",
    "HypothesisSummary",
    "EmpiricalLawhoodApi",
    'SourceProfileCompilationSummary',
    'IssueStudyRequest',
    'IssueExtensionsRequest',
    'IssueStudyBundleRequest',
    "MAX_AUTHORING_BYTES",
    "MAX_CAMPAIGN_PACKAGE_BYTES",
    "MAX_DATASET_PAGE_RECORDS",
    "OperationStatus",
    'CheckReadinessRequest',
    'StudyBundleStatusRequest',
    "NominationSummary",
    "ProposalSummary",
    'StudyIssueSummary',
    'ExecutableStudyIssueSummary',
    'MultiWorldStudyIssueSummary',
    "SystemSummary",
    "ScientificInspectionSummary",
    "WriteEffectSummary",
    "create_api",
    "create_cli_api",
    "load_authoring",
    "loads_authoring",
    "ResumeCampaignRequest",
    "RunCampaignRequest",
    "RunExecutionSummary",
    "RunStatusSummary",
]
