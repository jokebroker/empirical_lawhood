"""Artifact, catalog, execution, and environment implementations."""

from .authority import Ed25519DatasetAuthorizationSigner

from .catalog_mutation import (
    CATALOG_MUTATION_LOCK_FILENAME,
    CatalogMutationLock,
    CatalogMutationLockError,
    CatalogMutationLockUnavailable,
    production_catalog_mutation_lock,
    production_catalog_mutation_lock_path,
)
from .catalog_rebuild import (
    CatalogRebuildError,
    CatalogRebuildResult,
    rebuild_production_catalog,
)
from .dataset_projection import (
    decode_dataset_catalog_snapshot,
    decode_dataset_record_json,
    decode_unified_catalog_snapshot,
    encode_dataset_catalog_snapshot,
    encode_unified_catalog_snapshot,
)
from .dataset_catalog_access import (
    DatasetCatalogReadSessionProvider,
    load_local_dataset_projection_receipt_reference,
)
from .dataset_catalog_authority import (
    DatasetCatalogAuthorityBundle,
    DatasetCatalogAuthorityError,
    ExternalDatasetCatalogAuthorityLoader,
)
from .dataset_operations import (
    CleanRepositoryCommitPreflight,
    DatasetStorageRootRegistry,
    ExternalDatasetOperationAuthorityStore,
)
from .dataset_rebuild import (
    DatasetProjectionRebuildInstaller,
    ExternalDatasetProjectionRebuildAuthorityStore,
    LocalCatalogTargetInspector,
)
from .external_execution import (
    ExternalExecutionBackendKind,
    ExternalExecutionError,
    ExternalWorkerTransport,
    ExternalWorkerUnavailable,
    RegisteredExecutionBackend,
    RegisteredExternalTaskExecutor,
)
from .repository_identity import (
    RepositoryArtifactSpec,
    RepositoryIdentityError,
    load_clean_repository_artifacts,
)
from .recovery import (
    ExternalRunRecoveryStore,
    RunRecoveryError,
    decode_run_recovery_index,
    decode_run_recovery_terminal_event,
    decode_task_recovery_event,
)
from .execution import (
    ExecutionErrorKind,
    ExecutionResourceAdmission,
    ExecutionResourceCapacity,
    LocalExecutionResourceAdmitter,
    ResourceAdmissionError,
)

__all__ = [
    "CATALOG_MUTATION_LOCK_FILENAME",
    "CatalogMutationLock",
    "CatalogMutationLockError",
    "CatalogMutationLockUnavailable",
    "CatalogRebuildError",
    "CatalogRebuildResult",
    "CleanRepositoryCommitPreflight",
    "DatasetCatalogAuthorityBundle",
    "DatasetCatalogAuthorityError",
    "DatasetCatalogReadSessionProvider",
    "load_local_dataset_projection_receipt_reference",
    "DatasetStorageRootRegistry",
    "DatasetProjectionRebuildInstaller",
    "Ed25519DatasetAuthorizationSigner",
    "decode_dataset_catalog_snapshot",
    "decode_dataset_record_json",
    "decode_unified_catalog_snapshot",
    "encode_dataset_catalog_snapshot",
    "encode_unified_catalog_snapshot",
    "ExternalExecutionBackendKind",
    "ExternalExecutionError",
    "ExternalRunRecoveryStore",
    "ExternalDatasetCatalogAuthorityLoader",
    "ExternalDatasetOperationAuthorityStore",
    "ExternalDatasetProjectionRebuildAuthorityStore",
    "ExternalWorkerTransport",
    "ExternalWorkerUnavailable",
    "ExecutionErrorKind",
    "ExecutionResourceAdmission",
    "ExecutionResourceCapacity",
    "LocalExecutionResourceAdmitter",
    "LocalCatalogTargetInspector",
    "ResourceAdmissionError",
    "RegisteredExecutionBackend",
    "RegisteredExternalTaskExecutor",
    "RepositoryArtifactSpec",
    "RepositoryIdentityError",
    "RunRecoveryError",
    "decode_run_recovery_index",
    "decode_run_recovery_terminal_event",
    "decode_task_recovery_event",
    "load_clean_repository_artifacts",
    "production_catalog_mutation_lock",
    "production_catalog_mutation_lock_path",
    "rebuild_production_catalog",
]
