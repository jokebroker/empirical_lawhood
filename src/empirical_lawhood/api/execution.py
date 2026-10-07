"""Receipt-first campaign execution composed behind the public facade."""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec

from empirical_lawhood.api.models import RetrospectiveCampaignPackage

import base64
import binascii
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Final, Protocol

from sqlalchemy import Engine
from sqlalchemy.exc import NoResultFound, SQLAlchemyError

from empirical_lawhood.api.models import (
    CampaignMaintenanceInterruptionSummary,
    CampaignMaintenanceStopSummary,
)

from empirical_lawhood.infrastructure.artifacts import (
    ArtifactIdentityConflict,
    ArtifactPlaneError,
    ExternalArtifactPlane,
    ExternalStorageDiagnostic,
)
from empirical_lawhood.infrastructure.bounded_io import (
    BoundedFileIOError,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_CONTROL_PLANE_JSON_BYTES,
    MAX_RUNTIME_PLAN_JSON_BYTES,
    file_matches_bytes,
    read_bounded_bytes,
)
from empirical_lawhood.infrastructure.bounded_process import (
    BoundedProcessError,
    run_bounded_command,
)
from empirical_lawhood.infrastructure.catalog_projection import decode_catalog_snapshot
from empirical_lawhood.infrastructure.maintenance import ReceiptBoundaryStopper
from empirical_lawhood.infrastructure.execution import DurableExecutionEnvelopeCoordinator, DurableExecutionResourceEnvelopeCoordinator, ExecutionResourceAdmission, ExecutionErrorKind, FailureInjector, InjectedSchedulerCrash, LiveLeaseError, LocalExecutionResourceAdmitter, LocalProcessExecutor, LocalScheduler, ResourceAdmissionError, SchedulerConsistencyError, TaskProcessError
from empirical_lawhood.infrastructure.campaign_elapsed_budgets import (
    ExternalCampaignElapsedBudgetStore,
)
from empirical_lawhood.infrastructure.study_issue import PROGRAMME_EXECUTION_GRANTEE_ID, PROGRAMME_REVEAL_GRANTEE_ID, StudyOperationAuthorityStore
from empirical_lawhood.infrastructure.recovery import (
    ExternalRunRecoveryStore,
    RunRecoveryError,
)
from empirical_lawhood.infrastructure.sql import (
    DEFAULT_ATTEMPT_HISTORY_LIMIT,
    DEFAULT_OPERATIONAL_QUERY_WORK_LIMIT,
    MAX_ATTEMPT_HISTORY_LIMIT,
    MAX_OPERATIONAL_QUERY_WORK_LIMIT,
    OperationalAttemptCursor,
    OperationalQueryWorkLimitExceeded,
    OperationalWriteConflict,
    SQLiteOperationalRepository,
    catalog_query_preflight,
    create_read_only_catalog_engine,
)
from empirical_lawhood.infrastructure.task_receipts import (
    ExternalTaskReceiptStore,
    decode_artifact_manifest,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.discovery import SkepticReport
from empirical_lawhood.planning.exploration import ExploratoryFinding, HypothesisSet
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, SourceClosureKind, require_study_authority
from empirical_lawhood.planning.observation_order import ObservationOrderExperimentExtension
from empirical_lawhood.planning.retrospective_prediction import RetrospectivePredictionExperiment, RetrospectivePredictionRole
from empirical_lawhood.runtime.retrospective_prediction import RetrospectivePredictionBinding, derive_retained_prediction_topology
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.planning.native_source import NativeLawQualificationExperiment
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment, PredecessorBoundSourceQualificationExperiment, QualifiedSourceUseExperiment, ProspectiveRetainedSourceUse
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding, PredecessorBoundSourceQualificationSubstrateBinding, QualifiedSourceUseSubstrateBinding, ProspectiveRetainedSourceUseBinding, derive_source_qualification_topology
from empirical_lawhood.runtime.artifacts import (
    ArtifactManifest,
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactSemanticValidationRegistration,
    ArtifactSemanticValidationRegistry,
    ArtifactStreamWriteRequest,
    CanonicalTaskReceipt,
    MAX_ARTIFACT_SEMANTIC_VALIDATION_REGISTRATIONS,
    lineage_parent_sort_key,
    most_restrictive_outcome_access,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec, RosterCapacityDecision
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedReservationPlan, DurableCampaignElapsedBudgetCoordinator
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    AdjudicationReadoutState,
    ScientificAdjudicationContext,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
    decode_scientific_adjudication,
)
from empirical_lawhood.runtime.execution import validate_runner_resource_interface, ExecutionAssuranceProfile, ExecutorEnforcementCapability, ExternalInputResolver, OperationalAttempt, OperationalFailureClass, RunnerRegistry, SchedulerResult, TaskExecutor, TaskBlockReason, VerifiedArtifactInput
from empirical_lawhood.runtime.executable_bindings import CampaignRuntimeProviderResolver, ExecutableBindingRole, ExecutablePlatformPort, GeneratedExecutableBindingAggregate
from empirical_lawhood.runtime.exploration import (
    ExplorationWaveResult,
    validate_exploration_wave_result,
)
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityRegistry
from empirical_lawhood.runtime.input_access import planned_input_access_allowed
from empirical_lawhood.runtime.catalog import (
    CatalogVerificationStatus,
)
from empirical_lawhood.runtime.compiler import (
    compile_exploration_wave_execution_plan,
    compile_run_plan,
    lower_run_plan,
)
from empirical_lawhood.runtime.conditional_children import ConditionalChildInstantiation, FrozenParentInputBinding
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, CandidateExecutionPlan, EnvelopeExecutionPlan, ExecutionPlan, ProtocolExecutionTask, ExecutionTask, ExternalInputSpec, ProtocolRunPlan, ScientificStage
from empirical_lawhood.runtime.response_experiment import derive_response_experiment_topology
from empirical_lawhood.runtime.observation_order import ObservationOrderSubstrateBinding, derive_observation_order_topology
from empirical_lawhood.runtime.response_experiment_ports import NativeInteractionKind, ResponseSubstrateBinding
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CampaignRuntimeProviderRegistry,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    capability_semantic_validation_registry,
)
from empirical_lawhood.runtime.issued_extension_payloads import AuthenticatedExecutableRecord, ExecutableProviderReconstructionReceipt, RetrospectiveProviderReconstruction, ExecutableProviderRecoveryReceipt, IssuedExtensionPayloadResolver, reconstruct_executable_provider, verify_executable_provider_recovery
from empirical_lawhood.runtime.study_issue import StudySourceClosureInspector
from empirical_lawhood.runtime.preexecution_contract import ExperimentContractApplicability, ExperimentContractChronologyEvent, ExperimentContractChronologyKind, ExperimentContractClosureMatrix, ExperimentContractEvidence, ExperimentContractNativeConformance, ExperimentContractNativeDisposition, ExperimentContractPublicationBatch, ExperimentContractStageApplicability, derive_experiment_contract_applicability, validate_experiment_contract_closure
from empirical_lawhood.runtime.retry_amendment import OperationalSourceAmendment, LeaseExpiryRetryAmendment, DiagnosedPureTaskRetryAmendment, ChainedPureTaskRetryAmendment, CompletedRepairRetryAmendment, ElapsedClosureRetryAmendment, ResourcePureTaskRetryAmendment, ChainedResourceRetryAmendment, RetainedCustodyCompletionAmendment
from empirical_lawhood.runtime.recovery import RecoveryTerminalAttempt, RecoveryTerminalDisposition, ProtocolRunRecoveryIndex, CandidateRunRecoveryIndex, EnvelopeRunRecoveryIndex, RunRecoveryIndex, ProtocolRunRecoveryTerminalEvent, TaskRecoveryEvent, build_run_recovery_index
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, OperationalStatus, ScientificStatus
from empirical_lawhood.planning.approval import CompleteApprovalService

from .codecs import (
    MAX_AUTHORING_BYTES,
    MAX_CAMPAIGN_PACKAGE_BYTES,
    AuthoringCodecError,
    loads_campaign_package_authoring,
)
from .models import CampaignPackage, DualLoopPackage, ExplorationExecutionPackage, IssuedCampaignPackage, IssuedCampaignPackageRoot, IssuedStudyPackage, EnvelopeExperimentPackage, ExperimentPackage, ElapsedExperimentPackage, issued_base_candidate, issued_standard_candidate


CampaignExecutionPackage = CampaignPackage | IssuedCampaignPackageRoot
ExecutionPackage = (
    CampaignPackage | IssuedCampaignPackageRoot | DualLoopPackage | ExplorationExecutionPackage
)


class AuthenticatedExecutableRecordResolver(Protocol):
    """Replay owning qualification evidence for already-held ExperimentPackage factory inputs."""

    def resolve(
        self,
        *,
        package: ExperimentPackage,
        required_schema_ids: tuple[str, ...],
    ) -> tuple[AuthenticatedExecutableRecord, ...]: ...


class ExperimentContractEvidenceProvider(Protocol):
    """Resolve exact ephemeral closure inputs without constructing a provider."""

    def resolve(
        self,
        *,
        package: CampaignExecutionPackage,
        run_plan: ProtocolRunPlan,
        execution_plan: ProtocolExecutionPlan,
        authorization_replayed: bool,
    ) -> ExperimentContractEvidence: ...


def _issued_payload_schema_ids(
    package: CampaignExecutionPackage,
) -> tuple[str, ...]:
    if not isinstance(
        package,
        (EnvelopeExperimentPackage, ExperimentPackage),
    ):
        return ()
    return tuple(
        sorted(
            value.payload.object_schema
            for value in package.issued_study.candidate.authoring_package.extension_set.extensions
        )
    )


@dataclass(frozen=True, slots=True)
class IssuedExperimentContractEvidenceProvider:
    "Replay the exact issued measurement through controller-use inputs without constructing a provider."

    payload_resolver: IssuedExtensionPayloadResolver
    executable_aggregate: GeneratedExecutableBindingAggregate

    def resolve(
        self,
        *,
        package: CampaignExecutionPackage,
        run_plan: ProtocolRunPlan,
        execution_plan: ProtocolExecutionPlan,
        authorization_replayed: bool,
    ) -> ExperimentContractEvidence:
        if not authorization_replayed:
            raise PermissionError("contract evidence replay requires prior authority replay")
        if not isinstance(package, ExperimentPackage):
            raise ValueError("typed measurement through controller use contract evidence requires an ExperimentPackage")
        issued_identity = ObjectIdentity.from_record(
            package.issued_study.issued_extensions.issued_extension_set_id,
            package.issued_study.issued_extensions,
        )
        if (
            not isinstance(run_plan, ProtocolRunPlan)
            or run_plan.registry_sha256 != package.registry.fingerprint()
            or execution_plan.registry_sha256 != package.registry.fingerprint()
            or getattr(run_plan, "issued_extension_set", None) != issued_identity
            or getattr(execution_plan, "issued_extension_set", None) != issued_identity
        ):
            raise ValueError("contract evidence package/plan identity differs")
        carrier_schemas = set(_issued_payload_schema_ids(package)).intersection(
            {
                ResponseExperimentExtensionSet.SCHEMA,
                NativeLawQualificationExperiment.SCHEMA,
                ObservationOrderExperimentExtension.SCHEMA,
                FreshSourceQualificationExperiment.SCHEMA,
                PredecessorBoundSourceQualificationExperiment.SCHEMA,
                QualifiedSourceUseExperiment.SCHEMA,
                ProspectiveRetainedSourceUse.SCHEMA,
                RetrospectivePredictionExperiment.SCHEMA,
            }
        )
        if len(carrier_schemas) != 1:
            raise ValueError("contract evidence requires one supported issued carrier")
        carrier_schema = next(iter(carrier_schemas))
        substrate_schema = {
            ObservationOrderExperimentExtension.SCHEMA: ObservationOrderSubstrateBinding.SCHEMA,
            ResponseExperimentExtensionSet.SCHEMA: ResponseSubstrateBinding.SCHEMA,
            NativeLawQualificationExperiment.SCHEMA: ResponseSubstrateBinding.SCHEMA,
            FreshSourceQualificationExperiment.SCHEMA: FreshSourceQualificationSubstrateBinding.SCHEMA,
            PredecessorBoundSourceQualificationExperiment.SCHEMA: PredecessorBoundSourceQualificationSubstrateBinding.SCHEMA,
            QualifiedSourceUseExperiment.SCHEMA: QualifiedSourceUseSubstrateBinding.SCHEMA,
            ProspectiveRetainedSourceUse.SCHEMA: ProspectiveRetainedSourceUseBinding.SCHEMA,
            RetrospectivePredictionExperiment.SCHEMA: RetrospectivePredictionBinding.SCHEMA,
        }[carrier_schema]
        resolved = self.payload_resolver.resolve_declared_records(
            manifest=package.issued_study,
            aggregate=self.executable_aggregate,
            required_schema_ids=tuple(
                sorted(
                    (
                        carrier_schema,
                        substrate_schema,
                    )
                )
            ),
        )
        return self.from_records(
            registry=package.registry,
            records=resolved.records,
            execution_plan=execution_plan,
            historical_experiment=(
                package.experiment
                if isinstance(package.experiment, RetrospectiveExperimentSpec)
                else None
            ),
        )

    @classmethod
    def from_records(
        cls,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        execution_plan: ProtocolExecutionPlan,
        historical_experiment: RetrospectiveExperimentSpec | None = None,
    ) -> ExperimentContractEvidence:
        """Build the shared outcome-blind evidence from already authenticated records."""

        extensions = tuple(
            value
            for value in records
            if isinstance(
                value,
                (
                    ResponseExperimentExtensionSet,
                    ObservationOrderExperimentExtension,
                    FreshSourceQualificationExperiment,
                    RetrospectivePredictionExperiment,
                ),
            )
        )
        if len(extensions) != 1:
            raise ValueError("contract evidence requires one typed experiment carrier")
        extension = extensions[0]
        substrates = tuple(value for value in records if isinstance(value, ResponseSubstrateBinding))
        associations = tuple(
            value for value in records if isinstance(value, ObservationOrderSubstrateBinding)
        )
        qualifications = tuple(
            value for value in records if isinstance(value, FreshSourceQualificationSubstrateBinding)
        )
        historical_bindings = tuple(
            value for value in records if isinstance(value, RetrospectivePredictionBinding)
        )
        if historical_bindings and not isinstance(extension, RetrospectivePredictionExperiment):
            raise ValueError("another carrier received a historical prediction binding")
        if qualifications and not isinstance(extension, FreshSourceQualificationExperiment):
            raise ValueError("another carrier received a qualification substrate association")
        if isinstance(extension, RetrospectivePredictionExperiment):
            if len(historical_bindings) != 1 or substrates or associations or qualifications:
                raise ValueError("historical prediction lacks its sole held-source binding")
            historical = historical_bindings[0]
            projection = next(
                o
                for o in extension.owners
                if o.role is RetrospectivePredictionRole.TRAINING_PROJECTION
            )
            if (
                historical.prediction_carrier
                != ObjectIdentity.from_record(extension.extension_set_id, extension)
                or historical.projection_owner != projection.owner
                or historical.projection_config != projection.config
            ):
                raise ValueError("historical prediction binding changes its carrier or projection")
        elif isinstance(extension, ObservationOrderExperimentExtension):
            if len(associations) != 1 or substrates:
                raise ValueError("observation contract evidence lacks its substrate association")
            if associations[0].observation_carrier != ObjectIdentity.from_record(
                extension.extension_set_id,
                extension,
            ):
                raise ValueError("observation contract substrate selects another carrier")
            substrates = (associations[0].substrate,)
        elif isinstance(extension, FreshSourceQualificationExperiment):
            if len(qualifications) != 1 or substrates or associations:
                raise ValueError(
                    "qualification contract evidence lacks its sole substrate association"
                )
            association = qualifications[0]
            native_manifest = registry.resolve(
                association.substrate.provider_key, association.substrate.provider_version
            )
            if (
                association.qualification_carrier
                != ObjectIdentity.from_record(extension.extension_set_id, extension)
                or association.source_config != extension.source_config
                or ObjectIdentity.from_record(native_manifest.capability_key, native_manifest)
                != extension.source_capability
            ):
                raise ValueError("qualification substrate selects another carrier or source config")
            substrates = (association.substrate,)
        elif associations:
            raise ValueError("Measurement through controller use contract evidence received an observation association")
        requested_receivers = (
            extension.receiver_ids
            if isinstance(
                extension, (ObservationOrderExperimentExtension, FreshSourceQualificationExperiment)
            )
            else None
        )
        requested_clocks = (
            extension.clock_ids
            if isinstance(
                extension, (ObservationOrderExperimentExtension, FreshSourceQualificationExperiment)
            )
            else None
        )
        native = tuple(
            cls._native_conformance(
                registry,
                value,
                requested_receiver_ids=requested_receivers,
                requested_clock_ids=requested_clocks,
            )
            for value in substrates
        )
        tasks = {value.task_id: value for value in execution_plan.tasks}
        publication_batches = tuple(
            ExperimentContractPublicationBatch(
                batch_id=f"batch.{task_id}",
                producer_task_id=task_id,
                member_output_ids=tuple(value.output_id for value in tasks[task_id].outputs),
                lineage_edge_ids=tuple(
                    value.edge_id for value in getattr(tasks[task_id], "scientific_inputs", ())
                ),
                maximum_resident_bytes=(tasks[task_id].capability.requested_resources.output_bytes),
                custody_ordinal=index,
            )
            for index, task_id in enumerate(execution_plan.topological_task_ids(), start=1)
        )
        return ExperimentContractEvidence(
            stage_applicability=(
                ExperimentContractStageApplicability(
                    extension,
                    (
                        derive_retained_prediction_topology(extension)
                        if isinstance(extension, RetrospectivePredictionExperiment)
                        else derive_response_experiment_topology(extension)
                        if isinstance(extension, ResponseExperimentExtensionSet)
                        else (
                            derive_source_qualification_topology(extension)
                            if isinstance(extension, FreshSourceQualificationExperiment)
                            else derive_observation_order_topology(extension)
                        )
                    ),
                ),
            ),
            publication_batches=publication_batches,
            chronology=(
                ExperimentContractChronologyEvent(
                    event_id="event.specification-frozen",
                    ordinal=1,
                    kind=ExperimentContractChronologyKind.SPECIFICATION_FROZEN,
                ),
            ),
            authority_bindings=(),
            native_conformance=native,
            recovery_cuts=(),
            maximum_resident_bytes=max(
                value.capability.requested_resources.output_bytes for value in execution_plan.tasks
            ),
            historical_prediction_bindings=historical_bindings,
            historical_experiments=(
                () if historical_experiment is None else (historical_experiment,)
            ),
        )

    @staticmethod
    def _native_conformance(
        registry: CapabilityRegistry,
        binding: ResponseSubstrateBinding,
        *,
        requested_receiver_ids: tuple[str, ...] | None = None,
        requested_clock_ids: tuple[str, ...] | None = None,
    ) -> ExperimentContractNativeConformance:
        manifest = registry.resolve(binding.provider_key, binding.provider_version)
        action_links: tuple[str, ...] = ()
        if binding.interaction_kind is NativeInteractionKind.INTERACTIVE_EXECUTION:
            assert binding.action_contract is not None
            action_links = tuple(
                sorted(
                    name
                    for name, observed in (
                        ("requested", binding.action_contract.requested_observable),
                        ("accepted", binding.action_contract.accepted_observable),
                        ("applied", binding.action_contract.applied_observable),
                        ("realized", binding.action_contract.realized_observable),
                    )
                    if observed
                )
            )
        return ExperimentContractNativeConformance(
            binding=binding,
            requested_receiver_ids=(
                tuple(value.receiver_id for value in binding.receivers)
                if requested_receiver_ids is None
                else requested_receiver_ids
            ),
            requested_clock_ids=(
                tuple(value.clock_id for value in binding.clocks)
                if requested_clock_ids is None
                else requested_clock_ids
            ),
            preserved_dispositions=tuple(ExperimentContractNativeDisposition),
            preserved_action_link_ids=action_links,
            conformance_check_ids=manifest.conformance_check_ids,
        )


MAX_CATALOG_REBUILD_SEMANTIC_RUNS: Final[int] = 1_024
MAX_CATALOG_REBUILD_SEMANTIC_SOURCES: Final[int] = 2_048
MAX_CATALOG_REBUILD_SEMANTIC_SOURCE_BYTES: Final[int] = 64 * 1024 * 1024
# Projection semantic replay authenticates a composed package and its complete
# typed DAG in addition to compact sources. The measured 64-root DAG is
# 38,642,100 bytes, so that pair cannot share the compact 64 MiB budget. Keep
# independent per-record caps and a finite aggregate; unrelated replay retains
# MAX_CATALOG_REBUILD_SEMANTIC_SOURCE_BYTES.
MAX_CATALOG_REBUILD_PLAN_PACKAGE_SOURCE_BYTES: Final[int] = (
    MAX_CATALOG_REBUILD_SEMANTIC_SOURCE_BYTES
    + MAX_CAMPAIGN_PACKAGE_BYTES
    + MAX_RUNTIME_PLAN_JSON_BYTES
    + 3 * MAX_ARTIFACT_MANIFEST_BYTES
)
MAX_CATALOG_REBUILD_SEMANTIC_LOGICAL_ID_BYTES: Final[int] = 2_048
MAX_CATALOG_REBUILD_SEMANTIC_LOGICAL_ID_SEGMENTS: Final[int] = 64
MAX_CATALOG_REBUILD_RUN_CANDIDATES_PER_LOGICAL_ID: Final[int] = 32


class CampaignExecutionError(RuntimeError):
    kind = ExecutionErrorKind.CUSTODY_STORAGE
    reason_code = "EXECUTION_CUSTODY_OR_STORAGE_FAILED"

    def __init__(self, message: str = "execution boundary failed") -> None:
        super().__init__(message)


class CampaignIdentityError(CampaignExecutionError):
    kind = ExecutionErrorKind.IDENTITY
    reason_code = "EXECUTION_IDENTITY_REFUSED"


class CampaignCapabilityError(CampaignExecutionError):
    kind = ExecutionErrorKind.CAPABILITY
    reason_code = "CAPABILITY_NOT_REGISTERED"


class CampaignExecutableInputError(CampaignCapabilityError):
    reason_code = "AUTHENTICATED_EXECUTABLE_INPUTS_REQUIRED"


class CampaignConflictError(CampaignExecutionError):
    kind = ExecutionErrorKind.IMMUTABLE_CONFLICT
    reason_code = "IMMUTABLE_ARTIFACT_CONFLICT"


class CampaignConcurrencyError(CampaignExecutionError):
    kind = ExecutionErrorKind.LIVE_CONCURRENCY
    reason_code = "LIVE_EXECUTION_CONFLICT"


class CampaignValidationError(CampaignExecutionError):
    kind = ExecutionErrorKind.VALIDATION
    reason_code = "EXECUTION_CONTRACT_INVALID"


class CampaignRevealAuthorityError(CampaignValidationError):
    reason_code = "OUTCOME_REVEAL_AUTHORITY_REQUIRED"


class CampaignTaskError(CampaignExecutionError):
    kind = ExecutionErrorKind.TASK_FAILURE
    reason_code = "TASK_EXECUTION_FAILED"


class CampaignPrerequisiteError(CampaignExecutionError):
    kind = ExecutionErrorKind.PREREQUISITE
    reason_code = "EXECUTION_RESOURCE_UNAVAILABLE"


class CampaignOperationalQueryWorkLimitError(CampaignExecutionError):
    kind = ExecutionErrorKind.CAPABILITY
    reason_code = "OPERATIONAL_QUERY_WORK_LIMIT_EXCEEDED"


@dataclass(slots=True)
class _CatalogSemanticReplayBudget:
    remaining_bytes: int = MAX_CATALOG_REBUILD_SEMANTIC_SOURCE_BYTES
    source_count: int = 0

    def begin_source(self) -> None:
        self.source_count += 1
        if self.source_count > MAX_CATALOG_REBUILD_SEMANTIC_SOURCES:
            raise CampaignCapabilityError(
                "catalog semantic registry exceeds its aggregate source limit"
            )

    def charge_bytes(self, size_bytes: int) -> None:
        if size_bytes < 0 or size_bytes > self.remaining_bytes:
            raise CampaignCapabilityError(
                "catalog semantic registry exceeds its aggregate byte limit"
            )
        self.remaining_bytes -= size_bytes

    def read_limit(self, maximum_bytes: int) -> int:
        if maximum_bytes < 0:
            raise CampaignCapabilityError(
                "catalog semantic registry source has an invalid byte limit"
            )
        return min(maximum_bytes, self.remaining_bytes)


def _logical_semantic_run_candidates(logical_artifact_id: str) -> tuple[str, ...]:
    """Return bounded run candidates from compiler-reserved output identities."""

    dot_count = logical_artifact_id.count(".")
    if not (
        (logical_artifact_id.startswith("artifact.") and dot_count >= 3)
        or (logical_artifact_id.startswith("finding.") and dot_count >= 2)
        or (
            logical_artifact_id.startswith(("hypothesis-set.", "skeptic-report.", "wave-result."))
            and dot_count >= 1
        )
    ):
        return ()
    # Stable IDs are ASCII-only, so character count is also the encoded byte
    # count and does not require allocating a second hostile input-sized value.
    if len(logical_artifact_id) > MAX_CATALOG_REBUILD_SEMANTIC_LOGICAL_ID_BYTES:
        raise CampaignCapabilityError("catalog semantic logical ID exceeds its byte limit")
    segment_count = dot_count + 1
    if segment_count > MAX_CATALOG_REBUILD_SEMANTIC_LOGICAL_ID_SEGMENTS:
        raise CampaignCapabilityError("catalog semantic logical ID exceeds its segment limit")
    parts = logical_artifact_id.split(".")
    if any(not part for part in parts):
        raise CampaignIdentityError("catalog semantic logical ID has an empty segment")
    candidate_count = 0
    if parts[0] == "artifact" and len(parts) >= 4:
        # Campaign outputs are artifact.<run>.<step>.<output>. Run, step and
        # output IDs may themselves contain dots, so authenticate each possible
        # run prefix against its persisted plan rather than choosing by syntax.
        candidate_count = len(parts) - 3
    elif parts[0] == "finding" and len(parts) >= 3:
        # Exploration findings are finding.<run>.<analysis> with the same
        # possible dot ambiguity between run and analysis identities.
        candidate_count = len(parts) - 2
    elif parts[0] in {"hypothesis-set", "skeptic-report", "wave-result"} and len(parts) >= 2:
        candidate_count = 1
    if candidate_count > MAX_CATALOG_REBUILD_RUN_CANDIDATES_PER_LOGICAL_ID:
        raise CampaignCapabilityError("catalog semantic logical ID exceeds its run-candidate limit")
    candidates: set[str] = set()
    if parts[0] == "artifact":
        candidates.update(".".join(parts[1:index]) for index in range(2, len(parts) - 1))
    elif parts[0] == "finding":
        candidates.update(".".join(parts[1:index]) for index in range(2, len(parts)))
    else:
        candidates.add(".".join(parts[1:]))
    return tuple(
        sorted(
            validate_stable_id(candidate, field_name="catalog output run_id")
            for candidate in candidates
        )
    )


def _path_semantic_run_id(relative_path: str) -> str | None:
    """Retain the exact historical output-path claim as an additional binding."""

    parts = PurePosixPath(relative_path).parts
    if len(parts) < 4 or parts[0] != "runs" or parts[2] not in {"exploration", "outputs"}:
        return None
    return validate_stable_id(parts[1], field_name="catalog output run_id")


class CampaignInternalError(CampaignExecutionError):
    kind = ExecutionErrorKind.INTERNAL
    reason_code = "INTERNAL_EXECUTION_ERROR"

    def __init__(self) -> None:
        super().__init__("internal execution failure")


class ReferenceFixtureReplayDisabled(CampaignExecutionError):
    kind = ExecutionErrorKind.CAPABILITY
    reason_code = "REFERENCE_FIXTURE_REPLAY_DISABLED"


@dataclass(frozen=True, slots=True)
class RunExecutionSummary:
    run_id: str
    operational_status: str
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    adjudication_state: AdjudicationReadoutState
    adjudication_evaluability: AdjudicationEvaluability
    scientific_status: ScientificStatus | None
    admission_status: AdmissionStatus | None
    scientific_reason_codes: tuple[str, ...]
    adjudication_id: str | None = None
    adjudication_fingerprint: str | None = None
    adjudication_artifact_id: str | None = None
    adjudication_materialization_id: str | None = None
    adjudication_receipt_id: str | None = None
    evidence_world_id: str | None = None
    fixture_scope_id: str | None = None
    plumbing_only: bool = False
    provider_reconstruction_receipt_id: str | None = None
    provider_reconstruction_receipt_fingerprint: str | None = None
    provider_recovery_receipt_id: str | None = None
    provider_recovery_receipt_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class _ScientificAdjudicationBinding:
    task: ProtocolExecutionTask
    output_logical_artifact_id: str
    contract: ScientificAdjudicationOutputContract
    context: ScientificAdjudicationContext


@dataclass(frozen=True, slots=True)
class _ResolvedCampaignProvider:
    """One provider instance and any exact, non-authorizing reconstruction proof."""

    provider: CampaignRuntimeProvider
    reconstruction: ExecutableProviderReconstructionReceipt | None = None
    recovery: ExecutableProviderRecoveryReceipt | None = None


@dataclass(frozen=True, slots=True)
class AttemptHistorySummary:
    attempt_id: str
    task_id: str
    ordinal: int
    disposition: str
    reason_code: str | None
    block_kind: str | None
    implementation_commit: str | None = None
    execution_plan_id: str | None = None
    execution_plan_schema: str | None = None
    execution_plan_sha256: str | None = None
    failure_class: str | None = None
    retryable: bool | None = None
    sanitized_exception_type: str | None = None
    sanitized_message: str | None = None
    diagnostic_sha256: str | None = None
    recovery_event_id: str | None = None
    recovery_event_schema: str | None = None
    recovery_event_relative_path: str | None = None
    receipt_id: str | None = None
    receipt_relative_path: str | None = None
    receipt_sha256: str | None = None
    artifact_materialization_ids: tuple[str, ...] = ()
    adjudication_materialization_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunStatusSummary:
    run_id: str
    operational_status: str
    succeeded_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    blocked_task_ids: tuple[str, ...]
    running_task_ids: tuple[str, ...]
    attempt_history: tuple[AttemptHistorySummary, ...] = ()
    attempt_history_limit: int | None = None
    attempt_history_has_more: bool = False
    attempt_history_next_cursor: str | None = None
    operational_query_work_limit: int = DEFAULT_OPERATIONAL_QUERY_WORK_LIMIT
    implementation_commit: str | None = None
    execution_plan_id: str | None = None
    execution_plan_schema: str | None = None
    execution_plan_sha256: str | None = None
    recovery_index_id: str | None = None
    recovery_index_relative_path: str | None = None
    recovery_terminal_event_relative_path: str | None = None
    artifact_receipt_relative_paths: tuple[str, ...] = ()
    adjudication_receipt_relative_paths: tuple[str, ...] = ()
    status_source: str = "OPERATIONAL_CATALOG"


ATTEMPT_HISTORY_CURSOR_SCHEMA = 'empirical-lawhood/api/attempt-history-cursor'
MAX_ATTEMPT_HISTORY_CURSOR_BYTES = 2_048


def _strict_cursor_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("attempt history cursor contains a duplicate field")
        value[key] = item
    return value


def _attempt_history_run_sha256(run_id: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "run_id": run_id,
                "schema": ATTEMPT_HISTORY_CURSOR_SCHEMA,
            }
        )
    ).hexdigest()


def _encode_attempt_history_cursor(run_id: str, cursor: OperationalAttemptCursor) -> str:
    payload = canonical_json_bytes(
        {
            "after_attempt_id": cursor.attempt_id,
            "after_attempt_pk": cursor.attempt_pk,
            "after_task_id": cursor.task_id,
            "run_sha256": _attempt_history_run_sha256(run_id),
            "schema": ATTEMPT_HISTORY_CURSOR_SCHEMA,
            "snapshot_max_pk": cursor.snapshot_max_pk,
            "version": "1.0.0",
        }
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_attempt_history_cursor(
    run_id: str,
    cursor: str,
) -> OperationalAttemptCursor:
    if not cursor or len(cursor.encode("utf-8")) > MAX_ATTEMPT_HISTORY_CURSOR_BYTES:
        raise ValueError("attempt history cursor is empty or exceeds its byte limit")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(decoded, object_pairs_hook=_strict_cursor_object)
    except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as error:
        raise ValueError("attempt history cursor is invalid") from error
    expected_fields = {
        "after_attempt_id",
        "after_attempt_pk",
        "after_task_id",
        "run_sha256",
        "schema",
        "snapshot_max_pk",
        "version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("attempt history cursor shape is invalid")
    if value["schema"] != ATTEMPT_HISTORY_CURSOR_SCHEMA or value["version"] != "1.0.0":
        raise ValueError("attempt history cursor schema is unsupported")
    if value["run_sha256"] != _attempt_history_run_sha256(run_id):
        raise ValueError("attempt history cursor belongs to another run")
    attempt_id = value["after_attempt_id"]
    attempt_pk = value["after_attempt_pk"]
    snapshot_max_pk = value["snapshot_max_pk"]
    task_id = value["after_task_id"]
    if (
        not isinstance(attempt_id, str)
        or type(attempt_pk) is not int
        or type(snapshot_max_pk) is not int
        or not isinstance(task_id, str)
    ):
        raise ValueError("attempt history cursor fields have invalid types")
    return OperationalAttemptCursor(
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_pk=attempt_pk,
        snapshot_max_pk=snapshot_max_pk,
    )


class _StaticInputResolver(ExternalInputResolver):
    def __init__(self, values: tuple[VerifiedArtifactInput, ...]) -> None:
        self._values = {value.artifact_id: value for value in values}

    def resolve(
        self,
        logical_artifact_ids: tuple[str, ...],
    ) -> tuple[VerifiedArtifactInput, ...]:
        try:
            return tuple(self._values[artifact_id] for artifact_id in logical_artifact_ids)
        except KeyError as error:
            raise CampaignCapabilityError("runtime provider omitted a frozen input") from error


class CampaignExecutionService:
    """Execute/resume through guarded artifacts, receipts and operational SQLite."""

    def __init__(
        self,
        *,
        repo_root: Path,
        artifact_plane: ExternalArtifactPlane,
        engine_factory: Callable[[], Engine],
        read_only_engine_factory: Callable[[], Engine] | None = None,
        catalog_path: Path,
        providers: CampaignRuntimeProviderRegistry,
        provider_resolver: CampaignRuntimeProviderResolver | None = None,
        executable_binding_aggregate: GeneratedExecutableBindingAggregate | None = None,
        issued_extension_payload_resolver: IssuedExtensionPayloadResolver | None = None,
        authenticated_executable_record_resolver: (
            AuthenticatedExecutableRecordResolver | None
        ) = None,
        experiment_contract_evidence_provider: (ExperimentContractEvidenceProvider | None) = None,
        executable_platform_ports: tuple[ExecutablePlatformPort, ...] = (),
        approval_service: CompleteApprovalService | None = None,
        study_authority_store: StudyOperationAuthorityStore | None = None,
        study_execution_grantee_id: str = PROGRAMME_EXECUTION_GRANTEE_ID,
        study_reveal_grantee_id: str = PROGRAMME_REVEAL_GRANTEE_ID,
        executor: TaskExecutor | None = None,
        failure_injector: FailureInjector | None = None,
        resource_admitter: LocalExecutionResourceAdmitter | None = None,
        enforce_git_identity: bool = True,
        operational_query_work_limit: int = DEFAULT_OPERATIONAL_QUERY_WORK_LIMIT,
        catalog_semantic_validation_registry: ArtifactSemanticValidationRegistry | None = None,
        execution_assurance_profile: ExecutionAssuranceProfile = (
            ExecutionAssuranceProfile.TRUSTED_LOCAL
        ),
        maximum_parallel_tasks: int = 1,
        study_source_closure_inspector: StudySourceClosureInspector | None = None,
        execution_envelope_coordinator: DurableExecutionEnvelopeCoordinator | None = None,
        execution_resource_envelope_coordinator: (
            DurableExecutionResourceEnvelopeCoordinator | None
        ) = None,
        roster_capacity_decisions: tuple[RosterCapacityDecision, ...] = (),
    ) -> None:
        if (
            operational_query_work_limit <= 0
            or operational_query_work_limit > MAX_OPERATIONAL_QUERY_WORK_LIMIT
        ):
            raise ValueError("operational query work limit is outside its supported range")
        if maximum_parallel_tasks <= 0:
            raise ValueError("maximum parallel tasks must be positive")
        self.repo_root = repo_root.resolve(strict=True)
        self.artifact_plane = artifact_plane
        self._maintenance_stopper = ReceiptBoundaryStopper(artifact_plane)
        self.engine_factory = engine_factory
        self.catalog_path = catalog_path
        self.read_only_engine_factory = read_only_engine_factory or (
            lambda: create_read_only_catalog_engine(self.catalog_path)
        )
        self.providers = providers
        port_keys = tuple(value.port_key for value in executable_platform_ports)
        if tuple(sorted(set(port_keys))) != port_keys:
            raise ValueError("executable platform ports must be sorted and unique")
        self.provider_resolver = provider_resolver
        self.executable_binding_aggregate = executable_binding_aggregate
        self.issued_extension_payload_resolver = issued_extension_payload_resolver
        self.authenticated_executable_record_resolver = authenticated_executable_record_resolver
        self.experiment_contract_evidence_provider = experiment_contract_evidence_provider
        self.executable_platform_ports = executable_platform_ports
        self.approval_service = approval_service
        validate_stable_id(
            study_execution_grantee_id,
            field_name='study_execution_grantee_id',
        )
        validate_stable_id(
            study_reveal_grantee_id,
            field_name='study_reveal_grantee_id',
        )
        self.study_authority_store = study_authority_store
        self.study_execution_grantee_id = study_execution_grantee_id
        self.study_reveal_grantee_id = study_reveal_grantee_id
        self.executor = executor or LocalProcessExecutor(
            scratch_root=artifact_plane.root,
        )
        self.failure_injector = failure_injector
        self.resource_admitter = resource_admitter or LocalExecutionResourceAdmitter()
        self.enforce_git_identity = enforce_git_identity
        self.operational_query_work_limit = operational_query_work_limit
        self.execution_assurance_profile = execution_assurance_profile
        self.maximum_parallel_tasks = maximum_parallel_tasks
        self.study_source_closure_inspector = study_source_closure_inspector
        self.execution_envelope_coordinator = execution_envelope_coordinator
        self.execution_resource_envelope_coordinator = execution_resource_envelope_coordinator
        self.roster_capacity_decisions = roster_capacity_decisions
        # Catalog rebuild has no execution plan from which to derive capability
        # semantics.  Resolve it against a separately configured immutable
        # registry instead of trusting the declarations in persisted manifests.
        self.catalog_semantic_validation_registry = (
            catalog_semantic_validation_registry or artifact_plane.semantic_validations
        )

    @property
    def catalog_present(self) -> bool:
        return self.catalog_path.is_file()

    @property
    def capability_count(self) -> int:
        return self.providers.capability_count

    @property
    def provider_registry_fingerprint(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.providers.fingerprints)).hexdigest()

    def preexecution_contract_closure(
        self,
        *,
        package: CampaignExecutionPackage,
        run_plan: ProtocolRunPlan,
        execution_plan: ProtocolExecutionPlan,
    ) -> ExperimentContractClosureMatrix:
        """Replay authorized immutable inputs, then run the shared pure closure."""

        applicability = derive_experiment_contract_applicability(
            _issued_payload_schema_ids(package)
        )
        evidence = None
        if (
            applicability is ExperimentContractApplicability.APPLICABLE
            and self.experiment_contract_evidence_provider is not None
        ):
            self._validate_campaign_authorization(package)
            if not isinstance(package, ExperimentPackage):
                raise CampaignValidationError(
                    "applicable contract evidence requires a current issued package"
                )
            self._validate_issued_execution_authority(package, at_utc=None)
            try:
                evidence = self.experiment_contract_evidence_provider.resolve(
                    package=package,
                    run_plan=run_plan,
                    execution_plan=execution_plan,
                    authorization_replayed=True,
                )
            except (PermissionError, ValueError) as error:
                raise CampaignValidationError(
                    "authenticated experiment contract evidence replay failed"
                ) from error
        self._validate_output_profiles(execution_plan)
        return validate_experiment_contract_closure(
            run_plan=run_plan,
            execution_plan=execution_plan,
            capability_registry=package.registry,
            contract_evidence=evidence,
            applicability=applicability,
        )

    def preissue_readiness(
        self,
        *,
        registry: CapabilityRegistry,
        run_plan: ProtocolRunPlan,
        execution_plan: ProtocolExecutionPlan,
        decoded_records: tuple[CanonicalRecord, ...],
        resource_envelope: ExecutionResourceEnvelopeSpec,
        historical_experiment: RetrospectiveExperimentSpec | None = None,
    ) -> tuple[
        ExperimentContractClosureMatrix,
        tuple[str, ...],
        tuple[str, ...],
        ScientificAdjudicationOutputContract,
    ]:
        """Resolve the complete future runtime contract without effects or authority."""

        if self.provider_resolver is None:
            raise CampaignCapabilityError("runtime provider resolver is not composed")
        provider = self.provider_resolver.resolve(
            registry, decoded_records=decoded_records, platform_ports=self.executable_platform_ports
        )
        runners = provider.runners(registry)
        runner_registry = RunnerRegistry(runners, registry_sha256=registry.fingerprint())
        for task in execution_plan.tasks:
            runner = runner_registry.resolve(task)
            validate_runner_resource_interface(
                runner, resource_envelope.cell_for_task(task.task_id)
            )
        inputs = provider.external_inputs(execution_plan)
        try:
            expected_inputs = tuple(
                sorted(
                    {
                        value.logical_artifact_id
                        for task in execution_plan.tasks
                        for value in task.external_inputs
                    }
                )
            )
            input_ids = tuple(value.logical_artifact_id for value in inputs)
            if input_ids != expected_inputs:
                raise CampaignCapabilityError("runtime provider input set differs from plan")
        finally:
            for value in inputs:
                value.close()
        contracts = provider.output_semantic_contracts(registry, execution_plan)
        expected_keys = tuple(
            sorted(
                {
                    (
                        task.capability.capability_key,
                        task.capability.capability_version,
                        output.payload_schema,
                        output.profile,
                    )
                    for task in execution_plan.tasks
                    for output in task.outputs
                },
                key=str,
            )
        )
        if tuple(value.key for value in contracts) != expected_keys:
            raise CampaignCapabilityError("runtime provider semantic contracts differ from plan")
        self._semantic_artifact_plane(execution_plan, contracts)
        adjudication = provider.scientific_adjudication_contract(registry, execution_plan)
        matches = tuple(
            (task, output)
            for task in execution_plan.tasks
            for output in task.outputs
            if adjudication is not None
            and task.capability.capability_key == adjudication.capability_key
            and task.capability.capability_version == adjudication.capability_version
            and output.output_id == adjudication.output_id
        )
        if (
            adjudication is None
            or len(matches) != 1
            or matches[0][1].payload_schema != adjudication.payload_schema
        ):
            raise CampaignCapabilityError("scientific adjudication contract differs from plan")
        applicability = derive_experiment_contract_applicability(
            tuple(value.SCHEMA for value in decoded_records)
        )
        evidence = (
            IssuedExperimentContractEvidenceProvider.from_records(
                registry=registry,
                records=decoded_records,
                execution_plan=execution_plan,
                historical_experiment=historical_experiment,
            )
            if applicability is ExperimentContractApplicability.APPLICABLE
            else None
        )
        closure = validate_experiment_contract_closure(
            run_plan=run_plan,
            execution_plan=execution_plan,
            capability_registry=registry,
            contract_evidence=evidence,
            applicability=applicability,
        )
        return (
            closure,
            tuple(
                f"{value.manifest.capability_key}@{value.manifest.capability_version}"
                for value in runners
            ),
            input_ids,
            adjudication,
        )

    def _selected_runtime_binding_ids(
        self,
        registry: CapabilityRegistry,
    ) -> tuple[str, ...]:
        aggregate = self.executable_binding_aggregate
        if aggregate is None:
            raise CampaignCapabilityError("issued executable binding aggregate is not composed")
        selected = []
        for manifest in registry.capabilities:
            binding = aggregate.binding(
                manifest.capability_key,
                manifest.capability_version,
            )
            if binding is None:
                raise CampaignCapabilityError(
                    f"capability lacks an executable runtime binding: {manifest.registry_id}"
                )
            if (
                not binding.capability_backed
                or binding.role is not ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER
                or binding.capability_implementation_sha256 != manifest.implementation_sha256
                or binding.input_schema_ids != manifest.input_schema_ids
                or binding.output_schema_ids != manifest.output_schema_ids
            ):
                raise CampaignCapabilityError(
                    f"executable runtime binding differs from capability: {manifest.registry_id}"
                )
            selected.append(binding.binding_id)
        binding_ids = tuple(sorted(selected))
        if len(set(binding_ids)) != len(registry.capabilities):
            raise CampaignCapabilityError(
                "executable runtime bindings do not exactly cover the capability registry"
            )
        return binding_ids

    def _load_provider_reconstruction_receipt(
        self,
        run_id: str,
        *,
        budget: _CatalogSemanticReplayBudget | None = None,
    ) -> ExecutableProviderReconstructionReceipt | None:
        relative_path = f"runs/{run_id}/plans/provider-reconstruction.json"
        if not self._rebuild_control_pair_present(relative_path):
            return None
        payload, payload_sha256, payload_schema = self._verified_rebuild_control_payload(
            logical_artifact_id=f"provider-reconstruction.{run_id}",
            relative_path=relative_path,
            payload_schemas=tuple(
                sorted(
                    (
                        ExecutableProviderReconstructionReceipt.SCHEMA,
                        RetrospectiveProviderReconstruction.SCHEMA,
                    )
                )
            ),
            maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
            budget=budget or _CatalogSemanticReplayBudget(),
        )
        receipt = decode_canonical_bytes(
            payload,
            RetrospectiveProviderReconstruction
            if payload_schema == RetrospectiveProviderReconstruction.SCHEMA
            else ExecutableProviderReconstructionReceipt,
            maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
        )
        if (
            payload_schema != receipt.SCHEMA
            or payload_sha256 != receipt.fingerprint()
            or receipt.canonical_bytes() != payload
        ):
            raise CampaignIdentityError(
                "persisted provider reconstruction receipt differs from its identity"
            )
        return receipt

    def _resolve_campaign_provider(
        self,
        package: CampaignExecutionPackage,
        *,
        authorization_replayed: bool,
        require_persisted_reconstruction: bool = False,
        budget: _CatalogSemanticReplayBudget | None = None,
    ) -> _ResolvedCampaignProvider:
        """Resolve a precomposed package or reconstruct its issued executable provider.

        The explicit boolean is accepted only from call sites that have just
        replayed both durable campaign approval and programme execution
        authority.  It is checked before any issued payload source is read.
        """

        registry_sha256 = package.registry.fingerprint()
        try:
            provider = self.providers.resolve(registry_sha256)
        except KeyError:
            provider = None
        if provider is not None:
            return _ResolvedCampaignProvider(provider=provider)

        if not isinstance(package, ExperimentPackage):
            raise CampaignCapabilityError(
                "no runtime provider has an exact precomposed package binding"
            )
        if not authorization_replayed:
            raise PermissionError("issued provider reconstruction requires prior authority replay")
        aggregate = self.executable_binding_aggregate
        payload_resolver = self.issued_extension_payload_resolver
        provider_resolver = self.provider_resolver
        if aggregate is None or payload_resolver is None or provider_resolver is None:
            raise CampaignCapabilityError(
                "issued executable provider reconstruction is not fully composed"
            )
        selected_binding_ids = self._selected_runtime_binding_ids(package.registry)
        selected_bindings = tuple(
            next(value for value in aggregate.bindings if value.binding_id == binding_id)
            for binding_id in selected_binding_ids
        )
        required_authenticated_schemas = tuple(
            sorted(
                {
                    schema
                    for binding in selected_bindings
                    for schema in binding.required_authenticated_record_schemas
                }
            )
        )
        authenticated_records: tuple[AuthenticatedExecutableRecord, ...] = ()
        if required_authenticated_schemas:
            authenticated_resolver = self.authenticated_executable_record_resolver
            if authenticated_resolver is None:
                raise CampaignExecutableInputError(
                    "ExperimentPackage factory inputs require owning qualification replay"
                )
            try:
                authenticated_records = authenticated_resolver.resolve(
                    package=package,
                    required_schema_ids=required_authenticated_schemas,
                )
            except CampaignExecutionError:
                raise
            except Exception as error:
                raise CampaignExecutableInputError(
                    "ExperimentPackage factory input qualification replay failed"
                ) from error
            observed_schemas = tuple(value.record.SCHEMA for value in authenticated_records)
            if (
                observed_schemas != required_authenticated_schemas
                or tuple(sorted(set(observed_schemas))) != observed_schemas
            ):
                raise CampaignExecutableInputError(
                    "qualified ExperimentPackage input roster differs from its bindings"
                )
        try:
            reconstructed = reconstruct_executable_provider(
                registry=package.registry,
                manifest=package.issued_study,
                aggregate=aggregate,
                selected_binding_ids=selected_binding_ids,
                payload_resolver=payload_resolver,
                provider_resolver=provider_resolver,
                authenticated_records=authenticated_records,
                platform_ports=self.executable_platform_ports,
                authorization_replayed=True,
            )
        except PermissionError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignCapabilityError(
                "issued executable provider reconstruction failed closed"
            ) from error

        stored = self._load_provider_reconstruction_receipt(
            package.run_plan_id,
            budget=budget,
        )
        if require_persisted_reconstruction and stored is None:
            raise CampaignIdentityError(
                "persisted ExperimentPackage execution lacks its provider reconstruction receipt"
            )
        recovery = None
        if stored is not None:
            try:
                recovery = verify_executable_provider_recovery(
                    stored,
                    reconstructed.receipt,
                )
            except ValueError as error:
                raise CampaignCapabilityError(
                    "fresh provider reconstruction differs from persisted state"
                ) from error
        return _ResolvedCampaignProvider(
            provider=reconstructed.provider,
            reconstruction=reconstructed.receipt,
            recovery=recovery,
        )

    @property
    def executor_enforces_resources(self) -> bool:
        capability = getattr(self.executor, "enforcement_capability", None)
        return (
            isinstance(capability, ExecutorEnforcementCapability) and capability.enforces_resources
        )

    @property
    def supports_exploration_no_network(self) -> bool:
        capability = getattr(self.executor, "enforcement_capability", None)
        return (
            isinstance(capability, ExecutorEnforcementCapability)
            and capability.enforces_resources
            and capability.network_isolation
        )

    def write_effects(self, run_id: str) -> tuple[str, ...]:
        return (
            f"external:runs/{run_id}/inputs/*",
            f"external:runs/{run_id}/outputs/*",
            f"external:runs/{run_id}/plans/*",
            f"external:runs/{run_id}/receipts/*",
            "local:.empirical-lawhood/experiment_catalog.sqlite3",
        )

    def storage_diagnostic(
        self,
        *,
        operation_minimum_free_bytes: int = 0,
    ) -> ExternalStorageDiagnostic:
        return self.artifact_plane.root.diagnostic(
            operation_minimum_free_bytes=operation_minimum_free_bytes
        )

    def resource_admission(self, execution_plan: ProtocolExecutionPlan) -> ExecutionResourceAdmission:
        """Assess one plan without mutating storage or the operational catalog."""

        storage = self.artifact_plane.root.diagnostic(
            operation_minimum_free_bytes=execution_plan.minimum_free_bytes,
        )
        return self.resource_admitter.assess(
            execution_plan,
            available_scratch_bytes=storage.observed_free_bytes,
            executor_enforces_resources=self.executor_enforces_resources,
            executor_enforces_no_network=self.supports_exploration_no_network,
            assurance_profile=self.execution_assurance_profile,
        )

    def validate_external_projection(self, path: Path) -> Path:
        """Verify mount identity and resolve one existing projection without writes."""

        root = self.artifact_plane.root.verify(for_write=False)
        projection = path.resolve(strict=True)
        try:
            relative = projection.relative_to(root).as_posix()
        except ValueError as error:
            raise CampaignIdentityError("projection escapes the guarded external root") from error
        guarded = self.artifact_plane.root.resolve(relative, for_write=False)
        if guarded.resolve(strict=True) != projection or not guarded.is_file():
            raise CampaignIdentityError("projection is not a guarded external file")
        return projection

    def _verified_rebuild_control_payload(
        self,
        *,
        logical_artifact_id: str,
        relative_path: str,
        payload_schemas: tuple[str, ...],
        maximum_bytes: int,
        budget: _CatalogSemanticReplayBudget,
    ) -> tuple[bytes, str, str]:
        """Read one guarded plan/package independently of projection inclusion."""

        if not payload_schemas or tuple(sorted(set(payload_schemas))) != payload_schemas:
            raise ValueError("catalog rebuild payload schemas must be sorted and unique")
        budget.begin_source()
        manifest_path = self.artifact_plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        try:
            manifest_payload = read_bounded_bytes(
                manifest_path,
                maximum_bytes=budget.read_limit(MAX_ARTIFACT_MANIFEST_BYTES),
            )
            budget.charge_bytes(len(manifest_payload))
            manifest = decode_artifact_manifest(manifest_payload)
            materialization = manifest.materialization
            identity = manifest.logical
            if (
                materialization.logical_artifact_id != logical_artifact_id
                or materialization.storage_root_id
                != self.artifact_plane.root.contract.storage_root_id
                or materialization.relative_path != relative_path
                or materialization.size_bytes > maximum_bytes
                or identity.logical_artifact_id != logical_artifact_id
                or identity.payload_schema not in payload_schemas
                or identity.profile is not ArtifactProfile.CANONICAL_JSON
                or identity.media_type != "application/json"
            ):
                raise CampaignIdentityError(
                    "catalog semantic registry source violates its frozen identity"
                )
            payload_read_limit = budget.read_limit(maximum_bytes)
            if materialization.size_bytes > payload_read_limit:
                raise CampaignCapabilityError(
                    "catalog semantic registry exceeds its aggregate byte limit"
                )
            port = self.artifact_plane.open(
                VerifiedArtifactInput(identity, materialization),
                maximum_bytes=payload_read_limit,
            )
            try:
                payload = port.read()
            finally:
                port.close()
            if len(payload) != materialization.size_bytes:
                raise CampaignIdentityError(
                    "catalog semantic registry source differs from its declared size"
                )
            budget.charge_bytes(len(payload))
            return payload, identity.content_sha256, identity.payload_schema
        except CampaignExecutionError:
            raise
        except (ArtifactPlaneError, BoundedFileIOError, OSError, ValueError) as error:
            raise CampaignIdentityError(
                "catalog semantic registry source custody is invalid"
            ) from error

    def _projection_semantic_registrations(
        self,
        projection_payload: bytes,
    ) -> tuple[ArtifactSemanticValidationRegistration, ...]:
        """Recompute exact output registrations from authenticated plan ownership."""

        snapshot = decode_catalog_snapshot(projection_payload)
        claims: list[tuple[str, str, str | None, tuple[str, ...]]] = []
        candidate_run_ids: set[str] = set()
        for locator in snapshot.artifact_locators:
            if locator.verification_status is not CatalogVerificationStatus.VERIFIED:
                continue
            path_run_id = _path_semantic_run_id(locator.relative_path)
            logical_run_ids = _logical_semantic_run_candidates(locator.logical_artifact_id)
            if path_run_id is None and not logical_run_ids:
                continue
            run_id_candidates = set(logical_run_ids)
            if path_run_id is not None:
                run_id_candidates.add(path_run_id)
            candidate_run_ids.update(run_id_candidates)
            if len(candidate_run_ids) > MAX_CATALOG_REBUILD_SEMANTIC_RUNS:
                raise CampaignCapabilityError(
                    "catalog semantic registry exceeds its aggregate run limit"
                )
            claims.append(
                (
                    locator.logical_artifact_id,
                    locator.relative_path,
                    path_run_id,
                    tuple(sorted(run_id_candidates)),
                )
            )

        budget = _CatalogSemanticReplayBudget(
            remaining_bytes=MAX_CATALOG_REBUILD_PLAN_PACKAGE_SOURCE_BYTES
        )
        authenticated: dict[
            str,
            Mapping[str, tuple[str, ArtifactSemanticValidationRegistration]],
        ] = {}
        for run_id in sorted(candidate_run_ids):
            if self._rebuild_control_pair_present(f"runs/{run_id}/plans/execution-plan.json"):
                authenticated[run_id] = self._authenticated_rebuild_plan_semantics(
                    run_id,
                    budget=budget,
                )

        explicitly_registered = {
            value.logical_artifact_id
            for value in self.catalog_semantic_validation_registry.registrations
        }
        derived_by_logical_id: dict[str, ArtifactSemanticValidationRegistration] = {}
        for logical_artifact_id, relative_path, path_run_id, run_ids in claims:
            matches: list[
                tuple[
                    str,
                    str,
                    ArtifactSemanticValidationRegistration,
                ]
            ] = []
            for run_id in run_ids:
                output_index = authenticated.get(run_id)
                if output_index is None:
                    continue
                owned_output = output_index.get(logical_artifact_id)
                if owned_output is None:
                    continue
                expected_path, registration = owned_output
                matches.append((run_id, expected_path, registration))

            if not matches:
                has_candidate_control = any(run_id in authenticated for run_id in run_ids)
                if (
                    path_run_id is None
                    and not has_candidate_control
                    and logical_artifact_id in explicitly_registered
                ):
                    # An exact immutable static registration owns this otherwise
                    # reserved-looking identity. It is not inferred from a path.
                    continue
                raise CampaignIdentityError(
                    "authoritative projection output lacks authenticated plan ownership"
                )
            if len(matches) != 1:
                raise CampaignIdentityError(
                    "authoritative projection output has ambiguous plan ownership"
                )
            _run_id, expected_path, registration = matches[0]
            if relative_path != expected_path:
                raise CampaignIdentityError(
                    "authoritative projection output differs from its frozen plan path"
                )
            existing = derived_by_logical_id.get(logical_artifact_id)
            if existing is not None and existing != registration:
                raise CampaignIdentityError(
                    "authoritative projection output has ambiguous semantic ownership"
                )
            derived_by_logical_id[logical_artifact_id] = registration
            if len(derived_by_logical_id) > MAX_ARTIFACT_SEMANTIC_VALIDATION_REGISTRATIONS:
                raise CampaignCapabilityError(
                    "catalog semantic registry exceeds its registration limit"
                )
        return tuple(derived_by_logical_id[key] for key in sorted(derived_by_logical_id))

    def _rebuild_control_pair_present(self, relative_path: str) -> bool:
        """Detect one exact control pair without enumerating an external directory."""

        try:
            payload_path = self.artifact_plane.root.resolve(
                relative_path,
                for_write=False,
            )
            manifest_path = self.artifact_plane.root.resolve(
                f"{relative_path}.manifest.json",
                for_write=False,
            )
            return any(path.exists() or path.is_symlink() for path in (payload_path, manifest_path))
        except (OSError, ValueError) as error:
            raise CampaignIdentityError(
                "catalog semantic registry control locator is invalid"
            ) from error

    def _authenticated_rebuild_plan_semantics(
        self,
        run_id: str,
        *,
        budget: _CatalogSemanticReplayBudget,
    ) -> Mapping[str, tuple[str, ArtifactSemanticValidationRegistration]]:
        """Authenticate one plan/package pair and derive all exact output contracts."""

        plan_payload, plan_sha256, plan_schema = self._verified_rebuild_control_payload(
            logical_artifact_id=f"execution-plan.{run_id}",
            relative_path=f"runs/{run_id}/plans/execution-plan.json",
            payload_schemas=tuple(
                sorted(
                    (
                        ProtocolExecutionPlan.SCHEMA,
                        CandidateExecutionPlan.SCHEMA,
                        EnvelopeExecutionPlan.SCHEMA,
                        ExecutionPlan.SCHEMA,
                    )
                )
            ),
            maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
            budget=budget,
        )
        plan: ProtocolExecutionPlan
        if plan_schema == ExecutionPlan.SCHEMA:
            plan = decode_canonical_bytes(
                plan_payload,
                ExecutionPlan,
                maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
            )
        elif plan_schema == EnvelopeExecutionPlan.SCHEMA:
            plan = decode_canonical_bytes(
                plan_payload,
                EnvelopeExecutionPlan,
                maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
            )
        elif plan_schema == CandidateExecutionPlan.SCHEMA:
            plan = decode_canonical_bytes(
                plan_payload,
                CandidateExecutionPlan,
                maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
            )
        else:
            plan = decode_canonical_bytes(
                plan_payload,
                ProtocolExecutionPlan,
                maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
            )
        if plan.source_plan.object_id != run_id or plan.fingerprint() != plan_sha256:
            raise CampaignIdentityError(
                "persisted execution plan differs from its rebuild identity"
            )

        package_candidates = (
            "campaign-package",
            "exploration-execution-package",
        )
        available = tuple(
            package_name
            for package_name in package_candidates
            if self._rebuild_control_pair_present(f"runs/{run_id}/plans/{package_name}.json")
        )
        if len(available) != 1:
            raise CampaignIdentityError("execution plan lacks one unambiguous persisted package")
        package_name = available[0]
        package_schemas = (
            tuple(
                sorted(
                    (
                        CampaignPackage.SCHEMA,
                        IssuedCampaignPackage.SCHEMA,
                        IssuedStudyPackage.SCHEMA,
                        EnvelopeExperimentPackage.SCHEMA,
                        ExperimentPackage.SCHEMA,
                        RetrospectiveCampaignPackage.SCHEMA,
                        ElapsedExperimentPackage.SCHEMA,
                    )
                )
            )
            if package_name == "campaign-package"
            else (ExplorationExecutionPackage.SCHEMA,)
        )
        package_payload, package_sha256, package_schema = self._verified_rebuild_control_payload(
            logical_artifact_id=f"{package_name}.{run_id}",
            relative_path=f"runs/{run_id}/plans/{package_name}.json",
            payload_schemas=package_schemas,
            maximum_bytes=(
                MAX_CAMPAIGN_PACKAGE_BYTES
                if package_name == "campaign-package"
                else MAX_AUTHORING_BYTES
            ),
            budget=budget,
        )
        if package_schema == CampaignPackage.SCHEMA:
            package: CampaignExecutionPackage | ExplorationExecutionPackage = (
                decode_canonical_bytes(
                    package_payload,
                    CampaignPackage,
                    maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
                )
            )
        elif package_schema == IssuedCampaignPackage.SCHEMA:
            package = decode_canonical_bytes(
                package_payload,
                IssuedCampaignPackage,
                maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
            )
        elif package_schema == IssuedStudyPackage.SCHEMA:
            package = decode_canonical_bytes(
                package_payload,
                IssuedStudyPackage,
                maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
            )
        elif package_schema == ElapsedExperimentPackage.SCHEMA:
            package = decode_canonical_bytes(
                package_payload,
                ElapsedExperimentPackage,
                maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
            )
        elif package_schema == RetrospectiveCampaignPackage.SCHEMA:
            package = decode_canonical_bytes(
                package_payload,
                RetrospectiveCampaignPackage,
                maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
            )
        elif package_schema == ExperimentPackage.SCHEMA:
            package = decode_canonical_bytes(
                package_payload,
                ExperimentPackage,
                maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
            )
        elif package_schema == EnvelopeExperimentPackage.SCHEMA:
            package = decode_canonical_bytes(
                package_payload,
                EnvelopeExperimentPackage,
                maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES,
            )
        else:
            package = decode_canonical_bytes(
                package_payload,
                ExplorationExecutionPackage,
                maximum_bytes=MAX_AUTHORING_BYTES,
            )
        if (
            package.fingerprint() != package_sha256
            or package.execution_plan_id != plan.execution_plan_id
            or package.implementation_commit != plan.implementation_commit
            or package.registry.fingerprint() != plan.registry_sha256
        ):
            raise CampaignIdentityError("persisted package differs from its execution plan")
        try:
            expected_plan = self._recompile_rebuild_execution_plan(package)
        except CampaignExecutionError:
            raise
        except Exception as error:
            raise CampaignIdentityError(
                "catalog rebuild cannot authenticate its persisted execution plan"
            ) from error
        if expected_plan != plan or expected_plan.canonical_bytes() != plan_payload:
            raise CampaignIdentityError(
                "persisted execution plan differs from deterministic recompilation"
            )
        if not isinstance(package, ExplorationExecutionPackage):
            # Keep the recovery authority proof immediately adjacent to the
            # external payload read/factory boundary.  Deterministic plan
            # recompilation also replays these stores, but its successful
            # return is not treated as a transferable authority token.
            self._validate_campaign_authorization(package)
            if isinstance(
                package,
                (
                    IssuedCampaignPackage,
                    IssuedStudyPackage,
                    EnvelopeExperimentPackage,
                    ExperimentPackage,
                ),
            ):
                self._validate_issued_execution_authority(package, at_utc=None)
        try:
            if isinstance(package, ExplorationExecutionPackage):
                provider = self.providers.resolve(plan.registry_sha256)
            else:
                provider = self._resolve_campaign_provider(
                    package,
                    authorization_replayed=True,
                    require_persisted_reconstruction=True,
                    budget=budget,
                ).provider
            contracts = provider.output_semantic_contracts(package.registry, plan)
            registry = capability_semantic_validation_registry(
                registry_id=f"catalog-plan-semantics.{plan.execution_plan_id}",
                plan=plan,
                contracts=contracts,
            )
        except (KeyError, ValueError) as error:
            raise CampaignCapabilityError(
                "catalog rebuild capability semantics are not statically registered"
            ) from error
        registration_by_logical_id = {
            value.logical_artifact_id: value for value in registry.registrations
        }
        if len(registration_by_logical_id) != len(registry.registrations):
            raise CampaignIdentityError("authenticated plan has duplicate semantic registrations")
        output_by_logical_id: dict[
            str,
            tuple[str, ArtifactSemanticValidationRegistration],
        ] = {}
        for task in plan.tasks:
            for output in task.outputs:
                if output.logical_artifact_id in output_by_logical_id:
                    raise CampaignIdentityError(
                        "authenticated plan has duplicate logical output ownership"
                    )
                registration = registration_by_logical_id.pop(
                    output.logical_artifact_id,
                    None,
                )
                if registration is None:
                    raise CampaignIdentityError(
                        "authenticated plan output lacks an immutable semantic closure"
                    )
                output_by_logical_id[output.logical_artifact_id] = (
                    output.relative_path,
                    registration,
                )
        if registration_by_logical_id:
            raise CampaignIdentityError(
                "authenticated plan semantic registry has surplus output ownership"
            )
        return output_by_logical_id

    def _recompile_rebuild_execution_plan(
        self,
        package: CampaignExecutionPackage | ExplorationExecutionPackage,
    ) -> ProtocolExecutionPlan:
        """Authenticate a stored plan from its package and trusted compiler closure."""

        if isinstance(
            package,
            (
                CampaignPackage,
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            if self.approval_service is None:
                raise CampaignIdentityError("catalog rebuild lacks durable authorization replay")
            self._validate_campaign_authorization(package)
            if isinstance(
                package,
                (
                    IssuedCampaignPackage,
                    IssuedStudyPackage,
                    EnvelopeExperimentPackage,
                    ExperimentPackage,
                ),
            ):
                self._validate_issued_execution_authority(package, at_utc=None)
            issued_candidate = (
                issued_standard_candidate(package)
                if isinstance(
                    package,
                    (
                        IssuedCampaignPackage,
                        IssuedStudyPackage,
                        EnvelopeExperimentPackage,
                        ExperimentPackage,
                    ),
                )
                else None
            )
            base_candidate = (
                issued_base_candidate(package)
                if isinstance(
                    package,
                    (
                        IssuedCampaignPackage,
                        IssuedStudyPackage,
                        EnvelopeExperimentPackage,
                        ExperimentPackage,
                    ),
                )
                else None
            )
            run_plan = compile_run_plan(
                run_plan_id=package.run_plan_id,
                campaign=package.campaign,
                system=package.system,
                experiment=package.experiment,
                frozen_proposal=package.frozen_proposal,
                authorization=package.authorization,
                template=package.protocol,
                registry=package.registry,
                implementation_commit=package.implementation_commit,
                approval_service=self.approval_service,
                model_set=package.model_set,
                scientific_graph=(
                    base_candidate.scientific_graph if base_candidate is not None else None
                ),
                candidate=(
                    ObjectIdentity.from_record(
                        issued_candidate.candidate_id,
                        issued_candidate,
                    )
                    if issued_candidate is not None
                    else None
                ),
                issued_extension_set=(
                    package.issued_study.issued_extensions
                    if isinstance(
                        package,
                        (EnvelopeExperimentPackage, ExperimentPackage),
                    )
                    else None
                ),
                execution_envelope_spec=(
                    package.execution_envelope_spec
                    if isinstance(package, EnvelopeExperimentPackage)
                    else None
                ),
                execution_resource_envelope_spec=(
                    package.execution_resource_envelope_spec
                    if isinstance(package, ExperimentPackage)
                    else None
                ),
                predevelopment_jit_signature_census=(
                    package.predevelopment_jit_signature_census
                    if isinstance(package, ExperimentPackage)
                    else None
                ),
                jit_graph_signature_manifest=(
                    package.jit_graph_signature_manifest
                    if isinstance(package, ExperimentPackage)
                    else None
                ),
            )
            return lower_run_plan(run_plan, package.registry)

        skeptic = package.registry.resolve(
            package.skeptic_capability_key,
            package.skeptic_capability_version,
        )
        synthesis = package.registry.resolve(
            package.synthesis_capability_key,
            package.synthesis_capability_version,
        )
        return compile_exploration_wave_execution_plan(
            exploration_plan=package.exploration_plan,
            snapshot=package.snapshot,
            snapshot_verification=package.snapshot_verification,
            wave_input=package.wave_input,
            registry=package.registry,
            implementation_commit=package.implementation_commit,
            skeptic_capability_key=package.skeptic_capability_key,
            skeptic_capability_version=package.skeptic_capability_version,
            skeptic_budget=skeptic.resource_ceiling,
            synthesis_capability_key=package.synthesis_capability_key,
            synthesis_capability_version=package.synthesis_capability_version,
            synthesis_budget=synthesis.resource_ceiling,
            wave_result_schema=ExplorationWaveResult.SCHEMA,
        )

    def catalog_rebuild_artifact_plane(
        self,
        projection_payload: bytes,
    ) -> ExternalArtifactPlane:
        """Bind rebuild to static registrations plus persisted frozen plans."""

        derived = self._projection_semantic_registrations(projection_payload)
        if (
            len(self.catalog_semantic_validation_registry.registrations) + len(derived)
            > MAX_ARTIFACT_SEMANTIC_VALIDATION_REGISTRATIONS
        ):
            raise CampaignCapabilityError(
                "catalog semantic registry exceeds its registration limit"
            )
        registry = ArtifactSemanticValidationRegistry.from_registrations(
            registry_id=(
                f"catalog-rebuild-semantics.{hashlib.sha256(projection_payload).hexdigest()[:24]}"
            ),
            registrations=(
                *self.catalog_semantic_validation_registry.registrations,
                *derived,
            ),
        )
        return self.artifact_plane.with_semantic_validation_registry(registry)

    def _validate_output_profiles(self, execution_plan: ProtocolExecutionPlan) -> None:
        # Resolve the actual publication validators before any worker can run.
        # A provider's semantic declaration does not install a binary profile.
        for profile, schema in sorted(
            {
                (output.profile, output.payload_schema)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        ):
            try:
                self.artifact_plane.validators.generic_validation(
                    profile=profile, payload_schema=schema
                )
            except ArtifactPlaneError as error:
                raise CampaignCapabilityError(
                    f"output profile {profile.value} lacks its registered validator for {schema}"
                ) from error
        self._validate_planned_input_access(execution_plan)

    @staticmethod
    def _validate_planned_input_access(execution_plan: ProtocolExecutionPlan) -> None:
        by_task = {task.task_id: task for task in execution_plan.tasks}
        for task in execution_plan.tasks:
            selected = (
                {
                    edge.operational_logical_artifact_id
                    for edge in task.scientific_inputs
                    if edge.producer_task_id is not None
                }
                if isinstance(task, ExecutionTask)
                else None
            )
            accesses: list[tuple[str, OutcomeAccess, VisibilityCeiling | None]] = [
                (output.logical_artifact_id, output.outcome_access, output.visibility_ceiling)
                for parent in task.dependency_task_ids
                for output in by_task[parent].outputs
                if selected is None or output.logical_artifact_id in selected
            ]
            accesses.extend(
                (
                    value.logical_artifact_id,
                    value.expected_outcome_access,
                    value.expected_visibility_ceiling,
                )
                for value in task.external_inputs
                if value.expected_outcome_access is not None
            )
            for artifact_id, access, visibility in accesses:
                if not planned_input_access_allowed(
                    task,
                    artifact_id,
                    access,
                    visibility,
                    reveal_barrier_authorized=task.barrier is BarrierKind.REVEAL,
                ):
                    raise CampaignCapabilityError(
                        f"task {task.task_id} lacks read permission for planned input {artifact_id}"
                    )

    def _semantic_artifact_plane(
        self,
        execution_plan: ProtocolExecutionPlan,
        contracts: tuple[CapabilityOutputSemanticContract, ...],
    ) -> ExternalArtifactPlane:
        self._validate_output_profiles(execution_plan)
        registry = capability_semantic_validation_registry(
            registry_id=f"execution-semantics.{execution_plan.execution_plan_id}",
            plan=execution_plan,
            contracts=contracts,
        )
        return self.artifact_plane.with_semantic_validation_registry(registry)

    def execute(
        self,
        *,
        package: CampaignExecutionPackage,
        run_plan: ProtocolRunPlan,
        execution_plan: ProtocolExecutionPlan,
        at_utc: str | None = None,
        reveal_authority_id: str | None = None,
        conditional_instantiations: tuple[
            ConditionalChildInstantiation,
            ...,
        ] = (),
        parent_input_bindings: tuple[FrozenParentInputBinding, ...] = (),
        retry_amendment: LeaseExpiryRetryAmendment | None = None,
        source_amendment: OperationalSourceAmendment | None = None,
    ) -> RunExecutionSummary:
        try:
            # Refuse source drift before reading large issued control publications.
            # _execute_plan repeats this check immediately before effects.
            self._check_identity(
                package,
                retry_amendment=retry_amendment,
                source_amendment=source_amendment,
            )
            try:
                self.preexecution_contract_closure(
                    package=package,
                    run_plan=run_plan,
                    execution_plan=execution_plan,
                ).require_pass()
            except ValueError as error:
                raise CampaignValidationError(str(error)) from error
            authorized_barriers: tuple[BarrierKind, ...] = (BarrierKind.REVEAL,)
            self._validate_campaign_authorization(package)
            if conditional_instantiations:
                if not isinstance(
                    package,
                    (
                        IssuedCampaignPackage,
                        IssuedStudyPackage,
                        EnvelopeExperimentPackage,
                        ExperimentPackage,
                    ),
                ):
                    raise CampaignValidationError(
                        "conditional execution requires an issued programme"
                    )
                if len(conditional_instantiations) != 1:
                    raise CampaignValidationError(
                        "conditional execution requires exactly one instantiation"
                    )
                instantiation = conditional_instantiations[0]
                base_candidate = issued_base_candidate(package)
                if (
                    instantiation.child_candidate
                    != ObjectIdentity.from_record(
                        base_candidate.candidate_id,
                        base_candidate,
                    )
                    or base_candidate.conditional_successor is not None
                ):
                    raise CampaignValidationError(
                        "conditional instantiation binds another issued child"
                    )
            if parent_input_bindings:
                if not isinstance(
                    package,
                    (
                        IssuedCampaignPackage,
                        IssuedStudyPackage,
                        EnvelopeExperimentPackage,
                        ExperimentPackage,
                    ),
                ):
                    raise CampaignValidationError(
                        "parent input binding requires an issued programme"
                    )
                base_candidate = issued_base_candidate(package)
                base_candidate_identity = ObjectIdentity.from_record(
                    base_candidate.candidate_id,
                    base_candidate,
                )
                issued_candidate = issued_standard_candidate(package)
                issued_candidate_identity = ObjectIdentity.from_record(
                    issued_candidate.candidate_id,
                    issued_candidate,
                )
                if any(
                    value.candidate not in {base_candidate_identity, issued_candidate_identity}
                    or value.scientific_graph_sha256
                    != base_candidate.scientific_graph.fingerprint()
                    for value in parent_input_bindings
                ):
                    raise CampaignValidationError(
                        "parent input binding differs from the issued candidate"
                    )
                binding_ids = tuple(
                    sorted(value.external_input_id for value in parent_input_bindings)
                )
                if binding_ids != tuple(
                    value.external_input_id for value in parent_input_bindings
                ) or len(set(binding_ids)) != len(binding_ids):
                    raise CampaignValidationError("parent input bindings must be sorted and unique")
                authenticate_parent_inputs = getattr(
                    self.provider_resolver,
                    "authenticate_parent_inputs",
                    None,
                )
                if callable(authenticate_parent_inputs):
                    authenticate_parent_inputs(
                        registry=package.registry,
                        target_candidate=parent_input_bindings[0].candidate,
                        scientific_graph_sha256=(base_candidate.scientific_graph.fingerprint()),
                        bindings=parent_input_bindings,
                    )
            source_records: tuple[CanonicalRecord, ...] = ()
            resolved_provider: _ResolvedCampaignProvider | None = None
            authority_identities: tuple[ObjectIdentity, ...] = (
                ObjectIdentity.from_record(
                    package.authorization.authorization_id,
                    package.authorization,
                ),
            )
            if isinstance(
                package,
                (
                    IssuedCampaignPackage,
                    IssuedStudyPackage,
                    EnvelopeExperimentPackage,
                    ExperimentPackage,
                ),
            ):
                if at_utc is None:
                    raise CampaignValidationError(
                        "issued execution requires an explicit decision time"
                    )
                execution_authority = self._validate_issued_execution_authority(
                    package,
                    at_utc=at_utc,
                )
                authority_identities = (
                    ObjectIdentity.from_record(
                        package.scientific_approval.authorization_id,
                        package.scientific_approval,
                    ),
                    ObjectIdentity.from_record(
                        execution_authority.authority_id,
                        execution_authority,
                    ),
                )
                standard_package = (
                    package.base
                    if isinstance(
                        package,
                        (EnvelopeExperimentPackage, ExperimentPackage),
                    )
                    else package
                )
                if isinstance(standard_package, IssuedStudyPackage):
                    issued_draft = standard_package.issued_study.authoring_package.draft
                else:
                    assert isinstance(package, IssuedCampaignPackage)
                    issued_draft = package.issued_study.draft
                standard_records: tuple[CanonicalRecord, ...] = (
                    (
                        standard_package.issued_study.authoring_package,
                        standard_package.issued_study.authoring_package.entry_package,
                    )
                    if isinstance(standard_package, IssuedStudyPackage)
                    else ()
                )
                extension_records: tuple[CanonicalRecord, ...] = (
                    (
                        package.issued_study.candidate.authoring_package,
                        package.issued_study.issued_extensions,
                        package.issued_study.extension_validation,
                        package.execution_envelope_spec,
                    )
                    if isinstance(package, EnvelopeExperimentPackage)
                    else ()
                )
                if isinstance(package, ExperimentPackage):
                    extension_records = tuple(
                        record
                        for record in (
                            package.issued_study.candidate.authoring_package,
                            package.issued_study.issued_extensions,
                            package.issued_study.extension_validation,
                            package.execution_resource_envelope_spec,
                            package.predevelopment_jit_signature_census,
                            package.jit_graph_signature_manifest,
                            *(
                                (
                                    package.campaign_elapsed_budget,
                                    package.campaign_elapsed_reservation_plan,
                                )
                                if isinstance(package, ElapsedExperimentPackage)
                                else ()
                            ),
                        )
                        if record is not None
                    )
                available_source_records: tuple[CanonicalRecord, ...] = (
                    issued_draft,
                    *standard_records,
                    *extension_records,
                    package.issued_study.proposer_attestation,
                    *package.issued_study.materialization_qualifications,
                    package.scientific_approval,
                    package.experiment,
                    package.execution_authority,
                    *conditional_instantiations,
                    *parent_input_bindings,
                    *((package.model_set,) if package.model_set is not None else ()),
                )
                if self._plan_requires_reveal_authority(execution_plan):
                    reveal_authority = (
                        None
                        if isinstance(package, ExperimentPackage)
                        and reveal_authority_id is None
                        else self._validate_issued_reveal_authority(
                            package,
                            reveal_authority_id=reveal_authority_id,
                            at_utc=at_utc,
                        )
                    )
                    if (
                        isinstance(package, ExperimentPackage)
                        and reveal_authority is None
                    ):
                        authorized_barriers = ()
                    if reveal_authority is not None:
                        authorized_barriers = (BarrierKind.REVEAL,)
                        if not isinstance(package, ExperimentPackage):
                            authority_identities = (
                                *authority_identities,
                                ObjectIdentity.from_record(
                                    reveal_authority.authority_id,
                                    reveal_authority,
                                ),
                            )
                        available_source_records = (
                            *available_source_records,
                            reveal_authority,
                        )
                resolved_provider = self._resolve_campaign_provider(
                    package,
                    authorization_replayed=True,
                )
                provider = resolved_provider.provider
                if StudyOperationAuthority.SCHEMA in provider.issued_source_schema_ids:
                    authority_store = self.study_authority_store
                    if authority_store is None:
                        raise CampaignValidationError(
                            "issued custody-authority replay is not composed"
                        )
                    try:
                        custody_authority = authority_store.load(
                            package.issued_study.custody_authority.object_id
                        )
                    except Exception as error:
                        raise CampaignValidationError(
                            "stored programme custody authority replay failed"
                        ) from error
                    if (
                        ObjectIdentity.from_record(
                            custody_authority.authority_id,
                            custody_authority,
                        )
                        != package.issued_study.custody_authority
                    ):
                        raise CampaignValidationError(
                            "issued custody-authority store returned a substituted record"
                        )
                    available_source_records = (
                        *available_source_records,
                        custody_authority,
                    )
                requested_schemas = set(provider.issued_source_schema_ids)
                source_records = tuple(
                    value for value in available_source_records if value.SCHEMA in requested_schemas
                )
                if resolved_provider.reconstruction is not None:
                    source_records = tuple(sorted(source_records, key=lambda value: value.SCHEMA))
            return self._execute_plan(
                package=package,
                source_plan=run_plan,
                execution_plan=execution_plan,
                package_name="campaign-package",
                source_plan_name="run-plan",
                source_records=source_records,
                authority_identities=authority_identities,
                authorized_barriers=authorized_barriers,
                resolved_provider=resolved_provider,
                retry_amendment=retry_amendment,
                source_amendment=source_amendment,
            )
        except InjectedSchedulerCrash:
            raise
        except CampaignExecutionError:
            raise
        except ArtifactIdentityConflict as error:
            raise CampaignConflictError("immutable artifact identity conflict") from error
        except (LiveLeaseError, OperationalWriteConflict) as error:
            raise CampaignConcurrencyError("another execution owns the active operation") from error
        except SchedulerConsistencyError as error:
            raise CampaignValidationError(
                "execution result violated its frozen contract"
            ) from error
        except ResourceAdmissionError as error:
            raise CampaignPrerequisiteError(
                "frozen execution resources cannot be enforced locally"
            ) from error
        except (TaskProcessError, TimeoutError) as error:
            raise CampaignTaskError("task execution failed") from error
        except (ArtifactPlaneError, SQLAlchemyError) as error:
            raise CampaignExecutionError("execution custody or storage boundary failed") from error
        except ValueError as error:
            raise CampaignCapabilityError(
                "runtime provider violated its registered contract"
            ) from error
        except (OSError, subprocess.SubprocessError) as error:
            raise CampaignExecutionError("execution custody or storage boundary failed") from error
        except Exception as error:
            raise CampaignInternalError() from error

    def execute_exploration(
        self,
        *,
        package: DualLoopPackage | ExplorationExecutionPackage,
        execution_plan: ProtocolExecutionPlan,
    ) -> RunExecutionSummary:
        """Execute only the input-derived surface; keep the frozen replay disabled."""

        if isinstance(package, DualLoopPackage):
            raise ReferenceFixtureReplayDisabled(
                "REFERENCE_FIXTURE_REPLAY remains a read-only historical fixture"
            )
        selected = {
            selection.proposal_id
            for selection in package.exploration_plan.selections
            if selection.disposition.value == "SELECTED"
        }
        source_records: tuple[CanonicalRecord, ...] = (
            package.wave_input,
            package.exploration_plan,
            *(
                proposal.analysis
                for proposal in package.exploration_plan.proposals
                if proposal.proposal_id in selected
            ),
        )
        try:
            return self._execute_plan(
                package=package,
                source_plan=package.exploration_plan,
                execution_plan=execution_plan,
                package_name="exploration-execution-package",
                source_plan_name="exploration-plan",
                source_records=source_records,
                authority_identities=(
                    ObjectIdentity.from_record(
                        self._record_id(package.approval_request),
                        package.approval_request,
                    ),
                ),
            )
        except CampaignExecutionError:
            raise
        except ArtifactIdentityConflict as error:
            raise CampaignConflictError("immutable artifact identity conflict") from error
        except (LiveLeaseError, OperationalWriteConflict) as error:
            raise CampaignConcurrencyError("another execution owns the active operation") from error
        except SchedulerConsistencyError as error:
            raise CampaignValidationError(
                "execution result violated its frozen contract"
            ) from error
        except ResourceAdmissionError as error:
            raise CampaignPrerequisiteError(
                "frozen execution resources cannot be enforced locally"
            ) from error
        except (TaskProcessError, TimeoutError) as error:
            raise CampaignTaskError("task execution failed") from error
        except (ArtifactPlaneError, SQLAlchemyError) as error:
            raise CampaignExecutionError("execution custody or storage boundary failed") from error
        except ValueError as error:
            raise CampaignCapabilityError(
                "runtime provider violated its registered contract"
            ) from error
        except (OSError, subprocess.SubprocessError) as error:
            raise CampaignExecutionError("execution custody or storage boundary failed") from error
        except Exception as error:
            raise CampaignInternalError() from error

    def resolve_exploration_wave(
        self,
        package: ExplorationExecutionPackage,
        execution_plan: ProtocolExecutionPlan,
    ) -> ExplorationWaveResult:
        """Resolve one successful wave from its durable receipt and outputs."""

        run_id = package.exploration_plan.plan_id
        if (
            execution_plan.source_plan
            != ObjectIdentity.from_record(run_id, package.exploration_plan)
            or execution_plan.registry_sha256 != package.registry.fingerprint()
            or execution_plan.execution_plan_id != package.execution_plan_id
        ):
            raise CampaignIdentityError(
                "exploration result resolver received another execution plan"
            )
        try:
            provider = self.providers.resolve(execution_plan.registry_sha256)
            semantic_plane = self._semantic_artifact_plane(
                execution_plan,
                provider.output_semantic_contracts(package.registry, execution_plan),
            )
        except (KeyError, ValueError) as error:
            raise CampaignCapabilityError(
                "exploration semantic registry differs from the execution plan"
            ) from error
        self._verify_persisted_plan_record(
            run_id,
            "exploration-execution-package",
            package,
        )
        self._verify_persisted_plan_record(
            run_id,
            "exploration-plan",
            package.exploration_plan,
        )
        self._verify_persisted_plan_record(
            run_id,
            "execution-plan",
            execution_plan,
        )
        engine: Engine | None = None
        try:
            engine = self.read_only_engine_factory()
            if not catalog_query_preflight(engine).passed:
                raise CampaignValidationError("operational catalog schema is invalid")
            repository = SQLiteOperationalRepository(
                engine,
                query_work_limit=self.operational_query_work_limit,
            )
            if not repository.query_preflight().passed:
                raise CampaignValidationError("operational catalog schema is invalid")
            if repository.run_status(run_id) is not OperationalStatus.SUCCEEDED:
                raise CampaignValidationError("exploration run is not operationally complete")
            attempts = repository.current_attempts(run_id)
        except CampaignExecutionError:
            raise
        except OperationalQueryWorkLimitExceeded as error:
            raise CampaignOperationalQueryWorkLimitError(
                "operational query exhausted its bounded work budget"
            ) from error
        except NoResultFound as error:
            raise CampaignValidationError("exploration run is absent") from error
        except (OSError, SQLAlchemyError) as error:
            raise CampaignExecutionError("operational catalog is unavailable") from error
        finally:
            if engine is not None:
                engine.dispose()
        current_by_task = {attempt.task_id: attempt for attempt in attempts}
        expected_task_ids = tuple(task.task_id for task in execution_plan.tasks)
        if tuple(sorted(current_by_task)) != expected_task_ids or any(
            value.disposition.value != "SUCCEEDED" for value in current_by_task.values()
        ):
            raise CampaignValidationError(
                "exploration tasks lack complete successful current attempts"
            )
        receipt_store = ExternalTaskReceiptStore(semantic_plane)
        receipts: dict[str, CanonicalTaskReceipt] = {}
        for task in execution_plan.tasks:
            attempt = current_by_task[task.task_id]
            try:
                receipt = receipt_store.read(run_id, task.task_id, attempt.attempt_id)
            except (ArtifactPlaneError, OSError, ValueError) as error:
                raise CampaignValidationError(
                    "exploration task receipt custody is invalid"
                ) from error
            if (
                receipt is None
                or receipt.operational_status is not OperationalStatus.SUCCEEDED
                or receipt.receipt_id != f"receipt.{attempt.attempt_id}"
                or receipt.task_id != task.task_id
                or receipt.implementation_commit != execution_plan.implementation_commit
                or any(not check.passed for check in receipt.checks)
                or tuple(
                    sorted(value.logical_artifact_id for value in receipt.output_materializations)
                )
                != tuple(sorted(value.logical_artifact_id for value in task.outputs))
            ):
                raise CampaignValidationError("exploration task lacks a valid durable receipt")
            receipts[task.task_id] = receipt
        external_materialization_ids = self._verified_exploration_external_inputs(
            package,
            execution_plan,
        )
        for task in execution_plan.tasks:
            receipt = receipts[task.task_id]
            dependency_materialization_ids = {
                output.materialization_id
                for dependency in task.dependency_task_ids
                for output in receipts[dependency].output_materializations
            }
            expected_input_materialization_ids = dependency_materialization_ids | {
                external_materialization_ids[value.logical_artifact_id]
                for value in task.external_inputs
            }
            if set(receipt.input_materialization_ids) != expected_input_materialization_ids:
                raise CampaignValidationError(
                    "exploration receipt inputs differ from exact dependency outputs "
                    "and external materializations"
                )
        receipt = receipts["exploration-synthesis"]
        if tuple(
            sorted(value.logical_artifact_id for value in receipt.output_materializations)
        ) != (f"hypothesis-set.{run_id}", f"wave-result.{run_id}"):
            raise CampaignValidationError("exploration synthesis receipt names unexpected outputs")
        synthesis_task = next(
            (task for task in execution_plan.tasks if task.task_id == "exploration-synthesis"),
            None,
        )
        if synthesis_task is None:
            raise CampaignValidationError("exploration synthesis task is absent")
        maximum_bytes = synthesis_task.capability.requested_resources.output_bytes
        try:
            wave_bytes = self._receipt_output_bytes(
                receipt,
                artifact_plane=semantic_plane,
                logical_artifact_id=f"wave-result.{run_id}",
                payload_schema=ExplorationWaveResult.SCHEMA,
                maximum_bytes=maximum_bytes,
                visibility_ceiling=package.exploration_plan.visibility_ceiling,
                outcome_access=package.exploration_plan.outcome_access,
            )
            hypothesis_bytes = self._receipt_output_bytes(
                receipt,
                artifact_plane=semantic_plane,
                logical_artifact_id=f"hypothesis-set.{run_id}",
                payload_schema=HypothesisSet.SCHEMA,
                maximum_bytes=maximum_bytes,
                visibility_ceiling=package.exploration_plan.visibility_ceiling,
                outcome_access=package.exploration_plan.outcome_access,
            )
            wave_result = decode_canonical_bytes(
                wave_bytes,
                ExplorationWaveResult,
                maximum_bytes=maximum_bytes,
            )
            hypothesis_set = decode_canonical_bytes(
                hypothesis_bytes,
                HypothesisSet,
                maximum_bytes=maximum_bytes,
            )
            decoded_findings: list[ExploratoryFinding] = []
            for analysis_task in (
                task for task in execution_plan.tasks if task.task_id.startswith("explore.")
            ):
                if len(analysis_task.outputs) != 1:
                    raise CampaignValidationError(
                        "exploration analysis task has an invalid output contract"
                    )
                output = analysis_task.outputs[0]
                finding_bytes = self._receipt_output_bytes(
                    receipts[analysis_task.task_id],
                    artifact_plane=semantic_plane,
                    logical_artifact_id=output.logical_artifact_id,
                    payload_schema=ExploratoryFinding.SCHEMA,
                    maximum_bytes=analysis_task.capability.requested_resources.output_bytes,
                    visibility_ceiling=output.visibility_ceiling,
                    outcome_access=output.outcome_access,
                )
                decoded_findings.append(
                    decode_canonical_bytes(
                        finding_bytes,
                        ExploratoryFinding,
                        maximum_bytes=analysis_task.capability.requested_resources.output_bytes,
                    )
                )
            skeptic_task = next(
                (task for task in execution_plan.tasks if task.task_id == "exploration-skeptic"),
                None,
            )
            if skeptic_task is None or len(skeptic_task.outputs) != 1:
                raise CampaignValidationError(
                    "exploration skeptic task has an invalid output contract"
                )
            skeptic_output = skeptic_task.outputs[0]
            skeptic_bytes = self._receipt_output_bytes(
                receipts[skeptic_task.task_id],
                artifact_plane=semantic_plane,
                logical_artifact_id=skeptic_output.logical_artifact_id,
                payload_schema=SkepticReport.SCHEMA,
                maximum_bytes=skeptic_task.capability.requested_resources.output_bytes,
                visibility_ceiling=skeptic_output.visibility_ceiling,
                outcome_access=skeptic_output.outcome_access,
            )
            skeptic_report = decode_canonical_bytes(
                skeptic_bytes,
                SkepticReport,
                maximum_bytes=skeptic_task.capability.requested_resources.output_bytes,
            )
        except CampaignValidationError:
            raise
        except (ArtifactPlaneError, OSError, ValueError) as error:
            raise CampaignValidationError(
                "exploration receipt output custody is invalid"
            ) from error
        try:
            validate_exploration_wave_result(wave_result, package.wave_input)
        except ValueError as error:
            raise CampaignValidationError(
                "persisted exploration result lacks complete identity closure"
            ) from error
        if hypothesis_set != wave_result.hypothesis_synthesis.hypothesis_set:
            raise CampaignValidationError(
                "receipt-named hypothesis output differs from the wave result"
            )
        if tuple(sorted(decoded_findings, key=lambda value: value.finding_id)) != (
            wave_result.findings
        ):
            raise CampaignValidationError(
                "receipt-named finding outputs differ from the wave result"
            )
        if skeptic_report != wave_result.skeptic_report:
            raise CampaignValidationError(
                "receipt-named skeptic output differs from the wave result"
            )
        return wave_result

    def _verify_persisted_plan_record(
        self,
        run_id: str,
        name: str,
        expected: CanonicalRecord,
    ) -> None:
        relative_path = f"runs/{run_id}/plans/{name}.json"
        expected_bytes = expected.canonical_bytes()
        try:
            path = self.artifact_plane.root.resolve(relative_path, for_write=False)
            manifest_path = self.artifact_plane.root.resolve(
                f"{relative_path}.manifest.json",
                for_write=False,
            )
            if not path.is_file() or not manifest_path.is_file():
                raise CampaignIdentityError("persisted exploration freeze is absent")
            manifest = decode_artifact_manifest(
                read_bounded_bytes(
                    manifest_path,
                    maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
                )
            )
            if (
                manifest.materialization.relative_path != relative_path
                or manifest.materialization.size_bytes != len(expected_bytes)
                or manifest.logical.logical_artifact_id != f"{name}.{run_id}"
                or manifest.logical.payload_schema != expected.SCHEMA
                or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
                or manifest.logical.content_sha256 != expected.fingerprint()
            ):
                raise CampaignIdentityError("persisted exploration freeze identity differs")
            self.artifact_plane.verify_manifest(manifest)
            if not file_matches_bytes(path, expected_bytes):
                raise CampaignIdentityError("persisted exploration freeze bytes differ")
        except CampaignExecutionError:
            raise
        except (ArtifactPlaneError, BoundedFileIOError, OSError, ValueError) as error:
            raise CampaignIdentityError(
                "persisted exploration freeze custody is invalid"
            ) from error

    def _verified_exploration_external_inputs(
        self,
        package: ExplorationExecutionPackage,
        execution_plan: ProtocolExecutionPlan,
    ) -> dict[str, str]:
        """Resolve every frozen external input to its exact persisted evidence binding."""

        run_id = package.exploration_plan.plan_id
        selected = {
            selection.proposal_id
            for selection in package.exploration_plan.selections
            if selection.disposition.value == "SELECTED"
        }
        source_records: tuple[CanonicalRecord, ...] = (
            package.wave_input,
            package.exploration_plan,
            *(
                proposal.analysis
                for proposal in package.exploration_plan.proposals
                if proposal.proposal_id in selected
            ),
        )
        records_by_fingerprint: dict[str, CanonicalRecord] = {}
        for record in source_records:
            fingerprint = record.fingerprint()
            existing_record = records_by_fingerprint.get(fingerprint)
            if existing_record is not None and existing_record != record:
                raise CampaignValidationError(
                    "exploration source records have a conflicting content identity"
                )
            records_by_fingerprint[fingerprint] = record

        scope_parents = {
            package.exploration_plan.fingerprint(): ArtifactLineageParent(
                identity=ObjectIdentity.from_record(
                    package.exploration_plan.plan_id,
                    package.exploration_plan,
                ),
                visibility_ceiling=package.exploration_plan.visibility_ceiling,
                outcome_access=package.exploration_plan.outcome_access,
            ),
            package.snapshot.fingerprint(): ArtifactLineageParent(
                identity=ObjectIdentity.from_record(
                    package.snapshot.snapshot_id,
                    package.snapshot,
                ),
                visibility_ceiling=package.snapshot.visibility_ceiling,
                outcome_access=package.snapshot.outcome_access,
            ),
            package.wave_input.fingerprint(): ArtifactLineageParent(
                identity=ObjectIdentity.from_record(
                    package.wave_input.wave_input_id,
                    package.wave_input,
                ),
                visibility_ceiling=package.wave_input.plan.visibility_ceiling,
                outcome_access=package.wave_input.plan.outcome_access,
            ),
        }

        specifications: dict[str, ExternalInputSpec] = {}
        for task in execution_plan.tasks:
            for specification in task.external_inputs:
                existing = specifications.get(specification.logical_artifact_id)
                if existing is not None and existing != specification:
                    raise CampaignValidationError(
                        "exploration plan gives one external input conflicting identities"
                    )
                specifications[specification.logical_artifact_id] = specification
        resolved: dict[str, str] = {}
        for logical_artifact_id, specification in sorted(specifications.items()):
            relative_path = f"runs/{run_id}/inputs/{logical_artifact_id}.bin"
            try:
                manifest_path = self.artifact_plane.root.resolve(
                    f"{relative_path}.manifest.json",
                    for_write=False,
                )
                manifest = decode_artifact_manifest(
                    read_bounded_bytes(
                        manifest_path,
                        maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
                    )
                )
                logical = manifest.logical
                materialization = manifest.materialization
                if (
                    specification.expected_content_sha256 is None
                    or specification.expected_visibility_ceiling is None
                    or specification.expected_outcome_access is None
                    or logical.logical_artifact_id != logical_artifact_id
                    or materialization.logical_artifact_id != logical_artifact_id
                    or materialization.relative_path != relative_path
                    or logical.content_sha256 != specification.expected_content_sha256
                    or (
                        specification.expected_payload_schema is not None
                        and logical.payload_schema != specification.expected_payload_schema
                    )
                    or (
                        specification.expected_media_type is not None
                        and logical.media_type != specification.expected_media_type
                    )
                    or (
                        specification.expected_size_bytes is not None
                        and materialization.size_bytes != specification.expected_size_bytes
                    )
                    or (
                        specification.expected_visibility_ceiling is not None
                        and logical.visibility_ceiling
                        is not specification.expected_visibility_ceiling
                    )
                    or (
                        specification.expected_outcome_access is not None
                        and logical.outcome_access is not specification.expected_outcome_access
                    )
                ):
                    raise CampaignValidationError(
                        "persisted exploration external input differs from its frozen identity"
                    )
                evidence_record = records_by_fingerprint.get(specification.expected_content_sha256)
                evidence_parent = ArtifactLineageParent(
                    identity=(
                        ObjectIdentity.from_record(logical_artifact_id, evidence_record)
                        if evidence_record is not None
                        else ObjectIdentity.from_record(
                            specification.input_id,
                            specification,
                        )
                    ),
                    visibility_ceiling=specification.expected_visibility_ceiling,
                    outcome_access=specification.expected_outcome_access,
                )
                expected_parents = [evidence_parent]
                if specification.identity_scope_sha256 is not None:
                    scope_parent = scope_parents.get(specification.identity_scope_sha256)
                    if scope_parent is None:
                        raise CampaignValidationError(
                            "exploration external input names an unknown identity scope"
                        )
                    expected_parents.append(scope_parent)
                if logical.lineage_parents != tuple(
                    sorted(expected_parents, key=lineage_parent_sort_key)
                ):
                    raise CampaignValidationError(
                        "persisted exploration external input evidence lineage differs "
                        "from its frozen binding"
                    )
                self.artifact_plane.verify_manifest(manifest)
            except CampaignExecutionError:
                raise
            except (ArtifactPlaneError, BoundedFileIOError, OSError, ValueError) as error:
                raise CampaignValidationError(
                    "persisted exploration external input custody is invalid"
                ) from error
            resolved[logical_artifact_id] = materialization.materialization_id
        return resolved

    def _receipt_output_bytes(
        self,
        receipt: CanonicalTaskReceipt,
        *,
        artifact_plane: ExternalArtifactPlane | None = None,
        logical_artifact_id: str,
        payload_schema: str,
        maximum_bytes: int,
        visibility_ceiling: VisibilityCeiling,
        outcome_access: OutcomeAccess,
    ) -> bytes:
        matches = tuple(
            (logical, materialization)
            for logical, materialization in zip(
                receipt.output_logical_artifacts,
                receipt.output_materializations,
                strict=True,
            )
            if logical.logical_artifact_id == logical_artifact_id
            and materialization.logical_artifact_id == logical_artifact_id
        )
        if len(matches) != 1:
            raise CampaignValidationError(
                "exploration receipt output does not resolve exactly once"
            )
        logical, materialization = matches[0]
        if (
            logical.payload_schema != payload_schema
            or logical.profile is not ArtifactProfile.CANONICAL_JSON
            or logical.media_type != "application/json"
            or logical.visibility_ceiling is not visibility_ceiling
            or logical.outcome_access is not outcome_access
            or materialization.size_bytes > maximum_bytes
        ):
            raise CampaignValidationError(
                "exploration receipt output violates its bounded contract"
            )
        verifier = self.artifact_plane if artifact_plane is None else artifact_plane
        verifier.verify_manifest(ArtifactManifest(logical=logical, materialization=materialization))
        path = verifier.root.resolve(
            materialization.relative_path,
            for_write=False,
        )
        try:
            payload = read_bounded_bytes(path, maximum_bytes=maximum_bytes)
        except (BoundedFileIOError, OSError) as error:
            raise CampaignValidationError(
                "exploration receipt-named output violates its byte bound"
            ) from error
        if (
            len(payload) != materialization.size_bytes
            or hashlib.sha256(payload).hexdigest() != logical.content_sha256
        ):
            raise CampaignValidationError(
                "exploration receipt-named output bytes differ from the manifest"
            )
        return payload

    def _execute_plan(
        self,
        *,
        package: ExecutionPackage,
        source_plan: CanonicalRecord,
        execution_plan: ProtocolExecutionPlan,
        package_name: str,
        source_plan_name: str,
        source_records: tuple[CanonicalRecord, ...],
        authority_identities: tuple[ObjectIdentity, ...],
        authorized_barriers: tuple[BarrierKind, ...] = (BarrierKind.REVEAL,),
        resolved_provider: _ResolvedCampaignProvider | None = None,
        retry_amendment: LeaseExpiryRetryAmendment | None = None,
        source_amendment: OperationalSourceAmendment | None = None,
    ) -> RunExecutionSummary:
        self._check_identity(
            package,
            retry_amendment=retry_amendment,
            source_amendment=source_amendment,
        )
        source_plan_id = execution_plan.source_plan.object_id
        if execution_plan.source_plan != ObjectIdentity.from_record(source_plan_id, source_plan):
            raise CampaignIdentityError("execution plan binds another source plan")
        if resolved_provider is None:
            try:
                provider = self.providers.resolve(execution_plan.registry_sha256)
            except KeyError as error:
                raise CampaignCapabilityError(str(error)) from error
        else:
            provider = resolved_provider.provider
            if provider.registry_sha256 != execution_plan.registry_sha256:
                raise CampaignCapabilityError(
                    "resolved provider differs from the execution-plan registry"
                )
        adjudication_contract = provider.scientific_adjudication_contract(
            package.registry,
            execution_plan,
        )
        adjudication_binding = self._bind_scientific_adjudication(
            package,
            execution_plan,
            adjudication_contract,
        )
        output_semantic_contracts = provider.output_semantic_contracts(
            package.registry,
            execution_plan,
        )
        expected_semantic_keys = tuple(
            sorted(
                {
                    (
                        task.capability.capability_key,
                        task.capability.capability_version,
                        output.payload_schema,
                        output.profile,
                    )
                    for task in execution_plan.tasks
                    for output in task.outputs
                },
                key=str,
            )
        )
        observed_semantic_keys = tuple(contract.key for contract in output_semantic_contracts)
        if observed_semantic_keys != expected_semantic_keys:
            raise CampaignCapabilityError(
                "runtime provider semantic contracts differ from the execution plan"
            )
        if any(
            contract.capability_implementation_sha256
            != package.registry.resolve(
                contract.capability_key,
                contract.capability_version,
            ).implementation_sha256
            for contract in output_semantic_contracts
        ):
            raise CampaignCapabilityError(
                "runtime provider semantic validator differs from its registered implementation"
            )
        semantic_plane = self._semantic_artifact_plane(
            execution_plan,
            output_semantic_contracts,
        )
        operation_minimum_free_bytes = execution_plan.minimum_free_bytes
        admission = self.resource_admission(execution_plan)
        if not admission.admitted:
            raise ResourceAdmissionError(",".join(admission.reason_codes))
        self.artifact_plane.root.verify(
            for_write=True,
            operation_minimum_free_bytes=operation_minimum_free_bytes,
        )
        self._persist_plans(
            package,
            source_plan,
            execution_plan,
            package_name=package_name,
            source_plan_name=source_plan_name,
            minimum_free_bytes=operation_minimum_free_bytes,
            provider_reconstruction=(
                None if resolved_provider is None else resolved_provider.reconstruction
            ),
        )
        provider_inputs = provider.external_inputs(execution_plan, source_records)
        expected_inputs = tuple(
            sorted(
                {
                    artifact_id
                    for task in execution_plan.tasks
                    for artifact_id in task.external_input_artifact_ids
                }
            )
        )
        if tuple(value.logical_artifact_id for value in provider_inputs) != expected_inputs:
            for value in provider_inputs:
                value.close()
            raise CampaignCapabilityError("runtime provider input set differs from plan")
        external_inputs = self._persist_provider_inputs(
            source_plan_id,
            provider_inputs,
            minimum_free_bytes=operation_minimum_free_bytes,
        )
        try:
            engine = self.engine_factory()
        except Exception as error:
            raise CampaignExecutionError("operational catalog is unavailable") from error
        receipt_store = ExternalTaskReceiptStore(
            semantic_plane,
            minimum_free_bytes=operation_minimum_free_bytes,
        )
        recovery_index = build_run_recovery_index(
            execution_plan,
            run_id=source_plan_id,
            wave_id=f"wave.{package.package_id}",
            authority_identities=authority_identities,
        )
        recovery_store = ExternalRunRecoveryStore(
            semantic_plane,
            minimum_free_bytes=operation_minimum_free_bytes,
        )
        execution_envelope_id: str | None = None
        envelope_coordinator: DurableExecutionEnvelopeCoordinator | None = None
        execution_resource_envelope_id: str | None = None
        resource_envelope_coordinator: DurableExecutionResourceEnvelopeCoordinator | None = None
        elapsed_budget_coordinator: DurableCampaignElapsedBudgetCoordinator | None = None
        elapsed_reservation_plan: CampaignElapsedReservationPlan | None = None
        if isinstance(package, EnvelopeExperimentPackage):
            if not isinstance(execution_plan, EnvelopeExecutionPlan):
                raise CampaignValidationError("EnvelopeExperimentPackage did not compile an execution plan")
            envelope_coordinator = self.execution_envelope_coordinator
            if envelope_coordinator is None:
                raise CampaignPrerequisiteError(
                    "EnvelopeExperimentPackage execution requires a durable execution-envelope store"
                )
            execution_envelope_id = f"run-envelope.{source_plan_id}"
            envelope_coordinator.ensure_initialized(
                envelope_id=execution_envelope_id,
                spec=package.execution_envelope_spec,
                execution_plan=ObjectIdentity.from_record(
                    execution_plan.execution_plan_id,
                    execution_plan,
                ),
            )
        elif isinstance(package, ExperimentPackage):
            if not isinstance(execution_plan, ExecutionPlan):
                raise CampaignValidationError("ExperimentPackage did not compile an ExecutionPlan")
            resource_envelope_coordinator = self.execution_resource_envelope_coordinator
            if resource_envelope_coordinator is None:
                raise CampaignPrerequisiteError(
                    "ExperimentPackage execution requires a durable resource-envelope store"
                )
            execution_resource_envelope_id = f"run-resource-envelope.{source_plan_id}"
            if retry_amendment is None:
                resource_envelope_coordinator.ensure_initialized(
                    envelope_id=execution_resource_envelope_id,
                    spec=package.execution_resource_envelope_spec,
                    execution_plan=ObjectIdentity.from_record(
                        execution_plan.execution_plan_id,
                        execution_plan,
                    ),
                    jit_graph_signature_manifest=package.jit_graph_signature_manifest,
                )
            else:
                from empirical_lawhood.infrastructure.retry_amendment import prepare_operational_retry
                from empirical_lawhood.runtime.recovery import OperationalFailureDiagnostic

                prepared_retry = prepare_operational_retry(
                    retry_amendment,
                    plan=execution_plan,
                    original_index=recovery_index,
                    original_envelope=resource_envelope_coordinator.load(
                        execution_resource_envelope_id
                    ),
                    recovery_store=recovery_store,
                    receipt_store=receipt_store,
                    artifact_plane=semantic_plane,
                )
                resource_envelope_coordinator.ensure_checkpoint(
                    prepared_retry.resource_checkpoint,
                    jit_graph_signature_manifest=package.jit_graph_signature_manifest,
                )
                execution_resource_envelope_id = retry_amendment.envelope_id
                recovery_index = prepared_retry.index
                recovery_store.freeze(recovery_index)
                for event in prepared_retry.retained_failures:
                    assert event.failure is not None
                    assert event.attempt_id is not None
                    recovery_store.record_failure(
                        recovery_index,
                        task_id=event.task_id,
                        attempt_id=event.attempt_id,
                        reason_code=event.reason_codes[0],
                        failure=event.failure,
                    )
                diagnostic = OperationalFailureDiagnostic.create(
                    OperationalFailureClass.RECOVERY_OBSTRUCTION,
                    sanitized_exception_type=None,
                    sanitized_message="Owner-authorized prior operational failure; no task receipt or output custody",
                )
                for task_id in retry_amendment.task_ids:
                    if any(e.task_id == task_id for e in prepared_retry.retained_failures):
                        continue
                    recovery_store.record_failure(
                        recovery_index,
                        task_id=task_id,
                        attempt_id=recovery_index.task(task_id).attempts[0].attempt_id,
                        reason_code=retry_amendment.failure_reason,
                        failure=diagnostic,
                    )
            if isinstance(package, ElapsedExperimentPackage):
                elapsed_budget_coordinator = DurableCampaignElapsedBudgetCoordinator(
                    ExternalCampaignElapsedBudgetStore(
                        semantic_plane,
                        minimum_free_bytes=operation_minimum_free_bytes,
                    ),
                    budget=package.campaign_elapsed_budget,
                    ledger_id=(
                        f"campaign-elapsed-ledger.{package.campaign_elapsed_budget.budget_id}"
                    ),
                )
                elapsed_reservation_plan = package.campaign_elapsed_reservation_plan
                if isinstance(retry_amendment, ElapsedClosureRetryAmendment):
                    from empirical_lawhood.infrastructure.retry_amendment import closure_elapsed_coordinator

                    elapsed_budget_coordinator, _ = closure_elapsed_coordinator(
                        retry_amendment,
                        package.campaign_elapsed_budget,
                        ExternalCampaignElapsedBudgetStore(
                            semantic_plane, minimum_free_bytes=operation_minimum_free_bytes
                        ),
                    )
                    elapsed_reservation_plan = replace(
                        elapsed_reservation_plan, budget=elapsed_budget_coordinator.budget_identity
                    )
                if retry_amendment is not None:
                    retry_spec = retry_amendment.resource_spec(
                        package.execution_resource_envelope_spec
                    )
                    elapsed_reservation_plan = replace(
                        elapsed_reservation_plan,
                        plan_id=f"elapsed-reservations.{retry_amendment.amendment_id}",
                        stage_id=retry_amendment.amendment_id,
                        envelope_binding_id=f"elapsed-binding.{retry_amendment.amendment_id}",
                        governed_envelope=ObjectIdentity.from_record(
                            retry_spec.envelope_spec_id, retry_spec
                        ),
                    )
        try:
            repository = SQLiteOperationalRepository(engine)
            runners = provider.runners(package.registry, source_records)
            scheduler = LocalScheduler(
                operational_repository=repository,
                artifact_writer=semantic_plane,
                receipt_store=receipt_store,
                recovery_index=recovery_index,
                recovery_store=recovery_store,
                runner_registry=RunnerRegistry(
                    runners,
                    registry_sha256=execution_plan.registry_sha256,
                ),
                implementation_commit=execution_plan.implementation_commit,
                external_input_resolver=_StaticInputResolver(external_inputs),
                failure_injector=self.failure_injector,
                executor=self.executor,
                input_port_factory=semantic_plane,
                lane=execution_plan.lane,
                minimum_free_bytes=operation_minimum_free_bytes,
                output_semantic_contracts=output_semantic_contracts,
                adjudication_task_id=(
                    None if adjudication_binding is None else adjudication_binding.task.task_id
                ),
                scientific_adjudication_context=(
                    None if adjudication_binding is None else adjudication_binding.context
                ),
                assurance_profile=self.execution_assurance_profile,
                assurance_codes=admission.assurance_codes,
                maximum_parallel_tasks=self.maximum_parallel_tasks,
                parallel_resource_capacity=self.resource_admitter.capacity,
                execution_envelope_coordinator=envelope_coordinator,
                execution_envelope_event_store=(
                    recovery_store if envelope_coordinator is not None else None
                ),
                execution_envelope_id=execution_envelope_id,
                execution_resource_envelope_coordinator=(resource_envelope_coordinator),
                execution_resource_envelope_id=execution_resource_envelope_id,
                campaign_elapsed_budget_coordinator=elapsed_budget_coordinator,
                campaign_elapsed_reservation_plan=elapsed_reservation_plan,
                jit_graph_signature_manifest=(
                    package.jit_graph_signature_manifest
                    if isinstance(package, ExperimentPackage)
                    else None
                ),
                roster_capacity_decisions=(
                    self.roster_capacity_decisions
                    if isinstance(package, ExperimentPackage)
                    else ()
                ),
                authorized_barriers=authorized_barriers,
                retry_amendment=retry_amendment,
            )
            result = scheduler.execute(source_plan_id, execution_plan)
        finally:
            engine.dispose()
        return self._summary(
            result,
            binding=adjudication_binding,
            receipt_store=receipt_store,
            artifact_plane=semantic_plane,
            resolved_provider=resolved_provider,
        )

    def _persist_provider_inputs(
        self,
        run_id: str,
        values: tuple[ExternalInputPayload, ...],
        *,
        minimum_free_bytes: int,
    ) -> tuple[VerifiedArtifactInput, ...]:
        resolved: list[VerifiedArtifactInput] = []
        try:
            for value in values:
                result = self.artifact_plane.write_stream(
                    ArtifactStreamWriteRequest(
                        logical_artifact_id=value.logical_artifact_id,
                        relative_path=(f"runs/{run_id}/inputs/{value.logical_artifact_id}.bin"),
                        payload_schema=value.payload_schema,
                        profile=value.profile,
                        media_type=value.media_type,
                        publication_scope_id=f"campaign-inputs.{run_id}",
                        publication_scope_relative_root=f"runs/{run_id}/inputs",
                        chunks=value.chunks(),
                        maximum_bytes=value.maximum_bytes,
                        maximum_chunk_bytes=value.maximum_chunk_bytes,
                        expected_size_bytes=value.size_bytes,
                        expected_physical_sha256=value.source_sha256,
                        visibility_ceiling=value.visibility_ceiling,
                        parent_visibility_ceilings=value.parent_visibility_ceilings,
                        outcome_access=value.outcome_access,
                        logical_content_sha256=value.logical_content_sha256,
                        lineage_parents=value.lineage_parents,
                        minimum_free_bytes=minimum_free_bytes,
                    )
                )
                if (
                    result.materialization.size_bytes != value.size_bytes
                    or result.materialization.physical_sha256 != value.source_sha256
                ):
                    raise CampaignCapabilityError(
                        "runtime provider input identity changed during persistence"
                    )
                resolved.append(VerifiedArtifactInput(result.logical, result.materialization))
        finally:
            for value in values:
                value.close()
        return tuple(resolved)

    def interrupt_for_maintenance(
        self, authority_path: Path, *, confirmed: bool
    ) -> CampaignMaintenanceInterruptionSummary:
        from empirical_lawhood.runtime.maintenance import ContinuationBoundInterruptionAuthority, InterruptionAuthority

        root = self.artifact_plane.root.verify(for_write=confirmed)
        relative = authority_path.absolute().relative_to(root).as_posix()
        checked = self.artifact_plane.root.resolve(relative, for_write=False)
        payload = read_bounded_bytes(checked, maximum_bytes=64 * 1024)
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError("maintenance interruption authority must be a canonical record")
        schema = document.get("schema")
        if schema == InterruptionAuthority.SCHEMA:
            authority: ContinuationBoundInterruptionAuthority | InterruptionAuthority = (
                decode_canonical_bytes(
                    payload, InterruptionAuthority, maximum_bytes=64 * 1024
                )
            )
        elif schema == ContinuationBoundInterruptionAuthority.SCHEMA:
            authority = decode_canonical_bytes(
                payload, ContinuationBoundInterruptionAuthority, maximum_bytes=64 * 1024
            )
        else:
            raise ValueError("unsupported maintenance interruption authority schema")
        bound = (
            authority.service_stop
            if isinstance(authority, ContinuationBoundInterruptionAuthority)
            else authority.stop
        )
        observed = self._maintenance_stopper.interrupt(authority, confirmed=confirmed)
        return CampaignMaintenanceInterruptionSummary(
            bound.run_id,
            observed.stopped,
            observed.completed_tasks,
            observed.interrupted_tasks,
            observed.unstarted_tasks,
            authority.continuation_run_id
            if isinstance(authority, ContinuationBoundInterruptionAuthority)
            else None,
            observed.checkpoint_relative_path,
        )

    def stop_at_receipt_boundary(
        self, authority_path: Path, *, confirmed: bool
    ) -> CampaignMaintenanceStopSummary:
        from empirical_lawhood.runtime.maintenance import MaintenanceStopAuthority

        root = self.artifact_plane.root.verify(for_write=confirmed)
        relative = authority_path.absolute().relative_to(root).as_posix()
        checked = self.artifact_plane.root.resolve(relative, for_write=False)
        authority = decode_canonical_bytes(
            read_bounded_bytes(checked, maximum_bytes=64 * 1024),
            MaintenanceStopAuthority,
            maximum_bytes=64 * 1024,
        )
        observed = self._maintenance_stopper.try_stop(authority, confirmed=confirmed)
        return CampaignMaintenanceStopSummary(
            authority.run_id,
            observed.stopped,
            observed.ready,
            observed.completed_tasks,
            observed.unstarted_tasks,
            observed.checkpoint_relative_path,
        )

    def resume(
        self,
        run_id: str,
        compile_package: Callable[
            [CampaignExecutionPackage],
            tuple[ProtocolRunPlan, ProtocolExecutionPlan],
        ],
        *,
        at_utc: str | None = None,
        reveal_authority_id: str | None = None,
        conditional_instantiations: tuple[
            ConditionalChildInstantiation,
            ...,
        ] = (),
        parent_input_bindings: tuple[FrozenParentInputBinding, ...] = (),
        retry_amendment_path: Path | None = None,
        retry_authority_id: str | None = None,
        source_amendment_path: Path | None = None,
        source_authority_id: str | None = None,
    ) -> RunExecutionSummary:
        package = self._load_package(run_id)
        run_plan, execution_plan = compile_package(package)
        if run_plan.run_plan_id != run_id:
            raise CampaignIdentityError("persisted campaign package binds another run")
        retry_amendment = self._load_retry_amendment(
            package, retry_amendment_path, retry_authority_id, at_utc=at_utc
        )
        source_amendment = self._load_source_amendment(
            package,
            execution_plan,
            source_amendment_path,
            source_authority_id,
            at_utc=at_utc,
        )
        return self.execute(
            package=package,
            run_plan=run_plan,
            execution_plan=execution_plan,
            at_utc=at_utc,
            reveal_authority_id=reveal_authority_id,
            conditional_instantiations=conditional_instantiations,
            parent_input_bindings=parent_input_bindings,
            retry_amendment=retry_amendment,
            source_amendment=source_amendment,
        )

    def _load_source_amendment(
        self,
        package: CampaignExecutionPackage,
        execution_plan: ProtocolExecutionPlan,
        path: Path | None,
        authority_id: str | None,
        *,
        at_utc: str | None,
    ) -> OperationalSourceAmendment | None:
        if path is None and authority_id is None:
            return None
        if (
            path is None
            or authority_id is None
            or not isinstance(package, ExperimentPackage)
        ):
            raise CampaignValidationError(
                "operational source repair requires an exact amendment, authority and ExperimentPackage"
            )
        root = self.artifact_plane.root.verify(for_write=False)
        relative = path.absolute().relative_to(root).as_posix()
        amendment = decode_canonical_bytes(
            read_bounded_bytes(
                self.artifact_plane.root.resolve(relative, for_write=False),
                maximum_bytes=64 * 1024,
            ),
            OperationalSourceAmendment,
            maximum_bytes=64 * 1024,
        )
        store = self.study_authority_store
        if store is None:
            raise CampaignValidationError("operational source authority store is not composed")
        authority = store.load(authority_id)
        require_study_authority(
            authority,
            kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
            subject=amendment.identity,
            prerequisite_authority=package.execution_authority.prerequisite_authority,
            grantee_id=self.study_execution_grantee_id,
            at_utc=at_utc,
        )
        if (
            authority.issuer != package.execution_authority.issuer
            or authority.scope_id != package.execution_authority.scope_id
            or authority.allows_actuation
            or amendment.run_id != execution_plan.source_plan.object_id
            or amendment.execution_plan
            != ObjectIdentity.from_record(execution_plan.execution_plan_id, execution_plan)
            or amendment.original_implementation_commit != package.implementation_commit
        ):
            raise CampaignValidationError(
                "operational source amendment changes owner, plan, scope or scientific source"
            )
        self._check_identity(package, source_amendment=amendment)
        return amendment

    def preflight_retry(
        self,
        *,
        package: CampaignExecutionPackage,
        run_plan: ProtocolRunPlan,
        execution_plan: ProtocolExecutionPlan,
        amendment_path: Path | None,
        authority_id: str | None,
        at_utc: str,
    ) -> None:
        """Prove the exact continuation through the public no-effect resume route."""
        from empirical_lawhood.infrastructure.retry_amendment import prepare_operational_retry
        from empirical_lawhood.infrastructure.execution_resource_envelopes import _JournalCheckpoint
        from empirical_lawhood.runtime.execution_envelope import RunExecutionResourceEnvelope

        amendment = self._load_retry_amendment(package, amendment_path, authority_id, at_utc=at_utc)
        if amendment is None or not isinstance(package, ExperimentPackage):
            raise CampaignValidationError("retry preview requires an explicit amendment and an ExperimentPackage or ElapsedExperimentPackage")
        if not isinstance(package, ElapsedExperimentPackage) and not isinstance(
            amendment, (ResourcePureTaskRetryAmendment, ChainedResourceRetryAmendment)
        ):
            raise CampaignValidationError("ExperimentPackage retry preview requires ResourcePureTaskRetryAmendment or ChainedResourceRetryAmendment")
        self._validate_campaign_authorization(package)
        authority = self._validate_issued_execution_authority(package, at_utc=at_utc)
        self.preexecution_contract_closure(
            package=package, run_plan=run_plan, execution_plan=execution_plan
        ).require_pass()
        provider = self._resolve_campaign_provider(
            package, authorization_replayed=True, require_persisted_reconstruction=True
        ).provider
        runners = RunnerRegistry(
            provider.runners(package.registry), registry_sha256=package.registry.fingerprint()
        )
        for task in execution_plan.tasks:
            validate_runner_resource_interface(
                runners.resolve(task),
                package.execution_resource_envelope_spec.cell_for_task(task.task_id),
            )
        self._bind_scientific_adjudication(
            package,
            execution_plan,
            provider.scientific_adjudication_contract(package.registry, execution_plan),
        )
        plane = self._semantic_artifact_plane(
            execution_plan, provider.output_semantic_contracts(package.registry, execution_plan)
        )
        plane.root.verify(
            for_write=False, operation_minimum_free_bytes=execution_plan.minimum_free_bytes
        )
        if not self.resource_admission(execution_plan).admitted:
            raise CampaignPrerequisiteError("retry resources are unavailable")
        index = build_run_recovery_index(
            execution_plan,
            run_id=run_plan.run_plan_id,
            wave_id=f"wave.{package.package_id}",
            authority_identities=(
                ObjectIdentity.from_record(
                    package.scientific_approval.authorization_id, package.scientific_approval
                ),
                ObjectIdentity.from_record(authority.authority_id, authority),
            ),
        )
        coordinator = self.execution_resource_envelope_coordinator
        if coordinator is None:
            raise CampaignPrerequisiteError("retry resource journal is not composed")
        recovery = ExternalRunRecoveryStore(
            plane, minimum_free_bytes=execution_plan.minimum_free_bytes
        )
        receipts = ExternalTaskReceiptStore(
            plane, minimum_free_bytes=execution_plan.minimum_free_bytes
        )
        prepared = prepare_operational_retry(
            amendment,
            plan=execution_plan,
            original_index=index,
            original_envelope=coordinator.load(f"run-resource-envelope.{run_plan.run_plan_id}"),
            recovery_store=recovery,
            receipt_store=receipts,
            artifact_plane=plane,
        )
        index = prepared.prior_index
        tasks = {task.task_id: task for task in execution_plan.tasks}
        for task_id in prepared.prior_terminal.completed_task_ids:
            binding = index.task(task_id)
            successful = tuple(
                (attempt, receipt)
                for attempt in binding.attempts
                if (receipt := receipts.read(index.run_id, task_id, attempt.attempt_id)) is not None
            )
            if len(successful) != 1:
                raise CampaignValidationError("completed continuation input lacks its receipt")
            attempt, receipt = successful[0]
            recovery._validate_receipt_static(index, tasks[task_id], attempt, receipt)
            for logical, materialization in zip(
                receipt.output_logical_artifacts, receipt.output_materializations, strict=True
            ):
                plane.verify_manifest(
                    ArtifactManifest(logical=logical, materialization=materialization)
                )
        _JournalCheckpoint(prepared.resource_checkpoint, package.jit_graph_signature_manifest)
        payload = prepared.resource_checkpoint.canonical_bytes()
        if (
            decode_canonical_bytes(
                payload, RunExecutionResourceEnvelope, maximum_bytes=256 * 1024 * 1024
            )
            != prepared.resource_checkpoint
        ):
            raise CampaignValidationError("full-size retry resource checkpoint differs on decode")
        from empirical_lawhood.infrastructure.recovery import decode_run_recovery_index

        if decode_run_recovery_index(prepared.index.canonical_bytes()) != prepared.index:
            raise CampaignValidationError("full-size retry index differs on decode")
        if not isinstance(package, ElapsedExperimentPackage):
            return
        elapsed_coordinator = DurableCampaignElapsedBudgetCoordinator(
            ExternalCampaignElapsedBudgetStore(
                plane, minimum_free_bytes=execution_plan.minimum_free_bytes
            ),
            budget=package.campaign_elapsed_budget,
            ledger_id=f"campaign-elapsed-ledger.{package.campaign_elapsed_budget.budget_id}",
        )
        if isinstance(amendment, ElapsedClosureRetryAmendment):
            from empirical_lawhood.infrastructure.retry_amendment import closure_elapsed_coordinator

            elapsed_coordinator, elapsed = closure_elapsed_coordinator(
                amendment,
                package.campaign_elapsed_budget,
                ExternalCampaignElapsedBudgetStore(
                    plane, minimum_free_bytes=execution_plan.minimum_free_bytes
                ),
            )
        else:
            elapsed = elapsed_coordinator.current()
        needed = sum(
            package.campaign_elapsed_reservation_plan.projected_seconds(t)
            for t in amendment.task_ids
            if package.execution_resource_envelope_spec.cell_for_task(t).native_simulator_launch
        )
        if (
            elapsed.open_interval_id is not None
            or elapsed.exhausted
            or elapsed.accumulated_active_seconds + needed
            > elapsed_coordinator.budget.hard_ceiling_seconds
        ):
            raise CampaignPrerequisiteError(
                "retry lacks a closed, sufficient cumulative elapsed budget"
            )

    def _load_retry_amendment(
        self,
        package: CampaignExecutionPackage,
        path: Path | None,
        authority_id: str | None,
        *,
        at_utc: str | None,
    ) -> LeaseExpiryRetryAmendment | None:
        if path is None and authority_id is None:
            return None
        if (
            path is None
            or authority_id is None
            or not isinstance(
                package, (ExperimentPackage, ElapsedExperimentPackage)
            )
        ):
            raise CampaignValidationError(
                "retry requires an exact amendment, authority and ExperimentPackage or ElapsedExperimentPackage"
            )
        root = self.artifact_plane.root.verify(for_write=False)
        relative = path.absolute().relative_to(root).as_posix()
        raw = read_bounded_bytes(
            self.artifact_plane.root.resolve(relative, for_write=False), maximum_bytes=64 * 1024
        )
        schema = json.loads(raw).get("schema")
        record_types: dict[str, type[LeaseExpiryRetryAmendment]] = {
            LeaseExpiryRetryAmendment.SCHEMA: LeaseExpiryRetryAmendment,
            DiagnosedPureTaskRetryAmendment.SCHEMA: DiagnosedPureTaskRetryAmendment,
            ChainedPureTaskRetryAmendment.SCHEMA: ChainedPureTaskRetryAmendment,
            CompletedRepairRetryAmendment.SCHEMA: CompletedRepairRetryAmendment,
            ElapsedClosureRetryAmendment.SCHEMA: ElapsedClosureRetryAmendment,
            ResourcePureTaskRetryAmendment.SCHEMA: ResourcePureTaskRetryAmendment,
            ChainedResourceRetryAmendment.SCHEMA: ChainedResourceRetryAmendment,
            RetainedCustodyCompletionAmendment.SCHEMA: RetainedCustodyCompletionAmendment,
        }
        record_type = record_types.get(schema)
        if record_type is None:
            raise CampaignValidationError("unknown operational retry amendment schema")
        amendment = decode_canonical_bytes(raw, record_type, maximum_bytes=64 * 1024)
        store = self.study_authority_store
        if store is None:
            raise CampaignValidationError("retry authority store is not composed")
        authority = store.load(authority_id)
        require_study_authority(
            authority,
            kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
            subject=amendment.identity,
            prerequisite_authority=package.execution_authority.prerequisite_authority,
            grantee_id=self.study_execution_grantee_id,
            at_utc=at_utc,
        )
        if (
            authority.issuer != package.execution_authority.issuer
            or authority.scope_id != package.execution_authority.scope_id
            or authority.allows_actuation
            or amendment.original_implementation_commit != package.implementation_commit
        ):
            raise CampaignValidationError(
                "retry amendment changes owner, scope or scientific source"
            )
        self._check_identity(package, retry_amendment=amendment)
        return amendment

    def _load_status_execution_plan(self, run_id: str) -> ProtocolExecutionPlan:
        # Replay the complete typed DAG under the same finite plan envelope as
        # authoring and semantic recovery. Compact receipts keep their own bound.
        payload, expected_sha256, schema = self._verified_rebuild_control_payload(
            logical_artifact_id=f"execution-plan.{run_id}",
            relative_path=f"runs/{run_id}/plans/execution-plan.json",
            payload_schemas=tuple(
                sorted(
                    {
                        ProtocolExecutionPlan.SCHEMA,
                        CandidateExecutionPlan.SCHEMA,
                        EnvelopeExecutionPlan.SCHEMA,
                        ExecutionPlan.SCHEMA,
                    }
                )
            ),
            maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
            budget=_CatalogSemanticReplayBudget(
                remaining_bytes=MAX_RUNTIME_PLAN_JSON_BYTES + MAX_ARTIFACT_MANIFEST_BYTES,
            ),
        )
        record_type: type[ProtocolExecutionPlan]
        if schema == ExecutionPlan.SCHEMA:
            record_type = ExecutionPlan
        elif schema == EnvelopeExecutionPlan.SCHEMA:
            record_type = EnvelopeExecutionPlan
        elif schema == CandidateExecutionPlan.SCHEMA:
            record_type = CandidateExecutionPlan
        else:
            record_type = ProtocolExecutionPlan
        plan = decode_canonical_bytes(
            payload,
            record_type,
            maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
        )
        if (
            plan.source_plan.object_id != run_id
            or plan.fingerprint() != expected_sha256
            or plan.canonical_bytes() != payload
        ):
            raise CampaignIdentityError("persisted status plan differs from its identity")
        return plan

    def _load_status_index(
        self,
        run_id: str,
    ) -> tuple[ProtocolExecutionPlan, ProtocolRunRecoveryIndex, ExternalRunRecoveryStore]:
        package = self._load_package(run_id)
        plan = self._load_status_execution_plan(run_id)
        if package.execution_plan_id != plan.execution_plan_id:
            raise CampaignIdentityError("persisted status package binds another execution plan")
        wave_id = f"wave.{package.package_id}"
        recovery_root = f"runs/{run_id}/recovery/{wave_id}"
        if isinstance(plan, ExecutionPlan):
            schema = RunRecoveryIndex.SCHEMA
            relative_path = f"{recovery_root}/resource-run-recovery-index.json"
        elif isinstance(plan, EnvelopeExecutionPlan):
            schema = EnvelopeRunRecoveryIndex.SCHEMA
            relative_path = f"{recovery_root}/envelope-run-recovery-index.json"
        elif isinstance(plan, CandidateExecutionPlan):
            schema = CandidateRunRecoveryIndex.SCHEMA
            relative_path = f"{recovery_root}/run-recovery-index.json"
        else:
            schema = ProtocolRunRecoveryIndex.SCHEMA
            relative_path = f"{recovery_root}/run-recovery-index.json"
        store = ExternalRunRecoveryStore(self.artifact_plane)
        index = store.read_index(relative_path, expected_schema=schema)
        if (
            index.run_id != run_id
            or index.wave_id != wave_id
            or index.execution_plan != ObjectIdentity.from_record(plan.execution_plan_id, plan)
            or index.implementation_commit != plan.implementation_commit
        ):
            raise CampaignIdentityError("persisted recovery status binds another run version")
        return plan, index, store

    def _load_status_recovery(
        self,
        run_id: str,
    ) -> tuple[
        ProtocolExecutionPlan,
        ProtocolRunRecoveryIndex,
        ExternalRunRecoveryStore,
        ProtocolRunRecoveryTerminalEvent,
    ]:
        plan, index, store = self._load_status_index(run_id)
        terminal = store.read_terminal(index)
        if terminal is None:
            raise KeyError(run_id)
        return plan, index, store, terminal

    @staticmethod
    def _blocked_failure_class(reason_code: str | None) -> OperationalFailureClass | None:
        if reason_code is None:
            return None
        if reason_code == TaskBlockReason.RESOURCE_COMPUTABILITY_UNAVAILABLE.value:
            return OperationalFailureClass.RESOURCE_REFUSAL
        if reason_code == TaskBlockReason.UNKNOWN_COMPLETION.value:
            return OperationalFailureClass.RECOVERY_OBSTRUCTION
        return OperationalFailureClass.CONTRACT_REFUSAL

    @staticmethod
    def _candidate_event_path(
        index: ProtocolRunRecoveryIndex,
        attempt: RecoveryTerminalAttempt,
    ) -> str | None:
        binding = index.task(attempt.task_id)
        candidate = next(
            (value for value in binding.attempts if value.attempt_id == attempt.attempt_id),
            None,
        )
        return None if candidate is None else candidate.event_relative_path

    def _status_from_recovery(
        self,
        run_id: str,
        *,
        include_attempt_history: bool,
        attempt_history_limit: int,
        attempt_history_cursor: str | None,
    ) -> RunStatusSummary:
        plan, index, store, terminal = self._load_status_recovery(run_id)
        task_by_id = {task.task_id: task for task in plan.tasks}
        rows: list[AttemptHistorySummary] = []
        for terminal_attempt in terminal.attempts:
            event_path = self._candidate_event_path(index, terminal_attempt)
            event: TaskRecoveryEvent | None = (
                None if event_path is None else store.read_event(index, event_path)
            )
            if (
                terminal_attempt.disposition
                in {
                    RecoveryTerminalDisposition.SUCCEEDED,
                    RecoveryTerminalDisposition.FAILED,
                }
                and event is None
            ):
                raise RunRecoveryError("terminal status attempt lacks its current recovery event")
            failure = None if event is None else event.failure
            fallback_class = (
                self._blocked_failure_class(terminal_attempt.reason_code)
                if terminal_attempt.disposition is RecoveryTerminalDisposition.BLOCKED
                else None
            )
            failure_class = failure.failure_class if failure is not None else fallback_class
            task = task_by_id[terminal_attempt.task_id]
            adjudication = any(
                output.payload_schema == ScientificAdjudicationRecord.SCHEMA
                for output in task.outputs
            )
            materialization_ids = () if event is None else event.output_materialization_ids
            rows.append(
                AttemptHistorySummary(
                    attempt_id=terminal_attempt.attempt_id,
                    task_id=terminal_attempt.task_id,
                    ordinal=terminal_attempt.ordinal,
                    disposition=terminal_attempt.disposition.value,
                    reason_code=terminal_attempt.reason_code,
                    block_kind=(
                        TaskBlockReason(terminal_attempt.reason_code).kind.value
                        if terminal_attempt.disposition is RecoveryTerminalDisposition.BLOCKED
                        and terminal_attempt.reason_code is not None
                        else None
                    ),
                    implementation_commit=index.implementation_commit,
                    execution_plan_id=index.execution_plan.object_id,
                    execution_plan_schema=index.execution_plan.object_schema,
                    execution_plan_sha256=index.execution_plan.object_fingerprint,
                    failure_class=(None if failure_class is None else failure_class.value),
                    retryable=(None if failure_class is None else failure_class.retryable),
                    sanitized_exception_type=(
                        None if failure is None else failure.sanitized_exception_type
                    ),
                    sanitized_message=(None if failure is None else failure.sanitized_message),
                    diagnostic_sha256=(None if failure is None else failure.diagnostic_sha256),
                    recovery_event_id=(None if event is None else event.event_id),
                    recovery_event_schema=(None if event is None else event.SCHEMA),
                    recovery_event_relative_path=event_path,
                    receipt_id=terminal_attempt.receipt_id,
                    receipt_relative_path=(None if event is None else event.receipt_relative_path),
                    receipt_sha256=terminal_attempt.receipt_sha256,
                    artifact_materialization_ids=materialization_ids,
                    adjudication_materialization_ids=(materialization_ids if adjudication else ()),
                )
            )
        start = 0
        if attempt_history_cursor is not None:
            cursor = _decode_attempt_history_cursor(run_id, attempt_history_cursor)
            matches = tuple(
                index_value
                for index_value, row in enumerate(rows)
                if row.task_id == cursor.task_id and row.attempt_id == cursor.attempt_id
            )
            if (
                len(matches) != 1
                or cursor.attempt_pk != matches[0] + 1
                or cursor.snapshot_max_pk != len(rows)
            ):
                raise CampaignValidationError("attempt history cursor is invalid")
            start = matches[0] + 1
        selected = rows[start : start + attempt_history_limit]
        has_more = start + len(selected) < len(rows)
        next_cursor = None
        if include_attempt_history and has_more and selected:
            last_position = start + len(selected)
            last = selected[-1]
            next_cursor = _encode_attempt_history_cursor(
                run_id,
                OperationalAttemptCursor(
                    task_id=last.task_id,
                    attempt_id=last.attempt_id,
                    attempt_pk=last_position,
                    snapshot_max_pk=len(rows),
                ),
            )
        receipt_paths = tuple(
            sorted(
                value.receipt_relative_path
                for value in rows
                if value.receipt_relative_path is not None
            )
        )
        adjudication_paths = tuple(
            sorted(
                value.receipt_relative_path
                for value in rows
                if value.receipt_relative_path is not None
                and value.adjudication_materialization_ids
            )
        )
        return RunStatusSummary(
            run_id=run_id,
            operational_status=terminal.operational_status,
            succeeded_task_ids=terminal.completed_task_ids,
            failed_task_ids=terminal.failed_task_ids,
            blocked_task_ids=terminal.blocked_task_ids,
            running_task_ids=(),
            attempt_history=(tuple(selected) if include_attempt_history else ()),
            attempt_history_limit=(attempt_history_limit if include_attempt_history else None),
            attempt_history_has_more=(has_more if include_attempt_history else False),
            attempt_history_next_cursor=next_cursor,
            operational_query_work_limit=self.operational_query_work_limit,
            implementation_commit=index.implementation_commit,
            execution_plan_id=index.execution_plan.object_id,
            execution_plan_schema=index.execution_plan.object_schema,
            execution_plan_sha256=index.execution_plan.object_fingerprint,
            recovery_index_id=index.recovery_index_id,
            recovery_index_relative_path=index.index_relative_path,
            recovery_terminal_event_relative_path=index.terminal_event_relative_path,
            artifact_receipt_relative_paths=receipt_paths,
            adjudication_receipt_relative_paths=adjudication_paths,
            status_source="RECOVERY_TERMINAL",
        )

    def status(
        self,
        run_id: str,
        *,
        include_attempt_history: bool = False,
        attempt_history_limit: int = DEFAULT_ATTEMPT_HISTORY_LIMIT,
        attempt_history_cursor: str | None = None,
    ) -> RunStatusSummary:
        if attempt_history_limit <= 0 or attempt_history_limit > MAX_ATTEMPT_HISTORY_LIMIT:
            raise CampaignValidationError(
                f"attempt history limit must be in [1, {MAX_ATTEMPT_HISTORY_LIMIT}]"
            )
        if attempt_history_cursor is not None and not include_attempt_history:
            raise CampaignValidationError("attempt history cursor requires include_attempt_history")
        try:
            decoded_cursor = (
                None
                if attempt_history_cursor is None
                else _decode_attempt_history_cursor(run_id, attempt_history_cursor)
            )
        except ValueError as error:
            raise CampaignValidationError("attempt history cursor is invalid") from error
        try:
            return self._status_from_recovery(
                run_id,
                include_attempt_history=include_attempt_history,
                attempt_history_limit=attempt_history_limit,
                attempt_history_cursor=attempt_history_cursor,
            )
        except KeyError:
            if not self.catalog_present:
                raise KeyError(run_id) from None
        except CampaignValidationError:
            raise
        except (CampaignIdentityError, RunRecoveryError) as error:
            raise CampaignExecutionError("operational recovery evidence is obstructed") from error
        engine: Engine | None = None
        history_page = None
        try:
            engine = self.read_only_engine_factory()
            if not catalog_query_preflight(engine).passed:
                raise CampaignValidationError("operational catalog schema is invalid")
            repository = SQLiteOperationalRepository(
                engine,
                query_work_limit=self.operational_query_work_limit,
            )
            if not repository.query_preflight().passed:
                raise CampaignValidationError("operational catalog schema is invalid")
            try:
                status = repository.run_status(run_id)
            except NoResultFound as error:
                raise KeyError(run_id) from error
            attempts = repository.current_attempts(run_id)
            if include_attempt_history:
                try:
                    history_page = repository.attempt_history_page(
                        run_id,
                        limit=attempt_history_limit,
                        cursor=decoded_cursor,
                    )
                except ValueError as error:
                    raise CampaignValidationError("attempt history cursor is invalid") from error
        except (KeyError, CampaignExecutionError):
            raise
        except OperationalQueryWorkLimitExceeded as error:
            raise CampaignOperationalQueryWorkLimitError(
                "operational query exhausted its bounded work budget"
            ) from error
        except (OSError, SQLAlchemyError) as error:
            raise CampaignExecutionError("operational catalog is unavailable") from error
        except Exception as error:
            raise CampaignInternalError() from error
        finally:
            if engine is not None:
                engine.dispose()
        current_by_task = {attempt.task_id: attempt for attempt in attempts}
        by_status = {
            disposition: tuple(
                sorted(
                    task_id
                    for task_id, attempt in current_by_task.items()
                    if attempt.disposition.value == disposition
                )
            )
            for disposition in ("SUCCEEDED", "FAILED", "BLOCKED", "RUNNING")
        }
        status_plan: ProtocolExecutionPlan | None = None
        status_index: ProtocolRunRecoveryIndex | None = None
        status_store: ExternalRunRecoveryStore | None = None
        try:
            status_plan, status_index, status_store = self._load_status_index(run_id)
        except KeyError:
            pass
        except (CampaignIdentityError, RunRecoveryError) as error:
            raise CampaignExecutionError("operational recovery evidence is obstructed") from error

        def history_summary(attempt: OperationalAttempt) -> AttemptHistorySummary:
            event_path: str | None = None
            event: TaskRecoveryEvent | None = None
            task: ProtocolExecutionTask | None = None
            if status_index is not None and status_store is not None and status_plan is not None:
                binding = status_index.task(attempt.task_id)
                candidate = next(
                    (value for value in binding.attempts if value.attempt_id == attempt.attempt_id),
                    None,
                )
                if candidate is not None:
                    event_path = candidate.event_relative_path
                    event = status_store.read_event(status_index, event_path)
                task = next(
                    value for value in status_plan.tasks if value.task_id == attempt.task_id
                )
            failure = None if event is None else event.failure
            fallback_class = (
                self._blocked_failure_class(attempt.reason_code)
                if attempt.disposition.value == "BLOCKED"
                else None
            )
            failure_class = failure.failure_class if failure is not None else fallback_class
            materialization_ids = () if event is None else event.output_materialization_ids
            adjudication = task is not None and any(
                output.payload_schema == ScientificAdjudicationRecord.SCHEMA
                for output in task.outputs
            )
            return AttemptHistorySummary(
                attempt_id=attempt.attempt_id,
                task_id=attempt.task_id,
                ordinal=attempt.ordinal,
                disposition=attempt.disposition.value,
                reason_code=attempt.reason_code,
                block_kind=(None if attempt.block_kind is None else attempt.block_kind.value),
                implementation_commit=(
                    None if status_index is None else status_index.implementation_commit
                ),
                execution_plan_id=(
                    None if status_index is None else status_index.execution_plan.object_id
                ),
                execution_plan_schema=(
                    None if status_index is None else status_index.execution_plan.object_schema
                ),
                execution_plan_sha256=(
                    None if status_index is None else status_index.execution_plan.object_fingerprint
                ),
                failure_class=(None if failure_class is None else failure_class.value),
                retryable=(None if failure_class is None else failure_class.retryable),
                sanitized_exception_type=(
                    None if failure is None else failure.sanitized_exception_type
                ),
                sanitized_message=(None if failure is None else failure.sanitized_message),
                diagnostic_sha256=(None if failure is None else failure.diagnostic_sha256),
                recovery_event_id=(None if event is None else event.event_id),
                recovery_event_schema=(None if event is None else event.SCHEMA),
                recovery_event_relative_path=event_path,
                receipt_id=(None if event is None else event.receipt_id),
                receipt_relative_path=(None if event is None else event.receipt_relative_path),
                receipt_sha256=(None if event is None else event.receipt_sha256),
                artifact_materialization_ids=materialization_ids,
                adjudication_materialization_ids=(materialization_ids if adjudication else ()),
            )

        history_summaries = (
            tuple(history_summary(attempt) for attempt in history_page.attempts)
            if history_page is not None
            else ()
        )
        return RunStatusSummary(
            run_id=run_id,
            operational_status=status.value,
            succeeded_task_ids=by_status["SUCCEEDED"],
            failed_task_ids=by_status["FAILED"],
            blocked_task_ids=by_status["BLOCKED"],
            running_task_ids=by_status["RUNNING"],
            attempt_history=history_summaries,
            attempt_history_limit=(attempt_history_limit if history_page is not None else None),
            attempt_history_has_more=(False if history_page is None else history_page.has_more),
            attempt_history_next_cursor=(
                None
                if history_page is None or history_page.next_cursor is None
                else _encode_attempt_history_cursor(run_id, history_page.next_cursor)
            ),
            operational_query_work_limit=self.operational_query_work_limit,
            implementation_commit=(
                None if status_index is None else status_index.implementation_commit
            ),
            execution_plan_id=(
                None if status_index is None else status_index.execution_plan.object_id
            ),
            execution_plan_schema=(
                None if status_index is None else status_index.execution_plan.object_schema
            ),
            execution_plan_sha256=(
                None if status_index is None else status_index.execution_plan.object_fingerprint
            ),
            recovery_index_id=(None if status_index is None else status_index.recovery_index_id),
            recovery_index_relative_path=(
                None if status_index is None else status_index.index_relative_path
            ),
            recovery_terminal_event_relative_path=(
                None if status_index is None else status_index.terminal_event_relative_path
            ),
            artifact_receipt_relative_paths=tuple(
                sorted(
                    value.receipt_relative_path
                    for value in history_summaries
                    if value.receipt_relative_path is not None
                )
            ),
            adjudication_receipt_relative_paths=tuple(
                sorted(
                    value.receipt_relative_path
                    for value in history_summaries
                    if value.receipt_relative_path is not None
                    and value.adjudication_materialization_ids
                )
            ),
            status_source=(
                "OPERATIONAL_CATALOG"
                if status_index is None
                else "OPERATIONAL_CATALOG_WITH_RECOVERY_EVENTS"
            ),
        )

    def _check_identity(
        self,
        package: ExecutionPackage,
        *,
        retry_amendment: LeaseExpiryRetryAmendment | None = None,
        source_amendment: OperationalSourceAmendment | None = None,
    ) -> None:
        from empirical_lawhood.infrastructure.source_origin import require_executing_target_source

        try:
            require_executing_target_source(self.repo_root)
        except PermissionError as error:
            raise CampaignIdentityError("executing package does not match target source") from error
        if retry_amendment is not None and source_amendment is not None:
            raise CampaignIdentityError("source and retry amendments cannot be combined")
        if source_amendment is not None:
            from empirical_lawhood.infrastructure.retry_amendment import verify_operational_repair_source

            verify_operational_repair_source(self.repo_root, source_amendment)
            return
        if retry_amendment is not None:
            from empirical_lawhood.infrastructure.retry_amendment import verify_operational_repair_source

            verify_operational_repair_source(self.repo_root, retry_amendment)
            return
        if not self.enforce_git_identity:
            return
        if isinstance(
            package,
            (
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            source_closure = (
                package.base.issued_study.source_closure
                if isinstance(
                    package,
                    (EnvelopeExperimentPackage, ExperimentPackage),
                )
                else package.issued_study.source_closure
            )
            if source_closure.kind is SourceClosureKind.EXACT_SOURCE_CLOSURE:
                inspector = self.study_source_closure_inspector
                if inspector is None:
                    raise CampaignIdentityError(
                        "exact-source execution lacks its composed source inspector"
                    )
                try:
                    observed = inspector.observe(source_closure)
                except (OSError, PermissionError, RuntimeError, ValueError) as error:
                    raise CampaignIdentityError(
                        "exact-source execution identity inspection failed"
                    ) from error
                if observed != source_closure:
                    raise CampaignIdentityError("exact-source execution identity changed")
                return
        try:
            commit_result = run_bounded_command(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                timeout_seconds=5,
                maximum_stdout_bytes=4096,
                maximum_stderr_bytes=16 * 1024,
            )
            dirty_result = run_bounded_command(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                timeout_seconds=10,
                maximum_stdout_bytes=1024 * 1024,
                maximum_stderr_bytes=16 * 1024,
            )
            if commit_result.returncode != 0 or dirty_result.returncode != 0:
                raise BoundedProcessError("Git identity inspection failed")
            commit = commit_result.stdout.decode("utf-8").strip()
            dirty = dirty_result.stdout.decode("utf-8").strip()
        except (BoundedProcessError, UnicodeDecodeError) as error:
            raise CampaignIdentityError("Git identity inspection failed closed") from error
        if dirty:
            raise CampaignIdentityError("campaign execution requires a clean worktree")
        if commit != package.implementation_commit:
            raise CampaignIdentityError("campaign implementation commit differs")

    def _persist_plans(
        self,
        package: ExecutionPackage,
        source_plan: CanonicalRecord,
        execution_plan: ProtocolExecutionPlan,
        *,
        package_name: str,
        source_plan_name: str,
        minimum_free_bytes: int,
        provider_reconstruction: ExecutableProviderReconstructionReceipt | None = None,
    ) -> None:
        for plan in (source_plan, execution_plan):
            if (
                isinstance(plan, (ProtocolRunPlan, ProtocolExecutionPlan))
                and len(plan.canonical_bytes()) > MAX_RUNTIME_PLAN_JSON_BYTES
            ):
                raise CampaignValidationError(
                    "runtime plan exceeds its finite serialized byte envelope"
                )
        run_id = execution_plan.source_plan.object_id
        package_parents = self._package_lineage_parents(package)
        package_visibility = VisibilityCeiling.most_restrictive(
            *(parent.visibility_ceiling for parent in package_parents)
        )
        package_access = self._most_restrictive_access(package_parents)
        package_identity = ObjectIdentity.from_record(package.package_id, package)
        package_parent = ArtifactLineageParent(
            identity=package_identity,
            visibility_ceiling=package_visibility,
            outcome_access=package_access,
        )
        source_identity = ObjectIdentity.from_record(run_id, source_plan)
        source_parent = ArtifactLineageParent(
            identity=source_identity,
            visibility_ceiling=package_visibility,
            outcome_access=package_access,
        )
        provider_reconstruction_parents = (
            tuple(
                sorted(
                    (
                        package_parent,
                        ArtifactLineageParent(
                            identity=provider_reconstruction.payload_replay_set,
                            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                            outcome_access=OutcomeAccess.OUTCOME_BLIND,
                        ),
                    ),
                    key=lineage_parent_sort_key,
                )
            )
            if provider_reconstruction is not None
            else ()
        )
        values = (
            (package_name, package, package_parents),
            (source_plan_name, source_plan, (package_parent,)),
            (
                "execution-plan",
                execution_plan,
                tuple(
                    sorted(
                        (package_parent, source_parent),
                        key=lineage_parent_sort_key,
                    )
                ),
            ),
            *(
                (
                    (
                        "provider-reconstruction",
                        provider_reconstruction,
                        provider_reconstruction_parents,
                    ),
                )
                if provider_reconstruction is not None
                else ()
            ),
        )
        for name, value, parents in values:
            payload = value.canonical_bytes()
            maximum_chunk_bytes = min(1024 * 1024, len(payload))
            self.artifact_plane.write_stream(
                ArtifactStreamWriteRequest(
                    logical_artifact_id=f"{name}.{run_id}",
                    relative_path=f"runs/{run_id}/plans/{name}.json",
                    payload_schema=value.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    publication_scope_id=f"campaign-plans.{run_id}",
                    publication_scope_relative_root=f"runs/{run_id}/plans",
                    chunks=(
                        payload[offset : offset + maximum_chunk_bytes]
                        for offset in range(0, len(payload), maximum_chunk_bytes)
                    ),
                    maximum_bytes=len(payload),
                    maximum_chunk_bytes=maximum_chunk_bytes,
                    expected_size_bytes=len(payload),
                    expected_physical_sha256=value.fingerprint(),
                    visibility_ceiling=VisibilityCeiling.most_restrictive(
                        *(parent.visibility_ceiling for parent in parents)
                    ),
                    parent_visibility_ceilings=tuple(
                        parent.visibility_ceiling for parent in parents
                    ),
                    outcome_access=self._most_restrictive_access(parents),
                    logical_content_sha256=value.fingerprint(),
                    lineage_parents=parents,
                    minimum_free_bytes=minimum_free_bytes,
                )
            )

    @staticmethod
    def _record_id(value: CanonicalRecord) -> str:
        for attribute in (
            "system_id",
            "experiment_id",
            "campaign_id",
            "issue_id",
            "receipt_id",
            "authority_id",
            "authorization_id",
            "frozen_proposal_id",
            "template_id",
            "registry_id",
            "model_set_id",
            "snapshot_id",
            "verification_id",
            "plan_id",
            "finding_id",
            "hypothesis_set_id",
            "context_id",
            "wave_input_id",
            "budget_id",
        ):
            candidate = getattr(value, attribute, None)
            if isinstance(candidate, str):
                return candidate
        raise CampaignIdentityError(f"lineage record lacks a stable ID: {type(value).__name__}")

    @classmethod
    def _package_lineage_parents(
        cls,
        package: ExecutionPackage,
    ) -> tuple[ArtifactLineageParent, ...]:
        if isinstance(package, CampaignPackage):
            records = (
                package.system,
                package.experiment,
                package.campaign,
                package.frozen_proposal,
                package.authorization,
                package.protocol,
                package.registry,
                *((package.model_set,) if package.model_set is not None else ()),
                *(
                    (
                        package.campaign_elapsed_budget,
                        package.campaign_elapsed_reservation_plan,
                    )
                    if isinstance(package, ElapsedExperimentPackage)
                    else ()
                ),
            )
        elif isinstance(
            package,
            (
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            records = (
                package.issued_study,
                package.publication_receipt,
                package.system,
                package.experiment,
                package.campaign,
                package.frozen_proposal,
                package.scientific_approval,
                package.execution_authority,
                package.protocol,
                package.registry,
                *((package.model_set,) if package.model_set is not None else ()),
            )
        elif isinstance(package, DualLoopPackage):
            records = (
                package.system,
                package.snapshot,
                package.snapshot_verification,
                package.exploration_plan,
                *package.findings,
                package.hypotheses,
                package.design_context,
                package.approval_request,
                package.registry,
            )
        else:
            records = (
                package.system,
                package.snapshot,
                package.snapshot_verification,
                package.exploration_plan,
                package.wave_input,
                package.design_context,
                package.approval_request,
                package.registry,
            )
        parents = []
        for record in records:
            visibility = getattr(record, "visibility_ceiling", None)
            if not isinstance(visibility, VisibilityCeiling):
                visibility = getattr(
                    record,
                    "design_visibility_ceiling",
                    VisibilityCeiling.PROSPECTIVE,
                )
            if not isinstance(visibility, VisibilityCeiling):
                raise CampaignIdentityError("lineage record has invalid visibility")
            access = getattr(record, "outcome_access", None)
            if not isinstance(access, OutcomeAccess):
                access = (
                    OutcomeAccess.EVALUATION_REVEALED
                    if visibility
                    in {
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        VisibilityCeiling.PRIVILEGED_TRUTH,
                    }
                    else OutcomeAccess.OUTCOME_BLIND
                )
            parents.append(
                ArtifactLineageParent(
                    identity=ObjectIdentity.from_record(cls._record_id(record), record),
                    visibility_ceiling=visibility,
                    outcome_access=access,
                )
            )
        return tuple(sorted(parents, key=lineage_parent_sort_key))

    @staticmethod
    def _most_restrictive_access(
        parents: tuple[ArtifactLineageParent, ...],
    ) -> OutcomeAccess:
        return most_restrictive_outcome_access(*(parent.outcome_access for parent in parents))

    def load_persisted_package(self, run_id: str) -> CampaignExecutionPackage:
        """Read and verify the exact externally persisted package for recovery."""

        return self._load_package(run_id)

    def _load_package(self, run_id: str) -> CampaignExecutionPackage:
        relative = f"runs/{run_id}/plans/campaign-package.json"
        path = self.artifact_plane.root.resolve(relative, for_write=False)
        manifest = self.artifact_plane.root.resolve(f"{relative}.manifest.json", for_write=False)
        if not path.is_file() or not manifest.is_file():
            raise KeyError(run_id)
        try:
            manifest_record = decode_artifact_manifest(
                read_bounded_bytes(
                    manifest,
                    maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
                )
            )
        except (BoundedFileIOError, OSError, ValueError) as error:
            raise CampaignIdentityError("persisted campaign manifest is invalid") from error
        if manifest_record.materialization.relative_path != relative:
            raise CampaignIdentityError("persisted campaign manifest points elsewhere")
        if (
            manifest_record.materialization.size_bytes > MAX_CAMPAIGN_PACKAGE_BYTES
            or manifest_record.logical.logical_artifact_id != f"campaign-package.{run_id}"
            or manifest_record.logical.payload_schema
            not in {
                CampaignPackage.SCHEMA,
                IssuedCampaignPackage.SCHEMA,
                IssuedStudyPackage.SCHEMA,
                EnvelopeExperimentPackage.SCHEMA,
                ExperimentPackage.SCHEMA,
                RetrospectiveCampaignPackage.SCHEMA,
                ElapsedExperimentPackage.SCHEMA,
            }
            or manifest_record.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest_record.logical.media_type != "application/json"
        ):
            raise CampaignIdentityError(
                "persisted campaign manifest violates its bounded package contract"
            )
        self.artifact_plane.verify_manifest(manifest_record)
        try:
            payload = read_bounded_bytes(path, maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES)
        except (BoundedFileIOError, OSError) as error:
            raise CampaignIdentityError("persisted campaign package exceeds its bound") from error
        try:
            value = loads_campaign_package_authoring(
                payload.decode("utf-8"),
                media_type="application/json",
            )
        except (UnicodeDecodeError, AuthoringCodecError) as error:
            raise CampaignIdentityError("persisted campaign package is invalid") from error
        if not isinstance(
            value,
            (
                CampaignPackage,
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            raise CampaignIdentityError(
                "persisted run document is not a follow-up campaign package"
            )
        if manifest_record.logical.payload_schema != value.SCHEMA:
            raise CampaignIdentityError(
                "persisted campaign package schema differs from its manifest"
            )
        if value.canonical_bytes() != payload:
            raise CampaignIdentityError("persisted campaign package bytes are not canonical")
        if manifest_record.logical.content_sha256 != value.fingerprint():
            raise CampaignIdentityError("persisted campaign package identity differs")
        return value

    def _validate_campaign_authorization(
        self,
        package: CampaignExecutionPackage,
    ) -> None:
        if self.approval_service is None:
            raise CampaignValidationError(
                "stored durable campaign authorization replay is not composed"
            )
        try:
            replayed = self.approval_service.replay(
                system=package.system,
                frozen_proposal=ObjectIdentity.from_record(
                    package.frozen_proposal.frozen_proposal_id,
                    package.frozen_proposal,
                ),
                authorization=ObjectIdentity.from_record(
                    package.authorization.authorization_id,
                    package.authorization,
                ),
            )
        except Exception as error:
            raise CampaignValidationError(
                "stored durable campaign authorization replay failed"
            ) from error
        if replayed != package.experiment:
            raise CampaignValidationError(
                "campaign experiment differs from its durable authorization"
            )

    def _validate_issued_execution_authority(
        self,
        package: IssuedCampaignPackageRoot,
        *,
        at_utc: str | None,
    ) -> StudyOperationAuthority:
        store = self.study_authority_store
        if store is None:
            raise CampaignValidationError("issued execution-authority replay is not composed")
        try:
            authority = store.load(package.execution_authority.authority_id)
            if authority != package.execution_authority or ObjectIdentity.from_record(
                authority.authority_id, authority
            ) != ObjectIdentity.from_record(
                package.execution_authority.authority_id,
                package.execution_authority,
            ):
                raise PermissionError("execution authority store returned a substituted record")
            require_study_authority(
                authority,
                kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
                subject=ObjectIdentity.from_record(
                    package.issued_study.issue_id,
                    package.issued_study,
                ),
                prerequisite_authority=ObjectIdentity.from_record(
                    package.scientific_approval.authorization_id,
                    package.scientific_approval,
                ),
                grantee_id=self.study_execution_grantee_id,
                at_utc=at_utc,
            )
            return authority
        except Exception as error:
            raise CampaignValidationError(
                "stored programme execution authority replay failed"
            ) from error

    @staticmethod
    def _plan_requires_reveal_authority(plan: ProtocolExecutionPlan) -> bool:
        return any(
            task.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL} for task in plan.tasks
        )

    def _validate_issued_reveal_authority(
        self,
        package: IssuedCampaignPackageRoot,
        *,
        reveal_authority_id: str | None,
        at_utc: str,
    ) -> StudyOperationAuthority:
        if reveal_authority_id is None:
            raise CampaignRevealAuthorityError(
                "issued evaluator execution requires separate reveal authority"
            )
        try:
            validate_stable_id(
                reveal_authority_id,
                field_name="reveal_authority_id",
            )
            store = self.study_authority_store
            if store is None:
                raise PermissionError("issued reveal-authority replay is not composed")
            authority = store.load(reveal_authority_id)
            if authority.authority_id != reveal_authority_id:
                raise PermissionError("reveal authority store returned a substituted record")
            require_study_authority(
                authority,
                kind=StudyAuthorityKind.OUTCOME_REVEAL,
                subject=ObjectIdentity.from_record(
                    package.issued_study.issue_id,
                    package.issued_study,
                ),
                prerequisite_authority=ObjectIdentity.from_record(
                    package.execution_authority.authority_id,
                    package.execution_authority,
                ),
                grantee_id=self.study_reveal_grantee_id,
                at_utc=at_utc,
            )
            return authority
        except CampaignRevealAuthorityError:
            raise
        except Exception as error:
            raise CampaignRevealAuthorityError(
                "stored programme reveal authority replay failed"
            ) from error

    @staticmethod
    def _bind_scientific_adjudication(
        package: ExecutionPackage,
        execution_plan: ProtocolExecutionPlan,
        contract: ScientificAdjudicationOutputContract | None,
    ) -> _ScientificAdjudicationBinding | None:
        if not isinstance(
            package,
            (
                CampaignPackage,
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            if contract is not None:
                raise CampaignCapabilityError(
                    "non-campaign runtime cannot expose a campaign adjudication"
                )
            return None
        if contract is None:
            raise CampaignCapabilityError(
                "campaign runtime lacks a scientific adjudication output contract"
            )
        matches = tuple(
            task
            for task in execution_plan.tasks
            if task.capability.capability_key == contract.capability_key
            and task.capability.capability_version == contract.capability_version
            and any(output.output_id == contract.output_id for output in task.outputs)
        )
        if len(matches) != 1:
            raise CampaignCapabilityError(
                "scientific adjudication contract does not resolve one frozen task"
            )
        task = matches[0]
        output = next(value for value in task.outputs if value.output_id == contract.output_id)
        sealed_evaluation = (
            task.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
            and task.capability.kind is CapabilityKind.EVALUATOR
        )
        development_adjudication = (
            task.stage is ScientificStage.REPORT
            and task.capability.kind is CapabilityKind.REPORTER
            and output.outcome_access is OutcomeAccess.DEVELOPMENT_VISIBLE
            and output.visibility_ceiling
            in {
                VisibilityCeiling.DEVELOPMENT_ONLY,
                # Development analyses of already outcome-visible historical
                # evidence must retain that non-promotable lineage ceiling.
                VisibilityCeiling.OUTCOME_VISIBLE,
            }
        )
        revealed_adjudication = (
            task.stage is ScientificStage.REPORT
            and task.capability.kind is CapabilityKind.REPORTER
            and output.outcome_access is OutcomeAccess.EVALUATION_REVEALED
            and output.visibility_ceiling is VisibilityCeiling.OUTCOME_VISIBLE
        )
        admission_adjudication = (
            task.stage is ScientificStage.ADMISSION
            and task.capability.kind is CapabilityKind.ADMISSION_EVALUATOR
            and output.outcome_access is OutcomeAccess.EVALUATION_REVEALED
            and output.visibility_ceiling is VisibilityCeiling.OUTCOME_VISIBLE
        )
        if not task.dependency_task_ids or not (
            sealed_evaluation
            or development_adjudication
            or revealed_adjudication
            or admission_adjudication
        ):
            raise CampaignCapabilityError(
                "scientific adjudication must be a receipt-dependent sealed evaluator "
                "or a receipt-dependent development/revealed reporter or admission task"
            )
        if (
            output.payload_schema != contract.payload_schema
            or output.profile is not ArtifactProfile.CANONICAL_JSON
            or contract.maximum_bytes > task.capability.requested_resources.output_bytes
        ):
            raise CampaignCapabilityError(
                "scientific adjudication output differs from its frozen contract"
            )
        context = ScientificAdjudicationContext(
            execution_plan=ObjectIdentity.from_record(
                execution_plan.execution_plan_id,
                execution_plan,
            ),
            evidence_world_id=package.system.world.world_id,
            evidence_world_kind=package.system.world.kind,
            relation=ObjectIdentity.from_record(
                package.experiment.relation.relation_id,
                package.experiment.relation,
            ),
            independent_unit_id=package.experiment.independent_unit_id,
            information_cutoffs=tuple(
                ObjectIdentity.from_record(cutoff.cutoff_id, cutoff)
                for cutoff in package.experiment.information_cutoffs
            ),
            visibility_ceiling=output.visibility_ceiling,
            outcome_access=output.outcome_access,
            fixture_scope_id=contract.fixture_scope_id,
            plumbing_only=contract.plumbing_only,
        )
        return _ScientificAdjudicationBinding(
            task=task,
            output_logical_artifact_id=output.logical_artifact_id,
            contract=contract,
            context=context,
        )

    def _summary(
        self,
        result: SchedulerResult,
        *,
        binding: _ScientificAdjudicationBinding | None,
        receipt_store: ExternalTaskReceiptStore,
        artifact_plane: ExternalArtifactPlane,
        resolved_provider: _ResolvedCampaignProvider | None,
    ) -> RunExecutionSummary:
        def not_adjudicated(reason_code: str) -> RunExecutionSummary:
            return self._not_adjudicated(
                result,
                reason_code,
                resolved_provider=resolved_provider,
            )

        if result.status is not OperationalStatus.SUCCEEDED:
            return not_adjudicated("EXECUTION_NOT_SUCCEEDED")
        if binding is None:
            return not_adjudicated("SCIENTIFIC_ADJUDICATION_NOT_CONFIGURED")
        task_receipts = tuple(
            receipt for receipt in result.receipts if receipt.task_id == binding.task.task_id
        )
        if len(task_receipts) != 1:
            return not_adjudicated("ADJUDICATION_RECEIPT_MISSING")
        observed_receipt = task_receipts[0]
        try:
            durable_receipt = receipt_store.read(
                result.run_id,
                binding.task.task_id,
                observed_receipt.attempt_id,
            )
        except (ArtifactPlaneError, OSError, ValueError):
            return not_adjudicated("ADJUDICATION_RECEIPT_INVALID")
        if durable_receipt is None:
            return not_adjudicated("ADJUDICATION_RECEIPT_MISSING")
        if (
            durable_receipt != observed_receipt
            or durable_receipt.receipt_id not in result.receipt_ids
            or durable_receipt.operational_status is not OperationalStatus.SUCCEEDED
        ):
            return not_adjudicated("ADJUDICATION_RECEIPT_INVALID")
        if len(durable_receipt.output_logical_artifacts) != len(
            durable_receipt.output_materializations
        ):
            return not_adjudicated("ADJUDICATION_RECEIPT_INVALID")
        receipt_outputs = tuple(
            (logical, materialization)
            for logical, materialization in zip(
                durable_receipt.output_logical_artifacts,
                durable_receipt.output_materializations,
                strict=True,
            )
            if logical.logical_artifact_id == binding.output_logical_artifact_id
        )
        if len(receipt_outputs) != 1:
            return not_adjudicated("ADJUDICATION_MATERIALIZATION_MISMATCH")
        logical, materialization = receipt_outputs[0]
        if (
            logical.payload_schema != binding.contract.payload_schema
            or logical.profile is not ArtifactProfile.CANONICAL_JSON
            or logical.visibility_ceiling is not binding.context.visibility_ceiling
            or logical.outcome_access is not binding.context.outcome_access
        ):
            return not_adjudicated("ADJUDICATION_MATERIALIZATION_MISMATCH")
        verified_input = VerifiedArtifactInput(logical, materialization)
        try:
            port = artifact_plane.open(
                verified_input,
                maximum_bytes=binding.contract.maximum_bytes,
            )
            try:
                payload = port.read()
                if port.bytes_read != materialization.size_bytes:
                    raise ValueError("adjudication materialization was not read completely")
            finally:
                port.close()
        except (ArtifactPlaneError, OSError, ValueError):
            return not_adjudicated("ADJUDICATION_MATERIALIZATION_INVALID")
        try:
            record = decode_scientific_adjudication(
                payload,
                payload_schema=binding.contract.payload_schema,
            )
        except (ValueError, TypeError):
            return not_adjudicated("ADJUDICATION_RECORD_INVALID")
        receipt_by_task = {receipt.task_id: receipt for receipt in result.receipts}
        try:
            expected_required_receipts = tuple(
                sorted(
                    receipt_by_task[task_id].receipt_id
                    for task_id in binding.task.dependency_task_ids
                )
            )
        except KeyError:
            return not_adjudicated("ADJUDICATION_LINEAGE_MISMATCH")
        expected_output_ids = tuple(
            sorted(value.logical_artifact_id for value in durable_receipt.output_logical_artifacts)
        )
        if (
            record.adjudication_id != f"adjudication.{result.run_id}.{binding.task.task_id}"
            or record.run_id != result.run_id
            or record.adjudication_task_id != binding.task.task_id
            or not record.matches_context(binding.context)
            or record.input_materialization_ids != durable_receipt.input_materialization_ids
            or record.output_logical_artifact_ids != expected_output_ids
            or record.required_receipt_ids != expected_required_receipts
        ):
            return not_adjudicated("ADJUDICATION_LINEAGE_MISMATCH")
        return RunExecutionSummary(
            run_id=result.run_id,
            operational_status=result.status.value,
            completed_task_ids=result.completed_task_ids,
            failed_task_ids=result.failed_task_ids,
            blocked_task_ids=result.blocked_task_ids,
            receipt_ids=result.receipt_ids,
            adjudication_state=AdjudicationReadoutState.ADJUDICATED,
            adjudication_evaluability=record.evaluability,
            scientific_status=record.scientific_status,
            admission_status=record.admission_status,
            scientific_reason_codes=record.reason_codes,
            adjudication_id=record.adjudication_id,
            adjudication_fingerprint=record.fingerprint(),
            adjudication_artifact_id=logical.logical_artifact_id,
            adjudication_materialization_id=materialization.materialization_id,
            adjudication_receipt_id=durable_receipt.receipt_id,
            evidence_world_id=record.evidence_world_id,
            fixture_scope_id=record.fixture_scope_id,
            plumbing_only=record.plumbing_only,
            provider_reconstruction_receipt_id=(
                None
                if resolved_provider is None or resolved_provider.reconstruction is None
                else resolved_provider.reconstruction.receipt_id
            ),
            provider_reconstruction_receipt_fingerprint=(
                None
                if resolved_provider is None or resolved_provider.reconstruction is None
                else resolved_provider.reconstruction.fingerprint()
            ),
            provider_recovery_receipt_id=(
                None
                if resolved_provider is None or resolved_provider.recovery is None
                else resolved_provider.recovery.receipt_id
            ),
            provider_recovery_receipt_fingerprint=(
                None
                if resolved_provider is None or resolved_provider.recovery is None
                else resolved_provider.recovery.fingerprint()
            ),
        )

    @staticmethod
    def _not_adjudicated(
        result: SchedulerResult,
        reason_code: str,
        *,
        resolved_provider: _ResolvedCampaignProvider | None = None,
    ) -> RunExecutionSummary:
        return RunExecutionSummary(
            run_id=result.run_id,
            operational_status=result.status.value,
            completed_task_ids=result.completed_task_ids,
            failed_task_ids=result.failed_task_ids,
            blocked_task_ids=result.blocked_task_ids,
            receipt_ids=result.receipt_ids,
            adjudication_state=AdjudicationReadoutState.NOT_ADJUDICATED,
            adjudication_evaluability=AdjudicationEvaluability.UNEVALUABLE,
            scientific_status=None,
            admission_status=None,
            scientific_reason_codes=(reason_code,),
            provider_reconstruction_receipt_id=(
                None
                if resolved_provider is None or resolved_provider.reconstruction is None
                else resolved_provider.reconstruction.receipt_id
            ),
            provider_reconstruction_receipt_fingerprint=(
                None
                if resolved_provider is None or resolved_provider.reconstruction is None
                else resolved_provider.reconstruction.fingerprint()
            ),
            provider_recovery_receipt_id=(
                None
                if resolved_provider is None or resolved_provider.recovery is None
                else resolved_provider.recovery.receipt_id
            ),
            provider_recovery_receipt_fingerprint=(
                None
                if resolved_provider is None or resolved_provider.recovery is None
                else resolved_provider.recovery.fingerprint()
            ),
        )
