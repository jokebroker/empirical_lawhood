"""Expectation-only authoring manifests for the held Glenn 2026 release.

This module freezes source-backed facts that are already known about Zenodo
record 17163053 and the historical G1 transformation.  It deliberately does
not manufacture environment facts.  Callers must supply exact registered
storage-root identities, static capability bindings, selector identities, and
the immutable materialization/artifact identities produced by prior trusted
operations.

The historical CSV is a comparison-only artifact.  It is never added to the
transformation parent set, and these builders issue no scientific verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.dataset_authority import (
    DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA,
    DatasetControlLimits,
    DatasetFileLimits,
    DatasetStorageScope,
    DatasetWorkEnvelope,
)
from empirical_lawhood.planning.dataset_manifests import (
    DatasetAuthoringExceptionCode,
    DatasetCapabilityBinding,
    DatasetCapabilityKind,
    DatasetCausalContract,
    DatasetComparisonTarget,
    DatasetExpectedInvariant,
    DatasetFamilyInput,
    DatasetFieldContract,
    DatasetFieldDataType,
    DatasetFieldRole,
    DatasetInvariantKind,
    DatasetOutputContract,
    DatasetOutputRole,
    DatasetRegistrationManifest,
    DatasetReleaseInput,
    DatasetSelectorMemberContract,
    DatasetSplitContract,
    DatasetTransformInput,
    DatasetTransformationManifest,
    DatasetTypeNullNormalization,
)
from empirical_lawhood.planning.datasets import (
    AccessState,
    DatasetEvidenceClass,
    DatasetMaterialization,
    DatasetMaterializationClass,
    DatasetSelectorKind,
    DatasetSelectorRef,
    ExternalIdentifier,
    ExternalIdentifierKind,
    ReleaseResolutionClass,
)

from .contracts import (
    ACTION_COLUMNS,
    CANONICAL_COLUMNS,
    GLENN_2026_PROFILE,
    NUISANCE_COLUMNS,
)
from .transform import GLENN_OUTPUT_SCHEMA_ID, GLENN_PARQUET_PROFILE_ID


GLENN_ARCHIVE_RELATIVE_LOCATOR: Final = (
    "data/source/glenn-2026-v1/upstream/gdglenn_prr_2026.zip"
)
GLENN_ARCHIVE_SIZE_BYTES: Final = 549_154_725
GLENN_ARCHIVE_SHA256: Final = (
    "18af82987cfd5369ba670da6ab8413656d89ebaa1f0ed34b1034453ad2c8d92a"
)
GLENN_READ_SET_RELATIVE_LOCATOR: Final = (
    "runs/glenn-g1-selfserve-v1/source-read-set.csv"
)
GLENN_READ_SET_SIZE_BYTES: Final = 735
GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR: Final = (
    "runs/glenn-g1-selfserve-v1/observations.csv"
)
GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES: Final = 25_291
# ``ResourceBudget.source_scan_bytes`` counts each distinct authorized input
# once.  The production composition nevertheless performs two guard hashes and
# two adapter hashes of the archive.  This separate fixed limit makes that I/O
# amplification explicit without misrepresenting the scheduler's budget
# semantics; a fifth full pass requires a contract revision and review.
GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT: Final = 4
GLENN_ARCHIVE_FULL_SCAN_READ_LIMIT_BYTES: Final = (
    GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT * GLENN_ARCHIVE_SIZE_BYTES
)

_FAMILY_ID = "family.glenn-prr"
_RELEASE_ID = "release.glenn-2026"
_ARCHIVE_MATERIALIZATION_ID = "materialization.glenn-archive"
_OUTPUT_MATERIALIZATION_ID = "materialization.glenn-g1-observations"
_OUTPUT_ID = "output.glenn-observations"
_MAXIMUM_OUTPUT_BYTES = 64 * 1024**2
_REGISTRATION_CAPABILITY_KINDS = frozenset(
    {
        DatasetCapabilityKind.CONFIG,
        DatasetCapabilityKind.EVIDENCE_VERIFIER,
        DatasetCapabilityKind.ENVIRONMENT,
        DatasetCapabilityKind.INSPECTOR,
        DatasetCapabilityKind.RUNTIME,
    }
)
_TRANSFORMATION_CAPABILITY_KINDS = _REGISTRATION_CAPABILITY_KINDS | frozenset(
    {DatasetCapabilityKind.TRANSFORM}
)
_CAPABILITY_REGISTRY_KEYS = {
    DatasetCapabilityKind.CONFIG: "dataset.config.glenn",
    DatasetCapabilityKind.EVIDENCE_VERIFIER: "dataset.evidence-verifier.glenn",
    DatasetCapabilityKind.ENVIRONMENT: "dataset.environment.glenn",
    DatasetCapabilityKind.INSPECTOR: "dataset.inspector.glenn",
    DatasetCapabilityKind.RUNTIME: "dataset.runtime.glenn",
    DatasetCapabilityKind.TRANSFORM: "dataset.transform.glenn",
}


def _require_root(identity: ObjectIdentity, *, field_name: str) -> None:
    if not isinstance(identity, ObjectIdentity):
        raise ValueError(f"{field_name} must be an exact ObjectIdentity")
    if identity.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA:
        raise ValueError(f"{field_name} must identify a follow-up ExternalRootContract")


def _require_identity_schema(
    identity: ObjectIdentity,
    *,
    field_name: str,
    expected_schema: str,
) -> None:
    if not isinstance(identity, ObjectIdentity):
        raise ValueError(f"{field_name} must be an exact ObjectIdentity")
    if identity.object_schema != expected_schema:
        raise ValueError(f"{field_name} must identify {expected_schema!r}")


def _require_capabilities(
    capabilities: tuple[DatasetCapabilityBinding, ...],
    *,
    required_kinds: frozenset[DatasetCapabilityKind],
) -> None:
    if not isinstance(capabilities, tuple) or any(
        not isinstance(value, DatasetCapabilityBinding) for value in capabilities
    ):
        raise ValueError("capabilities must be a tuple of exact capability bindings")
    binding_ids = tuple(value.binding_id for value in capabilities)
    if tuple(sorted(set(binding_ids))) != binding_ids:
        raise ValueError("capabilities must have sorted unique binding IDs")
    kinds = tuple(value.kind for value in capabilities)
    if len(set(kinds)) != len(kinds) or set(kinds) != set(required_kinds):
        raise ValueError("capabilities do not bind the exact required kinds")
    if any(
        value.registry_key != _CAPABILITY_REGISTRY_KEYS[value.kind]
        for value in capabilities
    ):
        raise ValueError("capabilities do not bind the exact Glenn registry keys")


@dataclass(frozen=True, slots=True)
class GlennRetainedComparisonInputs(CanonicalRecord):
    """Caller-frozen read-set and comparison bytes; no historical defaults."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-retained-comparison-inputs'
    read_set_sha256: str
    observations_sha256: str

    def __post_init__(self) -> None:
        validate_sha256(self.read_set_sha256)
        validate_sha256(self.observations_sha256)


@dataclass(frozen=True, slots=True)
class GlennRegistrationManifestDependencies:
    """Exact identities supplied by the trusted registration composition root."""

    source_storage_root: ObjectIdentity
    control_storage_root: ObjectIdentity
    complete_release_selector_sha256: str
    capabilities: tuple[DatasetCapabilityBinding, ...]

    def __post_init__(self) -> None:
        _require_root(self.source_storage_root, field_name="source_storage_root")
        _require_root(self.control_storage_root, field_name="control_storage_root")
        validate_sha256(
            self.complete_release_selector_sha256,
            field_name="complete_release_selector_sha256",
        )
        _require_capabilities(
            self.capabilities,
            required_kinds=_REGISTRATION_CAPABILITY_KINDS,
        )


@dataclass(frozen=True, slots=True)
class GlennTransformationManifestDependencies:
    """Exact identities supplied after the archive registration has completed."""

    historical_inputs: GlennRetainedComparisonInputs
    source_storage_root: ObjectIdentity
    selector_storage_root: ObjectIdentity
    destination_storage_root: ObjectIdentity
    historical_storage_root: ObjectIdentity
    control_storage_root: ObjectIdentity
    archive_materialization: ObjectIdentity
    historical_observations_artifact: ObjectIdentity
    historical_complete_selector_sha256: str
    capabilities: tuple[DatasetCapabilityBinding, ...]
    expected_output_physical_sha256: str | None = None
    expected_output_byte_size: int | None = None

    def __post_init__(self) -> None:
        for field_name, identity in (
            ("source_storage_root", self.source_storage_root),
            ("selector_storage_root", self.selector_storage_root),
            ("destination_storage_root", self.destination_storage_root),
            ("historical_storage_root", self.historical_storage_root),
            ("control_storage_root", self.control_storage_root),
        ):
            _require_root(identity, field_name=field_name)
        _require_identity_schema(
            self.archive_materialization,
            field_name="archive_materialization",
            expected_schema=DatasetMaterialization.SCHEMA,
        )
        if self.archive_materialization.object_id != _ARCHIVE_MATERIALIZATION_ID:
            raise ValueError(
                "archive_materialization has the wrong Glenn materialization ID"
            )
        _require_identity_schema(
            self.historical_observations_artifact,
            field_name="historical_observations_artifact",
            expected_schema=ArtifactIdentity.SCHEMA,
        )
        validate_sha256(
            self.historical_complete_selector_sha256,
            field_name="historical_complete_selector_sha256",
        )
        _require_capabilities(
            self.capabilities,
            required_kinds=_TRANSFORMATION_CAPABILITY_KINDS,
        )
        output_identity_parts = (
            self.expected_output_physical_sha256 is not None,
            self.expected_output_byte_size is not None,
        )
        if any(output_identity_parts) and not all(output_identity_parts):
            raise ValueError(
                "expected output digest and byte size must be supplied together"
            )
        if self.expected_output_physical_sha256 is not None:
            validate_sha256(
                self.expected_output_physical_sha256,
                field_name="expected_output_physical_sha256",
            )
        if self.expected_output_byte_size is not None and (
            isinstance(self.expected_output_byte_size, bool)
            or not isinstance(self.expected_output_byte_size, int)
            or not 1 <= self.expected_output_byte_size <= _MAXIMUM_OUTPUT_BYTES
        ):
            raise ValueError(
                "expected_output_byte_size is outside the Glenn output bound"
            )


def _scope(
    scope_id: str,
    storage_root: ObjectIdentity,
    relative_prefix: str,
) -> DatasetStorageScope:
    return DatasetStorageScope(
        scope_id=scope_id,
        storage_root=storage_root,
        relative_prefix=relative_prefix,
    )


def _selector(
    selector_id: str,
    *,
    selector_sha256: str,
    complete_release: bool,
) -> DatasetSelectorRef:
    return DatasetSelectorRef(
        selector_id=selector_id,
        kind=(
            DatasetSelectorKind.COMPLETE_RELEASE
            if complete_release
            else DatasetSelectorKind.MANIFEST
        ),
        selector_schema='empirical-lawhood/planning/dataset-member-selector',
        selector_sha256=selector_sha256,
        complete_release=complete_release,
    )


def _invariants(
    rows: tuple[tuple[str, DatasetInvariantKind, tuple[str, ...]], ...],
) -> tuple[DatasetExpectedInvariant, ...]:
    values = tuple(
        DatasetExpectedInvariant(
            invariant_id=invariant_id,
            kind=kind,
            subject_ids=tuple(sorted(subject_ids)),
            checker_registry_key=(
                f"dataset.check.{kind.value.lower().replace('_', '-')}"
            ),
            required=True,
        )
        for invariant_id, kind, subject_ids in rows
    )
    return tuple(sorted(values, key=lambda value: value.invariant_id))


def _registration_work_envelope() -> DatasetWorkEnvelope:
    return DatasetWorkEnvelope(
        resources=ResourceBudget(
            cpu_cores=4,
            memory_bytes=4 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=3600,
            source_scan_bytes=GLENN_ARCHIVE_SIZE_BYTES,
            output_bytes=0,
        ),
        files=DatasetFileLimits(
            max_source_files=1,
            max_destination_files=0,
            max_archive_members=GLENN_2026_PROFILE.limits.maximum_members,
            max_single_file_bytes=GLENN_ARCHIVE_SIZE_BYTES,
        ),
        control=DatasetControlLimits(
            max_manifest_bytes=1024**2,
            max_metadata_records=4096,
            max_receipt_bytes=1024**2,
            max_reason_codes=64,
        ),
    )


def _transformation_work_envelope() -> DatasetWorkEnvelope:
    return DatasetWorkEnvelope(
        resources=ResourceBudget(
            cpu_cores=4,
            memory_bytes=4 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=3600,
            source_scan_bytes=(
                GLENN_ARCHIVE_SIZE_BYTES
                + GLENN_READ_SET_SIZE_BYTES
                + GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES
            ),
            output_bytes=_MAXIMUM_OUTPUT_BYTES,
        ),
        files=DatasetFileLimits(
            max_source_files=3,
            max_destination_files=1,
            max_archive_members=GLENN_2026_PROFILE.limits.maximum_members,
            max_single_file_bytes=GLENN_ARCHIVE_SIZE_BYTES,
        ),
        control=DatasetControlLimits(
            max_manifest_bytes=1024**2,
            max_metadata_records=4096,
            max_receipt_bytes=1024**2,
            max_reason_codes=64,
        ),
    )


def _family() -> DatasetFamilyInput:
    return DatasetFamilyInput(
        family_id=_FAMILY_ID,
        canonical_name="Generalized Data-driven Gradient-based Learning",
        provider_id="provider.zenodo",
        external_identifiers=(),
        description="Glenn deposited laser-water-sheet response archive.",
        keywords=("driven-dissipative", "glenn", "plasma-response"),
    )


def _release() -> DatasetReleaseInput:
    return DatasetReleaseInput(
        release_id=_RELEASE_ID,
        family_id=_FAMILY_ID,
        expected_resolution_class=ReleaseResolutionClass.PROVIDER_IMMUTABLE_RESOLVED,
        expected_access_state=AccessState.PUBLIC,
        provider_id="provider.zenodo",
        external_identifiers=(
            ExternalIdentifier(
                kind=ExternalIdentifierKind.DEPOSIT,
                namespace="zenodo",
                value="17163053",
            ),
        ),
        expected_publication_at_utc=None,
        expected_retrieved_at_utc=None,
        expected_mutable_snapshot=False,
        expected_format_profile_ids=("zip.safe-member-inventory",),
        expected_file_count_minimum=1,
        expected_file_count_maximum=1,
        expected_byte_count_minimum=GLENN_ARCHIVE_SIZE_BYTES,
        expected_byte_count_maximum=GLENN_ARCHIVE_SIZE_BYTES,
        expected_physical_sha256=GLENN_ARCHIVE_SHA256,
        expected_manifest_sha256=None,
        expected_licence_id="cc-by-4.0",
        expected_access_terms_id="public",
        expected_redistribution_allowed=True,
    )


def build_glenn_registration_manifest(
    dependencies: GlennRegistrationManifestDependencies,
) -> DatasetRegistrationManifest:
    """Build the exact no-download registration request for the held archive."""

    if not isinstance(dependencies, GlennRegistrationManifestDependencies):
        raise TypeError("dependencies must be GlennRegistrationManifestDependencies")
    return DatasetRegistrationManifest(
        manifest_id="manifest.glenn-registration",
        family=_family(),
        release=_release(),
        proposed_materialization_id=_ARCHIVE_MATERIALIZATION_ID,
        source_scope=_scope(
            "scope.glenn-source",
            dependencies.source_storage_root,
            GLENN_ARCHIVE_RELATIVE_LOCATOR,
        ),
        control_write_scope=_scope(
            "scope.glenn-registration-control",
            dependencies.control_storage_root,
            "dataset-operations/glenn-registration",
        ),
        selector=_selector(
            "selector.glenn-complete-release",
            selector_sha256=dependencies.complete_release_selector_sha256,
            complete_release=True,
        ),
        expected_physical_sha256=GLENN_ARCHIVE_SHA256,
        expected_byte_size=GLENN_ARCHIVE_SIZE_BYTES,
        expected_file_count=1,
        expected_media_type="application/zip",
        expected_format_profile_ids=("zip.safe-member-inventory",),
        materialization_class=DatasetMaterializationClass.AUTHORITATIVE_SOURCE,
        evidence_class=DatasetEvidenceClass.EMPIRICAL_SOURCE,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        capabilities=dependencies.capabilities,
        expected_invariants=_invariants(
            (
                (
                    "invariant.no-download",
                    DatasetInvariantKind.NO_DOWNLOAD,
                    (_ARCHIVE_MATERIALIZATION_ID,),
                ),
                (
                    "invariant.source-bytes-immutable",
                    DatasetInvariantKind.SOURCE_BYTES_IMMUTABLE,
                    (_ARCHIVE_MATERIALIZATION_ID,),
                ),
            )
        ),
        work_envelope=_registration_work_envelope(),
        exception_codes=(
            DatasetAuthoringExceptionCode.HISTORICAL_RETRIEVAL_TIME_UNAVAILABLE,
            DatasetAuthoringExceptionCode.RELEASE_TIMESTAMP_UNAVAILABLE,
        ),
    )


def _selected_members() -> tuple[DatasetSelectorMemberContract, ...]:
    return tuple(
        DatasetSelectorMemberContract(
            member_id=member.member_id,
            relative_locator=member.relative_locator,
            expected_physical_sha256=member.expected_physical_sha256,
            expected_compressed_size_bytes=member.expected_compressed_size_bytes,
            expected_uncompressed_size_bytes=member.expected_uncompressed_size_bytes,
            expected_crc32=member.expected_crc32,
            media_type=member.media_type,
            format_profile_id=member.format_profile_id,
            inspection_policy_id=member.inspection_policy_id,
        )
        for member in GLENN_2026_PROFILE.members
    )


def _field_id(column_name: str) -> str:
    return column_name.replace("_", "-")


def _field(
    column_name: str,
    *,
    role: DatasetFieldRole,
    causal_phase: CausalPhase,
    data_type: DatasetFieldDataType,
    nullable: bool = False,
    outcome_access: OutcomeAccess = OutcomeAccess.OUTCOME_BLIND,
    visibility_ceiling: VisibilityCeiling = VisibilityCeiling.PROSPECTIVE,
) -> DatasetFieldContract:
    quantity_role = role in {
        DatasetFieldRole.SOURCE_TIME,
        DatasetFieldRole.REQUESTED_ACTION,
        DatasetFieldRole.ACCEPTED_ACTION,
        DatasetFieldRole.APPLIED_ACTION,
        DatasetFieldRole.REALIZED_ACTION,
        DatasetFieldRole.RESPONSE,
        DatasetFieldRole.RECEIVER,
        DatasetFieldRole.COVARIATE,
        DatasetFieldRole.UNCERTAINTY,
    }
    if column_name in ACTION_COLUMNS:
        unit = "source-native-command"
    elif column_name in {"focal_r50_um", "prior_focal_r50_um"}:
        unit = "um"
    elif column_name == "acquisition_order":
        unit = "burst-index"
    elif quantity_role:
        unit = "source-native"
    else:
        unit = None
    return DatasetFieldContract(
        field_id=_field_id(column_name),
        column_name=column_name,
        column_index=CANONICAL_COLUMNS.index(column_name),
        data_type=data_type,
        role=role,
        unit=unit,
        frame="glenn-source-frame" if quantity_role else None,
        clock_id="glenn-burst-clock" if quantity_role else None,
        causal_phase=causal_phase,
        outcome_access=outcome_access,
        visibility_ceiling=visibility_ceiling,
        nullable=nullable,
    )


def _fields() -> tuple[DatasetFieldContract, ...]:
    fields = {
        "run": _field(
            "run",
            role=DatasetFieldRole.BATCH,
            causal_phase=CausalPhase.PREPARATION,
            data_type=DatasetFieldDataType.STRING,
        ),
        "burst": _field(
            "burst",
            role=DatasetFieldRole.INDEPENDENT_UNIT,
            causal_phase=CausalPhase.PREPARATION,
            data_type=DatasetFieldDataType.INT64,
        ),
        "acquisition_order": _field(
            "acquisition_order",
            role=DatasetFieldRole.SOURCE_TIME,
            causal_phase=CausalPhase.PRE_ACTION,
            data_type=DatasetFieldDataType.INT64,
        ),
        "prior_fitness": _field(
            "prior_fitness",
            role=DatasetFieldRole.COVARIATE,
            causal_phase=CausalPhase.PRE_ACTION,
            data_type=DatasetFieldDataType.FLOAT64,
            nullable=True,
        ),
        "prior_fitness_missing": _field(
            "prior_fitness_missing",
            role=DatasetFieldRole.VALIDITY,
            causal_phase=CausalPhase.PRE_ACTION,
            data_type=DatasetFieldDataType.BOOLEAN,
        ),
        "prior_focal_r50_um": _field(
            "prior_focal_r50_um",
            role=DatasetFieldRole.COVARIATE,
            causal_phase=CausalPhase.PRE_ACTION,
            data_type=DatasetFieldDataType.FLOAT64,
            nullable=True,
        ),
        "prior_focal_r50_missing": _field(
            "prior_focal_r50_missing",
            role=DatasetFieldRole.VALIDITY,
            causal_phase=CausalPhase.PRE_ACTION,
            data_type=DatasetFieldDataType.BOOLEAN,
        ),
    }
    for column_name in ACTION_COLUMNS:
        fields[column_name] = _field(
            column_name,
            role=DatasetFieldRole.REQUESTED_ACTION,
            causal_phase=CausalPhase.ACTION_REQUESTED,
            data_type=DatasetFieldDataType.FLOAT64,
        )
    for column_name in NUISANCE_COLUMNS:
        fields[column_name] = _field(
            column_name,
            role=DatasetFieldRole.COVARIATE,
            causal_phase=CausalPhase.PRE_ACTION,
            data_type=DatasetFieldDataType.FLOAT64,
        )
    fields["fitness"] = _field(
        "fitness",
        role=DatasetFieldRole.RECEIVER,
        causal_phase=CausalPhase.RECEIVER,
        data_type=DatasetFieldDataType.FLOAT64,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    fields["fitness_error"] = _field(
        "fitness_error",
        role=DatasetFieldRole.UNCERTAINTY,
        causal_phase=CausalPhase.RECEIVER,
        data_type=DatasetFieldDataType.FLOAT64,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    fields["focal_r50_um"] = _field(
        "focal_r50_um",
        role=DatasetFieldRole.RESPONSE,
        causal_phase=CausalPhase.RECEIVER,
        data_type=DatasetFieldDataType.FLOAT64,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    if set(fields) != set(CANONICAL_COLUMNS) or len(fields) != len(CANONICAL_COLUMNS):
        raise RuntimeError("Glenn manifest field registry drifted from canonical order")
    return tuple(sorted(fields.values(), key=lambda value: value.field_id))


def _split_contract() -> DatasetSplitContract:
    return DatasetSplitContract(
        split_id="split.glenn-burst",
        strategy_registry_key="dataset.split.independent-burst",
        independent_unit_field_ids=("burst",),
        group_field_ids=("run",),
        partition_ids=("development", "evidence-snapshot"),
        sealed_partition_ids=(),
        assignment_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        disjoint_by_independent_unit=True,
    )


def _causal_contract() -> DatasetCausalContract:
    return DatasetCausalContract(
        causal_contract_id="causal.glenn-burst",
        information_cutoff=InformationCutoff(
            cutoff_id="cutoff.glenn-pre-action",
            clock_id="glenn-burst-clock",
            phase=CausalPhase.ACTION_REQUESTED,
            coordinate=None,
            includes_coordinate=True,
        ),
        retained_history_field_ids=(
            "o2",
            "o3",
            "o4",
            "prior-fitness",
            "prior-fitness-missing",
            "prior-focal-r50-missing",
            "prior-focal-r50-um",
            "target-z",
        ),
        action_field_ids=tuple(sorted(_field_id(value) for value in ACTION_COLUMNS)),
        receiver_field_ids=("fitness", "focal-r50-um"),
        outcome_constructs_features=False,
        outcome_constructs_assignment=False,
    )


def _output_contract(
    dependencies: GlennTransformationManifestDependencies,
) -> DatasetOutputContract:
    exact_size = dependencies.expected_output_byte_size
    return DatasetOutputContract(
        output_id=_OUTPUT_ID,
        proposed_materialization_id=_OUTPUT_MATERIALIZATION_ID,
        role=DatasetOutputRole.CANONICAL_SOURCE,
        relative_locator="observations.parquet",
        output_schema=GLENN_OUTPUT_SCHEMA_ID,
        media_type="application/vnd.apache.parquet",
        format_profile_id=GLENN_PARQUET_PROFILE_ID,
        fields=_fields(),
        parent_materialization_ids=(_ARCHIVE_MATERIALIZATION_ID,),
        expected_physical_sha256=dependencies.expected_output_physical_sha256,
        expected_byte_count_minimum=exact_size if exact_size is not None else 1,
        expected_byte_count_maximum=(
            exact_size if exact_size is not None else _MAXIMUM_OUTPUT_BYTES
        ),
        expected_file_count=1,
        expected_row_count_minimum=GLENN_2026_PROFILE.expected_rows,
        expected_row_count_maximum=GLENN_2026_PROFILE.expected_rows,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def _normalizations() -> tuple[DatasetTypeNullNormalization, ...]:
    string_fields = {"run"}
    integer_fields = {"acquisition-order", "burst"}
    boolean_fields = {"prior-fitness-missing", "prior-focal-r50-missing"}
    nullable_fields = {"prior-fitness", "prior-focal-r50-um"}
    values = []
    for column_name in CANONICAL_COLUMNS:
        field_id = _field_id(column_name)
        output_type = (
            DatasetFieldDataType.STRING
            if field_id in string_fields
            else DatasetFieldDataType.INT64
            if field_id in integer_fields
            else DatasetFieldDataType.BOOLEAN
            if field_id in boolean_fields
            else DatasetFieldDataType.FLOAT64
        )
        comparison_type = (
            DatasetFieldDataType.STRING
            if field_id in string_fields
            else DatasetFieldDataType.INT64
            if field_id in integer_fields | boolean_fields
            else DatasetFieldDataType.FLOAT64
        )
        values.append(
            DatasetTypeNullNormalization(
                normalization_id=f"normalization.{field_id}",
                field_id=field_id,
                output_data_type=output_type,
                comparison_data_type=comparison_type,
                normalized_data_type=output_type,
                output_native_nulls=field_id in nullable_fields,
                comparison_native_nulls=False,
                output_null_tokens=(),
                comparison_null_tokens=(("",) if field_id in nullable_fields else ()),
                nulls_equal=True,
            )
        )
    return tuple(sorted(values, key=lambda value: value.normalization_id))


def _comparison_target(
    dependencies: GlennTransformationManifestDependencies,
) -> DatasetComparisonTarget:
    return DatasetComparisonTarget(
        comparison_id="comparison.glenn-historical-g1",
        output_id=_OUTPUT_ID,
        historical_artifact=dependencies.historical_observations_artifact,
        historical_scope=_scope(
            "scope.glenn-historical-comparison",
            dependencies.historical_storage_root,
            GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR,
        ),
        selector=_selector(
            "selector.glenn-historical-observations",
            selector_sha256=dependencies.historical_complete_selector_sha256,
            complete_release=True,
        ),
        expected_physical_sha256=dependencies.historical_inputs.observations_sha256,
        expected_byte_size=GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES,
        expected_media_type="text/csv",
        expected_format_profile_id="csv-rfc4180",
        expected_row_count=GLENN_2026_PROFILE.expected_rows,
        expected_column_count=len(CANONICAL_COLUMNS),
        type_null_normalizations=_normalizations(),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.UNBOUND_HISTORICAL,
    )


def _transformation_exceptions() -> tuple[DatasetAuthoringExceptionCode, ...]:
    return tuple(
        sorted(
            (
                DatasetAuthoringExceptionCode.FOCAL_R50_GENERATION_METHOD_UNAVAILABLE,
                DatasetAuthoringExceptionCode.FOCAL_R50_UNCERTAINTY_UNAVAILABLE,
                DatasetAuthoringExceptionCode.SOURCE_ACTION_LABEL_SEMANTICS_CAVEAT,
                DatasetAuthoringExceptionCode.UNSAFE_MEMBER_EXCLUDED,
            ),
            key=lambda value: value.value,
        )
    )


def build_glenn_transformation_manifest(
    dependencies: GlennTransformationManifestDependencies,
) -> DatasetTransformationManifest:
    """Build the parent-closed follow-up transform and comparison request.

    The returned parent set contains only the registered archive.  The frozen
    historical table remains a normalization-bound operational comparison.
    """

    if not isinstance(dependencies, GlennTransformationManifestDependencies):
        raise TypeError("dependencies must be GlennTransformationManifestDependencies")
    return DatasetTransformationManifest(
        manifest_id="manifest.glenn-transformation",
        release_id=_RELEASE_ID,
        inputs=(
            DatasetTransformInput(
                input_id="input.glenn-archive",
                materialization=dependencies.archive_materialization,
                scope=_scope(
                    "scope.glenn-transform-source",
                    dependencies.source_storage_root,
                    GLENN_ARCHIVE_RELATIVE_LOCATOR,
                ),
                selector=_selector(
                    "selector.glenn-safe-four-members",
                    selector_sha256=dependencies.historical_inputs.read_set_sha256,
                    complete_release=False,
                ),
                selector_manifest_scope=_scope(
                    "scope.glenn-source-read-set",
                    dependencies.selector_storage_root,
                    GLENN_READ_SET_RELATIVE_LOCATOR,
                ),
                selector_manifest_byte_size=GLENN_READ_SET_SIZE_BYTES,
                selected_members=_selected_members(),
            ),
        ),
        destination_scope=_scope(
            "scope.glenn-transform-destination",
            dependencies.destination_storage_root,
            "data/transformed/glenn-2026-g1",
        ),
        control_write_scope=_scope(
            "scope.glenn-transform-control",
            dependencies.control_storage_root,
            "dataset-operations/glenn-transformation",
        ),
        capabilities=dependencies.capabilities,
        outputs=(_output_contract(dependencies),),
        comparison_targets=(_comparison_target(dependencies),),
        split_contract=_split_contract(),
        causal_contract=_causal_contract(),
        expected_invariants=_invariants(
            (
                (
                    "invariant.action-clocks",
                    DatasetInvariantKind.ACTION_CLOCKS_DISTINCT,
                    tuple(_field_id(value) for value in ACTION_COLUMNS),
                ),
                (
                    "invariant.burst-independent-unit",
                    DatasetInvariantKind.INDEPENDENT_UNIT_PRESERVED,
                    ("burst",),
                ),
                (
                    "invariant.nested-shot-replication",
                    DatasetInvariantKind.NESTED_ROWS_DO_NOT_INFLATE_REPLICATION,
                    ("burst", "run"),
                ),
                (
                    "invariant.parent-closed",
                    DatasetInvariantKind.PARENT_CLOSED,
                    ("input.glenn-archive",),
                ),
                (
                    "invariant.safe-members",
                    DatasetInvariantKind.SAFE_MEMBERS_ONLY,
                    ("input.glenn-archive",),
                ),
                (
                    "invariant.units-frames-clocks",
                    DatasetInvariantKind.UNITS_FRAMES_CLOCKS_EXPLICIT,
                    ("fitness", "focal-r50-um", "o2", "target-z"),
                ),
            )
        ),
        work_envelope=_transformation_work_envelope(),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.UNBOUND_HISTORICAL,
        exception_codes=_transformation_exceptions(),
    )


__all__ = [
    "GLENN_ARCHIVE_FULL_SCAN_PASS_LIMIT",
    "GLENN_ARCHIVE_FULL_SCAN_READ_LIMIT_BYTES",
    "GLENN_ARCHIVE_RELATIVE_LOCATOR",
    "GLENN_ARCHIVE_SHA256",
    "GLENN_ARCHIVE_SIZE_BYTES",
    "GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR",
    "GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES",
    "GLENN_READ_SET_RELATIVE_LOCATOR",
    "GLENN_READ_SET_SIZE_BYTES",
    "GlennRegistrationManifestDependencies",
    "GlennTransformationManifestDependencies",
    "build_glenn_registration_manifest",
    "build_glenn_transformation_manifest",
]
