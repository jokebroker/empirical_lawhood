"""Exact preparation boundary for publishing the held Glenn reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import PurePosixPath
from typing import ClassVar, Final

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.planning.dataset_manifests import (
    DatasetCapabilityKind,
    DatasetTransformationManifest,
)
from empirical_lawhood.planning.datasets import (
    DatasetEvidenceClass,
    DatasetMaterializationClass,
    DatasetMaterializationVerificationSubject,
    DatasetSelectorKind,
    DatasetSelectorRef,
    EvidenceReferenceKind,
    LogicalContentIdentity,
    RegisteredImplementationIdentity,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.dataset_io import DatasetSourceOpener
from empirical_lawhood.runtime.dataset_transformation import (
    DatasetTransformationSupportingRecord,
    PreparedDatasetTransformation,
)
from empirical_lawhood.runtime.datasets import DatasetTransformRegistration

from .comparison import GlennLogicalEquivalence
from .contracts import (
    ACTION_COLUMNS,
    CANONICAL_COLUMNS,
    GLENN_2026_PROFILE,
    NUISANCE_COLUMNS,
)
from .manifests import GlennRetainedComparisonInputs, GLENN_ARCHIVE_RELATIVE_LOCATOR, GLENN_ARCHIVE_SIZE_BYTES, GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR, GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES, GLENN_READ_SET_RELATIVE_LOCATOR, GLENN_READ_SET_SIZE_BYTES
from .registry import GLENN_CAPABILITY_VERSION, GLENN_TRANSFORM_REGISTRY_KEY
from .reproduction import (
    GlennReproductionPreparationRequest,
    prepare_glenn_reproduction,
)
from .source import GlennArchiveAudit
from .transform import (
    GLENN_LOGICAL_PROFILE_ID,
    GLENN_OUTPUT_SCHEMA_ID,
    GLENN_PARQUET_PROFILE_ID,
    GlennParquetProfile,
    GlennTransformResult,
)


GLENN_TRANSFORMATION_VERIFICATION_POLICY_ID: Final = (
    "verification-policy.glenn-transformation"
)
GLENN_TRANSFORMATION_AUDIT_ID: Final = "audit.glenn-transformation"
GLENN_TRANSFORMATION_ARCHIVE_AUDIT_ID: Final = "audit.glenn-transformation-archive"
GLENN_TRANSFORMATION_EQUIVALENCE_ID: Final = "equivalence.glenn-historical-g1"
GLENN_TRANSFORMATION_WRITER_PROFILE_ID: Final = "profile.glenn-parquet-writer"


@dataclass(frozen=True, slots=True)
class GlennTransformationVerificationPolicy(CanonicalRecord):
    """Exact verifier contract for the one Glenn follow-up output."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/glenn/glenn-transformation-verification-policy'
    )

    policy_id: str
    manifest: ObjectIdentity
    transform_registration: ObjectIdentity
    transformer: RegisteredImplementationIdentity
    output_id: str
    output_schema_id: str
    output_format_profile_id: str
    output_media_type: str
    expected_row_count: int
    expected_column_count: int
    logical_profile_id: str
    writer_profile: ObjectIdentity
    comparison_is_non_lineage: bool
    scientific_verdict_allowed: bool

    def __post_init__(self) -> None:
        if self.policy_id != GLENN_TRANSFORMATION_VERIFICATION_POLICY_ID:
            raise ValueError(
                "Glenn transformation verification policy has another identity"
            )
        if (
            self.manifest.object_schema != DatasetTransformationManifest.SCHEMA
            or self.transform_registration.object_id != GLENN_TRANSFORM_REGISTRY_KEY
            or self.transform_registration.object_schema
            != DatasetTransformRegistration.SCHEMA
        ):
            raise ValueError(
                "Glenn verification policy has another manifest/registration type"
            )
        if not isinstance(self.transformer, RegisteredImplementationIdentity):
            raise TypeError("Glenn verification policy transformer is invalid")
        if (
            self.transformer.implementation_key != GLENN_TRANSFORM_REGISTRY_KEY
            or self.transformer.implementation_version != GLENN_CAPABILITY_VERSION
            or self.output_schema_id != GLENN_OUTPUT_SCHEMA_ID
            or self.output_format_profile_id != GLENN_PARQUET_PROFILE_ID
            or self.output_media_type != "application/vnd.apache.parquet"
            or self.expected_row_count != GLENN_2026_PROFILE.expected_rows
            or self.expected_column_count != len(CANONICAL_COLUMNS)
            or self.logical_profile_id != GLENN_LOGICAL_PROFILE_ID
        ):
            raise ValueError(
                "Glenn verification policy differs from the fixed output contract"
            )
        if not self.comparison_is_non_lineage or self.scientific_verdict_allowed:
            raise ValueError(
                "Glenn verification policy cannot promote comparison or science"
            )


@dataclass(frozen=True, slots=True)
class GlennTransformationAudit(CanonicalRecord):
    """Canonical operational audit derived from the in-memory transform result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-transformation-audit'

    manifest: ObjectIdentity
    archive_audit: ObjectIdentity
    output_schema_id: str
    output_physical_sha256: str
    output_logical_sha256: str
    output_size_bytes: int
    row_count: int
    column_count: int
    exception_codes: tuple[str, ...]
    action_columns: tuple[str, ...]
    nuisance_covariate_columns: tuple[str, ...]
    action_label_caveat: str
    focal_native_to_um: Decimal
    focal_is_post_command_mediator: bool
    focal_generation_method_available: bool
    focal_uncertainty_available: bool
    deposited_model_log_role: str
    writer_profile: ObjectIdentity
    scientific_verdict_issued: bool

    def __post_init__(self) -> None:
        if self.manifest.object_schema != DatasetTransformationManifest.SCHEMA:
            raise ValueError("Glenn transformation audit binds another manifest type")
        if self.archive_audit.object_schema != GlennArchiveAudit.SCHEMA:
            raise ValueError(
                "Glenn transformation audit binds another archive audit type"
            )
        if (
            self.output_schema_id != GLENN_OUTPUT_SCHEMA_ID
            or self.row_count != GLENN_2026_PROFILE.expected_rows
            or self.column_count != len(CANONICAL_COLUMNS)
            or self.action_columns != ACTION_COLUMNS
            or self.nuisance_covariate_columns != NUISANCE_COLUMNS
        ):
            raise ValueError(
                "Glenn transformation audit differs from the fixed table contract"
            )
        validate_sha256(
            self.output_physical_sha256,
            field_name="output_physical_sha256",
        )
        validate_sha256(
            self.output_logical_sha256,
            field_name="output_logical_sha256",
        )
        if self.output_size_bytes <= 0:
            raise ValueError("Glenn transformation audit output bytes must be positive")
        if (
            self.focal_generation_method_available
            or self.focal_uncertainty_available
            or not self.focal_is_post_command_mediator
            or self.scientific_verdict_issued
        ):
            raise ValueError(
                "Glenn transformation audit overstates source metadata or science"
            )


def _transformer_identity(
    registration: DatasetTransformRegistration,
) -> RegisteredImplementationIdentity:
    return RegisteredImplementationIdentity(
        implementation_key=registration.capability_key,
        implementation_version=registration.capability_version,
        implementation_sha256=registration.implementation_sha256,
    )


def _verification_policy(
    *,
    manifest: DatasetTransformationManifest,
    registration: DatasetTransformRegistration,
    writer_profile: GlennParquetProfile,
) -> GlennTransformationVerificationPolicy:
    output = manifest.outputs[0]
    return GlennTransformationVerificationPolicy(
        policy_id=GLENN_TRANSFORMATION_VERIFICATION_POLICY_ID,
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        transform_registration=ObjectIdentity.from_record(
            registration.capability_key,
            registration,
        ),
        transformer=_transformer_identity(registration),
        output_id=output.output_id,
        output_schema_id=output.output_schema,
        output_format_profile_id=output.format_profile_id,
        output_media_type=output.media_type,
        expected_row_count=GLENN_2026_PROFILE.expected_rows,
        expected_column_count=len(CANONICAL_COLUMNS),
        logical_profile_id=GLENN_LOGICAL_PROFILE_ID,
        writer_profile=ObjectIdentity.from_record(
            GLENN_TRANSFORMATION_WRITER_PROFILE_ID,
            writer_profile,
        ),
        comparison_is_non_lineage=True,
        scientific_verdict_allowed=False,
    )


def _transformation_audit(
    *,
    manifest: DatasetTransformationManifest,
    transformed: GlennTransformResult,
    physical_sha256: str,
    size_bytes: int,
    writer_profile: GlennParquetProfile,
) -> GlennTransformationAudit:
    return GlennTransformationAudit(
        manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        archive_audit=ObjectIdentity.from_record(
            GLENN_TRANSFORMATION_ARCHIVE_AUDIT_ID,
            transformed.archive_audit,
        ),
        output_schema_id=transformed.output_schema_id,
        output_physical_sha256=physical_sha256,
        output_logical_sha256=transformed.logical_sha256,
        output_size_bytes=size_bytes,
        row_count=transformed.table.num_rows,
        column_count=transformed.table.num_columns,
        exception_codes=tuple(
            sorted(value.value for value in transformed.exception_codes)
        ),
        action_columns=transformed.action_columns,
        nuisance_covariate_columns=transformed.nuisance_covariate_columns,
        action_label_caveat=transformed.action_label_caveat,
        focal_native_to_um=Decimal(str(transformed.focal_native_to_um)),
        focal_is_post_command_mediator=transformed.focal_is_post_command_mediator,
        focal_generation_method_available=transformed.focal_generation_method_available,
        focal_uncertainty_available=transformed.focal_uncertainty_available,
        deposited_model_log_role=transformed.deposited_model_log_role,
        writer_profile=ObjectIdentity.from_record(
            GLENN_TRANSFORMATION_WRITER_PROFILE_ID,
            writer_profile,
        ),
        scientific_verdict_issued=False,
    )


class GlennTransformationPreparer:
    """Run only the fixed held-source Glenn transformation in memory."""

    def __init__(
        self,
        *,
        manifest: DatasetTransformationManifest,
        registration: DatasetTransformRegistration,
        source_opener: DatasetSourceOpener,
    ) -> None:
        if not isinstance(manifest, DatasetTransformationManifest):
            raise TypeError("manifest must be a DatasetTransformationManifest")
        if not isinstance(registration, DatasetTransformRegistration):
            raise TypeError("registration must be a DatasetTransformRegistration")
        self._validate_static_contract(manifest, registration)
        self._manifest = manifest
        self._registration = registration
        self._source_opener = source_opener

    @staticmethod
    def _validate_static_contract(
        manifest: DatasetTransformationManifest,
        registration: DatasetTransformRegistration,
    ) -> None:
        if (
            manifest.manifest_id != "manifest.glenn-transformation"
            or len(manifest.inputs) != 1
            or len(manifest.outputs) != 1
            or len(manifest.comparison_targets) != 1
        ):
            raise ValueError(
                "Glenn transformation requires its exact one-input/output/comparison"
            )
        transform_input = manifest.inputs[0]
        output = manifest.outputs[0]
        comparison = manifest.comparison_targets[0]
        if (
            transform_input.scope.relative_prefix != GLENN_ARCHIVE_RELATIVE_LOCATOR
            or transform_input.selector_manifest_scope is None
            or transform_input.selector_manifest_scope.relative_prefix
            != GLENN_READ_SET_RELATIVE_LOCATOR
            or comparison.historical_scope.relative_prefix
            != GLENN_HISTORICAL_OBSERVATIONS_RELATIVE_LOCATOR
            or output.output_schema != GLENN_OUTPUT_SCHEMA_ID
            or output.format_profile_id != GLENN_PARQUET_PROFILE_ID
            or output.media_type != "application/vnd.apache.parquet"
            or output.expected_file_count != 1
            or output.expected_row_count_minimum != GLENN_2026_PROFILE.expected_rows
            or output.expected_row_count_maximum != GLENN_2026_PROFILE.expected_rows
        ):
            raise ValueError(
                "Glenn transformation manifest differs from fixed held inputs/output"
            )
        transform_bindings = tuple(
            value
            for value in manifest.capabilities
            if value.kind is DatasetCapabilityKind.TRANSFORM
        )
        if (
            registration.capability_key != GLENN_TRANSFORM_REGISTRY_KEY
            or registration.capability_version != GLENN_CAPABILITY_VERSION
            or registration.produced_media_types != (output.media_type,)
            or registration.destination_format_profile_ids
            != (output.format_profile_id,)
            or len(transform_bindings) != 1
            or transform_bindings[0].registry_key != registration.capability_key
            or transform_bindings[0].implementation
            != ObjectIdentity.from_record(
                registration.capability_key, registration.capability
            )
        ):
            raise ValueError(
                "Glenn transformation binds another registered implementation"
            )

    def prepare_transformation(
        self,
        request: ObjectIdentity,
        *,
        trusted_at_utc: str,
    ) -> PreparedDatasetTransformation:
        manifest_identity = ObjectIdentity.from_record(
            self._manifest.manifest_id,
            self._manifest,
        )
        if request != manifest_identity:
            raise ValueError("Glenn transformation request identifies another manifest")
        transform_input = self._manifest.inputs[0]
        assert transform_input.selector_manifest_scope is not None
        comparison = self._manifest.comparison_targets[0]
        prepared = prepare_glenn_reproduction(
            GlennReproductionPreparationRequest(
                historical_inputs=GlennRetainedComparisonInputs(
                    transform_input.selector.selector_sha256,
                    comparison.expected_physical_sha256,
                ),
                source_scope=transform_input.scope,
                selector_scope=transform_input.selector_manifest_scope,
                historical_scope=comparison.historical_scope,
                maximum_distinct_input_bytes=(
                    GLENN_ARCHIVE_SIZE_BYTES
                    + GLENN_READ_SET_SIZE_BYTES
                    + GLENN_HISTORICAL_OBSERVATIONS_SIZE_BYTES
                ),
                trusted_at_utc=trusted_at_utc,
            ),
            source_opener=self._source_opener,
        )
        output = self._manifest.outputs[0]
        artifact = prepared.artifact
        if (
            output.expected_byte_count_minimum is None
            or output.expected_byte_count_maximum is None
        ):
            raise ValueError("Glenn transformation requires bounded output bytes")
        if (
            artifact.row_count != GLENN_2026_PROFILE.expected_rows
            or artifact.column_count != len(CANONICAL_COLUMNS)
            or artifact.media_type != output.media_type
            or artifact.profile.profile_id != output.format_profile_id
            or output.expected_physical_sha256 not in {None, artifact.physical_sha256}
            or not (
                output.expected_byte_count_minimum
                <= artifact.size_bytes
                <= output.expected_byte_count_maximum
            )
        ):
            raise ValueError(
                "prepared Glenn artifact differs from its output expectations"
            )
        transformer = _transformer_identity(self._registration)
        selector = DatasetSelectorRef(
            selector_id="selector.glenn-g1-observations",
            kind=DatasetSelectorKind.MANIFEST,
            selector_schema=DatasetTransformationManifest.SCHEMA,
            selector_sha256=self._manifest.fingerprint(),
            complete_release=False,
        )
        relative_locator = PurePosixPath(
            self._manifest.destination_scope.relative_prefix,
            output.relative_locator,
        ).as_posix()
        subject = DatasetMaterializationVerificationSubject(
            materialization_id=output.proposed_materialization_id,
            release_id=self._manifest.release_id,
            materialization_class=DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
            evidence_class=DatasetEvidenceClass.DERIVED,
            outcome_access=output.outcome_access,
            storage_root_id=self._manifest.destination_scope.storage_root.object_id,
            relative_locator=relative_locator,
            selector=selector,
            physical_sha256=artifact.physical_sha256,
            byte_size=artifact.size_bytes,
            file_count=1,
            media_type=artifact.media_type,
            format_profile_id=artifact.profile.profile_id,
            logical_identity=LogicalContentIdentity(
                decoder=transformer,
                logical_schema=GLENN_OUTPUT_SCHEMA_ID,
                logical_sha256=artifact.logical_sha256,
            ),
        )
        policy = _verification_policy(
            manifest=self._manifest,
            registration=self._registration,
            writer_profile=artifact.profile,
        )
        audit = _transformation_audit(
            manifest=self._manifest,
            transformed=prepared.transform_audit,
            physical_sha256=artifact.physical_sha256,
            size_bytes=artifact.size_bytes,
            writer_profile=artifact.profile,
        )
        equivalence: GlennLogicalEquivalence = prepared.equivalence
        records = (
            DatasetTransformationSupportingRecord(
                record_id=GLENN_TRANSFORMATION_ARCHIVE_AUDIT_ID,
                record=prepared.transform_audit.archive_audit,
                evidence_kind=EvidenceReferenceKind.VERIFICATION,
            ),
            DatasetTransformationSupportingRecord(
                record_id=GLENN_TRANSFORMATION_AUDIT_ID,
                record=audit,
                evidence_kind=EvidenceReferenceKind.TRANSFORM,
            ),
            DatasetTransformationSupportingRecord(
                record_id=GLENN_TRANSFORMATION_EQUIVALENCE_ID,
                record=equivalence,
                evidence_kind=EvidenceReferenceKind.VERIFICATION,
            ),
            DatasetTransformationSupportingRecord(
                record_id=GLENN_TRANSFORMATION_WRITER_PROFILE_ID,
                record=artifact.profile,
                evidence_kind=EvidenceReferenceKind.MANIFEST,
            ),
        )
        return PreparedDatasetTransformation(
            manifest=manifest_identity,
            transformer=transformer,
            verification_policy_id=policy.policy_id,
            verification_policy_record=policy,
            output_id=output.output_id,
            artifact_profile=ArtifactProfile.PARQUET,
            payload=artifact.payload,
            subject=subject,
            row_count=artifact.row_count,
            column_count=artifact.column_count,
            source_guard_receipts=tuple(
                sorted(
                    (
                        prepared.source_guard_receipt,
                        prepared.selector_guard_receipt,
                        prepared.historical_guard_receipt,
                    ),
                    key=lambda value: value.guard_receipt_id,
                )
            ),
            supporting_records=tuple(
                sorted(records, key=lambda value: value.record_id)
            ),
            network_bytes=0,
            source_mutated=False,
            scientific_verdict_issued=False,
        )


__all__ = [
    "GLENN_TRANSFORMATION_ARCHIVE_AUDIT_ID",
    "GLENN_TRANSFORMATION_AUDIT_ID",
    "GLENN_TRANSFORMATION_EQUIVALENCE_ID",
    "GLENN_TRANSFORMATION_VERIFICATION_POLICY_ID",
    "GLENN_TRANSFORMATION_WRITER_PROFILE_ID",
    "GlennTransformationAudit",
    "GlennTransformationPreparer",
    "GlennTransformationVerificationPolicy",
]
