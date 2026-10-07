"""Strict expectation-only authoring roots for dataset operations.

These records can request registration, transformation, or a proposed binding.
They cannot assert observed custody, verification, receipts, results, scientific
evidence rungs, admission, or controller claims.
"""

from __future__ import annotations

import re
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff, parse_utc_timestamp

from .dataset_authority import DatasetStorageScope, DatasetWorkEnvelope
from .datasets import (
    AccessState,
    DatasetBindingRole,
    DatasetEvidenceClass,
    DatasetMaterialization,
    DatasetMaterializationClass,
    DatasetSelectorRef,
    ExperimentDatasetBinding,
    ExternalIdentifier,
    ExternalIdentifierKind,
    ReleaseResolutionClass,
)


_MAX_ITEMS = 256
MAX_DATASET_SELECTOR_MEMBERS: Final[int] = 10_000
_MAX_CAPABILITIES = 8
_MAX_EXCEPTION_CODES = 64
_MAX_TEXT_BYTES = 4096
_MAX_EXPECTED_BYTES = 1024**5
_MAX_EXPECTED_FILES = 10_000_000
_MAX_EXPECTED_ROWS = 10**12
_GLOB_METACHARACTERS = frozenset("*?[]{}")
_CRC32 = re.compile(r"^[0-9a-f]{8}$")
_CAPABILITY_MANIFEST_SCHEMA = 'empirical-lawhood/runtime/capability-manifest'
_CAPABILITY_CONFIG_REF_SCHEMA = 'empirical-lawhood/runtime/capability-config-ref'
_DATASET_EVIDENCE_VERIFIER_REGISTRATION_SCHEMA = (
    'empirical-lawhood/runtime/dataset-evidence-verifier-registration'
)
_EXPERIMENT_SPEC_SCHEMA = 'empirical-lawhood/kernel/experiment-spec'
EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE = "experiment-dataset-bindings"


def _validate_id(value: str, *, field_name: str) -> None:
    validate_stable_id(value, field_name=field_name)
    if len(value.encode("utf-8")) > 128:
        raise ValueError(f"{field_name} exceeds its byte limit")


def _validate_text(value: str, *, field_name: str) -> None:
    validate_nonempty(value, field_name=field_name)
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{field_name} exceeds its byte limit")


def _validate_literal_locator(value: str, *, field_name: str) -> None:
    validate_relative_locator(value)
    if any(character in value for character in _GLOB_METACHARACTERS):
        raise ValueError(f"{field_name} cannot contain glob metacharacters")
    if "://" in value:
        raise ValueError(f"{field_name} cannot be a URL")


def _validate_scope(value: DatasetStorageScope, *, field_name: str) -> None:
    if not isinstance(value, DatasetStorageScope):
        raise ValueError(f"{field_name} must be a DatasetStorageScope")
    _validate_literal_locator(value.relative_prefix, field_name=field_name)


def _validate_object_identity(value: ObjectIdentity, *, field_name: str) -> None:
    if not isinstance(value, ObjectIdentity):
        raise ValueError(f"{field_name} must be an exact ObjectIdentity")


def _validate_object_identity_schema(
    value: ObjectIdentity,
    *,
    field_name: str,
    expected_schema: str,
) -> None:
    _validate_object_identity(value, field_name=field_name)
    if value.object_schema != expected_schema:
        raise ValueError(f"{field_name} object schema must be {expected_schema!r}")


def _validate_sorted_ids(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple) or len(values) > _MAX_ITEMS:
        raise ValueError(f"{field_name} must be a bounded tuple")
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
    for value in values:
        _validate_id(value, field_name=field_name)


def _validate_sorted_records(
    values: tuple[CanonicalRecord, ...],
    *,
    attribute: str,
    field_name: str,
    expected_type: type[CanonicalRecord],
    allow_empty: bool = False,
    maximum: int = _MAX_ITEMS,
) -> None:
    if not isinstance(values, tuple) or len(values) > maximum:
        raise ValueError(f"{field_name} must be a bounded tuple")
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, expected_type) for value in values):
        raise ValueError(f"{field_name} contains an invalid record type")
    require_sorted_unique_ids(values, attribute=attribute, field_name=field_name)


def _validate_external_identifiers(
    values: tuple[ExternalIdentifier, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(values, tuple) or len(values) > _MAX_ITEMS:
        raise ValueError(f"{field_name} must be a bounded tuple")
    if any(not isinstance(value, ExternalIdentifier) for value in values):
        raise ValueError(f"{field_name} contains an invalid external identifier")
    keys = tuple((value.kind.value, value.namespace, value.value) for value in values)
    if tuple(sorted(set(keys))) != keys:
        raise ValueError(f"{field_name} must be sorted and unique")
    if any("://" in value.value for value in values):
        raise ValueError(f"{field_name} cannot contain URL-valued identifiers")


def _validate_expected_count(
    value: int | None,
    *,
    field_name: str,
    maximum: int,
    allow_zero: bool = False,
) -> None:
    if value is None:
        return
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field_name} is outside its expectation bound")


def _validate_visibility(
    outcome_access: OutcomeAccess,
    visibility_ceiling: VisibilityCeiling,
) -> None:
    if not isinstance(outcome_access, OutcomeAccess):
        raise ValueError("outcome_access must be an OutcomeAccess")
    if not isinstance(visibility_ceiling, VisibilityCeiling):
        raise ValueError("visibility_ceiling must be a VisibilityCeiling")
    required = inherited_visibility((), outcome_access)
    if not visibility_ceiling.is_at_least_as_restrictive_as(required):
        raise ValueError("visibility ceiling cannot be lower than outcome access")


_OUTCOME_ACCESS_INCLUDES: Mapping[OutcomeAccess, frozenset[OutcomeAccess]] = MappingProxyType(
    {
        OutcomeAccess.OUTCOME_BLIND: frozenset({OutcomeAccess.OUTCOME_BLIND}),
        OutcomeAccess.DEVELOPMENT_VISIBLE: frozenset(
            {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.DEVELOPMENT_VISIBLE}
        ),
        OutcomeAccess.EVALUATION_SEALED: frozenset(
            {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.EVALUATION_SEALED}
        ),
        OutcomeAccess.EVALUATOR_REVEAL: frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            }
        ),
        OutcomeAccess.EVALUATION_REVEALED: frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ),
        OutcomeAccess.PRIVILEGED_TRUTH: frozenset(OutcomeAccess),
    }
)


def _require_outcome_access_includes(
    aggregate: OutcomeAccess,
    contained: tuple[OutcomeAccess, ...],
    *,
    message: str,
) -> None:
    if any(value not in _OUTCOME_ACCESS_INCLUDES[aggregate] for value in contained):
        raise ValueError(message)


def _validate_materialization_classification(
    materialization_class: DatasetMaterializationClass,
    evidence_class: DatasetEvidenceClass,
) -> None:
    if not isinstance(materialization_class, DatasetMaterializationClass):
        raise ValueError("materialization_class must be a DatasetMaterializationClass")
    if not isinstance(evidence_class, DatasetEvidenceClass):
        raise ValueError("evidence_class must be a DatasetEvidenceClass")
    if (
        materialization_class
        in {
            DatasetMaterializationClass.HISTORICAL_DERIVATIVE,
            DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
        }
        and evidence_class is not DatasetEvidenceClass.DERIVED
    ):
        raise ValueError("derivative materializations must retain DERIVED evidence class")
    if (
        materialization_class is DatasetMaterializationClass.METADATA_OBSERVATION
        and evidence_class is not DatasetEvidenceClass.METADATA_ONLY
    ):
        raise ValueError("metadata observations must retain METADATA_ONLY evidence class")
    if (
        materialization_class is DatasetMaterializationClass.SOFTWARE_RUNTIME_MEDIUM
        and evidence_class is not DatasetEvidenceClass.SOFTWARE
    ):
        raise ValueError("software materializations must retain SOFTWARE evidence class")
    if (
        materialization_class is DatasetMaterializationClass.NON_AUTHORITATIVE_WORKING_COPY
        and evidence_class is not DatasetEvidenceClass.NON_AUTHORITATIVE
    ):
        raise ValueError("working copies must retain NON_AUTHORITATIVE evidence class")
    if materialization_class is DatasetMaterializationClass.GENERATED_OBSERVATION and (
        evidence_class
        not in {
            DatasetEvidenceClass.ANALYTIC_REFERENCE,
            DatasetEvidenceClass.SIMULATION,
            DatasetEvidenceClass.HIL,
            DatasetEvidenceClass.PHYSICAL_EXPERIMENT,
            DatasetEvidenceClass.GENERATED_TRUTH,
        }
    ):
        raise ValueError("generated observations cannot masquerade as source or derivative")
    if materialization_class is DatasetMaterializationClass.AUTHORITATIVE_SOURCE and (
        evidence_class
        in {
            DatasetEvidenceClass.DERIVED,
            DatasetEvidenceClass.METADATA_ONLY,
            DatasetEvidenceClass.SOFTWARE,
            DatasetEvidenceClass.NON_AUTHORITATIVE,
        }
    ):
        raise ValueError("authoritative source has an incompatible evidence class")


class DatasetCapabilityKind(StrEnum):
    INSPECTOR = "INSPECTOR"
    EVIDENCE_VERIFIER = "EVIDENCE_VERIFIER"
    TRANSFORM = "TRANSFORM"
    CONFIG = "CONFIG"
    RUNTIME = "RUNTIME"
    ENVIRONMENT = "ENVIRONMENT"
    BINDING_COMPILER = "BINDING_COMPILER"


@dataclass(frozen=True, slots=True)
class DatasetCapabilityBinding(CanonicalRecord):
    """Static registry key bound to an exact immutable implementation record."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-capability-binding'

    binding_id: str
    kind: DatasetCapabilityKind
    registry_key: str
    implementation: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_id(self.binding_id, field_name="binding_id")
        if not isinstance(self.kind, DatasetCapabilityKind):
            raise ValueError("kind must be a DatasetCapabilityKind")
        _validate_id(self.registry_key, field_name="registry_key")
        expected_schema = {
            DatasetCapabilityKind.INSPECTOR: _CAPABILITY_MANIFEST_SCHEMA,
            DatasetCapabilityKind.EVIDENCE_VERIFIER: (
                _DATASET_EVIDENCE_VERIFIER_REGISTRATION_SCHEMA
            ),
            DatasetCapabilityKind.TRANSFORM: _CAPABILITY_MANIFEST_SCHEMA,
            DatasetCapabilityKind.BINDING_COMPILER: _CAPABILITY_MANIFEST_SCHEMA,
            DatasetCapabilityKind.CONFIG: _CAPABILITY_CONFIG_REF_SCHEMA,
            DatasetCapabilityKind.RUNTIME: ArtifactIdentity.SCHEMA,
            DatasetCapabilityKind.ENVIRONMENT: ArtifactIdentity.SCHEMA,
        }[self.kind]
        _validate_object_identity_schema(
            self.implementation,
            field_name=f"{self.kind.value.lower().replace('_', '-')} identity",
            expected_schema=expected_schema,
        )
        if self.implementation.object_id != self.registry_key:
            raise ValueError("capability binding registry key and object identity differ")


def _validate_capabilities(
    values: tuple[DatasetCapabilityBinding, ...],
    *,
    required: frozenset[DatasetCapabilityKind],
) -> None:
    if not isinstance(values, tuple) or not values or len(values) > _MAX_CAPABILITIES:
        raise ValueError("capabilities must be a bounded nonempty tuple")
    if any(not isinstance(value, DatasetCapabilityBinding) for value in values):
        raise ValueError("capabilities contains an invalid record type")
    binding_ids = tuple(value.binding_id for value in values)
    if tuple(sorted(set(binding_ids))) != binding_ids:
        raise ValueError("capabilities must have sorted unique binding IDs")
    kinds = tuple(value.kind for value in values)
    if len(set(kinds)) != len(kinds):
        raise ValueError("capabilities cannot repeat a capability kind")
    if set(kinds) != set(required):
        raise ValueError("capabilities do not bind the exact required kinds")


class DatasetAuthoringExceptionCode(StrEnum):
    RELEASE_ID_UNRESOLVED = "RELEASE_ID_UNRESOLVED"
    RELEASE_TIMESTAMP_UNAVAILABLE = "RELEASE_TIMESTAMP_UNAVAILABLE"
    HISTORICAL_RETRIEVAL_TIME_UNAVAILABLE = "HISTORICAL_RETRIEVAL_TIME_UNAVAILABLE"
    UPSTREAM_CHECKSUM_UNAVAILABLE = "UPSTREAM_CHECKSUM_UNAVAILABLE"
    HTTP_VALIDATOR_UNAVAILABLE = "HTTP_VALIDATOR_UNAVAILABLE"
    LICENCE_UNSPECIFIED = "LICENCE_UNSPECIFIED"
    LICENCE_NOASSERTION = "LICENCE_NOASSERTION"
    HISTORICAL_MANIFEST_SEMANTICS_PARTIAL = "HISTORICAL_MANIFEST_SEMANTICS_PARTIAL"
    TRANSFORM_LINEAGE_PARTIAL = "TRANSFORM_LINEAGE_PARTIAL"
    STRUCTURAL_CONTRACT_UNPROVEN = "STRUCTURAL_CONTRACT_UNPROVEN"
    PARTIAL_SELECTOR_ONLY = "PARTIAL_SELECTOR_ONLY"
    PARTIAL_QUARANTINED = "PARTIAL_QUARANTINED"
    METADATA_ONLY_NO_SOURCE_BYTES = "METADATA_ONLY_NO_SOURCE_BYTES"
    MISSING_EXTERNAL_DEPENDENCY = "MISSING_EXTERNAL_DEPENDENCY"
    UPSTREAM_COMPRESSED_PARENT_UNAVAILABLE = "UPSTREAM_COMPRESSED_PARENT_UNAVAILABLE"
    SEALED_VERIFICATION_AUTHORITY_REQUIRED = "SEALED_VERIFICATION_AUTHORITY_REQUIRED"
    UNSAFE_MEMBER_EXCLUDED = "UNSAFE_MEMBER_EXCLUDED"
    AUXILIARY_SOFTWARE_NOT_USED = "AUXILIARY_SOFTWARE_NOT_USED"
    LOCATOR_MISSING = "LOCATOR_MISSING"
    FOCAL_R50_GENERATION_METHOD_UNAVAILABLE = "FOCAL_R50_GENERATION_METHOD_UNAVAILABLE"
    FOCAL_R50_UNCERTAINTY_UNAVAILABLE = "FOCAL_R50_UNCERTAINTY_UNAVAILABLE"
    SOURCE_ACTION_LABEL_SEMANTICS_CAVEAT = "SOURCE_ACTION_LABEL_SEMANTICS_CAVEAT"


def _validate_exception_codes(values: tuple[DatasetAuthoringExceptionCode, ...]) -> None:
    if not isinstance(values, tuple) or len(values) > _MAX_EXCEPTION_CODES:
        raise ValueError("exception_codes must be a bounded tuple")
    if any(not isinstance(value, DatasetAuthoringExceptionCode) for value in values):
        raise ValueError("exception_codes contains an invalid code")
    encoded = tuple(value.value for value in values)
    if tuple(sorted(set(encoded))) != encoded:
        raise ValueError("exception_codes must be sorted and unique")


@dataclass(frozen=True, slots=True)
class DatasetFamilyInput(CanonicalRecord):
    """Expectation-only family metadata supplied to a trusted resolver."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-family-input'

    family_id: str
    canonical_name: str
    provider_id: str | None
    external_identifiers: tuple[ExternalIdentifier, ...]
    description: str
    keywords: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id(self.family_id, field_name="family_id")
        _validate_text(self.canonical_name, field_name="canonical_name")
        if self.provider_id is not None:
            _validate_id(self.provider_id, field_name="provider_id")
        _validate_external_identifiers(
            self.external_identifiers,
            field_name="external_identifiers",
        )
        if self.description:
            _validate_text(self.description, field_name="description")
        elif not isinstance(self.description, str):
            raise ValueError("description must be text")
        if not isinstance(self.keywords, tuple) or len(self.keywords) > _MAX_ITEMS:
            raise ValueError("keywords must be a bounded tuple")
        require_sorted_unique_strings(self.keywords, field_name="keywords")
        for keyword in self.keywords:
            _validate_text(keyword, field_name="keywords")


_IMMUTABLE_RELEASE_IDENTIFIER_KINDS = frozenset(
    {
        ExternalIdentifierKind.PROVIDER_RELEASE,
        ExternalIdentifierKind.ACCESSION,
        ExternalIdentifierKind.DEPOSIT,
        ExternalIdentifierKind.REPOSITORY_COMMIT,
        ExternalIdentifierKind.IMMUTABLE_SNAPSHOT,
    }
)
_PROVIDER_OBJECT_IDENTIFIER_KINDS = frozenset(
    {
        ExternalIdentifierKind.PROVIDER_COLLECTION,
        ExternalIdentifierKind.PROVIDER_OBJECT,
    }
)


@dataclass(frozen=True, slots=True)
class DatasetReleaseInput(CanonicalRecord):
    """Expected release metadata; it does not assert trusted resolution or custody."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-release-input'

    release_id: str
    family_id: str
    expected_resolution_class: ReleaseResolutionClass
    expected_access_state: AccessState
    provider_id: str | None
    external_identifiers: tuple[ExternalIdentifier, ...]
    expected_publication_at_utc: str | None
    expected_retrieved_at_utc: str | None
    expected_mutable_snapshot: bool
    expected_format_profile_ids: tuple[str, ...]
    expected_file_count_minimum: int | None
    expected_file_count_maximum: int | None
    expected_byte_count_minimum: int | None
    expected_byte_count_maximum: int | None
    expected_physical_sha256: str | None
    expected_manifest_sha256: str | None
    expected_licence_id: str | None
    expected_access_terms_id: str | None
    expected_redistribution_allowed: bool | None

    def __post_init__(self) -> None:
        _validate_id(self.release_id, field_name="release_id")
        _validate_id(self.family_id, field_name="family_id")
        if not isinstance(self.expected_resolution_class, ReleaseResolutionClass):
            raise ValueError("expected_resolution_class must be a ReleaseResolutionClass")
        if not isinstance(self.expected_access_state, AccessState):
            raise ValueError("expected_access_state must be an AccessState")
        if self.provider_id is not None:
            _validate_id(self.provider_id, field_name="provider_id")
        _validate_external_identifiers(
            self.external_identifiers,
            field_name="external_identifiers",
        )
        published = (
            parse_utc_timestamp(
                self.expected_publication_at_utc,
                field_name="expected_publication_at_utc",
            )
            if self.expected_publication_at_utc is not None
            else None
        )
        retrieved = (
            parse_utc_timestamp(
                self.expected_retrieved_at_utc,
                field_name="expected_retrieved_at_utc",
            )
            if self.expected_retrieved_at_utc is not None
            else None
        )
        if published is not None and retrieved is not None and published > retrieved:
            raise ValueError("expected publication cannot follow expected retrieval")
        if not isinstance(self.expected_mutable_snapshot, bool):
            raise ValueError("expected_mutable_snapshot must be boolean")
        _validate_sorted_ids(
            self.expected_format_profile_ids,
            field_name="expected_format_profile_ids",
        )
        _validate_expected_range(
            self.expected_file_count_minimum,
            self.expected_file_count_maximum,
            field_name="expected_file_count",
            absolute_maximum=_MAX_EXPECTED_FILES,
        )
        _validate_expected_range(
            self.expected_byte_count_minimum,
            self.expected_byte_count_maximum,
            field_name="expected_byte_count",
            absolute_maximum=_MAX_EXPECTED_BYTES,
        )
        if self.expected_physical_sha256 is not None:
            validate_sha256(
                self.expected_physical_sha256,
                field_name="expected_physical_sha256",
            )
        if self.expected_manifest_sha256 is not None:
            validate_sha256(
                self.expected_manifest_sha256,
                field_name="expected_manifest_sha256",
            )
        for field_name, value in (
            ("expected_licence_id", self.expected_licence_id),
            ("expected_access_terms_id", self.expected_access_terms_id),
        ):
            if value is not None:
                _validate_id(value, field_name=field_name)
        if self.expected_redistribution_allowed is not None and not isinstance(
            self.expected_redistribution_allowed,
            bool,
        ):
            raise ValueError("expected_redistribution_allowed must be boolean or null")
        self._validate_expected_release_shape()

    def _validate_expected_release_shape(self) -> None:
        immutable_ids = tuple(
            value
            for value in self.external_identifiers
            if value.kind in _IMMUTABLE_RELEASE_IDENTIFIER_KINDS
        )
        immutable_keys = tuple((value.kind, value.namespace) for value in immutable_ids)
        if len(set(immutable_keys)) != len(immutable_keys):
            raise ValueError(
                "expected release has conflicting immutable identifiers of one kind/namespace"
            )
        provider_objects = tuple(
            value
            for value in self.external_identifiers
            if value.kind in _PROVIDER_OBJECT_IDENTIFIER_KINDS
        )
        if self.expected_resolution_class is ReleaseResolutionClass.PROVIDER_IMMUTABLE_RESOLVED:
            if self.provider_id is None or not immutable_ids:
                raise ValueError("expected immutable release requires provider and immutable ID")
            if self.expected_mutable_snapshot:
                raise ValueError("expected immutable release cannot be mutable")
        elif self.expected_resolution_class is ReleaseResolutionClass.CAPTURED_SNAPSHOT_RESOLVED:
            exact_bytes = (
                self.expected_byte_count_minimum is not None
                and self.expected_byte_count_minimum == self.expected_byte_count_maximum
            )
            if (
                self.provider_id is None
                or not provider_objects
                or self.expected_retrieved_at_utc is None
                or self.expected_physical_sha256 is None
                or not exact_bytes
                or not self.expected_mutable_snapshot
            ):
                raise ValueError(
                    "expected captured snapshot requires provider object, retrieval time, "
                    "exact bytes, physical digest, and mutable-snapshot declaration"
                )
            if immutable_ids:
                raise ValueError("expected captured snapshot cannot claim immutable release ID")
        elif self.expected_resolution_class is ReleaseResolutionClass.UNRESOLVED_LOCAL_CUSTODY:
            if immutable_ids:
                raise ValueError("expected unresolved custody cannot retain immutable release ID")
        elif (
            self.provider_id is not None
            or self.external_identifiers
            or self.expected_mutable_snapshot
        ):
            raise ValueError("expected generated release cannot claim provider identity")


class DatasetInvariantKind(StrEnum):
    SOURCE_BYTES_IMMUTABLE = "SOURCE_BYTES_IMMUTABLE"
    NO_DOWNLOAD = "NO_DOWNLOAD"
    NO_OVERWRITE = "NO_OVERWRITE"
    SAFE_MEMBERS_ONLY = "SAFE_MEMBERS_ONLY"
    EXACT_OUTPUT_SCHEMA = "EXACT_OUTPUT_SCHEMA"
    PARENT_CLOSED = "PARENT_CLOSED"
    INDEPENDENT_UNIT_PRESERVED = "INDEPENDENT_UNIT_PRESERVED"
    NESTED_ROWS_DO_NOT_INFLATE_REPLICATION = "NESTED_ROWS_DO_NOT_INFLATE_REPLICATION"
    SPLIT_DISJOINT = "SPLIT_DISJOINT"
    CAUSAL_ORDER_PRESERVED = "CAUSAL_ORDER_PRESERVED"
    ACTION_CLOCKS_DISTINCT = "ACTION_CLOCKS_DISTINCT"
    UNITS_FRAMES_CLOCKS_EXPLICIT = "UNITS_FRAMES_CLOCKS_EXPLICIT"


@dataclass(frozen=True, slots=True)
class DatasetExpectedInvariant(CanonicalRecord):
    """A required predeclared check, never a caller assertion that it passed."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-expected-invariant'

    invariant_id: str
    kind: DatasetInvariantKind
    subject_ids: tuple[str, ...]
    checker_registry_key: str
    required: bool

    def __post_init__(self) -> None:
        _validate_id(self.invariant_id, field_name="invariant_id")
        if not isinstance(self.kind, DatasetInvariantKind):
            raise ValueError("kind must be a DatasetInvariantKind")
        _validate_sorted_ids(self.subject_ids, field_name="subject_ids")
        _validate_id(self.checker_registry_key, field_name="checker_registry_key")
        if not isinstance(self.required, bool) or not self.required:
            raise ValueError("expected invariants must be required, not caller-adjudicated")


def _validate_invariants(values: tuple[DatasetExpectedInvariant, ...]) -> None:
    _validate_sorted_records(
        values,
        attribute="invariant_id",
        field_name="expected_invariants",
        expected_type=DatasetExpectedInvariant,
    )


def _validate_invariant_subjects(
    values: tuple[DatasetExpectedInvariant, ...],
    *,
    available_subject_ids: frozenset[str],
) -> None:
    for invariant in values:
        if not set(invariant.subject_ids).issubset(available_subject_ids):
            raise ValueError("expected invariant references an undeclared subject")


class DatasetFieldRole(StrEnum):
    INDEPENDENT_UNIT = "INDEPENDENT_UNIT"
    NESTED_OBSERVATION = "NESTED_OBSERVATION"
    BATCH = "BATCH"
    SOURCE_TIME = "SOURCE_TIME"
    REQUESTED_ACTION = "REQUESTED_ACTION"
    ACCEPTED_ACTION = "ACCEPTED_ACTION"
    APPLIED_ACTION = "APPLIED_ACTION"
    REALIZED_ACTION = "REALIZED_ACTION"
    RESPONSE = "RESPONSE"
    RECEIVER = "RECEIVER"
    COVARIATE = "COVARIATE"
    VALIDITY = "VALIDITY"
    UNCERTAINTY = "UNCERTAINTY"
    SPLIT = "SPLIT"


class DatasetFieldDataType(StrEnum):
    STRING = "STRING"
    INT64 = "INT64"
    FLOAT64 = "FLOAT64"
    BOOLEAN = "BOOLEAN"
    UTC_TIMESTAMP = "UTC_TIMESTAMP"


_QUANTITY_ROLES = frozenset(
    {
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
)
_ACTION_PHASES: Mapping[DatasetFieldRole, CausalPhase] = MappingProxyType(
    {
        DatasetFieldRole.REQUESTED_ACTION: CausalPhase.ACTION_REQUESTED,
        DatasetFieldRole.ACCEPTED_ACTION: CausalPhase.ACTION_REQUESTED,
        DatasetFieldRole.APPLIED_ACTION: CausalPhase.ACTION_APPLIED,
        DatasetFieldRole.REALIZED_ACTION: CausalPhase.ACTION_APPLIED,
    }
)


@dataclass(frozen=True, slots=True)
class DatasetFieldContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-field-contract'

    field_id: str
    column_name: str
    column_index: int
    data_type: DatasetFieldDataType
    role: DatasetFieldRole
    unit: str | None
    frame: str | None
    clock_id: str | None
    causal_phase: CausalPhase
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    nullable: bool

    def __post_init__(self) -> None:
        _validate_id(self.field_id, field_name="field_id")
        _validate_text(self.column_name, field_name="column_name")
        _validate_expected_count(
            self.column_index,
            field_name="column_index",
            maximum=_MAX_ITEMS - 1,
            allow_zero=True,
        )
        if not isinstance(self.data_type, DatasetFieldDataType):
            raise ValueError("data_type must be a DatasetFieldDataType")
        if not isinstance(self.role, DatasetFieldRole):
            raise ValueError("role must be a DatasetFieldRole")
        if not isinstance(self.causal_phase, CausalPhase):
            raise ValueError("causal_phase must be a CausalPhase")
        for field_name, value in (("unit", self.unit), ("frame", self.frame)):
            if value is not None:
                _validate_text(value, field_name=field_name)
        if self.clock_id is not None:
            _validate_id(self.clock_id, field_name="clock_id")
        if self.role in _QUANTITY_ROLES and None in {self.unit, self.frame, self.clock_id}:
            raise ValueError("quantity fields require explicit unit, frame, and clock")
        if self.role not in _QUANTITY_ROLES and any(
            value is not None for value in (self.unit, self.frame, self.clock_id)
        ):
            raise ValueError("structural fields cannot masquerade as physical quantities")
        required_phase = _ACTION_PHASES.get(self.role)
        if required_phase is not None and self.causal_phase is not required_phase:
            raise ValueError("action field role and causal phase differ")
        if self.role in {
            DatasetFieldRole.RESPONSE,
            DatasetFieldRole.RECEIVER,
        } and self.causal_phase not in {
            CausalPhase.RECEIVER,
            CausalPhase.POST_OUTCOME,
        }:
            raise ValueError("response/receiver fields require a receiver or post-outcome phase")
        _validate_visibility(self.outcome_access, self.visibility_ceiling)
        if not isinstance(self.nullable, bool):
            raise ValueError("nullable must be boolean")


class DatasetOutputRole(StrEnum):
    CANONICAL_SOURCE = "CANONICAL_SOURCE"
    COMMAND_FITNESS = "COMMAND_FITNESS"
    RECEIVER_CHECKPOINT = "RECEIVER_CHECKPOINT"
    TRANSITION = "TRANSITION"
    FEATURE = "FEATURE"
    AUXILIARY = "AUXILIARY"


@dataclass(frozen=True, slots=True)
class DatasetOutputContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-output-contract'

    output_id: str
    proposed_materialization_id: str
    role: DatasetOutputRole
    relative_locator: str
    output_schema: str
    media_type: str
    format_profile_id: str
    fields: tuple[DatasetFieldContract, ...]
    parent_materialization_ids: tuple[str, ...]
    expected_physical_sha256: str | None
    expected_byte_count_minimum: int | None
    expected_byte_count_maximum: int | None
    expected_file_count: int
    expected_row_count_minimum: int | None
    expected_row_count_maximum: int | None
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        _validate_id(self.output_id, field_name="output_id")
        _validate_id(
            self.proposed_materialization_id,
            field_name="proposed_materialization_id",
        )
        if not isinstance(self.role, DatasetOutputRole):
            raise ValueError("role must be a DatasetOutputRole")
        _validate_literal_locator(self.relative_locator, field_name="relative_locator")
        validate_schema(self.output_schema)
        _validate_text(self.media_type, field_name="media_type")
        _validate_id(self.format_profile_id, field_name="format_profile_id")
        _validate_sorted_records(
            self.fields,
            attribute="field_id",
            field_name="fields",
            expected_type=DatasetFieldContract,
        )
        column_names = tuple(field.column_name for field in self.fields)
        if len(set(column_names)) != len(column_names):
            raise ValueError("output column names must be unique")
        column_indices = tuple(field.column_index for field in self.fields)
        if set(column_indices) != set(range(len(self.fields))):
            raise ValueError("output column indices must be unique and contiguous from zero")
        _validate_sorted_ids(
            self.parent_materialization_ids,
            field_name="parent_materialization_ids",
            allow_empty=self.role is DatasetOutputRole.CANONICAL_SOURCE,
        )
        if self.expected_physical_sha256 is not None:
            validate_sha256(
                self.expected_physical_sha256,
                field_name="expected_physical_sha256",
            )
        _validate_expected_range(
            self.expected_byte_count_minimum,
            self.expected_byte_count_maximum,
            field_name="expected_byte_count",
            absolute_maximum=_MAX_EXPECTED_BYTES,
        )
        _validate_expected_count(
            self.expected_file_count,
            field_name="expected_file_count",
            maximum=_MAX_EXPECTED_FILES,
        )
        _validate_expected_range(
            self.expected_row_count_minimum,
            self.expected_row_count_maximum,
            field_name="expected_row_count",
            absolute_maximum=_MAX_EXPECTED_ROWS,
        )
        _validate_visibility(self.outcome_access, self.visibility_ceiling)
        field_ceiling = VisibilityCeiling.most_restrictive(
            *(field.visibility_ceiling for field in self.fields)
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(field_ceiling):
            raise ValueError("output visibility cannot be lower than a field visibility")
        _require_outcome_access_includes(
            self.outcome_access,
            tuple(field.outcome_access for field in self.fields),
            message="output cannot understate field outcome access",
        )


@dataclass(frozen=True, slots=True)
class DatasetSelectorMemberContract(CanonicalRecord):
    """One exact, safely inspectable archive member selected without a glob."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-selector-member-contract'

    member_id: str
    relative_locator: str
    expected_physical_sha256: str
    expected_compressed_size_bytes: int
    expected_uncompressed_size_bytes: int
    expected_crc32: str
    media_type: str
    format_profile_id: str
    inspection_policy_id: str

    def __post_init__(self) -> None:
        _validate_id(self.member_id, field_name="member_id")
        _validate_literal_locator(self.relative_locator, field_name="relative_locator")
        validate_sha256(
            self.expected_physical_sha256,
            field_name="expected_physical_sha256",
        )
        _validate_expected_count(
            self.expected_compressed_size_bytes,
            field_name="expected_compressed_size_bytes",
            maximum=_MAX_EXPECTED_BYTES,
            allow_zero=True,
        )
        _validate_expected_count(
            self.expected_uncompressed_size_bytes,
            field_name="expected_uncompressed_size_bytes",
            maximum=_MAX_EXPECTED_BYTES,
            allow_zero=True,
        )
        if (
            not isinstance(self.expected_crc32, str)
            or _CRC32.fullmatch(self.expected_crc32) is None
        ):
            raise ValueError("expected_crc32 must be eight lowercase hexadecimal characters")
        _validate_text(self.media_type, field_name="media_type")
        _validate_id(self.format_profile_id, field_name="format_profile_id")
        _validate_id(self.inspection_policy_id, field_name="inspection_policy_id")


@dataclass(frozen=True, slots=True)
class DatasetDirectoryMemberExpectation(CanonicalRecord):
    """One exact regular file in a bounded, literal directory selector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-directory-member-expectation'

    member_id: str
    relative_locator: str
    expected_physical_sha256: str
    expected_size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _validate_id(self.member_id, field_name="member_id")
        _validate_literal_locator(self.relative_locator, field_name="relative_locator")
        validate_sha256(
            self.expected_physical_sha256,
            field_name="expected_physical_sha256",
        )
        _validate_expected_count(
            self.expected_size_bytes,
            field_name="expected_size_bytes",
            maximum=_MAX_EXPECTED_BYTES,
            allow_zero=True,
        )
        _validate_text(self.media_type, field_name="media_type")


def dataset_directory_content_sha256(
    members: tuple[DatasetDirectoryMemberExpectation, ...],
) -> str:
    """Digest only ordered locator/size/content facts, not selector prose IDs."""

    return hashlib.sha256(
        canonical_json_bytes(
            tuple(
                {
                    "relative_locator": value.relative_locator,
                    "size_bytes": value.expected_size_bytes,
                    "sha256": value.expected_physical_sha256,
                }
                for value in members
            )
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetDirectorySelectorManifest(CanonicalRecord):
    """Exact allowlist for a directory-backed release or generated bundle."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-directory-selector-manifest'

    selector_id: str
    subject_id: str
    complete_subject: bool
    members: tuple[DatasetDirectoryMemberExpectation, ...]
    expected_content_sha256: str
    expected_total_size_bytes: int

    def __post_init__(self) -> None:
        _validate_id(self.selector_id, field_name="selector_id")
        _validate_id(self.subject_id, field_name="subject_id")
        if not isinstance(self.complete_subject, bool):
            raise ValueError("complete_subject must be boolean")
        _validate_sorted_records(
            self.members,
            attribute="member_id",
            field_name="members",
            expected_type=DatasetDirectoryMemberExpectation,
            maximum=MAX_DATASET_SELECTOR_MEMBERS,
        )
        locators = tuple(value.relative_locator for value in self.members)
        if len(set(locators)) != len(locators):
            raise ValueError("directory selector cannot repeat a locator")
        validate_sha256(
            self.expected_content_sha256,
            field_name="expected_content_sha256",
        )
        expected_size = sum(value.expected_size_bytes for value in self.members)
        _validate_expected_count(
            self.expected_total_size_bytes,
            field_name="expected_total_size_bytes",
            maximum=_MAX_EXPECTED_BYTES,
        )
        if self.expected_total_size_bytes != expected_size:
            raise ValueError("directory selector total size differs from its members")
        if self.expected_content_sha256 != dataset_directory_content_sha256(self.members):
            raise ValueError("directory selector content digest differs from its members")


@dataclass(frozen=True, slots=True)
class DatasetTypeNullNormalization(CanonicalRecord):
    """Predeclared type and null normalization for one comparison field."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-type-null-normalization'

    normalization_id: str
    field_id: str
    output_data_type: DatasetFieldDataType
    comparison_data_type: DatasetFieldDataType
    normalized_data_type: DatasetFieldDataType
    output_native_nulls: bool
    comparison_native_nulls: bool
    output_null_tokens: tuple[str, ...]
    comparison_null_tokens: tuple[str, ...]
    nulls_equal: bool

    def __post_init__(self) -> None:
        _validate_id(self.normalization_id, field_name="normalization_id")
        _validate_id(self.field_id, field_name="field_id")
        for type_field_name, data_type in (
            ("output_data_type", self.output_data_type),
            ("comparison_data_type", self.comparison_data_type),
            ("normalized_data_type", self.normalized_data_type),
        ):
            if not isinstance(data_type, DatasetFieldDataType):
                raise ValueError(f"{type_field_name} must be a DatasetFieldDataType")
        for token_field_name, null_tokens in (
            ("output_null_tokens", self.output_null_tokens),
            ("comparison_null_tokens", self.comparison_null_tokens),
        ):
            if not isinstance(null_tokens, tuple) or len(null_tokens) > 16:
                raise ValueError(f"{token_field_name} must be a bounded tuple")
            if any(not isinstance(token, str) for token in null_tokens):
                raise ValueError(f"{token_field_name} must contain text tokens")
            if tuple(sorted(set(null_tokens))) != null_tokens:
                raise ValueError(f"{token_field_name} must be sorted and unique")
            for token in null_tokens:
                if len(token.encode("utf-8")) > 64:
                    raise ValueError(f"{token_field_name} contains an oversized token")
        for boolean_field_name, boolean_value in (
            ("output_native_nulls", self.output_native_nulls),
            ("comparison_native_nulls", self.comparison_native_nulls),
        ):
            if not isinstance(boolean_value, bool):
                raise ValueError(f"{boolean_field_name} must be boolean")
        if not isinstance(self.nulls_equal, bool) or not self.nulls_equal:
            raise ValueError("comparison null equivalence must be explicitly enabled")


@dataclass(frozen=True, slots=True)
class DatasetComparisonTarget(CanonicalRecord):
    """Comparison-only historical bytes; never a transformation lineage parent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-comparison-target'

    comparison_id: str
    output_id: str
    historical_artifact: ObjectIdentity
    historical_scope: DatasetStorageScope
    selector: DatasetSelectorRef
    expected_physical_sha256: str
    expected_byte_size: int
    expected_media_type: str
    expected_format_profile_id: str
    expected_row_count: int
    expected_column_count: int
    type_null_normalizations: tuple[DatasetTypeNullNormalization, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        _validate_id(self.comparison_id, field_name="comparison_id")
        _validate_id(self.output_id, field_name="output_id")
        _validate_object_identity_schema(
            self.historical_artifact,
            field_name="historical artifact",
            expected_schema=ArtifactIdentity.SCHEMA,
        )
        _validate_scope(self.historical_scope, field_name="historical_scope")
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        if not self.selector.complete_release:
            raise ValueError("comparison target selector must cover the complete artifact")
        validate_sha256(
            self.expected_physical_sha256,
            field_name="expected_physical_sha256",
        )
        _validate_expected_count(
            self.expected_byte_size,
            field_name="expected_byte_size",
            maximum=_MAX_EXPECTED_BYTES,
        )
        _validate_text(self.expected_media_type, field_name="expected_media_type")
        _validate_id(self.expected_format_profile_id, field_name="expected_format_profile_id")
        _validate_expected_count(
            self.expected_row_count,
            field_name="expected_row_count",
            maximum=_MAX_EXPECTED_ROWS,
        )
        _validate_expected_count(
            self.expected_column_count,
            field_name="expected_column_count",
            maximum=_MAX_ITEMS,
        )
        _validate_sorted_records(
            self.type_null_normalizations,
            attribute="normalization_id",
            field_name="type_null_normalizations",
            expected_type=DatasetTypeNullNormalization,
        )
        normalized_fields = tuple(value.field_id for value in self.type_null_normalizations)
        if len(set(normalized_fields)) != len(normalized_fields):
            raise ValueError("comparison normalizations cannot repeat a field")
        if len(normalized_fields) != self.expected_column_count:
            raise ValueError("comparison column count must equal its normalization contract")
        _validate_visibility(self.outcome_access, self.visibility_ceiling)


def _validate_expected_range(
    minimum: int | None,
    maximum: int | None,
    *,
    field_name: str,
    absolute_maximum: int,
) -> None:
    if (minimum is None) != (maximum is None):
        raise ValueError(f"{field_name} bounds must be supplied together")
    if minimum is None or maximum is None:
        return
    _validate_expected_count(
        minimum,
        field_name=f"{field_name}_minimum",
        maximum=absolute_maximum,
        allow_zero=True,
    )
    _validate_expected_count(
        maximum,
        field_name=f"{field_name}_maximum",
        maximum=absolute_maximum,
        allow_zero=True,
    )
    if minimum > maximum:
        raise ValueError(f"{field_name} minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class DatasetSplitContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-split-contract'

    split_id: str
    strategy_registry_key: str
    independent_unit_field_ids: tuple[str, ...]
    group_field_ids: tuple[str, ...]
    partition_ids: tuple[str, ...]
    sealed_partition_ids: tuple[str, ...]
    assignment_outcome_access: OutcomeAccess
    disjoint_by_independent_unit: bool

    def __post_init__(self) -> None:
        _validate_id(self.split_id, field_name="split_id")
        _validate_id(self.strategy_registry_key, field_name="strategy_registry_key")
        _validate_sorted_ids(
            self.independent_unit_field_ids,
            field_name="independent_unit_field_ids",
        )
        _validate_sorted_ids(self.group_field_ids, field_name="group_field_ids", allow_empty=True)
        _validate_sorted_ids(self.partition_ids, field_name="partition_ids")
        _validate_sorted_ids(
            self.sealed_partition_ids,
            field_name="sealed_partition_ids",
            allow_empty=True,
        )
        if not set(self.sealed_partition_ids).issubset(self.partition_ids):
            raise ValueError("sealed partitions must be declared partitions")
        if self.assignment_outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("split assignment must be outcome-blind")
        if not isinstance(self.disjoint_by_independent_unit, bool) or not (
            self.disjoint_by_independent_unit
        ):
            raise ValueError("splits must be disjoint by physical independent unit")


@dataclass(frozen=True, slots=True)
class DatasetCausalContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-causal-contract'

    causal_contract_id: str
    information_cutoff: InformationCutoff
    retained_history_field_ids: tuple[str, ...]
    action_field_ids: tuple[str, ...]
    receiver_field_ids: tuple[str, ...]
    outcome_constructs_features: bool
    outcome_constructs_assignment: bool

    def __post_init__(self) -> None:
        _validate_id(self.causal_contract_id, field_name="causal_contract_id")
        if not isinstance(self.information_cutoff, InformationCutoff):
            raise ValueError("information_cutoff must be an InformationCutoff")
        if not self.information_cutoff.phase.precedes_or_equals(CausalPhase.ACTION_REQUESTED):
            raise ValueError("dataset causal cutoff cannot follow the requested action")
        _validate_sorted_ids(
            self.retained_history_field_ids,
            field_name="retained_history_field_ids",
            allow_empty=True,
        )
        _validate_sorted_ids(self.action_field_ids, field_name="action_field_ids")
        _validate_sorted_ids(self.receiver_field_ids, field_name="receiver_field_ids")
        groups = (
            set(self.retained_history_field_ids),
            set(self.action_field_ids),
            set(self.receiver_field_ids),
        )
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("history, action, and receiver fields must be distinct")
        for field_name, value in (
            ("outcome_constructs_features", self.outcome_constructs_features),
            ("outcome_constructs_assignment", self.outcome_constructs_assignment),
        ):
            if not isinstance(value, bool) or value:
                raise ValueError(f"{field_name} must be false")


_REGISTRATION_CAPABILITIES = frozenset(
    {
        DatasetCapabilityKind.INSPECTOR,
        DatasetCapabilityKind.EVIDENCE_VERIFIER,
        DatasetCapabilityKind.CONFIG,
        DatasetCapabilityKind.RUNTIME,
        DatasetCapabilityKind.ENVIRONMENT,
    }
)
_TRANSFORMATION_CAPABILITIES = _REGISTRATION_CAPABILITIES | frozenset(
    {DatasetCapabilityKind.TRANSFORM}
)
_BINDING_CAPABILITIES = frozenset(
    {
        DatasetCapabilityKind.BINDING_COMPILER,
        DatasetCapabilityKind.CONFIG,
        DatasetCapabilityKind.RUNTIME,
        DatasetCapabilityKind.ENVIRONMENT,
    }
)


@dataclass(frozen=True, slots=True)
class DatasetRegistrationManifest(CanonicalRecord):
    """Expectation-only registration of selected bytes already held."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-registration-manifest'

    manifest_id: str
    family: DatasetFamilyInput
    release: DatasetReleaseInput
    proposed_materialization_id: str
    source_scope: DatasetStorageScope
    control_write_scope: DatasetStorageScope
    selector: DatasetSelectorRef
    expected_physical_sha256: str
    expected_byte_size: int
    expected_file_count: int
    expected_media_type: str
    expected_format_profile_ids: tuple[str, ...]
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    capabilities: tuple[DatasetCapabilityBinding, ...]
    expected_invariants: tuple[DatasetExpectedInvariant, ...]
    work_envelope: DatasetWorkEnvelope
    exception_codes: tuple[DatasetAuthoringExceptionCode, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("manifest_id", self.manifest_id),
            ("proposed_materialization_id", self.proposed_materialization_id),
        ):
            _validate_id(value, field_name=field_name)
        if not isinstance(self.family, DatasetFamilyInput):
            raise ValueError("family must be a DatasetFamilyInput")
        if not isinstance(self.release, DatasetReleaseInput):
            raise ValueError("release must be a DatasetReleaseInput")
        if self.release.family_id != self.family.family_id:
            raise ValueError("release family ID must match the family input")
        if (
            self.family.provider_id is not None
            and self.release.provider_id is not None
            and self.family.provider_id != self.release.provider_id
        ):
            raise ValueError("family and release provider IDs differ")
        _validate_scope(self.source_scope, field_name="source_scope")
        _validate_scope(self.control_write_scope, field_name="control_write_scope")
        if self.source_scope.overlaps(self.control_write_scope):
            raise ValueError("registration source and control scopes must not overlap")
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        validate_sha256(
            self.expected_physical_sha256,
            field_name="expected_physical_sha256",
        )
        if (
            self.release.expected_physical_sha256 is not None
            and self.expected_physical_sha256 != self.release.expected_physical_sha256
        ):
            raise ValueError("materialization and release physical digest expectations differ")
        _validate_expected_count(
            self.expected_byte_size,
            field_name="expected_byte_size",
            maximum=_MAX_EXPECTED_BYTES,
        )
        _validate_expected_count(
            self.expected_file_count,
            field_name="expected_file_count",
            maximum=_MAX_EXPECTED_FILES,
        )
        _validate_text(self.expected_media_type, field_name="expected_media_type")
        _validate_sorted_ids(
            self.expected_format_profile_ids,
            field_name="expected_format_profile_ids",
        )
        if self.expected_format_profile_ids != self.release.expected_format_profile_ids:
            raise ValueError("materialization and release format expectations differ")
        file_minimum = self.release.expected_file_count_minimum
        file_maximum = self.release.expected_file_count_maximum
        if (
            file_minimum is not None
            and file_maximum is not None
            and not file_minimum <= self.expected_file_count <= file_maximum
        ):
            raise ValueError("materialization file count is outside release expectations")
        byte_minimum = self.release.expected_byte_count_minimum
        byte_maximum = self.release.expected_byte_count_maximum
        if (
            byte_minimum is not None
            and byte_maximum is not None
            and not byte_minimum <= self.expected_byte_size <= byte_maximum
        ):
            raise ValueError("materialization byte size is outside release expectations")
        _validate_materialization_classification(
            self.materialization_class,
            self.evidence_class,
        )
        _validate_visibility(self.outcome_access, self.visibility_ceiling)
        _validate_capabilities(self.capabilities, required=_REGISTRATION_CAPABILITIES)
        _validate_invariants(self.expected_invariants)
        _validate_invariant_subjects(
            self.expected_invariants,
            available_subject_ids=frozenset(
                {
                    self.manifest_id,
                    self.family.family_id,
                    self.release.release_id,
                    self.proposed_materialization_id,
                    self.source_scope.scope_id,
                    self.control_write_scope.scope_id,
                    self.selector.selector_id,
                    *(value.binding_id for value in self.capabilities),
                }
            ),
        )
        if not isinstance(self.work_envelope, DatasetWorkEnvelope):
            raise ValueError("work_envelope must be a DatasetWorkEnvelope")
        if self.work_envelope.files.max_source_files == 0:
            raise ValueError("registration must bound at least one source file")
        if self.work_envelope.files.max_destination_files != 0:
            raise ValueError("registration cannot request dataset destination writes")
        _validate_exception_codes(self.exception_codes)
        codes = frozenset(self.exception_codes)
        generated = (
            self.release.expected_resolution_class
            is ReleaseResolutionClass.NOT_APPLICABLE_GENERATED
        )
        structurally_implied = (
            (
                DatasetAuthoringExceptionCode.RELEASE_ID_UNRESOLVED,
                self.release.expected_resolution_class
                is ReleaseResolutionClass.UNRESOLVED_LOCAL_CUSTODY,
            ),
            (
                DatasetAuthoringExceptionCode.RELEASE_TIMESTAMP_UNAVAILABLE,
                self.release.expected_publication_at_utc is None and not generated,
            ),
            (
                DatasetAuthoringExceptionCode.HISTORICAL_RETRIEVAL_TIME_UNAVAILABLE,
                self.release.expected_retrieved_at_utc is None and not generated,
            ),
            (
                DatasetAuthoringExceptionCode.UPSTREAM_CHECKSUM_UNAVAILABLE,
                self.release.expected_physical_sha256 is None,
            ),
            (
                DatasetAuthoringExceptionCode.PARTIAL_SELECTOR_ONLY,
                not self.selector.complete_release,
            ),
        )
        for code, implied in structurally_implied:
            if (code in codes) != implied:
                raise ValueError(f"structural condition and {code.value} must agree")
        licence_codes = codes.intersection(
            {
                DatasetAuthoringExceptionCode.LICENCE_UNSPECIFIED,
                DatasetAuthoringExceptionCode.LICENCE_NOASSERTION,
            }
        )
        if self.release.expected_licence_id is None:
            if len(licence_codes) != 1:
                raise ValueError("missing licence expectation requires one typed licence exception")
        elif licence_codes:
            raise ValueError("known licence expectation cannot retain a missing-licence exception")


@dataclass(frozen=True, slots=True)
class DatasetTransformInput(CanonicalRecord):
    """One exact parent, selector, and storage scope for a transformation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-transform-input'

    input_id: str
    materialization: ObjectIdentity
    scope: DatasetStorageScope
    selector: DatasetSelectorRef
    selector_manifest_scope: DatasetStorageScope | None
    selector_manifest_byte_size: int | None
    selected_members: tuple[DatasetSelectorMemberContract, ...]

    def __post_init__(self) -> None:
        _validate_id(self.input_id, field_name="input_id")
        _validate_object_identity_schema(
            self.materialization,
            field_name="materialization",
            expected_schema=DatasetMaterialization.SCHEMA,
        )
        _validate_scope(self.scope, field_name="scope")
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        if self.selector_manifest_scope is not None:
            _validate_scope(self.selector_manifest_scope, field_name="selector_manifest_scope")
        if self.selector_manifest_byte_size is not None:
            _validate_expected_count(
                self.selector_manifest_byte_size,
                field_name="selector_manifest_byte_size",
                maximum=_MAX_EXPECTED_BYTES,
            )
        _validate_sorted_records(
            self.selected_members,
            attribute="member_id",
            field_name="selected_members",
            expected_type=DatasetSelectorMemberContract,
            allow_empty=True,
            maximum=MAX_DATASET_SELECTOR_MEMBERS,
        )
        member_locators = tuple(value.relative_locator for value in self.selected_members)
        if len(set(member_locators)) != len(member_locators):
            raise ValueError("selected members cannot repeat a locator")
        manifest_parts = (
            self.selector_manifest_scope is not None,
            self.selector_manifest_byte_size is not None,
            bool(self.selected_members),
        )
        if self.selector.complete_release:
            if any(manifest_parts):
                raise ValueError("complete-release selector cannot carry a member manifest")
        elif not all(manifest_parts):
            raise ValueError(
                "bounded partial selector requires manifest scope, size, and exact members"
            )


@dataclass(frozen=True, slots=True)
class DatasetTransformationManifest(CanonicalRecord):
    """Exact parent-closed transformation request; outputs remain expectations."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/dataset-transformation-manifest'

    manifest_id: str
    release_id: str
    inputs: tuple[DatasetTransformInput, ...]
    destination_scope: DatasetStorageScope
    control_write_scope: DatasetStorageScope
    capabilities: tuple[DatasetCapabilityBinding, ...]
    outputs: tuple[DatasetOutputContract, ...]
    comparison_targets: tuple[DatasetComparisonTarget, ...]
    split_contract: DatasetSplitContract
    causal_contract: DatasetCausalContract
    expected_invariants: tuple[DatasetExpectedInvariant, ...]
    work_envelope: DatasetWorkEnvelope
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    exception_codes: tuple[DatasetAuthoringExceptionCode, ...]

    def __post_init__(self) -> None:
        _validate_id(self.manifest_id, field_name="manifest_id")
        _validate_id(self.release_id, field_name="release_id")
        _validate_sorted_records(
            self.inputs,
            attribute="input_id",
            field_name="inputs",
            expected_type=DatasetTransformInput,
        )
        parent_ids = tuple(value.materialization.object_id for value in self.inputs)
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("transformation inputs cannot repeat a parent materialization")
        scope_ids = tuple(value.scope.scope_id for value in self.inputs)
        if len(set(scope_ids)) != len(scope_ids):
            raise ValueError("transformation inputs cannot repeat a source scope")
        _validate_scope(self.destination_scope, field_name="destination_scope")
        _validate_scope(self.control_write_scope, field_name="control_write_scope")
        for index, transform_input in enumerate(self.inputs):
            source_scope = transform_input.scope
            if any(source_scope.overlaps(other.scope) for other in self.inputs[index + 1 :]):
                raise ValueError("transformation source scopes overlap")
            if source_scope.overlaps(self.destination_scope):
                raise ValueError("transformation source and destination scopes overlap")
            if source_scope.overlaps(self.control_write_scope):
                raise ValueError("transformation source and control scopes overlap")
        if self.destination_scope.overlaps(self.control_write_scope):
            raise ValueError("transformation destination and control scopes overlap")
        selector_manifest_scopes = tuple(
            value.selector_manifest_scope
            for value in self.inputs
            if value.selector_manifest_scope is not None
        )
        for scope in selector_manifest_scopes:
            if scope.overlaps(self.destination_scope) or scope.overlaps(self.control_write_scope):
                raise ValueError("selector manifest scope overlaps a transformation write scope")
        _validate_capabilities(self.capabilities, required=_TRANSFORMATION_CAPABILITIES)
        _validate_sorted_records(
            self.outputs,
            attribute="output_id",
            field_name="outputs",
            expected_type=DatasetOutputContract,
        )
        output_materialization_ids = tuple(
            output.proposed_materialization_id for output in self.outputs
        )
        if len(set(output_materialization_ids)) != len(output_materialization_ids):
            raise ValueError("output materialization IDs must be unique")
        output_locators = tuple(output.relative_locator for output in self.outputs)
        if len(set(output_locators)) != len(output_locators):
            raise ValueError("output locators must be unique")
        if set(output_materialization_ids).intersection(parent_ids):
            raise ValueError(
                "output materialization IDs must be disjoint from parent materializations"
            )
        if set(output_locators).intersection(value.scope.relative_prefix for value in self.inputs):
            raise ValueError("output locators must be disjoint from parent source locators")
        expected_parents = tuple(sorted(parent_ids))
        for output in self.outputs:
            if output.parent_materialization_ids != expected_parents:
                raise ValueError("each output must bind the complete source parent set")
        _validate_sorted_records(
            self.comparison_targets,
            attribute="comparison_id",
            field_name="comparison_targets",
            expected_type=DatasetComparisonTarget,
            allow_empty=True,
        )
        output_by_id = {output.output_id: output for output in self.outputs}
        comparison_artifact_ids = tuple(
            target.historical_artifact.object_id for target in self.comparison_targets
        )
        if len(set(comparison_artifact_ids)) != len(comparison_artifact_ids):
            raise ValueError("comparison targets cannot repeat a historical artifact")
        if set(comparison_artifact_ids).intersection({*parent_ids, *output_materialization_ids}):
            raise ValueError(
                "comparison artifacts must be disjoint from lineage parents and outputs"
            )
        for target in self.comparison_targets:
            if target.output_id not in output_by_id:
                raise ValueError("comparison target references an undeclared output")
            output_field_ids = {field.field_id for field in output_by_id[target.output_id].fields}
            normalized_field_ids = {value.field_id for value in target.type_null_normalizations}
            if output_field_ids != normalized_field_ids:
                raise ValueError("comparison normalization must cover the exact output fields")
            output_fields = {
                field.field_id: field for field in output_by_id[target.output_id].fields
            }
            if any(
                normalization.output_data_type
                is not output_fields[normalization.field_id].data_type
                for normalization in target.type_null_normalizations
            ):
                raise ValueError(
                    "comparison replacement types must agree with the output field contract"
                )
            if any(
                normalization.output_native_nulls
                and not output_fields[normalization.field_id].nullable
                for normalization in target.type_null_normalizations
            ):
                raise ValueError(
                    "comparison native-null normalization requires a nullable output field"
                )
            if target.historical_scope.overlaps(self.destination_scope) or (
                target.historical_scope.overlaps(self.control_write_scope)
            ):
                raise ValueError("comparison target scope overlaps a transformation write scope")
        if not isinstance(self.split_contract, DatasetSplitContract):
            raise ValueError("split_contract must be a DatasetSplitContract")
        if not isinstance(self.causal_contract, DatasetCausalContract):
            raise ValueError("causal_contract must be a DatasetCausalContract")
        available_fields = {field.field_id for output in self.outputs for field in output.fields}
        field_contracts: dict[str, DatasetFieldContract] = {}
        for output in self.outputs:
            for field in output.fields:
                existing = field_contracts.setdefault(field.field_id, field)
                if existing != field:
                    raise ValueError("shared output field IDs require identical contracts")
        referenced_fields = (
            set(self.split_contract.independent_unit_field_ids)
            | set(self.split_contract.group_field_ids)
            | set(self.causal_contract.retained_history_field_ids)
            | set(self.causal_contract.action_field_ids)
            | set(self.causal_contract.receiver_field_ids)
        )
        if not referenced_fields.issubset(available_fields):
            raise ValueError("split or causal contract references an undeclared output field")
        if any(
            field_contracts[field_id].role is not DatasetFieldRole.INDEPENDENT_UNIT
            for field_id in self.split_contract.independent_unit_field_ids
        ):
            raise ValueError("split independent-unit fields require INDEPENDENT_UNIT roles")
        if any(
            field_contracts[field_id].role
            not in {DatasetFieldRole.NESTED_OBSERVATION, DatasetFieldRole.BATCH}
            for field_id in self.split_contract.group_field_ids
        ):
            raise ValueError("split group fields require nested-observation or batch roles")
        split_and_group_ids = (
            *self.split_contract.independent_unit_field_ids,
            *self.split_contract.group_field_ids,
        )
        if any(
            field_contracts[field_id].outcome_access is not OutcomeAccess.OUTCOME_BLIND
            for field_id in split_and_group_ids
        ):
            raise ValueError("split/group fields must be outcome-blind")
        if any(
            not field_contracts[field_id].causal_phase.precedes_or_equals(CausalPhase.PRE_ACTION)
            for field_id in split_and_group_ids
        ):
            raise ValueError("split/group fields must precede assignment")
        cutoff_phase = self.causal_contract.information_cutoff.phase
        if any(
            not field_contracts[field_id].causal_phase.precedes_or_equals(cutoff_phase)
            or field_contracts[field_id].causal_phase is cutoff_phase
            for field_id in self.causal_contract.retained_history_field_ids
        ):
            raise ValueError("retained history must precede the cutoff")
        cutoff_clock_id = self.causal_contract.information_cutoff.clock_id
        if any(
            field_contracts[field_id].clock_id is not None
            and field_contracts[field_id].clock_id != cutoff_clock_id
            for field_id in self.causal_contract.retained_history_field_ids
        ):
            raise ValueError("retained history clock must match the causal cutoff clock")
        if any(
            field_contracts[field_id].outcome_access is not OutcomeAccess.OUTCOME_BLIND
            for field_id in self.causal_contract.retained_history_field_ids
        ):
            raise ValueError("retained history must remain outcome-blind")
        if any(
            field_contracts[field_id].role not in _ACTION_PHASES
            for field_id in self.causal_contract.action_field_ids
        ):
            raise ValueError("causal action fields require typed action roles")
        if any(
            field_contracts[field_id].role
            not in {DatasetFieldRole.RESPONSE, DatasetFieldRole.RECEIVER}
            for field_id in self.causal_contract.receiver_field_ids
        ):
            raise ValueError("causal receiver fields require response or receiver roles")
        _validate_invariants(self.expected_invariants)
        _validate_invariant_subjects(
            self.expected_invariants,
            available_subject_ids=frozenset(
                {
                    self.manifest_id,
                    self.release_id,
                    self.destination_scope.scope_id,
                    self.control_write_scope.scope_id,
                    self.split_contract.split_id,
                    self.causal_contract.causal_contract_id,
                    *available_fields,
                    *(value.input_id for value in self.inputs),
                    *(value.materialization.object_id for value in self.inputs),
                    *(value.scope.scope_id for value in self.inputs),
                    *(value.selector.selector_id for value in self.inputs),
                    *(value.output_id for value in self.outputs),
                    *(value.proposed_materialization_id for value in self.outputs),
                    *(value.comparison_id for value in self.comparison_targets),
                    *(value.historical_artifact.object_id for value in self.comparison_targets),
                    *(value.historical_scope.scope_id for value in self.comparison_targets),
                    *(value.selector.selector_id for value in self.comparison_targets),
                    *(
                        normalization.normalization_id
                        for value in self.comparison_targets
                        for normalization in value.type_null_normalizations
                    ),
                    *(value.binding_id for value in self.capabilities),
                }
            ),
        )
        if not isinstance(self.work_envelope, DatasetWorkEnvelope):
            raise ValueError("work_envelope must be a DatasetWorkEnvelope")
        if (
            self.work_envelope.files.max_source_files == 0
            or self.work_envelope.files.max_destination_files == 0
        ):
            raise ValueError("transformation must bound source and destination files")
        _validate_visibility(self.outcome_access, self.visibility_ceiling)
        _require_outcome_access_includes(
            self.outcome_access,
            tuple(output.outcome_access for output in self.outputs)
            + tuple(target.outcome_access for target in self.comparison_targets),
            message="transformation cannot understate output or comparison outcome access",
        )
        output_ceiling = VisibilityCeiling.most_restrictive(
            *(output.visibility_ceiling for output in self.outputs),
            *(target.visibility_ceiling for target in self.comparison_targets),
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(output_ceiling):
            raise ValueError("transformation visibility cannot lower an output ceiling")
        _validate_exception_codes(self.exception_codes)


@dataclass(frozen=True, slots=True)
class ProposedExperimentDatasetBindingManifest(CanonicalRecord):
    """Proposed metadata binding; it is not a verified binding or scientific verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/proposed-experiment-dataset-binding-manifest'

    manifest_id: str
    experiment: ObjectIdentity
    release_id: str
    materialization: ObjectIdentity
    dataset_contract: ObjectIdentity
    transformation_manifest: ObjectIdentity | None
    selector: DatasetSelectorRef
    role: DatasetBindingRole
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    split_contract: DatasetSplitContract
    causal_contract: DatasetCausalContract
    control_write_scope: DatasetStorageScope
    capabilities: tuple[DatasetCapabilityBinding, ...]
    expected_invariants: tuple[DatasetExpectedInvariant, ...]
    work_envelope: DatasetWorkEnvelope
    exception_codes: tuple[DatasetAuthoringExceptionCode, ...]

    def __post_init__(self) -> None:
        _validate_id(self.manifest_id, field_name="manifest_id")
        _validate_object_identity_schema(
            self.experiment,
            field_name="experiment",
            expected_schema=_EXPERIMENT_SPEC_SCHEMA,
        )
        _validate_id(self.release_id, field_name="release_id")
        _validate_object_identity_schema(
            self.materialization,
            field_name="materialization",
            expected_schema=DatasetMaterialization.SCHEMA,
        )
        _validate_object_identity_schema(
            self.dataset_contract,
            field_name="dataset contract",
            expected_schema=DatasetOutputContract.SCHEMA,
        )
        if self.transformation_manifest is not None:
            _validate_object_identity_schema(
                self.transformation_manifest,
                field_name="transformation_manifest",
                expected_schema=DatasetTransformationManifest.SCHEMA,
            )
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        if not isinstance(self.role, DatasetBindingRole):
            raise ValueError("role must be a DatasetBindingRole")
        _validate_materialization_classification(
            self.materialization_class,
            self.evidence_class,
        )
        _validate_visibility(self.outcome_access, self.visibility_ceiling)
        if self.role is DatasetBindingRole.EVALUATION_SEALED and (
            self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("sealed evaluation binding requires sealed outcome access")
        if self.role is DatasetBindingRole.EVALUATOR_REVEAL and (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError("evaluator reveal binding requires evaluator-reveal access")
        if not isinstance(self.split_contract, DatasetSplitContract):
            raise ValueError("split_contract must be a DatasetSplitContract")
        if not isinstance(self.causal_contract, DatasetCausalContract):
            raise ValueError("causal_contract must be a DatasetCausalContract")
        _validate_scope(self.control_write_scope, field_name="control_write_scope")
        _validate_capabilities(self.capabilities, required=_BINDING_CAPABILITIES)
        _validate_invariants(self.expected_invariants)
        _validate_invariant_subjects(
            self.expected_invariants,
            available_subject_ids=frozenset(
                {
                    self.manifest_id,
                    self.experiment.object_id,
                    self.release_id,
                    self.materialization.object_id,
                    self.dataset_contract.object_id,
                    self.selector.selector_id,
                    self.split_contract.split_id,
                    self.causal_contract.causal_contract_id,
                    self.control_write_scope.scope_id,
                    *(value.binding_id for value in self.capabilities),
                    *(
                        (self.transformation_manifest.object_id,)
                        if self.transformation_manifest is not None
                        else ()
                    ),
                }
            ),
        )
        if not isinstance(self.work_envelope, DatasetWorkEnvelope):
            raise ValueError("work_envelope must be a DatasetWorkEnvelope")
        files = self.work_envelope.files
        if any(
            (
                files.max_source_files,
                files.max_destination_files,
                files.max_archive_members,
                files.max_single_file_bytes,
                self.work_envelope.resources.source_scan_bytes,
            )
        ):
            raise ValueError("binding authoring may request metadata/receipt work only")
        _validate_exception_codes(self.exception_codes)
        if frozenset(self.exception_codes).intersection(
            {
                DatasetAuthoringExceptionCode.TRANSFORM_LINEAGE_PARTIAL,
                DatasetAuthoringExceptionCode.STRUCTURAL_CONTRACT_UNPROVEN,
            }
        ):
            raise ValueError("a proposed experiment binding cannot retain incomplete lineage")


@dataclass(frozen=True, slots=True)
class ExperimentDatasetBindingRequirement(CanonicalRecord):
    """Path-free scientific input committed before experiment authorization."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-dataset-binding-requirement'

    binding_id: str
    run_id: str | None
    release_id: str
    materialization: ObjectIdentity
    selector: DatasetSelectorRef
    role: DatasetBindingRole
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    dataset_contract: ObjectIdentity
    transformation_manifest: ObjectIdentity | None
    parent_materializations: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _validate_id(self.binding_id, field_name="binding_id")
        if self.run_id is not None:
            _validate_id(self.run_id, field_name="run_id")
        _validate_id(self.release_id, field_name="release_id")
        _validate_object_identity_schema(
            self.materialization,
            field_name="materialization",
            expected_schema=DatasetMaterialization.SCHEMA,
        )
        if not isinstance(self.selector, DatasetSelectorRef):
            raise ValueError("selector must be a DatasetSelectorRef")
        if not isinstance(self.role, DatasetBindingRole):
            raise ValueError("role must be a DatasetBindingRole")
        _validate_materialization_classification(
            self.materialization_class,
            self.evidence_class,
        )
        _validate_visibility(self.outcome_access, self.visibility_ceiling)
        _validate_object_identity_schema(
            self.dataset_contract,
            field_name="dataset_contract",
            expected_schema=DatasetOutputContract.SCHEMA,
        )
        if self.transformation_manifest is not None:
            _validate_object_identity_schema(
                self.transformation_manifest,
                field_name="transformation_manifest",
                expected_schema=DatasetTransformationManifest.SCHEMA,
            )
        _validate_sorted_records(
            self.parent_materializations,
            attribute="object_id",
            field_name="parent_materializations",
            expected_type=ObjectIdentity,
            allow_empty=True,
        )
        if any(
            value.object_schema != DatasetMaterialization.SCHEMA
            for value in self.parent_materializations
        ):
            raise ValueError("binding parents must identify dataset materializations")
        if self.materialization.object_id in {
            value.object_id for value in self.parent_materializations
        }:
            raise ValueError("a binding materialization cannot parent itself")
        if (
            self.materialization_class
            in {
                DatasetMaterializationClass.HISTORICAL_DERIVATIVE,
                DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
                DatasetMaterializationClass.GENERATED_OBSERVATION,
            }
            and not self.parent_materializations
        ):
            raise ValueError("derived or generated binding requires exact parents")
        if self.role is DatasetBindingRole.EVALUATION_SEALED and (
            self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("sealed evaluation requirement requires sealed outcome access")
        if self.role is DatasetBindingRole.EVALUATOR_REVEAL and (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError("evaluator reveal requirement requires evaluator-reveal access")

    @classmethod
    def from_manifest(
        cls,
        *,
        binding_id: str,
        manifest: ProposedExperimentDatasetBindingManifest,
        parent_materializations: tuple[ObjectIdentity, ...] = (),
        run_id: str | None = None,
    ) -> ExperimentDatasetBindingRequirement:
        if not isinstance(manifest, ProposedExperimentDatasetBindingManifest):
            raise TypeError("manifest must be a proposed experiment binding")
        return cls(
            binding_id=binding_id,
            run_id=run_id,
            release_id=manifest.release_id,
            materialization=manifest.materialization,
            selector=manifest.selector,
            role=manifest.role,
            materialization_class=manifest.materialization_class,
            evidence_class=manifest.evidence_class,
            outcome_access=manifest.outcome_access,
            visibility_ceiling=manifest.visibility_ceiling,
            dataset_contract=manifest.dataset_contract,
            transformation_manifest=manifest.transformation_manifest,
            parent_materializations=parent_materializations,
        )

    @classmethod
    def from_binding(
        cls,
        binding: ExperimentDatasetBinding,
    ) -> ExperimentDatasetBindingRequirement:
        if not isinstance(binding, ExperimentDatasetBinding):
            raise TypeError("binding must be an ExperimentDatasetBinding")
        return cls(
            binding_id=binding.binding_id,
            run_id=binding.run_id,
            release_id=binding.release_id,
            materialization=binding.materialization,
            selector=binding.selector,
            role=binding.role,
            materialization_class=binding.materialization_class,
            evidence_class=binding.evidence_class,
            outcome_access=binding.outcome_access,
            visibility_ceiling=binding.visibility_ceiling,
            dataset_contract=binding.dataset_contract,
            transformation_manifest=binding.transformation_manifest,
            parent_materializations=binding.parent_materializations,
        )


def experiment_dataset_binding_set_id(experiment_spec_id: str) -> str:
    """Return a bounded deterministic ID without weakening the experiment ID."""

    _validate_id(experiment_spec_id, field_name="experiment_spec_id")
    digest = hashlib.sha256(experiment_spec_id.encode("utf-8")).hexdigest()[:32]
    return f"binding-set.{digest}"


@dataclass(frozen=True, slots=True)
class ExperimentDatasetBindingSet(CanonicalRecord):
    """Exact aggregate reconstructed from authenticated bindings at compilation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-dataset-binding-set'

    binding_set_id: str
    experiment_spec_id: str
    requirements: tuple[ExperimentDatasetBindingRequirement, ...]

    def __post_init__(self) -> None:
        _validate_id(self.binding_set_id, field_name="binding_set_id")
        _validate_id(self.experiment_spec_id, field_name="experiment_spec_id")
        if self.binding_set_id != experiment_dataset_binding_set_id(self.experiment_spec_id):
            raise ValueError("binding_set_id is not derived from experiment_spec_id")
        _validate_sorted_records(
            self.requirements,
            attribute="binding_id",
            field_name="requirements",
            expected_type=ExperimentDatasetBindingRequirement,
        )
        role_targets = tuple(
            (
                value.role,
                value.release_id,
                value.materialization.object_id,
                value.selector.selector_id,
            )
            for value in self.requirements
        )
        if len(set(role_targets)) != len(role_targets):
            raise ValueError("binding requirements repeat an exact role target")

    @classmethod
    def from_bindings(
        cls,
        *,
        experiment_spec_id: str,
        bindings: tuple[ExperimentDatasetBinding, ...],
    ) -> ExperimentDatasetBindingSet:
        return cls(
            binding_set_id=experiment_dataset_binding_set_id(experiment_spec_id),
            experiment_spec_id=experiment_spec_id,
            requirements=tuple(
                sorted(
                    (ExperimentDatasetBindingRequirement.from_binding(value) for value in bindings),
                    key=lambda value: value.binding_id,
                )
            ),
        )

    def extension(self) -> ExtensionBinding:
        return ExtensionBinding(
            namespace=EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE,
            schema=self.SCHEMA,
            payload_sha256=self.fingerprint(),
        )


DATASET_REGISTRATION_MANIFEST_SCHEMAS: Mapping[str, type[DatasetRegistrationManifest]] = (
    MappingProxyType({DatasetRegistrationManifest.SCHEMA: DatasetRegistrationManifest})
)
DATASET_TRANSFORMATION_MANIFEST_SCHEMAS: Mapping[str, type[DatasetTransformationManifest]] = (
    MappingProxyType({DatasetTransformationManifest.SCHEMA: DatasetTransformationManifest})
)
DATASET_BINDING_MANIFEST_SCHEMAS: Mapping[str, type[ProposedExperimentDatasetBindingManifest]] = (
    MappingProxyType(
        {
            ProposedExperimentDatasetBindingManifest.SCHEMA: (
                ProposedExperimentDatasetBindingManifest
            )
        }
    )
)
DATASET_MANIFEST_SCHEMAS: Mapping[str, type[CanonicalRecord]] = MappingProxyType(
    {
        DatasetRegistrationManifest.SCHEMA: DatasetRegistrationManifest,
        DatasetTransformationManifest.SCHEMA: DatasetTransformationManifest,
        ProposedExperimentDatasetBindingManifest.SCHEMA: (ProposedExperimentDatasetBindingManifest),
    }
)


__all__ = [
    "DATASET_BINDING_MANIFEST_SCHEMAS",
    "DATASET_MANIFEST_SCHEMAS",
    "DATASET_REGISTRATION_MANIFEST_SCHEMAS",
    "DATASET_TRANSFORMATION_MANIFEST_SCHEMAS",
    "MAX_DATASET_SELECTOR_MEMBERS",
    "EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE",
    "DatasetAuthoringExceptionCode",
    "DatasetCapabilityBinding",
    "DatasetCapabilityKind",
    "DatasetCausalContract",
    "DatasetComparisonTarget",
    "DatasetDirectoryMemberExpectation",
    "DatasetDirectorySelectorManifest",
    "DatasetExpectedInvariant",
    "ExperimentDatasetBindingRequirement",
    "ExperimentDatasetBindingSet",
    "DatasetFamilyInput",
    "DatasetFieldContract",
    "DatasetFieldDataType",
    "DatasetFieldRole",
    "DatasetInvariantKind",
    "DatasetOutputContract",
    "DatasetOutputRole",
    "DatasetRegistrationManifest",
    "DatasetReleaseInput",
    "DatasetSelectorMemberContract",
    "DatasetSplitContract",
    "DatasetTransformInput",
    "DatasetTransformationManifest",
    "DatasetTypeNullNormalization",
    "ProposedExperimentDatasetBindingManifest",
    "dataset_directory_content_sha256",
    "experiment_dataset_binding_set_id",
]
