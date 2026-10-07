"""Strict held-directory registration manifests for current simulators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import CausalPhase
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
    DatasetDirectorySelectorManifest,
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
)
from empirical_lawhood.planning.datasets import (
    DatasetEvidenceClass,
    DatasetMaterializationClass,
    DatasetSelectorKind,
    DatasetSelectorRef,
    ReleaseResolutionClass,
)

from .lineage import SimulatorGenerationLineageManifest
from .registry import (
    SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,
    SIMULATOR_DIRECTORY_MEDIA_TYPE,
    SIMULATOR_MAX_METADATA_BYTES,
    SIMULATOR_MAX_SINGLE_FILE_BYTES,
)


_REQUIRED_CAPABILITY_KINDS = frozenset(
    {
        DatasetCapabilityKind.CONFIG,
        DatasetCapabilityKind.ENVIRONMENT,
        DatasetCapabilityKind.EVIDENCE_VERIFIER,
        DatasetCapabilityKind.INSPECTOR,
        DatasetCapabilityKind.RUNTIME,
    }
)


@dataclass(frozen=True, slots=True)
class SimulatorHeldRegistrationDefinition(CanonicalRecord):
    """Exact authoring facts for one held software copy or generated bundle."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-held-registration-definition'

    definition_id: str
    family: DatasetFamilyInput
    release: DatasetReleaseInput
    proposed_materialization_id: str
    source_scope_id: str
    source_relative_prefix: str
    control_scope_id: str
    control_relative_prefix: str
    selector: DatasetDirectorySelectorManifest
    materialization_class: DatasetMaterializationClass
    evidence_class: DatasetEvidenceClass
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    generation_lineage: SimulatorGenerationLineageManifest | None
    exception_codes: tuple[DatasetAuthoringExceptionCode, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("definition_id", self.definition_id),
            ("proposed_materialization_id", self.proposed_materialization_id),
            ("source_scope_id", self.source_scope_id),
            ("control_scope_id", self.control_scope_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if not isinstance(self.family, DatasetFamilyInput):
            raise ValueError("family must be a DatasetFamilyInput")
        if not isinstance(self.release, DatasetReleaseInput):
            raise ValueError("release must be a DatasetReleaseInput")
        if self.release.family_id != self.family.family_id:
            raise ValueError("release family differs from the registration family")
        validate_relative_locator(self.source_relative_prefix)
        validate_relative_locator(self.control_relative_prefix)
        if not isinstance(self.selector, DatasetDirectorySelectorManifest):
            raise ValueError("selector must be a DatasetDirectorySelectorManifest")
        if self.selector.subject_id != self.release.release_id:
            raise ValueError("directory selector subject differs from the release")
        if self.release.expected_physical_sha256 != self.selector.expected_content_sha256:
            raise ValueError("release digest differs from the exact directory selector")
        if self.release.expected_file_count_minimum != len(self.selector.members) or (
            self.release.expected_file_count_maximum != len(self.selector.members)
        ):
            raise ValueError("release file bounds differ from the exact selector")
        if (
            self.release.expected_byte_count_minimum != self.selector.expected_total_size_bytes
            or self.release.expected_byte_count_maximum != self.selector.expected_total_size_bytes
        ):
            raise ValueError("release byte bounds differ from the exact selector")
        if self.release.expected_format_profile_ids != (SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,):
            raise ValueError("release has another directory format profile")
        generated = (
            self.release.expected_resolution_class
            is ReleaseResolutionClass.NOT_APPLICABLE_GENERATED
        )
        if generated != (self.generation_lineage is not None):
            raise ValueError("generated releases require exactly one lineage manifest")
        expected_manifest = (
            self.selector.fingerprint()
            if self.generation_lineage is None
            else self.generation_lineage.fingerprint()
        )
        if self.release.expected_manifest_sha256 != expected_manifest:
            raise ValueError("release manifest digest does not bind selector/lineage")
        if self.generation_lineage is not None:
            if (
                self.generation_lineage.proposed_materialization_id
                != self.proposed_materialization_id
                or self.generation_lineage.outcome_access is not self.outcome_access
                or self.generation_lineage.visibility_ceiling is not self.visibility_ceiling
            ):
                raise ValueError("generation lineage differs from its registration definition")
            self.generation_lineage.validate_selected_evidence(self.selector)
        if not isinstance(self.materialization_class, DatasetMaterializationClass):
            raise ValueError("materialization_class has another type")
        if not isinstance(self.evidence_class, DatasetEvidenceClass):
            raise ValueError("evidence_class has another type")
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise ValueError("outcome_access has another type")
        if not isinstance(self.visibility_ceiling, VisibilityCeiling):
            raise ValueError("visibility_ceiling has another type")
        if not isinstance(self.exception_codes, tuple) or any(
            not isinstance(value, DatasetAuthoringExceptionCode) for value in self.exception_codes
        ):
            raise ValueError("exception_codes has another type")
        encoded = tuple(value.value for value in self.exception_codes)
        if tuple(sorted(set(encoded))) != encoded:
            raise ValueError("exception_codes must be sorted and unique")


@dataclass(frozen=True, slots=True)
class SimulatorRegistrationManifestDependencies:
    """Trusted composition inputs kept outside scientific configuration."""

    storage_root: ObjectIdentity
    capabilities: tuple[DatasetCapabilityBinding, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.storage_root, ObjectIdentity)
            or self.storage_root.object_schema != DATASET_EXTERNAL_ROOT_CONTRACT_SCHEMA
        ):
            raise ValueError("storage_root must identify a current external root")
        if not isinstance(self.capabilities, tuple) or any(
            not isinstance(value, DatasetCapabilityBinding) for value in self.capabilities
        ):
            raise ValueError("capabilities must contain exact capability bindings")
        binding_ids = tuple(value.binding_id for value in self.capabilities)
        kinds = tuple(value.kind for value in self.capabilities)
        if tuple(sorted(set(binding_ids))) != binding_ids or (
            len(set(kinds)) != len(kinds) or set(kinds) != set(_REQUIRED_CAPABILITY_KINDS)
        ):
            raise ValueError("capabilities do not bind the exact registration closure")


def _scope(
    scope_id: str,
    root: ObjectIdentity,
    relative_prefix: str,
) -> DatasetStorageScope:
    return DatasetStorageScope(
        scope_id=scope_id,
        storage_root=root,
        relative_prefix=relative_prefix,
    )


def _invariants(
    definition: SimulatorHeldRegistrationDefinition,
) -> tuple[DatasetExpectedInvariant, ...]:
    kinds = [
        DatasetInvariantKind.NO_DOWNLOAD,
        DatasetInvariantKind.SOURCE_BYTES_IMMUTABLE,
    ]
    if definition.generation_lineage is not None:
        kinds.extend(
            (
                DatasetInvariantKind.PARENT_CLOSED,
                DatasetInvariantKind.INDEPENDENT_UNIT_PRESERVED,
                DatasetInvariantKind.NESTED_ROWS_DO_NOT_INFLATE_REPLICATION,
                DatasetInvariantKind.UNITS_FRAMES_CLOCKS_EXPLICIT,
            )
        )
        if definition.generation_lineage.action_clock_id is not None:
            kinds.append(DatasetInvariantKind.ACTION_CLOCKS_DISTINCT)
    values = tuple(
        DatasetExpectedInvariant(
            invariant_id=f"invariant.{value.value.lower().replace('_', '-')}",
            kind=value,
            subject_ids=(definition.proposed_materialization_id,),
            checker_registry_key=f"dataset.check.{value.value.lower().replace('_', '-')}",
            required=True,
        )
        for value in kinds
    )
    return tuple(sorted(values, key=lambda value: value.invariant_id))


def _work_envelope(
    selector: DatasetDirectorySelectorManifest,
) -> DatasetWorkEnvelope:
    return DatasetWorkEnvelope(
        resources=ResourceBudget(
            cpu_cores=4,
            memory_bytes=2 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=7200,
            source_scan_bytes=selector.expected_total_size_bytes,
            output_bytes=0,
        ),
        files=DatasetFileLimits(
            max_source_files=len(selector.members),
            max_destination_files=0,
            max_archive_members=0,
            max_single_file_bytes=min(
                SIMULATOR_MAX_SINGLE_FILE_BYTES,
                max(value.expected_size_bytes for value in selector.members),
            ),
        ),
        control=DatasetControlLimits(
            max_manifest_bytes=SIMULATOR_MAX_METADATA_BYTES,
            max_metadata_records=SIMULATOR_MAX_METADATA_BYTES,
            max_receipt_bytes=SIMULATOR_MAX_METADATA_BYTES,
            max_reason_codes=64,
        ),
    )


def build_simulator_held_registration_manifest(
    definition: SimulatorHeldRegistrationDefinition,
    dependencies: SimulatorRegistrationManifestDependencies,
) -> DatasetRegistrationManifest:
    """Lower one exact held registration definition into the generic dataset registration contract."""

    if not isinstance(definition, SimulatorHeldRegistrationDefinition):
        raise TypeError("definition must be a SimulatorHeldRegistrationDefinition")
    if not isinstance(dependencies, SimulatorRegistrationManifestDependencies):
        raise TypeError("dependencies must be SimulatorRegistrationManifestDependencies")
    selector = definition.selector
    return DatasetRegistrationManifest(
        manifest_id=f"manifest.{definition.definition_id}",
        family=definition.family,
        release=definition.release,
        proposed_materialization_id=definition.proposed_materialization_id,
        source_scope=_scope(
            definition.source_scope_id,
            dependencies.storage_root,
            definition.source_relative_prefix,
        ),
        control_write_scope=_scope(
            definition.control_scope_id,
            dependencies.storage_root,
            definition.control_relative_prefix,
        ),
        selector=DatasetSelectorRef(
            selector_id=selector.selector_id,
            kind=(
                DatasetSelectorKind.COMPLETE_RELEASE
                if selector.complete_subject
                else DatasetSelectorKind.MANIFEST
            ),
            selector_schema=selector.SCHEMA,
            selector_sha256=selector.fingerprint(),
            complete_release=selector.complete_subject,
        ),
        expected_physical_sha256=selector.expected_content_sha256,
        expected_byte_size=selector.expected_total_size_bytes,
        expected_file_count=len(selector.members),
        expected_media_type=SIMULATOR_DIRECTORY_MEDIA_TYPE,
        expected_format_profile_ids=(SIMULATOR_DIRECTORY_FORMAT_PROFILE_ID,),
        materialization_class=definition.materialization_class,
        evidence_class=definition.evidence_class,
        outcome_access=definition.outcome_access,
        visibility_ceiling=definition.visibility_ceiling,
        capabilities=dependencies.capabilities,
        expected_invariants=_invariants(definition),
        work_envelope=_work_envelope(selector),
        exception_codes=definition.exception_codes,
    )


def build_simulator_directory_output_contract(
    manifest: DatasetRegistrationManifest,
    *,
    generation_lineage: SimulatorGenerationLineageManifest | None = None,
) -> DatasetOutputContract:
    "Describe one exact held directory without pretending it is a table.\n\n    Scientific row/field contracts belong to later transformations.  This\n    source contract binds only the selected directory bytes and their exact\n    generated parents, which is sufficient for software/runtime and historical\n    bundle custody to participate in the generic dataset binding path.\n    "

    if not isinstance(manifest, DatasetRegistrationManifest):
        raise TypeError("manifest must be a DatasetRegistrationManifest")
    if generation_lineage is not None:
        if (
            not isinstance(generation_lineage, SimulatorGenerationLineageManifest)
            or generation_lineage.proposed_materialization_id
            != manifest.proposed_materialization_id
            or generation_lineage.outcome_access is not manifest.outcome_access
            or generation_lineage.visibility_ceiling is not manifest.visibility_ceiling
        ):
            raise ValueError("generation lineage differs from its registration manifest")
    output_id = f"contract.{manifest.proposed_materialization_id}"
    return DatasetOutputContract(
        output_id=output_id,
        proposed_materialization_id=manifest.proposed_materialization_id,
        role=(
            DatasetOutputRole.CANONICAL_SOURCE
            if generation_lineage is None
            else DatasetOutputRole.AUXILIARY
        ),
        relative_locator=manifest.source_scope.relative_prefix,
        output_schema=DatasetDirectorySelectorManifest.SCHEMA,
        media_type=manifest.expected_media_type,
        format_profile_id=manifest.expected_format_profile_ids[0],
        fields=(
            DatasetFieldContract(
                field_id="field.simulator-directory.member-relative-locator",
                column_name="member_relative_locator",
                column_index=0,
                data_type=DatasetFieldDataType.STRING,
                role=DatasetFieldRole.NESTED_OBSERVATION,
                unit=None,
                frame=None,
                clock_id=None,
                causal_phase=CausalPhase.PREPARATION,
                outcome_access=manifest.outcome_access,
                visibility_ceiling=manifest.visibility_ceiling,
                nullable=False,
            ),
        ),
        parent_materialization_ids=(
            ()
            if generation_lineage is None
            else tuple(value.object_id for value in generation_lineage.parent_materializations)
        ),
        expected_physical_sha256=manifest.expected_physical_sha256,
        expected_byte_count_minimum=manifest.expected_byte_size,
        expected_byte_count_maximum=manifest.expected_byte_size,
        expected_file_count=manifest.expected_file_count,
        expected_row_count_minimum=manifest.expected_file_count,
        expected_row_count_maximum=manifest.expected_file_count,
        outcome_access=manifest.outcome_access,
        visibility_ceiling=manifest.visibility_ceiling,
    )


__all__ = [
    "SimulatorHeldRegistrationDefinition",
    "SimulatorRegistrationManifestDependencies",
    "build_simulator_directory_output_contract",
    "build_simulator_held_registration_manifest",
]
