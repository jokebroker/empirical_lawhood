"""Static, source-neutral capability for exact experiment dataset binding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar, Final

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.dataset_manifests import (
    DatasetAuthoringExceptionCode,
    DatasetCapabilityBinding,
    DatasetCapabilityKind,
    DatasetCausalContract,
    DatasetExpectedInvariant,
    DatasetInvariantKind,
    DatasetOutputContract,
    DatasetSplitContract,
    ProposedExperimentDatasetBindingManifest,
)
from empirical_lawhood.planning.dataset_authority import (
    DatasetControlLimits,
    DatasetFileLimits,
    DatasetStorageScope,
    DatasetWorkEnvelope,
)
from empirical_lawhood.planning.datasets import (
    DatasetBindingRole,
    DatasetMaterialization,
    ExperimentDatasetBinding,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
    ImplementationSourceClosure,
)
from empirical_lawhood.runtime.datasets import (
    DatasetBindingCompilerRegistration,
    DatasetCapabilityLimits,
    DatasetCapabilityRegistry,
    DatasetSupportingIdentityRegistry,
)


DATASET_BINDING_COMPILER_KEY: Final = "dataset.binding-compiler.exact"
DATASET_BINDING_COMPILER_VERSION: Final = "1.0.0"
DATASET_BINDING_CONFIG_KEY: Final = "dataset.config.binding-compiler"
DATASET_BINDING_RUNTIME_KEY: Final = "dataset.runtime.binding-compiler"
DATASET_BINDING_ENVIRONMENT_KEY: Final = "dataset.environment.binding-compiler"
DATASET_BINDING_CONFIG_ARTIFACT_ID: Final = "artifact.dataset-binding-config"
DATASET_BINDING_COMPILER_CLOSURE_ID: Final = "implementation.dataset-binding-compiler"


@dataclass(frozen=True, slots=True)
class DatasetBindingCompilerConfig(CanonicalRecord):
    """Closed metadata-only compiler policy; it contains no locator or callable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/dataset-binding-compiler-config'

    profile_id: str
    current_verified_custody_required: bool
    source_authority_required: bool
    parent_closure_required: bool
    path_inputs_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        if self.profile_id != "exact-authenticated-binding":
            raise ValueError("dataset binding compiler config has another profile")
        if (
            self.current_verified_custody_required is not True
            or self.source_authority_required is not True
            or self.parent_closure_required is not True
            or self.path_inputs_allowed is not False
        ):
            raise ValueError("dataset binding compiler safety policy cannot be weakened")


def dataset_binding_compiler_config() -> DatasetBindingCompilerConfig:
    return DatasetBindingCompilerConfig(
        profile_id="exact-authenticated-binding",
        current_verified_custody_required=True,
        source_authority_required=True,
        parent_closure_required=True,
        path_inputs_allowed=False,
    )


@dataclass(frozen=True, slots=True)
class DatasetBindingCompilerStaticComposition(CanonicalRecord):
    """Exact capability and supporting identities for one clean implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/dataset-binding-compiler-static-composition'

    config: DatasetBindingCompilerConfig
    config_ref: CapabilityConfigRef
    runtime: ObjectIdentity
    environment: ObjectIdentity
    implementation_closure: ImplementationSourceClosure
    capability_registry: CapabilityRegistry
    dataset_registry: DatasetCapabilityRegistry
    supporting_registry: DatasetSupportingIdentityRegistry

    def __post_init__(self) -> None:
        if self.config != dataset_binding_compiler_config():
            raise ValueError("binding compiler config differs from its fixed profile")
        for value, object_id, field_name in (
            (self.runtime, DATASET_BINDING_RUNTIME_KEY, "runtime"),
            (self.environment, DATASET_BINDING_ENVIRONMENT_KEY, "environment"),
        ):
            if not isinstance(value, ObjectIdentity) or value.object_id != object_id:
                raise ValueError(f"{field_name} has another exact identity")
        if (
            not isinstance(self.implementation_closure, ImplementationSourceClosure)
            or self.implementation_closure.closure_id != DATASET_BINDING_COMPILER_CLOSURE_ID
        ):
            raise ValueError("binding compiler implementation closure differs")
        if (
            self.config_ref.config_id != DATASET_BINDING_CONFIG_KEY
            or self.config_ref.config_schema != self.config.SCHEMA
            or self.config_ref.content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("binding compiler config reference differs")
        capability = self.capability_registry.resolve(
            DATASET_BINDING_COMPILER_KEY,
            DATASET_BINDING_COMPILER_VERSION,
        )
        if (
            capability.implementation_sha256 != self.implementation_closure.fingerprint()
            or self.dataset_registry.binding_compiler(capability.registry_id).capability
            != capability
        ):
            raise ValueError("binding compiler registration differs from its source closure")
        if self.supporting_registry != DatasetSupportingIdentityRegistry(
            registry_id="dataset-support.binding-compiler",
            configs=(ObjectIdentity.from_record(DATASET_BINDING_CONFIG_KEY, self.config_ref),),
            runtimes=(self.runtime,),
            environments=(self.environment,),
        ):
            raise ValueError("binding compiler supporting identities differ")


def build_dataset_binding_compiler_static_composition(
    *,
    runtime: ObjectIdentity,
    environment: ObjectIdentity,
    implementation_closure: ImplementationSourceClosure,
) -> DatasetBindingCompilerStaticComposition:
    for value, object_id, field_name in (
        (runtime, DATASET_BINDING_RUNTIME_KEY, "runtime"),
        (environment, DATASET_BINDING_ENVIRONMENT_KEY, "environment"),
    ):
        if not isinstance(value, ObjectIdentity) or value.object_id != object_id:
            raise ValueError(f"{field_name} has another exact identity")
    if (
        not isinstance(implementation_closure, ImplementationSourceClosure)
        or implementation_closure.closure_id != DATASET_BINDING_COMPILER_CLOSURE_ID
    ):
        raise ValueError("implementation_closure has another exact identity")
    config = dataset_binding_compiler_config()
    config_ref = CapabilityConfigRef(
        config_id=DATASET_BINDING_CONFIG_KEY,
        config_schema=config.SCHEMA,
        config_schema_sha256=hashlib.sha256(config.SCHEMA.encode("utf-8")).hexdigest(),
        content_sha256=config.fingerprint(),
        artifact_id=DATASET_BINDING_CONFIG_ARTIFACT_ID,
    )
    capability = CapabilityManifest(
        capability_key=DATASET_BINDING_COMPILER_KEY,
        capability_version=DATASET_BINDING_COMPILER_VERSION,
        kind=CapabilityKind.TRANSFORM,
        config_schema=config_ref.config_schema,
        config_schema_sha256=config_ref.config_schema_sha256,
        input_schema_ids=(ProposedExperimentDatasetBindingManifest.SCHEMA,),
        output_schema_ids=(ExperimentDatasetBinding.SCHEMA,),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        resource_ceiling=ResourceBudget(
            cpu_cores=1,
            memory_bytes=64 * 1024**2,
            gpu_devices=0,
            wall_time_seconds=120,
            source_scan_bytes=10_000_000,
            output_bytes=10_000_000,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=runtime.object_id,
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "binding-current-custody",
            "binding-exact-experiment-materialization-selector-role",
            "binding-no-path-input",
            "binding-outcome-visibility-no-downgrade",
            "binding-parent-and-source-authority-closure",
        ),
        implementation_sha256=implementation_closure.fingerprint(),
    )
    registration = DatasetBindingCompilerRegistration(
        capability=capability,
        limits=DatasetCapabilityLimits(
            max_input_bytes=10_000_000,
            max_output_bytes=10_000_000,
            max_records=10_000,
            max_files=0,
            max_archive_members=0,
        ),
    )
    capability_registry = CapabilityRegistry(
        registry_id="capabilities.dataset-binding-compiler",
        capabilities=(capability,),
    )
    dataset_registry = DatasetCapabilityRegistry(
        registry_id="dataset-capabilities.binding-compiler",
        providers=(),
        inspectors=(),
        transforms=(),
        binding_compilers=(registration,),
        storage_verifiers=(),
        evidence_verifiers=(),
    )
    supporting_registry = DatasetSupportingIdentityRegistry(
        registry_id="dataset-support.binding-compiler",
        configs=(ObjectIdentity.from_record(DATASET_BINDING_CONFIG_KEY, config_ref),),
        runtimes=(runtime,),
        environments=(environment,),
    )
    return DatasetBindingCompilerStaticComposition(
        config=config,
        config_ref=config_ref,
        runtime=runtime,
        environment=environment,
        implementation_closure=implementation_closure,
        capability_registry=capability_registry,
        dataset_registry=dataset_registry,
        supporting_registry=supporting_registry,
    )


def dataset_binding_compiler_capability_bindings(
    composition: DatasetBindingCompilerStaticComposition,
) -> tuple[DatasetCapabilityBinding, ...]:
    capability = composition.capability_registry.resolve(
        DATASET_BINDING_COMPILER_KEY,
        DATASET_BINDING_COMPILER_VERSION,
    )
    return tuple(
        sorted(
            (
                DatasetCapabilityBinding(
                    binding_id="binding.dataset-binding-compiler",
                    kind=DatasetCapabilityKind.BINDING_COMPILER,
                    registry_key=capability.capability_key,
                    implementation=ObjectIdentity.from_record(
                        capability.capability_key,
                        capability,
                    ),
                ),
                DatasetCapabilityBinding(
                    binding_id="binding.dataset-binding-config",
                    kind=DatasetCapabilityKind.CONFIG,
                    registry_key=composition.config_ref.config_id,
                    implementation=ObjectIdentity.from_record(
                        composition.config_ref.config_id,
                        composition.config_ref,
                    ),
                ),
                DatasetCapabilityBinding(
                    binding_id="binding.dataset-binding-environment",
                    kind=DatasetCapabilityKind.ENVIRONMENT,
                    registry_key=composition.environment.object_id,
                    implementation=composition.environment,
                ),
                DatasetCapabilityBinding(
                    binding_id="binding.dataset-binding-runtime",
                    kind=DatasetCapabilityKind.RUNTIME,
                    registry_key=composition.runtime.object_id,
                    implementation=composition.runtime,
                ),
            ),
            key=lambda value: value.binding_id,
        )
    )


def build_proposed_experiment_dataset_binding_manifest(
    *,
    manifest_id: str,
    experiment: ExperimentSpec,
    materialization: DatasetMaterialization,
    dataset_contract: DatasetOutputContract,
    role: DatasetBindingRole,
    split_contract: DatasetSplitContract,
    causal_contract: DatasetCausalContract,
    control_write_scope: DatasetStorageScope,
    composition: DatasetBindingCompilerStaticComposition,
    transformation_manifest: ObjectIdentity | None = None,
    exception_codes: tuple[DatasetAuthoringExceptionCode, ...] = (),
) -> ProposedExperimentDatasetBindingManifest:
    "Lower exact authoring records to the source-neutral dataset manifest."

    if not isinstance(experiment, ExperimentSpec):
        raise TypeError("experiment must be an ExperimentSpec")
    if not isinstance(materialization, DatasetMaterialization):
        raise TypeError("materialization must be a DatasetMaterialization")
    if not isinstance(dataset_contract, DatasetOutputContract):
        raise TypeError("dataset_contract must be a DatasetOutputContract")
    if dataset_contract.proposed_materialization_id != materialization.materialization_id:
        raise ValueError("dataset contract names another materialization")
    if dataset_contract.outcome_access is not materialization.outcome_access:
        raise ValueError("dataset contract outcome access differs from materialization custody")
    if not isinstance(role, DatasetBindingRole):
        raise TypeError("role must be a DatasetBindingRole")
    parent_subjects = (
        materialization.materialization_id,
        *((transformation_manifest.object_id,) if transformation_manifest is not None else ()),
    )
    invariants = tuple(
        sorted(
            (
                DatasetExpectedInvariant(
                    invariant_id="invariant.binding.causal-order",
                    kind=DatasetInvariantKind.CAUSAL_ORDER_PRESERVED,
                    subject_ids=(causal_contract.causal_contract_id,),
                    checker_registry_key="dataset.check.binding-causal-order",
                    required=True,
                ),
                DatasetExpectedInvariant(
                    invariant_id="invariant.binding.parent-closed",
                    kind=DatasetInvariantKind.PARENT_CLOSED,
                    subject_ids=tuple(sorted(parent_subjects)),
                    checker_registry_key="dataset.check.binding-parent-closed",
                    required=True,
                ),
                DatasetExpectedInvariant(
                    invariant_id="invariant.binding.split-disjoint",
                    kind=DatasetInvariantKind.SPLIT_DISJOINT,
                    subject_ids=(split_contract.split_id,),
                    checker_registry_key="dataset.check.binding-split-disjoint",
                    required=True,
                ),
            ),
            key=lambda value: value.invariant_id,
        )
    )
    return ProposedExperimentDatasetBindingManifest(
        manifest_id=manifest_id,
        experiment=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        release_id=materialization.release_id,
        materialization=ObjectIdentity.from_record(
            materialization.materialization_id,
            materialization,
        ),
        dataset_contract=ObjectIdentity.from_record(
            dataset_contract.output_id,
            dataset_contract,
        ),
        transformation_manifest=transformation_manifest,
        selector=materialization.selector,
        role=role,
        materialization_class=materialization.materialization_class,
        evidence_class=materialization.evidence_class,
        outcome_access=materialization.outcome_access,
        visibility_ceiling=dataset_contract.visibility_ceiling,
        split_contract=split_contract,
        causal_contract=causal_contract,
        control_write_scope=control_write_scope,
        capabilities=dataset_binding_compiler_capability_bindings(composition),
        expected_invariants=invariants,
        work_envelope=DatasetWorkEnvelope(
            resources=ResourceBudget(
                cpu_cores=1,
                memory_bytes=64 * 1024**2,
                gpu_devices=0,
                wall_time_seconds=120,
                source_scan_bytes=0,
                output_bytes=0,
            ),
            files=DatasetFileLimits(
                max_source_files=0,
                max_destination_files=0,
                max_archive_members=0,
                max_single_file_bytes=0,
            ),
            control=DatasetControlLimits(
                max_manifest_bytes=10_000_000,
                max_metadata_records=5,
                max_receipt_bytes=10_000_000,
                max_reason_codes=64,
            ),
        ),
        exception_codes=exception_codes,
    )


__all__ = [
    "DATASET_BINDING_COMPILER_CLOSURE_ID",
    "DATASET_BINDING_COMPILER_KEY",
    "DATASET_BINDING_COMPILER_VERSION",
    "DATASET_BINDING_CONFIG_ARTIFACT_ID",
    "DATASET_BINDING_CONFIG_KEY",
    "DATASET_BINDING_ENVIRONMENT_KEY",
    "DATASET_BINDING_RUNTIME_KEY",
    "DatasetBindingCompilerConfig",
    "DatasetBindingCompilerStaticComposition",
    "build_dataset_binding_compiler_static_composition",
    "build_proposed_experiment_dataset_binding_manifest",
    "dataset_binding_compiler_capability_bindings",
    "dataset_binding_compiler_config",
]
