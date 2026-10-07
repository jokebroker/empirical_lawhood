from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.dataset_authority import (
    DATASET_TRANSFORMATION_MANIFEST_SCHEMA,
    DatasetOperationAuthorization,
    DatasetAuthorizationIssuerRegistration,
    DatasetAuthorizationSignatureAlgorithm,
    DatasetControlLimits,
    DatasetFileLimits,
    DatasetOperationPolicy,
    DatasetOperationRequest,
    DatasetStorageScope,
    DatasetWorkEnvelope,
    issue_dataset_operation_authorization,
)
from empirical_lawhood.planning.datasets import DatasetEvidenceClass, DatasetMaterializationClass
from empirical_lawhood.planning.study_authoring import CapabilitySelection
from empirical_lawhood.planning.source_pipelines import SourcePipelineArtifactProfile, SourcePipelineChangeSemantics, SourcePipelineCoordinate, SourcePipelineEdgeAccounting, SourcePipelineMode, SourcePipelineProfile, SourcePipelineTransformEdge
from empirical_lawhood.runtime.artifacts import (
    ArtifactGenericValidation,
    ArtifactManifest,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactPublicationBinding,
    ArtifactPublicationScope,
    LogicalArtifactIdentity,
    artifact_publication_batch_id,
    artifact_publication_commit_relative_path,
    artifact_publication_member,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCapabilityLimits,
    DatasetCapabilityRegistry,
    DatasetTransformRegistration,
)
from empirical_lawhood.runtime.source_pipelines import SourcePipelineExecutionEvidence, SourcePipelineCompilation, SourcePipelineValidatorEvidence, compile_source_pipeline
from empirical_lawhood.runtime.sources import (
    SourceCapabilityManifest,
    SourceMode,
    SourceResult,
)


class _Clock:
    def __init__(self, clock_id: str, timestamp: str) -> None:
        self.clock_id = clock_id
        self.timestamp = timestamp

    def now_utc(self) -> str:
        return self.timestamp


class _Signer:
    signature_algorithm = DatasetAuthorizationSignatureAlgorithm.ED25519
    signature_version = "1.0.0"

    def __init__(self) -> None:
        self._key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        self.verification_key_hex = (
            self._key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            .hex()
        )

    def sign(self, payload: bytes) -> bytes:
        return self._key.sign(payload)


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity(value: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=value,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=_digest(f"{value}:{schema}"),
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=1_000_000,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=1_000_000,
        output_bytes=1_000_000,
    )


def _coordinates() -> tuple[SourcePipelineCoordinate, ...]:
    return (
        SourcePipelineCoordinate(
            coordinate_id="coordinate.invalid-disposition",
            field_id="invalid-disposition",
            native_unit="1",
            frame_id="record",
            clock_id="source-clock",
        ),
        SourcePipelineCoordinate(
            coordinate_id="coordinate.physical-unit",
            field_id="physical-unit",
            native_unit="1",
            frame_id="preparation",
            clock_id="source-clock",
        ),
        SourcePipelineCoordinate(
            coordinate_id="coordinate.receiver",
            field_id="receiver",
            native_unit="dimensionless",
            frame_id="native-receiver",
            clock_id="source-clock",
        ),
    )


def _source_materialization(label: str) -> ArtifactMaterialization:
    return ArtifactMaterialization(
        materialization_id=f"materialization.source.{label}",
        logical_artifact_id=f"source.{label}",
        storage_root_id="storage.scientific",
        relative_path=f"sources/{label}/native.bin",
        physical_sha256=_digest(f"source-bytes:{label}"),
        size_bytes=128,
        compression="none",
    )


def _registries(
    source_format_profile_id: str,
) -> tuple[CapabilityRegistry, DatasetCapabilityRegistry, CapabilityManifest]:
    config_schema = 'empirical-lawhood/simulators/torax-native/source-normalization'
    transform = CapabilityManifest(
        capability_key="source-normalization",
        capability_version="1.0.0",
        kind=CapabilityKind.TRANSFORM,
        config_schema=config_schema,
        config_schema_sha256=_digest(config_schema),
        input_schema_ids=('empirical-lawhood/evidence/native-source',),
        output_schema_ids=('empirical-lawhood/evidence/canonical-source',),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-source-free-property-transport-source-pipeline",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=("preserve-native-evidence",),
        implementation_sha256=_digest("source-normalization-implementation"),
    )
    registration = DatasetTransformRegistration(
        capability=transform,
        accepted_media_types=("application/octet-stream",),
        produced_media_types=("application/vnd.apache.parquet",),
        source_format_profile_ids=(source_format_profile_id,),
        destination_format_profile_ids=('canonical-source',),
        limits=DatasetCapabilityLimits(
            max_input_bytes=1_000_000,
            max_output_bytes=1_000_000,
            max_records=10_000,
            max_files=2,
            max_archive_members=0,
        ),
    )
    return (
        CapabilityRegistry(
            registry_id="capability-registry.source-free-property-transport-source-pipeline",
            capabilities=(transform,),
        ),
        DatasetCapabilityRegistry(
            registry_id="dataset-registry.source-free-property-transport-source-pipeline",
            providers=(),
            inspectors=(),
            transforms=(registration,),
            binding_compilers=(),
            storage_verifiers=(),
            evidence_verifiers=(),
        ),
        transform,
    )


def _profile(
    *,
    label: str = "torax-native",
    mode: SourcePipelineMode = SourcePipelineMode.SIMULATED,
) -> tuple[
    SourcePipelineProfile,
    SourceCapabilityManifest,
    CapabilityRegistry,
    DatasetCapabilityRegistry,
]:
    source_format = f"{label}-native-source"
    capability_registry, dataset_registry, transform = _registries(source_format)
    coordinates = _coordinates()
    source_materialization = _source_materialization(label)
    validator = _identity(
        "validator.source-pipeline-semantics",
        'empirical-lawhood/runtime/artifact-semantic-validation',
    )
    edge = SourcePipelineTransformEdge(
        edge_id="edge.normalize-native-source",
        rank=0,
        dataset_transform_registry_id=transform.registry_id,
        selection=CapabilitySelection(
            capability_key=transform.capability_key,
            capability_version=transform.capability_version,
            implementation_sha256=transform.implementation_sha256,
        ),
        config=_identity("config.source-normalization", transform.config_schema),
        domain_schema='empirical-lawhood/evidence/native-source',
        codomain_schema='empirical-lawhood/evidence/canonical-source',
        source_media_type="application/octet-stream",
        destination_media_type="application/vnd.apache.parquet",
        source_format_profile_id=source_format,
        destination_format_profile_id='canonical-source',
        input_coordinates=coordinates,
        output_coordinates=coordinates,
        native_unit_semantics=SourcePipelineChangeSemantics.PRESERVED,
        native_unit_change_contract=None,
        frame_semantics=SourcePipelineChangeSemantics.PRESERVED,
        frame_change_contract=None,
        clock_semantics=SourcePipelineChangeSemantics.PRESERVED,
        clock_change_contract=None,
        row_semantics=SourcePipelineChangeSemantics.PRESERVED,
        row_change_contract=None,
        physical_unit_roster_preserved=True,
        invalid_rows_retained=True,
        semantic_validator=validator,
        resource_budget=_budget(),
    )
    profile = SourcePipelineProfile(
        profile_id=f"source-pipeline.{label}",
        profile_version="1.0.0",
        mode=mode,
        source_selection=CapabilitySelection(
            capability_key=f"source-adapter.{label}",
            capability_version="1.0.0",
            implementation_sha256=_digest(f"source-adapter:{label}"),
        ),
        source_config=_identity(
            f"config.source-adapter.{label}",
            'empirical-lawhood/methods/structural-transport/synthetic-source-adapter/config',
        ),
        source_id=f"source.{label}",
        source_materialization=ObjectIdentity.from_record(
            source_materialization.materialization_id,
            source_materialization,
        ),
        expected_source_sha256=source_materialization.physical_sha256,
        expected_source_size_bytes=source_materialization.size_bytes,
        licence_id="synthetic-fixture-only",
        operator_storage_root_id="storage.scientific",
        source_relative_locator=source_materialization.relative_path,
        source_schema=edge.domain_schema,
        source_media_type=edge.source_media_type,
        source_format_profile_id=edge.source_format_profile_id,
        input_coordinates=coordinates,
        transform_edges=(edge,),
        output_schema=edge.codomain_schema,
        output_media_type=edge.destination_media_type,
        output_filename_suffix=".parquet",
        output_artifact_profile=SourcePipelineArtifactProfile.PARQUET,
        output_coordinates=coordinates,
        semantic_validators=(validator,),
        physical_unit_field_id="physical-unit",
        invalid_disposition_field_id="invalid-disposition",
        causal_cutoff=_identity(
            f"cutoff.{label}",
            'empirical-lawhood/methods/structural-transport/synthetic-causal-cutoff-reference',
        ),
        dataset_transformation_manifest=_identity(
            f"transformation.{label}",
            DATASET_TRANSFORMATION_MANIFEST_SCHEMA,
        ),
        resource_ceiling=_budget(),
        maximum_row_count=10_000,
        maximum_physical_unit_count=100,
        scratch_namespace=f"scratch/source-pipelines/{label}",
        output_relative_locator=f"derived/{label}/canonical.parquet",
        evidence_world_id=(
            "deterministic-simulator" if mode is SourcePipelineMode.SIMULATED else "fixed-archive"
        ),
        observation_operator=_identity(
            f"observation-operator.{label}",
            'empirical-lawhood/planning/observation-operator',
        ),
        numerical_view_ids=("native-view",),
        receiver_semantics_id=f"receiver-semantics.{label}",
        validity_contract_id=f"validity-contract.{label}",
        uncertainty_contract_id=f"uncertainty-contract.{label}",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    source_manifest = SourceCapabilityManifest(
        capability_key=profile.source_selection.capability_key,
        capability_version=profile.source_selection.capability_version,
        mode=SourceMode.SIMULATED
        if mode is SourcePipelineMode.SIMULATED
        else SourceMode.ARCHIVAL,
        source_access=(
            SourceAccessClass.NONE
            if mode is SourcePipelineMode.SIMULATED
            else SourceAccessClass.PRIVATE_CREDENTIAL
        ),
        output_schema_ids=(profile.source_schema,),
        maximum_evidence=EvidenceCeiling.MEASUREMENT,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        deterministic=True,
        implementation_sha256=profile.source_selection.implementation_sha256,
        resource_budget=_budget(),
    )
    return profile, source_manifest, capability_registry, dataset_registry


def _storage_root() -> ObjectIdentity:
    return _identity(
        "storage.scientific",
        'empirical-lawhood/runtime/external-root-contract',
    )


def _dataset_request(profile: SourcePipelineProfile) -> DatasetOperationRequest:
    source_scope = DatasetStorageScope(
        scope_id="scope.source",
        storage_root=_storage_root(),
        relative_prefix="sources",
    )
    destination_scope = DatasetStorageScope(
        scope_id="scope.destination",
        storage_root=_storage_root(),
        relative_prefix="derived",
    )
    control_scope = DatasetStorageScope(
        scope_id="scope.control",
        storage_root=_storage_root(),
        relative_prefix="dataset-operations/source-pipeline",
    )
    envelope = DatasetWorkEnvelope(
        resources=_budget(),
        files=DatasetFileLimits(
            max_source_files=2,
            max_destination_files=2,
            max_archive_members=0,
            max_single_file_bytes=1_000_000,
        ),
        control=DatasetControlLimits(
            max_manifest_bytes=1_000_000,
            max_metadata_records=10_000,
            max_receipt_bytes=1_000_000,
            max_reason_codes=32,
        ),
    )
    policy = DatasetOperationPolicy(
        policy_id=f"policy.{profile.profile_id}",
        manifest=profile.dataset_transformation_manifest,
        action=AuthorityAction.DATASET_TRANSFORMATION,
        source_scopes=(source_scope,),
        destination_scope=destination_scope,
        control_write_scope=control_scope,
        work_envelope_ceiling=envelope,
        evidence_class=(
            DatasetEvidenceClass.SIMULATION
            if profile.mode is SourcePipelineMode.SIMULATED
            else DatasetEvidenceClass.EMPIRICAL_SOURCE
        ),
        materialization_class=DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
        world_kind=(
            WorldKind.NUMERICAL_SIMULATOR
            if profile.mode is SourcePipelineMode.SIMULATED
            else WorldKind.PHYSICAL_EXPERIMENT
        ),
        outcome_access=profile.outcome_access,
        visibility_ceiling=profile.visibility_ceiling,
        implementation_commit="1" * 40,
        issued_by="source-pipeline-policy-issuer",
        authorized_approver_id="source-pipeline-approver",
        valid_from_utc="2026-08-23T00:00:00Z",
        valid_until_utc="2026-08-23T03:00:00Z",
        reason_codes=("BOUNDED_SOURCE_PIPELINE",),
    )
    return DatasetOperationRequest(
        request_id=f"request.{profile.profile_id}",
        policy=ObjectIdentity.from_record(policy.policy_id, policy),
        manifest=profile.dataset_transformation_manifest,
        action=AuthorityAction.DATASET_TRANSFORMATION,
        source_scopes=(source_scope,),
        destination_scope=destination_scope,
        control_write_scope=control_scope,
        work_envelope=envelope,
        evidence_class=(
            DatasetEvidenceClass.SIMULATION
            if profile.mode is SourcePipelineMode.SIMULATED
            else DatasetEvidenceClass.EMPIRICAL_SOURCE
        ),
        materialization_class=DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
        world_kind=(
            WorldKind.NUMERICAL_SIMULATOR
            if profile.mode is SourcePipelineMode.SIMULATED
            else WorldKind.PHYSICAL_EXPERIMENT
        ),
        outcome_access=profile.outcome_access,
        visibility_ceiling=profile.visibility_ceiling,
        implementation_commit="1" * 40,
        requested_by="source-free-property-transport-source-pipeline",
        valid_from_utc="2026-08-23T00:00:00Z",
        valid_until_utc="2026-08-23T02:00:00Z",
        source_mutation_requested=False,
        download_requested=False,
        deletion_requested=False,
        reason_codes=("SYNTHETIC_SOURCE_PIPELINE_CONFORMANCE",),
    )


def _dataset_policy(
    profile: SourcePipelineProfile,
    request: DatasetOperationRequest,
) -> DatasetOperationPolicy:
    return DatasetOperationPolicy(
        policy_id=f"policy.{profile.profile_id}",
        manifest=request.manifest,
        action=request.action,
        source_scopes=request.source_scopes,
        destination_scope=request.destination_scope,
        control_write_scope=request.control_write_scope,
        work_envelope_ceiling=request.work_envelope,
        evidence_class=request.evidence_class,
        materialization_class=request.materialization_class,
        world_kind=request.world_kind,
        outcome_access=request.outcome_access,
        visibility_ceiling=request.visibility_ceiling,
        implementation_commit=request.implementation_commit,
        issued_by="source-pipeline-policy-issuer",
        authorized_approver_id="source-pipeline-approver",
        valid_from_utc="2026-08-23T00:00:00Z",
        valid_until_utc="2026-08-23T03:00:00Z",
        reason_codes=("BOUNDED_SOURCE_PIPELINE",),
    )


def _dataset_authority(
    profile: SourcePipelineProfile,
    request: DatasetOperationRequest,
) -> tuple[
    DatasetOperationPolicy,
    DatasetOperationAuthorization,
    DatasetAuthorizationIssuerRegistration,
]:
    policy = _dataset_policy(profile, request)
    signer = _Signer()
    issuer = DatasetAuthorizationIssuerRegistration(
        issuer_registration_id="issuer.source-free-property-transport-source-pipeline",
        issuer_id="source-pipeline-authority-issuer",
        allowed_actions=frozenset({AuthorityAction.DATASET_TRANSFORMATION}),
        implementation_sha256=_digest("source-pipeline-authority-issuer"),
        implementation_version="1.0.0",
        signature_algorithm=signer.signature_algorithm,
        signature_version=signer.signature_version,
        verification_key_hex=signer.verification_key_hex,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    authorization = issue_dataset_operation_authorization(
        policy=policy,
        request=request,
        authorization_id=f"authorization.{profile.profile_id}",
        approved_by=policy.authorized_approver_id,
        issuer_registration=issuer,
        signer=signer,
        clock=_Clock("source-pipeline-decision-clock", "2026-08-23T01:00:00Z"),
    )
    return policy, authorization, issuer


def _compile(
    profile: SourcePipelineProfile,
    source_manifest: SourceCapabilityManifest,
    capability_registry: CapabilityRegistry,
    dataset_registry: DatasetCapabilityRegistry,
) -> SourcePipelineCompilation:
    return compile_source_pipeline(
        profile,
        source_manifest=source_manifest,
        capability_registry=capability_registry,
        dataset_registry=dataset_registry,
        dataset_operation_request=_dataset_request(profile),
    )


def _output_manifest(profile: SourcePipelineProfile) -> ArtifactManifest:
    logical = LogicalArtifactIdentity(
        logical_artifact_id=f"output.{profile.profile_id}",
        content_sha256=_digest(f"canonical-output:{profile.profile_id}"),
        payload_schema=profile.output_schema,
        profile=ArtifactProfile.PARQUET,
        media_type=profile.output_media_type,
        visibility_ceiling=profile.visibility_ceiling,
        parent_visibility_ceilings=(),
        outcome_access=profile.outcome_access,
        generic_validation=ArtifactGenericValidation(
            validator_key="parquet-generic-validator",
            validator_version="1.0.0",
            validator_implementation_sha256=_digest("parquet-generic-validator"),
            payload_schema=profile.output_schema,
            profile=ArtifactProfile.PARQUET,
        ),
    )
    materialization = ArtifactMaterialization(
        materialization_id=f"materialization.{logical.logical_artifact_id}",
        logical_artifact_id=logical.logical_artifact_id,
        storage_root_id=profile.operator_storage_root_id,
        relative_path=profile.output_relative_locator,
        physical_sha256=_digest(f"physical-output:{profile.profile_id}"),
        size_bytes=256,
        compression="none",
    )
    scope = ArtifactPublicationScope(
        publication_scope_id=f"publication-scope.{profile.profile_id}",
        storage_root_id=profile.operator_storage_root_id,
        relative_root="derived",
        visibility_ceiling=profile.visibility_ceiling,
        outcome_access=profile.outcome_access,
    )
    members = (artifact_publication_member(logical, materialization),)
    batch_id = artifact_publication_batch_id(scope, members)
    publication = ArtifactPublicationBinding(
        publication_batch_id=batch_id,
        publication_scope=scope,
        commit_marker_relative_path=artifact_publication_commit_relative_path(
            scope,
            batch_id,
        ),
        commit_marker_sha256=_digest(f"commit:{batch_id}"),
        commit_marker_size_bytes=1,
        members=members,
    )
    return ArtifactManifest(
        logical=logical,
        materialization=materialization,
        publication=publication,
    )


def _execution_evidence(
    profile: SourcePipelineProfile,
    compilation: SourcePipelineCompilation,
) -> SourcePipelineExecutionEvidence:
    assert compilation.dataset_operation_request is not None
    policy, authorization, issuer = _dataset_authority(
        profile,
        compilation.dataset_operation_request,
    )
    source_materialization = _source_materialization(
        profile.profile_id.removeprefix("source-pipeline.")
    )
    source_result = SourceResult(
        request_id=compilation.source_request.request_id,
        source_id=profile.source_id,
        payload_schema=profile.source_schema,
        materialization=source_materialization,
        observed_sha256=source_materialization.physical_sha256,
        observed_size_bytes=source_materialization.size_bytes,
        custody_status="verified",
    )
    output_manifest = _output_manifest(profile)
    output_identity = ObjectIdentity.from_record(
        output_manifest.logical.logical_artifact_id,
        output_manifest,
    )
    roster_sha = _digest("physical-units:a,b")
    accounting = SourcePipelineEdgeAccounting(
        edge_id=profile.transform_edges[0].edge_id,
        input_artifact=profile.source_materialization,
        output_artifact=output_identity,
        input_row_count=12,
        output_row_count=12,
        input_invalid_row_count=2,
        output_invalid_row_count=2,
        physical_unit_count=2,
        input_physical_unit_roster_sha256=roster_sha,
        output_physical_unit_roster_sha256=roster_sha,
        dropped_without_disposition_count=0,
        observed_input_coordinates=profile.input_coordinates,
        observed_output_coordinates=profile.output_coordinates,
    )
    return SourcePipelineExecutionEvidence(
        evidence_id=f"execution-evidence.{profile.profile_id}",
        dataset_operation_policy=policy,
        dataset_operation_authorization=authorization,
        dataset_authorization_issuer=issuer,
        source_result=source_result,
        edge_accounting=(accounting,),
        output_manifest=output_manifest,
        validator_evidence=(
            SourcePipelineValidatorEvidence(
                validator=profile.semantic_validators[0],
                receipt=_identity(
                    f"validator-receipt.{profile.profile_id}",
                    'empirical-lawhood/methods/structural-transport/synthetic-semantic-validation-reference',
                ),
                passed=True,
            ),
        ),
        recovery=_identity(
            f"recovery.{profile.profile_id}",
            'empirical-lawhood/runtime/run-recovery-reference',
        ),
    )
