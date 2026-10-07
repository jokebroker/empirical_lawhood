"""Trusted compilation and campaign resolution of exact dataset bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeAlias

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.dataset_manifests import (
    EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE,
    DatasetCapabilityKind,
    DatasetOutputContract,
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
    ExperimentDatasetBindingSet,
    ProposedExperimentDatasetBindingManifest,
)
from empirical_lawhood.planning.dataset_authority import (
    DatasetOperationAuthorization,
    DatasetOperationPolicy,
    DatasetOperationRequest,
)
from empirical_lawhood.planning.datasets import (
    BindingState,
    CustodyState,
    DatasetBindingRole,
    DatasetMaterialization,
    DatasetMaterializationClass,
    EvidenceReference,
    EvidenceReferenceKind,
    ExperimentDatasetBinding,
    RegisteredImplementationIdentity,
)
from empirical_lawhood.runtime.dataset_registration import (
    DatasetRegistrationInstallation,
    DatasetRegistrationPublicationReceipt,
)
from empirical_lawhood.runtime.dataset_transformation import (
    DatasetTransformationInstallation,
    DatasetTransformationPublicationReceipt,
)
from empirical_lawhood.runtime.dataset_operations import DatasetOperationAuthorityBundle
from empirical_lawhood.runtime.datasets import (
    DatasetBindingCompilerRegistration,
    DatasetCatalogProjectionAnchor,
    DatasetCatalogSnapshot,
)
from empirical_lawhood.runtime.plans import ExternalInputSpec, ScientificStage


DatasetSourcePublicationReceipt: TypeAlias = (
    DatasetRegistrationPublicationReceipt | DatasetTransformationPublicationReceipt
)
DatasetSourceManifest: TypeAlias = DatasetRegistrationManifest | DatasetTransformationManifest


class DatasetBindingCompilationError(ValueError):
    """Typed fail-closed binding error safe to surface at compilation."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


class DatasetCampaignBindingResolver(Protocol):
    def resolve(
        self,
        experiment: ExperimentSpec,
        *,
        run_id: str,
    ) -> tuple[ExperimentDatasetBinding, ...]: ...


@dataclass(frozen=True, slots=True)
class DatasetBindingSourceAuthority:
    """Process-local, exact source closure loaded by trusted infrastructure.

    Receipt and manifest bodies remain typed canonical records.  The caller
    cannot replace their identities with a boolean assertion.
    """

    publication_receipt: DatasetSourcePublicationReceipt
    publication_evidence: EvidenceReference
    source_manifest: DatasetSourceManifest
    dataset_contract: DatasetOutputContract
    contract_evidence: EvidenceReference | None
    lineage_record: CanonicalRecord | None
    lineage_evidence: EvidenceReference | None

    def __post_init__(self) -> None:
        receipt = self.publication_receipt
        if not isinstance(
            receipt,
            (DatasetRegistrationPublicationReceipt, DatasetTransformationPublicationReceipt),
        ):
            raise TypeError("publication_receipt has another operation type")
        if (
            not isinstance(self.publication_evidence, EvidenceReference)
            or self.publication_evidence.kind is not EvidenceReferenceKind.RECEIPT
            or self.publication_evidence.evidence_id != receipt.receipt_id
            or self.publication_evidence.evidence_schema != receipt.SCHEMA
            or self.publication_evidence.evidence_sha256 != receipt.fingerprint()
        ):
            raise ValueError("publication_evidence differs from the exact source receipt")
        if not isinstance(
            self.source_manifest,
            (DatasetRegistrationManifest, DatasetTransformationManifest),
        ):
            raise TypeError("source_manifest has another operation type")
        manifest_identity = ObjectIdentity.from_record(
            self.source_manifest.manifest_id,
            self.source_manifest,
        )
        if receipt.manifest != manifest_identity:
            raise ValueError("source receipt and source manifest identities differ")
        if not isinstance(self.dataset_contract, DatasetOutputContract):
            raise TypeError("dataset_contract must be a DatasetOutputContract")
        if (self.lineage_record is None) != (self.lineage_evidence is None):
            raise ValueError("lineage record and evidence must be supplied together")
        if self.lineage_record is not None:
            assert self.lineage_evidence is not None
            if (
                not isinstance(self.lineage_record, CanonicalRecord)
                or self.lineage_evidence.evidence_sha256 != self.lineage_record.fingerprint()
                or self.lineage_evidence.evidence_schema != self.lineage_record.SCHEMA
            ):
                raise ValueError("lineage evidence differs from its exact record")
        if isinstance(receipt, DatasetRegistrationPublicationReceipt):
            if not isinstance(self.source_manifest, DatasetRegistrationManifest):
                raise ValueError("registration receipt requires a registration manifest")
            if self.contract_evidence is None:
                raise ValueError("registered source binding requires contract evidence")
            contract_identity = ObjectIdentity.from_record(
                self.dataset_contract.output_id,
                self.dataset_contract,
            )
            if (
                self.contract_evidence.evidence_id != contract_identity.object_id
                or self.contract_evidence.evidence_schema != contract_identity.object_schema
                or self.contract_evidence.evidence_sha256 != contract_identity.object_fingerprint
                or self.contract_evidence not in receipt.supporting_evidence
            ):
                raise ValueError("contract evidence is absent from source registration closure")
            if self.lineage_evidence is not None and (
                self.lineage_evidence not in receipt.supporting_evidence
            ):
                raise ValueError("lineage evidence is absent from source registration closure")
        else:
            if not isinstance(self.source_manifest, DatasetTransformationManifest):
                raise ValueError("transformation receipt requires a transformation manifest")
            matching_outputs = tuple(
                output
                for output in self.source_manifest.outputs
                if output.proposed_materialization_id == receipt.materialization.object_id
            )
            if matching_outputs != (self.dataset_contract,):
                raise ValueError(
                    "transformation source does not contain the exact dataset contract"
                )
            if self.contract_evidence is not None or self.lineage_record is not None:
                raise ValueError("transformation source already carries contract and lineage")

    @property
    def materialization(self) -> ObjectIdentity:
        return self.publication_receipt.materialization

    @property
    def parent_materializations(self) -> tuple[ObjectIdentity, ...]:
        if isinstance(self.publication_receipt, DatasetTransformationPublicationReceipt):
            return self.publication_receipt.parent_materializations
        if self.lineage_record is None:
            return ()
        parents = getattr(self.lineage_record, "parent_materializations", None)
        if not isinstance(parents, tuple) or any(
            not isinstance(value, ObjectIdentity)
            or value.object_schema != DatasetMaterialization.SCHEMA
            for value in parents
        ):
            raise ValueError("registered lineage lacks exact materialization parents")
        return parents


def dataset_binding_source_authority_from_registration(
    installation: DatasetRegistrationInstallation,
    *,
    manifest: DatasetRegistrationManifest,
    dataset_contract: DatasetOutputContract,
    lineage_record: CanonicalRecord | None = None,
) -> DatasetBindingSourceAuthority:
    """Recover a typed source closure from one installed registration batch."""

    if not isinstance(installation, DatasetRegistrationInstallation):
        raise TypeError("installation must be a DatasetRegistrationInstallation")
    contract_evidence = next(
        (
            value
            for value in installation.receipt.supporting_evidence
            if value.evidence_id == dataset_contract.output_id
        ),
        None,
    )
    lineage_evidence = None
    if lineage_record is not None:
        lineage_id = getattr(lineage_record, "lineage_id", None)
        if not isinstance(lineage_id, str):
            raise ValueError("lineage record lacks a stable lineage_id")
        lineage_evidence = next(
            (
                value
                for value in installation.receipt.supporting_evidence
                if value.evidence_id == lineage_id
            ),
            None,
        )
        if lineage_evidence is None:
            raise ValueError("registration installation lacks exact lineage evidence")
    if contract_evidence is None:
        raise ValueError("registration installation lacks exact dataset contract evidence")
    return DatasetBindingSourceAuthority(
        publication_receipt=installation.receipt,
        publication_evidence=installation.receipt_evidence,
        source_manifest=manifest,
        dataset_contract=dataset_contract,
        contract_evidence=contract_evidence,
        lineage_record=lineage_record,
        lineage_evidence=lineage_evidence,
    )


def dataset_binding_source_authority_from_transformation(
    installation: DatasetTransformationInstallation,
    *,
    manifest: DatasetTransformationManifest,
    dataset_contract: DatasetOutputContract,
) -> DatasetBindingSourceAuthority:
    """Recover a typed source closure from one installed transformation batch."""

    if not isinstance(installation, DatasetTransformationInstallation):
        raise TypeError("installation must be a DatasetTransformationInstallation")
    return DatasetBindingSourceAuthority(
        publication_receipt=installation.receipt,
        publication_evidence=installation.receipt_evidence,
        source_manifest=manifest,
        dataset_contract=dataset_contract,
        contract_evidence=None,
        lineage_record=None,
        lineage_evidence=None,
    )


class RegisteredDatasetBindingCompiler:
    """Source-neutral compiler selected by one static registration."""

    def __init__(self, registration: DatasetBindingCompilerRegistration) -> None:
        if not isinstance(registration, DatasetBindingCompilerRegistration):
            raise TypeError("registration must be a DatasetBindingCompilerRegistration")
        self.registration = registration
        self.capability_key = registration.capability_key
        self.capability_version = registration.capability_version
        self.implementation_sha256 = registration.implementation_sha256
        self.capability_manifest_fingerprint = registration.capability.fingerprint()
        self.registration_fingerprint = registration.fingerprint()
        self.implementation = RegisteredImplementationIdentity(
            implementation_key=self.capability_key,
            implementation_version=self.capability_version,
            implementation_sha256=self.implementation_sha256,
        )

    def compile(
        self,
        manifest: ProposedExperimentDatasetBindingManifest,
        *,
        snapshot: DatasetCatalogSnapshot,
        source_authority: DatasetBindingSourceAuthority,
        binding_id: str,
        run_id: str | None = None,
    ) -> ExperimentDatasetBinding:
        if not isinstance(manifest, ProposedExperimentDatasetBindingManifest):
            raise TypeError("manifest must be a ProposedExperimentDatasetBindingManifest")
        if not isinstance(snapshot, DatasetCatalogSnapshot):
            raise TypeError("snapshot must be a DatasetCatalogSnapshot")
        if not isinstance(source_authority, DatasetBindingSourceAuthority):
            raise TypeError("source_authority must be a DatasetBindingSourceAuthority")
        compiler_bindings = tuple(
            value
            for value in manifest.capabilities
            if value.kind is DatasetCapabilityKind.BINDING_COMPILER
        )
        expected_capability = ObjectIdentity.from_record(
            self.capability_key,
            self.registration.capability,
        )
        if (
            len(compiler_bindings) != 1
            or compiler_bindings[0].registry_key != self.capability_key
            or compiler_bindings[0].implementation != expected_capability
        ):
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_COMPILER_IDENTITY_MISMATCH",
                "binding manifest selects another compiler implementation",
            )

        release = next(
            (value for value in snapshot.releases if value.release_id == manifest.release_id),
            None,
        )
        materialization = next(
            (
                value
                for value in snapshot.materializations
                if value.materialization_id == manifest.materialization.object_id
            ),
            None,
        )
        if release is None or materialization is None:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_TARGET_MISSING",
                "binding release or materialization is absent from the authenticated snapshot",
            )
        if ObjectIdentity.from_record(materialization.materialization_id, materialization) != (
            manifest.materialization
        ):
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_MATERIALIZATION_STALE",
                "binding materialization identity differs from the authenticated snapshot",
            )
        if materialization.release_id != release.release_id:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_RELEASE_MISMATCH",
                "binding materialization belongs to another release",
            )
        if materialization.custody_state is not CustodyState.VERIFIED:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_CUSTODY_STALE",
                "binding materialization lacks current VERIFIED custody",
            )
        if materialization.selector != manifest.selector:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_SELECTOR_MISMATCH",
                "binding selector differs from verified custody",
            )
        if (
            materialization.materialization_class is not manifest.materialization_class
            or materialization.evidence_class is not manifest.evidence_class
        ):
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_CLASSIFICATION_MISMATCH",
                "binding classification differs from verified custody",
            )
        if materialization.outcome_access is not manifest.outcome_access:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_OUTCOME_ACCESS_DOWNGRADE",
                "binding cannot restate or lower materialization outcome access",
            )
        if source_authority.materialization != manifest.materialization:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_SOURCE_AUTHORITY_MISMATCH",
                "source authority receipt identifies another materialization",
            )
        source_manifest = source_authority.source_manifest
        if source_manifest.outcome_access is not manifest.outcome_access:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_SOURCE_ACCESS_MISMATCH",
                "binding outcome access differs from its authorized source operation",
            )
        if not manifest.visibility_ceiling.is_at_least_as_restrictive_as(
            source_manifest.visibility_ceiling
        ):
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_VISIBILITY_DOWNGRADE",
                "binding visibility is less restrictive than its source operation",
            )
        if (
            ObjectIdentity.from_record(
                source_authority.dataset_contract.output_id,
                source_authority.dataset_contract,
            )
            != manifest.dataset_contract
        ):
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_CONTRACT_MISMATCH",
                "binding dataset contract differs from source authority",
            )
        source_transform = (
            ObjectIdentity.from_record(source_manifest.manifest_id, source_manifest)
            if isinstance(source_manifest, DatasetTransformationManifest)
            else None
        )
        if source_transform != manifest.transformation_manifest:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_TRANSFORM_LINEAGE_MISMATCH",
                "binding transformation identity differs from source authority",
            )
        parents = source_authority.parent_materializations
        parent_by_id = {value.materialization_id: value for value in snapshot.materializations}
        for parent in parents:
            observed = parent_by_id.get(parent.object_id)
            if observed is None:
                raise DatasetBindingCompilationError(
                    "DATASET_BINDING_PARENT_MISSING",
                    "binding source lineage has a missing parent materialization",
                )
            if ObjectIdentity.from_record(observed.materialization_id, observed) != parent:
                raise DatasetBindingCompilationError(
                    "DATASET_BINDING_PARENT_STALE",
                    "binding source lineage parent identity is stale",
                )
        if (
            materialization.materialization_class
            in {
                DatasetMaterializationClass.HISTORICAL_DERIVATIVE,
                DatasetMaterializationClass.TRANSFORMED_DERIVATIVE,
                DatasetMaterializationClass.GENERATED_OBSERVATION,
            }
            and not parents
        ):
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_PARENT_MISSING",
                "derived or generated binding lacks exact source parents",
            )

        transform_reference = None
        transform_implementation = None
        if isinstance(source_manifest, DatasetTransformationManifest):
            manifest_evidence = source_authority.publication_receipt.manifest_evidence
            transform_reference = EvidenceReference(
                evidence_id=manifest_evidence.evidence_id,
                kind=EvidenceReferenceKind.TRANSFORM,
                evidence_schema=manifest_evidence.evidence_schema,
                evidence_sha256=manifest_evidence.evidence_sha256,
                storage_root_id=manifest_evidence.storage_root_id,
                relative_locator=manifest_evidence.relative_locator,
            )
            if materialization.verifier is None:
                raise DatasetBindingCompilationError(
                    "DATASET_BINDING_TRANSFORM_IMPLEMENTATION_MISSING",
                    "transformed materialization lacks its registered verifier identity",
                )
            transform_implementation = materialization.verifier
        evidence_refs = tuple(
            sorted(
                (
                    source_authority.publication_evidence,
                    *((transform_reference,) if transform_reference is not None else ()),
                ),
                key=lambda value: value.evidence_id,
            )
        )
        return ExperimentDatasetBinding(
            binding_id=binding_id,
            experiment_spec_id=manifest.experiment.object_id,
            experiment=manifest.experiment,
            run_id=run_id,
            release_id=manifest.release_id,
            materialization_id=materialization.materialization_id,
            materialization=manifest.materialization,
            selector=manifest.selector,
            role=manifest.role,
            materialization_class=manifest.materialization_class,
            evidence_class=manifest.evidence_class,
            outcome_access=manifest.outcome_access,
            visibility_ceiling=manifest.visibility_ceiling,
            dataset_contract=manifest.dataset_contract,
            transformation_manifest=manifest.transformation_manifest,
            parent_materializations=parents,
            transform_reference=transform_reference,
            transform_implementation=transform_implementation,
            source_authority=source_authority.publication_evidence,
            binding_implementation=self.implementation,
            binding_state=BindingState.PROPOSED,
            reason_codes=(),
            binding_receipt=None,
            evidence_refs=evidence_refs,
        )


@dataclass(frozen=True, slots=True)
class DatasetBindingVerificationReceipt(CanonicalRecord):
    """Proof over a proposed binding, before the final record references it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-binding-verification-receipt'

    receipt_id: str
    manifest: ObjectIdentity
    proposed_binding: ObjectIdentity
    compiler: RegisteredImplementationIdentity
    source_authority: EvidenceReference
    verified_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            not isinstance(self.manifest, ObjectIdentity)
            or self.manifest.object_schema != ProposedExperimentDatasetBindingManifest.SCHEMA
        ):
            raise ValueError("manifest must identify a proposed binding manifest")
        if (
            not isinstance(self.proposed_binding, ObjectIdentity)
            or self.proposed_binding.object_schema != ExperimentDatasetBinding.SCHEMA
        ):
            raise ValueError("proposed_binding must identify an ExperimentDatasetBinding")
        if not isinstance(self.compiler, RegisteredImplementationIdentity):
            raise TypeError("compiler must be a RegisteredImplementationIdentity")
        if (
            not isinstance(self.source_authority, EvidenceReference)
            or self.source_authority.kind is not EvidenceReferenceKind.RECEIPT
        ):
            raise ValueError("source_authority must be receipt evidence")
        parse_utc_timestamp(self.verified_at_utc, field_name="verified_at_utc")


@dataclass(frozen=True, slots=True)
class DatasetBindingPublicationReceipt(CanonicalRecord):
    """Metadata-only commit record for one exact verified experiment binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-binding-publication-receipt'

    receipt_id: str
    authority_bundle_id: str
    manifest: ObjectIdentity
    policy: ObjectIdentity
    request: ObjectIdentity
    authorization: ObjectIdentity
    source_authority_evidence: EvidenceReference
    manifest_evidence: EvidenceReference
    proposed_binding_evidence: EvidenceReference
    binding_verification_evidence: EvidenceReference
    dataset_snapshot_evidence: EvidenceReference
    binding: ObjectIdentity
    bound_at_utc: str
    network_bytes: int
    dataset_bytes_written: bool
    scientific_verdict_issued: bool
    control_batch_published: bool
    catalog_written: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.authority_bundle_id, field_name="authority_bundle_id")
        for field_name, identity, schema in (
            ("manifest", self.manifest, ProposedExperimentDatasetBindingManifest.SCHEMA),
            ("policy", self.policy, DatasetOperationPolicy.SCHEMA),
            ("request", self.request, DatasetOperationRequest.SCHEMA),
            ("authorization", self.authorization, DatasetOperationAuthorization.SCHEMA),
            ("binding", self.binding, ExperimentDatasetBinding.SCHEMA),
        ):
            if not isinstance(identity, ObjectIdentity) or identity.object_schema != schema:
                raise ValueError(f"{field_name} has another object schema")
        for field_name, reference, kind, expected_schema in (
            (
                "source_authority_evidence",
                self.source_authority_evidence,
                EvidenceReferenceKind.RECEIPT,
                None,
            ),
            (
                "manifest_evidence",
                self.manifest_evidence,
                EvidenceReferenceKind.MANIFEST,
                ProposedExperimentDatasetBindingManifest.SCHEMA,
            ),
            (
                "proposed_binding_evidence",
                self.proposed_binding_evidence,
                EvidenceReferenceKind.MANIFEST,
                ExperimentDatasetBinding.SCHEMA,
            ),
            (
                "binding_verification_evidence",
                self.binding_verification_evidence,
                EvidenceReferenceKind.BINDING,
                DatasetBindingVerificationReceipt.SCHEMA,
            ),
            (
                "dataset_snapshot_evidence",
                self.dataset_snapshot_evidence,
                EvidenceReferenceKind.MANIFEST,
                DatasetCatalogSnapshot.SCHEMA,
            ),
        ):
            if (
                not isinstance(reference, EvidenceReference)
                or reference.kind is not kind
                or (expected_schema is not None and reference.evidence_schema != expected_schema)
            ):
                raise ValueError(f"{field_name} has another evidence contract")
        if (
            self.manifest_evidence.evidence_id != self.manifest.object_id
            or self.manifest_evidence.evidence_sha256 != self.manifest.object_fingerprint
        ):
            raise ValueError("manifest evidence differs from its exact identity")
        parse_utc_timestamp(self.bound_at_utc, field_name="bound_at_utc")
        if self.network_bytes != 0:
            raise ValueError("dataset binding publication must use zero network bytes")
        for field_name in ("dataset_bytes_written", "scientific_verdict_issued", "catalog_written"):
            if getattr(self, field_name) is not False:
                raise ValueError(f"dataset binding publication requires {field_name}=false")
        if self.control_batch_published is not True:
            raise ValueError("dataset binding publication must commit its control batch")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class DatasetBindingInstallation(CanonicalRecord):
    """Trusted result after new binding publication or exact restart replay."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/dataset-binding-installation'

    receipt: DatasetBindingPublicationReceipt
    receipt_evidence: EvidenceReference
    snapshot: DatasetCatalogSnapshot
    binding: ExperimentDatasetBinding
    replayed_existing: bool
    new_external_artifacts_created: bool
    catalog_written: bool
    network_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, DatasetBindingPublicationReceipt):
            raise TypeError("receipt must be a DatasetBindingPublicationReceipt")
        if (
            not isinstance(self.receipt_evidence, EvidenceReference)
            or self.receipt_evidence.kind is not EvidenceReferenceKind.RECEIPT
            or self.receipt_evidence.evidence_id != self.receipt.receipt_id
            or self.receipt_evidence.evidence_schema != self.receipt.SCHEMA
            or self.receipt_evidence.evidence_sha256 != self.receipt.fingerprint()
        ):
            raise ValueError("receipt_evidence differs from the binding receipt")
        if not isinstance(self.snapshot, DatasetCatalogSnapshot):
            raise TypeError("snapshot must be a DatasetCatalogSnapshot")
        if not isinstance(self.binding, ExperimentDatasetBinding):
            raise TypeError("binding must be an ExperimentDatasetBinding")
        matches = tuple(
            value for value in self.snapshot.bindings if value.binding_id == self.binding.binding_id
        )
        if matches != (self.binding,):
            raise ValueError("installation binding differs from its snapshot")
        if (
            ObjectIdentity.from_record(self.binding.binding_id, self.binding)
            != self.receipt.binding
        ):
            raise ValueError("installation binding differs from its receipt")
        if self.receipt.dataset_snapshot_evidence.evidence_sha256 != self.snapshot.fingerprint():
            raise ValueError("installation snapshot differs from its receipt")
        if self.replayed_existing and self.new_external_artifacts_created:
            raise ValueError("replayed binding cannot create new artifacts")
        if self.catalog_written is not False or self.network_bytes != 0:
            raise ValueError("binding installation must not write catalog or network")


class DatasetBindingInstallerPort(Protocol):
    def install(
        self,
        bundle: DatasetOperationAuthorityBundle,
        *,
        manifest: ProposedExperimentDatasetBindingManifest,
        base_snapshot: DatasetCatalogSnapshot,
        source_authority: DatasetBindingSourceAuthority,
        binding_id: str,
        receipt_id: str,
        run_id: str | None = None,
    ) -> DatasetBindingInstallation: ...


def resolve_experiment_dataset_bindings(
    *,
    experiment: ExperimentSpec,
    run_id: str,
    snapshot: DatasetCatalogSnapshot,
    anchor: DatasetCatalogProjectionAnchor,
) -> tuple[ExperimentDatasetBinding, ...]:
    """Resolve the exact precommitted binding set from authenticated projection state."""

    if not isinstance(experiment, ExperimentSpec):
        raise TypeError("experiment must be an ExperimentSpec")
    if not isinstance(snapshot, DatasetCatalogSnapshot):
        raise TypeError("snapshot must be a DatasetCatalogSnapshot")
    if not isinstance(anchor, DatasetCatalogProjectionAnchor) or not anchor.validates_snapshot(
        snapshot
    ):
        raise DatasetBindingCompilationError(
            "DATASET_BINDING_PROJECTION_STALE",
            "dataset projection anchor differs from the supplied snapshot",
        )
    experiment_identity = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    candidates = tuple(
        sorted(
            (
                value
                for value in snapshot.bindings
                if value.experiment_spec_id == experiment.experiment_id
                and value.run_id in {None, run_id}
            ),
            key=lambda value: value.binding_id,
        )
    )
    if any(value.experiment != experiment_identity for value in candidates):
        raise DatasetBindingCompilationError(
            "DATASET_BINDING_EXPERIMENT_IDENTITY_MISMATCH",
            "dataset binding names a stale or substituted experiment identity",
        )
    if any(value.binding_state is not BindingState.VERIFIED for value in candidates):
        raise DatasetBindingCompilationError(
            "DATASET_BINDING_NOT_VERIFIED",
            "experiment dataset binding has not reached VERIFIED state",
        )
    extension = next(
        (
            value
            for value in experiment.extensions
            if value.namespace == EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE
        ),
        None,
    )
    if extension is None:
        if candidates:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_UNCOMMITTED_INPUT",
                "authenticated bindings exist but the experiment did not precommit them",
            )
        return ()
    if extension.schema != ExperimentDatasetBindingSet.SCHEMA:
        raise DatasetBindingCompilationError(
            "DATASET_BINDING_EXTENSION_SCHEMA_MISMATCH",
            "experiment dataset extension uses another schema",
        )
    observed = ExperimentDatasetBindingSet.from_bindings(
        experiment_spec_id=experiment.experiment_id,
        bindings=candidates,
    )
    if observed.fingerprint() != extension.payload_sha256:
        raise DatasetBindingCompilationError(
            "DATASET_BINDING_REQUIREMENT_SET_MISMATCH",
            "authenticated bindings differ from the experiment's exact precommitment",
        )
    return candidates


_DEVELOPMENT_ROLES = frozenset(
    {
        DatasetBindingRole.DEVELOPMENT,
        DatasetBindingRole.CALIBRATION,
        DatasetBindingRole.COMPARATOR,
    }
)
_EVALUATOR_ROLES = frozenset(
    {
        DatasetBindingRole.EVALUATION_SEALED,
        DatasetBindingRole.EVALUATOR_REVEAL,
        DatasetBindingRole.GENERATED_TRUTH,
    }
)
_ALL_STAGE_ROLES = frozenset({DatasetBindingRole.AUXILIARY, DatasetBindingRole.PARENT})


def dataset_external_inputs_for_stage(
    bindings: tuple[ExperimentDatasetBinding, ...],
    stage: ScientificStage,
) -> tuple[ExternalInputSpec, ...]:
    """Lower only role-compatible exact bindings to immutable external inputs."""

    if any(not isinstance(value, ExperimentDatasetBinding) for value in bindings):
        raise TypeError("bindings contains another record type")
    evaluator_stage = stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
    allowed_roles = _ALL_STAGE_ROLES | (_EVALUATOR_ROLES if evaluator_stage else _DEVELOPMENT_ROLES)
    selected = tuple(value for value in bindings if value.role in allowed_roles)
    if any(
        not evaluator_stage
        and value.outcome_access
        in {
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }
        for value in selected
    ):
        raise DatasetBindingCompilationError(
            "DATASET_BINDING_OUTCOME_ACCESS_INCOMPATIBLE",
            "non-evaluator stage cannot receive sealed, revealed, or privileged bindings",
        )
    inputs = tuple(
        sorted(
            (
                ExternalInputSpec(
                    input_id=f"dataset.{value.binding_id}",
                    logical_artifact_id=value.materialization_id,
                    expected_content_sha256=None,
                    expected_payload_schema=None,
                    expected_media_type=None,
                    expected_size_bytes=None,
                    expected_visibility_ceiling=value.visibility_ceiling,
                    expected_outcome_access=value.outcome_access,
                    identity_scope_sha256=value.fingerprint(),
                )
                for value in selected
            ),
            key=lambda value: value.input_id,
        )
    )
    require_sorted_unique_ids(inputs, attribute="input_id", field_name="dataset inputs")
    return inputs


__all__ = [
    "DatasetCampaignBindingResolver",
    "DatasetBindingCompilationError",
    "DatasetBindingInstallation",
    "DatasetBindingInstallerPort",
    "DatasetBindingPublicationReceipt",
    "DatasetBindingSourceAuthority",
    "DatasetBindingVerificationReceipt",
    "RegisteredDatasetBindingCompiler",
    "dataset_external_inputs_for_stage",
    "dataset_binding_source_authority_from_registration",
    "dataset_binding_source_authority_from_transformation",
    "resolve_experiment_dataset_bindings",
]
