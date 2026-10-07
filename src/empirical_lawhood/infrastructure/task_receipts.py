"""Canonical external task-receipt store and strict reconciliation decoder."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalizationError, validate_document_shape
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactGenericValidation,
    ArtifactManifest,
    ArtifactLineageParent,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactPublicationBinding,
    ArtifactPublicationIntent,
    ArtifactPublicationMember,
    ArtifactPublicationScope,
    ArtifactSemanticValidation,
    ArtifactWriteRequest,
    ArtifactWriteResult,
    CanonicalTaskReceipt,
    LogicalArtifactIdentity,
    ReceiptCheck,
)

from .artifacts import ArtifactIdentityConflict, ExternalArtifactPlane
from .bounded_io import (
    BoundedFileIOError,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_ARTIFACT_PUBLICATION_INTENT_BYTES,
    MAX_CONTROL_PLANE_JSON_BYTES,
    read_bounded_bytes,
)













def _read_control_plane_json(path: Path) -> bytes:
    try:
        return read_bounded_bytes(path, maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES)
    except (BoundedFileIOError, OSError) as error:
        raise ArtifactIdentityConflict("task receipt control-plane file is invalid") from error


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CanonicalizationError(f"{field_name} must be a string-keyed mapping")
    return value


def _record(
    value: object,
    *,
    schema: str,
    version: str,
    fields: frozenset[str],
) -> Mapping[str, object]:
    return validate_document_shape(
        _mapping(value, schema),
        expected_schema=schema,
        expected_version=version,
        field_names=fields,
    )


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise CanonicalizationError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    return None if value is None else _string(value, field_name)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalizationError(f"{field_name} must be an integer")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CanonicalizationError(f"{field_name} must be a boolean")
    return value


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise CanonicalizationError(f"{field_name} must be a JSON array")
    return value


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(_string(item, field_name) for item in _list(value, field_name))


def _materialization(document: object) -> ArtifactMaterialization:
    value = _record(
        document,
        schema=ArtifactMaterialization.SCHEMA,
        version=ArtifactMaterialization.VERSION,
        fields=frozenset(
            {
                "materialization_id",
                "logical_artifact_id",
                "storage_root_id",
                "relative_path",
                "physical_sha256",
                "size_bytes",
                "compression",
                "partition_selector",
            }
        ),
    )
    return ArtifactMaterialization(
        materialization_id=_string(value["materialization_id"], "materialization_id"),
        logical_artifact_id=_string(value["logical_artifact_id"], "logical_artifact_id"),
        storage_root_id=_string(value["storage_root_id"], "storage_root_id"),
        relative_path=_string(value["relative_path"], "relative_path"),
        physical_sha256=_string(value["physical_sha256"], "physical_sha256"),
        size_bytes=_integer(value["size_bytes"], "size_bytes"),
        compression=_string(value["compression"], "compression"),
        partition_selector=_optional_string(value["partition_selector"], "partition_selector"),
    )


def _receipt_check(document: object) -> ReceiptCheck:
    value = _record(
        document,
        schema=ReceiptCheck.SCHEMA,
        version=ReceiptCheck.VERSION,
        fields=frozenset({"check_id", "passed", "reason_codes"}),
    )
    return ReceiptCheck(
        check_id=_string(value["check_id"], "check_id"),
        passed=_boolean(value["passed"], "passed"),
        reason_codes=_strings(value["reason_codes"], "reason_codes"),
    )


def _json_document(payload: bytes, label: str) -> object:
    if not payload or len(payload) > MAX_CONTROL_PLANE_JSON_BYTES:
        raise CanonicalizationError(f"{label} violates its control-plane byte limit")
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise CanonicalizationError(f"{label} is not canonical UTF-8 JSON") from error


def decode_task_receipt(payload: bytes) -> CanonicalTaskReceipt:
    document = _json_document(payload, "task receipt")
    mapping = _mapping(document, "task receipt")
    observed_schema = _string(mapping.get("schema"), "schema")
    if observed_schema != CanonicalTaskReceipt.SCHEMA:
        raise CanonicalizationError("task receipt has an unsupported schema")
    fields = {
        "receipt_id",
        "run_id",
        "task_id",
        "attempt_id",
        "implementation_commit",
        "input_materialization_ids",
        "output_materializations",
        "checks",
        "operational_status",
        "reason_codes",
    }
    fields.add("output_logical_artifacts")
    value = _record(
        document,
        schema=observed_schema,
        version=CanonicalTaskReceipt.VERSION,
        fields=frozenset(fields),
    )
    try:
        status = OperationalStatus(_string(value["operational_status"], "operational_status"))
    except ValueError as error:
        raise CanonicalizationError("unsupported receipt operational status") from error
    receipt = CanonicalTaskReceipt(
        receipt_id=_string(value["receipt_id"], "receipt_id"),
        run_id=_string(value["run_id"], "run_id"),
        task_id=_string(value["task_id"], "task_id"),
        attempt_id=_string(value["attempt_id"], "attempt_id"),
        implementation_commit=_string(value["implementation_commit"], "implementation_commit"),
        input_materialization_ids=_strings(
            value["input_materialization_ids"], "input_materialization_ids"
        ),
        output_materializations=tuple(
            (
                _materialization(item)
                for item in _list(value["output_materializations"], "output_materializations")
            )
        ),
        output_logical_artifacts=tuple(
            (
                _logical_identity(item)
                for item in _list(value["output_logical_artifacts"], "output_logical_artifacts")
            )
        ),
        checks=tuple((_receipt_check(item) for item in _list(value["checks"], "checks"))),
        operational_status=status,
        reason_codes=_strings(value["reason_codes"], "reason_codes"),
    )
    canonical = receipt.canonical_bytes()
    if canonical != payload:
        raise CanonicalizationError("task receipt bytes are not canonical")
    return receipt


def _logical_identity(document: object) -> LogicalArtifactIdentity:
    fields = {
        "logical_artifact_id",
        "content_sha256",
        "payload_schema",
        "profile",
        "media_type",
        "visibility_ceiling",
        "parent_visibility_ceilings",
        "outcome_access",
    }
    fields.add("lineage_parents")
    fields.add("semantic_validation")
    fields.add("generic_validation")
    value = _record(
        document,
        schema=LogicalArtifactIdentity.SCHEMA,
        version=LogicalArtifactIdentity.VERSION,
        fields=frozenset(fields),
    )
    try:
        profile = ArtifactProfile(_string(value["profile"], "profile"))
        visibility = VisibilityCeiling(_string(value["visibility_ceiling"], "visibility_ceiling"))
        parents = tuple(
            (
                VisibilityCeiling(_string(item, "parent_visibility_ceilings"))
                for item in _list(
                    value["parent_visibility_ceilings"], "parent_visibility_ceilings"
                )
            )
        )
        outcome_access = OutcomeAccess(_string(value["outcome_access"], "outcome_access"))
    except ValueError as error:
        raise CanonicalizationError("unsupported artifact-manifest enum") from error
    logical_artifact_id = _string(value["logical_artifact_id"], "logical_artifact_id")
    content_sha256 = _string(value["content_sha256"], "content_sha256")
    payload_schema = _string(value["payload_schema"], "payload_schema")
    media_type = _string(value["media_type"], "media_type")
    generic_validation = _generic_validation(value["generic_validation"])
    lineage_parents = tuple(
        (_lineage_parent(item) for item in _list(value["lineage_parents"], "lineage_parents"))
    )
    semantic_validation = (
        _semantic_validation(value["semantic_validation"])
        if value["semantic_validation"] is not None
        else None
    )
    return LogicalArtifactIdentity(
        logical_artifact_id=logical_artifact_id,
        content_sha256=content_sha256,
        payload_schema=payload_schema,
        profile=profile,
        media_type=media_type,
        visibility_ceiling=visibility,
        parent_visibility_ceilings=parents,
        outcome_access=outcome_access,
        generic_validation=generic_validation,
        lineage_parents=lineage_parents,
        semantic_validation=semantic_validation,
    )


def _generic_validation(document: object) -> ArtifactGenericValidation:
    value = _record(
        document,
        schema=ArtifactGenericValidation.SCHEMA,
        version=ArtifactGenericValidation.VERSION,
        fields=frozenset(
            {
                "validator_key",
                "validator_version",
                "validator_implementation_sha256",
                "payload_schema",
                "profile",
            }
        ),
    )
    try:
        profile = ArtifactProfile(_string(value["profile"], "profile"))
    except ValueError as error:
        raise CanonicalizationError("unsupported generic-validation profile") from error
    return ArtifactGenericValidation(
        validator_key=_string(value["validator_key"], "validator_key"),
        validator_version=_string(value["validator_version"], "validator_version"),
        validator_implementation_sha256=_string(
            value["validator_implementation_sha256"],
            "validator_implementation_sha256",
        ),
        payload_schema=_string(value["payload_schema"], "payload_schema"),
        profile=profile,
    )


def _semantic_validation(document: object) -> ArtifactSemanticValidation:
    value = _record(
        document,
        schema=ArtifactSemanticValidation.SCHEMA,
        version=ArtifactSemanticValidation.VERSION,
        fields=frozenset(
            {
                "validator_key",
                "validator_version",
                "validator_implementation_sha256",
                "payload_schema",
                "profile",
                "top_level_keys",
                "value_keys",
                "field_bindings",
            }
        ),
    )
    try:
        profile = ArtifactProfile(_string(value["profile"], "profile"))
    except ValueError as error:
        raise CanonicalizationError("unsupported semantic-validation profile") from error
    bindings: list[tuple[str, str]] = []
    for item in _list(value["field_bindings"], "field_bindings"):
        pair = _list(item, "field_bindings item")
        if len(pair) != 2:
            raise CanonicalizationError("field_bindings items must contain two strings")
        bindings.append(
            (
                _string(pair[0], "field_bindings key"),
                _string(pair[1], "field_bindings value"),
            )
        )
    return ArtifactSemanticValidation(
        validator_key=_string(value["validator_key"], "validator_key"),
        validator_version=_string(value["validator_version"], "validator_version"),
        validator_implementation_sha256=_string(
            value["validator_implementation_sha256"],
            "validator_implementation_sha256",
        ),
        payload_schema=_string(value["payload_schema"], "payload_schema"),
        profile=profile,
        top_level_keys=_strings(value["top_level_keys"], "top_level_keys"),
        value_keys=_strings(value["value_keys"], "value_keys"),
        field_bindings=tuple(bindings),
    )


def _publication_binding(document: object) -> ArtifactPublicationBinding:
    value = _record(
        document,
        schema=ArtifactPublicationBinding.SCHEMA,
        version=ArtifactPublicationBinding.VERSION,
        fields=frozenset(
            {
                "publication_batch_id",
                "publication_scope",
                "commit_marker_relative_path",
                "commit_marker_sha256",
                "commit_marker_size_bytes",
                "members",
            }
        ),
    )
    return ArtifactPublicationBinding(
        publication_batch_id=_string(
            value["publication_batch_id"],
            "publication_batch_id",
        ),
        publication_scope=_publication_scope(value["publication_scope"]),
        commit_marker_relative_path=_string(
            value["commit_marker_relative_path"],
            "commit_marker_relative_path",
        ),
        commit_marker_sha256=_string(
            value["commit_marker_sha256"],
            "commit_marker_sha256",
        ),
        commit_marker_size_bytes=_integer(
            value["commit_marker_size_bytes"],
            "commit_marker_size_bytes",
        ),
        members=tuple(
            _publication_member(item) for item in _list(value["members"], "publication members")
        ),
    )


def _publication_member(document: object) -> ArtifactPublicationMember:
    value = _record(
        document,
        schema=ArtifactPublicationMember.SCHEMA,
        version=ArtifactPublicationMember.VERSION,
        fields=frozenset(
            {
                "materialization_id",
                "logical_artifact_id",
                "logical_identity_sha256",
                "storage_root_id",
                "relative_path",
                "physical_sha256",
                "size_bytes",
                "visibility_ceiling",
                "outcome_access",
            }
        ),
    )
    try:
        visibility = VisibilityCeiling(_string(value["visibility_ceiling"], "visibility_ceiling"))
        access = OutcomeAccess(_string(value["outcome_access"], "outcome_access"))
    except ValueError as error:
        raise CanonicalizationError("unsupported publication-member evidence class") from error
    return ArtifactPublicationMember(
        materialization_id=_string(value["materialization_id"], "materialization_id"),
        logical_artifact_id=_string(value["logical_artifact_id"], "logical_artifact_id"),
        logical_identity_sha256=_string(
            value["logical_identity_sha256"],
            "logical_identity_sha256",
        ),
        storage_root_id=_string(value["storage_root_id"], "storage_root_id"),
        relative_path=_string(value["relative_path"], "relative_path"),
        physical_sha256=_string(value["physical_sha256"], "physical_sha256"),
        size_bytes=_integer(value["size_bytes"], "size_bytes"),
        visibility_ceiling=visibility,
        outcome_access=access,
    )


def _publication_scope(document: object) -> ArtifactPublicationScope:
    value = _record(
        document,
        schema=ArtifactPublicationScope.SCHEMA,
        version=ArtifactPublicationScope.VERSION,
        fields=frozenset(
            {
                "publication_scope_id",
                "storage_root_id",
                "relative_root",
                "visibility_ceiling",
                "outcome_access",
            }
        ),
    )
    try:
        visibility = VisibilityCeiling(_string(value["visibility_ceiling"], "visibility_ceiling"))
        access = OutcomeAccess(_string(value["outcome_access"], "outcome_access"))
    except ValueError as error:
        raise CanonicalizationError("unsupported publication-scope evidence class") from error
    return ArtifactPublicationScope(
        publication_scope_id=_string(
            value["publication_scope_id"],
            "publication_scope_id",
        ),
        storage_root_id=_string(value["storage_root_id"], "storage_root_id"),
        relative_root=_string(value["relative_root"], "relative_root"),
        visibility_ceiling=visibility,
        outcome_access=access,
    )


def decode_artifact_publication_intent(payload: bytes) -> ArtifactPublicationIntent:
    if len(payload) > MAX_ARTIFACT_PUBLICATION_INTENT_BYTES:
        raise CanonicalizationError("artifact publication intent exceeds its byte limit")
    document = _json_document(payload, "artifact publication intent")
    value = _record(
        document,
        schema=ArtifactPublicationIntent.SCHEMA,
        version=ArtifactPublicationIntent.VERSION,
        fields=frozenset(
            {
                "publication_batch_id",
                "publication_scope",
                "members",
                "absent_relative_paths",
            }
        ),
    )
    intent = ArtifactPublicationIntent(
        publication_batch_id=_string(
            value["publication_batch_id"],
            "publication_batch_id",
        ),
        publication_scope=_publication_scope(value["publication_scope"]),
        members=tuple(
            _publication_member(item) for item in _list(value["members"], "publication members")
        ),
        absent_relative_paths=_strings(
            value["absent_relative_paths"],
            "absent_relative_paths",
        ),
    )
    if intent.canonical_bytes() != payload:
        raise CanonicalizationError("artifact publication intent bytes are not canonical")
    return intent


def _object_identity(document: object) -> ObjectIdentity:
    value = _record(
        document,
        schema=ObjectIdentity.SCHEMA,
        version=ObjectIdentity.VERSION,
        fields=frozenset({"object_id", "object_schema", "object_version", "object_fingerprint"}),
    )
    return ObjectIdentity(
        object_id=_string(value["object_id"], "object_id"),
        object_schema=_string(value["object_schema"], "object_schema"),
        object_version=_string(value["object_version"], "object_version"),
        object_fingerprint=_string(value["object_fingerprint"], "object_fingerprint"),
    )


def _lineage_parent(document: object) -> ArtifactLineageParent:
    value = _record(
        document,
        schema=ArtifactLineageParent.SCHEMA,
        version=ArtifactLineageParent.VERSION,
        fields=frozenset({"identity", "visibility_ceiling", "outcome_access"}),
    )
    try:
        visibility = VisibilityCeiling(_string(value["visibility_ceiling"], "visibility_ceiling"))
        access = OutcomeAccess(_string(value["outcome_access"], "outcome_access"))
    except ValueError as error:
        raise CanonicalizationError("unsupported artifact-lineage enum") from error
    return ArtifactLineageParent(
        identity=_object_identity(value["identity"]),
        visibility_ceiling=visibility,
        outcome_access=access,
    )


def decode_artifact_manifest(payload: bytes) -> ArtifactManifest:
    if len(payload) > MAX_ARTIFACT_MANIFEST_BYTES:
        raise CanonicalizationError("artifact manifest exceeds its byte limit")
    document = _json_document(payload, "artifact manifest")
    mapping = _mapping(document, "artifact manifest")
    observed_schema = _string(mapping.get("schema"), "schema")
    if observed_schema != ArtifactManifest.SCHEMA:
        raise CanonicalizationError("artifact manifest has an unsupported schema")
    fields = {"logical", "materialization"}
    fields.add("publication")
    value = _record(
        document,
        schema=observed_schema,
        version=ArtifactManifest.VERSION,
        fields=frozenset(fields),
    )
    manifest = ArtifactManifest(
        logical=_logical_identity(value["logical"]),
        materialization=_materialization(value["materialization"]),
        publication=_publication_binding(value["publication"])
        if value["publication"] is not None
        else None,
    )
    if manifest.publication is None:
        raise CanonicalizationError("current artifact manifest lacks a publication binding")
    canonical = manifest.canonical_bytes()
    if canonical != payload:
        raise CanonicalizationError("artifact manifest bytes are not canonical")
    return manifest


class ExternalTaskReceiptStore:
    def __init__(
        self,
        plane: ExternalArtifactPlane,
        *,
        minimum_free_bytes: int = 0,
    ) -> None:
        if minimum_free_bytes < 0:
            raise ValueError("receipt free-space floor must be nonnegative")
        self.plane = plane
        self.minimum_free_bytes = minimum_free_bytes

    @staticmethod
    def _relative_path(run_id: str, task_id: str, attempt_id: str) -> str:
        return f"runs/{run_id}/receipts/{task_id}/{attempt_id}.json"

    def commit(
        self,
        receipt: CanonicalTaskReceipt,
        *,
        visibility_ceiling: VisibilityCeiling,
        outcome_access: OutcomeAccess,
    ) -> ArtifactWriteResult:
        if len(receipt.output_logical_artifacts) != len(receipt.output_materializations):
            raise ArtifactIdentityConflict(
                "new task receipts require exact logical output identities"
            )
        lineage_parents = tuple(
            sorted(
                (
                    ArtifactLineageParent(
                        identity=ObjectIdentity.from_record(
                            logical.logical_artifact_id,
                            logical,
                        ),
                        visibility_ceiling=logical.visibility_ceiling,
                        outcome_access=logical.outcome_access,
                    )
                    for logical in receipt.output_logical_artifacts
                ),
                key=lambda parent: (
                    parent.identity.object_id,
                    parent.identity.object_schema,
                    parent.identity.object_version,
                    parent.identity.object_fingerprint,
                ),
            )
        )
        return self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=f"task-receipt.{receipt.receipt_id}",
                relative_path=self._relative_path(
                    receipt.run_id,
                    receipt.task_id,
                    receipt.attempt_id,
                ),
                payload_schema=receipt.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"task-receipts.{receipt.run_id}",
                publication_scope_relative_root=f"runs/{receipt.run_id}/receipts",
                payload=receipt.canonical_bytes(),
                visibility_ceiling=visibility_ceiling,
                parent_visibility_ceilings=tuple(
                    parent.visibility_ceiling for parent in lineage_parents
                ),
                outcome_access=outcome_access,
                lineage_parents=lineage_parents,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )

    def read_by_receipt_id(
        self,
        run_id: str,
        task_id: str,
        receipt_id: str,
    ) -> CanonicalTaskReceipt | None:
        """Read the current scheduler's exact committed receipt without scanning.

        This lookup conveys no execution or retry authority. Existing publication
        checks authenticate the selected attempt and its logical receipt identity.
        """
        from empirical_lawhood.kernel.serialization import validate_stable_id

        for name, value in (
            ("run_id", run_id),
            ("task_id", task_id),
            ("receipt_id", receipt_id),
        ):
            validate_stable_id(value, field_name=name)
        if not receipt_id.startswith("receipt."):
            raise ArtifactIdentityConflict("current receipt lacks its scheduler locator")
        attempt_id = receipt_id.removeprefix("receipt.")
        validate_stable_id(attempt_id, field_name="attempt_id")
        if not attempt_id.startswith(f"{run_id}.{task_id}."):
            raise ArtifactIdentityConflict("receipt locator substitutes its run or task")
        receipt = self.read(run_id, task_id, attempt_id)
        if receipt is not None and receipt.receipt_id != receipt_id:
            raise ArtifactIdentityConflict("receipt locator differs from committed identity")
        return receipt

    def read(
        self,
        run_id: str,
        task_id: str,
        attempt_id: str,
    ) -> CanonicalTaskReceipt | None:
        relative_path = self._relative_path(run_id, task_id, attempt_id)
        path = self.plane.root.resolve(relative_path, for_write=False)
        manifest_path = self.plane.root.resolve(f"{relative_path}.manifest.json", for_write=False)
        if not path.exists() and not manifest_path.exists():
            return None
        if not path.is_file() or not manifest_path.is_file():
            raise ArtifactIdentityConflict("task receipt/manifest pair is partial")
        manifest = decode_artifact_manifest(_read_control_plane_json(manifest_path))
        if (
            manifest.materialization.relative_path != relative_path
            or manifest.materialization.size_bytes > MAX_CONTROL_PLANE_JSON_BYTES
            or manifest.logical.payload_schema != CanonicalTaskReceipt.SCHEMA
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.media_type != "application/json"
        ):
            raise ArtifactIdentityConflict("task receipt manifest violates its bounded contract")
        self.plane.verify_manifest(manifest)
        receipt = decode_task_receipt(_read_control_plane_json(path))
        if (receipt.run_id, receipt.task_id, receipt.attempt_id) != (
            run_id,
            task_id,
            attempt_id,
        ):
            raise ArtifactIdentityConflict("task receipt path identity differs")
        expected_logical_id = f"task-receipt.{receipt.receipt_id}"
        if manifest.logical.logical_artifact_id != expected_logical_id:
            raise ArtifactIdentityConflict("task receipt logical identity differs")
        if manifest.logical.content_sha256 != receipt.fingerprint():
            raise ArtifactIdentityConflict("task receipt content identity differs")
        return receipt
