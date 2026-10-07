# SPDX-License-Identifier: MPL-2.0

"""prepared response grant replay in the existing guarded external publication store.

The operator selects this trust root separately from the held source tree.
Merely naming a JSON file (or a historical source grant) never imports authority.
Only owner tooling may populate this store; input checks are read-only.
"""

from pathlib import Path

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest

from .response_parent_custody import ResponseParentTargetGrant

_NAMESPACE = "authority/prepared-response-parent-grants"
_MAX_BYTES = 8 * 1024**2


class ResponseExternalParentAuthorityStore:
    """Resolve immutable target grants, including their publication manifests."""

    def __init__(self, plane: ExternalArtifactPlane) -> None:
        self.plane = plane

    def _relative(self, grant_id: str) -> str:
        validate_stable_id(grant_id, field_name="grant_id")
        return f"{_NAMESPACE}/{grant_id}.json"

    def resolve_grant(self, locator: Path) -> ResponseParentTargetGrant | None:
        if not locator.is_absolute() or locator.suffix != ".json":
            raise ValueError("RESPONSE_PARENT_TARGET_GRANT_LOCATOR_INVALID")
        relative = self._relative(locator.stem)
        expected = self.plane.root.resolve(relative, for_write=False)
        if locator != expected:
            raise ValueError("RESPONSE_PARENT_TARGET_GRANT_OUTSIDE_TRUSTED_STORE")
        sidecar = self.plane.root.resolve(relative + ".manifest.json", for_write=False)
        if not expected.is_file() or not sidecar.is_file():
            return None
        manifest = decode_artifact_manifest(
            read_bounded_bytes(sidecar, maximum_bytes=_MAX_BYTES)
        )
        if (
            manifest.logical.logical_artifact_id != locator.stem
            or manifest.logical.payload_schema != ResponseParentTargetGrant.SCHEMA
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.media_type != "application/json"
            or manifest.logical.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or manifest.logical.visibility_ceiling
            is not VisibilityCeiling.DEVELOPMENT_ONLY
            or manifest.materialization.relative_path != relative
            or manifest.materialization.size_bytes > _MAX_BYTES
        ):
            raise ValueError("RESPONSE_PARENT_TARGET_GRANT_PUBLICATION_MISMATCH")
        self.plane.verify_manifest(manifest)
        raw = read_bounded_bytes(expected, maximum_bytes=_MAX_BYTES)
        grant = decode_canonical_bytes(
            raw, ResponseParentTargetGrant, maximum_bytes=_MAX_BYTES
        )
        if (
            grant.grant_id != locator.stem
            or grant.canonical_bytes() != raw
            or grant.fingerprint() != manifest.logical.content_sha256
        ):
            raise ValueError("RESPONSE_PARENT_TARGET_GRANT_IDENTITY_MISMATCH")
        return grant

    def persist(self, grant: ResponseParentTargetGrant) -> ObjectIdentity:
        """Owner tooling only: persist or replay; this does not manufacture a grant."""
        if not isinstance(grant, ResponseParentTargetGrant):
            raise TypeError("RESPONSE_PARENT_TARGET_GRANT_TYPE_INVALID")
        relative = self._relative(grant.grant_id)
        locator = self.plane.root.resolve(relative, for_write=False)
        sidecar = self.plane.root.resolve(relative + ".manifest.json", for_write=False)
        if locator.exists() or sidecar.exists():
            if self.resolve_grant(locator) != grant:
                raise FileExistsError("RESPONSE_PARENT_TARGET_GRANT_IMMUTABLE")
        else:
            self.plane.write(
                ArtifactWriteRequest(
                    logical_artifact_id=grant.grant_id,
                    relative_path=relative,
                    payload_schema=grant.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    publication_scope_id=f"prepared-response-parent-authority.{grant.grant_id}",
                    publication_scope_relative_root=_NAMESPACE,
                    payload=grant.canonical_bytes(),
                    visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
                    parent_visibility_ceilings=(),
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                )
            )
        if self.resolve_grant(locator) != grant:
            raise RuntimeError("RESPONSE_PARENT_TARGET_GRANT_REPLAY_FAILED")
        return ObjectIdentity.from_record(grant.grant_id, grant)


__all__ = ['ResponseExternalParentAuthorityStore']
