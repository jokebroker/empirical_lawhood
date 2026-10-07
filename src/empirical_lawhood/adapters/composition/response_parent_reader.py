# SPDX-License-Identifier: MPL-2.0

"""Authenticated prepared response parent bytes; no schema or scientific identity rewriting."""

from dataclasses import dataclass
from pathlib import Path

from .response_parent_custody import ResponseParentTargetAuthorityStore, ResponseParentTargetGrant, replay_response_parent_custody
from .response_parent_input import ImportedResponseParentCustody, require_target_parent_authority


@dataclass(frozen=True)
class AuthenticatedResponseParent:
    grant: ResponseParentTargetGrant
    custody: ImportedResponseParentCustody
    raw: bytes


def read_authenticated_parent(
    *,
    route: str,
    source_root: Path | None,
    parent_manifest: Path | None,
    custody: Path | None,
    reveal_record: Path | None,
    analysis_record: Path | None,
    authority_store: ResponseParentTargetAuthorityStore | None,
) -> AuthenticatedResponseParent:
    grant = require_target_parent_authority(
        route=route,
        source_root=source_root,
        parent_manifest=parent_manifest,
        custody=custody,
        reveal_record=reveal_record,
        analysis_record=analysis_record,
        authority_store=authority_store,
    )
    assert source_root is not None and parent_manifest is not None
    assert (
        custody is not None
        and reveal_record is not None
        and analysis_record is not None
    )
    replay = replay_response_parent_custody(
        route=route,
        source_root=source_root,
        parent_manifest=parent_manifest,
        custody=custody,
        reveal_record=reveal_record,
        analysis_record=analysis_record,
        authority_store=authority_store,
        source_schema=grant.parent.source_schema,
    )
    if replay.grant != grant:
        raise ValueError("RESPONSE_PARENT_GRANT_CHANGED_DURING_REPLAY")
    return AuthenticatedResponseParent(grant, replay.custody, replay.raw)


__all__ = ['AuthenticatedResponseParent', "read_authenticated_parent"]
