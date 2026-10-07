# SPDX-License-Identifier: MPL-2.0
"""Bind existing bounded analysis files through the guarded publication owner.

Numeric transports are retained once. The independently committed completion
binds their exact bytes together with all selected control/report members.
"""

from hashlib import sha256
from pathlib import Path

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import bounded_file_sha256, read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest
from empirical_lawhood.runtime.execution import VerifiedArtifactInput
from empirical_lawhood.runtime.retained_analysis import (
    COMPLETION_FILENAME, MAX_RETAINED_ANALYSIS_COMPLETION_BYTES,
    RetainedAnalysisCompletion,
)


def _relative(directory, artifact_writer, *, writing):
    if not isinstance(artifact_writer, ExternalArtifactPlane):
        raise TypeError("Retained analysis requires the actual guarded external artifact plane")
    path = Path(directory)
    if ".." in path.parts or any(value.is_symlink() for value in (path, *path.parents)):
        raise ValueError("Retained analysis refuses traversal and symbolic links")
    artifact_writer.root.verify(for_write=writing)
    relative = path.absolute().relative_to(artifact_writer.root.contract.canonical_path).as_posix()
    artifact_writer.root.resolve(relative + "/" + COMPLETION_FILENAME, for_write=writing)
    return relative


def _verify_members(completion, *, relative, artifact_writer):
    for member in completion.members:
        actual = bounded_file_sha256(artifact_writer.root.resolve(relative + "/" + member.relative_path, for_write=False),
                                     maximum_bytes=member.artifact.size_bytes)
        if actual != (member.artifact.size_bytes, member.artifact.sha256):
            raise ValueError("Retained analysis member differs from its committed completion")


def publish_retained_analysis(*, directory, artifact_writer, completion_id, members):
    """Commit the exact closed member census last; grant no scientific authority."""
    relative = _relative(directory, artifact_writer, writing=True)
    completion = RetainedAnalysisCompletion(completion_id, tuple(sorted(members, key=lambda value: value.relative_path)))
    _verify_members(completion, relative=relative, artifact_writer=artifact_writer)
    payload = completion.canonical_bytes()
    if len(payload) > MAX_RETAINED_ANALYSIS_COMPLETION_BYTES:
        raise ValueError("Retained analysis completion exceeds its bounded custody record")
    artifact_writer.write(ArtifactWriteRequest(completion_id, relative + "/" + COMPLETION_FILENAME,
        completion.SCHEMA, ArtifactProfile.CANONICAL_JSON, "application/vnd.empirical-lawhood.canonical+json",
        completion_id + ".publication", relative, payload, VisibilityCeiling.DEVELOPMENT_ONLY, (),
        OutcomeAccess.DEVELOPMENT_VISIBLE, minimum_free_bytes=1))
    return completion


def read_retained_analysis(*, directory, artifact_writer):
    """Authenticate the durable completion and every member without calculation."""
    relative = _relative(directory, artifact_writer, writing=False)
    filename = relative + "/" + COMPLETION_FILENAME
    try:
        manifest = decode_artifact_manifest(read_bounded_bytes(artifact_writer.root.resolve(filename + ".manifest.json", for_write=False), maximum_bytes=1024**2))
    except FileNotFoundError as error:
        raise ValueError("Retained analysis has no authentic completed publication") from error
    if (manifest.logical.payload_schema != RetainedAnalysisCompletion.SCHEMA
            or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
            or manifest.logical.media_type != "application/vnd.empirical-lawhood.canonical+json"
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
            or manifest.logical.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or manifest.materialization.relative_path != filename
            or manifest.materialization.size_bytes > MAX_RETAINED_ANALYSIS_COMPLETION_BYTES):
        raise ValueError("Retained analysis completion changes its current exposed publication contract")
    port = artifact_writer.open(VerifiedArtifactInput(manifest.logical, manifest.materialization), maximum_bytes=MAX_RETAINED_ANALYSIS_COMPLETION_BYTES)
    try:
        completion = decode_canonical_bytes(port.read(MAX_RETAINED_ANALYSIS_COMPLETION_BYTES), RetainedAnalysisCompletion, maximum_bytes=MAX_RETAINED_ANALYSIS_COMPLETION_BYTES)
    finally:
        port.close()
    if completion.completion_id != manifest.logical.logical_artifact_id:
        raise ValueError("Retained analysis completion changes its logical identity")
    _verify_members(completion, relative=relative, artifact_writer=artifact_writer)
    return completion


def read_retained_analysis_member(completion, name, *, directory, artifact_writer, maximum_bytes):
    """Return the bounded member snapshot actually checked against custody."""
    relative = _relative(directory, artifact_writer, writing=False)
    member = next((value for value in completion.members if value.relative_path == name), None)
    if member is None or member.artifact.size_bytes > maximum_bytes:
        raise ValueError("Retained analysis lacks its exact required bounded member")
    raw = read_bounded_bytes(artifact_writer.root.resolve(relative + "/" + name, for_write=False), maximum_bytes=maximum_bytes)
    if (len(raw), sha256(raw).hexdigest()) != (member.artifact.size_bytes, member.artifact.sha256):
        raise ValueError("Retained analysis member differs from its committed completion")
    return raw
