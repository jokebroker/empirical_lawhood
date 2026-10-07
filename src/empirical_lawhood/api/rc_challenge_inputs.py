"""Publish current RC source/export operands through the existing artifact plane."""

from __future__ import annotations

from hashlib import sha256
from typing import Callable, Mapping

from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeConfig
from empirical_lawhood.adapters.simulator_morphism_challenges.numeric_inputs import (
    RCChallengeExposureCensus, RCChallengeNumericSource, RCChallengeNumericInput,
    RCChallengeNumericExportReceipt, allocate_numeric_source, export_numeric_source,
    RCChallengeSourceExportConfig,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactManifest, ArtifactMaterialization, ArtifactProfile, ArtifactWriter, ArtifactWriteRequest
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest


MAXIMUM_RC_NUMERIC_INPUT_BYTES = 8 * 1024**2
MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"


def authenticate_rc_exposure_census(census: RCChallengeExposureCensus, prior_source_payloads: Mapping[str, bytes]) -> None:
    """Authenticate every declared prior source and retain every exposed numeric seed."""
    if set(prior_source_payloads) != {a.artifact_id for a in census.prior_source_artifacts}:
        raise ValueError("prior source bytes differ from the complete declared exposure census")
    for artifact in census.prior_source_artifacts:
        payload = prior_source_payloads[artifact.artifact_id]
        if len(payload) != artifact.size_bytes or sha256(payload).hexdigest() != artifact.sha256 or artifact.payload_schema != RCChallengeNumericSource.SCHEMA:
            raise ValueError("prior RC source artifact bytes/schema fail authentication")
        source = decode_canonical_bytes(payload, RCChallengeNumericSource, maximum_bytes=MAXIMUM_RC_NUMERIC_INPUT_BYTES)
        if not {sha256(seed).hexdigest() for seed in source.seeds} <= set(census.exposed_seed_sha256s):
            raise ValueError("prior source seeds are missing from the exposure census")


def publish_rc_challenge_numeric_input(*, writer: ArtifactWriter,
                                      config: SimulatorMorphismChallengeConfig,
                                      exposure_census: RCChallengeExposureCensus,
                                      prior_source_payloads: Mapping[str, bytes],
                                      source_id: str, input_id: str,
                                      publication_scope_id: str,
                                      publication_scope_relative_root: str,
                                      relative_root: str,
                                      exposed_seed_hexes: tuple[str, ...] | None = None,
                                      manifest_reader: Callable[[ArtifactMaterialization], ArtifactManifest] | None = None) -> RCChallengeNumericInput:
    """Allocate, export and publish actual bytes; no historical receipts are invented.

    The caller supplies the guarded external writer and explicit selected paths.
    Supplied seeds are exposed examples; evaluation cannot promote their packet.
    """
    validate_stable_id(input_id)
    if manifest_reader is None:
        if not isinstance(writer, ExternalArtifactPlane):
            raise ValueError("numeric export requires the existing guarded plane or an explicit bounded manifest reader")
        manifest_reader = lambda materialization: decode_artifact_manifest(read_bounded_bytes(
            writer.root.resolve(materialization.relative_path, for_write=False), maximum_bytes=4 * 1024**2))
    authenticate_rc_exposure_census(exposure_census, prior_source_payloads)
    source = allocate_numeric_source(source_id=source_id, config=config, exposure_census=exposure_census,
                                     exposed_seed_hexes=exposed_seed_hexes)
    export = export_numeric_source(export_id=f"{input_id}.export", source=source)
    access = OutcomeAccess.DEVELOPMENT_VISIBLE if source.exposed_example else OutcomeAccess.EVALUATION_SEALED
    visibility = VisibilityCeiling.DEVELOPMENT_ONLY if source.exposed_example else VisibilityCeiling.PROSPECTIVE

    def request(name: str, record: CanonicalRecord) -> ArtifactWriteRequest:
        payload = record.canonical_bytes()
        if len(payload) > MAXIMUM_RC_NUMERIC_INPUT_BYTES:
            raise ValueError("RC numeric operand exceeds its closed publication byte bound")
        return ArtifactWriteRequest(
            logical_artifact_id=f"{input_id}.{name}", relative_path=f"{relative_root}/{name}.json",
            payload_schema=record.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
            media_type=MEDIA_TYPE, publication_scope_id=publication_scope_id,
            publication_scope_relative_root=publication_scope_relative_root, payload=payload,
            visibility_ceiling=visibility, parent_visibility_ceilings=(), outcome_access=access,
        )

    source_publication, export_publication = writer.write_batch((request("source", source), request("export", export)))
    receipt = RCChallengeNumericExportReceipt(f"{input_id}.export-receipt", source_publication, export_publication,
        manifest_reader(source_publication.manifest_materialization), manifest_reader(export_publication.manifest_materialization))
    receipt_publication = writer.write(request("receipt", receipt))
    result = RCChallengeNumericInput(input_id, source, export, receipt, receipt_publication,
        manifest_reader(receipt_publication.manifest_materialization))
    result.authenticate(writer)
    return result


def read_rc_challenge_numeric_input(payload: bytes, *, writer: ArtifactWriter) -> RCChallengeNumericInput:
    """Bounded typed replay with current payload/manifest custody verification."""
    result = decode_canonical_bytes(payload, RCChallengeNumericInput, maximum_bytes=MAXIMUM_RC_NUMERIC_INPUT_BYTES)
    result.authenticate(writer)
    return result


def publish_rc_source_export(*, request: RCChallengeSourceExportConfig, writer: ArtifactWriter,
                             prior_source_payloads: Mapping[str, bytes]) -> RCChallengeNumericInput:
    """Editable request entry point for current full-roster numerical operand delivery."""
    return publish_rc_challenge_numeric_input(
        writer=writer, config=request.config, exposure_census=request.exposure_census,
        prior_source_payloads=prior_source_payloads, source_id=request.source_id, input_id=request.input_id,
        publication_scope_id=request.publication_scope_id,
        publication_scope_relative_root=request.publication_scope_relative_root,
        relative_root=request.relative_root, exposed_seed_hexes=request.exposed_seed_hexes)
