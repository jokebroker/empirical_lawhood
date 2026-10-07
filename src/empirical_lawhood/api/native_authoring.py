"""Application-owned compilation of adapter-prepared native candidates.

SPDX-License-Identifier: MPL-2.0
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from empirical_lawhood.adapters.composition.prepared_response.native_authoring import PreparedResponseNativeAuthoringInput, prepare_prepared_response_native_authoring
from empirical_lawhood.api.authoring_handoff import create_authoring_api
from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.kernel.serialization import CanonicalRecord


def _write_records(
    directory: Path, bundle: object, exposure: CanonicalRecord
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    # Every supported authoring bundle has the same existing typed export seam.
    authoring = bundle.authoring  # type: ignore[attr-defined]
    payloads = bundle.payloads  # type: ignore[attr-defined]
    decoders = bundle.decoder_registrations  # type: ignore[attr-defined]
    author = directory / "authoring.json"
    author.write_bytes(authoring.canonical_bytes())
    exposure_path = directory / "exposure-inspection.json"
    exposure_path.write_bytes(exposure.canonical_bytes())
    payload_paths = []
    decoder_paths = []
    for index, record in enumerate(payloads):
        path = directory / f"payload-{index:02d}.json"
        path.write_bytes(record.canonical_bytes())
        payload_paths.append(path)
    for index, record in enumerate(decoders):
        path = directory / f"decoder-{index:02d}.json"
        path.write_bytes(record.canonical_bytes())
        decoder_paths.append(path)
    return author, tuple(payload_paths), tuple(decoder_paths)


def author_matrix_response(
    config: PreparedResponseNativeAuthoringInput,
    *,
    repo_root: Path,
    source_root: Path | None,
    plan: Path | None,
    design_packet: Path | None,
    prior_exposure: Path | None,
    model_bank: Path | None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Compile one source-owned candidate without native contact or authority."""
    prepared = prepare_prepared_response_native_authoring(
        config,
        repo_root=repo_root,
        source_root=source_root,
        plan=plan,
        design_packet=design_packet,
        prior_exposure=prior_exposure,
        model_bank=model_bank,
    )
    bundle = prepared.bundle
    exposure = prepared.exposure
    if output_dir is not None and (output_dir.is_symlink() or output_dir.exists()):
        raise FileExistsError("Matrix response authoring output must be a new directory")
    if output_dir is not None and output_dir.resolve().is_relative_to(
        repo_root.resolve()
    ):
        raise ValueError("Matrix response authoring output must be outside the target checkout")
    with TemporaryDirectory(prefix="matrix-response-authoring-") as scratch:
        scratch_path = Path(scratch)
        author, payloads, decoders = _write_records(scratch_path, bundle, exposure)
        api = create_authoring_api(
            repo_root=repo_root,
            candidate_context_provider=prepared.context_provider_type(bundle),
            candidate_capability_catalog=bundle.catalog,
        )
        compiled = api.compile_candidate(
            CompileCandidateRequest(author, payloads, decoders)
        )
        if not compiled.succeeded or compiled.payload is None:
            raise RuntimeError(
                f"Matrix response candidate refused: {compiled.reason_codes}; {compiled.errors}"
            )
        candidate = compiled.payload.report.candidate
        if candidate is None:
            raise RuntimeError("Matrix response candidate compiler returned no candidate")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=False)
            with report_incomplete_output(output_dir) as progress:
                progress.stage = "product export"
                _write_records(output_dir, bundle, exposure)
                (output_dir / "candidate.json").write_bytes(candidate.canonical_bytes())
                (output_dir / "candidate-report.json").write_bytes(
                    compiled.payload.report.canonical_bytes()
                )
                (output_dir / "selection.json").write_bytes(config.canonical_bytes())
    return {
        **prepared.selection_summary,
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate.fingerprint(),
        "campaign_candidate_compiled": True,
        "authoring_dir": str(output_dir) if output_dir is not None else None,
    }
