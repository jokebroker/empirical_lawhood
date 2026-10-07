"""Public bounded delivery and custody framing of reusable finite operands.

Exports materialize only explicitly selected new files. A manifest written last
is the completion marker; interruption can leave partial files, never a completed
manifest. Import authenticates bytes and typed identities, without native work,
qualification, source acquisition or a hidden private-mount prerequisite.
"""

from collections.abc import Mapping
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.api.configuration import _write_new_file
from empirical_lawhood.adapters._bounded_files import read_bounded_contained
from empirical_lawhood.adapters.methods.finite_response_law.original_f import OriginalFiniteResponseLaw
from empirical_lawhood.infrastructure.matrix_array_io import decode_matrix_arrays, encode_matrix_arrays
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.matrix_inputs import (
    MAXIMUM_MATRIX_ARRAY_BYTES,
    MATRIX_ARRAY_SCHEMA,
    MatrixAllocation,
    SavedMatrixArrays,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity

ORIGINAL_F_FILENAMES = (
    "payload.canonical.json", "manifest.json", "publication-commit.json",
    "calibration-report.json", "qualification-report.json",
)
NOMINATION_SOURCE_FILENAMES = (
    "original-development-report.json", "original-coefficients.transport.json",
    "original-development-manifest.json",
)
NOMINATION_CURRENT_FILENAMES = (
    "development-report.json", "coefficients.transport.json", "development-manifest.json",
)


def _read_selected(path: Path, *, maximum_bytes: int) -> bytes:
    if ".." in path.parts:
        raise ValueError("Operand input refuses parent-directory traversal")
    absolute = path.absolute()
    return read_bounded_contained(Path(absolute.anchor), absolute, maximum_bytes=maximum_bytes)


def _public_bytes(package: str, filenames: tuple[str, ...], sizes: tuple[int, ...]) -> tuple[bytes, ...]:
    root = files("empirical_lawhood").joinpath("resources", "finite_response_law", package)
    result = []
    for name, size in zip(filenames, sizes, strict=True):
        with root.joinpath(name).open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size:
            raise ValueError("Public finite source resource differs from its closed byte census")
        result.append(raw)
    return tuple(result)


def _export_files(destination: Path, members: Mapping[str, bytes], manifest_name: str, manifest: bytes) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("Operand destination must be an explicitly selected existing real directory")
    paths = tuple(destination / name for name in (*members, manifest_name))
    if any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("Operand export refuses existing members or manifest")
    for name, raw in members.items():
        _write_new_file(destination / name, raw)
    _write_new_file(destination / manifest_name, manifest)


def original_f_operand_export(destination: Path, *, payload_id: str):
    """Export the actual original bytes and distinct current original-F record."""
    from empirical_lawhood.adapters.methods.finite_response_law.original_f import (
        authenticate_original_f_sources, original_f_from_bytes, original_f_source_identities,
    )

    sources = _public_bytes("original_f", ORIGINAL_F_FILENAMES, (37921, 3919, 1428, 17166, 522736))
    identities = original_f_source_identities(tuple(f"{payload_id}.source.{index}" for index in range(5)))
    authenticate_original_f_sources(identities, sources)
    operand = original_f_from_bytes(sources[0], payload_id=payload_id, source_identities=identities)
    _export_files(destination, dict(zip(ORIGINAL_F_FILENAMES, sources, strict=True)), "original-f.canonical.json", operand.canonical_bytes())
    return operand


def original_f_operand_import(operand_path: Path, *, source_directory: Path):
    """Import supplied original files, authenticated against the current wrapper."""
    from empirical_lawhood.adapters.methods.finite_response_law.original_f import (
        OriginalFiniteResponseLaw, authenticate_original_f_sources, original_f_from_bytes,
    )

    operand = decode_canonical_bytes(_read_selected(operand_path, maximum_bytes=1024**2), OriginalFiniteResponseLaw, maximum_bytes=1024**2)
    sources = tuple(_read_selected(source_directory / name, maximum_bytes=identity.size_bytes) for name, identity in zip(ORIGINAL_F_FILENAMES, operand.source_identities, strict=True))
    authenticate_original_f_sources(operand.source_identities, sources)
    reconstructed = original_f_from_bytes(sources[0], payload_id=operand.payload_id, source_identities=operand.source_identities)
    if reconstructed != operand:
        raise ValueError("Original F current record differs from its authenticated source extraction")
    return operand


def current_nomination_operand_export(destination: Path, *, nomination_id: str):
    """Deliver config-compatible report/coefficient/manifest with original crosswalk."""
    from empirical_lawhood.adapters.methods.finite_response_law.nominated_package import current_nomination_from_sources

    sources = _public_bytes("nomination", ("development-report.json", "coefficients.transport.json", "development-manifest.json"), (32160, 763803, 227387))
    operand, current = current_nomination_from_sources(nomination_id, sources)
    members = dict(zip(NOMINATION_SOURCE_FILENAMES, sources, strict=True))
    members.update(zip(NOMINATION_CURRENT_FILENAMES, current, strict=True))
    _export_files(destination, members, "nomination.canonical.json", operand.canonical_bytes())
    return operand


def current_nomination_operand_import(operand_path: Path, *, source_directory: Path):
    from empirical_lawhood.adapters.methods.finite_response_law.nominated_package import CurrentNominationOperand, authenticate_current_nomination

    operand = decode_canonical_bytes(_read_selected(operand_path, maximum_bytes=65536), CurrentNominationOperand, maximum_bytes=65536)
    sources = tuple(_read_selected(source_directory / name, maximum_bytes=identity.size_bytes) for name, identity in zip(NOMINATION_SOURCE_FILENAMES, operand.original_artifacts, strict=True))
    current = tuple(_read_selected(source_directory / name, maximum_bytes=identity.size_bytes) for name, identity in zip(NOMINATION_CURRENT_FILENAMES, operand.current_artifacts, strict=True))
    authenticate_current_nomination(operand, sources, current)
    return operand


def saved_matrix_arrays_export(
    destination: Path,
    *,
    operand_id: str,
    allocation: MatrixAllocation,
    arrays: Mapping[str, NDArray[np.generic]],
    producers: tuple[ObjectIdentity, ...],
    source_artifacts: tuple[ArtifactIdentity, ...] = (),
    source_receipts: tuple[ObjectIdentity, ...] = (),
    original_f: ObjectIdentity | None = None,
    nonfinite_members: tuple[str, ...] = (),
) -> SavedMatrixArrays:
    """Retain caller-produced current arrays; this does not execute or grant authority."""
    raw, members = encode_matrix_arrays(arrays, nonfinite_members=nonfinite_members)
    artifact = ArtifactIdentity(f"{operand_id}.arrays", "matrix-arrays", MATRIX_ARRAY_SCHEMA, sha256(raw).hexdigest(), "application/x-npz", len(raw))
    manifest = SavedMatrixArrays(operand_id, allocation.identity, artifact, members, producers, source_artifacts, source_receipts, original_f)
    _export_files(destination, {"arrays.npz": raw, "allocation.canonical.json": allocation.canonical_bytes()}, "arrays.canonical.json", manifest.canonical_bytes())
    return manifest


def saved_matrix_arrays_import(
    manifest_path: Path, *, arrays_path: Path, allocation_path: Path, artifact_writer=None
) -> tuple[SavedMatrixArrays, MatrixAllocation, Mapping[str, NDArray[np.generic]]]:
    """Authenticate explicit files without consulting a catalog or a private mount."""
    manifest = decode_canonical_bytes(_read_selected(manifest_path, maximum_bytes=1024**2), SavedMatrixArrays, maximum_bytes=1024**2)
    allocation = decode_canonical_bytes(_read_selected(allocation_path, maximum_bytes=1024**2), MatrixAllocation, maximum_bytes=1024**2)
    if manifest.allocation != allocation.identity:
        raise ValueError("Saved matrix arrays differ from the supplied current allocation")
    _authenticate_current_source_completion(manifest_path.parent, manifest=manifest, allocation=allocation, artifact_writer=artifact_writer)
    arrays = decode_matrix_arrays(_read_selected(arrays_path, maximum_bytes=MAXIMUM_MATRIX_ARRAY_BYTES), manifest)
    return manifest, allocation, arrays




def _authenticate_current_source_completion(directory, *, manifest, allocation, artifact_writer):
    """Native source markers require independently committed complete custody."""
    from empirical_lawhood.kernel.numerical_provenance import NumericalProducingProvenance
    from empirical_lawhood.api.matrix_numerical_provenance import authenticate_recorded_matrix_provenance
    from empirical_lawhood.infrastructure.retained_analysis import read_retained_analysis, read_retained_analysis_member
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import (
        CurrentPreparationSourceConfig, MatrixPreparationSourceConfig,
        CurrentPreparationSourceReport, CurrentPreparationRootReport,
    )
    markers = ("source-config.canonical.json", "source-report.canonical.json", "numerical-provenance.canonical.json")
    if not any((directory / name).exists() or (directory / name).is_symlink() for name in markers):
        if any(item.payload_schema == NumericalProducingProvenance.SCHEMA for item in manifest.source_artifacts):
            raise ValueError("Native source lacks its committed producing provenance")
        return  # Explicit supplied exposed arrays retain their separate import contract.
    completion = read_retained_analysis(directory=directory, artifact_writer=artifact_writer)
    def raw(name):
        return read_retained_analysis_member(completion, name, directory=directory,
            artifact_writer=artifact_writer, maximum_bytes=16 * 1024**2)
    if decode_canonical_bytes(raw("arrays.canonical.json"), SavedMatrixArrays, maximum_bytes=16 * 1024**2) != manifest:
        raise ValueError("Native source manifest differs from its committed completion")
    provenance = authenticate_recorded_matrix_provenance(decode_canonical_bytes(
        raw("numerical-provenance.canonical.json"), NumericalProducingProvenance, maximum_bytes=16 * 1024**2))
    if not (directory / "source-config.canonical.json").exists():
        expected = {"bridge-input.canonical.json", "bridge-spec.canonical.json", "arrays.canonical.json",
                    "arrays.npz", "allocation.canonical.json", "numerical-provenance.canonical.json"}
        if {member.relative_path for member in completion.members} != expected:
            raise ValueError("Derived fit completion changes its complete member census")
        from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import TransientBridgeInput, TransientBridgeSpec
        source = decode_canonical_bytes(raw("bridge-input.canonical.json"), TransientBridgeInput, maximum_bytes=16 * 1024**2)
        spec = decode_canonical_bytes(raw("bridge-spec.canonical.json"), TransientBridgeSpec, maximum_bytes=16 * 1024**2)
        if source.allocation != allocation.identity or source.spec != spec or source.identity not in manifest.producers:
            raise ValueError("Derived fit completion changes its authenticated input/protocol")
        if not any(item.payload_schema == provenance.SCHEMA and item.sha256 == provenance.fingerprint()
                   for item in manifest.source_artifacts):
            raise ValueError("Derived fit completion omits its exact numerical provenance")
        return
    config_type = CurrentPreparationSourceConfig if manifest.original_f is not None else MatrixPreparationSourceConfig
    config = decode_canonical_bytes(raw("source-config.canonical.json"), config_type, maximum_bytes=16 * 1024**2)
    report = decode_canonical_bytes(raw("source-report.canonical.json"), CurrentPreparationSourceReport, maximum_bytes=16 * 1024**2)
    if (config.allocation != allocation.identity or report.allocation != allocation.identity
        or config.code_sources_sha256 != provenance.code_sources_sha256
        or config.dependency_lock_sha256 != provenance.dependency_lock_sha256
        or report.source != config.identity or report.original_f != manifest.original_f
        or manifest.producers != (config.identity, report.identity)
        or len(report.root_reports) != len(allocation.roots)):
        raise ValueError("Native source completion substitutes producing inputs or environment")
    by_name = {member.relative_path: member.artifact for member in completion.members}
    if manifest.source_artifacts != (by_name["numerical-provenance.canonical.json"], by_name["source-report.canonical.json"], *report.root_reports):
        raise ValueError("Native source changes its complete provenance/report artifact census")
    if decode_canonical_bytes(raw("source-allocation.canonical.json"), MatrixAllocation, maximum_bytes=16 * 1024**2) != allocation:
        raise ValueError("Native source allocation differs from its precontact frozen input")
    expected = set((*markers, "source-allocation.canonical.json", "allocation.canonical.json", "arrays.canonical.json", "arrays.npz"))
    for root, root_identity in zip(allocation.roots, report.root_reports, strict=True):
        name = root.root_id + ".source-report.canonical.json"
        root_raw = raw(name)
        if (len(root_raw), sha256(root_raw).hexdigest()) != (root_identity.size_bytes, root_identity.sha256):
            raise ValueError("Native source root report differs from its completion")
        root_report = decode_canonical_bytes(root_raw, CurrentPreparationRootReport, maximum_bytes=16 * 1024**2)
        if root_report.root_id != root.root_id or root_report.allocation != ObjectIdentity.from_record(root.root_id, root) or root_report.source != config.identity:
            raise ValueError("Native source root report substitutes its complete numeric allocation")
        expected.add(name)
        for cell in root_report.cells:
            if cell.artifact is not None:
                expected.add(cell.cell_id + ".canonical.json")
                bound = by_name.get(cell.cell_id + ".canonical.json")
                if bound != cell.artifact:
                    raise ValueError("Native source cell differs from its complete report census")
    if {member.relative_path for member in completion.members} != expected:
        raise ValueError("Native source completion changes its full typed member census")


def prepare_matrix_source_configuration(*, config_id: str, allocation: MatrixAllocation,
                                        original_f: OriginalFiniteResponseLaw | None = None, project_root: Path | None = None):
    """Bind an edited allocation to the actual current source/protocol owners."""
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import (
        CurrentPreparationSourceConfig, MatrixPreparationSourceConfig, MatrixPreparationSourceProtocol,
    )
    from empirical_lawhood.api.matrix_numerical_provenance import selected_dependency_lock_sha256
    root = Path(__file__).resolve().parents[3] if project_root is None else project_root
    lock = selected_dependency_lock_sha256(root)
    if original_f is None:
        return MatrixPreparationSourceConfig(config_id, allocation.identity,
            MatrixPreparationSourceProtocol(), current_preparation_code_sources_sha256(), dependency_lock_sha256=lock)
    return CurrentPreparationSourceConfig(config_id, allocation.identity, original_f.identity,
        current_preparation_protocol_identity(), current_preparation_code_sources_sha256(), dependency_lock_sha256=lock)


def current_preparation_code_sources_sha256() -> str:
    """Bind the complete current package; no partial native dependency list."""
    from empirical_lawhood.api.matrix_numerical_provenance import numerical_package_code_sha256
    return numerical_package_code_sha256()


def current_preparation_protocol_identity() -> ObjectIdentity:
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import TransientBridgeSpec

    return ObjectIdentity.from_record("finite-response-law.transient-bridge-protocol", TransientBridgeSpec())


def current_preparation_source_export(
    destination: Path,
    *,
    config,
    allocation: MatrixAllocation,
    original_f,
    project_root: Path,
    artifact_writer,
    progress=None,
) -> SavedMatrixArrays:
    """Explicitly execute and retain the current 24-root native development source.

    This effectful entry is separate from static validation and operand import.
    It grants no scientific execution authority or prospective eligibility.
    Native acquisition is not performed by export of already saved operands.
    """
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import CurrentPreparationSourceConfig

    if (
        not isinstance(config, CurrentPreparationSourceConfig)
        or config.allocation != allocation.identity
        or config.original_f != original_f.identity
        or config.protocol != current_preparation_protocol_identity()
        or config.code_sources_sha256 != current_preparation_code_sources_sha256()
        or tuple(root.cohort for root in allocation.roots) != ("q2",) * 8 + ("cir1",) * 16
        or any(root.source_prefix is not None for root in allocation.roots)
    ):
        raise ValueError("Current source changes its allocation, actual prefix kinds, F, protocol or code binding")
    return _matrix_source_export(destination, config=config, allocation=allocation, original_f_identity=original_f.identity, project_root=project_root, artifact_writer=artifact_writer, progress=progress)


def matrix_preparation_source_export(
    destination: Path, *, config, allocation: MatrixAllocation, project_root: Path, artifact_writer, progress=None
) -> SavedMatrixArrays:
    """Deliver a bounded declared current roster through the reusable native source.

    One call accepts up to32 roots, keeping complete nine-menu arrays below the
    shared64MiB expansion ceiling. Larger scientific campaigns compose explicit
    per-root/batch production; this helper never drops a configured root.
    """
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import MatrixPreparationSourceConfig

    if not isinstance(config, MatrixPreparationSourceConfig) or config.allocation != allocation.identity or config.code_sources_sha256 != current_preparation_code_sources_sha256() or any(root.cohort not in ("q2", "cir1") or root.source_prefix is not None for root in allocation.roots) or len(allocation.roots) > 32:
        raise ValueError("Matrix source changes its current roster, source kinds, code or bounded invocation size")
    return _matrix_source_export(destination, config=config, allocation=allocation, original_f_identity=None, project_root=project_root, artifact_writer=artifact_writer, progress=progress)


def _matrix_source_export(destination: Path, *, config, allocation: MatrixAllocation, original_f_identity: ObjectIdentity | None, project_root: Path, artifact_writer, progress=None) -> SavedMatrixArrays:
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source import acquire_current_preparation_root
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import CurrentPreparationSourceReport

    if not destination.is_dir() or destination.is_symlink() or tuple(destination.iterdir()):
        raise ValueError("Current source destination must be an explicitly selected empty real directory")

    from empirical_lawhood.infrastructure.retained_analysis import publish_retained_analysis
    from empirical_lawhood.runtime.retained_analysis import RetainedAnalysisMember
    from empirical_lawhood.api.matrix_numerical_provenance import preflight_matrix_output_directory
    preflight_matrix_output_directory(directory=destination, artifact_writer=artifact_writer,
        project_root=project_root, allow_existing_empty=True)
    members = []
    def bind(name, raw, schema):
        identity = ArtifactIdentity(name.replace("/", "."), "current-native-source-member", schema,
            sha256(raw).hexdigest(), "application/json", len(raw))
        members.append(RetainedAnalysisMember(name, identity))
        return identity

    def retain(cell_id, record):
        raw = record.canonical_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError("Current source cell exceeds its bounded canonical transport")
        identity = ArtifactIdentity(f"{cell_id}.native", "current-native-cell", record.SCHEMA, sha256(raw).hexdigest(), "application/json", len(raw))
        name = f"{cell_id}.canonical.json"
        _write_new_file(destination / name, raw)
        members.append(RetainedAnalysisMember(name, identity))
        return identity

    from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance
    provenance = capture_matrix_provenance(project_root=project_root,
        expected_code_sha256=config.code_sources_sha256,
        expected_lock_sha256=config.dependency_lock_sha256)
    provenance_raw = provenance.canonical_bytes()
    provenance_artifact = ArtifactIdentity("numerical.producing-provenance", "numerical-producing-provenance",
        provenance.SCHEMA, sha256(provenance_raw).hexdigest(), "application/json", len(provenance_raw))
    _write_new_file(destination / "numerical-provenance.canonical.json", provenance_raw)
    members.append(RetainedAnalysisMember("numerical-provenance.canonical.json", provenance_artifact))

    # Persist the predeclared source/allocation before the first native effect;
    # absence of arrays.canonical.json continues to mean incomplete acquisition.
    _write_new_file(destination / "source-config.canonical.json", config.canonical_bytes())
    _write_new_file(destination / "source-allocation.canonical.json", allocation.canonical_bytes())
    bind("source-config.canonical.json", config.canonical_bytes(), config.SCHEMA)
    bind("source-allocation.canonical.json", allocation.canonical_bytes(), allocation.SCHEMA)
    rows = []
    root_reports = []
    for root in allocation.roots:
        result = acquire_current_preparation_root(config=config, root=root, retain=retain, progress=progress)
        raw = result.report.canonical_bytes()
        identity = ArtifactIdentity(f"{root.root_id}.source-report", "current-root-source-report", result.report.SCHEMA, sha256(raw).hexdigest(), "application/json", len(raw))
        _write_new_file(destination / f"{root.root_id}.source-report.canonical.json", raw)
        members.append(RetainedAnalysisMember(f"{root.root_id}.source-report.canonical.json", identity))
        rows.append(result)
        root_reports.append(identity)
    if config.code_sources_sha256 != current_preparation_code_sources_sha256():
        raise ValueError("Current source code changed during native acquisition")
    report = CurrentPreparationSourceReport(f"{config.config_id}.report", config.identity, allocation.identity, original_f_identity, tuple(root_reports), "COMPLETE" if all(row.report.disposition == "COMPLETE" for row in rows) else "UNEVALUABLE")
    report_bytes = report.canonical_bytes()
    report_artifact = ArtifactIdentity(report.report_id, "current-source-report", report.SCHEMA, sha256(report_bytes).hexdigest(), "application/json", len(report_bytes))
    _write_new_file(destination / "source-report.canonical.json", report_bytes)
    members.append(RetainedAnalysisMember("source-report.canonical.json", report_artifact))
    arrays = {name: np.stack([row.arrays[name] for row in rows]) for name in rows[0].arrays}
    manifest = saved_matrix_arrays_export(destination, operand_id=f"{config.config_id}.native-arrays", allocation=allocation, arrays=arrays, producers=(config.identity, report.identity), source_artifacts=(provenance_artifact, report_artifact, *root_reports), original_f=original_f_identity, nonfinite_members=tuple(name for name, value in arrays.items() if not np.isfinite(value).all()))
    bind("arrays.canonical.json", manifest.canonical_bytes(), manifest.SCHEMA)
    bind("allocation.canonical.json", allocation.canonical_bytes(), allocation.SCHEMA)
    members.append(RetainedAnalysisMember("arrays.npz", manifest.array_artifact))
    publish_retained_analysis(directory=destination, artifact_writer=artifact_writer,
        completion_id=f"{config.config_id}.source-completion", members=tuple(members))
    return manifest


def preparation_operands_export(
    destination: Path,
    *,
    native_manifest_path: Path,
    native_arrays_path: Path,
    allocation_path: Path,
    original_f,
    operand_id: str,
    artifact_writer,
    project_root: Path,
) -> SavedMatrixArrays:
    """Produce the minimum current bridge/readiness operands from saved native arrays."""
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import TransientBridgeInput, TransientBridgeSpec, current_root_fold_arrays, fit_hold
    from empirical_lawhood.adapters.methods.finite_response_law.fitting import Normalizer
    from empirical_lawhood.adapters.methods.preparation_applicability.measurement import validity
    from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator

    if not destination.is_dir() or destination.is_symlink() or any(destination.iterdir()):
        raise ValueError("Derived operand output must be an explicitly selected empty real directory")
    from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance, preflight_matrix_output_directory
    preflight_matrix_output_directory(directory=destination, artifact_writer=artifact_writer,
        project_root=project_root, allow_existing_empty=True)
    provenance = capture_matrix_provenance(project_root=project_root)
    native, allocation, arrays = saved_matrix_arrays_import(native_manifest_path, arrays_path=native_arrays_path, allocation_path=allocation_path, artifact_writer=artifact_writer)
    if native.original_f != original_f.identity or tuple(root.cohort for root in allocation.roots) != ("q2",) * 8 + ("cir1",) * 16:
        raise ValueError("Current preparation arrays change original F or ordered native prefix kinds")
    expected = {"x": (24, 24, 2), "z": (24, 9, 24, 2), "y": (24, 9, 4, 8, 2, 2), "work": (24, 9, 2), "features": (24, 9, 2, 26, 24), "positions": (24, 9, 2, 26, 2, 3, 4, 4), "momenta": (24, 9, 2, 26, 2, 3, 4, 4)}
    for name, shape in expected.items():
        if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all() or arrays[name].dtype != np.dtype("complex128" if name in ("positions", "momenta") else "float64"):
            raise ValueError(f"UNEVALUABLE: incomplete required current {name} array")
    identity_bytes = native.canonical_bytes()
    native_record_artifact = ArtifactIdentity(f"{native.operand_id}.manifest", "current-native-array-manifest", native.SCHEMA, sha256(identity_bytes).hexdigest(), "application/json", len(identity_bytes))
    source = TransientBridgeInput(f"{operand_id}.input", allocation.identity, original_f.identity, native.array_artifact, native_record_artifact, tuple(root.root_id for root in allocation.roots), tuple(root.cohort for root in allocation.roots))
    fitted = current_root_fold_arrays(current_input=source, original_f=original_f, primary_prefix=arrays["x"][..., 0], native_handoff=arrays["z"], trajectories=arrays["features"])
    out = dict(arrays)
    out.update(fitted.arrays)
    out.update(center=np.asarray(original_f.center, dtype=np.float64), scale=np.asarray(original_f.scale, dtype=np.float64), operator=np.asarray(original_f.operator, dtype=np.float64).reshape(25, 32))
    # Current all-nine U2 baseline uses the same excluded-root ridge10 recipe.
    # It is a saved current fit, not an alias of an archival U2 prediction.
    upper_z = np.empty((24, 9, 24), dtype=np.float64)
    folds = fitted.arrays["folds"]
    prefix = arrays["x"][..., 0]
    for fold in range(4):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        normalizer = Normalizer.fit(prefix[train])
        operators = np.stack([fit_affine_operator(normalizer.apply(prefix[train]), arrays["z"][train, schedule, :, 0], ridge=10) for schedule in range(9)])
        upper_z[test] = np.stack([affine_prediction(operator, normalizer.apply(prefix[test])) for operator in operators], axis=1)
        np.testing.assert_allclose(upper_z[test, 0], fit_hold(prefix[train], arrays["z"][train, 0, :, 0], prefix[test]), rtol=1e-12, atol=1e-12)
        for suffix, value in (("center", normalizer.center), ("scale", normalizer.scale), ("operator", operators), ("train", train), ("excluded", test)):
            out[f"u2.fold.{fold}.{suffix}"] = np.asarray(value)
    out["upper_z"] = upper_z
    out["upper_mean"] = original_f.predict(upper_z.reshape(-1, 24)).mean.reshape(24, 9, 4, 8)
    summaries = [validity(original_f, arrays["z"][root], arrays["y"][root], arrays["work"][root]) for root in range(24)]
    for index, name in enumerate(("maxima", "conjuncts", "margin", "mean", "width", "supported")):
        out[name] = np.stack([summary[index] for summary in summaries])
    for row in fitted.fits:
        for name in ("training_roots", "test_roots", "normalizer_center", "normalizer_scale", "kernel_scale", "operator"):
            out[f"radial.all.fold.{row['fold']}.{name}"] = np.asarray(row[name])
        out[f"radial.all.fold.{row['fold']}.kernel_rank"] = np.asarray(row["kernel_rank"], dtype=np.int64)
        out[f"radial.all.fold.{row['fold']}.training_modal_projection_mse"] = np.asarray(row["training_modal_projection_mse"], dtype=np.float64)
    spec = TransientBridgeSpec()
    provenance_raw = provenance.canonical_bytes()
    provenance_artifact = ArtifactIdentity("numerical.producing-provenance", "derived-numerical-provenance",
        provenance.SCHEMA, sha256(provenance_raw).hexdigest(), "application/json", len(provenance_raw))
    _write_new_file(destination / "numerical-provenance.canonical.json", provenance_raw)
    _write_new_file(destination / "bridge-input.canonical.json", source.canonical_bytes())
    _write_new_file(destination / "bridge-spec.canonical.json", spec.canonical_bytes())
    manifest = saved_matrix_arrays_export(destination, operand_id=operand_id, allocation=allocation, arrays=out, producers=(source.identity, ObjectIdentity.from_record("finite-response-law.transient-bridge-protocol", spec)), source_artifacts=(native.array_artifact, native_record_artifact, provenance_artifact), source_receipts=native.source_receipts, original_f=original_f.identity)
    from empirical_lawhood.infrastructure.retained_analysis import publish_retained_analysis
    from empirical_lawhood.runtime.retained_analysis import RetainedAnalysisMember
    members = [RetainedAnalysisMember("arrays.npz", manifest.array_artifact)]
    for name, record in (("bridge-input.canonical.json", source), ("bridge-spec.canonical.json", spec),
                         ("arrays.canonical.json", manifest), ("allocation.canonical.json", allocation),
                         ("numerical-provenance.canonical.json", provenance)):
        raw = record.canonical_bytes()
        members.append(RetainedAnalysisMember(name, ArtifactIdentity(name, "derived-numerical-fit-member",
            record.SCHEMA, sha256(raw).hexdigest(), "application/json", len(raw))))
    publish_retained_analysis(directory=destination, artifact_writer=artifact_writer,
        completion_id=f"{operand_id}.fit-completion", members=tuple(members))
    return manifest


def preparation_bridge_fits(arrays: Mapping[str, NDArray[np.generic]]) -> tuple[Mapping[str, object], ...]:
    """Restore the four exact saved radial.all fit records for readiness."""
    result = []
    for fold in range(4):
        row = {"model": "radial.all", "fold": fold}
        for name in ("training_roots", "test_roots", "normalizer_center", "normalizer_scale", "kernel_scale", "operator"):
            row[name] = arrays[f"radial.all.fold.{fold}.{name}"]
        row["kernel_rank"] = int(arrays[f"radial.all.fold.{fold}.kernel_rank"])
        row["training_modal_projection_mse"] = float(arrays[f"radial.all.fold.{fold}.training_modal_projection_mse"])
        result.append(row)
    return tuple(result)
