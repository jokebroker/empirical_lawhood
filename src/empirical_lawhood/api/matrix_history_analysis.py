"""Explicit history production/import, descriptive calculation and durable replay.

Existing guarded publications supply custody. NPZ transports use the existing
uninterpreted ZIP custody profile; the bounded no-pickle matrix reader, not that
profile, authenticates their complete scientific array contract.
"""

from hashlib import sha256
from pathlib import Path

from empirical_lawhood.adapters.methods.matrix_history_analysis.records import (
    EXPOSED, HISTORY_FAMILIES, HISTORY_PURPOSES, HISTORY_TRANSPORT_SCHEMA,
    PUBLIC_HISTORY_MASTER_SEED, MatrixHistoryAllocation, MatrixHistoryAnalysisConfig,
    MatrixHistoryAnalysisReceipt, MatrixHistoryAnalysisRoot, MatrixHistoryArrayInput,
    MatrixHistoryEnvironment, MatrixHistoryProtocol, MatrixHistoryRootAllocation,
    MatrixHistorySeed, MatrixHistorySourceConfig, MatrixHistorySourceReceipt,
)
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.matrix_array_io import encode_matrix_arrays, decode_matrix_arrays
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes, validate_relative_locator, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest
from empirical_lawhood.runtime.execution import VerifiedArtifactInput

MAX_HISTORY_CONTROL_BYTES = 8 * 1024**2
def matrix_history_code_sources_sha256():
    from empirical_lawhood.api.matrix_numerical_provenance import numerical_package_code_sha256
    return numerical_package_code_sha256()


def _numeric_roots(namespace, master_seed):
    if type(master_seed) is not int or not 0 <= master_seed < 2**256:
        raise ValueError("History master input must be an explicit unsigned256-bit integer")
    validate_stable_id(namespace)
    return tuple(MatrixHistoryRootAllocation(f"{namespace}.h{index:03d}", index,
        tuple(MatrixHistorySeed(purpose, sha256(b"empirical-lawhood/history/numeric/v1\0" + master_seed.to_bytes(32, "big") + index.to_bytes(2, "big") + bytes([ordinal])).hexdigest()) for ordinal, purpose in enumerate(HISTORY_PURPOSES)))
        for index in range(256))


def prepare_matrix_history_allocation(*, allocation_id, namespace, master_seed,
                                      exposure=EXPOSED, prior_effective_seeds=()):
    """Bind numeric operands before effects. Namespace does not change draws.

    Proposed labels never prove freshness. Known original and supplied sample
    consumed streams are reserved regardless of labels or an omitted prior list.
    """
    roots = _numeric_roots(namespace, master_seed)
    allocation = MatrixHistoryAllocation(allocation_id, roots, exposure, tuple(sorted(set(prior_effective_seeds))))
    _check_exposure(allocation)
    return allocation


def _check_exposure(allocation):
    if allocation.exposure != "PROPOSED_UNRUN":
        return
    from empirical_lawhood.adapters.simulators.six_matrix_response.observation_order_scientific_inputs import observation_order_scientific_stream_inputs
    known = {int(s.scientific_seed_sha256[:32], 16) for qualification in (False, True) for s in observation_order_scientific_stream_inputs(qualification=qualification)}
    known.update(s.effective_seed for r in _numeric_roots("public.history", PUBLIC_HISTORY_MASTER_SEED) for s in r.seeds)
    known.update(allocation.prior_effective_seeds)
    if any(s.effective_seed in known for root in allocation.roots for s in root.seeds):
        raise ValueError("History proposed streams overlap original, public or declared exposed consumed bits")


def prepare_matrix_history_source_configuration(*, config_id, allocation, repo_root):
    return MatrixHistorySourceConfig(config_id, allocation.identity, MatrixHistoryProtocol(),
        matrix_history_code_sources_sha256(), sha256(_lock_bytes(Path(repo_root))).hexdigest())


def _lock_bytes(repo_root):
    from empirical_lawhood.adapters._bounded_files import read_bounded_contained
    return read_bounded_contained(repo_root, repo_root / "uv.lock", maximum_bytes=16 * 1024**2)


def _environment(config, repo_root):
    from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance
    provenance = capture_matrix_provenance(project_root=repo_root,
        expected_code_sha256=config.code_sources_sha256,
        expected_lock_sha256=config.dependency_lock_sha256)
    import json
    observed = json.loads(provenance.runtime_observation_json)
    return MatrixHistoryEnvironment(observed["python"], observed["numpy"], observed["scipy"],
        provenance.dependency_lock_sha256, provenance.code_sources_sha256, provenance=provenance)


def _plane(writer, relative_root, *, writing):
    if not isinstance(writer, ExternalArtifactPlane):
        raise TypeError("History custody requires the actual guarded external artifact plane")
    validate_relative_locator(relative_root)
    writer.root.verify(for_write=writing)
    writer.root.resolve(relative_root + "/preflight.json", for_write=writing)


def _publish(writer, *, name, path, relative_root, payload, schema, arrays=False):
    request = ArtifactWriteRequest(name, path, schema,
        ArtifactProfile.RAW_SOURCE_BYTES if arrays else ArtifactProfile.CANONICAL_JSON,
        "application/zip" if arrays else "application/vnd.empirical-lawhood.canonical+json",
        relative_root.replace("/", "."), relative_root, payload,
        VisibilityCeiling.DEVELOPMENT_ONLY, (), OutcomeAccess.DEVELOPMENT_VISIBLE,
        minimum_free_bytes=1)
    return writer.write_stream(request.as_stream()) if arrays else writer.write(request)


def _read_publication(writer, publication, *, maximum_bytes):
    if publication.materialization.size_bytes > maximum_bytes:
        raise ValueError("History publication exceeds its consuming byte bound")
    manifest = decode_artifact_manifest(read_bounded_bytes(writer.root.resolve(publication.manifest_materialization.relative_path, for_write=False), maximum_bytes=1024**2))
    writer.verify(publication.manifest_materialization)
    if manifest.logical != publication.logical or manifest.materialization != publication.materialization or manifest.logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY or manifest.logical.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
        raise ValueError("History current publication/manifest/evidence access differs")
    port = writer.open(VerifiedArtifactInput(manifest.logical, manifest.materialization), maximum_bytes=maximum_bytes)
    try:
        return port.read(maximum_bytes)
    finally:
        port.close()


def _read_record_path(writer, relative, kind):
    if not isinstance(writer, ExternalArtifactPlane):
        raise TypeError("History replay requires its actual guarded external artifact plane")
    validate_relative_locator(relative)
    writer.root.verify(for_write=False)
    manifest = decode_artifact_manifest(read_bounded_bytes(writer.root.resolve(relative + ".manifest.json", for_write=False), maximum_bytes=1024**2))
    if manifest.materialization.relative_path != relative or manifest.logical.payload_schema != kind.SCHEMA or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON or manifest.logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY or manifest.logical.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE or manifest.materialization.size_bytes > MAX_HISTORY_CONTROL_BYTES:
        raise ValueError("History retained record path/type/access differs")
    port = writer.open(VerifiedArtifactInput(manifest.logical, manifest.materialization), maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    try:
        return decode_canonical_bytes(port.read(MAX_HISTORY_CONTROL_BYTES), kind, maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    finally:
        port.close()


def _publish_arrays(writer, *, root, source, kind, arrays, relative_root, operand_id):
    import numpy as np
    nonfinite = tuple(k for k, v in arrays.items() if not np.isfinite(v).all())
    raw, members = encode_matrix_arrays(arrays, nonfinite_members=nonfinite)
    identity = ArtifactIdentity(operand_id + ".arrays", "history-array-transport", HISTORY_TRANSPORT_SCHEMA,
        sha256(raw).hexdigest(), "application/zip", len(raw))
    operand = MatrixHistoryArrayInput(operand_id, source, root, kind, identity, members)
    publication = _publish(writer, name=identity.artifact_id, path=relative_root + "/arrays.npz", relative_root=relative_root,
        payload=raw, schema=HISTORY_TRANSPORT_SCHEMA, arrays=True)
    record_publication = _publish(writer, name=operand_id, path=relative_root + "/operand.json", relative_root=relative_root,
        payload=operand.canonical_bytes(), schema=operand.SCHEMA)
    return operand, publication, record_publication


def _publish_matrix_history_arrays(*, config, allocation, root_index, arrays, writer, relative_root,
                                  repo_root, acquisition_origin="SUPPLIED_EXPOSED_ARRAYS",
                                  disposition="COMPLETE", completed_primary_steps=1024,
                                  completed_fine_steps=2048, reason=None, environment=None):
    """Publish supplied exposed phase-space; this never attests native acquisition."""
    _plane(writer, relative_root, writing=True)
    if config.allocation != allocation.identity or type(root_index) is not int or not 0 <= root_index < 256:
        raise ValueError("History supplied input changes source allocation/root")
    _check_exposure(allocation)
    env = _environment(config, repo_root) if environment is None else environment
    root = allocation.roots[root_index]
    from empirical_lawhood.adapters.simulators.six_matrix_response.history_source import validate_history_arrays
    validate_history_arrays(arrays, root, complete=disposition == "COMPLETE")
    source_publication, allocation_publication = _publish_inputs(writer, config, allocation, relative_root)
    operand, publication, operand_publication = _publish_arrays(writer, root=root, source=config.identity,
        kind="NATIVE_HISTORY", arrays=arrays, relative_root=relative_root, operand_id=root.history_id + ".native")
    receipt = MatrixHistorySourceReceipt(root.history_id + ".source-receipt", config, root, operand, publication,
        operand_publication, source_publication, allocation_publication, env, disposition, completed_primary_steps, completed_fine_steps, reason, acquisition_origin)
    _publish(writer, name=receipt.receipt_id, path=relative_root + "/receipt.json", relative_root=relative_root,
        payload=receipt.canonical_bytes(), schema=receipt.SCHEMA)
    return receipt


def publish_matrix_history_arrays(*, config, allocation, root_index, arrays, writer, relative_root, repo_root):
    """Import caller-supplied exposed arrays without claiming native acquisition."""
    return _publish_matrix_history_arrays(config=config, allocation=allocation, root_index=root_index,
        arrays=arrays, writer=writer, relative_root=relative_root, repo_root=repo_root,
        acquisition_origin="SUPPLIED_EXPOSED_ARRAYS")


def read_matrix_history_source_receipt(*, writer, relative_path, config, allocation):
    receipt = _read_record_path(writer, relative_path, MatrixHistorySourceReceipt)
    if receipt.source != config or config.allocation != allocation.identity or receipt.root != allocation.roots[receipt.root.history_index]:
        raise ValueError("History retained source receipt substitutes a config/root")
    _authenticate_source(receipt, writer)
    return receipt


def _authenticate_source(receipt, writer):
    from empirical_lawhood.api.matrix_numerical_provenance import authenticate_recorded_matrix_provenance
    authenticate_recorded_matrix_provenance(receipt.environment.provenance)
    allocation = _authenticate_inputs(writer, receipt.source, receipt.source_publication, receipt.allocation_publication)
    if receipt.root != allocation.roots[receipt.root.history_index]:
        raise ValueError("History source receipt changes its actual full numeric allocation")
    _require_array_publication(receipt.arrays, receipt.arrays_publication)
    raw_operand = _read_publication(writer, receipt.operand_publication, maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    if decode_canonical_bytes(raw_operand, MatrixHistoryArrayInput, maximum_bytes=MAX_HISTORY_CONTROL_BYTES) != receipt.arrays:
        raise ValueError("History supplied array record differs from actual published bytes")
    arrays = decode_matrix_arrays(_read_publication(writer, receipt.arrays_publication, maximum_bytes=64 * 1024**2), receipt.arrays)
    from empirical_lawhood.adapters.simulators.six_matrix_response.history_source import validate_history_arrays
    validate_history_arrays(arrays, receipt.root, complete=receipt.disposition == "COMPLETE")
    return arrays


def _publish_inputs(writer, config, allocation, relative_root):
    publications = []
    for name, record, identity in (("config", config, config.config_id), ("allocation", allocation, allocation.allocation_id)):
        publications.append(_publish(writer, name=identity, path=relative_root + f"/{name}.json", relative_root=relative_root,
            payload=record.canonical_bytes(), schema=record.SCHEMA))
    return tuple(publications)


def _authenticate_inputs(writer, config, config_publication, allocation_publication):
    actual_config = decode_canonical_bytes(_read_publication(writer, config_publication, maximum_bytes=MAX_HISTORY_CONTROL_BYTES), type(config), maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    allocation = decode_canonical_bytes(_read_publication(writer, allocation_publication, maximum_bytes=MAX_HISTORY_CONTROL_BYTES), MatrixHistoryAllocation, maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    expected_allocation = config.allocation if isinstance(config, MatrixHistorySourceConfig) else config.source.allocation
    if actual_config != config or allocation.identity != expected_allocation:
        raise ValueError("History actual retained source/allocation bytes differ")
    return allocation


def _require_array_publication(operand, publication):
    artifact = operand.array_artifact
    logical, physical = publication.logical, publication.materialization
    if (logical.logical_artifact_id != artifact.artifact_id or logical.payload_schema != artifact.payload_schema
        or logical.profile is not ArtifactProfile.RAW_SOURCE_BYTES or logical.media_type != artifact.media_type
        or logical.content_sha256 != artifact.sha256 or physical.physical_sha256 != artifact.sha256
        or physical.size_bytes != artifact.size_bytes):
        raise ValueError("History array physical/logical/current contract differs from its declared artifact")


def export_matrix_history_arrays(*, destination: Path, config, allocation, root_index, arrays):
    """Explicit bounded editable-file delivery; native validity is not attested."""
    import numpy as np
    from empirical_lawhood.api.configuration import _write_new_file
    from empirical_lawhood.adapters.simulators.six_matrix_response.history_source import validate_history_arrays
    destination = Path(destination)
    if ".." in destination.parts or any(p.is_symlink() for p in (destination, *destination.parents)) or not destination.is_dir():
        raise ValueError("History array export requires an explicitly selected existing real directory")
    if any((destination / name).exists() or (destination / name).is_symlink() for name in ("arrays.npz", "operand.json")):
        raise FileExistsError("History array export refuses existing members")
    if config.allocation != allocation.identity or type(root_index) is not int or not 0 <= root_index < 256:
        raise ValueError("History array export changes source/root allocation")
    root = allocation.roots[root_index]
    validate_history_arrays(arrays, root, complete=True)
    raw, members = encode_matrix_arrays(arrays)
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("History exposed export requires a complete finite native clock")
    artifact = ArtifactIdentity(root.history_id + ".supplied-arrays", "history-array-transport", HISTORY_TRANSPORT_SCHEMA,
        sha256(raw).hexdigest(), "application/zip", len(raw))
    operand = MatrixHistoryArrayInput(root.history_id + ".supplied", config.identity, root, "NATIVE_HISTORY", artifact, members)
    _write_new_file(destination / "arrays.npz", raw)
    _write_new_file(destination / "operand.json", operand.canonical_bytes())
    return operand


def import_matrix_history_arrays(*, operand_path: Path, arrays_path: Path, config, allocation,
                                writer, relative_root, repo_root):
    """Bounded typed import of explicit NPZ files into exposed current custody."""
    _plane(writer, relative_root, writing=True)
    from empirical_lawhood.adapters._bounded_files import read_bounded_contained
    def read_selected(path, limit):
        path = Path(path)
        if ".." in path.parts:
            raise ValueError("History supplied array input refuses parent traversal")
        absolute = path.absolute()
        return read_bounded_contained(Path(absolute.anchor), absolute, maximum_bytes=limit)
    operand = decode_canonical_bytes(read_selected(operand_path, MAX_HISTORY_CONTROL_BYTES), MatrixHistoryArrayInput, maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    if operand.source != config.identity or operand.kind != "NATIVE_HISTORY" or operand.root != allocation.roots[operand.root.history_index]:
        raise ValueError("History supplied-file source/root/mathematical identity differs")
    arrays = decode_matrix_arrays(read_selected(arrays_path, 64 * 1024**2), operand)
    return publish_matrix_history_arrays(config=config, allocation=allocation, root_index=operand.root.history_index,
        arrays=arrays, writer=writer, relative_root=relative_root, repo_root=repo_root)


def produce_matrix_histories(*, config, allocation, writer, relative_root, repo_root,
                             root_indices=tuple(range(256)), recover=False, progress=None):
    """Produce selected declared histories, or verify completed receipts without rerun."""
    _plane(writer, relative_root, writing=True)
    if config.allocation != allocation.identity or root_indices != tuple(sorted(set(root_indices))) or not root_indices or any(type(i) is not int or not 0 <= i < 256 for i in root_indices):
        raise ValueError("History source needs its exact allocation and ordered selected numeric roots")
    _check_exposure(allocation)
    # Preflight every selected member before the first native effect.
    pending, completed = [], []
    for index in root_indices:
        relative = f"{relative_root}/h{index:03d}"
        receipt_path = writer.root.resolve(relative + "/receipt.json", for_write=True)
        if receipt_path.exists():
            if not recover:
                raise FileExistsError("History source already exists; select receipt recovery explicitly")
            completed.append(read_matrix_history_source_receipt(writer=writer, relative_path=relative + "/receipt.json", config=config, allocation=allocation))
        else:
            if any(writer.root.resolve(relative + "/" + name, for_write=True).exists() for name in ("arrays.npz", "operand.json", "config.json", "allocation.json")):
                raise FileExistsError("Interrupted history has no completion receipt; native work was not repeated")
            pending.append(index)
    if not pending:
        return tuple(sorted(completed, key=lambda r: r.root.history_index))
    env = _environment(config, repo_root)
    from empirical_lawhood.adapters.simulators.six_matrix_response.history_source import acquire_history
    for index in pending:
        # Bind full numeric inputs before the first native effect in this root.
        _publish_inputs(writer, config, allocation, f"{relative_root}/h{index:03d}")
        arrays, disposition, primary, fine, reason = acquire_history(allocation.roots[index], config.protocol, progress=progress)
        completed.append(_publish_matrix_history_arrays(config=config, allocation=allocation, root_index=index,
            arrays=arrays, writer=writer, relative_root=f"{relative_root}/h{index:03d}", repo_root=repo_root,
            acquisition_origin="CURRENT_NATIVE_PRODUCER", disposition=disposition,
            completed_primary_steps=primary, completed_fine_steps=fine, reason=reason, environment=env))
    if config.code_sources_sha256 != matrix_history_code_sources_sha256():
        raise ValueError("History producing source bytes changed during native acquisition")
    return tuple(sorted(completed, key=lambda r: r.root.history_index))


def _decimal_metrics(metrics):
    from decimal import Decimal
    import math
    return tuple(sorted((name, Decimal(str(float(value))) if value is not None and math.isfinite(value) else None) for name, value in metrics.items()))


def _authenticate_analysis_root(root, config, writer):
    if root.arrays is None:
        return
    if root.arrays.source != config.identity or root.arrays.kind != config.mode or root.arrays.root.history_index != root.root_index or root.arrays.root.history_id != root.history_id:
        raise ValueError("History retained analysis substitutes its scientific source/root/mode")
    operand = decode_canonical_bytes(_read_publication(writer, root.operand_publication, maximum_bytes=MAX_HISTORY_CONTROL_BYTES), MatrixHistoryArrayInput, maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    if operand != root.arrays:
        raise ValueError("History analysis actual operand publication differs")
    _require_array_publication(operand, root.arrays_publication)
    arrays = decode_matrix_arrays(_read_publication(writer, root.arrays_publication, maximum_bytes=64 * 1024**2), operand)
    from empirical_lawhood.adapters.methods.matrix_history_analysis.analysis import validate_analysis_arrays
    validate_analysis_arrays(arrays, operand.root, mode=config.mode,
        compare_commutant_altered=config.compare_commutant_altered)


def analyze_matrix_histories(*, config: MatrixHistoryAnalysisConfig, allocation, source_receipt_paths, writer,
                            relative_root, repo_root, recover=False):
    """Authenticate explicit sources, calculate once and retain all256 root roles.

    Missing/incomplete roots remain UNEVALUABLE. The summary never turns a
    partial cohort into a complete family median or a qualified prediction law.
    """
    _plane(writer, relative_root, writing=True)
    if config.source.allocation != allocation.identity:
        raise ValueError("History analysis allocation/source identity differs")
    final = writer.root.resolve(relative_root + "/receipt.json", for_write=True)
    if final.exists():
        if not recover:
            raise FileExistsError("History analysis exists; select receipt recovery explicitly")
        return read_matrix_history_analysis(writer=writer, relative_path=relative_root + "/receipt.json", config=config)
    env = _environment(config.source, repo_root)
    if type(source_receipt_paths) not in (tuple, list) or len(source_receipt_paths) > 256 or len(set(source_receipt_paths)) != len(source_receipt_paths):
        raise ValueError("History analysis explicit source receipt locators differ")
    sources, source_paths = {}, {}
    for path in source_receipt_paths:
        validate_relative_locator(path)
        payload_path = writer.root.resolve(path, for_write=False)
        manifest_path = writer.root.resolve(path + ".manifest.json", for_write=False)
        if not payload_path.exists() and not manifest_path.exists():
            continue
        source = read_matrix_history_source_receipt(writer=writer, relative_path=path, config=config.source, allocation=allocation)
        if source.root.history_index in sources:
            raise ValueError("History analysis has duplicate physical source roots")
        sources[source.root.history_index] = source
        source_paths[source.root.history_index] = path
    config_publication, allocation_publication = _publish_inputs(writer, config, allocation, relative_root)
    from empirical_lawhood.adapters.methods.matrix_history_analysis.analysis import algebra_history, passive_history, algebra_sample_indices, passive_cells
    roots = []
    for root in allocation.roots:
        relative = f"{relative_root}/h{root.history_index:03d}"
        prior = writer.root.resolve(relative + "/result.json", for_write=True)
        source = sources.get(root.history_index)
        if prior.exists():
            if not recover:
                raise FileExistsError("History partial analysis exists; select receipt recovery explicitly")
            result = _read_record_path(writer, relative + "/result.json", MatrixHistoryAnalysisRoot)
            expected_source = None if source is None else ObjectIdentity.from_record(source.receipt_id, source)
            if result.history_id != root.history_id or result.root_index != root.history_index or result.source_receipt != expected_source:
                raise ValueError("History recovered root changes its original explicit source receipt")
            _authenticate_analysis_root(result, config, writer)
            roots.append(result)
            continue
        if any(writer.root.resolve(relative + "/" + name, for_write=True).exists() for name in ("arrays.npz", "operand.json")):
            raise FileExistsError("Interrupted analysis lacks its root receipt; work was not silently repeated")
        identity = None if source is None else ObjectIdentity.from_record(source.receipt_id, source)
        expected_samples = sum(len(algebra_sample_indices(root, m)) for m in (1, 2)) if config.mode == "ALGEBRA" else 2 * len(passive_cells(root))
        if source is None or source.disposition != "COMPLETE":
            result = MatrixHistoryAnalysisRoot(root.history_index, root.history_id, "UNENTERED" if source is None else "UNEVALUABLE",
                "SOURCE_NOT_SUPPLIED" if source is None else source.reason, identity, None, None, None, expected_samples, 0, 0,
                source_receipt_path=source_paths.get(root.history_index))
        else:
            arrays = _authenticate_source(source, writer)
            import numpy as np
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    calculated, requested, identifiable, metrics = (algebra_history(arrays, root, compare_commutant_altered=config.compare_commutant_altered) if config.mode == "ALGEBRA" else passive_history(arrays, root))
            except (FloatingPointError, np.linalg.LinAlgError):
                result = MatrixHistoryAnalysisRoot(root.history_index, root.history_id, "UNEVALUABLE", "ANALYSIS_NUMERICAL_INVALID",
                    identity, None, None, None, expected_samples, 0, 0, source_receipt_path=source_paths[root.history_index])
            else:
                operand, publication, record_publication = _publish_arrays(writer, root=root, source=config.identity,
                    kind=config.mode, arrays=calculated, relative_root=relative, operand_id=config.config_id + f".h{root.history_index:03d}")
                result = MatrixHistoryAnalysisRoot(root.history_index, root.history_id, "COMPLETE", None, identity,
                    operand, publication, record_publication, requested, requested, identifiable, _decimal_metrics(metrics),
                    source_paths[root.history_index])
        _publish(writer, name=config.config_id + f".h{root.history_index:03d}.result", path=relative + "/result.json",
            relative_root=relative, payload=result.canonical_bytes(), schema=result.SCHEMA)
        roots.append(result)
    medians = []
    from statistics import median
    for index, family in enumerate(HISTORY_FAMILIES):
        group = roots[index * 64:(index + 1) * 64]
        names = sorted({name for r in group for name, _ in r.metrics})
        values = []
        for name in names:
            sample = [dict(r.metrics).get(name) for r in group]
            values.append((name, median(sample) if all(r.disposition == "COMPLETE" for r in group) and all(v is not None for v in sample) else None))
        medians.append((family, tuple(values)))
    columns = _publish(writer, name=config.config_id + ".columns", path=relative_root + "/columns.json", relative_root=relative_root,
        payload=canonical_json_bytes(_column_document(config.mode)), schema="empirical-lawhood/matrix-history-analysis/column-contract")
    receipt = MatrixHistoryAnalysisReceipt(config.config_id + ".receipt", config, allocation.identity, env,
        tuple(roots), tuple(medians), columns, config_publication, allocation_publication, tuple(source_receipt_paths),
        "COMPLETE" if all(r.disposition == "COMPLETE" for r in roots) else "UNEVALUABLE")
    _publish(writer, name=receipt.receipt_id, path=relative_root + "/receipt.json", relative_root=relative_root,
        payload=receipt.canonical_bytes(), schema=receipt.SCHEMA)
    return receipt


def read_matrix_history_analysis(*, writer, relative_path, config=None):
    """Verify exact retained publications; never reacquire, fit or recalculate."""
    receipt = _read_record_path(writer, relative_path, MatrixHistoryAnalysisReceipt)
    from empirical_lawhood.api.matrix_numerical_provenance import authenticate_recorded_matrix_provenance
    authenticate_recorded_matrix_provenance(receipt.environment.provenance)
    if config is not None and receipt.config != config:
        raise ValueError("History result differs from selected analysis configuration")
    allocation = _authenticate_inputs(writer, receipt.config, receipt.config_publication, receipt.allocation_publication)
    columns = _read_publication(writer, receipt.column_contract_publication, maximum_bytes=MAX_HISTORY_CONTROL_BYTES)
    if columns != canonical_json_bytes(_column_document(receipt.config.mode)):
        raise ValueError("History result column meanings differ from their owned contract")
    for root in receipt.roots:
        if root.history_id != allocation.roots[root.root_index].history_id:
            raise ValueError("History result has a foreign root within its full allocation")
        if root.arrays is not None and root.arrays.root != allocation.roots[root.root_index]:
            raise ValueError("History result changes its full numeric source allocation")
        if root.source_receipt_path is not None:
            source = _read_record_path(writer, root.source_receipt_path, MatrixHistorySourceReceipt)
            if ObjectIdentity.from_record(source.receipt_id, source) != root.source_receipt or source.source != receipt.config.source or source.root.history_index != root.root_index or source.root.history_id != root.history_id:
                raise ValueError("History result actual source receipt/current scientific source differs")
            _authenticate_source(source, writer)
        _authenticate_analysis_root(root, receipt.config, writer)
    return receipt


def _column_document(mode):
    from empirical_lawhood.adapters.methods.matrix_history_analysis.analysis import result_column_contract
    return {"schema": "empirical-lawhood/matrix-history-analysis/column-contract", "version": "1.0.0", "value": result_column_contract(mode)}


def matrix_history_example_records(repo_root):
    """Fixed exposed editable examples; changing labels cannot make them fresh."""
    allocation = prepare_matrix_history_allocation(allocation_id="public.matrix-history.allocation",
        namespace="public.matrix-history", master_seed=PUBLIC_HISTORY_MASTER_SEED)
    source = prepare_matrix_history_source_configuration(config_id="public.matrix-history.source",
        allocation=allocation, repo_root=repo_root)
    return {"allocation.json": allocation, "source.json": source,
        "algebra.json": MatrixHistoryAnalysisConfig("public.matrix-history.algebra", "ALGEBRA", source),
        "passive.json": MatrixHistoryAnalysisConfig("public.matrix-history.passive", "PASSIVE", source)}


def matrix_history_result_summary(receipt):
    return {"receipt_id": receipt.receipt_id, "receipt_sha256": receipt.fingerprint(),
        "mode": receipt.config.mode, "disposition": receipt.disposition, "evidence_role": receipt.evidence_role,
        "requested_histories": 256, "histories_per_family": 64,
        "completed_histories": sum(r.disposition == "COMPLETE" for r in receipt.roots),
        "unevaluable_histories": sum(r.disposition == "UNEVALUABLE" for r in receipt.roots),
        "unentered_histories": sum(r.disposition == "UNENTERED" for r in receipt.roots),
        "requested_nested_samples": sum(r.requested_samples for r in receipt.roots),
        "retained_nested_samples": sum(r.retained_samples for r in receipt.roots),
        "identifiable_y_samples": sum(r.identifiable_y_samples for r in receipt.roots),
        "family_medians": tuple((family, tuple((name, str(value) if value is not None else None) for name, value in values)) for family, values in receipt.family_medians),
        "historical_source_closure_verified": False, "original_selected_excursion_reproduced": False,
        "scientific_qualification_granted": False, "reader_performs_scientific_execution": False}
