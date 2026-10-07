"""Explicit current preparation inputs, analyses, retained results and recovery.

These are exposed development calculations. No issue, execution or reveal grant
is manufactured. Current native data and original F are authenticated separately;
historical filenames, recipes and documentation never substitute for inputs.
"""

from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from importlib.resources import files

import numpy as np

from empirical_lawhood.api.authoring_handoff import preflight_authoring_output_directory
from empirical_lawhood.api.configuration import _write_new_file
from empirical_lawhood.api.finite_operands import (
    _read_selected, original_f_operand_import, saved_matrix_arrays_export,
    saved_matrix_arrays_import,
)
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import TransientBridgeInput, TransientBridgeSpec
from empirical_lawhood.adapters.methods.matrix_preparation_analysis.records import (
    BaselineSupportConfig, PreparationAnalysisMetric, PreparationAnalysisReport, PreparationAnalysisInputs,
    TransientAnalysisConfig,
    SPECIFICATION,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.api.matrix_numerical_provenance import (
    capture_matrix_provenance, numerical_package_code_sha256,
    selected_dependency_lock_sha256, authenticate_recorded_matrix_provenance,
)
from empirical_lawhood.infrastructure.retained_analysis import (
    publish_retained_analysis, read_retained_analysis, read_retained_analysis_member,
)
from empirical_lawhood.infrastructure.matrix_array_io import decode_matrix_arrays
from empirical_lawhood.kernel.matrix_inputs import SavedMatrixArrays, MatrixAllocation, MAXIMUM_MATRIX_ARRAY_BYTES
from empirical_lawhood.runtime.retained_analysis import RetainedAnalysisMember



def current_preparation_analysis_sources_sha256() -> str:
    """Actual installed producing-owner bytes, without native import/discovery."""
    return numerical_package_code_sha256()


def _require_specification():
    with files("empirical_lawhood").joinpath("resources", "matrix_preparation_analysis", "specification.canonical.json").open("rb") as source:
        raw = source.read(64 * 1024 + 1)
    if raw != SPECIFICATION.canonical_bytes():
        raise ValueError("Installed current preparation specification differs from its closed operative identity")


def _record(path, record_type, maximum=1024**2):
    return decode_canonical_bytes(_read_selected(path, maximum_bytes=maximum), record_type, maximum_bytes=maximum)


def _manifest_artifact(manifest):
    raw = manifest.canonical_bytes()
    return ArtifactIdentity(manifest.operand_id + ".manifest", "current-array-manifest", manifest.SCHEMA,
                            sha256(raw).hexdigest(), "application/json", len(raw))


def _saved(directory: Path, *, artifact_writer):
    return saved_matrix_arrays_import(directory / "arrays.canonical.json", arrays_path=directory / "arrays.npz",
                                     allocation_path=directory / "allocation.canonical.json", artifact_writer=artifact_writer)


def _original(directory: Path):
    return original_f_operand_import(directory / "original-f.canonical.json", source_directory=directory)


def _metrics(values):
    return tuple(PreparationAnalysisMetric(name, None if value is None or not np.isfinite(value) else Decimal(str(float(value))))
                 for name, value in sorted(values.items()))


def _record_artifact(record, record_id, role):
    return ArtifactIdentity(record_id, role, record.SCHEMA, record.fingerprint(), "application/json", len(record.canonical_bytes()))


def _complete(output, *, artifact_writer, report, config, manifest, analysis_kind, roles,
              arrays, allocation=None, source_members=(), upstream_members=()):
    inputs = PreparationAnalysisInputs(report.report_id + ".inputs", config.identity, analysis_kind, report.inputs, roles, upstream_members)
    _validate_report_semantics(report, config=config, arrays=arrays, inputs=inputs)
    _write_new_file(output / "analysis-inputs.canonical.json", inputs.canonical_bytes())
    members = [
        RetainedAnalysisMember("analysis-report.canonical.json", _record_artifact(report, report.report_id, "analysis-report")),
        RetainedAnalysisMember("analysis-config.canonical.json", _record_artifact(config, report.report_id + ".config", "analysis-config")),
        RetainedAnalysisMember("analysis-inputs.canonical.json", _record_artifact(inputs, inputs.input_id, "analysis-inputs")),
        RetainedAnalysisMember("arrays.canonical.json", _manifest_artifact(manifest)),
        RetainedAnalysisMember("arrays.npz", manifest.array_artifact),
        *source_members,
    ]
    if allocation is not None:
        members.append(RetainedAnalysisMember("allocation.canonical.json", _record_artifact(allocation, report.report_id + ".allocation", "allocation")))
    publish_retained_analysis(directory=output, artifact_writer=artifact_writer,
                              completion_id=report.report_id + ".completion", members=tuple(members))


def _retained_record(completion, name, kind, *, directory, artifact_writer, maximum=1024**2):
    raw = read_retained_analysis_member(completion, name, directory=directory, artifact_writer=artifact_writer, maximum_bytes=maximum)
    return decode_canonical_bytes(raw, kind, maximum_bytes=maximum)


def _read_terminal(directory, *, artifact_writer):
    completion = read_retained_analysis(directory=directory, artifact_writer=artifact_writer)
    report = _retained_record(completion, "analysis-report.canonical.json", PreparationAnalysisReport, directory=directory, artifact_writer=artifact_writer, maximum=4 * 1024**2)
    inputs = _retained_record(completion, "analysis-inputs.canonical.json", PreparationAnalysisInputs, directory=directory, artifact_writer=artifact_writer)
    authenticate_recorded_matrix_provenance(report.provenance)
    if (completion.completion_id != report.report_id + ".completion" or inputs.configuration != report.configuration
            or inputs.artifacts != report.inputs):
        raise ValueError("Retained preparation terminal changes its exact input/configuration custody")
    return completion, report, inputs


def _validate_terminal_members(completion, *, report, config, manifest, inputs, allocation=None):
    expected = {
        "analysis-report.canonical.json": _record_artifact(report, report.report_id, "analysis-report"),
        "analysis-config.canonical.json": _record_artifact(config, report.report_id + ".config", "analysis-config"),
        "analysis-inputs.canonical.json": _record_artifact(inputs, inputs.input_id, "analysis-inputs"),
        "arrays.canonical.json": _manifest_artifact(manifest), "arrays.npz": manifest.array_artifact,
    }
    if allocation is not None:
        expected["allocation.canonical.json"] = _record_artifact(allocation, report.report_id + ".allocation", "allocation")
    if inputs.analysis_kind == "TANGENT_SOURCE":
        expected.update((value.artifact_id, value) for value in manifest.source_artifacts)
    if {value.relative_path: value.artifact for value in completion.members} != expected:
        raise ValueError("Retained preparation completion changes its exact required member census/schema")


_TRANSIENT_INPUT_ROLES = (
    "native-arrays", "native-manifest", "bridge-arrays", "bridge-manifest", "bridge-input", "bridge-spec",
    "original-f-operand", "original-payload", "original-manifest", "original-commit", "original-calibration", "original-qualification",
)


def _metric_names(kind, *, incomplete):
    if kind == "TANGENT_SOURCE":
        return {"independent_source_roots", "complete_source_roots", "independent_mechanism_roots"}
    if kind == "TANGENT_ANALYSIS":
        return set(("independent_source_roots", "independent_view_roots", "independent_mechanism_roots",
                    "nested_mechanism_comparisons", "complete_source_roots", "complete_mechanism_roots",
                    "native_view_passing_roots", "numerical_passing_roots", "absolute_view_failing_roots",
                    "odd_view_failing_roots", "secant_failing_roots", "coupled_tangent_mse", "scalar_tangent_mse",
                    "scalar_minus_coupled_mse"))
    if kind == "BASELINE_SUPPORT":
        if incomplete:
            return {"support.total_cells", "independent_roots", "complete_roots"}
        return {"independent_roots", "complete_roots", "support.total_cells", "support.failure_roots",
                "support.true_supported", "support.true_unsupported", "support.false_supported", "support.false_unsupported",
                "feature.baseline_mse", "feature.transient_mse", "feature.cross_term", "feature.total_mse",
                "response.lower_mse", "response.baseline_mse", "response.transient_mse", "response.total_mse",
                "response.cross_lower_baseline", "response.cross_lower_transient", "response.cross_baseline_transient"}
    if incomplete:
        return {"independent_roots", "complete_roots", "outer_fits", "inner_fits"}
    prefixes = tuple(model + "." + regime for model in ("radial", "nonrestoring")
                     for regime in ("all", "early_time", "duration_transfer", "recovery_transfer"))
    names = {"independent_roots", "complete_roots", "outer_fits", "new_outer_fits", "reused_outer_fits", "inner_fits"}
    names.update(prefix + "." + part + "_mse_by_root.summary"
                 for prefix in ("actual_handoff_f", "current_u2", "predicted_hold_only", *prefixes)
                 for part in ("feature", "response"))
    names.update(prefix + "." + part + "_mse_by_root.summary" for prefix in prefixes
                 for part in ("measured_hold_diagnostic.response", "increment", "transfer", "trajectory", "late"))
    names.update(model + ".all." + role for model in ("radial", "nonrestoring")
                 for role in ("exploratory_covered_roots", "enclosed_support_cells"))
    return names


def _validate_report_semantics(report, *, config, arrays, inputs):
    kind = inputs.analysis_kind
    values = {metric.name: metric.value for metric in report.metrics}
    if set(values) != _metric_names(kind, incomplete=bool(report.incomplete_roots)):
        raise ValueError("Preparation report changes its closed metric roles")
    if any(value is not None and value < 0 and ("cross" not in name and name != "scalar_minus_coupled_mse")
           for name, value in values.items()):
        raise ValueError("Preparation report has an impossible nonnegative diagnostic/count")
    def count(name, expected):
        if values.get(name) != expected:
            raise ValueError("Preparation report count differs from its retained observation census")
    def boolean(name, shape):
        value = arrays.get(name)
        if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
            raise ValueError("Preparation report lacks its exact retained missingness census")
        return value
    if kind.startswith("TANGENT"):
        from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import TangentPreparationConfig
        if (type(config) is not TangentPreparationConfig or report.provenance.dependency_lock_sha256 != config.dependency_lock_sha256
                or report.provenance.code_sources_sha256 != config.code_sources_sha256):
            raise ValueError("Tangent report changes its selected configuration/lock")
        shard_names = tuple(root.root_id + suffix for root in config.roots
                            for suffix in (".prefix.canonical.json", ".prefix.h5", ".native.canonical.json", ".native.h5"))
        expected_roles = shard_names if kind == "TANGENT_SOURCE" else ("source-report", "source-arrays", "source-manifest", *shard_names)
        if inputs.roles != expected_roles or tuple(value.artifact_id for value in inputs.artifacts[-512:]) != shard_names:
            raise ValueError("Tangent report changes its full source input role census")
        complete = boolean("native_complete", (128,))
        if kind == "TANGENT_SOURCE":
            complete = complete & boolean("prefix_complete", (128, 2)).all(axis=1)
            incomplete = ~complete
        else:
            selected = boolean("numerical_subset", (128,))
            if not np.array_equal(selected, [root.numerical_semantics for root in config.roots]):
                raise ValueError("Tangent report changes its pre-outcome numerical subset")
            mechanism = boolean("mechanism_complete", (128,))
            numerical_complete = np.isfinite(arrays["absolute_view_error"]).reshape(128, -1).all(axis=1) & np.isfinite(arrays["odd_view_error"]).reshape(128, -1).all(axis=1)
            incomplete = ~complete | ~numerical_complete | (selected & ~mechanism)
            count("independent_view_roots", 128)
            count("nested_mechanism_comparisons", 6400)
            count("complete_mechanism_roots", int(np.sum(selected & mechanism)))
            count("native_view_passing_roots", int(boolean("native_view_pass", (128,)).sum()))
            count("numerical_passing_roots", int(boolean("numerical_pass", (128,)).sum()))
        count("independent_source_roots", 128)
        count("independent_mechanism_roots", 32)
        count("complete_source_roots", int(complete.sum()))
    else:
        expected_kind = "TRANSIENT_ANALYSIS" if type(config) is TransientAnalysisConfig else "BASELINE_SUPPORT"
        expected_roles = _TRANSIENT_INPUT_ROLES if kind == "TRANSIENT_ANALYSIS" else ("transient-report", "transient-arrays", "transient-manifest", *("parent." + role for role in _TRANSIENT_INPUT_ROLES))
        if kind != expected_kind or inputs.roles != expected_roles:
            raise ValueError("Preparation report changes its closed scientific input roles")
        incomplete = boolean("missing_roots", (24,))
        count("independent_roots", 24)
        count("complete_roots", 24 - int(incomplete.sum()))
        if kind == "TRANSIENT_ANALYSIS":
            count("outer_fits", 0 if incomplete.any() else 32)
            count("inner_fits", 0 if incomplete.any() else 24)
        else:
            count("support.total_cells", 216)
            if not incomplete.any():
                actual, predicted = boolean("actual_support", (24, 9)), boolean("predicted_support", (24, 9))
                count("support.true_supported", int(np.sum(actual & predicted)))
                count("support.true_unsupported", int(np.sum(~actual & ~predicted)))
                count("support.false_supported", int(np.sum(~actual & predicted)))
                count("support.false_unsupported", int(np.sum(actual & ~predicted)))
                count("support.failure_roots", int(np.any(~actual, axis=1).sum()))
    expected_missing = tuple(sorted(root for root, missing in zip(report.root_ids, incomplete, strict=True) if missing))
    if report.incomplete_roots != expected_missing or (report.disposition == "COMPLETE") != (not incomplete.any()):
        raise ValueError("Preparation report disposition differs from its retained array missingness")


def _retain(destination, *, config, allocation, arrays, inputs, lower, incomplete, metrics,
            artifact_writer, provenance, analysis_kind, roles, upstream_members=()):
    destination.mkdir()
    _write_new_file(destination / "analysis-config.canonical.json", config.canonical_bytes())
    manifest = saved_matrix_arrays_export(destination, operand_id=config.analysis_id + ".arrays", allocation=allocation,
                                         arrays=arrays, producers=(config.identity,), source_artifacts=inputs,
                                         original_f=lower.identity,
                                         nonfinite_members=tuple(sorted(name for name, value in arrays.items() if value.dtype.kind in "fc" and not np.isfinite(value).all())))
    report = PreparationAnalysisReport(config.analysis_id + ".report", config.identity, inputs, manifest.array_artifact,
                                       tuple(root.root_id for root in allocation.roots), tuple(sorted(incomplete)),
                                       _metrics(metrics), "UNEVALUABLE" if incomplete else "COMPLETE",
                                       implementation_sources_sha256=provenance.code_sources_sha256, provenance=provenance)
    # This terminal is written last. A partial attempt cannot become a result.
    _write_new_file(destination / "analysis-report.canonical.json", report.canonical_bytes())
    _complete(destination, artifact_writer=artifact_writer, report=report, config=config, manifest=manifest,
              arrays=arrays, allocation=allocation, analysis_kind=analysis_kind, roles=roles, upstream_members=upstream_members)
    return report


def transient_response_analysis(output_directory: Path, *, config: TransientAnalysisConfig,
                                native_directory: Path, bridge_directory: Path,
                                original_f_directory: Path, artifact_writer, project_root: Path) -> PreparationAnalysisReport:
    """Fit fixed transfer/inner folds from authenticated I2R current operands."""
    output = preflight_authoring_output_directory(directory=output_directory, repo_root=project_root, artifact_writer=artifact_writer)
    if type(config) is not TransientAnalysisConfig:
        raise ValueError("Transient analysis requires its strict configuration")
    _require_specification()
    provenance = capture_matrix_provenance(project_root=project_root)
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.transient import transient_analysis, validate_operands

    lower = _original(original_f_directory)
    native, native_allocation, native_arrays = _saved(native_directory, artifact_writer=artifact_writer)
    bridge, allocation, arrays = _saved(bridge_directory, artifact_writer=artifact_writer)
    source = _record(bridge_directory / "bridge-input.canonical.json", TransientBridgeInput)
    spec = _record(bridge_directory / "bridge-spec.canonical.json", TransientBridgeSpec)
    native_artifact = _manifest_artifact(native)
    if (native_allocation != allocation or native.original_f != lower.identity or bridge.original_f != lower.identity
            or source.original_f != lower.identity or source.allocation != allocation.identity
            or source.root_ids != tuple(root.root_id for root in allocation.roots)
            or source.root_cohorts != tuple(root.cohort for root in allocation.roots)
            or source.spec != spec or source.identity not in bridge.producers
            or source.trajectory != native.array_artifact
            or source.native_panel.sha256 != native_artifact.sha256
            or source.native_panel.size_bytes != native_artifact.size_bytes
            or native.array_artifact not in bridge.source_artifacts
            or source.native_panel not in bridge.source_artifacts):
        raise ValueError("Transient analysis current native/bridge/original-F provenance differs")
    for name in ("x", "z", "y", "features"):
        if name not in native_arrays or name not in arrays or not np.array_equal(native_arrays[name], arrays[name], equal_nan=True):
            raise ValueError("Transient bridge changes authenticated current native operands")
    if ("requested_ticks" not in native_arrays or not np.array_equal(native_arrays["requested_ticks"], np.tile(np.arange(0, 401, 16), (24, 1)) + 4096)
            or tuple(root.cohort for root in allocation.roots) != ("q2",) * 8 + ("cir1",) * 16):
        raise ValueError("Transient analysis changes source clocks or complete cohort order")
    incomplete_indices = set(validate_operands(arrays, lower))
    for name in ("parent_complete", "response_observed"):
        value = native_arrays.get(name)
        expected = (24, 9, 2) if name == "parent_complete" else (24, 9, 4, 8, 2, 2)
        if not isinstance(value, np.ndarray) or value.shape != expected or value.dtype != np.bool_:
            raise ValueError("Transient analysis lacks the complete native observation census")
        incomplete_indices.update(np.flatnonzero(~value.reshape(24, -1).all(axis=1)).tolist())
    inputs = (native.array_artifact, native_artifact, bridge.array_artifact, _manifest_artifact(bridge),
              _record_artifact(source, source.operand_id, "bridge-input"),
              _record_artifact(spec, config.analysis_id + ".bridge-spec", "bridge-spec"),
              _record_artifact(lower, lower.payload_id, "original-f-operand"), *lower.source_identities)
    roles = ("native-arrays", "native-manifest", "bridge-arrays", "bridge-manifest", "bridge-input", "bridge-spec",
             "original-f-operand", "original-payload", "original-manifest", "original-commit", "original-calibration", "original-qualification")
    incomplete = tuple(allocation.roots[index].root_id for index in sorted(incomplete_indices))
    if incomplete:
        out = {"missing_roots": np.asarray([index in incomplete_indices for index in range(24)], dtype=np.bool_)}
        metrics = {"independent_roots": 24, "complete_roots": 24 - len(incomplete), "outer_fits": 0, "inner_fits": 0}
    else:
        out = transient_analysis(arrays, lower)
        out["missing_roots"] = np.zeros(24, dtype=np.bool_)
        selection = slice(None) if config.summary_population == "all" else slice(0, 8) if config.summary_population == "q2" else slice(8, 24)
        metrics = {"independent_roots": 24, "complete_roots": 24, "outer_fits": 32, "new_outer_fits": 28, "reused_outer_fits": 4, "inner_fits": 24}
        for name, value in out.items():
            if name.endswith("mse_by_root"):
                metrics[name + ".summary"] = float(np.mean(value[selection]))
        for key in ("radial.all", "nonrestoring.all"):
            metrics[key + ".exploratory_covered_roots"] = int(out[key + ".exploratory_response_covered"].sum())
            metrics[key + ".enclosed_support_cells"] = int(out[key + ".enclosed_support"].sum())
    return _retain(output, config=config, allocation=allocation, arrays=out, inputs=inputs, lower=lower,
                   incomplete=incomplete, metrics=metrics, artifact_writer=artifact_writer, provenance=provenance,
                   analysis_kind="TRANSIENT_ANALYSIS", roles=roles, upstream_members=(*native.source_artifacts, *bridge.source_artifacts))


def preparation_analysis_read(directory: Path, *, artifact_writer) -> tuple[PreparationAnalysisReport, object, object]:
    """Authenticate a completed result; absence means partial-output recovery."""
    completion, report, inputs = _read_terminal(directory, artifact_writer=artifact_writer)
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import TangentPreparationConfig
    if report.configuration.object_schema == TangentPreparationConfig.SCHEMA:
        return tangent_analysis_read(directory, artifact_writer=artifact_writer)
    types = {TransientAnalysisConfig.SCHEMA: TransientAnalysisConfig, BaselineSupportConfig.SCHEMA: BaselineSupportConfig}
    record_type = types.get(report.configuration.object_schema)
    if record_type is None:
        raise ValueError("Unsupported preparation analysis configuration")
    config = _retained_record(completion, "analysis-config.canonical.json", record_type, directory=directory, artifact_writer=artifact_writer)
    manifest = _retained_record(completion, "arrays.canonical.json", SavedMatrixArrays, directory=directory, artifact_writer=artifact_writer)
    allocation = _retained_record(completion, "allocation.canonical.json", MatrixAllocation, directory=directory, artifact_writer=artifact_writer)
    arrays = decode_matrix_arrays(read_retained_analysis_member(completion, "arrays.npz", directory=directory,
        artifact_writer=artifact_writer, maximum_bytes=MAXIMUM_MATRIX_ARRAY_BYTES), manifest)
    if (config.identity != report.configuration or manifest.producers != (config.identity,)
            or report.arrays != manifest.array_artifact or report.inputs != manifest.source_artifacts
            or report.root_ids != tuple(root.root_id for root in allocation.roots) or manifest.allocation != allocation.identity):
        raise ValueError("Retained preparation analysis changes its result/config/input/array identity")
    _validate_terminal_members(completion, report=report, config=config, manifest=manifest, inputs=inputs, allocation=allocation)
    _validate_report_semantics(report, config=config, arrays=arrays, inputs=inputs)
    return report, allocation, arrays


def baseline_support_analysis(output_directory: Path, *, config: BaselineSupportConfig,
                              transient_directory: Path, original_f_directory: Path, artifact_writer, project_root: Path) -> PreparationAnalysisReport:
    """Evaluate saved P08 forecasts; performs no fit or native replay."""
    output = preflight_authoring_output_directory(directory=output_directory, repo_root=project_root, artifact_writer=artifact_writer)
    if type(config) is not BaselineSupportConfig:
        raise ValueError("Baseline analysis requires its strict configuration")
    _require_specification()
    provenance = capture_matrix_provenance(project_root=project_root)
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.transient import baseline_support_analysis as evaluate

    lower = _original(original_f_directory)
    parent, allocation, arrays = preparation_analysis_read(transient_directory, artifact_writer=artifact_writer)
    parent_completion, _, parent_inputs = _read_terminal(transient_directory, artifact_writer=artifact_writer)
    manifest = _retained_record(parent_completion, "arrays.canonical.json", SavedMatrixArrays, directory=transient_directory, artifact_writer=artifact_writer)
    if parent.configuration.object_schema != TransientAnalysisConfig.SCHEMA or manifest.original_f != lower.identity:
        raise ValueError("Baseline analysis requires the completed current transient output and unchanged original F")
    inputs = (_record_artifact(parent, parent.report_id, "transient-report"), parent.arrays, _manifest_artifact(manifest), *parent.inputs)
    roles = ("transient-report", "transient-arrays", "transient-manifest", *("parent." + role for role in parent_inputs.roles))
    if parent.disposition == "UNEVALUABLE":
        out = {"missing_roots": np.asarray([root.root_id in parent.incomplete_roots for root in allocation.roots], dtype=np.bool_)}
        metrics = {"support.total_cells": 216, "independent_roots": 24, "complete_roots": 24 - len(parent.incomplete_roots)}
    else:
        out, metrics = evaluate(arrays, lower, model=config.model)
        metrics.update(independent_roots=24, complete_roots=24)
    return _retain(output, config=config, allocation=allocation, arrays=out, inputs=inputs, lower=lower,
                   incomplete=parent.incomplete_roots, metrics=metrics, artifact_writer=artifact_writer,
                   provenance=provenance, analysis_kind="BASELINE_SUPPORT", roles=roles, upstream_members=parent_inputs.upstream_members)


def tangent_preparation_configuration(*, config_id: str, namespace: str, master_seed: int, project_root: Path, summary_context="all"):
    """Prepare the explicit full current numerical census without native contact."""
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import proposed_tangent_configuration
    return proposed_tangent_configuration(config_id=config_id, namespace=namespace, master_seed=master_seed,
                                          summary_context=summary_context, code_sources_sha256=current_preparation_analysis_sources_sha256(),
                                          dependency_lock_sha256=selected_dependency_lock_sha256(project_root))


def _tangent_export_arrays(destination, *, config, arrays, sources, operand_id):
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import SavedTangentArrays
    from empirical_lawhood.infrastructure.matrix_array_io import encode_matrix_arrays
    from empirical_lawhood.kernel.matrix_inputs import MATRIX_ARRAY_SCHEMA
    raw, members = encode_matrix_arrays(arrays, nonfinite_members=tuple(sorted(name for name, value in arrays.items() if value.dtype.kind in "fc" and not np.isfinite(value).all())))
    artifact = ArtifactIdentity(operand_id + ".arrays", "current-tangent-arrays", MATRIX_ARRAY_SCHEMA,
                                sha256(raw).hexdigest(), "application/x-npz", len(raw))
    manifest = SavedTangentArrays(operand_id, config.identity, artifact, members, sources)
    _write_new_file(destination / "arrays.npz", raw)
    _write_new_file(destination / "arrays.canonical.json", manifest.canonical_bytes())
    return manifest


def _tangent_saved(directory, *, completion, artifact_writer):
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import SavedTangentArrays, TangentPreparationConfig
    from empirical_lawhood.infrastructure.matrix_array_io import decode_matrix_arrays
    from empirical_lawhood.kernel.matrix_inputs import MAXIMUM_MATRIX_ARRAY_BYTES
    manifest = _retained_record(completion, "arrays.canonical.json", SavedTangentArrays, directory=directory, artifact_writer=artifact_writer)
    config = _retained_record(completion, "analysis-config.canonical.json", TangentPreparationConfig, directory=directory, artifact_writer=artifact_writer)
    if config.identity != manifest.configuration:
        raise ValueError("Saved current tangent input changes its exact numerical configuration")
    arrays = decode_matrix_arrays(read_retained_analysis_member(completion, "arrays.npz", directory=directory,
        artifact_writer=artifact_writer, maximum_bytes=MAXIMUM_MATRIX_ARRAY_BYTES), manifest)
    return manifest, config, arrays


def tangent_preparation_source_export(output_directory: Path, *, config, artifact_writer, project_root: Path, progress=None):
    """Acquire the fixed full128 current source into one explicit fresh directory.

    Per-root source and native artifacts precede the final manifest. Interruption
    retains completed members but no completed source report. Recovery uses a new
    output path; no resume, overwrite, scheduler or authority is synthesized.
    """
    output = preflight_authoring_output_directory(directory=output_directory, repo_root=project_root, artifact_writer=artifact_writer)
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent import acquire_tangent_root
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import TangentPreparationConfig
    if type(config) is not TangentPreparationConfig:
        raise ValueError("Tangent source requires its strict full current configuration")
    _require_specification()
    provenance = capture_matrix_provenance(project_root=project_root, expected_code_sha256=config.code_sources_sha256,
                                           expected_lock_sha256=config.dependency_lock_sha256)
    output.mkdir()
    _write_new_file(output / "analysis-config.canonical.json", config.canonical_bytes())
    projected = []
    source_artifacts = []
    for root in config.roots:
        prefix, prefix_hdf5, native, native_hdf5, arrays = acquire_tangent_root(config, root, progress=progress)
        for name, raw, schema, media in (
            (root.root_id + ".prefix.canonical.json", prefix.canonical_bytes(), prefix.SCHEMA, "application/json"),
            (root.root_id + ".prefix.h5", prefix_hdf5, "empirical-lawhood/methods/matrix-preparation-analysis/current-prefix-hdf5", "application/x-hdf5"),
            (root.root_id + ".native.canonical.json", native.canonical_bytes(), native.SCHEMA, "application/json"),
            (root.root_id + ".native.h5", native_hdf5, "empirical-lawhood/simulators/matrix-preparation/native-observations-hdf5", "application/x-hdf5"),
        ):
            _write_new_file(output / name, raw)
            source_artifacts.append(ArtifactIdentity(name, "current-tangent-source", schema, sha256(raw).hexdigest(), media, len(raw)))
        projected.append(arrays)
    arrays = {name: np.stack([row[name] for row in projected]) for name in projected[0]}
    manifest = _tangent_export_arrays(output, config=config, arrays=arrays, sources=tuple(source_artifacts), operand_id=config.config_id + ".source")
    incomplete = tuple(root.root_id for index, root in enumerate(config.roots) if not arrays["native_complete"][index] or not arrays["prefix_complete"][index].all())
    report = PreparationAnalysisReport(config.config_id + ".source-report", config.identity, tuple(source_artifacts),
                                       manifest.array_artifact, tuple(root.root_id for root in config.roots), tuple(sorted(incomplete)),
                                       _metrics({"independent_source_roots": 128, "complete_source_roots": 128 - len(incomplete),
                                                 "independent_mechanism_roots": 32}), "UNEVALUABLE" if incomplete else "COMPLETE",
                                       implementation_sources_sha256=provenance.code_sources_sha256, provenance=provenance)
    _write_new_file(output / "analysis-report.canonical.json", report.canonical_bytes())
    _complete(output, artifact_writer=artifact_writer, report=report, config=config, manifest=manifest,
              arrays=arrays, analysis_kind="TANGENT_SOURCE", roles=tuple(value.artifact_id for value in source_artifacts),
              source_members=tuple(RetainedAnalysisMember(value.artifact_id, value) for value in source_artifacts))
    return report


def tangent_response_analysis(output_directory: Path, *, config, source_directory: Path, artifact_writer, project_root: Path):
    """Evaluate current realized paths; tangent inputs remain retrospective."""
    output = preflight_authoring_output_directory(directory=output_directory, repo_root=project_root, artifact_writer=artifact_writer)
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent import tangent_analysis
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import TangentPreparationConfig, TangentNativeResult, TangentPrefixResult
    if type(config) is not TangentPreparationConfig:
        raise ValueError("Tangent evaluator requires its strict configuration")
    _require_specification()
    provenance = capture_matrix_provenance(project_root=project_root, expected_code_sha256=config.code_sources_sha256,
                                           expected_lock_sha256=config.dependency_lock_sha256)
    source_completion, source_report, source_inputs = _read_terminal(source_directory, artifact_writer=artifact_writer)
    if source_inputs.analysis_kind != "TANGENT_SOURCE":
        raise ValueError("Tangent evaluator requires an authenticated source terminal")
    manifest, source_config, arrays = _tangent_saved(source_directory, completion=source_completion, artifact_writer=artifact_writer)
    _validate_terminal_members(source_completion, report=source_report, config=source_config, manifest=manifest, inputs=source_inputs)
    _validate_report_semantics(source_report, config=source_config, arrays=arrays, inputs=source_inputs)
    if (source_config.roots != config.roots or source_report.configuration != source_config.identity
            or source_report.arrays != manifest.array_artifact
            or source_report.inputs != manifest.source_artifacts
            or source_report.root_ids != tuple(root.root_id for root in config.roots)):
        raise ValueError("Tangent evaluator changes the authenticated current source census")
    # No native replay occurs. Authenticate every requested root shard first.
    expected_names = tuple(root.root_id + suffix for root in config.roots for suffix in (".prefix.canonical.json", ".prefix.h5", ".native.canonical.json", ".native.h5"))
    if tuple(item.artifact_id for item in manifest.source_artifacts) != expected_names:
        raise ValueError("Tangent source manifest changes the complete per-root member census")
    for index, root in enumerate(config.roots):
        group = manifest.source_artifacts[4 * index:4 * index + 4]
        canonical = []
        for identity in group:
            raw = read_retained_analysis_member(source_completion, identity.artifact_id, directory=source_directory,
                                               artifact_writer=artifact_writer, maximum_bytes=96 * 1024**2)
            if len(raw) != identity.size_bytes or sha256(raw).hexdigest() != identity.sha256:
                raise ValueError("Current tangent source member differs from its authenticated bytes")
            if identity.media_type == "application/json":
                canonical.append(raw)
        prefix = decode_canonical_bytes(canonical[0], TangentPrefixResult, maximum_bytes=1024**2)
        native = decode_canonical_bytes(canonical[1], TangentNativeResult, maximum_bytes=1024**2)
        if (prefix.root != root or prefix.configuration != source_config.identity or native.root != root
                or native.source_config != source_config.identity or native.prefix.sha256 != group[0].sha256
                or native.prefix.size_bytes != group[0].size_bytes or native.observations_sha256 != group[3].sha256
                or not np.array_equal(arrays["prefix_complete"][index], [view.disposition == "COMPLETE" for view in prefix.views])
                or arrays["native_complete"][index] != all(delivery.disposition == "COMPLETE" for delivery in native.deliveries)):
            raise ValueError("Current tangent source records/projection/root/clock joins differ")
    out, metrics, incomplete = tangent_analysis(config, arrays)
    output.mkdir()
    _write_new_file(output / "analysis-config.canonical.json", config.canonical_bytes())
    input_artifact = _manifest_artifact(manifest)
    sources = (_record_artifact(source_report, source_report.report_id, "tangent-source-report"), manifest.array_artifact,
               input_artifact, *manifest.source_artifacts)
    retained = _tangent_export_arrays(output, config=config, arrays=out, sources=sources, operand_id=config.config_id + ".analysis")
    report = PreparationAnalysisReport(config.config_id + ".analysis-report", config.identity, sources, retained.array_artifact,
                                       tuple(root.root_id for root in config.roots), tuple(sorted(incomplete)), _metrics(metrics),
                                       "UNEVALUABLE" if incomplete else "COMPLETE",
                                       implementation_sources_sha256=provenance.code_sources_sha256, provenance=provenance)
    _write_new_file(output / "analysis-report.canonical.json", report.canonical_bytes())
    _complete(output, artifact_writer=artifact_writer, report=report, config=config, manifest=retained,
              arrays=out, analysis_kind="TANGENT_ANALYSIS", roles=("source-report", "source-arrays", "source-manifest", *source_inputs.roles))
    return report


def tangent_analysis_read(directory: Path, *, artifact_writer):
    """Read exact completed tangent bytes, with partial-output refusal."""
    completion, report, inputs = _read_terminal(directory, artifact_writer=artifact_writer)
    manifest, config, arrays = _tangent_saved(directory, completion=completion, artifact_writer=artifact_writer)
    if (report.configuration != config.identity or report.arrays != manifest.array_artifact
            or report.inputs != manifest.source_artifacts
            or report.root_ids != tuple(root.root_id for root in config.roots)):
        raise ValueError("Retained tangent result differs from its configuration/census/bytes")
    _validate_terminal_members(completion, report=report, config=config, manifest=manifest, inputs=inputs)
    _validate_report_semantics(report, config=config, arrays=arrays, inputs=inputs)
    return report, config, arrays
