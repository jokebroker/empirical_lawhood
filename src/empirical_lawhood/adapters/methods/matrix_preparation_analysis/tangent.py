"""Current prefix/native producer and retrospective tangent evaluator.

Native integration and every tangent/secant step stay with their existing
owners. This module selects the fixed chart and composes those mechanics; it
does not replace realized paths with an analytical surrogate.
"""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO

import numpy as np

from empirical_lawhood.adapters.methods.matrix_preparation.projection import PROJECTION_SHAPES, project_preparation_arrays
from empirical_lawhood.adapters.methods.matrix_preparation.numerical_semantics import native_view_errors
from empirical_lawhood.adapters.simulators.matrix_preparation.source import execute_preparation_panel
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode, _encode
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import _run_view, response_hdf5_group, response_hdf5_text, response_hdf5_writer
from empirical_lawhood.kernel.references import ArtifactIdentity
from .tangent_records import TangentNativeResult, TangentPreparationConfig, TangentPrefixResult, TangentPrefixSegment


def acquire_tangent_root(config: TangentPreparationConfig, root, *, progress=None):
    """Full current native root; imported historical prefixes are not accepted."""
    if type(config) is not TangentPreparationConfig or root not in config.roots:
        raise ValueError("Current tangent source requires its strict complete allocation")
    segment = TangentPrefixSegment(root)
    prefix_stream = BytesIO()
    with response_hdf5_writer(prefix_stream) as artifact:
        response_hdf5_text(artifact, "schema", "empirical-lawhood/methods/matrix-preparation-analysis/current-prefix-hdf5")
        response_hdf5_text(artifact, "source_config_sha256", config.fingerprint())
        response_hdf5_text(artifact, "root_sha256", root.fingerprint())
        views = tuple(_run_view(config, segment, refinement, None, response_hdf5_group(artifact, f"r{refinement}"), progress=progress)
                      for refinement in (1, 2))
    primary = views[0].checkpoint
    mode = None if primary is None or primary.frozen_mode_base64 is None else _decode(primary.frozen_mode_base64, (3, 4, 4))
    views = tuple(replace(view, checkpoint=replace(view.checkpoint, frozen_mode_base64=None if mode is None else _encode(mode)))
                  if view.checkpoint is not None else view for view in views)
    prefix = TangentPrefixResult(root.root_id + ".prefix-result", config.identity, root, views)
    prefix_raw = prefix.canonical_bytes()
    prefix_artifact = ArtifactIdentity(root.root_id + ".prefix-result", "current-prefix", prefix.SCHEMA, sha256(prefix_raw).hexdigest(), "application/json", len(prefix_raw))
    checkpoints = {view.refinement: view.checkpoint for view in views if view.disposition == "COMPLETE"}
    payload, deliveries, mode_sha, completed = execute_preparation_panel(config, root, checkpoints, prefix_sha256=prefix.fingerprint(), progress=progress)
    result = TangentNativeResult(root.root_id + ".native-result", config.identity, root, prefix_artifact,
                                 mode_sha, deliveries, sha256(payload).hexdigest(), completed)
    projections = tuple(project_preparation_arrays(result, payload, refinement, source_identity=config.identity) for refinement in (1, 2))
    arrays = {name: np.stack([projection[0][name] for projection in projections]) for name in PROJECTION_SHAPES}
    arrays["prefix_complete"] = np.asarray([view.disposition == "COMPLETE" for view in views], dtype=np.bool_)
    arrays["mechanism_complete"] = np.asarray([not projection[1] for projection in projections], dtype=np.bool_)
    arrays["native_complete"] = np.asarray(all(delivery.disposition == "COMPLETE" for delivery in deliveries), dtype=np.bool_)
    arrays["prefix_completed_updates"] = np.asarray([view.delivery.completed_intervals for view in views], dtype=np.int64)
    return prefix, prefix_stream.getvalue(), result, payload, arrays


def tangent_analysis(config: TangentPreparationConfig, arrays):
    """Full 128-root accounting and 6,400 nested mechanism comparisons.

    Paired native-view agreement retains all 128 roots. The mechanism subset
    is pre-outcome ranked: 16 roots per context. Coupled and scalar tangent
    comparisons remain diagnostics, never use-time forecasts.
    """
    for name, shape in PROJECTION_SHAPES.items():
        if name not in arrays or arrays[name].shape != (128, 2, *shape) or arrays[name].dtype != np.float64:
            raise ValueError("Tangent analysis changes the complete native projection axes")
    for name, shape in (("prefix_complete", (128, 2)), ("mechanism_complete", (128, 2)), ("native_complete", (128,))):
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype != np.bool_:
            raise ValueError("Tangent analysis lacks complete native disposition census")
    selected = np.asarray([root.numerical_semantics for root in config.roots], dtype=np.bool_)
    complete = arrays["native_complete"] & arrays["prefix_complete"].all(axis=1)
    mechanism_complete = complete & arrays["mechanism_complete"].all(axis=1)
    for name in ("amplitude_chi", "amplitude_even", "amplitude_position_error", "amplitude_momentum_error", "coupled_tangent", "scalar_tangent", "secant_receiver", "secant_position_error", "secant_momentum_error", "secant_position_norm", "secant_momentum_norm"):
        mechanism_complete &= np.isfinite(arrays[name]).reshape(128, -1).all(axis=1)
    response = arrays["response"].transpose(0, 2, 1, 3, 4, 5)
    absolute, odd = native_view_errors(response)
    numerical_complete = np.isfinite(absolute).reshape(128, -1).all(axis=1) & np.isfinite(odd).reshape(128, -1).all(axis=1)
    absolute_pass = np.max(absolute.reshape(128, -1), axis=1) <= 1 / 128
    odd_pass = np.max(odd.reshape(128, -1), axis=1) <= 1 / 256
    native_view_pass = complete & numerical_complete & absolute_pass & odd_pass
    secant_position = arrays["secant_position_error"] / (1e-8 + 1e-5 * arrays["secant_position_norm"])
    secant_momentum = arrays["secant_momentum_error"] / (1e-8 + 1e-5 * arrays["secant_momentum_norm"])
    secant_pass = (np.max(secant_position.reshape(128, -1), axis=1) <= 1) & (np.max(secant_momentum.reshape(128, -1), axis=1) <= 1)
    tangent = arrays["coupled_tangent"][:, :, :, None]
    scalar = arrays["scalar_tangent"][:, :, :, None]
    coupled_error = arrays["amplitude_chi"] - tangent
    scalar_error = arrays["amplitude_chi"] - scalar
    full_pass = selected & mechanism_complete & native_view_pass & secant_pass
    out = {"numerical_subset": selected, "native_complete": complete, "mechanism_complete": mechanism_complete,
           "absolute_view_error": absolute, "odd_view_error": odd, "absolute_pass": absolute_pass,
           "odd_pass": odd_pass, "native_view_pass": native_view_pass, "secant_position_ratio": secant_position,
           "secant_momentum_ratio": secant_momentum, "secant_pass": secant_pass,
           "numerical_pass": full_pass, "coupled_amplitude_error": coupled_error,
           "scalar_amplitude_error": scalar_error,
           "amplitude_even": arrays["amplitude_even"],
           "scalar_coupled_receiver_difference": arrays["scalar_tangent"] - arrays["coupled_tangent"],
           "finite_action_secant_tangent_difference": arrays["secant_receiver"] - arrays["coupled_tangent"],
           "development_role": np.asarray([("fit", "interval", "screen").index(root.development_role) for root in config.roots], dtype=np.int64)}
    metrics = {"independent_source_roots": 128, "independent_view_roots": 128, "independent_mechanism_roots": 32,
               "nested_mechanism_comparisons": 6400, "complete_source_roots": int(complete.sum()),
               "complete_mechanism_roots": int(np.sum(selected & mechanism_complete)),
               "native_view_passing_roots": int(native_view_pass.sum()),
               "numerical_passing_roots": int(full_pass.sum()),
               "absolute_view_failing_roots": int(np.sum(~complete | ~numerical_complete | ~absolute_pass)),
               "odd_view_failing_roots": int(np.sum(~complete | ~numerical_complete | ~odd_pass)),
               "secant_failing_roots": int(np.sum(selected & (~mechanism_complete | ~secant_pass)))}
    cohort = np.asarray([config.summary_context == "all" or root.context == config.summary_context for root in config.roots])
    usable = selected & mechanism_complete & cohort
    metrics["coupled_tangent_mse"] = float(np.mean(coupled_error[usable] ** 2)) if usable.any() else None
    metrics["scalar_tangent_mse"] = float(np.mean(scalar_error[usable] ** 2)) if usable.any() else None
    metrics["scalar_minus_coupled_mse"] = None if not usable.any() else metrics["scalar_tangent_mse"] - metrics["coupled_tangent_mse"]
    incomplete = tuple(root.root_id for index, root in enumerate(config.roots) if not complete[index] or not numerical_complete[index] or (selected[index] and not mechanism_complete[index]))
    return out, metrics, incomplete
