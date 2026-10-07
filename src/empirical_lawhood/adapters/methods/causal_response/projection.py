"""Exact root projection of committed forecasts and paired native futures."""

from decimal import Decimal
import json

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.methods.prepared_response.projection import project_prepared_future
from empirical_lawhood.adapters.methods.prepared_response.qualification_projection import _force_error
from empirical_lawhood.adapters.simulators.causal_response.contracts import native_invocations
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from .comparison import compare_context
from .models import Array, primary_contrast
from .records import CausalResponseCommittedPrediction, CausalResponseEvaluationConfig, CausalResponseEvaluation, CausalResponseProjectionConfig, CausalResponseViewObservation


def scalars(values: Array) -> tuple[Decimal | None, ...]:
    return tuple(Decimal(str(float(v))) if np.isfinite(v) else None for v in values.ravel())


def project_root(
    config: CausalResponseProjectionConfig,
    root: PreparedRoot,
    refinement: int,
    inputs: tuple[tuple[PreparedNativeTaskResult, bytes, CausalResponseCommittedPrediction], ...],
) -> CausalResponseViewObservation:
    from empirical_lawhood.adapters.simulators.causal_response.provider import validate_prediction

    source = config.native_spec
    if root not in source.roots or refinement not in (1, 2):
        raise ValueError("causal response prediction projection outside assigned root/view")
    tasks = {t.task_id: t for t in native_invocations(source) if t.root == root}
    if len(inputs) != len(tasks) or {r.invocation.task_id for r, _, _ in inputs} != set(tasks):
        raise ValueError("causal response prediction projection changes complete native input census")
    identity = ObjectIdentity.from_record
    results = {r.invocation.task_id: r for r, _, _ in inputs}
    predictions = {r.invocation.task_id: p for r, _, p in inputs}
    common = results[f"{root.root_id}.prefix.native"].common_start
    data = {}
    for result, payload, prediction in inputs:
        task = tasks[result.invocation.task_id]
        if (
            result.invocation != task
            or result.common_start != common
            or result.predecessors
            != tuple(identity(results[d].result_id, results[d]) for d in task.dependency_task_ids)
        ):
            raise ValueError(
                "causal response prediction native result changes common-start/predecessor/invocation lineage"
            )
        validate_prediction(source, result, prediction)
        data[task.task_id] = decode_prepared_task_native(result, payload)
    observed = np.full((5, 2, 5, 2, 2), np.nan)
    predicted = np.full((5, 5, 2, 2), np.nan)
    parent_work = np.full(5, np.nan)
    force_work = np.full((5, 2, 4, 2), np.nan)
    reasons = set()
    innovations: dict[str, set[str]] = {p: set() for p in source.purposes}
    for pi, parent in enumerate(PARENTS):
        parent_id = f"{root.root_id}.{parent}.parent.native"
        pair = data[parent_id]
        prediction = predictions[parent_id]
        predicted[pi] = np.asarray(prediction.gain, dtype=float).reshape(2, 5, 2, 2)[refinement - 1]
        if prediction.disposition != "COMMITTED_AT_HANDOFF":
            reasons.add("PREDICTION_UNAVAILABLE")
        if (
            pair is None
            or pair[refinement - 1].checkpoint is None
            or common is None
            or common.frame is None
        ):
            reasons.add("HANDOFF_OR_FRAME_UNAVAILABLE")
            continue
        native_parent = pair[refinement - 1]
        parent_work[pi] = float(native_parent.delivery.parent_absolute_density_work)
        handoff = bind_prepared_native_handoff(native_parent)
        for ki, purpose in enumerate(source.purposes):
            values = np.full((4, 5, 2), np.nan)
            for wi, word in enumerate(source.words):
                task_id = f"{root.root_id}.{parent}.{purpose}.{word.word_id}.native"
                future_pair = data[task_id]
                if future_pair is None or not results[task_id].native_complete:
                    reasons.add("FUTURE_DELIVERY_UNAVAILABLE")
                    continue
                future = future_pair[refinement - 1]
                error = _force_error(future)
                if error is None or error > Decimal("0.000000000001"):
                    reasons.add("FORCE_DELIVERY_MISMATCH")
                    continue
                measured = project_prepared_future(common, handoff, future, matched_hold=None)
                values[wi] = measured.outputs[:, :2]
                force_work[pi, ki, wi] = measured.outputs[-1, 5:]
                innovations[purpose].add(future.delivery.innovation_sha256)
            observed[pi, ki] = np.stack(
                ((values[1] - values[0]) / 2, (values[3] - values[2]) / 2), axis=-1
            )
    if any(len(v) > 1 for v in innovations.values()) or (
        all(innovations.values())
        and innovations[source.purposes[0]] == innovations[source.purposes[1]]
    ):
        raise ValueError("causal response prediction innovation pairing/independent-replicate contract differs")
    return CausalResponseViewObservation(
        identity(config.config_id, config),
        root,
        refinement,
        tuple(identity(r.result_id, r) for _, r in sorted(results.items())),
        tuple(
            identity(
                predictions[f"{root.root_id}.{p}.parent.native"].prediction_id,
                predictions[f"{root.root_id}.{p}.parent.native"],
            )
            for p in PARENTS
        ),
        tuple(sorted(reasons)),
        scalars(observed),
        scalars(predicted),
        scalars(parent_work),
        scalars(force_work),
    )


def evaluate(
    config: CausalResponseEvaluationConfig, reports: tuple[CausalResponseViewObservation, ...]
) -> CausalResponseEvaluation:
    identity = ObjectIdentity.from_record
    source = config.projection.native_spec
    expected = {(r, v) for r in source.roots for v in (1, 2)}
    if (
        len(reports) != 128
        or {(r.root, r.refinement) for r in reports} != expected
        or any(
            r.projection_config != identity(config.projection.config_id, config.projection)
            for r in reports
        )
    ):
        raise ValueError("causal response prediction evaluator changes its complete assigned projection census")
    by_slot = {(r.root.context, r.root.index, r.refinement): r for r in reports}
    results = {}
    for context in source.model_bank.contexts:
        name = context.recipe.context
        observed = np.empty((32, 2, 2, 32))
        predicted = np.empty((32, 2, 32))
        for i in range(32):
            for vi in range(2):
                report = by_slot[name, i, vi + 1]
                gain = np.asarray(report.observed_gain, dtype=float).reshape(5, 2, 5, 2, 2)
                observed[i, vi] = primary_contrast(np.swapaxes(gain, 0, 1)).reshape(2, 32)
                predicted[i, vi] = primary_contrast(
                    np.asarray(report.predicted_gain, dtype=float).reshape(5, 5, 2, 2)
                ).ravel()
        baseline = np.broadcast_to(
            primary_contrast(context.baseline_gain).ravel(), (32, 2, 32)
        ).copy()
        results[name] = compare_context(observed, predicted, baseline)
    return CausalResponseEvaluation(
        identity(config.config_id, config),
        tuple(identity(r.report_id, r) for r in sorted(reports, key=lambda r: r.report_id)),
        json.dumps(results, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
