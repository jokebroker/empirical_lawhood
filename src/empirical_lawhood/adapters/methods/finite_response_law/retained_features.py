# SPDX-License-Identifier: MPL-2.0
"""Original retained causal feature assembly from authenticated native operands.

The caller authenticates the 16 prepared-response/32 information-prediction root inventory, successful receipts,
logical/materialized result and payload hashes, source freeze and analysis
permission before entry. The library keeps native prefix/parent lineage and
view/cutoff checks and delegates measurements to the existing compact instrument.
It opens no files, integrates no source, fits no model and grants no authority.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT, compact_interface
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .fitting import Features
from .science import PARENTS


@dataclass(frozen=True, slots=True)
class RetainedFeatureRootOperands:
    """One authenticated retained prefix and its five ordered parent payloads."""

    root_id: str
    prefix: PreparedNativeTaskResult
    parents: tuple[tuple[PreparedNativeTaskResult, bytes], ...]


def assemble_retained_features(
    roots: tuple[RetainedFeatureRootOperands, ...],
) -> tuple[Features, dict[str, Any]]:
    """Keep the original 48-root prefix/handoff chart and causal feature recipe.

    A separately verified export must bind original hashes and interpretation
    to current target identity and custody. Original grants never transfer.
    Root and receipt authentication is a caller prerequisite. Exact native task,
    checkpoint, source, numerical-view and clock joins remain checked here.
    """
    expected = tuple(
        f"{cohort}.prepared.r{r:03d}"
        for cohort, count in (("prepared-response", 16), ("information-response-prediction", 32))
        for r in range(count)
    )
    if tuple(v.root_id for v in roots) != expected or any(
        len(v.parents) != 5 for v in roots
    ):
        raise ValueError(
            "Retained features change the complete original root/parent census"
        )
    if any(
        operands.prefix.invocation.task_id
        != f"{CAMPAIGN}.{stage}.prepared.r{index:03d}.prefix.native"
        for operands, (stage, index) in zip(
            roots,
            ((stage, index)
             for stage, count in (("qualification", 16), ("prospective-evaluation", 32))
             for index in range(count)),
            strict=True,
        )
    ):
        raise ValueError("FINITE_RESPONSE_FEATURES_VERIFIED_TARGET_EXPORT_REQUIRED")
    x, z = np.empty((48, 24, 2)), np.empty((48, 5, 24, 2))
    lineage, ids = [], []
    for r, operands in enumerate(roots):
        prefix = operands.prefix
        stage, native_root_index = ("qualification", r) if r < 16 else ("prospective-evaluation", r - 16)
        native_root = f"{CAMPAIGN}.{stage}.prepared.r{native_root_index:03d}"
        if prefix.invocation.task_id != f"{native_root}.prefix.native":
            raise ValueError("Feature prefix changes the original native root census")
        ids.append(operands.root_id)
        cohort = operands.root_id.split(".", 1)[0]
        prefix_identity = ObjectIdentity.from_record(prefix.result_id, prefix)
        if (
            not prefix.native_complete
            or prefix.common_start is None
            or prefix.common_start.frame is None
        ):
            raise ValueError("Retained prefix is unavailable")
        common = prefix.common_start
        for p, parent in enumerate(PARENTS):
            record, payload = operands.parents[p]
            if (
                record.invocation.task_id
                != f"{prefix.invocation.root.root_id}.{parent}.parent.native"
            ):
                raise ValueError(
                    "Feature parent task differs from the supplied prefix/root/parent join"
                )
            if (
                record.predecessors != (prefix_identity,)
                or record.common_start != common
                or (not record.native_complete)
            ):
                raise ValueError("Feature parent/prefix lineage differs")
            pair = decode_prepared_task_native(record, payload)
            assert pair is not None and common.frame is not None
            for v, phase in enumerate(pair):
                assert phase.checkpoint is not None
                for kind, checkpoint, cutoff in (
                    ("prefix", common.checkpoints[v], 4096),
                    ("handoff", phase.checkpoint, 4368),
                ):
                    if (
                        checkpoint.passive.refinement != v + 1
                        or checkpoint.native.step_index != cutoff * (v + 1)
                        or checkpoint.root != record.invocation.root
                        or (checkpoint.source_spec != record.invocation.source_spec)
                    ):
                        raise ValueError(
                            "Feature checkpoint changes source/root/view/cutoff"
                        )
                    interface = compact_interface(
                        frame=common.frame,
                        ticks=checkpoint.history_ticks,
                        positions=_decode(
                            checkpoint.history_positions_base64, (31, 2, 3, 4, 4)
                        ),
                        momenta=_decode(
                            checkpoint.history_momenta_base64, (31, 2, 3, 4, 4)
                        ),
                        cutoff_tick=cutoff,
                    )
                    if kind == "prefix":
                        if p == 0:
                            x[r, :, v] = interface.values
                        else:
                            np.testing.assert_array_equal(x[r, :, v], interface.values)
                    else:
                        z[r, p, :, v] = interface.values
                    lineage.append(
                        {
                            "root": ids[-1],
                            "cohort": cohort,
                            "parent": parent,
                            "view": v,
                            "role": kind,
                            "cutoff_tick": cutoff,
                            "backward_tick": cutoff - 32,
                            "frame_cutoff_tick": 4096,
                            "checkpoint": json.loads(
                                ObjectIdentity.from_record(
                                    checkpoint.checkpoint_id, checkpoint
                                ).canonical_bytes()
                            ),
                            "source": json.loads(
                                checkpoint.source_spec.canonical_bytes()
                            ),
                            "native_result": json.loads(
                                ObjectIdentity.from_record(
                                    record.result_id, record
                                ).canonical_bytes()
                            ),
                            "native_payload": {
                                "sha256": sha256(payload).hexdigest(),
                                "bytes": len(payload),
                            },
                            "prefix_record": json.loads(
                                prefix_identity.canonical_bytes()
                            ),
                            "feature_values_sha256": sha256(
                                interface.values.tobytes()
                            ).hexdigest(),
                        }
                    )
    features = Features(x, z, tuple(ids))
    return (
        features,
        {
            "schema": "finite-response-law-causal-features",
            "instrument": json.loads(REFERENCE_INSTRUMENT.canonical_bytes()),
            "axes": {
                "x": ["root", "native_feature", "view"],
                "z": ["root", "parent", "native_feature", "view"],
            },
            "primary_fit_view": 0,
            "independent_roots": 48,
            "prefix_feature_rows": 96,
            "handoff_feature_rows": 480,
            "rows": lineage,
            "new_native_updates": 0,
            "model_fits": 0,
        },
    )
