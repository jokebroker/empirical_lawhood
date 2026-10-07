# SPDX-License-Identifier: MPL-2.0
"""Original causal-response development labels from authenticated retained source qualification.

Callers authenticate the original scalar cache, complete native root inventory,
receipt/result/payload hashes and analysis permission before entry. The original
negative source qualification may supply exposed development operands, never prospective support.
This library opens no files, fits no bank and performs no native work or issue.
"""

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, PARENTS, prepared_words
from empirical_lawhood.adapters.simulators.prepared_response.source import bind_prepared_native_handoff
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .exposed_projection import exposed_source_qualification_transition_labels
from .models import CausalResponsePanel


def assemble_retained_development(
    cached: dict[str, Any],
    native_inputs: Mapping[str, tuple[bytes, bytes]],
) -> tuple[tuple[CausalResponsePanel, ...], dict[str, Any]]:
    """Reconstruct both original 16-root contexts and all transition labels.

    Separately verified original-source exports must bind original hashes and
    interpretation to these current task identities and target custody.
    The supplied 200-scalar cache retains handoff history/sketch cutoffs. All
    retained native result/payload pairs must be authenticated before entry;
    this check does not grant reveal, publication or new experimental authority.
    """
    required = {
        task
        for context in ("assembling", "prepared")
        for index in range(16)
        for root in (f"{CAMPAIGN}.qualification.{context}.r{index:03d}",)
        for task in (
            root + ".prefix.native",
            *(f"{root}.{parent}.parent.native" for parent in PARENTS),
            *(f"{root}.{parent}.common-response.{word.word_id}.native"
              for parent in PARENTS for word in prepared_words(Decimal(16))),
        )
    }
    if not required.issubset(native_inputs):
        raise ValueError("CAUSAL_RESPONSE_RETAINED_VERIFIED_TARGET_EXPORT_REQUIRED")
    history = np.swapaxes(cached["features"][:, :, 1:, :192], 1, 2).reshape(
        32, 5, 2, 16, 12
    )
    sketch = np.swapaxes(cached["features"][:, :, 1:, 192:], 1, 2)
    observed = np.swapaxes(cached["original_channels"], 1, 2)
    transitions = np.empty((32, 5, 2, 9, 21, 12))
    roots = []
    for ri in range(32):
        context = ("assembling", "prepared")[ri // 16]
        root_id = f"{CAMPAIGN}.qualification.{context}.r{ri % 16:03d}"

        def native(task_id: str) -> tuple[PreparedNativeTaskResult, Any]:
            record_bytes, payload = native_inputs[task_id]
            record = decode_canonical_bytes(
                record_bytes, PreparedNativeTaskResult, maximum_bytes=16 * 1024**2
            )
            if record.invocation.task_id != task_id:
                raise ValueError(
                    "Retained causal record changes its declared native task"
                )
            pair = decode_prepared_task_native(record, payload)
            if pair is None or not record.native_complete:
                raise ValueError("causal response prediction original source qualification source is not complete")
            return (record, pair)

        prefix, _ = native(root_id + ".prefix.native")
        common = prefix.common_start
        assert common is not None
        roots.append(common.root)
        for pi, parent in enumerate(PARENTS):
            parent_record, parent_pair = native(f"{root_id}.{parent}.parent.native")
            if parent_record.common_start != common:
                raise ValueError("causal response prediction parent lost original common-start lineage")
            for wi, word in enumerate(prepared_words(Decimal(16))):
                future_record, future_pair = native(
                    f"{root_id}.{parent}.common-response.{word.word_id}.native"
                )
                if future_record.common_start != common:
                    raise ValueError("causal response prediction future lost original common-start lineage")
                for vi in range(2):
                    transitions[ri, pi, vi, wi] = exposed_source_qualification_transition_labels(
                        common,
                        bind_prepared_native_handoff(parent_pair[vi]),
                        future_pair[vi],
                    )
    panels = []
    for start in (0, 16):
        panels.append(
            CausalResponsePanel(
                tuple(roots[start : start + 16]),
                history[start : start + 16],
                sketch[start : start + 16],
                observed[start : start + 16],
                transitions[start : start + 16],
            )
        )
    return (
        tuple(panels),
        {
            "history": history,
            "sketch": sketch,
            "observed": observed,
            "transitions": transitions,
        },
    )
