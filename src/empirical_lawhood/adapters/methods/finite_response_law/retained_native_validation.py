# SPDX-License-Identifier: MPL-2.0
"""Independent retained native lineage and panel-array validation.

Authenticate the original prepared-response/information-prediction arrays, source inventory, successful receipts,
custody and analysis authority before entry. Keep the original all-root native
word/clock/common-start/innovation checks and sampled endpoint comparisons.
The function opens no files, integrates no source and creates no qualification.
"""

from collections.abc import Mapping
from hashlib import sha256
import json
from typing import Any

import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode as decode_complex
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .retained import FiniteResponseLawObservedPanel, import_retained
from .science import PARENTS


def unpack(value: Any) -> Any:
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"schema", "value", "version"}:
            return unpack(value["value"])
        if set(value) == {"decimal"}:
            return float(value["decimal"])
        return {k: unpack(v) for k, v in value.items()}
    return value


def receiver(modes: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("pabc,...abc->...p", modes.conj(), values))


def validate_retained_native_panel(
    source_qualification: dict[str, Any],
    source_qualification_response: dict[str, Any],
    information_prediction: dict[str, Any],
    native_inputs: Mapping[str, tuple[bytes, bytes]],
) -> tuple[FiniteResponseLawObservedPanel, dict[str, Any]]:
    """Check complete supplied native lineage before importing the retained panel.

    The caller authenticates each canonical result/payload and original array
    hash, exact source inventory, prepared-response custody and failure summaries, information-prediction complete
    status/64-root extraction, and separate analysis authority. Native inputs
    are keyed by current task ID after a separately verified export binds
    original hashes and interpretation to target identity and target custody.
    Labels alone are not those custody proofs; original grants do not transfer.
    Preserve the original 300MiB/file,100GiB cumulative and3600s authentication
    budget in the caller's active resource ledger. This pure readout does not
    grant a new allowance. The bounded decoder also checks native pair layout.
    """

    required = {
        task
        for count, stage in ((16, "qualification"), (32, "prospective-evaluation"))
        for index in range(count)
        for root in (f"{CAMPAIGN}.{stage}.prepared.r{index:03d}",)
        for parent in PARENTS
        for task in (
            f"{root}.{parent}.parent.native",
            *(f"{root}.{parent}.{purpose}.{CAMPAIGN}.word.{word}.native"
              for purpose in (("common-response",) if stage == "qualification"
                              else ("common-response", "prospective-task"))
              for word in (
                  *(("hold",) if stage == "qualification" else ()),
                  *(f"a{amplitude}.d{direction}.{sign}"
                    for amplitude in ((8, 16) if stage == "qualification" else (16,))
                    for direction in (0, 1) for sign in ("negative", "positive")),
              )),
        )
    }
    if not required.issubset(native_inputs):
        raise ValueError("FINITE_RESPONSE_RETAINED_VERIFIED_TARGET_EXPORT_REQUIRED")

    def read_result(task: str) -> tuple[bytes, bytes, dict[str, Any]]:
        raw, payload = native_inputs[task]
        record = decode_canonical_bytes(
            raw, PreparedNativeTaskResult, maximum_bytes=16 * 1024**2
        )
        if record.invocation.task_id != task or not record.native_complete:
            raise ValueError(
                "Retained native result changes its complete task identity"
            )
        return raw, payload, unpack(json.loads(raw))

    def native_views(raw: bytes, payload: bytes) -> tuple[dict[str, Any], ...]:
        record = decode_canonical_bytes(
            raw, PreparedNativeTaskResult, maximum_bytes=16 * 1024**2
        )
        pair = decode_prepared_task_native(record, payload)
        assert pair is not None
        return tuple({"ticks": v.ticks, "positions": v.positions} for v in pair)

    lineage = []
    seeds = set()
    native_checks = 0
    for cohort, n, stage in (("prepared-response", 16, "qualification"), ("information-response-prediction", 32, "prospective-evaluation")):
        for r in range(n):
            root = f"{CAMPAIGN}.{stage}.prepared.r{r:03d}"
            common_digest = None
            for p, parent in enumerate(PARENTS):
                parent_task = f"{root}.{parent}.parent.native"
                parent_record = read_result(parent_task)[2]
                common = parent_record["common_start"]
                digest = sha256(
                    json.dumps(common, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if common_digest is None:
                    common_digest = digest
                    seed = f"{root}.seed.{common['root']['seed_sha256']}"
                    if seed in seeds:
                        raise ValueError(
                            "Repeated physical independent unit across cohorts"
                        )
                    seeds.add(seed)
                if digest != common_digest or common["mode_disposition"] != "RESOLVED":
                    raise ValueError("Common frame/start changed within root")
                if parent_record["unentered_reason"] is not None:
                    raise ValueError("Unaccounted parent failure")
                parent_views = parent_record["native_pair"]["views"]
                for v, view in enumerate(parent_views):
                    actual = view["delivery"]["parent_absolute_density_work"]
                    expected = (
                        source_qualification["parent_work"][16 + r, v, p]
                        if cohort == "prepared-response"
                        else information_prediction["parent_work"][1, r, v, p]
                    )
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
                sample = {}
                words = [
                    (a, d, s)
                    for a in ((8, 16) if cohort == "prepared-response" else (16,))
                    for d in (0, 1)
                    for s in (-1, 1)
                ]
                if cohort == "prepared-response":
                    words.insert(0, (0, 0, 0))
                for f, purpose in enumerate(
                    ("common-response",) if cohort == "prepared-response" else ("common-response", "prospective-task")
                ):
                    innovations: dict[int, Any] = {}
                    for amplitude, direction, sign in words:
                        word = (
                            "hold"
                            if amplitude == 0
                            else f"a{amplitude}.d{direction}.{('negative' if sign < 0 else 'positive')}"
                        )
                        task = f"{root}.{parent}.{purpose}.{CAMPAIGN}.word.{word}.native"
                        raw, payload, rec = read_result(task)
                        rec_ref = {
                            "task_id": task,
                            "sha256": sha256(raw).hexdigest(),
                            "bytes": len(raw),
                        }
                        payload_ref = {
                            "task_id": task,
                            "sha256": sha256(payload).hexdigest(),
                            "bytes": len(payload),
                        }
                        pair = rec["native_pair"]
                        invocation = rec["invocation"]
                        if (
                            invocation["root"]["context"],
                            invocation["root"]["index"],
                            invocation["root"]["stage"],
                            invocation["parent"],
                            invocation["purpose"],
                        ) != ("prepared", r, stage, parent, purpose):
                            raise ValueError(
                                "Native invocation changes root/parent/future"
                            )
                        if invocation["word"] != {
                            "magnitude": float(amplitude),
                            "direction_index": direction,
                            "sign": sign,
                        }:
                            raise ValueError("Native action differs from array chart")
                        if (
                            rec["unentered_reason"] is not None
                            or rec["common_start"] != common
                            or pair["data_sha256"] != payload_ref["sha256"]
                            or (pair["purpose"] != purpose)
                        ):
                            raise ValueError(
                                "Invalid common-start/payload/future lineage"
                            )
                        for v, view in enumerate(pair["views"]):
                            delivery = view["delivery"]
                            if (
                                delivery["incoming_checkpoint"]
                                != parent_views[v]["checkpoint"]
                                or delivery["disposition"] != "COMPLETE"
                                or (not delivery["accepted"])
                            ):
                                raise ValueError(
                                    "Delivery/start is not the authenticated retained complete panel"
                                )
                            if (
                                delivery["start_tick"],
                                delivery["requested_end_tick"],
                                delivery["refinement"],
                            ) != (4368, 4688, v + 1):
                                raise ValueError("Native readout clocks/views differ")
                            innovation = (
                                delivery["innovation_sha256"],
                                delivery["innovation_streams"],
                            )
                            if v in innovations and innovations[v] != innovation:
                                raise ValueError("Signed mates/HOLD innovations differ")
                            innovations[v] = innovation
                        lineage.append(
                            {
                                "root": f"{cohort}.prepared.r{r:03d}",
                                "parent": parent,
                                "handoff_tick": 4368,
                                "readout_tick": 4560,
                                "future": purpose,
                                "word": word,
                                "native_result": rec_ref,
                                "native_payload": payload_ref,
                                "source_spec": invocation["source_spec"],
                                "common_start_sha256": common_digest,
                                "view_incoming_checkpoints": [
                                    x["delivery"]["incoming_checkpoint"]
                                    for x in pair["views"]
                                ],
                            }
                        )
                        if r == 0 and p == 0:
                            sample[f, amplitude, direction, sign] = native_views(
                                raw, payload
                            )
                    if r == 0 and p == 0:
                        modes = decode_complex(
                            common["frozen_ports_base64"], (2, 3, 4, 4)
                        )
                        for amplitude, direction, sign in words:
                            if amplitude == 0:
                                continue
                            branch = sample[f, amplitude, direction, sign]
                            for v, data in enumerate(branch):
                                index = int(np.flatnonzero(data["ticks"] == 4560)[0])
                                endpoint = receiver(
                                    modes,
                                    data["positions"][index, 0]
                                    - data["positions"][0, 0],
                                )
                                if cohort == "prepared-response":
                                    w = (
                                        1
                                        + (8 if amplitude == 16 else 0)
                                        + 2 * direction
                                        + (sign == 1)
                                    )
                                    expected = source_qualification["values"][16 + r, v, p, w, 2, :2]
                                else:
                                    w = 2 * direction + (sign == 1)
                                    expected = information_prediction["endpoints"][1, r, v, p, f, w, 2]
                                np.testing.assert_allclose(
                                    endpoint, expected, rtol=0, atol=2e-12
                                )
                                native_checks += 1
    panel = import_retained(source_qualification, source_qualification_response, information_prediction, lineage_authenticated=True)
    return (
        panel,
        {
            "root_counts": {"prepared-response": 16, "information-response-prediction": 32, "total": 48},
            "independent_root_ids": len(seeds),
            "native_endpoint_checks": native_checks,
            "source_rows": lineage,
            "new_native_updates": 0,
            "native_reexecution": False,
        },
    )
