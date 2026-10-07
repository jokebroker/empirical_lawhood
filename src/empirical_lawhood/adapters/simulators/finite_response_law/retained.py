"""Authenticate the native semantics of an exposed prepared-response parent continuation."""

from hashlib import sha256

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedNativeSpec, prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import decode_prepared_native_pair
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedNativePhaseData
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.source_qualification import SourceQualificationRetainedPredecessor
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.source_qualification import validate_retained_qualification_receipt


def authenticate_retained_handoff(
    *,
    declaration: SourceQualificationRetainedPredecessor,
    source: PreparedNativeSpec,
    receipt: CanonicalTaskReceipt,
    record: PreparedNativeTaskResult,
    payload: bytes,
) -> tuple[PreparedNativePhaseData, PreparedNativePhaseData]:
    """Decode old custody, without historical execution or new native updates.

    The caller authenticates receipt, record and payload through the declared
    external ports. This repeats their content and source/clock joins before a
    current-source continuation may restore either checkpoint.
    """
    validate_retained_qualification_receipt(declaration, receipt)
    root = record.invocation.root
    source_identity = ObjectIdentity.from_record(source.spec_id, source)
    if (
        source.stage != 'qualification'
        or source.fingerprint() != record.invocation.source_spec.object_fingerprint
        or source.member != prepared_native_member()
        or source.numerical_views
        != tuple(prepared_numerical_view(r) for r in (1, 2))
        or root.stage != 'qualification'
        or root.context != "prepared"
        or root.index not in range(16)
        or root not in source.roots
        or record.invocation.phase != "parent"
        or record.invocation.parent not in PARENTS
        or record.invocation.source_spec != source_identity
        or record.invocation.task_id != declaration.segment_id
        or declaration.physical_independent_unit_id != root.physical_unit_id
        or declaration.end != root.handoff
        or declaration.native_clock_id != "finite-response-law.reference-clock"
        or declaration.view_ids
        != tuple(f"{root.root_id}.flh-project.r{r}" for r in (1, 2))
        or record.unentered_reason is not None
        or record.native_pair is None
        or record.common_start is None
        or record.common_start.frame is None
        or record.common_start.root != root
        or any(
            c.source_spec != source_identity for c in record.common_start.checkpoints
        )
    ):
        raise ValueError(
            "Finite response-law retained handoff changes its frozen prepared-response source/root/parent/clock/view"
        )
    records = tuple(
        a for a in declaration.artifacts if a.payload_schema == record.SCHEMA
    )
    native = tuple(
        a for a in declaration.artifacts if a.sha256 == record.native_pair.data_sha256
    )
    if (
        len(declaration.artifacts) != 2
        or len(records) != 1
        or len(native) != 1
        or sha256(record.canonical_bytes()).hexdigest() != records[0].sha256
        or len(record.canonical_bytes()) != records[0].size_bytes
        or sha256(payload).hexdigest() != native[0].sha256
        or len(payload) != native[0].size_bytes
    ):
        raise ValueError(
            "Finite response-law retained native record or pair differs from its exact receipt"
        )
    pair = decode_prepared_native_pair(record.native_pair, payload)
    for refinement, value in enumerate(pair, start=1):
        checkpoint = value.checkpoint
        if (
            value.delivery.disposition != "COMPLETE"
            or checkpoint is None
            or checkpoint.native.step_index != 4368 * refinement
            or checkpoint.passive.refinement != refinement
            or checkpoint.source_spec != source_identity
            or value.delivery.common_start
            != ObjectIdentity.from_record(
                record.common_start.common_start_id, record.common_start
            )
            or value.delivery.incoming_checkpoint
            != ObjectIdentity.from_record(
                record.common_start.checkpoints[refinement - 1].checkpoint_id,
                record.common_start.checkpoints[refinement - 1],
            )
            or 4336 not in checkpoint.history_ticks
        ):
            raise ValueError(
                "Finite response-law retained parent checkpoint lost native continuation or causal history"
            )
    return pair
