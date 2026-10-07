"""Current RC result/prerequisite inspection from exact committed task receipts."""

from dataclasses import dataclass

from empirical_lawhood.adapters.simulator_morphism_challenges.retained_results import RCChallengeRetainedResult, RCChallengeResultAuthorityContext
from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeRecurrenceResult
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore, decode_artifact_manifest
from empirical_lawhood.runtime.artifacts import ArtifactManifest, ArtifactWriter, CanonicalTaskReceipt


@dataclass(frozen=True, slots=True)
class RCChallengeResultView:
    retained: RCChallengeRetainedResult
    record: CanonicalRecord
    summary: dict[str, object]


def read_rc_challenge_result(payload: bytes, *, writer: ArtifactWriter) -> RCChallengeResultView:
    """Read complete outcomes/stops without rerunning an observer, solver or inference."""
    retained = decode_canonical_bytes(payload, RCChallengeRetainedResult, maximum_bytes=16 * 1024**2)
    authenticate_rc_challenge_result_authority(retained, writer=writer)
    retained.authenticate(writer)
    record = retained.decode()
    summary = {"result_id": retained.result_id, "payload_schema": record.SCHEMA,
               "payload_sha256": record.fingerprint(), "run_id": retained.task_receipt.run_id,
               "task_id": retained.task_receipt.task_id,
               "operational_status": retained.task_receipt.operational_status.value,
               "implementation_commit": retained.task_receipt.implementation_commit,
               "reason_codes": getattr(record, "reason_codes", ()),
               "scientific_execution_performed_by_reader": False}
    if isinstance(record, SimulatorMorphismChallengeRecurrenceResult):
        summary.update({"requested_independent_units": sum(cell.requested_count for cell in record.cells if cell.scale_cells == 64 and cell.endpoint_id == "dynamical-closure"),
                        "recurrence_cells": len(record.cells),
                        "evidence_ceiling": record.evidence_ceiling.value,
                        "dynamical_existence_recurs": record.dynamical_existence_recurs,
                        "decision_existence_recurs": record.decision_existence_recurs,
                        "dynamical_majority_opposition_recurs": record.dynamical_majority_opposition_recurs,
                        "decision_majority_opposition_recurs": record.decision_majority_opposition_recurs,
                        "physical_claim_allowed": record.physical_claim_allowed,
                        "controller_claim_allowed": record.controller_claim_allowed,
                        "cells": tuple(cell.to_document()["value"] for cell in record.cells)})
    else:
        for name in ("passed", "issue_evaluation", "phase", "scientific_status", "terminal"):
            if hasattr(record, name):
                value = getattr(record, name)
                summary[name] = getattr(value, "value", value)
    return RCChallengeResultView(retained, record, summary)


def bind_rc_challenge_retained_result(*, result_id: str, record_payload: bytes,
                                     output_manifest: ArtifactManifest, task_receipt: CanonicalTaskReceipt,
                                     receipt_manifest: ArtifactManifest, writer: ArtifactWriter,
                                     authority_context: RCChallengeResultAuthorityContext | None = None) -> RCChallengeRetainedResult:
    """Export a receipt-bound selected result for inspection or a dependent phase."""
    result = RCChallengeRetainedResult(result_id, record_payload.decode("utf-8"), output_manifest,
                                       task_receipt, receipt_manifest, authority_context)
    authenticate_rc_challenge_result_authority(result, writer=writer)
    result.authenticate(writer)
    result.decode()
    return result


def _authenticate_result_access(*, context, receipt, logical, writer) -> None:
    if logical.outcome_access not in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL, OutcomeAccess.EVALUATION_REVEALED):
        from empirical_lawhood.api.current_result_custody import authenticate_unprotected_current_result_if_issued
        authenticate_unprotected_current_result_if_issued(writer=writer, receipt=receipt,
            logical=logical, capability_prefixes=('simulator-morphism-challenges.',))
        return
    if context is None or not isinstance(writer, ExternalArtifactPlane):
        raise PermissionError("RC_PROTECTED_RESULT_ACTUAL_CURRENT_AUTHORITY_REQUIRED")
    from empirical_lawhood.api.current_result_custody import _authenticate_protected_current_result
    _authenticate_protected_current_result(issued_study=context.issued_study,
        execution_authority=context.execution_authority, reveal_authority=context.reveal_authority,
        receipt=receipt, logical=logical, writer=writer, family='rc')


def authenticate_rc_challenge_result_authority(retained: RCChallengeRetainedResult, *, writer: ArtifactWriter) -> None:
    """Custody is separate from a current issued-study outcome access grant."""
    _authenticate_result_access(context=retained.authority_context, receipt=retained.task_receipt,
        logical=retained.output_manifest.logical, writer=writer)


def bind_rc_challenge_selected_output(*, result_id: str, run_id: str, task_id: str,
                                      receipt_id: str, output_artifact_id: str,
                                      writer: ExternalArtifactPlane,
                                      authority_context: RCChallengeResultAuthorityContext | None = None) -> RCChallengeRetainedResult:
    """Bind exactly one output from the actual stored receipt and full publication.

    The scheduler receipt lookup never selects a first/latest attempt. Protected
    output authority is verified before its bytes are read.
    """
    receipt = ExternalTaskReceiptStore(writer).read_by_receipt_id(run_id, task_id, receipt_id)
    if receipt is None:
        raise ValueError("RC_RESULT_EXACT_STORED_RECEIPT_REQUIRED")
    matches = tuple((logical, physical) for logical, physical in zip(receipt.output_logical_artifacts,
        receipt.output_materializations, strict=True) if logical.logical_artifact_id == output_artifact_id)
    if len(matches) != 1:
        raise ValueError("RC_RESULT_SELECTED_OUTPUT_ABSENT_FROM_EXACT_RECEIPT")
    logical, physical = matches[0]
    _authenticate_result_access(context=authority_context, receipt=receipt, logical=logical, writer=writer)
    manifests = tuple(decode_artifact_manifest(read_bounded_bytes(
        writer.root.resolve(materialization.relative_path + ".manifest.json", for_write=False), maximum_bytes=16 * 1024**2))
        for materialization in receipt.output_materializations)
    if tuple(value.materialization for value in manifests) != receipt.output_materializations or tuple(value.logical for value in manifests) != receipt.output_logical_artifacts:
        raise ValueError("RC_RESULT_FULL_RECEIPT_PUBLICATION_MISMATCH")
    writer.verify_manifests(manifests)
    output_manifest = next(value for value in manifests if value.materialization == physical)
    receipt_path = f"runs/{run_id}/receipts/{task_id}/{receipt.attempt_id}.json"
    receipt_manifest = decode_artifact_manifest(read_bounded_bytes(
        writer.root.resolve(receipt_path + ".manifest.json", for_write=False), maximum_bytes=16 * 1024**2))
    raw = read_bounded_bytes(writer.root.resolve(physical.relative_path, for_write=False), maximum_bytes=8 * 1024**2)
    return bind_rc_challenge_retained_result(result_id=result_id, record_payload=raw, output_manifest=output_manifest,
        task_receipt=receipt, receipt_manifest=receipt_manifest, writer=writer, authority_context=authority_context)
