"""Compact append-only catalog projection for one automated exploration wave."""

from __future__ import annotations

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.planning.discovery import ExplorationLifecycleLedger
from empirical_lawhood.planning.exploration import ExplorationPlan
from empirical_lawhood.runtime.artifacts import ArtifactWriteResult, ExternalRootContract
from empirical_lawhood.runtime.catalog import (
    ArtifactLocatorRecord,
    CatalogSnapshot,
    CatalogVerificationStatus,
    ExplorationAttemptRecord,
    LogicalArtifactRecord,
    ReceiptLocatorRecord,
    ScientificObjectKind,
    ScientificObjectRecord,
    StorageRootRecord,
)


def _status(value: str) -> str:
    return value.lower().replace("_", "-")


def exploration_catalog_snapshot(
    *,
    root_contract: ExternalRootContract,
    plan: ExplorationPlan,
    ledger: ExplorationLifecycleLedger,
    ledger_write: ArtifactWriteResult,
    receipt_sha256: str,
    receipt_relative_path: str,
    receipt_schema: str,
) -> CatalogSnapshot:
    """Project full selection/execution lifecycle metadata without row-level data."""

    validate_sha256(receipt_sha256, field_name="receipt_sha256")
    if ledger.plan is None or ledger.plan.object_fingerprint != plan.fingerprint():
        raise ValueError("catalog exploration ledger binds another plan")
    if ledger_write.logical.visibility_ceiling != ledger.visibility_ceiling:
        raise ValueError("catalog exploration artifact visibility differs from its ledger")
    materialization_id = ledger_write.materialization.materialization_id
    receipt_id = f"receipt.{plan.plan_id}"
    objects: list[ScientificObjectRecord] = []

    def add_object(
        kind: ScientificObjectKind,
        record: CanonicalRecord,
        categorical_status: str,
    ) -> str:
        scientific_object_id = f"object.{_status(kind.value)}.{record.fingerprint()[:24]}"
        objects.append(
            ScientificObjectRecord(
                scientific_object_id=scientific_object_id,
                kind=kind,
                object_schema=record.SCHEMA,
                object_sha256=record.fingerprint(),
                visibility_ceiling=ledger.visibility_ceiling,
                categorical_status=categorical_status,
                artifact_materialization_id=materialization_id,
                receipt_id=receipt_id,
            )
        )
        return scientific_object_id

    selection_by_id = {selection.proposal_id: selection for selection in ledger.selections}
    analysis_object_ids: dict[str, str] = {}
    add_object(
        ScientificObjectKind.EXPLORATION_PLAN,
        plan,
        "frozen-outcome-visible",
    )
    for proposal in plan.proposals:
        selection = selection_by_id[proposal.proposal_id]
        disposition = _status(selection.disposition.value)
        analysis_object_ids[proposal.analysis.analysis_id] = add_object(
            ScientificObjectKind.ANALYSIS_SPEC,
            proposal.analysis,
            disposition,
        )
        add_object(
            ScientificObjectKind.ANALYSIS_PROPOSAL,
            proposal,
            disposition,
        )
    for finding in ledger.findings:
        add_object(
            ScientificObjectKind.EXPLORATORY_FINDING,
            finding,
            _status(finding.status.value),
        )
    for synthesis in ledger.hypothesis_syntheses:
        hypothesis_set = synthesis.hypothesis_set
        add_object(
            ScientificObjectKind.HYPOTHESIS_SET,
            hypothesis_set,
            "competing-hypotheses",
        )
        for hypothesis in hypothesis_set.hypotheses:
            add_object(
                ScientificObjectKind.HYPOTHESIS,
                hypothesis,
                _status(hypothesis.disposition.value),
            )

    selection_records = tuple(
        ExplorationAttemptRecord(
            attempt_id=f"selection.{selection.proposal_id}",
            analysis_spec_object_id=analysis_object_ids[
                next(
                    proposal.analysis.analysis_id
                    for proposal in plan.proposals
                    if proposal.proposal_id == selection.proposal_id
                )
            ],
            disposition=_status(selection.disposition.value),
            reason_codes=selection.reason_codes,
            artifact_materialization_ids=(materialization_id,),
        )
        for selection in ledger.selections
    )
    execution_records = tuple(
        ExplorationAttemptRecord(
            attempt_id=attempt.attempt_id,
            analysis_spec_object_id=analysis_object_ids[attempt.analysis_id],
            disposition=_status(attempt.status.value),
            reason_codes=attempt.reason_codes,
            artifact_materialization_ids=(materialization_id,),
        )
        for attempt in ledger.attempts
    )
    materialization = ledger_write.materialization
    return CatalogSnapshot(
        storage_roots=(
            StorageRootRecord(
                storage_root_id=root_contract.storage_root_id,
                logical_name=root_contract.logical_name,
                canonical_path=root_contract.canonical_path,
                mount_contract_schema=root_contract.mount_contract_schema,
                verification_status=CatalogVerificationStatus.VERIFIED,
            ),
        ),
        logical_artifacts=(
            LogicalArtifactRecord(
                logical_artifact_id=ledger_write.logical.logical_artifact_id,
                content_sha256=ledger_write.logical.content_sha256,
                payload_schema=ledger_write.logical.payload_schema,
                profile=ledger_write.logical.profile,
                media_type=ledger_write.logical.media_type,
                visibility_ceiling=ledger_write.logical.visibility_ceiling,
            ),
        ),
        artifact_locators=(
            ArtifactLocatorRecord(
                materialization_id=materialization.materialization_id,
                logical_artifact_id=materialization.logical_artifact_id,
                storage_root_id=materialization.storage_root_id,
                relative_path=materialization.relative_path,
                physical_sha256=materialization.physical_sha256,
                size_bytes=materialization.size_bytes,
                compression=materialization.compression,
                partition_selector=materialization.partition_selector,
                verification_status=CatalogVerificationStatus.VERIFIED,
            ),
        ),
        receipts=(
            ReceiptLocatorRecord(
                receipt_id=receipt_id,
                run_id=plan.plan_id,
                storage_root_id=root_contract.storage_root_id,
                relative_path=receipt_relative_path,
                sha256=receipt_sha256,
                receipt_schema=receipt_schema,
                verification_status=CatalogVerificationStatus.VERIFIED,
            ),
        ),
        scientific_objects=tuple(sorted(objects, key=lambda record: record.scientific_object_id)),
        knowledge_edges=(),
        metric_definitions=(),
        metric_observations=(),
        exploration_attempts=tuple(
            sorted((*selection_records, *execution_records), key=lambda record: record.attempt_id)
        ),
    )
