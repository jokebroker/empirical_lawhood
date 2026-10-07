"""Held-out evaluator; the only FSM adapter permitted to open sealed receivers."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from empirical_lawhood.kernel.atlases import AtlasGap, AtlasGapKind
from empirical_lawhood.kernel.evidence import EvidenceRung, VisibilityCeiling
from empirical_lawhood.kernel.status import ReadinessStatus, ScientificStatus

from .contracts import FineSteeringMirrorAdversarialAudit, FineSteeringMirrorBlockResponse, FineSteeringMirrorDevelopmentModel, FineSteeringMirrorProtocolSpec, FineSteeringMirrorRungAssessment, FineSteeringMirrorTransportStatus, FineSteeringMirrorVerticalSliceResult
from .observation import load_safe_array, receiver_errors, reduce_block
from .system import fine_steering_mirror_system


def _assessment(
    rung: EvidenceRung,
    scientific: ScientificStatus,
    readiness: ReadinessStatus,
    reasons: tuple[str, ...],
) -> FineSteeringMirrorRungAssessment:
    rank = tuple(EvidenceRung).index(rung)
    return FineSteeringMirrorRungAssessment(
        assessment_id=f'fine-steering-mirror-{tuple(EvidenceRung)[rank].value.lower().replace("_", "-")}-assessment',
        rung=rung,
        scientific_status=scientific,
        readiness_status=readiness,
        reason_codes=tuple(sorted(reasons)),
    )


def _within_action_peak_range(
    blocks: tuple[FineSteeringMirrorBlockResponse, ...],
) -> Decimal:
    maximum = Decimal("0")
    levels = sorted({block.action_level_volts for block in blocks})
    for level in levels:
        current = [block for block in blocks if block.action_level_volts == level]
        if len(current) != 2:
            raise ValueError("development recurrence requires two blocks per action level")
        for output_index in range(3):
            values = [block.receiver_peak_hz[output_index] for block in current]
            maximum = max(maximum, max(values) - min(values))
    return maximum


def evaluate_vertical_slice(
    *,
    protocol: FineSteeringMirrorProtocolSpec,
    protocol_sha256: str,
    freeze_commit: str,
    source_member_sha256: tuple[str, ...],
    development_blocks: tuple[FineSteeringMirrorBlockResponse, ...],
    development_model: FineSteeringMirrorDevelopmentModel,
    evaluation_blocks: tuple[FineSteeringMirrorBlockResponse, ...],
) -> FineSteeringMirrorVerticalSliceResult:
    "Grade the exact frozen evidence ladder without constructing a local law."

    errors, correct_rmse, reversed_rmse = receiver_errors(
        development_model,
        evaluation_blocks,
        protocol.action_levels_volts,
    )
    all_blocks = (*development_blocks, *evaluation_blocks)
    source_geometry_pass = all(
        block.maximum_input_matrix_condition <= protocol.maximum_input_matrix_condition
        for block in all_blocks
    )
    period_recurrence_pass = all(
        block.period_relative_difference <= protocol.maximum_period_relative_difference
        for block in all_blocks
    )
    block_recurrence_pass = (
        _within_action_peak_range(development_blocks)
        <= protocol.maximum_within_action_peak_range_hz
    )
    slope_pass = all(value < 0 for value in development_model.slope_hz_per_v) and (
        development_model.slope_hz_per_v[protocol.primary_receiver_index]
        <= protocol.maximum_primary_slope_hz_per_v
    )
    heldout_error_pass = all(
        value <= protocol.maximum_heldout_absolute_error_hz for row in errors for value in row
    )
    wrong_action_advantage = reversed_rmse - correct_rmse
    wrong_action_pass = wrong_action_advantage >= protocol.minimum_primary_wrong_action_advantage_hz
    order_relation_pass = source_geometry_pass and period_recurrence_pass and block_recurrence_pass
    action_response_pass = order_relation_pass and slope_pass and heldout_error_pass and wrong_action_pass
    order_relation_reasons = (
        ("independent-block-and-period-recurrence-cleared",)
        if order_relation_pass
        else tuple(
            reason
            for condition, reason in (
                (source_geometry_pass, "input-matrix-condition-failed"),
                (period_recurrence_pass, "period-recurrence-failed"),
                (block_recurrence_pass, "development-block-recurrence-failed"),
            )
            if not condition
        )
    )
    action_response_reasons = (
        ("heldout-action-response-and-wrong-action-falsifier-cleared",)
        if action_response_pass
        else tuple(
            reason
            for condition, reason in (
                (slope_pass, "development-action-slope-gate-failed"),
                (heldout_error_pass, "heldout-prediction-error-gate-failed"),
                (wrong_action_pass, "wrong-action-falsifier-failed"),
                (order_relation_pass, "order-relation-prerequisite-failed"),
            )
            if not condition
        )
    )
    assessments = (
        _assessment(
            EvidenceRung.MEASUREMENT,
            ScientificStatus.SUPPORTED,
            ReadinessStatus.READY,
            ("exact-safe-source-measurements-adopted",),
        ),
        _assessment(
            EvidenceRung.ORDER_RELATION,
            ScientificStatus.SUPPORTED if order_relation_pass else ScientificStatus.NOT_SUPPORTED,
            ReadinessStatus.READY,
            order_relation_reasons,
        ),
        _assessment(
            EvidenceRung.RESPONSE,
            ScientificStatus.SUPPORTED if action_response_pass else ScientificStatus.NOT_SUPPORTED,
            ReadinessStatus.READY,
            action_response_reasons,
        ),
        _assessment(
            EvidenceRung.LOCAL_LAW,
            ScientificStatus.UNEVALUABLE,
            ReadinessStatus.REQUIRES_NEW_DATA,
            (
                "evaluation-recurrence-one-block-per-action",
                "no-one-factor-denominator-exchange",
                "wrong-time-unevaluable-for-phase-invariant-peak",
            ),
        ),
        _assessment(
            EvidenceRung.ADMISSION,
            ScientificStatus.NOT_TESTED,
            ReadinessStatus.PREREQUISITE_NOT_MET,
            (
                "complete-admission-intersection-absent",
                'law-qualification-prerequisite-not-met',
            ),
        ),
        _assessment(
            EvidenceRung.CONTROLLER_USE,
            ScientificStatus.NOT_TESTED,
            ReadinessStatus.PREREQUISITE_NOT_MET,
            (
                "no-controller-synthesized",
                'admission-prerequisite-not-met',
                "physical-actuation-authority-not-granted",
            ),
        ),
    )
    gap = AtlasGap(
        gap_id='fine-steering-mirror-law-qualification-evidence-gap',
        kind=AtlasGapKind.NO_DATA,
        chart_ids=("fsm-source-rms-family-chart",),
        denominator_cell_ids=("fsm-published-preparation-cell",),
        reason_codes=(
            "evaluation-recurrence-insufficient",
            "one-factor-denominator-exchange-absent",
            "wrong-time-falsifier-unevaluable",
        ),
        evidence_link_ids=(),
    )
    audit = FineSteeringMirrorAdversarialAudit(
        audit_id='fine-steering-mirror-adversarial-audit',
        check_ids=(
            "action-chart-not-inflated-to-arbitrary-three-axis-control",
            "admission-not-substituted-by-predictivity",
            "delivery-chain-denominator-retained",
            "nested-units-not-inflated",
            "no-law-emitted-across-evidence-gap",
            "published-prior-not-relabelled-discovery",
            "simulator-transport-explicitly-not-applicable",
        ),
        failed_check_ids=(),
        conclusion_codes=(
            "physical-response-evidence-does-not-establish-control",
            "fresh-evidence-requires-fresh-blocks-and-delivery-measurement",
        ),
    )
    return FineSteeringMirrorVerticalSliceResult(
        result_id="fine-steering-mirror-empirical-slice",
        system_id=fine_steering_mirror_system().system_id,
        protocol_sha256=protocol_sha256,
        freeze_commit=freeze_commit,
        source_member_sha256=tuple(sorted(source_member_sha256)),
        development_model=development_model,
        evaluation_blocks=evaluation_blocks,
        heldout_absolute_error_hz=errors,
        primary_correct_action_rmse_hz=correct_rmse,
        primary_reversed_action_rmse_hz=reversed_rmse,
        rung_assessments=assessments,
        atlas_gap=gap,
        transport_status=FineSteeringMirrorTransportStatus.NOT_APPLICABLE_NO_SIMULATOR,
        controller_status=ScientificStatus.NOT_TESTED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        adversarial_audit=audit,
    )


def reveal_evaluation_blocks(
    *,
    action_paths: tuple[Path, Path, Path],
    sealed_receiver_paths: tuple[Path, Path, Path],
    protocol: FineSteeringMirrorProtocolSpec,
) -> tuple[FineSteeringMirrorBlockResponse, ...]:
    """Open sealed receivers only inside the evaluator and reduce one block per action."""

    blocks: list[FineSteeringMirrorBlockResponse] = []
    for index, level in enumerate(protocol.action_levels_volts):
        actions = load_safe_array(
            action_paths[index],
            expected_realizations=3,
            operator=protocol.observation_operator,
        )
        receivers = load_safe_array(
            sealed_receiver_paths[index],
            expected_realizations=3,
            operator=protocol.observation_operator,
        )
        blocks.append(
            reduce_block(
                block_id=f"fsm-evaluation-{int(level * 1000)}mv-block-1",
                action_level_volts=level,
                actions=actions,
                receivers=receivers,
                realization_start=0,
                operator=protocol.observation_operator,
            )
        )
    return tuple(blocks)
