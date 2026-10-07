# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import timezone
from decimal import Decimal

import pytest

from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import (
    CausalPhase,
    ClockRelationKind,
    ClockRelationSpec,
    InformationCutoff,
    parse_utc_timestamp,
    validate_utc_timestamp,
)


def _claim(system: SystemSpec, **changes: object) -> ClaimSpec:
    values: dict[str, object] = {
        "claim_id": "thermal-law-claim",
        "world_id": system.world.world_id,
        "relation_id": system.relation.relation_id,
        "proposition": "Applied power has a supported local thermal response.",
        "estimand": "Ten-second temperature displacement per applied watt.",
        "physical_independent_unit_id": system.independent_unit.unit_id,
        "requested_rung": EvidenceRung.LOCAL_LAW,
        "evidence_ceiling": EvidenceCeiling.LOCAL_LAW,
        "outcome_access": OutcomeAccess.EVALUATION_SEALED,
        "visibility_ceiling": VisibilityCeiling.PROSPECTIVE,
        "promotion_rule": "All predeclared recurrence and wrong-action gates pass.",
        "assumption_ids": ("causal-cutoff", "local-linearity"),
        "numerical_view_ids": ("thermal-view-fine",),
    }
    values.update(changes)
    return ClaimSpec(**values)  # type: ignore[arg-type]


def test_prospective_claim_validates_against_system(
    numerical_system: SystemSpec,
) -> None:
    numerical_system.validate_claim(_claim(numerical_system))


def test_claim_cannot_exceed_world_ceiling(numerical_system: SystemSpec) -> None:
    restricted_world = replace(numerical_system.world, maximum_evidence=EvidenceCeiling.RESPONSE)
    restricted_system = replace(numerical_system, world=restricted_world)
    with pytest.raises(ValueError, match="world ceiling"):
        restricted_system.validate_claim(_claim(restricted_system))


def test_declared_claim_ceiling_cannot_overstate_world(
    numerical_system: SystemSpec,
) -> None:
    restricted_world = replace(numerical_system.world, maximum_evidence=EvidenceCeiling.RESPONSE)
    restricted_system = replace(numerical_system, world=restricted_world)
    claim = _claim(
        restricted_system,
        requested_rung=EvidenceRung.RESPONSE,
        evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
    )
    with pytest.raises(ValueError, match="world ceiling"):
        restricted_system.validate_claim(claim)


def test_outcome_visible_claim_is_structurally_nonpromotable(
    numerical_system: SystemSpec,
) -> None:
    with pytest.raises(ValueError, match="non-promotable"):
        _claim(
            numerical_system,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
    finding = _claim(
        numerical_system,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    numerical_system.validate_claim(finding)


def test_visibility_can_only_propagate_toward_more_restrictive() -> None:
    assert (
        inherited_visibility((VisibilityCeiling.OUTCOME_VISIBLE,), OutcomeAccess.OUTCOME_BLIND)
        is VisibilityCeiling.OUTCOME_VISIBLE
    )


@pytest.mark.parametrize(
    "value",
    (
        "2026-07-16T00:00:00Z",
        "2026-07-16T00:00:00.1Z",
        "2026-07-16T00:00:00.123456Z",
    ),
)
def test_strict_utc_timestamp_parser_and_validator(value: str) -> None:
    parsed = parse_utc_timestamp(value, field_name="retrieved_at_utc")

    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    assert validate_utc_timestamp(value, field_name="retrieved_at_utc") == value


@pytest.mark.parametrize(
    "value",
    (
        "",
        "2026-07-16",
        "2026-07-16 00:00:00Z",
        "2026-07-16T00:00Z",
        "2026-07-16T00:00:00z",
        "2026-07-16T00:00:00+00:00",
        "2026-07-16T00:00:00.1234567Z",
        "2026-02-30T00:00:00Z",
        "2026-07-16T24:00:00Z",
        "2026-07-16T00:00:60Z",
    ),
)
def test_strict_utc_timestamp_rejects_noncanonical_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="retrieved_at_utc"):
        parse_utc_timestamp(value, field_name="retrieved_at_utc")


def test_claim_cannot_launder_parent_visibility(numerical_system: SystemSpec) -> None:
    with pytest.raises(ValueError, match="cannot be lowered"):
        _claim(
            numerical_system,
            derivation_parent_ids=("posthoc-parent",),
            parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
        )


def test_pre_action_cutoff_rejects_applied_action_and_receiver(
    numerical_system: SystemSpec,
) -> None:
    cutoff = InformationCutoff(
        cutoff_id="pre-action-cutoff",
        clock_id="experiment-clock",
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal("5"),
    )
    numerical_system.require_inputs_available(cutoff, ("ambient-boundary", "substrate"))
    with pytest.raises(ValueError, match="exceeds cutoff"):
        numerical_system.require_inputs_available(cutoff, ("action-power",))
    with pytest.raises(ValueError, match="exceeds cutoff"):
        numerical_system.require_inputs_available(cutoff, ("receiver-temperature",))


def test_cutoff_rejects_wrong_clock(numerical_system: SystemSpec) -> None:
    cutoff = InformationCutoff(
        cutoff_id="other-clock-cutoff",
        clock_id="other-clock",
        phase=CausalPhase.POST_OUTCOME,
        coordinate=None,
    )
    with pytest.raises(ValueError, match="exceeds cutoff"):
        numerical_system.require_inputs_available(cutoff, ("substrate",))


def test_clock_relation_rejects_acausal_negative_delay() -> None:
    with pytest.raises(ValueError, match="at least"):
        ClockRelationSpec(
            relation_id="acausal-clock-map",
            source_clock_id="source-clock",
            target_clock_id="target-clock",
            kind=ClockRelationKind.FIXED_DELAY,
            delay=Decimal("-0.1"),
            tolerance=Decimal("0"),
            evidence_contract_id="clock-evidence",
        )
