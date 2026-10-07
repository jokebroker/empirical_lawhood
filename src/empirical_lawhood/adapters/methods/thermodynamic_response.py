"""Transparent method-side records for arbitrary finite-word trajectory panels."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.response_algebra import ActionStage
from empirical_lawhood.kernel.thermodynamic_response import ApplicabilityDisposition, FiniteWordMode, ScalarDoseActionWordApplicabilityCheck, ScalarDoseActionWordApplicabilityCriterion, ScalarDoseActionWordProjectionWitness, ScalarDoseActionWordSignatureStatus
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.thermodynamic_response import FiniteWordFamilySpec

from .contracts import DataSplit
from .response_algebra import (
    ResponseAlgebraMethodConfig,
    ResponseAlgebraMethodInput,
    ResponseAlgebraMethodResult,
    identify_response_algebra,
    response_algebra_unit_contrasts,
)
from .response_algebra_posthoc import WORD_IDS


_SCALAR_DOSE_WORD_SEMANTICS = {
    "a-early": ("SEQUENTIAL", ("a",)),
    "a-late": ("SEQUENTIAL", ("a",)),
    "a-repeat": ("SEQUENTIAL", ("a", "a")),
    "a-then-b": ("SEQUENTIAL", ("a", "b")),
    "b-early": ("SEQUENTIAL", ("b",)),
    "b-late": ("SEQUENTIAL", ("b",)),
    "b-repeat": ("SEQUENTIAL", ("b", "b")),
    "b-then-a": ("SEQUENTIAL", ("b", "a")),
    "identity": ("IDENTITY", ()),
    "simultaneous-a-b": ("SIMULTANEOUS", ("a", "b")),
}


def _check(
    criterion: ScalarDoseActionWordApplicabilityCriterion,
    disposition: ApplicabilityDisposition,
    *reason_codes: str,
) -> ScalarDoseActionWordApplicabilityCheck:
    return ScalarDoseActionWordApplicabilityCheck(
        check_id=f"check-{criterion.value.lower().replace('_', '-')}",
        criterion=criterion,
        disposition=disposition,
        reason_codes=tuple(sorted(reason_codes)),
    )


def _passing_checks() -> tuple[ScalarDoseActionWordApplicabilityCheck, ...]:
    return tuple(
        sorted(
            (
                _check(criterion, ApplicabilityDisposition.PASS)
                for criterion in ScalarDoseActionWordApplicabilityCriterion
            ),
            key=lambda value: value.check_id,
        )
    )


def _assert_exact_delivered_family(method_input: ResponseAlgebraMethodInput) -> None:
    held_out = tuple(
        value
        for value in method_input.delivered_words
        if value.split is DataSplit.HELD_OUT and value.dose == Decimal("1")
    )
    by_unit: dict[str, dict[str, object]] = {}
    for record in held_out:
        words = by_unit.setdefault(record.independent_unit_id, {})
        if record.family_word_id in words:
            raise ValueError("scalar-dose action-word delivered family duplicates one unit/word branch")
        words[record.family_word_id] = record
    if not by_unit or any(set(words) != set(WORD_IDS) for words in by_unit.values()):
        raise ValueError("scalar-dose action-word delivery lacks the exact complete ten-word family")
    for words in by_unit.values():
        action_words = {
            word_id: getattr(record, "delivered_word") for word_id, record in words.items()
        }
        for word_id, action_word in action_words.items():
            mode, letter_ids = _SCALAR_DOSE_WORD_SEMANTICS[word_id]
            if (
                action_word.mode.value != mode
                or tuple(letter.letter_id for letter in action_word.letters) != letter_ids
            ):
                raise ValueError("scalar-dose action-word delivered family changes frozen word semantics")
        early = action_words["a-early"].letters[0].applied
        late = action_words["a-late"].letters[0].applied
        if early.clock_id != late.clock_id or early.clock_coordinate >= late.clock_coordinate:
            raise ValueError("scalar-dose action-word delivered family changes early/late chronology")
        expected_clocks = {
            "b-early": (early.clock_id, early.clock_coordinate),
            "b-late": (late.clock_id, late.clock_coordinate),
        }
        for word_id, expected in expected_clocks.items():
            observed = action_words[word_id].letters[0].applied
            if (observed.clock_id, observed.clock_coordinate) != expected:
                raise ValueError("scalar-dose action-word delivered family changes matched word schedules")
        for word_id in ("a-repeat", "a-then-b", "b-repeat", "b-then-a"):
            observed = tuple(
                (letter.applied.clock_id, letter.applied.clock_coordinate)
                for letter in action_words[word_id].letters
            )
            if observed != (
                (early.clock_id, early.clock_coordinate),
                (late.clock_id, late.clock_coordinate),
            ):
                raise ValueError("scalar-dose action-word delivered family changes matched word schedules")
        simultaneous = action_words["simultaneous-a-b"].letters
        if any(
            (letter.applied.clock_id, letter.applied.clock_coordinate)
            != (early.clock_id, early.clock_coordinate)
            for letter in simultaneous
        ):
            raise ValueError("scalar-dose action-word delivered family changes exact simultaneity")
        receiver_horizons = {(word.receiver_id, word.horizon_id) for word in action_words.values()}
        if len(receiver_horizons) != 1:
            raise ValueError("scalar-dose action-word delivered family changes receiver or horizon")


def exact_projection_witness(
    *,
    witness_id: str,
    method_input: ResponseAlgebraMethodInput,
    method_config: ResponseAlgebraMethodConfig,
    method_result: ResponseAlgebraMethodResult,
    parent_receipt_id: str | None = None,
) -> ScalarDoseActionWordProjectionWitness:
    "Re-execute and byte-compare an exact scalar-dose action-word result before exposing its signature."

    if not (
        method_config.require_exact_stage_value_equality
        and method_config.require_exact_stage_clock_equality
    ):
        raise ValueError("exact scalar-dose action-word projection requires exact stage value and clock checks")
    _assert_exact_delivered_family(method_input)
    response_algebra_unit_contrasts(method_input, method_config)
    reproduced = identify_response_algebra(method_input, method_config)
    if reproduced.canonical_bytes() != method_result.canonical_bytes():
        raise ValueError("scalar-dose action-word method result does not reproduce byte-for-byte")
    status = (
        ScalarDoseActionWordSignatureStatus.RECEIPT_BOUND_SCALAR_DOSE_PARENT
        if parent_receipt_id is not None
        else ScalarDoseActionWordSignatureStatus.EXACT_SCALAR_DOSE
    )
    return ScalarDoseActionWordProjectionWitness(
        witness_id=witness_id,
        source_schema=method_result.SCHEMA,
        source_object_id=method_result.result_id,
        status=status,
        checks=_passing_checks(),
        response_signature=method_result.signature,
        parent_receipt_id=parent_receipt_id,
        reason_codes=(),
    )


def nonapplicability_witness_for(
    *,
    witness_id: str,
    family: FiniteWordFamilySpec,
    panel: FiniteTrajectoryPanel,
) -> ScalarDoseActionWordProjectionWitness:
    "Return a typed scalar-dose action-word applicability failure without coercing a richer finite-word family.\n\n    If every structural criterion passes, callers must construct and execute an\n    exact scalar-dose action-word method input; this helper deliberately refuses to invent a scalar-dose action-word\n    response signature from finite-word metadata or trajectories.\n    "

    dispositions: dict[ScalarDoseActionWordApplicabilityCriterion, tuple[bool, str]] = {}
    deliveries = tuple(delivery for word in family.words for delivery in word.deliveries)
    stage_unit_compatible = all(
        len({stage.native_unit for stage in delivery.stages}) == 1 for delivery in deliveries
    )
    dispositions[ScalarDoseActionWordApplicabilityCriterion.STAGE_UNIT_CONTRACT] = (
        stage_unit_compatible,
        "V1_STAGE_UNIT_CONTRACT_FAILED",
    )
    scalar_nonnegative = all(
        len(delivery.delivered_dose) == 1
        and delivery.sign >= 0
        and delivery.delivered_dose[0].value >= 0
        for delivery in deliveries
    )
    dispositions[ScalarDoseActionWordApplicabilityCriterion.SCALAR_NONNEGATIVE_DOSE] = (
        scalar_nonnegative,
        "V1_SCALAR_NONNEGATIVE_DOSE_FAILED",
    )
    words_by_id = {word.word_id: word for word in family.words}
    matched_semantics = set(words_by_id) == set(WORD_IDS)
    if matched_semantics:
        for word_id, (mode, letter_ids) in _SCALAR_DOSE_WORD_SEMANTICS.items():
            word = words_by_id[word_id]
            expected_mode = {
                "IDENTITY": FiniteWordMode.IDENTITY,
                "SEQUENTIAL": FiniteWordMode.SEQUENTIAL,
                "SIMULTANEOUS": FiniteWordMode.SIMULTANEOUS,
            }[mode]
            if (
                word.mode is not expected_mode
                or tuple(delivery.letter_id for delivery in word.deliveries) != letter_ids
            ):
                matched_semantics = False
                break
    matched_semantics = matched_semantics and panel.word_ids == WORD_IDS
    dispositions[ScalarDoseActionWordApplicabilityCriterion.MATCHED_WORD_SCHEDULES] = (
        matched_semantics,
        "V1_MATCHED_WORD_SCHEDULES_FAILED",
    )
    simultaneous = words_by_id.get("simultaneous-a-b")
    exact_simultaneity = simultaneous is not None and len(simultaneous.deliveries) == 2
    if exact_simultaneity and simultaneous is not None:
        stage_clocks = {
            (stage.clock_id, stage.clock_coordinate)
            for delivery in simultaneous.deliveries
            for stage in delivery.stages
            if stage.stage is ActionStage.APPLIED
        }
        starts = {delivery.realized_interval.start for delivery in simultaneous.deliveries}
        exact_simultaneity = len(stage_clocks) == 1 and len(starts) == 1
    dispositions[ScalarDoseActionWordApplicabilityCriterion.EXACT_SIMULTANEITY] = (
        exact_simultaneity,
        "V1_EXACT_SIMULTANEITY_FAILED",
    )
    complete_controls = (
        set(words_by_id) == set(WORD_IDS)
        and "identity" in words_by_id
        and all(word.composition_status.value == "DEFINED" for word in family.words)
    )
    dispositions[ScalarDoseActionWordApplicabilityCriterion.COMPLETE_PREFIXES_AND_CONTROLS] = (
        complete_controls,
        "V1_COMPLETE_PREFIXES_AND_CONTROLS_FAILED",
    )
    receiver_horizon = (
        len(family.receiver_ids) == 1
        and len(family.horizon_ids) == 1
        and panel.receiver_id == family.receiver_ids[0]
        and panel.horizon_id == family.horizon_ids[0]
    )
    dispositions[ScalarDoseActionWordApplicabilityCriterion.RECEIVER_AND_HORIZON_CONTRACT] = (
        receiver_horizon,
        "V1_RECEIVER_AND_HORIZON_CONTRACT_FAILED",
    )
    complete_units = (
        panel.word_ids == WORD_IDS
        and all(value is not None for value in panel.source_episode_ids)
        and all(value is not None for value in panel.values)
    )
    dispositions[ScalarDoseActionWordApplicabilityCriterion.COMPLETE_UNIT_AGGREGATION] = (
        complete_units,
        "V1_COMPLETE_UNIT_AGGREGATION_FAILED",
    )
    if all(passed for passed, _reason in dispositions.values()):
        raise ValueError("finite-word family may be applicable to the scalar-dose action-word method; exact scalar-dose action-word execution is required")
    checks = tuple(
        sorted(
            (
                _check(
                    criterion,
                    ApplicabilityDisposition.PASS if passed else ApplicabilityDisposition.FAIL,
                    *(() if passed else (reason,)),
                )
                for criterion, (passed, reason) in dispositions.items()
            ),
            key=lambda value: value.check_id,
        )
    )
    reasons = tuple(sorted(reason for passed, reason in dispositions.values() if not passed))
    return ScalarDoseActionWordProjectionWitness(
        witness_id=witness_id,
        source_schema=family.SCHEMA,
        source_object_id=family.family_id,
        status=ScalarDoseActionWordSignatureStatus.SCALAR_DOSE_NOT_APPLICABLE,
        checks=checks,
        response_signature=None,
        parent_receipt_id=None,
        reason_codes=reasons,
    )


@dataclass(frozen=True, slots=True)
class FiniteTrajectoryPanel(CanonicalRecord):
    """Exact finite-word trajectories with explicit incomplete assignment blocks.

    Values are flattened in ``[independent_unit, word, time, coordinate]``
    order. The word axis preserves the frozen inventory order. A missing block
    is represented by a null episode identity, null digest, null values and a
    declared reason; it never manufactures a complete counterfactual branch.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-trajectory-panel'

    panel_id: str
    system_id: str
    relation_id: str
    evidence_world_id: str
    world_kind: WorldKind
    split: DataSplit
    numerical_view_id: str
    state_view_id: str
    receiver_id: str
    horizon_id: str
    independent_unit_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    identity_word_id: str
    times: tuple[Decimal, ...]
    clock_id: str
    time_unit: str
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    values: tuple[Decimal | None, ...]
    source_episode_ids: tuple[str | None, ...]
    source_episode_sha256s: tuple[str | None, ...]
    matched_identity_episode_ids: tuple[str | None, ...]
    block_reason_codes: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("panel_id", self.panel_id),
            ("system_id", self.system_id),
            ("relation_id", self.relation_id),
            ("evidence_world_id", self.evidence_world_id),
            ("numerical_view_id", self.numerical_view_id),
            ("state_view_id", self.state_view_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        for unit_id in self.independent_unit_ids:
            validate_stable_id(unit_id, field_name="independent_unit_ids")
        if not self.word_ids or len(set(self.word_ids)) != len(self.word_ids):
            raise ValueError("ordered word inventory must be nonempty and unique")
        for word_id in self.word_ids:
            validate_stable_id(word_id, field_name="word_ids")
        validate_stable_id(self.identity_word_id, field_name="identity_word_id")
        if self.identity_word_id not in self.word_ids:
            raise ValueError("panel identity word is outside its frozen inventory")
        if not self.times:
            raise ValueError("finite trajectory panel requires observation times")
        for time_value in self.times:
            validate_decimal(time_value, field_name="times", minimum=Decimal("0"))
        if any(later <= earlier for earlier, later in zip(self.times, self.times[1:])):
            raise ValueError("finite trajectory panel times must increase")
        validate_nonempty(self.time_unit, field_name="time_unit")
        if not self.coordinate_ids or len(set(self.coordinate_ids)) != len(self.coordinate_ids):
            raise ValueError("panel coordinates must be nonempty and unique")
        for coordinate_id in self.coordinate_ids:
            validate_stable_id(coordinate_id, field_name="coordinate_ids")
        if len(self.native_units) != len(self.coordinate_ids):
            raise ValueError("panel coordinate identities and native units differ")
        for native_unit in self.native_units:
            validate_nonempty(native_unit, field_name="native_units")

        block_count = len(self.independent_unit_ids) * len(self.word_ids)
        if not (
            len(self.source_episode_ids)
            == len(self.source_episode_sha256s)
            == len(self.matched_identity_episode_ids)
            == len(self.block_reason_codes)
            == block_count
        ):
            raise ValueError("panel episode metadata differs from its unit-word blocks")
        value_block_size = len(self.times) * len(self.coordinate_ids)
        if len(self.values) != block_count * value_block_size:
            raise ValueError("panel value count differs from its declared tensor shape")
        if len(self.values) > 10_000_000:
            raise ValueError("finite trajectory panel exceeds its bounded value count")
        for panel_value in self.values:
            if panel_value is not None:
                validate_decimal(panel_value, field_name="values")

        observed_episode_ids: set[str] = set()
        observed_unit_indices: set[int] = set()
        observed_identity_episodes = {
            episode_id
            for unit_index in range(len(self.independent_unit_ids))
            for word_index, word_id in enumerate(self.word_ids)
            if word_id == self.identity_word_id
            for episode_id in (
                self.source_episode_ids[unit_index * len(self.word_ids) + word_index],
            )
            if episode_id is not None
        }
        for block_index, (episode_id, sha256, matched_identity_id, reasons) in enumerate(
            zip(
                self.source_episode_ids,
                self.source_episode_sha256s,
                self.matched_identity_episode_ids,
                self.block_reason_codes,
                strict=True,
            )
        ):
            require_sorted_unique_strings(reasons, field_name="block_reason_codes")
            start = block_index * value_block_size
            block_values = self.values[start : start + value_block_size]
            if episode_id is None:
                if (
                    sha256 is not None
                    or matched_identity_id is not None
                    or any(value is not None for value in block_values)
                ):
                    raise ValueError("unassigned panel block cannot carry episode data")
                if not reasons:
                    raise ValueError("unassigned panel block requires a reason")
                continue
            validate_stable_id(episode_id, field_name="source_episode_ids")
            if episode_id in observed_episode_ids:
                raise ValueError("one realized episode cannot populate several word branches")
            observed_episode_ids.add(episode_id)
            observed_unit_indices.add(block_index // len(self.word_ids))
            if sha256 is None:
                raise ValueError("assigned panel block requires a source digest")
            validate_sha256(sha256, field_name="source_episode_sha256s")
            if matched_identity_id is None:
                raise ValueError("assigned panel block requires a frozen matched identity")
            validate_stable_id(
                matched_identity_id,
                field_name="matched_identity_episode_ids",
            )
            word_index = block_index % len(self.word_ids)
            if self.word_ids[word_index] == self.identity_word_id:
                if matched_identity_id != episode_id:
                    raise ValueError("identity episode must match itself")
            elif matched_identity_id not in observed_identity_episodes:
                raise ValueError("word block references an absent matched identity episode")
            if any(value is None for value in block_values) and not reasons:
                raise ValueError("incomplete assigned panel block requires a reason")
        if observed_unit_indices != set(range(len(self.independent_unit_ids))):
            raise ValueError("every declared physical unit requires an assigned episode")


__all__ = [
    'FiniteTrajectoryPanel',
    'exact_projection_witness',
    'nonapplicability_witness_for',
]
