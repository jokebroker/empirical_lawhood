"""Oracle-separated sixteen-case thermodynamic-response reference catalogue."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import DataSplit
from empirical_lawhood.adapters.methods.thermodynamic_response import FiniteTrajectoryPanel
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.thermodynamic_response import (
    BalanceClosureStatus,
    EntropyEligibilityStatus,
    ExchangeEstimandKind,
    FiniteAxisDisposition,
    ReceiverMapDisposition,
    ReturnQualificationStatus,
    ThermodynamicCoordinateRole,
    ThermodynamicLedgerObservation,
    ThermodynamicObservationStatus,
    ThermodynamicRoleBinding,
    ThermodynamicSignConvention,
    TransportDisposition,
    validate_thermodynamic_role_bindings,
)
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.thermodynamic_response import (
    THERMODYNAMIC_RESPONSE_CONFORMANCE_CASE_TOKENS,
    ThermodynamicResponseConformanceSpec,
    reference_thermodynamic_response_conformance_spec,
)


class ThermodynamicReferenceTruthKind(StrEnum):
    REVERSIBLE_COMMUTING = "REVERSIBLE_COMMUTING"
    RETURN_QUALIFIED_DISSIPATIVE_COMMUTING = "RETURN_QUALIFIED_DISSIPATIVE_COMMUTING"
    RETURN_QUALIFIED_FINITE_ORDER = "RETURN_QUALIFIED_FINITE_ORDER"
    TIMING_COUNTERFEIT = "TIMING_COUNTERFEIT"
    HIDDEN_BATH = "HIDDEN_BATH"
    PROJECTED_CURVATURE = "PROJECTED_CURVATURE"
    NON_RETURN_COUNTERFEIT = "NON_RETURN_COUNTERFEIT"
    MISSING_BALANCE_TERM = "MISSING_BALANCE_TERM"
    COLLAPSED_ACTION_FIBRE = "COLLAPSED_ACTION_FIBRE"
    NONSTATIONARY_COCYCLE = "NONSTATIONARY_COCYCLE"
    HYBRID_SWITCHING = "HYBRID_SWITCHING"
    STOCHASTIC_FINITE_KERNEL = "STOCHASTIC_FINITE_KERNEL"
    HOUSEKEEPING_COUNTERFEIT = "HOUSEKEEPING_COUNTERFEIT"
    ACTION_EXCHANGE_DOUBLE_COUNT = "ACTION_EXCHANGE_DOUBLE_COUNT"
    AGEING_CARRYOVER_COUNTERFEIT = "AGEING_CARRYOVER_COUNTERFEIT"
    FINITE_COHERENCE_AND_TRANSPORT = "FINITE_COHERENCE_AND_TRANSPORT"


class FiniteRepresentationKind(StrEnum):
    DETERMINISTIC_FINITE = "DETERMINISTIC_FINITE"
    STOCHASTIC_FINITE = "STOCHASTIC_FINITE"
    HYBRID_FINITE = "HYBRID_FINITE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ThermodynamicTruthBlindInput(CanonicalRecord):
    """Generated observations without truth names or expected dispositions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/thermodynamic-truth-blind-input'

    input_id: str
    panel: FiniteTrajectoryPanel
    ledger_observations: tuple[ThermodynamicLedgerObservation, ...]
    diagnostic_values: tuple[NamedDecimal, ...]
    observed_context_ids: tuple[str, ...]
    role_bindings: tuple[ThermodynamicRoleBinding, ...]
    housekeeping_model_id: str | None
    system_boundary_complete: bool
    chemical_terms_complete: bool
    synchronization_valid: bool
    constitutive_assumptions_valid: bool
    local_equilibrium_justified: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        if self.input_id != self.panel.panel_id:
            raise ValueError("truth-blind input identity differs from its trajectory panel")
        if self.panel.world_kind is not WorldKind.ANALYTIC_REFERENCE:
            raise ValueError("truth-known conformance requires an analytic reference world")
        require_sorted_unique_ids(
            self.ledger_observations,
            attribute="observation_id",
            field_name="ledger_observations",
        )
        if any(value.system_id != self.panel.system_id for value in self.ledger_observations):
            raise ValueError("truth-blind ledger crosses its reference-system boundary")
        require_sorted_unique_ids(
            self.diagnostic_values,
            attribute="value_id",
            field_name="diagnostic_values",
        )
        require_sorted_unique_strings(
            self.observed_context_ids,
            field_name="observed_context_ids",
        )
        validate_thermodynamic_role_bindings(self.role_bindings)
        if self.housekeeping_model_id is not None:
            validate_stable_id(self.housekeeping_model_id, field_name="housekeeping_model_id")


@dataclass(frozen=True, slots=True)
class PrivilegedThermodynamicOracle(CanonicalRecord):
    """Expected scientific dispositions, never an input to the analysis method."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/privileged-thermodynamic-oracle'

    oracle_id: str
    input_id: str
    truth_kind: ThermodynamicReferenceTruthKind
    controlled_order: FiniteAxisDisposition
    state_bath_return: ReturnQualificationStatus
    energy_balance: BalanceClosureStatus
    entropy_eligibility: EntropyEligibilityStatus
    action_quotient: FiniteAxisDisposition
    temporal_cocycle: FiniteAxisDisposition
    equal_lag_stationarity: FiniteAxisDisposition
    receiver_map: ReceiverMapDisposition
    cycle_interpretation_allowed: bool
    context_augmentation: FiniteAxisDisposition
    representation: FiniteRepresentationKind
    perturbation_exchange: FiniteAxisDisposition
    denominator_stability: FiniteAxisDisposition
    finite_coherence: FiniteAxisDisposition
    transport: TransportDisposition
    double_count_rejected: bool
    lie_claim_allowed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.oracle_id, field_name="oracle_id")
        validate_stable_id(self.input_id, field_name="input_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.lie_claim_allowed:
            raise ValueError("finite truth-known cases never establish an infinitesimal Lie claim")


_WORD_IDS = (
    "a",
    "ab",
    "b",
    "ba",
    "cycle",
    "identity",
    "longer-aba",
    "overlap-ab",
    "repeat-a",
    "timing-ab",
    "timing-ba",
)


def _trajectory(
    case_index: int,
    unit_index: int,
    word_id: str,
    word_index: int,
    time_index: int,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    time = Decimal(time_index)
    fraction = time / Decimal("3")
    bath = Decimal(unit_index) / Decimal("10")
    mode = Decimal("0")
    receiver_x = Decimal(unit_index + 1)
    receiver_y = Decimal(2 * (unit_index + 1))
    if case_index == 9:
        receiver_x += time**2
        receiver_y += (time**2) / Decimal("2")
    else:
        receiver_x += time / Decimal("10")
        receiver_y += time / Decimal("20")

    ordinary_effects = {
        "a": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0")),
        "ab": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("1")),
        "b": (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("1")),
        "ba": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("1")),
        "identity": (Decimal("0"),) * 4,
        "longer-aba": (
            Decimal("0"),
            Decimal("0"),
            Decimal("2"),
            Decimal("1"),
        ),
        "overlap-ab": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("1")),
        "repeat-a": (Decimal("0"), Decimal("0"), Decimal("2"), Decimal("0")),
        "timing-ab": (Decimal("0"),) * 4,
        "timing-ba": (Decimal("0"),) * 4,
    }
    if word_id == "cycle":
        if time_index == 1:
            effect = (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0"))
        elif time_index == 2:
            effect = (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("1"))
        else:
            effect = (Decimal("0"),) * 4
    else:
        effect = ordinary_effects[word_id]

    if case_index in {2, 3, 4, 10} and word_id in {"ab", "ba"}:
        direction = Decimal("1") if word_id == "ab" else Decimal("-1")
        effect = (
            effect[0] + (direction * Decimal("0.8") if case_index == 4 else 0),
            effect[1] + (direction if case_index == 10 else 0),
            effect[2] + direction * Decimal("0.8"),
            effect[3],
        )
    if case_index == 3 and word_id in {"timing-ab", "timing-ba"}:
        direction = Decimal("1") if word_id == "timing-ab" else Decimal("-1")
        effect = (Decimal("0"), Decimal("0"), direction * Decimal("0.8"), Decimal("0"))
    if case_index == 5 and word_id in {"ab", "ba"}:
        direction = Decimal("1") if word_id == "ab" else Decimal("-1")
        effect = (effect[0], effect[1], Decimal("0"), effect[3] + direction * Decimal("0.8"))
    if case_index == 6 and word_id == "cycle" and time_index == 3:
        effect = (Decimal("0.3"), Decimal("0"), Decimal("0.8"), Decimal("0"))
    if case_index == 8 and word_id != "identity":
        effect = (Decimal("0"), Decimal("0"), Decimal("0.01"), Decimal("0.01"))
    if case_index == 11 and word_id not in {"cycle", "identity"}:
        stochastic_sign = Decimal("1") if unit_index % 2 == 0 else Decimal("-1")
        effect = (
            effect[0],
            effect[1],
            effect[2] + stochastic_sign * Decimal("0.2"),
            effect[3] - stochastic_sign * Decimal("0.1"),
        )
    if case_index == 14:
        receiver_x += Decimal(word_index) * Decimal("0.2")
        bath += Decimal(word_index) * Decimal("0.05")
    if case_index == 15:
        if word_id == "a":
            effect = (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("1"))
        elif word_id == "b":
            effect = (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("-1"))

    return (
        bath + effect[0] * fraction,
        mode + effect[1] * fraction,
        receiver_x + effect[2] * fraction,
        receiver_y + effect[3] * fraction,
    )


def _panel(case_index: int, spec: ThermodynamicResponseConformanceSpec) -> FiniteTrajectoryPanel:
    token = spec.case_tokens[case_index]
    unit_ids = tuple(f"reference-unit-{index:02d}" for index in range(8))
    values: list[Decimal] = []
    episode_ids: list[str] = []
    digests: list[str] = []
    matched_identity_ids: list[str] = []
    reasons: list[tuple[str, ...]] = []
    for unit_index, unit_id in enumerate(unit_ids):
        identity_episode = f"episode-{case_index:02d}-{unit_id}-identity"
        for word_index, word_id in enumerate(_WORD_IDS):
            episode_id = f"episode-{case_index:02d}-{unit_id}-{word_id}"
            episode_ids.append(episode_id)
            digests.append(f"{case_index * 1000 + unit_index * 20 + word_index + 1:064x}")
            matched_identity_ids.append(episode_id if word_id == "identity" else identity_episode)
            reasons.append(())
            for time_index in range(len(spec.time_coordinates)):
                values.extend(_trajectory(case_index, unit_index, word_id, word_index, time_index))
    return FiniteTrajectoryPanel(
        panel_id=token,
        system_id="thermodynamic-reference-system",
        relation_id="thermodynamic-reference-relation",
        evidence_world_id=f"world-{token}",
        world_kind=WorldKind.ANALYTIC_REFERENCE,
        split=DataSplit.HELD_OUT,
        numerical_view_id=(
            "stochastic-transition-reference-view"
            if case_index == 11
            else ("paired-transport-reference-view" if case_index == 15 else "exact-reference-view")
        ),
        state_view_id=(
            "context-augmented-reference-state"
            if case_index in {4, 10}
            else (
                "full-and-coarse-reference-state"
                if case_index in {5, 8, 15}
                else "full-reference-state"
            )
        ),
        receiver_id="full-reference-receiver",
        horizon_id="three-second-horizon",
        independent_unit_ids=unit_ids,
        word_ids=_WORD_IDS,
        identity_word_id="identity",
        times=spec.time_coordinates,
        clock_id="reference-clock",
        time_unit=spec.time_unit,
        coordinate_ids=spec.coordinate_ids,
        native_units=spec.native_units,
        values=tuple(values),
        source_episode_ids=tuple(episode_ids),
        source_episode_sha256s=tuple(digests),
        matched_identity_episode_ids=tuple(matched_identity_ids),
        block_reason_codes=tuple(reasons),
    )


def _ledger_observation(
    *,
    case_index: int,
    word_id: str,
    term_id: str,
    role: ThermodynamicCoordinateRole,
    value: Decimal | None,
    unit: str,
) -> ThermodynamicLedgerObservation:
    status = (
        ThermodynamicObservationStatus.OBSERVED
        if value is not None
        else ThermodynamicObservationStatus.TERM_UNOBSERVED
    )
    storage = role in {
        ThermodynamicCoordinateRole.STORED_ENERGY,
        ThermodynamicCoordinateRole.STATE_ENTROPY,
    }
    estimand = (
        ExchangeEstimandKind.ABSOLUTE_STORAGE_CHANGE
        if role is ThermodynamicCoordinateRole.STORED_ENERGY
        else (
            ExchangeEstimandKind.ABSOLUTE_STATE_ENTROPY_CHANGE
            if role is ThermodynamicCoordinateRole.STATE_ENTROPY
            else ExchangeEstimandKind.TOTAL_EXCHANGE
        )
    )
    return ThermodynamicLedgerObservation(
        observation_id=f"observation-{case_index:02d}-{word_id}-{term_id}",
        system_id="thermodynamic-reference-system",
        independent_unit_id="reference-unit-00",
        episode_id=f"episode-{case_index:02d}-reference-unit-00-{word_id}",
        word_id=word_id,
        term_id=term_id,
        physical_event_id=f"event-{case_index:02d}-{word_id}-{term_id}",
        coordinate_role=role,
        estimand_kind=estimand,
        source_native_value=value,
        source_native_unit=unit,
        source_native_positive_direction="canonical-reference-direction",
        canonical_value=value,
        canonical_unit=unit,
        canonical_sign_convention=(
            ThermodynamicSignConvention.FINAL_MINUS_INITIAL
            if storage
            else ThermodynamicSignConvention.INTO_SYSTEM_POSITIVE
        ),
        canonical_sign_multiplier=1,
        sign_transform_rule_id="identity-reference-sign",
        uncertainty=Decimal("0.01") if value is not None else None,
        matched_identity_observation_id=None,
        decomposition_model_id=None,
        status=status,
        reason_codes=() if value is not None else ("REFERENCE_TERM_UNOBSERVED",),
    )


def _ledger(case_index: int) -> tuple[ThermodynamicLedgerObservation, ...]:
    energy: tuple[Decimal | None, Decimal | None]
    identity_energy: tuple[Decimal | None, Decimal | None]
    entropy: tuple[Decimal | None, Decimal | None]
    if case_index == 0:
        energy = identity_energy = (Decimal("0"), Decimal("0"))
        entropy = (Decimal("0"), Decimal("0"))
    elif case_index in {1, 2}:
        energy = (Decimal("2"), Decimal("2"))
        identity_energy = (Decimal("1"), Decimal("1"))
        entropy = (Decimal("1.2"), Decimal("1"))
    elif case_index == 6:
        energy = (Decimal("2"), Decimal("2"))
        identity_energy = (Decimal("0"), Decimal("0"))
        entropy = (Decimal("0"), Decimal("0"))
    elif case_index == 7:
        energy = (None, Decimal("2"))
        identity_energy = (None, Decimal("0"))
        entropy = (None, None)
    elif case_index == 12:
        energy = identity_energy = (Decimal("5"), Decimal("5"))
        entropy = (Decimal("0"), Decimal("0"))
    elif case_index == 13:
        energy = (Decimal("2"), Decimal("2"))
        identity_energy = (Decimal("0"), Decimal("0"))
        entropy = (Decimal("0"), Decimal("0"))
    else:
        energy = identity_energy = (Decimal("0"), Decimal("0"))
        entropy = (None, None)
    observations: list[ThermodynamicLedgerObservation] = []
    for word_id, (storage_value, heat_value) in (
        ("cycle", energy),
        ("identity", identity_energy),
    ):
        observations.extend(
            (
                _ledger_observation(
                    case_index=case_index,
                    word_id=word_id,
                    term_id="boundary-heat",
                    role=ThermodynamicCoordinateRole.HEAT_TRANSFER,
                    value=heat_value,
                    unit="J",
                ),
                _ledger_observation(
                    case_index=case_index,
                    word_id=word_id,
                    term_id="stored-energy",
                    role=ThermodynamicCoordinateRole.STORED_ENERGY,
                    value=storage_value,
                    unit="J",
                ),
            )
        )
    observations.extend(
        (
            _ledger_observation(
                case_index=case_index,
                word_id="cycle",
                term_id="entropy-state",
                role=ThermodynamicCoordinateRole.STATE_ENTROPY,
                value=entropy[0],
                unit="J/K",
            ),
            _ledger_observation(
                case_index=case_index,
                word_id="cycle",
                term_id="entropy-transfer",
                role=ThermodynamicCoordinateRole.HEAT_ENTROPY_TRANSFER,
                value=entropy[1],
                unit="J/K",
            ),
        )
    )
    return tuple(sorted(observations, key=lambda value: value.observation_id))


def _diagnostics(case_index: int) -> tuple[NamedDecimal, ...]:
    values = {
        "associativity-residual": (Decimal("0") if case_index == 15 else Decimal("0.01"), "K"),
        "carryover-initial-spread": (
            Decimal("1.8") if case_index == 14 else Decimal("0"),
            "K",
        ),
        "context-conditioned-order-residual": (
            Decimal("0") if case_index in {4, 10} else Decimal("0.01"),
            "K",
        ),
        "return-bath-residual": (Decimal("0.3") if case_index == 6 else Decimal("0.01"), "K"),
        "return-state-residual": (Decimal("0.8") if case_index == 6 else Decimal("0.01"), "K"),
        "stochastic-kernel-residual": (
            Decimal("0.01") if case_index == 11 else Decimal("1"),
            "probability",
        ),
        "transport-residual": (Decimal("0") if case_index == 15 else Decimal("1"), "axis-native"),
    }
    return tuple(
        NamedDecimal(value_id=value_id, value=value, unit=unit)
        for value_id, (value, unit) in sorted(values.items())
    )


def _role_bindings(case_index: int) -> tuple[ThermodynamicRoleBinding, ...]:
    shared_event = "shared-command-transfer-event" if case_index == 13 else "command-event"
    return (
        ThermodynamicRoleBinding(
            binding_id="binding-boundary-heat",
            quantity_id="boundary-heat",
            physical_event_id=shared_event if case_index == 13 else "heat-event",
            role=ThermodynamicCoordinateRole.HEAT_TRANSFER,
        ),
        ThermodynamicRoleBinding(
            binding_id="binding-heater-command",
            quantity_id="heater-command",
            physical_event_id=shared_event,
            role=ThermodynamicCoordinateRole.ACTION_COMMAND,
        ),
        ThermodynamicRoleBinding(
            binding_id="binding-stored-energy",
            quantity_id="stored-energy",
            physical_event_id="storage-event",
            role=ThermodynamicCoordinateRole.STORED_ENERGY,
        ),
    )


def thermodynamic_truth_blind_inputs(
    spec: ThermodynamicResponseConformanceSpec | None = None,
) -> tuple[ThermodynamicTruthBlindInput, ...]:
    frozen = spec or reference_thermodynamic_response_conformance_spec()
    result = []
    for case_index, token in enumerate(frozen.case_tokens):
        contexts = ["bath-state-observed", "receiver-state-observed"]
        if case_index == 10:
            contexts.append("mode-state-observed")
        if case_index == 11:
            contexts.append("transition-counts-observed")
        result.append(
            ThermodynamicTruthBlindInput(
                input_id=token,
                panel=_panel(case_index, frozen),
                ledger_observations=_ledger(case_index),
                diagnostic_values=_diagnostics(case_index),
                observed_context_ids=tuple(sorted(contexts)),
                role_bindings=_role_bindings(case_index),
                housekeeping_model_id=(
                    "reference-housekeeping-model" if case_index == 12 else None
                ),
                system_boundary_complete=case_index != 7,
                chemical_terms_complete=case_index != 7,
                synchronization_valid=True,
                constitutive_assumptions_valid=case_index in {0, 1, 2},
                local_equilibrium_justified=case_index in {0, 1, 2},
            )
        )
    return tuple(result)


def _oracle(
    index: int,
    truth: ThermodynamicReferenceTruthKind,
    **changes: object,
) -> PrivilegedThermodynamicOracle:
    defaults: dict[str, object] = {
        "controlled_order": FiniteAxisDisposition.EQUIVALENT,
        "state_bath_return": ReturnQualificationStatus.RETURN_QUALIFIED,
        "energy_balance": BalanceClosureStatus.CLOSED,
        "entropy_eligibility": EntropyEligibilityStatus.LEDGER_PARTIAL,
        "action_quotient": FiniteAxisDisposition.MATERIAL,
        "temporal_cocycle": FiniteAxisDisposition.COCYCLIC,
        "equal_lag_stationarity": FiniteAxisDisposition.STATIONARY,
        "receiver_map": ReceiverMapDisposition.FAITHFUL_AT_TESTED_RESOLUTION,
        "cycle_interpretation_allowed": True,
        "context_augmentation": FiniteAxisDisposition.NOT_APPLICABLE,
        "representation": FiniteRepresentationKind.DETERMINISTIC_FINITE,
        "perturbation_exchange": FiniteAxisDisposition.EQUIVALENT,
        "denominator_stability": FiniteAxisDisposition.STATIONARY,
        "finite_coherence": FiniteAxisDisposition.EQUIVALENT,
        "transport": TransportDisposition.UNEVALUABLE,
        "double_count_rejected": True,
        "lie_claim_allowed": False,
        "reason_codes": (),
    }
    defaults.update(changes)
    token = THERMODYNAMIC_RESPONSE_CONFORMANCE_CASE_TOKENS[index]
    return PrivilegedThermodynamicOracle(
        oracle_id=f"privileged-oracle-{index:02d}",
        input_id=token,
        truth_kind=truth,
        **defaults,  # type: ignore[arg-type]
    )


def privileged_thermodynamic_oracles() -> tuple[PrivilegedThermodynamicOracle, ...]:
    return (
        _oracle(
            0,
            ThermodynamicReferenceTruthKind.REVERSIBLE_COMMUTING,
            energy_balance=BalanceClosureStatus.CLOSED,
            entropy_eligibility=EntropyEligibilityStatus.ELIGIBLE,
            perturbation_exchange=FiniteAxisDisposition.EQUIVALENT,
        ),
        _oracle(
            1,
            ThermodynamicReferenceTruthKind.RETURN_QUALIFIED_DISSIPATIVE_COMMUTING,
            energy_balance=BalanceClosureStatus.CLOSED,
            entropy_eligibility=EntropyEligibilityStatus.ELIGIBLE,
            perturbation_exchange=FiniteAxisDisposition.MATERIAL,
        ),
        _oracle(
            2,
            ThermodynamicReferenceTruthKind.RETURN_QUALIFIED_FINITE_ORDER,
            controlled_order=FiniteAxisDisposition.MATERIAL,
            energy_balance=BalanceClosureStatus.CLOSED,
            entropy_eligibility=EntropyEligibilityStatus.ELIGIBLE,
            perturbation_exchange=FiniteAxisDisposition.MATERIAL,
        ),
        _oracle(3, ThermodynamicReferenceTruthKind.TIMING_COUNTERFEIT),
        _oracle(
            4,
            ThermodynamicReferenceTruthKind.HIDDEN_BATH,
            controlled_order=FiniteAxisDisposition.MATERIAL,
            context_augmentation=FiniteAxisDisposition.EQUIVALENT,
        ),
        _oracle(
            5,
            ThermodynamicReferenceTruthKind.PROJECTED_CURVATURE,
            controlled_order=FiniteAxisDisposition.MATERIAL,
            receiver_map=ReceiverMapDisposition.PARTIALLY_COLLAPSING,
        ),
        _oracle(
            6,
            ThermodynamicReferenceTruthKind.NON_RETURN_COUNTERFEIT,
            state_bath_return=ReturnQualificationStatus.NON_RETURNING,
            energy_balance=BalanceClosureStatus.CLOSED,
            entropy_eligibility=EntropyEligibilityStatus.ENERGY_BALANCE_ONLY,
            cycle_interpretation_allowed=False,
            perturbation_exchange=FiniteAxisDisposition.MATERIAL,
        ),
        _oracle(
            7,
            ThermodynamicReferenceTruthKind.MISSING_BALANCE_TERM,
            state_bath_return=ReturnQualificationStatus.UNEVALUABLE,
            energy_balance=BalanceClosureStatus.TERM_UNOBSERVED,
            cycle_interpretation_allowed=False,
            perturbation_exchange=FiniteAxisDisposition.MATERIAL,
        ),
        _oracle(
            8,
            ThermodynamicReferenceTruthKind.COLLAPSED_ACTION_FIBRE,
            controlled_order=FiniteAxisDisposition.BELOW_RESOLUTION,
            action_quotient=FiniteAxisDisposition.BELOW_RESOLUTION,
            receiver_map=ReceiverMapDisposition.ACTION_BLIND,
        ),
        _oracle(
            9,
            ThermodynamicReferenceTruthKind.NONSTATIONARY_COCYCLE,
            equal_lag_stationarity=FiniteAxisDisposition.NONSTATIONARY,
        ),
        _oracle(
            10,
            ThermodynamicReferenceTruthKind.HYBRID_SWITCHING,
            controlled_order=FiniteAxisDisposition.MATERIAL,
            context_augmentation=FiniteAxisDisposition.EQUIVALENT,
            representation=FiniteRepresentationKind.HYBRID_FINITE,
        ),
        _oracle(
            11,
            ThermodynamicReferenceTruthKind.STOCHASTIC_FINITE_KERNEL,
            representation=FiniteRepresentationKind.STOCHASTIC_FINITE,
        ),
        _oracle(
            12,
            ThermodynamicReferenceTruthKind.HOUSEKEEPING_COUNTERFEIT,
            energy_balance=BalanceClosureStatus.CLOSED,
            entropy_eligibility=EntropyEligibilityStatus.ENERGY_BALANCE_ONLY,
            perturbation_exchange=FiniteAxisDisposition.EQUIVALENT,
        ),
        _oracle(
            13,
            ThermodynamicReferenceTruthKind.ACTION_EXCHANGE_DOUBLE_COUNT,
            energy_balance=BalanceClosureStatus.CLOSED,
            entropy_eligibility=EntropyEligibilityStatus.ENERGY_BALANCE_ONLY,
            perturbation_exchange=FiniteAxisDisposition.MATERIAL,
        ),
        _oracle(
            14,
            ThermodynamicReferenceTruthKind.AGEING_CARRYOVER_COUNTERFEIT,
            controlled_order=FiniteAxisDisposition.UNEVALUABLE,
            state_bath_return=ReturnQualificationStatus.UNEVALUABLE,
            cycle_interpretation_allowed=False,
            denominator_stability=FiniteAxisDisposition.NONSTATIONARY,
            finite_coherence=FiniteAxisDisposition.UNEVALUABLE,
        ),
        _oracle(
            15,
            ThermodynamicReferenceTruthKind.FINITE_COHERENCE_AND_TRANSPORT,
            receiver_map=ReceiverMapDisposition.PARTIALLY_COLLAPSING,
            transport=TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION,
        ),
    )


__all__ = [
    "FiniteRepresentationKind",
    "PrivilegedThermodynamicOracle",
    "ThermodynamicReferenceTruthKind",
    "ThermodynamicTruthBlindInput",
    "privileged_thermodynamic_oracles",
    "thermodynamic_truth_blind_inputs",
]
