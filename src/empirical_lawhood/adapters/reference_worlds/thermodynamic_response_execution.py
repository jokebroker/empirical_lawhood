"""Truth-blind execution and separate privileged scoring for thermodynamic response conformance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.thermodynamic_response import FiniteTrajectoryPanel, nonapplicability_witness_for
from empirical_lawhood.adapters.methods.thermodynamic_response_analysis import (
    EntropyBalanceResult,
    ThermodynamicResponseMethodConfig,
    ThermodynamicResponseMethodInput,
    ThermodynamicResponseMethodResult,
    TransportAxisComparison,
    assess_entropy_balance,
    build_absolute_energy_balance,
    build_structural_transport_witness,
    compose_thermodynamic_response_signature,
    controlled_order_contrast,
    cycle_exchange,
    evaluate_finite_contrast,
    generalized_information_geometry,
    identity_coherence,
    qualify_grouped_return,
    receiver_naturality,
    temporal_propagation,
    validate_method_contract,
)
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.response_algebra import ActionStage
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import RelationalIdentity
from empirical_lawhood.kernel.thermodynamic_response import ActionDelivery, ActionInterval, ActionStageObservation, BalanceClosureAssessment, BalanceClosureStatus, ChronologyConvention, CompositeSignatureAxis, DeliveredDoseComponent, DeliveryQualification, EnergyTransferMode, EntropyEligibilityStatus, EntropyTermRole, EntropyTermSpec, ExchangeEstimandKind, FiniteAxisDisposition, FiniteResponseSignature, FiniteWordMode, FiniteWord, OntologyCompatibilityStatus, OntologyCompatibilityWitness, PartialCompositionStatus, PreparedResponseObject, ReceiverMapDisposition, ReturnQualificationStatus, StoredEnergyTermSpec, ThermodynamicClaimCeiling, ThermodynamicCoordinateRole, ThermodynamicObservationStatus, ThermodynamicSignConvention, ThermodynamicTermSpec, TransportDisposition
from empirical_lawhood.kernel.time import HorizonSpec
from empirical_lawhood.planning.thermodynamic_response import (
    FiniteContrastSpec,
    FiniteWordFamilySpec,
    ThermodynamicResponseConformanceSpec,
    WordRole,
    WordRoleMap,
    reference_thermodynamic_response_conformance_spec,
)

from .thermodynamic_response import (
    FiniteRepresentationKind,
    PrivilegedThermodynamicOracle,
    privileged_thermodynamic_oracles,
    thermodynamic_truth_blind_inputs,
)


@dataclass(frozen=True, slots=True)
class ThermodynamicReferenceInvocation:
    method_input: ThermodynamicResponseMethodInput
    method_config: ThermodynamicResponseMethodConfig


@dataclass(frozen=True, slots=True)
class ThermodynamicReferenceMethodReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/thermodynamic-reference-method-readout'

    readout_id: str
    input_id: str
    method_result: ThermodynamicResponseMethodResult
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
        validate_stable_id(self.readout_id, field_name="readout_id")
        validate_stable_id(self.input_id, field_name="input_id")
        if self.method_result.input_id != self.input_id:
            raise ValueError("reference readout differs from its method-result input")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.lie_claim_allowed:
            raise ValueError("finite conformance readout cannot make an infinitesimal Lie claim")


@dataclass(frozen=True, slots=True)
class ThermodynamicReferenceCaseScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/thermodynamic-reference-case-score'

    score_id: str
    input_id: str
    passed: bool
    mismatched_field_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.input_id, field_name="input_id")
        require_sorted_unique_strings(
            self.mismatched_field_ids,
            field_name="mismatched_field_ids",
        )
        if self.passed == bool(self.mismatched_field_ids):
            raise ValueError("reference case pass status differs from its mismatches")


@dataclass(frozen=True, slots=True)
class ThermodynamicReferenceConformanceScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/thermodynamic-reference-conformance-score'

    score_id: str
    conformance_id: str
    case_scores: tuple[ThermodynamicReferenceCaseScore, ...]
    passed_cases: int
    failed_cases: int
    false_noncommutativity: int
    false_entropy_production: int
    false_cycle_qualification: int
    gate_passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.conformance_id, field_name="conformance_id")
        require_sorted_unique_ids(
            self.case_scores,
            attribute="score_id",
            field_name="case_scores",
        )
        if self.passed_cases + self.failed_cases != len(self.case_scores):
            raise ValueError("conformance case counts differ from score rows")
        if self.passed_cases != sum(value.passed for value in self.case_scores):
            raise ValueError("conformance passing-case count differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_gate = self.failed_cases == 0 and not (
            self.false_noncommutativity
            or self.false_entropy_production
            or self.false_cycle_qualification
        )
        if self.gate_passed is not expected_gate:
            raise ValueError("conformance gate differs from failures and false promotions")
        if self.gate_passed and self.reason_codes:
            raise ValueError("passing conformance score cannot carry failure reasons")
        if not self.gate_passed and not self.reason_codes:
            raise ValueError("failed conformance score requires reasons")


def _prepared_object(index: int) -> PreparedResponseObject:
    return PreparedResponseObject(
        object_id=f"reference-object-{index:02d}",
        relation_id="thermodynamic-reference-relation",
        preparation_id="reference-preparation",
        denominator_id="reference-denominator",
        bath_view_id="reference-bath-view",
        retained_history_id="reference-history",
        clock_id="reference-clock",
        clock_coordinate=Decimal(index),
        support_id="reference-support",
        valid=True,
        reason_codes=(),
    )


def _delivery(
    *,
    word_id: str,
    delivery_index: int,
    letter_id: str,
    source_index: int,
    target_index: int,
    sign: int = 1,
    overlap: bool = False,
) -> ActionDelivery:
    start = Decimal("0.5") if overlap else Decimal(source_index) + Decimal("0.1")
    end = Decimal("2.5") if overlap else Decimal(target_index) - Decimal("0.1")
    if overlap and delivery_index == 0:
        end = Decimal("2")
    elif overlap:
        start = Decimal("1")
    delivery_id = f"delivery-{word_id}-{delivery_index:02d}"
    stages = tuple(
        sorted(
            (
                ActionStageObservation(
                    stage=stage,
                    quantity_id=f"{letter_id}-{stage.value.lower()}-quantity",
                    value=Decimal(sign),
                    native_unit=(
                        "V"
                        if stage is ActionStage.REQUESTED
                        else ("A" if stage is ActionStage.APPLIED else "W")
                    ),
                    clock_id=f"{stage.value.lower()}-clock",
                    clock_coordinate=start,
                    uncertainty=Decimal("0.001"),
                    observation_id=f"{delivery_id}-{stage.value.lower()}",
                )
                for stage in ActionStage
            ),
            key=lambda value: value.observation_id,
        )
    )
    interval = ActionInterval(
        start=start,
        end=end,
        clock_id="reference-clock",
        time_unit="s",
    )
    return ActionDelivery(
        delivery_id=delivery_id,
        letter_id=letter_id,
        port_id=f"port-{letter_id}",
        stages=stages,
        planned_interval=interval,
        realized_interval=interval,
        pulse_shape_id="rectangular-reference-pulse",
        delivered_dose=(
            DeliveredDoseComponent(
                quantity_id=f"dose-{letter_id}",
                value=Decimal(sign),
                native_unit="J",
                uncertainty=Decimal("0.001"),
            ),
        ),
        sign=sign,
        slew_rate=None,
        slew_rate_unit=None,
        preparation_id="reference-preparation",
        causal_prefix_id=f"reference-object-{source_index:02d}",
        support_id="reference-delivery-support",
        interlock_active=False,
        clipped=False,
        saturated=False,
        calibration_id="reference-delivery-calibration",
        actuator_id=f"reference-actuator-{letter_id}",
        synchronization_id="reference-clock-map",
        qualification=DeliveryQualification.QUALIFIED,
        reason_codes=(),
    )


def _word(
    word_id: str,
    letters: tuple[tuple[str, int], ...],
    *,
    simultaneous: bool = False,
) -> FiniteWord:
    prefix_indices: tuple[int, ...]
    if not letters:
        prefix_indices = (0, 3)
        deliveries: tuple[ActionDelivery, ...] = ()
        mode = FiniteWordMode.IDENTITY
    elif simultaneous:
        prefix_indices = (0, 3)
        deliveries = tuple(
            _delivery(
                word_id=word_id,
                delivery_index=index,
                letter_id=letter,
                source_index=0,
                target_index=3,
                sign=sign,
                overlap=True,
            )
            for index, (letter, sign) in enumerate(letters)
        )
        mode = FiniteWordMode.SIMULTANEOUS
    else:
        prefix_indices = (
            (0, 3) if len(letters) == 1 else ((0, 1, 3) if len(letters) == 2 else (0, 1, 2, 3))
        )
        deliveries = tuple(
            _delivery(
                word_id=word_id,
                delivery_index=index,
                letter_id=letter,
                source_index=prefix_indices[index],
                target_index=prefix_indices[index + 1],
                sign=sign,
            )
            for index, (letter, sign) in enumerate(letters)
        )
        mode = FiniteWordMode.SEQUENTIAL
    return FiniteWord(
        word_id=word_id,
        mode=mode,
        source_object_id="reference-object-00",
        target_object_id="reference-object-03",
        deliveries=deliveries,
        prefix_object_ids=tuple(f"reference-object-{value:02d}" for value in prefix_indices),
        chronology_convention=ChronologyConvention.LEFT_TO_RIGHT_DELIVERY,
        composition_status=PartialCompositionStatus.DEFINED,
        overlap_tolerance=Decimal("0.01"),
        overlap_tolerance_unit="s",
        elapsed_time=Decimal("3"),
        elapsed_time_unit="s",
        reason_codes=(),
    )


def reference_finite_word_family() -> FiniteWordFamilySpec:
    words = (
        _word("a", (("a", 1),)),
        _word("ab", (("a", 1), ("b", 1))),
        _word("b", (("b", 1),)),
        _word("ba", (("b", 1), ("a", 1))),
        _word("cycle", (("a", 1), ("a-return", -1))),
        _word("identity", ()),
        _word("longer-aba", (("a", 1), ("b", 1), ("a", 1))),
        _word("overlap-ab", (("a", 1), ("b", 1)), simultaneous=True),
        _word("repeat-a", (("a", 1), ("a", 1))),
        _word("timing-ab", (("a", 1), ("b", 1))),
        _word("timing-ba", (("b", 1), ("a", 1))),
    )
    assignments = tuple(
        sorted(
            (
                ("a", (WordRole.SINGLE,)),
                ("ab", (WordRole.ORDER,)),
                ("b", (WordRole.SINGLE,)),
                ("ba", (WordRole.ORDER,)),
                ("cycle", (WordRole.CYCLE,)),
                ("identity", (WordRole.IDENTITY,)),
                ("longer-aba", (WordRole.LONGER_WORD,)),
                ("overlap-ab", (WordRole.OVERLAP,)),
                ("repeat-a", (WordRole.REPEAT,)),
                ("timing-ab", (WordRole.SHAM,)),
                ("timing-ba", (WordRole.SHAM,)),
            )
        )
    )
    return FiniteWordFamilySpec(
        family_id="thermodynamic-reference-word-family",
        relation_id="thermodynamic-reference-relation",
        objects=tuple(_prepared_object(index) for index in range(4)),
        words=words,
        declared_action_pairs=(("a", "b"),),
        role_map=WordRoleMap(
            role_map_id="thermodynamic-reference-word-roles",
            family_id="thermodynamic-reference-word-family",
            assignments=assignments,
        ),
        receiver_ids=("full-reference-receiver",),
        horizon_ids=("three-second-horizon",),
    )


def _ontology() -> OntologyCompatibilityWitness:
    return OntologyCompatibilityWitness(
        witness_id="thermodynamic-reference-ontology",
        relation=RelationalIdentity(
            relation_id="thermodynamic-reference-relation",
            denominator_quantity_ids=("bath-state", "prepared-denominator"),
            history_quantity_ids=("denominator-history",),
            memoryless=False,
            action_quantity_ids=("action-a", "action-b"),
            receiver_quantity_ids=("receiver-x", "receiver-y"),
            horizon=HorizonSpec(
                horizon_id="three-second-horizon",
                clock_id="reference-clock",
                duration=Decimal("3"),
                time_unit="s",
            ),
        ),
        bath_denominator_quantity_ids=("bath-state",),
        bath_history_quantity_ids=("denominator-history",),
        thermodynamic_annotation_quantity_ids=(
            "boundary-heat",
            "state-entropy",
            "stored-energy",
        ),
        status=OntologyCompatibilityStatus.COMPATIBLE,
        reason_codes=(),
    )


def _boundary_terms() -> tuple[ThermodynamicTermSpec, ...]:
    return (
        ThermodynamicTermSpec(
            term_id="boundary-heat",
            system_id="thermodynamic-reference-system",
            quantity_id="boundary-heat",
            port_owner_id="thermodynamic-reference-system",
            port_id="boundary-heat-port",
            interface_id=None,
            physical_event_id="boundary-heat-event",
            transfer_mode=EnergyTransferMode.HEAT,
            native_unit="J",
            source_native_positive_direction="into-reference-system",
            canonical_sign_convention=ThermodynamicSignConvention.INTO_SYSTEM_POSITIVE,
            canonical_sign_multiplier=1,
            integration_method_id="exact-reference-integration",
            calibration_id="exact-reference-calibration",
            uncertainty_contract_id="reference-energy-uncertainty",
            validity_contract_id="reference-energy-validity",
        ),
    )


def _stored_energy_terms() -> tuple[StoredEnergyTermSpec, ...]:
    return (
        StoredEnergyTermSpec(
            term_id="stored-energy",
            system_id="thermodynamic-reference-system",
            quantity_id="stored-energy",
            native_unit="J",
            constitutive_model_id="exact-reference-storage-model",
            calibration_id="exact-reference-calibration",
            uncertainty_contract_id="reference-energy-uncertainty",
            discrepancy_contract_id="zero-reference-discrepancy",
            directly_observed=True,
            sign_convention=ThermodynamicSignConvention.FINAL_MINUS_INITIAL,
        ),
    )


def _entropy_terms() -> tuple[EntropyTermSpec, ...]:
    return (
        EntropyTermSpec(
            term_id="entropy-state",
            system_id="thermodynamic-reference-system",
            role=EntropyTermRole.STATE_ENTROPY,
            entropy_quantity_id="state-entropy",
            heat_term_id=None,
            matter_term_id=None,
            boundary_temperature_quantity_id=None,
            composition_quantity_ids=(),
            constitutive_model_ids=("exact-reference-entropy-model",),
            assumption_ids=("reference-local-equilibrium",),
            uncertainty_contract_id="reference-entropy-uncertainty",
            validity_contract_id="reference-entropy-validity",
            synchronization_id="reference-clock-map",
            local_equilibrium_assumed=True,
        ),
        EntropyTermSpec(
            term_id="entropy-transfer",
            system_id="thermodynamic-reference-system",
            role=EntropyTermRole.HEAT_OVER_BOUNDARY_TEMPERATURE,
            entropy_quantity_id="heat-entropy-transfer",
            heat_term_id="boundary-heat",
            matter_term_id=None,
            boundary_temperature_quantity_id="boundary-temperature",
            composition_quantity_ids=(),
            constitutive_model_ids=(),
            assumption_ids=("reference-boundary-temperature",),
            uncertainty_contract_id="reference-entropy-uncertainty",
            validity_contract_id="reference-entropy-validity",
            synchronization_id="reference-clock-map",
            local_equilibrium_assumed=True,
        ),
    )


def _contrast(
    contrast_id: str,
    coefficients: tuple[tuple[str, Decimal], ...],
    panel: FiniteTrajectoryPanel,
) -> FiniteContrastSpec:
    return FiniteContrastSpec(
        contrast_id=contrast_id,
        family_id="thermodynamic-reference-word-family",
        relation_id="thermodynamic-reference-relation",
        estimand_id=contrast_id,
        receiver_id=panel.receiver_id,
        horizon_id=panel.horizon_id,
        coefficients=coefficients,
        coordinate_ids=panel.coordinate_ids,
        response_units=panel.native_units,
        zero_sum_required=True,
    )


def truth_blind_thermodynamic_invocations(
    spec: ThermodynamicResponseConformanceSpec | None = None,
) -> tuple[ThermodynamicReferenceInvocation, ...]:
    frozen = spec or reference_thermodynamic_response_conformance_spec()
    family = reference_finite_word_family()
    result = []
    thresholds = {value.value_id: value for value in frozen.decision_thresholds}
    for source in thermodynamic_truth_blind_inputs(frozen):
        contrasts = (
            _contrast(
                "raw-order",
                (("ab", Decimal("1")), ("ba", Decimal("-1"))),
                source.panel,
            ),
            _contrast(
                "timing-bath-null",
                (("timing-ab", Decimal("1")), ("timing-ba", Decimal("-1"))),
                source.panel,
            ),
        )
        method_input = ThermodynamicResponseMethodInput(
            input_id=source.input_id,
            ontology_witness=_ontology(),
            family=family,
            panel=source.panel,
            contrasts=contrasts,
            role_bindings=source.role_bindings,
            boundary_terms=_boundary_terms(),
            stored_energy_terms=_stored_energy_terms(),
            entropy_terms=_entropy_terms(),
            ledger_observations=source.ledger_observations,
            scalar_dose_projection_witness=nonapplicability_witness_for(
                witness_id=f"scalar-dose-nonapplicability-{source.input_id}",
                family=family,
                panel=source.panel,
            ),
        )
        config = ThermodynamicResponseMethodConfig(
            config_id=f"config-{source.input_id}",
            method_version="1.0.0",
            identity_word_id="identity",
            minimum_independent_units=frozen.independent_units_per_case,
            numerical_floor=frozen.numerical_floors,
            affine_ridge=Decimal("0.000000000001"),
            cocycle_tolerance=thresholds["cocycle"].value,
            stationarity_tolerance=thresholds["stationarity"].value,
            balance_tolerance=thresholds["balance-closure"].value,
            balance_unit=thresholds["balance-closure"].unit,
            system_boundary_complete=source.system_boundary_complete,
            chemical_terms_complete=source.chemical_terms_complete,
            synchronization_valid=source.synchronization_valid,
            constitutive_assumptions_valid=source.constitutive_assumptions_valid,
            local_equilibrium_justified=source.local_equilibrium_justified,
            prohibited_component_ids=("llm", "rl"),
        )
        validate_method_contract(method_input, config)
        result.append(ThermodynamicReferenceInvocation(method_input, config))
    return tuple(result)


def _panel_array(panel: FiniteTrajectoryPanel) -> np.ndarray:
    return np.asarray(panel.values, dtype=object).reshape(
        len(panel.independent_unit_ids),
        len(panel.word_ids),
        len(panel.times),
        len(panel.coordinate_ids),
    )


def _scaled_max(values: np.ndarray, floors: np.ndarray) -> float:
    return float(np.max(np.abs(values / floors[None, :])))


def _projected_panel(
    panel: FiniteTrajectoryPanel,
    coordinate_indices: tuple[int, ...],
) -> FiniteTrajectoryPanel:
    raw = _panel_array(panel)
    projected = raw[..., coordinate_indices]
    return replace(
        panel,
        panel_id=f"projected-{panel.panel_id}",
        state_view_id="coarse-reference-state",
        receiver_id="coarse-reference-receiver",
        coordinate_ids=tuple(panel.coordinate_ids[index] for index in coordinate_indices),
        native_units=tuple(panel.native_units[index] for index in coordinate_indices),
        values=tuple(projected.reshape(-1)),
    )


def _context_augmentation(
    panel: FiniteTrajectoryPanel, tolerance: float
) -> tuple[FiniteAxisDisposition, float]:
    raw = np.asarray(_panel_array(panel), dtype=np.float64)
    word_index = {value: index for index, value in enumerate(panel.word_ids)}
    coordinate_index = {value: index for index, value in enumerate(panel.coordinate_ids)}
    order = raw[:, word_index["ab"]] - raw[:, word_index["ba"]]
    target = order[..., coordinate_index["receiver-x"]].reshape(-1, 1)
    context = order[
        ...,
        (coordinate_index["bath-state"], coordinate_index["mode-state"]),
    ].reshape(-1, 2)
    operator = np.linalg.pinv(context) @ target
    residual = float(np.sqrt(np.mean((target - context @ operator) ** 2)))
    return (
        FiniteAxisDisposition.EQUIVALENT
        if residual <= tolerance
        else FiniteAxisDisposition.MATERIAL,
        residual,
    )


def _balance_and_entropy(
    method_input: ThermodynamicResponseMethodInput,
    config: ThermodynamicResponseMethodConfig,
) -> tuple[BalanceClosureAssessment, EntropyBalanceResult]:
    cycle_energy = tuple(
        value
        for value in method_input.ledger_observations
        if value.word_id == "cycle" and value.term_id in {"boundary-heat", "stored-energy"}
    )
    balance = build_absolute_energy_balance(
        assessment_id=f"balance-{method_input.input_id}",
        expected_storage_term_ids=("stored-energy",),
        expected_boundary_term_ids=("boundary-heat",),
        observations=cycle_energy,
        closure_tolerance=config.balance_tolerance,
        energy_unit=config.balance_unit,
        entropy_eligibility=EntropyEligibilityStatus.ENERGY_BALANCE_ONLY,
    )
    entropy_observations = tuple(
        value
        for value in method_input.ledger_observations
        if value.word_id == "cycle" and value.term_id.startswith("entropy-")
    )
    entropy = assess_entropy_balance(
        entropy_terms=method_input.entropy_terms,
        observations=entropy_observations,
        absolute_energy_balance=balance,
        system_boundary_complete=config.system_boundary_complete,
        chemical_terms_complete=config.chemical_terms_complete,
        synchronization_valid=config.synchronization_valid,
        constitutive_assumptions_valid=config.constitutive_assumptions_valid,
        local_equilibrium_justified=config.local_equilibrium_justified,
    )
    if entropy.status is EntropyEligibilityStatus.ELIGIBLE:
        balance = replace(
            balance,
            entropy_eligibility=EntropyEligibilityStatus.ELIGIBLE,
            claim_ceiling=ThermodynamicClaimCeiling.ENTROPY_PRODUCTION_ELIGIBLE,
        )
    return balance, entropy


def identify_truth_blind_thermodynamic_response(
    invocation: ThermodynamicReferenceInvocation,
    spec: ThermodynamicResponseConformanceSpec | None = None,
) -> ThermodynamicReferenceMethodReadout:
    frozen = spec or reference_thermodynamic_response_conformance_spec()
    method_input = invocation.method_input
    config = invocation.method_config
    validate_method_contract(method_input, config)
    panel = method_input.panel
    floors = np.asarray([float(value.value) for value in frozen.numerical_floors], dtype=np.float64)
    threshold = float(
        next(
            value.value
            for value in frozen.decision_thresholds
            if value.value_id == "controlled-order"
        )
    )
    contrast_by_id = {value.contrast_id: value for value in method_input.contrasts}
    raw_order = evaluate_finite_contrast(panel, contrast_by_id["raw-order"])
    timing_null = evaluate_finite_contrast(panel, contrast_by_id["timing-bath-null"])
    controlled = controlled_order_contrast(raw_order, timing_null)
    raw_metric = _scaled_max(raw_order.values[:, -1], floors)
    timing_metric = _scaled_max(timing_null.values[:, -1], floors)
    controlled_metric = _scaled_max(controlled.values[:, -1], floors)

    raw = np.asarray(_panel_array(panel), dtype=np.float64)
    word_index = {value: index for index, value in enumerate(panel.word_ids)}
    identity_index = word_index[panel.identity_word_id]
    starts = raw[:, :, 0]
    initial_spread = float(
        np.max(np.abs((starts - starts[:, identity_index : identity_index + 1]) / floors))
    )
    denominator_stability = (
        FiniteAxisDisposition.STATIONARY
        if initial_spread <= 1
        else FiniteAxisDisposition.NONSTATIONARY
    )
    constituent = np.stack(
        (
            raw[:, word_index["a"], -1] - raw[:, identity_index, -1],
            raw[:, word_index["b"], -1] - raw[:, identity_index, -1],
        )
    )
    constituent_metric = float(np.max(np.abs(constituent / floors)))
    constituent_disposition = (
        FiniteAxisDisposition.MATERIAL
        if constituent_metric > 1
        else FiniteAxisDisposition.BELOW_RESOLUTION
    )
    if denominator_stability is FiniteAxisDisposition.NONSTATIONARY:
        controlled_disposition = FiniteAxisDisposition.UNEVALUABLE
    elif constituent_disposition is FiniteAxisDisposition.BELOW_RESOLUTION:
        controlled_disposition = FiniteAxisDisposition.BELOW_RESOLUTION
    elif controlled_metric <= threshold:
        controlled_disposition = FiniteAxisDisposition.EQUIVALENT
    else:
        controlled_disposition = FiniteAxisDisposition.MATERIAL

    numerical_floor = np.tile(floors, (len(panel.times), 1))
    geometry = generalized_information_geometry(
        panel,
        identity_word_id=panel.identity_word_id,
        numerical_floor=numerical_floor,
    )
    action_rank = max(geometry.floor_certified_ranks)
    action_quotient = (
        FiniteAxisDisposition.BELOW_RESOLUTION
        if constituent_disposition is FiniteAxisDisposition.BELOW_RESOLUTION
        else FiniteAxisDisposition.MATERIAL
    )
    action_kernel_ids = tuple(
        sorted(
            word_id
            for word_id in panel.word_ids
            if word_id != panel.identity_word_id
            and float(
                np.max(
                    np.abs((raw[:, word_index[word_id], -1] - raw[:, identity_index, -1]) / floors)
                )
            )
            <= 1
        )
    )
    temporal = temporal_propagation(
        panel,
        identity_word_id=panel.identity_word_id,
        selected_time_indices=tuple(range(len(panel.times))),
        ridge=float(config.affine_ridge),
        cocycle_tolerance=float(config.cocycle_tolerance),
        stationarity_tolerance=float(config.stationarity_tolerance),
    )
    identity_result = identity_coherence(
        panel,
        identity_word_id=panel.identity_word_id,
        ridge=float(config.affine_ridge),
    )
    coherence_metric = float(np.max(np.abs(identity_result.residuals / floors)))
    finite_coherence = (
        FiniteAxisDisposition.UNEVALUABLE
        if denominator_stability is FiniteAxisDisposition.NONSTATIONARY
        else (
            FiniteAxisDisposition.EQUIVALENT
            if coherence_metric <= 1
            else FiniteAxisDisposition.MATERIAL
        )
    )

    if panel.state_view_id == "full-and-coarse-reference-state":
        x_index = panel.coordinate_ids.index("receiver-x")
        projected = _projected_panel(panel, (x_index,))
        receiver_result = receiver_naturality(
            panel,
            projected,
            np.eye(len(panel.coordinate_ids), dtype=np.float64)[(x_index,), :],
            identity_word_id=panel.identity_word_id,
            full_floor=numerical_floor,
            projected_floor=numerical_floor[:, (x_index,)],
            naturality_tolerance=1e-6,
        )
        receiver_map = receiver_result.disposition
    else:
        receiver_map = ReceiverMapDisposition.FAITHFUL_AT_TESTED_RESOLUTION
    if action_quotient is FiniteAxisDisposition.BELOW_RESOLUTION:
        receiver_map = ReceiverMapDisposition.ACTION_BLIND

    if panel.state_view_id == "context-augmented-reference-state":
        context_augmentation, context_residual = _context_augmentation(panel, 1e-6)
    else:
        context_augmentation = FiniteAxisDisposition.NOT_APPLICABLE
        context_residual = 0.0
    if panel.numerical_view_id == "stochastic-transition-reference-view":
        representation = FiniteRepresentationKind.STOCHASTIC_FINITE
    elif (
        panel.state_view_id == "context-augmented-reference-state"
        and float(np.max(np.abs(raw[..., panel.coordinate_ids.index("mode-state")]))) > 0
    ):
        representation = FiniteRepresentationKind.HYBRID_FINITE
    else:
        representation = FiniteRepresentationKind.DETERMINISTIC_FINITE

    balance, entropy = _balance_and_entropy(method_input, config)
    cycle_final = raw[0, word_index["cycle"], -1]
    identity_final = raw[0, identity_index, -1]
    residual = cycle_final - identity_final
    state_ids = ("receiver-x", "receiver-y")
    bath_ids = ("bath-state", "mode-state")
    coordinate_index = {value: index for index, value in enumerate(panel.coordinate_ids)}
    state_residuals = tuple(
        NamedDecimal(
            value_id=value,
            value=Decimal(str(residual[coordinate_index[value]])),
            unit=panel.native_units[coordinate_index[value]],
        )
        for value in state_ids
    )
    bath_residuals = tuple(
        NamedDecimal(
            value_id=value,
            value=Decimal(str(residual[coordinate_index[value]])),
            unit=panel.native_units[coordinate_index[value]],
        )
        for value in bath_ids
    )
    tolerances = tuple(
        NamedDecimal(
            value_id=value,
            value=Decimal("0.1"),
            unit=panel.native_units[coordinate_index[value]],
        )
        for value in tuple(sorted((*state_ids, *bath_ids)))
    )
    uncertainties = tuple(
        NamedDecimal(
            value_id=value,
            value=Decimal("0.01"),
            unit=panel.native_units[coordinate_index[value]],
        )
        for value in tuple(sorted((*state_ids, *bath_ids)))
    )
    qualification = qualify_grouped_return(
        qualification_id=f"return-{method_input.input_id}",
        system_id=panel.system_id,
        independent_unit_id="reference-unit-00",
        episode_id=f"episode-{method_input.input_id[-2:]}-reference-unit-00-cycle",
        word_id="cycle",
        recovery_horizon_id="three-second-horizon",
        recovery_horizon_predeclared=True,
        state_residuals=state_residuals,
        bath_residuals=bath_residuals,
        component_tolerances=tolerances,
        component_uncertainties=uncertainties,
        support_valid=True,
        delivery_valid=True,
        interlock_clear=True,
        data_complete=denominator_stability is FiniteAxisDisposition.STATIONARY,
        balance_assessment=balance,
        unobserved_coordinate_ids=(),
    )
    cycle_heat = tuple(
        value
        for value in method_input.ledger_observations
        if value.word_id == "cycle"
        and value.term_id == "boundary-heat"
        and value.status is ThermodynamicObservationStatus.OBSERVED
        and value.estimand_kind is ExchangeEstimandKind.TOTAL_EXCHANGE
    )
    identity_heat = tuple(
        value
        for value in method_input.ledger_observations
        if value.word_id == "identity"
        and value.term_id == "boundary-heat"
        and value.status is ThermodynamicObservationStatus.OBSERVED
        and value.estimand_kind is ExchangeEstimandKind.TOTAL_EXCHANGE
    )
    if cycle_heat and identity_heat:
        exchange = cycle_exchange(
            identity_word_id="identity",
            word_observations=cycle_heat,
            identity_observations=identity_heat,
        )
        exchange_value = exchange.identity_differenced_observations[0].canonical_value
        perturbation_exchange = (
            FiniteAxisDisposition.MATERIAL
            if exchange_value is not None and abs(exchange_value) > config.balance_tolerance
            else FiniteAxisDisposition.EQUIVALENT
        )
        loop_exchange_ids = tuple(value.observation_id for value in cycle_heat)
    else:
        perturbation_exchange = FiniteAxisDisposition.UNEVALUABLE
        loop_exchange_ids = ()

    transport_residual = 0.0
    if panel.numerical_view_id == "paired-transport-reference-view":
        transport_witness = build_structural_transport_witness(
            witness_id=f"transport-{method_input.input_id}",
            source_signature_id="source-reference-signature",
            target_signature_id="target-reference-signature",
            source_system_id="source-reference-medium",
            target_system_id="target-reference-medium",
            source_world_kind=panel.world_kind,
            target_world_kind=panel.world_kind,
            action_role_map=(("source-action", "target-action"),),
            horizon_map=(("source-horizon", "target-horizon"),),
            receiver_map=(("source-receiver", "target-receiver"),),
            thermodynamic_role_map=(("source-heat", "target-heat"),),
            common_support_object_map=(("source-object", "target-object"),),
            axis_comparisons=(
                TransportAxisComparison(
                    axis=CompositeSignatureAxis.RECEIVER_NATURALITY_AND_FAITHFULNESS,
                    source_then_map=raw[:, :, -1, 2:4],
                    map_then_target=raw[:, :, -1, 2:4].copy(),
                    uncertainty=Decimal("0"),
                    tolerance=Decimal("0.000001"),
                    native_unit="floor-scaled-receiver",
                ),
            ),
            faithfulness_limit_ids=("coarse-receiver-collapses-y",),
            decisive_counterexample_ids=(),
        )
        transport_status = transport_witness.status
    else:
        transport_status = TransportDisposition.UNEVALUABLE

    double_count_rejected = (
        len(
            {
                value.quantity_id
                for value in method_input.role_bindings
                if value.role
                in {
                    ThermodynamicCoordinateRole.ACTION_COMMAND,
                    ThermodynamicCoordinateRole.HEAT_TRANSFER,
                }
            }
        )
        == 2
        and sum(
            value.word_id == "cycle" and value.term_id == "boundary-heat"
            for value in method_input.ledger_observations
        )
        == 1
    )

    metrics = tuple(
        sorted(
            (
                NamedDecimal(
                    "action-materiality-max", Decimal(str(constituent_metric)), "floor-scaled-max"
                ),
                NamedDecimal(
                    "carryover-initial-spread", Decimal(str(initial_spread)), "floor-scaled-max"
                ),
                NamedDecimal(
                    "coherence-residual-max", Decimal(str(coherence_metric)), "floor-scaled-max"
                ),
                NamedDecimal("context-residual", Decimal(str(context_residual)), "K"),
                NamedDecimal(
                    "controlled-order-max", Decimal(str(controlled_metric)), "floor-scaled-max"
                ),
                NamedDecimal("raw-order-max", Decimal(str(raw_metric)), "floor-scaled-max"),
                NamedDecimal("timing-null-max", Decimal(str(timing_metric)), "floor-scaled-max"),
                NamedDecimal("transport-residual", Decimal(str(transport_residual)), "axis-native"),
            ),
            key=lambda value: value.value_id,
        )
    )
    reasons = set()
    if controlled_disposition is FiniteAxisDisposition.UNEVALUABLE:
        reasons.add("DENOMINATOR_DRIFT_BARS_WITHIN_UNIT_CONTRAST")
    finite_signature = FiniteResponseSignature(
        signature_id=f"finite-signature-{method_input.input_id}",
        relation_id=panel.relation_id,
        delivery_qualification=DeliveryQualification.QUALIFIED,
        constituent_port_materiality=constituent_disposition,
        action_image_rank=action_rank,
        action_kernel_word_ids=action_kernel_ids,
        simultaneous_defect=FiniteAxisDisposition.NOT_APPLICABLE,
        controlled_sequential_defect=controlled_disposition,
        repeat_and_longer_word_curvature=FiniteAxisDisposition.EQUIVALENT,
        temporal_cocycle=temporal.cocycle_disposition,
        equal_lag_stationarity=temporal.stationarity_disposition,
        receiver_map=receiver_map,
        lie_axis=FiniteAxisDisposition.NOT_APPLICABLE,
        metrics=metrics,
        evidence_link_ids=(method_input.input_id,),
        reason_codes=tuple(sorted(reasons)),
    )
    composite = compose_thermodynamic_response_signature(
        signature_id=f"composite-signature-{method_input.input_id}",
        relation_id=panel.relation_id,
        finite_response_signature=finite_signature,
        scalar_dose_projection_witness=method_input.scalar_dose_projection_witness,
        evidence_world_id=panel.evidence_world_id,
        world_kind=panel.world_kind,
        denominator_type_id="prepared-reference-denominator",
        bath_type_id="driven-reference-bath",
        thermodynamic_coordinate_sufficiency=entropy.status,
        return_qualification=qualification.status,
        energy_balance_closure=balance.status,
        loop_exchange_observation_ids=loop_exchange_ids,
        action_quotient=action_quotient,
        resolution_class=constituent_disposition,
        evidence_link_ids=(method_input.input_id,),
        reason_codes=tuple(sorted(reasons)),
    )
    method_result = ThermodynamicResponseMethodResult(
        result_id=f"method-result-{method_input.input_id}",
        input_id=method_input.input_id,
        config_id=config.config_id,
        signature=composite,
        contrast_metrics=metrics,
        balance_assessments=(balance,),
        return_qualifications=(qualification,),
        entropy_eligibility=entropy.status,
        generated_entropy=(
            NamedDecimal(
                "generated-entropy",
                entropy.generated_entropy,
                entropy.entropy_unit or "J/K",
            )
            if entropy.generated_entropy is not None
            else None
        ),
        independent_unit_ids=panel.independent_unit_ids,
        reason_codes=tuple(sorted(reasons)),
    )
    return ThermodynamicReferenceMethodReadout(
        readout_id=f"readout-{method_input.input_id}",
        input_id=method_input.input_id,
        method_result=method_result,
        controlled_order=controlled_disposition,
        state_bath_return=qualification.status,
        energy_balance=balance.status,
        entropy_eligibility=entropy.status,
        action_quotient=action_quotient,
        temporal_cocycle=temporal.cocycle_disposition,
        equal_lag_stationarity=temporal.stationarity_disposition,
        receiver_map=receiver_map,
        cycle_interpretation_allowed=(
            qualification.status is ReturnQualificationStatus.RETURN_QUALIFIED
            and balance.status is BalanceClosureStatus.CLOSED
        ),
        context_augmentation=context_augmentation,
        representation=representation,
        perturbation_exchange=perturbation_exchange,
        denominator_stability=denominator_stability,
        finite_coherence=finite_coherence,
        transport=transport_status,
        double_count_rejected=double_count_rejected,
        lie_claim_allowed=False,
        reason_codes=tuple(sorted(reasons)),
    )


_SCORED_FIELDS = (
    "action_quotient",
    "context_augmentation",
    "controlled_order",
    "cycle_interpretation_allowed",
    "denominator_stability",
    "double_count_rejected",
    "energy_balance",
    "entropy_eligibility",
    "equal_lag_stationarity",
    "finite_coherence",
    "lie_claim_allowed",
    "perturbation_exchange",
    "receiver_map",
    "representation",
    "state_bath_return",
    "temporal_cocycle",
    "transport",
)


def score_thermodynamic_response_conformance(
    readouts: tuple[ThermodynamicReferenceMethodReadout, ...],
    oracles: tuple[PrivilegedThermodynamicOracle, ...] | None = None,
    spec: ThermodynamicResponseConformanceSpec | None = None,
) -> ThermodynamicReferenceConformanceScore:
    frozen = spec or reference_thermodynamic_response_conformance_spec()
    truth = oracles or privileged_thermodynamic_oracles()
    if (
        tuple(value.input_id for value in readouts) != frozen.case_tokens
        or tuple(value.input_id for value in truth) != frozen.case_tokens
    ):
        raise ValueError("thermodynamic conformance case sets differ")
    scores = []
    false_noncommutativity = 0
    false_entropy = 0
    false_cycle = 0
    for index, (readout, oracle) in enumerate(zip(readouts, truth, strict=True)):
        mismatches = tuple(
            field_name
            for field_name in _SCORED_FIELDS
            if getattr(readout, field_name) != getattr(oracle, field_name)
        )
        if (
            oracle.controlled_order is not FiniteAxisDisposition.MATERIAL
            and readout.controlled_order is FiniteAxisDisposition.MATERIAL
        ):
            false_noncommutativity += 1
        if (
            oracle.entropy_eligibility is not EntropyEligibilityStatus.ELIGIBLE
            and readout.entropy_eligibility is EntropyEligibilityStatus.ELIGIBLE
        ):
            false_entropy += 1
        if not oracle.cycle_interpretation_allowed and readout.cycle_interpretation_allowed:
            false_cycle += 1
        scores.append(
            ThermodynamicReferenceCaseScore(
                score_id=f"case-score-{index:02d}",
                input_id=readout.input_id,
                passed=not mismatches,
                mismatched_field_ids=mismatches,
            )
        )
    case_scores = tuple(scores)
    failed = sum(not value.passed for value in case_scores)
    reasons = set()
    if failed:
        reasons.add("REFERENCE_CASE_MISMATCH")
    if false_noncommutativity:
        reasons.add("FALSE_NONCOMMUTATIVITY")
    if false_entropy:
        reasons.add("FALSE_ENTROPY_PRODUCTION")
    if false_cycle:
        reasons.add("FALSE_CYCLE_QUALIFICATION")
    gate = (
        failed == 0
        and false_noncommutativity <= frozen.maximum_false_noncommutativity
        and false_entropy <= frozen.maximum_false_entropy_production
        and false_cycle <= frozen.maximum_false_cycle_qualification
    )
    return ThermodynamicReferenceConformanceScore(
        score_id="thermodynamic-reference-conformance-score",
        conformance_id=frozen.conformance_id,
        case_scores=case_scores,
        passed_cases=len(case_scores) - failed,
        failed_cases=failed,
        false_noncommutativity=false_noncommutativity,
        false_entropy_production=false_entropy,
        false_cycle_qualification=false_cycle,
        gate_passed=gate,
        reason_codes=tuple(sorted(reasons)),
    )


def execute_truth_blind_thermodynamic_conformance(
    spec: ThermodynamicResponseConformanceSpec | None = None,
) -> tuple[ThermodynamicReferenceMethodReadout, ...]:
    frozen = spec or reference_thermodynamic_response_conformance_spec()
    return tuple(
        identify_truth_blind_thermodynamic_response(value, frozen)
        for value in truth_blind_thermodynamic_invocations(frozen)
    )


__all__ = [
    "ThermodynamicReferenceCaseScore",
    "ThermodynamicReferenceConformanceScore",
    "ThermodynamicReferenceInvocation",
    "ThermodynamicReferenceMethodReadout",
    "execute_truth_blind_thermodynamic_conformance",
    "identify_truth_blind_thermodynamic_response",
    "reference_finite_word_family",
    "score_thermodynamic_response_conformance",
    "truth_blind_thermodynamic_invocations",
]
