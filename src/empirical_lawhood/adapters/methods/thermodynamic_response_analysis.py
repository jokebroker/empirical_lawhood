"""Transparent grouped estimators for finite thermodynamic-response experiments.

The routines in this module operate on declared words and physical independent
units. They contain no source access, outcome reveal, persistence, actuation or
privileged truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.thermodynamic_response import BalanceClosureAssessment, BalanceClosureStatus, CompositeSignatureAxis, EntropyEligibilityStatus, EntropyTermRole, EntropyTermSpec, ExchangeEstimandKind, FiniteAxisDisposition, FiniteResponseSignature, OntologyCompatibilityWitness, ReturnQualification, ReturnQualificationStatus, ReceiverMapDisposition, StoredEnergyTermSpec, StructuralTransportWitness, ThermodynamicClaimCeiling, ThermodynamicCoordinateRole, ThermodynamicLedgerObservation, ThermodynamicObservationStatus, ThermodynamicResponseSignature, ThermodynamicRoleBinding, ThermodynamicTermSpec, TransportDisposition, ScalarDoseActionWordProjectionWitness, ScalarDoseActionWordSignatureStatus, validate_balance_observations, validate_thermodynamic_role_bindings
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.thermodynamic_response import (
    FiniteContrastSpec,
    FiniteWordFamilySpec,
    WordRole,
)

from .response_algebra import _apply_homogeneous, _homogeneous_fit
from .thermodynamic_response import FiniteTrajectoryPanel


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ThermodynamicResponseMethodInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/thermodynamic-response-method-input'

    input_id: str
    ontology_witness: OntologyCompatibilityWitness
    family: FiniteWordFamilySpec
    panel: FiniteTrajectoryPanel
    contrasts: tuple[FiniteContrastSpec, ...]
    role_bindings: tuple[ThermodynamicRoleBinding, ...]
    boundary_terms: tuple[ThermodynamicTermSpec, ...]
    stored_energy_terms: tuple[StoredEnergyTermSpec, ...]
    entropy_terms: tuple[EntropyTermSpec, ...]
    ledger_observations: tuple[ThermodynamicLedgerObservation, ...]
    scalar_dose_projection_witness: ScalarDoseActionWordProjectionWitness | None

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        relation_id = self.ontology_witness.relation.relation_id
        if self.family.relation_id != relation_id or self.panel.relation_id != relation_id:
            raise ValueError("method input changes its relational identity")
        if tuple(value.word_id for value in self.family.words) != self.panel.word_ids:
            raise ValueError("method panel differs from the frozen ordered word inventory")
        identity_word_ids = {
            word_id
            for word_id, roles in self.family.role_map.assignments
            if WordRole.IDENTITY in roles
        }
        if identity_word_ids != {self.panel.identity_word_id}:
            raise ValueError("method panel identity differs from the declared identity role")
        require_sorted_unique_ids(
            self.contrasts,
            attribute="contrast_id",
            field_name="contrasts",
        )
        family_word_ids = {value.word_id for value in self.family.words}
        for contrast in self.contrasts:
            if contrast.family_id != self.family.family_id:
                raise ValueError("finite contrast belongs to another word family")
            if contrast.relation_id != relation_id:
                raise ValueError("finite contrast belongs to another relation")
            if not {value for value, _coefficient in contrast.coefficients} <= family_word_ids:
                raise ValueError("finite contrast references a word outside its family")
            if contrast.receiver_id != self.panel.receiver_id:
                raise ValueError("finite contrast receiver differs from its panel")
            if contrast.horizon_id != self.panel.horizon_id:
                raise ValueError("finite contrast horizon differs from its panel")
            if contrast.coordinate_ids != self.panel.coordinate_ids:
                raise ValueError("finite contrast coordinates differ from its panel")
            if contrast.response_units != self.panel.native_units:
                raise ValueError("finite contrast native units differ from its panel")
        validate_thermodynamic_role_bindings(self.role_bindings)
        for values, attribute, name in (
            (self.boundary_terms, "term_id", "boundary_terms"),
            (self.stored_energy_terms, "term_id", "stored_energy_terms"),
            (self.entropy_terms, "term_id", "entropy_terms"),
            (self.ledger_observations, "observation_id", "ledger_observations"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)
        system_ids = {
            self.panel.system_id,
            *(value.system_id for value in self.boundary_terms),
            *(value.system_id for value in self.stored_energy_terms),
            *(value.system_id for value in self.entropy_terms),
            *(value.system_id for value in self.ledger_observations),
        }
        if len(system_ids) != 1:
            raise ValueError("method input crosses system/evidence boundaries")
        if not {value.independent_unit_id for value in self.ledger_observations} <= set(
            self.panel.independent_unit_ids
        ):
            raise ValueError("method ledger references a unit outside its trajectory panel")


@dataclass(frozen=True, slots=True)
class ThermodynamicResponseMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/thermodynamic-response-method-config'

    config_id: str
    method_version: str
    identity_word_id: str
    minimum_independent_units: int
    numerical_floor: tuple[NamedDecimal, ...]
    affine_ridge: Decimal
    cocycle_tolerance: Decimal
    stationarity_tolerance: Decimal
    balance_tolerance: Decimal
    balance_unit: str
    system_boundary_complete: bool
    chemical_terms_complete: bool
    synchronization_valid: bool
    constitutive_assumptions_valid: bool
    local_equilibrium_justified: bool
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_semantic_version(self.method_version)
        if self.method_version != "1.0.0":
            raise ValueError("unsupported thermodynamic-response method version")
        validate_stable_id(self.identity_word_id, field_name="identity_word_id")
        if not 2 <= self.minimum_independent_units <= 100_000:
            raise ValueError("minimum independent-unit count must be in [2, 100000]")
        require_sorted_unique_ids(
            self.numerical_floor,
            attribute="value_id",
            field_name="numerical_floor",
        )
        if not self.numerical_floor:
            raise ValueError("method config requires native-coordinate floors")
        for name, value in (
            ("affine_ridge", self.affine_ridge),
            ("cocycle_tolerance", self.cocycle_tolerance),
            ("stationarity_tolerance", self.stationarity_tolerance),
            ("balance_tolerance", self.balance_tolerance),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
        validate_nonempty(self.balance_unit, field_name="balance_unit")
        require_sorted_unique_strings(
            self.prohibited_component_ids,
            field_name="prohibited_component_ids",
            allow_empty=False,
        )
        if not {"llm", "rl"} <= set(self.prohibited_component_ids):
            raise ValueError("thermodynamic-response method must prohibit LLM and RL")


@dataclass(frozen=True, slots=True)
class ThermodynamicResponseMethodResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/thermodynamic-response-method-result'

    result_id: str
    input_id: str
    config_id: str
    signature: ThermodynamicResponseSignature
    contrast_metrics: tuple[NamedDecimal, ...]
    balance_assessments: tuple[BalanceClosureAssessment, ...]
    return_qualifications: tuple[ReturnQualification, ...]
    entropy_eligibility: EntropyEligibilityStatus
    generated_entropy: NamedDecimal | None
    independent_unit_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("input_id", self.input_id),
            ("config_id", self.config_id),
        ):
            validate_stable_id(value, field_name=name)
        for values, attribute, name in (
            (self.contrast_metrics, "value_id", "contrast_metrics"),
            (self.balance_assessments, "assessment_id", "balance_assessments"),
            (self.return_qualifications, "qualification_id", "return_qualifications"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.entropy_eligibility is EntropyEligibilityStatus.ELIGIBLE:
            if self.generated_entropy is None:
                raise ValueError("eligible entropy result requires generated entropy")
        elif self.generated_entropy is not None:
            raise ValueError("ineligible entropy result cannot report generated entropy")


@dataclass(frozen=True, slots=True)
class GroupedFiniteContrast:
    contrast_id: str
    independent_unit_ids: tuple[str, ...]
    excluded_unit_ids: tuple[str, ...]
    times: FloatArray
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        validate_stable_id(self.contrast_id, field_name="contrast_id")
        expected = (len(self.independent_unit_ids), len(self.times), len(self.coordinate_ids))
        if self.values.shape != expected or not np.all(np.isfinite(self.values)):
            raise ValueError("grouped finite contrast values are incomplete or nonfinite")
        if set(self.independent_unit_ids) & set(self.excluded_unit_ids):
            raise ValueError("included and excluded contrast units overlap")


@dataclass(frozen=True, slots=True)
class IncompleteBlockContrastResult:
    contrast_id: str
    model_id: str
    independent_unit_ids: tuple[str, ...]
    leave_one_out_unit_ids: tuple[str, ...]
    aggregate_estimate: FloatArray
    leave_one_unit_out_estimates: FloatArray
    incidence_rank: int
    estimable: bool


@dataclass(frozen=True, slots=True)
class TemporalPropagationResult:
    independent_unit_ids: tuple[str, ...]
    cocycle_defects: FloatArray
    stationarity_defects: FloatArray
    cocycle_disposition: FiniteAxisDisposition
    stationarity_disposition: FiniteAxisDisposition


@dataclass(frozen=True, slots=True)
class IdentityCoherenceResult:
    independent_unit_ids: tuple[str, ...]
    residuals: FloatArray
    native_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FittedAffineMap:
    map_id: str
    training_unit_ids: tuple[str, ...]
    operator: FloatArray

    def __post_init__(self) -> None:
        validate_stable_id(self.map_id, field_name="map_id")
        require_sorted_unique_strings(
            self.training_unit_ids,
            field_name="training_unit_ids",
            allow_empty=False,
        )
        if self.operator.ndim != 2 or not np.all(np.isfinite(self.operator)):
            raise ValueError("fitted affine map must be a finite matrix")


@dataclass(frozen=True, slots=True)
class ParenthesizationCoherenceResult:
    evaluation_unit_ids: tuple[str, ...]
    left_predictions: FloatArray
    right_predictions: FloatArray
    observed_finals: FloatArray
    parenthesization_residuals: FloatArray
    left_observation_errors: FloatArray
    right_observation_errors: FloatArray


@dataclass(frozen=True, slots=True)
class ReceiverNaturalityResult:
    independent_unit_ids: tuple[str, ...]
    maximum_value_defect: float
    full_resolved_distinction_count: int
    projected_retained_distinction_count: int
    disposition: ReceiverMapDisposition


@dataclass(frozen=True, slots=True)
class GeneralizedGeometryResult:
    independent_unit_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    singular_values_by_time: tuple[tuple[float, ...], ...]
    floor_certified_ranks: tuple[int, ...]
    participation_dimensions: tuple[float, ...]
    metric_label: str


@dataclass(frozen=True, slots=True)
class EntropyBalanceResult:
    status: EntropyEligibilityStatus
    generated_entropy: Decimal | None
    uncertainty: Decimal | None
    entropy_unit: str | None
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CycleExchangeResult:
    word_observations: tuple[ThermodynamicLedgerObservation, ...]
    identity_observations: tuple[ThermodynamicLedgerObservation, ...]
    identity_differenced_observations: tuple[ThermodynamicLedgerObservation, ...]


@dataclass(frozen=True, slots=True)
class TransportAxisComparison:
    axis: CompositeSignatureAxis
    source_then_map: FloatArray
    map_then_target: FloatArray
    uncertainty: Decimal
    tolerance: Decimal
    native_unit: str


def commuting_square_residual(
    source_then_map: FloatArray,
    map_then_target: FloatArray,
    *,
    uncertainty: Decimal,
    tolerance: Decimal,
    native_unit: str,
) -> tuple[Decimal, Decimal, Decimal, str, FiniteAxisDisposition]:
    """Compute one native-unit commuting residual without cross-medium pooling."""

    if source_then_map.shape != map_then_target.shape or source_then_map.size == 0:
        raise ValueError("commuting-square paths have different or empty shapes")
    if not np.all(np.isfinite(source_then_map)) or not np.all(np.isfinite(map_then_target)):
        raise ValueError("commuting-square paths must be finite")
    validate_decimal(uncertainty, field_name="uncertainty", minimum=Decimal("0"))
    validate_decimal(tolerance, field_name="tolerance", minimum=Decimal("0"))
    validate_nonempty(native_unit, field_name="native_unit")
    residual = Decimal(f"{float(np.sqrt(np.mean((source_then_map - map_then_target) ** 2))):.15g}")
    disposition = (
        FiniteAxisDisposition.EQUIVALENT
        if residual + uncertainty <= tolerance
        else FiniteAxisDisposition.MATERIAL
    )
    return residual, uncertainty, tolerance, native_unit, disposition


def build_structural_transport_witness(
    *,
    witness_id: str,
    source_signature_id: str,
    target_signature_id: str,
    source_system_id: str,
    target_system_id: str,
    source_world_kind: WorldKind,
    target_world_kind: WorldKind,
    action_role_map: tuple[tuple[str, str], ...],
    horizon_map: tuple[tuple[str, str], ...],
    receiver_map: tuple[tuple[str, str], ...],
    thermodynamic_role_map: tuple[tuple[str, str], ...],
    common_support_object_map: tuple[tuple[str, str], ...],
    axis_comparisons: tuple[TransportAxisComparison, ...],
    faithfulness_limit_ids: tuple[str, ...],
    decisive_counterexample_ids: tuple[str, ...],
) -> StructuralTransportWitness:
    """Build a nonpooling transport witness from native per-axis comparisons."""

    axes = tuple(value.axis for value in axis_comparisons)
    if not axes or tuple(sorted(set(axes), key=lambda value: value.value)) != axes:
        raise ValueError("transport comparisons must be nonempty, sorted and unique by axis")
    residuals = tuple(
        (
            comparison.axis,
            *commuting_square_residual(
                comparison.source_then_map,
                comparison.map_then_target,
                uncertainty=comparison.uncertainty,
                tolerance=comparison.tolerance,
                native_unit=comparison.native_unit,
            ),
        )
        for comparison in axis_comparisons
    )
    dispositions = tuple(value[-1] for value in residuals)
    reasons: tuple[str, ...]
    if decisive_counterexample_ids:
        status = TransportDisposition.FAILED
        reasons = ("DECISIVE_TRANSPORT_COUNTEREXAMPLE",)
    elif all(value is FiniteAxisDisposition.EQUIVALENT for value in dispositions):
        status = TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION
        reasons = ()
    elif any(value is FiniteAxisDisposition.EQUIVALENT for value in dispositions):
        status = TransportDisposition.PARTIAL
        reasons = ("SOME_TRANSPORT_AXES_EXCEED_NATIVE_TOLERANCE",)
    else:
        status = TransportDisposition.FAILED
        reasons = ("TRANSPORT_AXES_EXCEED_NATIVE_TOLERANCE",)
    return StructuralTransportWitness(
        witness_id=witness_id,
        source_signature_id=source_signature_id,
        target_signature_id=target_signature_id,
        source_system_id=source_system_id,
        target_system_id=target_system_id,
        source_world_kind=source_world_kind,
        target_world_kind=target_world_kind,
        action_role_map=action_role_map,
        horizon_map=horizon_map,
        receiver_map=receiver_map,
        thermodynamic_role_map=thermodynamic_role_map,
        common_support_object_map=common_support_object_map,
        axis_residuals=residuals,
        faithfulness_limit_ids=faithfulness_limit_ids,
        decisive_counterexample_ids=decisive_counterexample_ids,
        native_numeric_pooling_performed=False,
        status=status,
        reason_codes=reasons,
    )


def _panel_values(panel: FiniteTrajectoryPanel) -> npt.NDArray[np.object_]:
    shape = (
        len(panel.independent_unit_ids),
        len(panel.word_ids),
        len(panel.times),
        len(panel.coordinate_ids),
    )
    return np.asarray(panel.values, dtype=object).reshape(shape)


def _block_is_complete(
    panel: FiniteTrajectoryPanel,
    values: npt.NDArray[np.object_],
    unit_index: int,
    word_index: int,
) -> bool:
    block_index = unit_index * len(panel.word_ids) + word_index
    return panel.source_episode_ids[block_index] is not None and all(
        value is not None for value in values[unit_index, word_index].reshape(-1)
    )


def _incidence_design(
    panel: FiniteTrajectoryPanel,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    raw = _panel_values(panel)
    rows: list[FloatArray] = []
    trajectories: list[FloatArray] = []
    observed_units: set[str] = set()
    column_count = len(panel.independent_unit_ids) + len(panel.word_ids)
    for unit_index, unit_id in enumerate(panel.independent_unit_ids):
        for word_index in range(len(panel.word_ids)):
            if not _block_is_complete(panel, raw, unit_index, word_index):
                continue
            row = np.zeros(column_count, dtype=np.float64)
            row[unit_index] = 1.0
            row[len(panel.independent_unit_ids) + word_index] = 1.0
            rows.append(row)
            trajectories.append(np.asarray(raw[unit_index, word_index], dtype=np.float64))
            observed_units.add(unit_id)
    if not rows:
        raise ValueError("finite panel has no complete assigned blocks")
    return np.stack(rows), np.stack(trajectories), tuple(sorted(observed_units))


def _contrast_vector(panel: FiniteTrajectoryPanel, spec: FiniteContrastSpec) -> FloatArray:
    word_index = {value: index for index, value in enumerate(panel.word_ids)}
    vector = np.zeros(len(panel.independent_unit_ids) + len(panel.word_ids), dtype=np.float64)
    for word_id, coefficient in spec.coefficients:
        try:
            index = word_index[word_id]
        except KeyError as error:
            raise ValueError("finite contrast references a missing word") from error
        vector[len(panel.independent_unit_ids) + index] = float(coefficient)
    return vector


def _is_estimable(design: FloatArray, functional: FloatArray) -> bool:
    projection = np.linalg.pinv(design) @ design
    return bool(np.linalg.norm(functional - functional @ projection) <= 1e-9)


def estimate_incomplete_block_contrast(
    panel: FiniteTrajectoryPanel,
    spec: FiniteContrastSpec,
) -> IncompleteBlockContrastResult:
    """Estimate a connected unit/word fixed-effect contrast with grouped LOO sensitivity."""

    if not spec.zero_sum_required:
        raise ValueError("incomplete-block word effects identify only zero-sum contrasts")
    if spec.coordinate_ids != panel.coordinate_ids or spec.response_units != panel.native_units:
        raise ValueError("incomplete-block contrast units differ from its panel")
    design, trajectories, observed_unit_ids = _incidence_design(panel)
    functional = _contrast_vector(panel, spec)
    estimable = _is_estimable(design, functional)
    shape = (len(panel.times), len(panel.coordinate_ids))
    if not estimable:
        return IncompleteBlockContrastResult(
            contrast_id=spec.contrast_id,
            model_id="additive-physical-unit-and-word-fixed-effects",
            independent_unit_ids=observed_unit_ids,
            leave_one_out_unit_ids=(),
            aggregate_estimate=np.full(shape, np.nan),
            leave_one_unit_out_estimates=np.empty((0, *shape)),
            incidence_rank=int(np.linalg.matrix_rank(design)),
            estimable=False,
        )
    coefficients = np.linalg.pinv(design) @ trajectories.reshape(len(design), -1)
    aggregate = (functional @ coefficients).reshape(shape)
    leave_one_out: list[FloatArray] = []
    retained_units: list[str] = []
    for unit_index, unit_id in enumerate(panel.independent_unit_ids):
        keep = design[:, unit_index] == 0
        reduced = design[keep]
        if not np.any(keep) or not _is_estimable(reduced, functional):
            continue
        reduced_coefficients = np.linalg.pinv(reduced) @ trajectories[keep].reshape(
            int(np.sum(keep)), -1
        )
        leave_one_out.append((functional @ reduced_coefficients).reshape(shape))
        retained_units.append(unit_id)
    return IncompleteBlockContrastResult(
        contrast_id=spec.contrast_id,
        model_id="additive-physical-unit-and-word-fixed-effects",
        independent_unit_ids=observed_unit_ids,
        leave_one_out_unit_ids=tuple(retained_units),
        aggregate_estimate=aggregate,
        leave_one_unit_out_estimates=(
            np.stack(leave_one_out) if leave_one_out else np.empty((0, *shape))
        ),
        incidence_rank=int(np.linalg.matrix_rank(design)),
        estimable=True,
    )


def _word_effect_responses(
    panel: FiniteTrajectoryPanel,
    identity_word_id: str,
) -> tuple[tuple[str, ...], FloatArray]:
    design, trajectories, observed_unit_ids = _incidence_design(panel)
    try:
        identity_index = panel.word_ids.index(identity_word_id)
    except ValueError as error:
        raise ValueError("word-effect identity is absent") from error
    coefficients = np.linalg.pinv(design) @ trajectories.reshape(len(design), -1)
    word_effects = []
    for word_index in range(len(panel.word_ids)):
        functional = np.zeros(design.shape[1], dtype=np.float64)
        functional[len(panel.independent_unit_ids) + word_index] = 1
        functional[len(panel.independent_unit_ids) + identity_index] -= 1
        if not _is_estimable(design, functional):
            raise ValueError("word-incidence graph does not identify every identity contrast")
        word_effects.append(
            (functional @ coefficients).reshape(len(panel.times), len(panel.coordinate_ids))
        )
    return observed_unit_ids, np.stack(word_effects)


def evaluate_finite_contrast(
    panel: FiniteTrajectoryPanel,
    spec: FiniteContrastSpec,
) -> GroupedFiniteContrast:
    """Evaluate one static contrast on complete participating physical units."""

    if spec.receiver_id != panel.receiver_id or spec.horizon_id != panel.horizon_id:
        raise ValueError("finite contrast receiver/horizon differs from its panel")
    if spec.coordinate_ids != panel.coordinate_ids or spec.response_units != panel.native_units:
        raise ValueError("finite contrast coordinates/native units differ from its panel")
    word_index = {value: index for index, value in enumerate(panel.word_ids)}
    if not {value for value, _coefficient in spec.coefficients} <= set(word_index):
        raise ValueError("finite contrast references a missing word")
    raw = _panel_values(panel)
    included: list[str] = []
    excluded: list[str] = []
    results: list[FloatArray] = []
    for unit_index, unit_id in enumerate(panel.independent_unit_ids):
        selected = tuple(
            (word_index[word_id], coefficient) for word_id, coefficient in spec.coefficients
        )
        if not all(
            _block_is_complete(panel, raw, unit_index, selected_index)
            for selected_index, _coefficient in selected
        ):
            excluded.append(unit_id)
            continue
        total = np.zeros((len(panel.times), len(panel.coordinate_ids)), dtype=np.float64)
        for selected_index, coefficient in selected:
            total += np.asarray(raw[unit_index, selected_index], dtype=np.float64) * float(
                coefficient
            )
        included.append(unit_id)
        results.append(total)
    values = (
        np.stack(results)
        if results
        else np.empty((0, len(panel.times), len(panel.coordinate_ids)), dtype=np.float64)
    )
    return GroupedFiniteContrast(
        contrast_id=spec.contrast_id,
        independent_unit_ids=tuple(included),
        excluded_unit_ids=tuple(excluded),
        times=np.asarray(tuple(float(value) for value in panel.times), dtype=np.float64),
        coordinate_ids=panel.coordinate_ids,
        native_units=panel.native_units,
        values=values,
    )


def controlled_order_contrast(
    raw_order: GroupedFiniteContrast,
    timing_and_bath_null: GroupedFiniteContrast,
) -> GroupedFiniteContrast:
    """Subtract the predeclared timing/bath null without calling raw AB-BA causal."""

    if (
        raw_order.independent_unit_ids != timing_and_bath_null.independent_unit_ids
        or raw_order.coordinate_ids != timing_and_bath_null.coordinate_ids
        or raw_order.native_units != timing_and_bath_null.native_units
        or not np.array_equal(raw_order.times, timing_and_bath_null.times)
    ):
        raise ValueError("controlled order requires the same complete physical-unit blocks")
    return GroupedFiniteContrast(
        contrast_id=f"controlled-{raw_order.contrast_id}",
        independent_unit_ids=raw_order.independent_unit_ids,
        excluded_unit_ids=tuple(
            sorted(set(raw_order.excluded_unit_ids) | set(timing_and_bath_null.excluded_unit_ids))
        ),
        times=raw_order.times.copy(),
        coordinate_ids=raw_order.coordinate_ids,
        native_units=raw_order.native_units,
        values=raw_order.values - timing_and_bath_null.values,
    )


def _complete_word_tensor(
    panel: FiniteTrajectoryPanel,
    word_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], FloatArray]:
    raw = _panel_values(panel)
    word_index = {value: index for index, value in enumerate(panel.word_ids)}
    if not set(word_ids) <= set(word_index):
        raise ValueError("requested word tensor references a missing word")
    included_indices = tuple(
        unit_index
        for unit_index in range(len(panel.independent_unit_ids))
        if all(
            _block_is_complete(panel, raw, unit_index, word_index[word_id]) for word_id in word_ids
        )
    )
    unit_ids = tuple(panel.independent_unit_ids[index] for index in included_indices)
    values = np.asarray(
        [[raw[index, word_index[word_id]] for word_id in word_ids] for index in included_indices],
        dtype=np.float64,
    )
    return unit_ids, values


def temporal_propagation(
    panel: FiniteTrajectoryPanel,
    *,
    identity_word_id: str,
    selected_time_indices: tuple[int, ...],
    ridge: float,
    cocycle_tolerance: float,
    stationarity_tolerance: float,
) -> TemporalPropagationResult:
    """Cross-fitted cocycle and equal-lag stationarity as separate axes."""

    if tuple(sorted(set(selected_time_indices))) != selected_time_indices:
        raise ValueError("temporal indices must be sorted and unique")
    if len(selected_time_indices) < 3 or selected_time_indices[-1] >= len(panel.times):
        raise ValueError("temporal propagation requires at least three in-range anchors")
    unit_ids, tensor = _complete_word_tensor(panel, (identity_word_id,))
    identity = tensor[:, 0][:, selected_time_indices]
    if len(unit_ids) < 3:
        raise ValueError("temporal propagation requires at least three physical units")
    scale = np.std(identity.reshape(-1, identity.shape[-1]), axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    identity = identity / scale
    cocycle: list[float] = []
    for first, middle, last in combinations(range(len(selected_time_indices)), 3):
        for held_out in range(len(unit_ids)):
            train = np.arange(len(unit_ids)) != held_out
            first_map = _homogeneous_fit(identity[train, first], identity[train, middle], ridge)
            second_map = _homogeneous_fit(identity[train, middle], identity[train, last], ridge)
            direct_map = _homogeneous_fit(identity[train, first], identity[train, last], ridge)
            start_state = identity[held_out : held_out + 1, first]
            via = _apply_homogeneous(second_map @ first_map, start_state)
            direct = _apply_homogeneous(direct_map, start_state)
            cocycle.append(float(np.sqrt(np.mean((via - direct) ** 2))))
    stationarity: list[float] = []
    times = tuple(panel.times[index] for index in selected_time_indices)
    intervals: dict[Decimal, list[tuple[int, int]]] = {}
    for start_index in range(len(times) - 1):
        for end in range(start_index + 1, len(times)):
            intervals.setdefault(times[end] - times[start_index], []).append((start_index, end))
    for same_lag in intervals.values():
        if len(same_lag) < 2:
            continue
        for first_interval, second_interval in combinations(same_lag, 2):
            for held_out in range(len(unit_ids)):
                train = np.arange(len(unit_ids)) != held_out
                first_map = _homogeneous_fit(
                    identity[train, first_interval[0]],
                    identity[train, first_interval[1]],
                    ridge,
                )
                second_map = _homogeneous_fit(
                    identity[train, second_interval[0]],
                    identity[train, second_interval[1]],
                    ridge,
                )
                start = identity[held_out : held_out + 1, first_interval[0]]
                stationarity.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (
                                    _apply_homogeneous(first_map, start)
                                    - _apply_homogeneous(second_map, start)
                                )
                                ** 2
                            )
                        )
                    )
                )
    cocycle_values = np.asarray(cocycle, dtype=np.float64)
    stationarity_values = np.asarray(stationarity, dtype=np.float64)
    cocycle_status = (
        FiniteAxisDisposition.COCYCLIC
        if cocycle_values.size and float(np.max(cocycle_values)) <= cocycle_tolerance
        else FiniteAxisDisposition.NONCLOSED
    )
    stationarity_status = (
        FiniteAxisDisposition.STATIONARY
        if stationarity_values.size and float(np.max(stationarity_values)) <= stationarity_tolerance
        else (
            FiniteAxisDisposition.NONSTATIONARY
            if stationarity_values.size
            else FiniteAxisDisposition.UNEVALUABLE
        )
    )
    return TemporalPropagationResult(
        independent_unit_ids=unit_ids,
        cocycle_defects=cocycle_values,
        stationarity_defects=stationarity_values,
        cocycle_disposition=cocycle_status,
        stationarity_disposition=stationarity_status,
    )


def identity_coherence(
    panel: FiniteTrajectoryPanel,
    *,
    identity_word_id: str,
    ridge: float,
) -> IdentityCoherenceResult:
    """Leave-one-physical-unit-out fitted natural-evolution identity residuals."""

    unit_ids, tensor = _complete_word_tensor(panel, (identity_word_id,))
    identity = tensor[:, 0]
    if len(unit_ids) < 3:
        raise ValueError("identity coherence requires at least three physical units")
    residuals = []
    for held_out in range(len(unit_ids)):
        train = np.arange(len(unit_ids)) != held_out
        operator = _homogeneous_fit(identity[train, 0], identity[train, -1], ridge)
        predicted = _apply_homogeneous(operator, identity[held_out : held_out + 1, 0])
        residuals.append(predicted[0] - identity[held_out, -1])
    return IdentityCoherenceResult(
        independent_unit_ids=unit_ids,
        residuals=np.asarray(residuals, dtype=np.float64),
        native_units=panel.native_units,
    )


def fit_affine_map(
    *,
    map_id: str,
    training_unit_ids: tuple[str, ...],
    starts: FloatArray,
    ends: FloatArray,
    ridge: float,
) -> FittedAffineMap:
    if starts.shape != ends.shape or starts.shape[0] != len(training_unit_ids):
        raise ValueError("affine-map training states differ from physical-unit identities")
    return FittedAffineMap(
        map_id=map_id,
        training_unit_ids=training_unit_ids,
        operator=_homogeneous_fit(starts, ends, ridge),
    )


def parenthesization_coherence(
    *,
    action_a: FittedAffineMap,
    action_c: FittedAffineMap,
    composite_ab: FittedAffineMap,
    composite_bc: FittedAffineMap,
    evaluation_unit_ids: tuple[str, ...],
    starts: FloatArray,
    observed_finals: FloatArray,
) -> ParenthesizationCoherenceResult:
    """Compare C o (BA) with (CB) o A from separately fitted maps."""

    maps = (action_a, action_c, composite_ab, composite_bc)
    if len({value.map_id for value in maps}) != len(maps):
        raise ValueError("parenthesization requires separately identified map records")
    if starts.shape != observed_finals.shape or starts.shape[0] != len(evaluation_unit_ids):
        raise ValueError("coherence evaluation states differ from physical-unit identities")
    shapes = {value.operator.shape for value in maps}
    if len(shapes) != 1:
        raise ValueError("parenthesization map dimensions differ")
    left = _apply_homogeneous(action_c.operator @ composite_ab.operator, starts)
    right = _apply_homogeneous(composite_bc.operator @ action_a.operator, starts)
    return ParenthesizationCoherenceResult(
        evaluation_unit_ids=evaluation_unit_ids,
        left_predictions=left,
        right_predictions=right,
        observed_finals=observed_finals,
        parenthesization_residuals=left - right,
        left_observation_errors=left - observed_finals,
        right_observation_errors=right - observed_finals,
    )


def receiver_naturality(
    full_panel: FiniteTrajectoryPanel,
    projected_panel: FiniteTrajectoryPanel,
    projection: FloatArray,
    *,
    identity_word_id: str,
    full_floor: FloatArray,
    projected_floor: FloatArray,
    naturality_tolerance: float,
) -> ReceiverNaturalityResult:
    """Test projection commutation and distinction retention on the same episodes."""

    if (
        full_panel.system_id != projected_panel.system_id
        or full_panel.relation_id != projected_panel.relation_id
        or full_panel.independent_unit_ids != projected_panel.independent_unit_ids
        or full_panel.word_ids != projected_panel.word_ids
        or full_panel.times != projected_panel.times
        or full_panel.source_episode_ids != projected_panel.source_episode_ids
        or full_panel.matched_identity_episode_ids != projected_panel.matched_identity_episode_ids
    ):
        raise ValueError("receiver panels are not nested views of the same episodes")
    if projection.shape != (
        len(projected_panel.coordinate_ids),
        len(full_panel.coordinate_ids),
    ):
        raise ValueError("receiver projection shape differs from declared views")
    if full_floor.shape != (len(full_panel.times), len(full_panel.coordinate_ids)):
        raise ValueError("full receiver floor shape differs")
    if projected_floor.shape != (
        len(projected_panel.times),
        len(projected_panel.coordinate_ids),
    ):
        raise ValueError("projected receiver floor shape differs")
    if np.any(full_floor <= 0) or np.any(projected_floor <= 0):
        raise ValueError("receiver floors must be positive")
    full_raw = _panel_values(full_panel)
    projected_raw = _panel_values(projected_panel)
    defects = []
    for unit_index in range(len(full_panel.independent_unit_ids)):
        for word_index in range(len(full_panel.word_ids)):
            full_complete = _block_is_complete(full_panel, full_raw, unit_index, word_index)
            projected_complete = _block_is_complete(
                projected_panel, projected_raw, unit_index, word_index
            )
            if full_complete != projected_complete:
                raise ValueError("receiver panels have different complete block support")
            if not full_complete:
                continue
            full_block = np.asarray(full_raw[unit_index, word_index], dtype=np.float64)
            projected_block = np.asarray(projected_raw[unit_index, word_index], dtype=np.float64)
            defects.append(np.einsum("tc,pc->tp", full_block, projection) - projected_block)
    if not defects:
        raise ValueError("receiver naturality has no common complete blocks")
    maximum_defect = float(np.max(np.abs(np.stack(defects))))
    if identity_word_id not in full_panel.word_ids:
        raise ValueError("receiver naturality identity word is absent")
    unit_ids, full_response = _word_effect_responses(full_panel, identity_word_id)
    projected_unit_ids, projected_response = _word_effect_responses(
        projected_panel, identity_word_id
    )
    if unit_ids != projected_unit_ids:
        raise ValueError("receiver panels have different connected unit support")
    full_scaled = full_response / full_floor[None, :, :]
    projected_scaled = projected_response / projected_floor[None, :, :]
    full_resolved = 0
    retained = 0
    for left in range(len(full_panel.word_ids)):
        for right in range(left + 1, len(full_panel.word_ids)):
            if float(np.linalg.norm(full_scaled[left] - full_scaled[right])) > 1:
                full_resolved += 1
                if float(np.linalg.norm(projected_scaled[left] - projected_scaled[right])) > 1:
                    retained += 1
    if maximum_defect > naturality_tolerance:
        disposition = ReceiverMapDisposition.NON_NATURAL_AT_TESTED_RESOLUTION
    elif full_resolved and retained == 0:
        disposition = ReceiverMapDisposition.ACTION_BLIND
    elif retained < full_resolved:
        disposition = ReceiverMapDisposition.PARTIALLY_COLLAPSING
    else:
        disposition = ReceiverMapDisposition.FAITHFUL_AT_TESTED_RESOLUTION
    return ReceiverNaturalityResult(
        independent_unit_ids=unit_ids,
        maximum_value_defect=maximum_defect,
        full_resolved_distinction_count=full_resolved,
        projected_retained_distinction_count=retained,
        disposition=disposition,
    )


def generalized_information_geometry(
    panel: FiniteTrajectoryPanel,
    *,
    identity_word_id: str,
    numerical_floor: FloatArray,
) -> GeneralizedGeometryResult:
    """Arbitrary-word receiver-pullback geometry, explicitly not Fisher information."""

    unit_ids, response = _word_effect_responses(panel, identity_word_id)
    if len(unit_ids) < 2:
        raise ValueError("generalized geometry requires at least two physical units")
    if numerical_floor.shape != (len(panel.times), len(panel.coordinate_ids)):
        raise ValueError("numerical floor differs from panel time/coordinate shape")
    if not np.all(np.isfinite(numerical_floor)) or np.any(numerical_floor <= 0):
        raise ValueError("numerical floor must be finite and positive")
    singular_rows: list[tuple[float, ...]] = []
    ranks: list[int] = []
    participation: list[float] = []
    for time_index in range(len(panel.times)):
        scaled = response[:, time_index] / numerical_floor[time_index]
        singular = np.linalg.svd(scaled, compute_uv=False)
        perturbation_bound = float(np.sqrt(scaled.size))
        active = singular > perturbation_bound
        active_values = singular[active]
        singular_rows.append(tuple(float(value) for value in singular))
        ranks.append(int(np.sum(active)))
        participation.append(
            float((np.sum(active_values**2) ** 2) / np.sum(active_values**4))
            if active_values.size and np.sum(active_values**4) > 0
            else 0.0
        )
    return GeneralizedGeometryResult(
        independent_unit_ids=unit_ids,
        word_ids=panel.word_ids,
        singular_values_by_time=tuple(singular_rows),
        floor_certified_ranks=tuple(ranks),
        participation_dimensions=tuple(participation),
        metric_label="receiver-pullback-floor-scaled-distinguishability-not-fisher-information",
    )


def build_absolute_energy_balance(
    *,
    assessment_id: str,
    expected_storage_term_ids: tuple[str, ...],
    expected_boundary_term_ids: tuple[str, ...],
    observations: tuple[ThermodynamicLedgerObservation, ...],
    closure_tolerance: Decimal,
    energy_unit: str,
    entropy_eligibility: EntropyEligibilityStatus,
) -> BalanceClosureAssessment:
    """Construct absolute first-law closure from storage and total exchange only."""

    validate_decimal(
        closure_tolerance,
        field_name="closure_tolerance",
        minimum=Decimal("0"),
    )
    validate_nonempty(energy_unit, field_name="energy_unit")
    require_sorted_unique_strings(
        expected_storage_term_ids,
        field_name="expected_storage_term_ids",
        allow_empty=False,
    )
    require_sorted_unique_strings(
        expected_boundary_term_ids,
        field_name="expected_boundary_term_ids",
        allow_empty=False,
    )
    require_sorted_unique_ids(
        observations,
        attribute="observation_id",
        field_name="observations",
    )
    if not observations:
        raise ValueError("energy balance requires ledger observations")
    identity = {
        (value.system_id, value.independent_unit_id, value.episode_id, value.word_id)
        for value in observations
    }
    if len(identity) != 1:
        raise ValueError("energy balance observations cross episode/unit/system boundaries")
    system_id, independent_unit_id, episode_id, word_id = next(iter(identity))
    if len({value.term_id for value in observations}) != len(observations):
        raise ValueError("energy balance contains duplicate observations for one term")
    by_term = {value.term_id: value for value in observations}
    expected = set(expected_storage_term_ids) | set(expected_boundary_term_ids)
    if not set(by_term) <= expected:
        raise ValueError("energy balance contains an undeclared term")
    missing = tuple(sorted(expected - set(by_term)))
    storage = tuple(by_term[value] for value in expected_storage_term_ids if value in by_term)
    boundary = tuple(by_term[value] for value in expected_boundary_term_ids if value in by_term)
    if any(
        value.coordinate_role is not ThermodynamicCoordinateRole.STORED_ENERGY
        or value.estimand_kind is not ExchangeEstimandKind.ABSOLUTE_STORAGE_CHANGE
        for value in storage
    ):
        raise ValueError("absolute energy balance storage uses a nonstorage estimand")
    permitted_boundary_roles = {
        ThermodynamicCoordinateRole.WORK_TRANSFER,
        ThermodynamicCoordinateRole.HEAT_TRANSFER,
        ThermodynamicCoordinateRole.MATTER_TRANSFER,
        ThermodynamicCoordinateRole.OTHER_BOUNDARY_TRANSFER,
    }
    if any(
        value.coordinate_role not in permitted_boundary_roles
        or value.estimand_kind is not ExchangeEstimandKind.TOTAL_EXCHANGE
        for value in boundary
    ):
        raise ValueError("absolute energy balance requires total boundary exchange")
    if any(value.canonical_unit != energy_unit for value in (*storage, *boundary)):
        raise ValueError("energy balance canonical units differ")
    if len({value.physical_event_id for value in boundary}) != len(boundary):
        raise ValueError("energy balance counts one physical event more than once")
    nonobserved = tuple(
        value.term_id
        for value in (*storage, *boundary)
        if value.status is not ThermodynamicObservationStatus.OBSERVED
    )
    missing = tuple(sorted(set(missing) | set(nonobserved)))
    complete = not missing
    delta: Decimal | None
    exchange: Decimal | None
    residual: Decimal | None
    uncertainty: Decimal | None
    if complete:
        delta = sum(
            (value.canonical_value for value in storage if value.canonical_value is not None),
            start=Decimal("0"),
        )
        exchange = sum(
            (value.canonical_value for value in boundary if value.canonical_value is not None),
            start=Decimal("0"),
        )
        residual = delta - exchange
        uncertainty = sum(
            (value.uncertainty for value in (*storage, *boundary) if value.uncertainty is not None),
            start=Decimal("0"),
        )
        if abs(residual) + uncertainty <= closure_tolerance:
            status = BalanceClosureStatus.CLOSED
            reasons: tuple[str, ...] = ()
        elif abs(residual) - uncertainty > closure_tolerance:
            status = BalanceClosureStatus.NOT_CLOSED
            reasons = ("ABSOLUTE_ENERGY_RESIDUAL_EXCEEDS_TOLERANCE",)
        else:
            status = BalanceClosureStatus.ANNULAR
            reasons = ("ABSOLUTE_ENERGY_RESIDUAL_ANNULAR",)
        claim_ceiling = (
            ThermodynamicClaimCeiling.ENTROPY_PRODUCTION_ELIGIBLE
            if status is BalanceClosureStatus.CLOSED
            and entropy_eligibility is EntropyEligibilityStatus.ELIGIBLE
            else ThermodynamicClaimCeiling.ENERGY_BALANCE_ONLY
        )
    else:
        delta = exchange = residual = uncertainty = None
        status = BalanceClosureStatus.TERM_UNOBSERVED
        reasons = ("BALANCE_TERM_UNOBSERVED",)
        claim_ceiling = ThermodynamicClaimCeiling.THERMODYNAMIC_LEDGER_PARTIAL
        entropy_eligibility = EntropyEligibilityStatus.LEDGER_PARTIAL
    assessment = BalanceClosureAssessment(
        assessment_id=assessment_id,
        system_id=system_id,
        independent_unit_id=independent_unit_id,
        episode_id=episode_id,
        word_id=word_id,
        stored_energy_observation_ids=tuple(sorted(value.observation_id for value in storage)),
        boundary_transfer_observation_ids=tuple(sorted(value.observation_id for value in boundary)),
        unobserved_term_ids=missing,
        delta_energy=delta,
        total_boundary_exchange=exchange,
        residual=residual,
        uncertainty=uncertainty,
        closure_tolerance=closure_tolerance,
        energy_unit=energy_unit,
        status=status,
        entropy_eligibility=entropy_eligibility,
        claim_ceiling=claim_ceiling,
        reason_codes=reasons,
    )
    if complete:
        validate_balance_observations(assessment, observations)
    return assessment


def assess_entropy_balance(
    *,
    entropy_terms: tuple[EntropyTermSpec, ...],
    observations: tuple[ThermodynamicLedgerObservation, ...],
    absolute_energy_balance: BalanceClosureAssessment,
    system_boundary_complete: bool,
    chemical_terms_complete: bool,
    synchronization_valid: bool,
    constitutive_assumptions_valid: bool,
    local_equilibrium_justified: bool,
) -> EntropyBalanceResult:
    """Report generated entropy only after every independent eligibility gate passes."""

    require_sorted_unique_ids(entropy_terms, attribute="term_id", field_name="entropy_terms")
    require_sorted_unique_ids(
        observations,
        attribute="observation_id",
        field_name="observations",
    )
    if not entropy_terms:
        return EntropyBalanceResult(
            status=EntropyEligibilityStatus.UNEVALUABLE,
            generated_entropy=None,
            uncertainty=None,
            entropy_unit=None,
            reason_codes=("ENTROPY_TERM_INVENTORY_EMPTY",),
        )
    if any(value.system_id != absolute_energy_balance.system_id for value in entropy_terms):
        raise ValueError("entropy terms and energy balance belong to different systems")
    if any(
        value.system_id != absolute_energy_balance.system_id
        or value.independent_unit_id != absolute_energy_balance.independent_unit_id
        or value.episode_id != absolute_energy_balance.episode_id
        or value.word_id != absolute_energy_balance.word_id
        for value in observations
    ):
        raise ValueError("entropy observations and energy balance identities differ")
    if absolute_energy_balance.status is not BalanceClosureStatus.CLOSED:
        return EntropyBalanceResult(
            status=EntropyEligibilityStatus.LEDGER_PARTIAL,
            generated_entropy=None,
            uncertainty=None,
            entropy_unit=None,
            reason_codes=("ABSOLUTE_ENERGY_BALANCE_NOT_CLOSED",),
        )
    expected_ids = {value.term_id for value in entropy_terms}
    by_term = {value.term_id: value for value in observations}
    if set(by_term) - expected_ids:
        raise ValueError("entropy balance contains an undeclared term")
    if len(by_term) != len(observations):
        raise ValueError("entropy balance contains duplicate observations for one term")
    missing = expected_ids - set(by_term)
    if missing or any(
        by_term[value].status is not ThermodynamicObservationStatus.OBSERVED
        for value in expected_ids - missing
    ):
        return EntropyBalanceResult(
            status=EntropyEligibilityStatus.LEDGER_PARTIAL,
            generated_entropy=None,
            uncertainty=None,
            entropy_unit=None,
            reason_codes=("ENTROPY_TERM_UNOBSERVED",),
        )
    audit = {
        "CHEMICAL_TERMS_INCOMPLETE": chemical_terms_complete,
        "CONSTITUTIVE_ASSUMPTIONS_INVALID": constitutive_assumptions_valid,
        "LOCAL_EQUILIBRIUM_UNJUSTIFIED": local_equilibrium_justified,
        "SYNCHRONIZATION_INVALID": synchronization_valid,
        "SYSTEM_BOUNDARY_INCOMPLETE": system_boundary_complete,
    }
    failures = tuple(sorted(reason for reason, passed in audit.items() if not passed))
    if failures:
        return EntropyBalanceResult(
            status=EntropyEligibilityStatus.ENERGY_BALANCE_ONLY,
            generated_entropy=None,
            uncertainty=None,
            entropy_unit=None,
            reason_codes=failures,
        )
    state_ids = {
        value.term_id for value in entropy_terms if value.role is EntropyTermRole.STATE_ENTROPY
    }
    if len(state_ids) != 1:
        return EntropyBalanceResult(
            status=EntropyEligibilityStatus.UNEVALUABLE,
            generated_entropy=None,
            uncertainty=None,
            entropy_unit=None,
            reason_codes=("STATE_ENTROPY_TERM_NOT_UNIQUE",),
        )
    state = by_term[next(iter(state_ids))]
    transfers = tuple(by_term[value] for value in expected_ids - state_ids)
    if (
        state.coordinate_role is not ThermodynamicCoordinateRole.STATE_ENTROPY
        or state.estimand_kind is not ExchangeEstimandKind.ABSOLUTE_STATE_ENTROPY_CHANGE
    ):
        raise ValueError("state entropy observation has the wrong ledger role")
    permitted_transfer_roles = {
        ThermodynamicCoordinateRole.HEAT_ENTROPY_TRANSFER,
        ThermodynamicCoordinateRole.MATTER_ENTROPY_TRANSFER,
        ThermodynamicCoordinateRole.OTHER_BOUNDARY_TRANSFER,
    }
    if any(
        value.coordinate_role not in permitted_transfer_roles
        or value.estimand_kind is not ExchangeEstimandKind.TOTAL_EXCHANGE
        for value in transfers
    ):
        raise ValueError("entropy transfer observation has the wrong ledger role")
    observed = (state, *transfers)
    role_by_term = {value.term_id: value.role for value in entropy_terms}
    expected_coordinate_roles = {
        EntropyTermRole.STATE_ENTROPY: ThermodynamicCoordinateRole.STATE_ENTROPY,
        EntropyTermRole.HEAT_OVER_BOUNDARY_TEMPERATURE: (
            ThermodynamicCoordinateRole.HEAT_ENTROPY_TRANSFER
        ),
        EntropyTermRole.MATTER_ENTROPY_TRANSFER: (
            ThermodynamicCoordinateRole.MATTER_ENTROPY_TRANSFER
        ),
    }
    if any(
        role_by_term[value.term_id] in expected_coordinate_roles
        and value.coordinate_role is not expected_coordinate_roles[role_by_term[value.term_id]]
        for value in observed
    ):
        raise ValueError("entropy observation differs from its declared term role")
    if len({value.physical_event_id for value in transfers}) != len(transfers):
        raise ValueError("entropy balance counts one transfer event more than once")
    units = {value.canonical_unit for value in observed}
    if len(units) != 1:
        raise ValueError("entropy ledger native units differ")
    if any(value.canonical_value is None or value.uncertainty is None for value in observed):
        raise ValueError("observed entropy ledger lacks values or uncertainty")
    if state.canonical_value is None:
        raise ValueError("observed state entropy lacks a canonical value")
    generated = state.canonical_value - sum(
        (value.canonical_value for value in transfers if value.canonical_value is not None),
        start=Decimal("0"),
    )
    uncertainty = sum(
        (value.uncertainty for value in observed if value.uncertainty is not None),
        start=Decimal("0"),
    )
    return EntropyBalanceResult(
        status=EntropyEligibilityStatus.ELIGIBLE,
        generated_entropy=generated,
        uncertainty=uncertainty,
        entropy_unit=next(iter(units)),
        reason_codes=(),
    )


def cycle_exchange(
    *,
    identity_word_id: str,
    word_observations: tuple[ThermodynamicLedgerObservation, ...],
    identity_observations: tuple[ThermodynamicLedgerObservation, ...],
) -> CycleExchangeResult:
    """Retain absolute cycle/identity ledgers and derive, never substitute, their difference."""

    for values, name in (
        (word_observations, "word_observations"),
        (identity_observations, "identity_observations"),
    ):
        require_sorted_unique_ids(values, attribute="observation_id", field_name=name)
        if any(
            value.status is not ThermodynamicObservationStatus.OBSERVED
            or value.estimand_kind is not ExchangeEstimandKind.TOTAL_EXCHANGE
            for value in values
        ):
            raise ValueError("cycle exchange requires observed absolute total exchange")
    word_by_term = {value.term_id: value for value in word_observations}
    identity_by_term = {value.term_id: value for value in identity_observations}
    if any(value.word_id != identity_word_id for value in identity_observations):
        raise ValueError("identity exchange observations do not use the declared identity word")
    if any(value.word_id == identity_word_id for value in word_observations):
        raise ValueError("cycle exchange word cannot be the identity word")
    if set(word_by_term) != set(identity_by_term):
        raise ValueError("cycle and identity ledgers have different exchange terms")
    differences = []
    for term_id in sorted(word_by_term):
        word = word_by_term[term_id]
        identity = identity_by_term[term_id]
        if (
            word.canonical_unit != identity.canonical_unit
            or word.coordinate_role is not identity.coordinate_role
            or word.system_id != identity.system_id
            or word.independent_unit_id != identity.independent_unit_id
        ):
            raise ValueError("cycle and identity exchange terms are not matched")
        if word.canonical_value is None or identity.canonical_value is None:
            raise ValueError("observed cycle exchange lacks canonical values")
        differences.append(
            ThermodynamicLedgerObservation(
                observation_id=f"difference-{word.observation_id}",
                system_id=word.system_id,
                independent_unit_id=word.independent_unit_id,
                episode_id=word.episode_id,
                word_id=word.word_id,
                term_id=word.term_id,
                physical_event_id=f"difference-{word.physical_event_id}",
                coordinate_role=word.coordinate_role,
                estimand_kind=ExchangeEstimandKind.IDENTITY_DIFFERENCED_EXCHANGE,
                source_native_value=word.canonical_value - identity.canonical_value,
                source_native_unit=word.canonical_unit,
                source_native_positive_direction="canonical-identity-difference",
                canonical_value=word.canonical_value - identity.canonical_value,
                canonical_unit=word.canonical_unit,
                canonical_sign_convention=word.canonical_sign_convention,
                canonical_sign_multiplier=1,
                sign_transform_rule_id="identity-difference-in-canonical-sign",
                uncertainty=(word.uncertainty or Decimal("0"))
                + (identity.uncertainty or Decimal("0")),
                matched_identity_observation_id=identity.observation_id,
                decomposition_model_id=None,
                status=ThermodynamicObservationStatus.OBSERVED,
                reason_codes=(),
            )
        )
    return CycleExchangeResult(
        word_observations=word_observations,
        identity_observations=identity_observations,
        identity_differenced_observations=tuple(differences),
    )


def validate_method_contract(
    method_input: ThermodynamicResponseMethodInput,
    config: ThermodynamicResponseMethodConfig,
) -> None:
    """Validate one analysis input/config pairing without accessing outcomes."""

    if config.identity_word_id != method_input.panel.identity_word_id:
        raise ValueError("method config identity differs from its frozen panel")
    if len(method_input.panel.independent_unit_ids) < config.minimum_independent_units:
        raise ValueError("method input has too few physical independent units")
    floor_ids = tuple(value.value_id for value in config.numerical_floor)
    if set(floor_ids) != set(method_input.panel.coordinate_ids):
        raise ValueError("method numerical floors differ from panel coordinates")
    floor_by_id = {value.value_id: value.unit for value in config.numerical_floor}
    if any(
        floor_by_id[coordinate_id] != unit
        for coordinate_id, unit in zip(
            method_input.panel.coordinate_ids,
            method_input.panel.native_units,
            strict=True,
        )
    ):
        raise ValueError("method numerical-floor units differ from panel native units")


def qualify_grouped_return(
    *,
    qualification_id: str,
    system_id: str,
    independent_unit_id: str,
    episode_id: str,
    word_id: str,
    recovery_horizon_id: str,
    recovery_horizon_predeclared: bool,
    state_residuals: tuple[NamedDecimal, ...],
    bath_residuals: tuple[NamedDecimal, ...],
    component_tolerances: tuple[NamedDecimal, ...],
    component_uncertainties: tuple[NamedDecimal, ...],
    support_valid: bool,
    delivery_valid: bool,
    interlock_clear: bool,
    data_complete: bool,
    balance_assessment: BalanceClosureAssessment,
    unobserved_coordinate_ids: tuple[str, ...],
) -> ReturnQualification:
    """Adjudicate state-and-bath return for one physical unit as one group."""

    if (
        balance_assessment.system_id != system_id
        or balance_assessment.independent_unit_id != independent_unit_id
        or balance_assessment.episode_id != episode_id
        or balance_assessment.word_id != word_id
    ):
        raise ValueError("grouped return and energy balance identities differ")
    residuals = (*state_residuals, *bath_residuals)
    residual_by_id = {value.value_id: value for value in residuals}
    tolerance_by_id = {value.value_id: value for value in component_tolerances}
    uncertainty_by_id = {value.value_id: value for value in component_uncertainties}
    if len(residual_by_id) != len(residuals) or not residuals:
        raise ValueError("grouped return coordinates must be nonempty and unique")
    if set(residual_by_id) != set(tolerance_by_id) or set(residual_by_id) != set(uncertainty_by_id):
        raise ValueError("grouped return residual, tolerance and uncertainty IDs differ")
    for coordinate_id, residual in residual_by_id.items():
        if not (
            residual.unit
            == tolerance_by_id[coordinate_id].unit
            == uncertainty_by_id[coordinate_id].unit
        ):
            raise ValueError("grouped return native units differ")

    gate_reasons = {
        "BALANCE_NOT_CLOSED": balance_assessment.status is BalanceClosureStatus.CLOSED,
        "DATA_INCOMPLETE": data_complete,
        "DELIVERY_INVALID": delivery_valid,
        "INTERLOCK_NOT_CLEAR": interlock_clear,
        "RECOVERY_HORIZON_NOT_PREDECLARED": recovery_horizon_predeclared,
        "SUPPORT_INVALID": support_valid,
    }
    reasons = {reason for reason, passed in gate_reasons.items() if not passed}
    if unobserved_coordinate_ids:
        reasons.add("RETURN_COORDINATE_UNOBSERVED")
    if reasons:
        status = ReturnQualificationStatus.UNEVALUABLE
    else:
        upper_pass = all(
            abs(residual.value) + uncertainty_by_id[coordinate_id].value
            <= tolerance_by_id[coordinate_id].value
            for coordinate_id, residual in residual_by_id.items()
        )
        decisive_fail = any(
            abs(residual.value) - uncertainty_by_id[coordinate_id].value
            > tolerance_by_id[coordinate_id].value
            for coordinate_id, residual in residual_by_id.items()
        )
        if upper_pass:
            status = ReturnQualificationStatus.RETURN_QUALIFIED
        elif decisive_fail:
            status = ReturnQualificationStatus.NON_RETURNING
            reasons.add("RETURN_RESIDUAL_EXCEEDS_TOLERANCE")
        else:
            status = ReturnQualificationStatus.ANNULAR
            reasons.add("RETURN_RESIDUAL_ANNULAR")
    return ReturnQualification(
        qualification_id=qualification_id,
        system_id=system_id,
        independent_unit_id=independent_unit_id,
        episode_id=episode_id,
        word_id=word_id,
        recovery_horizon_id=recovery_horizon_id,
        recovery_horizon_predeclared=recovery_horizon_predeclared,
        state_residuals=state_residuals,
        bath_residuals=bath_residuals,
        component_tolerances=component_tolerances,
        component_uncertainties=component_uncertainties,
        support_valid=support_valid,
        delivery_valid=delivery_valid,
        interlock_clear=interlock_clear,
        data_complete=data_complete,
        balance_assessment_id=balance_assessment.assessment_id,
        balance_status=balance_assessment.status,
        unobserved_coordinate_ids=unobserved_coordinate_ids,
        status=status,
        reason_codes=tuple(sorted(reasons)),
    )


def compose_thermodynamic_response_signature(
    *,
    signature_id: str,
    relation_id: str,
    finite_response_signature: FiniteResponseSignature,
    scalar_dose_projection_witness: ScalarDoseActionWordProjectionWitness | None,
    evidence_world_id: str,
    world_kind: WorldKind,
    denominator_type_id: str,
    bath_type_id: str,
    thermodynamic_coordinate_sufficiency: EntropyEligibilityStatus,
    return_qualification: ReturnQualificationStatus,
    energy_balance_closure: BalanceClosureStatus,
    loop_exchange_observation_ids: tuple[str, ...],
    action_quotient: FiniteAxisDisposition,
    resolution_class: FiniteAxisDisposition,
    explicit_unevaluable_axes: tuple[CompositeSignatureAxis, ...] = (),
    evidence_link_ids: tuple[str, ...] = (),
    reason_codes: tuple[str, ...] = (),
) -> ThermodynamicResponseSignature:
    """Assemble one finite response/thermodynamic signature without promotion."""

    v1_status = (
        scalar_dose_projection_witness.status
        if scalar_dose_projection_witness is not None
        else ScalarDoseActionWordSignatureStatus.UNEVALUABLE
    )
    response_signature = (
        scalar_dose_projection_witness.response_signature if scalar_dose_projection_witness is not None else None
    )
    if (
        thermodynamic_coordinate_sufficiency
        in {
            EntropyEligibilityStatus.ELIGIBLE,
            EntropyEligibilityStatus.ENERGY_BALANCE_ONLY,
        }
        and energy_balance_closure is not BalanceClosureStatus.CLOSED
    ):
        raise ValueError("thermodynamic sufficiency requires closed absolute energy balance")
    if (
        thermodynamic_coordinate_sufficiency is EntropyEligibilityStatus.ELIGIBLE
        and energy_balance_closure is BalanceClosureStatus.CLOSED
    ):
        claim_ceiling = ThermodynamicClaimCeiling.ENTROPY_PRODUCTION_ELIGIBLE
    elif (
        thermodynamic_coordinate_sufficiency is EntropyEligibilityStatus.ENERGY_BALANCE_ONLY
        and energy_balance_closure is BalanceClosureStatus.CLOSED
    ):
        claim_ceiling = ThermodynamicClaimCeiling.ENERGY_BALANCE_ONLY
    elif (
        thermodynamic_coordinate_sufficiency
        is EntropyEligibilityStatus.DISSIPATIVE_CLOSURE_PROXY_ONLY
    ):
        claim_ceiling = ThermodynamicClaimCeiling.DISSIPATIVE_CLOSURE_PROXY_ONLY
    elif thermodynamic_coordinate_sufficiency is EntropyEligibilityStatus.LEDGER_PARTIAL:
        claim_ceiling = ThermodynamicClaimCeiling.THERMODYNAMIC_LEDGER_PARTIAL
    else:
        claim_ceiling = ThermodynamicClaimCeiling.THERMODYNAMIC_INTERPRETATION_UNEVALUABLE

    axes = set(explicit_unevaluable_axes)
    if finite_response_signature.temporal_cocycle is FiniteAxisDisposition.UNEVALUABLE:
        axes.add(CompositeSignatureAxis.TEMPORAL_COCYCLE_CLOSURE)
    if finite_response_signature.equal_lag_stationarity is FiniteAxisDisposition.UNEVALUABLE:
        axes.add(CompositeSignatureAxis.EQUAL_LAG_STATIONARITY)
    if finite_response_signature.receiver_map is ReceiverMapDisposition.UNEVALUABLE:
        axes.add(CompositeSignatureAxis.RECEIVER_NATURALITY_AND_FAITHFULNESS)
    if return_qualification is ReturnQualificationStatus.UNEVALUABLE:
        axes.add(CompositeSignatureAxis.PREPARATION_STATE_AND_BATH_RETURN)
    if energy_balance_closure in {
        BalanceClosureStatus.TERM_UNOBSERVED,
        BalanceClosureStatus.UNEVALUABLE,
    }:
        axes.add(CompositeSignatureAxis.ABSOLUTE_ENERGY_CLOSURE)
    if thermodynamic_coordinate_sufficiency in {
        EntropyEligibilityStatus.LEDGER_PARTIAL,
        EntropyEligibilityStatus.UNEVALUABLE,
    }:
        axes.add(CompositeSignatureAxis.ENTROPY_PRODUCTION_ELIGIBILITY)
    if action_quotient is FiniteAxisDisposition.UNEVALUABLE:
        axes.add(CompositeSignatureAxis.ACTION_IMAGE_QUOTIENT)
    if resolution_class is FiniteAxisDisposition.UNEVALUABLE:
        axes.add(CompositeSignatureAxis.CONSTITUENT_PORT_MATERIALITY)
    reasons = set(reason_codes)
    if scalar_dose_projection_witness is None:
        reasons.add("V1_PROJECTION_UNEVALUATED")
    if axes and not reasons:
        reasons.add("COMPOSITE_AXIS_UNEVALUABLE")
    if return_qualification is not ReturnQualificationStatus.RETURN_QUALIFIED and (
        loop_exchange_observation_ids
    ):
        reasons.add("LOOP_EXCHANGE_IS_OPEN_PATH_NOT_QUALIFIED_CYCLE")
    return ThermodynamicResponseSignature(
        signature_id=signature_id,
        relation_id=relation_id,
        response_signature=response_signature,
        response_signature_status=v1_status,
        scalar_dose_projection_witness=scalar_dose_projection_witness,
        finite_response_signature=finite_response_signature,
        evidence_world_id=evidence_world_id,
        world_kind=world_kind,
        denominator_type_id=denominator_type_id,
        bath_type_id=bath_type_id,
        delivery_qualification=finite_response_signature.delivery_qualification,
        thermodynamic_coordinate_sufficiency=thermodynamic_coordinate_sufficiency,
        return_qualification=return_qualification,
        energy_balance_closure=energy_balance_closure,
        entropy_claim_ceiling=claim_ceiling,
        loop_exchange_observation_ids=loop_exchange_observation_ids,
        receiver_faithfulness=finite_response_signature.receiver_map,
        temporal_composition=finite_response_signature.temporal_cocycle,
        equal_lag_stationarity=finite_response_signature.equal_lag_stationarity,
        action_quotient=action_quotient,
        resolution_class=resolution_class,
        explicit_unevaluable_axes=tuple(sorted(axes, key=lambda value: value.value)),
        evidence_link_ids=evidence_link_ids,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    "CycleExchangeResult",
    "EntropyBalanceResult",
    "FittedAffineMap",
    "GeneralizedGeometryResult",
    "GroupedFiniteContrast",
    "IdentityCoherenceResult",
    "IncompleteBlockContrastResult",
    "ParenthesizationCoherenceResult",
    "ReceiverNaturalityResult",
    "TemporalPropagationResult",
    "ThermodynamicResponseMethodConfig",
    "ThermodynamicResponseMethodInput",
    "ThermodynamicResponseMethodResult",
    "TransportAxisComparison",
    "assess_entropy_balance",
    "build_absolute_energy_balance",
    "build_structural_transport_witness",
    "commuting_square_residual",
    "compose_thermodynamic_response_signature",
    "controlled_order_contrast",
    "cycle_exchange",
    "evaluate_finite_contrast",
    "estimate_incomplete_block_contrast",
    "fit_affine_map",
    "generalized_information_geometry",
    "identity_coherence",
    "parenthesization_coherence",
    "qualify_grouped_return",
    "receiver_naturality",
    "temporal_propagation",
    "validate_method_contract",
]
