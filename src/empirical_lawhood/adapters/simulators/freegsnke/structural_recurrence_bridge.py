'FreeGSNKE compatibility bridge into the frozen margin structural recurrence forecast method records.\n\nThe bridge is deliberately target owned.  It reduces compact native records;\nit never opens simulator artifacts and never changes the structural recurrence predictor.  Every\nmapping/denominator alternative receives a byte-exact structural recurrence target design before\ndevelopment, so development may select an existing design but cannot author a\npost-outcome compatibility rescue.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from statistics import median
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateDenominatorCandidate, IndependentSubstrateTargetMappingCandidate, IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceNativeThreshold, StructuralRecurrenceQualificationBranch, StructuralRecurrenceSplitRoster, StructuralRecurrenceSourceQualification, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage, StructuralRecurrenceUnitObservation
from empirical_lawhood.adapters.methods.structural_recurrence import UNIVERSAL_ROLES, NativeRoleBinding, NativeRoleCompatibilityRecord, TargetLevel, TargetSlotRecord, UniversalRole
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .contracts import FreeGsnkeBranchKind, FreeGsnkePhase
from .design import FreeGsnkeActionDesign
from .target_analysis import FreeGsnkeBranchReduction, FreeGsnkePhaseReduction, FreeGsnkePreparationReduction, FreeGsnkeTargetReductionConfig
from .target_design import FreeGsnkePowerFreeze, FreeGsnkeTargetDesignFreeze


FREEGSNKE_REQUIRED_STRUCTURAL_RECURRENCE_THRESHOLD_IDS = (
    "denominator-common-max",
    "denominator-view-local-max",
    "direct-composed-max",
    "effort-max",
    "history-close-max",
    "realization-error-max",
    "sink-margin-min",
    "target-effect-min",
    "timing-offset-max",
    "unit-pass-rate-min",
)


@dataclass(frozen=True, slots=True)
class FreeGsnkeStructuralRecurrenceActionBinding(CanonicalRecord):
    'One atomic target branch and its normalized structural recurrence target-effect coordinate.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-structural-recurrence-action-binding'

    binding_id: str
    branch_id: str
    structural_recurrence_action_id: str
    target_coordinate_id: str
    target_horizon_s: Decimal
    target_native_unit: str
    response_direction: Decimal
    target_materiality_scale: Decimal

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "branch_id",
            'structural_recurrence_action_id',
            "target_coordinate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(
            self.target_horizon_s,
            field_name="target_horizon_s",
            minimum=Decimal(0),
        )
        validate_nonempty(self.target_native_unit, field_name="target_native_unit")
        validate_decimal(self.response_direction, field_name="response_direction")
        if self.response_direction not in {Decimal(-1), Decimal(1)}:
            raise ValueError("FreeGSNKE target response direction must be signed")
        validate_decimal(
            self.target_materiality_scale,
            field_name="target_materiality_scale",
            minimum=Decimal(0),
        )
        if self.target_materiality_scale == 0:
            raise ValueError("FreeGSNKE target materiality scale must be positive")


@dataclass(frozen=True, slots=True)
class FreeGsnkeStructuralRecurrenceSinkGate(CanonicalRecord):
    """One native sink/current boundary retained as a normalized signed margin."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-structural-recurrence-sink-gate'

    gate_id: str
    coordinate_id: str
    horizon_s: Decimal
    native_unit: str
    direction: str
    boundary_value: Decimal
    normalization_scale: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.direction not in {"AT_LEAST", "AT_MOST", "ABS_AT_MOST"}:
            raise ValueError("FreeGSNKE sink-gate direction differs")
        validate_decimal(self.boundary_value, field_name="boundary_value")
        validate_decimal(
            self.normalization_scale,
            field_name="normalization_scale",
            minimum=Decimal(0),
        )
        if self.normalization_scale == 0:
            raise ValueError("FreeGSNKE sink-gate scale must be positive")


@dataclass(frozen=True, slots=True)
class FreeGsnkeStructuralRecurrenceCompositionBinding(CanonicalRecord):
    """One predeclared joint-versus-component local composition comparison."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-structural-recurrence-composition-binding'

    binding_id: str
    joint_branch_id: str
    component_branch_ids: tuple[str, ...]
    coordinate_id: str
    horizon_s: Decimal
    native_unit: str
    normalization_scale: Decimal

    def __post_init__(self) -> None:
        for name in ("binding_id", "joint_branch_id", "coordinate_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.component_branch_ids,
            field_name="component_branch_ids",
            allow_empty=False,
        )
        if len(self.component_branch_ids) < 2 or self.joint_branch_id in (
            self.component_branch_ids
        ):
            raise ValueError("FreeGSNKE composition binding requires distinct components")
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(
            self.normalization_scale,
            field_name="normalization_scale",
            minimum=Decimal(0),
        )
        if self.normalization_scale == 0:
            raise ValueError("FreeGSNKE composition scale must be positive")


@dataclass(frozen=True, slots=True)
class FreeGsnkeDenominatorStratumProjection(CanonicalRecord):
    'One outcome-blind native-to-candidate denominator-stratum map.\n\n    Denominator alternatives are scientific transformations, not labels.  A\n    merge/omission candidate must actually collapse native strata, while a\n    proposed or split candidate must retain its prospectively frozen\n    partition.  The projection is therefore part of each predevelopment structural recurrence\n    design candidate and is applied before either D or H distances are built.\n    '

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-denominator-stratum-projection'

    projection_id: str
    native_denominator_stratum_id: str
    candidate_denominator_stratum_id: str

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "native_denominator_stratum_id",
            "candidate_denominator_stratum_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class FreeGsnkeStructuralRecurrenceDesignCandidate(CanonicalRecord):
    """One operational predevelopment design for a mapping/denominator pair."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-structural-recurrence-design-candidate'

    candidate_id: str
    mapping_candidate_id: str
    denominator_candidate_id: str
    denominator_stratum_projections: tuple[FreeGsnkeDenominatorStratumProjection, ...]
    denominator_exchange_claim_bearing: bool
    action_bindings: tuple[FreeGsnkeStructuralRecurrenceActionBinding, ...]
    sink_gates: tuple[FreeGsnkeStructuralRecurrenceSinkGate, ...]
    composition_bindings: tuple[FreeGsnkeStructuralRecurrenceCompositionBinding, ...]
    history_checkpoint_horizon_s: Decimal
    history_future_horizon_s: Decimal
    design: StructuralRecurrenceTargetDesignFreeze

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "mapping_candidate_id",
            "denominator_candidate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.denominator_stratum_projections,
            attribute="projection_id",
            field_name="denominator_stratum_projections",
        )
        if not self.denominator_stratum_projections:
            raise ValueError("FreeGSNKE denominator projection is empty")
        native_ids = tuple(
            value.native_denominator_stratum_id for value in self.denominator_stratum_projections
        )
        if len(native_ids) != len(set(native_ids)):
            raise ValueError("FreeGSNKE denominator projection repeats a native stratum")
        candidate_strata = {
            value.candidate_denominator_stratum_id for value in self.denominator_stratum_projections
        }
        if self.denominator_exchange_claim_bearing != (len(candidate_strata) >= 2):
            raise ValueError("FreeGSNKE denominator exchange claim is not projection-derived")
        require_sorted_unique_ids(
            self.action_bindings,
            attribute="binding_id",
            field_name="action_bindings",
        )
        if (
            len({value.branch_id for value in self.action_bindings}) != len(self.action_bindings)
            or len({value.structural_recurrence_action_id for value in self.action_bindings})
            != len(self.action_bindings)
            or sum(value.structural_recurrence_action_id == "hold" for value in self.action_bindings) != 1
        ):
            raise ValueError('FreeGSNKE candidate branch/structural recurrence action mapping differs')
        require_sorted_unique_ids(
            self.sink_gates,
            attribute="gate_id",
            field_name="sink_gates",
        )
        if not self.sink_gates:
            raise ValueError("FreeGSNKE candidate requires native sink gates")
        require_sorted_unique_ids(
            self.composition_bindings,
            attribute="binding_id",
            field_name="composition_bindings",
        )
        for name in ("history_checkpoint_horizon_s", "history_future_horizon_s"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.history_checkpoint_horizon_s >= self.history_future_horizon_s:
            raise ValueError("FreeGSNKE candidate history horizons must advance")
        if self.design.native_action_ids != tuple(
            sorted(value.structural_recurrence_action_id for value in self.action_bindings)
        ):
            raise ValueError('FreeGSNKE candidate action chart differs from structural recurrence design')
        if not self.design.frozen_before_development:
            raise ValueError('FreeGSNKE structural recurrence candidate was not frozen before development')

    def project_denominator_stratum(self, native_stratum_id: str) -> str:
        try:
            return next(
                value.candidate_denominator_stratum_id
                for value in self.denominator_stratum_projections
                if value.native_denominator_stratum_id == native_stratum_id
            )
        except StopIteration as error:
            raise ValueError("FreeGSNKE denominator projection omits a native stratum") from error


@dataclass(frozen=True, slots=True)
class FreeGsnkeStructuralRecurrenceBridgeFreeze(CanonicalRecord):
    """Complete finite compatibility reducer freeze before development."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-structural-recurrence-bridge-freeze'

    freeze_id: str
    target_design: ObjectIdentity
    action_design: ObjectIdentity
    development_reduction_config: ObjectIdentity
    evaluation_reduction_config: ObjectIdentity
    mapping_candidate_ids: tuple[str, ...]
    denominator_candidate_ids: tuple[str, ...]
    native_denominator_stratum_ids: tuple[str, ...]
    design_candidates: tuple[FreeGsnkeStructuralRecurrenceDesignCandidate, ...]
    hold_branch_id: str
    action_scalarization_rule: str
    denominator_distance_rule: str
    history_distance_rule: str
    failure_distance: Decimal
    target_native_values_pooled: bool
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_stable_id(self.hold_branch_id, field_name="hold_branch_id")
        if self.target_design.object_schema != FreeGsnkeTargetDesignFreeze.SCHEMA:
            raise ValueError("FreeGSNKE bridge target design identity differs")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE bridge action design identity differs")
        for name in ("development_reduction_config", "evaluation_reduction_config"):
            if getattr(self, name).object_schema != FreeGsnkeTargetReductionConfig.SCHEMA:
                raise ValueError(f"FreeGSNKE bridge {name} identity differs")
        for name in ("mapping_candidate_ids", "denominator_candidate_ids"):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        require_sorted_unique_strings(
            self.native_denominator_stratum_ids,
            field_name="native_denominator_stratum_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.design_candidates,
            attribute="candidate_id",
            field_name="design_candidates",
        )
        expected_pairs = {
            (mapping_id, denominator_id)
            for mapping_id in self.mapping_candidate_ids
            for denominator_id in self.denominator_candidate_ids
        }
        observed_pairs = {
            (value.mapping_candidate_id, value.denominator_candidate_id)
            for value in self.design_candidates
        }
        if observed_pairs != expected_pairs or len(observed_pairs) != len(self.design_candidates):
            raise ValueError('FreeGSNKE structural recurrence designs do not cover the finite pair roster')
        for candidate in self.design_candidates:
            projected_native_ids = tuple(
                sorted(
                    value.native_denominator_stratum_id
                    for value in candidate.denominator_stratum_projections
                )
            )
            if projected_native_ids != self.native_denominator_stratum_ids:
                raise ValueError(
                    'FreeGSNKE structural recurrence denominator projection differs from the frozen native roster'
                )
        if self.action_scalarization_rule != "ATOMIC_BRANCH_IDENTITY_FIDELITY_INDICATOR":
            raise ValueError('FreeGSNKE structural recurrence action scalarization differs')
        if self.denominator_distance_rule != (
            "MAX_PAIRWISE_DENOMINATOR_STRATUM_MEDIAN_TARGET_EFFECT_DISTANCE"
        ):
            raise ValueError('FreeGSNKE structural recurrence denominator-distance rule differs')
        if self.history_distance_rule != (
            "MAX_DENOMINATOR_CONDITIONAL_HISTORY_STRATUM_MEDIAN_DISTANCE"
        ):
            raise ValueError('FreeGSNKE structural recurrence history-distance rule differs')
        validate_decimal(
            self.failure_distance,
            field_name="failure_distance",
            minimum=Decimal(0),
        )
        if self.failure_distance == 0:
            raise ValueError("FreeGSNKE bridge failure distance must be positive")
        threshold_fingerprints = set()
        mapping_signatures: dict[str, set[tuple[object, ...]]] = {}
        for candidate in self.design_candidates:
            design = candidate.design
            hold_branch_id = next(
                value.branch_id
                for value in candidate.action_bindings
                if value.structural_recurrence_action_id == "hold"
            )
            if hold_branch_id != self.hold_branch_id:
                raise ValueError("FreeGSNKE candidate hold branch differs")
            signature = (
                tuple(value.fingerprint() for value in candidate.action_bindings),
                tuple(value.fingerprint() for value in candidate.sink_gates),
                tuple(value.fingerprint() for value in candidate.composition_bindings),
                candidate.history_checkpoint_horizon_s,
                candidate.history_future_horizon_s,
            )
            mapping_signatures.setdefault(candidate.mapping_candidate_id, set()).add(signature)
            threshold_fingerprints.add(tuple(value.fingerprint() for value in design.thresholds))
            failure_threshold_ids = (
                "denominator-view-local-max",
                "direct-composed-max",
                "history-close-max",
                "realization-error-max",
                "timing-offset-max",
            )
            if self.failure_distance <= max(
                design.threshold(value) for value in failure_threshold_ids
            ):
                raise ValueError('FreeGSNKE failure distance does not clear structural recurrence thresholds')
        if len(threshold_fingerprints) != 1:
            raise ValueError('FreeGSNKE structural recurrence alternatives change native thresholds')
        if any(len(signatures) != 1 for signatures in mapping_signatures.values()):
            raise ValueError(
                "FreeGSNKE one mapping changes operational semantics across denominators"
            )
        unique_mapping_signatures = {
            next(iter(signatures)) for signatures in mapping_signatures.values()
        }
        if len(unique_mapping_signatures) != len(mapping_signatures):
            raise ValueError("FreeGSNKE distinct mapping candidates are operationally identical")
        if self.target_native_values_pooled:
            raise ValueError("FreeGSNKE native values cannot be pooled")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError('FreeGSNKE structural recurrence bridge crossed development access')

    def candidate(
        self,
        *,
        mapping_candidate_id: str,
        denominator_candidate_id: str,
    ) -> FreeGsnkeStructuralRecurrenceDesignCandidate:
        return next(
            value
            for value in self.design_candidates
            if value.mapping_candidate_id == mapping_candidate_id
            and value.denominator_candidate_id == denominator_candidate_id
        )


_ROLE_NATIVE_UNITS = {
    UniversalRole.ACTION_REALIZATION: "1",
    UniversalRole.AUTHORITY: "1",
    UniversalRole.DYNAMICS: "1",
    UniversalRole.EFFORT: "1",
    UniversalRole.HOLD: "V",
    UniversalRole.HORIZON_CLOCK: "s",
    UniversalRole.NATIVE_ACTION: "V",
    UniversalRole.PREPARED_DENOMINATOR: "freegsnke-preparation",
    UniversalRole.PRESERVATION: "1",
    UniversalRole.REACHABILITY: "1",
    UniversalRole.RECEIVER_SINK: "1",
    UniversalRole.RECEIVER_TARGET: "1",
    UniversalRole.RETAINED_HISTORY: "A",
    UniversalRole.SUPPORT: "1",
    UniversalRole.UNCERTAINTY: "1",
    UniversalRole.VALIDITY: "1",
}


def _native_compatibility(
    *,
    design_id: str,
    target_slot: TargetSlotRecord,
    unavailable_roles: tuple[UniversalRole, ...],
) -> NativeRoleCompatibilityRecord:
    unavailable = set(unavailable_roles)
    bindings = tuple(
        NativeRoleBinding(
            binding_id=(f"binding.{design_id}.{role.value.lower().replace('_', '-')}"),
            native_object_id=(f"native.{design_id}.{role.value.lower().replace('_', '-')}"),
            role=role,
            native_unit=_ROLE_NATIVE_UNITS[role],
            native_direction="TARGET_NATIVE_SIGNED_OR_CATEGORICAL",
            clock_id=(f"clock.{design_id}.{role.value.lower().replace('_', '-')}"),
            missingness_rule="Missing native role remains typed and lowers the ceiling.",
            claim_ceiling=EvidenceCeiling.ADMISSION,
            available=role not in unavailable,
            reuse_compatibility_id=None,
        )
        for role in UNIVERSAL_ROLES
    )
    return NativeRoleCompatibilityRecord(
        map_id=f"compatibility.{design_id}",
        target_slot_id=target_slot.target_slot_id,
        ontology=target_slot.ontology,
        bindings=bindings,
        required_new_role_ids=(),
        map_valid=True,
        reason_codes=(),
    )


def build_freegsnke_structural_recurrence_target_design(
    *,
    design_id: str,
    target_slot: TargetSlotRecord,
    qualification: StructuralRecurrenceSourceQualification,
    mapping_candidate: IndependentSubstrateTargetMappingCandidate,
    denominator_candidate: IndependentSubstrateDenominatorCandidate,
    action_bindings: tuple[FreeGsnkeStructuralRecurrenceActionBinding, ...],
    power_freeze: FreeGsnkePowerFreeze,
    thresholds: tuple[StructuralRecurrenceNativeThreshold, ...],
    selected_source_implementation: ObjectIdentity,
    evaluator_implementation: ObjectIdentity,
    unavailable_roles: tuple[UniversalRole, ...] = (),
) -> StructuralRecurrenceTargetDesignFreeze:
    'Materialize one member of the finite predevelopment structural recurrence design roster.'

    validate_stable_id(design_id, field_name="design_id")
    if mapping_candidate.target_slot is not IndependentSubstrateTargetKind.FREEGSNKE:
        raise ValueError('FreeGSNKE structural recurrence design received another target mapping')
    if denominator_candidate.impossible:
        raise ValueError('an impossible denominator cannot produce a structural recurrence design')
    if target_slot.target_level is not TargetLevel.ADMISSION:
        raise ValueError('FreeGSNKE structural recurrence target slot must request admission')
    if qualification.branch is not StructuralRecurrenceQualificationBranch.ENTER_LAW_QUALIFICATION:
        raise ValueError('FreeGSNKE structural recurrence design requires source qualification')
    if not power_freeze.issue_eligible:
        raise ValueError('FreeGSNKE structural recurrence design requires a jointly powered roster')
    require_sorted_unique_ids(
        action_bindings,
        attribute="binding_id",
        field_name="action_bindings",
    )
    action_ids = tuple(sorted(value.structural_recurrence_action_id for value in action_bindings))
    if len(action_ids) != len(set(action_ids)) or "hold" not in action_ids:
        raise ValueError('FreeGSNKE structural recurrence action bindings require exact hold and unique IDs')
    require_sorted_unique_ids(
        thresholds,
        attribute="threshold_id",
        field_name="thresholds",
    )
    if tuple(value.threshold_id for value in thresholds) != (
        FREEGSNKE_REQUIRED_STRUCTURAL_RECURRENCE_THRESHOLD_IDS
    ) or any(value.native_unit != "1" for value in thresholds):
        raise ValueError('FreeGSNKE structural recurrence threshold roster/normalization differs')
    rosters = (
        StructuralRecurrenceSplitRoster(
            roster_id=f"roster.{design_id}.development",
            stage=StructuralRecurrenceTargetStage.DEVELOPMENT,
            unit_ids=power_freeze.development_unit_ids,
            seeds=power_freeze.development_seeds,
        ),
        StructuralRecurrenceSplitRoster(
            roster_id=f"roster.{design_id}.evaluation",
            stage=StructuralRecurrenceTargetStage.EVALUATION,
            unit_ids=power_freeze.evaluation_unit_ids,
            seeds=power_freeze.evaluation_seeds,
        ),
        StructuralRecurrenceSplitRoster(
            roster_id=f"roster.{design_id}.prospective-validation",
            stage=StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION,
            unit_ids=power_freeze.prospective_validation_unit_ids,
            seeds=power_freeze.prospective_validation_seeds,
        ),
    )
    mapping_by_role = {value.structural_recurrence_role_id: value for value in mapping_candidate.role_bindings}
    return StructuralRecurrenceTargetDesignFreeze(
        design_id=design_id,
        target_slot=target_slot,
        qualification=qualification,
        prepared_denominator=(
            f"{denominator_candidate.candidate_id}:"
            f"{','.join(denominator_candidate.factor_ids)}:"
            f"{','.join(denominator_candidate.stratum_ids)}"
        ),
        retained_history=",".join(mapping_by_role["H"].target_native_role_ids),
        horizon_clock=",".join(mapping_by_role["tau"].target_native_role_ids),
        independent_unit=("one outcome-blind prepared equilibrium/profile/current state"),
        observation_operator=",".join(mapping_by_role["R"].target_native_role_ids),
        native_action_ids=action_ids,
        receiver_ids=power_freeze.required_receiver_ids,
        thresholds=thresholds,
        rosters=rosters,
        compatibility=_native_compatibility(
            design_id=design_id,
            target_slot=target_slot,
            unavailable_roles=unavailable_roles,
        ),
        selected_source_implementation=selected_source_implementation,
        evaluator_implementation=evaluator_implementation,
        maximum_level=TargetLevel.ADMISSION,
        frozen_before_development=True,
        protected_outcome_access_count=0,
    )


def _branch(
    unit: FreeGsnkePreparationReduction,
    branch_id: str,
) -> FreeGsnkeBranchReduction:
    return next(value for value in unit.branches if value.branch_id == branch_id)


def _contrast_value(
    unit: FreeGsnkePreparationReduction,
    *,
    branch_id: str,
    coordinate_id: str,
    horizon_s: Decimal,
) -> Decimal | None:
    return next(
        (
            value.delta
            for value in unit.contrasts
            if value.branch_id == branch_id
            and value.coordinate_id == coordinate_id
            and value.horizon_s == horizon_s
        ),
        None,
    )


def _normalized_effect(
    unit: FreeGsnkePreparationReduction,
    binding: FreeGsnkeStructuralRecurrenceActionBinding,
    *,
    horizon_s: Decimal | None = None,
) -> Decimal | None:
    if binding.structural_recurrence_action_id == "hold":
        return Decimal(0)
    value = _contrast_value(
        unit,
        branch_id=binding.branch_id,
        coordinate_id=binding.target_coordinate_id,
        horizon_s=horizon_s or binding.target_horizon_s,
    )
    if value is None:
        return None
    return binding.response_direction * value / binding.target_materiality_scale


def _pairwise_median_distance(groups: dict[str, list[Decimal]]) -> Decimal | None:
    if len(groups) < 2 or any(not values for values in groups.values()):
        return None
    medians = tuple(median(values) for _, values in sorted(groups.items()))
    return max(abs(left - right) for left, right in combinations(medians, 2))


def _denominator_distance(
    units: tuple[FreeGsnkePreparationReduction, ...],
    action_bindings: tuple[FreeGsnkeStructuralRecurrenceActionBinding, ...],
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
) -> Decimal | None:
    distances = []
    expected_native_strata = {value.denominator_stratum_id for value in units}
    projection_native_strata = {
        value.native_denominator_stratum_id
        for value in design_candidate.denominator_stratum_projections
    }
    if expected_native_strata != projection_native_strata:
        return None
    expected_strata = {
        design_candidate.project_denominator_stratum(value) for value in expected_native_strata
    }
    expected_history_strata = {value.history_stratum_id for value in units}
    if not expected_history_strata:
        return None
    if len(expected_strata) == 1:
        return Decimal(0)
    for binding in action_bindings:
        if binding.structural_recurrence_action_id == "hold":
            continue
        for history_stratum in sorted(expected_history_strata):
            groups: dict[str, list[Decimal]] = {}
            for unit in units:
                if not unit.complete_for_inference or unit.history_stratum_id != history_stratum:
                    continue
                effect = _normalized_effect(unit, binding)
                if effect is not None:
                    groups.setdefault(
                        design_candidate.project_denominator_stratum(unit.denominator_stratum_id),
                        [],
                    ).append(effect)
            if set(groups) != expected_strata:
                return None
            distance = _pairwise_median_distance(groups)
            if distance is None:
                return None
            distances.append(distance)
    return max(distances) if distances else None


def _history_distance(
    units: tuple[FreeGsnkePreparationReduction, ...],
    action_bindings: tuple[FreeGsnkeStructuralRecurrenceActionBinding, ...],
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    *,
    horizon_s: Decimal,
) -> Decimal | None:
    distances = []
    native_denominator_strata = {value.denominator_stratum_id for value in units}
    projection_native_strata = {
        value.native_denominator_stratum_id
        for value in design_candidate.denominator_stratum_projections
    }
    if native_denominator_strata != projection_native_strata:
        return None
    denominator_strata = tuple(
        sorted(
            {
                design_candidate.project_denominator_stratum(value)
                for value in native_denominator_strata
            }
        )
    )
    expected_history_strata = {value.history_stratum_id for value in units}
    if len(denominator_strata) < 1 or len(expected_history_strata) < 2:
        return None
    for binding in action_bindings:
        if binding.structural_recurrence_action_id == "hold":
            continue
        for denominator_stratum in denominator_strata:
            groups: dict[str, list[Decimal]] = {}
            for unit in units:
                if (
                    not unit.complete_for_inference
                    or design_candidate.project_denominator_stratum(unit.denominator_stratum_id)
                    != denominator_stratum
                ):
                    continue
                effect = _normalized_effect(unit, binding, horizon_s=horizon_s)
                if effect is not None:
                    groups.setdefault(unit.history_stratum_id, []).append(effect)
            if set(groups) != expected_history_strata:
                return None
            distance = _pairwise_median_distance(groups)
            if distance is None:
                return None
            distances.append(distance)
    return max(distances) if distances else None


def _sink_margin(
    branch: FreeGsnkeBranchReduction,
    gates: tuple[FreeGsnkeStructuralRecurrenceSinkGate, ...],
    *,
    failure_distance: Decimal,
) -> Decimal:
    points = {(value.coordinate_id, value.horizon_s): value for value in branch.native_points}
    margins = []
    for gate in gates:
        point = points.get((gate.coordinate_id, gate.horizon_s))
        if point is None or point.native_unit != gate.native_unit:
            return -failure_distance
        if gate.direction == "AT_LEAST":
            signed = point.value - gate.boundary_value
        elif gate.direction == "AT_MOST":
            signed = gate.boundary_value - point.value
        else:
            signed = gate.boundary_value - abs(point.value)
        margins.append(signed / gate.normalization_scale)
    return min(margins)


def _composition_distance(
    unit: FreeGsnkePreparationReduction,
    bindings: tuple[FreeGsnkeStructuralRecurrenceCompositionBinding, ...],
    *,
    failure_distance: Decimal,
) -> Decimal:
    if not bindings:
        return Decimal(0)
    distances = []
    for binding in bindings:
        joint = _contrast_value(
            unit,
            branch_id=binding.joint_branch_id,
            coordinate_id=binding.coordinate_id,
            horizon_s=binding.horizon_s,
        )
        components = tuple(
            _contrast_value(
                unit,
                branch_id=branch_id,
                coordinate_id=binding.coordinate_id,
                horizon_s=binding.horizon_s,
            )
            for branch_id in binding.component_branch_ids
        )
        if joint is None or any(value is None for value in components):
            return failure_distance
        distances.append(
            abs(joint - sum((value for value in components if value is not None), Decimal(0)))
            / binding.normalization_scale
        )
    return max(distances)


def _current_bounds_passed(
    branch: FreeGsnkeBranchReduction,
    config: FreeGsnkeTargetReductionConfig,
) -> bool:
    points_by_coordinate: dict[str, list[Decimal]] = {}
    for point in branch.native_points:
        points_by_coordinate.setdefault(point.coordinate_id, []).append(point.value)
    for bound in config.source_current_bounds:
        circuit_id = bound.quantity_id.removesuffix("-current")
        values = points_by_coordinate.get(f"current-{circuit_id}", [])
        if not values or any(
            (bound.lower is not None and value < bound.lower)
            or (bound.upper is not None and value > bound.upper)
            for value in values
        ):
            return False
    return True


def lower_freegsnke_phase_to_structural_recurrence(
    *,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    reduction_config: FreeGsnkeTargetReductionConfig,
    phase_reduction: FreeGsnkePhaseReduction,
    action_design: FreeGsnkeActionDesign,
    execution_authority_verified: bool,
) -> StructuralRecurrenceStageEvidence:
    """Lower every issued preparation/branch, including adverse outcomes."""

    if design_candidate not in bridge.design_candidates:
        raise ValueError('FreeGSNKE structural recurrence design candidate is outside the bridge freeze')
    expected_config = {
        FreeGsnkePhase.DEVELOPMENT: bridge.development_reduction_config,
        FreeGsnkePhase.EVALUATION: bridge.evaluation_reduction_config,
    }.get(phase_reduction.phase)
    if expected_config is None:
        raise ValueError('FreeGSNKE structural recurrence bridge supports development/evaluation only')
    config_identity = ObjectIdentity.from_record(
        reduction_config.config_id,
        reduction_config,
    )
    if (
        expected_config != config_identity
        or phase_reduction.config != config_identity
        or bridge.action_design
        != ObjectIdentity.from_record(action_design.design_id, action_design)
        or reduction_config.action_design != bridge.action_design
    ):
        raise ValueError('FreeGSNKE structural recurrence lowering changed a frozen target operand')
    branch_ids = {value.branch_id for value in action_design.branches}
    if branch_ids != {value.branch_id for value in design_candidate.action_bindings}:
        raise ValueError('FreeGSNKE structural recurrence action bindings differ from the target chart')
    by_branch_spec = {value.branch_id: value for value in action_design.branches}
    hold_spec = by_branch_spec[bridge.hold_branch_id]
    if hold_spec.branch_kind is not FreeGsnkeBranchKind.COMPARATOR:
        raise ValueError('FreeGSNKE structural recurrence hold is not the exact comparator')
    stage = {
        FreeGsnkePhase.DEVELOPMENT: StructuralRecurrenceTargetStage.DEVELOPMENT,
        FreeGsnkePhase.EVALUATION: StructuralRecurrenceTargetStage.EVALUATION,
    }[phase_reduction.phase]
    design = design_candidate.design
    roster = design.roster(stage)
    units_by_id = {value.unit_id: value for value in phase_reduction.units}
    if set(units_by_id) != set(roster.unit_ids):
        raise ValueError('FreeGSNKE structural recurrence evidence differs from the issued roster')
    observed_native_denominator_strata = tuple(
        sorted({value.denominator_stratum_id for value in phase_reduction.units})
    )
    if observed_native_denominator_strata != bridge.native_denominator_stratum_ids:
        raise ValueError('FreeGSNKE structural recurrence evidence changed the frozen native denominator roster')
    denominator_distance = _denominator_distance(
        phase_reduction.units,
        design_candidate.action_bindings,
        design_candidate,
    )
    checkpoint_distance = _history_distance(
        phase_reduction.units,
        design_candidate.action_bindings,
        design_candidate,
        horizon_s=design_candidate.history_checkpoint_horizon_s,
    )
    future_distance = _history_distance(
        phase_reduction.units,
        design_candidate.action_bindings,
        design_candidate,
        horizon_s=design_candidate.history_future_horizon_s,
    )
    failure = bridge.failure_distance
    observations = []
    seed_by_unit = dict(zip(roster.unit_ids, roster.seeds, strict=True))
    for unit_id in roster.unit_ids:
        unit = units_by_id[unit_id]
        composition_distance = _composition_distance(
            unit,
            design_candidate.composition_bindings,
            failure_distance=failure,
        )
        outcomes = []
        for binding in design_candidate.action_bindings:
            branch = _branch(unit, binding.branch_id)
            audit = branch.action_audit
            requested = Decimal(1)
            accepted = Decimal(audit.accepted_observed and audit.requested_accepted_exact)
            applied = Decimal(
                accepted == 1
                and audit.applied_observed
                and audit.accepted_applied_exact
                and not audit.clipping_observed
            )
            realized = Decimal(
                applied == 1
                and audit.realized_observed
                and audit.receiver_realization_clocks_aligned
            )
            effect = _normalized_effect(unit, binding)
            target_effect = effect if effect is not None else Decimal(0)
            sink_margin = _sink_margin(
                branch,
                design_candidate.sink_gates,
                failure_distance=failure,
            )
            branch_spec = by_branch_spec[binding.branch_id]
            effort = max(
                abs(branch_spec.p4_dose_fraction),
                abs(branch_spec.p5_dose_fraction),
            )
            timing_passed = all(
                (
                    audit.temporal_order_passed,
                    audit.receiver_realization_clocks_aligned,
                    audit.causal_cutoff_passed,
                )
            )
            current_bounds_passed = _current_bounds_passed(branch, reduction_config)
            target_passed = binding.structural_recurrence_action_id == "hold" or target_effect >= 1
            support = unit.complete_for_inference and target_passed
            preservation = branch.analysis_grid_complete and sink_margin >= 0
            reachability = bool(applied) and current_bounds_passed
            reasons = {
                *branch.reason_codes,
                *(() if effect is not None else ("TARGET_EFFECT_UNOBSERVED",)),
                *(
                    ()
                    if denominator_distance is not None
                    else ("DENOMINATOR_CONTRAST_UNEVALUABLE",)
                ),
                *(
                    ()
                    if checkpoint_distance is not None
                    else ("HISTORY_CHECKPOINT_CONTRAST_UNEVALUABLE",)
                ),
                *(() if future_distance is not None else ("HISTORY_FUTURE_CONTRAST_UNEVALUABLE",)),
                *(() if current_bounds_passed else ("SOURCE_CURRENT_BOUND_FAILED_OR_UNOBSERVED",)),
                *(() if execution_authority_verified else ("EXECUTION_AUTHORITY_UNVERIFIED",)),
                *(() if support else ("TARGET_LOCAL_SUPPORT_FAILED",)),
            }
            outcomes.append(
                StructuralRecurrenceActionOutcome(
                    outcome_id=f"{unit_id}.{binding.structural_recurrence_action_id}",
                    action_id=binding.structural_recurrence_action_id,
                    requested_action=requested,
                    accepted_action=accepted,
                    applied_action=applied,
                    realized_action=realized,
                    target_effect=target_effect,
                    sink_margin=sink_margin,
                    effort=effort,
                    realization_error=max(
                        abs(requested - accepted),
                        abs(accepted - applied),
                        abs(applied - realized),
                    ),
                    denominator_distance=(
                        denominator_distance if denominator_distance is not None else failure
                    ),
                    history_checkpoint_distance=(
                        checkpoint_distance if checkpoint_distance is not None else failure
                    ),
                    history_future_distance=(
                        future_distance if future_distance is not None else failure
                    ),
                    direct_composed_distance=composition_distance,
                    timing_offset=Decimal(0) if timing_passed else failure,
                    support_preserved=support,
                    validity_passed=branch.analysis_grid_complete,
                    preservation_passed=preservation,
                    dynamics_passed=branch.analysis_grid_complete,
                    reachability_passed=reachability,
                    authority_passed=execution_authority_verified,
                    uncertainty_evaluable=(
                        unit.complete_for_inference and not phase_reduction.panel_envelope_limited
                    ),
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        observations.append(
            StructuralRecurrenceUnitObservation(
                unit_id=unit_id,
                target_slot_id=design.target_slot.target_slot_id,
                stage=stage,
                seed=seed_by_unit[unit_id],
                preparation_fingerprint=unit.preparation.object_fingerprint,
                outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            )
        )
    ordered = tuple(sorted(observations, key=lambda value: value.unit_id))
    return StructuralRecurrenceStageEvidence(
        evidence_id=(
            f"evidence.{design.target_slot.target_slot_id}.{stage.value.lower()}."
            'freegsnke-structural-recurrence-bridge'
        ),
        target_design=ObjectIdentity.from_record(design.design_id, design),
        target_slot_id=design.target_slot.target_slot_id,
        stage=stage,
        units=ordered,
        independent_unit_count=len(ordered),
        nested_action_outcome_count=sum(len(value.outcomes) for value in ordered),
        source_implementation=design.selected_source_implementation,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if stage is StructuralRecurrenceTargetStage.DEVELOPMENT
            else OutcomeAccess.EVALUATION_SEALED
        ),
    )


__all__ = [
    'FREEGSNKE_REQUIRED_STRUCTURAL_RECURRENCE_THRESHOLD_IDS',
    'FreeGsnkeDenominatorStratumProjection',
    'FreeGsnkeStructuralRecurrenceActionBinding',
    'FreeGsnkeStructuralRecurrenceBridgeFreeze',
    'FreeGsnkeStructuralRecurrenceCompositionBinding',
    'FreeGsnkeStructuralRecurrenceDesignCandidate',
    'FreeGsnkeStructuralRecurrenceSinkGate',
    'build_freegsnke_structural_recurrence_target_design',
    'lower_freegsnke_phase_to_structural_recurrence',
]
