"Evidence-derived current admission and reachability contracts.\n\nThe historical geometry path aggregates adapter-authored categorical values.\nThis module instead binds raw typed observations and registered evaluator\nidentities, then checks every categorical value against a deterministic\nderivation.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionOccurrence, OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.admission import (
    AdmissionCellResult,
    AdmissionGateKind,
    AdmissionGateResult,
    AdmissionSet,
    GateStatus,
    ReachabilityResult,
    ReachabilityStatus,
    validate_admission_against_atlas,
    validate_reachability_against_admission,
)
from empirical_lawhood.kernel.atlases import ResponseAtlas
from empirical_lawhood.kernel.causal_contracts import (
    PredicateDirection,
    ReceiverInterval,
    TemporalPredicateAssessment,
    TemporalPredicateSemantics,
)
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.models import ModelMemberLawBinding, ViewModelSetSpec, ModelSetSpec
from empirical_lawhood.kernel.obligations import (
    ComputabilityEvidence,
    ObligationStatus,
    StructuralConvergenceSpec,
)
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    QuantityBound,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ReadinessStatus
from empirical_lawhood.kernel.time import HorizonSpec, InformationCutoff
from empirical_lawhood.planning.geometry import (
    AdmissionCandidateCell,
    AdmissionComparison,
    ReachabilityComparison,
)


def _exact_evidence_map(values: tuple[EvidenceLink, ...]) -> dict[str, EvidenceLink]:
    result: dict[str, EvidenceLink] = {}
    for value in values:
        existing = result.get(value.link_id)
        if existing is not None and existing != value:
            raise ValueError("evidence ID is reused with different exact evidence")
        result[value.link_id] = value
    return result


class GatePredicateKind(StrEnum):
    SCALAR_AT_LEAST = "SCALAR_AT_LEAST"
    SCALAR_AT_MOST = "SCALAR_AT_MOST"
    SCALAR_WITHIN_CLOSED_INTERVAL = "SCALAR_WITHIN_CLOSED_INTERVAL"
    BOOLEAN_EQUALS = "BOOLEAN_EQUALS"
    IDENTITY_EQUALS = "IDENTITY_EQUALS"


@dataclass(frozen=True, slots=True)
class GatePredicateSpec(CanonicalRecord):
    """Exact receiver predicate evaluated by one registered implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/gate-predicate-spec'

    predicate_id: str
    gate_kind: AdmissionGateKind
    receiver_id: str
    direction: PredicateDirection
    quantity_id: str
    protected_interval: ReceiverInterval
    predicate_kind: GatePredicateKind
    lower: NamedDecimal | None
    upper: NamedDecimal | None
    expected_boolean: bool | None
    expected_identity: ObjectIdentity | None
    temporal_semantics: TemporalPredicateSemantics | None
    constraint_ids: tuple[str, ...]
    evaluator: ExecutableReference

    def __post_init__(self) -> None:
        for name, value in (
            ("predicate_id", self.predicate_id),
            ("receiver_id", self.receiver_id),
            ("quantity_id", self.quantity_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.constraint_ids,
            field_name="constraint_ids",
            allow_empty=False,
        )
        if not self.evaluator.deterministic:
            raise ValueError("gate predicate evaluator must be deterministic")
        scalar = self.predicate_kind in {
            GatePredicateKind.SCALAR_AT_LEAST,
            GatePredicateKind.SCALAR_AT_MOST,
            GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
        }
        if scalar:
            if self.expected_boolean is not None or self.expected_identity is not None:
                raise ValueError("scalar predicate cannot carry categorical expectations")
            if self.predicate_kind is GatePredicateKind.SCALAR_AT_LEAST:
                if self.lower is None or self.upper is not None:
                    raise ValueError("at-least predicate requires only a lower bound")
                if self.direction is not PredicateDirection.AT_LEAST:
                    raise ValueError("at-least predicate has the wrong receiver direction")
            elif self.predicate_kind is GatePredicateKind.SCALAR_AT_MOST:
                if self.upper is None or self.lower is not None:
                    raise ValueError("at-most predicate requires only an upper bound")
                if self.direction is not PredicateDirection.AT_MOST:
                    raise ValueError("at-most predicate has the wrong receiver direction")
            else:
                if self.lower is None or self.upper is None:
                    raise ValueError("closed-interval predicate requires two bounds")
                if self.lower.unit != self.upper.unit:
                    raise ValueError("closed-interval bounds use unlike native units")
                if self.lower.value > self.upper.value:
                    raise ValueError("closed-interval predicate bounds are reversed")
                if self.direction is not PredicateDirection.WITHIN_ABSOLUTE_TOLERANCE:
                    raise ValueError("closed-interval predicate has the wrong direction")
        elif self.predicate_kind is GatePredicateKind.BOOLEAN_EQUALS:
            if (
                self.expected_boolean is None
                or self.expected_identity is not None
                or self.lower is not None
                or self.upper is not None
            ):
                raise ValueError("boolean predicate requires only its expected value")
        elif (
            self.expected_identity is None
            or self.expected_boolean is not None
            or self.lower is not None
            or self.upper is not None
        ):
            raise ValueError("identity predicate requires only its expected identity")
        if (
            self.gate_kind is AdmissionGateKind.BASELINE_PRESERVATION
            and self.temporal_semantics is None
        ):
            raise ValueError("baseline-preservation predicate requires temporal semantics")

    @property
    def native_unit(self) -> str | None:
        bound = self.lower if self.lower is not None else self.upper
        return bound.unit if bound is not None else None


class PreservationCompatibilityRule(StrEnum):
    EXACT_SEMANTIC_EQUALITY = "EXACT_SEMANTIC_EQUALITY"


@dataclass(frozen=True, slots=True)
class BaselinePreservationCompatibility(CanonicalRecord):
    "Exact law-qualification obligation to admission preservation-predicate compatibility."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/baseline-preservation-compatibility'

    compatibility_id: str
    gate_predicate: GatePredicateSpec
    obligation: TemporalPredicateAssessment
    rule: PreservationCompatibilityRule
    evaluator: ExecutableReference
    evidence_link_ids: tuple[str, ...]
    compatible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.compatibility_id, field_name="compatibility_id")
        if self.gate_predicate.gate_kind is not AdmissionGateKind.BASELINE_PRESERVATION:
            raise ValueError("baseline compatibility requires a preservation gate")
        if not self.evaluator.deterministic:
            raise ValueError("baseline compatibility evaluator must be deterministic")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        obligation_evidence = tuple(link.link_id for link in self.obligation.evidence_links)
        if self.evidence_link_ids != obligation_evidence:
            raise ValueError("baseline compatibility omits exact obligation evidence")
        threshold_compatible = (
            self.gate_predicate.direction is PredicateDirection.AT_LEAST
            and self.gate_predicate.predicate_kind is GatePredicateKind.SCALAR_AT_LEAST
            and self.gate_predicate.lower == self.obligation.criterion
            and self.gate_predicate.upper is None
        ) or (
            self.gate_predicate.direction is PredicateDirection.AT_MOST
            and self.gate_predicate.predicate_kind is GatePredicateKind.SCALAR_AT_MOST
            and self.gate_predicate.upper == self.obligation.criterion
            and self.gate_predicate.lower is None
        )
        observed = (
            self.gate_predicate.receiver_id == self.obligation.receiver_id
            and self.gate_predicate.direction is self.obligation.direction
            and self.gate_predicate.protected_interval == self.obligation.protected_domain
            and self.gate_predicate.temporal_semantics is self.obligation.semantics
            and threshold_compatible
        )
        if self.compatible != observed:
            raise ValueError("baseline compatibility flag differs from exact semantics")
        expected_reasons = () if observed else ("PRESERVATION_SEMANTICS_MISMATCH",)
        if self.reason_codes != expected_reasons:
            raise ValueError("baseline compatibility reasons differ from exact semantics")


@dataclass(frozen=True, slots=True)
class AtlasGateReceipt(CanonicalRecord):
    """Raw typed gate observation plus its mechanically derived disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-gate-receipt'

    receipt_id: str
    cell_id: str
    model_member_id: str
    numerical_view_id: str
    predicate: GatePredicateSpec
    observed_scalar: NamedDecimal | None
    observed_boolean: bool | None
    observed_identity: ObjectIdentity | None
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    baseline_compatibility: BaselinePreservationCompatibility | None
    status: GateStatus
    margin: NamedDecimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("cell_id", self.cell_id),
            ("model_member_id", self.model_member_id),
            ("numerical_view_id", self.numerical_view_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("gate receipt requires exact input artifacts and evidence")
        if self.evaluator != self.predicate.evaluator:
            raise ValueError("gate receipt uses a foreign evaluator implementation")
        artifact_ids = {artifact.artifact_id for artifact in self.input_artifacts}
        linked_artifact_ids = {
            artifact_id for link in self.evidence_links for artifact_id in link.artifact_ids
        }
        if artifact_ids != linked_artifact_ids:
            raise ValueError("gate receipt artifact corpus differs from its evidence")
        if self.evaluator.payload.artifact_id not in artifact_ids:
            raise ValueError("gate evaluator payload is absent from the input corpus")
        if (
            self.baseline_compatibility is not None
            and self.baseline_compatibility.evaluator.payload.artifact_id not in artifact_ids
        ):
            raise ValueError("baseline compatibility evaluator payload is absent from inputs")
        if self.outcome_access in {
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("outcome-visible inputs cannot derive admission gate truth")
        if self.predicate.gate_kind is AdmissionGateKind.BASELINE_PRESERVATION:
            if (
                self.baseline_compatibility is None
                or self.baseline_compatibility.gate_predicate != self.predicate
                or not self.baseline_compatibility.compatible
            ):
                raise ValueError("baseline-preservation gate lacks exact law qualification compatibility")
        elif self.baseline_compatibility is not None:
            raise ValueError("only baseline preservation may bind law qualification compatibility")
        expected_status, expected_margin, expected_reasons = self._derive()
        if (
            self.status is not expected_status
            or self.margin != expected_margin
            or self.reason_codes != expected_reasons
        ):
            raise ValueError("gate status/margin is not derived from its raw observation")

    def _derive(self) -> tuple[GateStatus, NamedDecimal | None, tuple[str, ...]]:
        supplied = sum(
            value is not None
            for value in (
                self.observed_scalar,
                self.observed_boolean,
                self.observed_identity,
            )
        )
        if supplied == 0:
            return GateStatus.UNEVALUABLE, None, ("GATE_OBSERVATION_UNAVAILABLE",)
        if supplied != 1:
            raise ValueError("gate receipt must contain exactly one observation kind")
        predicate = self.predicate
        margin: NamedDecimal | None = None
        passed: bool
        if predicate.predicate_kind in {
            GatePredicateKind.SCALAR_AT_LEAST,
            GatePredicateKind.SCALAR_AT_MOST,
            GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
        }:
            if self.observed_scalar is None:
                raise ValueError("scalar gate predicate requires a scalar observation")
            if self.observed_scalar.unit != predicate.native_unit:
                raise ValueError("gate observation uses a foreign native unit")
            if predicate.predicate_kind is GatePredicateKind.SCALAR_AT_LEAST:
                if predicate.lower is None:  # pragma: no cover - predicate invariant
                    raise AssertionError("at-least predicate lost its lower bound")
                value = self.observed_scalar.value - predicate.lower.value
            elif predicate.predicate_kind is GatePredicateKind.SCALAR_AT_MOST:
                if predicate.upper is None:  # pragma: no cover - predicate invariant
                    raise AssertionError("at-most predicate lost its upper bound")
                value = predicate.upper.value - self.observed_scalar.value
            else:
                if predicate.lower is None or predicate.upper is None:  # pragma: no cover
                    raise AssertionError("closed-interval predicate lost a bound")
                value = min(
                    self.observed_scalar.value - predicate.lower.value,
                    predicate.upper.value - self.observed_scalar.value,
                )
            margin = NamedDecimal(
                value_id=f"margin.{self.receipt_id}",
                value=value,
                unit=self.observed_scalar.unit,
            )
            passed = value >= 0
        elif predicate.predicate_kind is GatePredicateKind.BOOLEAN_EQUALS:
            if self.observed_boolean is None:
                raise ValueError("boolean gate predicate requires a boolean observation")
            passed = self.observed_boolean is predicate.expected_boolean
        else:
            if self.observed_identity is None:
                raise ValueError("identity gate predicate requires an identity observation")
            passed = self.observed_identity == predicate.expected_identity
        if passed:
            return GateStatus.PASS, margin, ()
        return GateStatus.FAIL, margin, ("GATE_PREDICATE_FAILED",)

    @property
    def coordinate_id(self) -> str:
        return f"{self.cell_id}.{self.model_member_id}.{self.predicate.gate_kind.value.lower()}"


@dataclass(frozen=True, slots=True)
class AtlasAdmissionSpec(CanonicalRecord):
    "Complete admission receipt grid for nominal and robust intersections."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-admission-spec'

    evaluation_id: str
    atlas: ResponseAtlas
    model_set: ViewModelSetSpec
    nominal_model_member_id: str
    receiver_quantity_ids: tuple[str, ...]
    candidate_cells: tuple[AdmissionCandidateCell, ...]
    receipts: tuple[AtlasGateReceipt, ...]
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(
            self.nominal_model_member_id,
            field_name="nominal_model_member_id",
        )
        if self.atlas.world_id != self.model_set.target_world_id:
            raise ValueError("admission model set targets another evidence world")
        if self.nominal_model_member_id not in self.model_set.member_view_ids:
            raise ValueError("nominal admission member is absent from the model set")
        require_sorted_unique_strings(
            self.receiver_quantity_ids,
            field_name="receiver_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.candidate_cells,
            attribute="cell_id",
            field_name="candidate_cells",
        )
        require_sorted_unique_ids(
            self.receipts,
            attribute="receipt_id",
            field_name="receipts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.candidate_cells or not self.evidence_links:
            raise ValueError("admission requires cells and exact evidence")
        expected = {
            f"{cell.cell_id}.{member}.{kind.value.lower()}"
            for cell in self.candidate_cells
            for member in self.model_set.member_view_ids
            for kind in AdmissionGateKind
        }
        if (
            len(self.receipts) != len(expected)
            or {receipt.coordinate_id for receipt in self.receipts} != expected
        ):
            raise ValueError("admission gate-receipt grid is incomplete or contains extras")
        if any(receipt.numerical_view_id != receipt.model_member_id for receipt in self.receipts):
            raise ValueError("gate receipt model member and numerical view differ")
        known_evidence = _exact_evidence_map(self.evidence_links)
        receipt_evidence = _exact_evidence_map(
            tuple(link for receipt in self.receipts for link in receipt.evidence_links)
        )
        if receipt_evidence != known_evidence:
            raise ValueError("admission evidence differs from its exact receipt corpus")
        gap_coordinates = {
            (chart_id, denominator_id)
            for gap in self.atlas.gaps
            for chart_id in gap.chart_ids
            for denominator_id in gap.denominator_cell_ids
        }
        for cell in self.candidate_cells:
            coordinate = (cell.chart_id, cell.denominator_cell_id)
            if coordinate in gap_coordinates:
                raise ValueError("admission candidate occupies an explicit atlas gap")
            if not self.atlas.laws_for_coordinate(*coordinate):
                raise ValueError("admission candidate lies outside atlas law support")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("evidence-derived admission must have an admission ceiling")
        inherited = VisibilityCeiling.most_restrictive(
            *(
                inherited_visibility(
                    tuple(link.visibility_ceiling for link in receipt.evidence_links),
                    receipt.outcome_access,
                )
                for receipt in self.receipts
            )
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("admission visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible evidence cannot establish admission")


def _aggregate_gate_receipts(
    *,
    cell_id: str,
    kind: AdmissionGateKind,
    receipts: tuple[AtlasGateReceipt, ...],
) -> AdmissionGateResult:
    statuses = {receipt.status for receipt in receipts}
    status = (
        GateStatus.FAIL
        if GateStatus.FAIL in statuses
        else (GateStatus.UNEVALUABLE if GateStatus.UNEVALUABLE in statuses else GateStatus.PASS)
    )
    margins = tuple(receipt.margin for receipt in receipts if receipt.margin is not None)
    margin: NamedDecimal | None = None
    gate_id = f"gate.{cell_id}.{kind.value.lower()}"
    if margins:
        units = {value.unit for value in margins}
        if len(units) != 1:
            raise ValueError("gate receipts use unlike native margin units")
        margin = NamedDecimal(
            value_id=f"margin.{gate_id}",
            value=min(value.value for value in margins),
            unit=margins[0].unit,
        )
    return AdmissionGateResult(
        gate_id=gate_id,
        kind=kind,
        status=status,
        constraint_ids=tuple(
            sorted(
                {
                    constraint
                    for receipt in receipts
                    for constraint in receipt.predicate.constraint_ids
                }
            )
        ),
        margin=margin,
        reason_codes=(
            ()
            if status is GateStatus.PASS
            else tuple(
                sorted(
                    {
                        reason
                        for receipt in receipts
                        if receipt.status is not GateStatus.PASS
                        for reason in receipt.reason_codes
                    }
                )
            )
        ),
        evidence_link_ids=tuple(
            sorted({link.link_id for receipt in receipts for link in receipt.evidence_links})
        ),
    )


def _derive_atlas_admission_set(
    spec: AtlasAdmissionSpec,
    *,
    member_ids: tuple[str, ...],
    suffix: str,
) -> AdmissionSet:
    cells = []
    for candidate in spec.candidate_cells:
        gates = tuple(
            sorted(
                (
                    _aggregate_gate_receipts(
                        cell_id=candidate.cell_id,
                        kind=kind,
                        receipts=tuple(
                            receipt
                            for receipt in spec.receipts
                            if receipt.cell_id == candidate.cell_id
                            and receipt.model_member_id in member_ids
                            and receipt.predicate.gate_kind is kind
                        ),
                    )
                    for kind in AdmissionGateKind
                ),
                key=lambda gate: gate.gate_id,
            )
        )
        cells.append(
            AdmissionCellResult(
                cell_id=candidate.cell_id,
                denominator_cell_id=candidate.denominator_cell_id,
                chart_id=candidate.chart_id,
                action_bound_ids=candidate.action_bound_ids,
                gates=gates,
                admitted=all(gate.status is GateStatus.PASS for gate in gates),
            )
        )
    cell_tuple = tuple(sorted(cells, key=lambda cell: cell.cell_id))
    admitted = tuple(cell for cell in cell_tuple if cell.admitted)
    status = (
        AdmissionStatus.ADMITTED
        if len(admitted) == len(cell_tuple)
        else (
            AdmissionStatus.PARTIAL
            if admitted
            else (
                AdmissionStatus.UNEVALUABLE
                if any(
                    gate.status is GateStatus.UNEVALUABLE
                    for cell in cell_tuple
                    for gate in cell.gates
                )
                else AdmissionStatus.EMPTY
            )
        )
    )
    admission = AdmissionSet(
        admission_id=f"admission.{spec.evaluation_id}.{suffix}",
        atlas=ObjectIdentity.from_record(spec.atlas.atlas_id, spec.atlas),
        model_set_id=(
            spec.model_set.model_set_id
            if suffix == "robust"
            else f"{spec.model_set.model_set_id}.nominal"
        ),
        receiver_quantity_ids=spec.receiver_quantity_ids,
        cells=cell_tuple,
        status=status,
        evidence_links=spec.evidence_links,
        evidence_ceiling=spec.evidence_ceiling,
        visibility_ceiling=VisibilityCeiling.most_restrictive(
            spec.visibility_ceiling,
            spec.model_set.visibility_ceiling,
            spec.atlas.visibility_ceiling,
        ),
    )
    validate_admission_against_atlas(spec.atlas, admission)
    return admission


def derive_atlas_admission_comparison(
    spec: AtlasAdmissionSpec,
) -> AdmissionComparison:
    "Derive nominal and robust admission intersections from raw receipt bytes."

    nominal = _derive_atlas_admission_set(
        spec,
        member_ids=(spec.nominal_model_member_id,),
        suffix="nominal",
    )
    robust = _derive_atlas_admission_set(
        spec,
        member_ids=spec.model_set.member_view_ids,
        suffix="robust",
    )
    disagreement = tuple(
        sorted(set(nominal.admitted_cell_ids).symmetric_difference(robust.admitted_cell_ids))
    )
    return AdmissionComparison(
        comparison_id=f"admission-comparison.{spec.evaluation_id}",
        evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
        nominal=nominal,
        robust=robust,
        structurally_stable=not disagreement,
        disagreement_cell_ids=disagreement,
    )


@dataclass(frozen=True, slots=True)
class DirectionConstraintReceipt(CanonicalRecord):
    """Complete constraint receipts for one candidate reachable direction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/direction-constraint-receipt'

    direction_id: str
    gate_receipts: tuple[AtlasGateReceipt, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.direction_id, field_name="direction_id")
        require_sorted_unique_ids(
            self.gate_receipts,
            attribute="receipt_id",
            field_name="gate_receipts",
        )
        if not self.gate_receipts:
            raise ValueError("reachable direction requires constraint receipts")
        if any(
            receipt.predicate.gate_kind is not AdmissionGateKind.REACHABILITY
            for receipt in self.gate_receipts
        ):
            raise ValueError("direction constraint uses a non-reachability gate")
        coordinates = {
            (
                receipt.cell_id,
                receipt.model_member_id,
                receipt.numerical_view_id,
            )
            for receipt in self.gate_receipts
        }
        if len(coordinates) != 1:
            raise ValueError("direction constraints cross cell/model/view coordinates")

    @property
    def status(self) -> GateStatus:
        statuses = {receipt.status for receipt in self.gate_receipts}
        if GateStatus.FAIL in statuses:
            return GateStatus.FAIL
        if GateStatus.UNEVALUABLE in statuses:
            return GateStatus.UNEVALUABLE
        return GateStatus.PASS


class ReachabilityRepresentationStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ReachabilityCellDisposition(StrEnum):
    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class AtlasReachabilityReceipt(CanonicalRecord):
    """Registered method output with mechanically derived cell disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-reachability-receipt'

    receipt_id: str
    admission_cell_id: str
    model_member_id: str
    numerical_view_id: str
    admission: ObjectIdentity
    initial_set_id: str
    action_chart_ids: tuple[str, ...]
    dynamics_law_ids: tuple[str, ...]
    horizon: HorizonSpec
    constraint_ids: tuple[str, ...]
    structural_convergence: StructuralConvergenceSpec
    computability: ComputabilityEvidence
    method: ExecutableReference
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    representation_status: ReachabilityRepresentationStatus
    candidate_direction_ids: tuple[str, ...]
    direction_receipts: tuple[DirectionConstraintReceipt, ...]
    independent_basis_direction_ids: tuple[str, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    status: ReachabilityCellDisposition
    viable_direction_rank: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("admission_cell_id", self.admission_cell_id),
            ("model_member_id", self.model_member_id),
            ("numerical_view_id", self.numerical_view_id),
            ("initial_set_id", self.initial_set_id),
        ):
            validate_stable_id(value, field_name=name)
        for field_name, values in (
            ("action_chart_ids", self.action_chart_ids),
            ("dynamics_law_ids", self.dynamics_law_ids),
            ("constraint_ids", self.constraint_ids),
            ("candidate_direction_ids", self.candidate_direction_ids),
            (
                "independent_basis_direction_ids",
                self.independent_basis_direction_ids,
            ),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        require_sorted_unique_ids(
            self.direction_receipts,
            attribute="direction_id",
            field_name="direction_receipts",
        )
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.method.deterministic:
            raise ValueError("reachability method must be deterministic")
        if self.outcome_access in {
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("outcome-visible inputs cannot derive reachability")
        artifact_ids = {artifact.artifact_id for artifact in self.input_artifacts}
        linked_artifact_ids = {
            artifact_id for link in self.evidence_links for artifact_id in link.artifact_ids
        }
        if artifact_ids != linked_artifact_ids:
            raise ValueError("reachability artifacts differ from exact evidence")
        if self.method.payload.artifact_id not in artifact_ids:
            raise ValueError("reachability method payload is absent from its inputs")
        expected: tuple[
            ReachabilityCellDisposition,
            int,
            tuple[str, ...],
        ]
        if self.representation_status is ReachabilityRepresentationStatus.UNAVAILABLE:
            expected = (
                ReachabilityCellDisposition.UNEVALUABLE,
                0,
                ("REACHABILITY_REPRESENTATION_UNAVAILABLE",),
            )
            if (
                self.candidate_direction_ids
                or self.direction_receipts
                or self.independent_basis_direction_ids
            ):
                raise ValueError("unavailable representation cannot impute directions")
        else:
            if not self.candidate_direction_ids:
                raise ValueError("available reachability requires candidate directions")
            if {receipt.direction_id for receipt in self.direction_receipts} != set(
                self.candidate_direction_ids
            ):
                raise ValueError("reachability directions lack complete constraint receipts")
            for direction in self.direction_receipts:
                for receipt in direction.gate_receipts:
                    if (
                        receipt.cell_id != self.admission_cell_id
                        or receipt.model_member_id != self.model_member_id
                        or receipt.numerical_view_id != self.numerical_view_id
                    ):
                        raise ValueError("reachability constraint uses a foreign cell/model/view")
                    if (
                        receipt.information_cutoff != self.information_cutoff
                        or receipt.outcome_access is not self.outcome_access
                    ):
                        raise ValueError(
                            "reachability constraint uses a foreign cutoff/access boundary"
                        )
            direction_evidence = _exact_evidence_map(
                tuple(
                    link
                    for direction in self.direction_receipts
                    for receipt in direction.gate_receipts
                    for link in receipt.evidence_links
                )
            )
            if direction_evidence != _exact_evidence_map(self.evidence_links):
                raise ValueError("reachability direction evidence differs from method inputs")
            observed_constraint_ids = {
                constraint
                for direction in self.direction_receipts
                for receipt in direction.gate_receipts
                for constraint in receipt.predicate.constraint_ids
            }
            if observed_constraint_ids != set(self.constraint_ids):
                raise ValueError("reachability method constraints differ from direction receipts")
            viable = {
                receipt.direction_id
                for receipt in self.direction_receipts
                if receipt.status is GateStatus.PASS
            }
            if not set(self.independent_basis_direction_ids) <= viable:
                raise ValueError("reachability basis contains a nonviable direction")
            if viable:
                if not self.independent_basis_direction_ids:
                    raise ValueError("viable reachability requires an independent basis")
                expected = (
                    ReachabilityCellDisposition.REACHABLE,
                    len(self.independent_basis_direction_ids),
                    (),
                )
            elif any(
                receipt.status is GateStatus.UNEVALUABLE for receipt in self.direction_receipts
            ):
                expected = (
                    ReachabilityCellDisposition.UNEVALUABLE,
                    0,
                    ("REACHABILITY_DIRECTION_UNEVALUABLE",),
                )
            else:
                expected = (
                    ReachabilityCellDisposition.UNREACHABLE,
                    0,
                    ("REACHABILITY_NO_VIABLE_DIRECTION",),
                )
        if (
            self.status is not expected[0]
            or self.viable_direction_rank != expected[1]
            or self.reason_codes != expected[2]
        ):
            raise ValueError("reachability status/rank is not derived from method receipts")

    @property
    def coordinate_id(self) -> str:
        return f"{self.admission_cell_id}.{self.model_member_id}"


@dataclass(frozen=True, slots=True)
class AtlasReachabilitySpec(CanonicalRecord):
    "Complete registered-method receipt grid for one admission."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-reachability-spec'

    evaluation_id: str
    admission: AdmissionComparison
    model_set: ViewModelSetSpec
    nominal_model_member_id: str
    initial_set_id: str
    action_chart_ids: tuple[str, ...]
    dynamics_law_ids: tuple[str, ...]
    horizon: HorizonSpec
    constraint_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    structural_convergence: StructuralConvergenceSpec
    computability: ComputabilityEvidence
    method: ExecutableReference
    receipts: tuple[AtlasReachabilityReceipt, ...]
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("nominal_model_member_id", self.nominal_model_member_id),
            ("initial_set_id", self.initial_set_id),
        ):
            validate_stable_id(value, field_name=name)
        for field_name, values in (
            ("action_chart_ids", self.action_chart_ids),
            ("dynamics_law_ids", self.dynamics_law_ids),
            ("constraint_ids", self.constraint_ids),
            ("numerical_view_ids", self.numerical_view_ids),
        ):
            require_sorted_unique_strings(
                values,
                field_name=field_name,
                allow_empty=False,
            )
        if self.numerical_view_ids != self.model_set.member_view_ids:
            raise ValueError("reachability views differ from the exact model set")
        if self.nominal_model_member_id not in self.model_set.member_view_ids:
            raise ValueError("nominal reachability member is absent from model set")
        if (
            self.admission.robust.model_set_id != self.model_set.model_set_id
            or self.admission.nominal.model_set_id != f"{self.model_set.model_set_id}.nominal"
        ):
            raise ValueError("reachability admission uses another model set")
        if not set(self.action_chart_ids) <= {
            cell.chart_id for cell in self.admission.robust.cells
        }:
            raise ValueError("reachability action chart lies outside admission")
        require_sorted_unique_ids(
            self.receipts,
            attribute="receipt_id",
            field_name="receipts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        expected = {
            f"{cell.cell_id}.{member}"
            for cell in self.admission.robust.cells
            for member in self.model_set.member_view_ids
        }
        if (
            len(self.receipts) != len(expected)
            or {receipt.coordinate_id for receipt in self.receipts} != expected
        ):
            raise ValueError("reachability receipt grid is incomplete or contains extras")
        admission_identity = ObjectIdentity.from_record(
            self.admission.comparison_id,
            self.admission,
        )
        for receipt in self.receipts:
            if (
                receipt.admission != admission_identity
                or receipt.initial_set_id != self.initial_set_id
                or receipt.action_chart_ids != self.action_chart_ids
                or receipt.dynamics_law_ids != self.dynamics_law_ids
                or receipt.horizon != self.horizon
                or receipt.constraint_ids != self.constraint_ids
                or receipt.structural_convergence != self.structural_convergence
                or receipt.computability != self.computability
                or receipt.method != self.method
                or receipt.numerical_view_id != receipt.model_member_id
            ):
                raise ValueError("reachability method receipt uses foreign frozen operands")
        receipt_evidence = _exact_evidence_map(
            tuple(link for receipt in self.receipts for link in receipt.evidence_links)
        )
        if receipt_evidence != _exact_evidence_map(self.evidence_links):
            raise ValueError("reachability evidence differs from receipt corpus")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("evidence-derived reachability must have an admission ceiling")
        inherited = VisibilityCeiling.most_restrictive(
            *(
                inherited_visibility(
                    tuple(link.visibility_ceiling for link in receipt.evidence_links),
                    receipt.outcome_access,
                )
                for receipt in self.receipts
            )
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("reachability visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible evidence cannot establish reachability")


def _derive_reachability_result(
    spec: AtlasReachabilitySpec,
    *,
    admission: AdmissionSet,
    member_ids: tuple[str, ...],
    suffix: str,
) -> ReachabilityResult:
    reachable: list[str] = []
    ranks: list[int] = []
    unevaluable = False
    for cell in admission.cells:
        if not cell.admitted:
            continue
        receipts = tuple(
            receipt
            for receipt in spec.receipts
            if receipt.admission_cell_id == cell.cell_id and receipt.model_member_id in member_ids
        )
        statuses = {receipt.status for receipt in receipts}
        if statuses == {ReachabilityCellDisposition.REACHABLE}:
            reachable.append(cell.cell_id)
            ranks.append(min(receipt.viable_direction_rank for receipt in receipts))
        elif ReachabilityCellDisposition.UNEVALUABLE in statuses:
            unevaluable = True
    if spec.computability.readiness is not ReadinessStatus.READY:
        status = ReachabilityStatus.COMPUTABILITY_BOUNDARY
    elif spec.structural_convergence.status is not ObligationStatus.SATISFIED:
        status = ReachabilityStatus.UNEVALUABLE
    elif admission.status is AdmissionStatus.UNEVALUABLE:
        status = ReachabilityStatus.UNEVALUABLE
    elif admission.status is AdmissionStatus.EMPTY:
        status = ReachabilityStatus.EMPTY
    elif len(reachable) == len(admission.admitted_cell_ids):
        status = ReachabilityStatus.REACHABLE
    elif reachable:
        status = ReachabilityStatus.PARTIAL
    elif unevaluable:
        status = ReachabilityStatus.UNEVALUABLE
    else:
        status = ReachabilityStatus.EMPTY
    if status in {ReachabilityStatus.REACHABLE, ReachabilityStatus.PARTIAL}:
        reachable_ids = tuple(sorted(reachable))
        rank = min(ranks)
    else:
        reachable_ids = ()
        rank = 0
    result = ReachabilityResult(
        reachability_id=f"reachability.{spec.evaluation_id}.{suffix}",
        admission=ObjectIdentity.from_record(admission.admission_id, admission),
        initial_set_id=spec.initial_set_id,
        action_chart_ids=spec.action_chart_ids,
        dynamics_law_ids=spec.dynamics_law_ids,
        horizon=spec.horizon,
        constraint_ids=spec.constraint_ids,
        numerical_view_ids=spec.numerical_view_ids,
        structural_convergence=spec.structural_convergence,
        computability=spec.computability,
        reachable_cell_ids=reachable_ids,
        viable_direction_rank=rank,
        status=status,
        evidence_links=spec.evidence_links,
        evidence_ceiling=spec.evidence_ceiling,
        visibility_ceiling=VisibilityCeiling.most_restrictive(
            spec.visibility_ceiling,
            spec.model_set.visibility_ceiling,
            admission.visibility_ceiling,
        ),
    )
    validate_reachability_against_admission(admission, result)
    return result


def derive_atlas_reachability_comparison(
    spec: AtlasReachabilitySpec,
) -> ReachabilityComparison:
    """Derive nominal/robust reachable geometry from registered method receipts."""

    nominal = _derive_reachability_result(
        spec,
        admission=spec.admission.nominal,
        member_ids=(spec.nominal_model_member_id,),
        suffix="nominal",
    )
    robust = _derive_reachability_result(
        spec,
        admission=spec.admission.robust,
        member_ids=spec.model_set.member_view_ids,
        suffix="robust",
    )
    disagreement = tuple(
        sorted(set(nominal.reachable_cell_ids).symmetric_difference(robust.reachable_cell_ids))
    )
    return ReachabilityComparison(
        comparison_id=f"reachability-comparison.{spec.evaluation_id}",
        evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
        nominal=nominal,
        robust=robust,
        structurally_stable=(
            not disagreement and nominal.viable_direction_rank == robust.viable_direction_rank
        ),
        disagreement_cell_ids=disagreement,
    )


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionActionFibre(CanonicalRecord):
    """Exact controller binding and current ActionWord occurrence semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receipt-admission-action-fibre'

    action_binding_id: str
    action_word: OccurrenceActionWord
    occurrence_ids: tuple[str, ...]
    required_delivery_stages: tuple[ActionDeliveryStage, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.action_binding_id, field_name="action_binding_id")
        require_sorted_unique_strings(self.occurrence_ids, field_name="occurrence_ids")
        if self.occurrence_ids != tuple(
            sorted(value.occurrence_id for value in self.action_word.occurrences)
        ):
            raise ValueError("admission action fibre occurrence roster differs from its ActionWord")
        if self.required_delivery_stages != tuple(ActionDeliveryStage):
            raise ValueError("admission action fibre requires all four delivery stages in order")
        if self.action_word.support_status is not ActionWordSupportStatus.SUPPORTED:
            raise ValueError("admission action fibre requires a supported current ActionWord")

    @property
    def action_word_identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.action_word.word_id, self.action_word)


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionSupportCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receipt-admission-support-cell'

    cell_id: str
    chart_id: str
    denominator_cell_id: str
    action_bound_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("chart_id", self.chart_id),
            ("denominator_cell_id", self.denominator_cell_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.action_bound_ids,
            field_name="action_bound_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionPlannedCoordinate(CanonicalRecord):
    """One exact member/version/action/support coordinate in the frozen roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receipt-admission-planned-coordinate'

    coordinate_id: str
    law_member_binding: ObjectIdentity
    response_law: ObjectIdentity
    qualification_result: ObjectIdentity
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    action_fibre: ObjectIdentity
    support_cell: ObjectIdentity

    def __post_init__(self) -> None:
        for name, value in (
            ("coordinate_id", self.coordinate_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.qualification_view_ids,
            field_name="qualification_view_ids",
            allow_empty=False,
        )
        if self.law_member_binding.object_schema != ModelMemberLawBinding.SCHEMA:
            raise ValueError("admission coordinate requires a model-member law binding")
        if self.response_law.object_schema != 'empirical-lawhood/kernel/response-law':
            raise ValueError("admission coordinate requires a ResponseLaw identity")
        if self.qualification_result.object_schema != (
            'empirical-lawhood/kernel/law-qualification-result'
        ):
            raise ValueError("admission coordinate requires authoritative law qualification")
        if self.action_fibre.object_schema != ReceiptAdmissionActionFibre.SCHEMA:
            raise ValueError("admission coordinate requires an exact admission action fibre")
        if self.support_cell.object_schema != ReceiptAdmissionSupportCell.SCHEMA:
            raise ValueError("admission coordinate requires an exact admission support cell")

    @property
    def product_key(self) -> tuple[str, str, str, str]:
        return (
            self.denominator_member_id,
            self.candidate_version_id,
            self.action_fibre.object_id,
            self.support_cell.object_id,
        )


class LawEvaluationBindingDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    MEMBER_ABSENT = "MEMBER_ABSENT"
    OPERAND_ABSENT = "OPERAND_ABSENT"
    PRODUCT_NOT_CLAIMED = "PRODUCT_NOT_CLAIMED"
    RESOURCE_REFUSED = "RESOURCE_REFUSED"


@dataclass(frozen=True, slots=True)
class LawMemberEvaluationBinding(CanonicalRecord):
    "Exact law evaluation continuity for one planned admission coordinate."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/law-member-evaluation-binding'

    binding_id: str
    planned_coordinate: ObjectIdentity
    law_member_binding: ObjectIdentity
    response_law: ObjectIdentity
    qualification_result: ObjectIdentity
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_id: str
    action_fibre: ObjectIdentity
    action_word: ObjectIdentity
    support_cell: ObjectIdentity
    law_evaluation_request: ObjectIdentity
    law_evaluation_result: ObjectIdentity | None
    evaluator_implementation: ObjectIdentity | None
    evaluation_disposition: LawEvaluationBindingDisposition | None

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("qualification_view_id", self.qualification_view_id),
        ):
            validate_stable_id(value, field_name=name)
        expected_schemas = (
            (self.planned_coordinate, ReceiptAdmissionPlannedCoordinate.SCHEMA, "planned coordinate"),
            (self.law_member_binding, ModelMemberLawBinding.SCHEMA, "law member"),
            (self.response_law, 'empirical-lawhood/kernel/response-law', "response law"),
            (
                self.qualification_result,
                'empirical-lawhood/kernel/law-qualification-result',
                "qualification result",
            ),
            (self.action_fibre, ReceiptAdmissionActionFibre.SCHEMA, "action fibre"),
            (self.action_word, OccurrenceActionWord.SCHEMA, "action word"),
            (self.support_cell, ReceiptAdmissionSupportCell.SCHEMA, "support cell"),
            (
                self.law_evaluation_request,
                'empirical-lawhood/methods/law-evaluation-request',
                "law evaluation request",
            ),
        )
        for identity, schema, label in expected_schemas:
            if identity.object_schema != schema:
                raise ValueError(f"law evaluation binding requires an exact {label}")
        terminal_values = (
            self.law_evaluation_result,
            self.evaluator_implementation,
            self.evaluation_disposition,
        )
        if all(value is None for value in terminal_values):
            return
        if any(value is None for value in terminal_values):
            raise ValueError("law evaluation binding terminal fields must be all present or absent")
        if self.law_evaluation_result is None:  # pragma: no cover - narrowed above
            raise AssertionError("law evaluation result unexpectedly absent")
        if self.law_evaluation_result.object_schema != (
            'empirical-lawhood/methods/law-evaluation-result'
        ):
            raise ValueError("law evaluation binding requires the exact result schema")


def validate_law_evaluation_binding(
    coordinate: ReceiptAdmissionPlannedCoordinate,
    binding: LawMemberEvaluationBinding,
) -> None:
    """Check binding fields that planning can prove without adapter imports."""

    if (
        binding.planned_coordinate
        != ObjectIdentity.from_record(coordinate.coordinate_id, coordinate)
        or binding.law_member_binding != coordinate.law_member_binding
        or binding.response_law != coordinate.response_law
        or binding.qualification_result != coordinate.qualification_result
        or binding.denominator_member_id != coordinate.denominator_member_id
        or binding.candidate_version_id != coordinate.candidate_version_id
        or binding.qualification_view_id not in coordinate.qualification_view_ids
        or binding.action_fibre != coordinate.action_fibre
        or binding.support_cell != coordinate.support_cell
    ):
        raise ValueError("law evaluation binding rewrites its planned admission coordinate")


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionReceiptProductionPlan(CanonicalRecord):
    """Frozen complete raw-receipt roster and proof/resource ownership."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receipt-admission-receipt-production-plan'

    plan_id: str
    atlas: ResponseAtlas
    qualification_batch: ObjectIdentity
    model_set: ModelSetSpec
    nominal_denominator_member_id: str
    action_fibres: tuple[ReceiptAdmissionActionFibre, ...]
    support_cells: tuple[ReceiptAdmissionSupportCell, ...]
    coordinates: tuple[ReceiptAdmissionPlannedCoordinate, ...]
    gate_predicates: tuple[GatePredicateSpec, ...]
    initial_set_id: str
    horizon: HorizonSpec
    reachability_constraint_ids: tuple[str, ...]
    gate_receipt_producer: ObjectIdentity
    reachability_receipt_producer: ObjectIdentity
    utility_receipt_producer: ObjectIdentity
    gate_resource_envelope: ObjectIdentity
    reachability_resource_envelope: ObjectIdentity
    utility_resource_envelope: ObjectIdentity
    evidence_domain: ObjectIdentity
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    authority_boundary: ObjectIdentity
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("plan_id", self.plan_id),
            ("nominal_denominator_member_id", self.nominal_denominator_member_id),
            ("initial_set_id", self.initial_set_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.qualification_batch.object_schema != (
            'empirical-lawhood/methods/law-qualification-batch'
        ):
            raise ValueError("admission plan requires a LawQualificationBatch identity")
        member_ids = tuple(value.denominator_member_id for value in self.model_set.members)
        if self.nominal_denominator_member_id not in member_ids:
            raise ValueError("nominal admission member is absent from ModelSetSpec")
        if self.model_set.target_world_id != self.atlas.world_id:
            raise ValueError("admission model set and atlas use different evidence worlds")
        atlas_laws = {ObjectIdentity.from_record(value.law_id, value) for value in self.atlas.laws}
        if any(value.response_law not in atlas_laws for value in self.model_set.members):
            raise ValueError("admission model member law is absent from the exact atlas")
        require_sorted_unique_ids(
            self.action_fibres,
            attribute="action_binding_id",
            field_name="action_fibres",
        )
        require_sorted_unique_ids(
            self.support_cells,
            attribute="cell_id",
            field_name="support_cells",
        )
        require_sorted_unique_ids(
            self.coordinates,
            attribute="coordinate_id",
            field_name="coordinates",
        )
        if not self.action_fibres or not self.support_cells:
            raise ValueError("admission receipt plan requires actions and support cells")
        atlas_charts = {value.chart_id for value in self.atlas.laws}
        if any(value.chart_id not in atlas_charts for value in self.support_cells):
            raise ValueError("admission support cell uses a chart outside the exact atlas")
        require_sorted_unique_ids(
            self.gate_predicates,
            attribute="predicate_id",
            field_name="gate_predicates",
        )
        if len(self.gate_predicates) != len(AdmissionGateKind) or {
            value.gate_kind for value in self.gate_predicates
        } != set(AdmissionGateKind):
            raise ValueError("admission receipt plan requires exactly one predicate for every gate")
        require_sorted_unique_strings(
            self.reachability_constraint_ids,
            field_name="reachability_constraint_ids",
            allow_empty=False,
        )
        self._validate_coordinates()
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("admission receipt production must be outcome-blind")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("admission receipt production ceiling must be exactly admission")
        inherited_visibility = VisibilityCeiling.most_restrictive(
            self.atlas.visibility_ceiling,
            self.model_set.visibility_ceiling,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited_visibility):
            raise ValueError("admission receipt-plan visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible inputs cannot produce admission receipts")

    def _validate_coordinates(self) -> None:
        action_identities = {
            value.action_binding_id: ObjectIdentity.from_record(value.action_binding_id, value)
            for value in self.action_fibres
        }
        support_identities = {
            value.cell_id: ObjectIdentity.from_record(value.cell_id, value)
            for value in self.support_cells
        }
        members = {value.denominator_member_id: value for value in self.model_set.members}
        atlas_laws = {
            ObjectIdentity.from_record(value.law_id, value): value for value in self.atlas.laws
        }
        expected = {
            (member.denominator_member_id, version, action_id, cell_id)
            for member in self.model_set.members
            for version in member.candidate_version_ids
            for action_id in action_identities
            for cell_id in support_identities
        }
        if (
            len(self.coordinates) != len(expected)
            or {value.product_key for value in self.coordinates} != expected
        ):
            raise ValueError("admission planned Cartesian coordinate roster is incomplete or has extras")
        for coordinate in self.coordinates:
            member = members[coordinate.denominator_member_id]
            action_fibre = self.action_fibre(coordinate.action_fibre)
            support_identity = support_identities.get(coordinate.support_cell.object_id)
            law = atlas_laws.get(coordinate.response_law)
            if (
                coordinate.law_member_binding
                != ObjectIdentity.from_record(member.binding_id, member)
                or coordinate.response_law != member.response_law
                or coordinate.qualification_result != member.qualification_result
                or coordinate.candidate_version_id not in member.candidate_version_ids
                or coordinate.qualification_view_ids != member.qualification_view_ids
                or coordinate.action_fibre != action_identities[coordinate.action_fibre.object_id]
                or coordinate.support_cell != support_identity
                or law is None
            ):
                raise ValueError("admission coordinate rewrites its model/action/support operands")
            if law is None:  # pragma: no cover - guarded above for type narrowing
                raise AssertionError("admission coordinate lost its atlas law")
            word = action_fibre.action_word
            if (
                word.horizon_id != law.relation.horizon.horizon_id
                or word.receiver_id not in law.relation.receiver_quantity_ids
                or any(
                    occurrence.channel.controller_quantity_id
                    not in law.relation.action_quantity_ids
                    for occurrence in word.occurrences
                )
            ):
                raise ValueError("admission action word differs from its law H/A/R/tau binding")

    def action_fibre(self, identity: ObjectIdentity) -> ReceiptAdmissionActionFibre:
        value = next(
            (
                value
                for value in self.action_fibres
                if ObjectIdentity.from_record(value.action_binding_id, value) == identity
            ),
            None,
        )
        if value is None:
            raise ValueError("admission coordinate references an absent or altered action fibre")
        return value

    def support_cell(self, identity: ObjectIdentity) -> ReceiptAdmissionSupportCell:
        value = next(
            (
                value
                for value in self.support_cells
                if ObjectIdentity.from_record(value.cell_id, value) == identity
            ),
            None,
        )
        if value is None:
            raise ValueError("admission coordinate references an absent or altered support cell")
        return value


class ReceiptAdmissionRawDisposition(StrEnum):
    EVALUATED = "EVALUATED"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    AUTHORITY_ABSENT = "AUTHORITY_ABSENT"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    UNEVALUABLE = "UNEVALUABLE"


_ADMISSION_DISPOSITION_REASONS = {
    ReceiptAdmissionRawDisposition.TECHNICAL_INVALID: "ADMISSION_TECHNICAL_INVALID",
    ReceiptAdmissionRawDisposition.EVIDENCE_UNAVAILABLE: "ADMISSION_EVIDENCE_UNAVAILABLE",
    ReceiptAdmissionRawDisposition.AUTHORITY_ABSENT: "ADMISSION_AUTHORITY_ABSENT",
    ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT: "ADMISSION_OUTSIDE_SUPPORT",
    ReceiptAdmissionRawDisposition.UNEVALUABLE: "ADMISSION_UNEVALUABLE",
}


def admission_disposition_reason(disposition: ReceiptAdmissionRawDisposition) -> str:
    if disposition is ReceiptAdmissionRawDisposition.EVALUATED:
        raise ValueError("evaluated admission rows have predicate-derived reasons")
    return _ADMISSION_DISPOSITION_REASONS[disposition]


def derive_admission_gate_observation(
    *,
    receipt_id: str,
    predicate: GatePredicateSpec,
    observed_scalar: NamedDecimal | None,
    observed_boolean: bool | None,
    observed_identity: ObjectIdentity | None,
) -> tuple[GateStatus, NamedDecimal | None, tuple[str, ...]]:
    supplied = sum(
        value is not None for value in (observed_scalar, observed_boolean, observed_identity)
    )
    if supplied != 1:
        raise ValueError("evaluated admission gate requires exactly one observation kind")
    if predicate.predicate_kind in {
        GatePredicateKind.SCALAR_AT_LEAST,
        GatePredicateKind.SCALAR_AT_MOST,
        GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
    }:
        if observed_scalar is None:
            raise ValueError("scalar admission predicate requires a scalar observation")
        if observed_scalar.unit != predicate.native_unit:
            raise ValueError("admission gate observation uses a foreign native unit")
        if predicate.predicate_kind is GatePredicateKind.SCALAR_AT_LEAST:
            if predicate.lower is None:  # pragma: no cover - predicate invariant
                raise AssertionError("at-least admission predicate lost its lower bound")
            raw_margin = observed_scalar.value - predicate.lower.value
        elif predicate.predicate_kind is GatePredicateKind.SCALAR_AT_MOST:
            if predicate.upper is None:  # pragma: no cover - predicate invariant
                raise AssertionError("at-most admission predicate lost its upper bound")
            raw_margin = predicate.upper.value - observed_scalar.value
        else:
            if predicate.lower is None or predicate.upper is None:  # pragma: no cover
                raise AssertionError("interval admission predicate lost a bound")
            raw_margin = min(
                observed_scalar.value - predicate.lower.value,
                predicate.upper.value - observed_scalar.value,
            )
        margin = NamedDecimal(
            value_id=f"margin.{receipt_id}",
            value=raw_margin,
            unit=observed_scalar.unit,
        )
        passed = raw_margin >= 0
    elif predicate.predicate_kind is GatePredicateKind.BOOLEAN_EQUALS:
        if observed_boolean is None:
            raise ValueError("boolean admission predicate requires a boolean observation")
        margin = None
        passed = observed_boolean is predicate.expected_boolean
    else:
        if observed_identity is None:
            raise ValueError("identity admission predicate requires an identity observation")
        margin = None
        passed = observed_identity == predicate.expected_identity
    return (
        (GateStatus.PASS if passed else GateStatus.FAIL),
        margin,
        (() if passed else ("ADMISSION_GATE_PREDICATE_FAILED",)),
    )


def _validate_admission_receipt_evidence(
    *,
    evaluator: ExecutableReference,
    input_artifacts: tuple[ArtifactIdentity, ...],
    evidence_links: tuple[EvidenceLink, ...],
) -> None:
    require_sorted_unique_ids(
        input_artifacts,
        attribute="artifact_id",
        field_name="input_artifacts",
    )
    require_sorted_unique_ids(
        evidence_links,
        attribute="link_id",
        field_name="evidence_links",
    )
    if not input_artifacts or not evidence_links:
        raise ValueError("admission receipt requires evaluator/manifest artifacts and evidence")
    artifact_ids = {value.artifact_id for value in input_artifacts}
    linked_ids = {value for link in evidence_links for value in link.artifact_ids}
    if artifact_ids != linked_ids or evaluator.payload.artifact_id not in artifact_ids:
        raise ValueError("admission receipt artifacts differ from its exact evidence/evaluator")


def _action_occurrence_inside_support(
    occurrence: ActionOccurrence,
    bounds: dict[str, QuantityBound],
) -> bool:
    bound = bounds.get(occurrence.channel.controller_quantity_id)
    if bound is None or occurrence.channel.native_unit != bound.native_unit:
        return False
    return (bound.lower is None or occurrence.realized.value >= bound.lower) and (
        bound.upper is None or occurrence.realized.value <= bound.upper
    )


def admission_coordinate_outside_support(
    plan: ReceiptAdmissionReceiptProductionPlan,
    coordinate: ReceiptAdmissionPlannedCoordinate,
    *,
    finite_evaluation_bindings: tuple[LawMemberEvaluationBinding, ...] = (),
) -> bool:
    "Derive exact L(D,H,A,R,tau) support for one frozen admission coordinate."

    if coordinate not in plan.coordinates:
        raise ValueError("admission support derivation requires an exact planned coordinate")
    from .finite_action_support import NAMESPACE, mapped_coordinate_outside_support

    if any(e.namespace == NAMESPACE for e in plan.model_set.extensions):
        return mapped_coordinate_outside_support(plan, coordinate, finite_evaluation_bindings)
    action = plan.action_fibre(coordinate.action_fibre).action_word
    cell = plan.support_cell(coordinate.support_cell)
    law = next(
        value
        for value in plan.atlas.laws
        if ObjectIdentity.from_record(value.law_id, value) == coordinate.response_law
    )
    bounds = {value.quantity_id: value for value in law.obligations.support.action_bounds}
    return (
        action.denominator_id != cell.denominator_cell_id
        or law.chart_id != cell.chart_id
        or cell.denominator_cell_id not in law.obligations.support.denominator_cell_ids
        or not set(cell.action_bound_ids)
        <= {value.bound_id for value in law.obligations.support.action_bounds}
        or not all(
            _action_occurrence_inside_support(occurrence, bounds)
            for occurrence in action.occurrences
        )
    )


@dataclass(frozen=True, slots=True)
class AdmissionCoordinateGateReceipt(CanonicalRecord):
    "Exact raw admission gate receipt at one planned product coordinate."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-coordinate-gate-receipt'

    receipt_id: str
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    predicate: GatePredicateSpec
    disposition: ReceiptAdmissionRawDisposition
    observed_scalar: NamedDecimal | None
    observed_boolean: bool | None
    observed_identity: ObjectIdentity | None
    law_evaluation_binding: LawMemberEvaluationBinding
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    evidence_domain: ObjectIdentity
    authority_boundary: ObjectIdentity
    producer: ObjectIdentity
    resource_envelope: ObjectIdentity
    evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    baseline_compatibility: BaselinePreservationCompatibility | None
    status: GateStatus
    margin: NamedDecimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        _validate_admission_receipt_evidence(
            evaluator=self.evaluator,
            input_artifacts=self.input_artifacts,
            evidence_links=self.evidence_links,
        )
        if self.evaluator != self.predicate.evaluator:
            raise ValueError("admission gate receipt uses a foreign predicate evaluator")
        validate_law_evaluation_binding(
            self.planned_coordinate,
            self.law_evaluation_binding,
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("outcome-visible evidence cannot derive admission gate truth")
        if self.predicate.gate_kind is AdmissionGateKind.BASELINE_PRESERVATION:
            if (
                self.baseline_compatibility is None
                or self.baseline_compatibility.gate_predicate != self.predicate
                or not self.baseline_compatibility.compatible
            ):
                raise ValueError("admission baseline gate lacks exact law qualification compatibility")
        elif self.baseline_compatibility is not None:
            raise ValueError("only the admission baseline gate may bind law qualification compatibility")
        if self.disposition is ReceiptAdmissionRawDisposition.EVALUATED:
            if (
                self.law_evaluation_binding.law_evaluation_result is None
                or self.law_evaluation_binding.evaluation_disposition
                is not LawEvaluationBindingDisposition.SUPPORTED
            ):
                raise ValueError("evaluated admission gate requires a supported law evaluation binding")
            expected = derive_admission_gate_observation(
                receipt_id=self.receipt_id,
                predicate=self.predicate,
                observed_scalar=self.observed_scalar,
                observed_boolean=self.observed_boolean,
                observed_identity=self.observed_identity,
            )
        else:
            if any(
                value is not None
                for value in (
                    self.observed_scalar,
                    self.observed_boolean,
                    self.observed_identity,
                )
            ):
                raise ValueError("non-evaluated admission gate cannot impute scientific values")
            reason = admission_disposition_reason(self.disposition)
            expected = (
                (
                    GateStatus.FAIL
                    if self.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                    else GateStatus.UNEVALUABLE
                ),
                None,
                (reason,),
            )
        if (self.status, self.margin, self.reason_codes) != expected:
            raise ValueError("admission gate status/margin is not derived from its raw disposition")

    @property
    def product_key(self) -> tuple[str, AdmissionGateKind]:
        return (self.planned_coordinate.coordinate_id, self.predicate.gate_kind)


class ReceiptAdmissionReachabilityReferenceKind(StrEnum):
    ACTIVE_MINUS_QUALIFIED_HOLD = "ACTIVE_MINUS_QUALIFIED_HOLD"
    DECLARED_NATIVE_REFERENCE = "DECLARED_NATIVE_REFERENCE"


@dataclass(frozen=True, slots=True)
class ControlledMapReachabilityReceipt(CanonicalRecord):
    """Full controlled-map reachability; receiver-visible witnesses cannot inhabit it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-reachability-receipt'

    receipt_id: str
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    law_evaluation_binding: LawMemberEvaluationBinding
    admission_direction_gate_receipt_id: str
    reference_kind: ReceiptAdmissionReachabilityReferenceKind
    reachability_request: ObjectIdentity
    qualified_hold_action_fibre: ObjectIdentity | None
    qualified_hold_law_evaluation_binding: LawMemberEvaluationBinding | None
    declared_native_reference_direction_id: str | None
    controlled_map_evaluation: ObjectIdentity | None
    horizon: HorizonSpec
    constraint_ids: tuple[str, ...]
    candidate_direction_ids: tuple[str, ...]
    viable_direction_ids: tuple[str, ...]
    independent_basis_direction_ids: tuple[str, ...]
    controlled_map_rank: int | None
    condition_number: NamedDecimal | None
    maximum_condition_number: NamedDecimal
    disposition: ReceiptAdmissionRawDisposition
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    evidence_domain: ObjectIdentity
    authority_boundary: ObjectIdentity
    producer: ObjectIdentity
    resource_envelope: ObjectIdentity
    method: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    status: ReachabilityCellDisposition
    viable_direction_rank: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("admission_direction_gate_receipt_id", self.admission_direction_gate_receipt_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.declared_native_reference_direction_id is not None:
            validate_stable_id(
                self.declared_native_reference_direction_id,
                field_name="declared_native_reference_direction_id",
            )
        if self.reachability_request.object_schema != (
            'empirical-lawhood/geometry/controlled-io-reachability-request'
        ):
            raise ValueError("admission reachability requires its exact controlled-map request")
        validate_law_evaluation_binding(
            self.planned_coordinate,
            self.law_evaluation_binding,
        )
        for name, values in (
            ("constraint_ids", self.constraint_ids),
            ("candidate_direction_ids", self.candidate_direction_ids),
            ("viable_direction_ids", self.viable_direction_ids),
            ("independent_basis_direction_ids", self.independent_basis_direction_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if not self.constraint_ids:
            raise ValueError("admission reachability requires frozen constraints")
        validate_decimal(
            self.maximum_condition_number.value,
            field_name="maximum_condition_number",
            minimum=Decimal(1),
        )
        if self.maximum_condition_number.unit != "1":
            raise ValueError("reachability condition threshold must be dimensionless")
        if self.reference_kind is ReceiptAdmissionReachabilityReferenceKind.ACTIVE_MINUS_QUALIFIED_HOLD:
            if (
                self.qualified_hold_action_fibre is None
                or self.qualified_hold_action_fibre.object_schema != ReceiptAdmissionActionFibre.SCHEMA
                or self.qualified_hold_law_evaluation_binding is None
                or self.qualified_hold_law_evaluation_binding.evaluation_disposition
                is not LawEvaluationBindingDisposition.SUPPORTED
                or self.qualified_hold_action_fibre == self.planned_coordinate.action_fibre
                or self.declared_native_reference_direction_id is not None
            ):
                raise ValueError(
                    "active-minus-HOLD reachability requires another supported exact HOLD fibre"
                )
        elif (
            self.qualified_hold_action_fibre is not None
            or self.qualified_hold_law_evaluation_binding is not None
            or self.declared_native_reference_direction_id is None
        ):
            raise ValueError("native-reference reachability requires only its direction identity")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("outcome-visible evidence cannot derive admission reachability")
        _validate_admission_receipt_evidence(
            evaluator=self.method,
            input_artifacts=self.input_artifacts,
            evidence_links=self.evidence_links,
        )
        expected = self._derive()
        if (self.status, self.viable_direction_rank, self.reason_codes) != expected:
            raise ValueError("admission reachability status/rank is not mechanically derived")

    def _derive(self) -> tuple[ReachabilityCellDisposition, int, tuple[str, ...]]:
        if self.disposition is not ReceiptAdmissionRawDisposition.EVALUATED:
            if any(
                (
                    self.controlled_map_evaluation is not None,
                    self.candidate_direction_ids,
                    self.viable_direction_ids,
                    self.independent_basis_direction_ids,
                    self.controlled_map_rank is not None,
                    self.condition_number is not None,
                )
            ):
                raise ValueError("non-evaluated reachability cannot impute controlled geometry")
            reason = admission_disposition_reason(self.disposition)
            return (
                (
                    ReachabilityCellDisposition.UNREACHABLE
                    if self.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                    else ReachabilityCellDisposition.UNEVALUABLE
                ),
                0,
                (reason,),
            )
        if self.controlled_map_evaluation is None:
            raise ValueError("evaluated reachability requires a full controlled-map result")
        if (
            self.law_evaluation_binding.law_evaluation_result is None
            or self.law_evaluation_binding.evaluation_disposition
            is not LawEvaluationBindingDisposition.SUPPORTED
        ):
            raise ValueError("evaluated reachability requires a supported law evaluation binding")
        if self.controlled_map_evaluation.object_schema != (
            'empirical-lawhood/methods/receiver-conditioned-io/markov-kernel-family'
        ):
            raise ValueError(
                "controlled reachability requires a controlled-IO/map identity, not a witness"
            )
        if not self.candidate_direction_ids or self.controlled_map_rank is None:
            raise ValueError("evaluated reachability requires candidate directions and map rank")
        if self.controlled_map_rank < 0:
            raise ValueError("controlled-map rank must be nonnegative")
        if self.condition_number is None or self.condition_number.unit != "1":
            raise ValueError("evaluated reachability requires dimensionless conditioning")
        validate_decimal(
            self.condition_number.value,
            field_name="condition_number",
            minimum=Decimal(1),
        )
        if not set(self.viable_direction_ids) <= set(self.candidate_direction_ids):
            raise ValueError("viable direction lies outside the candidate roster")
        if not set(self.independent_basis_direction_ids) <= set(self.viable_direction_ids):
            raise ValueError("reachability basis contains a nonviable direction")
        reasons: set[str] = set()
        if self.controlled_map_rank == 0:
            reasons.add("ADMISSION_CONTROLLED_MAP_ZERO_RANK")
        if self.condition_number.value > self.maximum_condition_number.value:
            reasons.add("ADMISSION_CONTROLLED_MAP_ILL_CONDITIONED")
        if not self.viable_direction_ids:
            reasons.add("ADMISSION_NO_VIABLE_DIRECTION")
        if not self.independent_basis_direction_ids:
            reasons.add("ADMISSION_NO_INDEPENDENT_DIRECTION_BASIS")
        if reasons:
            return ReachabilityCellDisposition.UNREACHABLE, 0, tuple(sorted(reasons))
        rank = len(self.independent_basis_direction_ids)
        if rank > self.controlled_map_rank:
            raise ValueError("viable direction rank exceeds the controlled-map rank")
        return ReachabilityCellDisposition.REACHABLE, rank, ()

    @property
    def product_key(self) -> str:
        return self.planned_coordinate.coordinate_id


class ReceiptAdmissionUtilityDirection(StrEnum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class ReceiptAdmissionUtilityStatus(StrEnum):
    VIABLE = "VIABLE"
    NOT_VIABLE = "NOT_VIABLE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class UtilityEvaluationReceipt(CanonicalRecord):
    """Member/version-local robust utility derived from raw native values."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/utility-evaluation-receipt'

    receipt_id: str
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    law_evaluation_binding: LawMemberEvaluationBinding
    utility_definition: ObjectIdentity
    direction: ReceiptAdmissionUtilityDirection
    minimum_utility: NamedDecimal
    predicted_response: NamedDecimal | None
    reference_response: NamedDecimal | None
    uncertainty_allowance: NamedDecimal | None
    derived_utility: NamedDecimal | None
    disposition: ReceiptAdmissionRawDisposition
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    evidence_domain: ObjectIdentity
    authority_boundary: ObjectIdentity
    producer: ObjectIdentity
    resource_envelope: ObjectIdentity
    evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    status: ReceiptAdmissionUtilityStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        _validate_admission_receipt_evidence(
            evaluator=self.evaluator,
            input_artifacts=self.input_artifacts,
            evidence_links=self.evidence_links,
        )
        validate_law_evaluation_binding(
            self.planned_coordinate,
            self.law_evaluation_binding,
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("outcome-visible evidence cannot derive admission utility")
        values = (
            self.predicted_response,
            self.reference_response,
            self.uncertainty_allowance,
            self.derived_utility,
        )
        if self.disposition is ReceiptAdmissionRawDisposition.EVALUATED:
            if any(value is None for value in values) or (
                self.law_evaluation_binding.law_evaluation_result is None
            ):
                raise ValueError("evaluated admission utility requires all raw operands")
            if (
                self.law_evaluation_binding.evaluation_disposition
                is not LawEvaluationBindingDisposition.SUPPORTED
            ):
                raise ValueError("evaluated admission utility requires a supported law evaluation binding")
            predicted = self.predicted_response
            reference = self.reference_response
            uncertainty = self.uncertainty_allowance
            derived = self.derived_utility
            if predicted is None or reference is None or uncertainty is None or derived is None:
                raise AssertionError("evaluated utility lost a required operand")
            units = {
                predicted.unit,
                reference.unit,
                uncertainty.unit,
                derived.unit,
                self.minimum_utility.unit,
            }
            if len(units) != 1:
                raise ValueError("admission utility mixes native units")
            validate_decimal(
                uncertainty.value,
                field_name="uncertainty_allowance",
                minimum=Decimal(0),
            )
            signed = (
                predicted.value - reference.value
                if self.direction is ReceiptAdmissionUtilityDirection.HIGHER_IS_BETTER
                else reference.value - predicted.value
            )
            expected_value = signed - uncertainty.value
            if derived.value != expected_value:
                raise ValueError("admission utility is not derived from response/reference/uncertainty")
            expected = (
                (
                    ReceiptAdmissionUtilityStatus.VIABLE
                    if derived.value >= self.minimum_utility.value
                    else ReceiptAdmissionUtilityStatus.NOT_VIABLE
                ),
                (
                    ()
                    if derived.value >= self.minimum_utility.value
                    else ("ADMISSION_UTILITY_BELOW_MINIMUM",)
                ),
            )
        else:
            if any(value is not None for value in values):
                raise ValueError("non-evaluated admission utility cannot impute scientific values")
            expected = (
                (
                    ReceiptAdmissionUtilityStatus.NOT_VIABLE
                    if self.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                    else ReceiptAdmissionUtilityStatus.UNEVALUABLE
                ),
                (admission_disposition_reason(self.disposition),),
            )
        if (self.status, self.reason_codes) != expected:
            raise ValueError("admission utility status is not mechanically derived")

    @property
    def product_key(self) -> str:
        return self.planned_coordinate.coordinate_id


@dataclass(frozen=True, slots=True)
class ControlledMapAdmissionReceiptCorpus(CanonicalRecord):
    """Exactly complete raw gate/reachability/utility product corpus."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-admission-receipt-corpus'

    corpus_id: str
    plan: ReceiptAdmissionReceiptProductionPlan
    gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...]
    reachability_receipts: tuple[ControlledMapReachabilityReceipt, ...]
    utility_receipts: tuple[UtilityEvaluationReceipt, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.corpus_id, field_name="corpus_id")
        for name, values in (
            ("gate_receipts", self.gate_receipts),
            ("reachability_receipts", self.reachability_receipts),
            ("utility_receipts", self.utility_receipts),
        ):
            require_sorted_unique_ids(values, attribute="receipt_id", field_name=name)
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        coordinate_ids = {value.coordinate_id for value in self.plan.coordinates}
        expected_gates = {
            (coordinate_id, kind) for coordinate_id in coordinate_ids for kind in AdmissionGateKind
        }
        if (
            len(self.gate_receipts) != len(expected_gates)
            or {value.product_key for value in self.gate_receipts} != expected_gates
        ):
            raise ValueError("admission gate corpus is incomplete or contains extras")
        if (
            len(self.reachability_receipts) != len(coordinate_ids)
            or {value.product_key for value in self.reachability_receipts} != coordinate_ids
        ):
            raise ValueError("admission reachability corpus is incomplete or contains extras")
        if (
            len(self.utility_receipts) != len(coordinate_ids)
            or {value.product_key for value in self.utility_receipts} != coordinate_ids
        ):
            raise ValueError("admission utility corpus is incomplete or contains extras")
        self._validate_bindings()
        all_receipts: tuple[
            AdmissionCoordinateGateReceipt | ControlledMapReachabilityReceipt | UtilityEvaluationReceipt,
            ...,
        ] = (*self.gate_receipts, *self.reachability_receipts, *self.utility_receipts)
        nested_artifacts: dict[str, ArtifactIdentity] = {}
        for receipt in all_receipts:
            for value in receipt.input_artifacts:
                prior = nested_artifacts.setdefault(value.artifact_id, value)
                if prior != value:
                    raise ValueError(
                        "admission receipts reuse one artifact ID for different exact artifacts"
                    )
        if {value.artifact_id: value for value in self.input_artifacts} != nested_artifacts:
            raise ValueError("admission corpus artifact roster differs from its exact receipts")
        nested_evidence = _exact_evidence_map(
            tuple(value for receipt in all_receipts for value in receipt.evidence_links)
        )
        if _exact_evidence_map(self.evidence_links) != nested_evidence:
            raise ValueError("admission corpus evidence roster differs from its exact receipts")

    def _validate_bindings(self) -> None:
        validate_admission_gate_and_utility_receipts(self.plan, self.gate_receipts, self.utility_receipts)
        coordinates = {value.coordinate_id: value for value in self.plan.coordinates}
        gate_by_id = {value.receipt_id: value for value in self.gate_receipts}
        for reachability_receipt in self.reachability_receipts:
            self._validate_evaluation_binding(reachability_receipt)
            direction_gate = gate_by_id.get(
                reachability_receipt.admission_direction_gate_receipt_id
            )
            if (
                reachability_receipt.planned_coordinate
                != coordinates[reachability_receipt.planned_coordinate.coordinate_id]
                or reachability_receipt.horizon != self.plan.horizon
                or reachability_receipt.constraint_ids != self.plan.reachability_constraint_ids
                or reachability_receipt.information_cutoff != self.plan.information_cutoff
                or reachability_receipt.outcome_access is not self.plan.outcome_access
                or reachability_receipt.evidence_domain != self.plan.evidence_domain
                or reachability_receipt.authority_boundary != self.plan.authority_boundary
                or reachability_receipt.producer != self.plan.reachability_receipt_producer
                or reachability_receipt.resource_envelope
                != self.plan.reachability_resource_envelope
            ):
                raise ValueError("admission reachability receipt rewrites a frozen plan operand")
            self._validate_support_disposition(
                reachability_receipt.planned_coordinate,
                reachability_receipt.disposition,
            )
            if (
                direction_gate is None
                or direction_gate.planned_coordinate != reachability_receipt.planned_coordinate
                or direction_gate.predicate.gate_kind is not AdmissionGateKind.REACHABILITY
            ):
                raise ValueError("admission controlled reachability uses another direction gate")
            if reachability_receipt.status is ReachabilityCellDisposition.REACHABLE and (
                direction_gate.status is not GateStatus.PASS
            ):
                raise ValueError(
                    "controlled reachability cannot bypass its admission direction gate"
                )
            if direction_gate.status is GateStatus.UNEVALUABLE and (
                reachability_receipt.status is not ReachabilityCellDisposition.UNEVALUABLE
            ):
                raise ValueError("unevaluable direction compatibility cannot yield reachability")
            if (
                reachability_receipt.reference_kind
                is ReceiptAdmissionReachabilityReferenceKind.ACTIVE_MINUS_QUALIFIED_HOLD
            ):
                hold_identity = reachability_receipt.qualified_hold_action_fibre
                hold_binding = reachability_receipt.qualified_hold_law_evaluation_binding
                coordinate = reachability_receipt.planned_coordinate
                hold_coordinate = next(
                    (
                        value
                        for value in self.plan.coordinates
                        if value.denominator_member_id == coordinate.denominator_member_id
                        and value.candidate_version_id == coordinate.candidate_version_id
                        and value.qualification_view_ids == coordinate.qualification_view_ids
                        and value.support_cell == coordinate.support_cell
                        and value.action_fibre == hold_identity
                    ),
                    None,
                )
                if hold_coordinate is None or hold_binding is None:
                    raise ValueError("admission reachability lacks its exact HOLD coordinate binding")
                validate_law_evaluation_binding(hold_coordinate, hold_binding)
                if (
                    hold_binding.action_word
                    != self.plan.action_fibre(hold_coordinate.action_fibre).action_word_identity
                ):
                    raise ValueError("admission reachability HOLD binding uses another ActionWord")

    def _validate_evaluation_binding(
        self,
        receipt: (
            AdmissionCoordinateGateReceipt | ControlledMapReachabilityReceipt | UtilityEvaluationReceipt
        ),
    ) -> None:
        coordinate = receipt.planned_coordinate
        action = self.plan.action_fibre(coordinate.action_fibre)
        if receipt.law_evaluation_binding.action_word != action.action_word_identity:
            raise ValueError("admission receipt law evaluation uses another exact ActionWord")

    def _validate_support_disposition(
        self,
        coordinate: ReceiptAdmissionPlannedCoordinate,
        disposition: ReceiptAdmissionRawDisposition,
    ) -> None:
        outside = admission_coordinate_outside_support(self.plan, coordinate)
        if outside != (disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT):
            raise ValueError(
                "admission receipt outside-support disposition differs from exact L(D,H,A,R,tau)"
            )


def validate_admission_gate_and_utility_receipts(
    plan: ReceiptAdmissionReceiptProductionPlan,
    gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    utility_receipts: tuple[UtilityEvaluationReceipt, ...],
    *,
    finite_evaluation_bindings: tuple[LawMemberEvaluationBinding, ...] = (),
) -> None:
    """Check unchanged raw gate/utility semantics for either reachability certificate kind."""

    coordinates = {value.coordinate_id: value for value in plan.coordinates}
    predicates = {value.gate_kind: value for value in plan.gate_predicates}
    for gate_receipt in gate_receipts:
        if (
            gate_receipt.law_evaluation_binding.action_word
            != plan.action_fibre(gate_receipt.planned_coordinate.action_fibre).action_word_identity
        ):
            raise ValueError("admission receipt law evaluation uses another exact ActionWord")
        if (
            gate_receipt.planned_coordinate
            != coordinates[gate_receipt.planned_coordinate.coordinate_id]
            or gate_receipt.predicate != predicates[gate_receipt.predicate.gate_kind]
            or gate_receipt.information_cutoff != plan.information_cutoff
            or gate_receipt.outcome_access is not plan.outcome_access
            or gate_receipt.evidence_domain != plan.evidence_domain
            or gate_receipt.authority_boundary != plan.authority_boundary
            or gate_receipt.producer != plan.gate_receipt_producer
            or gate_receipt.resource_envelope != plan.gate_resource_envelope
        ):
            raise ValueError("admission gate receipt rewrites a frozen plan operand")
        if admission_coordinate_outside_support(
            plan,
            gate_receipt.planned_coordinate,
            finite_evaluation_bindings=finite_evaluation_bindings,
        ) != (gate_receipt.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT):
            raise ValueError(
                "admission receipt outside-support disposition differs from exact L(D,H,A,R,tau)"
            )
    for utility_receipt in utility_receipts:
        if (
            utility_receipt.law_evaluation_binding.action_word
            != plan.action_fibre(
                utility_receipt.planned_coordinate.action_fibre
            ).action_word_identity
        ):
            raise ValueError("admission receipt law evaluation uses another exact ActionWord")
        if (
            utility_receipt.planned_coordinate
            != coordinates[utility_receipt.planned_coordinate.coordinate_id]
            or utility_receipt.information_cutoff != plan.information_cutoff
            or utility_receipt.outcome_access is not plan.outcome_access
            or utility_receipt.evidence_domain != plan.evidence_domain
            or utility_receipt.authority_boundary != plan.authority_boundary
            or utility_receipt.producer != plan.utility_receipt_producer
            or utility_receipt.resource_envelope != plan.utility_resource_envelope
        ):
            raise ValueError("admission utility receipt rewrites a frozen plan operand")
        if admission_coordinate_outside_support(
            plan,
            utility_receipt.planned_coordinate,
            finite_evaluation_bindings=finite_evaluation_bindings,
        ) != (utility_receipt.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT):
            raise ValueError(
                "admission receipt outside-support disposition differs from exact L(D,H,A,R,tau)"
            )


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionAdmissionCandidateCell(CanonicalRecord):
    "Action-specific admission cell whose raw products remain in the bound corpus."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receipt-admission-admission-candidate-cell'

    candidate_cell_id: str
    action_fibre: ObjectIdentity
    support_cell: ObjectIdentity
    planned_coordinate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_cell_id, field_name="candidate_cell_id")
        if self.action_fibre.object_schema != ReceiptAdmissionActionFibre.SCHEMA:
            raise ValueError("admission cell requires an exact action fibre")
        if self.support_cell.object_schema != ReceiptAdmissionSupportCell.SCHEMA:
            raise ValueError("admission cell requires an exact support cell")
        require_sorted_unique_strings(
            self.planned_coordinate_ids,
            field_name="planned_coordinate_ids",
            allow_empty=False,
        )

    @property
    def product_key(self) -> tuple[str, str]:
        return (self.action_fibre.object_id, self.support_cell.object_id)


@dataclass(frozen=True, slots=True)
class ReceiptAdmissionSpec(CanonicalRecord):
    "Controller-consumed admission truth derived only from one raw corpus."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/receipt-admission-spec'

    evaluation_id: str
    corpus: ControlledMapAdmissionReceiptCorpus
    nominal_denominator_member_id: str
    receiver_quantity_ids: tuple[str, ...]
    candidate_cells: tuple[ReceiptAdmissionAdmissionCandidateCell, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_admission_admission_inputs(self)


class ReceiptAdmissionAdmissionCorpusInputs(Protocol):
    @property
    def plan(self) -> ReceiptAdmissionReceiptProductionPlan: ...

    @property
    def gate_receipts(self) -> tuple[AdmissionCoordinateGateReceipt, ...]: ...

    @property
    def evidence_links(self) -> tuple[EvidenceLink, ...]: ...


class ReceiptAdmissionAdmissionInputs(Protocol):
    @property
    def evaluation_id(self) -> str: ...

    @property
    def corpus(self) -> ReceiptAdmissionAdmissionCorpusInputs: ...

    @property
    def nominal_denominator_member_id(self) -> str: ...

    @property
    def receiver_quantity_ids(self) -> tuple[str, ...]: ...

    @property
    def candidate_cells(self) -> tuple[ReceiptAdmissionAdmissionCandidateCell, ...]: ...

    @property
    def evidence_ceiling(self) -> EvidenceCeiling: ...

    @property
    def visibility_ceiling(self) -> VisibilityCeiling: ...


def validate_admission_admission_inputs(spec: ReceiptAdmissionAdmissionInputs) -> None:
    """Common exact admission roster and visibility check, independent of map semantics."""

    validate_stable_id(spec.evaluation_id, field_name="evaluation_id")
    validate_stable_id(
        spec.nominal_denominator_member_id,
        field_name="nominal_denominator_member_id",
    )
    if spec.nominal_denominator_member_id != (spec.corpus.plan.nominal_denominator_member_id):
        raise ValueError("receipt-derived admission nominal member differs from the admission plan")
    require_sorted_unique_strings(
        spec.receiver_quantity_ids,
        field_name="receiver_quantity_ids",
        allow_empty=False,
    )
    require_sorted_unique_ids(
        spec.candidate_cells,
        attribute="candidate_cell_id",
        field_name="candidate_cells",
    )
    expected = {
        (action.action_binding_id, cell.cell_id)
        for action in spec.corpus.plan.action_fibres
        for cell in spec.corpus.plan.support_cells
    }
    if (
        len(spec.candidate_cells) != len(expected)
        or {value.product_key for value in spec.candidate_cells} != expected
    ):
        raise ValueError("receipt-derived admission action/support roster is incomplete or has extras")
    coordinates = {value.coordinate_id: value for value in spec.corpus.plan.coordinates}
    for candidate in spec.candidate_cells:
        expected_ids = tuple(
            value.coordinate_id
            for value in spec.corpus.plan.coordinates
            if value.action_fibre == candidate.action_fibre
            and value.support_cell == candidate.support_cell
        )
        if candidate.planned_coordinate_ids != expected_ids or any(
            coordinates[value].action_fibre != candidate.action_fibre
            or coordinates[value].support_cell != candidate.support_cell
            for value in candidate.planned_coordinate_ids
        ):
            raise ValueError("receipt-derived admission cell differs from its exact Cartesian products")
    if spec.evidence_ceiling is not EvidenceCeiling.ADMISSION:
        raise ValueError("receipt-derived admission must have an admission ceiling")
    inherited = VisibilityCeiling.most_restrictive(
        spec.corpus.plan.visibility_ceiling,
        spec.corpus.plan.model_set.visibility_ceiling,
        spec.corpus.plan.atlas.visibility_ceiling,
        *(value.visibility_ceiling for value in spec.corpus.evidence_links),
    )
    if not spec.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
        raise ValueError("receipt-derived admission visibility cannot be lowered")
    if not spec.visibility_ceiling.is_promotable:
        raise ValueError("outcome-visible evidence cannot establish receipt-derived admission")


def _aggregate_controlled_map_gate(
    candidate_cell_id: str,
    kind: AdmissionGateKind,
    receipts: tuple[AdmissionCoordinateGateReceipt, ...],
) -> AdmissionGateResult:
    statuses = {value.status for value in receipts}
    if GateStatus.FAIL in statuses:
        status = GateStatus.FAIL
    elif GateStatus.UNEVALUABLE in statuses:
        status = GateStatus.UNEVALUABLE
    else:
        status = GateStatus.PASS
    margins = tuple(value.margin for value in receipts if value.margin is not None)
    if margins:
        if len({value.unit for value in margins}) != 1:
            raise ValueError("receipt-derived admission cannot aggregate unlike native-unit margins")
        margin = NamedDecimal(
            value_id=f"receipt-admission-gate-margin.{candidate_cell_id}.{kind.value.lower()}",
            value=min(value.value for value in margins),
            unit=margins[0].unit,
        )
    else:
        margin = None
    return AdmissionGateResult(
        gate_id=f"receipt-admission-gate.{candidate_cell_id}.{kind.value.lower()}",
        kind=kind,
        status=status,
        constraint_ids=tuple(
            sorted(
                {
                    constraint
                    for receipt in receipts
                    for constraint in receipt.predicate.constraint_ids
                }
            )
        ),
        margin=margin,
        reason_codes=(
            ()
            if status is GateStatus.PASS
            else tuple(sorted({reason for value in receipts for reason in value.reason_codes}))
        ),
        evidence_link_ids=tuple(
            sorted({link.link_id for value in receipts for link in value.evidence_links})
        ),
    )


def _derive_receipt_admission_set(
    spec: ReceiptAdmissionAdmissionInputs,
    *,
    member_ids: tuple[str, ...],
    suffix: str,
) -> AdmissionSet:
    coordinate_map = {value.coordinate_id: value for value in spec.corpus.plan.coordinates}
    cells: list[AdmissionCellResult] = []
    for candidate in spec.candidate_cells:
        selected_coordinate_ids = {
            value
            for value in candidate.planned_coordinate_ids
            if coordinate_map[value].denominator_member_id in member_ids
        }
        gates = tuple(
            sorted(
                (
                    _aggregate_controlled_map_gate(
                        candidate.candidate_cell_id,
                        kind,
                        tuple(
                            value
                            for value in spec.corpus.gate_receipts
                            if value.planned_coordinate.coordinate_id in selected_coordinate_ids
                            and value.predicate.gate_kind is kind
                        ),
                    )
                    for kind in AdmissionGateKind
                ),
                key=lambda value: value.gate_id,
            )
        )
        support = spec.corpus.plan.support_cell(candidate.support_cell)
        cells.append(
            AdmissionCellResult(
                cell_id=candidate.candidate_cell_id,
                denominator_cell_id=support.denominator_cell_id,
                chart_id=support.chart_id,
                action_bound_ids=support.action_bound_ids,
                gates=gates,
                admitted=all(value.status is GateStatus.PASS for value in gates),
            )
        )
    if all(value.admitted for value in cells):
        status = AdmissionStatus.ADMITTED
    elif any(value.admitted for value in cells):
        status = AdmissionStatus.PARTIAL
    elif any(gate.status is GateStatus.UNEVALUABLE for value in cells for gate in value.gates):
        status = AdmissionStatus.UNEVALUABLE
    else:
        status = AdmissionStatus.EMPTY
    admission = AdmissionSet(
        admission_id=f"receipt-admission-set.{spec.evaluation_id}.{suffix}",
        atlas=ObjectIdentity.from_record(
            spec.corpus.plan.atlas.atlas_id,
            spec.corpus.plan.atlas,
        ),
        model_set_id=(
            spec.corpus.plan.model_set.model_set_id
            if suffix == "robust"
            else f"{spec.corpus.plan.model_set.model_set_id}.nominal"
        ),
        receiver_quantity_ids=spec.receiver_quantity_ids,
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        status=status,
        evidence_links=spec.corpus.evidence_links,
        evidence_ceiling=spec.evidence_ceiling,
        visibility_ceiling=VisibilityCeiling.most_restrictive(
            spec.visibility_ceiling,
            spec.corpus.plan.model_set.visibility_ceiling,
            spec.corpus.plan.atlas.visibility_ceiling,
        ),
    )
    _validate_controlled_map_admission_against_atlas(spec.corpus.plan.atlas, admission)
    return admission


def _validate_controlled_map_admission_against_atlas(
    atlas: ResponseAtlas,
    admission: AdmissionSet,
) -> None:
    """Allow typed outside-support failures while forbidding invented admission."""

    if admission.atlas != ObjectIdentity.from_record(atlas.atlas_id, atlas):
        raise ValueError("receipt-derived admission binds another response atlas")
    output_quantity_ids = {
        quantity_id for law in atlas.laws for quantity_id in law.interface_output_quantity_ids
    }
    if not set(admission.receiver_quantity_ids) <= output_quantity_ids:
        raise ValueError("receipt-derived admission receiver is not produced by its atlas")
    chart_ids = {value.chart_id for value in atlas.laws}
    for cell in admission.cells:
        if cell.chart_id not in chart_ids:
            raise ValueError("receipt-derived admission cell uses a chart outside its atlas")
        if cell.admitted and not atlas.laws_for_coordinate(
            cell.chart_id,
            cell.denominator_cell_id,
        ):
            raise ValueError("receipt-derived admission invents support outside its atlas laws")
        if cell.admitted and atlas.gaps_for_coordinate(
            cell.chart_id,
            cell.denominator_cell_id,
        ):
            raise ValueError("receipt-derived admitted cell occupies an explicit atlas gap")


def derive_receipt_admission_comparison(
    spec: ReceiptAdmissionSpec,
) -> AdmissionComparison:
    """Intersect all candidate versions and members from the exact raw corpus."""

    nominal = _derive_receipt_admission_set(
        spec,
        member_ids=(spec.nominal_denominator_member_id,),
        suffix="nominal",
    )
    robust = _derive_receipt_admission_set(
        spec,
        member_ids=tuple(
            value.denominator_member_id for value in spec.corpus.plan.model_set.members
        ),
        suffix="robust",
    )
    disagreement = tuple(
        sorted(set(nominal.admitted_cell_ids).symmetric_difference(robust.admitted_cell_ids))
    )
    return AdmissionComparison(
        comparison_id=f"receipt-admission-comparison.{spec.evaluation_id}",
        evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
        nominal=nominal,
        robust=robust,
        structurally_stable=not disagreement,
        disagreement_cell_ids=disagreement,
    )


class ReceiptAdmissionReachabilityAggregateDisposition(StrEnum):
    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_ADMITTED = "NOT_ADMITTED"


@dataclass(frozen=True, slots=True)
class ControlledMapReachabilityCellResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-reachability-cell-result'

    candidate_cell_id: str
    action_fibre: ObjectIdentity
    support_cell: ObjectIdentity
    receipt_ids: tuple[str, ...]
    disposition: ReceiptAdmissionReachabilityAggregateDisposition
    viable_direction_rank: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_cell_id, field_name="candidate_cell_id")
        require_sorted_unique_strings(
            self.receipt_ids,
            field_name="receipt_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.viable_direction_rank < 0:
            raise ValueError("controlled-map reachability rank must be nonnegative")
        if self.disposition is ReceiptAdmissionReachabilityAggregateDisposition.REACHABLE:
            if self.viable_direction_rank == 0 or self.reason_codes:
                raise ValueError("reachable controlled-map cell requires rank and no exclusion reasons")
        elif self.viable_direction_rank != 0 or not self.reason_codes:
            raise ValueError("non-reachable controlled-map cell requires zero rank and reasons")


@dataclass(frozen=True, slots=True)
class ControlledMapReachabilitySet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-reachability-set'

    reachability_id: str
    admission: ObjectIdentity
    corpus: ObjectIdentity
    model_set: ObjectIdentity
    denominator_member_ids: tuple[str, ...]
    candidate_version_ids: tuple[str, ...]
    cells: tuple[ControlledMapReachabilityCellResult, ...]
    admission_status: AdmissionStatus
    reachable_cell_ids: tuple[str, ...]
    viable_direction_rank: int
    status: ReachabilityStatus
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling
    safe_abstention_required_outside: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.reachability_id, field_name="reachability_id")
        for name, values in (
            ("denominator_member_ids", self.denominator_member_ids),
            ("candidate_version_ids", self.candidate_version_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.cells,
            attribute="candidate_cell_id",
            field_name="cells",
        )
        require_sorted_unique_strings(
            self.reachable_cell_ids,
            field_name="reachable_cell_ids",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        reachable = tuple(
            value
            for value in self.cells
            if value.disposition is ReceiptAdmissionReachabilityAggregateDisposition.REACHABLE
        )
        observed_ids = tuple(value.candidate_cell_id for value in reachable)
        observed_rank = min(
            (value.viable_direction_rank for value in reachable),
            default=0,
        )
        if self.reachable_cell_ids != observed_ids or self.viable_direction_rank != observed_rank:
            raise ValueError("controlled-map reachable cells/rank differ from cell assessments")
        admitted_cells = tuple(
            value
            for value in self.cells
            if value.disposition is not ReceiptAdmissionReachabilityAggregateDisposition.NOT_ADMITTED
        )
        if self.admission_status is AdmissionStatus.UNEVALUABLE and not admitted_cells:
            expected_status = ReachabilityStatus.UNEVALUABLE
        elif not admitted_cells:
            expected_status = ReachabilityStatus.EMPTY
        elif len(reachable) == len(admitted_cells):
            expected_status = ReachabilityStatus.REACHABLE
        elif reachable:
            expected_status = ReachabilityStatus.PARTIAL
        elif any(
            value.disposition is ReceiptAdmissionReachabilityAggregateDisposition.UNEVALUABLE
            for value in admitted_cells
        ):
            expected_status = ReachabilityStatus.UNEVALUABLE
        else:
            expected_status = ReachabilityStatus.EMPTY
        if self.status is not expected_status:
            raise ValueError("controlled-map reachability status differs from exact cell aggregation")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("controlled-map reachability must have an admission ceiling")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible evidence cannot establish controlled-map reachability")
        if not self.safe_abstention_required_outside:
            raise ValueError("controlled-map reachability requires HOLD or NONATTEMPT outside support")


@dataclass(frozen=True, slots=True)
class ControlledMapReachabilityComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-reachability-comparison'

    comparison_id: str
    evaluation: ObjectIdentity
    nominal: ControlledMapReachabilitySet
    robust: ControlledMapReachabilitySet
    structurally_stable: bool
    disagreement_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.comparison_id, field_name="comparison_id")
        require_sorted_unique_strings(
            self.disagreement_cell_ids,
            field_name="disagreement_cell_ids",
        )
        expected_disagreement = tuple(
            sorted(
                set(self.nominal.reachable_cell_ids).symmetric_difference(
                    self.robust.reachable_cell_ids
                )
            )
        )
        expected_stability = (
            not expected_disagreement
            and self.nominal.viable_direction_rank == self.robust.viable_direction_rank
        )
        if (
            self.disagreement_cell_ids != expected_disagreement
            or self.structurally_stable != expected_stability
        ):
            raise ValueError("controlled-map reachability comparison differs from nominal/robust results")


@dataclass(frozen=True, slots=True)
class ControlledMapReachabilitySpec(CanonicalRecord):
    "Full-map reachability derived from the same corpus as receipt-derived admission."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-reachability-spec'

    evaluation_id: str
    admission_spec: ReceiptAdmissionSpec
    admission: AdmissionComparison
    initial_set_id: str
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("initial_set_id", self.initial_set_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.admission != derive_receipt_admission_comparison(self.admission_spec):
            raise ValueError("controlled-map reachability admission is not derived from its raw corpus")
        if self.initial_set_id != self.admission_spec.corpus.plan.initial_set_id:
            raise ValueError("controlled-map reachability initial set differs from the admission plan")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("evidence-derived controlled-map reachability must have an admission ceiling")
        inherited_visibility = VisibilityCeiling.most_restrictive(
            self.admission_spec.visibility_ceiling,
            self.admission.nominal.visibility_ceiling,
            self.admission.robust.visibility_ceiling,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited_visibility):
            raise ValueError("controlled-map reachability visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible evidence cannot establish controlled-map reachability")


def _derive_controlled_map_reachability_set(
    spec: ControlledMapReachabilitySpec,
    *,
    admission: AdmissionSet,
    member_ids: tuple[str, ...],
    suffix: str,
) -> ControlledMapReachabilitySet:
    corpus = spec.admission_spec.corpus
    coordinate_map = {value.coordinate_id: value for value in corpus.plan.coordinates}
    admission_cells = {value.cell_id: value for value in admission.cells}
    cells: list[ControlledMapReachabilityCellResult] = []
    for candidate in spec.admission_spec.candidate_cells:
        selected_ids = {
            value
            for value in candidate.planned_coordinate_ids
            if coordinate_map[value].denominator_member_id in member_ids
        }
        receipts = tuple(
            value
            for value in corpus.reachability_receipts
            if value.planned_coordinate.coordinate_id in selected_ids
        )
        admission_cell = admission_cells[candidate.candidate_cell_id]
        reasons: tuple[str, ...]
        if not admission_cell.admitted:
            disposition = ReceiptAdmissionReachabilityAggregateDisposition.NOT_ADMITTED
            rank = 0
            reasons = ("ADMISSION_NOT_ADMITTED",)
        elif all(value.status is ReachabilityCellDisposition.REACHABLE for value in receipts):
            disposition = ReceiptAdmissionReachabilityAggregateDisposition.REACHABLE
            rank = min(value.viable_direction_rank for value in receipts)
            reasons = ()
        elif any(value.status is ReachabilityCellDisposition.UNEVALUABLE for value in receipts):
            disposition = ReceiptAdmissionReachabilityAggregateDisposition.UNEVALUABLE
            rank = 0
            reasons = tuple(sorted({reason for value in receipts for reason in value.reason_codes}))
        else:
            disposition = ReceiptAdmissionReachabilityAggregateDisposition.UNREACHABLE
            rank = 0
            reasons = tuple(sorted({reason for value in receipts for reason in value.reason_codes}))
        cells.append(
            ControlledMapReachabilityCellResult(
                candidate_cell_id=candidate.candidate_cell_id,
                action_fibre=candidate.action_fibre,
                support_cell=candidate.support_cell,
                receipt_ids=tuple(value.receipt_id for value in receipts),
                disposition=disposition,
                viable_direction_rank=rank,
                reason_codes=reasons,
            )
        )
    member_set = set(member_ids)
    candidate_versions = tuple(
        sorted(
            {
                version
                for member in corpus.plan.model_set.members
                if member.denominator_member_id in member_set
                for version in member.candidate_version_ids
            }
        )
    )
    provisional = tuple(sorted(cells, key=lambda value: value.candidate_cell_id))
    reachable_ids = tuple(
        value.candidate_cell_id
        for value in provisional
        if value.disposition is ReceiptAdmissionReachabilityAggregateDisposition.REACHABLE
    )
    ranks = tuple(
        value.viable_direction_rank
        for value in provisional
        if value.disposition is ReceiptAdmissionReachabilityAggregateDisposition.REACHABLE
    )
    admitted_count = sum(
        value.disposition is not ReceiptAdmissionReachabilityAggregateDisposition.NOT_ADMITTED
        for value in provisional
    )
    if admission.status is AdmissionStatus.UNEVALUABLE and admitted_count == 0:
        status = ReachabilityStatus.UNEVALUABLE
    elif admitted_count == 0:
        status = ReachabilityStatus.EMPTY
    elif len(reachable_ids) == admitted_count:
        status = ReachabilityStatus.REACHABLE
    elif reachable_ids:
        status = ReachabilityStatus.PARTIAL
    elif any(
        value.disposition is ReceiptAdmissionReachabilityAggregateDisposition.UNEVALUABLE for value in provisional
    ):
        status = ReachabilityStatus.UNEVALUABLE
    else:
        status = ReachabilityStatus.EMPTY
    return ControlledMapReachabilitySet(
        reachability_id=f"controlled-map-reachability.{spec.evaluation_id}.{suffix}",
        admission=ObjectIdentity.from_record(admission.admission_id, admission),
        corpus=ObjectIdentity.from_record(corpus.corpus_id, corpus),
        model_set=ObjectIdentity.from_record(
            corpus.plan.model_set.model_set_id,
            corpus.plan.model_set,
        ),
        denominator_member_ids=tuple(sorted(member_ids)),
        candidate_version_ids=candidate_versions,
        cells=provisional,
        admission_status=admission.status,
        reachable_cell_ids=reachable_ids,
        viable_direction_rank=min(ranks, default=0),
        status=status,
        evidence_links=corpus.evidence_links,
        evidence_ceiling=spec.evidence_ceiling,
        visibility_ceiling=VisibilityCeiling.most_restrictive(
            spec.visibility_ceiling,
            corpus.plan.model_set.visibility_ceiling,
            admission.visibility_ceiling,
        ),
    )


def derive_controlled_map_reachability_comparison(
    spec: ControlledMapReachabilitySpec,
) -> ControlledMapReachabilityComparison:
    nominal = _derive_controlled_map_reachability_set(
        spec,
        admission=spec.admission.nominal,
        member_ids=(spec.admission_spec.nominal_denominator_member_id,),
        suffix="nominal",
    )
    robust = _derive_controlled_map_reachability_set(
        spec,
        admission=spec.admission.robust,
        member_ids=tuple(
            value.denominator_member_id
            for value in spec.admission_spec.corpus.plan.model_set.members
        ),
        suffix="robust",
    )
    disagreement = tuple(
        sorted(set(nominal.reachable_cell_ids).symmetric_difference(robust.reachable_cell_ids))
    )
    return ControlledMapReachabilityComparison(
        comparison_id=f"controlled-map-reachability-comparison.{spec.evaluation_id}",
        evaluation=ObjectIdentity.from_record(spec.evaluation_id, spec),
        nominal=nominal,
        robust=robust,
        structurally_stable=(
            not disagreement and nominal.viable_direction_rank == robust.viable_direction_rank
        ),
        disagreement_cell_ids=disagreement,
    )


@dataclass(frozen=True, slots=True)
class VerifiedAdmissionReachabilityCategoricalProjection(CanonicalRecord):
    "Equality-checked categorical view; never an input to raw admission reconstruction."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/verified-admission-reachability-categorical-projection'

    projection_id: str
    admission_spec: ReceiptAdmissionSpec
    reachability_spec: ControlledMapReachabilitySpec
    corpus: ObjectIdentity
    admission: AdmissionComparison
    reachability: ControlledMapReachabilityComparison
    nominal_admission_status: AdmissionStatus
    robust_admission_status: AdmissionStatus
    nominal_admitted_cell_ids: tuple[str, ...]
    robust_admitted_cell_ids: tuple[str, ...]
    nominal_reachability_status: ReachabilityStatus
    robust_reachability_status: ReachabilityStatus
    nominal_reachable_cell_ids: tuple[str, ...]
    robust_reachable_cell_ids: tuple[str, ...]
    nominal_viable_direction_rank: int
    robust_viable_direction_rank: int
    equality_checked: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        for name, values in (
            ("nominal_admitted_cell_ids", self.nominal_admitted_cell_ids),
            ("robust_admitted_cell_ids", self.robust_admitted_cell_ids),
            ("nominal_reachable_cell_ids", self.nominal_reachable_cell_ids),
            ("robust_reachable_cell_ids", self.robust_reachable_cell_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.reachability_spec.admission_spec != self.admission_spec:
            raise ValueError("verified admission/reachability projection specs do not share one raw corpus")
        expected_corpus = ObjectIdentity.from_record(
            self.admission_spec.corpus.corpus_id,
            self.admission_spec.corpus,
        )
        if self.corpus != expected_corpus:
            raise ValueError("verified admission/reachability projection binds another raw corpus")
        if self.admission != derive_receipt_admission_comparison(self.admission_spec):
            raise ValueError("projected admission is not derived from its raw corpus")
        if self.reachability != derive_controlled_map_reachability_comparison(self.reachability_spec):
            raise ValueError("projected reachability is not derived from its raw corpus")
        observed = (
            self.admission.nominal.status,
            self.admission.robust.status,
            self.admission.nominal.admitted_cell_ids,
            self.admission.robust.admitted_cell_ids,
            self.reachability.nominal.status,
            self.reachability.robust.status,
            self.reachability.nominal.reachable_cell_ids,
            self.reachability.robust.reachable_cell_ids,
            self.reachability.nominal.viable_direction_rank,
            self.reachability.robust.viable_direction_rank,
        )
        declared = (
            self.nominal_admission_status,
            self.robust_admission_status,
            self.nominal_admitted_cell_ids,
            self.robust_admitted_cell_ids,
            self.nominal_reachability_status,
            self.robust_reachability_status,
            self.nominal_reachable_cell_ids,
            self.robust_reachable_cell_ids,
            self.nominal_viable_direction_rank,
            self.robust_viable_direction_rank,
        )
        if not self.equality_checked or declared != observed:
            raise ValueError("verified admission/reachability categorical projection differs from the raw-receipt derivation")


def derive_controlled_map_categorical_projection(
    admission_spec: ReceiptAdmissionSpec,
    reachability_spec: ControlledMapReachabilitySpec,
) -> VerifiedAdmissionReachabilityCategoricalProjection:
    if reachability_spec.admission_spec != admission_spec:
        raise ValueError("verified admission/reachability projection specs do not share one exact raw admission corpus")
    admission = derive_receipt_admission_comparison(admission_spec)
    reachability = derive_controlled_map_reachability_comparison(reachability_spec)
    return VerifiedAdmissionReachabilityCategoricalProjection(
        projection_id=f"verified-admission-reachability-projection.{admission_spec.evaluation_id}",
        admission_spec=admission_spec,
        reachability_spec=reachability_spec,
        corpus=ObjectIdentity.from_record(admission_spec.corpus.corpus_id, admission_spec.corpus),
        admission=admission,
        reachability=reachability,
        nominal_admission_status=admission.nominal.status,
        robust_admission_status=admission.robust.status,
        nominal_admitted_cell_ids=admission.nominal.admitted_cell_ids,
        robust_admitted_cell_ids=admission.robust.admitted_cell_ids,
        nominal_reachability_status=reachability.nominal.status,
        robust_reachability_status=reachability.robust.status,
        nominal_reachable_cell_ids=reachability.nominal.reachable_cell_ids,
        robust_reachable_cell_ids=reachability.robust.reachable_cell_ids,
        nominal_viable_direction_rank=reachability.nominal.viable_direction_rank,
        robust_viable_direction_rank=reachability.robust.viable_direction_rank,
        equality_checked=True,
    )


__all__ = [
    "BaselinePreservationCompatibility",
    "DirectionConstraintReceipt",
    'AtlasAdmissionSpec',
    'ReceiptAdmissionSpec',
    'AtlasReachabilitySpec',
    'ControlledMapReachabilitySpec',
    'AtlasGateReceipt',
    'AdmissionCoordinateGateReceipt',
    "GatePredicateKind",
    "GatePredicateSpec",
    "LawEvaluationBindingDisposition",
    "LawMemberEvaluationBinding",
    'ReceiptAdmissionActionFibre',
    'ReceiptAdmissionAdmissionCandidateCell',
    'ReceiptAdmissionPlannedCoordinate',
    'ReceiptAdmissionRawDisposition',
    'ReceiptAdmissionReachabilityReferenceKind',
    'ReceiptAdmissionReachabilityAggregateDisposition',
    'ControlledMapReachabilityCellResult',
    'ControlledMapAdmissionReceiptCorpus',
    'ReceiptAdmissionReceiptProductionPlan',
    'ReceiptAdmissionSupportCell',
    'ReceiptAdmissionUtilityDirection',
    'ReceiptAdmissionUtilityStatus',
    "PreservationCompatibilityRule",
    'VerifiedAdmissionReachabilityCategoricalProjection',
    "ReachabilityCellDisposition",
    'AtlasReachabilityReceipt',
    'ControlledMapReachabilityReceipt',
    "ReachabilityRepresentationStatus",
    'ControlledMapReachabilityComparison',
    'ControlledMapReachabilitySet',
    'UtilityEvaluationReceipt',
    'derive_atlas_admission_comparison',
    'derive_receipt_admission_comparison',
    'derive_controlled_map_categorical_projection',
    'derive_atlas_reachability_comparison',
    'derive_controlled_map_reachability_comparison',
    'admission_coordinate_outside_support',
    "validate_law_evaluation_binding",
]
