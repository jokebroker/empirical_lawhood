"Additive parametric response-method reference preparations and canonical conformance outcomes."

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.reference_worlds import ReferenceWorldKind, get_reference_world
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawIdentificationResult, LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.models import QuantityAlignment, TransportStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal, QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import ClockRelationKind, ClockRelationSpec
from empirical_lawhood.kernel.worlds import WorldKind

from .baselines import (
    LocalLinearIdentifier,
    LocalStateSpaceIdentifier,
    NonlinearLocalIdentifier,
)
from .contracts import (
    DataSplit,
    IdentificationDataset,
    LawIdentificationConfig,
    LawIdentifier,
    LawObservation,
    ObservationRole,
    OneFactorExchange,
    ReceiverCriterion,
    ResponseSlopeTolerance,
    StructuralCoefficientTolerance,
)
from .conformance_payloads import EphemeralCandidatePayloadPlane
from .identification import build_response_method_identification_service
from .transport import (
    DirectionalDiscrepancyEstimator,
    DirectionalTransportAssessment,
    DirectionalTransportDataset,
    DiscrepancyEstimatorConfig,
    PairedTransportObservation,
    TransportCriterion,
)


@dataclass(frozen=True)
class IdentificationCase:
    case_id: str
    system: SystemSpec
    dataset: IdentificationDataset
    config: LawIdentificationConfig
    identifier: LawIdentifier
    expected_status: ScientificStatus


@dataclass(frozen=True)
class TransportCase:
    case_id: str
    source_system: SystemSpec
    target_system: SystemSpec
    dataset: DirectionalTransportDataset
    config: DiscrepancyEstimatorConfig
    expected_status: TransportStatus


@dataclass(frozen=True, slots=True)
class LawConformanceOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-conformance-outcome'

    case_id: str
    world_id: str
    method_key: str
    expected_status: ScientificStatus
    observed_status: ScientificStatus
    response_law_emitted: bool
    failed_check_ids: tuple[str, ...]
    physical_unit_count: int
    numerical_view_count: int
    result_fingerprint: str

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("world_id", self.world_id),
            ("method_key", self.method_key),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.failed_check_ids, field_name="failed_check_ids")
        validate_sha256(self.result_fingerprint, field_name="result_fingerprint")
        if self.physical_unit_count <= 0 or self.numerical_view_count < 2:
            raise ValueError("law conformance requires physical units and refined views")


@dataclass(frozen=True, slots=True)
class TransportConformanceOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transport-conformance-outcome'

    case_id: str
    source_world_id: str
    target_world_id: str
    expected_status: TransportStatus
    observed_status: TransportStatus
    calibration_unit_ids: tuple[str, ...]
    held_out_unit_ids: tuple[str, ...]
    failed_test_ids: tuple[str, ...]
    assessment_fingerprint: str

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("source_world_id", self.source_world_id),
            ("target_world_id", self.target_world_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("calibration_unit_ids", self.calibration_unit_ids),
            ("held_out_unit_ids", self.held_out_unit_ids),
            ("failed_test_ids", self.failed_test_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if set(self.calibration_unit_ids) & set(self.held_out_unit_ids):
            raise ValueError("transport conformance calibration and held-out units overlap")
        validate_sha256(self.assessment_fingerprint, field_name="assessment_fingerprint")


@dataclass(frozen=True, slots=True)
class ResponseMethodReferenceConformanceReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/methods/response-method-reference-conformance-report'

    report_id: str
    law_outcomes: tuple[LawConformanceOutcome, ...]
    transport_outcomes: tuple[TransportConformanceOutcome, ...]
    capability_registry_fingerprint: str
    status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_ids(
            self.law_outcomes,
            attribute="case_id",
            field_name="law_outcomes",
        )
        require_sorted_unique_ids(
            self.transport_outcomes,
            attribute="case_id",
            field_name="transport_outcomes",
        )
        if not self.law_outcomes or not self.transport_outcomes:
            raise ValueError("Parametric response-method conformance requires law and transport outcomes")
        validate_sha256(
            self.capability_registry_fingerprint,
            field_name="capability_registry_fingerprint",
        )
        if self.status != "PASS":
            raise ValueError("Parametric response-method conformance report represents only a passing execution")


def _named(value_id: str, value: Decimal) -> tuple[NamedDecimal, ...]:
    return (NamedDecimal(value_id=value_id, value=value, unit="1"),)


def build_identification_dataset(
    kind: ReferenceWorldKind,
    *,
    dataset_id: str,
    response: Callable[[Decimal, Decimal, str], Decimal],
    wrong_action_effect: Decimal = Decimal(0),
) -> tuple[SystemSpec, IdentificationDataset]:
    system = get_reference_world(kind).system
    observations: list[LawObservation] = []
    actions = tuple(Decimal(value) for value in ("-2", "-1", "0", "1", "2"))
    histories: tuple[Decimal | None, ...] = (
        (Decimal("-1"), Decimal("1")) if system.relation.history_quantity_ids else (None,)
    )
    for split_index, split in enumerate(DataSplit):
        for cell_index, cell_id in enumerate(("response-method-cell-a", "response-method-cell-b")):
            for action_index, action in enumerate(actions):
                for history_index, history in enumerate(histories):
                    for replicate in range(2):
                        unit_id = (
                            f"response-method-unit-{dataset_id}-{split_index}-{cell_index}-"
                            f"{action_index}-{history_index}-{replicate}"
                        )
                        for view in system.numerical_views:
                            receiver = response(action, history or Decimal(0), view.view_id)
                            observations.append(
                                LawObservation(
                                    observation_id=f"observation.{unit_id}.{view.view_id}",
                                    physical_unit_instance_id=unit_id,
                                    denominator_cell_id=cell_id,
                                    chart_id="response-method-local-chart",
                                    numerical_view_id=view.view_id,
                                    split=split,
                                    role=ObservationRole.PRIMARY,
                                    denominator_values=_named("denominator", Decimal(cell_index)),
                                    history_values=(
                                        _named("history", history) if history is not None else ()
                                    ),
                                    action_values=_named("action", action),
                                    receiver_values=_named("receiver", receiver),
                                    tags=("response-method-additive-independent-preparation",),
                                )
                            )
    falsifier_history = histories[0]
    for view in system.numerical_views:
        observations.append(
            LawObservation(
                observation_id=f"observation.{dataset_id}.wrong-action.{view.view_id}",
                physical_unit_instance_id=f"response-method-unit-{dataset_id}-wrong-action",
                denominator_cell_id="response-method-cell-a",
                chart_id="response-method-local-chart",
                numerical_view_id=view.view_id,
                split=DataSplit.HELD_OUT,
                role=ObservationRole.WRONG_ACTION,
                denominator_values=_named("denominator", Decimal(0)),
                history_values=(
                    _named("history", falsifier_history) if falsifier_history is not None else ()
                ),
                action_values=_named("action", Decimal(1)),
                receiver_values=_named("receiver", wrong_action_effect),
                tags=("response-method-decisive-wrong-action",),
            )
        )
    dataset = IdentificationDataset(
        dataset_id=dataset_id,
        system=ObjectIdentity.from_record(system.system_id, system),
        relation=system.relation,
        information_cutoff_id=f"cutoff.{dataset_id}",
        observations=tuple(sorted(observations, key=lambda value: value.observation_id)),
        one_factor_exchanges=(
            OneFactorExchange(
                exchange_id=f"exchange.{dataset_id}",
                left_denominator_cell_id="response-method-cell-a",
                right_denominator_cell_id="response-method-cell-b",
                exchanged_factor_id="denominator",
                slope_tolerances=(
                    ResponseSlopeTolerance(
                        tolerance_id="tolerance.action.receiver",
                        action_quantity_id="action",
                        receiver_quantity_id="receiver",
                        native_unit="1/1",
                        maximum_absolute_difference=Decimal("0.01"),
                    ),
                ),
            ),
        ),
        evidence_artifacts=(
            ArtifactIdentity(
                artifact_id=f"artifact.{dataset_id}",
                role="canonical-evidence",
                payload_schema='empirical-lawhood/table/synthetic-law-conformance-observations',
                sha256=hashlib.sha256(f"{dataset_id}:canonical-rows".encode()).hexdigest(),
                media_type="application/vnd.apache.parquet",
                size_bytes=len(observations) * 128,
            ),
        ),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return system, dataset


def build_identification_config(
    system: SystemSpec,
    *,
    config_id: str,
    method_key: str,
    method_kind: LawMethodKind,
    polynomial_degree: int,
    include_history: bool,
    maximum_held_out_rmse: Decimal = Decimal("0.01"),
) -> LawIdentificationConfig:
    features = ["action"]
    if include_history:
        features.append("history")
    terms = ["intercept", *(f"linear.{value}" for value in features)]
    if polynomial_degree == 2:
        terms.extend(f"quadratic.{value}" for value in features)

    def coefficient_unit(term_id: str) -> str:
        if term_id == "intercept":
            return "1"
        if term_id.startswith("linear."):
            return "1/(1)"
        return "1/(1^2)"

    return LawIdentificationConfig(
        config_id=config_id,
        method_key=method_key,
        method_version="1.0.0",
        method_kind=method_kind,
        chart_id="response-method-local-chart",
        feature_quantity_ids=tuple(sorted(features)),
        receiver_quantity_ids=("receiver",),
        receiver_criteria=(
            ReceiverCriterion(
                criterion_id="criterion.receiver",
                receiver_quantity_id="receiver",
                native_unit="1",
                maximum_held_out_rmse=maximum_held_out_rmse,
                maximum_wrong_action_effect=Decimal("0.1"),
                minimum_baseline_improvement=Decimal("0.5"),
                aleatoric_uncertainty_bound=Decimal(0),
                aleatoric_bound_method_id="analytic-deterministic-zero-noise",
                observation_uncertainty_bound=Decimal(0),
                observation_bound_method_id="analytic-exact-observation",
            ),
        ),
        action_bounds=(
            QuantityBound(
                bound_id="bound.action",
                quantity_id="action",
                native_unit="1",
                lower=Decimal("-2"),
                upper=Decimal("2"),
            ),
        ),
        structural_tolerances=tuple(
            sorted(
                (
                    StructuralCoefficientTolerance(
                        tolerance_id=f"tolerance.receiver.{term_id}",
                        receiver_quantity_id="receiver",
                        term_id=term_id,
                        native_unit=coefficient_unit(term_id),
                        zero_absolute_tolerance=Decimal("1e-10"),
                        maximum_refinement_difference=Decimal("0.01"),
                    )
                    for term_id in terms
                ),
                key=lambda value: value.tolerance_id,
            )
        ),
        mapping_assumption_ids=(
            "frozen-response-method-coordinate-chart",
            "independent-analytic-preparations",
        ),
        polynomial_degree=polynomial_degree,
        minimum_recurrence_units=2,
        minimum_action_levels=5,
        uncertainty_confidence_level=Decimal("0.95"),
        maximum_residual_history_correlation=Decimal("0.1"),
        causal_strength=(
            CausalStrength.SIMULATOR_INTERVENTION
            if system.world.kind is WorldKind.NUMERICAL_SIMULATOR
            else CausalStrength.RANDOMIZED_INTERVENTION
        ),
        representation_kind=(
            LawRepresentationKind.LOCAL_STATE_SPACE
            if include_history
            else LawRepresentationKind.FINITE_ACTION_OPERATOR
        ),
    )


def identification_cases() -> tuple[IdentificationCase, ...]:
    cases = []
    specifications = (
        (
            "response-method-nonlinear-local",
            ReferenceWorldKind.NONLINEAR_CHARTED,
            NonlinearLocalIdentifier(),
            lambda action, _history, _view: Decimal(1) + action + Decimal("0.5") * action * action,
            True,
            Decimal("0.01"),
            Decimal(0),
        ),
        (
            "response-method-stable-linear",
            ReferenceWorldKind.STABLE_LINEAR,
            LocalLinearIdentifier(),
            lambda action, _history, _view: Decimal(1) + Decimal(2) * action,
            True,
            Decimal("0.01"),
            Decimal(0),
        ),
        (
            "response-method-state-space",
            ReferenceWorldKind.HYSTERETIC_MEMORY,
            LocalStateSpaceIdentifier(),
            lambda action, history, _view: (
                Decimal(1) + Decimal(2) * action + Decimal("0.5") * history
            ),
            True,
            Decimal("0.01"),
            Decimal(0),
        ),
        (
            "response-method-non-closing",
            ReferenceWorldKind.HYSTERETIC_MEMORY,
            LocalLinearIdentifier(),
            lambda action, history, _view: (
                Decimal(1) + Decimal(2) * action + Decimal("0.2") * history
            ),
            False,
            Decimal("0.3"),
            Decimal(0),
        ),
        (
            "response-method-structurally-unstable",
            ReferenceWorldKind.NUMERICAL_FALSE_STRUCTURE,
            LocalLinearIdentifier(),
            lambda action, _history, view: (
                Decimal(1) + (Decimal(2) if view == "coarse-view" else Decimal(-2)) * action
            ),
            False,
            Decimal("0.01"),
            Decimal(0),
        ),
        (
            "response-method-wrong-action-rejection",
            ReferenceWorldKind.STABLE_LINEAR,
            LocalLinearIdentifier(),
            lambda action, _history, _view: Decimal(1) + Decimal(2) * action,
            False,
            Decimal("0.01"),
            Decimal(1),
        ),
    )
    for (
        case_id,
        kind,
        identifier,
        response,
        supported,
        maximum_rmse,
        wrong_effect,
    ) in specifications:
        system, dataset = build_identification_dataset(
            kind,
            dataset_id=case_id,
            response=response,
            wrong_action_effect=wrong_effect,
        )
        include_history = isinstance(identifier, LocalStateSpaceIdentifier)
        cases.append(
            IdentificationCase(
                case_id=case_id,
                system=system,
                dataset=dataset,
                config=build_identification_config(
                    system,
                    config_id=f"config.{case_id}",
                    method_key=identifier.method_key,
                    method_kind=identifier.method_kind,
                    polynomial_degree=2 if isinstance(identifier, NonlinearLocalIdentifier) else 1,
                    include_history=include_history,
                    maximum_held_out_rmse=maximum_rmse,
                ),
                identifier=identifier,
                expected_status=(
                    ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED
                ),
            )
        )
    return tuple(sorted(cases, key=lambda value: value.case_id))


def _transport_values(action: Decimal, receiver: Decimal) -> tuple[NamedDecimal, ...]:
    return (
        NamedDecimal(value_id="action", value=action, unit="1"),
        NamedDecimal(value_id="receiver", value=receiver, unit="1"),
    )


def build_transport_case(
    *,
    dataset_id: str,
    held_out_offset: Decimal = Decimal("0.2"),
    reverse: bool = False,
) -> TransportCase:
    source = get_reference_world(ReferenceWorldKind.STABLE_LINEAR).system
    target = get_reference_world(ReferenceWorldKind.DISCREPANCY_EXPLOITATION).system
    observations = []
    for split, actions in (
        (DataSplit.CALIBRATION, (Decimal("-1"), Decimal(0), Decimal(1))),
        (DataSplit.HELD_OUT, (Decimal("-1.5"), Decimal("0.5"), Decimal("1.5"))),
    ):
        for index, action in enumerate(actions):
            source_receiver = Decimal(2) * action
            offset = Decimal("0.2") if split is DataSplit.CALIBRATION else held_out_offset
            source_values = _transport_values(action, source_receiver)
            target_values = _transport_values(action, source_receiver + offset)
            if reverse:
                source_values, target_values = target_values, source_values
            observations.append(
                PairedTransportObservation(
                    pair_id=f"pair.{dataset_id}.{split.value.lower()}.{index}",
                    source_unit_instance_id=f"source-unit.{dataset_id}.{split.value.lower()}.{index}",
                    target_unit_instance_id=f"target-unit.{dataset_id}.{split.value.lower()}.{index}",
                    split=split,
                    denominator_cell_id="response-method-transport-cell",
                    horizon_id="reference-horizon",
                    source_values=source_values,
                    target_values=target_values,
                )
            )
    if reverse:
        source, target = target, source
    alignments = (
        QuantityAlignment(
            alignment_id="alignment.action",
            source_quantity_id="action",
            target_quantity_id="action",
            source_native_unit="1",
            target_native_unit="1",
        ),
        QuantityAlignment(
            alignment_id="alignment.receiver",
            source_quantity_id="receiver",
            target_quantity_id="receiver",
            source_native_unit="1",
            target_native_unit="1",
        ),
    )
    dataset = DirectionalTransportDataset(
        dataset_id=dataset_id,
        source_system=ObjectIdentity.from_record(source.system_id, source),
        target_system=ObjectIdentity.from_record(target.system_id, target),
        source_numerical_view_ids=("fine-view",),
        target_numerical_view_ids=("fine-view",),
        alignments=alignments,
        clock_relations=(
            ClockRelationSpec(
                relation_id=f"clock-relation.{dataset_id}",
                source_clock_id="reference-clock",
                target_clock_id="reference-clock",
                kind=ClockRelationKind.IDENTITY,
                delay=Decimal(0),
                tolerance=Decimal(0),
                evidence_contract_id="response-method-clock-identity-evidence",
            ),
        ),
        observations=tuple(sorted(observations, key=lambda value: value.pair_id)),
        evidence_artifacts=(
            ArtifactIdentity(
                artifact_id=f"artifact.{dataset_id}",
                role="paired-transport-evidence",
                payload_schema='empirical-lawhood/table/synthetic-paired-transport-conformance',
                sha256=hashlib.sha256(f"{dataset_id}:paired-evidence".encode()).hexdigest(),
                media_type="application/vnd.apache.parquet",
                size_bytes=len(observations) * 96,
            ),
        ),
        information_cutoff_id=f"cutoff.{dataset_id}",
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    config = DiscrepancyEstimatorConfig(
        config_id=f"config.{dataset_id}",
        estimator_key=DirectionalDiscrepancyEstimator.estimator_key,
        estimator_version=DirectionalDiscrepancyEstimator.estimator_version,
        criteria=(
            TransportCriterion(
                criterion_id="criterion.action",
                alignment_id="alignment.action",
                native_unit="1",
                maximum_held_out_rmse=Decimal(0),
                observation_uncertainty_bound=Decimal(0),
            ),
            TransportCriterion(
                criterion_id="criterion.receiver",
                alignment_id="alignment.receiver",
                native_unit="1",
                maximum_held_out_rmse=Decimal("0.01"),
                observation_uncertainty_bound=Decimal("0.01"),
            ),
        ),
        validity_domain_ids=("response-method-transport-cell",),
        assumption_ids=("identity-clock", "identity-native-unit-operator"),
        maximum_evidence=EvidenceCeiling.LOCAL_LAW,
    )
    return TransportCase(
        case_id=dataset_id,
        source_system=source,
        target_system=target,
        dataset=dataset,
        config=config,
        expected_status=(
            TransportStatus.SUPPORTED
            if held_out_offset == Decimal("0.2")
            else TransportStatus.NOT_SUPPORTED
        ),
    )


def transport_cases() -> tuple[TransportCase, ...]:
    return tuple(
        sorted(
            (
                build_transport_case(dataset_id="response-method-transport-forward"),
                build_transport_case(
                    dataset_id="response-method-transport-heldout-failure",
                    held_out_offset=Decimal(1),
                ),
                build_transport_case(dataset_id="response-method-transport-reverse", reverse=True),
            ),
            key=lambda value: value.case_id,
        )
    )


def _law_outcome(
    case: IdentificationCase, result: LawIdentificationResult
) -> LawConformanceOutcome:
    failed = tuple(sorted(check.check_id for check in result.checks if not check.passed))
    if result.scientific_status is not case.expected_status:
        raise ValueError(f'Parametric response-method law conformance status mismatch: {case.case_id}')
    emitted = result.response_law is not None
    if emitted is not (case.expected_status is ScientificStatus.SUPPORTED):
        raise ValueError(f'Parametric response-method law conformance emission mismatch: {case.case_id}')
    return LawConformanceOutcome(
        case_id=case.case_id,
        world_id=case.system.world.world_id,
        method_key=case.identifier.method_key,
        expected_status=case.expected_status,
        observed_status=result.scientific_status,
        response_law_emitted=emitted,
        failed_check_ids=failed,
        physical_unit_count=len(case.dataset.physical_unit_instance_ids),
        numerical_view_count=len(case.system.numerical_views),
        result_fingerprint=result.fingerprint(),
    )


def _transport_outcome(
    case: TransportCase,
    result: DirectionalTransportAssessment,
) -> TransportConformanceOutcome:
    observed = result.transport_result.status
    if observed is not case.expected_status:
        raise ValueError(f'Parametric response-method transport conformance status mismatch: {case.case_id}')
    return TransportConformanceOutcome(
        case_id=case.case_id,
        source_world_id=result.model_relation.source_world_id,
        target_world_id=result.model_relation.target_world_id,
        expected_status=case.expected_status,
        observed_status=observed,
        calibration_unit_ids=result.model_relation.calibration_unit_ids,
        held_out_unit_ids=result.model_relation.held_out_validation_unit_ids,
        failed_test_ids=result.transport_result.failed_test_ids,
        assessment_fingerprint=result.fingerprint(),
    )


def run_response_method_reference_conformance(
    capability_registry_fingerprint: str,
) -> ResponseMethodReferenceConformanceReport:
    payload_plane = EphemeralCandidatePayloadPlane()
    law_service = build_response_method_identification_service(
        payload_publisher=payload_plane,
        payload_reader=payload_plane,
    )
    law_outcomes = tuple(
        _law_outcome(
            case,
            law_service.identify(case.system, case.dataset, case.config, case.identifier),
        )
        for case in identification_cases()
    )
    transport_service = DirectionalDiscrepancyEstimator()
    transport_outcomes = tuple(
        _transport_outcome(
            case,
            transport_service.evaluate(
                case.source_system,
                case.target_system,
                case.dataset,
                case.config,
            ),
        )
        for case in transport_cases()
    )
    return ResponseMethodReferenceConformanceReport(
        report_id="response-method-reference-conformance",
        law_outcomes=law_outcomes,
        transport_outcomes=transport_outcomes,
        capability_registry_fingerprint=capability_registry_fingerprint,
        status="PASS",
    )
