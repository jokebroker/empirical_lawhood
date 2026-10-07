"""Calibration/held-out directional discrepancy and transport qualification."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.models import (
    DirectionalDiscrepancyEstimate,
    DiscrepancyBound,
    ModelRelation,
    QuantityAlignment,
    TransportResult,
    TransportStatus,
    validate_transport_result,
)
from empirical_lawhood.kernel.obligations import DiscrepancySpec, ObligationStatus, ValiditySpec
from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    ObjectIdentity,
)
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import ClockRelationSpec

from .contracts import DataSplit


@dataclass(frozen=True, slots=True)
class PairedTransportObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/paired-transport-observation'

    pair_id: str
    source_unit_instance_id: str
    target_unit_instance_id: str
    split: DataSplit
    denominator_cell_id: str
    horizon_id: str
    source_values: tuple[NamedDecimal, ...]
    target_values: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_id", self.pair_id),
            ("source_unit_instance_id", self.source_unit_instance_id),
            ("target_unit_instance_id", self.target_unit_instance_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("source_values", self.source_values),
            ("target_values", self.target_values),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
            if not values:
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class DirectionalTransportDataset(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/directional-transport-dataset'

    dataset_id: str
    source_system: ObjectIdentity
    target_system: ObjectIdentity
    source_numerical_view_ids: tuple[str, ...]
    target_numerical_view_ids: tuple[str, ...]
    alignments: tuple[QuantityAlignment, ...]
    clock_relations: tuple[ClockRelationSpec, ...]
    observations: tuple[PairedTransportObservation, ...]
    evidence_artifacts: tuple[ArtifactIdentity, ...]
    information_cutoff_id: str
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.dataset_id, field_name="dataset_id")
        validate_stable_id(self.information_cutoff_id, field_name="information_cutoff_id")
        if self.source_system == self.target_system:
            raise ValueError("directional transport requires distinct source and target systems")
        for name, values in (
            ("source_numerical_view_ids", self.source_numerical_view_ids),
            ("target_numerical_view_ids", self.target_numerical_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        require_sorted_unique_ids(
            self.alignments,
            attribute="alignment_id",
            field_name="alignments",
        )
        if not self.alignments:
            raise ValueError("directional transport requires quantity alignments")
        require_sorted_unique_ids(
            self.clock_relations,
            attribute="relation_id",
            field_name="clock_relations",
        )
        if not self.clock_relations:
            raise ValueError("directional transport requires clock relations")
        require_sorted_unique_ids(
            self.observations,
            attribute="pair_id",
            field_name="observations",
        )
        if {observation.split for observation in self.observations} != set(DataSplit):
            raise ValueError("transport dataset requires calibration and held-out pairs")
        require_sorted_unique_ids(
            self.evidence_artifacts,
            attribute="artifact_id",
            field_name="evidence_artifacts",
        )
        if not self.evidence_artifacts:
            raise ValueError("transport dataset requires external evidence identities")
        calibration = tuple(
            value for value in self.observations if value.split is DataSplit.CALIBRATION
        )
        held_out = tuple(value for value in self.observations if value.split is DataSplit.HELD_OUT)
        for attribute in ("pair_id", "source_unit_instance_id", "target_unit_instance_id"):
            if {getattr(value, attribute) for value in calibration} & {
                getattr(value, attribute) for value in held_out
            }:
                raise ValueError("calibration and held-out transport identities overlap")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("transport dataset visibility cannot be lowered")

    def split(self, split: DataSplit) -> tuple[PairedTransportObservation, ...]:
        return tuple(value for value in self.observations if value.split is split)


@dataclass(frozen=True, slots=True)
class TransportCriterion(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transport-criterion'

    criterion_id: str
    alignment_id: str
    native_unit: str
    maximum_held_out_rmse: Decimal
    observation_uncertainty_bound: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.criterion_id, field_name="criterion_id")
        validate_stable_id(self.alignment_id, field_name="alignment_id")
        if not self.native_unit.strip():
            raise ValueError("transport criterion native_unit must not be empty")
        for name, value in (
            ("maximum_held_out_rmse", self.maximum_held_out_rmse),
            ("observation_uncertainty_bound", self.observation_uncertainty_bound),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class DiscrepancyEstimatorConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/discrepancy-estimator-config'

    config_id: str
    estimator_key: str
    estimator_version: str
    criteria: tuple[TransportCriterion, ...]
    validity_domain_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    maximum_evidence: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.estimator_key, field_name="estimator_key")
        validate_semantic_version(self.estimator_version)
        require_sorted_unique_ids(
            self.criteria,
            attribute="criterion_id",
            field_name="criteria",
        )
        if not self.criteria:
            raise ValueError("discrepancy estimator requires aligned criteria")
        for name, values in (
            ("validity_domain_ids", self.validity_domain_ids),
            ("assumption_ids", self.assumption_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)


@dataclass(frozen=True, slots=True)
class ObservationOperatorSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observation-operator-spec'

    operator_id: str
    alignments: tuple[QuantityAlignment, ...]
    clock_relation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.operator_id, field_name="operator_id")
        require_sorted_unique_ids(
            self.alignments,
            attribute="alignment_id",
            field_name="alignments",
        )
        require_sorted_unique_strings(
            self.clock_relation_ids,
            field_name="clock_relation_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class DirectionalTransportAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/directional-transport-assessment'

    assessment_id: str
    estimate: DirectionalDiscrepancyEstimate
    model_relation: ModelRelation
    transport_result: TransportResult

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.model_relation.discrepancy != self.estimate.discrepancy:
            raise ValueError("transport assessment relation and estimate discrepancy differ")
        validate_transport_result(self.model_relation, self.transport_result)


def _values(values: tuple[NamedDecimal, ...]) -> dict[str, NamedDecimal]:
    return {value.value_id: value for value in values}


class DirectionalDiscrepancyEstimator:
    estimator_key = "baseline.directional-max-residual"
    estimator_version = "1.0.0"

    def evaluate(
        self,
        source_system: SystemSpec,
        target_system: SystemSpec,
        dataset: DirectionalTransportDataset,
        config: DiscrepancyEstimatorConfig,
    ) -> DirectionalTransportAssessment:
        self._validate_inputs(source_system, target_system, dataset, config)
        calibration = dataset.split(DataSplit.CALIBRATION)
        held_out = dataset.split(DataSplit.HELD_OUT)
        calibration_ids = tuple(value.pair_id for value in calibration)
        held_out_ids = tuple(value.pair_id for value in held_out)
        criteria = {value.alignment_id: value for value in config.criteria}
        bounds: list[DiscrepancyBound] = []
        metrics: list[NamedDecimal] = []
        failed: list[str] = []
        for alignment in dataset.alignments:
            criterion = criteria[alignment.alignment_id]
            residuals = tuple(
                _values(value.target_values)[alignment.target_quantity_id].value
                - _values(value.source_values)[alignment.source_quantity_id].value
                for value in calibration
            )
            bias = sum(residuals, Decimal(0)) / Decimal(len(residuals))
            deviation = max((abs(value - bias) for value in residuals), default=Decimal(0))
            bound_id = f"bound.{dataset.dataset_id}.{alignment.alignment_id}"
            bounds.append(
                DiscrepancyBound(
                    bound_id=bound_id,
                    alignment_id=alignment.alignment_id,
                    source_quantity_id=alignment.source_quantity_id,
                    target_quantity_id=alignment.target_quantity_id,
                    native_unit=criterion.native_unit,
                    directional_bias=bias,
                    absolute_deviation_bound=deviation,
                    calibration_unit_ids=calibration_ids,
                )
            )
            held_out_errors = tuple(
                _values(value.target_values)[alignment.target_quantity_id].value
                - (_values(value.source_values)[alignment.source_quantity_id].value + bias)
                for value in held_out
            )
            rmse = (
                sum((value * value for value in held_out_errors), Decimal(0))
                / Decimal(len(held_out_errors))
            ).sqrt()
            allowed = deviation + criterion.observation_uncertainty_bound
            violations = sum(abs(value) > allowed for value in held_out_errors)
            metrics.extend(
                (
                    NamedDecimal(
                        value_id=f"held-out-rmse.{alignment.alignment_id}",
                        value=rmse,
                        unit=criterion.native_unit,
                    ),
                    NamedDecimal(
                        value_id=f"observation-bound.{alignment.alignment_id}",
                        value=criterion.observation_uncertainty_bound,
                        unit=criterion.native_unit,
                    ),
                    NamedDecimal(
                        value_id=f"violation-count.{alignment.alignment_id}",
                        value=Decimal(violations),
                        unit="1",
                    ),
                )
            )
            if rmse > criterion.maximum_held_out_rmse:
                failed.append(f"held-out-rmse.{alignment.alignment_id}")
            if violations:
                failed.append(f"held-out-bound.{alignment.alignment_id}")
        evidence_link, operator = self._evidence(dataset, source_system, target_system)
        status = ObligationStatus.SATISFIED if not failed else ObligationStatus.FAILED
        effective_ceiling = (
            config.maximum_evidence
            if dataset.visibility_ceiling.is_promotable
            else EvidenceCeiling.NON_PROMOTABLE
        )
        discrepancy = DiscrepancySpec(
            discrepancy_id=f"discrepancy.{dataset.dataset_id}",
            source_world_id=source_system.world.world_id,
            target_world_id=target_system.world.world_id,
            aligned_quantity_ids=tuple(
                sorted(alignment.source_quantity_id for alignment in dataset.alignments)
            ),
            aligned_clock_relation_ids=tuple(
                relation.relation_id for relation in dataset.clock_relations
            ),
            calibration_unit_ids=calibration_ids,
            held_out_validation_unit_ids=held_out_ids,
            discrepancy_bound_ids=tuple(sorted(value.bound_id for value in bounds)),
            failed_test_ids=tuple(sorted(set(failed))),
            evidence_ceiling=effective_ceiling,
            status=status,
            evidence_link_ids=(evidence_link.link_id,),
        )
        estimate = DirectionalDiscrepancyEstimate(
            estimate_id=f"estimate.{dataset.dataset_id}",
            dataset=ObjectIdentity.from_record(dataset.dataset_id, dataset),
            config=ObjectIdentity.from_record(config.config_id, config),
            discrepancy=discrepancy,
            bounds=tuple(sorted(bounds, key=lambda value: value.bound_id)),
            held_out_metrics=tuple(sorted(metrics, key=lambda value: value.value_id)),
            status=status,
            evidence_links=(evidence_link,),
            outcome_access=dataset.outcome_access,
            parent_visibility_ceilings=dataset.parent_visibility_ceilings,
            visibility_ceiling=dataset.visibility_ceiling,
        )
        validity = ValiditySpec(
            validity_id=f"validity.{dataset.dataset_id}",
            validity_domain_ids=config.validity_domain_ids,
            assumption_ids=config.assumption_ids,
            exclusion_reason_codes=tuple(sorted(set(failed))),
            status=status,
            evidence_link_ids=(evidence_link.link_id,),
        )
        relation = ModelRelation(
            model_relation_id=f"relation.{dataset.dataset_id}",
            source_world_id=source_system.world.world_id,
            target_world_id=target_system.world.world_id,
            source_numerical_view_ids=dataset.source_numerical_view_ids,
            target_numerical_view_ids=dataset.target_numerical_view_ids,
            quantity_alignments=dataset.alignments,
            clock_relations=dataset.clock_relations,
            observation_operator=operator,
            calibration_unit_ids=calibration_ids,
            held_out_validation_unit_ids=held_out_ids,
            discrepancy=discrepancy,
            validity=validity,
            maximum_evidence=effective_ceiling,
            outcome_access=dataset.outcome_access,
            parent_visibility_ceilings=dataset.parent_visibility_ceilings,
            visibility_ceiling=dataset.visibility_ceiling,
            evidence_links=(evidence_link,),
        )
        transport = TransportResult(
            transport_result_id=f"transport.{dataset.dataset_id}",
            model_relation=ObjectIdentity.from_record(relation.model_relation_id, relation),
            status=(
                TransportStatus.SUPPORTED
                if status is ObligationStatus.SATISFIED
                else TransportStatus.NOT_SUPPORTED
            ),
            metrics=estimate.held_out_metrics,
            failed_test_ids=discrepancy.failed_test_ids,
            achieved_evidence_ceiling=(
                EvidenceCeiling.RESPONSE
                if effective_ceiling.allows(EvidenceRung.RESPONSE)
                else EvidenceCeiling.NON_PROMOTABLE
            ),
            evidence_links=(evidence_link,),
            outcome_access=dataset.outcome_access,
            parent_visibility_ceiling=dataset.visibility_ceiling,
            visibility_ceiling=dataset.visibility_ceiling,
        )
        return DirectionalTransportAssessment(
            assessment_id=f"assessment.{dataset.dataset_id}",
            estimate=estimate,
            model_relation=relation,
            transport_result=transport,
        )

    def _validate_inputs(
        self,
        source_system: SystemSpec,
        target_system: SystemSpec,
        dataset: DirectionalTransportDataset,
        config: DiscrepancyEstimatorConfig,
    ) -> None:
        self._validate_selection(source_system, target_system, dataset, config)
        self._validate_alignments(source_system, target_system, dataset, config)
        self._validate_observations(source_system, target_system, dataset)
        self._validate_clocks(source_system, target_system, dataset)

    def _validate_selection(
        self,
        source_system: SystemSpec,
        target_system: SystemSpec,
        dataset: DirectionalTransportDataset,
        config: DiscrepancyEstimatorConfig,
    ) -> None:
        if config.estimator_key != self.estimator_key or (
            config.estimator_version != self.estimator_version
        ):
            raise ValueError("discrepancy configuration selects another estimator")
        if dataset.source_system != ObjectIdentity.from_record(
            source_system.system_id, source_system
        ) or dataset.target_system != ObjectIdentity.from_record(
            target_system.system_id, target_system
        ):
            raise ValueError("transport dataset is bound to another directional system pair")
        if source_system.world.world_id == target_system.world.world_id:
            raise ValueError("directional transport requires distinct evidence worlds")
        if not set(dataset.source_numerical_view_ids).issubset(
            {value.view_id for value in source_system.numerical_views}
        ) or not set(dataset.target_numerical_view_ids).issubset(
            {value.view_id for value in target_system.numerical_views}
        ):
            raise ValueError("transport dataset uses an unknown numerical view")

    @staticmethod
    def _validate_alignments(
        source_system: SystemSpec,
        target_system: SystemSpec,
        dataset: DirectionalTransportDataset,
        config: DiscrepancyEstimatorConfig,
    ) -> None:
        criteria = {value.alignment_id: value for value in config.criteria}
        if set(criteria) != {value.alignment_id for value in dataset.alignments}:
            raise ValueError("transport criteria do not cover every aligned quantity")
        source_quantities = {value.quantity_id: value for value in source_system.quantities}
        target_quantities = {value.quantity_id: value for value in target_system.quantities}
        for alignment in dataset.alignments:
            source_quantity = source_quantities.get(alignment.source_quantity_id)
            target_quantity = target_quantities.get(alignment.target_quantity_id)
            if source_quantity is None or target_quantity is None:
                raise ValueError("transport alignment references an unknown system quantity")
            if source_quantity.native_unit != alignment.source_native_unit or (
                target_quantity.native_unit != alignment.target_native_unit
            ):
                raise ValueError("transport alignment unit differs from its system quantity")
            if alignment.conversion_reference_id is not None:
                raise ValueError("baseline identity operator cannot execute unit conversion")
            criterion = criteria[alignment.alignment_id]
            if criterion.native_unit != alignment.target_native_unit:
                raise ValueError("transport criterion unit differs from target native unit")

    @staticmethod
    def _validate_observations(
        source_system: SystemSpec,
        target_system: SystemSpec,
        dataset: DirectionalTransportDataset,
    ) -> None:
        for observation in dataset.observations:
            if (
                observation.horizon_id != source_system.relation.horizon.horizon_id
                or observation.horizon_id != target_system.relation.horizon.horizon_id
                or source_system.relation.horizon.duration
                != target_system.relation.horizon.duration
                or source_system.relation.horizon.time_unit
                != target_system.relation.horizon.time_unit
            ):
                raise ValueError("transport pair uses an unaligned response horizon")
            source = _values(observation.source_values)
            target = _values(observation.target_values)
            for alignment in dataset.alignments:
                if source.get(alignment.source_quantity_id) is None or (
                    target.get(alignment.target_quantity_id) is None
                ):
                    raise ValueError("transport pair omits an aligned quantity")
                if source[alignment.source_quantity_id].unit != alignment.source_native_unit or (
                    target[alignment.target_quantity_id].unit != alignment.target_native_unit
                ):
                    raise ValueError("transport pair uses the wrong native unit")

    @staticmethod
    def _validate_clocks(
        source_system: SystemSpec,
        target_system: SystemSpec,
        dataset: DirectionalTransportDataset,
    ) -> None:
        source_clocks = {value.clock_id for value in source_system.clocks}
        target_clocks = {value.clock_id for value in target_system.clocks}
        if any(
            relation.source_clock_id not in source_clocks
            or relation.target_clock_id not in target_clocks
            for relation in dataset.clock_relations
        ):
            raise ValueError("transport clock relation references an unknown directional clock")

    @staticmethod
    def _evidence(
        dataset: DirectionalTransportDataset,
        source_system: SystemSpec,
        target_system: SystemSpec,
    ) -> tuple[EvidenceLink, ExecutableReference]:
        operator_spec = ObservationOperatorSpec(
            operator_id=f"operator.{dataset.dataset_id}",
            alignments=dataset.alignments,
            clock_relation_ids=tuple(value.relation_id for value in dataset.clock_relations),
        )
        payload = ArtifactIdentity(
            artifact_id=f"artifact.{operator_spec.operator_id}",
            role="observation-operator",
            payload_schema=operator_spec.SCHEMA,
            sha256=operator_spec.fingerprint(),
            media_type="application/json",
            size_bytes=len(operator_spec.canonical_bytes()),
        )
        operator = ExecutableReference(
            reference_id=f"evaluator.{operator_spec.operator_id}",
            capability_key="baseline.identity-observation-operator",
            capability_version="1.0.0",
            evaluator_key="identity-quantity-alignment",
            payload=payload,
            payload_format=SafePayloadFormat.CANONICAL_JSON,
            input_schema=PairedTransportObservation.SCHEMA,
            output_schema='empirical-lawhood/methods/aligned-transport-observation',
            deterministic=True,
        )
        link = EvidenceLink(
            link_id=f"evidence.{dataset.dataset_id}",
            relation=EvidenceRelation.DERIVED_FROM,
            source=ObjectIdentity.from_record(dataset.dataset_id, dataset),
            target=ObjectIdentity.from_record(target_system.system_id, target_system),
            artifact_ids=tuple(
                sorted(
                    {
                        payload.artifact_id,
                        *(value.artifact_id for value in dataset.evidence_artifacts),
                    }
                )
            ),
            world_id=target_system.world.world_id,
            information_cutoff_id=dataset.information_cutoff_id,
            outcome_access=dataset.outcome_access,
            visibility_ceiling=dataset.visibility_ceiling,
            parent_visibility_ceilings=dataset.parent_visibility_ceilings,
            reason=(
                f"Directional {source_system.world.world_id} to "
                f"{target_system.world.world_id} calibration and held-out assessment."
            ),
        )
        return link, operator
