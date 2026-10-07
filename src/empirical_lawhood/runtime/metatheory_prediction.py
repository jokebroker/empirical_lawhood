"""Issue-before-contact and evaluator-only structural adjudication services."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryPredictiveLevel
from empirical_lawhood.planning.metatheory_prediction import CategoricalForecast, DynamicalForecast, MetatheoryAdjudicationCell, MetatheoryAdjudicationResult, CustodyBoundMetatheoryAdjudicationResult, MetatheoryAdjudicationSpec, MetatheoryAdjudicationStopKind, MetatheoryAdjudicationStop, MetatheoryForecastCell, MetatheoryMissingTargetPolicy, MetatheoryPredictionIssueReceipt, MetatheoryPredictionPackage, MetatheoryRevealAuthorizationBinding, MetatheorySealedTarget, MetatheoryTargetAcquisitionIndex, MetatheoryTargetOutcome, MetricForecast, RevealedTargetOutcomeBinding
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class MetatheoryPredictionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MetatheoryPredictionIssueService:
    def issue(
        self,
        *,
        package: MetatheoryPredictionPackage,
        issue_clock_id: str,
        publication: ObjectIdentity,
        recovery: ObjectIdentity,
    ) -> MetatheoryPredictionIssueReceipt:
        if package.target_outcome_access_count:
            raise MetatheoryPredictionError("PREDICTION_PACKAGE_READ_TARGET_OUTCOMES")
        if package.planned_issue_id != issue_clock_id:
            raise MetatheoryPredictionError("PREDICTION_ISSUE_IDENTITY_MISMATCH")
        return MetatheoryPredictionIssueReceipt(
            receipt_id=f"receipt.{package.planned_issue_id}",
            prediction_package=ObjectIdentity.from_record(package.package_id, package),
            issue_clock_id=issue_clock_id,
            issued_before_target_contact=True,
            target_contact_count=0,
            target_outcome_access_count=0,
            publication=publication,
            recovery=recovery,
            grants_execution_or_reveal_authority=False,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryTargetAcquisitionService:
    """Finalize an exact post-issue sealed-target roster without revealing outcomes."""

    def finalize(
        self,
        *,
        package: MetatheoryPredictionPackage,
        issue: MetatheoryPredictionIssueReceipt,
        sealed_targets: tuple[MetatheorySealedTarget, ...],
    ) -> MetatheoryTargetAcquisitionIndex:
        issue_identity = ObjectIdentity.from_record(issue.receipt_id, issue)
        if issue.prediction_package != ObjectIdentity.from_record(package.package_id, package):
            raise MetatheoryPredictionError("TARGET_ACQUISITION_ISSUE_IDENTITY_MISMATCH")
        targets = {value.target_id: value for value in package.targets}
        sealed = {value.target_id: value for value in sealed_targets}
        if len(sealed) != len(sealed_targets) or set(sealed) - set(targets):
            raise MetatheoryPredictionError("TARGET_ACQUISITION_ROSTER_MISMATCH")
        for target_id, target in targets.items():
            acquired = sealed.get(target_id)
            if acquired is None:
                if target.missing_target_policy is MetatheoryMissingTargetPolicy.REQUIRE_COMPLETE:
                    raise MetatheoryPredictionError("REQUIRED_TARGET_NOT_ACQUIRED")
                continue
            if acquired.acquisition_complete != (
                acquired.acquired_physical_unit_ids == target.physical_unit_ids
            ):
                raise MetatheoryPredictionError("TARGET_ACQUISITION_COMPLETENESS_MISMATCH")
            if not set(acquired.acquired_physical_unit_ids).issubset(set(target.physical_unit_ids)):
                raise MetatheoryPredictionError("TARGET_ACQUISITION_UNIT_ROSTER_MISMATCH")
            if (
                not acquired.acquisition_complete
                and target.missing_target_policy is MetatheoryMissingTargetPolicy.REQUIRE_COMPLETE
            ):
                raise MetatheoryPredictionError("REQUIRED_TARGET_ACQUISITION_INCOMPLETE")
        return MetatheoryTargetAcquisitionIndex(
            index_id=f"acquisition.{issue.receipt_id}",
            prediction_issue=issue_identity,
            sealed_targets=tuple(sorted(sealed_targets, key=lambda value: value.target_id)),
            acquired_after_issue=True,
            outcomes_exposed=False,
            visibility=VisibilityCeiling.PROSPECTIVE,
        )


@dataclass(frozen=True, slots=True)
class MetatheoryAdjudicationService:
    capability_registry: CapabilityRegistry

    def adjudicate_outcomes(
        self,
        *,
        package: MetatheoryPredictionPackage,
        issue: MetatheoryPredictionIssueReceipt,
        acquisition: MetatheoryTargetAcquisitionIndex,
        reveal: MetatheoryRevealAuthorizationBinding,
        spec: MetatheoryAdjudicationSpec,
        dependence_assessment: EvidenceDependenceAssessment,
        outcomes: tuple[MetatheoryTargetOutcome, ...],
    ) -> MetatheoryAdjudicationResult | MetatheoryAdjudicationStop:
        issue_identity = ObjectIdentity.from_record(issue.receipt_id, issue)
        acquisition_identity = ObjectIdentity.from_record(acquisition.index_id, acquisition)
        if issue.prediction_package != ObjectIdentity.from_record(package.package_id, package):
            return self._stop(
                issue_identity, MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH
            )
        if (
            acquisition.prediction_issue != issue_identity
            or reveal.prediction_issue != issue_identity
            or reveal.acquisition_index != acquisition_identity
            or spec.prediction_package != issue.prediction_package
            or dependence_assessment.dependence_spec != package.dependence_spec
            or spec.scoring_methods != package.scoring_methods
        ):
            return self._stop(
                issue_identity, MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH
            )
        if not reveal.reveal_permitted:
            return self._stop(
                issue_identity,
                MetatheoryAdjudicationStopKind.REVEAL_AUTHORITY_REQUIRED,
                reveal.reason_codes,
            )
        for binding in spec.scoring_methods:
            method = binding.method
            self._require_method(
                method.capability_key,
                method.capability_version,
                method.implementation_sha256,
                method.config.object_schema,
            )

        targets = {value.target_id: value for value in package.targets}
        sealed = {value.target_id: value for value in acquisition.sealed_targets}
        if set(sealed) - set(targets):
            return self._stop(
                issue_identity, MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH
            )
        for target_id, target in targets.items():
            acquired = sealed.get(target_id)
            if acquired is None or not acquired.acquisition_complete:
                if target.missing_target_policy is MetatheoryMissingTargetPolicy.REQUIRE_COMPLETE:
                    return self._stop(
                        issue_identity,
                        MetatheoryAdjudicationStopKind.ACQUISITION_INCOMPLETE,
                    )
            elif acquired.acquired_physical_unit_ids != target.physical_unit_ids:
                return self._stop(
                    issue_identity,
                    MetatheoryAdjudicationStopKind.ACQUISITION_INCOMPLETE,
                )

        outcome_by_cell = {value.forecast_cell_id: value for value in outcomes}
        if len(outcome_by_cell) != len(outcomes) or set(outcome_by_cell) - {
            value.cell_id for value in package.forecast_cells
        }:
            return self._stop(
                issue_identity, MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH
            )
        cells = []
        for forecast in package.forecast_cells:
            outcome = outcome_by_cell.get(forecast.cell_id)
            if outcome is None:
                target = targets[forecast.target_id]
                if target.missing_target_policy is MetatheoryMissingTargetPolicy.REQUIRE_COMPLETE:
                    return self._stop(
                        issue_identity,
                        MetatheoryAdjudicationStopKind.REQUIRED_TARGET_MISSING,
                    )
                cells.append(self._missing_cell(forecast))
                continue
            if (
                outcome.target_id != forecast.target_id
                or outcome.physical_unit_id != forecast.physical_unit_id
                or outcome.predictive_level is not forecast.predictive_level
            ):
                return self._stop(
                    issue_identity,
                    MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH,
                )
            cells.append(self._score(forecast, outcome))

        by_level = {
            level: self._aggregate(
                tuple(value.disposition for value in cells if value.predictive_level is level)
            )
            for level in MetatheoryPredictiveLevel
        }
        active_levels = tuple(value for value in by_level.values() if value is not None)
        if dependence_assessment.achieved_class is None:
            aggregate = MetatheoryAggregateDisposition.UNEVALUABLE
        elif MetatheoryAggregateDisposition.OPPOSED in active_levels:
            aggregate = (
                MetatheoryAggregateDisposition.OPPOSED
                if set(active_levels) == {MetatheoryAggregateDisposition.OPPOSED}
                else MetatheoryAggregateDisposition.MIXED
            )
        elif MetatheoryAggregateDisposition.UNEVALUABLE in active_levels:
            aggregate = (
                MetatheoryAggregateDisposition.UNEVALUABLE
                if set(active_levels) == {MetatheoryAggregateDisposition.UNEVALUABLE}
                else MetatheoryAggregateDisposition.MIXED
            )
        else:
            aggregate = MetatheoryAggregateDisposition.SUPPORTED
        # The complete-unit denominator is the union, never the number of views/cells.
        units = {unit for target in package.targets for unit in target.physical_unit_ids}
        return MetatheoryAdjudicationResult(
            result_id=f"result.{spec.spec_id}",
            prediction_issue=issue_identity,
            reveal_authorization=ObjectIdentity.from_record(reveal.binding_id, reveal),
            acquisition_index=acquisition_identity,
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            complete_physical_unit_count=len(units),
            dependence_assessment=dependence_assessment,
            achieved_dependence_class=dependence_assessment.achieved_class,
            categorical_disposition=by_level[MetatheoryPredictiveLevel.CATEGORICAL],
            metric_disposition=by_level[MetatheoryPredictiveLevel.METRIC],
            dynamical_disposition=by_level[MetatheoryPredictiveLevel.DYNAMICAL],
            disposition=aggregate,
            maximum_ordinary_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            maximum_structural_evidence_ceiling=package.maximum_structural_evidence_ceiling,
            obstruction_refs=(),
            parent_promotion_permitted=False,
        )

    def adjudicate_custodied_outcomes(
        self,
        *,
        package: MetatheoryPredictionPackage,
        issue: MetatheoryPredictionIssueReceipt,
        acquisition: MetatheoryTargetAcquisitionIndex,
        reveal: MetatheoryRevealAuthorizationBinding,
        spec: MetatheoryAdjudicationSpec,
        dependence_assessment: EvidenceDependenceAssessment,
        outcomes: tuple[MetatheoryTargetOutcome, ...],
        outcome_bindings: tuple[RevealedTargetOutcomeBinding, ...],
    ) -> CustodyBoundMetatheoryAdjudicationResult | MetatheoryAdjudicationStop:
        """Adjudicate only after every revealed outcome joins sealed custody."""

        issue_identity = ObjectIdentity.from_record(issue.receipt_id, issue)
        acquisition_identity = ObjectIdentity.from_record(acquisition.index_id, acquisition)
        reveal_identity = ObjectIdentity.from_record(reveal.binding_id, reveal)
        if issue.prediction_package != ObjectIdentity.from_record(package.package_id, package):
            return self._stop(
                issue_identity,
                MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH,
            )
        if (
            acquisition.prediction_issue != issue_identity
            or reveal.prediction_issue != issue_identity
            or reveal.acquisition_index != acquisition_identity
            or spec.prediction_package != issue.prediction_package
            or dependence_assessment.dependence_spec != package.dependence_spec
            or spec.scoring_methods != package.scoring_methods
        ):
            return self._stop(
                issue_identity,
                MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH,
            )
        if not reveal.reveal_permitted:
            return self._stop(
                issue_identity,
                MetatheoryAdjudicationStopKind.REVEAL_AUTHORITY_REQUIRED,
                reveal.reason_codes,
            )

        outcome_by_identity = {
            ObjectIdentity.from_record(value.outcome_id, value): value for value in outcomes
        }
        binding_by_outcome = {value.outcome: value for value in outcome_bindings}
        if (
            len(outcome_by_identity) != len(outcomes)
            or len(binding_by_outcome) != len(outcome_bindings)
            or set(binding_by_outcome) != set(outcome_by_identity)
        ):
            return self._custody_stop(issue_identity)
        forecasts = {value.cell_id: value for value in package.forecast_cells}
        sealed = {value.target_id: value for value in acquisition.sealed_targets}
        for outcome_identity, outcome in outcome_by_identity.items():
            binding = binding_by_outcome[outcome_identity]
            forecast = forecasts.get(outcome.forecast_cell_id)
            sealed_target = sealed.get(outcome.target_id)
            if (
                forecast is None
                or sealed_target is None
                or not sealed_target.acquisition_complete
                or outcome.physical_unit_id not in set(sealed_target.acquired_physical_unit_ids)
                or binding.prediction_issue != issue_identity
                or binding.acquisition_index != acquisition_identity
                or binding.sealed_target
                != ObjectIdentity.from_record(sealed_target.target_id, sealed_target)
                or binding.sealed_artifact != sealed_target.sealed_artifact
                or binding.forecast_cell != ObjectIdentity.from_record(forecast.cell_id, forecast)
                or binding.outcome != outcome_identity
                or binding.target_id != outcome.target_id
                or binding.physical_unit_id != outcome.physical_unit_id
                or binding.predictive_level is not outcome.predictive_level
                or binding.reveal_authorization != reveal_identity
            ):
                return self._custody_stop(issue_identity)

        adjudication_result = self.adjudicate_outcomes(
            package=package,
            issue=issue,
            acquisition=acquisition,
            reveal=reveal,
            spec=spec,
            dependence_assessment=dependence_assessment,
            outcomes=outcomes,
        )
        if isinstance(adjudication_result, MetatheoryAdjudicationStop):
            return adjudication_result
        return CustodyBoundMetatheoryAdjudicationResult(
            result_id=f"custody-bound-metatheory-result.{spec.spec_id}",
            adjudication_result=adjudication_result,
            outcome_bindings=tuple(sorted(outcome_bindings, key=lambda value: value.binding_id)),
        )

    @staticmethod
    def _missing_cell(forecast: MetatheoryForecastCell) -> MetatheoryAdjudicationCell:
        return MetatheoryAdjudicationCell(
            cell_id=f"adjudication.{forecast.cell_id}",
            forecast_cell=ObjectIdentity.from_record(forecast.cell_id, forecast),
            outcome=None,
            predictive_level=forecast.predictive_level,
            score=None,
            decisive_falsifier_ids=(),
            disposition=MetatheoryCellDisposition.UNEVALUABLE,
            reason_codes=("PERMITTED_TARGET_MISSING",),
            evidence_links=(),
        )

    @staticmethod
    def _score(
        forecast: MetatheoryForecastCell,
        outcome: MetatheoryTargetOutcome,
    ) -> MetatheoryAdjudicationCell:
        supported: bool | None
        decisive = False
        score: NamedDecimal | None
        if forecast.categorical is not None:
            categorical: CategoricalForecast = forecast.categorical
            assert outcome.observed_category_id is not None
            supported = outcome.observed_category_id == categorical.predicted_category_id
            decisive = outcome.observed_category_id in categorical.unsafe_observed_category_ids
            score = NamedDecimal(
                f"score.{forecast.cell_id}",
                Decimal(1 if supported else 0),
                "1",
            )
        elif forecast.metric is not None:
            metric: MetricForecast = forecast.metric
            assert outcome.observed_metric is not None
            if outcome.observed_metric.unit != metric.native_unit:
                raise MetatheoryPredictionError("METRIC_OUTCOME_NATIVE_UNIT_MISMATCH")
            supported = (
                metric.interval_lower.value
                <= outcome.observed_metric.value
                <= metric.interval_upper.value
            )
            score = NamedDecimal(
                f"score.{forecast.cell_id}",
                outcome.observed_metric.value,
                metric.native_unit,
            )
        else:
            dynamical: DynamicalForecast = forecast.dynamical  # type: ignore[assignment]
            if outcome.censored:
                supported = None
                score = None
            else:
                assert outcome.observed_transition_id is not None
                assert outcome.observed_first_passage is not None
                if outcome.observed_first_passage.unit != dynamical.horizon.unit:
                    raise MetatheoryPredictionError("DYNAMICAL_OUTCOME_NATIVE_UNIT_MISMATCH")
                supported = (
                    outcome.observed_transition_id == dynamical.transition_id
                    and dynamical.first_passage_lower.value
                    <= outcome.observed_first_passage.value
                    <= dynamical.first_passage_upper.value
                )
                score = outcome.observed_first_passage
        reasons: tuple[str, ...]
        falsifiers: tuple[str, ...]
        if supported is True:
            disposition = MetatheoryCellDisposition.SUPPORTED
            reasons = ()
            falsifiers = ()
        elif supported is False:
            disposition = MetatheoryCellDisposition.OPPOSED
            reasons = (
                ("DECISIVE_UNSAFE_CATEGORY_OBSERVED" if decisive else "FROZEN_FORECAST_OPPOSED"),
            )
            falsifiers = forecast.falsifier_ids
        else:
            disposition = MetatheoryCellDisposition.UNEVALUABLE
            reasons = ("DYNAMICAL_OUTCOME_CENSORED",)
            falsifiers = ()
        return MetatheoryAdjudicationCell(
            cell_id=f"adjudication.{forecast.cell_id}",
            forecast_cell=ObjectIdentity.from_record(forecast.cell_id, forecast),
            outcome=ObjectIdentity.from_record(outcome.outcome_id, outcome),
            predictive_level=forecast.predictive_level,
            score=score,
            decisive_falsifier_ids=falsifiers,
            disposition=disposition,
            reason_codes=reasons,
            evidence_links=outcome.evidence_links,
        )

    @staticmethod
    def _aggregate(
        dispositions: tuple[MetatheoryCellDisposition, ...],
    ) -> MetatheoryAggregateDisposition | None:
        if not dispositions:
            return None
        values = set(dispositions)
        if values == {MetatheoryCellDisposition.SUPPORTED}:
            return MetatheoryAggregateDisposition.SUPPORTED
        if values == {MetatheoryCellDisposition.OPPOSED}:
            return MetatheoryAggregateDisposition.OPPOSED
        if values == {MetatheoryCellDisposition.UNEVALUABLE}:
            return MetatheoryAggregateDisposition.UNEVALUABLE
        return MetatheoryAggregateDisposition.MIXED

    @staticmethod
    def _stop(
        issue: ObjectIdentity,
        kind: MetatheoryAdjudicationStopKind,
        reasons: tuple[str, ...] = (),
    ) -> MetatheoryAdjudicationStop:
        return MetatheoryAdjudicationStop(
            stop_id=f"stop.{issue.object_id}.{kind.value.lower()}",
            prediction_issue=issue,
            stop_kind=kind,
            reason_codes=tuple(sorted(reasons or (kind.value,))),
            outcome_access=(
                OutcomeAccess.EVALUATION_SEALED
                if kind is MetatheoryAdjudicationStopKind.REVEAL_AUTHORITY_REQUIRED
                else OutcomeAccess.OUTCOME_BLIND
            ),
            scientific_result_constructed=False,
        )

    @staticmethod
    def _custody_stop(issue: ObjectIdentity) -> MetatheoryAdjudicationStop:
        return MetatheoryAdjudicationStop(
            stop_id=f"stop.{issue.object_id}.sealed-outcome-custody-mismatch",
            prediction_issue=issue,
            stop_kind=MetatheoryAdjudicationStopKind.INPUT_IDENTITY_MISMATCH,
            reason_codes=("SEALED_OUTCOME_CUSTODY_MISMATCH",),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            scientific_result_constructed=False,
        )

    def _require_method(
        self,
        key: str,
        version: str,
        implementation_sha256: str,
        config_schema: str,
    ) -> None:
        try:
            manifest = self.capability_registry.resolve(key, version)
        except KeyError as error:
            raise MetatheoryPredictionError("METATHEORY_SCORER_UNREGISTERED") from error
        if (
            manifest.implementation_sha256 != implementation_sha256
            or manifest.config_schema != config_schema
        ):
            raise MetatheoryPredictionError("METATHEORY_SCORER_BINDING_DRIFT")


__all__ = [
    "MetatheoryAdjudicationService",
    "MetatheoryPredictionError",
    "MetatheoryPredictionIssueService",
    "MetatheoryTargetAcquisitionService",
]
