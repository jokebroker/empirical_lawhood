"""Deterministic complete-unit decision comparison finalization."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceApplicability, DecisionAssuranceCell, DecisionAssuranceResult, DecisionAssuranceSpec, DecisionComparisonEvidence, DecisionDisposition, DecisionErrorInterval, DecisionErrorKind
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryMethodSelection
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class DecisionAssuranceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecisionAssuranceService:
    capability_registry: CapabilityRegistry

    def finalize(
        self,
        *,
        spec: DecisionAssuranceSpec,
        evidence: tuple[DecisionComparisonEvidence, ...] = (),
    ) -> DecisionAssuranceResult:
        if spec.applicability is DecisionAssuranceApplicability.NOT_APPLICABLE:
            if evidence:
                raise DecisionAssuranceError("N/A_DECISION_ASSURANCE_RECEIVED_EVIDENCE")
            return self._applicability_result(spec, proof=None)
        assert spec.candidate_evaluator is not None
        self._require_method(spec.candidate_evaluator)
        if spec.applicability is DecisionAssuranceApplicability.DIRECT_NATIVE_COMPLETE_UNIT:
            if evidence:
                raise DecisionAssuranceError("DIRECT_NATIVE_DECISION_RECEIVED_COMPARISON")
            assert spec.direct_native_proof is not None
            return self._applicability_result(spec, proof=spec.direct_native_proof)

        assert spec.reference_evaluator is not None
        assert spec.uncertainty_method is not None
        self._require_method(spec.reference_evaluator)
        self._require_method(spec.uncertainty_method)
        return self._comparison_result(spec, evidence)

    def _applicability_result(
        self,
        spec: DecisionAssuranceSpec,
        *,
        proof: ObjectIdentity | None,
    ) -> DecisionAssuranceResult:
        direct = spec.applicability is DecisionAssuranceApplicability.DIRECT_NATIVE_COMPLETE_UNIT
        links = () if proof is None else (proof,)
        return DecisionAssuranceResult(
            result_id=f"result.{spec.spec_id}",
            assurance_spec=ObjectIdentity.from_record(spec.spec_id, spec),
            applicability=spec.applicability,
            applicability_proof=proof,
            comparison_performed=False,
            evidence=(),
            cells=(),
            complete_physical_unit_count=len(spec.physical_unit_ids),
            false_admission_count=None,
            false_safe_hold_count=None,
            false_hold_count=None,
            false_nonattempt_count=None,
            action_substitution_count=None,
            decision_mismatch_count=None,
            error_intervals=(),
            decisive_veto_cell_ids=(),
            disposition=(
                MetatheoryAggregateDisposition.SUPPORTED
                if direct
                else MetatheoryAggregateDisposition.PREREQUISITE_NONATTEMPT
            ),
            achieved_structural_evidence_ceiling=spec.maximum_structural_evidence_ceiling,
            evidence_links=links,
        )

    def _comparison_result(
        self,
        spec: DecisionAssuranceSpec,
        evidence: tuple[DecisionComparisonEvidence, ...],
    ) -> DecisionAssuranceResult:
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        targets = {value.target_id: value for value in spec.targets}
        observed = {value.target.object_id: value for value in evidence}
        if len(observed) != len(evidence) or set(observed) != set(targets):
            raise DecisionAssuranceError("DECISION_TARGET_ROSTER_MISMATCH")
        rules = {value.error_kind: value for value in spec.error_rules}
        cells = []
        links: dict[str, ObjectIdentity] = {}
        units_by_error: dict[DecisionErrorKind, set[str]] = {
            kind: set() for kind in DecisionErrorKind
        }
        for target_id in sorted(targets):
            target = targets[target_id]
            value = observed[target_id]
            if (
                value.assurance_spec != spec_identity
                or value.target != ObjectIdentity.from_record(target.target_id, target)
                or value.physical_unit_id != target.physical_unit_id
                or value.action_fibre_id != target.action_fibre_id
                or value.candidate_evaluator != spec.candidate_evaluator
                or value.reference_evaluator != spec.reference_evaluator
                or value.uncertainty_method != spec.uncertainty_method
            ):
                raise DecisionAssuranceError("DECISION_EVIDENCE_IDENTITY_MISMATCH")
            errors, unevaluable_reason = self._derive_errors(value)
            for kind in errors:
                units_by_error[kind].add(target.physical_unit_id)
            decisive = any(rules[kind].decisive_veto for kind in errors)
            reasons: tuple[str, ...]
            if unevaluable_reason is not None:
                disposition = MetatheoryCellDisposition.UNEVALUABLE
                reasons = (unevaluable_reason,)
            elif errors:
                disposition = MetatheoryCellDisposition.OPPOSED
                reasons = tuple(f"DECISION_{kind.value}" for kind in errors)
            else:
                disposition = MetatheoryCellDisposition.SUPPORTED
                reasons = ()
            cells.append(
                DecisionAssuranceCell(
                    cell_id=f"cell.{target.target_id}",
                    assurance_spec=spec_identity,
                    target=ObjectIdentity.from_record(target.target_id, target),
                    evidence=ObjectIdentity.from_record(value.evidence_id, value),
                    physical_unit_id=target.physical_unit_id,
                    action_fibre_id=target.action_fibre_id,
                    candidate_decision=value.candidate_decision,
                    reference_decision=value.reference_decision,
                    error_kinds=errors,
                    decisive_veto=decisive,
                    disposition=disposition,
                    reason_codes=reasons,
                )
            )
            for link in (value.publication, value.recovery, *value.evidence_links):
                links[link.object_id] = link

        opposed = [
            cell for cell in cells if cell.disposition is MetatheoryCellDisposition.OPPOSED
        ]
        unevaluable = [
            cell for cell in cells if cell.disposition is MetatheoryCellDisposition.UNEVALUABLE
        ]
        decisive_cells = tuple(sorted(cell.cell_id for cell in cells if cell.decisive_veto))
        rate_failures = {
            kind
            for kind, units in units_by_error.items()
            if Decimal(len(units)) / Decimal(len(spec.physical_unit_ids))
            > rules[kind].maximum_complete_unit_rate.value
        }
        if decisive_cells or rate_failures:
            aggregate = MetatheoryAggregateDisposition.OPPOSED
        elif unevaluable:
            aggregate = (
                MetatheoryAggregateDisposition.MIXED
                if opposed
                else MetatheoryAggregateDisposition.UNEVALUABLE
            )
        elif opposed:
            aggregate = MetatheoryAggregateDisposition.MIXED
        else:
            aggregate = MetatheoryAggregateDisposition.SUPPORTED
        intervals = tuple(
            self._interval(
                kind=kind,
                count=len(units_by_error[kind]),
                denominator=len(spec.physical_unit_ids),
                interval_rule_id=rules[kind].interval_rule_id,
            )
            for kind in sorted(DecisionErrorKind, key=lambda value: value.value)
        )
        counts = {kind: len(units_by_error[kind]) for kind in DecisionErrorKind}
        return DecisionAssuranceResult(
            result_id=f"result.{spec.spec_id}",
            assurance_spec=spec_identity,
            applicability=spec.applicability,
            applicability_proof=None,
            comparison_performed=True,
            evidence=evidence,
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            complete_physical_unit_count=len(spec.physical_unit_ids),
            false_admission_count=counts[DecisionErrorKind.FALSE_ADMISSION],
            false_safe_hold_count=counts[DecisionErrorKind.FALSE_SAFE_HOLD],
            false_hold_count=counts[DecisionErrorKind.FALSE_HOLD],
            false_nonattempt_count=counts[DecisionErrorKind.FALSE_NONATTEMPT],
            action_substitution_count=counts[DecisionErrorKind.ACTION_SUBSTITUTION],
            decision_mismatch_count=counts[DecisionErrorKind.DECISION_MISMATCH],
            error_intervals=intervals,
            decisive_veto_cell_ids=decisive_cells,
            disposition=aggregate,
            achieved_structural_evidence_ceiling=spec.maximum_structural_evidence_ceiling,
            evidence_links=tuple(links[key] for key in sorted(links)),
        )

    @staticmethod
    def _derive_errors(
        evidence: DecisionComparisonEvidence,
    ) -> tuple[tuple[DecisionErrorKind, ...], str | None]:
        candidate = evidence.candidate_decision
        reference = evidence.reference_decision
        if DecisionDisposition.UNEVALUABLE in (candidate, reference):
            return (), "DECISION_REFERENCE_OR_CANDIDATE_UNEVALUABLE"
        errors: set[DecisionErrorKind] = set()
        if candidate is DecisionDisposition.ACTIVE_ACTION:
            if reference is DecisionDisposition.ACTIVE_ACTION:
                if (
                    evidence.candidate_action_occurrence != evidence.reference_action_occurrence
                    or evidence.candidate_delivery != evidence.reference_delivery
                ):
                    errors.add(DecisionErrorKind.ACTION_SUBSTITUTION)
            else:
                errors.add(DecisionErrorKind.FALSE_ADMISSION)
        elif candidate is DecisionDisposition.HOLD:
            if reference is DecisionDisposition.HOLD:
                if evidence.reference_hold_supported_and_viable is not True:
                    errors.add(DecisionErrorKind.FALSE_SAFE_HOLD)
            elif reference is DecisionDisposition.UNSAFE:
                errors.add(DecisionErrorKind.FALSE_SAFE_HOLD)
            else:
                errors.add(DecisionErrorKind.FALSE_HOLD)
        elif candidate is DecisionDisposition.NONATTEMPT:
            if reference is not DecisionDisposition.NONATTEMPT:
                errors.add(DecisionErrorKind.FALSE_NONATTEMPT)
        elif candidate is DecisionDisposition.UNSAFE:
            if reference is not DecisionDisposition.UNSAFE:
                errors.add(DecisionErrorKind.DECISION_MISMATCH)
        if errors:
            errors.add(DecisionErrorKind.DECISION_MISMATCH)
        return tuple(sorted(errors, key=lambda value: value.value)), None

    @staticmethod
    def _interval(
        *,
        kind: DecisionErrorKind,
        count: int,
        denominator: int,
        interval_rule_id: str,
    ) -> DecisionErrorInterval:
        # The core contract records an exact empirical point interval. Registered
        # uncertainty adapters may replace this only through a current method.
        rate = Decimal(count) / Decimal(denominator)
        return DecisionErrorInterval(
            error_kind=kind,
            complete_unit_count=count,
            lower_rate=NamedDecimal(f"rate.lower.{kind.value.lower()}", rate, "1"),
            upper_rate=NamedDecimal(f"rate.upper.{kind.value.lower()}", rate, "1"),
            interval_rule_id=interval_rule_id,
        )

    def _require_method(self, method: MetatheoryMethodSelection) -> None:
        try:
            manifest = self.capability_registry.resolve(
                method.capability_key,
                method.capability_version,
            )
        except KeyError as error:
            raise DecisionAssuranceError("DECISION_METHOD_UNREGISTERED") from error
        if (
            manifest.implementation_sha256 != method.implementation_sha256
            or manifest.config_schema != method.config.object_schema
        ):
            raise DecisionAssuranceError("DECISION_METHOD_BINDING_DRIFT")


__all__ = ["DecisionAssuranceError", "DecisionAssuranceService"]
