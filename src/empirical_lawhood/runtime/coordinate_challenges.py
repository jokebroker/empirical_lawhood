"""Exact finalization for prospective coordinate challenge evidence."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeCellResult, CoordinateChallengeEvidence, CoordinateChallengeNomination, CoordinateChallengeResult, CoordinateChallengeSpec, CoordinateSamplingMode
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryMethodSelection
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class CoordinateChallengeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CoordinateChallengeService:
    capability_registry: CapabilityRegistry

    def finalize(
        self,
        *,
        spec: CoordinateChallengeSpec,
        nominations: tuple[CoordinateChallengeNomination, ...],
        evidence: tuple[CoordinateChallengeEvidence, ...],
    ) -> CoordinateChallengeResult:
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        expected_keys = {
            (ObjectIdentity.from_record(candidate.candidate_id, candidate), physical_unit_id)
            for candidate in spec.candidates
            for physical_unit_id in spec.physical_unit_ids
        }
        nomination_by_key = {
            (value.candidate, value.physical_unit_id): value for value in nominations
        }
        if len(nomination_by_key) != len(nominations):
            raise CoordinateChallengeError("COORDINATE_NOMINATION_ROSTER_DUPLICATE")
        if set(nomination_by_key) != expected_keys:
            raise CoordinateChallengeError("COORDINATE_NOMINATION_ROSTER_MISMATCH")
        evidence_by_key = {(value.candidate, value.physical_unit_id): value for value in evidence}
        if len(evidence_by_key) != len(evidence):
            raise CoordinateChallengeError("COORDINATE_EVIDENCE_ROSTER_DUPLICATE")
        if set(evidence_by_key) != expected_keys:
            raise CoordinateChallengeError("COORDINATE_EVIDENCE_ROSTER_MISMATCH")
        self._require_method(spec.construction_method)
        self._require_method(spec.adjudication_method)
        cells = []
        links: dict[str, ObjectIdentity] = {}
        for key in sorted(
            expected_keys,
            key=lambda value: (
                value[0].object_id,
                value[0].object_fingerprint,
                value[1],
            ),
        ):
            nomination = nomination_by_key[key]
            observed = evidence_by_key[key]
            reasons: tuple[str, ...]
            if (
                nomination.challenge_spec != spec_identity
                or observed.challenge_spec != spec_identity
                or observed.nomination
                != ObjectIdentity.from_record(nomination.nomination_id, nomination)
                or nomination.candidate != observed.candidate
                or nomination.physical_unit_id != observed.physical_unit_id
                or nomination.construction_method != spec.construction_method
                or observed.adjudication_method != spec.adjudication_method
                or observed.sampling_mode is not spec.sampling_mode
            ):
                raise CoordinateChallengeError("COORDINATE_EVIDENCE_IDENTITY_MISMATCH")
            for link in (*observed.evidence_links, observed.publication, observed.recovery):
                previous = links.get(link.object_id)
                if previous is not None and previous != link:
                    raise CoordinateChallengeError("COORDINATE_EVIDENCE_LINK_CONFLICT")
                links[link.object_id] = link
            if not nomination.targetable:
                if observed.informative_collision_entered:
                    raise CoordinateChallengeError("UNTARGETABLE_UNIT_ENTERED_COLLISION")
                present = dynamical = decision = MetatheoryCellDisposition.UNEVALUABLE
                disposition = MetatheoryCellDisposition.UNEVALUABLE
                reasons = ("COORDINATE_UNTARGETABLE",)
                adverse = consistent = 0
            elif not observed.informative_collision_entered:
                present = dynamical = decision = MetatheoryCellDisposition.UNEVALUABLE
                disposition = MetatheoryCellDisposition.UNEVALUABLE
                reasons = ("NO_INFORMATIVE_COLLISION_ENTERED",)
                adverse = consistent = 0
            else:
                present = MetatheoryCellDisposition.SUPPORTED
                dynamical = self._comparison_disposition(observed.future_response_equal)
                decision = self._comparison_disposition(observed.decision_equal)
                values = (present, dynamical, decision)
                adverse = int(MetatheoryCellDisposition.OPPOSED in values)
                consistent = int(
                    all(value is MetatheoryCellDisposition.SUPPORTED for value in values)
                )
                if observed.endpoint_saturated:
                    disposition = MetatheoryCellDisposition.UNEVALUABLE
                    reasons = ("ENDPOINT_SATURATED_BY_UNCHALLENGED_LABEL",)
                elif adverse:
                    disposition = MetatheoryCellDisposition.OPPOSED
                    reasons = ("ADVERSE_INFORMATIVE_COLLISION",)
                elif consistent:
                    disposition = MetatheoryCellDisposition.SUPPORTED
                    reasons = ()
                else:
                    disposition = MetatheoryCellDisposition.UNEVALUABLE
                    reasons = ("SUFFICIENCY_OPERAND_UNRESOLVED",)
            cells.append(
                CoordinateChallengeCellResult(
                    cell_id=f"cell.{nomination.nomination_id}",
                    challenge_spec=spec_identity,
                    candidate=nomination.candidate,
                    physical_unit_id=nomination.physical_unit_id,
                    evidence=ObjectIdentity.from_record(observed.evidence_id, observed),
                    informative_collision_entered=observed.informative_collision_entered,
                    present_sufficiency=present,
                    dynamical_sufficiency=dynamical,
                    decision_sufficiency=decision,
                    adverse_collision_count=adverse,
                    consistent_collision_count=consistent,
                    targetable=nomination.targetable,
                    prevalence_applicable=(
                        spec.sampling_mode is CoordinateSamplingMode.UNTOUCHED
                        and nomination.targetable
                    ),
                    endpoint_saturated=observed.endpoint_saturated,
                    exact_structural_rank=observed.exact_structural_rank,
                    numerical_rank_lower=observed.numerical_rank_lower,
                    numerical_rank_upper=observed.numerical_rank_upper,
                    usable_conditioned_rank=observed.usable_conditioned_rank,
                    disposition=disposition,
                    decisive_witness_ids=tuple(
                        sorted(value.object_id for value in observed.evidence_links)
                    ),
                    reason_codes=reasons,
                )
            )
        dispositions = {value.disposition for value in cells}
        if dispositions == {MetatheoryCellDisposition.SUPPORTED}:
            aggregate = MetatheoryAggregateDisposition.SUPPORTED
        elif MetatheoryCellDisposition.OPPOSED in dispositions:
            aggregate = (
                MetatheoryAggregateDisposition.OPPOSED
                if dispositions == {MetatheoryCellDisposition.OPPOSED}
                else MetatheoryAggregateDisposition.MIXED
            )
        else:
            aggregate = MetatheoryAggregateDisposition.UNEVALUABLE
        return CoordinateChallengeResult(
            result_id=f"result.{spec.spec_id}",
            challenge_spec=spec_identity,
            nominations=tuple(sorted(nominations, key=lambda value: value.nomination_id)),
            evidence=tuple(sorted(evidence, key=lambda value: value.evidence_id)),
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            targeted_cell_count=(
                len(cells) if spec.sampling_mode is CoordinateSamplingMode.TARGETED else 0
            ),
            untouched_cell_count=(
                len(cells) if spec.sampling_mode is CoordinateSamplingMode.UNTOUCHED else 0
            ),
            untargetable_unit_count=sum(
                not value.targetable for value in nomination_by_key.values()
            ),
            disposition=aggregate,
            maximum_structural_evidence_ceiling=spec.maximum_structural_evidence_ceiling,
            evidence_links=tuple(links[key] for key in sorted(links)),
        )

    def _require_method(self, method: MetatheoryMethodSelection) -> None:
        try:
            manifest = self.capability_registry.resolve(
                method.capability_key,
                method.capability_version,
            )
        except KeyError as error:
            raise CoordinateChallengeError("COORDINATE_METHOD_UNREGISTERED") from error
        if manifest.implementation_sha256 != method.implementation_sha256:
            raise CoordinateChallengeError("COORDINATE_METHOD_BINDING_DRIFT")

    @staticmethod
    def _comparison_disposition(value: bool | None) -> MetatheoryCellDisposition:
        if value is True:
            return MetatheoryCellDisposition.SUPPORTED
        if value is False:
            return MetatheoryCellDisposition.OPPOSED
        return MetatheoryCellDisposition.UNEVALUABLE


__all__ = ["CoordinateChallengeError", "CoordinateChallengeService"]
