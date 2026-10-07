"""Registered outcome-blind construction of coordinate challenge operands."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeEvidence, CoordinateChallengeNomination, CoordinateChallengeSpec, MetatheoryCoordinateConstructionProduct
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceQualificationResult
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


class MetatheoryCoordinateConstructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MetatheoryCoordinateConstructionService:
    """Validate a registered construction method and preserve its typed operands."""

    capability_registry: CapabilityRegistry

    def construct(
        self,
        *,
        spec: CoordinateChallengeSpec,
        source_qualification: ScientificSourceQualificationResult | None,
        nominations: tuple[CoordinateChallengeNomination, ...],
        evidence: tuple[CoordinateChallengeEvidence, ...],
    ) -> MetatheoryCoordinateConstructionProduct:
        method = spec.construction_method
        try:
            manifest = self.capability_registry.resolve(
                method.capability_key,
                method.capability_version,
            )
        except KeyError as error:
            raise MetatheoryCoordinateConstructionError(
                "COORDINATE_CONSTRUCTION_METHOD_UNREGISTERED"
            ) from error
        if (
            manifest.implementation_sha256 != method.implementation_sha256
            or manifest.config_schema != method.config.object_schema
        ):
            raise MetatheoryCoordinateConstructionError(
                "COORDINATE_CONSTRUCTION_METHOD_BINDING_DRIFT"
            )
        spec_identity = ObjectIdentity.from_record(spec.spec_id, spec)
        source_identity = (
            None
            if source_qualification is None
            else ObjectIdentity.from_record(
                source_qualification.result_id,
                source_qualification,
            )
        )
        if spec.source_qualification != source_identity:
            raise MetatheoryCoordinateConstructionError(
                "COORDINATE_CONSTRUCTION_SOURCE_QUALIFICATION_MISMATCH"
            )
        expected = {
            (ObjectIdentity.from_record(candidate.candidate_id, candidate), physical_unit_id)
            for candidate in spec.candidates
            for physical_unit_id in spec.physical_unit_ids
        }
        nomination_keys = {(value.candidate, value.physical_unit_id) for value in nominations}
        evidence_keys = {(value.candidate, value.physical_unit_id) for value in evidence}
        if (
            len(nomination_keys) != len(nominations)
            or len(evidence_keys) != len(evidence)
            or nomination_keys != expected
            or evidence_keys != expected
        ):
            raise MetatheoryCoordinateConstructionError(
                "COORDINATE_CONSTRUCTION_OPERAND_ROSTER_MISMATCH"
            )
        nominations_by_key = {
            (value.candidate, value.physical_unit_id): value for value in nominations
        }
        for value in evidence:
            key = (value.candidate, value.physical_unit_id)
            nomination = nominations_by_key[key]
            if (
                nomination.challenge_spec != spec_identity
                or value.challenge_spec != spec_identity
                or nomination.construction_method != method
                or value.nomination
                != ObjectIdentity.from_record(nomination.nomination_id, nomination)
            ):
                raise MetatheoryCoordinateConstructionError(
                    "COORDINATE_CONSTRUCTION_OPERAND_IDENTITY_MISMATCH"
                )
        return MetatheoryCoordinateConstructionProduct(
            product_id=f"construction.{spec.spec_id}",
            challenge_spec=spec_identity,
            source_qualification=source_identity,
            nominations=tuple(sorted(nominations, key=lambda value: value.nomination_id)),
            evidence=tuple(sorted(evidence, key=lambda value: value.evidence_id)),
            construction_method=method,
            target_outcomes_read=False,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            grants_authority=False,
        )


__all__ = [
    "MetatheoryCoordinateConstructionError",
    "MetatheoryCoordinateConstructionService",
]
