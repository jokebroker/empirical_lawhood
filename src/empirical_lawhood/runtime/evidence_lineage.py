"""Pure identity-derived evidence-dependence assessment."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec, EvidenceLineageBundle, EvidenceSharedComponentType, EvidenceSharedComponent, EvidenceSubstrateClass
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass


_RANK = {value: index for index, value in enumerate(EvidenceDependenceClass)}


@dataclass(frozen=True, slots=True)
class EvidenceDependenceService:
    def assess(self, spec: EvidenceDependenceSpec) -> EvidenceDependenceAssessment:
        shared = self._shared_components(spec.predecessor, spec.target)
        achieved, reasons = self._derive(spec.predecessor, spec.target)
        if achieved is None:
            refused = tuple(sorted(EvidenceDependenceClass, key=lambda value: value.value))
        else:
            refused = tuple(
                sorted(
                    (
                        value
                        for value in EvidenceDependenceClass
                        if _RANK[value] > _RANK[achieved]
                    ),
                    key=lambda value: value.value,
                )
            )
            if _RANK[achieved] < _RANK[spec.requested_class]:
                reasons.append("REQUESTED_DEPENDENCE_CLASS_NOT_ACHIEVED")
        evidence_by_id = {value.identity.object_id: value.identity for value in shared}
        return EvidenceDependenceAssessment(
            assessment_id=f"assessment.{spec.spec_id}",
            dependence_spec=ObjectIdentity.from_record(spec.spec_id, spec),
            predecessor_lineage=ObjectIdentity.from_record(
                spec.predecessor.bundle_id,
                spec.predecessor,
            ),
            target_lineage=ObjectIdentity.from_record(spec.target.bundle_id, spec.target),
            requested_class=spec.requested_class,
            achieved_class=achieved,
            shared_components=shared,
            refused_stronger_classes=refused,
            evidence_links=tuple(evidence_by_id[key] for key in sorted(evidence_by_id)),
            reason_codes=tuple(sorted(set(reasons))),
        )

    @staticmethod
    def _derive(
        predecessor: EvidenceLineageBundle,
        target: EvidenceLineageBundle,
    ) -> tuple[EvidenceDependenceClass | None, list[str]]:
        if (
            predecessor.unavailable_component_reason_codes
            or target.unavailable_component_reason_codes
        ):
            return None, [
                "DEPENDENCE_LINEAGE_INCOMPLETE",
                *predecessor.unavailable_component_reason_codes,
                *target.unavailable_component_reason_codes,
            ]
        assert predecessor.implementation is not None
        assert predecessor.generator is not None
        assert predecessor.preparation is not None
        assert predecessor.observation is not None
        assert target.implementation is not None
        assert target.generator is not None
        assert target.preparation is not None
        assert target.observation is not None

        same_implementation = (
            predecessor.implementation.implementation_closure
            == target.implementation.implementation_closure
            or predecessor.implementation.executable_provider
            == target.implementation.executable_provider
        )
        independent_implementation = (
            predecessor.implementation.implementation_closure
            != target.implementation.implementation_closure
            and predecessor.implementation.executable_provider
            != target.implementation.executable_provider
        )
        same_generator = (
            predecessor.generator.generator_family == target.generator.generator_family
            or predecessor.generator.algorithm == target.generator.algorithm
            or predecessor.generator.implementation == target.generator.implementation
        )
        independent_generator = not same_generator
        same_substrate = (
            predecessor.preparation.substrate_or_source_family
            == target.preparation.substrate_or_source_family
        )
        independent_substrate = not same_substrate
        shared_ontology = bool(
            set(predecessor.generator.shared_ontology_identities)
            & set(target.generator.shared_ontology_identities)
        )

        if (
            target.preparation.substrate_class is EvidenceSubstrateClass.PHYSICAL
            and independent_substrate
            and independent_generator
            and independent_implementation
        ):
            return EvidenceDependenceClass.PHYSICAL_GROUNDING, []
        if independent_substrate and independent_generator and independent_implementation:
            return EvidenceDependenceClass.INDEPENDENT_SUBSTRATE, []
        if same_substrate and independent_generator and independent_implementation:
            return EvidenceDependenceClass.INDEPENDENT_IMPLEMENTATION_SHARED_SUBSTRATE, []
        if independent_generator and shared_ontology:
            return EvidenceDependenceClass.INDEPENDENT_GENERATOR_SHARED_ONTOLOGY, []
        if same_implementation or same_generator:
            return EvidenceDependenceClass.SAME_IMPLEMENTATION_RESAMPLE, []
        return None, ["DEPENDENCE_RELATION_NOT_PROVABLE_FROM_LINEAGE"]

    @staticmethod
    def _shared_components(
        predecessor: EvidenceLineageBundle,
        target: EvidenceLineageBundle,
    ) -> tuple[EvidenceSharedComponent, ...]:
        left = EvidenceDependenceService._typed_identities(predecessor)
        right = EvidenceDependenceService._typed_identities(target)
        shared = []
        for component_type in EvidenceSharedComponentType:
            right_values = set(right[component_type])
            for identity in left[component_type]:
                if identity in right_values:
                    shared.append(
                        EvidenceSharedComponent(
                            component_id=(
                                f"shared.{component_type.value.lower()}.{identity.object_id}"
                            ),
                            component_type=component_type,
                            identity=identity,
                        )
                    )
        return tuple(sorted(shared, key=lambda value: value.component_id))

    @staticmethod
    def _typed_identities(
        bundle: EvidenceLineageBundle,
    ) -> dict[EvidenceSharedComponentType, tuple[ObjectIdentity, ...]]:
        result: dict[EvidenceSharedComponentType, tuple[ObjectIdentity, ...]] = {
            value: () for value in EvidenceSharedComponentType
        }
        if bundle.implementation is not None:
            result[EvidenceSharedComponentType.IMPLEMENTATION] = (
                bundle.implementation.implementation_closure,
            )
            result[EvidenceSharedComponentType.EXECUTABLE_CAPABILITY] = (
                bundle.implementation.executable_capability,
            )
            result[EvidenceSharedComponentType.EXECUTABLE_PROVIDER] = (
                bundle.implementation.executable_provider,
            )
            result[EvidenceSharedComponentType.LIBRARY_OR_RUNTIME] = (
                bundle.implementation.library_or_runtime_identities
            )
        if bundle.generator is not None:
            result[EvidenceSharedComponentType.GENERATOR] = (bundle.generator.generator_family,)
            result[EvidenceSharedComponentType.ALGORITHM] = (bundle.generator.algorithm,)
            result[EvidenceSharedComponentType.GENERATOR_IMPLEMENTATION] = (
                bundle.generator.implementation,
            )
            result[EvidenceSharedComponentType.CONFIGURATION] = (bundle.generator.configuration,)
            result[EvidenceSharedComponentType.ONTOLOGY] = (
                bundle.generator.shared_ontology_identities
            )
        if bundle.preparation is not None:
            result[EvidenceSharedComponentType.SUBSTRATE_OR_SOURCE] = (
                bundle.preparation.substrate_or_source_family,
            )
            result[EvidenceSharedComponentType.PREPARATION] = (
                bundle.preparation.preparation_mechanism,
            )
            result[EvidenceSharedComponentType.SAMPLER] = (bundle.preparation.sampler,)
            result[EvidenceSharedComponentType.SOURCE_MATERIALIZATION] = (
                bundle.preparation.source_materialization_parents
            )
        if bundle.observation is not None:
            result[EvidenceSharedComponentType.RECEIVER] = (
                bundle.observation.receiver_construction,
            )
            result[EvidenceSharedComponentType.OBSERVATION_OPERATOR] = (
                bundle.observation.observation_operator,
            )
            result[EvidenceSharedComponentType.NUMERICAL_VIEW] = (
                bundle.observation.numerical_view,
            )
            result[EvidenceSharedComponentType.TRANSFORMATION] = (
                bundle.observation.transformation,
            )
            result[EvidenceSharedComponentType.ANALYSIS_METHOD] = (
                bundle.observation.analysis_method,
            )
        return result


__all__ = ["EvidenceDependenceService"]
