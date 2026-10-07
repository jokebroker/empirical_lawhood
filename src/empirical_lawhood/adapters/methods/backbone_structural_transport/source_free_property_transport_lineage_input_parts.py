from __future__ import annotations

import hashlib


from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec, EvidenceGeneratorLineage, EvidenceImplementationLineage, EvidenceLineageBundle, EvidenceObservationLineage, EvidencePreparationLineage, EvidenceSubstrateClass
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass
from empirical_lawhood.runtime.evidence_lineage import EvidenceDependenceService


def _identity(role: str, tag: str) -> ObjectIdentity:
    object_id = f"{role}.{tag}"
    schema = f'empirical-lawhood/methods/structural-transport/synthetic-input/{role}'
    digest = hashlib.sha256(f"{object_id}:{schema}".encode()).hexdigest()
    return ObjectIdentity(object_id, schema, "1.0.0", digest)


def _bundle(
    role: str,
    *,
    implementation: str,
    provider: str,
    generator: str,
    algorithm: str,
    generator_implementation: str,
    configuration: str,
    substrate: str,
    substrate_class: EvidenceSubstrateClass = EvidenceSubstrateClass.SIMULATOR,
    ontology: str = "common",
) -> EvidenceLineageBundle:
    return EvidenceLineageBundle(
        bundle_id=f"lineage-bundle.{role}",
        implementation=EvidenceImplementationLineage(
            lineage_id=f"implementation-lineage.{role}",
            implementation_closure=_identity("closure", implementation),
            executable_capability=_identity("capability", provider),
            executable_provider=_identity("provider", provider),
            library_or_runtime_identities=(_identity("runtime", "cpython-3-11"),),
        ),
        generator=EvidenceGeneratorLineage(
            lineage_id=f"generator-lineage.{role}",
            generator_family=_identity("generator", generator),
            algorithm=_identity("algorithm", algorithm),
            implementation=_identity("generator-implementation", generator_implementation),
            configuration=_identity("configuration", configuration),
            shared_ontology_identities=(_identity("ontology", ontology),),
        ),
        preparation=EvidencePreparationLineage(
            lineage_id=f"preparation-lineage.{role}",
            substrate_class=substrate_class,
            substrate_or_source_family=_identity("substrate", substrate),
            preparation_mechanism=_identity("preparation", substrate),
            sampler=_identity("sampler", substrate),
            roster=_identity("roster", role),
            source_materialization_parents=(_identity("materialization", role),),
        ),
        observation=EvidenceObservationLineage(
            lineage_id=f"observation-lineage.{role}",
            receiver_construction=_identity("receiver", "common"),
            observation_operator=_identity("observation", "common"),
            numerical_view=_identity("numerical-view", role),
            transformation=_identity("transformation", role),
            analysis_method=_identity("analysis", role),
        ),
        unavailable_component_reason_codes=(),
    )


def _base_pair() -> tuple[EvidenceLineageBundle, EvidenceLineageBundle]:
    predecessor = _bundle(
        "predecessor",
        implementation="shared",
        provider="shared",
        generator="shared",
        algorithm="shared",
        generator_implementation="shared",
        configuration="seed-1",
        substrate="shared",
    )
    target = _bundle(
        "target",
        implementation="shared",
        provider="shared",
        generator="shared",
        algorithm="shared",
        generator_implementation="shared",
        configuration="seed-2",
        substrate="shared",
    )
    return predecessor, target


def _assess(
    predecessor: EvidenceLineageBundle,
    target: EvidenceLineageBundle,
    requested: EvidenceDependenceClass = EvidenceDependenceClass.PHYSICAL_GROUNDING,
) -> EvidenceDependenceAssessment:
    return EvidenceDependenceService().assess(
        EvidenceDependenceSpec(
            spec_id='dependence.synthetic',
            requested_class=requested,
            predecessor=predecessor,
            target=target,
            stronger_class_refusal_required=True,
        )
    )
