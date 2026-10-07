"Staged-pulse candidate custody and results, finalised only by the shared law service."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import CandidateFamilyLedger, LawCandidateEvidence
from empirical_lawhood.adapters.methods.finite_law_assembly import (
    assemble_finite_law,
    require_native_custody,
)
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.config import ClassicalNativeConfig
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from .config import ClassicalDesign, roots
from .law_family import law_family
from .law_payload import ClassicalLawDecoder, ClassicalLawPayload
from .law_profile import ClassicalQualificationProfile
from .qualification import ClassicalQualificationRow, ClassicalQualification
from .science import CUTOFF, METHOD, staged_pulse_reactor_system


@dataclass(frozen=True, slots=True)
class ClassicalLaw(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-law'
    payload: ClassicalLawPayload
    family: CandidateFamilyLedger
    candidate: LawCandidateEvidence
    qualification: LawQualificationResult
    operands: ClassicalQualificationRow

    def __post_init__(self) -> None:
        if (
            self.payload.recipe != self.operands.recipe
            or self.payload.qualification
            != ObjectIdentity.from_record(self.operands.record_id, self.operands)
            or self.family.dataset_or_projection != self.payload.qualification
            or self.qualification.dataset_or_projection != self.payload.qualification
            or self.candidate.payload_publication.content_sha256 != self.payload.fingerprint()
            or self.candidate.axis_map != self.family.axis_map
        ):
            raise ValueError("law report substitutes its sole-owner payload or qualification chain")


@dataclass(frozen=True, slots=True)
class ClassicalLaws(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-laws'
    operands: ClassicalQualification
    rows: tuple[ClassicalLaw, ...]

    def __post_init__(self) -> None:
        if tuple(r.operands for r in self.rows) != self.operands.rows:
            raise ValueError("law census drops a failed or nonentered assigned relation")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.operands.nomination.block}.laws"

    @property
    def qualified(self) -> tuple[str, ...]:
        return tuple(
            r.payload.recipe.recipe_id
            for r in self.rows
            if r.qualification.scientific_status is ScientificStatus.SUPPORTED
        )


def qualify_laws(
    *,
    native: ClassicalNativeConfig,
    qualification: ClassicalQualification,
    manifests: tuple[ArtifactManifest, ...],
    receipts: tuple[CanonicalTaskReceipt, ...],
    capability: CapabilityManifest,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
    design_artifact: ArtifactIdentity,
) -> ClassicalLaws:
    block = qualification.nomination.block
    assigned = roots(block, "qualification")
    require_native_custody(
        manifests,
        receipts,
        tuple(f"classical.assay.{r}" for r in assigned),
        tuple(a.object_fingerprint for a in qualification.assays),
    )
    if design_artifact.sha256 != ClassicalDesign().fingerprint():
        raise ValueError("qualification changes the predeclared staged-pulse design")
    decoder = ClassicalLawDecoder()
    results = []
    for row in qualification.rows:
        payload = ClassicalLawPayload(
            block,
            row.recipe,
            native.prepared_domain,
            ObjectIdentity.from_record(
                qualification.nomination.record_id, qualification.nomination
            ),
            ObjectIdentity.from_record(row.record_id, row),
        )
        owner = QualificationProofOwner(
            f"{payload.stem}.proof-owner",
            capability.capability_key,
            capability.capability_version,
            capability.implementation_sha256,
        )
        binding, family = law_family(
            payload,
            ObjectIdentity.from_record(capability.capability_key, capability),
            ObjectIdentity.from_record(owner.owner_id, owner),
        )
        candidate, result = assemble_finite_law(
            system=staged_pulse_reactor_system(row.recipe.bound.coordinate),
            stem=payload.stem,
            method=METHOD,
            cutoff=CUTOFF,
            payload=payload,
            decoder=decoder,
            family=family,
            binding=binding,
            capability=capability,
            manifests=manifests,
            receipts=receipts,
            metrics=tuple(
                (
                    NamedDecimal("joint-event", D(int(r[2])), "1"),
                    NamedDecimal("native-unsafe", D(int(r[3])), "1"),
                )
                for r in row.roots
            ),
            evidence_description="Assigned raw native measurements, both views, frozen causal response/thermal map and exact local or induced-history chart.",
            profile=lambda candidate: ClassicalQualificationProfile(
                owner, row, payload, candidate, design_artifact
            ),
            publisher=publisher,
            reader=reader,
        )
        results.append(ClassicalLaw(payload, family, candidate, result, row))
    return ClassicalLaws(qualification, tuple(results))
