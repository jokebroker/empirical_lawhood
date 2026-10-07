"Authenticated finite-row inputs; only the shared service finalizes local law."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.contracts import (
    CandidateFamilyLedger,
    LawCandidateEvidence,
)
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.config import FrontierNativeConfig
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    NamedDecimal,
)
from empirical_lawhood.kernel.status import OperationalStatus, ScientificStatus
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.adapters.methods.finite_law_assembly import assemble_finite_law
from .config import COORDINATES, FrontierDesign
from .law_family import law_family
from .law_payload import FrontierLawDecoder, FrontierLawPayload
from .law_profile import FrontierQualificationProfile
from .qualification import QUALIFICATION_ROOTS, FrontierQualification, FrontierQualificationRow
from .science import CUTOFF, METHOD, frontier_system


@dataclass(frozen=True, slots=True)
class FrontierLaw(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-law'
    payload: FrontierLawPayload
    family: CandidateFamilyLedger
    candidate: LawCandidateEvidence
    qualification: LawQualificationResult
    operands: FrontierQualificationRow

    def __post_init__(self) -> None:
        if (
            self.payload.bound != self.operands.bound
            or self.payload.qualification
            != ObjectIdentity.from_record(self.operands.record_id, self.operands)
            or self.family.dataset_or_projection != self.payload.qualification
            or self.qualification.dataset_or_projection != self.payload.qualification
            or self.candidate.payload_publication.content_sha256 != self.payload.fingerprint()
            or self.candidate.axis_map != self.family.axis_map
        ):
            raise ValueError("frontier law lost its exact sole-owner qualification chain")


@dataclass(frozen=True, slots=True)
class FrontierLaws(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-laws'
    source: ObjectIdentity
    rows: tuple[FrontierLaw, ...]

    def __post_init__(self) -> None:
        if (
            self.source.object_schema != FrontierQualification.SCHEMA
            or tuple(r.payload.bound.coordinate for r in self.rows) != COORDINATES
        ):
            raise ValueError("law census changed the fixed 72-coordinate denominator")

    @property
    def record_id(self) -> str:
        return "reactor-finite-control-frontier.laws"

    @property
    def qualified(self) -> tuple[str, ...]:
        return tuple(
            r.payload.bound.coordinate.coordinate_id
            for r in self.rows
            if r.qualification.scientific_status is ScientificStatus.SUPPORTED
        )


def qualify_laws(
    *,
    native: FrontierNativeConfig,
    qualification: FrontierQualification,
    manifests: tuple[ArtifactManifest, ...],
    receipts: tuple[CanonicalTaskReceipt, ...],
    capability: CapabilityManifest,
    publisher: CandidatePayloadPublisher,
    reader: CandidatePayloadReader,
    design_artifact: ArtifactIdentity,
) -> FrontierLaws:
    if (
        len(manifests) != 96
        or len(receipts) != 96
        or tuple(r.task_id for r in receipts)
        != tuple(f"frontier.assay.{r}" for r in QUALIFICATION_ROOTS)
        or len({r.run_id for r in receipts}) != 1
        or len({r.implementation_commit for r in receipts}) != 1
        or design_artifact.sha256 != FrontierDesign().fingerprint()
        or tuple(m.logical.content_sha256 for m in manifests)
        != tuple(a.object_fingerprint for a in qualification.assays)
    ):
        raise ValueError("Local law needs the complete immutable 96-root native receipt roster")
    for manifest, receipt in zip(manifests, receipts, strict=True):
        if (
            receipt.operational_status is not OperationalStatus.SUCCEEDED
            or receipt.output_materializations != (manifest.materialization,)
            or receipt.output_logical_artifacts != (manifest.logical,)
            or manifest.publication is None
            or manifest.logical.outcome_access
            not in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL)
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("frontier local law received substituted native evidence")
    system = frontier_system()
    decoder = FrontierLawDecoder()
    results = []
    for row in qualification.rows:
        payload = FrontierLawPayload(
            row.bound,
            native.prepared_domain,
            qualification.development,
            ObjectIdentity.from_record(row.record_id, row),
        )
        stem = payload.stem
        owner = QualificationProofOwner(
            f"{stem}.proof-owner",
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
            system=system,
            stem=stem,
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
                    NamedDecimal("joint-event", D(int(root[2])), "1"),
                    NamedDecimal("native-unsafe", D(int(root[3])), "1"),
                )
                for root in row.roots
            ),
            evidence_description="Assigned native pulse chart; this row preserves both numerical views and its complete root denominator.",
            profile=lambda candidate: FrontierQualificationProfile(
                owner, row, payload, candidate, design_artifact
            ),
            publisher=publisher,
            reader=reader,
        )
        results.append(FrontierLaw(payload, family, candidate, result, row))
    return FrontierLaws(
        ObjectIdentity.from_record(qualification.record_id, qualification), tuple(results)
    )
