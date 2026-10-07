"""Sole method-neutral terminal response-law qualification authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import ClaimSpec, EvidenceCeiling, EvidenceRung
from empirical_lawhood.kernel.identification import LawIdentificationResult, LawQualificationResult, ResponseQualificationFacetAssessment, ResponseQualificationTrace, QualificationOperationalDisposition, StructuralConvergenceResult, TerminalLawObligationAssessment, TerminalObligationDisposition
from empirical_lawhood.kernel.laws import ResponseLaw, validate_law_against_system
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.provenance import Claim, EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, ExtensionBinding
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus, ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt

from .contracts import ComponentUncertaintyFamilyAssessment, JointUncertaintyFamilyAssessment, CandidateFamilyLedger, CandidateQualificationEligibility, CandidateSelectionDisposition, CandidateSelectionReceipt, FiniteActionCompatibilitySetExtension, ComponentUncertaintyCandidateAssessment, JointUncertaintyCandidateAssessment, LawCandidateEvidence, LawObligationTemplate, ParametricLawModel, ComponentQualificationAssessment, JointQualificationAssessment
from .receiver_conditioned_io.contracts import ControlledIOVersionSet
from .qualification_profiles import ComponentQualificationProfile, JointQualificationProfile


class CandidatePayloadReadError(RuntimeError):
    """Authoritative evaluator payload could not be read or verified."""


class CandidatePayloadReader(Protocol):
    def read_candidate_payload(self, receipt: CandidatePayloadPublicationReceipt) -> bytes: ...


class CandidatePayloadPublisher(Protocol):
    def publish_candidate_payload(
        self,
        *,
        payload: bytes,
        evaluator: ExecutableReference,
        implementation: ObjectIdentity,
        decoder_schema: str,
        decoder_version: str,
        maximum_decode_bytes: int,
    ) -> CandidatePayloadPublicationReceipt: ...


class CandidatePayloadDecoder(Protocol):
    @property
    def extension_namespace(self) -> str | None: ...

    @property
    def decoder_schema(self) -> str: ...

    @property
    def decoder_version(self) -> str: ...

    @property
    def payload_schema(self) -> str: ...

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord: ...


class CandidateQualificationProfileEvaluator(Protocol):
    "Registered proof owner for a candidate family outside the baseline parametric response-method profile."

    @property
    def profile(self) -> ComponentQualificationProfile: ...

    def evaluate_candidate(
        self,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        payload: CanonicalRecord,
    ) -> ComponentQualificationAssessment: ...


@dataclass(frozen=True, slots=True)
class QualificationProfileEvaluatorRegistry:
    """Closed static proof-owner registry; callers select an exact profile identity."""

    evaluators: tuple[CandidateQualificationProfileEvaluator, ...]

    def __post_init__(self) -> None:
        keys = tuple(value.profile.profile_id for value in self.evaluators)
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("qualification profile evaluator registry must be sorted and unique")

    def require(self, profile: ObjectIdentity) -> CandidateQualificationProfileEvaluator:
        for evaluator in self.evaluators:
            if profile == ObjectIdentity.from_record(
                evaluator.profile.profile_id,
                evaluator.profile,
            ):
                return evaluator
        raise ValueError("qualification profile evaluator is not registered")


@dataclass(frozen=True, slots=True)
class CanonicalParametricLawModelDecoder:
    extension_namespace: str | None = None
    decoder_schema: str = 'empirical-lawhood/methods/parametric-law/model-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = ParametricLawModel.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload,
            ParametricLawModel,
            maximum_bytes=maximum_bytes,
        )


@dataclass(frozen=True, slots=True)
class CanonicalFiniteActionCompatibilitySetDecoder:
    extension_namespace: str | None = "finite-action-compatibility-set"
    decoder_schema: str = 'empirical-lawhood/methods/finite-action/compatibility-set-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = FiniteActionCompatibilitySetExtension.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload,
            FiniteActionCompatibilitySetExtension,
            maximum_bytes=maximum_bytes,
        )


@dataclass(frozen=True, slots=True)
class CanonicalControlledIOVersionSetDecoder:
    extension_namespace: str | None = "controlled-io-version-set"
    decoder_schema: str = 'empirical-lawhood/methods/controlled-input-output/compatible-model-set-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = ControlledIOVersionSet.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload,
            ControlledIOVersionSet,
            maximum_bytes=maximum_bytes,
        )


@dataclass(frozen=True, slots=True)
class CandidatePayloadDecoderRegistry:
    """Closed static decoder registry; configuration never supplies callables."""

    decoders: tuple[CandidatePayloadDecoder, ...]

    def __post_init__(self) -> None:
        keys = tuple(
            (value.decoder_schema, value.decoder_version, value.payload_schema)
            for value in self.decoders
        )
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("candidate payload decoder registry must be sorted and unique")

    def require(self, receipt: CandidatePayloadPublicationReceipt) -> CandidatePayloadDecoder:
        key = (
            receipt.decoder_schema,
            receipt.decoder_version,
            receipt.artifact.payload_schema,
        )
        for decoder in self.decoders:
            if (decoder.decoder_schema, decoder.decoder_version, decoder.payload_schema) == key:
                return decoder
        raise CandidatePayloadReadError("EVALUATOR_DECODER_UNAVAILABLE")


def _verified_candidate_payload(
    *,
    reader: CandidatePayloadReader,
    decoder_registry: CandidatePayloadDecoderRegistry,
    receipt: CandidatePayloadPublicationReceipt,
) -> CanonicalRecord:
    try:
        payload = reader.read_candidate_payload(receipt)
    except Exception as error:
        raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_UNREADABLE") from error
    if len(payload) > receipt.maximum_decode_bytes:
        raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_EXCEEDS_BOUND")
    if hashlib.sha256(payload).hexdigest() != receipt.content_sha256:
        raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_DIGEST_MISMATCH")
    decoder = decoder_registry.require(receipt)
    try:
        value = decoder.decode(payload, maximum_bytes=receipt.maximum_decode_bytes)
    except Exception as error:
        raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_DECODE_FAILED") from error
    if value.fingerprint() != receipt.content_sha256:
        raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_CANONICAL_IDENTITY_MISMATCH")
    return value


def _verified_candidate_extension(
    *,
    decoder_registry: CandidatePayloadDecoderRegistry,
    candidate: LawCandidateEvidence,
) -> tuple[ExtensionBinding, ...]:
    decoder = decoder_registry.require(candidate.payload_publication)
    extension = candidate.candidate_extension
    if decoder.extension_namespace is None:
        if extension is not None:
            raise CandidatePayloadReadError("EVALUATOR_EXTENSION_UNEXPECTED")
        return ()
    if extension is None:
        raise CandidatePayloadReadError("EVALUATOR_EXTENSION_ABSENT")
    if (
        extension.namespace != decoder.extension_namespace
        or extension.schema != decoder.payload_schema
        or extension.payload_sha256 != candidate.payload_publication.content_sha256
    ):
        raise CandidatePayloadReadError("EVALUATOR_EXTENSION_DRIFT")
    return (extension,)


@dataclass(frozen=True, slots=True)
class CandidateFamilyAssembler:
    """Complete-roster, least-complex-passing family reducer."""

    def assemble(
        self,
        ledger: CandidateFamilyLedger,
        assessments: tuple[ComponentUncertaintyCandidateAssessment, ...],
    ) -> ComponentUncertaintyFamilyAssessment:
        ordered = tuple(sorted(assessments, key=lambda value: value.candidate_id))
        required_ids = {
            value.candidate_id
            for value in ledger.members
            if value.disposition.value == "ASSESS_REQUIRED"
        }
        if {value.candidate_id for value in ordered} != required_ids:
            raise ValueError("candidate family is not completely assessed")
        rank = {value.candidate_id: value.complexity_rank for value in ledger.members}
        passing = tuple(
            value
            for value in ordered
            if value.qualification_eligibility is CandidateQualificationEligibility.PASSING
        )
        selected = (
            min(passing, key=lambda value: (rank[value.candidate_id], value.candidate_id))
            if passing
            else None
        )
        selection = CandidateSelectionReceipt(
            receipt_id=f"selection.{ledger.family_id}",
            family=ObjectIdentity.from_record(ledger.family_id, ledger),
            assessed_candidate_ids=tuple(value.candidate_id for value in ordered),
            selected_candidate_id=selected.candidate_id if selected is not None else None,
            disposition=(
                CandidateSelectionDisposition.SELECTED
                if selected is not None
                else CandidateSelectionDisposition.NO_SELECTION
            ),
            candidate_generation_rule_ids=ledger.candidate_generation_rule_ids,
            selection_threshold_ids=ledger.selection_threshold_ids,
            development_input_ids=ledger.development_input_ids,
            selector_capability=ledger.selector_capability,
            selector_config=ledger.selector_config,
            selector_implementation=ledger.selector_implementation,
            multiplicity_family_id=ledger.multiplicity_family_id,
            multiplicity_rule_id=ledger.multiplicity_rule_id,
            randomness_seed_ids=ledger.randomness_seed_ids,
            tie_break_rule=ledger.tie_break_rule,
        )
        if any(isinstance(value, JointUncertaintyCandidateAssessment) for value in ordered):
            joint = tuple(value for value in ordered if isinstance(value, JointUncertaintyCandidateAssessment))
            if len(joint) != len(ordered):
                raise ValueError("candidate family cannot mix qualification uncertainty bases")
            return JointUncertaintyFamilyAssessment(
                assessment_id=f"family-assessment.{ledger.family_id}",
                ledger=ledger,
                assessments=joint,
                selection=selection,
            )
        return ComponentUncertaintyFamilyAssessment(
            assessment_id=f"family-assessment.{ledger.family_id}",
            ledger=ledger,
            assessments=ordered,
            selection=selection,
        )


@dataclass(frozen=True, slots=True)
class LawAssessmentAssembler:
    """Pre-fitted candidate assembler; cannot construct claims/results/laws."""

    payload_reader: CandidatePayloadReader
    decoder_registry: CandidatePayloadDecoderRegistry
    profile_registry: QualificationProfileEvaluatorRegistry

    def assemble(
        self,
        *,
        system: SystemSpec,
        candidate: LawCandidateEvidence,
        qualification_profile: ObjectIdentity,
        metrics: tuple[NamedDecimal, ...] = (),
    ) -> ComponentUncertaintyCandidateAssessment:
        payload = _verified_candidate_payload(
            reader=self.payload_reader,
            decoder_registry=self.decoder_registry,
            receipt=candidate.payload_publication,
        )
        _verified_candidate_extension(
            decoder_registry=self.decoder_registry,
            candidate=candidate,
        )
        if candidate.system != ObjectIdentity.from_record(system.system_id, system):
            raise ValueError("law assessment candidate is bound to another system")
        profile_evaluator = self.profile_registry.require(qualification_profile)
        profile_contract = profile_evaluator.profile
        if candidate.method_key not in profile_contract.applicable_method_keys:
            raise ValueError("qualification profile does not own the candidate method")
        if candidate.representation_kind not in (profile_contract.applicable_representation_kinds):
            raise ValueError("qualification profile does not own the candidate representation")
        if candidate.payload_publication.artifact.payload_schema not in (
            profile_contract.applicable_payload_schemas
        ):
            raise ValueError("qualification profile does not own the candidate payload schema")
        observed_extension_schemas = (
            () if candidate.candidate_extension is None else (candidate.candidate_extension.schema,)
        )
        if observed_extension_schemas != profile_contract.applicable_extension_schemas:
            raise ValueError("qualification profile extension family differs from the candidate")
        profile = profile_evaluator.evaluate_candidate(system, candidate, payload)
        if profile.profile != qualification_profile:
            raise ValueError("qualification assessment changes its registered profile identity")
        if profile.candidate_evidence != ObjectIdentity.from_record(
            candidate.evidence_id,
            candidate,
        ):
            raise ValueError("qualification profile assessed another candidate payload")
        if isinstance(profile_contract, JointQualificationProfile) != isinstance(
            profile, JointQualificationAssessment
        ):
            raise ValueError("qualification assessment changes its registered uncertainty basis")
        if isinstance(profile, JointQualificationAssessment):
            return JointUncertaintyCandidateAssessment(
                assessment_id=f"candidate-assessment.{candidate.candidate_id}",
                candidate_id=candidate.candidate_id,
                candidate_evidence=candidate,
                profile_assessment=profile,
                metrics=tuple(sorted(metrics, key=lambda value: value.value_id)),
            )
        return ComponentUncertaintyCandidateAssessment(
            assessment_id=f"candidate-assessment.{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            candidate_evidence=candidate,
            profile_assessment=profile,
            metrics=tuple(sorted(metrics, key=lambda value: value.value_id)),
        )


def _failed_reasons(assessment: ComponentUncertaintyCandidateAssessment) -> tuple[str, ...]:
    profile = assessment.profile_assessment
    reasons = {
        reason for check in profile.checks if not check.passed for reason in check.reason_codes
    }
    if isinstance(profile, JointQualificationAssessment):
        reasons.update(profile.joint_uncertainty.reason_codes)
    else:
        reasons.update(
            reason
            for component in profile.uncertainty.components
            if component.status not in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}
            for reason in component.limitation_codes
        )
    reasons.update(
        reason
        for value in profile.property_qualifications
        if value.status is not ScientificStatus.SUPPORTED
        for reason in value.reason_codes
    )
    return tuple(sorted(reasons))


def _build_claim_spec(
    *,
    system: SystemSpec,
    ledger: CandidateFamilyLedger,
    promotable: bool,
) -> ClaimSpec:
    template = ledger.claim_template
    obligation = ledger.obligation_template
    claim = ClaimSpec(
        claim_id=template.claim_id,
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=template.proposition,
        estimand=template.estimand,
        physical_independent_unit_id=obligation.independent_unit_id,
        requested_rung=EvidenceRung.LOCAL_LAW if promotable else None,
        evidence_ceiling=(
            template.requested_evidence_ceiling if promotable else EvidenceCeiling.NON_PROMOTABLE
        ),
        outcome_access=ledger.outcome_access,
        visibility_ceiling=ledger.visibility_ceiling,
        promotion_rule=template.promotion_rule,
        assumption_ids=template.mapping_assumption_ids,
        numerical_view_ids=obligation.numerical_view_ids,
    )
    system.validate_claim(claim)
    return claim


def _build_claim(
    *,
    claim_spec: ClaimSpec,
    status: ScientificStatus,
    highest_rung: EvidenceRung | None,
    evidence_links: tuple[EvidenceLink, ...],
    reasons: tuple[str, ...],
    omit_failed_observed_rung: bool,
) -> Claim:
    claim_reasons: tuple[str, ...]
    observed: EvidenceRung | None
    if status is ScientificStatus.SUPPORTED:
        summary = "All frozen gates supported the local relational response law."
        claim_reasons = ("all-frozen-gates-passed",)
        observed = EvidenceRung.LOCAL_LAW
    else:
        summary = "The fitted candidate did not satisfy the frozen local-law obligations."
        claim_reasons = reasons
        observed = None if omit_failed_observed_rung else highest_rung
    return Claim(
        claim=claim_spec,
        scientific_status=status,
        observed_rung=observed,
        result_summary=summary,
        evidence_links=evidence_links,
        reason_codes=claim_reasons,
        lifecycle_status=LifecycleStatus.TERMINAL,
    )


def _build_obligations(
    template: LawObligationTemplate,
    evidence_links: tuple[EvidenceLink, ...],
) -> ScientificObligations:
    evidence_ids = tuple(value.link_id for value in evidence_links)
    return ScientificObligations(
        obligations_id=template.obligations_id,
        support=SupportSpec(
            support_id=template.support_id,
            relation_id=evidence_links[0].target.object_id,
            independent_unit_id=template.independent_unit_id,
            physical_unit_count=template.physical_unit_count,
            nested_numerical_view_count=template.nested_numerical_view_count,
            information_cutoff_id=template.information_cutoff_id,
            chart_ids=template.chart_ids,
            denominator_cell_ids=template.denominator_cell_ids,
            action_bounds=template.action_bounds,
            status=ObligationStatus.SATISFIED,
            evidence_link_ids=evidence_ids,
        ),
        validity=ValiditySpec(
            validity_id=template.validity_id,
            validity_domain_ids=template.validity_domain_ids,
            assumption_ids=template.assumption_ids,
            exclusion_reason_codes=(),
            status=ObligationStatus.SATISFIED,
            evidence_link_ids=evidence_ids,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id=template.uncertainty_id,
            method_key=template.uncertainty_method_key,
            independent_unit_id=template.independent_unit_id,
            confidence_level=template.uncertainty_confidence_level,
            interval_quantity_ids=template.interval_quantity_ids,
            limitation_codes=template.uncertainty_limitation_codes,
            status=ObligationStatus.SATISFIED,
            evidence_link_ids=evidence_ids,
        ),
        falsifiers=tuple(
            FalsifierSpec(
                falsifier_id=value.falsifier_id,
                kind=value.kind,
                capability_key=value.capability_key,
                description=value.description,
                decisive_rule=value.decisive_rule,
                status=ObligationStatus.SATISFIED,
                evidence_link_ids=evidence_ids,
            )
            for value in template.falsifiers
        ),
        closure=ClosureSpec(
            closure_id=template.closure_id,
            recurrence_cell_ids=template.recurrence_cell_ids,
            exchange_factor_ids=template.exchange_factor_ids,
            retained_history_ids=template.retained_history_ids,
            status=ObligationStatus.SATISFIED,
            evidence_link_ids=evidence_ids,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id=template.structural_convergence_id,
            required_structure_ids=template.required_structure_ids,
            numerical_view_ids=template.numerical_view_ids,
            tolerances=template.structural_tolerances,
            status=ObligationStatus.SATISFIED,
            evidence_link_ids=evidence_ids,
        ),
        computability=ComputabilityEvidence(
            computability_id=template.computability_id,
            envelope_id=template.computability_envelope_id,
            numerical_view_ids=template.numerical_view_ids,
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=evidence_ids,
        ),
    )


def _status_for_no_selection(family: ComponentUncertaintyFamilyAssessment) -> ScientificStatus:
    facet_statuses = {facet.status for facet in _aggregate_trace(family).facets}
    if ScientificStatus.MIXED in facet_statuses:
        return ScientificStatus.MIXED
    if ScientificStatus.PARTIAL in facet_statuses:
        return ScientificStatus.PARTIAL
    if family.assessments and all(
        value.qualification_eligibility is CandidateQualificationEligibility.UNEVALUABLE
        for value in family.assessments
    ):
        return ScientificStatus.UNEVALUABLE
    return ScientificStatus.NOT_SUPPORTED


def _family_evidence(family: ComponentUncertaintyFamilyAssessment) -> tuple[EvidenceLink, ...]:
    by_id: dict[str, EvidenceLink] = {}
    for assessment in family.assessments:
        for link in assessment.candidate_evidence.evidence_links:
            prior = by_id.setdefault(link.link_id, link)
            if prior != link:
                raise ValueError("candidate family reuses an evidence ID with different content")
    return tuple(by_id[value] for value in sorted(by_id))


def _family_metrics(family: ComponentUncertaintyFamilyAssessment) -> tuple[NamedDecimal, ...]:
    by_id: dict[str, NamedDecimal] = {}
    for assessment in family.assessments:
        for metric in assessment.metrics:
            prior = by_id.setdefault(metric.value_id, metric)
            if prior != metric:
                raise ValueError("candidate family reuses a metric ID with different content")
    return tuple(by_id[value] for value in sorted(by_id))


def _aggregate_scientific_status(
    statuses: tuple[ScientificStatus, ...],
) -> ScientificStatus:
    observed = set(statuses)
    if len(observed) == 1:
        return statuses[0]
    if (
        ScientificStatus.MIXED in observed
        or {
            ScientificStatus.SUPPORTED,
            ScientificStatus.NOT_SUPPORTED,
        }
        <= observed
    ):
        return ScientificStatus.MIXED
    return ScientificStatus.PARTIAL


def _aggregate_trace(family: ComponentUncertaintyFamilyAssessment) -> ResponseQualificationTrace:
    traces = tuple(value.profile_assessment.qualification_trace for value in family.assessments)
    first = traces[0]
    for trace in traces[1:]:
        if (
            trace.physical_independent_unit_ids != first.physical_independent_unit_ids
            or trace.candidate_version_member_ids != first.candidate_version_member_ids
            or trace.denominator_member_ids != first.denominator_member_ids
            or trace.qualification_view_ids != first.qualification_view_ids
            or trace.information_cutoff_id != first.information_cutoff_id
            or trace.outcome_access is not first.outcome_access
            or {value.facet_id for value in trace.facets}
            != {value.facet_id for value in first.facets}
        ):
            raise ValueError("candidate family qualification traces change common scope/topology")
    facets_by_trace = tuple({value.facet_id: value for value in trace.facets} for trace in traces)
    facets: list[ResponseQualificationFacetAssessment] = []
    for source in first.facets:
        values = tuple(by_id[source.facet_id] for by_id in facets_by_trace)
        if any(
            (
                value.rung is not source.rung
                or value.proof_owner != source.proof_owner
                or value.prerequisite_facet_ids != source.prerequisite_facet_ids
                or value.required_for_public_rung is not source.required_for_public_rung
            )
            for value in values[1:]
        ):
            raise ValueError("candidate family qualification facets change proof topology")
        status = _aggregate_scientific_status(tuple(value.status for value in values))
        if status is ScientificStatus.SUPPORTED:
            evidence_ids = tuple(
                sorted({link for value in values for link in value.evidence_link_ids})
            )
            reasons: tuple[str, ...] = ()
            supported_object = f"object.{source.facet_id}.{family.selection.receipt_id}"
        else:
            evidence_ids = ()
            reasons = tuple(
                sorted(
                    {
                        f"{assessment.candidate_id}--{reason}"
                        for assessment, value in zip(
                            family.assessments,
                            values,
                            strict=True,
                        )
                        for reason in value.reason_codes
                    }
                    | {f"family-facet-{status.value.lower()}"}
                )
            )
            supported_object = None
        facets.append(
            ResponseQualificationFacetAssessment(
                facet_id=source.facet_id,
                rung=source.rung,
                status=status,
                proof_owner=source.proof_owner,
                prerequisite_facet_ids=source.prerequisite_facet_ids,
                evidence_link_ids=evidence_ids,
                reason_codes=reasons,
                maximum_supported_object_id=supported_object,
                required_for_public_rung=source.required_for_public_rung,
            )
        )
    ordered_facets = tuple(sorted(facets, key=lambda value: value.facet_id))
    rung_order = (
        EvidenceRung.MEASUREMENT,
        EvidenceRung.ORDER_RELATION,
        EvidenceRung.RESPONSE,
        EvidenceRung.LOCAL_LAW,
    )
    rung_rank = {value: index for index, value in enumerate(rung_order)}
    highest_rung: EvidenceRung | None = None
    for rung in rung_order:
        applicable = tuple(
            value
            for value in ordered_facets
            if value.required_for_public_rung and rung_rank[value.rung] <= rung_rank[rung]
        )
        if applicable and all(value.status is ScientificStatus.SUPPORTED for value in applicable):
            highest_rung = rung
        else:
            break
    return ResponseQualificationTrace(
        trace_id=f"trace.{family.selection.receipt_id}",
        candidate_id=family.selection.receipt_id,
        facets=ordered_facets,
        physical_independent_unit_ids=first.physical_independent_unit_ids,
        candidate_version_member_ids=first.candidate_version_member_ids,
        denominator_member_ids=first.denominator_member_ids,
        qualification_view_ids=first.qualification_view_ids,
        information_cutoff_id=first.information_cutoff_id,
        outcome_access=first.outcome_access,
        highest_supported_rung=highest_rung,
    )


def _aggregate_obligation_status(
    statuses: tuple[ObligationStatus, ...],
) -> ObligationStatus:
    passing = {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}
    if all(value is ObligationStatus.NOT_APPLICABLE for value in statuses):
        return ObligationStatus.NOT_APPLICABLE
    if all(value in passing for value in statuses):
        return ObligationStatus.SATISFIED
    if ObligationStatus.FAILED in statuses:
        return ObligationStatus.FAILED
    if ObligationStatus.UNEVALUABLE in statuses:
        return ObligationStatus.UNEVALUABLE
    return ObligationStatus.REQUIRED


def _aggregate_terminal_obligations(
    family: ComponentUncertaintyFamilyAssessment,
) -> TerminalLawObligationAssessment:
    assessments = tuple(
        value.profile_assessment.terminal_obligations for value in family.assessments
    )
    first = assessments[0]
    dispositions: list[TerminalObligationDisposition] = []
    for source in first.dispositions:
        values = tuple(value.disposition(source.kind) for value in assessments)
        if any(value.proof_owner != source.proof_owner for value in values[1:]):
            raise ValueError("candidate family changes an obligation proof owner")
        status = _aggregate_obligation_status(tuple(value.status for value in values))
        if status is ObligationStatus.SATISFIED:
            evidence_ids = tuple(
                sorted({link for value in values for link in value.evidence_link_ids})
            )
            reasons: tuple[str, ...] = ()
        elif status is ObligationStatus.NOT_APPLICABLE:
            evidence_ids = ()
            reasons = ()
        else:
            evidence_ids = ()
            reasons = tuple(
                sorted(
                    {
                        f"{assessment.candidate_id}--{reason}"
                        for assessment, value in zip(
                            family.assessments,
                            values,
                            strict=True,
                        )
                        for reason in value.reason_codes
                    }
                    | {f"family-obligation-{status.value.lower()}"}
                )
            )
        dispositions.append(
            TerminalObligationDisposition(
                disposition_id=(
                    f"obligation.{family.selection.receipt_id}."
                    f"{source.kind.value.lower().replace('_', '-')}"
                ),
                kind=source.kind,
                status=status,
                proof_owner=source.proof_owner,
                evidence_link_ids=evidence_ids,
                reason_codes=reasons,
            )
        )
    return TerminalLawObligationAssessment(
        assessment_id=f"terminal-obligations.{family.selection.receipt_id}",
        dispositions=tuple(sorted(dispositions, key=lambda value: value.disposition_id)),
    )


def _representative(family: ComponentUncertaintyFamilyAssessment) -> ComponentUncertaintyCandidateAssessment:
    if not family.assessments:
        raise ValueError("a terminal family requires at least one assessed candidate")
    rank = {value.candidate_id: value.complexity_rank for value in family.ledger.members}
    rung_rank = {
        None: -1,
        EvidenceRung.MEASUREMENT: 0,
        EvidenceRung.ORDER_RELATION: 1,
        EvidenceRung.RESPONSE: 2,
        EvidenceRung.LOCAL_LAW: 3,
    }
    return min(
        family.assessments,
        key=lambda value: (
            -rung_rank[value.profile_assessment.qualification_trace.highest_supported_rung],
            rank[value.candidate_id],
            value.candidate_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class ResponseLawQualificationService:
    """Only service allowed to construct terminal qualification results/laws."""

    payload_reader: CandidatePayloadReader
    decoder_registry: CandidatePayloadDecoderRegistry

    def qualify(
        self,
        system: SystemSpec,
        dataset_or_projection: ObjectIdentity,
        family: ComponentUncertaintyFamilyAssessment,
    ) -> LawQualificationResult:
        ledger = family.ledger
        if ledger.system != ObjectIdentity.from_record(system.system_id, system):
            raise ValueError("candidate family is bound to another system")
        if ledger.dataset_or_projection != dataset_or_projection:
            raise ValueError("candidate family is bound to another evidence projection")
        selected = family.selected_assessment()
        if selected is None:
            return self._no_selection(system, dataset_or_projection, family)
        self._verify_payload(selected.candidate_evidence.payload_publication)
        self._verify_extension(selected.candidate_evidence)
        return self._supported(system, dataset_or_projection, family, selected)

    def _verify_payload(self, receipt: CandidatePayloadPublicationReceipt) -> CanonicalRecord:
        return _verified_candidate_payload(
            reader=self.payload_reader,
            decoder_registry=self.decoder_registry,
            receipt=receipt,
        )

    def _verify_extension(
        self,
        candidate: LawCandidateEvidence,
    ) -> tuple[ExtensionBinding, ...]:
        return _verified_candidate_extension(
            decoder_registry=self.decoder_registry,
            candidate=candidate,
        )

    def _supported(
        self,
        system: SystemSpec,
        dataset_or_projection: ObjectIdentity,
        family: ComponentUncertaintyFamilyAssessment,
        selected: ComponentUncertaintyCandidateAssessment,
    ) -> LawQualificationResult:
        if selected.qualification_eligibility is not CandidateQualificationEligibility.PASSING:
            raise ValueError("selected candidate is not qualification-eligible")
        ledger = family.ledger
        evidence = selected.candidate_evidence
        profile = selected.profile_assessment
        extensions = self._verify_extension(evidence)
        claim_spec = _build_claim_spec(system=system, ledger=ledger, promotable=True)
        claim = _build_claim(
            claim_spec=claim_spec,
            status=ScientificStatus.SUPPORTED,
            highest_rung=EvidenceRung.LOCAL_LAW,
            evidence_links=evidence.evidence_links,
            reasons=(),
            omit_failed_observed_rung=False,
        )
        obligations = _build_obligations(ledger.obligation_template, evidence.evidence_links)
        law = ResponseLaw(
            law_id=ledger.claim_template.law_id,
            system_id=system.system_id,
            world_id=system.world.world_id,
            relation=system.relation,
            chart_id=ledger.chart_id,
            representation_kind=ledger.representation_kind,
            causal_strength=ledger.claim_template.causal_strength,
            evaluator=evidence.candidate_evaluator,
            interface_input_quantity_ids=ledger.claim_template.interface_input_quantity_ids,
            interface_output_quantity_ids=ledger.claim_template.interface_output_quantity_ids,
            mapping_assumption_ids=ledger.claim_template.mapping_assumption_ids,
            joint_response_sink_effort_distribution_identified=(
                ledger.claim_template.joint_response_sink_effort_distribution_identified
            ),
            obligations=obligations,
            claim=claim,
            evidence_links=evidence.evidence_links,
            scientific_status=ScientificStatus.SUPPORTED,
            evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            outcome_access=evidence.outcome_access,
            parent_visibility_ceilings=evidence.parent_visibility_ceilings,
            visibility_ceiling=evidence.visibility_ceiling,
            extensions=extensions,
        )
        validate_law_against_system(law, system)
        response_method_projection = self._response_method_projection(
            system=system,
            family=family,
            assessment=selected,
            claim=claim,
            law=law,
            reasons=(),
        )
        return LawQualificationResult(
            result_id=ledger.claim_template.terminal_result_id,
            dataset_or_projection=dataset_or_projection,
            config=selected.candidate_evidence.config,
            candidate_family_assessment=ObjectIdentity.from_record(
                family.assessment_id,
                family,
            ),
            selection_receipt=ObjectIdentity.from_record(
                family.selection.receipt_id,
                family.selection,
            ),
            axis_map=ObjectIdentity.from_record(ledger.axis_map.axis_map_id, ledger.axis_map),
            claim_unit_binding=ledger.claim_unit_binding,
            payload_publication=ObjectIdentity.from_record(
                evidence.payload_publication.receipt_id,
                evidence.payload_publication,
            ),
            system_id=system.system_id,
            world_id=system.world.world_id,
            relation=system.relation,
            chart_id=ledger.chart_id,
            method_key=ledger.method_key,
            method_version=ledger.method_version,
            method_kind=ledger.method_kind,
            representation_kind=ledger.representation_kind,
            selected_candidate_id=selected.candidate_id,
            candidate_evaluator=evidence.candidate_evaluator,
            qualification_trace=profile.qualification_trace,
            terminal_obligations=profile.terminal_obligations,
            metrics=selected.metrics,
            claim=claim,
            response_law=law,
            operational_disposition=QualificationOperationalDisposition.COMPLETE,
            scientific_status=ScientificStatus.SUPPORTED,
            highest_supported_rung=EvidenceRung.LOCAL_LAW,
            reason_codes=(),
            evidence_links=evidence.evidence_links,
            outcome_access=evidence.outcome_access,
            parent_visibility_ceilings=evidence.parent_visibility_ceilings,
            visibility_ceiling=evidence.visibility_ceiling,
            response_method_projection=response_method_projection,
            extensions=extensions,
        )

    def _no_selection(
        self,
        system: SystemSpec,
        dataset_or_projection: ObjectIdentity,
        family: ComponentUncertaintyFamilyAssessment,
    ) -> LawQualificationResult:
        representative = _representative(family)
        ledger = family.ledger
        evidence = representative.candidate_evidence
        family_evidence = _family_evidence(family)
        family_metrics = _family_metrics(family)
        family_trace = _aggregate_trace(family)
        family_obligations = _aggregate_terminal_obligations(family)
        status = _status_for_no_selection(family)
        reasons = tuple(
            sorted(
                {
                    "no-selection",
                    *(
                        reason
                        for assessment in family.assessments
                        for reason in _failed_reasons(assessment)
                    ),
                }
            )
        )
        selection_identity = ObjectIdentity.from_record(
            family.selection.receipt_id,
            family.selection,
        )
        promotable = ledger.visibility_ceiling.is_promotable
        claim_spec = _build_claim_spec(system=system, ledger=ledger, promotable=promotable)
        claim = _build_claim(
            claim_spec=claim_spec,
            status=status,
            highest_rung=family_trace.highest_supported_rung,
            evidence_links=family_evidence,
            reasons=reasons,
            omit_failed_observed_rung=False,
        )
        representative_failure_reasons = _failed_reasons(representative)
        representative_failure_claim = _build_claim(
            claim_spec=claim_spec,
            status=ScientificStatus.NOT_SUPPORTED,
            highest_rung=None,
            evidence_links=evidence.evidence_links,
            reasons=representative_failure_reasons,
            omit_failed_observed_rung=True,
        )
        response_method_projection = self._response_method_projection(
            system=system,
            family=family,
            assessment=representative,
            claim=representative_failure_claim,
            law=None,
            reasons=representative_failure_reasons,
        )
        return LawQualificationResult(
            result_id=ledger.claim_template.terminal_result_id,
            dataset_or_projection=dataset_or_projection,
            config=(
                representative.candidate_evidence.config
                if ledger.claim_template.compatibility_result_id is not None
                else ledger.selector_config
            ),
            candidate_family_assessment=ObjectIdentity.from_record(
                family.assessment_id,
                family,
            ),
            selection_receipt=selection_identity,
            axis_map=ObjectIdentity.from_record(ledger.axis_map.axis_map_id, ledger.axis_map),
            claim_unit_binding=ledger.claim_unit_binding,
            payload_publication=None,
            system_id=system.system_id,
            world_id=system.world.world_id,
            relation=system.relation,
            chart_id=ledger.chart_id,
            method_key=ledger.method_key,
            method_version=ledger.method_version,
            method_kind=ledger.method_kind,
            representation_kind=ledger.representation_kind,
            selected_candidate_id=None,
            candidate_evaluator=None,
            qualification_trace=family_trace,
            terminal_obligations=family_obligations,
            metrics=family_metrics,
            claim=claim,
            response_law=None,
            operational_disposition=QualificationOperationalDisposition.COMPLETE,
            scientific_status=status,
            highest_supported_rung=family_trace.highest_supported_rung,
            reason_codes=reasons,
            evidence_links=family_evidence,
            outcome_access=ledger.outcome_access,
            parent_visibility_ceilings=ledger.parent_visibility_ceilings,
            visibility_ceiling=ledger.visibility_ceiling,
            response_method_projection=response_method_projection,
        )

    def _response_method_projection(
        self,
        *,
        system: SystemSpec,
        family: ComponentUncertaintyFamilyAssessment,
        assessment: ComponentUncertaintyCandidateAssessment,
        claim: Claim,
        law: ResponseLaw | None,
        reasons: tuple[str, ...],
    ) -> LawIdentificationResult | None:
        result_id = family.ledger.claim_template.compatibility_result_id
        convergence: StructuralConvergenceResult | None = assessment.response_method_structural_convergence
        if result_id is None:
            return None
        if convergence is None:
            raise ValueError("LawIdentificationResult compatibility projection requires structural convergence")
        evidence = assessment.candidate_evidence
        profile = assessment.profile_assessment
        return LawIdentificationResult(
            result_id=result_id,
            dataset=evidence.dataset_or_projection,
            config=evidence.config,
            system_id=system.system_id,
            world_id=system.world.world_id,
            relation=system.relation,
            chart_id=family.ledger.chart_id,
            method_key=family.ledger.method_key,
            method_version=family.ledger.method_version,
            method_kind=family.ledger.method_kind,
            representation_kind=family.ledger.representation_kind,
            candidate_evaluator=evidence.candidate_evaluator,
            checks=profile.checks,
            uncertainty=profile.uncertainty,
            convergence=convergence,
            metrics=assessment.metrics,
            claim=claim,
            response_law=law,
            scientific_status=(
                ScientificStatus.SUPPORTED if law is not None else ScientificStatus.NOT_SUPPORTED
            ),
            reason_codes=reasons,
            evidence_links=evidence.evidence_links,
            outcome_access=evidence.outcome_access,
            parent_visibility_ceilings=evidence.parent_visibility_ceilings,
            visibility_ceiling=evidence.visibility_ceiling,
        )


def project_response_method(result: LawQualificationResult) -> LawIdentificationResult:
    """Sole pure compatibility projection; never a second proof owner."""

    if result.response_method_projection is None:
        raise ValueError("qualification result has no LawIdentificationResult compatibility projection")
    return result.response_method_projection
