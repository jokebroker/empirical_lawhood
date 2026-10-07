"Frozen conditional child instantiation without scientific redesign."

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from collections.abc import Mapping
from typing import ClassVar, Protocol, TypeVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_authoring import ConditionalTerminalDisposition

from .candidate_compiler import CandidateCompilationReport, CandidateCompilationDisposition, CandidateGraphExternalInput, ContentIdentityPolicy, DraftStudyCandidate, StudyCompilationReport, ExecutableStudyCandidate
from .plans import ScientificInputRole


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


@dataclass(frozen=True, slots=True)
class ConditionalChildResolution:
    """One provider-validated conditional decision and its exact child binding."""

    instantiation: ConditionalChildInstantiation
    compilation: StudyCompilationReport | None
    parent_input_binding: FrozenParentInputBinding | None

    def __post_init__(self) -> None:
        if (self.compilation is None) != (self.parent_input_binding is None):
            raise ValueError("conditional child and parent binding must be present together")
        if self.compilation is None:
            if self.instantiation.child_candidate is not None:
                raise ValueError("terminal conditional resolution retained a child identity")
            return
        candidate = self.compilation.candidate
        if candidate is None:
            raise ValueError("conditional compilation lacks its exact child candidate")
        base = candidate.base_candidate
        child_identity = ObjectIdentity.from_record(base.candidate_id, base)
        if (
            self.instantiation.child_candidate != child_identity
            or self.parent_input_binding is None
            or self.parent_input_binding.candidate != child_identity
        ):
            raise ValueError("conditional resolution child identities differ")


class ConditionalChildResolver(Protocol):
    "Closed composition seam for outcome-derived conditional child instantiation."

    @property
    def parent_record_schemas(self) -> Mapping[str, type[CanonicalRecord]]: ...

    def resolve(
        self,
        *,
        parent_compilation: StudyCompilationReport,
        parent_record: CanonicalRecord,
    ) -> ConditionalChildResolution: ...


def _canonical_json_payload(record: CanonicalRecord) -> str:
    return record.canonical_bytes().decode("utf-8")


def _validate_canonical_payload(
    payload: str,
    identity: ObjectIdentity,
) -> None:
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("conditional parent payload is not canonical JSON") from error
    canonical = (
        json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if canonical.decode("utf-8") != payload:
        raise ValueError("conditional parent payload is not canonical JSON")
    if sha256(canonical).hexdigest() != identity.object_fingerprint:
        raise ValueError("conditional parent payload differs from its identity")


@dataclass(frozen=True, slots=True)
class FrozenParentInputBinding(CanonicalRecord):
    """Bind a later parent record to an already-frozen candidate input slot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/frozen-parent-input-binding'

    binding_id: str
    candidate: ObjectIdentity
    scientific_graph_sha256: str
    external_input_id: str
    parent_record: ObjectIdentity
    parent_record_payload: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.external_input_id, field_name="external_input_id")
        validate_sha256(
            self.scientific_graph_sha256,
            field_name="scientific_graph_sha256",
        )
        _validate_canonical_payload(self.parent_record_payload, self.parent_record)
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("parent input binding has unsupported outcome access")
        expected_visibility = {
            OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
        }[self.outcome_access]
        if self.visibility_ceiling is not expected_visibility:
            raise ValueError("parent input binding visibility differs from parent access")

    def decode_parent(
        self,
        record_type: type[_RecordT],
        *,
        maximum_bytes: int,
    ) -> _RecordT:
        if record_type.SCHEMA != self.parent_record.object_schema:
            raise ValueError("parent input decoder names another schema")
        value = decode_canonical_bytes(
            self.parent_record_payload.encode("utf-8"),
            record_type,
            maximum_bytes=maximum_bytes,
        )
        if ObjectIdentity.from_record(self.parent_record.object_id, value) != self.parent_record:
            raise ValueError("bound parent record identity changed")
        return value


@dataclass(frozen=True, slots=True)
class ConditionalChildInstantiation(CanonicalRecord):
    "Self-contained binding of one frozen conditional child to one parent record."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/conditional-child-instantiation'

    instantiation_id: str
    parent_candidate: ObjectIdentity
    frozen_successor: ObjectIdentity
    parent_record: ObjectIdentity
    parent_record_payload: str
    scientific_template_sha256: str
    disposition: ConditionalTerminalDisposition
    child_candidate: ObjectIdentity | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.instantiation_id, field_name="instantiation_id")
        validate_sha256(
            self.scientific_template_sha256,
            field_name="scientific_template_sha256",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        _validate_canonical_payload(self.parent_record_payload, self.parent_record)
        if self.disposition is ConditionalTerminalDisposition.EXECUTION_ELIGIBLE:
            if self.child_candidate is None or self.reason_codes:
                raise ValueError("eligible conditional instantiation is incomplete")
        elif self.child_candidate is not None or not self.reason_codes:
            raise ValueError("terminal conditional instantiation lacks its reason")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("conditional instantiation has unsupported outcome access")
        expected_visibility = {
            OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
        }[self.outcome_access]
        if self.visibility_ceiling is not expected_visibility:
            raise ValueError("conditional instantiation visibility differs from parent access")

    def decode_parent(
        self,
        record_type: type[_RecordT],
        *,
        maximum_bytes: int,
    ) -> _RecordT:
        if record_type.SCHEMA != self.parent_record.object_schema:
            raise ValueError("conditional parent decoder names another schema")
        value = decode_canonical_bytes(
            self.parent_record_payload.encode("utf-8"),
            record_type,
            maximum_bytes=maximum_bytes,
        )
        if ObjectIdentity.from_record(self.parent_record.object_id, value) != self.parent_record:
            raise ValueError("conditional parent record identity changed")
        return value


def bind_frozen_parent_input(
    *,
    candidate: DraftStudyCandidate | ExecutableStudyCandidate,
    external_input_id: str,
    parent_record_id: str,
    parent_record: CanonicalRecord,
    outcome_access: OutcomeAccess | None = None,
) -> FrozenParentInputBinding:
    """Materialize one declared parent-input slot without changing the graph."""

    validate_stable_id(external_input_id, field_name="external_input_id")
    validate_stable_id(parent_record_id, field_name="parent_record_id")
    scientific_graph = (
        candidate.scientific_graph
        if isinstance(candidate, DraftStudyCandidate)
        else candidate.base_candidate.base_candidate.scientific_graph
    )
    values = tuple(
        value
        for value in scientific_graph.external_inputs
        if value.input_id == external_input_id
        and value.scientific_role is ScientificInputRole.PARENT_RECEIPT
        and value.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
    )
    if len(values) != 1:
        raise ValueError("candidate lacks one frozen parent-input substitution")
    specification = values[0]
    resolved_access = specification.outcome_access if outcome_access is None else outcome_access
    payload = parent_record.canonical_bytes()
    if specification.payload_schema != parent_record.SCHEMA:
        raise ValueError("bound parent record has another frozen schema")
    if (
        specification.outcome_access is not resolved_access
        or specification.visibility_ceiling
        is not {
            OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
        }.get(resolved_access)
    ):
        raise ValueError("bound parent access differs from its frozen input slot")
    if len(payload) > specification.maximum_size_bytes:
        raise ValueError("bound parent record exceeds its frozen size")
    candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
    parent_identity = ObjectIdentity.from_record(parent_record_id, parent_record)
    seed = {
        "candidate": candidate_identity,
        "scientific_graph_sha256": scientific_graph.fingerprint(),
        "external_input_id": external_input_id,
        "parent_record": parent_identity,
    }
    digest = sha256(canonical_json_bytes(seed)).hexdigest()
    return FrozenParentInputBinding(
        binding_id=f"parent-input-binding.{digest[:24]}",
        candidate=candidate_identity,
        scientific_graph_sha256=scientific_graph.fingerprint(),
        external_input_id=external_input_id,
        parent_record=parent_identity,
        parent_record_payload=_canonical_json_payload(parent_record),
        outcome_access=resolved_access,
        visibility_ceiling={
            OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
        }[resolved_access],
    )


def _parent_input(candidate: DraftStudyCandidate) -> CandidateGraphExternalInput:
    successor = candidate.conditional_successor
    if successor is None:
        raise ValueError("candidate has no frozen conditional child")
    values = tuple(
        value
        for value in successor.scientific_graph.external_inputs
        if value.input_id == successor.request.parent_receipt_input_id
        and value.scientific_role is ScientificInputRole.PARENT_RECEIPT
        and value.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
    )
    if len(values) != 1:
        raise ValueError("frozen conditional child lacks one parent-receipt substitution")
    return values[0]


def instantiate_conditional_child(
    *,
    parent_compilation: CandidateCompilationReport,
    parent_record_id: str,
    parent_record: CanonicalRecord,
    eligible: bool,
    reason_codes: tuple[str, ...] = (),
    outcome_access: OutcomeAccess | None = None,
) -> tuple[
    ConditionalChildInstantiation,
    CandidateCompilationReport | None,
]:
    "Instantiate only the parent record and derived candidate identity.\n\n    The system, experiment, campaign, sources, capabilities, protocol, graph,\n    obligations, thresholds and resource ceiling are copied byte-for-byte from\n    the pre-outcome frozen conditional child.\n    "

    validate_stable_id(parent_record_id, field_name="parent_record_id")
    if (
        parent_compilation.disposition
        is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
        or parent_compilation.candidate is None
    ):
        raise ValueError("conditional parent is not an exact ready candidate")
    parent_candidate = parent_compilation.candidate
    successor = parent_candidate.conditional_successor
    if successor is None:
        raise ValueError("candidate has no frozen conditional child")
    parent_input = _parent_input(parent_candidate)
    resolved_access = parent_input.outcome_access if outcome_access is None else outcome_access
    if parent_input.payload_schema != parent_record.SCHEMA:
        raise ValueError("conditional parent record has another frozen schema")
    if (
        parent_input.outcome_access is not resolved_access
        or parent_input.visibility_ceiling
        is not {
            OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
        }.get(resolved_access)
    ):
        raise ValueError("conditional parent access differs from its frozen input slot")
    payload = parent_record.canonical_bytes()
    if len(payload) > parent_input.maximum_size_bytes:
        raise ValueError("conditional parent record exceeds its frozen size bound")
    parent_identity = ObjectIdentity.from_record(parent_record_id, parent_record)
    successor_identity = ObjectIdentity.from_record(
        successor.request.request_id,
        successor,
    )
    disposition = (
        successor.request.eligible_disposition
        if eligible
        else successor.request.ineligible_disposition
    )
    normalized_reasons = tuple(sorted(set(reason_codes)))
    if eligible and normalized_reasons:
        raise ValueError("eligible conditional instantiation cannot retain stop reasons")
    if not eligible and not normalized_reasons:
        raise ValueError("ineligible conditional instantiation requires a typed reason")
    child: DraftStudyCandidate | None = None
    if eligible:
        seed = {
            "parent_candidate": ObjectIdentity.from_record(
                parent_candidate.candidate_id,
                parent_candidate,
            ),
            "frozen_successor": successor_identity,
            "parent_record": parent_identity,
            "scientific_template_sha256": successor.scientific_template_sha256,
        }
        digest = sha256(canonical_json_bytes(seed)).hexdigest()
        child = replace(
            parent_candidate,
            candidate_id=f"candidate.conditional.{digest[:24]}",
            protocol=successor.protocol,
            scientific_graph=successor.scientific_graph,
            obligation_coverage=successor.obligation_coverage,
            conditional_successor=None,
        )
    instantiation_seed = {
        "parent_candidate": ObjectIdentity.from_record(
            parent_candidate.candidate_id,
            parent_candidate,
        ),
        "frozen_successor": successor_identity,
        "parent_record": parent_identity,
        "disposition": disposition,
        "child_candidate": (
            None if child is None else ObjectIdentity.from_record(child.candidate_id, child)
        ),
    }
    instantiation_digest = sha256(canonical_json_bytes(instantiation_seed)).hexdigest()
    instantiation = ConditionalChildInstantiation(
        instantiation_id=f"conditional-instantiation.{instantiation_digest[:24]}",
        parent_candidate=ObjectIdentity.from_record(
            parent_candidate.candidate_id,
            parent_candidate,
        ),
        frozen_successor=successor_identity,
        parent_record=parent_identity,
        parent_record_payload=_canonical_json_payload(parent_record),
        scientific_template_sha256=successor.scientific_template_sha256,
        disposition=disposition,
        child_candidate=(
            None if child is None else ObjectIdentity.from_record(child.candidate_id, child)
        ),
        reason_codes=normalized_reasons,
        outcome_access=resolved_access,
        visibility_ceiling={
            OutcomeAccess.OUTCOME_BLIND: VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.DEVELOPMENT_VISIBLE: VisibilityCeiling.DEVELOPMENT_ONLY,
            OutcomeAccess.EVALUATION_REVEALED: VisibilityCeiling.OUTCOME_VISIBLE,
        }[resolved_access],
    )
    if child is None:
        return instantiation, None
    child_report = replace(
        parent_compilation,
        protocol_sha256=child.protocol.fingerprint(),
        candidate=child,
        provisional_protocol=child.protocol,
        provisional_graph=child.scientific_graph,
        provisional_coverage=child.obligation_coverage,
        provisional_graph_sha256=child.scientific_graph.fingerprint(),
        provisional_coverage_sha256=child.obligation_coverage.fingerprint(),
    )
    return instantiation, child_report


__all__ = [
    'ConditionalChildResolution',
    'ConditionalChildResolver',
    'ConditionalChildInstantiation',
    "FrozenParentInputBinding",
    "bind_frozen_parent_input",
    'instantiate_conditional_child',
]
