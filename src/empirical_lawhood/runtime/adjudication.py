"""Receipt-bound scientific adjudication records and strict transport decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.kernel.worlds import WorldKind


MAX_ADJUDICATION_BYTES = 1024 * 1024


class AdjudicationEvaluability(StrEnum):
    EVALUABLE = "EVALUABLE"
    UNEVALUABLE = "UNEVALUABLE"


class AdjudicationReadoutState(StrEnum):
    ADJUDICATED = "ADJUDICATED"
    NOT_ADJUDICATED = "NOT_ADJUDICATED"


@dataclass(frozen=True, slots=True)
class ScientificAdjudicationContext(CanonicalRecord):
    """Outcome-free denominator and lineage context supplied to one evaluator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/scientific-adjudication-context'

    execution_plan: ObjectIdentity
    evidence_world_id: str
    evidence_world_kind: WorldKind
    relation: ObjectIdentity
    independent_unit_id: str
    information_cutoffs: tuple[ObjectIdentity, ...]
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess
    fixture_scope_id: str | None = None
    plumbing_only: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_world_id, field_name="evidence_world_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        require_sorted_unique_ids(
            self.information_cutoffs,
            attribute="object_id",
            field_name="information_cutoffs",
        )
        if not self.information_cutoffs:
            raise ValueError("adjudication context requires an information cutoff")
        if self.fixture_scope_id is not None:
            validate_stable_id(self.fixture_scope_id, field_name="fixture_scope_id")
        if self.plumbing_only != (self.fixture_scope_id is not None):
            raise ValueError("plumbing-only adjudication requires an exact fixture scope")
        if self.plumbing_only and self.evidence_world_kind is not WorldKind.ANALYTIC_REFERENCE:
            raise ValueError("only an analytic reference may be labeled plumbing-only")
        if self.outcome_access in {
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATOR_REVEAL,
        }:
            raise ValueError("adjudication context cannot retain a sealed or in-reveal state")
        if self.outcome_access in {
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        } and not self.visibility_ceiling.is_at_least_as_restrictive_as(
            VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("revealed adjudication context must remain outcome-visible")


@dataclass(frozen=True, slots=True)
class ScientificAdjudicationOutputContract(CanonicalRecord):
    """Static task/output locator; it cannot declare a verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/scientific-adjudication-output-contract'

    capability_key: str
    capability_version: str
    output_id: str
    payload_schema: str
    maximum_bytes: int = MAX_ADJUDICATION_BYTES
    fixture_scope_id: str | None = None
    plumbing_only: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_stable_id(self.output_id, field_name="output_id")
        validate_schema(self.payload_schema)
        if self.maximum_bytes <= 0 or self.maximum_bytes > MAX_ADJUDICATION_BYTES:
            raise ValueError("adjudication output byte bound is invalid")
        if self.fixture_scope_id is not None:
            validate_stable_id(self.fixture_scope_id, field_name="fixture_scope_id")
        if self.plumbing_only != (self.fixture_scope_id is not None):
            raise ValueError("plumbing-only contract requires an exact fixture scope")


@dataclass(frozen=True, slots=True)
class ScientificAdjudicationRecord(CanonicalRecord):
    """Closed scientific/admission readout bound to exact executed evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/scientific-adjudication-record'

    adjudication_id: str
    run_id: str
    adjudication_task_id: str
    execution_plan: ObjectIdentity
    input_materialization_ids: tuple[str, ...]
    output_logical_artifact_ids: tuple[str, ...]
    required_receipt_ids: tuple[str, ...]
    evidence_world_id: str
    evidence_world_kind: WorldKind
    relation: ObjectIdentity
    independent_unit_id: str
    information_cutoffs: tuple[ObjectIdentity, ...]
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess
    evaluability: AdjudicationEvaluability
    scientific_status: ScientificStatus
    admission_status: AdmissionStatus
    reason_codes: tuple[str, ...]
    fixture_scope_id: str | None = None
    plumbing_only: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("adjudication_id", self.adjudication_id),
            ("run_id", self.run_id),
            ("adjudication_task_id", self.adjudication_task_id),
            ("evidence_world_id", self.evidence_world_id),
            ("independent_unit_id", self.independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.input_materialization_ids,
            field_name="input_materialization_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.output_logical_artifact_ids,
            field_name="output_logical_artifact_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.required_receipt_ids,
            field_name="required_receipt_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.information_cutoffs,
            attribute="object_id",
            field_name="information_cutoffs",
        )
        if not self.information_cutoffs:
            raise ValueError("adjudication requires an information cutoff")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.fixture_scope_id is not None:
            validate_stable_id(self.fixture_scope_id, field_name="fixture_scope_id")
        if self.plumbing_only != (self.fixture_scope_id is not None):
            raise ValueError("plumbing-only adjudication requires an exact fixture scope")
        if self.plumbing_only and self.evidence_world_kind is not WorldKind.ANALYTIC_REFERENCE:
            raise ValueError("only an analytic reference may be labeled plumbing-only")
        if self.outcome_access in {
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATOR_REVEAL,
        }:
            raise ValueError("adjudication cannot retain a sealed or in-reveal state")
        if self.outcome_access in {
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        } and not self.visibility_ceiling.is_at_least_as_restrictive_as(
            VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("revealed adjudication must remain outcome-visible")
        if self.evaluability is AdjudicationEvaluability.UNEVALUABLE and (
            self.scientific_status is not ScientificStatus.UNEVALUABLE
            or self.admission_status is not AdmissionStatus.UNEVALUABLE
        ):
            raise ValueError("unevaluable adjudication must retain unevaluable statuses")
        if self.evaluability is AdjudicationEvaluability.EVALUABLE and (
            self.scientific_status in {ScientificStatus.NOT_TESTED, ScientificStatus.UNEVALUABLE}
            or self.admission_status is AdmissionStatus.UNEVALUABLE
        ):
            raise ValueError("evaluable adjudication requires evaluated statuses")

    def matches_context(self, context: ScientificAdjudicationContext) -> bool:
        return all(
            (
                self.execution_plan == context.execution_plan,
                self.evidence_world_id == context.evidence_world_id,
                self.evidence_world_kind is context.evidence_world_kind,
                self.relation == context.relation,
                self.independent_unit_id == context.independent_unit_id,
                self.information_cutoffs == context.information_cutoffs,
                self.visibility_ceiling is context.visibility_ceiling,
                self.outcome_access is context.outcome_access,
                self.fixture_scope_id == context.fixture_scope_id,
                self.plumbing_only == context.plumbing_only,
            )
        )


def encode_scientific_adjudication(
    record: ScientificAdjudicationRecord,
    *,
    payload_schema: str,
) -> bytes:
    """Encode a record under the frozen capability-output schema alias."""

    validate_schema(payload_schema)
    document = json.loads(record.canonical_bytes())
    document["schema"] = payload_schema
    return _canonical_json_document_bytes(document)


def _canonical_json_document_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("adjudication document is not canonical JSON data") from error


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("adjudication document contains a duplicate field")
        result[key] = value
    return result


def _identity(value: object, *, field_name: str) -> ObjectIdentity:
    if not isinstance(value, dict) or set(value) != {"schema", "value", "version"}:
        raise ValueError(f"{field_name} identity envelope is invalid")
    if (
        value["schema"] != ObjectIdentity.SCHEMA
        or value["version"] != ObjectIdentity.VERSION
        or not isinstance(value["value"], dict)
    ):
        raise ValueError(f"{field_name} identity schema is invalid")
    body = value["value"]
    if set(body) != {
        "object_fingerprint",
        "object_id",
        "object_schema",
        "object_version",
    }:
        raise ValueError(f"{field_name} identity fields are invalid")
    object_id = body["object_id"]
    object_schema = body["object_schema"]
    object_version = body["object_version"]
    object_fingerprint = body["object_fingerprint"]
    if not all(
        isinstance(item, str)
        for item in (object_id, object_schema, object_version, object_fingerprint)
    ):
        raise ValueError(f"{field_name} identity fields are invalid")
    return ObjectIdentity(
        object_id=object_id,
        object_schema=object_schema,
        object_version=object_version,
        object_fingerprint=object_fingerprint,
    )


def _strings(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must contain strings")
    return tuple(value)


def _string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name=field_name)


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def decode_scientific_adjudication(
    payload: bytes,
    *,
    payload_schema: str,
) -> ScientificAdjudicationRecord:
    """Strictly decode one bounded canonical adjudication output."""

    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_ADJUDICATION_BYTES:
        raise ValueError("adjudication payload exceeds its byte bound")
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("adjudication payload is not strict UTF-8 JSON") from error
    if _canonical_json_document_bytes(document) != payload:
        raise ValueError("adjudication payload bytes are not canonical")
    if not isinstance(document, dict) or set(document) != {"schema", "value", "version"}:
        raise ValueError("adjudication envelope fields differ")
    if document["schema"] != payload_schema or document["version"] != "1.0.0":
        raise ValueError("adjudication output schema differs from its contract")
    body = document["value"]
    expected_fields = {field.name for field in fields(ScientificAdjudicationRecord)}
    if not isinstance(body, dict) or set(body) != expected_fields:
        raise ValueError("adjudication record fields differ")
    cutoffs = body["information_cutoffs"]
    if not isinstance(cutoffs, list):
        raise ValueError("information_cutoffs must be a list")
    return ScientificAdjudicationRecord(
        adjudication_id=_string(body["adjudication_id"], field_name="adjudication_id"),
        run_id=_string(body["run_id"], field_name="run_id"),
        adjudication_task_id=_string(
            body["adjudication_task_id"], field_name="adjudication_task_id"
        ),
        execution_plan=_identity(body["execution_plan"], field_name="execution_plan"),
        input_materialization_ids=_strings(
            body["input_materialization_ids"],
            field_name="input_materialization_ids",
        ),
        output_logical_artifact_ids=_strings(
            body["output_logical_artifact_ids"],
            field_name="output_logical_artifact_ids",
        ),
        required_receipt_ids=_strings(
            body["required_receipt_ids"],
            field_name="required_receipt_ids",
        ),
        evidence_world_id=_string(body["evidence_world_id"], field_name="evidence_world_id"),
        evidence_world_kind=WorldKind(
            _string(body["evidence_world_kind"], field_name="evidence_world_kind")
        ),
        relation=_identity(body["relation"], field_name="relation"),
        independent_unit_id=_string(body["independent_unit_id"], field_name="independent_unit_id"),
        information_cutoffs=tuple(
            _identity(value, field_name="information_cutoff") for value in cutoffs
        ),
        visibility_ceiling=VisibilityCeiling(
            _string(body["visibility_ceiling"], field_name="visibility_ceiling")
        ),
        outcome_access=OutcomeAccess(_string(body["outcome_access"], field_name="outcome_access")),
        evaluability=AdjudicationEvaluability(
            _string(body["evaluability"], field_name="evaluability")
        ),
        scientific_status=ScientificStatus(
            _string(body["scientific_status"], field_name="scientific_status")
        ),
        admission_status=AdmissionStatus(
            _string(body["admission_status"], field_name="admission_status")
        ),
        reason_codes=_strings(body["reason_codes"], field_name="reason_codes"),
        fixture_scope_id=_optional_string(body["fixture_scope_id"], field_name="fixture_scope_id"),
        plumbing_only=_boolean(body["plumbing_only"], field_name="plumbing_only"),
    )
