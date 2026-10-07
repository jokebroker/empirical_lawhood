"""Bounded non-executable UTF-8 table decode for held NREL archive members."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO

from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateTargetPhase
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import NRELActionKind, NRELArchiveDecodeResult, NRELArchiveMemberFormat, NRELArchiveSourceBinding, NRELColumnRole, NRELNativeSample, NRELPreparationTrace, NRELSafeDecodeProfile


def _optional(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _integer(value: str, *, field_name: str, optional: bool = False) -> int | None:
    stripped = value.strip()
    if optional and not stripped:
        return None
    try:
        parsed = int(stripped)
    except ValueError as error:
        raise ValueError(f"NREL {field_name} is not an exact integer") from error
    if str(parsed) != stripped:
        raise ValueError(f"NREL {field_name} is not canonically encoded")
    return parsed


def _decimal(value: str, *, field_name: str) -> Decimal | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = Decimal(stripped)
    except InvalidOperation as error:
        raise ValueError(f"NREL {field_name} is not an exact decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"NREL {field_name} must be finite")
    return parsed


def _boolean(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"NREL {field_name} is not an exact boolean")


def _cell(
    row: tuple[str, ...],
    indexes: dict[NRELColumnRole, int],
    role: NRELColumnRole,
) -> str:
    return row[indexes[role]]


def _sample(
    row: tuple[str, ...],
    indexes: dict[NRELColumnRole, int],
) -> NRELNativeSample:
    observation_valid = _boolean(
        _cell(row, indexes, NRELColumnRole.OBSERVATION_VALID),
        field_name="observation_valid",
    )
    saturated = _boolean(
        _cell(row, indexes, NRELColumnRole.SATURATED),
        field_name="saturated",
    )
    tripped = _boolean(
        _cell(row, indexes, NRELColumnRole.TRIPPED),
        field_name="tripped",
    )
    reasons: set[str] = set()
    if not observation_valid:
        reasons.add("NREL_OBSERVATION_INVALID")
    if saturated:
        reasons.add("NREL_RECEIVER_SATURATION")
    request_tick = _integer(
        _cell(row, indexes, NRELColumnRole.REQUEST_TICK),
        field_name="request_tick",
    )
    assert request_tick is not None
    sequence_index = _integer(
        _cell(row, indexes, NRELColumnRole.SEQUENCE_INDEX),
        field_name="sequence_index",
    )
    assert sequence_index is not None
    return NRELNativeSample(
        sample_id=_cell(row, indexes, NRELColumnRole.SAMPLE_ID).strip(),
        preparation_id=_cell(row, indexes, NRELColumnRole.PREPARATION_ID).strip(),
        run_id=_cell(row, indexes, NRELColumnRole.RUN_ID).strip(),
        sequence_index=sequence_index,
        timestamp_utc=_cell(row, indexes, NRELColumnRole.TIMESTAMP_UTC).strip(),
        action_kind=NRELActionKind(_cell(row, indexes, NRELColumnRole.ACTION_KIND).strip()),
        action_id=_cell(row, indexes, NRELColumnRole.ACTION_ID).strip(),
        requested_action_code=_cell(
            row,
            indexes,
            NRELColumnRole.REQUESTED_ACTION,
        ).strip(),
        action_accepted=_boolean(
            _cell(row, indexes, NRELColumnRole.ACTION_ACCEPTED_FLAG),
            field_name="action_accepted",
        ),
        accepted_action_code=_optional(_cell(row, indexes, NRELColumnRole.ACCEPTED_ACTION)),
        applied_action_code=_optional(_cell(row, indexes, NRELColumnRole.APPLIED_ACTION)),
        realized_action_code=_optional(_cell(row, indexes, NRELColumnRole.REALIZED_ACTION)),
        request_tick=request_tick,
        acceptance_tick=_integer(
            _cell(row, indexes, NRELColumnRole.ACCEPTANCE_TICK),
            field_name="acceptance_tick",
            optional=True,
        ),
        application_tick=_integer(
            _cell(row, indexes, NRELColumnRole.APPLICATION_TICK),
            field_name="application_tick",
            optional=True,
        ),
        realization_tick=_integer(
            _cell(row, indexes, NRELColumnRole.REALIZATION_TICK),
            field_name="realization_tick",
            optional=True,
        ),
        ac_power_w=_decimal(
            _cell(row, indexes, NRELColumnRole.AC_POWER_W),
            field_name="ac_power_w",
        ),
        dc_power_w=_decimal(
            _cell(row, indexes, NRELColumnRole.DC_POWER_W),
            field_name="dc_power_w",
        ),
        ac_uncertainty_w=_decimal(
            _cell(row, indexes, NRELColumnRole.AC_UNCERTAINTY_W),
            field_name="ac_uncertainty_w",
        ),
        dc_uncertainty_w=_decimal(
            _cell(row, indexes, NRELColumnRole.DC_UNCERTAINTY_W),
            field_name="dc_uncertainty_w",
        ),
        tripped=tripped,
        saturated=saturated,
        observation_valid=observation_valid and not saturated,
        reason_codes=tuple(sorted(reasons)),
    )


def safe_decode_nrel_table(
    *,
    payload: bytes,
    source: NRELArchiveSourceBinding,
    profile: NRELSafeDecodeProfile,
    sealed_evaluation_decode_authorized: bool = False,
    evaluator_reveal_authorized: bool = False,
) -> NRELArchiveDecodeResult:
    """Decode one exact selected member without object loading or hidden filtering."""

    source_identity = ObjectIdentity.from_record(source.binding_id, source)
    if profile.source_binding != source_identity:
        raise ValueError("NREL decode source binding differs")
    members = {value.member_id: value for value in source.members}
    member = members.get(profile.member.object_id)
    if member is None or profile.member != ObjectIdentity.from_record(
        member.member_id,
        member,
    ):
        raise ValueError("NREL decode member is absent or differs")
    expected_delimiter = {
        NRELArchiveMemberFormat.CSV_UTF8: ",",
        NRELArchiveMemberFormat.TSV_UTF8: "\t",
    }[member.format]
    if profile.delimiter != expected_delimiter:
        raise ValueError("NREL decode delimiter differs from member format")
    if len(payload) != member.size_bytes or len(payload) > profile.maximum_payload_bytes:
        raise ValueError("NREL member size differs or exceeds decode limit")
    predecode_sha256 = sha256(payload).hexdigest()
    if predecode_sha256 != member.content_sha256:
        raise ValueError("NREL member SHA-256 differs")
    if profile.phase is IndependentSubstrateTargetPhase.EVALUATION and not (
        sealed_evaluation_decode_authorized or evaluator_reveal_authorized
    ):
        raise PermissionError("sealed NREL evaluation decode requires sealed decode authority")
    try:
        text = payload.decode(profile.encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("NREL member is not strict UTF-8") from error
    if "\x00" in text:
        raise ValueError("NREL member contains a NUL byte")
    reader = csv.reader(StringIO(text, newline=""), delimiter=profile.delimiter)
    try:
        headers = tuple(next(reader))
    except StopIteration as error:
        raise ValueError("NREL member is empty") from error
    if headers != profile.expected_headers:
        raise ValueError("NREL member header/order differs from frozen profile")
    header_indexes = {value: index for index, value in enumerate(headers)}
    indexes = {binding.role: header_indexes[binding.header] for binding in profile.column_bindings}
    included = set(profile.included_preparation_ids)
    by_preparation: dict[str, list[NRELNativeSample]] = {
        preparation_id: [] for preparation_id in profile.included_preparation_ids
    }
    scanned = 0
    selected = 0
    for row_number, raw_row in enumerate(reader, start=2):
        scanned += 1
        if scanned > profile.maximum_rows:
            raise ValueError("NREL member exceeds row limit")
        row = tuple(raw_row)
        if len(row) != profile.maximum_columns:
            raise ValueError(f"NREL row {row_number} column count differs")
        if any(len(value.encode("utf-8")) > profile.maximum_cell_bytes for value in row):
            raise ValueError(f"NREL row {row_number} contains an oversized cell")
        preparation_id = _cell(
            row,
            indexes,
            NRELColumnRole.PREPARATION_ID,
        ).strip()
        if preparation_id not in included:
            continue
        by_preparation[preparation_id].append(_sample(row, indexes))
        selected += 1
    absent = tuple(
        preparation_id for preparation_id, samples in by_preparation.items() if not samples
    )
    if absent:
        raise ValueError("NREL frozen preparation partition is absent from the member")
    traces = tuple(
        sorted(
            (
                NRELPreparationTrace(
                    trace_id=f"trace.nrel.{preparation_id}",
                    preparation_id=preparation_id,
                    run_id=samples[0].run_id,
                    apparatus_id=source.apparatus_id,
                    site_id=source.site_id,
                    samples=tuple(sorted(samples, key=lambda value: value.sequence_index)),
                    scientific_unit_count=1,
                )
                for preparation_id, samples in by_preparation.items()
            ),
            key=lambda value: value.trace_id,
        )
    )
    postdecode_sha256 = sha256(payload).hexdigest()
    return NRELArchiveDecodeResult(
        result_id=f"decode-result.{profile.profile_id}",
        source_binding=source_identity,
        decode_profile=ObjectIdentity.from_record(profile.profile_id, profile),
        member_predecode_sha256=predecode_sha256,
        member_postdecode_sha256=postdecode_sha256,
        scanned_row_count=scanned,
        selected_row_count=selected,
        traces=traces,
        scientific_unit_count=len(traces),
        outcome_access=profile.outcome_access,
        sealed_evaluation_decode_authorized=sealed_evaluation_decode_authorized,
        evaluator_reveal_authorized=evaluator_reveal_authorized,
    )


__all__ = ["safe_decode_nrel_table"]
