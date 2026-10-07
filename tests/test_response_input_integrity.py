# SPDX-License-Identifier: MPL-2.0
"""External six-matrix response bytes retain containment, bounds and unambiguous exclusions."""

from __future__ import annotations

from empirical_lawhood.api import native_authoring as native_application

import json
import os
from pathlib import Path
from tests.prepared_seed_fixtures import fixture_seed_census

import pytest

from empirical_lawhood.adapters import _bounded_files
from empirical_lawhood.adapters.composition.finite_response_law.native_input import (
    FiniteResponseLawCalibrationInput,
    check_finite_calibration_input,
)
from empirical_lawhood.adapters.composition.prepared_response import native_authoring
from empirical_lawhood.adapters.methods.finite_response_law import science


def test_response_external_read_exact_bound_and_empty_file(tmp_path: Path) -> None:
    member = tmp_path / "census.json"
    member.write_bytes(b"1234")
    assert native_authoring._read_external(tmp_path, member, 4) == (
        b"1234",
        "census.json",
    )
    with pytest.raises(ValueError, match="byte limit"):
        native_authoring._read_external(tmp_path, member, 3)
    member.write_bytes(b"")
    with pytest.raises(ValueError, match="byte ceiling"):
        native_authoring._read_external(tmp_path, member, 4)


@pytest.mark.parametrize("reader", ("external", "plan"))
@pytest.mark.parametrize("replacement", ("member", "ancestor", "fifo"))
def test_response_replacement_between_path_check_and_open_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, replacement: str
) -> None:
    held = tmp_path / "held"
    held.mkdir()
    member = held / "input.json"
    member.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / member.name).write_bytes(b"outside held root")
    original_open = os.open
    replaced = False

    def replace_before_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            if replacement == "ancestor":
                held.rename(tmp_path / "original-held")
                held.symlink_to(outside, target_is_directory=True)
            else:
                member.unlink()
                if replacement == "member":
                    member.symlink_to(outside / member.name)
                else:
                    os.mkfifo(member)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(_bounded_files.os, "open", replace_before_open)
    with pytest.raises(ValueError, match="safely|regular file"):
        if reader == "external":
            native_authoring._read_external(held, member, 64)
        else:
            native_authoring._read_plan(member)
    assert replaced


@pytest.mark.parametrize("change", ("grow", "truncate"))
def test_response_growth_and_truncation_are_bounded_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    member = tmp_path / "input.json"
    member.write_bytes(b"1234")
    original_read = os.read
    requests = []
    returned_bytes = []

    def mutate_before_read(fd, size):
        requests.append(size)
        if len(requests) == 1:
            member.write_bytes(b"x" * 1024 if change == "grow" else b"x")
        value = original_read(fd, size)
        returned_bytes.append(len(value))
        return value

    monkeypatch.setattr(_bounded_files.os, "read", mutate_before_read)
    with pytest.raises(ValueError, match="grew|truncated|changed"):
        native_authoring._read_external(tmp_path, member, 4)
    assert max(requests) <= 4
    assert sum(returned_bytes) <= 5


def _external_call(route: str, held: Path, prior: Path) -> None:
    if route == "prepared":
        plan = held / "plan.txt"
        plan.write_text("exposed development input; no scientific claim\n")
        native_application.author_matrix_response(
            native_authoring.PreparedResponseNativeAuthoringInput(
                "empirical-lawhood-prepared-response-integrity-test",
                "source-qualification",
                "empirical-lawhood.prepared-response.integrity-test",
                "empirical-lawhood-prepared-response-integrity-test",
                'a57a528e3aab3a16eafb2f320ea2d343a4530b9bd916e9b0e20012ee570a39b5',
                root_seed_census=fixture_seed_census("qualification", 'a57a528e3aab3a16eafb2f320ea2d343a4530b9bd916e9b0e20012ee570a39b5'),
            ),
            repo_root=Path(__file__).parents[1],
            source_root=held,
            plan=plan,
            design_packet=plan,
            prior_exposure=prior,
            model_bank=None,
        )
    else:
        check_finite_calibration_input(
            FiniteResponseLawCalibrationInput(
                "empirical-lawhood-finite-response-law-calibration-integrity"
            ),
            source_root=held,
            plan=Path(science.__file__).with_name("specification.md"),
            prior_exposure=prior,
        )


@pytest.mark.parametrize("route", ("prepared", "finite"))
@pytest.mark.parametrize(
    "field",
    (
        "excluded_unit_ids",
        "proposed_unit_ids",
        "excluded_seed_ids",
        "proposed_seed_ids",
        "schema",
        "value",
    ),
)
def test_duplicate_census_fields_refuse_at_public_input_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, field: str
) -> None:
    def no_compilation(*args, **kwargs):
        pytest.fail("malformed exposure reached implementation identity or compilation")

    monkeypatch.setattr(native_authoring, "_implementation_digest", no_compilation)
    schema = (
        "empirical-lawhood/composition/prepared-response/prepared-exposure-inspection"
        if route == "prepared"
        else "empirical-lawhood/methods/finite-response-law/native-exposure-metadata"
    )
    fields = {
        "excluded_unit_ids": ["unit.exposed"],
        "proposed_unit_ids": ["unit.prior"],
        "excluded_seed_ids": ["seed.exposed"],
        "proposed_seed_ids": ["seed.prior"],
    }
    value = json.dumps(fields)
    if field not in ("schema", "value"):
        value = value[:-1] + f', "{field}": ["unrelated.identity"]}}'
    if route == "prepared":
        raw = f'{{"schema":"{schema}","value":{value}}}'
    else:
        raw = f'{{"schema":"{schema}",' + value[1:]
    if field == "schema":
        raw = raw[:-1] + f',"schema":"{schema}"}}'
    if field == "value":
        raw = raw[:-1] + (
            ',"value":{}}' if route == "prepared" else ',"value":{},"value":{}}'
        )
    prior = tmp_path / "census.json"
    prior.write_text(raw)
    with pytest.raises(ValueError, match=f"duplicate JSON mapping key: {field}"):
        _external_call(route, tmp_path, prior)


def test_model_bank_nested_duplicate_is_rejected_before_namespace_adaptation() -> None:
    from empirical_lawhood.adapters._strict_json import loads_external_json

    with pytest.raises(ValueError, match="duplicate JSON mapping key: coefficient"):
        loads_external_json(
            b'{"schema":"empirical-lawhood/test/model-bank","value":{"coefficient":1,"coefficient":2}}'
        )
    with pytest.raises(ValueError, match="non-finite"):
        loads_external_json(b'{"value":NaN}')


@pytest.mark.parametrize("route", ("prepared", "finite"))
@pytest.mark.parametrize(
    "entries", (["unit.z", "unit.a"], ["unit.a", "unit.a"], [0], "unit.a")
)
def test_malformed_census_entries_refuse_before_compilation(
    tmp_path, monkeypatch, route, entries
):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed census reached source compilation")

    monkeypatch.setattr(native_authoring, "_implementation_digest", forbidden)
    fields = {
        "excluded_unit_ids": entries,
        "proposed_unit_ids": ["unit.prior"],
        "excluded_seed_ids": ["seed.exposed"],
        "proposed_seed_ids": ["seed.prior"],
    }
    if route == "prepared":
        document = {
            "schema": "empirical-lawhood/composition/prepared-response/prepared-exposure-inspection",
            "value": fields,
        }
    else:
        document = {
            "schema": "empirical-lawhood/methods/finite-response-law/native-exposure-metadata",
            **fields,
        }
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(document))
    with pytest.raises((TypeError, ValueError)):
        _external_call(route, tmp_path, prior)


@pytest.mark.parametrize("route", ("information", "causal"))
@pytest.mark.parametrize("field", ("schema", "value", "coefficient"))
def test_model_bank_duplicate_refuses_through_authoring_before_target_import(
    tmp_path, monkeypatch, route, field
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "ambiguous model bank reached target import or source compilation"
        )

    monkeypatch.setattr(native_authoring, "_require_target_bank_document", forbidden)
    monkeypatch.setattr(native_authoring, "_implementation_digest", forbidden)
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema": "empirical-lawhood/composition/prepared-response/prepared-exposure-inspection",
                "value": {
                    "excluded_unit_ids": ["unit.exposed"],
                    "proposed_unit_ids": ["unit.prior"],
                    "excluded_seed_ids": ["seed.exposed"],
                    "proposed_seed_ids": ["seed.prior"],
                },
            }
        )
    )
    plan = tmp_path / "plan.txt"
    plan.write_text("synthetic pre-contact design")
    raw = {
        "schema": '{"schema":"empirical-lawhood/test/model-bank","schema":"empirical-lawhood/test/model-bank","value":{}}',
        "value": '{"schema":"empirical-lawhood/test/model-bank","value":{},"value":{}}',
        "coefficient": '{"schema":"empirical-lawhood/test/model-bank","value":{"coefficient":1,"coefficient":2}}',
    }[field]
    bank = tmp_path / "bank.json"
    bank.write_text(raw)
    config = native_authoring.PreparedResponseNativeAuthoringInput(
        f"empirical-lawhood-{route}-response-integrity",
        {"information": "information-prediction", "causal": "causal-response-prediction"}[route],
        f"empirical-lawhood.{route}-response.integrity",
        f"empirical-lawhood-{route}-response-integrity",
        {"information": '97cbce4f0dfba51f5b480c14d5e3415ba0538681e3ef97ae2549cbc7e8074923', "causal": '0d2865eed2380770a1f64e8b3cff0c9eaab6aae69b9388f695f3d474e7222bcd'}[route],
    )
    with pytest.raises(ValueError, match=f"duplicate JSON mapping key: {field}"):
        native_application.author_matrix_response(
            config,
            repo_root=Path(__file__).parents[1],
            source_root=tmp_path,
            plan=plan,
            design_packet=plan,
            prior_exposure=prior,
            model_bank=bank,
        )


@pytest.mark.parametrize("route", ("prepared", "finite"))
@pytest.mark.parametrize("collision", ("unit", "seed"))
def test_public_input_seam_refuses_collisions_before_compilation_or_native_contact(
    tmp_path,
    monkeypatch,
    route,
    collision,
):
    """Use actual retained roster identities, without running any native task."""
    from hashlib import sha256
    from empirical_lawhood.adapters.composition.finite_response_law import native_input
    from empirical_lawhood.adapters.composition.finite_response_law.exposure import (
        native_seed_ids,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("colliding ledger reached authoring or native task enumeration")

    if route == "prepared":
        # Match _external_call's frozen synthetic pre-contact input exactly.
        source = native_authoring.PreparedNativeSpec(
            'qualification',
            'a57a528e3aab3a16eafb2f320ea2d343a4530b9bd916e9b0e20012ee570a39b5',
            sha256(b"exposed development input; no scientific claim\n").hexdigest(),
            sha256((Path(__file__).parents[1] / "uv.lock").read_bytes()).hexdigest(),
            native_authoring.ObjectIdentity.from_record(
                native_authoring.SOURCE_QUALIFICATION_SOURCE_CAPABILITY.capability_key,
                native_authoring.SOURCE_QUALIFICATION_SOURCE_CAPABILITY,
            ),
            native_authoring.prepared_native_member(),
            tuple(native_authoring.prepared_numerical_view(r) for r in (1, 2)),
            None,
            root_seed_census=fixture_seed_census("qualification", 'a57a528e3aab3a16eafb2f320ea2d343a4530b9bd916e9b0e20012ee570a39b5'),
        )
        units = tuple(root.physical_unit_id for root in source.roots)
        seeds = native_authoring.prepared_seed_ids(source)
        monkeypatch.setattr(
            native_authoring,
            "build_prepared_response_source_qualification_authoring",
            forbidden,
        )
    else:
        source = native_input.FiniteResponseLawCalibrationConfig(
            'calibration',
            science.FiniteResponseLawScienceSpec(),
            native_input.ObjectIdentity.from_record(
                native_input.SOURCE_CAPABILITY.capability_key,
                native_input.SOURCE_CAPABILITY,
            ),
            None,
            (),
        )
        units = tuple(root.physical_unit_id for root in source.roots)
        seeds = native_seed_ids(source)
        monkeypatch.setattr(native_input, "calibration_invocations", forbidden)
    fields = {
        "excluded_unit_ids": [units[0] if collision == "unit" else "unit.exposed"],
        "proposed_unit_ids": ["unit.prior"],
        "excluded_seed_ids": [seeds[0] if collision == "seed" else "seed.exposed"],
        "proposed_seed_ids": ["seed.prior"],
    }
    schema = (
        "empirical-lawhood/composition/prepared-response/prepared-exposure-inspection"
        if route == "prepared"
        else "empirical-lawhood/methods/finite-response-law/native-exposure-metadata"
    )
    document = {
        "schema": f"{schema}",
        **({"value": fields} if route == "prepared" else fields),
    }
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="COLLISION|EXPOSED_CALIBRATION_ROSTER"):
        _external_call(route, tmp_path, prior)
