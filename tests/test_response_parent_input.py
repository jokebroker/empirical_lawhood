"""Historical prepared response parent paths and typed declarations cannot grant authority."""

from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.response_parent_input import ImportedResponseParentCustody
from empirical_lawhood.adapters.composition.prepared_response.dependent_input import PreparedResponseDependentInput, check_dependent_input
from empirical_lawhood.adapters.composition.response_composition.input import ResponseCompositionInput, check_response_composition_input
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1] / "experiments"


def _dependent(route: str) -> PreparedResponseDependentInput:
    return load_registered_authoring(
        ROOT / "prepared-response" / f"prepared-{route}-input.json",
        root_schemas={PreparedResponseDependentInput.SCHEMA: PreparedResponseDependentInput},
        maximum_bytes=16 * 1024,
    )


def _response_composition() -> ResponseCompositionInput:
    return load_registered_authoring(
        ROOT / 'causal-transfer-audit/analysis-input.json',
        root_schemas={ResponseCompositionInput.SCHEMA: ResponseCompositionInput},
        maximum_bytes=16 * 1024,
    )


def test_shipped_dependent_and_response_composition_selections_require_original_roles() -> None:
    assert {_dependent(route).route for route in ("dependent-refinement", "fresh-response-calibration")} == {"dependent-refinement", "fresh-response-calibration"}
    assert _response_composition().evidence_role == "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
    with pytest.raises(ValueError, match="PREPARED_RESPONSE_DEPENDENT_INPUT_INVALID"):
        replace(_dependent("dependent-refinement"), route="fresh-response-calibration")
    with pytest.raises(ValueError, match="RESPONSE_COMPOSITION_INPUT_INVALID"):
        replace(_response_composition(), evidence_role="QUALIFICATION")


def test_complete_looking_witness_and_paths_still_refuse_before_outcome_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = tmp_path / "held"
    held.mkdir()
    manifest = held / "parent-manifest.json"
    manifest.write_bytes(b"outcome-sensitive bytes must stay unread")
    witness = ImportedResponseParentCustody(
        'empirical-lawhood/testing/fixtures/response-parent-custody/native-input-manifest',
        "a" * 64,
        ("b" * 64,),
        ('prepared-response.qualification.prepared.r000',),
        ("c" * 64,),
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    )
    assert witness.source_manifest_sha256 == "a" * 64
    supplied = {
        "source_root": held,
        "parent_manifest": manifest,
        "custody": tmp_path / "self-declared-custody.json",
        "reveal_record": tmp_path / "self-declared-reveal.json",
        "analysis_record": tmp_path / "self-declared-analysis.json",
    }
    dependent = {route: _dependent(route) for route in ("dependent-refinement", "fresh-response-calibration")}
    response_composition = _response_composition()

    def no_read(self: Path) -> bytes:
        raise AssertionError(f"read outcome before target authority: {self}")

    monkeypatch.setattr(Path, "read_bytes", no_read)
    for route in ("dependent-refinement", "fresh-response-calibration"):
        with pytest.raises(
            ValueError, match={"dependent-refinement": 'DEPENDENT_REFINEMENT_TARGET_CUSTODY_STORE_REQUIRED', "fresh-response-calibration": 'FRESH_RESPONSE_CALIBRATION_TARGET_CUSTODY_STORE_REQUIRED'}[route]
        ):
            check_dependent_input(dependent[route], **supplied)
    with pytest.raises(ValueError, match="RESPONSE_COMPOSITION_TARGET_CUSTODY_STORE_REQUIRED"):
        check_response_composition_input(response_composition, **supplied)


def test_missing_parent_and_escaped_manifest_refuse_without_bytes(
    tmp_path: Path,
) -> None:
    held = tmp_path / "held"
    held.mkdir()
    with pytest.raises(ValueError, match="DEPENDENT_REFINEMENT_PARENT_MANIFEST_REQUIRED"):
        check_dependent_input(
            _dependent("dependent-refinement"),
            source_root=held,
            parent_manifest=None,
            custody=None,
            reveal_record=None,
            analysis_record=None,
        )
    escaped = tmp_path / "outside.json"
    escaped.write_bytes(b"{}")
    with pytest.raises(ValueError, match="RESPONSE_COMPOSITION_PARENT_MANIFEST_INVALID"):
        check_response_composition_input(
            _response_composition(),
            source_root=held,
            parent_manifest=escaped,
            custody=None,
            reveal_record=None,
            analysis_record=None,
        )
