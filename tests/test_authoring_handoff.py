"""Shared handoff leaves preserve explicit family routing and strict projections.

SPDX-License-Identifier: MPL-2.0
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.api import authoring_handoff, composition
from empirical_lawhood.kernel.serialization import CanonicalizationError
from empirical_lawhood.runtime.plans import CandidateExecutionPlan


ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "tests/fixtures/authoring-execution-plan.json"


def test_shared_projection_loader_preserves_original_canonical_product(tmp_path):
    path = tmp_path / "preissue-execution-plan.json"
    path.write_bytes(PROJECTION.read_bytes())
    value = authoring_handoff.load_authoring_execution_projection(tmp_path)
    assert isinstance(value, CandidateExecutionPlan)
    assert value.canonical_bytes() == PROJECTION.read_bytes()
    assert value.execution_plan_id == "execution.maintenance.wrapper.rc"
    path.write_bytes(PROJECTION.read_bytes() + b" ")
    with pytest.raises(CanonicalizationError):
        authoring_handoff.load_authoring_execution_projection(tmp_path)


def test_shared_exclusive_export_never_replaces_an_existing_record(tmp_path):
    # Decode through the actual common reader with its fixed source filename.
    (tmp_path / "preissue-execution-plan.json").write_bytes(PROJECTION.read_bytes())
    record = authoring_handoff.load_authoring_execution_projection(tmp_path)
    path = authoring_handoff.write_exclusive_record(tmp_path, "candidate-projection.json", record)
    assert path.read_bytes() == PROJECTION.read_bytes()
    with pytest.raises(FileExistsError):
        authoring_handoff.write_exclusive_record(tmp_path, path.name, record)
    assert path.read_bytes() == PROJECTION.read_bytes()


def _configure_composition(monkeypatch, tmp_path):
    contract = SimpleNamespace(canonical_path=str(tmp_path))
    monkeypatch.setattr(authoring_handoff, "resolve_external_root_contract", lambda *args, **kwargs: contract)
    monkeypatch.setattr(composition, "create_api", lambda **kwargs: kwargs)
    monkeypatch.setattr(composition, "GuardedExternalRoot", lambda contract: contract)
    monkeypatch.setattr(composition, "ExternalArtifactPlane", lambda root: root)
    return contract


@pytest.mark.parametrize(("selector", "loader", "provider", "reactor"), (
    ("reactor_authoring_dir", "load_fresh_reactor_bundle", "ReactorPrefixResponseCandidateContextProvider", True),
    ("circuit_authoring_dir", "load_fresh_rc_bundle", "ResistorCapacitorCandidateContextProvider", False),
    ("electron_gas_authoring_dir", "load_uniform_electron_gas_analytic_authoring_bundle", "UniformElectronGasAnalyticCandidateContextProvider", False),
    ("synthetic_material_authoring_dir", "load_synthetic_material_response_bundle", "SyntheticMaterialResponseMethodCandidateContextProvider", False),
))
def test_four_explicit_cli_authoring_selections_retain_context_and_reactor_ports(tmp_path, monkeypatch, selector, loader, provider, reactor):
    _configure_composition(monkeypatch, tmp_path)
    directory = tmp_path / "authoring"
    directory.mkdir()
    (directory / "preissue-execution-plan.json").write_bytes(PROJECTION.read_bytes())
    experiment = SimpleNamespace(experiment_id="maintenance.wrapper.rc")
    bundle = SimpleNamespace(authoring=SimpleNamespace(base=SimpleNamespace(draft=SimpleNamespace(experiment=experiment))), standard_context=SimpleNamespace(base=SimpleNamespace(registry="synthetic-registry")), catalog="synthetic-catalog")
    seen = []
    monkeypatch.setattr(composition, loader, lambda path: bundle)
    monkeypatch.setattr(composition, provider, lambda selected: seen.append(selected) or "synthetic-provider")
    monkeypatch.setattr(composition, "fresh_reactor_platform_ports", lambda **kwargs: ("synthetic-port",))
    trust = Path("synthetic-approval-trust")
    result = composition.create_cli_api(repo_root=ROOT, operator_storage_profile=object(), approval_checker_trust_path=trust, **{selector: directory})
    assert seen == [bundle]
    assert result["candidate_context_provider"] == "synthetic-provider"
    assert result["candidate_capability_catalog"] == "synthetic-catalog"
    assert result["study_bundle_registry"] == "synthetic-registry"
    assert result["approval_checker_trust_path"] is trust
    assert ("executable_platform_ports" in result) is reactor
    if reactor:
        assert result["executable_platform_ports"] == ("synthetic-port",)
    experiment.experiment_id = "synthetic.wrong-experiment"
    with pytest.raises(ValueError, match="projection binds another experiment"):
        composition.create_cli_api(repo_root=ROOT, operator_storage_profile=object(), **{selector: directory})


def test_cli_zero_selection_and_conflict_refusal_are_unchanged(tmp_path, monkeypatch):
    _configure_composition(monkeypatch, tmp_path)
    result = composition.create_cli_api(repo_root=ROOT, operator_storage_profile=object())
    assert "candidate_context_provider" not in result
    with pytest.raises(ValueError, match="select exactly one"):
        composition.create_cli_api(repo_root=ROOT, operator_storage_profile=object(), reactor_authoring_dir=tmp_path, circuit_authoring_dir=tmp_path)


def test_shared_directory_containment_preserves_family_refusal(tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(authoring_handoff, "resolve_external_root_contract", lambda *args, **kwargs: SimpleNamespace(canonical_path=str(root)))
    with pytest.raises(ValueError, match="fresh RC authoring lies outside explicit external storage"):
        authoring_handoff.resolve_authoring_directory(outside, repo_root=ROOT, storage_profile=object(), escape_message="fresh RC authoring lies outside explicit external storage")
