"""Public exposed imports/calculation/receipts; no native cohort campaign."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pytest
from typer.testing import CliRunner

from empirical_lawhood.api import matrix_history_analysis as api
from empirical_lawhood.adapters.methods.matrix_history_analysis.records import MatrixHistoryAnalysisConfig, MatrixHistoryAllocation, MatrixHistoryProtocol, MatrixHistorySeed
from empirical_lawhood.adapters.simulators.six_matrix_response import history_source as native
from empirical_lawhood.adapters.simulators.six_matrix_response.history_preparation import four_family_preparation_couplings
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import BAOABGradientCache
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.artifacts import ExternalRootContract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plane(tmp_path):
    path = tmp_path / "exposed-software-store"
    path.mkdir()
    contract = ExternalRootContract("test.matrix-history-root", "fictional test store", str(path), "/", "empirical-lawhood/test/local-root", 1)
    return ExternalArtifactPlane(GuardedExternalRoot(contract))


def allocation(master=481):
    return api.prepare_matrix_history_allocation(allocation_id="test.history.allocation", namespace="test.history", master_seed=master)


def arrays_for(root):
    """Engineered exposed arrays, explicitly not a simulated native history."""
    state = native.initial_history_state(root)
    result = {}
    for view, multiplier in (("primary", 1), ("fine", 2)):
        count = 1024 * multiplier + 1
        result[f"{view}_positions"] = np.repeat(state.positions[None], count, axis=0).astype("<c16")
        result[f"{view}_momenta"] = np.zeros_like(result[f"{view}_positions"])
        result[f"{view}_ticks"] = np.arange(count, dtype="<i8")
        result[f"{view}_alpha"] = np.array([four_family_preparation_couplings(root.family_id, tick,
            ramp_steps=256, integrator_multiplier=multiplier, target_x=2/3, target_y=22/3,
            decreasing_coupling_start=8) for tick in range(count)], dtype="<f8")
    return result


def test_numeric_allocation_names_do_not_supply_freshness():
    first = allocation()
    renamed = api.prepare_matrix_history_allocation(allocation_id="new.labels", namespace="new.labels", master_seed=481)
    assert first.roots[0].seeds == renamed.roots[0].seeds
    assert first.roots[0].seeds != allocation(482).roots[0].seeds
    with pytest.raises(ValueError, match="overlap"):
        api.prepare_matrix_history_allocation(allocation_id="renamed.public", namespace="renamed.public", master_seed=706256, exposure="PROPOSED_UNRUN")
    effective = first.roots[0].seeds[0].effective_seed
    with pytest.raises(ValueError, match="exposed"):
        replace(first, exposure="PROPOSED_UNRUN", prior_effective_seeds=(effective,))
    # Same high128 consumed bits/different low128 still collide.
    seeds = list(first.roots[0].seeds)
    seeds[1] = MatrixHistorySeed(seeds[1].purpose, seeds[0].digest_sha256[:32] + "0"*32)
    with pytest.raises(ValueError, match="consumed"):
        replace(first.roots[0], seeds=tuple(seeds))
    assert decode_canonical_bytes(first.canonical_bytes(), MatrixHistoryAllocation, maximum_bytes=1024**2) == first


@pytest.mark.parametrize("mode", ("ALGEBRA", "PASSIVE"))
def test_public_actual_calculation_retention_recovery_and_missing_roots(plane, monkeypatch, mode):
    inputs = allocation()
    source = api.prepare_matrix_history_source_configuration(config_id="test.history.source", allocation=inputs, repo_root=ROOT)
    supplied = api.publish_matrix_history_arrays(config=source, allocation=inputs, root_index=0,
        arrays=arrays_for(inputs.roots[0]), writer=plane, relative_root="source/h000", repo_root=ROOT)
    assert supplied.acquisition_origin == "SUPPLIED_EXPOSED_ARRAYS"
    config = MatrixHistoryAnalysisConfig("test.history." + mode.lower(), mode, source)
    result = api.analyze_matrix_histories(config=config, allocation=inputs,
        source_receipt_paths=tuple(f"source/h{i:03d}/receipt.json" for i in range(256)), writer=plane, relative_root="analysis", repo_root=ROOT)
    assert len(result.roots) == 256
    assert result.roots[0].disposition == "COMPLETE"
    assert result.roots[1].disposition == "UNENTERED"
    assert result.roots[1].requested_samples == (130 if mode == "ALGEBRA" else 8)
    assert result.disposition == "UNEVALUABLE"
    assert result.roots[0].retained_samples == (130 if mode == "ALGEBRA" else 8)
    summary = api.matrix_history_result_summary(result)
    assert summary["completed_histories"] == 1 and not summary["scientific_qualification_granted"]
    assert summary["unentered_histories"] == 255
    assert len(result.source_receipt_locators) == 256
    if mode == "ALGEBRA":
        assert summary["requested_nested_samples"] == 36224
    if mode == "PASSIVE":
        assert all(v is None for _, values in result.family_medians for _, v in values)
    from empirical_lawhood.adapters.methods.matrix_history_analysis import analysis
    def forbidden(*args, **kwargs):
        raise AssertionError("receipt recovery repeated scientific work")
    monkeypatch.setattr(analysis, "algebra_history", forbidden)
    monkeypatch.setattr(analysis, "passive_history", forbidden)
    monkeypatch.setattr(native, "acquire_history", forbidden)
    assert api.read_matrix_history_analysis(writer=plane, relative_path="analysis/receipt.json", config=config) == result
    assert api.analyze_matrix_histories(config=config, allocation=inputs, source_receipt_paths=(),
        writer=plane, relative_root="analysis", repo_root=ROOT, recover=True) == result
    # Verify actual input source custody on every read, before numeric result contact.
    path = plane.root.resolve(supplied.arrays_publication.materialization.relative_path, for_write=False)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises((ValueError, RuntimeError)):
        api.read_matrix_history_analysis(writer=plane, relative_path="analysis/receipt.json", config=config)


def test_native_one_step_source_and_purpose_pairing():
    root = allocation().roots[64]
    member, views = native.history_native_parameters()
    def run(root):
        streams = {s.purpose: np.random.Generator(np.random.PCG64DXSM(s.effective_seed)) for s in root.seeds}
        initial = native.initial_history_state(root)
        return native.advance_history_pair(initial, initial, root=root, coarse_step=1, protocol=MatrixHistoryProtocol(),
            streams=streams, caches=(BAOABGradientCache(), BAOABGradientCache()), member=member, views=views), streams
    (primary, halfway, fine), streams = run(root)
    renamed = replace(root, history_id="renamed.history")
    (again, _, _), _ = run(renamed)
    np.testing.assert_array_equal(primary.positions, again.positions)
    assert (primary.step_index, halfway.step_index, fine.step_index) == (1, 1, 2)
    assert primary.alpha_tilde_x == fine.alpha_tilde_x
    # Autonomous stream has not been consumed by a preparation step.
    untouched = np.random.Generator(np.random.PCG64DXSM(root.seed_for("autonomous-coarse-driver")))
    assert streams["autonomous-coarse-driver"].bit_generator.state == untouched.bit_generator.state
    (changed, _, _), _ = run(allocation(482).roots[64])
    assert not np.array_equal(primary.positions, changed.positions)


def test_source_preflight_recovery_and_interruption_never_reacquire(plane, monkeypatch):
    inputs = allocation()
    source = api.prepare_matrix_history_source_configuration(config_id="test.history.source", allocation=inputs, repo_root=ROOT)
    supplied = api.publish_matrix_history_arrays(config=source, allocation=inputs, root_index=0,
        arrays=arrays_for(inputs.roots[0]), writer=plane, relative_root="source/h000", repo_root=ROOT)
    def forbidden(*args, **kwargs):
        raise AssertionError("existing source recovery reacquired history")
    monkeypatch.setattr(native, "acquire_history", forbidden)
    assert api.produce_matrix_histories(config=source, allocation=inputs, writer=plane, relative_root="source",
        repo_root=ROOT, root_indices=(0,), recover=True) == (supplied,)
    with pytest.raises(FileExistsError):
        api.produce_matrix_histories(config=source, allocation=inputs, writer=plane, relative_root="source", repo_root=ROOT, root_indices=(0,))
    # Receipt omission is an interrupted source, not permission to sample again.
    receipt = plane.root.resolve("source/h000/receipt.json", for_write=False)
    receipt.unlink()
    with pytest.raises(FileExistsError, match="Interrupted"):
        api.produce_matrix_histories(config=source, allocation=inputs, writer=plane, relative_root="source", repo_root=ROOT, root_indices=(0,), recover=True)
    # A remaining manifest makes this a corrupt/partial pair, never UNENTERED.
    config = MatrixHistoryAnalysisConfig("test.history.algebra", "ALGEBRA", source)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        api.analyze_matrix_histories(config=config, allocation=inputs,
            source_receipt_paths=("source/h000/receipt.json",), writer=plane, relative_root="refused-analysis", repo_root=ROOT)
    assert not plane.root.resolve("refused-analysis/receipt.json", for_write=True).exists()
    with pytest.raises(ValueError):
        api.produce_matrix_histories(config=source, allocation=inputs, writer=plane, relative_root="source/../escape", repo_root=ROOT, root_indices=(1,))


def test_source_wrong_clock_couplings_identity_and_code_refuse_before_effect(plane):
    inputs = allocation()
    source = api.prepare_matrix_history_source_configuration(config_id="test.history.source", allocation=inputs, repo_root=ROOT)
    arrays = arrays_for(inputs.roots[0])
    arrays["fine_ticks"][1] = 99
    with pytest.raises(ValueError, match="clock"):
        api.publish_matrix_history_arrays(config=source, allocation=inputs, root_index=0, arrays=arrays, writer=plane,
            relative_root="refused-clock", repo_root=ROOT)
    arrays = arrays_for(inputs.roots[0])
    arrays["primary_alpha"][500, 0] = 8
    with pytest.raises(ValueError, match="schedule"):
        api.publish_matrix_history_arrays(config=source, allocation=inputs, root_index=0, arrays=arrays, writer=plane,
            relative_root="refused-couplings", repo_root=ROOT)
    with pytest.raises(ValueError, match="code"):
        api.publish_matrix_history_arrays(config=replace(source, code_sources_sha256="f"*64), allocation=inputs, root_index=0,
            arrays=arrays_for(inputs.roots[0]), writer=plane, relative_root="refused-code", repo_root=ROOT)
    assert not plane.root.resolve("refused-clock/arrays.npz", for_write=True).exists()
    assert not plane.root.resolve("refused-couplings/arrays.npz", for_write=True).exists()


def test_explicit_bounded_array_delivery_import_and_byte_refusal(plane, tmp_path):
    inputs = allocation()
    source = api.prepare_matrix_history_source_configuration(config_id="test.history.source", allocation=inputs, repo_root=ROOT)
    destination = tmp_path / "selected-delivery"
    destination.mkdir()
    operand = api.export_matrix_history_arrays(destination=destination, config=source, allocation=inputs,
        root_index=64, arrays=arrays_for(inputs.roots[64]))
    received = api.import_matrix_history_arrays(operand_path=destination / "operand.json", arrays_path=destination / "arrays.npz",
        config=source, allocation=inputs, writer=plane, relative_root="imported/h064", repo_root=ROOT)
    assert received.acquisition_origin == "SUPPLIED_EXPOSED_ARRAYS"
    assert received.arrays.members == operand.members
    with pytest.raises(FileExistsError):
        api.export_matrix_history_arrays(destination=destination, config=source, allocation=inputs,
            root_index=64, arrays=arrays_for(inputs.roots[64]))
    raw = bytearray((destination / "arrays.npz").read_bytes())
    raw[-1] ^= 1
    (destination / "arrays.npz").write_bytes(raw)
    with pytest.raises(ValueError, match="transport"):
        api.import_matrix_history_arrays(operand_path=destination / "operand.json", arrays_path=destination / "arrays.npz",
            config=source, allocation=inputs, writer=plane, relative_root="refused-import", repo_root=ROOT)
    assert not plane.root.resolve("refused-import/arrays.npz", for_write=True).exists()


def test_real_bounded_prefix_cancellation_has_no_completed_native_receipt(plane):
    inputs = allocation()
    source = api.prepare_matrix_history_source_configuration(config_id="test.history.source", allocation=inputs, repo_root=ROOT)
    calls = []
    def interrupt(history_id, tick):
        calls.append((history_id, tick))
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        api.produce_matrix_histories(config=source, allocation=inputs, writer=plane, relative_root="cancelled",
            repo_root=ROOT, root_indices=(0,), progress=interrupt)
    assert calls == [(inputs.roots[0].history_id, 1)]
    assert not plane.root.resolve("cancelled/h000/receipt.json", for_write=True).exists()
    assert not plane.root.resolve("cancelled/h000/arrays.npz", for_write=True).exists()
    with pytest.raises(FileExistsError, match="Interrupted"):
        api.produce_matrix_histories(config=source, allocation=inputs, writer=plane, relative_root="cancelled",
            repo_root=ROOT, root_indices=(0,), recover=True)


def test_saved_result_census_refuses_missing_directions_and_reordered_cells():
    from empirical_lawhood.adapters.methods.matrix_history_analysis.analysis import passive_history, validate_analysis_arrays
    root = allocation().roots[0]
    arrays, _, _, _ = passive_history(arrays_for(root), root)
    validate_analysis_arrays(arrays, root, mode="PASSIVE", compare_commutant_altered=True)
    missing = dict(arrays)
    missing.pop("increment_singular_values")
    with pytest.raises(ValueError, match="census"):
        validate_analysis_arrays(missing, root, mode="PASSIVE", compare_commutant_altered=True)
    changed = dict(arrays)
    changed["cell_role"] = arrays["cell_role"][::-1]
    with pytest.raises(ValueError, match="census"):
        validate_analysis_arrays(changed, root, mode="PASSIVE", compare_commutant_altered=True)


@pytest.mark.parametrize("mode", ("ALGEBRA", "PASSIVE"))
def test_cli_saved_import_actual_analysis_and_retained_read(plane, tmp_path, monkeypatch, mode):
    """Real CLI parsers/science/custody with only the explicit storage selection injected."""
    from empirical_lawhood.adapters.methods.matrix_history_analysis import analysis
    from empirical_lawhood.cli.app import app

    inputs = allocation()
    source = api.prepare_matrix_history_source_configuration(
        config_id="test.cli.history.source", allocation=inputs, repo_root=ROOT)
    config = MatrixHistoryAnalysisConfig("test.cli.history." + mode.lower(), mode, source)
    paths = {name: tmp_path / (name + ".json") for name in ("allocation", "source", "analysis")}
    for name, record in (("allocation", inputs), ("source", source), ("analysis", config)):
        paths[name].write_bytes(record.canonical_bytes())
    delivery = tmp_path / "explicit-delivery"
    delivery.mkdir()
    api.export_matrix_history_arrays(destination=delivery, config=source, allocation=inputs,
        root_index=0, arrays=arrays_for(inputs.roots[0]))
    monkeypatch.setattr("empirical_lawhood.cli.integrations._storage", lambda: (ROOT, None, plane))

    def forbidden(*args, **kwargs):
        raise AssertionError("saved-array CLI route entered native acquisition or repeated analysis")

    monkeypatch.setattr(native, "acquire_history", forbidden)
    runner = CliRunner()

    def invoke(command, *arguments, attempt):
        result = runner.invoke(app, ["campaign", command, *map(str, arguments),
            "--attempt-dir", str(attempt)])
        assert result.exit_code == 0, (result.output, result.exception)
        retained = json.loads((attempt / "invocation.json").read_bytes())
        assert retained["operation_completed"] and retained["native_contact"] == "none"
        expected_inputs = {"matrix-history-analysis": paths["analysis"]}
        if command == "matrix-history-import":
            expected_inputs = {"matrix-history-source": paths["source"]}
        if command != "matrix-history-result":
            expected_inputs["matrix-history-allocation"] = paths["allocation"]
        snapshots = {row["role"]: row for row in retained["inputs"]}
        for role, original in expected_inputs.items():
            snapshot = snapshots[role]
            payload = original.read_bytes()
            assert snapshot["bytes"] == len(payload) and snapshot["sha256"] == sha256(payload).hexdigest()
            if len(payload) <= 64 * 1024:
                assert snapshot["accepted"] and snapshot["used_snapshot"]
                assert (attempt / snapshot["retained_file"]).read_bytes() == payload
            else:
                # The complete allocation remains scientific custody, not an expanded attempt snapshot.
                assert not snapshot["used_snapshot"] and "retained_file" not in snapshot
                assert snapshot["identity_scope"] == "observed_before_operation"
        return json.loads(result.stdout)

    imported = invoke("matrix-history-import", "--input", paths["source"],
        "--allocation", paths["allocation"], "--operand", delivery / "operand.json",
        "--arrays", delivery / "arrays.npz", "--relative-root", "cli-source/h000",
        attempt=tmp_path / "import-attempt")
    assert imported["status"] == "SUPPLIED_EXPOSED_ARRAYS"
    source_receipt = api.read_matrix_history_source_receipt(writer=plane,
        relative_path="cli-source/h000/receipt.json", config=source, allocation=inputs)
    assert imported["receipt_sha256"] == source_receipt.fingerprint()
    assert source_receipt.acquisition_origin == "SUPPLIED_EXPOSED_ARRAYS"

    summary = invoke("matrix-history-analyze", "--input", paths["analysis"],
        "--allocation", paths["allocation"], "--source-root", "cli-source",
        "--relative-root", "cli-result", attempt=tmp_path / "analysis-attempt")
    assert summary["mode"] == mode and summary["disposition"] == "UNEVALUABLE"
    assert summary["requested_histories"] == 256 and summary["completed_histories"] == 1
    assert summary["unentered_histories"] == 255
    assert summary["retained_nested_samples"] == (130 if mode == "ALGEBRA" else 8)
    assert not summary["scientific_qualification_granted"]

    # Readout/recovery must authenticate actual retained arrays, not recalculate them.
    monkeypatch.setattr(analysis, "algebra_history", forbidden)
    monkeypatch.setattr(analysis, "passive_history", forbidden)
    readout = invoke("matrix-history-result", "--input", paths["analysis"],
        "--relative-path", "cli-result/receipt.json", attempt=tmp_path / "read-attempt")
    recovered = invoke("matrix-history-analyze", "--input", paths["analysis"],
        "--allocation", paths["allocation"], "--source-root", "cli-source",
        "--relative-root", "cli-result", "--recover", attempt=tmp_path / "recovery-attempt")
    assert readout == summary == recovered

    data_path = plane.root.resolve(source_receipt.arrays_publication.materialization.relative_path,
        for_write=False)
    changed = bytearray(data_path.read_bytes())
    changed[-1] ^= 1
    data_path.write_bytes(changed)
    refused = runner.invoke(app, ["campaign", "matrix-history-result", "--input",
        str(paths["analysis"]), "--relative-path", "cli-result/receipt.json"])
    assert refused.exit_code == 2 and not refused.stdout
