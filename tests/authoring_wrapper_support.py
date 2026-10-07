"""Synthetic, fixed-identity public authoring handoffs; never qualification.

SPDX-License-Identifier: MPL-2.0
"""

from datetime import UTC, datetime
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path

from empirical_lawhood.api.codecs import load_registered_authoring
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("rc", "electron-gas", "material", "matrix")
MODULES = {
    "rc": "empirical_lawhood.api.rc_authoring",
    "electron-gas": "empirical_lawhood.api.uniform_electron_gas_authoring",
    "material": "empirical_lawhood.api.synthetic_material_authoring",
    "matrix": "empirical_lawhood.api.native_authoring",
}

# Producing uv.lock identity of the original frozen matrix products. The golden
# test freezes this external dependency identity; normal authoring still reads
# and binds the current checkout's actual lock bytes.
MATRIX_BASELINE_LOCK_SHA256 = "5f0da9affb34be4e56cdb37e219af85fb81e9094b1eed18de2ad472e6510f494"


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 6, 0, 0, 0, tzinfo=UTC)


def export_wrapper(family, directory, monkeypatch, *, module=None, persist=True, frozen_matrix_lock=False):
    module = module or importlib.import_module(MODULES[family])
    output = directory / "export"
    experiment_id = f"maintenance.wrapper.{family}"
    if family != "matrix":
        def closure(root, closure_id):
            return ImplementationSourceClosure(closure_id, SourceClosureKind.CLEAN_GIT_COMMIT, "a" * 40, "b" * 64, "c" * 64, True)

        monkeypatch.setattr(module, "capture_clean_target_closure", closure)
    if family == "rc":
        study = decode_canonical_bytes((ROOT / "experiments/rc-ladder-response/study.json").read_bytes(), module.ResistorCapacitorLadderStudyConfig, maximum_bytes=65536)
        summary = module.author_fresh_rc(root=ROOT, study=study, experiment_id=experiment_id, output_dir=output)
    elif family == "electron-gas":
        config = decode_canonical_bytes((ROOT / "experiments/electron-gas-response/config.json").read_bytes(), module.UniformElectronGasAnalyticReferenceConfig, maximum_bytes=65536)
        summary = module.author_uniform_electron_gas_analytic_reference(root=ROOT, config=config, experiment_id=experiment_id, output_dir=output)
    elif family == "material":
        config = decode_canonical_bytes((ROOT / "experiments/lattice-pairing-method/authoring.json").read_bytes(), module.SyntheticMaterialResponseMethodConfig, maximum_bytes=128 * 1024)
        summary = module.author_synthetic_material_response(root=ROOT, config=config, experiment_id=experiment_id, output_dir=output)
    else:
        preparation = importlib.import_module("empirical_lawhood.adapters.composition.prepared_response.native_authoring")
        monkeypatch.setattr(preparation, "_implementation_digest", lambda root: "b" * 64)
        monkeypatch.setattr(preparation, "datetime", FrozenDatetime)
        if frozen_matrix_lock:
            source_spec = preparation.PreparedNativeSpec

            def original_producing_spec(*args, **kwargs):
                observed = source_spec(*args, **kwargs)
                assert observed.dependency_lock_sha256 == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
                return replace(observed, dependency_lock_sha256=MATRIX_BASELINE_LOCK_SHA256)

            monkeypatch.setattr(preparation, "PreparedNativeSpec", original_producing_spec)
        config = load_registered_authoring(ROOT / "experiments/prepared-response/prepared-source-qualification-author.json", root_schemas={preparation.PreparedResponseNativeAuthoringInput.SCHEMA: preparation.PreparedResponseNativeAuthoringInput}, maximum_bytes=16 * 1024)
        held = directory / "held"
        held.mkdir(parents=True)
        prior = held / "inspection.json"
        prior.write_text(json.dumps({"schema": "empirical-lawhood/composition/prepared-response/prepared-exposure-inspection", "value": {"excluded_unit_ids": ["unit.synthetic-excluded-a", "unit.synthetic-excluded-b"], "proposed_unit_ids": ["unit.synthetic-prior"], "excluded_seed_ids": ["seed.synthetic-excluded-a", "seed.synthetic-excluded-b"], "proposed_seed_ids": ["seed.synthetic-prior"]}}))
        plan = held / "synthetic-plan.md"
        plan.write_text("Synthetic software handoff plan. No native qualification.\n")
        summary = module.author_matrix_response(config, repo_root=ROOT, source_root=held, plan=plan, design_packet=plan, prior_exposure=prior, model_bank=None, output_dir=output if persist else None)
    summary = {**summary, "authoring_dir": "$AUTHORING_DIR" if summary["authoring_dir"] is not None else None}
    files = [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(output.iterdir())] if output.exists() else []
    return {"summary": summary, "files": files}
