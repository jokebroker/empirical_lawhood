# SPDX-License-Identifier: MPL-2.0
"""Generate explicit exposed integration inputs; never create current authority."""

import argparse
from hashlib import sha256
from pathlib import Path

from empirical_lawhood.adapters.composition.finite_response_law.assignment import proposed_scientific_seeds
from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import FiniteResponseLawCurrentAllocation, FiniteResponseLawCurrentExposure, FiniteResponseLawRerunInput, PUBLIC_CALIBRATION_MASTER_SEED, PUBLIC_EVALUATION_MASTER_SEED, original_development_exposure, public_rerun_sample_exposure
from empirical_lawhood.adapters.methods.finite_response_law.nominated_package import current_nomination_from_sources
from empirical_lawhood.adapters.methods.finite_response_law.original_f import original_f_source_identities, authenticate_original_f_sources, original_f_from_bytes
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import CurrentPreparationSourceConfig, MatrixPreparationSourceConfig, MatrixPreparationSourceProtocol
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import FIXED_PROBE_SEED
from empirical_lawhood.adapters.simulator_morphism_challenges.numeric_inputs import exposed_source_export_example, exposed_phase_config_examples
from empirical_lawhood.api.finite_operands import ORIGINAL_F_FILENAMES, _public_bytes, current_preparation_code_sources_sha256, current_preparation_protocol_identity
from empirical_lawhood.kernel.matrix_inputs import MATRIX_NATIVE_PURPOSES, MatrixAllocation, MatrixRootAllocation, MatrixPurposeSeed

ROOT = Path(__file__).resolve().parents[1]


def _root(index: int, cohort: str, label: str):
    seeds = []
    for purpose in sorted(MATRIX_NATIVE_PURPOSES):
        digest = sha256(f'public-exposed-matrix-input|{label}|{index}|{purpose}'.encode()).digest()
        seeds.append(MatrixPurposeSeed(purpose, int.from_bytes(digest[:16], 'big')))
    conditioning = ()
    if cohort == 'constructed':
        conditioning = (MatrixPurposeSeed('passive-probes', FIXED_PROBE_SEED, 'PCG64DXSM'),)
    else:
        seeds.append(MatrixPurposeSeed('passive-probes', int.from_bytes(sha256(f'public-exposed-matrix-probe|{label}|{index}'.encode()).digest(), 'big'), 'PCG64DXSM'))
    return MatrixRootAllocation(f'example.exposed.{label}.root-{index:03d}', cohort, tuple(sorted(seeds, key=lambda seed: seed.purpose_id)), conditioning_seeds=conditioning)


def examples() -> dict[str, bytes]:
    sources = _public_bytes('original_f', ORIGINAL_F_FILENAMES, (37921, 3919, 1428, 17166, 522736))
    identities = original_f_source_identities(tuple(f'example.original-f.source.{index}' for index in range(5)))
    authenticate_original_f_sources(identities, sources)
    lower = original_f_from_bytes(sources[0], payload_id='example.original-f', source_identities=identities)
    nomination, _ = current_nomination_from_sources('example.original-nomination', _public_bytes('nomination', ('development-report.json', 'coefficients.transport.json', 'development-manifest.json'), (32160, 763803, 227387)))
    allocation = MatrixAllocation('example.exposed.matrix-allocation', tuple(_root(index, 'q2' if index < 8 else 'cir1', 'preparation') for index in range(24)))
    conformance = MatrixAllocation('example.exposed.native-conformance-allocation', tuple(_root(index, cohort, 'conformance') for index, cohort in enumerate(('q2', 'cir1', 'constructed'))))
    lock_sha256 = sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    source = CurrentPreparationSourceConfig('example.exposed.preparation-source', allocation.identity, lower.identity, current_preparation_protocol_identity(), current_preparation_code_sources_sha256(), dependency_lock_sha256=lock_sha256)
    generic = MatrixPreparationSourceConfig('example.exposed.matrix-source', allocation.identity, MatrixPreparationSourceProtocol(), current_preparation_code_sources_sha256(), dependency_lock_sha256=lock_sha256)
    original_units, original_seeds = original_development_exposure()
    sample_units, sample_seeds = public_rerun_sample_exposure()
    exposure = FiniteResponseLawCurrentExposure('example.original-and-public-exposure', tuple(sorted(set(original_units) | set(sample_units))), tuple(sorted(set(original_seeds) | set(sample_seeds))))
    allocations = tuple(FiniteResponseLawCurrentAllocation(stage, f'empirical-lawhood.finite-response-law.{stage}.public-sample', FiniteResponseLawScienceSpec().plan_sha256, count, 'EXPOSED_DEVELOPMENT_NONPROMOTABLE', proposed_scientific_seeds(stage, master)) for stage, count, master in (('calibration', 32, PUBLIC_CALIBRATION_MASTER_SEED), ('prospective-evaluation', 64, PUBLIC_EVALUATION_MASTER_SEED)))
    rerun = FiniteResponseLawRerunInput('example.exposed.finite-calibration', 'calibration', allocations[0], nomination, original_units, original_seeds)
    records = {
        'experiments/matrix-inputs/allocation.json': allocation,
        'experiments/matrix-inputs/preparation-source.json': source,
        'experiments/matrix-inputs/matrix-source.json': generic,
        'experiments/preparation-applicability/conformance-allocation.json': conformance,
        'experiments/rc-challenges/source-export.json': exposed_source_export_example(),
        'experiments/finite-response-law/current-calibration-allocation.json': allocations[0],
        'experiments/finite-response-law/current-evaluation-allocation.json': allocations[1],
        'experiments/finite-response-law/current-exposure.json': exposure,
        'experiments/finite-response-law/rerun-calibration.json': rerun,
    }
    records.update({f'experiments/rc-challenges/{config.phase.value.lower()}.json': config for config in exposed_phase_config_examples()})
    from empirical_lawhood.api.matrix_geometry import matrix_geometry_example_input
    from empirical_lawhood.api.selected_events import selected_parent_example_input
    from empirical_lawhood.api.matrix_history_analysis import matrix_history_example_records
    lock_sha256 = sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    records["experiments/matrix-geometry/input.json"] = matrix_geometry_example_input(environment_lock_sha256=lock_sha256)
    records["experiments/selected-events/parent.json"] = selected_parent_example_input(environment_lock_sha256=lock_sha256)
    records.update({f"experiments/matrix-history-analysis/{name}": record for name, record in matrix_history_example_records(ROOT).items()})
    from empirical_lawhood.api.matrix_preparation_analysis import tangent_preparation_configuration
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.records import TransientAnalysisConfig, BaselineSupportConfig
    records["experiments/matrix-tangent/input.json"] = tangent_preparation_configuration(config_id="example.matrix-tangent.exposed", namespace="example.matrix-tangent.exposed", master_seed=70409, project_root=ROOT)
    records["experiments/matrix-transient/input.json"] = TransientAnalysisConfig("example.matrix-transient.exposed")
    records["experiments/matrix-transient/baseline.json"] = BaselineSupportConfig("example.matrix-baseline.exposed")
    return {name: record.canonical_bytes() for name, record in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for relative, payload in examples().items():
        path = ROOT / relative
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                raise SystemExit(f'Integration example drifted: {relative}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


if __name__ == '__main__':
    main()
