"""Current six-matrix development scans, full precontact proof and exact recovery."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.composition.matrix_response_study.design import _source_config
from empirical_lawhood.adapters.methods.matrix_response_study import anisotropic_feasibility as science
from empirical_lawhood.adapters.methods.matrix_response_study.geometry_inputs import MatrixGeometryAllocation, allocate_matrix_geometry, PUBLIC_GEOMETRY_MASTER_SEED, validate_geometry_source
from empirical_lawhood.adapters.methods.matrix_response_study.scientific_seed_inputs import anisotropic_member_scientific_ordinal
from empirical_lawhood.adapters.methods.matrix_response_study.numerical_qualification import MatrixResponseNumericalQualificationQualificationReport, MatrixResponseNumericalQualificationDisposition
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactManifest

from .matrix_native_custody import MatrixNativeCustody, matrix_scientific_code_sha256, read_matrix_native_payload, preflight_matrix_native_store, retain_matrix_native_runtime

MATRIX_GEOMETRY_INPUT_MAXIMUM_BYTES = 8 * 1024**2
MATRIX_GEOMETRY_RESULT_MAXIMUM_BYTES = 64 * 1024**2
_NUMERICAL_CHECK_NAMES = (
    "action-reference-optimized", "gradient-finite-difference", "hermitian-noise-covariance",
    "ideal-background-stationarity", "ideal-spectrum", "laplacian-construction-crosscheck",
    "receiver-hidden-transform-invariance", "canonical-action-ordering", "canonical-history-convergence",
    "chain-recurrence", "integrator-hermiticity", "numerical-view-phase-concordance",
    "numerical-view-radius-difference", "metropolis-acceptance-lower", "metropolis-acceptance-upper",
    "metropolis-langevin-radius", "anisotropic-feasibility-projected-wall-seconds",
)


@dataclass(frozen=True, slots=True)
class MatrixGeometryScanInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-geometry/scan-input"
    config_id: str
    allocation: MatrixGeometryAllocation
    source: SixMatrixResponseSixMatrixSourceConfig
    scientific_code_sha256: str
    environment_lock_sha256: str
    numerical_qualification: ArtifactManifest | None = None
    workers: int = 1
    minimum_free_bytes: int = 1024**3
    maximum_output_bytes: int = MATRIX_GEOMETRY_RESULT_MAXIMUM_BYTES

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_geometry_source(self.source)
        for name in ("scientific_code_sha256", "environment_lock_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if type(self.workers) is not int or not 1 <= self.workers <= self.source.maximum_worker_processes:
            raise ValueError("Geometry worker count exceeds its existing source contract")
        if self.minimum_free_bytes < 1 or self.maximum_output_bytes != MATRIX_GEOMETRY_RESULT_MAXIMUM_BYTES:
            raise ValueError("Geometry operation lacks its bounded output/free-space contract")

    @property
    def run_id(self) -> str:
        return f"matrix-geometry.run-{self.fingerprint()[:32]}"


def matrix_geometry_example_input(*, environment_lock_sha256: str) -> MatrixGeometryScanInput:
    return prepare_matrix_geometry_scan(config_id="matrix-geometry.exposed-example",
        master_seed=PUBLIC_GEOMETRY_MASTER_SEED, environment_lock_sha256=environment_lock_sha256)


def prepare_matrix_geometry_scan(*, config_id: str, master_seed: int,
                                environment_lock_sha256: str,
                                prior_exposed_seed_sha256s: tuple[str, ...] = (),
                                workers: int = 1,
                                numerical_qualification: ArtifactManifest | None = None) -> MatrixGeometryScanInput:
    """Allocate the complete current preset; labels never determine numerical draws."""
    return MatrixGeometryScanInput(config_id,
        allocate_matrix_geometry(allocation_id=f"{config_id}.allocation", master_seed=master_seed,
            prior_exposed_seed_sha256s=prior_exposed_seed_sha256s),
        _source_config(), matrix_scientific_code_sha256(), environment_lock_sha256,
        numerical_qualification, workers)


def matrix_native_runtime_observation() -> dict[str, object]:
    """Observe the existing numerical contract through its shared owner."""
    from empirical_lawhood.api.matrix_numerical_provenance import matrix_native_runtime_observation as observe
    return observe()


def authenticate_matrix_native_environment(config, *, project_root: Path) -> None:
    if config.scientific_code_sha256 != matrix_scientific_code_sha256():
        raise ValueError("Matrix operation scientific source owners changed after input preparation")
    lock = read_bounded_bytes(project_root / "uv.lock", maximum_bytes=16 * 1024**2)
    if sha256(lock).hexdigest() != config.environment_lock_sha256:
        raise ValueError("Matrix operation selected environment lock differs")


def _all_tasks(config: MatrixGeometryScanInput):
    return tuple(task for member in config.source.anisotropic_model.family_members
        for stage in science.MatrixResponseAnisotropicFeasibilityStage
        for task in science._tasks(source=config.source, member=member, stage=stage, allocation=config.allocation))


def prove_matrix_geometry_scan(config: MatrixGeometryScanInput) -> dict[str, object]:
    """Construct the installed native owner and exact complete graph without execution."""
    if config.scientific_code_sha256 != matrix_scientific_code_sha256():
        raise ValueError("Geometry proof scientific source owners changed after preparation")
    from empirical_lawhood.adapters.simulators.six_matrix_response.executable_binding import MATRIX_RESPONSE_SIMULATOR_EXECUTABLE_BINDING
    from empirical_lawhood.adapters.composition.generated_executable_bindings import EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import SIX_MATRIX_RESPONSE_CAPABILITY
    from empirical_lawhood.runtime.capabilities import CapabilityRegistry
    registry = CapabilityRegistry("matrix-geometry.native-proof", (SIX_MATRIX_RESPONSE_CAPABILITY,))
    factory = EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY.factory(MATRIX_RESPONSE_SIMULATOR_EXECUTABLE_BINDING.binding_id)
    provider = factory.build_provider(registry=registry, records=(config.source,), platform_ports=())
    if len(provider.runners(registry)) != 1:
        raise ValueError("Geometry proof lacks the exact registered native owner")
    tasks = _all_tasks(config)
    if len(tasks) != 17118 or len({science._rollout_id(t) for t in tasks}) != len(tasks):
        raise ValueError("Geometry proof changes the complete potential task census")
    for task in tasks:
        validate_stable_id(science._rollout_id(task), field_name="rollout_id")
    feasibility = tuple(t for t in tasks if t.stage is science.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY)
    confirmation = tuple(t for t in tasks if t.stage is science.MatrixResponseAnisotropicFeasibilityStage.CONFIRMATION)
    secondary = tuple(t for t in tasks if t.stage is science.MatrixResponseAnisotropicFeasibilityStage.NUMERICAL_CONCORDANCE)
    maximum_steps = sum(t.schedule.total_steps * t.step_multiplier for t in feasibility) + sum(t.schedule.total_steps * t.step_multiplier for t in (*confirmation[:1296], *secondary[:36]))
    if maximum_steps != 12146688 or maximum_steps > config.source.feasibility_envelope.maximum_anisotropic_feasibility_integration_steps:
        raise ValueError("Geometry proof exceeds the requested native integration work")
    return {"config_sha256": config.fingerprint(), "allocation_sha256": config.allocation.fingerprint(),
        "source_sha256": config.source.fingerprint(), "installed_source_capability": SIX_MATRIX_RESPONSE_CAPABILITY.fingerprint(),
        "scientific_code_sha256": config.scientific_code_sha256, "environment_lock_sha256": config.environment_lock_sha256,
        "feasibility_roots": len(feasibility), "potential_confirmation_roots": len(confirmation),
        "actual_selected_confirmation_roots": 1296, "secondary_nested_views": 36,
        "potential_task_count": len(tasks), "maximum_integration_steps": maximum_steps,
        "memory_bytes_per_worker": 4 * 1024**3, "maximum_workers": config.workers,
        "maximum_aggregate_memory_bytes": config.workers * 4 * 1024**3,
        "maximum_retained_rollout_bytes": (9126 + 1296 + 36) * 16384,
        "maximum_output_bytes_per_rollout": 16384,
        "maximum_report_bytes": config.maximum_output_bytes, "checkpoint_interval_steps": 256,
        "graph_sha256": sha256(canonical_json_bytes(tuple((science._rollout_id(t), t.scientific_seed_sha256, t.schedule.total_steps * t.step_multiplier) for t in tasks))).hexdigest(),
        "exposed_example": config.allocation.exposed_example, "evidence_ceiling": "NON_PROMOTABLE", "scientific_execution_performed": False}


def _custody(config, writer, implementation_commit, *, read_only=False):
    return MatrixNativeCustody(writer=writer, config=config, run_id=config.run_id,
        implementation_commit=implementation_commit, minimum_free_bytes=config.minimum_free_bytes,
        maximum_output_bytes=config.maximum_output_bytes, read_only=read_only)


def _qualification(config, writer, implementation_commit):
    manifest = config.numerical_qualification
    if manifest is None:
        raise ValueError("Geometry scan requires an authenticated current numerical qualification; use matrix numerical-qualification first")
    logical = manifest.logical
    if (logical.payload_schema != MatrixResponseNumericalQualificationQualificationReport.SCHEMA
        or logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        or logical.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or manifest.materialization.size_bytes > 4 * 1024**2):
        raise ValueError("Geometry numerical qualification has another schema/access/size")
    parts = manifest.materialization.relative_path.split("/")
    if len(parts) != 4 or parts[0] != "runs" or parts[2:] != ["outputs", "numerical-qualification.json"]:
        raise ValueError("Geometry numerical qualification lacks its exact public task locator")
    from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
    input_path = f"runs/{parts[1]}/inputs/config.json"
    input_manifest = decode_artifact_manifest(read_bounded_bytes(writer.root.resolve(input_path + ".manifest.json", for_write=False), maximum_bytes=1024**2))
    original = decode_canonical_bytes(read_matrix_native_payload(writer, input_manifest, maximum_bytes=MATRIX_GEOMETRY_INPUT_MAXIMUM_BYTES), MatrixGeometryScanInput, maximum_bytes=MATRIX_GEOMETRY_INPUT_MAXIMUM_BYTES)
    original_custody = _custody(original, writer, implementation_commit, read_only=True)
    if (original.run_id != parts[1] or original_custody._manifest(manifest.materialization.relative_path) != manifest
        or original.scientific_code_sha256 != config.scientific_code_sha256
        or original.environment_lock_sha256 != config.environment_lock_sha256):
        raise ValueError("Geometry qualification changes its producing current input or publication")
    record = original_custody.load("numerical-qualification", MatrixResponseNumericalQualificationQualificationReport)
    if record is None:
        raise ValueError("Geometry qualification lacks its exact completed source receipt")
    if (record.source_config != ObjectIdentity.from_record(config.source.config_id, config.source)
        or record.implementation_commit != implementation_commit
        or record.disposition is not MatrixResponseNumericalQualificationDisposition.NUMERICALLY_QUALIFIED):
        raise ValueError("Geometry current numerical source is unqualified or belongs to another source/commit")
    from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import SIX_MATRIX_RESPONSE_CAPABILITY
    if ({check.check_id for check in record.checks} != {
        f"matrix-response-numerical-qualification.{name}" for name in _NUMERICAL_CHECK_NAMES}
        or len(record.canonical_points) != len(config.source.canonical_reference_models)
        or {point.canonical_model for point in record.canonical_points} != {
            ObjectIdentity.from_record(model.model_id, model) for model in config.source.canonical_reference_models}
        or tuple(benchmark.q for benchmark in record.benchmarks) != (2, 3, 4)
        or record.simulator_capability != ObjectIdentity.from_record(SIX_MATRIX_RESPONSE_CAPABILITY.capability_key, SIX_MATRIX_RESPONSE_CAPABILITY)):
        raise ValueError("Geometry numerical prerequisite omits its complete source-defined check/model/benchmark census")
    return record


def bind_matrix_geometry_qualification(*, config: MatrixGeometryScanInput,
                                      qualification_config: MatrixGeometryScanInput,
                                      artifact_writer: ExternalArtifactPlane,
                                      implementation_commit: str) -> MatrixGeometryScanInput:
    """Bind an actual current prerequisite receipt; no acquisition or authority is granted."""
    prove_matrix_geometry_scan(config)
    custody = _custody(qualification_config, artifact_writer, implementation_commit, read_only=True)
    if custody.load("numerical-qualification", MatrixResponseNumericalQualificationQualificationReport) is None:
        raise ValueError("Geometry input binding lacks an actual completed numerical qualification")
    manifest = custody._manifest(f"runs/{qualification_config.run_id}/outputs/numerical-qualification.json")
    bound = replace(config, numerical_qualification=manifest)
    _qualification(bound, artifact_writer, implementation_commit)
    return bound


def _validate_rollout(task, record):
    if len(record.canonical_bytes()) > 16384:
        raise ValueError("Geometry cell exceeds its declared bounded scientific product")
    expected = (science._rollout_id(task), task.stage, task.member.member_id, task.q,
        task.alpha_x_index, task.alpha_y_index, task.alpha_tilde_x, task.alpha_tilde_y,
        task.history_id, task.seed_index, task.numerical_view.view_id, task.schedule.total_steps * task.step_multiplier, task.scientific_seed_sha256)
    actual = (record.rollout_id, record.stage, record.member_id, record.q,
        record.alpha_x_index, record.alpha_y_index, record.alpha_tilde_x, record.alpha_tilde_y,
        record.history_id, record.seed_index, record.numerical_view_id, record.requested_steps, record.rng_seed_sha256)
    if actual != expected:
        raise ValueError("Geometry rollout changes its allocated history/root/view/seed identity")


def run_matrix_geometry_scan(*, config: MatrixGeometryScanInput, artifact_writer: ExternalArtifactPlane,
                             project_root: Path, implementation_commit: str, progress=None):
    """Execute the locked scan; complete exact receipts are reused without native repetition."""
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    prove_matrix_geometry_scan(config)
    authenticate_matrix_native_environment(config, project_root=project_root)
    qualification = _qualification(config, artifact_writer, implementation_commit)
    custody = _custody(config, artifact_writer, implementation_commit)
    existing = custody.load("terminal", science.MatrixResponseAnisotropicFeasibilityQualificationReport)
    if existing is not None:
        return read_matrix_geometry_result(config=config, artifact_writer=artifact_writer, implementation_commit=implementation_commit)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        runtime = matrix_native_runtime_observation()
        retain_matrix_native_runtime(custody, runtime)
        def execute(tasks, *, workers, progress):
            results = []
            pending = []
            for task in tasks:
                record = custody.load(science._rollout_id(task), science.MatrixResponseAnisotropicFeasibilityRolloutSummary)
                if record is None:
                    pending.append(task)
                else:
                    _validate_rollout(task, record)
                    results.append(record)
            # Existing scientific worker implementation; publish each returned
            # cell before continuing. Serial mode gives cell-level interruption recovery.
            if workers == 1:
                for index, task in enumerate(pending):
                    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
                    custody.begin_native_effect(science._rollout_id(task))
                    record = science._run_rollout(task)
                    _validate_rollout(task, record)
                    results.append(custody.retain(science._rollout_id(task), record))
                    if progress is not None:
                        progress(len(results), len(tasks))
            else:
                for task in pending:
                    custody.begin_native_effect(science._rollout_id(task))
                for record in science._execute_tasks(tuple(pending), workers=workers, progress=progress):
                    task = next(t for t in pending if science._rollout_id(t) == record.rollout_id)
                    _validate_rollout(task, record)
                    results.append(custody.retain(record.rollout_id, record))
            return tuple(sorted(results, key=lambda r: r.rollout_id))
        report = science.run_matrix_response_study_anisotropic_feasibility_feasibility(source=config.source,
            implementation_commit=implementation_commit, numerical_qualification_report_sha256=qualification.fingerprint(),
            allocation=config.allocation, workers=config.workers, progress=progress, task_executor=execute)
        custody.retain("terminal", report)
    return {"report": report, "summary": matrix_geometry_summary(report, config), "runtime_observation": runtime}


def matrix_geometry_summary(report, config):
    return {"run_id": config.run_id, "report_sha256": report.fingerprint(), "requested_feasibility_roots": 9126,
        "retained_rollouts": len(report.rollouts), "valid_rollouts": sum(row.valid for row in report.rollouts),
        "invalid_rollouts": sum(not row.valid for row in report.rollouts), "selected_member_id": report.selected_member_id,
        "disposition": report.disposition.value, "reason_codes": report.reason_codes,
        "member_scores": tuple(row.to_document()["value"] for row in report.member_scores),
        "evidence_ceiling": "NON_PROMOTABLE", "exposed_example": config.allocation.exposed_example}


def read_matrix_geometry_result(*, config: MatrixGeometryScanInput, artifact_writer: ExternalArtifactPlane, implementation_commit: str):
    prove_matrix_geometry_scan(config)
    custody = _custody(config, artifact_writer, implementation_commit, read_only=True)
    report = custody.load("terminal", science.MatrixResponseAnisotropicFeasibilityQualificationReport)
    if report is None:
        raise ValueError("Geometry scan has no completed terminal receipt; resume the same selected input")
    qualification = _qualification(config, artifact_writer, implementation_commit)
    if report.numerical_qualification_report_sha256 != qualification.fingerprint():
        raise ValueError("Geometry terminal substitutes its current numerical qualification")
    tasks = {science._rollout_id(task): task for task in _all_tasks(config)}
    expected_ids = {key for key, task in tasks.items() if task.stage is science.MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY
        or task.member.member_id == report.selected_member_id}
    if {row.rollout_id for row in report.rollouts} != expected_ids or report.source_config != ObjectIdentity.from_record(config.source.config_id, config.source):
        raise ValueError("Geometry result omits a requested root or changes its selected confirmation/source")
    for row in report.rollouts:
        _validate_rollout(tasks[row.rollout_id], row)
        if custody.load(row.rollout_id, type(row)) != row:
            raise ValueError("Geometry terminal differs from an exact completed cell receipt")
    rows = {row.rollout_id: row for row in report.rollouts}
    expected_report = science.run_matrix_response_study_anisotropic_feasibility_feasibility(
        source=config.source, implementation_commit=implementation_commit,
        numerical_qualification_report_sha256=qualification.fingerprint(), allocation=config.allocation,
        workers=config.workers, task_executor=lambda tasks, **_: tuple(rows[science._rollout_id(task)] for task in tasks))
    if report != expected_report:
        raise ValueError("Geometry terminal differs from its complete current scientific reduction")
    return {"report": report, "summary": matrix_geometry_summary(report, config)}


def run_matrix_geometry_conformance(*, config: MatrixGeometryScanInput, coordinate: tuple[int, ...],
                                    artifact_writer: ExternalArtifactPlane, project_root: Path, implementation_commit: str):
    """One explicitly selected exposed source cell; this is not the full scan."""
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    prove_matrix_geometry_scan(config)
    authenticate_matrix_native_environment(config, project_root=project_root)
    task = next((t for t in _all_tasks(config) if (
        anisotropic_member_scientific_ordinal(
            mass_x=t.member.mass_x, mass_y=t.member.mass_y, cross_coupling_gamma=t.member.cross_coupling_gamma),
        tuple(science.MatrixResponseAnisotropicFeasibilityStage).index(t.stage), t.alpha_x_index, t.alpha_y_index,
        science.FOUR_FAMILY_PREPARATION_IDS.index(t.history_id), t.seed_index, int(t.step_multiplier == 2)) == coordinate), None)
    if task is None:
        raise ValueError("Geometry conformance coordinate is outside the complete declared allocation")
    custody = _custody(config, artifact_writer, implementation_commit)
    record = custody.load(science._rollout_id(task), science.MatrixResponseAnisotropicFeasibilityRolloutSummary)
    if record is None:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            retain_matrix_native_runtime(custody, matrix_native_runtime_observation())
            custody.begin_native_effect(science._rollout_id(task))
            record = science._run_rollout(task)
        _validate_rollout(task, record)
        custody.retain(science._rollout_id(task), record)
    _validate_rollout(task, record)
    return record


def run_matrix_numerical_qualification(*, config: MatrixGeometryScanInput, artifact_writer: ExternalArtifactPlane,
                                     project_root: Path, implementation_commit: str):
    """Public acquisition of the actual current prerequisite, never inherited C0 approval."""
    preflight_matrix_native_store(artifact_writer, minimum_free_bytes=config.minimum_free_bytes)
    prove_matrix_geometry_scan(config)
    authenticate_matrix_native_environment(config, project_root=project_root)
    custody = _custody(config, artifact_writer, implementation_commit)
    report = custody.load("numerical-qualification", MatrixResponseNumericalQualificationQualificationReport)
    if report is None:
        from threadpoolctl import threadpool_limits
        from empirical_lawhood.adapters.methods.matrix_response_study.numerical_qualification import run_matrix_response_study_numerical_qualification_qualification
        with threadpool_limits(limits=1):
            retain_matrix_native_runtime(custody, matrix_native_runtime_observation())
            custody.begin_native_effect("numerical-qualification")
            report = run_matrix_response_study_numerical_qualification_qualification(source=config.source, implementation_commit=implementation_commit)
        custody.retain("numerical-qualification", report)
    manifest = custody._manifest(f"runs/{config.run_id}/outputs/numerical-qualification.json")
    return {"report": report, "manifest": manifest, "disposition": report.disposition.value,
        "reason_codes": report.reason_codes, "grants_authority": False}
