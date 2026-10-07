"""Application-owned candidate compilation and plan projection.

SPDX-License-Identifier: MPL-2.0
"""

from decimal import Decimal
from pathlib import Path
from empirical_lawhood.adapters.composition.task_resources import (
    progress_resource_envelope,
)
from empirical_lawhood.api.authoring_handoff import create_authoring_api, write_exclusive_record as _save_new
from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.api.results import CompileCandidateRequest
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.runtime.compiler import compile_preissue_run_plan, lower_run_plan
from empirical_lawhood.runtime.study_issue import project_standard_study_extensions
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_authoring import SyntheticMaterialResponseMethodCandidateContextProvider, build_synthetic_material_response_method_authoring
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_contracts import SyntheticMaterialResponseMethodConfig
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_design import BUDGET
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_provider import native_task_id
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_packet import SyntheticMaterialMethodProfile
from empirical_lawhood.infrastructure.source_closure import capture_clean_target_closure


def author_synthetic_material_response(
    *, root: Path, config: SyntheticMaterialResponseMethodConfig, experiment_id: str, output_dir: Path
) -> dict[str, object]:
    """Compile strict executable study authoring and project a plan without executing the method suite."""

    validate_stable_id(experiment_id, field_name="experiment_id")
    closure = capture_clean_target_closure(root, f"{experiment_id}.source-closure")
    bundle = build_synthetic_material_response_method_authoring(
        config=config,
        experiment_id=experiment_id,
        implementation_sha256=closure.implementation_sha256,
    )
    if output_dir.is_symlink() or output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("ambient pressure superconductor authoring output must be a new empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_incomplete_output(output_dir) as progress:
        profile = SyntheticMaterialMethodProfile(
            f"{experiment_id}.profile", experiment_id, config.fingerprint()
        )
        _save_new(output_dir, "profile.json", profile)
        _save_new(output_dir, "config.json", config)
        authoring = _save_new(output_dir, "authoring.json", bundle.authoring)
        _save_new(output_dir, "base-authoring.json", bundle.authoring.base)
        payloads = tuple(
            _save_new(output_dir, f"payload-{index}.json", record)
            for index, record in enumerate(bundle.payloads)
        )
        decoders = tuple(
            _save_new(output_dir, f"decoder-{index}.json", record)
            for index, record in enumerate(bundle.decoder_registrations)
        )
        progress.stage = "candidate compilation"
        api = create_authoring_api(
            repo_root=root,
            candidate_context_provider=SyntheticMaterialResponseMethodCandidateContextProvider(bundle),
            candidate_capability_catalog=bundle.catalog,
        )
        compiled = api.compile_candidate(
            CompileCandidateRequest(authoring, payloads, decoders)
        )
        if not compiled.succeeded or compiled.payload is None:
            raise RuntimeError(
                f"ambient pressure superconductor gauge covariant response method candidate refused: {compiled.reason_codes}"
            )
        report = compiled.payload.report
        candidate = report.candidate
        if candidate is None:
            raise RuntimeError("ambient pressure superconductor gauge covariant response method candidate compiler returned no candidate")
        progress.stage = "plan projection"
        base = candidate.base_candidate.base_candidate
        issued_extensions, _, _ = project_standard_study_extensions(
            candidate=candidate,
            extension_payload_bytes=tuple(
                record.canonical_bytes() for record in bundle.payloads
            ),
            decoder_registrations=bundle.decoder_registrations,
            extension_materializations=report.extension_materializations,
        )
        protocol = base.protocol
        resources = progress_resource_envelope(
            prefix=experiment_id,
            design=config,
            protocol=protocol,
            issued_extensions=ObjectIdentity.from_record(
                issued_extensions.issued_extension_set_id, issued_extensions
            ),
            cpu_seconds=BUDGET.wall_time_seconds,
            native_preparations=(
                (
                    native_task_id(config),
                    config.independent_unit_id,
                    f"{config.independent_unit_id}.method-suite",
                ),
            ),
            progress_counter="ambient-pressure-superconductor-completed-gauge-covariant-response-method-suite",
            heartbeat_seconds=Decimal(60),
            maximum_concurrency=1,
            maximum_memory_bytes=BUDGET.memory_bytes,
            terminal_failure_codes=("SYNTHETIC_MATERIAL_RESPONSE_METHOD_FAILURE",),
        )
        projected = compile_preissue_run_plan(
            run_plan_id=experiment_id,
            candidate_record=base,
            candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
            registry=bundle.standard_context.base.registry,
            implementation_commit=closure.implementation_commit,
            issued_extension_set=issued_extensions,
            resource_envelope=resources,
            jit_census=None,
            jit_manifest=None,
        )
        lowered = lower_run_plan(projected, bundle.standard_context.base.registry)
        progress.stage = "product export"
        for name, record in (
            ("candidate-report.json", report),
            ("candidate.json", candidate),
            ("base-candidate.json", candidate.base_candidate),
            ("source-closure.json", closure),
            ("resources.json", resources),
            ("preissue-run-plan.json", projected),
            ("preissue-execution-plan.json", lowered),
        ):
            _save_new(output_dir, name, record)
        return {
            "experiment_id": experiment_id,
            "implementation_commit": closure.implementation_commit,
            "candidate_sha256": candidate.fingerprint(),
            "task_count": len(lowered.tasks),
            "output_count": sum(len(task.outputs) for task in lowered.tasks),
            "independent_method_units": 1,
            "nested_gauge_covariant_response_fixtures": 9,
            "authoring_dir": str(output_dir),
            "public_config_exposed": config.config_id
            == "empirical-lawhood-ambient-pressure-superconductor-gauge-covariant-response-method-development",
            "source_contacted": False,
            "native_tasks_executed": 0,
        }
