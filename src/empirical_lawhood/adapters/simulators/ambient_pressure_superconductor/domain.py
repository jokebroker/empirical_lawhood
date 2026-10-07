'Pure excluded solver control material preparation and truth-known solver-smoke adjudication.'

from __future__ import annotations

from decimal import Decimal

from .contracts import ExcludedSolverControlConfig, ExcludedSolverControlSolverSmokeResult, CORRECTIVE_CAMPAIGN_ID, ConfigLifecycle, MaterialPreparationRecord, NumericalViewRecord, SolverControlInputRecord, SolverRunObservation, SolverSmokeDisposition


REQUIRED_OPERANDS = tuple(
    sorted(
        (
            "atom-count",
            "electron-count",
            "estimated-scf-accuracy-ry",
            "scf-convergence",
            "total-energy-ry",
        )
    )
)


def build_control_input(
    config: ExcludedSolverControlConfig,
) -> tuple[MaterialPreparationRecord, NumericalViewRecord, SolverControlInputRecord]:
    preparation = MaterialPreparationRecord(
        preparation_id='preparation.ambient-pressure-superconductor-excluded-solver-control-pb-control',
        config_sha256=config.payload_sha256,
        candidate_id='candidate.ambient-pressure-superconductor-excluded-solver-control-pb-excluded-control',
        source_release_id="epw-6.1-qe-7.6",
        control_case_id=config.control_case_id,
        prototype_family_id="family.elemental-fcc-lead",
        parent_preparation_id="preparation.none-truth-known-source-control",
        composition=(("pb", Decimal("1")),),
        requested_action="HOLD",
        accepted_action="HOLD",
        applied_action="HOLD",
        realized_action="HOLD",
        mechanism_lane="conventional-electron-phonon",
        split_partition="excluded-workflow-control",
        outcome_visibility="truth-known-nonpromotable",
    )
    view = NumericalViewRecord(
        view_id='view.ambient-pressure-superconductor-excluded-solver-control-qe76-pb-scf',
        config_sha256=config.payload_sha256,
        solver_profile_id=config.solver_profile_id,
        source_release_id="epw-6.1-qe-7.6",
        source_archive_sha256=config.source_archive_sha256,
        container_image_digest=config.container_image_digest,
        environment_image_sha256=config.environment_image_sha256,
        executor_implementation_id=config.executor_implementation_id,
        qe_binary_sha256=config.qe_binary_sha256,
        functional="pz-lda",
        pseudopotential_family="epw-bundled-pb-fully-relativistic-nosoc",
        pseudopotential_sha256=config.pseudopotential_sha256,
        input_sha256=config.input_sha256,
        precision="double",
        wavefunction_cutoff_Ry=config.wavefunction_cutoff_Ry,
        charge_density_cutoff_Ry=config.charge_density_cutoff_Ry,
        k_mesh=config.k_mesh,
        smearing_Ry=config.smearing_Ry,
        convergence_threshold_Ry=config.convergence_threshold_Ry,
        device_class="cpu",
        network_required=False,
    )
    control = SolverControlInputRecord(
        input_id='solver-input.ambient-pressure-superconductor-excluded-solver-control-pb-scf',
        config_sha256=config.payload_sha256,
        preparation_fingerprint=preparation.fingerprint(),
        numerical_view_fingerprint=view.fingerprint(),
        control_case_id=config.control_case_id,
        expected_atom_count=config.expected_atom_count,
        expected_electron_count=config.expected_electron_count,
        required_output_operands=REQUIRED_OPERANDS,
    )
    return preparation, view, control


def adjudicate_solver_smoke(
    config: ExcludedSolverControlConfig,
    control: SolverControlInputRecord,
    observation: SolverRunObservation,
) -> ExcludedSolverControlSolverSmokeResult:
    if control.config_sha256 != config.payload_sha256:
        raise ValueError('excluded solver control control input is not bound to the exact config')
    if observation.config_sha256 != config.payload_sha256:
        raise ValueError('excluded solver control observation is not bound to the exact config')
    if control.control_case_id != observation.control_case_id:
        raise ValueError('excluded solver control control and observation case identities differ')
    if observation.solver_profile_id != config.solver_profile_id:
        raise ValueError('excluded solver control observation came from another solver profile')

    values = {
        "atom-count": observation.atom_count is not None,
        "electron-count": observation.electron_count is not None,
        "estimated-scf-accuracy-ry": observation.estimated_accuracy_Ry is not None,
        "scf-convergence": observation.converged,
        "total-energy-ry": observation.total_energy_Ry is not None,
    }
    extracted = tuple(sorted(key for key, present in values.items() if present))
    missing = tuple(sorted(set(control.required_output_operands) - set(extracted)))
    reasons = list(observation.reason_codes)
    if observation.atom_count != control.expected_atom_count:
        reasons.append("atom-count-control-mismatch")
    if observation.electron_count != control.expected_electron_count:
        reasons.append("electron-count-control-mismatch")
    if observation.estimated_accuracy_Ry is not None and (
        observation.estimated_accuracy_Ry > config.convergence_threshold_Ry
    ):
        reasons.append("scf-accuracy-threshold-failed")
    exact_binding = (
        config.lifecycle is ConfigLifecycle.FROZEN
        and bool(config.qe_binary_sha256)
        and observation.executor_implementation_id == config.executor_implementation_id
        and observation.qe_binary_sha256 == config.qe_binary_sha256
    )
    if not exact_binding:
        reasons.append("qualified-binary-binding-required")
    passed = (
        observation.completed
        and observation.converged
        and observation.exit_code == 0
        and not missing
        and observation.atom_count == control.expected_atom_count
        and observation.electron_count == control.expected_electron_count
        and observation.estimated_accuracy_Ry is not None
        and observation.estimated_accuracy_Ry <= config.convergence_threshold_Ry
        and exact_binding
    )
    if config.campaign_id == CORRECTIVE_CAMPAIGN_ID:
        disposition = (
            SolverSmokeDisposition.CORRECTIVE_PASS
            if passed
            else SolverSmokeDisposition.CORRECTIVE_FAIL
        )
        result_id = 'result.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke-corrective-solver-control'
    else:
        disposition = SolverSmokeDisposition.PASS if passed else SolverSmokeDisposition.FAIL
        result_id = 'result.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke'
    if observation.exit_code in {124, 137}:
        disposition = SolverSmokeDisposition.RESOURCE_INADEQUATE
    if passed:
        reasons.extend(
            (
                "excluded-truth-known-control-only",
                "no-superconductivity-claim",
                'solver-route-qualified-for-material-source-design-consideration',
            )
        )
    else:
        reasons.append("broad-material-campaign-not-qualified")
    return ExcludedSolverControlSolverSmokeResult(
        result_id=result_id,
        config_sha256=config.payload_sha256,
        observation_fingerprint=observation.fingerprint(),
        disposition=disposition,
        truth_known_control=True,
        target_contact_count=0,
        extracted_operands=extracted,
        missing_operands=missing,
        units_verified=not missing,
        exact_executor_binding_verified=exact_binding,
        restart_tested=False,
        restart_disposition="not-attempted-clean-control-completion",
        resource_enforcement_limitations=(
            "aggregate-descendant-limits-not-enforced-by-platform",
            "privileged-loop-mount-helper-required-by-vfat-source-volume",
            "runtime-environment-image-scan-is-not-a-scientific-input-port",
            "trusted-local-not-strict-isolated",
        ),
        reason_codes=tuple(sorted(set(reasons))),
    )


__all__ = ["REQUIRED_OPERANDS", "adjudicate_solver_smoke", "build_control_input"]
