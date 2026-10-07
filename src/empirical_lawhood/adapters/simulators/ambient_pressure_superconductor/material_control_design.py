'Pure material control workflow-to-control and noncompensating panel transformations.'

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from hashlib import sha256
from pathlib import PurePosixPath
import tarfile

from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .gauge_covariant_response_contracts import GaugeCovariantResponseConformanceResult, GaugeCovariantResponseResponseObservation

from .material_control_contracts import MaterialControlCalibrationFreeze, MaterialControlCloseout, MaterialControlConfig, MaterialControlControlObservation, MaterialControlControlPanel, MaterialControlDisposition, MaterialControlScienceFreeze, MaterialControlStageResult, MaterialControlTutorialReproduction, MaterialControlTruthWorldConformance, BridgeDisposition, ControlClass, ControlDisposition, MultibandStrongCouplingBridgeQualification, MaterialGaugeCovariantViewDisposition, MultibandStrongCouplingMaterialViewResult
from .material_control_solver import MaterialControlWorkflowOperands, MaterialControlWorkflowProfile, WorkflowKind, WorkflowSource
from .material_control_source import _external_path, CONTROL_INPUT_ARCHIVE, TUTORIAL_ARCHIVE, TUTORIAL_ARCHIVE_SHA256
from empirical_lawhood._required_inputs import required_external_sha256


def profile_source_identity(profile: MaterialControlWorkflowProfile) -> tuple[str, str, str]:
    archive_relative = (
        TUTORIAL_ARCHIVE
        if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
        else CONTROL_INPUT_ARCHIVE
    )
    archive_path = _external_path(archive_relative)
    source_sha = (
        TUTORIAL_ARCHIVE_SHA256
        if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
        else required_external_sha256(
            'EMPIRICAL_LAWHOOD_MATERIAL_CONTROL_CONTROL_INPUT_ARCHIVE_SHA256'
        )
    )
    rows: list[tuple[str, int, str]] = []
    primary_payload: bytes | None = None
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            selected = (
                profile.source is WorkflowSource.SSSP_CONTROL
                and profile.source_root == "/".join(path.parts[:3])
            ) or (
                profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
                and path.parts[:2]
                == (
                    "tutorial04",
                    "exercise1" if profile.prefix == "pb" else "exercise2",
                )
            )
            if not selected:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('material control profile source member cannot be read')
            payload = stream.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError('material control profile source member size differs')
            digest = sha256(payload).hexdigest()
            rows.append((member.name, member.size, digest))
            if path.name == "scf.in" and (
                "phonon" in path.parts
                or (
                    profile.kind is WorkflowKind.SCF_CLASS and path.parent.name == "scf"
                )
            ):
                primary_payload = payload
    if not rows or primary_payload is None:
        raise ValueError('material control profile source members are incomplete')
    preparation_sha = sha256(canonical_json_bytes(tuple(sorted(rows)))).hexdigest()
    return source_sha, sha256(primary_payload).hexdigest(), preparation_sha


def build_tutorial_reproduction(
    *,
    profile: MaterialControlWorkflowProfile,
    operands: MaterialControlWorkflowOperands,
    source_identity: tuple[str, str, str] | None = None,
) -> MaterialControlTutorialReproduction:
    if profile.source is not WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE:
        raise ValueError("tutorial reproduction received an SSSP profile")
    source_sha, _input_sha, preparation_sha = (
        profile_source_identity(profile) if source_identity is None else source_identity
    )
    if profile.control_class is ControlClass.POSITIVE_ISOTROPIC:
        recovered = all(
            (
                operands.exit_code == 0,
                not operands.timed_out,
                operands.scf_repeat_evaluable,
                operands.scf_converged,
                operands.phonon_evaluable,
                operands.minimum_phonon_frequency_cm1 >= Decimal("-5"),
                operands.epw_evaluable,
                Decimal("0.9") <= operands.electron_phonon_lambda <= Decimal("1.4"),
                Decimal("4") <= operands.tc_estimate_K <= Decimal("6"),
                Decimal("0.6") <= operands.gap_max_meV <= Decimal("1.3"),
            )
        )
    else:
        recovered = all(
            (
                operands.exit_code == 0,
                operands.scf_converged,
                operands.phonon_evaluable,
                operands.minimum_phonon_frequency_cm1 >= Decimal("-5"),
                operands.epw_evaluable,
                Decimal("30") <= operands.tc_estimate_K <= Decimal("50"),
                Decimal("0.5") <= operands.gap_min_meV <= Decimal("3"),
                Decimal("6") <= operands.gap_max_meV <= Decimal("10"),
                operands.gap_max_meV >= Decimal("1.5") * operands.gap_min_meV,
            )
        )
    reasons = list(operands.reason_codes)
    reasons.append(
        "reason.official-workflow-qualitatively-recovered"
        if recovered
        else "reason.official-workflow-not-recovered"
    )
    reasons.append("reason.tutorial-view-excluded-from-sssp-calibration")
    return MaterialControlTutorialReproduction(
        reproduction_id=f"reproduction.{profile.profile_id.removeprefix('workflow.')}",
        structure_id=profile.structure_id,
        control_class=profile.control_class,
        view_id=profile.view_id,
        source_asset_sha256=source_sha,
        preparation_sha256=preparation_sha,
        exit_code=operands.exit_code,
        timed_out=operands.timed_out,
        terminal_step_id=operands.terminal_step_id,
        wall_time_seconds=operands.wall_time_seconds,
        peak_scratch_bytes=operands.peak_scratch_bytes,
        workflow_completed=operands.exit_code == 0 and not operands.timed_out,
        pairing_recovered=recovered,
        electron_phonon_lambda=operands.electron_phonon_lambda,
        tc_estimate_K=operands.tc_estimate_K,
        gap_min_meV=operands.gap_min_meV,
        gap_max_meV=operands.gap_max_meV,
        raw_archive_sha256=operands.raw_archive_sha256,
        raw_archive_bytes=operands.raw_archive_bytes,
        science_calibration_authorized=False,
        physical_independent_unit_count=1,
        reason_codes=tuple(sorted(set(reasons))),
    )


def build_control_observation(
    *,
    profile: MaterialControlWorkflowProfile,
    operands: MaterialControlWorkflowOperands,
    material_gauge_covariant_response: MultibandStrongCouplingMaterialViewResult | None,
    source_identity: tuple[str, str, str] | None = None,
) -> MaterialControlControlObservation:
    if profile.source is not WorkflowSource.SSSP_CONTROL:
        raise ValueError("SSSP control observation received a tutorial profile")
    source_sha, input_sha, preparation_sha = (
        profile_source_identity(profile) if source_identity is None else source_identity
    )
    if profile.control_class in {
        ControlClass.POSITIVE_ISOTROPIC,
        ControlClass.POSITIVE_ANISOTROPIC,
    }:
        recovered = all(
            (
                operands.exit_code == 0,
                operands.scf_converged,
                operands.phonon_evaluable,
                operands.minimum_phonon_frequency_cm1 >= Decimal("-5"),
                operands.epw_evaluable,
                operands.tc_estimate_K > 0,
                operands.gap_max_meV > 0,
            )
        )
    elif profile.control_class is ControlClass.NORMAL_METAL:
        recovered = all(
            (
                operands.exit_code == 0,
                not operands.timed_out,
                operands.scf_repeat_evaluable,
                operands.scf_converged,
                operands.fermi_evaluable,
            )
        )
    elif profile.control_class is ControlClass.INSULATOR:
        recovered = all(
            (
                operands.exit_code == 0,
                not operands.timed_out,
                operands.scf_repeat_evaluable,
                operands.scf_converged,
                operands.band_gap_eV > 0,
            )
        )
    else:
        recovered = (
            operands.exit_code == 0
            and not operands.timed_out
            and operands.scf_repeat_evaluable
            and operands.phonon_evaluable
            and operands.minimum_phonon_frequency_cm1 < Decimal("-5")
        )
    gauge_covariant_response_evaluable = (
        material_gauge_covariant_response is not None
        and material_gauge_covariant_response.disposition is MaterialGaugeCovariantViewDisposition.PASS
    )
    reasons = list(operands.reason_codes)
    if material_gauge_covariant_response is not None:
        reasons.extend(material_gauge_covariant_response.reason_codes)
    reasons.append(
        "reason.control-material-class-recovered"
        if recovered
        else "reason.control-material-class-not-recovered"
    )
    return MaterialControlControlObservation(
        observation_id=f"observation.{profile.profile_id.removeprefix('workflow.')}",
        structure_id=profile.structure_id,
        control_class=profile.control_class,
        view_id=profile.view_id,
        source_asset_sha256=source_sha,
        input_member_sha256=input_sha,
        requested_preparation_sha256=preparation_sha,
        accepted_preparation_sha256=preparation_sha,
        applied_preparation_sha256=preparation_sha,
        realized_preparation_sha256=preparation_sha,
        requested_realized_match=True,
        exit_code=operands.exit_code,
        timed_out=operands.timed_out,
        terminal_step_id=operands.terminal_step_id,
        wall_time_seconds=operands.wall_time_seconds,
        peak_scratch_bytes=operands.peak_scratch_bytes,
        workflow_completed=operands.exit_code == 0 and not operands.timed_out,
        scf_completed=operands.scf_completed,
        scf_converged=operands.scf_converged,
        scf_repeat_evaluable=operands.scf_repeat_evaluable,
        phonon_evaluable=operands.phonon_evaluable,
        epw_evaluable=operands.epw_evaluable,
        gauge_covariant_response_evaluable=gauge_covariant_response_evaluable,
        metallic=operands.fermi_evaluable,
        dynamically_stable=(
            operands.phonon_evaluable
            and operands.minimum_phonon_frequency_cm1 >= Decimal("-5")
        ),
        positive_superconductor_recovered=(
            recovered
            and profile.control_class
            in {ControlClass.POSITIVE_ISOTROPIC, ControlClass.POSITIVE_ANISOTROPIC}
        ),
        total_energy_Ry=operands.total_energy_Ry,
        scf_accuracy_Ry=operands.scf_accuracy_Ry,
        scf_repeat_residual_Ry=operands.scf_repeat_residual_Ry,
        minimum_phonon_frequency_cm1=operands.minimum_phonon_frequency_cm1,
        electron_phonon_lambda=operands.electron_phonon_lambda,
        tc_estimate_K=operands.tc_estimate_K,
        gap_min_meV=operands.gap_min_meV,
        gap_max_meV=operands.gap_max_meV,
        band_gap_eV=operands.band_gap_eV,
        gauge_covariant_stiffness_lower_per_m2=(
            Decimal("0") if material_gauge_covariant_response is None else material_gauge_covariant_response.stiffness_lower_per_m2
        ),
        gauge_covariant_stiffness_upper_per_m2=(
            Decimal("0") if material_gauge_covariant_response is None else material_gauge_covariant_response.stiffness_upper_per_m2
        ),
        raw_archive_sha256=operands.raw_archive_sha256,
        raw_archive_bytes=operands.raw_archive_bytes,
        reason_codes=tuple(sorted(set(reasons))),
    )


def build_control_panel(
    *, structure_id: str, observations: tuple[MaterialControlControlObservation, ...]
) -> MaterialControlControlPanel:
    ordered = tuple(sorted(observations, key=lambda value: value.view_id))
    if len(ordered) != 2 or any(
        value.structure_id != structure_id for value in ordered
    ):
        raise ValueError('material control panel requires two exact SSSP views of one material')
    control_class = ordered[0].control_class
    if any(value.control_class is not control_class for value in ordered):
        raise ValueError('material control panel mixes control classes')
    if control_class in {
        ControlClass.POSITIVE_ISOTROPIC,
        ControlClass.POSITIVE_ANISOTROPIC,
    }:
        passed = all(
            value.positive_superconductor_recovered and value.gauge_covariant_response_evaluable
            for value in ordered
        )
        passed = passed and all(
            value.workflow_completed and value.scf_repeat_evaluable for value in ordered
        )
        passed = (
            passed and max(value.gauge_covariant_stiffness_lower_per_m2 for value in ordered) > 0
        )
        passed = passed and (
            max(value.gauge_covariant_stiffness_lower_per_m2 for value in ordered)
            <= min(value.gauge_covariant_stiffness_upper_per_m2 for value in ordered)
        )
    elif control_class is ControlClass.NORMAL_METAL:
        passed = all(
            value.workflow_completed
            and value.scf_repeat_evaluable
            and value.scf_converged
            and value.metallic
            for value in ordered
        )
    elif control_class is ControlClass.INSULATOR:
        passed = all(
            value.workflow_completed
            and value.scf_repeat_evaluable
            and value.scf_converged
            and not value.metallic
            and value.band_gap_eV > 0
            for value in ordered
        )
    else:
        passed = all(
            value.workflow_completed
            and value.scf_repeat_evaluable
            and value.phonon_evaluable
            and not value.dynamically_stable
            for value in ordered
        )
    resource_unresolved = any(
        reason
        in {
            "reason.solver-wall-time-exceeded",
            "reason.solver-exit-nonzero",
            "reason.scf-total-energy-missing",
            "reason.scf-accuracy-missing",
            "reason.scf-repeat-operands-missing",
            "reason.phonon-frequency-operands-missing",
            "reason.epw-pairing-operands-missing",
        }
        for value in ordered
        for reason in value.reason_codes
    )
    method_unresolved = control_class in {
        ControlClass.POSITIVE_ISOTROPIC,
        ControlClass.POSITIVE_ANISOTROPIC,
    } and any(
        value.positive_superconductor_recovered and not value.gauge_covariant_response_evaluable
        for value in ordered
    )
    disposition = (
        ControlDisposition.PASS
        if passed
        else (
            ControlDisposition.UNEVALUABLE
            if resource_unresolved or method_unresolved
            else ControlDisposition.FAIL
        )
    )
    return MaterialControlControlPanel(
        panel_id=f"panel.material-control-{structure_id.removeprefix('structure.')}",
        structure_id=structure_id,
        control_class=control_class,
        observations=ordered,
        physical_independent_unit_count=1,
        nested_numerical_view_count=2,
        disposition=disposition,
        reason_codes=(
            "reason.control-panel-intersection-pass"
            if passed
            else (
                "reason.control-panel-intersection-unevaluable"
                if disposition is ControlDisposition.UNEVALUABLE
                else "reason.control-panel-intersection-fail"
            ),
        ),
    )


def _round_up_three_significant(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal("0")
    exponent = value.adjusted() - 2
    quantum = Decimal(1).scaleb(exponent)
    return value.quantize(quantum, rounding=ROUND_CEILING)


def _relative_distance(first: Decimal, second: Decimal) -> Decimal:
    scale = max(abs(first), abs(second))
    return Decimal("0") if scale == 0 else abs(first - second) / scale


def build_calibration_freeze(
    *,
    panels: tuple[MaterialControlControlPanel, ...],
    material_results: tuple[MultibandStrongCouplingMaterialViewResult, ...],
    gauge_covariant_observations: tuple[GaugeCovariantResponseResponseObservation, ...],
) -> MaterialControlCalibrationFreeze:
    ordered = tuple(sorted(panels, key=lambda value: value.structure_id))
    if len(ordered) != 5 or any(
        value.disposition is not ControlDisposition.PASS for value in ordered
    ):
        raise ValueError('material control calibration requires five passed calibration panels')
    observations = tuple(value for panel in ordered for value in panel.observations)
    numerical_noise = _round_up_three_significant(
        max(
            max(value.scf_accuracy_Ry, value.scf_repeat_residual_Ry)
            for value in observations
        )
    )
    disagreements: list[Decimal] = []
    for panel in ordered:
        first, second = panel.observations
        fields = {
            ControlClass.POSITIVE_ISOTROPIC: (
                "electron_phonon_lambda",
                "tc_estimate_K",
                "gap_max_meV",
            ),
            ControlClass.POSITIVE_ANISOTROPIC: (
                "electron_phonon_lambda",
                "tc_estimate_K",
                "gap_max_meV",
            ),
            # Different pseudopotentials do not share an absolute energy zero.
            # Treat the normal-metal cross-view result categorically instead
            # of comparing pseudopotential-dependent total energies.
            ControlClass.NORMAL_METAL: (),
            ControlClass.INSULATOR: ("band_gap_eV",),
            ControlClass.DYNAMICALLY_UNSTABLE: ("minimum_phonon_frequency_cm1",),
        }[panel.control_class]
        disagreements.extend(
            _relative_distance(getattr(first, field_name), getattr(second, field_name))
            for field_name in fields
        )
    for structure_id in sorted({value.structure_id for value in material_results}):
        views = tuple(
            sorted(
                (
                    value
                    for value in material_results
                    if value.structure_id == structure_id
                ),
                key=lambda value: value.view_id,
            )
        )
        if len(views) != 2:
            raise ValueError(
                'material control material cross-view calibration requires paired views'
            )
        disagreements.append(
            _relative_distance(
                views[0].inverse_penetration_depth_sq_per_m2,
                views[1].inverse_penetration_depth_sq_per_m2,
            )
        )
    cross_view = _round_up_three_significant(max(disagreements, default=Decimal("0")))
    stable_controls = tuple(
        value
        for panel in ordered
        if panel.control_class
        in {ControlClass.POSITIVE_ISOTROPIC, ControlClass.POSITIVE_ANISOTROPIC}
        for value in panel.observations
    )
    phonon_tolerance = _round_up_three_significant(
        max(
            (
                max(Decimal("0"), -value.minimum_phonon_frequency_cm1)
                for value in stable_controls
            ),
            default=Decimal("0"),
        )
    )
    nonsuperconducting = tuple(
        value
        for value in gauge_covariant_observations
        if value.fixture_id in {'fixture.gauge-covariant-response-insulator', 'fixture.gauge-covariant-response-normal'}
    )
    if len(nonsuperconducting) != 2:
        raise ValueError(
            'material control receiver calibration lacks normal and insulating gauge covariant response fixtures'
        )
    receiver_tolerance = _round_up_three_significant(
        max(
            max(
                abs(value.normal_total_eV_per_link),
                abs(value.stiffness_base_eV_per_link),
                abs(value.stiffness_refined_eV_per_link),
            )
            for value in nonsuperconducting
        )
    )
    receiver_identity_limit = _round_up_three_significant(
        max(
            (
                *(value.ward_absolute_residual_eV for value in gauge_covariant_observations),
                *(
                    value.gauge_covariance_residual_eV_per_cell
                    for value in material_results
                ),
            ),
            default=Decimal("0"),
        )
    )
    finite_q_limit = _round_up_three_significant(
        max(
            (
                value.finite_q_fit_residual_eV_per_cell
                / max(value.intercept_eV_per_cell, Decimal("1e-99"))
                for value in material_results
            ),
            default=Decimal("0"),
        )
    )
    return MaterialControlCalibrationFreeze(
        freeze_id='freeze.ambient-pressure-superconductor-material-control-calibration-calibration',
        control_panel_sha256s=tuple(sorted(value.fingerprint() for value in ordered)),
        numerical_noise_Ry=numerical_noise,
        cross_view_relative_limit=cross_view,
        phonon_stability_tolerance_cm1=phonon_tolerance,
        receiver_tolerance=receiver_tolerance,
        receiver_identity_residual_limit_eV_per_cell=receiver_identity_limit,
        finite_q_fit_relative_limit=finite_q_limit,
        rounding_rule_ids=tuple(
            sorted(
                (
                    "rounding.outward-three-significant-digits",
                    "rounding.upward-three-significant-digits",
                )
            )
        ),
        target_outcomes_used=False,
        all_predeclared_fields_resolved=True,
        reason_codes=tuple(
            sorted(
                (
                    'reason.calibration-derived-from-calibration-and-frozen-gauge-covariant-response-fixtures-only',
                    "reason.low-temperature-controls-do-not-calibrate-300k-positive-threshold",
                )
            )
        ),
    )


def build_material_control_closeout(
    *,
    config: MaterialControlConfig,
    control_bundle_sha256: str,
    bridge: MultibandStrongCouplingBridgeQualification,
    gauge_covariant_conformance: GaugeCovariantResponseConformanceResult,
    gauge_covariant_observations: tuple[GaugeCovariantResponseResponseObservation, ...],
    tutorials: tuple[MaterialControlTutorialReproduction, ...],
    material_results: tuple[MultibandStrongCouplingMaterialViewResult, ...],
    panels: tuple[MaterialControlControlPanel, ...],
    constructive_search: MaterialControlTruthWorldConformance,
) -> MaterialControlCloseout:
    ordered_tutorials = tuple(sorted(tutorials, key=lambda value: value.structure_id))
    ordered_material = tuple(
        sorted(material_results, key=lambda value: (value.structure_id, value.view_id))
    )
    ordered_panels = tuple(sorted(panels, key=lambda value: value.structure_id))
    if (
        len(ordered_tutorials) != 2
        or len(ordered_material) != 4
        or len(ordered_panels) != 5
    ):
        raise ValueError('material control closeout received an incomplete calibration evidence roster')
    resource_failure = any(
        "wall-time" in reason or "solver-exit" in reason
        for tutorial in ordered_tutorials
        for reason in tutorial.reason_codes
    ) or any(
        "wall-time" in reason or "solver-exit" in reason
        for panel in ordered_panels
        for observation in panel.observations
        for reason in observation.reason_codes
    )
    method_pass = (
        bridge.disposition is BridgeDisposition.CONDITIONAL_METHOD_PASS
        and gauge_covariant_conformance.strict_method_pass
        and gauge_covariant_conformance.fingerprint() == config.gauge_covariant_conformance_sha256
        and constructive_search.disposition_id == 'constructive-search.exploration-conformance-pass'
    )
    controls_pass = all(
        value.disposition is ControlDisposition.PASS for value in ordered_panels
    ) and all(value.pairing_recovered for value in ordered_tutorials)
    material_pass = all(
        value.disposition is MaterialGaugeCovariantViewDisposition.PASS
        for value in ordered_material
    )
    passed = method_pass and controls_pass and material_pass and not resource_failure
    calibration: MaterialControlCalibrationFreeze | None = None
    science: MaterialControlScienceFreeze | None = None
    if passed:
        calibration = build_calibration_freeze(
            panels=ordered_panels,
            material_results=ordered_material,
            gauge_covariant_observations=gauge_covariant_observations,
        )
        science = MaterialControlScienceFreeze(
            freeze_id='freeze.ambient-pressure-superconductor-material-control-science',
            config_sha256=config.payload_sha256,
            gauge_covariant_development_freeze_sha256=config.gauge_covariant_development_freeze_sha256,
            control_bundle_sha256=control_bundle_sha256,
            bridge_qualification_sha256=bridge.fingerprint(),
            gauge_covariant_conformance_sha256=gauge_covariant_conformance.fingerprint(),
            tutorial_reproduction_sha256s=tuple(
                sorted(value.fingerprint() for value in ordered_tutorials)
            ),
            material_gauge_covariant_result_sha256s=tuple(
                sorted(value.fingerprint() for value in ordered_material)
            ),
            control_panel_sha256s=tuple(
                sorted(value.fingerprint() for value in ordered_panels)
            ),
            calibration_freeze_sha256=calibration.fingerprint(),
            constructive_search_conformance_sha256=constructive_search.fingerprint(),
            development_roster_sha256=config.predecessors.development_roster_sha256,
            development_source_sha256=config.predecessors.development_source_sha256,
            resource_envelope_sha256=sha256(
                canonical_json_bytes(config.resources)
            ).hexdigest(),
            target_contact_count=0,
            calibration_control_count=5,
            all_controls_passed=True,
            constructive_search_passed=True,
            science_frozen_before_development_contact=True,
            reason_codes=tuple(
                sorted(
                    (
                        'reason.material-control-five-calibration-control-panels-passed',
                        'reason.material-control-low-temperature-controls-qualify-method-not-300k-material',
                        'reason.material-control-science-frozen-before-development-atlas-contact',
                        'reason.material-control-constructive-search-six-world-conformance-passed',
                    )
                )
            ),
        )
    if passed:
        disposition = MaterialControlDisposition.PASS
    elif resource_failure:
        disposition = MaterialControlDisposition.RESOURCE_STOP
    elif not method_pass or any(
        value.disposition is ControlDisposition.UNEVALUABLE for value in ordered_panels
    ):
        disposition = MaterialControlDisposition.METHOD_FAIL
    else:
        disposition = MaterialControlDisposition.CONTROL_FAIL
    reason_codes = [
        {
            MaterialControlDisposition.PASS: 'reason.material-control-noncompensating-intersection-passed',
            MaterialControlDisposition.CONTROL_FAIL: 'reason.material-control-control-recovery-failed',
            MaterialControlDisposition.SOURCE_FAIL: 'reason.material-control-source-custody-failed',
            MaterialControlDisposition.METHOD_FAIL: 'reason.material-control-method-or-validity-failed',
            MaterialControlDisposition.RESOURCE_STOP: 'reason.material-control-resource-path-inadequate',
        }[disposition],
        'reason.material-control-target-contact-count-zero',
    ]
    stage = MaterialControlStageResult(
        result_id='result.ambient-pressure-superconductor-material-control-control-science-freeze',
        disposition=disposition,
        attempted=True,
        evaluable=disposition
        in {
            MaterialControlDisposition.PASS,
            MaterialControlDisposition.CONTROL_FAIL,
            MaterialControlDisposition.METHOD_FAIL,
        },
        science_freeze_sha256="" if science is None else science.fingerprint(),
        calibration_control_count=5,
        development_material_count=0,
        prospective_material_count=0,
        target_contact_count=0,
        reason_codes=tuple(sorted(reason_codes)),
    )
    return MaterialControlCloseout(
        closeout_id='closeout.ambient-pressure-superconductor-material-control-control-science-freeze',
        control_bundle_sha256=control_bundle_sha256,
        bridge_qualification_sha256=bridge.fingerprint(),
        gauge_covariant_conformance_sha256=gauge_covariant_conformance.fingerprint(),
        tutorial_reproduction_sha256s=tuple(
            sorted(value.fingerprint() for value in ordered_tutorials)
        ),
        material_gauge_covariant_result_sha256s=tuple(
            sorted(value.fingerprint() for value in ordered_material)
        ),
        control_panel_sha256s=tuple(
            sorted(value.fingerprint() for value in ordered_panels)
        ),
        constructive_search_conformance_sha256=constructive_search.fingerprint(),
        calibration_freeze=calibration,
        science_freeze=science,
        stage_result=stage,
        target_contact_count=0,
        reason_codes=stage.reason_codes,
    )


__all__ = [
    "build_control_observation",
    "build_control_panel",
    'build_material_control_closeout',
    "build_calibration_freeze",
    "build_tutorial_reproduction",
    "profile_source_identity",
]
