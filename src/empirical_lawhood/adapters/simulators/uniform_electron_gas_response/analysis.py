"""Truth-blind transverse-law identification and noncompensating admission."""

from __future__ import annotations

from decimal import Decimal
from itertools import combinations

from .contracts import AdmissionGateResult, BranchInput, BranchLawResult, BranchScreenResult, ClosureDiagnosticInput, EvidenceLane, FiniteQEstimate, GateName, UniformElectronGasMeasurementClass, UniformElectronGasObservationOrderClass, UniformElectronGasResponseClass, UniformElectronGasLawQualificationClass, UniformElectronGasAdmissionClass, UniformElectronGasTransverseScreenConfig, TransversePanel, ViewLawResult, ViewScreenResult
from .physics import finite_q_estimate, maximum_field, penetration_depth_from_kernel, q_intercept_stability, relative_vector_difference, row_map, shielding_score, transverse_relative_dot, ueg_scales, vector_potential_from_u, vector_scale


def _sorted_reasons(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted({reason for group in groups for reason in group}))


def _panel_contract_reasons(panel: TransversePanel, config: UniformElectronGasTransverseScreenConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        panel.r_s != config.r_s
        or panel.temperature_K != config.temperature_K
        or panel.q_direction != config.q_direction
        or panel.action_direction != config.action_direction
        or panel.view_id not in config.views
        or not panel.source_semantics_valid
    ):
        reasons.append("panel-denominator-or-semantics-mismatch")
    expected_keys = {
        (q_value, u_value) for q_value in config.q_over_kf for u_value in config.u_values
    }
    rows = row_map(panel.rows)
    if set(rows) != expected_keys:
        reasons.append("panel-action-roster-incomplete")
        return tuple(sorted(set(reasons)))
    controls = {value.q_over_kf: value for value in panel.controls}
    if len(controls) != len(panel.controls) or set(controls) != set(config.q_over_kf):
        reasons.append("panel-control-roster-incomplete")
    scales = ueg_scales(config.r_s)
    action_floor = abs(vector_potential_from_u(config.u0, scales)) * Decimal("1e-18")
    for key, row in rows.items():
        q_over_kf, u_value = key
        expected_amplitude = vector_potential_from_u(u_value, scales)
        expected_vector = vector_scale(config.action_direction, expected_amplitude)
        if (
            row.q_over_kf != q_over_kf
            or row.u != u_value
            or row.requested_clock != config.requested_clock
            or row.accepted_clock != config.accepted_clock
            or row.applied_clock != config.applied_clock
            or row.receiver_clock != config.receiver_clock
            or not (
                row.requested_clock < row.accepted_clock < row.applied_clock < row.receiver_clock
            )
            or not row.accepted
            or not row.valid
        ):
            reasons.append("action-or-clock-contract-invalid")
            continue
        for stage_vector in (
            row.requested_A_T,
            row.accepted_A_T,
            row.applied_A_T,
            row.realized_A_T,
        ):
            if relative_vector_difference(
                stage_vector,
                expected_vector,
                floor=action_floor,
            ) > config.threshold("action_realization_relative"):
                reasons.append("action-realization-mismatch")
        q_vector = vector_scale(config.q_direction, q_over_kf * scales.k_f_m1)
        if transverse_relative_dot(q_vector, row.realized_A_T) > config.threshold(
            "transverse_dot_relative"
        ):
            reasons.append("action-not-transverse")
        if transverse_relative_dot(q_vector, row.current_density) > config.threshold(
            "transverse_dot_relative"
        ):
            reasons.append("current-not-transverse")
    return tuple(sorted(set(reasons)))


def _normal_cancellation(panel: TransversePanel) -> Decimal:
    values = []
    for control in panel.controls:
        denominator = max(abs(control.diamagnetic_normal), Decimal("1e-300"))
        values.append(abs(control.diamagnetic_normal + control.paramagnetic_normal) / denominator)
    return max(values, default=Decimal("Infinity"))


def _ward_max(panel: TransversePanel) -> Decimal:
    return max((value.ward_residual for value in panel.controls), default=Decimal("Infinity"))


def _order_class(panel: TransversePanel, config: UniformElectronGasTransverseScreenConfig) -> UniformElectronGasObservationOrderClass:
    if panel.order_zero is None:
        return UniformElectronGasObservationOrderClass.ORDER_OPERAND_REQUIRED
    order_floor = config.threshold("response_floor_relative_to_diamagnetic")
    if panel.order_zero <= order_floor:
        return UniformElectronGasObservationOrderClass.NO_ORDER_OPPORTUNITY
    return UniformElectronGasObservationOrderClass.ORDER_OPPORTUNITY_SUPPORTED


def _view_result(panel: TransversePanel, config: UniformElectronGasTransverseScreenConfig) -> ViewLawResult:
    contract_reasons = list(_panel_contract_reasons(panel, config))
    measurement_class = UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL if not contract_reasons else UniformElectronGasMeasurementClass.PARTIAL_ROLE_LIMITED
    observation_order_class = _order_class(panel, config)
    normal_residual = _normal_cancellation(panel)
    ward_residual = _ward_max(panel)
    scales = ueg_scales(config.r_s)
    max_field = maximum_field(panel.rows, scales)
    estimates: tuple[FiniteQEstimate, ...] = ()
    kernel0 = None
    kernel0_error = None
    kernel0_lower = None
    c2 = None
    q_stability = None
    order_preservation = None
    reasons = list(contract_reasons)

    if panel.order_zero is not None and panel.order_at_max_action is not None:
        if panel.order_zero == 0:
            order_preservation = Decimal("1") if panel.order_at_max_action == 0 else Decimal("0")
        else:
            order_preservation = panel.order_at_max_action / panel.order_zero

    if measurement_class is not UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL:
        response_class = UniformElectronGasResponseClass.UNEVALUABLE
        law_qualification_class = UniformElectronGasLawQualificationClass.UNEVALUABLE
        reasons.append("response-prerequisite-measurement")
    else:
        rows = row_map(panel.rows)
        estimates = tuple(
            finite_q_estimate(
                panel_id=panel.panel_id,
                q_over_kf=q_value,
                rows=rows,
                config=config,
                action_direction=panel.action_direction,
            )
            for q_value in config.q_over_kf
        )
        controls_converged = all(
            value.converged and value.limit_order_supported for value in panel.controls
        )
        a_max = abs(vector_potential_from_u(config.u0, scales))
        diamagnetic = max(
            (abs(value.diamagnetic_normal) for value in panel.controls),
            default=scales.diamagnetic_kernel_A_T_m3,
        )
        kernel_floor = config.threshold("response_floor_relative_to_diamagnetic") * diamagnetic
        current_floor = kernel_floor * a_max
        wrong_polarization = max(
            (abs(value.wrong_polarization_current_A_m2) for value in panel.controls),
            default=Decimal("Infinity"),
        )
        if (
            normal_residual > config.threshold("normal_cancellation_relative")
            or ward_residual > config.threshold("ward_residual")
            or wrong_polarization > current_floor
            or not controls_converged
        ):
            response_class = UniformElectronGasResponseClass.CONTROL_FAILED
            if normal_residual > config.threshold("normal_cancellation_relative"):
                reasons.append("normal-cancellation-failed")
            if ward_residual > config.threshold("ward_residual"):
                reasons.append("ward-conformance-failed")
            if wrong_polarization > current_floor:
                reasons.append("wrong-polarization-control-failed")
            if not controls_converged:
                reasons.append("static-convergence-or-limit-failed")
        elif all(abs(value.kernel_full) <= kernel_floor for value in estimates):
            response_class = UniformElectronGasResponseClass.RESPONSE_BELOW_FLOOR
            reasons.append("finite-response-below-floor")
        elif any(value.kernel_full <= kernel_floor for value in estimates):
            response_class = UniformElectronGasResponseClass.WRONG_SIGN
            reasons.append("diamagnetic-direction-failed")
        else:
            response_class = UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE

        if response_class is not UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE:
            law_qualification_class = UniformElectronGasLawQualificationClass.NOT_SUPPORTED
        elif not all(
            value.locality_pass and value.even_remainder_pass and value.zero_offset_pass
            for value in estimates
        ):
            law_qualification_class = UniformElectronGasLawQualificationClass.NONLINEAR_BOUNDARY_LIMITED
            if not all(value.zero_offset_pass for value in estimates):
                reasons.append("zero-action-offset-failed")
            if not all(value.locality_pass and value.even_remainder_pass for value in estimates):
                reasons.append("amplitude-locality-or-even-remainder-failed")
        else:
            all_fit, _small_fit, q_stability = q_intercept_stability(
                estimates, kernel_floor=kernel_floor
            )
            kernel0 = all_fit.kernel0
            kernel0_error = all_fit.error_bound
            kernel0_lower = all_fit.kernel0 - all_fit.error_bound
            c2 = all_fit.c2
            if q_stability > config.threshold("q_intercept_stability_relative"):
                law_qualification_class = UniformElectronGasLawQualificationClass.FINITE_Q_ONLY
                reasons.append("q-intercept-unstable")
            else:
                law_qualification_class = UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED

    return ViewLawResult(
        result_id=f"view-law-{panel.panel_id}",
        panel_id=panel.panel_id,
        view_id=panel.view_id,
        measurement_class=measurement_class,
        observation_order_class=observation_order_class,
        response_class=response_class,
        law_qualification_class=law_qualification_class,
        finite_q=estimates,
        kernel0=kernel0,
        kernel0_error_bound=kernel0_error,
        kernel0_lower_bound=kernel0_lower,
        c2=c2,
        q_intercept_stability_relative=q_stability,
        normal_cancellation_max_relative=normal_residual,
        ward_residual_max=ward_residual,
        maximum_field_T=max_field,
        order_preservation_fraction=order_preservation,
        reason_codes=tuple(sorted(set(reasons))),
    )


def _aggregate_source_qualification(values: tuple[ViewLawResult, ...]) -> UniformElectronGasMeasurementClass:
    return (
        UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL
        if all(value.measurement_class is UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL for value in values)
        else UniformElectronGasMeasurementClass.PARTIAL_ROLE_LIMITED
    )


def _aggregate_observation(values: tuple[ViewLawResult, ...]) -> UniformElectronGasObservationOrderClass:
    classes = {value.observation_order_class for value in values}
    if classes == {UniformElectronGasObservationOrderClass.ORDER_OPPORTUNITY_SUPPORTED}:
        return UniformElectronGasObservationOrderClass.ORDER_OPPORTUNITY_SUPPORTED
    if classes == {UniformElectronGasObservationOrderClass.NO_ORDER_OPPORTUNITY}:
        return UniformElectronGasObservationOrderClass.NO_ORDER_OPPORTUNITY
    if UniformElectronGasObservationOrderClass.ORDER_OPERAND_REQUIRED in classes:
        return UniformElectronGasObservationOrderClass.ORDER_OPERAND_REQUIRED
    return UniformElectronGasObservationOrderClass.VIEW_OPPOSED


def _aggregate_response_identification(values: tuple[ViewLawResult, ...]) -> UniformElectronGasResponseClass:
    classes = {value.response_class for value in values}
    for decisive in (
        UniformElectronGasResponseClass.UNEVALUABLE,
        UniformElectronGasResponseClass.CONTROL_FAILED,
        UniformElectronGasResponseClass.WRONG_SIGN,
        UniformElectronGasResponseClass.RESPONSE_BELOW_FLOOR,
        UniformElectronGasResponseClass.PARTIAL,
    ):
        if decisive in classes:
            return decisive
    return UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE


def _view_agreement(values: tuple[ViewLawResult, ...]) -> Decimal | None:
    kernels = tuple(value.kernel0 for value in values if value.kernel0 is not None)
    if len(kernels) != len(values) or len(kernels) < 2:
        return None
    return max(
        (
            abs(left - right) / max(abs(left), abs(right), Decimal("1e-300"))
            for left, right in combinations(kernels, 2)
        ),
        default=Decimal("0"),
    )


def _aggregate_law_qualification(
    values: tuple[ViewLawResult, ...], *, view_agreement: Decimal | None, config: UniformElectronGasTransverseScreenConfig
) -> UniformElectronGasLawQualificationClass:
    classes = {value.law_qualification_class for value in values}
    if UniformElectronGasLawQualificationClass.UNEVALUABLE in classes:
        return UniformElectronGasLawQualificationClass.UNEVALUABLE
    if UniformElectronGasLawQualificationClass.NONLINEAR_BOUNDARY_LIMITED in classes:
        return UniformElectronGasLawQualificationClass.NONLINEAR_BOUNDARY_LIMITED
    if UniformElectronGasLawQualificationClass.FINITE_Q_ONLY in classes:
        return UniformElectronGasLawQualificationClass.FINITE_Q_ONLY
    if classes != {UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED}:
        return UniformElectronGasLawQualificationClass.NOT_SUPPORTED
    if view_agreement is None or view_agreement > config.threshold("view_k0_agreement_relative"):
        return UniformElectronGasLawQualificationClass.VIEW_LOCAL_ONLY
    return UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED


def _gate(
    gate: GateName, *, passed: bool, margin: Decimal | None, reason: str
) -> AdmissionGateResult:
    return AdmissionGateResult(gate=gate, passed=passed, margin=margin, reason_code=reason)


def identify_branch(branch: BranchInput, config: UniformElectronGasTransverseScreenConfig) -> BranchLawResult:
    """Identify measurement, observation/order opportunity, transverse response and local law qualification without constructing receiver admission."""

    view_results = tuple(
        sorted(
            (_view_result(panel, config) for panel in branch.panels),
            key=lambda value: value.result_id,
        )
    )
    branch_reasons: list[str] = []
    view_ids = tuple(sorted(panel.view_id for panel in branch.panels))
    if view_ids != config.views:
        branch_reasons.append("branch-view-roster-incomplete")
    if (
        len({panel.branch_id for panel in branch.panels}) != 1
        or len({panel.source_id for panel in branch.panels}) != 1
    ):
        branch_reasons.append("branch-source-or-denominator-mismatch")
    if any(not panel.deterministic for panel in branch.panels):
        # The entered truth-known conformance act has only producer-certified deterministic bounds.
        # Refuse stochastic rows rather than treating nested q/action/view rows
        # as replication; a stochastic follow-up needs an independent-unit
        # batch schema and registered estimator.
        branch_reasons.append("stochastic-independent-unit-estimator-required")
    elif len({panel.independent_unit_id for panel in branch.panels}) != 1:
        branch_reasons.append("deterministic-views-must-share-acquisition-unit")
    measurement_class = _aggregate_source_qualification(view_results) if not branch_reasons else UniformElectronGasMeasurementClass.PARTIAL_ROLE_LIMITED
    observation_order_class = _aggregate_observation(view_results)
    response_class = (
        _aggregate_response_identification(view_results) if measurement_class is UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL else UniformElectronGasResponseClass.UNEVALUABLE
    )
    agreement = _view_agreement(view_results)
    law_qualification_class = (
        _aggregate_law_qualification(view_results, view_agreement=agreement, config=config)
        if measurement_class is UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL
        else UniformElectronGasLawQualificationClass.UNEVALUABLE
    )
    lower_bounds = tuple(
        value.kernel0_lower_bound for value in view_results if value.kernel0_lower_bound is not None
    )
    robust_lower = min(lower_bounds) if len(lower_bounds) == len(view_results) else None
    reasons = _sorted_reasons(*(list(value.reason_codes) for value in view_results), branch_reasons)
    return BranchLawResult(
        result_id=f"law-{branch.case_id}",
        input_id=branch.input_id,
        case_id=branch.case_id,
        measurement_class=measurement_class,
        observation_order_class=observation_order_class,
        response_class=response_class,
        law_qualification_class=law_qualification_class,
        view_results=view_results,
        robust_kernel0_lower_bound=robust_lower,
        view_k0_agreement_relative=agreement,
        reason_codes=reasons,
    )


def adjudicate_branch(
    branch: BranchInput, law: BranchLawResult, config: UniformElectronGasTransverseScreenConfig
) -> BranchScreenResult:
    """Adjudicate every view, then form the exact robust intersection."""

    expected_law = identify_branch(branch, config)
    if law != expected_law:
        raise ValueError("law result differs from the frozen method output for its input")

    panel_by_view = {panel.view_id: panel for panel in branch.panels}
    law_by_view = {value.view_id: value for value in law.view_results}
    if set(panel_by_view) != set(law_by_view):
        raise ValueError("law result and branch input views differ")

    def screen_view(panel: TransversePanel, value: ViewLawResult) -> ViewScreenResult:
        penetration = None
        shielding = None
        lower = value.kernel0_lower_bound
        if lower is not None and lower > 0:
            penetration = penetration_depth_from_kernel(lower)
            shielding = shielding_score(config.slab_thickness_m, penetration)
        target_pass = (
            value.law_qualification_class is UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED
            and shielding is not None
            and shielding >= config.threshold("shielding_score_minimum")
        )
        target_margin = (
            None if shielding is None else shielding - config.threshold("shielding_score_minimum")
        )
        preservation = value.order_preservation_fraction
        sink_pass = (
            value.observation_order_class is UniformElectronGasObservationOrderClass.ORDER_OPPORTUNITY_SUPPORTED
            and preservation is not None
            and preservation >= config.threshold("order_preservation_fraction")
        )
        sink_reason = (
            "order-preserved"
            if sink_pass
            else (
                "preservation-operand-required"
                if preservation is None
                else "order-preservation-failed"
            )
        )
        effort_pass = value.maximum_field_T <= config.field_ceiling_T
        observation_pass = (
            value.measurement_class is UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL
            and value.response_class is UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE
        )
        observation_margin = min(
            config.threshold("normal_cancellation_relative")
            - value.normal_cancellation_max_relative,
            config.threshold("ward_residual") - value.ward_residual_max,
        )
        uncertainty_pass = lower is not None and lower > 0
        baseline_pass = panel.baseline_preserved is True
        dynamics_pass = all(
            control.converged and control.limit_order_supported for control in panel.controls
        )
        reachability_pass = value.measurement_class is UniformElectronGasMeasurementClass.TYPED_TRANSVERSE_PANEL
        gates = tuple(
            sorted(
                (
                    _gate(
                        GateName.AUTHORITY,
                        passed=branch.authority_valid,
                        margin=Decimal("1") if branch.authority_valid else Decimal("-1"),
                        reason=(
                            "authority-valid" if branch.authority_valid else "authority-required"
                        ),
                    ),
                    _gate(
                        GateName.BASELINE_PRESERVATION,
                        passed=baseline_pass,
                        margin=Decimal("1") if baseline_pass else Decimal("-1"),
                        reason=(
                            "baseline-preserved"
                            if baseline_pass
                            else "baseline-preservation-failed"
                        ),
                    ),
                    _gate(
                        GateName.DYNAMICS,
                        passed=dynamics_pass,
                        margin=Decimal("1") if dynamics_pass else Decimal("-1"),
                        reason=(
                            "static-limit-supported"
                            if dynamics_pass
                            else "static-limit-unsupported"
                        ),
                    ),
                    _gate(
                        GateName.EFFORT,
                        passed=effort_pass,
                        margin=config.field_ceiling_T - value.maximum_field_T,
                        reason=("weak-field-pass" if effort_pass else "weak-field-failed"),
                    ),
                    _gate(
                        GateName.OBSERVATION_VALIDITY,
                        passed=observation_pass,
                        margin=observation_margin,
                        reason=(
                            "observation-valid"
                            if observation_pass
                            else "observation-validity-failed"
                        ),
                    ),
                    _gate(
                        GateName.PHYSICAL_SINK,
                        passed=sink_pass,
                        margin=(
                            None
                            if preservation is None
                            else preservation - config.threshold("order_preservation_fraction")
                        ),
                        reason=sink_reason,
                    ),
                    _gate(
                        GateName.REACHABILITY,
                        passed=reachability_pass,
                        margin=(Decimal("1") if reachability_pass else Decimal("-1")),
                        reason=(
                            "actions-realized-in-support"
                            if reachability_pass
                            else "action-realization-failed"
                        ),
                    ),
                    _gate(
                        GateName.TARGET,
                        passed=target_pass,
                        margin=target_margin,
                        reason=(
                            "shielding-target-pass" if target_pass else "shielding-target-failed"
                        ),
                    ),
                    _gate(
                        GateName.UNCERTAINTY,
                        passed=uncertainty_pass,
                        margin=lower,
                        reason=(
                            "positive-deterministic-bound"
                            if uncertainty_pass
                            else "nonpositive-or-missing-bound"
                        ),
                    ),
                ),
                key=lambda gate: gate.gate.value,
            )
        )
        if all(gate.passed for gate in gates):
            admission_class = UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED
        elif sink_reason == "preservation-operand-required":
            admission_class = UniformElectronGasAdmissionClass.PRESERVATION_OPERAND_REQUIRED
        elif value.law_qualification_class is not UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED:
            admission_class = UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW
        elif not branch.authority_valid:
            admission_class = UniformElectronGasAdmissionClass.AUTHORITY_REQUIRED
        else:
            admission_class = UniformElectronGasAdmissionClass.EMPTY_HOLD
        return ViewScreenResult(
            result_id=f"view-screen-{panel.panel_id}",
            panel_id=panel.panel_id,
            view_id=panel.view_id,
            admission_class=admission_class,
            gates=gates,
            kernel0_lower_bound=lower,
            penetration_depth_upper_m=penetration,
            shielding_score_lower=shielding,
            hold=admission_class is not UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED,
            reason_codes=tuple(sorted(gate.reason_code for gate in gates if not gate.passed)),
        )

    view_screens = tuple(
        screen_view(panel_by_view[view_id], law_by_view[view_id])
        for view_id in sorted(panel_by_view)
    )
    agreement = law.view_k0_agreement_relative
    gates_list: list[AdmissionGateResult] = []
    pass_reasons = {
        GateName.AUTHORITY: "authority-valid",
        GateName.BASELINE_PRESERVATION: "baseline-preserved",
        GateName.DYNAMICS: "static-limit-supported",
        GateName.EFFORT: "weak-field-pass",
        GateName.OBSERVATION_VALIDITY: "observation-valid",
        GateName.PHYSICAL_SINK: "order-preserved",
        GateName.REACHABILITY: "actions-realized-in-support",
        GateName.TARGET: "shielding-target-pass",
        GateName.UNCERTAINTY: "view-robust-positive-bound",
    }
    for gate_name in GateName:
        members = tuple(
            next(gate for gate in value.gates if gate.gate is gate_name) for value in view_screens
        )
        passed = bool(members) and all(value.passed for value in members)
        margins = tuple(value.margin for value in members if value.margin is not None)
        margin = min(margins) if len(margins) == len(members) and margins else None
        if gate_name is GateName.UNCERTAINTY:
            agreement_margin = (
                None
                if agreement is None
                else config.threshold("view_k0_agreement_relative") - agreement
            )
            passed = passed and agreement_margin is not None and agreement_margin >= 0
            if agreement_margin is not None:
                margin = agreement_margin if margin is None else min(margin, agreement_margin)
            failure_reason = "view-or-bound-uncertainty-failed"
        else:
            failed = tuple(value for value in members if not value.passed)
            failure_reason = failed[0].reason_code if failed else f"{gate_name.value}-failed"
        gates_list.append(
            _gate(
                gate_name,
                passed=passed,
                margin=margin,
                reason=pass_reasons[gate_name] if passed else failure_reason,
            )
        )
    gates = tuple(gates_list)
    robust_lower = law.robust_kernel0_lower_bound
    penetration = None
    shielding = None
    if robust_lower is not None and robust_lower > 0:
        penetration = penetration_depth_from_kernel(robust_lower)
        shielding = shielding_score(config.slab_thickness_m, penetration)

    if law.law_qualification_class is UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED and all(value.passed for value in gates):
        admission_class = UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED
    elif any(value.admission_class is UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED for value in view_screens):
        admission_class = UniformElectronGasAdmissionClass.BRANCH_LOCAL_ONLY
    elif any(value.admission_class is UniformElectronGasAdmissionClass.PRESERVATION_OPERAND_REQUIRED for value in view_screens):
        admission_class = UniformElectronGasAdmissionClass.PRESERVATION_OPERAND_REQUIRED
    elif law.law_qualification_class is not UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED:
        admission_class = UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW
    elif not branch.authority_valid:
        admission_class = UniformElectronGasAdmissionClass.AUTHORITY_REQUIRED
    else:
        admission_class = UniformElectronGasAdmissionClass.EMPTY_HOLD
    reasons = _sorted_reasons(
        list(law.reason_codes),
        [value.reason_code for value in gates if not value.passed],
    )
    return BranchScreenResult(
        result_id=f"screen-{branch.case_id}",
        input_id=branch.input_id,
        case_id=branch.case_id,
        evidence_lane=EvidenceLane.GAUGE_CLOSED_TRANSVERSE,
        measurement_class=law.measurement_class,
        observation_order_class=law.observation_order_class,
        response_class=law.response_class,
        law_qualification_class=law.law_qualification_class,
        admission_class=admission_class,
        view_results=law.view_results,
        view_screens=view_screens,
        gates=gates,
        robust_kernel0_lower_bound=robust_lower,
        penetration_depth_upper_m=penetration,
        shielding_score_lower=shielding,
        hold=admission_class is not UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED,
        reason_codes=reasons,
    )


def analyze_branch(branch: BranchInput, config: UniformElectronGasTransverseScreenConfig) -> BranchScreenResult:
    """Compose the separately registered measurement, observation/order opportunity, response, law qualification and admission acts."""

    return adjudicate_branch(branch, identify_branch(branch, config), config)


def evaluate_closure_only(value: ClosureDiagnosticInput, config: UniformElectronGasTransverseScreenConfig) -> BranchScreenResult:
    """Classify a supplied Tc closure without ever constructing stiffness."""

    del config
    gates = tuple(
        _gate(
            gate,
            passed=False,
            margin=None,
            reason=(
                "transverse-current-operand-required"
                if gate is not GateName.AUTHORITY
                else "closure-lane-nonpromotable"
            ),
        )
        for gate in GateName
    )
    return BranchScreenResult(
        result_id=f"screen-{value.branch_id}-closure",
        input_id=value.input_id,
        case_id=value.branch_id,
        evidence_lane=EvidenceLane.TC_CLOSURE_DIAGNOSTIC,
        measurement_class=UniformElectronGasMeasurementClass.CLOSURE_ONLY,
        observation_order_class=UniformElectronGasObservationOrderClass.PHENOMENOLOGICAL_CLOSURE_ONLY,
        response_class=UniformElectronGasResponseClass.NOT_ATTEMPTED_PREREQUISITE,
        law_qualification_class=UniformElectronGasLawQualificationClass.NOT_ATTEMPTED_PREREQUISITE,
        admission_class=UniformElectronGasAdmissionClass.NOT_ATTEMPTED_PREREQUISITE,
        view_results=(),
        view_screens=(),
        gates=gates,
        robust_kernel0_lower_bound=None,
        penetration_depth_upper_m=None,
        shielding_score_lower=None,
        hold=True,
        reason_codes=("transverse-current-operand-required",),
    )


__all__ = [
    "adjudicate_branch",
    "analyze_branch",
    "evaluate_closure_only",
    "identify_branch",
]
