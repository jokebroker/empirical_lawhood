"""Truth-known reference families for uniform electron gas transverse receiver screen conformance."""

from __future__ import annotations

from decimal import Decimal

from .analysis import adjudicate_branch, evaluate_closure_only, identify_branch
from .contracts import ActionCurrentRow, BranchInput, BranchScreenResult, CaseConformance, ClosureDiagnosticInput, EvidenceLane, GaugeControlRow, MethodConformanceResult, UniformElectronGasResponseClass, UniformElectronGasLawQualificationClass, UniformElectronGasAdmissionClass, UniformElectronGasTransverseScreenConfig, TransversePanel, TruthCase, TruthInputBatch, TruthKnownCase, TruthLawBatch, TruthOracle, TruthOracleBatch, TruthScreenBatch
from .physics import kernel_from_penetration_depth, ueg_scales, vector_potential_from_u, vector_scale


_CASE_INDEX = {
    value: index
    for index, value in enumerate(
        (
            TruthCase.MISSING_PRESERVATION,
            TruthCase.NONLINEAR,
            TruthCase.NORMAL,
            TruthCase.POSITIVE,
            TruthCase.Q_DRIFT,
            TruthCase.SPURIOUS_NORMAL,
            TruthCase.VIEW_DISAGREEMENT,
            TruthCase.WARD_FAIL,
            TruthCase.WRONG_SIGN,
        ),
        start=1,
    )
}


def _opaque_id(case: TruthCase) -> str:
    return f"truth-case-{_CASE_INDEX[case]:03d}"


def _expected(case: TruthCase) -> tuple[UniformElectronGasResponseClass, UniformElectronGasLawQualificationClass, UniformElectronGasAdmissionClass, str]:
    return {
        TruthCase.MISSING_PRESERVATION: (
            UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE,
            UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED,
            UniformElectronGasAdmissionClass.PRESERVATION_OPERAND_REQUIRED,
            "preservation-operand-required",
        ),
        TruthCase.NONLINEAR: (
            UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE,
            UniformElectronGasLawQualificationClass.NONLINEAR_BOUNDARY_LIMITED,
            UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW,
            "amplitude-locality-or-even-remainder-failed",
        ),
        TruthCase.NORMAL: (
            UniformElectronGasResponseClass.RESPONSE_BELOW_FLOOR,
            UniformElectronGasLawQualificationClass.NOT_SUPPORTED,
            UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW,
            "finite-response-below-floor",
        ),
        TruthCase.POSITIVE: (
            UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE,
            UniformElectronGasLawQualificationClass.TRANSVERSE_LAW_SUPPORTED,
            UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED,
            "all-gates-pass",
        ),
        TruthCase.Q_DRIFT: (
            UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE,
            UniformElectronGasLawQualificationClass.FINITE_Q_ONLY,
            UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW,
            "q-intercept-unstable",
        ),
        TruthCase.SPURIOUS_NORMAL: (
            UniformElectronGasResponseClass.CONTROL_FAILED,
            UniformElectronGasLawQualificationClass.NOT_SUPPORTED,
            UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW,
            "normal-cancellation-failed",
        ),
        TruthCase.VIEW_DISAGREEMENT: (
            UniformElectronGasResponseClass.FINITE_TRANSVERSE_RESPONSE,
            UniformElectronGasLawQualificationClass.VIEW_LOCAL_ONLY,
            UniformElectronGasAdmissionClass.BRANCH_LOCAL_ONLY,
            "view-or-bound-uncertainty-failed",
        ),
        TruthCase.WARD_FAIL: (
            UniformElectronGasResponseClass.CONTROL_FAILED,
            UniformElectronGasLawQualificationClass.NOT_SUPPORTED,
            UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW,
            "ward-conformance-failed",
        ),
        TruthCase.WRONG_SIGN: (
            UniformElectronGasResponseClass.WRONG_SIGN,
            UniformElectronGasLawQualificationClass.NOT_SUPPORTED,
            UniformElectronGasAdmissionClass.NOT_ENTERED_LOCAL_LAW,
            "diamagnetic-direction-failed",
        ),
    }[case]


def _kernel_at_q(
    *,
    case: TruthCase,
    view_id: str,
    q_over_kf: Decimal,
    base_kernel: Decimal,
    q_max: Decimal,
) -> Decimal:
    view_factor = Decimal("1")
    if case is TruthCase.VIEW_DISAGREEMENT and view_id == "refined":
        view_factor = Decimal("0.80")
    if case is TruthCase.NORMAL:
        return Decimal("0")
    sign = Decimal("-1") if case is TruthCase.WRONG_SIGN else Decimal("1")
    if case is TruthCase.Q_DRIFT:
        # A quartic bend cannot be represented by the frozen q^2 model.  The
        # scale is selected before evaluation and keeps every finite-q kernel
        # positive while shifting the all-q intercept away from the
        # three-smallest-q intercept.
        bend = Decimal("8") * (q_over_kf / q_max) ** 4
        return sign * view_factor * base_kernel * (Decimal("1") + bend)
    return sign * view_factor * base_kernel * (Decimal("1") + Decimal("4") * q_over_kf**2)


def _panel(case: TruthCase, view_id: str, config: UniformElectronGasTransverseScreenConfig) -> TransversePanel:
    case_id = _opaque_id(case)
    index = _CASE_INDEX[case]
    scales = ueg_scales(config.r_s)
    base_kernel = kernel_from_penetration_depth(config.positive_penetration_depth_m)
    q_max = max(config.q_over_kf)
    rows = []
    for q_index, q_value in enumerate(config.q_over_kf):
        kernel_q = _kernel_at_q(
            case=case,
            view_id=view_id,
            q_over_kf=q_value,
            base_kernel=base_kernel,
            q_max=q_max,
        )
        for u_index, u_value in enumerate(config.u_values):
            a_amplitude = vector_potential_from_u(u_value, scales)
            a_vector = vector_scale(config.action_direction, a_amplitude)
            nonlinear_factor = Decimal("1")
            if case is TruthCase.NONLINEAR and config.u0 != 0:
                nonlinear_factor += Decimal("0.20") * (u_value / config.u0) ** 2
            current_amplitude = -kernel_q * a_amplitude * nonlinear_factor
            current = vector_scale(config.action_direction, current_amplitude)
            error_kernel = (
                max(abs(kernel_q), scales.diamagnetic_kernel_A_T_m3)
                * config.deterministic_kernel_relative_error
            )
            error_current = error_kernel * max(
                abs(a_amplitude),
                abs(vector_potential_from_u(config.u0, scales)),
            )
            rows.append(
                ActionCurrentRow(
                    row_id=(f"row-{case_id}-{view_id}-q{q_index:02d}-u{u_index:02d}"),
                    q_over_kf=q_value,
                    u=u_value,
                    requested_A_T=a_vector,
                    accepted_A_T=a_vector,
                    applied_A_T=a_vector,
                    realized_A_T=a_vector,
                    current_density=current,
                    current_error_bound_A_m2=error_current,
                    requested_clock=config.requested_clock,
                    accepted_clock=config.accepted_clock,
                    applied_clock=config.applied_clock,
                    receiver_clock=config.receiver_clock,
                    accepted=True,
                    valid=True,
                )
            )
    normal_residual_fraction = (
        Decimal("0.02") if case is TruthCase.SPURIOUS_NORMAL else Decimal("0")
    )
    controls = tuple(
        GaugeControlRow(
            control_id=f"control-{case_id}-{view_id}-q{q_index:02d}",
            q_over_kf=q_value,
            diamagnetic_normal=scales.diamagnetic_kernel_A_T_m3,
            paramagnetic_normal=(
                -scales.diamagnetic_kernel_A_T_m3 * (Decimal("1") - normal_residual_fraction)
            ),
            ward_residual=(Decimal("1e-3") if case is TruthCase.WARD_FAIL else Decimal("1e-8")),
            wrong_polarization_current_A_m2=Decimal("0"),
            converged=True,
            limit_order_supported=True,
        )
        for q_index, q_value in enumerate(config.q_over_kf)
    )
    if case is TruthCase.NORMAL:
        order_zero: Decimal | None = Decimal("0")
        order_max: Decimal | None = Decimal("0")
    elif case is TruthCase.MISSING_PRESERVATION:
        order_zero = Decimal("1")
        order_max = None
    else:
        order_zero = Decimal("1")
        order_max = Decimal("0.99")
    return TransversePanel(
        panel_id=f"panel-{case_id}-{view_id}",
        source_id="source-uniform-electron-gas-response-truth-known",
        branch_id=f"branch-truth-{index:03d}",
        view_id=view_id,
        evidence_lane=EvidenceLane.GAUGE_CLOSED_TRANSVERSE,
        evidence_world="analytic-reference",
        independent_unit_id=f"acquisition-{case_id}",
        deterministic=True,
        r_s=config.r_s,
        temperature_K=config.temperature_K,
        q_direction=config.q_direction,
        action_direction=config.action_direction,
        rows=tuple(sorted(rows, key=lambda value: value.row_id)),
        controls=tuple(sorted(controls, key=lambda value: value.control_id)),
        order_zero=order_zero,
        order_at_max_action=order_max,
        baseline_preserved=True,
        source_semantics_valid=True,
    )


def build_truth_case(case: TruthCase, config: UniformElectronGasTransverseScreenConfig) -> TruthKnownCase:
    """Build method input and a separately held privileged oracle."""

    case_id = _opaque_id(case)
    expected_response_class, expected_law_qualification_class, expected_admission_class, reason = _expected(case)
    method_input = BranchInput(
        input_id=f"input-{case_id}",
        case_id=case_id,
        panels=tuple(
            sorted(
                (_panel(case, view_id, config) for view_id in config.views),
                key=lambda value: value.panel_id,
            )
        ),
        authority_valid=True,
    )
    oracle = TruthOracle(
        oracle_id=f"oracle-{case_id}",
        case_id=case_id,
        truth_family=case,
        expected_response_class=expected_response_class,
        expected_law_qualification_class=expected_law_qualification_class,
        expected_admission_class=expected_admission_class,
        true_penetration_depth_m=(
            config.positive_penetration_depth_m
            if case
            in {
                TruthCase.MISSING_PRESERVATION,
                TruthCase.NONLINEAR,
                TruthCase.POSITIVE,
                TruthCase.SPURIOUS_NORMAL,
                TruthCase.VIEW_DISAGREEMENT,
                TruthCase.WARD_FAIL,
            }
            else None
        ),
        decisive_reason_code=reason,
    )
    # The method payload must not carry any privileged family label.  Physical
    # planted behavior remains visible because that is the method's input.
    payload_text = method_input.canonical_bytes().decode("utf-8")
    if any(f'"{value.value}"' in payload_text for value in TruthCase):
        raise ValueError("truth family leaked into the method input")
    return TruthKnownCase(case_id=case_id, method_input=method_input, privileged_oracle=oracle)


def _score_case(
    *, result: BranchScreenResult, oracle: TruthOracle, config: UniformElectronGasTransverseScreenConfig
) -> CaseConformance:
    screen = result
    reasons = []
    if screen.response_class is not oracle.expected_response_class:
        reasons.append("response-class-mismatch")
    if screen.law_qualification_class is not oracle.expected_law_qualification_class:
        reasons.append("response-law-qualification-class-mismatch")
    if screen.admission_class is not oracle.expected_admission_class:
        reasons.append("response-screen-admission-class-mismatch")
    if oracle.decisive_reason_code == "all-gates-pass":
        if not all(gate.passed for gate in screen.gates):
            reasons.append("positive-gate-intersection-failed")
    elif oracle.decisive_reason_code not in screen.reason_codes:
        reasons.append("decisive-reason-not-recovered")
    if oracle.truth_family is TruthCase.POSITIVE:
        if screen.penetration_depth_upper_m is None:
            reasons.append("positive-penetration-depth-missing")
        else:
            relative = (
                abs(screen.penetration_depth_upper_m - config.positive_penetration_depth_m)
                / config.positive_penetration_depth_m
            )
            if relative > config.threshold("slab_fit_relative"):
                reasons.append("positive-penetration-depth-recovery-failed")
    return CaseConformance(
        case_id=oracle.case_id,
        result=screen,
        oracle=oracle,
        passed=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def build_truth_batches(config: UniformElectronGasTransverseScreenConfig) -> tuple[TruthInputBatch, TruthOracleBatch]:
    cases = tuple(build_truth_case(case, config) for case in config.truth_cases)
    inputs = TruthInputBatch(
        batch_id="uniform-electron-gas-transverse-screen-truth-inputs",
        config_sha256=config.payload_sha256,
        inputs=tuple(
            sorted((value.method_input for value in cases), key=lambda value: value.case_id)
        ),
    )
    oracles = TruthOracleBatch(
        batch_id="uniform-electron-gas-transverse-screen-truth-oracles",
        config_sha256=config.payload_sha256,
        oracles=tuple(
            sorted((value.privileged_oracle for value in cases), key=lambda value: value.case_id)
        ),
    )
    if any(f'"{value.value}"' in inputs.canonical_bytes().decode("utf-8") for value in TruthCase):
        raise ValueError("privileged truth family leaked into the method batch")
    return inputs, oracles


def identify_truth_batch(value: TruthInputBatch, config: UniformElectronGasTransverseScreenConfig) -> TruthLawBatch:
    if value.config_sha256 != config.payload_sha256:
        raise ValueError("truth input batch config differs")
    return TruthLawBatch(
        batch_id="uniform-electron-gas-transverse-screen-truth-laws",
        config_sha256=config.payload_sha256,
        laws=tuple(
            sorted(
                (identify_branch(branch, config) for branch in value.inputs),
                key=lambda law: law.case_id,
            )
        ),
    )


def adjudicate_truth_batch(
    inputs: TruthInputBatch, laws: TruthLawBatch, config: UniformElectronGasTransverseScreenConfig
) -> TruthScreenBatch:
    if inputs.config_sha256 != config.payload_sha256 or laws.config_sha256 != config.payload_sha256:
        raise ValueError("truth admission batch config differs")
    branch_by_case = {value.case_id: value for value in inputs.inputs}
    law_by_case = {value.case_id: value for value in laws.laws}
    if set(branch_by_case) != set(law_by_case):
        raise ValueError("truth input and law case rosters differ")
    return TruthScreenBatch(
        batch_id="uniform-electron-gas-transverse-screen-truth-screens",
        config_sha256=config.payload_sha256,
        screens=tuple(
            adjudicate_branch(branch_by_case[case_id], law_by_case[case_id], config)
            for case_id in sorted(branch_by_case)
        ),
    )


def evaluate_truth_batches(
    screens: TruthScreenBatch, oracles: TruthOracleBatch, config: UniformElectronGasTransverseScreenConfig
) -> MethodConformanceResult:
    if (
        screens.config_sha256 != config.payload_sha256
        or oracles.config_sha256 != config.payload_sha256
    ):
        raise ValueError("truth evaluator batch config differs")
    screen_by_case = {value.case_id: value for value in screens.screens}
    oracle_by_case = {value.case_id: value for value in oracles.oracles}
    if set(screen_by_case) != set(oracle_by_case):
        raise ValueError("truth screen and oracle case rosters differ")
    cases = tuple(
        _score_case(
            result=screen_by_case[case_id],
            oracle=oracle_by_case[case_id],
            config=config,
        )
        for case_id in sorted(screen_by_case)
    )
    closure = ClosureDiagnosticInput(
        input_id="closure-conformance-input",
        source_id="closure-conformance-source",
        branch_id="closure-conformance-branch",
        r_s=config.r_s,
        temperature_K=config.temperature_K,
        supplied_tc_K=Decimal("600"),
        stiffness_closure="bcs-phenomenological-closure",
    )
    closure_result = evaluate_closure_only(closure, config)
    closure_pass = (
        closure_result.evidence_lane is EvidenceLane.TC_CLOSURE_DIAGNOSTIC
        and closure_result.admission_class is UniformElectronGasAdmissionClass.NOT_ATTEMPTED_PREREQUISITE
        and closure_result.hold
        and closure_result.penetration_depth_upper_m is None
        and closure_result.robust_kernel0_lower_bound is None
        and not closure_result.view_screens
    )
    passed = closure_pass and all(value.passed for value in cases)
    return MethodConformanceResult(
        result_id="uniform-electron-gas-transverse-screen-method-conformance",
        config_sha256=config.payload_sha256,
        cases=cases,
        closure_nonpromotion_pass=closure_pass,
        passed=passed,
        status=("TRANSVERSE_SCREEN_METHOD_CONFORMANCE_PASS" if passed else "TRANSVERSE_SCREEN_METHOD_CONFORMANCE_FAIL"),
    )


def run_truth_known_conformance(config: UniformElectronGasTransverseScreenConfig) -> MethodConformanceResult:
    """Generate, identify and evaluator-reveal all nine truth-known families."""
    inputs, oracles = build_truth_batches(config)
    laws = identify_truth_batch(inputs, config)
    screens = adjudicate_truth_batch(inputs, laws, config)
    return evaluate_truth_batches(screens, oracles, config)


__all__ = [
    "adjudicate_truth_batch",
    "build_truth_batches",
    "build_truth_case",
    "evaluate_truth_batches",
    "identify_truth_batch",
    "run_truth_known_conformance",
]
