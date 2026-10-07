'Pure gauge covariant response design-freeze, staged-stop and closeout transformations.'

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path, required_external_sha256

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate

from .gauge_covariant_response_contracts import GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY, GaugeCovariantResponseCloseout, GaugeCovariantResponseConfig, GaugeCovariantResponseDevelopmentBasisFreeze, GaugeCovariantResponseSourceQualification, GaugeCovariantResponseStage, GaugeCovariantResponseStageDisposition, GaugeCovariantResponseStageResult, GaugeCovariantResponseConformanceResult
from .contracts import ExcludedSolverControlSolverSmokeResult, SolverSmokeDisposition


def build_development_basis_freeze(
    *,
    config: GaugeCovariantResponseConfig,
    source: GaugeCovariantResponseSourceQualification,
    conformance: GaugeCovariantResponseConformanceResult,
    development_protocol: ProtocolTemplate,
    development_registry: CapabilityRegistry,
    conditional_protocols: tuple[ProtocolTemplate, ...],
    conditional_registry: CapabilityRegistry,
) -> GaugeCovariantResponseDevelopmentBasisFreeze:
    if source.fingerprint() != config.source_extension_sha256:
        raise ValueError('gauge covariant response source extension differs from frozen config')
    if conformance.formalism_sha256 != config.gauge_covariant_formalism_sha256:
        raise ValueError('gauge covariant response gauge covariant response formalism differs from frozen config')
    if conformance.fixture_suite_sha256 != config.gauge_covariant_fixture_suite_sha256:
        raise ValueError('gauge covariant response gauge covariant response fixture suite differs from frozen config')
    if not conformance.strict_method_pass:
        raise ValueError('gauge covariant response design basis cannot freeze a failed strict-gauge covariant response method')
    if development_protocol.template_id != config.development_protocol_id:
        raise ValueError('gauge covariant response development protocol identity differs from config')
    provider_ids = tuple(
        sorted(
            {
                manifest.capability_key
                for registry in (development_registry, conditional_registry)
                for manifest in registry.capabilities
            }
        )
    )
    expected = tuple(
        sorted(
            (
                GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY,
                GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY,
            )
        )
    )
    if provider_ids != expected:
        raise ValueError('gauge covariant response development and conditional provider sets are incomplete')
    return GaugeCovariantResponseDevelopmentBasisFreeze(
        freeze_id='freeze.ambient-pressure-superconductor-gauge-covariant-response-staged-gauge-covariant-response-development',
        config_sha256=config.payload_sha256,
        amendment_sha256=config.amendment_sha256,
        base_source_design_result_sha256=config.base_source_design_result_sha256,
        base_source_sha256=config.base_source_sha256,
        base_roster_sha256=config.base_roster_sha256,
        base_exploration_sha256=config.base_exploration_sha256,
        base_science_sha256=config.base_science_sha256,
        source_extension_sha256=source.fingerprint(),
        gauge_covariant_formalism_sha256=conformance.formalism_sha256,
        gauge_covariant_conformance_sha256=conformance.fingerprint(),
        development_protocol_id=development_protocol.template_id,
        development_protocol_sha256=development_protocol.fingerprint(),
        development_registry_sha256=development_registry.fingerprint(),
        conditional_protocol_sha256s=tuple(
            sorted(protocol.fingerprint() for protocol in conditional_protocols)
        ),
        provider_capability_ids=provider_ids,
        compatibility_map_ids=tuple(
            sorted(
                (
                    'compatibility.gauge-covariant-response-base-science-design-additive-uniform-pairing-current',
                    'compatibility.material-source-design-transport-nomination-gauge-covariant-response-nomination-to-computational-admission-slot',
                    'compatibility.dft-wannier-hopping-frame-to-gauge-covariant-response-lattice',
                    'compatibility.epw-or-scdft-gap-temperature-to-gauge-covariant-response-pairing',
                    'compatibility.gauge-covariant-response-stiffness-interval-to-analytic-slab-receiver',
                    "compatibility.sixteen-material-gates-to-nine-generic-gates",
                )
            )
        ),
        calibration_derived_field_ids=tuple(
            sorted(
                (
                    "calibration.base-refined-view-agreement-limit",
                    "calibration.normal-cancellation-limit",
                    "calibration.numerical-noise-limit",
                    "calibration.receiver-tolerance",
                    "calibration.ward-residual-limit",
                )
            )
        ),
        target_contact_count=0,
        full_development_template_present=True,
        exact_provider_set_present=True,
        gauge_covariant_response_provider_present=True,
        computational_admission_slots_frozen=True,
        reason_codes=tuple(
            sorted(
                (
                    'reason.gauge-covariant-response-additive-compatibility-map-frozen',
                    'reason.gauge-covariant-response-full-static-development-template-frozen',
                    'reason.gauge-covariant-response-provider-set-and-conditional-slots-frozen',
                    'reason.gauge-covariant-response-gauge-covariant-response-conformance-passed-method-scope-only',
                    'reason.gauge-covariant-response-target-contact-count-zero',
                )
            )
        ),
    )


def _load_excluded_solver_control_result() -> ExcludedSolverControlSolverSmokeResult | None:
    result_path = required_external_path('EMPIRICAL_LAWHOOD_EXCLUDED_SOLVER_CONTROL_CORRECTIVE_RESULT')
    expected_sha256 = required_external_sha256('EMPIRICAL_LAWHOOD_EXCLUDED_SOLVER_CONTROL_CORRECTIVE_RESULT_SHA256')
    if not result_path.is_file() or result_path.is_symlink():
        return None
    payload = result_path.read_bytes()
    result = decode_canonical_bytes(
        payload,
        ExcludedSolverControlSolverSmokeResult,
        maximum_bytes=max(len(payload), 1),
    )
    if result.fingerprint() != expected_sha256:
        raise ValueError('immutable excluded solver control corrective solver control result fingerprint differs')
    return result


def build_material_control_control_result(
    *, design: GaugeCovariantResponseDevelopmentBasisFreeze, conformance: GaugeCovariantResponseConformanceResult
) -> GaugeCovariantResponseStageResult:
    if design.gauge_covariant_conformance_sha256 != conformance.fingerprint():
        raise ValueError('material control received another gauge covariant response conformance result')
    excluded_solver_control = _load_excluded_solver_control_result()
    reasons = [
        'reason.material-control-material-specific-300k-pairing-state-absent',
        'reason.material-control-material-specific-wannier-hamiltonians-absent',
        'reason.material-control-mgb2-phonon-and-epw-control-operands-absent',
        'reason.material-control-pb-phonon-and-epw-control-operands-absent',
        'reason.material-control-uniform-pairing-current-material-bloch-operands-absent',
        'reason.material-control-science-freeze-prohibited',
        'reason.material-control-si-current-compatibility-operands-absent',
        'reason.material-control-source-operand-required-before-control-execution',
    ]
    control_count = 0
    if excluded_solver_control is None:
        reasons.append('reason.material-control-excluded-solver-control-corrective-solver-control-control-receipt-unavailable')
    elif (
        excluded_solver_control.disposition is not SolverSmokeDisposition.CORRECTIVE_PASS
        or excluded_solver_control.target_contact_count != 0
    ):
        reasons.append('reason.material-control-excluded-solver-control-corrective-solver-control-control-not-qualified')
    else:
        reasons.append('reason.material-control-excluded-solver-control-corrective-solver-control-pb-scf-route-reused-as-excluded-control')
        control_count = 1
    if conformance.strict_method_pass:
        reasons.append('reason.material-control-gauge-covariant-response-method-fixture-suite-passed')
    else:
        reasons.append('reason.material-control-gauge-covariant-response-method-fixture-suite-failed')
    return GaugeCovariantResponseStageResult(
        result_id='result.ambient-pressure-superconductor-gauge-covariant-response-material-control-control-science-freeze',
        stage=GaugeCovariantResponseStage.MATERIAL_CONTROL,
        disposition=GaugeCovariantResponseStageDisposition.SOURCE_OPERAND_REQUIRED,
        attempted=True,
        evaluable=False,
        upstream_result_sha256=design.fingerprint(),
        target_contact_count=0,
        calibration_control_count=control_count,
        development_material_count=0,
        prospective_material_count=0,
        gauge_covariant_nomination_count=0,
        admission_count=0,
        controller_count=0,
        reason_codes=tuple(sorted(reasons)),
    )


def build_nonattempt(*, stage: GaugeCovariantResponseStage, upstream: GaugeCovariantResponseStageResult) -> GaugeCovariantResponseStageResult:
    return GaugeCovariantResponseStageResult(
        result_id=f"result.ambient-pressure-superconductor-gauge-covariant-response-{stage.value.lower().replace('_', '-')}-nonattempt-material-linked-receiver",
        stage=stage,
        disposition=GaugeCovariantResponseStageDisposition.NONATTEMPT,
        attempted=False,
        evaluable=False,
        upstream_result_sha256=upstream.fingerprint(),
        target_contact_count=upstream.target_contact_count,
        calibration_control_count=upstream.calibration_control_count,
        development_material_count=0,
        prospective_material_count=0,
        gauge_covariant_nomination_count=0,
        admission_count=0,
        controller_count=0,
        reason_codes=tuple(
            sorted(
                (
                    "reason.downstream-nonattempt-preserves-causal-cutoff",
                    f"reason.{stage.value.lower().replace('_', '-')}-blocked-by-material-control-source-stop",
                    "reason.no-material-outcome-accessed",
                )
            )
        ),
    )


def build_closeout(
    *, config: GaugeCovariantResponseConfig, design: GaugeCovariantResponseDevelopmentBasisFreeze, computational_admission: GaugeCovariantResponseStageResult
) -> GaugeCovariantResponseCloseout:
    if computational_admission.stage is not GaugeCovariantResponseStage.COMPUTATIONAL_ADMISSION:
        raise ValueError('gauge covariant response closeout requires the terminal computational admission development record')
    return GaugeCovariantResponseCloseout(
        closeout_id='closeout.ambient-pressure-superconductor-gauge-covariant-response-staged',
        config_sha256=config.payload_sha256,
        design_freeze_sha256=design.fingerprint(),
        terminal_parent_sha256=computational_admission.fingerprint(),
        operational_status="operational.succeeded",
        constructive_path_disposition_id="constructive-path.unevaluable",
        discovery_advantage_disposition_id="discovery-advantage.unevaluable",
        material_target_disposition_id='material-target.unevaluable-no-development-atlas-contact',
        maximum_claim_ceiling_id='ceiling.method-and-design-readiness',
        target_contact_count=computational_admission.target_contact_count,
        admission_count=0,
        controller_use_validation_count=0,
        computational_admission_candidate_count=0,
        downstream_nonattempt_stage_ids=tuple(
            sorted(
                (
                    'stage.ambient-pressure-superconductor-diverse-seed-actions',
                    'stage.ambient-pressure-superconductor-response-guided-exploration-wave-1',
                    'stage.ambient-pressure-superconductor-response-guided-exploration-wave-2',
                    'stage.ambient-pressure-superconductor-transport-nomination',
                    'stage.ambient-pressure-superconductor-computational-admission',
                    'stage.ambient-pressure-superconductor-prospective-controller-validation',
                    'stage.ambient-pressure-superconductor-independent-recurrence',
                )
            )
        ),
        reason_codes=tuple(
            sorted(
                (
                    'reason.development-closeout-closeout-after-material-control-source-operand-stop',
                    'reason.material-source-design-design-basis-frozen',
                    'reason.material-control-science-freeze-not-issued',
                    'reason.no-development-atlas-or-sealed-prospective-target-contact',
                    "reason.target-not-found-status-not-warranted",
                )
            )
        ),
    )


__all__ = [
    'build_material_control_control_result',
    "build_closeout",
    "build_development_basis_freeze",
    "build_nonattempt",
]
