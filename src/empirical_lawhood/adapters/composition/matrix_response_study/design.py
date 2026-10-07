"""Code-owned, outcome-blind Six-matrix response-0 design binding.

The factory performs no source contact, storage write, issue, execution or
reveal.  It is the single exact constructor for the repository-held design
configuration and for Six-matrix response-0A reconstruction tests.
"""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.adapters.control.matrix_response_study.contracts import MatrixResponseActionDomain, MatrixResponseActionGrammar, MatrixResponseOuterArmDisposition, MatrixResponseOuterArm, MatrixResponseOuterStudyAuthoringConfig, MatrixResponsePrimitiveAction, MatrixResponseProspectiveEvaluationTopology, ProtectedAccessClaim, ProtectedFieldAccessRule, ProtectedObservableAccessManifest
from empirical_lawhood.adapters.methods.matrix_response_study.contracts import MatrixResponseLawFamilyConfig, MatrixResponseLawMethodConfig, MatrixResponsePairedPanelReducerConfig, MatrixResponsePropertyFace, MatrixResponseRoleEquivarianceConfig, MatrixResponseRoleSpace, MatrixResponseStructuralFacePlan, ConstitutiveLawFamily, CrossSizeMapKind, RoleSpaceKind
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import AnisotropicSixMatrixConstitutiveModel, BetaCouplingRule, SixMatrixResponseFeasibilityEnvelope, SixMatrixResponseSizeCompatibility, SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalThresholds, SixMatrixResponseNumericalView, SixMatrixResponsePreparationSchedule, SixMatrixResponseSixMatrixSourceConfig, CanonicalSixMatrixIsotropicModel, MatrixIntegratorKind, MatrixPrecision
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind

from .contracts import SixMatrixResponseArtifactDataset, SixMatrixResponseArtifactProfileConfig, MatrixResponseCampaignPackageDag, MatrixResponseCampaignPackageNode, MatrixResponseDesignBinding, MatrixResponseResourceEnvelope
from .lineage_inputs import MatrixResponseLineageInputs


_TWO_THIRDS = Decimal(2) / Decimal(3)


def _canonical_models() -> tuple[CanonicalSixMatrixIsotropicModel, ...]:
    points = (
        *((2, mass, alpha) for mass in ("0.5", "1", "10") for alpha in ("0", "4.2", "8")),
        *((3, "1", alpha) for alpha in ("0", "4.2", "8")),
        *((4, "1", alpha) for alpha in ("0", "8")),
    )
    values = tuple(
        CanonicalSixMatrixIsotropicModel(
            model_id=(f"six-matrix-response.canonical.q{q}.mass-{mass.replace('.', 'p')}.scaled-coupling-{alpha.replace('.', 'p')}"),
            matrix_dimension_rule="n=q^2",
            representation_dimension_q=q,
            mass_m=Decimal(mass),
            cross_coupling_gamma=Decimal(1),
            beta_rule=BetaCouplingRule.PUBLISHED_DETERMINISTIC,
            scaled_coupling_id="six-matrix-response.quantity.alpha-tilde",
            scaled_coupling_alpha_tilde=Decimal(alpha),
        )
        for q, mass, alpha in points
    )
    return tuple(sorted(values, key=lambda value: value.model_id))


def _model_family() -> AnisotropicSixMatrixConstitutiveModel:
    members = tuple(
        SixMatrixResponseModelFamilyMember(
            member_id=(f"six-matrix-response.member.mass-{mass.replace('.', 'p')}.cross-coupling-{gamma.replace('.', 'p')}"),
            mass_x=Decimal(mass),
            mass_y=Decimal(mass),
            cross_coupling_gamma=Decimal(gamma),
            beta_rule=BetaCouplingRule.PUBLISHED_DETERMINISTIC,
        )
        for mass in ("0.5", "1", "10")
        for gamma in ("0.25", "1")
    )
    return AnisotropicSixMatrixConstitutiveModel(
        model_family_id="six-matrix-response.model-family.anisotropic-six-matrix",
        matrix_dimension_rule="n=q^2",
        scaled_coupling_id_x="six-matrix-response.quantity.alpha-tilde-x",
        scaled_coupling_id_y="six-matrix-response.quantity.alpha-tilde-y",
        beta_rule=BetaCouplingRule.PUBLISHED_DETERMINISTIC,
        family_members=tuple(sorted(members, key=lambda value: value.member_id)),
        protected_action_quantity_ids=(
            "six-matrix-response.quantity.alpha-tilde-x",
            "six-matrix-response.quantity.alpha-tilde-y",
        ),
    )


def _source_config() -> SixMatrixResponseSixMatrixSourceConfig:
    primary = SixMatrixResponseNumericalView(
        view_id="six-matrix-response.view.baoab-dt-0p001",
        integrator=MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN,
        timestep=Decimal("0.001"),
        time_unit="dimensionless-langevin-time",
        friction_gamma=Decimal(1),
        bath_temperature=Decimal(1),
        precision=MatrixPrecision.COMPLEX128,
        rng_algorithm="numpy-pcg64dxsm",
        rng_version="2.4.6",
        stream_derivation_rule_id="six-matrix-response.rng.explicit-original-full-seed-pcg64dxsm",
        checkpoint_interval_steps=256,
    )
    secondary = SixMatrixResponseNumericalView(
        view_id="six-matrix-response.view.baoab-dt-0p0005",
        integrator=MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN,
        timestep=Decimal("0.0005"),
        time_unit="dimensionless-langevin-time",
        friction_gamma=Decimal(1),
        bath_temperature=Decimal(1),
        precision=MatrixPrecision.COMPLEX128,
        rng_algorithm="numpy-pcg64dxsm",
        rng_version="2.4.6",
        stream_derivation_rule_id="six-matrix-response.rng.explicit-original-full-seed-pcg64dxsm",
        checkpoint_interval_steps=256,
    )
    feasibility_schedule = SixMatrixResponsePreparationSchedule(
        schedule_id="six-matrix-response.schedule.short-feasibility",
        ramp_steps=256,
        dwell_steps=512,
        qualification_steps=256,
        fast_receiver_cadence_steps=16,
        spectral_receiver_cadence_steps=256,
    )
    confirmation_schedule = SixMatrixResponsePreparationSchedule(
        schedule_id="six-matrix-response.schedule.feasibility-confirmation",
        ramp_steps=512,
        dwell_steps=1024,
        qualification_steps=512,
        fast_receiver_cadence_steps=16,
        spectral_receiver_cadence_steps=256,
    )
    feasibility = SixMatrixResponseFeasibilityEnvelope(
        envelope_id="six-matrix-response.feasibility-envelope",
        feasibility_alpha_tilde_min=Decimal(0),
        feasibility_alpha_tilde_max=Decimal(8),
        feasibility_axis_count=13,
        feasibility_history_ids=(
            "matrix-history.joint-increasing-coupling",
            "matrix-history.x-first-increasing-coupling",
            "matrix-history.y-first-increasing-coupling",
        ),
        feasibility_seed_count=3,
        feasibility_schedule=feasibility_schedule,
        confirmation_alpha_tilde_min=Decimal(0),
        confirmation_alpha_tilde_max=Decimal(8),
        confirmation_axis_count=9,
        confirmation_history_ids=(
            "matrix-history.joint-increasing-coupling",
            "matrix-history.joint-decreasing-coupling",
            "matrix-history.x-first-increasing-coupling",
            "matrix-history.y-first-increasing-coupling",
        ),
        confirmation_seed_count=4,
        confirmation_schedule=confirmation_schedule,
        secondary_view_stratification_rule_id="six-matrix-response.stratification.boundary-interior",
        family_selection_metric_ids=(
            "six-matrix-response.selection.01-all-four-constitutions",
            "six-matrix-response.selection.02-minimum-axial-interior-width",
            "six-matrix-response.selection.03-seed-recurrence",
            "six-matrix-response.selection.04-history-diversity",
            "six-matrix-response.selection.05-numerical-concordance",
            "six-matrix-response.selection.06-kernel-separation",
            "six-matrix-response.selection.07-negative-false-admission",
        ),
        family_selection_tie_break="ascending-member-id",
        maximum_anisotropic_feasibility_integration_steps=15_000_000,
    )
    thresholds = SixMatrixResponseNumericalThresholds(
        thresholds_id="six-matrix-response.numerical-thresholds",
        gradient_relative_error_max=Decimal("1e-7"),
        hermiticity_residual_max=Decimal("1e-12"),
        invariance_relative_error_max=Decimal("1e-10"),
        ideal_spectrum_absolute_error_max=Decimal("1e-10"),
        radius_phi_min=Decimal("0.35"),
        radius_phi_max=Decimal("0.95"),
        su2_closure_ratio_max=Decimal("0.30"),
        approximate_kernel_band_ratio_max=Decimal("0.25"),
        persistence_fraction_min=Decimal("0.75"),
        confirmation_seed_recurrence_minimum=3,
        confirmation_history_recurrence_minimum=2,
        confirmation_minimum_cells_per_axis=2,
        numerical_view_phase_concordance_min=Decimal(1),
    )
    compatibility = SixMatrixResponseSizeCompatibility(
        compatibility_id="six-matrix-response.finite-size-compatibility",
        assay_roster=(2, 3, 4),
        raw_coupling_formula="alpha=alpha_tilde/q",
        scaled_coupling_formula="alpha_tilde=q*alpha",
        action_normalization_formula="S_density=S/n",
        time_normalization_id="six-matrix-response.normalization.dimensionless-langevin-time",
        friction_normalization_id="six-matrix-response.normalization.unit-friction",
        bath_noise_normalization_id="six-matrix-response.normalization.hs-isotropic-unit-bath",
        receiver_normalization_ids=(
            "six-matrix-response.normalization.action-density-by-n",
            "six-matrix-response.normalization.heat-trace-by-q4",
            "six-matrix-response.normalization.radius-by-alpha-casimir",
            "six-matrix-response.normalization.spectrum-by-radius-squared",
        ),
        effort_normalization_id="six-matrix-response.normalization.generalized-work-by-n",
        spectral_transport_kind="quotient-invariant-normalized-bands",
        projector_transport_kind="not-applicable-across-q",
        decisive_falsifier_ids=(
            "six-matrix-response.falsifier.cross-q-constitution-loss",
            "six-matrix-response.falsifier.history-only-one-factor-support",
            "six-matrix-response.falsifier.numerical-view-phase-disagreement",
            "six-matrix-response.falsifier.spectral-band-order-reversal",
        ),
    )
    return SixMatrixResponseSixMatrixSourceConfig(
        config_id="six-matrix-response.six-matrix-source-config",
        config_version="1.0.0",
        canonical_reference_models=_canonical_models(),
        anisotropic_model=_model_family(),
        primary_view=primary,
        secondary_view=secondary,
        feasibility_envelope=feasibility,
        numerical_thresholds=thresholds,
        finite_size_compatibility=compatibility,
        independent_unit_kind="rng-seeded-matrix-trajectory",
        acquisition_group_rule_id="six-matrix-response.acquisition.one-effect-per-view",
        maximum_total_integration_steps=30_000_000,
        maximum_worker_processes=8,
        blas_threads_per_worker=1,
        grants_authority=False,
    )


def _role_space(
    *,
    suffix: str,
    kind: RoleSpaceKind,
    dimension: str,
    ambient: str,
    scalar: str,
    cross_size: CrossSizeMapKind,
) -> MatrixResponseRoleSpace:
    return MatrixResponseRoleSpace(
        role_space_id=f"six-matrix-response.role-space.{suffix}",
        kind=kind,
        real_dimension_formula=dimension,
        ambient_space_id_formula=ambient,
        basis_frame_id=f"six-matrix-response.frame.{suffix}",
        scalar_convention=scalar,
        inner_product_id=f"six-matrix-response.inner-product.{suffix}",
        gauge_quotient_id="six-matrix-response.quotient.simultaneous-unitary-conjugation",
        cross_size_map_kind=cross_size,
        degeneracy_rule_id="six-matrix-response.degeneracy.full-eigenspace-projector",
        projector_rank_rule_id="six-matrix-response.projector-rank.numerical-qualification-noise-calibrated",
        projector_tolerance_rule_id="six-matrix-response.projector-tolerance.band-gap-calibrated",
        response_map_id=f"six-matrix-response.response-map.{suffix}",
        realizable_force_map_id="six-matrix-response.force-map.hs-projected-hermitian",
    )


def _role_config() -> MatrixResponseRoleEquivarianceConfig:
    spaces = (
        _role_space(
            suffix="configuration",
            kind=RoleSpaceKind.CONFIGURATION,
            dimension="6*q^4",
            ambient="six-matrix-response.ambient.configuration.q{q}",
            scalar="complex-hermitian-realification",
            cross_size=CrossSizeMapKind.NOT_APPLICABLE,
        ),
        _role_space(
            suffix="force",
            kind=RoleSpaceKind.HERMITIAN_FORCE,
            dimension="6*q^4",
            ambient="six-matrix-response.ambient.hermitian-force.q{q}",
            scalar="complex-hermitian-realification",
            cross_size=CrossSizeMapKind.NOT_APPLICABLE,
        ),
        _role_space(
            suffix="latent",
            kind=RoleSpaceKind.LOCAL_LAW_LATENT,
            dimension="24",
            ambient="six-matrix-response.ambient.local-law-latent",
            scalar="real",
            cross_size=CrossSizeMapKind.QUOTIENT_INVARIANT_SUMMARY,
        ),
        _role_space(
            suffix="phase-space",
            kind=RoleSpaceKind.PHASE_SPACE,
            dimension="12*q^4",
            ambient="six-matrix-response.ambient.phase-space.q{q}",
            scalar="complex-hermitian-realification",
            cross_size=CrossSizeMapKind.NOT_APPLICABLE,
        ),
        _role_space(
            suffix="probe",
            kind=RoleSpaceKind.HERMITIAN_PROBE,
            dimension="q^4",
            ambient="six-matrix-response.ambient.hermitian-probe.q{q}",
            scalar="complex-hermitian-realification",
            cross_size=CrossSizeMapKind.NOT_APPLICABLE,
        ),
        _role_space(
            suffix="receiver",
            kind=RoleSpaceKind.RECEIVER_RESPONSE,
            dimension="36",
            ambient="six-matrix-response.ambient.receiver-response",
            scalar="real",
            cross_size=CrossSizeMapKind.EXACT_SCALAR_NORMALIZATION,
        ),
    )
    return MatrixResponseRoleEquivarianceConfig(
        config_id="six-matrix-response.role-equivariance-config",
        role_spaces=tuple(sorted(spaces, key=lambda value: value.role_space_id)),
        transformation_ids=(
            "six-matrix-response.transform.anonymous-mode-permutation",
            "six-matrix-response.transform.degenerate-eigenspace-order",
            "six-matrix-response.transform.factor-exchange",
            "six-matrix-response.transform.simultaneous-unitary-conjugation",
            "six-matrix-response.transform.triplet-so3-x",
            "six-matrix-response.transform.triplet-so3-y",
        ),
        invariant_feature_ids=(
            "six-matrix-response.feature.factor-gram-spectrum",
            "six-matrix-response.feature.heat-trace",
            "six-matrix-response.feature.joint-laplacian-spectrum",
            "six-matrix-response.feature.normalized-action-radius",
            "six-matrix-response.feature.response-projector",
            "six-matrix-response.feature.triplet-laplacian-spectrum",
        ),
        factor_anonymization_rule_id="six-matrix-response.anonymization.invariant-signature-then-ambiguity",
        degenerate_subspace_rule_id="six-matrix-response.degeneracy.compare-whole-projector",
        same_q_projector_comparison_id="six-matrix-response.projector.principal-angles-hs",
        cross_q_projector_disposition=CrossSizeMapKind.NOT_APPLICABLE,
        adversarial_conformance_ids=(
            "six-matrix-response.conformance.factor-exchange-heldout",
            "six-matrix-response.conformance.mode-permutation-heldout",
            "six-matrix-response.conformance.so3-heldout",
            "six-matrix-response.conformance.unitary-heldout",
        ),
        maximum_role_slots_per_constitution=4,
    )


def _law_family(
    *,
    family_id: str,
    family: ConstitutiveLawFamily,
    system_id: str,
    denominator_id: str,
    history_id: str,
    chart_id: str,
    receiver_id: str,
    horizon_id: str,
    relation_id: str,
    action_quantity_ids: tuple[str, ...],
    receiver_quantity_ids: tuple[str, ...],
) -> MatrixResponseLawFamilyConfig:
    return MatrixResponseLawFamilyConfig(
        law_family_id=family_id,
        family=family,
        system_id=system_id,
        prepared_denominator_id=denominator_id,
        retained_history_id=history_id,
        action_chart_id=chart_id,
        receiver_id=receiver_id,
        horizon_id=horizon_id,
        relation_id=relation_id,
        action_quantity_ids=action_quantity_ids,
        receiver_quantity_ids=receiver_quantity_ids,
        candidate_payload_schema='empirical-lawhood/composition/matrix-response-study/local-law-candidate-payload',
        candidate_payload_version="1.0.0",
        candidate_decoder_id="six-matrix-response.decoder.local-law-candidate",
        method_kind=LawMethodKind.NONLINEAR_LOCAL,
        representation_kind=(
            LawRepresentationKind.FINITE_ACTION_OPERATOR
            if family is ConstitutiveLawFamily.L_UP
            else LawRepresentationKind.LOCAL_STATE_SPACE
        ),
        causal_strength_ceiling=CausalStrength.SIMULATOR_INTERVENTION,
        mapping_assumption_ids=(
            "six-matrix-response.assumption.local-support-only",
            "six-matrix-response.assumption.same-implementation-resample",
            "six-matrix-response.assumption.typed-action-receiver-continuity",
        ),
        decisive_falsifier_ids=(
            "six-matrix-response.falsifier.heldout-equivariance",
            "six-matrix-response.falsifier.history-nonrecurrence",
            "six-matrix-response.falsifier.numerical-member-disagreement",
            "six-matrix-response.falsifier.out-of-support-action",
            "six-matrix-response.falsifier.residual-leakage",
        ),
        candidate_version_ids=("six-matrix-response.candidate.nonlinear-local",),
        numerical_member_ids=("six-matrix-response.member.primary-dt", "six-matrix-response.member.secondary-half-dt"),
        qualification_view_ids=(
            "six-matrix-response.qualification.factor-exchange",
            "six-matrix-response.qualification.history-recurrence",
            "six-matrix-response.qualification.numerical-concordance",
            "six-matrix-response.qualification.seed-recurrence",
        ),
    )


def _law_method_config() -> MatrixResponseLawMethodConfig:
    parent = _law_family(
        family_id="six-matrix-response.law-family.l-up.parent",
        family=ConstitutiveLawFamily.L_UP,
        system_id="six-matrix-response.system.parent-anisotropic",
        denominator_id="six-matrix-response.denominator.parent-anisotropic",
        history_id="six-matrix-response.history.parent-window",
        chart_id="six-matrix-response.chart.outer",
        receiver_id="six-matrix-response.receiver.parent",
        horizon_id="six-matrix-response.horizon.outer",
        relation_id="six-matrix-response.relation.l-up",
        action_quantity_ids=(
            "six-matrix-response.quantity.alpha-tilde-x",
            "six-matrix-response.quantity.alpha-tilde-y",
        ),
        receiver_quantity_ids=(
            "six-matrix-response.receiver.constitution-support",
            "six-matrix-response.receiver.cross-commutator",
            "six-matrix-response.receiver.generalized-work",
            "six-matrix-response.receiver.geometry-preservation",
            "six-matrix-response.receiver.role-projectors",
            "six-matrix-response.receiver.validity",
        ),
    )
    lower = tuple(
        _law_family(
            family_id=f"six-matrix-response.law-family.l-down.{constitution}.mode-{mode}",
            family=ConstitutiveLawFamily.L_DOWN,
            system_id=f"six-matrix-response.system.lower.{constitution}.mode-{mode}",
            denominator_id=f"six-matrix-response.denominator.lower.{constitution}.mode-{mode}",
            history_id=f"six-matrix-response.history.lower.{constitution}.mode-{mode}",
            chart_id=f"six-matrix-response.chart.inner.{constitution}.mode-{mode}",
            receiver_id=f"six-matrix-response.receiver.lower.{constitution}.mode-{mode}",
            horizon_id="six-matrix-response.horizon.inner",
            relation_id=f"six-matrix-response.relation.l-down.{constitution}.mode-{mode}",
            action_quantity_ids=("six-matrix-response.quantity.force-amplitude",),
            receiver_quantity_ids=(
                "six-matrix-response.receiver.geometry-preservation",
                "six-matrix-response.receiver.inner-effort",
                "six-matrix-response.receiver.matter-target-error",
                "six-matrix-response.receiver.mode-response",
            ),
        )
        for constitution in ("01", "10", "11")
        for mode in range(4)
    )
    return MatrixResponseLawMethodConfig(
        config_id="six-matrix-response.law-method-config",
        parent_law=parent,
        lower_laws=tuple(sorted(lower, key=lambda value: value.law_family_id)),
        qualification_profile_id="six-matrix-response.qualification-profile.nonlinear-local",
        method_evidence_producer_id="six-matrix-response.method-evidence.equivariant-local-law",
        sole_finalizer_service_id="response-law-qualification-service",
        batch_owner_id="law-qualification-batch-owner",
        atlas_owner_id="response-atlas-owner",
        leakage_statistic_id="six-matrix-response.leakage.out-of-role-energy-fraction",
        leakage_upper_bound=Decimal("0.10"),
        grants_law_truth=False,
    )


def _structural_plan() -> MatrixResponseStructuralFacePlan:
    definitions = (
        (
            "factor-exchange",
            "anonymous-factor-conversion",
            "quotient-equivariance",
            "quotient-factor-map",
            "exact-factor-exchange-roster",
            "exchange-categorical-match",
            CrossSizeMapKind.QUOTIENT_INVARIANT_SUMMARY,
        ),
        (
            "interaction-response",
            "interaction-response-calibration",
            "response-calibration",
            "normalized-scalar-map",
            "exact-interaction-target-roster",
            "calibration-interval-distance",
            CrossSizeMapKind.EXACT_SCALAR_NORMALIZATION,
        ),
        (
            "kernel-dimension",
            "kernel-interval-and-degeneracy",
            "interval-categorical",
            "quotient-kernel-summary",
            "exact-kernel-target-roster",
            "integer-interval-distance",
            CrossSizeMapKind.QUOTIENT_INVARIANT_SUMMARY,
        ),
        (
            "matter-response",
            "matter-response-calibration",
            "response-calibration",
            "normalized-scalar-map",
            "exact-matter-target-roster",
            "calibration-interval-distance",
            CrossSizeMapKind.EXACT_SCALAR_NORMALIZATION,
        ),
        (
            "projector-overlap",
            "role-projector-principal-angles",
            "projector-principal-angles",
            "same-q-hs-ambient-map",
            "exact-projector-target-roster",
            "maximum-principal-angle",
            CrossSizeMapKind.NOT_APPLICABLE,
        ),
        (
            "spectral-bands",
            "normalized-band-wasserstein",
            "spectral-wasserstein",
            "quotient-normalized-spectrum",
            "exact-spectral-target-roster",
            "wasserstein-band-distance",
            CrossSizeMapKind.QUOTIENT_INVARIANT_SUMMARY,
        ),
    )
    faces = tuple(
        MatrixResponsePropertyFace(
            face_id=f"six-matrix-response.face.{suffix}",
            property_kind_id=f"six-matrix-response.property.{property_kind}",
            method_id=f"six-matrix-response.method.{method}",
            ambient_map_id=f"six-matrix-response.ambient-map.{ambient}",
            target_roster_rule_id=f"six-matrix-response.roster.{roster}",
            metric_id=f"six-matrix-response.metric.{metric}",
            uncertainty_operation_id="six-matrix-response.uncertainty.panel-bootstrap-plus-numerical-envelope",
            categorical_companion_id=f"six-matrix-response.categorical.{suffix}",
            decisive_falsifier_ids=(
                f"six-matrix-response.falsifier.{suffix}-direction",
                f"six-matrix-response.falsifier.{suffix}-support",
                f"six-matrix-response.falsifier.{suffix}-uncertainty",
            ),
            cross_q_applicability=cross_size,
        )
        for suffix, property_kind, method, ambient, roster, metric, cross_size in definitions
    )
    return MatrixResponseStructuralFacePlan(
        plan_id="six-matrix-response.structural-face-plan",
        faces=tuple(sorted(faces, key=lambda value: value.face_id)),
        complete_unit_roster_rule_id="six-matrix-response.structural.complete-panel-roster",
        evidence_dependence_id="six-matrix-response.same-implementation-resample",
        prediction_cutoff_id="six-matrix-response.cutoff.before-protected-future",
        supported_terminal_id="matrix-response-study.terminal.structural-supported",
        opposed_terminal_id="matrix-response-study.terminal.structural-opposed",
        unevaluable_terminal_id="matrix-response-study.terminal.structural-unevaluable",
        missing_terminal_id="matrix-response-study.terminal.structural-missing",
        optional_admission_and_controller_evaluation_independent=True,
    )


def _action_grammars() -> tuple[MatrixResponseActionGrammar, MatrixResponseActionGrammar]:
    outer_values = (
        ("hold", "0", "0", False, True),
        ("x-down", str(-_TWO_THIRDS), "0", False, False),
        ("x-down-y-up", str(-_TWO_THIRDS), str(_TWO_THIRDS), True, False),
        ("x-up", str(_TWO_THIRDS), "0", False, False),
        ("x-up-y-down", str(_TWO_THIRDS), str(-_TWO_THIRDS), True, False),
        ("xy-down", str(-_TWO_THIRDS), str(-_TWO_THIRDS), True, False),
        ("xy-up", str(_TWO_THIRDS), str(_TWO_THIRDS), True, False),
        ("y-down", "0", str(-_TWO_THIRDS), False, False),
        ("y-up", "0", str(_TWO_THIRDS), False, False),
    )
    outer_primitives = tuple(
        MatrixResponsePrimitiveAction(
            primitive_id=f"six-matrix-response.outer-primitive.{suffix}",
            domain=MatrixResponseActionDomain.PARENT_COUPLING,
            delta_x=Decimal(dx),
            delta_y=Decimal(dy),
            simultaneous_channels=simultaneous,
            measured_hold=hold,
            native_unit="dimensionless-alpha-tilde",
            native_frame="anonymous-parent-factor-frame",
        )
        for suffix, dx, dy, simultaneous, hold in outer_values
    )
    outer = MatrixResponseActionGrammar(
        chart_id="six-matrix-response.chart.outer",
        domain=MatrixResponseActionDomain.PARENT_COUPLING,
        primitives=tuple(sorted(outer_primitives, key=lambda value: value.primitive_id)),
        maximum_active_depth=3,
        allow_all_ordered_active_compositions=True,
        hold_standalone_only=True,
        maximum_occurrences_per_word=6,
        ramp_steps=256,
        dwell_steps=512,
        qualification_steps=256,
        maximum_rate_per_step=_TWO_THIRDS / Decimal(256),
        generalized_work_includes_explicit_parameter_term=True,
    )
    inner_primitives = tuple(
        MatrixResponsePrimitiveAction(
            primitive_id=f"six-matrix-response.inner-primitive.{suffix}",
            domain=MatrixResponseActionDomain.LOWER_WORLD_FORCE,
            delta_x=Decimal(value),
            delta_y=Decimal(0),
            simultaneous_channels=False,
            measured_hold=hold,
            native_unit="dimensionless-force",
            native_frame="anonymous-hermitian-mode-frame",
        )
        for suffix, value, hold in (
            ("hold", "0", True),
            ("neg", "-0.1", False),
            ("pos", "0.1", False),
        )
    )
    inner = MatrixResponseActionGrammar(
        chart_id="six-matrix-response.chart.inner.template",
        domain=MatrixResponseActionDomain.LOWER_WORLD_FORCE,
        primitives=tuple(sorted(inner_primitives, key=lambda value: value.primitive_id)),
        maximum_active_depth=2,
        allow_all_ordered_active_compositions=True,
        hold_standalone_only=True,
        maximum_occurrences_per_word=2,
        ramp_steps=128,
        dwell_steps=256,
        qualification_steps=256,
        maximum_rate_per_step=Decimal("0.1") / Decimal(128),
        generalized_work_includes_explicit_parameter_term=True,
    )
    return outer, inner


def _study_authoring() -> MatrixResponseOuterStudyAuthoringConfig:
    outer, inner = _action_grammars()
    return MatrixResponseOuterStudyAuthoringConfig(
        config_id="six-matrix-response.outer-programme-authoring-config",
        config_version="1.0.0",
        law_family_id="six-matrix-response.law-family.l-up.parent",
        outer_action_grammar=outer,
        inner_action_grammar=inner,
        numerical_member_ids=("six-matrix-response.member.primary-dt", "six-matrix-response.member.secondary-half-dt"),
        candidate_version_ids=("six-matrix-response.candidate.nonlinear-local",),
        outer_support_cell_ids=(
            "six-matrix-response.support.00.joint-increasing-coupling",
            "six-matrix-response.support.00.joint-decreasing-coupling",
            "six-matrix-response.support.01.joint-increasing-coupling",
            "six-matrix-response.support.01.joint-decreasing-coupling",
            "six-matrix-response.support.10.joint-increasing-coupling",
            "six-matrix-response.support.10.joint-decreasing-coupling",
            "six-matrix-response.support.11.joint-increasing-coupling",
            "six-matrix-response.support.11.joint-decreasing-coupling",
        ),
        inner_support_cell_ids=(
            "six-matrix-response.inner-support.history-a-target-high",
            "six-matrix-response.inner-support.history-a-target-low",
            "six-matrix-response.inner-support.history-b-target-high",
            "six-matrix-response.inner-support.history-b-target-low",
        ),
        outer_admission_cartesian_count=9360,
        inner_admission_cartesian_count_per_law=56,
        admission_gate_ids=(
            "six-matrix-response.programme-admission-gate.authority",
            "six-matrix-response.programme-admission-gate.dynamics",
            "six-matrix-response.programme-admission-gate.effort",
            "six-matrix-response.programme-admission-gate.preservation",
            "six-matrix-response.programme-admission-gate.reachability",
            "six-matrix-response.programme-admission-gate.sink",
            "six-matrix-response.programme-admission-gate.target",
            "six-matrix-response.programme-admission-gate.uncertainty",
            "six-matrix-response.programme-admission-gate.validity",
        ),
        controller_study_schema='empirical-lawhood/planning/admission-controller-study',
        controller_compiler_schema='empirical-lawhood/runtime/compiled-admission-controller-study',
        controller_runtime_id="controller-runtime.sole-owner",
        arbitrary_chart_binder_id="six-matrix-response.programme-author.arbitrary-chart",
        expected_study_identity_rule_id="six-matrix-response.programme.canonical-fingerprint",
        grants_authority=False,
    )


def _prospective_topology() -> MatrixResponseProspectiveEvaluationTopology:
    definitions = (
        ("maximin", MatrixResponseOuterArmDisposition.COMPARATOR_ONLY, False),
        ("measured-hold", MatrixResponseOuterArmDisposition.HOLD_CONTROL, False),
        ("stratified-random", MatrixResponseOuterArmDisposition.COMPARATOR_ONLY, False),
        ("typed-constitutive", MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME, True),
        ("untyped-mpc", MatrixResponseOuterArmDisposition.COMPARATOR_ONLY, False),
    )
    arms = tuple(
        MatrixResponseOuterArm(
            arm_id=f"six-matrix-response.arm.{suffix}",
            disposition=disposition,
            information_contract_id=f"six-matrix-response.information-contract.{suffix}",
            capacity_contract_id=f"six-matrix-response.capacity-contract.{suffix}",
            law_family_id=(
                "six-matrix-response.law-family.l-up.parent"
                if disposition is MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME
                else None
            ),
            study_id_rule=(
                "six-matrix-response.programme.typed-constitutive-by-programme-admission"
                if disposition is MatrixResponseOuterArmDisposition.QUALIFIED_PROSPECTIVE_PROGRAMME
                else None
            ),
            inner_controller_use_applicable=inner,
        )
        for suffix, disposition, inner in definitions
    )
    return MatrixResponseProspectiveEvaluationTopology(
        topology_id="six-matrix-response.prospective-evaluation-topology",
        physical_panel_count=80,
        target_constitution_ids=("00", "01", "10", "11"),
        panels_per_target_constitution=20,
        outer_arms=tuple(sorted(arms, key=lambda value: value.arm_id)),
        possible_parent_action_word_count=585,
        contingent_template_rule_id="six-matrix-response.prospective-evaluation.contingent-panel-arm-action",
        activation_rule_id="six-matrix-response.prospective-evaluation.post-commit-bytewise-stratum",
        within_stratum_role_ids=("efficacy", "hold-control", "reserve"),
        minimum_efficacy_units_per_stratum=2,
        minimum_hold_controls_per_stratum=1,
        minimum_reserves_per_stratum=1,
        below_minimum_terminal_id="matrix-response-study.terminal.controller-use-stratum-unevaluable",
        matched_hold_rule_id="six-matrix-response.hold.same-checkpoint-common-random-numbers",
        common_random_number_rule_id="six-matrix-response.rng.panel-purpose-arm",
        sealed_reference_rule_id="six-matrix-response.reference.sealed-before-action",
        inner_constitution_ids=("01", "10", "11"),
        inner_modes_per_constitution=4,
        inner_action_hold_child_per_applicable_cell=True,
        zero_constitution_inner_terminal_id="matrix-response-study.terminal.inner-controller-use-nonattempt.00",
        panel_is_sole_inference_unit=True,
        frozen_before_admission_outcome_visibility=True,
        grants_authority=False,
    )


def _reducer() -> MatrixResponsePairedPanelReducerConfig:
    return MatrixResponsePairedPanelReducerConfig(
        config_id="six-matrix-response.paired-panel-reducer-config",
        independent_unit_id="six-matrix-response.unit.protected-panel",
        panel_count=80,
        outer_arm_ids=(
            "six-matrix-response.arm.maximin",
            "six-matrix-response.arm.measured-hold",
            "six-matrix-response.arm.stratified-random",
            "six-matrix-response.arm.typed-constitutive",
            "six-matrix-response.arm.untyped-mpc",
        ),
        primary_contrast_id="six-matrix-response.contrast.typed-vs-capacity-matched-untyped",
        primary_contrast_direction="greater",
        paired_binary_interval_id="six-matrix-response.interval.exact-mcnemar-score",
        continuous_interval_id="six-matrix-response.interval.whole-panel-bootstrap",
        bootstrap_resample_unit="whole-panel",
        bootstrap_replicates=19_999,
        familywise_alpha=Decimal("0.05"),
        multiplicity_method_id="six-matrix-response.multiplicity.holm-six-gates",
        missingness_rule_id="six-matrix-response.missing.intent-to-treat-failure-or-unevaluable",
        invalidity_rule_id="six-matrix-response.invalid.noncompensating-denominator-stop",
        intent_to_treat=True,
    )


def _protected_access() -> ProtectedObservableAccessManifest:
    excluded = (
        "six-matrix-response.field.checkpoint-parity",
        "six-matrix-response.field.coarse-stationarity",
        "six-matrix-response.field.delivery-free-validity",
        "six-matrix-response.field.nonfinite-count",
        "six-matrix-response.field.preparation-success",
    )
    protected = (
        "six-matrix-response.field.canonical-constitution-label",
        "six-matrix-response.field.interaction-response",
        "six-matrix-response.field.kernel-dimensions",
        "six-matrix-response.field.matter-response",
        "six-matrix-response.field.role-projectors",
        "six-matrix-response.field.spectral-bands",
        "six-matrix-response.field.target-attainment",
    )
    policy = (
        "six-matrix-response.field.current-scaled-couplings",
        "six-matrix-response.field.fast-validity",
        "six-matrix-response.field.history-summary",
        "six-matrix-response.field.normalized-radii",
        "six-matrix-response.field.operational-target-descriptor",
        "six-matrix-response.field.support-margin",
    )
    rules = (
        ProtectedFieldAccessRule(
            rule_id="six-matrix-response.access.adjudicator-after-reveal",
            principal_id="six-matrix-response.principal.adjudicator",
            field_ids=protected,
            access_mode="read",
            earliest_stage_id="six-matrix-response.stage.authorized-reveal",
        ),
        ProtectedFieldAccessRule(
            rule_id="six-matrix-response.access.excluded-q4-qualifier",
            principal_id="six-matrix-response.principal.excluded-q4-qualifier",
            field_ids=excluded,
            access_mode="read",
            earliest_stage_id="six-matrix-response.stage.q4-excluded-qualification",
        ),
        ProtectedFieldAccessRule(
            rule_id="six-matrix-response.access.prediction-policy",
            principal_id="six-matrix-response.principal.prediction-policy",
            field_ids=policy,
            access_mode="read",
            earliest_stage_id="six-matrix-response.stage.structural-prediction-prediction",
        ),
        ProtectedFieldAccessRule(
            rule_id="six-matrix-response.access.sealed-writer",
            principal_id="six-matrix-response.principal.sealed-writer",
            field_ids=protected,
            access_mode="write-only",
            earliest_stage_id="six-matrix-response.stage.prospective-execution-execution",
        ),
    )
    return ProtectedObservableAccessManifest(
        manifest_id="six-matrix-response.protected-observable-access-manifest",
        access_claim=ProtectedAccessClaim.PROCEDURAL_BLINDNESS_ONLY,
        operating_system_boundary_id="six-matrix-response.boundary.single-owner-vfat-no-acl",
        rules=tuple(sorted(rules, key=lambda value: value.rule_id)),
        q4_excluded_allowed_field_ids=excluded,
        protected_outcome_field_ids=protected,
        prediction_policy_field_ids=policy,
        reveal_authority_id="six-matrix-response.authority.reveal-required",
        confidentiality_claimed=False,
        grants_authority=False,
    )


def _dataset(
    path: str,
    *,
    dtype: str,
    minimum: tuple[int, ...],
    maximum: tuple[int, ...],
    chunk: tuple[int, ...],
    unit: str,
    frame: str,
    clock: str,
    role: str,
) -> SixMatrixResponseArtifactDataset:
    suffix = path.strip("/").replace("/", "-")
    return SixMatrixResponseArtifactDataset(
        dataset_id=f"six-matrix-response.dataset.{suffix}",
        path=path,
        dtype=dtype,
        minimum_shape=minimum,
        maximum_shape=maximum,
        chunk_shape=chunk,
        fill_value_id="six-matrix-response.hdf5.fill.zero-by-dtype",
        native_unit=unit,
        frame_id=frame,
        clock_id=clock,
        key_role_id=role,
    )


def _artifact_profiles() -> tuple[SixMatrixResponseArtifactProfileConfig, ...]:
    checkpoint_datasets = tuple(
        sorted(
            (
                _dataset(
                    "/momentum/x",
                    dtype="<c16",
                    minimum=(3, 4, 4),
                    maximum=(3, 16, 16),
                    chunk=(1, 4, 4),
                    unit="dimensionless-momentum",
                    frame="six-matrix-response.frame.hidden-matrix-basis",
                    clock="six-matrix-response.clock.checkpoint-step",
                    role="six-matrix-response.key.momentum-x",
                ),
                _dataset(
                    "/momentum/y",
                    dtype="<c16",
                    minimum=(3, 4, 4),
                    maximum=(3, 16, 16),
                    chunk=(1, 4, 4),
                    unit="dimensionless-momentum",
                    frame="six-matrix-response.frame.hidden-matrix-basis",
                    clock="six-matrix-response.clock.checkpoint-step",
                    role="six-matrix-response.key.momentum-y",
                ),
                _dataset(
                    "/rng/state",
                    dtype="<u1",
                    minimum=(1,),
                    maximum=(4096,),
                    chunk=(1,),
                    unit="canonical-rng-state-byte",
                    frame="six-matrix-response.frame.rng-state",
                    clock="six-matrix-response.clock.checkpoint-step",
                    role="six-matrix-response.key.rng-state",
                ),
                _dataset(
                    "/state/x",
                    dtype="<c16",
                    minimum=(3, 4, 4),
                    maximum=(3, 16, 16),
                    chunk=(1, 4, 4),
                    unit="dimensionless-matrix-coordinate",
                    frame="six-matrix-response.frame.hidden-matrix-basis",
                    clock="six-matrix-response.clock.checkpoint-step",
                    role="six-matrix-response.key.state-x",
                ),
                _dataset(
                    "/state/y",
                    dtype="<c16",
                    minimum=(3, 4, 4),
                    maximum=(3, 16, 16),
                    chunk=(1, 4, 4),
                    unit="dimensionless-matrix-coordinate",
                    frame="six-matrix-response.frame.hidden-matrix-basis",
                    clock="six-matrix-response.clock.checkpoint-step",
                    role="six-matrix-response.key.state-y",
                ),
            ),
            key=lambda value: value.path,
        )
    )
    receiver_datasets = tuple(
        sorted(
            (
                _dataset(
                    "/fast/time",
                    dtype="<f8",
                    minimum=(1,),
                    maximum=(8192,),
                    chunk=(1,),
                    unit="dimensionless-langevin-time",
                    frame="six-matrix-response.frame.receiver-scalar",
                    clock="six-matrix-response.clock.fast-receiver",
                    role="six-matrix-response.key.fast-time",
                ),
                _dataset(
                    "/fast/value",
                    dtype="<f8",
                    minimum=(1, 12),
                    maximum=(8192, 12),
                    chunk=(1, 12),
                    unit="receiver-specific-normalized",
                    frame="six-matrix-response.frame.receiver-vector",
                    clock="six-matrix-response.clock.fast-receiver",
                    role="six-matrix-response.key.fast-values",
                ),
                _dataset(
                    "/spectral/eigenvalues",
                    dtype="<f8",
                    minimum=(1, 3, 16),
                    maximum=(512, 3, 256),
                    chunk=(1, 3, 16),
                    unit="normalized-laplacian-eigenvalue",
                    frame="six-matrix-response.frame.anonymous-spectrum",
                    clock="six-matrix-response.clock.spectral-receiver",
                    role="six-matrix-response.key.spectral-eigenvalues",
                ),
                _dataset(
                    "/spectral/time",
                    dtype="<f8",
                    minimum=(1,),
                    maximum=(512,),
                    chunk=(1,),
                    unit="dimensionless-langevin-time",
                    frame="six-matrix-response.frame.receiver-scalar",
                    clock="six-matrix-response.clock.spectral-receiver",
                    role="six-matrix-response.key.spectral-time",
                ),
            ),
            key=lambda value: value.path,
        )
    )
    compatibility_datasets = (
        _dataset(
            "/probe/value",
            dtype="<f8",
            minimum=(1, 4),
            maximum=(16, 4),
            chunk=(1, 4),
            unit="dimensionless-synthetic-probe",
            frame="six-matrix-response.frame.synthetic-compatibility",
            clock="six-matrix-response.clock.synthetic-step",
            role="six-matrix-response.key.synthetic-probe",
        ),
    )
    values = (
        SixMatrixResponseArtifactProfileConfig(
            config_id="six-matrix-response.artifact-profile.checkpoint",
            payload_schema='empirical-lawhood/composition/matrix-response-study/checkpoint-hdf5',
            media_type="application/x-hdf5",
            datasets=checkpoint_datasets,
            maximum_objects=9,
            maximum_dataset_elements=4096,
            maximum_total_elements=16384,
            maximum_decoded_bytes=1_073_741_824,
            maximum_shard_bytes=536_870_912,
            deterministic_writer_id="six-matrix-response.writer.checkpoint-hdf5",
            compression_filter_id="six-matrix-response.hdf5.compression.none",
            shuffle_filter_enabled=False,
            checksum_filter_id="six-matrix-response.hdf5.checksum.none",
            object_creation_order_id="six-matrix-response.hdf5.creation-order.lexicographic-path",
            root_track_object_times=False,
            dataset_track_object_times=False,
            group_object_time_policy_id=("six-matrix-response.hdf5.group-time-disable-request-byte-parity"),
            hdf5_library_version_policy_id="six-matrix-response.hdf5.library-lock-receipt-exact",
            libver_bounds_id="six-matrix-response.hdf5.libver-earliest-to-hdf5-1-14",
            one_writer_per_shard=True,
            extra_objects_forbidden=True,
            external_links_forbidden=True,
            variable_length_values_forbidden=True,
        ),
        SixMatrixResponseArtifactProfileConfig(
            config_id="six-matrix-response.artifact-profile.compatibility",
            payload_schema='empirical-lawhood/composition/matrix-response-study/compatibility-probe-hdf5',
            media_type="application/x-hdf5",
            datasets=compatibility_datasets,
            maximum_objects=4,
            maximum_dataset_elements=64,
            maximum_total_elements=64,
            maximum_decoded_bytes=1_073_741_824,
            maximum_shard_bytes=536_870_912,
            deterministic_writer_id="six-matrix-response.writer.compatibility-hdf5",
            compression_filter_id="six-matrix-response.hdf5.compression.none",
            shuffle_filter_enabled=False,
            checksum_filter_id="six-matrix-response.hdf5.checksum.none",
            object_creation_order_id="six-matrix-response.hdf5.creation-order.lexicographic-path",
            root_track_object_times=False,
            dataset_track_object_times=False,
            group_object_time_policy_id=("six-matrix-response.hdf5.group-time-disable-request-byte-parity"),
            hdf5_library_version_policy_id="six-matrix-response.hdf5.library-lock-receipt-exact",
            libver_bounds_id="six-matrix-response.hdf5.libver-earliest-to-hdf5-1-14",
            one_writer_per_shard=True,
            extra_objects_forbidden=True,
            external_links_forbidden=True,
            variable_length_values_forbidden=True,
        ),
        SixMatrixResponseArtifactProfileConfig(
            config_id="six-matrix-response.artifact-profile.receiver",
            payload_schema='empirical-lawhood/composition/matrix-response-study/receiver-hdf5',
            media_type="application/x-hdf5",
            datasets=receiver_datasets,
            maximum_objects=8,
            maximum_dataset_elements=393_216,
            maximum_total_elements=600_000,
            maximum_decoded_bytes=1_073_741_824,
            maximum_shard_bytes=536_870_912,
            deterministic_writer_id="six-matrix-response.writer.receiver-hdf5",
            compression_filter_id="six-matrix-response.hdf5.compression.none",
            shuffle_filter_enabled=False,
            checksum_filter_id="six-matrix-response.hdf5.checksum.none",
            object_creation_order_id="six-matrix-response.hdf5.creation-order.lexicographic-path",
            root_track_object_times=False,
            dataset_track_object_times=False,
            group_object_time_policy_id=("six-matrix-response.hdf5.group-time-disable-request-byte-parity"),
            hdf5_library_version_policy_id="six-matrix-response.hdf5.library-lock-receipt-exact",
            libver_bounds_id="six-matrix-response.hdf5.libver-earliest-to-hdf5-1-14",
            one_writer_per_shard=True,
            extra_objects_forbidden=True,
            external_links_forbidden=True,
            variable_length_values_forbidden=True,
        ),
    )
    return tuple(sorted(values, key=lambda value: value.config_id))


def _package_node(
    package_id: str,
    parents: tuple[str, ...],
    product_id: str,
    schemas: tuple[str, ...],
) -> MatrixResponseCampaignPackageNode:
    return MatrixResponseCampaignPackageNode(
        package_id=package_id,
        parent_package_ids=tuple(sorted(parents)),
        product_id=product_id,
        ordinary_stage_disposition_id="six-matrix-response.stage-disposition.explicit-applicability",
        required_extension_schema_ids=tuple(sorted(schemas)),
        grants_authority=False,
    )


def _package_dag() -> MatrixResponseCampaignPackageDag:
    source_schema = SixMatrixResponseSixMatrixSourceConfig.SCHEMA
    law_schema = MatrixResponseLawMethodConfig.SCHEMA
    structural_schema = MatrixResponseStructuralFacePlan.SCHEMA
    programme_schema = MatrixResponseOuterStudyAuthoringConfig.SCHEMA
    topology_schema = MatrixResponseProspectiveEvaluationTopology.SCHEMA
    reducer_schema = MatrixResponsePairedPanelReducerConfig.SCHEMA
    artifact_schema = SixMatrixResponseArtifactProfileConfig.SCHEMA
    access_schema = ProtectedObservableAccessManifest.SCHEMA
    nodes: list[MatrixResponseCampaignPackageNode] = [
        _package_node(
            "matrix-numerical-qualification-numerical", (), "six-matrix-response.product.numerical-qualification-numerical", (source_schema, artifact_schema)
        ),
        _package_node(
            "matrix-anisotropic-feasibility-substrate",
            ("matrix-numerical-qualification-numerical",),
            "six-matrix-response.product.anisotropic-feasibility-substrate",
            (source_schema, artifact_schema),
        ),
        _package_node(
            "six-matrix-response-q4-excluded-qualification",
            ("matrix-anisotropic-feasibility-substrate",),
            "six-matrix-response.product.q4-excluded",
            (source_schema, access_schema, artifact_schema),
        ),
        _package_node(
            "six-matrix-response-lup-qualification",
            ("matrix-anisotropic-feasibility-substrate",),
            "six-matrix-response.product.lup-laws",
            (law_schema, source_schema),
        ),
        _package_node(
            "six-matrix-response-ldown-qualification",
            ("matrix-anisotropic-feasibility-substrate",),
            "six-matrix-response.product.ldown-laws",
            (law_schema, source_schema),
        ),
        _package_node(
            "six-matrix-response-structural-development",
            ("six-matrix-response-ldown-qualification", "six-matrix-response-lup-qualification"),
            "six-matrix-response.product.structural-plan",
            (structural_schema, law_schema),
        ),
        _package_node(
            "matrix-outer-programme-admission",
            ("six-matrix-response-lup-qualification",),
            "six-matrix-response.product.outer-programme-admission-programme",
            (programme_schema, law_schema),
        ),
    ]
    inner_admission_ids: list[str] = []
    inner_controller_evaluation_ids: list[str] = []
    for constitution in ("01", "10", "11"):
        for mode in range(4):
            admission_config_id = f"matrix-inner-programme-admission-{constitution}-mode-{mode}"
            controller_evaluation_config_id = f"matrix-inner-prospective-evaluation-typed-{constitution}-mode-{mode}"
            inner_admission_ids.append(admission_config_id)
            inner_controller_evaluation_ids.append(controller_evaluation_config_id)
            nodes.append(
                _package_node(
                    admission_config_id,
                    ("six-matrix-response-ldown-qualification",),
                    f"six-matrix-response.product.inner-programme-admission.{constitution}.mode-{mode}",
                    (programme_schema, law_schema),
                )
            )
    nodes.extend(
        (
            _package_node(
                "six-matrix-response-protected-preparation",
                tuple(
                    (
                        "six-matrix-response-q4-excluded-qualification",
                        "six-matrix-response-structural-development",
                        "matrix-outer-programme-admission",
                        *inner_admission_ids,
                    )
                ),
                "six-matrix-response.product.protected-checkpoints",
                (source_schema, access_schema, artifact_schema, topology_schema),
            ),
            _package_node(
                "six-matrix-response-structural-prediction",
                ("six-matrix-response-protected-preparation", "six-matrix-response-structural-development"),
                "six-matrix-response.product.structural-predictions",
                (structural_schema, access_schema),
            ),
            _package_node(
                "matrix-outer-prospective-evaluation",
                ("matrix-outer-programme-admission", "six-matrix-response-protected-preparation", "six-matrix-response-structural-prediction"),
                "six-matrix-response.product.outer-prospective-evaluation",
                (topology_schema, reducer_schema, access_schema),
            ),
        )
    )
    for constitution in ("01", "10", "11"):
        for mode in range(4):
            admission_config_id = f"matrix-inner-programme-admission-{constitution}-mode-{mode}"
            controller_evaluation_config_id = f"matrix-inner-prospective-evaluation-typed-{constitution}-mode-{mode}"
            nodes.append(
                _package_node(
                    controller_evaluation_config_id,
                    (admission_config_id, "matrix-outer-prospective-evaluation"),
                    f"six-matrix-response.product.inner-prospective-evaluation.{constitution}.mode-{mode}",
                    (topology_schema, law_schema, reducer_schema),
                )
            )
    nodes.extend(
        (
            _package_node(
                "matrix-size-compatibility-transport",
                tuple(("matrix-outer-prospective-evaluation", *inner_controller_evaluation_ids)),
                "six-matrix-response.product.size-compatibility-transport",
                (structural_schema, topology_schema, reducer_schema),
            ),
            _package_node(
                "six-matrix-response-adjudication",
                ("matrix-size-compatibility-transport",),
                "six-matrix-response.product.terminal-adjudication",
                (structural_schema, reducer_schema, access_schema),
            ),
        )
    )
    return MatrixResponseCampaignPackageDag(
        dag_id="six-matrix-response.campaign-package-dag",
        nodes=tuple(sorted(nodes, key=lambda value: value.package_id)),
        candidate_role_ids=(
            "six-matrix-response.candidate-role.analysis",
            "six-matrix-response.candidate-role.evaluation",
            "six-matrix-response.candidate-role.reporter",
            "six-matrix-response.candidate-role.source",
        ),
        issued_package_schema='empirical-lawhood/api/issued-compilation-package-reference',
        run_plan_schema='empirical-lawhood/runtime/compiled-run-plan-reference',
        execution_plan_schema='empirical-lawhood/runtime/frozen-compilation-execution-plan',
        cross_package_binding_schema='empirical-lawhood/runtime/frozen-parent-input-binding',
        total_catalog_loss_reconstructible=True,
    )


def build_matrix_response_study_design_binding(lineage: MatrixResponseLineageInputs) -> MatrixResponseDesignBinding:
    """Build the Six-matrix response design from caller-supplied parent identities."""

    return MatrixResponseDesignBinding(
        design_id="six-matrix-response.design-binding",
        design_version="1.0.0",
        plan_id="six-matrix-response-platform",
        scientific_parent_sha256=lineage.sha256("scientific_parent_sha256"),
        metatheory_contract_sha256=lineage.sha256("metatheory_contract_sha256"),
        platform_acceptance_commit=lineage.git_commit("platform_acceptance_commit"),
        source_config=_source_config(),
        role_config=_role_config(),
        law_method_config=_law_method_config(),
        structural_plan=_structural_plan(),
        study_authoring=_study_authoring(),
        prospective_topology=_prospective_topology(),
        reducer=_reducer(),
        artifact_profiles=_artifact_profiles(),
        protected_access=_protected_access(),
        package_dag=_package_dag(),
        resource_envelope=MatrixResponseResourceEnvelope(
            envelope_id="six-matrix-response.resource-envelope",
            physical_cores=8,
            worker_processes=8,
            blas_threads_per_worker=1,
            memory_bytes=33_496_653_824,
            external_free_bytes_at_design=974_122_745_856,
            maximum_output_bytes=500_000_000_000,
            maximum_integration_steps=30_000_000,
            preferred_wall_seconds=Decimal(64_800),
            hard_wall_seconds=Decimal(86_400),
            benchmark_forecast_rule_id="six-matrix-response.resource.numerical-qualification-measured-complete-roster",
            over_ceiling_terminal_id="matrix-response-study.terminal.resource-ceiling-nonadmitted",
            contraction_allowed=False,
        ),
        stable_package_names=(
            'empirical_lawhood.adapters.composition.matrix_response_study',
            'empirical_lawhood.adapters.control.matrix_response_study',
            'empirical_lawhood.adapters.methods.matrix_response_study',
            'empirical_lawhood.adapters.simulators.six_matrix_response',
        ),
        planned_write_roots=lineage.locators("planned_write_roots"),
        external_artifact_root_id="external-root.operator-external-root-six-matrix-response",
        evidence_dependence_id="six-matrix-response.same-implementation-resample",
        maximum_claim_id="six-matrix-response.claim.simulator-local-support-limited-constitutive-closure",
        unresolved_scientific_decisions=(),
        grants_issue_authority=False,
        grants_execution_authority=False,
        grants_reveal_authority=False,
    )


__all__ = ['build_matrix_response_study_design_binding']
