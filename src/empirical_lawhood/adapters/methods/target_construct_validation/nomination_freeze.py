"""Factual outcome-blind outcome-blind nomination nomination freeze for target construct validation."""

from __future__ import annotations

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .native_dossier import TargetConstructValidationIndependenceDossier, TargetConstructValidationTargetNativeDossier
from .nomination import TargetConstructValidationCandidate, TargetConstructValidationCandidateFeasibility, TargetConstructValidationCandidateRegistry, TargetConstructValidationContaminationLedger, TargetConstructValidationNominationFreeze, TargetConstructValidationTargetEnvelope, select_nominees


BRIAN2_SDIST_SHA256 = "e4ddeec1cc23f37542e3d067acea2770623c264fbded144ca6d7a1b6cc3ca397"
FIPY_SDIST_SHA256 = "7d43ad8a4f61984deccbf12ec9fd39849e8f36943c233a9e2d772810b491f0df"
CANTERA_SDIST_SHA256 = "d811175e052251a7d2ea70f60fa19a5b8a61bf32d3b5565d2404b22ba57d8d41"


def _candidates() -> tuple[TargetConstructValidationCandidate, ...]:
    return (
        TargetConstructValidationCandidate(
            candidate_id="candidate.brian2-lif",
            domain_id="computational-neuroscience",
            solver_family_id="brian2-clock-driven-euler",
            generator_family_id="brian2",
            task_statement=(
                "Estimate the firing and membrane-potential response of a reset leaky "
                "integrate-and-fire preparation to bounded current steps after rest or "
                "a native conditioning pulse."
            ),
            generator_provenance=(
                "Brian2 2.9.0, independently maintained by the Brian team; the clock-driven "
                "spiking-neuron simulator and governing equation API predate target construct validation."
            ),
            specification_provenance=(
                "The leaky integrate-and-fire current-response and firing-rate task is a "
                'standard Brian2 documented example, not a structural recurrence-derived task.'
            ),
            preparation_unit_definition=(
                "One independently seeded, fully reset neuron trial block containing the "
                "complete frozen condition roster."
            ),
            native_intervention_definition="A bounded injected-current step in picoampere.",
            action_realization_chain=(
                "requested amplitude -> bounds acceptance -> TimedArray current -> recorded "
                "realized current at the neuron clock"
            ),
            receiver_definitions=(
                "post-step mean membrane potential in millivolt",
                "spike count over the post-step observation interval",
            ),
            horizon_definitions=("80 millisecond post-step horizon",),
            causal_cutoff_definition="All features end at the current-step start clock.",
            likely_denominator_factor_ids=("membrane-time-constant",),
            likely_history_factor_ids=("conditioning-pulse-history",),
            source_cost_class=1,
            execution_cost_class=1,
            prior_project_exposure_ids=(),
            prior_investigator_exposure_ids=(),
            construct_threat_ids=(
                "current-command-echo",
                "deterministic-threshold-tautology",
            ),
            maximum_claim="Finite Brian2 LIF simulator-local categorical response evidence.",
            construct_non_tautological=True,
            task_specification_independent=True,
            new_generator_family=True,
            complete_action_realization_observable=True,
            independent_units_defensible=True,
            nontrivial_contrasts_available=True,
            sealed_evaluation_feasible=True,
            joint_power_feasible=True,
            source_metadata_complete=True,
            recurrence_eligible=True,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        TargetConstructValidationCandidate(
            candidate_id="candidate.cantera-cstr",
            domain_id="chemical-reactor-kinetics",
            solver_family_id="sundials-cvodes",
            generator_family_id="cantera",
            task_statement=(
                "Estimate temperature and species response of an open continuously stirred "
                "reactor to bounded inlet-flow steps at fixed residence-time conditions."
            ),
            generator_provenance=(
                "Cantera 3.2.0, independently maintained open chemical-kinetics software."
            ),
            specification_provenance=(
                "The open CSTR residence-time response is an official Cantera reactor example."
            ),
            preparation_unit_definition=(
                "One fully initialized reactor-network trajectory under one frozen inlet state."
            ),
            native_intervention_definition="A bounded mass-flow-controller coefficient step.",
            action_realization_chain=(
                "requested flow -> nonnegative acceptance -> controller coefficient -> "
                "reported mass flow"
            ),
            receiver_definitions=(
                "reactor temperature in kelvin",
                "selected product mole fraction",
            ),
            horizon_definitions=("five residence-time transient horizon",),
            causal_cutoff_definition="All features end before the inlet-flow step.",
            likely_denominator_factor_ids=("residence-time",),
            likely_history_factor_ids=("reactor-initial-state",),
            source_cost_class=1,
            execution_cost_class=2,
            prior_project_exposure_ids=(),
            prior_investigator_exposure_ids=(),
            construct_threat_ids=("flow-command-echo", "ignition-regime-instability"),
            maximum_claim="Finite Cantera CSTR simulator-local categorical response evidence.",
            construct_non_tautological=True,
            task_specification_independent=True,
            new_generator_family=True,
            complete_action_realization_observable=True,
            independent_units_defensible=True,
            nontrivial_contrasts_available=True,
            sealed_evaluation_feasible=True,
            joint_power_feasible=True,
            source_metadata_complete=True,
            recurrence_eligible=True,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        TargetConstructValidationCandidate(
            candidate_id="candidate.fipy-diffusion",
            domain_id="transport-pde",
            solver_family_id="fipy-scipy-sparse-lu",
            generator_family_id="fipy",
            task_statement=(
                "Estimate downstream field mass and gradient response of a reset transient "
                "diffusion-decay preparation to bounded localized source pulses."
            ),
            generator_provenance=(
                "FiPy 4.0.3, independently maintained by NIST as a finite-volume PDE solver."
            ),
            specification_provenance=(
                "Transient diffusion with standard source terms is a documented FiPy task "
                "that predates target construct validation."
            ),
            preparation_unit_definition=(
                "One independently seeded, fully reset field trajectory block containing the "
                "complete frozen condition roster."
            ),
            native_intervention_definition="A bounded localized scalar-source pulse.",
            action_realization_chain=(
                "requested source -> bounds acceptance -> finite-volume source vector -> "
                "recorded integrated applied source"
            ),
            receiver_definitions=(
                "downstream-window mean field concentration",
                "right-boundary field gradient",
            ),
            horizon_definitions=("0.5 second post-pulse horizon",),
            causal_cutoff_definition="All features end at the localized-source onset.",
            likely_denominator_factor_ids=("diffusivity",),
            likely_history_factor_ids=("residual-field-history",),
            source_cost_class=1,
            execution_cost_class=1,
            prior_project_exposure_ids=(),
            prior_investigator_exposure_ids=(),
            construct_threat_ids=("mesh-dependent-category", "source-command-echo"),
            maximum_claim="Finite FiPy transport simulator-local categorical response evidence.",
            construct_non_tautological=True,
            task_specification_independent=True,
            new_generator_family=True,
            complete_action_realization_observable=True,
            independent_units_defensible=True,
            nontrivial_contrasts_available=True,
            sealed_evaluation_feasible=True,
            joint_power_feasible=True,
            source_metadata_complete=True,
            recurrence_eligible=True,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )


def _native_dossiers(
    registry: TargetConstructValidationCandidateRegistry,
) -> tuple[TargetConstructValidationTargetNativeDossier, ...]:
    candidates = {value.candidate_id: value for value in registry.candidates}
    brian = candidates["candidate.brian2-lif"]
    fipy = candidates["candidate.fipy-diffusion"]
    return (
        TargetConstructValidationTargetNativeDossier(
            dossier_id='dossier.target-brian2.leaky-integrate-and-fire',
            target_id='target-brian2.leaky-integrate-and-fire',
            source_candidate=ObjectIdentity.from_record(brian.candidate_id, brian),
            domain_id=brian.domain_id,
            generator_family_id=brian.generator_family_id,
            task_statement=brian.task_statement,
            preparation_unit_id="brian2-reset-trial-block",
            preparation_unit_definition=brian.preparation_unit_definition,
            causal_cutoff_definition=brian.causal_cutoff_definition,
            native_action_ids=("action.high-current", "action.hold", "action.low-current"),
            hold_action_id="action.hold",
            receiver_ids=("receiver.mean-membrane-potential", "receiver.spike-count"),
            receiver_direction_definition=(
                "Increasing values mean greater depolarization or firing; the two receivers "
                "remain separate and are never compensated."
            ),
            horizon_ids=("horizon.post-step-80ms",),
            native_unit_ids=("millisecond", "millivolt", "picoampere", "spike-count"),
            native_frame_ids=("neuron-clock", "neuron-rest-potential-frame"),
            baseline_policy_id="baseline.lif-monotone-current-response",
            decisive_falsifier_ids=(
                "falsifier.action-current-not-realized",
                "falsifier.receiver-direct-command-echo",
                "falsifier.reset-memory-cross-unit",
            ),
            requested_action_recorded=True,
            accepted_action_recorded=True,
            applied_action_recorded=True,
            realized_action_recorded=True,
            uses_structural_recurrence_role_vocabulary=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        TargetConstructValidationTargetNativeDossier(
            dossier_id='dossier.target-fipy.source-diffusion',
            target_id='target-fipy.source-diffusion',
            source_candidate=ObjectIdentity.from_record(fipy.candidate_id, fipy),
            domain_id=fipy.domain_id,
            generator_family_id=fipy.generator_family_id,
            task_statement=fipy.task_statement,
            preparation_unit_id="fipy-reset-field-trajectory-block",
            preparation_unit_definition=fipy.preparation_unit_definition,
            causal_cutoff_definition=fipy.causal_cutoff_definition,
            native_action_ids=("action.high-source", "action.hold", "action.low-source"),
            hold_action_id="action.hold",
            receiver_ids=("receiver.downstream-mean", "receiver.right-gradient"),
            receiver_direction_definition=(
                "Increasing downstream mean and outward right gradient are positive in their "
                "own native gauges; neither receiver compensates the other."
            ),
            horizon_ids=("horizon.post-source-0p5s",),
            native_unit_ids=("field-concentration", "field-gradient", "second", "source-rate"),
            native_frame_ids=("one-dimensional-domain", "simulation-clock"),
            baseline_policy_id="baseline.diffusion-monotone-source-response",
            decisive_falsifier_ids=(
                "falsifier.applied-source-integral-mismatch",
                "falsifier.receiver-source-vector-echo",
                "falsifier.reset-field-cross-unit",
            ),
            requested_action_recorded=True,
            accepted_action_recorded=True,
            applied_action_recorded=True,
            realized_action_recorded=True,
            uses_structural_recurrence_role_vocabulary=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )


def _independence_dossiers(
    native: tuple[TargetConstructValidationTargetNativeDossier, ...],
) -> tuple[TargetConstructValidationIndependenceDossier, ...]:
    by_target = {value.target_id: value for value in native}
    brian = by_target['target-brian2.leaky-integrate-and-fire']
    fipy = by_target['target-fipy.source-diffusion']
    return (
        TargetConstructValidationIndependenceDossier(
            dossier_id='independence.target-brian2.leaky-integrate-and-fire',
            target_id=brian.target_id,
            source_candidate=brian.source_candidate,
            generator_family_id=brian.generator_family_id,
            authorship_evidence="Brian2 is maintained by the external Brian development team.",
            chronology_evidence=(
                "Brian2 and its documented LIF current-response examples predate target construct validation."
            ),
            solver_family_evidence=(
                "Clock-driven Brian2 Euler state update, distinct from FiPy finite-volume "
                "assembly and SciPy sparse LU."
            ),
            generator_not_authored_for_construct_validation=True,
            generator_predates_construct_validation=True,
            domain_independent_of_other_target=True,
            solver_family_independent_of_other_target=True,
            generator_family_independent_of_other_target=True,
            direct_echo_of_structural_recurrence_fixture=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        TargetConstructValidationIndependenceDossier(
            dossier_id='independence.target-fipy.source-diffusion',
            target_id=fipy.target_id,
            source_candidate=fipy.source_candidate,
            generator_family_id=fipy.generator_family_id,
            authorship_evidence="FiPy is maintained externally by NIST contributors.",
            chronology_evidence=(
                "FiPy transient diffusion/source examples and finite-volume implementation "
                "predate target construct validation."
            ),
            solver_family_evidence=(
                "FiPy finite-volume assembly with SciPy sparse LU, distinct from Brian2's "
                "clock-driven neuron state updater."
            ),
            generator_not_authored_for_construct_validation=True,
            generator_predates_construct_validation=True,
            domain_independent_of_other_target=True,
            solver_family_independent_of_other_target=True,
            generator_family_independent_of_other_target=True,
            direct_echo_of_structural_recurrence_fixture=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )


def frozen_nomination() -> TargetConstructValidationNominationFreeze:
    """Return the deterministic candidate freeze without importing either simulator."""

    registry = TargetConstructValidationCandidateRegistry(
        registry_id="target-construct-validation.candidate-registry",
        candidates=_candidates(),
        frozen_before_candidate_outcomes=True,
        candidate_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    ledger = TargetConstructValidationContaminationLedger(
        ledger_id="target-construct-validation.contamination-ledger",
        candidate_registry=ObjectIdentity.from_record(registry.registry_id, registry),
        entries=(),
        transitive_exposure_audited=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    nomination = select_nominees(registry)
    native = _native_dossiers(registry)
    independence = _independence_dossiers(native)
    feasibility = (
        TargetConstructValidationCandidateFeasibility(
            feasibility_id="feasibility.candidate.brian2-lif",
            candidate_id="candidate.brian2-lif",
            package_name="brian2",
            package_version="2.9.0",
            source_distribution_sha256=BRIAN2_SDIST_SHA256,
            documentation_identity="brian2-official-docs-2.9-and-pypi-release-2.9.0",
            license_identity="cecill-2.1",
            python_compatible=True,
            action_api_static_readable=True,
            reset_semantics_static_readable=True,
            receiver_api_static_readable=True,
            complete_unit_static_resolvable=True,
            generator_contacted=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        TargetConstructValidationCandidateFeasibility(
            feasibility_id="feasibility.candidate.fipy-diffusion",
            candidate_id="candidate.fipy-diffusion",
            package_name="fipy",
            package_version="4.0.3",
            source_distribution_sha256=FIPY_SDIST_SHA256,
            documentation_identity="nist-fipy-official-docs-4.0.3-and-pypi-release-4.0.3",
            license_identity="nist-software-terms-public-domain",
            python_compatible=True,
            action_api_static_readable=True,
            reset_semantics_static_readable=True,
            receiver_api_static_readable=True,
            complete_unit_static_resolvable=True,
            generator_contacted=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )
    envelopes = (
        TargetConstructValidationTargetEnvelope(
            envelope_id='envelope.target-brian2.leaky-integrate-and-fire',
            target_slot_id="target-n1",
            candidate_id="candidate.brian2-lif",
            source_package_name="brian2",
            source_package_version="2.9.0",
            source_distribution_sha256=BRIAN2_SDIST_SHA256,
            maximum_parallel_units=4,
            cpu_cores_per_unit=1,
            memory_bytes_per_unit=536_870_912,
            unit_timeout_seconds=120,
            phase_wall_time_seconds=14_400,
            external_storage_root_id="external-root.target-construct-validation.brian2-lif",
            source_acquisition_authority_id="authority.owner.target-construct-validation.brian2-source",
            execution_authority_id="authority.owner.target-construct-validation.brian2-execution",
            reveal_authority_id="authority.owner.target-construct-validation.brian2-reveal",
            evaluator_identity="evaluator.target-construct-validation.brian2-sealed",
            network_after_source_binding=False,
            live_or_physical_authority_granted=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
        TargetConstructValidationTargetEnvelope(
            envelope_id='envelope.target-fipy.source-diffusion',
            target_slot_id="target-n2",
            candidate_id="candidate.fipy-diffusion",
            source_package_name="fipy",
            source_package_version="4.0.3",
            source_distribution_sha256=FIPY_SDIST_SHA256,
            maximum_parallel_units=4,
            cpu_cores_per_unit=1,
            memory_bytes_per_unit=536_870_912,
            unit_timeout_seconds=120,
            phase_wall_time_seconds=14_400,
            external_storage_root_id="external-root.target-construct-validation.fipy-diffusion",
            source_acquisition_authority_id="authority.owner.target-construct-validation.fipy-source",
            execution_authority_id="authority.owner.target-construct-validation.fipy-execution",
            reveal_authority_id="authority.owner.target-construct-validation.fipy-reveal",
            evaluator_identity="evaluator.target-construct-validation.fipy-sealed",
            network_after_source_binding=False,
            live_or_physical_authority_granted=False,
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )
    return TargetConstructValidationNominationFreeze(
        freeze_id="target-construct-validation.nomination-freeze.outcome-blind-nomination",
        registry=registry,
        contamination_ledger=ledger,
        native_dossiers=native,
        independence_dossiers=independence,
        feasibility_records=feasibility,
        nomination_result=nomination,
        target_envelopes=envelopes,
        metadata_source_ids=(
            "metadata.brian2-docs-2.9",
            "metadata.brian2-pypi-2.9.0",
            "metadata.cantera-docs-3.2.0",
            "metadata.cantera-pypi-3.2.0",
            "metadata.fipy-nist-docs-4.0.3",
            "metadata.fipy-pypi-4.0.3",
        ),
        frozen_before_candidate_outcomes=True,
        candidate_generator_contact_count=0,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    "BRIAN2_SDIST_SHA256",
    "CANTERA_SDIST_SHA256",
    "FIPY_SDIST_SHA256",
    "frozen_nomination",
]
