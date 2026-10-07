"""Pure phase services joining simulator morphism challenges records without hiding scientific steps."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetUnitScientificInput

from dataclasses import dataclass
from hashlib import sha256
import secrets

from empirical_lawhood.adapters.methods.simulator_morphism_challenges.evaluator import adjudicate_unit_scale
from empirical_lawhood.adapters.methods.simulator_morphism_challenges.observer import nominate_from_history, observe_and_nominate, observe_history
from empirical_lawhood.adapters.simulators.rc_ladder_morphism_challenges.generator import generate_outcomes
from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .array_io import PackedArrays, simulator_morphism_challenges_array_semantics, logical_arrays_digest, pack_arrays, unpack_arrays
from .contracts import SimulatorMorphismChallengeConfig, SimulatorMorphismChallengeChallengeNomination, SimulatorMorphismChallengeGate, SimulatorMorphismChallengeHistoryRankForecast, SimulatorMorphismChallengeMethodFreeze, SimulatorMorphismChallengePhase, SimulatorMorphismChallengeSeedRosterCommitment
from .descriptors import canary_unit_ids, development_unit_ids, family_from_unit_id, generate_descriptor, reserved_non_evaluation_seed_digests
from .runtime_contracts import SimulatorMorphismChallengeAdjudicationBundle, SimulatorMorphismChallengeCanaryReport, SimulatorMorphismChallengeDenominatorBundle, SimulatorMorphismChallengeDevelopmentGate, SimulatorMorphismChallengeDevelopmentGateCount, SimulatorMorphismChallengeDevelopmentLedger, SimulatorMorphismChallengeEvaluationDesignFreeze, SimulatorMorphismChallengeGeneratorBundle, SimulatorMorphismChallengeHistoryBundle, SimulatorMorphismChallengeNominationFreeze, SimulatorMorphismChallengeObserverBundle, SimulatorMorphismChallengeRequestedUnitLedger, SimulatorMorphismChallengeSeedEntry, SimulatorMorphismChallengeSeedRoster


@dataclass(frozen=True, slots=True)
class ObserverBundleExecution:
    bundle: SimulatorMorphismChallengeObserverBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class HistoryBundleExecution:
    bundle: SimulatorMorphismChallengeHistoryBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class GeneratorBundleExecution:
    bundle: SimulatorMorphismChallengeGeneratorBundle
    packed_arrays: PackedArrays


def requested_unit_ledger(config: SimulatorMorphismChallengeConfig) -> SimulatorMorphismChallengeRequestedUnitLedger:
    return SimulatorMorphismChallengeRequestedUnitLedger(
        ledger_id=f"ledger.simulator-morphism-challenges.{config.phase.value.lower()}.requested-units",
        phase=config.phase,
        config_sha256=config.fingerprint(),
        unit_ids=config.unit_ids,
        scale_cells=config.scale_cells,
        outcome_count=0,
    )


def create_seed_roster(
    config: SimulatorMorphismChallengeConfig,
    *,
    injected_seeds: tuple[bytes, ...] | None = None,
) -> tuple[SimulatorMorphismChallengeSeedRoster, SimulatorMorphismChallengeSeedRosterCommitment]:
    if config.phase is not SimulatorMorphismChallengePhase.NOMINATION:
        raise ValueError("seed roster creation requires the nomination config")
    seeds = (
        tuple(secrets.token_bytes(32) for _ in config.unit_ids)
        if injected_seeds is None
        else injected_seeds
    )
    if len(seeds) != len(config.unit_ids) or any(len(value) != 32 for value in seeds):
        raise ValueError("simulator morphism challenges injected seed roster differs")
    if set(config.unit_ids) & {*development_unit_ids(), *canary_unit_ids()}:
        raise ValueError("simulator morphism challenges evaluation unit IDs overlap a prior phase")
    seed_digests = tuple(sha256(value).hexdigest() for value in seeds)
    if len(set(seed_digests)) != len(seed_digests):
        raise ValueError("simulator morphism challenges evaluation seeds repeat")
    if set(seed_digests) & reserved_non_evaluation_seed_digests():
        raise ValueError("simulator morphism challenges evaluation seeds overlap a canary/development seed")
    entries = tuple(
        SimulatorMorphismChallengeSeedEntry(
            unit_id=unit_id,
            seed_hex=seed.hex(),
            seed_sha256=seed_digest,
        )
        for unit_id, seed, seed_digest in zip(
            config.unit_ids,
            seeds,
            seed_digests,
            strict=True,
        )
    )
    roster = SimulatorMorphismChallengeSeedRoster(
        roster_id="simulator-morphism-challenges.evaluation-seed-roster",
        entries=entries,
        outcome_count_at_generation=0,
    )
    payload_sha256 = sha256(roster.canonical_bytes()).hexdigest()
    commitment = SimulatorMorphismChallengeSeedRosterCommitment(
        roster_id="simulator-morphism-challenges.evaluation-seed-roster-commitment",
        evaluation_unit_ids=config.unit_ids,
        seed_payload_sha256=payload_sha256,
        commitment_sha256=sha256(
            b"simulator-morphism-challenges-seed-roster-commitment\0" + roster.canonical_bytes()
        ).hexdigest(),
        seed_bytes_per_unit=32,
        public_seed_count=0,
        outcome_count_at_commitment=0,
    )
    return roster, commitment


def validate_seed_roster_commitment(
    roster: SimulatorMorphismChallengeSeedRoster,
    commitment: SimulatorMorphismChallengeSeedRosterCommitment,
) -> None:
    if (
        tuple(value.unit_id for value in roster.entries) != commitment.evaluation_unit_ids
        or sha256(roster.canonical_bytes()).hexdigest() != commitment.seed_payload_sha256
        or sha256(b"simulator-morphism-challenges-seed-roster-commitment\0" + roster.canonical_bytes()).hexdigest()
        != commitment.commitment_sha256
    ):
        raise ValueError("simulator morphism challenges seed roster differs from its public commitment")


def denominator_bundle(
    *,
    config: SimulatorMorphismChallengeConfig,
    unit_id: str,
    seed: bytes,
    scientific_input: HistoryBudgetUnitScientificInput | None = None,
) -> SimulatorMorphismChallengeDenominatorBundle:
    if unit_id not in config.unit_ids:
        raise ValueError("simulator morphism challenges descriptor unit lies outside the phase roster")
    if not isinstance(scientific_input, HistoryBudgetUnitScientificInput):
        raise ValueError("denominator generation requires authenticated complete original numeric inputs before scientific work")
    scientific_input.require_current_binding(programme_ordinal=0, unit_id=unit_id, seed=seed, scale_cells=config.scale_cells)
    family = family_from_unit_id(unit_id)
    descriptors = tuple(
        generate_descriptor(
            unit_id=unit_id,
            family=family,
            scale_cells=scale,
            seed=seed,
        )
        for scale in config.scale_cells
    )
    return SimulatorMorphismChallengeDenominatorBundle(
        bundle_id=f"denominator-bundle.{unit_id}",
        unit_id=unit_id,
        descriptors=descriptors,
        scientific_input=scientific_input,
    )


def observer_bundle(
    *,
    config: SimulatorMorphismChallengeConfig,
    denominators: SimulatorMorphismChallengeDenominatorBundle,
    implementation_sha256: str,
) -> ObserverBundleExecution:
    forecasts: list[SimulatorMorphismChallengeHistoryRankForecast] = []
    nominations: list[SimulatorMorphismChallengeChallengeNomination] = []
    arrays = {}
    for descriptor in denominators.descriptors:
        execution = observe_and_nominate(
            config=config,
            descriptor=descriptor,
            implementation_sha256=implementation_sha256,
            scientific_input=denominators.scientific_input.descriptor_input(descriptor.fingerprint()),
        )
        forecasts.append(execution.forecast)
        nominations.extend(execution.nominations)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.observer",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: simulator_morphism_challenges_array_semantics(key) for key in arrays},
    )
    bundle = SimulatorMorphismChallengeObserverBundle(
        bundle_id=f"observer-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=tuple(sorted(forecasts, key=lambda value: value.forecast_id)),
        nominations=tuple(sorted(nominations, key=lambda value: value.nomination_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_count_at_nomination=0,
    )
    return ObserverBundleExecution(bundle=bundle, packed_arrays=packed)


def history_bundle(
    *,
    config: SimulatorMorphismChallengeConfig,
    denominators: SimulatorMorphismChallengeDenominatorBundle,
) -> HistoryBundleExecution:
    forecasts = []
    arrays = {}
    for descriptor in denominators.descriptors:
        execution = observe_history(config=config, descriptor=descriptor)
        forecasts.append(execution.forecast)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.history",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: simulator_morphism_challenges_array_semantics(key) for key in arrays},
    )
    bundle = SimulatorMorphismChallengeHistoryBundle(
        bundle_id=f"history-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=tuple(sorted(forecasts, key=lambda value: value.forecast_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_count=0,
    )
    return HistoryBundleExecution(bundle=bundle, packed_arrays=packed)


def targeter_bundle(
    *,
    config: SimulatorMorphismChallengeConfig,
    denominators: SimulatorMorphismChallengeDenominatorBundle,
    history: SimulatorMorphismChallengeHistoryBundle,
    implementation_sha256: str,
) -> ObserverBundleExecution:
    if history.denominator_bundle_sha256 != denominators.fingerprint():
        raise ValueError("simulator morphism challenges targeter received another history denominator")
    by_id = {value.forecast_id: value for value in history.forecasts}
    nominations: list[SimulatorMorphismChallengeChallengeNomination] = []
    arrays = {}
    for descriptor in denominators.descriptors:
        execution = nominate_from_history(
            config=config,
            descriptor=descriptor,
            history_forecast=by_id[f"rank-forecast.{descriptor.unit_id}.n{descriptor.scale_cells}"],
            implementation_sha256=implementation_sha256,
            scientific_input=denominators.scientific_input.descriptor_input(descriptor.fingerprint()),
        )
        nominations.extend(execution.nominations)
        arrays.update(
            {f"n{descriptor.scale_cells}.nomination-modes": execution.arrays["nomination-modes"]}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.targeter",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: simulator_morphism_challenges_array_semantics(key) for key in arrays},
    )
    bundle = SimulatorMorphismChallengeObserverBundle(
        bundle_id=f"observer-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=history.forecasts,
        nominations=tuple(sorted(nominations, key=lambda value: value.nomination_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_count_at_nomination=0,
    )
    return ObserverBundleExecution(bundle=bundle, packed_arrays=packed)


def freeze_nominations(
    *,
    history: SimulatorMorphismChallengeHistoryBundle,
    observer: SimulatorMorphismChallengeObserverBundle,
    method_freeze: SimulatorMorphismChallengeMethodFreeze | None,
) -> SimulatorMorphismChallengeNominationFreeze:
    if (
        history.unit_id != observer.unit_id
        or history.denominator_bundle_sha256 != observer.denominator_bundle_sha256
        or history.forecasts != observer.forecasts
    ):
        raise ValueError("simulator morphism challenges nomination freeze inputs differ")
    return SimulatorMorphismChallengeNominationFreeze(
        freeze_id=f"nomination-freeze.{observer.unit_id}",
        observer_bundle=observer,
        history_bundle_sha256=history.fingerprint(),
        method_freeze_sha256=(None if method_freeze is None else method_freeze.fingerprint()),
        outcome_count_at_freeze=0,
    )


def generator_bundle(
    *,
    config: SimulatorMorphismChallengeConfig,
    denominators: SimulatorMorphismChallengeDenominatorBundle,
    observer: SimulatorMorphismChallengeObserverBundle,
    implementation_sha256: str,
) -> GeneratorBundleExecution:
    if observer.denominator_bundle_sha256 != denominators.fingerprint():
        raise ValueError("simulator morphism challenges generator received another observer denominator")
    outcomes = []
    arrays = {}
    for descriptor in denominators.descriptors:
        nominations = tuple(
            value for value in observer.nominations if value.scale_cells == descriptor.scale_cells
        )
        execution = generate_outcomes(
            config=config,
            descriptor=descriptor,
            nominations=nominations,
            implementation_sha256=implementation_sha256,
            outcome_access=config.outcome_access,
        )
        outcomes.append(execution.outcome)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.generator",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: simulator_morphism_challenges_array_semantics(key) for key in arrays},
    )
    bundle = SimulatorMorphismChallengeGeneratorBundle(
        bundle_id=f"generator-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        observer_bundle_sha256=observer.fingerprint(),
        outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_access=config.outcome_access,
    )
    return GeneratorBundleExecution(bundle=bundle, packed_arrays=packed)


def adjudication_bundle(
    *,
    config: SimulatorMorphismChallengeConfig,
    denominators: SimulatorMorphismChallengeDenominatorBundle,
    observer: SimulatorMorphismChallengeObserverBundle,
    generator: SimulatorMorphismChallengeGeneratorBundle,
    generator_arrays_payload: bytes,
    generator_arrays_manifest: object,
    method_freeze: SimulatorMorphismChallengeMethodFreeze | None,
    maximum_array_bytes: int,
) -> SimulatorMorphismChallengeAdjudicationBundle:
    from .runtime_contracts import SimulatorMorphismChallengeArrayManifest

    if not isinstance(generator_arrays_manifest, SimulatorMorphismChallengeArrayManifest):
        raise TypeError("simulator morphism challenges evaluator requires the typed generator array manifest")
    if (
        generator.denominator_bundle_sha256 != denominators.fingerprint()
        or generator.observer_bundle_sha256 != observer.fingerprint()
        or generator.arrays_manifest_sha256 != generator_arrays_manifest.fingerprint()
    ):
        raise ValueError("simulator morphism challenges evaluator input identities differ")
    arrays = unpack_arrays(
        generator_arrays_payload,
        generator_arrays_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    forecasts = {value.forecast_id: value for value in observer.forecasts}
    outcomes = {value.outcome_id: value for value in generator.outcomes}
    rows = []
    for descriptor in denominators.descriptors:
        scale = descriptor.scale_cells
        scale_arrays = {
            key.split(".", 1)[1]: value
            for key, value in arrays.items()
            if key.startswith(f"n{scale}.")
        }
        outcome_id = f"generator-outcome.{descriptor.unit_id}.n{scale}"
        forecast_id = f"rank-forecast.{descriptor.unit_id}.n{scale}"
        outcome = outcomes[outcome_id]
        if logical_arrays_digest(scale_arrays) != outcome.arrays_sha256:
            raise ValueError("simulator morphism challenges sealed generator arrays differ from the outcome record")
        nominations = tuple(value for value in observer.nominations if value.scale_cells == scale)
        rows.append(
            adjudicate_unit_scale(
                config=config,
                descriptor=descriptor,
                forecast=forecasts[forecast_id],
                nominations=nominations,
                generator_outcome=outcome,
                method_freeze=method_freeze,
            )
        )
    return SimulatorMorphismChallengeAdjudicationBundle(
        bundle_id=f"adjudication-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        method_freeze_sha256=(None if method_freeze is None else method_freeze.fingerprint()),
        adjudications=tuple(sorted(rows, key=lambda value: value.adjudication_id)),
        raw_generator_payload_exposed=False,
    )


def development_ledger(
    *,
    canary: SimulatorMorphismChallengeCanaryReport,
    adjudication_bundles: tuple[SimulatorMorphismChallengeAdjudicationBundle, ...],
) -> SimulatorMorphismChallengeDevelopmentLedger:
    if not canary.passed:
        raise ValueError("development cannot close after a failed canary")
    adjudications_by_unit = {value.unit_id: value for value in adjudication_bundles}
    if len(adjudications_by_unit) != 6:
        raise ValueError("simulator morphism challenges development unit roster is incomplete")
    gate_counts = []
    for family in sorted(
        {value.family for bundle in adjudication_bundles for value in bundle.adjudications},
        key=lambda value: value.value,
    ):
        pair_adjudications = tuple(
            pair
            for bundle in adjudication_bundles
            for adjudication in bundle.adjudications
            if adjudication.family is family
            for pair in adjudication.pair_adjudications
        )
        gate_counts.append(
            SimulatorMorphismChallengeDevelopmentGateCount(
                family=family,
                target_nomination_count=sum(
                    value.gate is SimulatorMorphismChallengeGate.TARGET for value in pair_adjudications
                ),
                sink_nomination_count=sum(
                    value.gate is SimulatorMorphismChallengeGate.SINK for value in pair_adjudications
                ),
            )
        )
    adjudications = tuple(
        sorted(
            (value for bundle in adjudication_bundles for value in bundle.adjudications),
            key=lambda value: value.adjudication_id,
        )
    )
    return SimulatorMorphismChallengeDevelopmentLedger(
        ledger_id="simulator-morphism-challenges.development-ledger",
        canary_report_sha256=canary.fingerprint(),
        adjudications=adjudications,
        gate_counts=tuple(sorted(gate_counts, key=lambda value: value.gate_count_id)),
        requested_unit_count=6,
        complete_unit_count=len(adjudication_bundles),
        evaluation_roster_access_count=0,
        reason_codes=(),
    )


def evaluate_development_gate(ledger: SimulatorMorphismChallengeDevelopmentLedger) -> SimulatorMorphismChallengeDevelopmentGate:
    reasons: set[str] = set()
    if ledger.complete_unit_count != ledger.requested_unit_count:
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_DEVELOPMENT_INCOMPLETE")
    total_target = sum(value.target_nomination_count for value in ledger.gate_counts)
    total_sink = sum(value.sink_nomination_count for value in ledger.gate_counts)
    if (
        any(
            value.target_nomination_count == 0 or value.sink_nomination_count == 0
            for value in ledger.gate_counts
        )
        or total_target < 12
        or total_sink < 12
    ):
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_BOUNDARY_TARGETING_UNPOWERED")
    if any(not value.generator_observer_agreement for value in ledger.adjudications):
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_CANARY_CONFORMANCE_FAILED")
    return SimulatorMorphismChallengeDevelopmentGate(
        gate_id="simulator-morphism-challenges.development-power-and-correctness-gate",
        development_ledger_sha256=ledger.fingerprint(),
        issue_evaluation=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def freeze_method(
    *,
    development_config: SimulatorMorphismChallengeConfig,
    evaluation_config: SimulatorMorphismChallengeConfig,
    seed_roster_commitment: SimulatorMorphismChallengeSeedRosterCommitment,
    development_gate: SimulatorMorphismChallengeDevelopmentGate,
    observer_implementation_sha256: str,
    generator_implementation_sha256: str,
    evaluator_implementation_sha256: str,
    development_receipt_closure_sha256: str,
) -> SimulatorMorphismChallengeMethodFreeze:
    if (
        development_config.phase is not SimulatorMorphismChallengePhase.DEVELOPMENT
        or evaluation_config.phase is not SimulatorMorphismChallengePhase.EVALUATION
        or not development_gate.issue_evaluation
        or evaluation_config.seed_roster_commitment_sha256 != seed_roster_commitment.fingerprint()
    ):
        raise ValueError("simulator morphism challenges method freeze prerequisites differ")
    return SimulatorMorphismChallengeMethodFreeze(
        freeze_id="simulator-morphism-challenges.method-freeze",
        development_config_sha256=development_config.fingerprint(),
        evaluation_config_sha256=evaluation_config.fingerprint(),
        seed_roster_commitment_sha256=seed_roster_commitment.fingerprint(),
        observer_implementation_sha256=observer_implementation_sha256,
        generator_implementation_sha256=generator_implementation_sha256,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
        development_receipt_closure_sha256=development_receipt_closure_sha256,
        frozen_threshold_ids=(
            "action-panel-5x5",
            "simulator-morphism-challenges-boundary-target-sink",
            "complete-unit-intention-to-treat",
            "history-rank-relative-1e-12",
            "midpoint-boundary-tie-1e-10",
            "receiver-r8-capacitance-weighted",
        ),
        evaluation_outcome_count=0,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )


def freeze_evaluation_design(
    *,
    method_freeze: SimulatorMorphismChallengeMethodFreeze,
    evaluation_config: SimulatorMorphismChallengeConfig,
    seed_roster_commitment: SimulatorMorphismChallengeSeedRosterCommitment,
    implementation_source_closure_sha256: str,
) -> SimulatorMorphismChallengeEvaluationDesignFreeze:
    if (
        evaluation_config.phase is not SimulatorMorphismChallengePhase.EVALUATION
        or method_freeze.evaluation_config_sha256 != evaluation_config.fingerprint()
        or evaluation_config.seed_roster_commitment_sha256 != seed_roster_commitment.fingerprint()
        or method_freeze.seed_roster_commitment_sha256 != seed_roster_commitment.fingerprint()
    ):
        raise ValueError("simulator morphism challenges evaluation design freeze prerequisites differ")
    return SimulatorMorphismChallengeEvaluationDesignFreeze(
        freeze_id="simulator-morphism-challenges.evaluation-design-freeze",
        method_freeze=method_freeze,
        evaluation_config_sha256=evaluation_config.fingerprint(),
        seed_roster_commitment=seed_roster_commitment,
        implementation_source_closure_sha256=implementation_source_closure_sha256,
        evaluation_outcome_count=0,
    )


def receipt_closure_sha256(
    receipt_ids: tuple[str, ...], materialization_ids: tuple[str, ...]
) -> str:
    return sha256(
        canonical_json_bytes(
            {
                "receipt_ids": tuple(sorted(receipt_ids)),
                "materialization_ids": tuple(sorted(materialization_ids)),
            }
        )
    ).hexdigest()


__all__ = [
    "GeneratorBundleExecution",
    "HistoryBundleExecution",
    "ObserverBundleExecution",
    "adjudication_bundle",
    "create_seed_roster",
    "denominator_bundle",
    "development_ledger",
    "evaluate_development_gate",
    "freeze_evaluation_design",
    "freeze_method",
    "freeze_nominations",
    "generator_bundle",
    "history_bundle",
    "observer_bundle",
    "receipt_closure_sha256",
    "requested_unit_ledger",
    "targeter_bundle",
    "validate_seed_roster_commitment",
]
