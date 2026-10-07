"""Pure phase services joining history budget phase diagram records without hiding scientific steps."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetUnitScientificInput

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
import secrets
from typing import cast

import numpy as np

from empirical_lawhood.adapters.methods.history_budget_phase_diagram.conditioning import conditioning_step
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.discrete_rank import discrete_rank_bracket
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.evaluator import adjudicate_targeted_coordinates, adjudicate_unit_scale, adjudicate_untouched_scale
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.history import assemble_dense_operator, coordinate_labels, observe_history
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.preparation_sampler import sample_untouched_preparations
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.structural_rank import structural_rank_step
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.targeting import nominate_targeted_challenges
from empirical_lawhood.adapters.simulators.rc_ladder_history_budget.generator import generate_outcomes, generate_untouched_outcomes
from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .array_io import FloatArray, IntArray, PackedArrays, TypedArray, history_budget_phase_diagram_array_semantics, logical_arrays_digest, pack_arrays, unpack_arrays
from .contracts import HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramChallengeNomination, HistoryBudgetPhaseDiagramConditioningStep, HistoryBudgetPhaseDiagramDiscreteRankBracket, HistoryBudgetPhaseDiagramGate, HistoryBudgetPhaseDiagramMethodFreeze, HistoryBudgetPhaseDiagramOptimizationCertificate, HistoryBudgetPhaseDiagramPhase, HistoryBudgetPhaseDiagramSeedRosterCommitment, HistoryBudgetPhaseDiagramStructuralRankStep, HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication, HistoryBudgetPhaseDiagramUnitAdjudication, HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication
from .descriptors import canary_unit_ids, development_unit_ids, family_from_unit_id, generate_descriptor, reserved_non_evaluation_seed_digests
from .runtime_contracts import HistoryBudgetPhaseDiagramAdjudicationBundle, HistoryBudgetPhaseDiagramCanaryReport, HistoryBudgetPhaseDiagramDenominatorBundle, HistoryBudgetPhaseDiagramDevelopmentGate, HistoryBudgetPhaseDiagramDevelopmentGateCount, HistoryBudgetPhaseDiagramDevelopmentLedger, HistoryBudgetPhaseDiagramEvaluationDesignFreeze, HistoryBudgetPhaseDiagramGeneratorBundle, HistoryBudgetPhaseDiagramHistoryBundle, HistoryBudgetPhaseDiagramNominationFreeze, HistoryBudgetPhaseDiagramObserverBundle, HistoryBudgetPhaseDiagramRequestedUnitLedger, HistoryBudgetPhaseDiagramSeedEntry, HistoryBudgetPhaseDiagramSeedRoster, HistoryBudgetPhaseDiagramUntouchedBundle


@dataclass(frozen=True, slots=True)
class ObserverBundleExecution:
    bundle: HistoryBudgetPhaseDiagramObserverBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class HistoryBundleExecution:
    bundle: HistoryBudgetPhaseDiagramHistoryBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class GeneratorBundleExecution:
    bundle: HistoryBudgetPhaseDiagramGeneratorBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class UntouchedBundleExecution:
    bundle: HistoryBudgetPhaseDiagramUntouchedBundle
    float_arrays: PackedArrays
    int_arrays: PackedArrays


def _require_float_array(value: TypedArray, *, array_id: str) -> FloatArray:
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"history budget phase diagram array {array_id} must be float64")
    return cast(FloatArray, value)


def _require_int_array(value: TypedArray, *, array_id: str) -> IntArray:
    if value.dtype != np.dtype(np.int64):
        raise ValueError(f"history budget phase diagram array {array_id} must be int64")
    return cast(IntArray, value)


def requested_unit_ledger(config: HistoryBudgetPhaseDiagramConfig) -> HistoryBudgetPhaseDiagramRequestedUnitLedger:
    return HistoryBudgetPhaseDiagramRequestedUnitLedger(
        ledger_id=f"ledger.history-budget-phase-diagram.{config.phase.value.lower()}.requested-units",
        phase=config.phase,
        config_sha256=config.fingerprint(),
        unit_ids=config.unit_ids,
        scale_cells=config.scale_cells,
        outcome_count=0,
    )


def create_seed_roster(
    config: HistoryBudgetPhaseDiagramConfig,
    *,
    injected_seeds: tuple[bytes, ...] | None = None,
) -> tuple[HistoryBudgetPhaseDiagramSeedRoster, HistoryBudgetPhaseDiagramSeedRosterCommitment]:
    if config.phase is not HistoryBudgetPhaseDiagramPhase.NOMINATION:
        raise ValueError("seed roster creation requires the nomination config")
    seeds = (
        tuple(secrets.token_bytes(32) for _ in config.unit_ids)
        if injected_seeds is None
        else injected_seeds
    )
    if len(seeds) != len(config.unit_ids) or any(len(value) != 32 for value in seeds):
        raise ValueError("history budget phase diagram injected seed roster differs")
    if set(config.unit_ids) & {*development_unit_ids(), *canary_unit_ids()}:
        raise ValueError("history budget phase diagram evaluation unit IDs overlap a prior phase")
    seed_digests = tuple(sha256(value).hexdigest() for value in seeds)
    if len(set(seed_digests)) != len(seed_digests):
        raise ValueError("history budget phase diagram evaluation seeds repeat")
    if set(seed_digests) & reserved_non_evaluation_seed_digests():
        raise ValueError("history budget phase diagram evaluation seeds overlap a canary/development seed")
    entries = tuple(
        HistoryBudgetPhaseDiagramSeedEntry(
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
    roster = HistoryBudgetPhaseDiagramSeedRoster(
        roster_id="history-budget-phase-diagram.evaluation-seed-roster",
        entries=entries,
        outcome_count_at_generation=0,
    )
    payload_sha256 = sha256(roster.canonical_bytes()).hexdigest()
    commitment = HistoryBudgetPhaseDiagramSeedRosterCommitment(
        roster_id="history-budget-phase-diagram.evaluation-seed-roster-commitment",
        evaluation_unit_ids=config.unit_ids,
        seed_payload_sha256=payload_sha256,
        commitment_sha256=sha256(
            b"history-budget-phase-diagram-seed-roster-commitment\0" + roster.canonical_bytes()
        ).hexdigest(),
        seed_bytes_per_unit=32,
        public_seed_count=0,
        outcome_count_at_commitment=0,
    )
    return roster, commitment


def validate_seed_roster_commitment(
    roster: HistoryBudgetPhaseDiagramSeedRoster,
    commitment: HistoryBudgetPhaseDiagramSeedRosterCommitment,
) -> None:
    if (
        tuple(value.unit_id for value in roster.entries) != commitment.evaluation_unit_ids
        or sha256(roster.canonical_bytes()).hexdigest() != commitment.seed_payload_sha256
        or sha256(b"history-budget-phase-diagram-seed-roster-commitment\0" + roster.canonical_bytes()).hexdigest()
        != commitment.commitment_sha256
    ):
        raise ValueError("history budget phase diagram seed roster differs from its public commitment")


def denominator_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    unit_id: str,
    seed: bytes,
    scientific_input: HistoryBudgetUnitScientificInput | None = None,
) -> HistoryBudgetPhaseDiagramDenominatorBundle:
    if unit_id not in config.unit_ids:
        raise ValueError("history budget phase diagram descriptor unit lies outside the phase roster")
    if not isinstance(scientific_input, HistoryBudgetUnitScientificInput):
        raise ValueError("denominator generation requires authenticated complete original numeric inputs before scientific work")
    scientific_input.require_current_binding(programme_ordinal=1, unit_id=unit_id, seed=seed, scale_cells=config.scale_cells)
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
    if scientific_input.preparation_input is None:
        raise ValueError("untouched preparation requires the complete supplied original substream input")
    preparation_seed = bytes.fromhex(scientific_input.preparation_input.preparation_substream_seed_hex)
    return HistoryBudgetPhaseDiagramDenominatorBundle(
        bundle_id=f"denominator-bundle.{unit_id}",
        unit_id=unit_id,
        preparation_substream_seed_hex=preparation_seed.hex(),
        preparation_substream_seed_sha256=sha256(preparation_seed).hexdigest(),
        descriptors=descriptors,
        scientific_input=scientific_input,
    )


def observer_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    denominators: HistoryBudgetPhaseDiagramDenominatorBundle,
    implementation_sha256: str,
) -> ObserverBundleExecution:
    history = history_bundle(config=config, denominators=denominators)
    return targeter_bundle(
        config=config,
        denominators=denominators,
        history=history.bundle,
        implementation_sha256=implementation_sha256,
    )


def history_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    denominators: HistoryBudgetPhaseDiagramDenominatorBundle,
) -> HistoryBundleExecution:
    forecasts = []
    coordinates = []
    structural = []
    discrete = []
    conditioning = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        execution = observe_history(config=config, descriptor=descriptor)
        forecasts.append(execution.forecast)
        operator = assemble_dense_operator(descriptor)
        inverse_lag = execution.arrays["inverse-lag-propagator"]
        blocks = [operator.receiver.copy()]
        current = operator.receiver.copy()
        matrices_by_depth: dict[int, FloatArray] = {0: current.copy()}
        for depth in range(1, config.history_max_depth + 1):
            current = current @ inverse_lag
            blocks.append(current.copy())
            matrices_by_depth[depth] = np.vstack(blocks)
        structural_by_depth: dict[int, HistoryBudgetPhaseDiagramStructuralRankStep] = {}
        discrete_by_depth: dict[int, HistoryBudgetPhaseDiagramDiscreteRankBracket] = {}
        conditioning_by_depth_resolution: dict[tuple[int, Decimal], HistoryBudgetPhaseDiagramConditioningStep] = {}
        for coordinate in coordinate_labels(config, descriptor.scale_cells):
            structural_step = structural_by_depth.get(coordinate.depth)
            if structural_step is None:
                structural_step = structural_rank_step(descriptor, coordinate)
                structural_by_depth[coordinate.depth] = structural_step
            else:
                structural_step = replace(
                    structural_step,
                    coordinate_id=coordinate.coordinate_id,
                )
            coordinates.append(coordinate)
            structural.append(structural_step)
            discrete_step = discrete_by_depth.get(coordinate.depth)
            if discrete_step is None:
                discrete_step = discrete_rank_bracket(
                    config,
                    descriptor,
                    coordinate,
                    structural_step,
                    matrix=matrices_by_depth[coordinate.depth],
                )
                discrete_by_depth[coordinate.depth] = discrete_step
            else:
                discrete_step = replace(
                    discrete_step,
                    coordinate_id=coordinate.coordinate_id,
                )
            discrete.append(discrete_step)
            conditioning_key = (coordinate.depth, coordinate.resolution_epsilon)
            conditioning_value = conditioning_by_depth_resolution.get(conditioning_key)
            if conditioning_value is None:
                conditioning_value = conditioning_step(
                    config,
                    descriptor,
                    coordinate,
                    matrix=matrices_by_depth[coordinate.depth],
                )
                conditioning_by_depth_resolution[conditioning_key] = conditioning_value
            else:
                conditioning_value = replace(
                    conditioning_value,
                    coordinate_id=coordinate.coordinate_id,
                )
            conditioning.append(conditioning_value)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.history",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: history_budget_phase_diagram_array_semantics(key) for key in arrays},
    )
    bundle = HistoryBudgetPhaseDiagramHistoryBundle(
        bundle_id=f"history-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=tuple(sorted(forecasts, key=lambda value: value.forecast_id)),
        coordinates=tuple(sorted(coordinates, key=lambda value: value.coordinate_id)),
        structural_rank_steps=tuple(sorted(structural, key=lambda value: value.coordinate_id)),
        discrete_rank_brackets=tuple(sorted(discrete, key=lambda value: value.coordinate_id)),
        conditioning_steps=tuple(sorted(conditioning, key=lambda value: value.coordinate_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_count=0,
    )
    return HistoryBundleExecution(bundle=bundle, packed_arrays=packed)


def targeter_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    denominators: HistoryBudgetPhaseDiagramDenominatorBundle,
    history: HistoryBudgetPhaseDiagramHistoryBundle,
    implementation_sha256: str,
) -> ObserverBundleExecution:
    if history.denominator_bundle_sha256 != denominators.fingerprint():
        raise ValueError("history budget phase diagram targeter received another history denominator")
    by_id = {value.forecast_id: value for value in history.forecasts}
    nominations: list[HistoryBudgetPhaseDiagramChallengeNomination] = []
    certificates: list[HistoryBudgetPhaseDiagramOptimizationCertificate] = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        execution = nominate_targeted_challenges(
            config=config,
            descriptor=descriptor,
            history_forecast=by_id[f"rank-forecast.{descriptor.unit_id}.n{descriptor.scale_cells}"],
            implementation_sha256=implementation_sha256,
        )
        nominations.extend(execution.nominations)
        certificates.extend(execution.certificates)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.targeter",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: history_budget_phase_diagram_array_semantics(key) for key in arrays},
    )
    bundle = HistoryBudgetPhaseDiagramObserverBundle(
        bundle_id=f"observer-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=history.forecasts,
        nominations=tuple(sorted(nominations, key=lambda value: value.nomination_id)),
        optimization_certificates=tuple(
            sorted(certificates, key=lambda value: value.certificate_id)
        ),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_count_at_nomination=0,
    )
    return ObserverBundleExecution(bundle=bundle, packed_arrays=packed)


def untouched_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    denominators: HistoryBudgetPhaseDiagramDenominatorBundle,
) -> UntouchedBundleExecution:
    descriptors = {value.scale_cells: value for value in denominators.descriptors}
    execution = sample_untouched_preparations(
        config=config,
        unit_id=denominators.unit_id,
        descriptors=descriptors,
        seed=denominators.preparation_substream_seed,
        scientific_input=denominators.scientific_input.preparation_input,
    )
    float_arrays = dict(execution.float_arrays)
    int_arrays = dict(execution.int_arrays)
    packed_float = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.untouched-float",
        unit_id=denominators.unit_id,
        arrays=float_arrays,
        semantics={key: history_budget_phase_diagram_array_semantics(key) for key in float_arrays},
    )
    packed_int = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.untouched-int",
        unit_id=denominators.unit_id,
        arrays=int_arrays,
        semantics={key: history_budget_phase_diagram_array_semantics(key) for key in int_arrays},
    )
    bundle = HistoryBudgetPhaseDiagramUntouchedBundle(
        bundle_id=f"untouched-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        preparation_manifest=execution.manifest,
        coordinates=execution.coordinates,
        float_arrays_manifest_sha256=packed_float.manifest.fingerprint(),
        int_arrays_manifest_sha256=packed_int.manifest.fingerprint(),
        outcome_count_at_freeze=0,
    )
    return UntouchedBundleExecution(
        bundle=bundle,
        float_arrays=packed_float,
        int_arrays=packed_int,
    )


def freeze_nominations(
    *,
    history: HistoryBudgetPhaseDiagramHistoryBundle,
    observer: HistoryBudgetPhaseDiagramObserverBundle,
    untouched: HistoryBudgetPhaseDiagramUntouchedBundle,
    method_freeze: HistoryBudgetPhaseDiagramMethodFreeze | None,
) -> HistoryBudgetPhaseDiagramNominationFreeze:
    if (
        history.unit_id != observer.unit_id
        or history.denominator_bundle_sha256 != observer.denominator_bundle_sha256
        or history.denominator_bundle_sha256 != untouched.denominator_bundle_sha256
        or history.forecasts != observer.forecasts
    ):
        raise ValueError("history budget phase diagram nomination freeze inputs differ")
    return HistoryBudgetPhaseDiagramNominationFreeze(
        freeze_id=f"nomination-freeze.{observer.unit_id}",
        observer_bundle=observer,
        untouched_bundle_sha256=untouched.fingerprint(),
        preparation_manifest_sha256=untouched.preparation_manifest.fingerprint(),
        history_bundle_sha256=history.fingerprint(),
        method_freeze_sha256=(None if method_freeze is None else method_freeze.fingerprint()),
        outcome_count_at_freeze=0,
    )


def generator_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    denominators: HistoryBudgetPhaseDiagramDenominatorBundle,
    observer: HistoryBudgetPhaseDiagramObserverBundle,
    untouched: HistoryBudgetPhaseDiagramUntouchedBundle,
    untouched_float_payload: bytes,
    untouched_float_manifest: object,
    implementation_sha256: str,
    maximum_array_bytes: int = 256 * 1024**2,
) -> GeneratorBundleExecution:
    from .runtime_contracts import HistoryBudgetPhaseDiagramArrayManifest

    if not isinstance(untouched_float_manifest, HistoryBudgetPhaseDiagramArrayManifest):
        raise TypeError("history budget phase diagram generator requires the typed untouched array manifest")
    if observer.denominator_bundle_sha256 != denominators.fingerprint():
        raise ValueError("history budget phase diagram generator received another observer denominator")
    if (
        untouched.denominator_bundle_sha256 != denominators.fingerprint()
        or untouched.float_arrays_manifest_sha256 != untouched_float_manifest.fingerprint()
    ):
        raise ValueError("history budget phase diagram generator received another untouched cohort")
    untouched_arrays = unpack_arrays(
        untouched_float_payload,
        untouched_float_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    outcomes = []
    untouched_outcomes = []
    arrays: dict[str, TypedArray] = {}
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
        untouched_execution = generate_untouched_outcomes(
            config=config,
            descriptor=descriptor,
            preparation_manifest=untouched.preparation_manifest,
            earliest_states=_require_float_array(
                untouched_arrays[f"initial-states-n{descriptor.scale_cells}"],
                array_id=f"initial-states-n{descriptor.scale_cells}",
            ),
            implementation_sha256=implementation_sha256,
            outcome_access=config.outcome_access,
        )
        untouched_outcomes.append(untouched_execution.outcome)
        arrays.update(
            {
                f"n{descriptor.scale_cells}.untouched-{key}": value
                for key, value in untouched_execution.arrays.items()
            }
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.generator",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: history_budget_phase_diagram_array_semantics(key) for key in arrays},
    )
    bundle = HistoryBudgetPhaseDiagramGeneratorBundle(
        bundle_id=f"generator-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        observer_bundle_sha256=observer.fingerprint(),
        outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
        untouched_outcomes=tuple(sorted(untouched_outcomes, key=lambda value: value.outcome_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_access=config.outcome_access,
    )
    return GeneratorBundleExecution(bundle=bundle, packed_arrays=packed)


def adjudication_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    denominators: HistoryBudgetPhaseDiagramDenominatorBundle,
    history: HistoryBudgetPhaseDiagramHistoryBundle,
    observer: HistoryBudgetPhaseDiagramObserverBundle,
    untouched: HistoryBudgetPhaseDiagramUntouchedBundle,
    untouched_int_payload: bytes,
    untouched_int_manifest: object,
    generator: HistoryBudgetPhaseDiagramGeneratorBundle,
    generator_arrays_payload: bytes,
    generator_arrays_manifest: object,
    method_freeze: HistoryBudgetPhaseDiagramMethodFreeze | None,
    maximum_array_bytes: int,
) -> HistoryBudgetPhaseDiagramAdjudicationBundle:
    from .runtime_contracts import HistoryBudgetPhaseDiagramArrayManifest

    if not isinstance(generator_arrays_manifest, HistoryBudgetPhaseDiagramArrayManifest) or not isinstance(
        untouched_int_manifest, HistoryBudgetPhaseDiagramArrayManifest
    ):
        raise TypeError("history budget phase diagram evaluator requires the typed generator array manifest")
    if (
        generator.denominator_bundle_sha256 != denominators.fingerprint()
        or history.denominator_bundle_sha256 != denominators.fingerprint()
        or history.forecasts != observer.forecasts
        or generator.observer_bundle_sha256 != observer.fingerprint()
        or generator.arrays_manifest_sha256 != generator_arrays_manifest.fingerprint()
        or untouched.int_arrays_manifest_sha256 != untouched_int_manifest.fingerprint()
    ):
        raise ValueError("history budget phase diagram evaluator input identities differ")
    arrays = unpack_arrays(
        generator_arrays_payload,
        generator_arrays_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    untouched_int_arrays = unpack_arrays(
        untouched_int_payload,
        untouched_int_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    forecasts = {value.forecast_id: value for value in observer.forecasts}
    outcomes = {value.outcome_id: value for value in generator.outcomes}
    untouched_outcomes = {value.outcome_id: value for value in generator.untouched_outcomes}
    rows: list[HistoryBudgetPhaseDiagramUnitAdjudication] = []
    targeted_rows: list[HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication] = []
    untouched_rows: list[HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication] = []
    for descriptor in denominators.descriptors:
        scale = descriptor.scale_cells
        scale_arrays = {
            key.split(".", 1)[1]: value
            for key, value in arrays.items()
            if key.startswith(f"n{scale}.") and not key.startswith(f"n{scale}.untouched-")
        }
        outcome_id = f"generator-outcome.{descriptor.unit_id}.n{scale}"
        forecast_id = f"rank-forecast.{descriptor.unit_id}.n{scale}"
        outcome = outcomes[outcome_id]
        if logical_arrays_digest(scale_arrays) != outcome.arrays_sha256:
            raise ValueError("history budget phase diagram sealed generator arrays differ from the outcome record")
        nominations = tuple(value for value in observer.nominations if value.scale_cells == scale)
        unit_adjudication = adjudicate_unit_scale(
            config=config,
            descriptor=descriptor,
            forecast=forecasts[forecast_id],
            nominations=nominations,
            generator_outcome=outcome,
            method_freeze=method_freeze,
        )
        rows.append(unit_adjudication)
        targeted_rows.extend(
            adjudicate_targeted_coordinates(
                descriptor=descriptor,
                coordinates=coordinate_labels(config, scale),
                unit_adjudication=unit_adjudication,
                certificates=tuple(
                    value
                    for value in observer.optimization_certificates
                    if value.coordinate_id.startswith(f"coordinate.n{scale}.")
                ),
                expected_action_ids=tuple(
                    sorted(
                        f"action.a{amplitude_index:02d}.d{duration_index:02d}"
                        for amplitude_index, _ in enumerate(config.action_amplitudes_u_star)
                        for duration_index, _ in enumerate(config.action_durations_t_star)
                    )
                ),
            )
        )
        untouched_scale_arrays = {
            key.split(".untouched-", 1)[1]: value
            for key, value in arrays.items()
            if key.startswith(f"n{scale}.untouched-")
        }
        untouched_outcome = untouched_outcomes[
            f"untouched-generator-outcome.{descriptor.unit_id}.n{scale}"
        ]
        if logical_arrays_digest(untouched_scale_arrays) != untouched_outcome.arrays_sha256:
            raise ValueError("history budget phase diagram untouched generator arrays differ from their record")
        coordinates = tuple(value for value in untouched.coordinates if value.scale_cells == scale)
        untouched_rows.extend(
            adjudicate_untouched_scale(
                config=config,
                descriptor=descriptor,
                coordinates=coordinates,
                collision_edges=_require_int_array(
                    untouched_int_arrays[f"collision-edges-n{scale}"],
                    array_id=f"collision-edges-n{scale}",
                ),
                receiver_history=_require_float_array(
                    untouched_scale_arrays["receiver-history"],
                    array_id=f"n{scale}.untouched-receiver-history",
                ),
                action_metrics=_require_float_array(
                    untouched_scale_arrays["action-metrics"],
                    array_id=f"n{scale}.untouched-action-metrics",
                ),
                action_states=_require_float_array(
                    untouched_scale_arrays["action-states"],
                    array_id=f"n{scale}.untouched-action-states",
                ),
                generator_outcome=untouched_outcome,
            )
        )
    return HistoryBudgetPhaseDiagramAdjudicationBundle(
        bundle_id=f"adjudication-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        method_freeze_sha256=(None if method_freeze is None else method_freeze.fingerprint()),
        history_bundle=history,
        adjudications=tuple(sorted(rows, key=lambda value: value.adjudication_id)),
        targeted_adjudications=tuple(
            sorted(targeted_rows, key=lambda value: value.adjudication_id)
        ),
        untouched_adjudications=tuple(
            sorted(untouched_rows, key=lambda value: value.adjudication_id)
        ),
        raw_generator_payload_exposed=False,
    )


def development_ledger(
    *,
    adjudication_bundles: tuple[HistoryBudgetPhaseDiagramAdjudicationBundle, ...],
) -> HistoryBudgetPhaseDiagramDevelopmentLedger:
    adjudications_by_unit = {value.unit_id: value for value in adjudication_bundles}
    if len(adjudications_by_unit) != 12:
        raise ValueError("history budget phase diagram development unit roster is incomplete")
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
            HistoryBudgetPhaseDiagramDevelopmentGateCount(
                family=family,
                target_nomination_count=sum(
                    value.gate is HistoryBudgetPhaseDiagramGate.TARGET for value in pair_adjudications
                ),
                sink_nomination_count=sum(
                    value.gate is HistoryBudgetPhaseDiagramGate.SINK for value in pair_adjudications
                ),
            )
        )
    adjudications = tuple(
        sorted(
            (value for bundle in adjudication_bundles for value in bundle.adjudications),
            key=lambda value: value.adjudication_id,
        )
    )
    return HistoryBudgetPhaseDiagramDevelopmentLedger(
        ledger_id="history-budget-phase-diagram.development-ledger",
        canary_report_sha256=None,
        adjudications=adjudications,
        gate_counts=tuple(sorted(gate_counts, key=lambda value: value.gate_count_id)),
        requested_unit_count=12,
        complete_unit_count=len(adjudication_bundles),
        evaluation_roster_access_count=0,
        reason_codes=(),
    )


def evaluate_development_gate(
    ledger: HistoryBudgetPhaseDiagramDevelopmentLedger,
    canary: HistoryBudgetPhaseDiagramCanaryReport,
) -> HistoryBudgetPhaseDiagramDevelopmentGate:
    reasons: set[str] = set()
    if not canary.passed:
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_CONFORMANCE_FAILED")
    if ledger.complete_unit_count != ledger.requested_unit_count:
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_DEVELOPMENT_INCOMPLETE")
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
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_BOUNDARY_TARGETING_UNPOWERED")
    if any(not value.generator_observer_agreement for value in ledger.adjudications):
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_CONFORMANCE_FAILED")
    return HistoryBudgetPhaseDiagramDevelopmentGate(
        gate_id="history-budget-phase-diagram.development-power-and-correctness-gate",
        development_ledger_sha256=ledger.fingerprint(),
        issue_evaluation=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def freeze_method(
    *,
    development_config: HistoryBudgetPhaseDiagramConfig,
    development_gate: HistoryBudgetPhaseDiagramDevelopmentGate,
    observer_implementation_sha256: str,
    generator_implementation_sha256: str,
    evaluator_implementation_sha256: str,
    development_receipt_closure_sha256: str,
) -> HistoryBudgetPhaseDiagramMethodFreeze:
    if (
        development_config.phase is not HistoryBudgetPhaseDiagramPhase.DEVELOPMENT
        or not development_gate.issue_evaluation
    ):
        raise ValueError("history budget phase diagram method freeze prerequisites differ")
    return HistoryBudgetPhaseDiagramMethodFreeze(
        freeze_id="history-budget-phase-diagram.method-freeze",
        development_config_sha256=development_config.fingerprint(),
        observer_implementation_sha256=observer_implementation_sha256,
        generator_implementation_sha256=generator_implementation_sha256,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
        development_receipt_closure_sha256=development_receipt_closure_sha256,
        frozen_threshold_ids=(
            "action-panel-5x5",
            "history-budget-phase-diagram-boundary-target-sink",
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
    method_freeze: HistoryBudgetPhaseDiagramMethodFreeze,
    evaluation_config: HistoryBudgetPhaseDiagramConfig,
    seed_roster_commitment: HistoryBudgetPhaseDiagramSeedRosterCommitment,
    implementation_source_closure_sha256: str,
) -> HistoryBudgetPhaseDiagramEvaluationDesignFreeze:
    if (
        evaluation_config.phase is not HistoryBudgetPhaseDiagramPhase.EVALUATION
        or evaluation_config.seed_roster_commitment_sha256 != seed_roster_commitment.fingerprint()
    ):
        raise ValueError("history budget phase diagram evaluation design freeze prerequisites differ")
    return HistoryBudgetPhaseDiagramEvaluationDesignFreeze(
        freeze_id="history-budget-phase-diagram.evaluation-design-freeze",
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
    "untouched_bundle",
    "UntouchedBundleExecution",
    "validate_seed_roster_commitment",
]
