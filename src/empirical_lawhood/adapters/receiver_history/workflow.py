"""Pure phase services joining receiver-history records without hiding scientific steps."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import secrets
from typing import cast

import numpy as np

from empirical_lawhood.adapters.methods.receiver_history_closure.evaluator import (
    adjudicate_targeted_coordinates,
    adjudicate_unit_scale,
    adjudicate_untouched_scale,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.history import (
    coordinate_labels,
    observe_history,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.preparation_sampler import (
    sample_untouched_preparations,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.lexical_targeting import (
    nominate_targeted_challenges,
)
from empirical_lawhood.adapters.simulators.rc_ladder.generator import (
    generate_outcomes,
    generate_untouched_outcomes,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .array_io import (
    FloatArray,
    IntArray,
    PackedArrays,
    TypedArray,
    receiver_history_array_semantics,
    logical_arrays_digest,
    pack_arrays,
    unpack_arrays,
)
from .contracts import (
    ReceiverHistoryConfig,
    ReceiverHistoryChallengeGeometry,
    ReceiverHistoryChallengeNomination,
    ReceiverHistoryEndpoint,
    ReceiverHistoryGate,
    ReceiverHistoryMethodFreeze,
    ReceiverHistoryOptimizationCertificate,
    ReceiverHistoryPhase,
    ReceiverHistoryPowerState,
    ReceiverHistorySeedRosterCommitment,
    ReceiverHistoryTargetedCoordinateAdjudication,
    ReceiverHistoryUnitAdjudication,
    ReceiverHistoryUntouchedCoordinateAdjudication,
)
from .descriptors import (
    canary_unit_ids,
    development_unit_ids,
    family_from_unit_id,
    generate_descriptor,
    reserved_non_evaluation_seed_digests,
)
from .runtime_contracts import (
    ReceiverHistoryAdjudicationBundle,
    ReceiverHistoryCanaryReport,
    ReceiverHistoryDenominatorBundle,
    ReceiverHistoryDevelopmentGate,
    ReceiverHistoryDevelopmentGateCount,
    ReceiverHistoryDevelopmentLedger,
    ReceiverHistoryEndpointCoordinatePowerAtlas,
    ReceiverHistoryEvaluationDesignFreeze,
    ReceiverHistoryGeneratorBundle,
    ReceiverHistoryHistoryBundle,
    ReceiverHistoryNominationFreeze,
    ReceiverHistoryObserverBundle,
    ReceiverHistoryRequestedUnitLedger,
    ReceiverHistorySeedEntry,
    ReceiverHistorySeedRoster,
    ReceiverHistoryUntouchedBundle,
    ReceiverHistoryUntouchedFreeze,
)


@dataclass(frozen=True, slots=True)
class ObserverBundleExecution:
    bundle: ReceiverHistoryObserverBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class HistoryBundleExecution:
    bundle: ReceiverHistoryHistoryBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class GeneratorBundleExecution:
    bundle: ReceiverHistoryGeneratorBundle
    packed_arrays: PackedArrays


@dataclass(frozen=True, slots=True)
class UntouchedBundleExecution:
    bundle: ReceiverHistoryUntouchedBundle
    float_arrays: PackedArrays
    int_arrays: PackedArrays


def _require_float_array(value: TypedArray, *, array_id: str) -> FloatArray:
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"receiver-history array {array_id} must be float64")
    return cast(FloatArray, value)


def _require_int_array(value: TypedArray, *, array_id: str) -> IntArray:
    if value.dtype != np.dtype(np.int64):
        raise ValueError(f"receiver-history array {array_id} must be int64")
    return cast(IntArray, value)


def requested_unit_ledger(config: ReceiverHistoryConfig) -> ReceiverHistoryRequestedUnitLedger:
    return ReceiverHistoryRequestedUnitLedger(
        ledger_id=f"ledger.receiver-history.{config.phase.value.lower()}.requested-units",
        phase=config.phase,
        config_sha256=config.fingerprint(),
        unit_ids=config.unit_ids,
        scale_cells=config.scale_cells,
        outcome_count=0,
    )


def create_seed_roster(
    config: ReceiverHistoryConfig,
    *,
    injected_seeds: tuple[bytes, ...] | None = None,
) -> tuple[ReceiverHistorySeedRoster, ReceiverHistorySeedRosterCommitment]:
    if config.phase is not ReceiverHistoryPhase.NOMINATION:
        raise ValueError("seed roster creation requires the nomination config")
    seeds = (
        tuple(secrets.token_bytes(32) for _ in config.unit_ids)
        if injected_seeds is None
        else injected_seeds
    )
    if len(seeds) != len(config.unit_ids) or any(len(value) != 32 for value in seeds):
        raise ValueError("receiver-history injected seed roster differs")
    if set(config.unit_ids) & {*development_unit_ids(), *canary_unit_ids()}:
        raise ValueError("receiver-history evaluation unit IDs overlap a prior phase")
    seed_digests = tuple(sha256(value).hexdigest() for value in seeds)
    if len(set(seed_digests)) != len(seed_digests):
        raise ValueError("receiver-history evaluation seeds repeat")
    if set(seed_digests) & reserved_non_evaluation_seed_digests():
        raise ValueError("receiver-history evaluation seeds overlap a canary/development seed")
    entries = tuple(
        ReceiverHistorySeedEntry(
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
    roster = ReceiverHistorySeedRoster(
        roster_id="receiver-history.evaluation-seed-roster",
        entries=entries,
        outcome_count_at_generation=0,
    )
    payload_sha256 = sha256(roster.canonical_bytes()).hexdigest()
    commitment = ReceiverHistorySeedRosterCommitment(
        roster_id="receiver-history.evaluation-seed-roster-commitment",
        evaluation_unit_ids=config.unit_ids,
        seed_payload_sha256=payload_sha256,
        commitment_sha256=sha256(
            b'receiver-history-seed-roster-commitment\x00' + roster.canonical_bytes()
        ).hexdigest(),
        seed_bytes_per_unit=32,
        public_seed_count=0,
        outcome_count_at_commitment=0,
    )
    return roster, commitment


def validate_seed_roster_commitment(
    roster: ReceiverHistorySeedRoster,
    commitment: ReceiverHistorySeedRosterCommitment,
) -> None:
    if (
        tuple(value.unit_id for value in roster.entries) != commitment.evaluation_unit_ids
        or sha256(roster.canonical_bytes()).hexdigest() != commitment.seed_payload_sha256
        or sha256(
            b'receiver-history-seed-roster-commitment\x00' + roster.canonical_bytes()
        ).hexdigest()
        != commitment.commitment_sha256
    ):
        raise ValueError("receiver-history seed roster differs from its public commitment")


def denominator_bundle(
    *,
    config: ReceiverHistoryConfig,
    unit_id: str,
    seed: bytes,
) -> ReceiverHistoryDenominatorBundle:
    if unit_id not in config.unit_ids:
        raise ValueError("receiver-history descriptor unit lies outside the phase roster")
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
    preparation_seed = sha256(
        seed + b"\0receiver-history-untouched-preparation-substream\0" + unit_id.encode("ascii")
    ).digest()
    return ReceiverHistoryDenominatorBundle(
        bundle_id=f"denominator-bundle.{unit_id}",
        unit_id=unit_id,
        preparation_substream_seed_hex=preparation_seed.hex(),
        preparation_substream_seed_sha256=sha256(preparation_seed).hexdigest(),
        descriptors=descriptors,
    )


def observer_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
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
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
) -> HistoryBundleExecution:
    forecasts = []
    coordinates = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        execution = observe_history(config=config, descriptor=descriptor)
        forecasts.append(execution.forecast)
        for coordinate in coordinate_labels(config, descriptor.scale_cells):
            coordinates.append(coordinate)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.history",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: receiver_history_array_semantics(key) for key in arrays},
    )
    bundle = ReceiverHistoryHistoryBundle(
        bundle_id=f"history-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=tuple(sorted(forecasts, key=lambda value: value.forecast_id)),
        coordinates=tuple(sorted(coordinates, key=lambda value: value.coordinate_id)),
        structural_rank_steps=(),
        discrete_rank_brackets=(),
        conditioning_steps=(),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_count=0,
    )
    return HistoryBundleExecution(bundle=bundle, packed_arrays=packed)


def targeter_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    history: ReceiverHistoryHistoryBundle,
    implementation_sha256: str,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas | None = None,
    endpoint_mask: frozenset[ReceiverHistoryEndpoint] | None = None,
) -> ObserverBundleExecution:
    if history.denominator_bundle_sha256 != denominators.fingerprint():
        raise ValueError("receiver-history targeter received another history denominator")
    by_id = {value.forecast_id: value for value in history.forecasts}
    nominations: list[ReceiverHistoryChallengeNomination] = []
    geometries: list[ReceiverHistoryChallengeGeometry] = []
    certificates: list[ReceiverHistoryOptimizationCertificate] = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        eligible_endpoints_by_coordinate = None
        if power_atlas is not None:
            capable = {
                ReceiverHistoryPowerState.WITNESS_CAPABLE,
                ReceiverHistoryPowerState.NONADVERSITY_CAPABLE,
                ReceiverHistoryPowerState.MIXED_CAPABLE,
            }
            eligible_endpoints_by_coordinate = {
                coordinate.coordinate_id: frozenset(
                    cell.endpoint
                    for cell in power_atlas.cells
                    if cell.family is descriptor.family
                    and cell.scale_cells == descriptor.scale_cells
                    and cell.coordinate_id == coordinate.coordinate_id
                    and cell.power_state in capable
                )
                for coordinate in history.coordinates
                if coordinate.scale_cells == descriptor.scale_cells
            }
        elif endpoint_mask is not None:
            if not endpoint_mask or not endpoint_mask.issubset(frozenset(ReceiverHistoryEndpoint)):
                raise ValueError("receiver-history endpoint projection mask differs")
            eligible_endpoints_by_coordinate = {
                coordinate.coordinate_id: endpoint_mask
                for coordinate in history.coordinates
                if coordinate.scale_cells == descriptor.scale_cells
            }
        execution = nominate_targeted_challenges(
            config=config,
            descriptor=descriptor,
            history_forecast=by_id[f"rank-forecast.{descriptor.unit_id}.n{descriptor.scale_cells}"],
            implementation_sha256=implementation_sha256,
            eligible_endpoints_by_coordinate=eligible_endpoints_by_coordinate,
        )
        nominations.extend(execution.nominations)
        geometries.extend(execution.geometries)
        certificates.extend(execution.certificates)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.targeter",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: receiver_history_array_semantics(key) for key in arrays},
    )
    bundle = ReceiverHistoryObserverBundle(
        bundle_id=f"observer-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        forecasts=history.forecasts,
        geometries=tuple(sorted(geometries, key=lambda value: value.nomination_id)),
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
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
) -> UntouchedBundleExecution:
    descriptors = {value.scale_cells: value for value in denominators.descriptors}
    execution = sample_untouched_preparations(
        config=config,
        unit_id=denominators.unit_id,
        descriptors=descriptors,
        seed=denominators.preparation_substream_seed,
    )
    float_arrays = dict(execution.float_arrays)
    int_arrays = dict(execution.int_arrays)
    packed_float = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.untouched-float",
        unit_id=denominators.unit_id,
        arrays=float_arrays,
        semantics={key: receiver_history_array_semantics(key) for key in float_arrays},
    )
    packed_int = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.untouched-int",
        unit_id=denominators.unit_id,
        arrays=int_arrays,
        semantics={key: receiver_history_array_semantics(key) for key in int_arrays},
    )
    bundle = ReceiverHistoryUntouchedBundle(
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
    history: ReceiverHistoryHistoryBundle,
    observer: ReceiverHistoryObserverBundle,
    untouched: ReceiverHistoryUntouchedBundle | None,
    method_freeze: ReceiverHistoryMethodFreeze | None,
) -> ReceiverHistoryNominationFreeze:
    if (
        history.unit_id != observer.unit_id
        or history.denominator_bundle_sha256 != observer.denominator_bundle_sha256
        or (
            untouched is not None
            and history.denominator_bundle_sha256 != untouched.denominator_bundle_sha256
        )
        or history.forecasts != observer.forecasts
    ):
        raise ValueError("receiver-history nomination freeze inputs differ")
    return ReceiverHistoryNominationFreeze(
        freeze_id=f"nomination-freeze.{observer.unit_id}",
        geometries=observer.geometries,
        certificate_request_keys=tuple(
            value.certificate_id for value in observer.optimization_certificates
        ),
        observer_bundle_sha256=observer.fingerprint(),
        untouched_bundle_sha256=None if untouched is None else untouched.fingerprint(),
        preparation_manifest_sha256=(
            None if untouched is None else untouched.preparation_manifest.fingerprint()
        ),
        history_bundle_sha256=history.fingerprint(),
        method_freeze_sha256=(None if method_freeze is None else method_freeze.fingerprint()),
        outcome_count_at_freeze=0,
    )


def freeze_untouched(
    *,
    untouched: ReceiverHistoryUntouchedBundle,
    method_freeze: ReceiverHistoryMethodFreeze,
) -> ReceiverHistoryUntouchedFreeze:
    return ReceiverHistoryUntouchedFreeze(
        freeze_id=f"untouched-freeze.{untouched.unit_id}",
        unit_id=untouched.unit_id,
        untouched_bundle_sha256=untouched.fingerprint(),
        preparation_manifest_sha256=untouched.preparation_manifest.fingerprint(),
        method_freeze_sha256=method_freeze.fingerprint(),
        outcome_count_at_freeze=0,
    )


def generator_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    nomination_freeze: ReceiverHistoryNominationFreeze,
    untouched: ReceiverHistoryUntouchedBundle,
    untouched_float_payload: bytes,
    untouched_float_manifest: object,
    untouched_int_payload: bytes,
    untouched_int_manifest: object,
    implementation_sha256: str,
    maximum_array_bytes: int = 256 * 1024**2,
) -> GeneratorBundleExecution:
    from .runtime_contracts import ReceiverHistoryArrayManifest

    if not isinstance(untouched_float_manifest, ReceiverHistoryArrayManifest) or not isinstance(
        untouched_int_manifest, ReceiverHistoryArrayManifest
    ):
        raise TypeError("receiver-history generator requires both typed untouched array manifests")
    if (
        untouched.denominator_bundle_sha256 != denominators.fingerprint()
        or untouched.float_arrays_manifest_sha256 != untouched_float_manifest.fingerprint()
        or untouched.int_arrays_manifest_sha256 != untouched_int_manifest.fingerprint()
    ):
        raise ValueError("receiver-history generator received another untouched cohort")
    untouched_arrays = unpack_arrays(
        untouched_float_payload,
        untouched_float_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    untouched_int_arrays = unpack_arrays(
        untouched_int_payload,
        untouched_int_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    outcomes = []
    untouched_outcomes = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        nominations = tuple(
            value
            for value in nomination_freeze.geometries
            if value.scale_cells == descriptor.scale_cells
        )
        execution = generate_outcomes(
            config=config,
            descriptor=descriptor,
            nominations=nominations,
            implementation_sha256=implementation_sha256,
            outcome_access=config.outcome_access,
            certificate_request_keys=tuple(
                value
                for value in nomination_freeze.certificate_request_keys
                if f"coordinate.n{descriptor.scale_cells}." in value
            ),
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
            coordinates=tuple(
                value
                for value in untouched.coordinates
                if value.scale_cells == descriptor.scale_cells
            ),
            validity_mask=_require_int_array(
                untouched_int_arrays[f"validity-mask-n{descriptor.scale_cells}"],
                array_id=f"validity-mask-n{descriptor.scale_cells}",
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
        semantics={key: receiver_history_array_semantics(key) for key in arrays},
    )
    bundle = ReceiverHistoryGeneratorBundle(
        bundle_id=f"generator-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        denominator_bundle_sha256=denominators.fingerprint(),
        observer_bundle_sha256=nomination_freeze.observer_bundle_sha256,
        outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
        untouched_outcomes=tuple(sorted(untouched_outcomes, key=lambda value: value.outcome_id)),
        arrays_manifest_sha256=packed.manifest.fingerprint(),
        outcome_access=config.outcome_access,
    )
    return GeneratorBundleExecution(bundle=bundle, packed_arrays=packed)


def target_generator_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    nomination_freeze: ReceiverHistoryNominationFreeze,
    implementation_sha256: str,
) -> GeneratorBundleExecution:
    if config.phase not in {ReceiverHistoryPhase.DEVELOPMENT, ReceiverHistoryPhase.TARGETED_EVALUATION}:
        raise ValueError("receiver-history target generator requires development or evaluation")
    outcomes = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        geometries = tuple(
            value
            for value in nomination_freeze.geometries
            if value.scale_cells == descriptor.scale_cells
        )
        execution = generate_outcomes(
            config=config,
            descriptor=descriptor,
            nominations=geometries,
            implementation_sha256=implementation_sha256,
            outcome_access=config.outcome_access,
            certificate_request_keys=tuple(
                value
                for value in nomination_freeze.certificate_request_keys
                if f"coordinate.n{descriptor.scale_cells}." in value
            ),
        )
        outcomes.append(execution.outcome)
        arrays.update(
            {f"n{descriptor.scale_cells}.{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.target-generator",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: receiver_history_array_semantics(key) for key in arrays},
    )
    return GeneratorBundleExecution(
        bundle=ReceiverHistoryGeneratorBundle(
            bundle_id=f"target-generator-bundle.{denominators.unit_id}",
            unit_id=denominators.unit_id,
            denominator_bundle_sha256=denominators.fingerprint(),
            observer_bundle_sha256=nomination_freeze.observer_bundle_sha256,
            outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            untouched_outcomes=(),
            arrays_manifest_sha256=packed.manifest.fingerprint(),
            outcome_access=config.outcome_access,
        ),
        packed_arrays=packed,
    )


def untouched_generator_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    untouched: ReceiverHistoryUntouchedBundle,
    untouched_float_payload: bytes,
    untouched_float_manifest: object,
    untouched_int_payload: bytes,
    untouched_int_manifest: object,
    implementation_sha256: str,
    maximum_array_bytes: int,
) -> GeneratorBundleExecution:
    from .runtime_contracts import ReceiverHistoryArrayManifest

    if config.phase is not ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
        raise ValueError("receiver-history untouched generator requires untouched evaluation")
    if not isinstance(untouched_float_manifest, ReceiverHistoryArrayManifest) or not isinstance(
        untouched_int_manifest, ReceiverHistoryArrayManifest
    ):
        raise TypeError("receiver-history untouched generator requires typed array manifests")
    if (
        untouched.denominator_bundle_sha256 != denominators.fingerprint()
        or untouched.float_arrays_manifest_sha256 != untouched_float_manifest.fingerprint()
        or untouched.int_arrays_manifest_sha256 != untouched_int_manifest.fingerprint()
    ):
        raise ValueError("receiver-history untouched generator identity differs")
    floats = unpack_arrays(
        untouched_float_payload, untouched_float_manifest, maximum_bytes=maximum_array_bytes
    )
    integers = unpack_arrays(
        untouched_int_payload, untouched_int_manifest, maximum_bytes=maximum_array_bytes
    )
    outcomes = []
    arrays: dict[str, TypedArray] = {}
    for descriptor in denominators.descriptors:
        scale = descriptor.scale_cells
        execution = generate_untouched_outcomes(
            config=config,
            descriptor=descriptor,
            preparation_manifest=untouched.preparation_manifest,
            earliest_states=_require_float_array(
                floats[f"initial-states-n{scale}"], array_id=f"initial-states-n{scale}"
            ),
            coordinates=tuple(
                value for value in untouched.coordinates if value.scale_cells == scale
            ),
            validity_mask=_require_int_array(
                integers[f"validity-mask-n{scale}"], array_id=f"validity-mask-n{scale}"
            ),
            implementation_sha256=implementation_sha256,
            outcome_access=config.outcome_access,
        )
        outcomes.append(execution.outcome)
        arrays.update(
            {f"n{scale}.untouched-{key}": value for key, value in execution.arrays.items()}
        )
    packed = pack_arrays(
        manifest_id=f"array-manifest.{denominators.unit_id}.untouched-generator",
        unit_id=denominators.unit_id,
        arrays=arrays,
        semantics={key: receiver_history_array_semantics(key) for key in arrays},
    )
    return GeneratorBundleExecution(
        bundle=ReceiverHistoryGeneratorBundle(
            bundle_id=f"untouched-generator-bundle.{denominators.unit_id}",
            unit_id=denominators.unit_id,
            denominator_bundle_sha256=denominators.fingerprint(),
            observer_bundle_sha256=untouched.fingerprint(),
            outcomes=(),
            untouched_outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            arrays_manifest_sha256=packed.manifest.fingerprint(),
            outcome_access=config.outcome_access,
        ),
        packed_arrays=packed,
    )


def adjudication_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    history: ReceiverHistoryHistoryBundle,
    observer: ReceiverHistoryObserverBundle,
    untouched: ReceiverHistoryUntouchedBundle,
    untouched_int_payload: bytes,
    untouched_int_manifest: object,
    generator: ReceiverHistoryGeneratorBundle,
    generator_arrays_payload: bytes,
    generator_arrays_manifest: object,
    method_freeze: ReceiverHistoryMethodFreeze | None,
    maximum_array_bytes: int,
) -> ReceiverHistoryAdjudicationBundle:
    from .runtime_contracts import ReceiverHistoryArrayManifest

    if not isinstance(generator_arrays_manifest, ReceiverHistoryArrayManifest) or not isinstance(
        untouched_int_manifest, ReceiverHistoryArrayManifest
    ):
        raise TypeError("receiver-history evaluator requires the typed generator array manifest")
    if (
        generator.denominator_bundle_sha256 != denominators.fingerprint()
        or history.denominator_bundle_sha256 != denominators.fingerprint()
        or history.forecasts != observer.forecasts
        or generator.observer_bundle_sha256 != observer.fingerprint()
        or generator.arrays_manifest_sha256 != generator_arrays_manifest.fingerprint()
        or untouched.int_arrays_manifest_sha256 != untouched_int_manifest.fingerprint()
    ):
        raise ValueError("receiver-history evaluator input identities differ")
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
    rows: list[ReceiverHistoryUnitAdjudication] = []
    targeted_rows: list[ReceiverHistoryTargetedCoordinateAdjudication] = []
    untouched_rows: list[ReceiverHistoryUntouchedCoordinateAdjudication] = []
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
            raise ValueError(
                "receiver-history sealed generator arrays differ from the outcome record"
            )
        nominations = tuple(value for value in observer.nominations if value.scale_cells == scale)
        geometries = tuple(value for value in observer.geometries if value.scale_cells == scale)
        unit_adjudication = adjudicate_unit_scale(
            config=config,
            descriptor=descriptor,
            forecast=forecasts[forecast_id],
            geometries=geometries,
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
                sparse_certificate_checks=outcome.sparse_certificate_checks,
                expected_action_ids=tuple(
                    sorted(
                        f"action.a{amplitude_index:02d}.d{duration_index:02d}"
                        for amplitude_index, _ in enumerate(config.action_amplitudes_u_star)
                        for duration_index, _ in enumerate(config.action_durations_t_star)
                    )
                ),
                optimization_tolerance=config.optimization_tolerance,
                eligible_endpoints_by_coordinate={
                    coordinate.coordinate_id: frozenset(ReceiverHistoryEndpoint)
                    for coordinate in coordinate_labels(config, scale)
                },
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
            raise ValueError("receiver-history untouched generator arrays differ from their record")
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
                action_receiver=_require_float_array(
                    untouched_scale_arrays["action-receiver"],
                    array_id=f"n{scale}.untouched-action-receiver",
                ),
                validity_mask=_require_int_array(
                    untouched_int_arrays[f"validity-mask-n{scale}"],
                    array_id=f"validity-mask-n{scale}",
                ),
                generator_outcome=untouched_outcome,
            )
        )
    return ReceiverHistoryAdjudicationBundle(
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


def target_adjudication_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    history: ReceiverHistoryHistoryBundle,
    observer: ReceiverHistoryObserverBundle,
    generator: ReceiverHistoryGeneratorBundle,
    generator_arrays_payload: bytes,
    generator_arrays_manifest: object,
    method_freeze: ReceiverHistoryMethodFreeze | None,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas | None,
    maximum_array_bytes: int,
) -> ReceiverHistoryAdjudicationBundle:
    from .runtime_contracts import ReceiverHistoryArrayManifest

    if config.phase not in {
        ReceiverHistoryPhase.DEVELOPMENT,
        ReceiverHistoryPhase.TARGETED_EVALUATION,
    } or not isinstance(generator_arrays_manifest, ReceiverHistoryArrayManifest):
        raise ValueError("receiver-history target evaluator input type differs")
    if (
        history.denominator_bundle_sha256 != denominators.fingerprint()
        or observer.denominator_bundle_sha256 != denominators.fingerprint()
        or generator.denominator_bundle_sha256 != denominators.fingerprint()
        or generator.observer_bundle_sha256 != observer.fingerprint()
        or generator.arrays_manifest_sha256 != generator_arrays_manifest.fingerprint()
        or generator.untouched_outcomes
    ):
        raise ValueError("receiver-history target evaluator identities differ")
    arrays = unpack_arrays(
        generator_arrays_payload,
        generator_arrays_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    forecasts = {value.forecast_id: value for value in observer.forecasts}
    outcomes = {value.outcome_id: value for value in generator.outcomes}
    rows: list[ReceiverHistoryUnitAdjudication] = []
    targeted_rows: list[ReceiverHistoryTargetedCoordinateAdjudication] = []
    expected_actions = tuple(
        sorted(
            f"action.a{amplitude_index:02d}.d{duration_index:02d}"
            for amplitude_index, _ in enumerate(config.action_amplitudes_u_star)
            for duration_index, _ in enumerate(config.action_durations_t_star)
        )
    )
    for descriptor in denominators.descriptors:
        scale = descriptor.scale_cells
        outcome = outcomes[f"generator-outcome.{descriptor.unit_id}.n{scale}"]
        scale_arrays = {
            key.split(".", 1)[1]: value
            for key, value in arrays.items()
            if key.startswith(f"n{scale}.")
        }
        if logical_arrays_digest(scale_arrays) != outcome.arrays_sha256:
            raise ValueError("receiver-history target generator arrays differ")
        nominations = tuple(value for value in observer.nominations if value.scale_cells == scale)
        geometries = tuple(value for value in observer.geometries if value.scale_cells == scale)
        unit = adjudicate_unit_scale(
            config=config,
            descriptor=descriptor,
            forecast=forecasts[f"rank-forecast.{descriptor.unit_id}.n{scale}"],
            geometries=geometries,
            nominations=nominations,
            generator_outcome=outcome,
            method_freeze=method_freeze,
        )
        rows.append(unit)
        targeted_rows.extend(
            adjudicate_targeted_coordinates(
                descriptor=descriptor,
                coordinates=tuple(
                    value for value in history.coordinates if value.scale_cells == scale
                ),
                unit_adjudication=unit,
                certificates=tuple(
                    value
                    for value in observer.optimization_certificates
                    if value.coordinate_id.startswith(f"coordinate.n{scale}.")
                ),
                sparse_certificate_checks=outcome.sparse_certificate_checks,
                expected_action_ids=expected_actions,
                optimization_tolerance=config.optimization_tolerance,
                eligible_endpoints_by_coordinate={
                    coordinate.coordinate_id: (
                        frozenset(ReceiverHistoryEndpoint)
                        if power_atlas is None
                        else frozenset(
                            cell.endpoint
                            for cell in power_atlas.cells
                            if cell.family is descriptor.family
                            and cell.scale_cells == descriptor.scale_cells
                            and cell.coordinate_id == coordinate.coordinate_id
                            and cell.power_state
                            in {
                                ReceiverHistoryPowerState.WITNESS_CAPABLE,
                                ReceiverHistoryPowerState.NONADVERSITY_CAPABLE,
                                ReceiverHistoryPowerState.MIXED_CAPABLE,
                            }
                        )
                    )
                    for coordinate in history.coordinates
                    if coordinate.scale_cells == scale
                },
            )
        )
    return ReceiverHistoryAdjudicationBundle(
        bundle_id=f"target-adjudication-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        method_freeze_sha256=(None if method_freeze is None else method_freeze.fingerprint()),
        history_bundle=history,
        adjudications=tuple(sorted(rows, key=lambda value: value.adjudication_id)),
        targeted_adjudications=tuple(
            sorted(targeted_rows, key=lambda value: value.adjudication_id)
        ),
        untouched_adjudications=(),
        raw_generator_payload_exposed=False,
    )


def untouched_adjudication_bundle(
    *,
    config: ReceiverHistoryConfig,
    denominators: ReceiverHistoryDenominatorBundle,
    untouched: ReceiverHistoryUntouchedBundle,
    untouched_int_payload: bytes,
    untouched_int_manifest: object,
    generator: ReceiverHistoryGeneratorBundle,
    generator_arrays_payload: bytes,
    generator_arrays_manifest: object,
    method_freeze: ReceiverHistoryMethodFreeze,
    maximum_array_bytes: int,
) -> ReceiverHistoryAdjudicationBundle:
    from .runtime_contracts import ReceiverHistoryArrayManifest

    if config.phase is not ReceiverHistoryPhase.UNTOUCHED_EVALUATION or not all(
        isinstance(value, ReceiverHistoryArrayManifest)
        for value in (untouched_int_manifest, generator_arrays_manifest)
    ):
        raise ValueError("receiver-history untouched evaluator input type differs")
    assert isinstance(untouched_int_manifest, ReceiverHistoryArrayManifest)
    assert isinstance(generator_arrays_manifest, ReceiverHistoryArrayManifest)
    if (
        untouched.denominator_bundle_sha256 != denominators.fingerprint()
        or generator.denominator_bundle_sha256 != denominators.fingerprint()
        or generator.observer_bundle_sha256 != untouched.fingerprint()
        or generator.outcomes
        or generator.arrays_manifest_sha256 != generator_arrays_manifest.fingerprint()
        or untouched.int_arrays_manifest_sha256 != untouched_int_manifest.fingerprint()
    ):
        raise ValueError("receiver-history untouched evaluator identities differ")
    arrays = unpack_arrays(
        generator_arrays_payload,
        generator_arrays_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    integers = unpack_arrays(
        untouched_int_payload,
        untouched_int_manifest,
        maximum_bytes=maximum_array_bytes,
    )
    outcomes = {value.scale_cells: value for value in generator.untouched_outcomes}
    rows: list[ReceiverHistoryUntouchedCoordinateAdjudication] = []
    for descriptor in denominators.descriptors:
        scale = descriptor.scale_cells
        scale_arrays = {
            key.split(".untouched-", 1)[1]: value
            for key, value in arrays.items()
            if key.startswith(f"n{scale}.untouched-")
        }
        outcome = outcomes[scale]
        if logical_arrays_digest(scale_arrays) != outcome.arrays_sha256:
            raise ValueError("receiver-history untouched generator arrays differ")
        rows.extend(
            adjudicate_untouched_scale(
                config=config,
                descriptor=descriptor,
                coordinates=tuple(
                    value for value in untouched.coordinates if value.scale_cells == scale
                ),
                collision_edges=_require_int_array(
                    integers[f"collision-edges-n{scale}"],
                    array_id=f"collision-edges-n{scale}",
                ),
                receiver_history=_require_float_array(
                    scale_arrays["receiver-history"],
                    array_id=f"n{scale}.untouched-receiver-history",
                ),
                action_metrics=_require_float_array(
                    scale_arrays["action-metrics"],
                    array_id=f"n{scale}.untouched-action-metrics",
                ),
                action_receiver=_require_float_array(
                    scale_arrays["action-receiver"],
                    array_id=f"n{scale}.untouched-action-receiver",
                ),
                validity_mask=_require_int_array(
                    integers[f"validity-mask-n{scale}"],
                    array_id=f"validity-mask-n{scale}",
                ),
                generator_outcome=outcome,
            )
        )
    return ReceiverHistoryAdjudicationBundle(
        bundle_id=f"untouched-adjudication-bundle.{denominators.unit_id}",
        unit_id=denominators.unit_id,
        method_freeze_sha256=method_freeze.fingerprint(),
        history_bundle=None,
        adjudications=(),
        targeted_adjudications=(),
        untouched_adjudications=tuple(sorted(rows, key=lambda value: value.adjudication_id)),
        raw_generator_payload_exposed=False,
    )


def development_ledger(
    *,
    adjudication_bundles: tuple[ReceiverHistoryAdjudicationBundle, ...],
) -> ReceiverHistoryDevelopmentLedger:
    adjudications_by_unit = {value.unit_id: value for value in adjudication_bundles}
    if len(adjudications_by_unit) != 12:
        raise ValueError("receiver-history development unit roster is incomplete")
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
            ReceiverHistoryDevelopmentGateCount(
                family=family,
                target_nomination_count=sum(
                    value.gate is ReceiverHistoryGate.TARGET for value in pair_adjudications
                ),
                sink_nomination_count=sum(
                    value.gate is ReceiverHistoryGate.SINK for value in pair_adjudications
                ),
            )
        )
    adjudications = tuple(
        sorted(
            (value for bundle in adjudication_bundles for value in bundle.adjudications),
            key=lambda value: value.adjudication_id,
        )
    )
    targeted = tuple(
        sorted(
            (value for bundle in adjudication_bundles for value in bundle.targeted_adjudications),
            key=lambda value: value.adjudication_id,
        )
    )
    return ReceiverHistoryDevelopmentLedger(
        ledger_id="receiver-history.development-ledger",
        canary_report_sha256=None,
        adjudications=adjudications,
        targeted_adjudications=targeted,
        gate_counts=tuple(sorted(gate_counts, key=lambda value: value.gate_count_id)),
        requested_unit_count=12,
        complete_unit_count=len(adjudication_bundles),
        evaluation_roster_access_count=0,
        reason_codes=(),
    )


def evaluate_development_gate(
    ledger: ReceiverHistoryDevelopmentLedger,
    canary: ReceiverHistoryCanaryReport,
) -> ReceiverHistoryDevelopmentGate:
    reasons: set[str] = set()
    if not canary.passed:
        reasons.add('RECEIVER_HISTORY_CANARY_CONFORMANCE_FAILED')
    if ledger.complete_unit_count != ledger.requested_unit_count:
        reasons.add('RECEIVER_HISTORY_DEVELOPMENT_INCOMPLETE')
    if any(not value.generator_observer_agreement for value in ledger.adjudications):
        reasons.add('RECEIVER_HISTORY_CANARY_CONFORMANCE_FAILED')
    if any(not value.valid for value in ledger.targeted_adjudications):
        reasons.add('RECEIVER_HISTORY_ENDPOINT_IMPLEMENTATION_DISAGREEMENT')
    return ReceiverHistoryDevelopmentGate(
        gate_id="receiver-history.development-power-and-correctness-gate",
        development_ledger_sha256=ledger.fingerprint(),
        issue_evaluation=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def freeze_method(
    *,
    development_config: ReceiverHistoryConfig,
    development_gate: ReceiverHistoryDevelopmentGate,
    observer_implementation_sha256: str,
    generator_implementation_sha256: str,
    evaluator_implementation_sha256: str,
    development_receipt_closure_sha256: str,
) -> ReceiverHistoryMethodFreeze:
    if (
        development_config.phase is not ReceiverHistoryPhase.DEVELOPMENT
        or not development_gate.issue_evaluation
    ):
        raise ValueError("receiver-history method freeze prerequisites differ")
    return ReceiverHistoryMethodFreeze(
        freeze_id="receiver-history.method-freeze",
        development_config_sha256=development_config.fingerprint(),
        observer_implementation_sha256=observer_implementation_sha256,
        generator_implementation_sha256=generator_implementation_sha256,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
        development_receipt_closure_sha256=development_receipt_closure_sha256,
        frozen_threshold_ids=(
            (
                'action-panel-3x3-low-mid-high'
                if len(development_config.action_amplitudes_u_star) == 3
                else 'action-panel-5x5'
            ),
            'boundary-target-sink',
            'complete-unit-intention-to-treat',
            'history-rank-relative-1e-12',
            'midpoint-boundary-tie-1e-10',
            'receiver-r8-capacitance-weighted',
        ),
        evaluation_outcome_count=0,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
    )


def freeze_evaluation_design(
    *,
    method_freeze: ReceiverHistoryMethodFreeze,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas,
    evaluation_config: ReceiverHistoryConfig,
    seed_roster_commitment: ReceiverHistorySeedRosterCommitment,
    implementation_source_closure_sha256: str,
) -> ReceiverHistoryEvaluationDesignFreeze:
    if (
        evaluation_config.phase is not ReceiverHistoryPhase.TARGETED_EVALUATION
        or evaluation_config.seed_roster_commitment_sha256 != seed_roster_commitment.fingerprint()
        or power_atlas.invalid_cell_count
    ):
        raise ValueError("receiver-history evaluation design freeze prerequisites differ")
    return ReceiverHistoryEvaluationDesignFreeze(
        freeze_id="receiver-history.evaluation-design-freeze",
        method_freeze=method_freeze,
        power_atlas=power_atlas,
        evaluation_config_sha256=evaluation_config.fingerprint(),
        continuation_predicate_id='receiver-history-powered-endpoint-mask',
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
    "target_adjudication_bundle",
    "untouched_adjudication_bundle",
    "create_seed_roster",
    "denominator_bundle",
    "development_ledger",
    "evaluate_development_gate",
    "freeze_evaluation_design",
    "freeze_method",
    "freeze_nominations",
    "freeze_untouched",
    "generator_bundle",
    "target_generator_bundle",
    "untouched_generator_bundle",
    "history_bundle",
    "observer_bundle",
    "receipt_closure_sha256",
    "requested_unit_ledger",
    "targeter_bundle",
    "untouched_bundle",
    "UntouchedBundleExecution",
    "validate_seed_roster_commitment",
]
