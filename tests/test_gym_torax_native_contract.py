from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from empirical_lawhood.adapters.simulators.gym_torax_native.action_word import GYM_TORAX_LOWER_IP_ACTION_WORD_ID, build_gym_torax_action_word, build_gym_torax_native_schedule
from empirical_lawhood.adapters.simulators.gym_torax_native.diagnostic_contracts import GymToraxObservationDisposition, GymToraxPreparation, GymToraxSourceDisposition, encode_gym_torax_float64_block as encode_gym_torax_diagnostic_float64_block, gym_torax_numerical_members
from empirical_lawhood.adapters.simulators.gym_torax_native.field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxRequestRole, encode_gym_torax_float64_block
from empirical_lawhood.adapters.simulators.gym_torax_native.extraction_manifest import GymToraxBoundedExtractionManifest, build_gym_torax_extraction_manifest
from empirical_lawhood.adapters.simulators.gym_torax_native.field_metadata import GymToraxFieldMetadataManifest, build_gym_torax_field_metadata_manifest
from empirical_lawhood.adapters.simulators.gym_torax_native.metadata_barrier import GymToraxNativeMetadataCanaryDisposition, adjudicate_gym_torax_native_metadata_canary
from empirical_lawhood.adapters.simulators.gym_torax_native.runtime import GymToraxNativeActionCodec, GymToraxRuntimePreflightError, GymToraxRuntimeProbe, acquire_gym_torax_episode
from empirical_lawhood.adapters.simulators.gym_torax_native.source_qualification import GymToraxNativeSourceQualificationDisposition, qualify_gym_torax_native_source
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity


def _probe() -> GymToraxRuntimeProbe:
    return GymToraxRuntimeProbe(
        python_version="3.11.14",
        numpy_version="2.4.6",
        scipy_version="1.17.1",
        xarray_version="2026.7.0",
        gymtorax_version="1.1.1",
        torax_version="1.4.2",
        jax_version="0.10.2",
        jaxlib_version="0.10.2",
        backend="cpu",
        x64_enabled=True,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _request() -> tuple[
    GymToraxFieldMetadataEpisodeRequest,
    GymToraxBoundedExtractionManifest,
    GymToraxFieldMetadataManifest,
]:
    extraction_manifest = build_gym_torax_extraction_manifest(_REPOSITORY_ROOT)
    field_manifest = build_gym_torax_field_metadata_manifest()
    request = GymToraxFieldMetadataEpisodeRequest(
        request_id='request.tokamak-control.injected-metadata-canary',
        request_role=GymToraxRequestRole.METADATA_CANARY,
        source_qualification=None,
        extraction_manifest=ObjectIdentity.from_record(
            extraction_manifest.manifest_id,
            extraction_manifest,
        ),
        field_metadata_manifest=ObjectIdentity.from_record(
            field_manifest.manifest_id,
            field_manifest,
        ),
        preparation=GymToraxPreparation(
            preparation_id='preparation.tokamak-control.injected',
            physical_independent_unit_id='unit.tokamak-control.injected',
            environment_seed=17,
            initial_temperature_scale=Decimal(1),
            initial_density_nbar=Decimal("0.85"),
            bootstrap_multiplier=Decimal(1),
            inner_transport_scale=Decimal(1),
        ),
        numerical_member=next(
            value
            for value in gym_torax_numerical_members()
            if value.member_id == 'member.tokamak-control.primary'
        ),
        schedule=build_gym_torax_native_schedule(
            build_gym_torax_action_word(GYM_TORAX_LOWER_IP_ACTION_WORD_ID)
        ),
        maximum_output_bytes=64 * 1024 * 1024,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    return request, extraction_manifest, field_manifest


class _MetadataCompleteEnvironment:
    def __init__(self, *, drift_q_dimension: bool = False) -> None:
        self.current_time = 0
        self.state: Mapping[str, object] = {}
        self.drift_q_dimension = drift_q_dimension
        self.closed = False

    def _state(self) -> Mapping[str, object]:
        profile = np.asarray((1.0, 1.1, 1.2), dtype=np.float64)
        return {
            "profiles": {
                "T_e": profile,
                "T_i": profile,
                "n_e": profile,
                "psi": profile,
                "q": np.asarray((1.0, 1.1), dtype=np.float64),
            },
            "scalars": {
                field_id: np.asarray((1.0,), dtype=np.float64)
                for field_id in (
                    "H98",
                    "P_heat_total",
                    "P_radiation_e",
                    "Q_fusion",
                    "beta_N",
                    "fgw_n_e_volume_avg",
                    "q95",
                    "q_min",
                )
            },
            "numerics": {
                "inner_solver_iterations": np.asarray((2.0,)),
                "outer_solver_iterations": np.asarray((1.0,)),
                "sawtooth_crash": np.asarray((0.0,)),
                "solver_error_state": np.asarray((0.0,)),
            },
        }

    def source_dimensions(
        self,
    ) -> tuple[
        Mapping[tuple[str, str], tuple[str, ...]],
        Mapping[str, npt.NDArray[np.float64]],
    ]:
        dimensions: dict[tuple[str, str], tuple[str, ...]] = {
            ("profiles", field_id): ("rho_norm",)
            for field_id in ("T_e", "T_i", "n_e", "psi")
        }
        dimensions[("profiles", "q")] = (
            ("rho_norm",) if self.drift_q_dimension else ("rho_face_norm",)
        )
        dimensions.update(
            {
                (category, field_id): ("value",)
                for category, field_ids in (
                    (
                        "scalars",
                        (
                            "H98",
                            "P_heat_total",
                            "P_radiation_e",
                            "Q_fusion",
                            "beta_N",
                            "fgw_n_e_volume_avg",
                            "q95",
                            "q_min",
                        ),
                    ),
                    (
                        "numerics",
                        (
                            "inner_solver_iterations",
                            "outer_solver_iterations",
                            "sawtooth_crash",
                            "solver_error_state",
                        ),
                    ),
                )
                for field_id in field_ids
            }
        )
        return dimensions, {
            "rho_norm": np.asarray((0.0, 0.5, 1.0), dtype=np.float64),
            "rho_face_norm": np.asarray((0.0, 1.0), dtype=np.float64),
        }

    def reset(self, *, seed: int) -> tuple[Mapping[str, object], Mapping[str, object]]:
        assert seed == 17
        self.current_time = 0
        self.state = self._state()
        return self.state, {}

    def step(
        self,
        action: Mapping[str, npt.NDArray[np.float64]],
    ) -> tuple[Mapping[str, object], float, bool, bool, Mapping[str, object]]:
        GymToraxNativeActionCodec.decode(action)
        self.current_time += 1
        self.state = self._state()
        return (
            self.state,
            0.0,
            False,
            False,
            {
                "applied_action": action,
                "realized_action": action,
            },
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.native("gymtorax", "torax")
def test_injected_canary_crosses_the_complete_metadata_barrier() -> None:
    request, extraction_manifest, manifest = _request()
    environment = _MetadataCompleteEnvironment()

    episode = acquire_gym_torax_episode(
        request,
        extraction_manifest=extraction_manifest,
        field_metadata_manifest=manifest,
        repository_root=_REPOSITORY_ROOT,
        environment_factory=lambda _preparation, _member: environment,
        runtime_inspector=_probe,
    )

    assert environment.closed
    assert episode.state_clocks == tuple(range(121))
    assert episode.observation_disposition is GymToraxObservationDisposition.COMPLETE
    assert "OPERATOR_API_UNAVAILABLE" in episode.reason_codes
    source_blocks = tuple(
        value for value in episode.blocks if value.category.startswith("source-")
    )
    assert len(source_blocks) == 17
    assert all(value.field_metadata_id for value in source_blocks)
    assert all(
        value.native_unit != "source-unit-unspecified" for value in source_blocks
    )
    assert all(value.native_frame_id for value in source_blocks)
    receipt = adjudicate_gym_torax_native_metadata_canary(
        episode=episode,
        field_metadata_manifest=manifest,
    )
    assert receipt.disposition is GymToraxNativeMetadataCanaryDisposition.PASS
    assert not receipt.grants_source_assessment_freeze_authority
    assert not receipt.grants_source_assessment_execution_authority
    qualification = qualify_gym_torax_native_source(
        episode=episode,
        canary=receipt,
        extraction_manifest=extraction_manifest,
        field_metadata_manifest=manifest,
    )
    assert (
        qualification.disposition
        is GymToraxNativeSourceQualificationDisposition.QUALIFIED
    )
    assert not qualification.grants_source_assessment_freeze_authority
    assert not qualification.grants_source_assessment_execution_authority


@pytest.mark.native("gymtorax", "torax")
def test_source_dimension_drift_is_partial_and_named() -> None:
    request, extraction_manifest, manifest = _request()

    episode = acquire_gym_torax_episode(
        request,
        extraction_manifest=extraction_manifest,
        field_metadata_manifest=manifest,
        repository_root=_REPOSITORY_ROOT,
        environment_factory=lambda _preparation, _member: _MetadataCompleteEnvironment(
            drift_q_dimension=True
        ),
        runtime_inspector=_probe,
    )

    assert episode.observation_disposition is GymToraxObservationDisposition.PARTIAL
    assert "SOURCE_FIELD_DIMENSION_METADATA_MISMATCH" in episode.reason_codes
    receipt = adjudicate_gym_torax_native_metadata_canary(
        episode=episode,
        field_metadata_manifest=manifest,
    )
    assert receipt.disposition is GymToraxNativeMetadataCanaryDisposition.FAIL
    assert "CANARY_OBSERVATION_INCOMPLETE" in receipt.reason_codes


def test_manifest_identity_drift_refuses_before_environment_construction() -> None:
    request, extraction_manifest, manifest = _request()
    bad_request = replace(
        request,
        field_metadata_manifest=replace(
            request.field_metadata_manifest,
            object_fingerprint="0" * 64,
        ),
    )
    factory_called = False

    def factory(_preparation: object, _member: object) -> _MetadataCompleteEnvironment:
        nonlocal factory_called
        factory_called = True
        return _MetadataCompleteEnvironment()

    with pytest.raises(GymToraxRuntimePreflightError) as error:
        acquire_gym_torax_episode(
            bad_request,
            extraction_manifest=extraction_manifest,
            field_metadata_manifest=manifest,
            repository_root=_REPOSITORY_ROOT,
            environment_factory=factory,
            runtime_inspector=_probe,
        )

    assert error.value.reason_codes == ("FIELD_METADATA_MANIFEST_DIVERGENCE",)
    assert not factory_called


@pytest.mark.native("gymtorax", "torax")
def test_reset_observer_records_the_exact_attempt_boundary() -> None:
    request, extraction_manifest, manifest = _request()
    environment = _MetadataCompleteEnvironment()
    observed = 0

    def observe() -> None:
        nonlocal observed
        observed += 1

    def fail_reset(*, seed: int) -> object:
        assert seed == 17
        raise RuntimeError("injected reset failure")

    environment.reset = fail_reset  # type: ignore[method-assign]
    episode = acquire_gym_torax_episode(
        request,
        extraction_manifest=extraction_manifest,
        field_metadata_manifest=manifest,
        repository_root=_REPOSITORY_ROOT,
        environment_factory=lambda _preparation, _member: environment,
        runtime_inspector=_probe,
        source_reset_observer=observe,
    )

    assert observed == 1
    assert episode.source_reset_attempted
    assert episode.observation_disposition is GymToraxObservationDisposition.UNEVALUABLE
    assert environment.closed


@pytest.mark.native("gymtorax", "torax")
def test_post_reset_missing_source_blocks_remain_typed_evidence() -> None:
    request, extraction_manifest, manifest = _request()
    environment = _MetadataCompleteEnvironment()

    def reset_without_retained_source(
        *, seed: int
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        assert seed == 17
        environment.current_time = 0
        environment.state = {"profiles": {}, "scalars": {}, "numerics": {}}
        return environment.state, {}

    def fail_first_step(
        _action: Mapping[str, npt.NDArray[np.float64]],
    ) -> tuple[Mapping[str, object], float, bool, bool, Mapping[str, object]]:
        raise ValueError("injected post-reset source failure")

    environment.reset = reset_without_retained_source  # type: ignore[method-assign]
    environment.step = fail_first_step  # type: ignore[method-assign]
    episode = acquire_gym_torax_episode(
        request,
        extraction_manifest=extraction_manifest,
        field_metadata_manifest=manifest,
        repository_root=_REPOSITORY_ROOT,
        environment_factory=lambda _preparation, _member: environment,
        runtime_inspector=_probe,
    )

    assert environment.closed
    assert episode.state_clocks == (0,)
    assert episode.source_disposition is GymToraxSourceDisposition.SOURCE_UNAVAILABLE
    assert episode.observation_disposition is GymToraxObservationDisposition.PARTIAL
    assert "SIMULATOR_VALUEERROR" in episode.reason_codes
    assert not tuple(
        value for value in episode.blocks if value.category.startswith("source-")
    )


def test_diagnostic_unspecified_block_is_not_a_valid_metadata_block() -> None:
    diagnostic_block = encode_gym_torax_diagnostic_float64_block(
        block_id="block.source-profile.t-e",
        category="source-profile",
        native_field_id="T_e",
        native_unit="source-unit-unspecified",
        dimension_ids=("state-clock", "source-axis-0"),
        values=np.ones((1, 1), dtype=np.float64),
        clock_values=(0,),
    )
    assert diagnostic_block.native_unit == "source-unit-unspecified"

    with pytest.raises(ValueError, match="cannot carry unspecified units"):
        encode_gym_torax_float64_block(
            block_id=diagnostic_block.block_id,
            category=diagnostic_block.category,
            native_field_id=diagnostic_block.native_field_id,
            native_unit=diagnostic_block.native_unit,
            native_frame_id='frame.tokamak-control.rho-norm',
            field_metadata_id='field-metadata.tokamak-control.profiles.t-e',
            dimension_ids=("state-clock", "rho_norm"),
            values=diagnostic_block.array(),
            clock_values=(0,),
        )
