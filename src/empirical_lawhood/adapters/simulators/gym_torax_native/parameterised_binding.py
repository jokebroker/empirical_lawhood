"""Gym--TORAX-owned parameterised native binding constants."""

from __future__ import annotations

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding


def build_gym_torax_source_qualification_prospective_binding(
    *,
    binding_id: str,
    native_config: ObjectIdentity,
    installed_executable_binding: ObjectIdentity,
    provider_implementation_sha256: str,
    receivers: tuple[NativeReceiverContract, ...],
) -> ResponseSubstrateBinding:
    """Build import-light interactive semantics without importing Gym or TORAX."""

    clocks = (
        NativeClockContract(
            clock_id="gym-torax.env-clock",
            native_unit="step",
            frame_id="gym-torax.environment-frame",
        ),
        NativeClockContract(
            clock_id="gym-torax.solver-clock",
            native_unit="s",
            frame_id="gym-torax.solver-time-frame",
        ),
    )
    return ResponseSubstrateBinding(
        binding_id=binding_id,
        medium_id='medium.gym-torax-native.prepared-episode',
        interaction_kind=NativeInteractionKind.INTERACTIVE_EXECUTION,
        provider_key='gym-torax-native.parameterised-source-acquisition',
        provider_version="1.0.0",
        provider_implementation_sha256=provider_implementation_sha256,
        installed_executable_binding=installed_executable_binding,
        source_profile_schema='empirical-lawhood/simulators/gym-torax-native/source-profile-interface',
        native_request_schema=('empirical-lawhood/simulators/gym-torax-native/execution-request-interface'),
        native_object_schema=None,
        native_episode_schema='empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-native-episode',
        native_projection_schema='empirical-lawhood/simulators/gym-torax-native/gym-torax-native-projection',
        native_config=native_config,
        receivers=receivers,
        clocks=clocks,
        action_contract=NativeActionContract(
            action_coordinate_id="gym-torax.action-profile",
            requested_observable=True,
            accepted_observable=True,
            applied_observable=True,
            realized_observable=True,
            native_hold_token="ZERO_PERTURBATION_HOLD",
        ),
        external_source_contacted=False,
        scientific_verdict_constructed=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = ['build_gym_torax_source_qualification_prospective_binding']
