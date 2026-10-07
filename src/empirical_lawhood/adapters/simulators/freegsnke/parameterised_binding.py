"""FreeGSNKE binding to the neutral parameterised campaign contract."""

from __future__ import annotations

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.response_experiment_ports import NativeActionContract, NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding


def build_freegsnke_source_qualification_prospective_binding(
    *,
    binding_id: str,
    native_config: ObjectIdentity,
    installed_executable_binding: ObjectIdentity,
    provider_implementation_sha256: str,
    native_request_schema: str,
    native_episode_schema: str,
    native_projection_schema: str,
    receivers: tuple[NativeReceiverContract, ...],
    clocks: tuple[NativeClockContract, ...],
    native_hold_token: str | None,
) -> ResponseSubstrateBinding:
    """Declare interactive action-chain truth without importing FreeGSNKE."""

    return ResponseSubstrateBinding(
        binding_id=binding_id,
        medium_id="medium.mastu-freegsnke-dynamic",
        interaction_kind=NativeInteractionKind.INTERACTIVE_EXECUTION,
        provider_key="freegsnke.parameterised-response-acquisition",
        provider_version="1.0.0",
        provider_implementation_sha256=provider_implementation_sha256,
        installed_executable_binding=installed_executable_binding,
        source_profile_schema='empirical-lawhood/planning/source-pipeline-profile',
        native_request_schema=native_request_schema,
        native_object_schema=None,
        native_episode_schema=native_episode_schema,
        native_projection_schema=native_projection_schema,
        native_config=native_config,
        receivers=receivers,
        clocks=clocks,
        action_contract=NativeActionContract(
            action_coordinate_id="freegsnke.pf-voltage-vector",
            requested_observable=True,
            accepted_observable=True,
            applied_observable=True,
            realized_observable=True,
            native_hold_token=native_hold_token,
        ),
        external_source_contacted=False,
        scientific_verdict_constructed=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = ['build_freegsnke_source_qualification_prospective_binding']
