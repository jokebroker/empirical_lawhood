"""MAST-U archive binding to the neutral parameterised campaign contract."""

from __future__ import annotations

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.response_experiment_ports import NativeClockContract, NativeInteractionKind, NativeReceiverContract, ResponseSubstrateBinding


def build_mastu_archive_source_qualification_prospective_binding(
    *,
    binding_id: str,
    native_config: ObjectIdentity,
    installed_executable_binding: ObjectIdentity,
    provider_implementation_sha256: str,
    native_request_schema: str,
    native_object_schema: str,
    native_projection_schema: str,
    receivers: tuple[NativeReceiverContract, ...],
    clocks: tuple[NativeClockContract, ...],
) -> ResponseSubstrateBinding:
    """Build outcome-blind archive semantics without importing a UDA client."""

    return ResponseSubstrateBinding(
        binding_id=binding_id,
        medium_id="medium.mastu-archive-pf",
        interaction_kind=NativeInteractionKind.READ_ONLY_ACQUISITION,
        provider_key='mastu-archive.parameterised-acquisition',
        provider_version="1.0.0",
        provider_implementation_sha256=provider_implementation_sha256,
        installed_executable_binding=installed_executable_binding,
        source_profile_schema='empirical-lawhood/planning/source-pipeline-profile',
        native_request_schema=native_request_schema,
        native_object_schema=native_object_schema,
        native_episode_schema=None,
        native_projection_schema=native_projection_schema,
        native_config=native_config,
        receivers=receivers,
        clocks=clocks,
        action_contract=None,
        external_source_contacted=False,
        scientific_verdict_constructed=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = ['build_mastu_archive_source_qualification_prospective_binding']
