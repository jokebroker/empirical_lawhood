"""Neutral materialisation of parameterised native acquisition/execution intents."""

from __future__ import annotations

from typing import TypedDict

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionGroup, ResponseAcquisitionView, ResponsePreparationUnit, ResponseExperimentExtensionSet
from empirical_lawhood.planning.observation_order import ObservationAcquisitionGroup, ObservationPhysicalUnit, ObservationOrderExperimentExtension
from empirical_lawhood.runtime.response_experiment_ports import NativeAcquisitionRequest, NativeExecutionRequest, NativeInteractionKind, ResponseSubstrateBinding


class _NativeRequestFields(TypedDict):
    request_id: str
    experiment_extension_set: ObjectIdentity
    substrate_binding: ObjectIdentity
    stage_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    preparation_coordinate_id: str
    preparation_instance_id: str
    preparation_sha256: str
    adapter_realization_id: str | None
    native_request_schema: str
    native_config: ObjectIdentity
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    causal_cutoff_id: str
    grants_execution_authority: bool
    outcome_access: OutcomeAccess


def _validate_lineage(
    *,
    extension_set: ResponseExperimentExtensionSet,
    binding: ResponseSubstrateBinding,
    installed_executable_binding: ObjectIdentity,
    physical_unit: ResponsePreparationUnit,
    acquisition_group: ResponseAcquisitionGroup,
) -> None:
    bound_identification_config = extension_set.identification_config
    if binding.installed_executable_binding != installed_executable_binding:
        raise ValueError("native binding differs from its installed executable descriptor")
    if physical_unit not in bound_identification_config.physical_units or acquisition_group not in bound_identification_config.acquisition_groups:
        raise ValueError("native request inputs lie outside the frozen measurement through local law config")
    if (
        acquisition_group.physical_independent_unit_id != physical_unit.physical_independent_unit_id
        or acquisition_group.preparation_instance_id != physical_unit.preparation_instance_id
    ):
        raise ValueError("native acquisition changes physical preparation lineage")


def _request_fields(
    *,
    request_id: str,
    extension_set: ResponseExperimentExtensionSet,
    binding: ResponseSubstrateBinding,
    stage_id: str,
    causal_cutoff_id: str,
    physical_unit: ResponsePreparationUnit,
    acquisition_group: ResponseAcquisitionGroup,
) -> _NativeRequestFields:
    return {
        "request_id": request_id,
        "experiment_extension_set": ObjectIdentity.from_record(
            extension_set.extension_set_id,
            extension_set,
        ),
        "substrate_binding": ObjectIdentity.from_record(binding.binding_id, binding),
        "stage_id": stage_id,
        "physical_independent_unit_id": physical_unit.physical_independent_unit_id,
        "acquisition_group_id": acquisition_group.acquisition_group_id,
        "preparation_coordinate_id": physical_unit.preparation_coordinate_id,
        "preparation_instance_id": physical_unit.preparation_instance_id,
        "preparation_sha256": physical_unit.preparation_sha256,
        "adapter_realization_id": physical_unit.adapter_realization_id,
        "native_request_schema": binding.native_request_schema,
        "native_config": binding.native_config,
        "receiver_ids": tuple(value.receiver_id for value in binding.receivers),
        "clock_ids": tuple(value.clock_id for value in binding.clocks),
        "causal_cutoff_id": causal_cutoff_id,
        "grants_execution_authority": False,
        "outcome_access": OutcomeAccess.OUTCOME_BLIND,
    }


def materialise_native_acquisition_request(
    *,
    request_id: str,
    extension_set: ResponseExperimentExtensionSet,
    binding: ResponseSubstrateBinding,
    installed_executable_binding: ObjectIdentity,
    stage_id: str,
    causal_cutoff_id: str,
    physical_unit: ResponsePreparationUnit,
    acquisition_group: ResponseAcquisitionGroup,
) -> NativeAcquisitionRequest:
    """Materialise one read-only acquisition intent without contacting its source."""

    _validate_lineage(
        extension_set=extension_set,
        binding=binding,
        installed_executable_binding=installed_executable_binding,
        physical_unit=physical_unit,
        acquisition_group=acquisition_group,
    )
    if binding.interaction_kind is not NativeInteractionKind.READ_ONLY_ACQUISITION:
        raise ValueError("read-only acquisition requires a read-only native binding")
    return NativeAcquisitionRequest(
        **_request_fields(
            request_id=request_id,
            extension_set=extension_set,
            binding=binding,
            stage_id=stage_id,
            causal_cutoff_id=causal_cutoff_id,
            physical_unit=physical_unit,
            acquisition_group=acquisition_group,
        )
    )


def materialise_observation_acquisition_request(
    *,
    request_id: str,
    extension_set: ObservationOrderExperimentExtension,
    binding: ResponseSubstrateBinding,
    installed_executable_binding: ObjectIdentity,
    stage_id: str,
    causal_cutoff_id: str,
    physical_unit: ObservationPhysicalUnit,
    acquisition_group: ObservationAcquisitionGroup,
) -> NativeAcquisitionRequest:
    "Materialise the same neutral native request for an observation-order-only carrier."

    if binding.installed_executable_binding != installed_executable_binding:
        raise ValueError("native binding differs from its installed executable descriptor")
    if (
        physical_unit not in extension_set.physical_units
        or acquisition_group not in extension_set.acquisition_groups
        or acquisition_group.physical_independent_unit_id
        != physical_unit.physical_independent_unit_id
        or acquisition_group.preparation_instance_id != physical_unit.preparation_instance_id
    ):
        raise ValueError("native observation request changes preparation lineage")
    if binding.interaction_kind is not NativeInteractionKind.READ_ONLY_ACQUISITION:
        raise ValueError("observation acquisition requires a read-only native binding")
    return NativeAcquisitionRequest(
        request_id=request_id,
        experiment_extension_set=ObjectIdentity.from_record(
            extension_set.extension_set_id,
            extension_set,
        ),
        substrate_binding=ObjectIdentity.from_record(binding.binding_id, binding),
        stage_id=stage_id,
        physical_independent_unit_id=physical_unit.physical_independent_unit_id,
        acquisition_group_id=acquisition_group.acquisition_group_id,
        preparation_coordinate_id=physical_unit.preparation_coordinate_id,
        preparation_instance_id=physical_unit.preparation_instance_id,
        preparation_sha256=physical_unit.preparation_sha256,
        adapter_realization_id=physical_unit.adapter_realization_id,
        native_request_schema=binding.native_request_schema,
        native_config=binding.native_config,
        receiver_ids=tuple(value.receiver_id for value in binding.receivers),
        clock_ids=tuple(value.clock_id for value in binding.clocks),
        causal_cutoff_id=causal_cutoff_id,
        grants_execution_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def materialise_native_execution_request(
    *,
    request_id: str,
    extension_set: ResponseExperimentExtensionSet,
    binding: ResponseSubstrateBinding,
    installed_executable_binding: ObjectIdentity,
    stage_id: str,
    causal_cutoff_id: str,
    physical_unit: ResponsePreparationUnit,
    acquisition_group: ResponseAcquisitionGroup,
    nested_view: ResponseAcquisitionView,
) -> NativeExecutionRequest:
    """Materialise one interactive intent without importing or running a simulator."""

    _validate_lineage(
        extension_set=extension_set,
        binding=binding,
        installed_executable_binding=installed_executable_binding,
        physical_unit=physical_unit,
        acquisition_group=acquisition_group,
    )
    if binding.interaction_kind is not NativeInteractionKind.INTERACTIVE_EXECUTION:
        raise ValueError("interactive execution requires an interactive native binding")
    if binding.action_contract is None:
        raise ValueError("interactive native binding lacks its action contract")
    if nested_view not in extension_set.identification_config.nested_views:
        raise ValueError("native execution view lies outside the frozen measurement through local law config")
    if (
        nested_view.acquisition_group_id != acquisition_group.acquisition_group_id
        or nested_view.view_id not in acquisition_group.view_ids
        or nested_view.action_word_id is None
    ):
        raise ValueError("native execution view changes acquisition/action lineage")
    views_by_id = {value.view_id: value for value in extension_set.identification_config.nested_views}
    group_request_identities = {
        (
            views_by_id[view_id].action_word_id,
            views_by_id[view_id].numerical_member_id,
        )
        for view_id in acquisition_group.view_ids
    }
    if group_request_identities != {(nested_view.action_word_id, nested_view.numerical_member_id)}:
        raise ValueError("native execution acquisition group changes request identity")
    return NativeExecutionRequest(
        **_request_fields(
            request_id=request_id,
            extension_set=extension_set,
            binding=binding,
            stage_id=stage_id,
            causal_cutoff_id=causal_cutoff_id,
            physical_unit=physical_unit,
            acquisition_group=acquisition_group,
        ),
        nested_view_id=nested_view.view_id,
        action_word_id=nested_view.action_word_id,
        action_contract=binding.action_contract,
    )


__all__ = [
    'materialise_native_acquisition_request',
    'materialise_native_execution_request',
    'materialise_observation_acquisition_request',
]
