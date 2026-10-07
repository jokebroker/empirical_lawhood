"""Public-route source provider for paired observation order histories."""

from __future__ import annotations

from dataclasses import fields
from typing import cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.observation_order import ObservationOrderExperimentExtension
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_binding import materialise_observation_acquisition_request
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .observation_order import MatrixObservationOrderAcquisitionDisposition, MatrixObservationOrderPairedHistoryRequest, MatrixObservationOrderPairedHistoryResult, MatrixObservationOrderPairedHistorySourceConfig, OBSERVATION_ORDER_CANONICAL_MEDIA_TYPE, OBSERVATION_ORDER_HDF5_SCHEMA, execute_observation_order_paired_history


OBSERVATION_ORDER_SOURCE_TASK_PREFIX = "matrix-observation-order.acquire"
OBSERVATION_ORDER_SOURCE_CAPABILITY_KEY = "matrix-observation-order.paired-history-acquisition"
OBSERVATION_ORDER_CAPABILITY_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_BYTES = 1024 * 1024


def _read(port: object, maximum: int) -> bytes:
    size = cast(int, getattr(port, "size_bytes"))
    if size < 1 or size > maximum:
        raise ValueError("observation order source input exceeds its byte bound")
    try:
        payload = cast(bytes, getattr(port, "read")(size + 1))
        if len(payload) != size or getattr(port, "read")(1):
            raise ValueError("observation order source input size differs")
        return payload
    finally:
        getattr(port, "close")()


class MatrixObservationOrderPairedHistoryTask:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        carrier: ObservationOrderExperimentExtension,
        substrate: ResponseSubstrateBinding,
        source: MatrixObservationOrderPairedHistorySourceConfig,
        installed_binding: ObjectIdentity,
    ) -> None:
        self.manifest = manifest
        self._carrier = carrier
        self._substrate = substrate
        self._source = source
        self._installed_binding = installed_binding

    def execute(self, context: TaskContext) -> RunnerResult:
        configs = tuple(
            port
            for port in context.input_ports
            if port.payload_schema == MatrixObservationOrderPairedHistorySourceConfig.SCHEMA
        )
        if len(configs) != 1 or len(context.input_ports) != 1:
            for port in context.input_ports:
                port.close()
            raise ValueError("observation order acquisition lacks its sole source config")
        config = decode_canonical_bytes(
            _read(configs[0], _MAXIMUM_CANONICAL_BYTES),
            MatrixObservationOrderPairedHistorySourceConfig,
            maximum_bytes=_MAXIMUM_CANONICAL_BYTES,
        )
        if config != self._source or not context.task_id.startswith(f"{OBSERVATION_ORDER_SOURCE_TASK_PREFIX}."):
            raise ValueError("observation order acquisition config or task identity differs")
        group_id = context.task_id.removeprefix(f"{OBSERVATION_ORDER_SOURCE_TASK_PREFIX}.")
        group = next(
            (
                value
                for value in self._carrier.acquisition_groups
                if value.acquisition_group_id == group_id
            ),
            None,
        )
        slot = next(
            (value for value in self._source.slots if value.acquisition_group_id == group_id),
            None,
        )
        if (
            group is None
            or slot is None
            or group.physical_independent_unit_id != slot.physical_independent_unit_id
            or group.preparation_instance_id != slot.preparation_instance_id
            or set(group.view_ids) != {slot.primary_view_id, slot.fine_view_id}
        ):
            raise ValueError("observation order acquisition group differs from the paired source slot")
        unit = next(
            value
            for value in self._carrier.physical_units
            if value.physical_independent_unit_id == group.physical_independent_unit_id
        )
        neutral = materialise_observation_acquisition_request(
            request_id=f"neutral-request.{context.task_id}",
            extension_set=self._carrier,
            binding=self._substrate,
            installed_executable_binding=self._installed_binding,
            stage_id="matrix-observation-order.paired-native-history",
            causal_cutoff_id=self._carrier.causal_cutoff_ids[0],
            physical_unit=unit,
            acquisition_group=group,
        )
        request = MatrixObservationOrderPairedHistoryRequest(
            request_id=f"paired-history-request.{slot.history_id}",
            task_id=context.task_id,
            source_config=ObjectIdentity.from_record(self._source.config_id, self._source),
            neutral_request=neutral,
            slot=slot,
            evidence_ceiling=self.manifest.maximum_evidence_ceiling,
            grants_authority=False,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        result, hdf5 = execute_observation_order_paired_history(self._source, request)
        complete = result.disposition is MatrixObservationOrderAcquisitionDisposition.COMPLETE
        envelope = LinkedCampaignStageEnvelope(
            envelope_id=f"stage-envelope.{context.task_id}",
            role=LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
            disposition=(
                LinkedCampaignDisposition.SUPPORTED
                if complete
                else LinkedCampaignDisposition.UNEVALUABLE
            ),
            scientific_product=ObjectIdentity.from_record(result.result_id, result),
            reason_codes=result.reason_codes,
        )
        values = {
            MatrixObservationOrderPairedHistoryResult.SCHEMA: result.canonical_bytes(),
            OBSERVATION_ORDER_HDF5_SCHEMA: hdf5,
            LinkedCampaignStageEnvelope.SCHEMA: envelope.canonical_bytes(),
        }
        if {value.payload_schema for value in context.output_ports} != set(values):
            raise ValueError("observation order acquisition output roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(port.output_id, values[port.payload_schema])
                for port in sorted(context.output_ports, key=lambda value: value.output_id)
            ),
            checks=(
                ReceiptCheck("one-acquisition-group-per-history", True, ()),
                ReceiptCheck("paired-views-share-native-driver", True, ()),
                ReceiptCheck("source-constructs-no-scientific-verdict", True, ()),
            ),
        )


class MatrixObservationOrderPairedHistoryProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        carrier: ObservationOrderExperimentExtension,
        substrate: ResponseSubstrateBinding,
        source: MatrixObservationOrderPairedHistorySourceConfig,
        installed_binding: ObjectIdentity,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("observation order source manifest differs from the selected registry")
        if (
            substrate.installed_executable_binding != installed_binding
            or substrate.native_config != ObjectIdentity.from_record(source.config_id, source)
        ):
            raise ValueError("observation order source native binding differs from its issued config")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.carrier = carrier
        self.substrate = substrate
        self.source = source
        self.installed_binding = installed_binding

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("observation order source runner inputs differ")
        return (
            cast(
                TaskRunner,
                MatrixObservationOrderPairedHistoryTask(
                    manifest=self.manifest,
                    carrier=self.carrier,
                    substrate=self.substrate,
                    source=self.source,
                    installed_binding=self.installed_binding,
                ),
            ),
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("observation order source execution plan differs")
        observed = {
            task.task_id
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
        }
        expected = {
            f"{OBSERVATION_ORDER_SOURCE_TASK_PREFIX}.{value.acquisition_group_id}"
            for value in self.carrier.acquisition_groups
        }
        if observed != expected:
            raise ValueError("observation order source task roster differs from issued acquisition groups")
        specs = {
            item.logical_artifact_id: item
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            for item in task.external_inputs
            if item.expected_payload_schema == self.source.SCHEMA
        }
        if set(specs) != {
            self.source.config_id,
            f"config-artifact.{self.source.config_id}",
        } or any(
            value.expected_content_sha256 != self.source.fingerprint() for value in specs.values()
        ):
            raise ValueError("observation order source plan lacks its exact external config")
        identity = ObjectIdentity.from_record(self.source.config_id, self.source)
        parent = ArtifactLineageParent(
            identity,
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=logical_artifact_id,
                payload_schema=self.source.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=OBSERVATION_ORDER_CANONICAL_MEDIA_TYPE,
                payload=self.source.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                lineage_parents=(parent,),
                logical_content_sha256=self.source.fingerprint(),
            )
            for logical_artifact_id in sorted(specs)
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("observation order source semantic registry differs")
        del execution_plan
        records = (MatrixObservationOrderPairedHistoryResult, LinkedCampaignStageEnvelope)
        contracts = [
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=value.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(field.name for field in fields(value))),
            )
            for value in records
        ]
        contracts.append(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=OBSERVATION_ORDER_HDF5_SCHEMA,
                profile=ArtifactProfile.AUDITED_HDF5,
            )
        )
        return tuple(sorted(contracts, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("observation order source adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "OBSERVATION_ORDER_CAPABILITY_VERSION",
    "OBSERVATION_ORDER_SOURCE_CAPABILITY_KEY",
    "OBSERVATION_ORDER_SOURCE_TASK_PREFIX",
    'MatrixObservationOrderPairedHistoryProvider',
    'MatrixObservationOrderPairedHistoryTask',
]
