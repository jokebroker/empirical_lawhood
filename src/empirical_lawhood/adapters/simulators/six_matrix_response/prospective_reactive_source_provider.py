"""Production acquisition provider for the Six-matrix response prospective source tranche."""

from __future__ import annotations

from dataclasses import fields
from typing import cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_binding import materialise_native_acquisition_request
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANONICAL_MEDIA_TYPE, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID, SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal, SixMatrixResponseProspectiveReactiveSourceLawNumericalView, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, ProspectiveReactiveSourceLawRoutingService, execute_six_matrix_response_reactive_source_history


PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_TASK_PREFIX = "matrix-response-prospective-reactive-source-law-acquire"
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_KEY = "matrix-response-prospective-reactive-source-law.complete-history-acquisition"
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_BYTES = 16 * 1024 * 1024


def _read(port: object) -> bytes:
    size = cast(int, getattr(port, "size_bytes"))
    if size > _MAXIMUM_CANONICAL_BYTES:
        raise ValueError("matrix response prospective reactive source law source input exceeds its bound")
    try:
        payload = cast(bytes, getattr(port, "read")(size + 1))
        if len(payload) != size or getattr(port, "read")(1):
            raise ValueError("matrix response prospective reactive source law source input size differs")
        return payload
    finally:
        getattr(port, "close")()


class SixMatrixResponseProspectiveReactiveSourceLawHistoryTask:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        source_config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
        installed_executable_binding: ObjectIdentity,
        routing_service: ProspectiveReactiveSourceLawRoutingService | None,
    ) -> None:
        self.manifest = manifest
        self._extension = extension_set
        self._substrate = substrate_binding
        self._source = source_config
        self._binding = installed_executable_binding
        self._routing_service = routing_service

    def execute(self, context: TaskContext) -> RunnerResult:
        config_ports = tuple(
            value for value in context.input_ports if value.payload_schema == SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA
        )
        if len(config_ports) != 1:
            for value in context.input_ports:
                value.close()
            raise ValueError("matrix response prospective reactive source law source task lacks its exact config")
        for value in context.input_ports:
            if value is not config_ports[0]:
                value.close()
        decoded = decode_canonical_bytes(_read(config_ports[0]), SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, maximum_bytes=_MAXIMUM_CANONICAL_BYTES)
        if decoded != self._source:
            raise ValueError("matrix response prospective reactive source law source config was substituted")
        prefix = f"{PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_TASK_PREFIX}."
        if not context.task_id.startswith(prefix):
            raise ValueError("matrix response prospective reactive source law source task identity differs")
        group_id = context.task_id.removeprefix(prefix)
        identification_config = self._extension.identification_config
        group = next((value for value in identification_config.acquisition_groups if value.acquisition_group_id == group_id), None)
        if group is None or len(group.view_ids) != 1:
            raise ValueError("matrix response prospective reactive source law acquisition group differs")
        view = next(value for value in identification_config.nested_views if value.view_id == group.view_ids[0])
        unit = next(value for value in identification_config.physical_units if value.physical_independent_unit_id == group.physical_independent_unit_id)
        slot = next(
            (
                value
                for value in self._source.slots
                if group_id in {value.primary_acquisition_group_id, value.fine_acquisition_group_id}
            ),
            None,
        )
        if slot is None or slot.physical_independent_unit_id != group.physical_independent_unit_id:
            raise ValueError("matrix response prospective reactive source law acquisition changes its physical history")
        primary = group_id == slot.primary_acquisition_group_id
        numerical = SixMatrixResponseProspectiveReactiveSourceLawNumericalView.PRIMARY if primary else SixMatrixResponseProspectiveReactiveSourceLawNumericalView.SAME_DRIVER_HALF
        expected_view = slot.primary_view_id if primary else slot.fine_view_id
        expected_member = SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_PRIMARY_MEMBER_ID if primary else SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_FINE_MEMBER_ID
        if view.view_id != expected_view or view.numerical_member_id != expected_member:
            raise ValueError("matrix response prospective reactive source law numerical view differs")
        neutral = materialise_native_acquisition_request(
            request_id=f"neutral-request.{context.task_id}",
            extension_set=self._extension,
            binding=self._substrate,
            installed_executable_binding=self._binding,
            stage_id=f"matrix-response-prospective-reactive-source-law-{self._source.tranche.value.lower()}-source",
            causal_cutoff_id="matrix-response-prospective-reactive-source-law.routing-before-future-rng",
            physical_unit=unit,
            acquisition_group=group,
        )
        request = SixMatrixResponseProspectiveReactiveSourceLawHistoryRequest(
            request_id=f"history-request.{context.task_id}", task_id=context.task_id,
            result_output_id="history-result", trajectory_output_id="history-panel",
            source_config=ObjectIdentity.from_record(self._source.config_id, self._source),
            neutral_request=neutral, slot=slot, numerical_view=numerical,
            acquisition_group_id=group_id, scientific_view_id=view.view_id,
            evidence_ceiling=self.manifest.maximum_evidence_ceiling,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE, grants_authority=False,
        )
        result, payload = execute_six_matrix_response_reactive_source_history(
            config=self._source, request=request, routing_service=self._routing_service
        )
        envelope = LinkedCampaignStageEnvelope(
            envelope_id=f"stage-envelope.{context.task_id}",
            role=LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
            disposition=(
                LinkedCampaignDisposition.SUPPORTED
                if result.terminal is SixMatrixResponseProspectiveReactiveSourceLawHistoryTerminal.COMPLETE
                else LinkedCampaignDisposition.UNEVALUABLE
            ),
            scientific_product=ObjectIdentity.from_record(result.result_id, result),
            reason_codes=result.reason_codes,
        )
        values = {
            "history-result": result.canonical_bytes(),
            "history-panel": payload,
            "stage-envelope": envelope.canonical_bytes(),
        }
        prefix = f"{context.task_id}."
        ports = {
            value.output_id.removeprefix(prefix) if value.output_id.startswith(prefix) else value.output_id: value
            for value in context.output_ports
        }
        if set(ports) != set(values):
            raise ValueError("matrix response prospective reactive source law source output roster differs")
        return RunnerResult(
            outputs=tuple(TaskOutputPayload(ports[key].output_id, values[key]) for key in sorted(values)),
            checks=(
                ReceiptCheck("real-neutral-acquisition-request", True, ()),
                ReceiptCheck("routing-digest-precedes-future-seeds", True, ()),
                ReceiptCheck("no-source-scientific-terminal", True, ()),
            ),
        )


class SixMatrixResponseProspectiveReactiveSourceLawHistoryProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        source_config: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
        installed_executable_binding: ObjectIdentity,
        routing_service: ProspectiveReactiveSourceLawRoutingService | None,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("matrix response prospective reactive source law source manifest differs from registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.capability_count = 1
        self.extension_set = extension_set
        self.substrate_binding = substrate_binding
        self.source_config = source_config
        self.installed_binding = installed_executable_binding
        self.routing_service = routing_service

    def runners(self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("matrix response prospective reactive source law source runner inputs differ")
        return (cast(TaskRunner, SixMatrixResponseProspectiveReactiveSourceLawHistoryTask(
            manifest=self.manifest, extension_set=self.extension_set,
            substrate_binding=self.substrate_binding, source_config=self.source_config,
            installed_executable_binding=self.installed_binding,
            routing_service=self.routing_service,
        )),)

    def external_inputs(self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("matrix response prospective reactive source law source plan differs")
        observed = {value.task_id for value in plan.tasks if value.capability.capability_key == self.manifest.capability_key}
        expected = {
            f"{PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_TASK_PREFIX}.{group}"
            for slot in self.source_config.slots
            for group in (slot.primary_acquisition_group_id, slot.fine_acquisition_group_id)
            if group is not None
        }
        if observed != expected:
            raise ValueError("matrix response prospective reactive source law execution plan changes its 320 acquisitions")
        identity = ObjectIdentity.from_record(self.source_config.config_id, self.source_config)
        parent = ArtifactLineageParent(identity, VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND)
        return (ExternalInputPayload.from_bytes(
            logical_artifact_id=f"config-artifact.{self.source_config.config_id}",
            payload_schema=self.source_config.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
            media_type=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANONICAL_MEDIA_TYPE, payload=self.source_config.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE, outcome_access=OutcomeAccess.OUTCOME_BLIND,
            parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,), lineage_parents=(parent,),
            logical_content_sha256=self.source_config.fingerprint(),
        ),)

    def output_semantic_contracts(self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("matrix response prospective reactive source law source semantic registry differs")
        del execution_plan
        records = (SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, LinkedCampaignStageEnvelope)
        contracts = [
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest, payload_schema=value.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(field.name for field in fields(value))),
            )
            for value in records
        ]
        contracts.append(CapabilityOutputSemanticContract.from_manifest(
            self.manifest, payload_schema=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, profile=ArtifactProfile.AUDITED_HDF5
        ))
        return tuple(sorted(contracts, key=lambda value: value.key))

    def scientific_adjudication_contract(self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None) -> None:
        if registry != self.registry:
            raise ValueError("matrix response prospective reactive source law source adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_KEY", "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_VERSION",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_TASK_PREFIX", 'SixMatrixResponseProspectiveReactiveSourceLawHistoryProvider', 'SixMatrixResponseProspectiveReactiveSourceLawHistoryTask',
]
