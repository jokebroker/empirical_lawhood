"""Consolidated campaign provider for the matrix response reactive entrance precursor source.

This module is deliberately narrow: it materialises the platform's neutral
read-only acquisition request, invokes the already-defined reactive entrance precursor
physics, and publishes the native result plus its dense HDF5 trajectory.  It
does not label entrances or adjudicate reactive entrance.
"""

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

from .reactive_entrance import REACTIVE_ENTRANCE_CANONICAL_MEDIA_TYPE, REACTIVE_ENTRANCE_FINE_MEMBER_ID, REACTIVE_ENTRANCE_FINE_STEPS, REACTIVE_ENTRANCE_HDF5_SCHEMA, REACTIVE_ENTRANCE_PRIMARY_MEMBER_ID, REACTIVE_ENTRANCE_PRIMARY_STEPS, REACTIVE_ENTRANCE_RECEIVER_CADENCE, SixMatrixResponseReactiveEntranceNumericalViewKind, SixMatrixResponseReactiveEntrancePrecursorRequest, SixMatrixResponseReactiveEntrancePrecursorResult, SixMatrixResponseReactiveEntrancePrecursorTerminal, SixMatrixResponseReactiveEntranceSourceConfig, execute_six_matrix_response_reactive_entrance_precursor


REACTIVE_ENTRANCE_SOURCE_TASK_PREFIX = "matrix-response-reactive-entrance-acquire"
REACTIVE_ENTRANCE_SOURCE_CAPABILITY_KEY = "matrix-response-reactive-entrance.reactive-precursor"
REACTIVE_ENTRANCE_SOURCE_CAPABILITY_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_INPUT_BYTES = 16 * 1024 * 1024


def _read(port: object) -> bytes:
    size = cast(int, getattr(port, "size_bytes"))
    if size > _MAXIMUM_CANONICAL_INPUT_BYTES:
        raise ValueError("matrix response reactive entrance canonical input exceeds its bound")
    try:
        payload = cast(bytes, getattr(port, "read")(size + 1))
        if len(payload) != size or getattr(port, "read")(1):
            raise ValueError("matrix response reactive entrance canonical input size differs")
        return payload
    finally:
        getattr(port, "close")()


def _source_envelope(
    task_id: str,
    result: SixMatrixResponseReactiveEntrancePrecursorResult,
) -> LinkedCampaignStageEnvelope:
    complete = result.terminal is SixMatrixResponseReactiveEntrancePrecursorTerminal.COMPLETED
    reasons = () if complete else (result.reason_codes or ("NUMERICAL_SOURCE_INVALID",))
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
        disposition=(
            LinkedCampaignDisposition.SUPPORTED
            if complete
            else LinkedCampaignDisposition.UNEVALUABLE
        ),
        scientific_product=ObjectIdentity.from_record(result.result_id, result),
        reason_codes=tuple(sorted(reasons)),
    )


class SixMatrixResponseReactiveEntrancePrecursorTask:
    """Execute exactly one issued primary or same-driver fine precursor."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        source_config: SixMatrixResponseReactiveEntranceSourceConfig,
        installed_executable_binding: ObjectIdentity,
    ) -> None:
        self.manifest = manifest
        self._extension = extension_set
        self._substrate = substrate_binding
        self._source = source_config
        self._installed_binding = installed_executable_binding

    def execute(self, context: TaskContext) -> RunnerResult:
        config_ports = tuple(
            value
            for value in context.input_ports
            if value.payload_schema == SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA
        )
        if len(config_ports) != 1:
            for value in context.input_ports:
                value.close()
            raise ValueError("matrix response reactive entrance source task requires one exact source config")
        for value in context.input_ports:
            if value is not config_ports[0]:
                value.close()
        decoded = decode_canonical_bytes(
            _read(config_ports[0]),
            SixMatrixResponseReactiveEntranceSourceConfig,
            maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES,
        )
        if decoded != self._source:
            raise ValueError("matrix response reactive entrance source task received a substituted config")
        prefix = f"{REACTIVE_ENTRANCE_SOURCE_TASK_PREFIX}."
        if not context.task_id.startswith(prefix):
            raise ValueError("matrix response reactive entrance source task identity differs")
        group_id = context.task_id[len(prefix) :]
        identification_config = self._extension.identification_config
        group = next(
            (value for value in identification_config.acquisition_groups if value.acquisition_group_id == group_id),
            None,
        )
        if group is None or len(group.view_ids) != 1:
            raise ValueError("matrix response reactive entrance acquisition group differs from the frozen roster")
        view = next(value for value in identification_config.nested_views if value.view_id == group.view_ids[0])
        unit = next(
            value
            for value in identification_config.physical_units
            if value.physical_independent_unit_id == group.physical_independent_unit_id
        )
        slot = next(
            (
                value
                for value in self._source.slots
                if group_id
                in {
                    value.primary_acquisition_group_id,
                    value.fine_acquisition_group_id,
                }
            ),
            None,
        )
        if slot is None or view.physical_independent_unit_id != slot.physical_independent_unit_id:
            raise ValueError("matrix response reactive entrance acquisition changes its conditional physical unit")
        primary = group_id == slot.primary_acquisition_group_id
        expected_view = slot.primary_view_id if primary else slot.fine_view_id
        expected_member = REACTIVE_ENTRANCE_PRIMARY_MEMBER_ID if primary else REACTIVE_ENTRANCE_FINE_MEMBER_ID
        if view.view_id != expected_view or view.numerical_member_id != expected_member:
            raise ValueError("matrix response reactive entrance acquisition changes its numerical view")
        neutral = materialise_native_acquisition_request(
            request_id=f"neutral-request.{context.task_id}",
            extension_set=self._extension,
            binding=self._substrate,
            installed_executable_binding=self._installed_binding,
            stage_id="reactive-entrance-source-materialization",
            causal_cutoff_id="matrix-response-reactive-entrance.pre-outcome-source-freeze",
            physical_unit=unit,
            acquisition_group=group,
        )
        request = SixMatrixResponseReactiveEntrancePrecursorRequest(
            request_id=f"precursor-request.{context.task_id}",
            task_id=context.task_id,
            result_output_id="precursor-result",
            trajectory_output_id="precursor-trajectory",
            source_config=ObjectIdentity.from_record(self._source.config_id, self._source),
            neutral_request=neutral,
            slot=slot,
            numerical_view_kind=(
                SixMatrixResponseReactiveEntranceNumericalViewKind.PRIMARY
                if primary
                else SixMatrixResponseReactiveEntranceNumericalViewKind.SAME_DRIVER_HALF
            ),
            numerical_member_id=expected_member,
            acquisition_group_id=group_id,
            scientific_view_id=view.view_id,
            integration_steps=REACTIVE_ENTRANCE_PRIMARY_STEPS if primary else REACTIVE_ENTRANCE_FINE_STEPS,
            receiver_cadence_parent_steps=REACTIVE_ENTRANCE_RECEIVER_CADENCE,
            evidence_ceiling=self.manifest.maximum_evidence_ceiling,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            grants_authority=False,
        )
        executed = execute_six_matrix_response_reactive_entrance_precursor(config=self._source, request=request)
        envelope = _source_envelope(context.task_id, executed.result)
        values = {
            "precursor-result": executed.result.canonical_bytes(),
            "precursor-trajectory": executed.hdf5_payload,
            "stage-envelope": envelope.canonical_bytes(),
        }
        expected_ports = {
            "precursor-result": (SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA, ArtifactProfile.CANONICAL_JSON),
            "precursor-trajectory": (REACTIVE_ENTRANCE_HDF5_SCHEMA, ArtifactProfile.AUDITED_HDF5),
            "stage-envelope": (LinkedCampaignStageEnvelope.SCHEMA, ArtifactProfile.CANONICAL_JSON),
        }
        prefix = f"{context.task_id}."
        ports = {
            (
                value.output_id.removeprefix(prefix)
                if value.output_id.startswith(prefix)
                else value.output_id
            ): value
            for value in context.output_ports
        }
        if len(ports) != len(context.output_ports) or set(ports) != set(values):
            raise ValueError("matrix response reactive entrance source output roster differs")
        for output_id, port in ports.items():
            if (port.payload_schema, port.profile) != expected_ports[output_id]:
                raise ValueError("matrix response reactive entrance source output contract differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(ports[output_id].output_id, values[output_id])
                for output_id in sorted(ports)
            ),
            checks=(
                ReceiptCheck("complete-phase-space-custody", True, ()),
                ReceiptCheck("neutral-request-materialised", True, ()),
                ReceiptCheck("no-source-scientific-verdict", True, ()),
            ),
        )


class SixMatrixResponseReactiveEntrancePrecursorProvider(CampaignRuntimeProvider):
    """Installed provider for the exact reactive entrance acquisition task family."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        source_config: SixMatrixResponseReactiveEntranceSourceConfig,
        installed_executable_binding: ObjectIdentity,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("matrix response reactive entrance source manifest differs from installed registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.capability_count = 1
        self.extension_set = extension_set
        self.substrate_binding = substrate_binding
        self.source_config = source_config
        self.installed_executable_binding = installed_executable_binding

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("matrix response reactive entrance source runner inputs differ")
        return (
            cast(
                TaskRunner,
                SixMatrixResponseReactiveEntrancePrecursorTask(
                    manifest=self.manifest,
                    extension_set=self.extension_set,
                    substrate_binding=self.substrate_binding,
                    source_config=self.source_config,
                    installed_executable_binding=self.installed_executable_binding,
                ),
            ),
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("matrix response reactive entrance source plan/input registry differs")
        task_ids = {
            value.task_id
            for value in plan.tasks
            if value.capability.capability_key == self.manifest.capability_key
        }
        expected = {
            f"{REACTIVE_ENTRANCE_SOURCE_TASK_PREFIX}.{value.primary_acquisition_group_id}"
            for value in self.source_config.slots
        } | {
            f"{REACTIVE_ENTRANCE_SOURCE_TASK_PREFIX}.{value.fine_acquisition_group_id}"
            for value in self.source_config.slots
            if value.fine_acquisition_group_id is not None
        }
        if task_ids != expected:
            raise ValueError("matrix response reactive entrance execution plan changes the 512+128 source roster")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.source_config.config_id, self.source_config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return (
            ExternalInputPayload.from_bytes(
                logical_artifact_id=f"config-artifact.{self.source_config.config_id}",
                payload_schema=self.source_config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=REACTIVE_ENTRANCE_CANONICAL_MEDIA_TYPE,
                payload=self.source_config.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                lineage_parents=(parent,),
                logical_content_sha256=self.source_config.fingerprint(),
            ),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("matrix response reactive entrance source semantic registry differs")
        del execution_plan
        records = {
            SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA: SixMatrixResponseReactiveEntrancePrecursorResult,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
        }
        contracts = [
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(record_type))),
            )
            for schema, record_type in records.items()
        ]
        contracts.append(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=REACTIVE_ENTRANCE_HDF5_SCHEMA,
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
            raise ValueError("matrix response reactive entrance source adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "REACTIVE_ENTRANCE_SOURCE_CAPABILITY_KEY",
    "REACTIVE_ENTRANCE_SOURCE_CAPABILITY_VERSION",
    "REACTIVE_ENTRANCE_SOURCE_TASK_PREFIX",
    'SixMatrixResponseReactiveEntrancePrecursorProvider',
    'SixMatrixResponseReactiveEntrancePrecursorTask',
]
