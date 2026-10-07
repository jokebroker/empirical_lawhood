"Noncontact compilation for the action-free observation-order carrier."

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.observation_order import MAX_OBSERVATION_ORDER_CONFIG_BYTES, ObservationOrderOwnerRole, ObservationOrderExperimentExtension
from empirical_lawhood.planning.prospective_config import ProspectiveAuthorityOperation
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.response_experiment_ports import NativeInteractionKind, ResponseSubstrateBinding
from empirical_lawhood.runtime.static_codecs import (
    CanonicalRecordCodecRegistry,
    build_canonical_record_codec_registry,
)


OBSERVATION_ORDER_OBJECTIVE_PROFILE_ID = "objective.observation-order"


@dataclass(frozen=True, slots=True)
class ObservationOrderSubstrateBinding(CanonicalRecord):
    """Bind one observation carrier to the existing neutral native contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/observation-order-substrate-binding'

    binding_id: str
    observation_carrier: ObjectIdentity
    substrate: ResponseSubstrateBinding
    source_config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.observation_carrier.object_schema != ObservationOrderExperimentExtension.SCHEMA:
            raise ValueError("observation substrate binding requires the observation-order carrier")
        if (
            self.substrate.interaction_kind is not NativeInteractionKind.READ_ONLY_ACQUISITION
            or self.substrate.action_contract is not None
            or self.substrate.native_config != self.source_config
        ):
            raise ValueError("observation substrate binding changes neutral native semantics")


class ObservationOrderStageRole(StrEnum):
    ACQUISITION = "ACQUISITION"
    PROJECTION = "PROJECTION"
    SEALED_EVALUATION = "SEALED_EVALUATION"


@dataclass(frozen=True, slots=True)
class ObservationOrderStageSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/observation-order-stage-spec'

    task_id: str
    role: ObservationOrderStageRole
    subject_id: str
    acquisition_group_id: str | None
    scientific_view_id: str | None
    dependency_task_ids: tuple[str, ...]
    owner: ObjectIdentity
    config: ObjectIdentity
    authority_operation: ProspectiveAuthorityOperation | None
    maximum_attempts: int

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        validate_stable_id(self.subject_id, field_name="subject_id")
        for name in ("acquisition_group_id", "scientific_view_id"):
            value = getattr(self, name)
            if value is not None:
                validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.dependency_task_ids,
            field_name="dependency_task_ids",
        )
        expected = {
            ObservationOrderStageRole.ACQUISITION: (
                True,
                False,
                ProspectiveAuthorityOperation.EXECUTE,
            ),
            ObservationOrderStageRole.PROJECTION: (True, True, None),
            ObservationOrderStageRole.SEALED_EVALUATION: (
                False,
                False,
                ProspectiveAuthorityOperation.REVEAL,
            ),
        }[self.role]
        if (
            (self.acquisition_group_id is not None) != expected[0]
            or (self.scientific_view_id is not None) != expected[1]
            or self.authority_operation is not expected[2]
        ):
            raise ValueError("observation stage role changes its lineage or authority operation")
        if self.maximum_attempts != 1:
            raise ValueError("observation science permits one task identity attempt")


@dataclass(frozen=True, slots=True)
class CompiledObservationOrderExperiment(CanonicalRecord):
    """Exact 1:views:1 topology derived without source contact or authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-observation-order-experiment'

    plan_id: str
    extension_set: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    stages: tuple[ObservationOrderStageSpec, ...]
    topology_sha256: str
    maximum_evidence_ceiling: EvidenceCeiling
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        require_sorted_unique_ids(self.stages, attribute="task_id", field_name="stages")
        validate_sha256(self.topology_sha256, field_name="topology_sha256")
        if self.topology_sha256 != _topology_sha256(self.stages):
            raise ValueError("compiled observation topology digest differs")
        if {value.role for value in self.stages} != set(ObservationOrderStageRole):
            raise ValueError("compiled observation topology omits a role")
        if (
            sum(
                value.role is ObservationOrderStageRole.SEALED_EVALUATION for value in self.stages
            )
            != 1
        ):
            raise ValueError("compiled observation topology requires one sealed evaluator")
        task_ids = {value.task_id for value in self.stages}
        if any(not set(value.dependency_task_ids) <= task_ids for value in self.stages):
            raise ValueError("compiled observation topology has a ghost dependency")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.ORDER_RELATION:
            raise ValueError("compiled observation topology exceeds ORDER_RELATION")
        if (
            self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("compiled observation topology exceeds prospective authority")


@dataclass(frozen=True, slots=True)
class ObservationOrderCompilationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/observation-order-compilation-receipt'

    receipt_id: str
    extension_set: ObjectIdentity
    compiled_plan: ObjectIdentity
    strict_root_identities: tuple[ObjectIdentity, ...]
    config_identities: tuple[ObjectIdentity, ...]
    physical_independent_unit_count: int
    acquisition_group_count: int
    nested_view_count: int
    sealed_evaluation_count: int
    topology_sha256: str
    issued: bool
    execution_ready: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(self.topology_sha256, field_name="topology_sha256")
        if len(self.strict_root_identities) != 2 or len(set(self.strict_root_identities)) != 2:
            raise ValueError("observation compilation requires two strict roots")
        if not self.config_identities:
            raise ValueError("observation compilation requires owned configs")
        if (
            min(
                self.physical_independent_unit_count,
                self.acquisition_group_count,
                self.nested_view_count,
            )
            < 1
            or self.sealed_evaluation_count != 1
        ):
            raise ValueError("observation compilation topology counts differ")
        if self.physical_independent_unit_count != self.acquisition_group_count:
            raise ValueError("observation compilation changes one-group-per-unit")
        if self.issued or self.execution_ready or self.grants_authority:
            raise ValueError("observation compilation cannot issue, authorize or claim readiness")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("observation compilation receipt must remain prospective")


def _topology_sha256(stages: tuple[ObservationOrderStageSpec, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(stages)).hexdigest()


def derive_observation_order_topology(
    extension: ObservationOrderExperimentExtension,
) -> CompiledObservationOrderExperiment:
    owners = {value.role: value for value in extension.owners}
    acquisition = tuple(
        ObservationOrderStageSpec(
            task_id=f"{extension.acquisition_task_prefix}.{group.acquisition_group_id}",
            role=ObservationOrderStageRole.ACQUISITION,
            subject_id=group.physical_independent_unit_id,
            acquisition_group_id=group.acquisition_group_id,
            scientific_view_id=None,
            dependency_task_ids=(),
            owner=owners[ObservationOrderOwnerRole.SOURCE].owner,
            config=owners[ObservationOrderOwnerRole.SOURCE].config,
            authority_operation=ProspectiveAuthorityOperation.EXECUTE,
            maximum_attempts=1,
        )
        for group in extension.acquisition_groups
    )
    by_group = {value.acquisition_group_id: value.task_id for value in acquisition}
    projections = tuple(
        ObservationOrderStageSpec(
            task_id=f"{extension.projection_task_prefix}.{view.view_id}",
            role=ObservationOrderStageRole.PROJECTION,
            subject_id=view.physical_independent_unit_id,
            acquisition_group_id=view.acquisition_group_id,
            scientific_view_id=view.view_id,
            dependency_task_ids=(by_group[view.acquisition_group_id],),
            owner=owners[ObservationOrderOwnerRole.EVIDENCE_PROJECTION].owner,
            config=owners[ObservationOrderOwnerRole.EVIDENCE_PROJECTION].config,
            authority_operation=None,
            maximum_attempts=1,
        )
        for view in extension.nested_views
    )
    evaluator = ObservationOrderStageSpec(
        task_id=extension.sealed_evaluation_task_id,
        role=ObservationOrderStageRole.SEALED_EVALUATION,
        subject_id=extension.experiment_id,
        acquisition_group_id=None,
        scientific_view_id=None,
        dependency_task_ids=tuple(value.task_id for value in projections),
        owner=owners[ObservationOrderOwnerRole.SEALED_EVALUATOR].owner,
        config=owners[ObservationOrderOwnerRole.SEALED_EVALUATOR].config,
        authority_operation=ProspectiveAuthorityOperation.REVEAL,
        maximum_attempts=1,
    )
    stages = tuple(sorted((*acquisition, *projections, evaluator), key=lambda value: value.task_id))
    return CompiledObservationOrderExperiment(
        plan_id=f"observation-order-plan.{extension.extension_set_id}",
        extension_set=ObjectIdentity.from_record(extension.extension_set_id, extension),
        evidence_profile_selection=extension.evidence_profile_selection,
        source_pipeline_profile=extension.source_pipeline_profile,
        stages=stages,
        topology_sha256=_topology_sha256(stages),
        maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def compile_observation_order_experiment(
    *,
    extension: ObservationOrderExperimentExtension,
    evidence_profile_selection: EvidenceProfileSelection,
    source_pipeline_profile: SourcePipelineProfile,
) -> tuple[CompiledObservationOrderExperiment, ObservationOrderCompilationReceipt]:
    """Compile exact roots into acquisition/projection/evaluation topology only."""

    evidence = ObjectIdentity.from_record(
        evidence_profile_selection.selection_id,
        evidence_profile_selection,
    )
    source = ObjectIdentity.from_record(source_pipeline_profile.profile_id, source_pipeline_profile)
    if (
        extension.evidence_profile_selection != evidence
        or extension.source_pipeline_profile != source
    ):
        raise ValueError("observation carrier differs from its decoded strict roots")
    if (
        evidence_profile_selection.objective_profile_id != OBSERVATION_ORDER_OBJECTIVE_PROFILE_ID
        or evidence_profile_selection.requested_rungs != (EvidenceRung.ORDER_RELATION,)
    ):
        raise ValueError("observation carrier requires the strict observation-order objective selection")
    compiled = derive_observation_order_topology(extension)
    receipt = ObservationOrderCompilationReceipt(
        receipt_id=f"observation-order-compilation.{extension.extension_set_id}",
        extension_set=compiled.extension_set,
        compiled_plan=ObjectIdentity.from_record(compiled.plan_id, compiled),
        strict_root_identities=(evidence, source),
        config_identities=extension.config_identities,
        physical_independent_unit_count=len(extension.physical_units),
        acquisition_group_count=len(extension.acquisition_groups),
        nested_view_count=len(extension.nested_views),
        sealed_evaluation_count=1,
        topology_sha256=compiled.topology_sha256,
        issued=False,
        execution_ready=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return compiled, receipt


def build_observation_order_decoder_registrations(
    *, implementation_sha256: str
) -> tuple[StudyExtensionDecoderRegistration, ...]:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    record_type = ObservationOrderExperimentExtension
    decoder_key = 'observation-order.observation-experiment-extension'
    config_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "decoder_key": decoder_key,
                "decoder_version": "1.0.0",
                "mode": "exact-canonical-record",
                "payload_schema": record_type.SCHEMA,
            }
        )
    ).hexdigest()
    return (
        StudyExtensionDecoderRegistration(
            registration_id='decoder-registration.observation-experiment-extension',
            decoder_key=decoder_key,
            decoder_version="1.0.0",
            payload_schema=record_type.SCHEMA,
            payload_version=record_type.VERSION,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            maximum_payload_bytes=MAX_OBSERVATION_ORDER_CONFIG_BYTES,
        ),
    )


def build_observation_order_codec_registry(
    *, implementation_sha256: str
) -> CanonicalRecordCodecRegistry:
    return build_canonical_record_codec_registry(
        registry_id="canonical-codecs.observation-order",
        record_types=(ObservationOrderExperimentExtension,),
        maximum_bytes_by_schema={
            ObservationOrderExperimentExtension.SCHEMA: MAX_OBSERVATION_ORDER_CONFIG_BYTES
        },
        decoder_implementation_sha256=implementation_sha256,
    )


__all__ = [
    'CompiledObservationOrderExperiment',
    "OBSERVATION_ORDER_OBJECTIVE_PROFILE_ID",
    'ObservationOrderCompilationReceipt',
    'ObservationOrderStageRole',
    'ObservationOrderStageSpec',
    'ObservationOrderSubstrateBinding',
    'build_observation_order_codec_registry',
    'build_observation_order_decoder_registrations',
    'compile_observation_order_experiment',
    'derive_observation_order_topology',
]
