"""Bounded evidence-manifest projection and strict R6 compatibility lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import (
    ClaimUnitBinding,
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
    IdentificationManifestObservation,
    ObservationDisposition,
    ProjectedIdentificationObservation,
)

from .contracts import (
    DataSplit,
    IdentificationDataset,
    LawObservation,
    ObservationRole,
    OneFactorExchange,
)


class CompactObservationSource(Protocol):
    """Injected bounded reader over already-authorized external payloads."""

    def read_compact_values(
        self,
        manifest: IdentificationEvidenceManifest,
        observation: IdentificationManifestObservation,
        required_value_ids: tuple[str, ...],
    ) -> tuple[NamedDecimal, ...]: ...


@dataclass(frozen=True, slots=True)
class ObservationValuePartition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observation-value-partition'

    partition_id: str
    denominator_value_ids: tuple[str, ...]
    history_value_ids: tuple[str, ...]
    action_value_ids: tuple[str, ...]
    receiver_value_ids: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.partition_id, field_name="partition_id")
        partitions = (
            self.denominator_value_ids,
            self.history_value_ids,
            self.action_value_ids,
            self.receiver_value_ids,
        )
        for name, values, allow_empty in (
            ("denominator_value_ids", self.denominator_value_ids, False),
            ("history_value_ids", self.history_value_ids, True),
            ("action_value_ids", self.action_value_ids, False),
            ("receiver_value_ids", self.receiver_value_ids, False),
            ("tags", self.tags, True),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=allow_empty)
        if any(
            set(left) & set(right)
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        ):
            raise ValueError("observation value partitions overlap")

    @property
    def required_value_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.denominator_value_ids,
                    *self.history_value_ids,
                    *self.action_value_ids,
                    *self.receiver_value_ids,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class IdentificationProjectionPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/identification-projection-plan'

    plan_id: str
    manifest: ObjectIdentity
    allowed_consumer_id: str
    projection_capability: ObjectIdentity
    projection_config: ObjectIdentity
    projection_implementation: ObjectIdentity
    claim_unit_binding: ClaimUnitBinding
    value_partition: ObservationValuePartition

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.allowed_consumer_id, field_name="allowed_consumer_id")
        if self.manifest.object_schema != IdentificationEvidenceManifest.SCHEMA:
            raise ValueError("projection plan requires an identification manifest")


@dataclass(frozen=True, slots=True)
class ObservationTagBinding(CanonicalRecord):
    """Exact observation-local metadata retained by a lossless projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/observation-tag-binding'

    observation_id: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        require_sorted_unique_strings(self.tags, field_name="tags")


@dataclass(frozen=True, slots=True)
class TaggedIdentificationProjectionPlan(CanonicalRecord):
    "Evidence projection with an exact observation-local tag roster."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/tagged-identification-projection-plan'

    plan_id: str
    manifest: ObjectIdentity
    allowed_consumer_id: str
    projection_capability: ObjectIdentity
    projection_config: ObjectIdentity
    projection_implementation: ObjectIdentity
    claim_unit_binding: ClaimUnitBinding
    value_partition: ObservationValuePartition
    observation_tags: tuple[ObservationTagBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.allowed_consumer_id, field_name="allowed_consumer_id")
        if self.manifest.object_schema != IdentificationEvidenceManifest.SCHEMA:
            raise ValueError("projection plan requires an identification manifest")
        require_sorted_unique_ids(
            self.observation_tags,
            attribute="observation_id",
            field_name="observation_tags",
        )
        if not self.observation_tags:
            raise ValueError("Observation-local projection plan requires observation-local tag accounting")


@dataclass(frozen=True, slots=True)
class IdentificationEvidenceProjector:
    """Pure topology validation around one injected bounded payload reader."""

    def project(
        self,
        system: SystemSpec,
        manifest: IdentificationEvidenceManifest,
        plan: IdentificationProjectionPlan | TaggedIdentificationProjectionPlan,
        source: CompactObservationSource,
    ) -> IdentificationEvidenceProjection:
        if manifest.system != ObjectIdentity.from_record(system.system_id, system):
            raise ValueError("evidence manifest binds another prepared system")
        if manifest.relation != system.relation:
            raise ValueError("evidence manifest changes the relational identity")
        if manifest.evidence_world_id != system.world.world_id:
            raise ValueError("evidence manifest changes the evidence world")
        if manifest.outcome_access not in system.world.available_outcome_access:
            raise ValueError("evidence manifest uses unavailable outcome access")
        if plan.manifest != ObjectIdentity.from_record(manifest.manifest_id, manifest):
            raise ValueError("projection plan binds another manifest")
        if plan.allowed_consumer_id not in manifest.allowed_consumer_ids:
            raise ValueError("projection consumer is not allowed by the evidence domain")
        scopes = {value.scope_id: value for value in manifest.qualification_scopes}
        scope_id = plan.claim_unit_binding.scope.object_id
        if scope_id not in scopes:
            raise ValueError("projection claim scope is absent from the manifest")
        scope = scopes[scope_id]
        if plan.claim_unit_binding.scope != ObjectIdentity.from_record(scope.scope_id, scope):
            raise ValueError("projection changes the exact qualification scope identity")
        if plan.claim_unit_binding.claim_id != scope.claim_id:
            raise ValueError("projection claim/unit binding differs from its scope")
        if plan.claim_unit_binding.independent_unit_instance_ids != (
            scope.independent_unit_instance_ids
        ):
            raise ValueError("projection changes the independent-unit roster")
        if plan.claim_unit_binding.aggregation_level_id != scope.aggregation_level_id:
            raise ValueError("projection changes the claim aggregation level")
        payloads = {value.payload_id: value for value in manifest.payloads}
        tags_by_observation = (
            {value.observation_id: value.tags for value in plan.observation_tags}
            if isinstance(plan, TaggedIdentificationProjectionPlan)
            else None
        )
        if tags_by_observation is not None and set(tags_by_observation) != {
            value.observation_id for value in manifest.observations
        }:
            raise ValueError("Observation-local projection tag roster differs from manifest observations")
        rows = []
        for observation in manifest.observations:
            values = source.read_compact_values(
                manifest,
                observation,
                plan.value_partition.required_value_ids,
            )
            if observation.disposition is not ObservationDisposition.COMPLETE and values:
                raise ValueError("invalid/missing manifest observation cannot acquire values")
            rows.append(
                ProjectedIdentificationObservation(
                    projected_observation_id=f"projected.{observation.observation_id}",
                    manifest_observation_id=observation.observation_id,
                    physical_unit_instance_id=observation.physical_unit_instance_id,
                    nested_coordinate_ids=observation.nested_coordinate_ids,
                    denominator_cell_id=observation.denominator_cell_id,
                    chart_id=observation.chart_id,
                    split_id=observation.split_id,
                    role_id=observation.role_id,
                    member_id=observation.member_id,
                    candidate_version_id=observation.candidate_version_id,
                    qualification_view_id=observation.qualification_view_id,
                    receiver_id=observation.receiver_id,
                    receiver_clock_id=observation.receiver_clock_id,
                    native_frame_id=observation.native_frame_id,
                    payload_id=observation.payload_id,
                    payload_member_locator=observation.payload_member_locator,
                    action_delivery=observation.action_delivery,
                    disposition=observation.disposition,
                    compact_values=values,
                    denominator_value_ids=plan.value_partition.denominator_value_ids,
                    history_value_ids=plan.value_partition.history_value_ids,
                    action_value_ids=plan.value_partition.action_value_ids,
                    receiver_value_ids=plan.value_partition.receiver_value_ids,
                    tags=(
                        plan.value_partition.tags
                        if tags_by_observation is None
                        else tags_by_observation[observation.observation_id]
                    ),
                    reason_codes=observation.reason_codes,
                )
            )
        return IdentificationEvidenceProjection(
            projection_id=f"projection.{manifest.manifest_id}.{plan.allowed_consumer_id}",
            manifest=plan.manifest,
            allowed_consumer_id=plan.allowed_consumer_id,
            projection_capability=plan.projection_capability,
            projection_config=plan.projection_config,
            projection_implementation=plan.projection_implementation,
            claim_unit_binding=plan.claim_unit_binding,
            information_cutoff_id=manifest.information_cutoff_id,
            observations=tuple(sorted(rows, key=lambda value: value.projected_observation_id)),
            source_payload_ids=tuple(sorted(payloads)),
            outcome_access=manifest.outcome_access,
            parent_visibility_ceilings=manifest.parent_visibility_ceilings,
            visibility_ceiling=manifest.visibility_ceiling,
        )


_RESPONSE_METHOD_SPLITS = {
    "calibration": DataSplit.CALIBRATION,
    "held-out": DataSplit.HELD_OUT,
}
_RESPONSE_METHOD_ROLES = {
    "negative-control": ObservationRole.NEGATIVE_CONTROL,
    "primary": ObservationRole.PRIMARY,
    "wrong-action": ObservationRole.WRONG_ACTION,
    "wrong-time": ObservationRole.WRONG_TIME,
}


def project_response_method_identification_dataset(
    *,
    dataset_id: str,
    manifest: IdentificationEvidenceManifest,
    projection: IdentificationEvidenceProjection,
    one_factor_exchanges: tuple[OneFactorExchange, ...],
) -> IdentificationDataset:
    "Strict lowering only for the subset exactly representable by the response-method identification dataset."

    validate_stable_id(dataset_id, field_name="dataset_id")
    if projection.manifest != ObjectIdentity.from_record(manifest.manifest_id, manifest):
        raise ValueError("R6 projection binds another manifest")
    values_by_id = {
        row.projected_observation_id: {value.value_id: value for value in row.compact_values}
        for row in projection.observations
    }
    observations = []
    for row in projection.observations:
        if row.disposition is not ObservationDisposition.COMPLETE:
            raise ValueError("the response-method identification dataset cannot represent non-complete observation dispositions")
        if row.action_word is not None:
            raise ValueError("the response-method identification dataset cannot losslessly represent current exact ActionWord identity")
        try:
            split = _RESPONSE_METHOD_SPLITS[row.split_id]
            role = _RESPONSE_METHOD_ROLES[row.role_id]
        except KeyError as error:
            raise ValueError("R6 projection uses an unknown split or role") from error
        values = values_by_id[row.projected_observation_id]

        def selected(ids: tuple[str, ...]) -> tuple[NamedDecimal, ...]:
            return tuple(values[value_id] for value_id in ids)

        observations.append(
            LawObservation(
                observation_id=row.manifest_observation_id,
                physical_unit_instance_id=row.physical_unit_instance_id,
                denominator_cell_id=row.denominator_cell_id,
                chart_id=row.chart_id,
                numerical_view_id=row.qualification_view_id,
                split=split,
                role=role,
                denominator_values=selected(row.denominator_value_ids),
                history_values=selected(row.history_value_ids),
                action_values=selected(row.action_value_ids),
                receiver_values=selected(row.receiver_value_ids),
                tags=row.tags,
            )
        )
    return IdentificationDataset(
        dataset_id=dataset_id,
        system=manifest.system,
        relation=manifest.relation,
        information_cutoff_id=manifest.information_cutoff_id,
        observations=tuple(sorted(observations, key=lambda value: value.observation_id)),
        one_factor_exchanges=one_factor_exchanges,
        evidence_artifacts=tuple(
            sorted(
                (value.artifact for value in manifest.payloads),
                key=lambda value: value.artifact_id,
            )
        ),
        outcome_access=projection.outcome_access,
        parent_visibility_ceilings=projection.parent_visibility_ceilings,
        visibility_ceiling=projection.visibility_ceiling,
    )
