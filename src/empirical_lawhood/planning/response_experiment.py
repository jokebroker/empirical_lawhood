"Outcome-blind extension configs for reusable parameterised response experiments.\n\nThe records in this module are shallow joins over existing scientific owners.\nThey do not hand-author ``RunPlan``/``ExecutionPlan``, contain executable\nobjects, grant authority, or carry observed outcomes.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignProfile
from empirical_lawhood.planning.local_support_compatibility import PreparedDenominatorLocalSupportCompatibilitySpec
from empirical_lawhood.planning.matched_action_hold_controller_evaluation import MatchedActionHoldControllerEvaluationTemplate
from empirical_lawhood.planning.prospective_config import ProspectivePackageLineage, ProspectiveSequentialActionWordChart, ProspectiveTerminalMatrix
from empirical_lawhood.planning.prospective_controller_study import AdmissionControllerStudyTemplate
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile


MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES = 32 * 1024 * 1024


def _require_identity_schema(
    value: ObjectIdentity,
    schema: str,
    *,
    field_name: str,
) -> None:
    if value.object_schema != schema:
        raise ValueError(f"{field_name} requires {schema}")


def _require_prospective_authoring(
    *,
    outcome_access: OutcomeAccess,
    visibility_ceiling: VisibilityCeiling,
    grants_authority: bool,
) -> None:
    if outcome_access is not OutcomeAccess.OUTCOME_BLIND:
        raise ValueError("parameterised experiment config must remain outcome-blind")
    if visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
        raise ValueError("parameterised experiment config must remain prospective")
    if grants_authority:
        raise ValueError("parameterised experiment config cannot grant authority")


class ResponseDataSplitRole(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


@dataclass(frozen=True, slots=True)
class ResponsePreparationUnit(CanonicalRecord):
    """One independent preparation instance; nested views never add replication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-preparation-unit'

    physical_independent_unit_id: str
    preparation_coordinate_id: str
    preparation_instance_id: str
    preparation_sha256: str
    adapter_realization_id: str | None
    split_role: ResponseDataSplitRole

    def __post_init__(self) -> None:
        validate_stable_id(
            self.physical_independent_unit_id,
            field_name="physical_independent_unit_id",
        )
        validate_stable_id(
            self.preparation_coordinate_id,
            field_name="preparation_coordinate_id",
        )
        validate_stable_id(
            self.preparation_instance_id,
            field_name="preparation_instance_id",
        )
        if self.adapter_realization_id is not None:
            validate_stable_id(
                self.adapter_realization_id,
                field_name="adapter_realization_id",
            )
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")


@dataclass(frozen=True, slots=True)
class ResponseAcquisitionView(CanonicalRecord):
    """One scientific view of a declared native acquisition group."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-acquisition-view'

    view_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    numerical_member_id: str
    action_word_id: str | None

    def __post_init__(self) -> None:
        for name in (
            "view_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
            "numerical_member_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.action_word_id is not None:
            validate_stable_id(self.action_word_id, field_name="action_word_id")


@dataclass(frozen=True, slots=True)
class ResponseAcquisitionGroup(CanonicalRecord):
    """One source effect/object shared by one or more scientific views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-acquisition-group'

    acquisition_group_id: str
    physical_independent_unit_id: str
    preparation_instance_id: str
    view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "acquisition_group_id",
            "physical_independent_unit_id",
            "preparation_instance_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.view_ids,
            field_name="view_ids",
            allow_empty=False,
        )
        for view_id in self.view_ids:
            validate_stable_id(view_id, field_name="view_ids")


@dataclass(frozen=True, slots=True)
class ResponseQualificationConfig(CanonicalRecord):
    "Response qualification extension inputs over the existing strict profile roots."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-qualification-config'
    SOURCE_ROOT_SCHEMA: ClassVar[str] = SourcePipelineProfile.SCHEMA
    CAMPAIGN_ROOT_SCHEMA: ClassVar[str] = LinkedCampaignProfile.SCHEMA
    GROUP_NUMERICAL_VIEWS: ClassVar[bool] = False
    REQUIRES_PERMANENT_EVALUATION: ClassVar[bool] = True

    config_id: str
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    linked_campaign_profile: ObjectIdentity
    action_chart: ProspectiveSequentialActionWordChart | None
    package_lineage: ProspectivePackageLineage
    terminal_matrix: ProspectiveTerminalMatrix
    physical_units: tuple[ResponsePreparationUnit, ...]
    acquisition_groups: tuple[ResponseAcquisitionGroup, ...]
    nested_views: tuple[ResponseAcquisitionView, ...]
    identification_admission_physical_independent_unit_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    numerical_member_ids: tuple[str, ...]
    projection_config: ObjectIdentity
    method_config: ObjectIdentity
    qualification_profile: ObjectIdentity
    law_finalizer_owner: ObjectIdentity
    batch_atlas_owner: ObjectIdentity | None
    maximum_evidence_ceiling: EvidenceCeiling
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.CAMPAIGN_ROOT_SCHEMA == LinkedCampaignProfile.SCHEMA
            and self.batch_atlas_owner is None
        ):
            raise ValueError("linked response qualification requires its batch/atlas owner")
        _require_identity_schema(
            self.evidence_profile_selection,
            EvidenceProfileSelection.SCHEMA,
            field_name="evidence_profile_selection",
        )
        _require_identity_schema(
            self.source_pipeline_profile,
            self.SOURCE_ROOT_SCHEMA,
            field_name="source_pipeline_profile",
        )
        _require_identity_schema(
            self.linked_campaign_profile,
            self.CAMPAIGN_ROOT_SCHEMA,
            field_name="linked_campaign_profile",
        )
        for name, values in (
            (
                'identification_admission_physical_independent_unit_ids',
                self.identification_admission_physical_independent_unit_ids,
            ),
            ("development_unit_ids", self.development_unit_ids),
            ("evaluation_unit_ids", self.evaluation_unit_ids),
            ("numerical_member_ids", self.numerical_member_ids),
        ):
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=(
                    name == "evaluation_unit_ids" and not self.REQUIRES_PERMANENT_EVALUATION
                ),
            )
            for value in values:
                validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.physical_units,
            attribute="physical_independent_unit_id",
            field_name="physical_units",
        )
        require_sorted_unique_ids(
            self.nested_views,
            attribute="view_id",
            field_name="nested_views",
        )
        require_sorted_unique_ids(
            self.acquisition_groups,
            attribute="acquisition_group_id",
            field_name="acquisition_groups",
        )
        if not self.physical_units or not self.acquisition_groups or not self.nested_views:
            raise ValueError(
                "response qualification config requires physical units, acquisition groups and nested views"
            )
        unit_ids = {value.physical_independent_unit_id for value in self.physical_units}
        if unit_ids != set(self.identification_admission_physical_independent_unit_ids):
            raise ValueError("response qualification physical-unit records differ from their roster")
        if len({value.preparation_instance_id for value in self.physical_units}) != len(
            self.physical_units
        ):
            raise ValueError("response qualification physical units reuse preparation-instance identity")
        realization_ids = {
            value.adapter_realization_id
            for value in self.physical_units
            if value.adapter_realization_id is not None
        }
        if len(realization_ids) != sum(
            value.adapter_realization_id is not None for value in self.physical_units
        ):
            raise ValueError("response qualification physical units reuse adapter realization identity")
        development = set(self.development_unit_ids)
        evaluation = set(self.evaluation_unit_ids)
        if development & evaluation:
            raise ValueError("response qualification development and evaluation units overlap")
        if development | evaluation != set(self.identification_admission_physical_independent_unit_ids):
            raise ValueError("response qualification split units do not partition the physical-unit roster")
        if development != {
            value.physical_independent_unit_id
            for value in self.physical_units
            if value.split_role is ResponseDataSplitRole.DEVELOPMENT
        } or evaluation != {
            value.physical_independent_unit_id
            for value in self.physical_units
            if value.split_role is ResponseDataSplitRole.EVALUATION
        }:
            raise ValueError("response qualification split labels differ from physical-unit records")
        chart_word_ids = (
            set()
            if self.action_chart is None
            else {value.word_id for value in self.action_chart.action_words}
        )
        if self.action_chart is None and any(
            value.action_word_id is not None for value in self.nested_views
        ):
            raise ValueError("read-only response qualification view cannot declare an action word")
        if self.action_chart is not None and any(
            value.action_word_id is None for value in self.nested_views
        ):
            raise ValueError("interactive response qualification view requires an action word")
        if any(
            value.physical_independent_unit_id not in unit_ids
            or value.numerical_member_id not in self.numerical_member_ids
            or (value.action_word_id is not None and value.action_word_id not in chart_word_ids)
            for value in self.nested_views
        ):
            raise ValueError("response qualification nested view lies outside a declared roster")
        unit_by_id = {value.physical_independent_unit_id: value for value in self.physical_units}
        view_by_id = {value.view_id: value for value in self.nested_views}
        declared_view_ids: list[str] = []
        for group in self.acquisition_groups:
            unit = unit_by_id.get(group.physical_independent_unit_id)
            if unit is None or unit.preparation_instance_id != group.preparation_instance_id:
                raise ValueError("response qualification acquisition group changes its preparation lineage")
            group_views: list[ResponseAcquisitionView] = []
            for view_id in group.view_ids:
                view = view_by_id.get(view_id)
                if (
                    view is None
                    or view.acquisition_group_id != group.acquisition_group_id
                    or view.physical_independent_unit_id != group.physical_independent_unit_id
                ):
                    raise ValueError("response qualification acquisition group changes its view lineage")
                group_views.append(view)
                declared_view_ids.append(view_id)
            if (
                self.action_chart is not None
                and len(
                    {
                        (
                            view.action_word_id,
                            None if self.GROUP_NUMERICAL_VIEWS else view.numerical_member_id,
                        )
                        for view in group_views
                    }
                )
                != 1
            ):
                raise ValueError(
                    "interactive response qualification acquisition group requests heterogeneous episodes"
                )
        if len(declared_view_ids) != len(set(declared_view_ids)):
            raise ValueError("response qualification scientific view belongs to multiple acquisitions")
        if set(declared_view_ids) != set(view_by_id):
            raise ValueError("response qualification acquisition groups do not cover every scientific view")
        if {value.numerical_member_id for value in self.nested_views} != set(
            self.numerical_member_ids
        ):
            raise ValueError("response qualification nested views omit a numerical member")
        if {value.physical_independent_unit_id for value in self.nested_views} != unit_ids:
            raise ValueError("response qualification nested views omit a physical independent unit")
        if {value.action_word_id for value in self.nested_views if value.action_word_id} != (
            chart_word_ids
        ):
            raise ValueError("response qualification nested views omit an action word")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("response qualification config must stop at the local-law ceiling")
        _require_prospective_authoring(
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
            grants_authority=self.grants_authority,
        )


class AdmissionRouteKind(StrEnum):
    CONTROLLED_IO_REACHABILITY = "CONTROLLED_IO_REACHABILITY"
    FINITE_INTERVENTION_ADMISSION = "FINITE_INTERVENTION_ADMISSION"


@dataclass(frozen=True, slots=True)
class AdmissionExperimentConfig(CanonicalRecord):
    "Conditional admission extension inputs; all scientific owners remain external."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-experiment-config'

    config_id: str
    identification_config: ObjectIdentity
    route_kind: AdmissionRouteKind
    supported_law_binding_rule_id: str
    local_support_spec: PreparedDenominatorLocalSupportCompatibilitySpec
    controlled_io_member_config: ObjectIdentity | None
    controlled_io_excluded_feasibility_spec: ObjectIdentity | None
    raw_admission_receipt_owner: ObjectIdentity
    admission_owner: ObjectIdentity
    reachability_owner: ObjectIdentity | None
    study_template: AdmissionControllerStudyTemplate
    controller_compiler_owner: ObjectIdentity
    controller_runtime_owner: ObjectIdentity
    admission_gate_ids: tuple[str, ...]
    finite_intervention_claim_ceiling: str
    maximum_evidence_ceiling: EvidenceCeiling
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(
            self.supported_law_binding_rule_id,
            field_name="supported_law_binding_rule_id",
        )
        _require_identity_schema(
            self.identification_config,
            ResponseQualificationConfig.SCHEMA,
            field_name='identification_config',
        )
        require_sorted_unique_strings(
            self.admission_gate_ids,
            field_name='admission_gate_ids',
            allow_empty=False,
        )
        for gate_id in self.admission_gate_ids:
            validate_stable_id(gate_id, field_name='admission_gate_ids')
        validate_stable_id(
            self.finite_intervention_claim_ceiling,
            field_name="finite_intervention_claim_ceiling",
        )
        if (
            self.local_support_spec.prepared_denominator_id
            != self.study_template.prepared_denominator_id
        ):
            raise ValueError("admission local support and programme change prepared denominator")
        if self.route_kind is AdmissionRouteKind.CONTROLLED_IO_REACHABILITY:
            if (
                self.controlled_io_member_config is None
                or self.controlled_io_excluded_feasibility_spec is None
                or self.reachability_owner is None
            ):
                raise ValueError("controlled-I/O admission requires construction, obstruction and owner")
        elif (
            self.controlled_io_member_config is not None
            or self.controlled_io_excluded_feasibility_spec is not None
            or self.reachability_owner is not None
        ):
            raise ValueError("finite-intervention admission cannot claim controlled-I/O operands")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("admission config must remain at the admission ceiling")
        _require_prospective_authoring(
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
            grants_authority=self.grants_authority,
        )


@dataclass(frozen=True, slots=True)
class ProspectiveUseExperimentConfig(CanonicalRecord):
    "Pre-admission frozen matched ACTION/HOLD prospective-evaluation extension."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-use-experiment-config'

    config_id: str
    admission_config: ObjectIdentity
    evaluation_template: MatchedActionHoldControllerEvaluationTemplate
    frozen_before_admission_outcome_visibility: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        _require_identity_schema(
            self.admission_config,
            AdmissionExperimentConfig.SCHEMA,
            field_name='admission_config',
        )
        if not self.frozen_before_admission_outcome_visibility:
            raise ValueError("controller use design must freeze before admission outcome visibility")
        _require_prospective_authoring(
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
            grants_authority=self.grants_authority,
        )


@dataclass(frozen=True, slots=True)
class ResponseExperimentExtensionSet(CanonicalRecord):
    "Closed response experiment extension set over three existing strict root identities."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-experiment-extension-set'

    extension_set_id: str
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    linked_campaign_profile: ObjectIdentity
    identification_config: ResponseQualificationConfig
    admission_config: AdmissionExperimentConfig | None
    prospective_evaluation_config: ProspectiveUseExperimentConfig | None
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_set_id, field_name="extension_set_id")
        if (
            type(self) is ResponseExperimentExtensionSet
            and type(self.identification_config) is not ResponseQualificationConfig
        ):
            raise ValueError("static response experiment extension requires its exact static-root response qualification schema")
        identification_config_identity = ObjectIdentity.from_record(
            self.identification_config.config_id,
            self.identification_config,
        )
        if (
            self.identification_config.evidence_profile_selection != self.evidence_profile_selection
            or self.identification_config.source_pipeline_profile != self.source_pipeline_profile
            or self.identification_config.linked_campaign_profile != self.linked_campaign_profile
        ):
            raise ValueError("response qualification config differs from strict root identities")
        if self.prospective_evaluation_config is not None and self.admission_config is None:
            raise ValueError("controller use applicability requires admission applicability")
        if self.admission_config is not None:
            admission_config_identity = ObjectIdentity.from_record(self.admission_config.config_id, self.admission_config)
            if self.admission_config.identification_config != identification_config_identity:
                raise ValueError("admission config binds another response qualification config")
            chart = self.identification_config.action_chart
            if chart is None:
                raise ValueError("applicable admission requires an interactive response qualification action chart")
            chart_word_ids = {value.word_id for value in chart.action_words}
            programme = self.admission_config.study_template
            if (
                self.admission_config.local_support_spec.chart_id != chart.chart_id
                or {value.object_id for value in self.admission_config.local_support_spec.action_words}
                - chart_word_ids
            ):
                raise ValueError("admission local-support actions differ from the prospective chart")
            if {
                programme.active_action_word_id,
                programme.hold_action_word_id,
            } - chart_word_ids:
                raise ValueError("admission action words lie outside the prospective chart")
            if self.prospective_evaluation_config is not None:
                template = self.prospective_evaluation_config.evaluation_template
                if self.prospective_evaluation_config.admission_config != admission_config_identity:
                    raise ValueError("controller use config binds another admission config")
                if {
                    template.qualified_action_word.word_id,
                    template.qualified_hold_word.word_id,
                } - chart_word_ids:
                    raise ValueError("controller use action words lie outside the prospective chart")
                if (
                    programme.active_action_word_id != template.qualified_action_word.word_id
                    or programme.hold_action_word_id != template.qualified_hold_word.word_id
                    or programme.expected_study_id != template.expected_eligible_admission_study_id
                ):
                    raise ValueError("admission/controller use programme or action continuity differs")
                prospective_evaluation_independent_unit_ids = {value.physical_independent_unit_id for value in template.units}
                if prospective_evaluation_independent_unit_ids & set(self.identification_config.identification_admission_physical_independent_unit_ids):
                    raise ValueError("identification and admission and controller use physical independent units overlap")
        _require_prospective_authoring(
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
            grants_authority=self.grants_authority,
        )

    @property
    def config_identities(self) -> tuple[ObjectIdentity, ...]:
        records: tuple[
            ResponseQualificationConfig | AdmissionExperimentConfig | ProspectiveUseExperimentConfig,
            ...,
        ] = tuple(
            value
            for value in (self.identification_config, self.admission_config, self.prospective_evaluation_config)
            if value is not None
        )
        return tuple(ObjectIdentity.from_record(value.config_id, value) for value in records)


def decode_response_qualification_config(payload: bytes) -> ResponseQualificationConfig:
    return decode_canonical_bytes(
        payload,
        ResponseQualificationConfig,
        maximum_bytes=MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES,
    )


def decode_admission_experiment_config(payload: bytes) -> AdmissionExperimentConfig:
    return decode_canonical_bytes(
        payload,
        AdmissionExperimentConfig,
        maximum_bytes=MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES,
    )


def decode_prospective_use_experiment_config(payload: bytes) -> ProspectiveUseExperimentConfig:
    return decode_canonical_bytes(
        payload,
        ProspectiveUseExperimentConfig,
        maximum_bytes=MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES,
    )


def decode_response_experiment_extension_set(
    payload: bytes,
) -> ResponseExperimentExtensionSet:
    return decode_canonical_bytes(
        payload,
        ResponseExperimentExtensionSet,
        maximum_bytes=MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES,
    )


__all__ = [
    "MAX_PARAMETERISED_RESPONSE_CONFIG_BYTES",
    'ResponseQualificationConfig',
    'ResponseAcquisitionGroup',
    'ResponseAcquisitionView',
    'ResponsePreparationUnit',
    'ResponseDataSplitRole',
    'ResponseExperimentExtensionSet',
    'AdmissionExperimentConfig',
    'AdmissionRouteKind',
    'ProspectiveUseExperimentConfig',
    'decode_response_qualification_config',
    'decode_response_experiment_extension_set',
    'decode_admission_experiment_config',
    'decode_prospective_use_experiment_config',
]
