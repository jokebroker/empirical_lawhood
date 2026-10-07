"Strict prospective native-source roots for interactive response qualification acquisition.\n\nUnlike a static source/ETL profile, this declares effects that have no observation\nmaterialization or content digest yet. The exact native configuration, installed\nselection, independent-unit roster and work bounds are fixed before contact.\n"

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    require_sorted_unique_ids,
    validate_schema,
    validate_stable_id,
)
from .study_authoring import CapabilitySelection
from .campaigns import CampaignSpec
from .response_experiment import ResponseQualificationConfig, ResponseExperimentExtensionSet


@dataclass(frozen=True, slots=True)
class NativeSourceProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-source-profile'
    profile_id: str
    source_selection: CapabilitySelection
    source_config: ObjectIdentity
    observation_schema: str
    physical_independent_unit_ids: tuple[str, ...]
    maximum_native_segment_count: int
    maximum_native_update_count: int
    resource_ceiling: ResourceBudget
    prerequisite_qualification: ObjectIdentity
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_schema(self.observation_schema)
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        for unit in self.physical_independent_unit_ids:
            validate_stable_id(unit, field_name="physical_independent_unit_ids")
        if any(
            type(value) is not int or value <= 0
            for value in (self.maximum_native_segment_count, self.maximum_native_update_count)
        ):
            raise ValueError("native source requires positive finite segment/update bounds")
        if (
            self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("native source profile cannot acquire, reveal or grant authority")


@dataclass(frozen=True, slots=True)
class NativeProjectionGroup(CanonicalRecord):
    """One pure artifact-producing task over named views of one unit/member."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-projection-group'
    projection_task_id: str
    scientific_view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_task_id, field_name="projection_task_id")
        require_sorted_unique_strings(
            self.scientific_view_ids, field_name="scientific_view_ids", allow_empty=False
        )
        for view in self.scientific_view_ids:
            validate_stable_id(view, field_name="scientific_view_ids")


@dataclass(frozen=True, slots=True)
class NativeLawQualificationConfig(ResponseQualificationConfig):
    "Use the unchanged unit/action laws with an explicitly native source root.\n\n    Compatibility map: inherited ``source_pipeline_profile`` names the exact\n    NativeSourceProfile identity in this schema only. It never denotes an ETL\n    request or a promise of already materialized observation bytes. The inherited\n    ``linked_campaign_profile`` names the actual CampaignSpec, not the four-package\n    controller-use topology. No batch/atlas or admission/controller-use owner is consumed at this rung.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-law-qualification-config'
    SOURCE_ROOT_SCHEMA: ClassVar[str] = NativeSourceProfile.SCHEMA
    CAMPAIGN_ROOT_SCHEMA: ClassVar[str] = CampaignSpec.SCHEMA
    GROUP_NUMERICAL_VIEWS: ClassVar[bool] = True
    projection_groups: tuple[NativeProjectionGroup, ...]

    def __post_init__(self) -> None:
        super(NativeLawQualificationConfig, self).__post_init__()
        if (
            self.action_chart is None
            or self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or self.batch_atlas_owner is not None
        ):
            raise ValueError("native response qualification requires an interactive chart, local-law ceiling and no atlas")
        require_sorted_unique_ids(
            self.projection_groups, attribute="projection_task_id", field_name="projection_groups"
        )
        views = {v.view_id: v for v in self.nested_views}
        grouped = [v for g in self.projection_groups for v in g.scientific_view_ids]
        if len(grouped) != len(views) or set(grouped) != set(views):
            raise ValueError("native projection groups must partition the scientific views exactly")
        for group in self.projection_groups:
            if (
                len(
                    {
                        (views[v].physical_independent_unit_id, views[v].numerical_member_id)
                        for v in group.scientific_view_ids
                    }
                )
                != 1
            ):
                raise ValueError(
                    "native projection group mixes independent units or numerical members"
                )


@dataclass(frozen=True, slots=True)
class NativeLawQualificationExperiment(ResponseExperimentExtensionSet):
    "Native strict-root variant; admission and controller-use stages are absent, with the same base topology."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-law-qualification-experiment'
    CONFIG_TYPE: ClassVar[type[NativeLawQualificationConfig]] = NativeLawQualificationConfig
    identification_config: NativeLawQualificationConfig

    def __post_init__(self) -> None:
        super(NativeLawQualificationExperiment, self).__post_init__()
        if (
            type(self.identification_config) is not self.CONFIG_TYPE
            or self.admission_config is not None
            or self.prospective_evaluation_config is not None
        ):
            raise ValueError("native response qualification extension cannot embed static roots or admission/controller-use overlays")


@dataclass(frozen=True, slots=True)
class NativeCrossfitFold(CanonicalRecord):
    """Whole independent units held from one fit; views inherit their unit's role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-crossfit-fold'
    fold_id: str
    training_unit_ids: tuple[str, ...]
    heldout_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fold_id, field_name="fold_id")
        for name in ("training_unit_ids", "heldout_unit_ids"):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        if set(self.training_unit_ids) & set(self.heldout_unit_ids):
            raise ValueError("native crossfit training and held-out units overlap")


@dataclass(frozen=True, slots=True)
class NativeCrossfitPartition(CanonicalRecord):
    """A complete fold partition of one declared fitting population."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-crossfit-partition'
    partition_id: str
    population_unit_ids: tuple[str, ...]
    folds: tuple[NativeCrossfitFold, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.partition_id, field_name="partition_id")
        require_sorted_unique_strings(
            self.population_unit_ids, field_name="population_unit_ids", allow_empty=False
        )
        require_sorted_unique_ids(self.folds, attribute="fold_id", field_name="folds")
        if not 2 <= len(self.folds) <= 16 or len(self.population_unit_ids) > 4096:
            raise ValueError("native crossfit partition exceeds its finite population/fold bounds")
        population = set(self.population_unit_ids)
        held = tuple(unit for fold in self.folds for unit in fold.heldout_unit_ids)
        if tuple(sorted(held)) != self.population_unit_ids or any(
            set(fold.training_unit_ids) != population - set(fold.heldout_unit_ids)
            for fold in self.folds
        ):
            raise ValueError("native crossfit folds do not exactly partition their population")


@dataclass(frozen=True, slots=True)
class NativeNestedCrossfitPopulation(CanonicalRecord):
    """Each outer training population has its own inner partition, plus all-D selection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-nested-crossfit-population'
    population_id: str
    outer_partition: NativeCrossfitPartition
    inner_partitions: tuple[NativeCrossfitPartition, ...]
    final_selection_partition: NativeCrossfitPartition
    final_refit_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.population_id, field_name="population_id")
        require_sorted_unique_ids(
            self.inner_partitions, attribute="partition_id", field_name="inner_partitions"
        )
        expected = {fold.fold_id: fold.training_unit_ids for fold in self.outer_partition.folds}
        if (
            {part.partition_id: part.population_unit_ids for part in self.inner_partitions}
            != expected
            or self.final_selection_partition.population_unit_ids
            != self.outer_partition.population_unit_ids
            or self.final_refit_unit_ids != self.outer_partition.population_unit_ids
        ):
            raise ValueError("native nested crossfit leaks an outer root or changes final all-D refit")
        partition_ids = (
            self.outer_partition.partition_id,
            self.final_selection_partition.partition_id,
            *(part.partition_id for part in self.inner_partitions),
        )
        if len(set(partition_ids)) != len(partition_ids):
            raise ValueError("native nested crossfit partition identities overlap")


@dataclass(frozen=True, slots=True)
class NativeNestedCrossfitPlan(CanonicalRecord):
    """Nominal development followed by a separately issued fresh qualification population."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-nested-crossfit-plan'
    plan_id: str
    populations: tuple[NativeNestedCrossfitPopulation, ...]
    qualification_study_id: str
    qualification_unit_ids: tuple[str, ...]
    learning_scope: str = "NOMINAL_DEVELOPMENT_ONLY_SEPARATE_FRESH_QUALIFICATION"

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.qualification_study_id, field_name="qualification_study_id")
        require_sorted_unique_ids(
            self.populations, attribute="population_id", field_name="populations"
        )
        require_sorted_unique_strings(
            self.qualification_unit_ids, field_name="qualification_unit_ids", allow_empty=False
        )
        for value in self.qualification_unit_ids:
            validate_stable_id(value, field_name="qualification_unit_ids")
        units = self.development_unit_ids
        if (
            not 1 <= len(self.populations) <= 16
            or len(units) > 4096
            or len(units) != len(set(units))
            or len(self.qualification_unit_ids) > 4096
            or set(units) & set(self.qualification_unit_ids)
            or self.learning_scope != "NOMINAL_DEVELOPMENT_ONLY_SEPARATE_FRESH_QUALIFICATION"
        ):
            raise ValueError("native crossfit plan changes its independent populations or ceiling")

    @property
    def development_unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted(unit for pop in self.populations for unit in pop.final_refit_unit_ids))


@dataclass(frozen=True, slots=True)
class NativeCrossfitConfig(NativeLawQualificationConfig):
    "All acquired units are development units; fold-specific roles replace a permanent evaluation split.\n\n    The inherited local-law value is a maximum ceiling, not a qualification\n    result. This nominal stage cannot publish a protected calibrated law. Its\n    finalizer/qualification bindings name the separate qualification child's owners.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-crossfit-config'
    REQUIRES_PERMANENT_EVALUATION: ClassVar[bool] = False
    crossfit_plan: NativeNestedCrossfitPlan

    def __post_init__(self) -> None:
        super(NativeCrossfitConfig, self).__post_init__()
        if (
            self.evaluation_unit_ids
            or self.development_unit_ids != self.crossfit_plan.development_unit_ids
        ):
            raise ValueError("native crossfit config requires all and only its development units")


@dataclass(frozen=True, slots=True)
class NativeCrossfitExperiment(NativeLawQualificationExperiment):
    "Exact native cross-fitting root; no admission/controller-use overlay or permanent held-out split."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-crossfit-experiment'
    CONFIG_TYPE: ClassVar[type[NativeLawQualificationConfig]] = NativeCrossfitConfig
    identification_config: NativeCrossfitConfig
