"""Strict adapter-local contracts for the Virtual Cell Challenge programme.

These records add transcriptomic source, prediction, scoring, placement and
empirical-replay semantics without changing the repository kernel ontology.
Large matrices remain external artifacts; only bounded identities and compact
scientific contracts cross the application platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_relative_locator,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind


class VirtualCellContractError(ValueError):
    """Raised when VCC source or experiment semantics fail closed."""


class VirtualCellSplit(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    SHARED = "SHARED"


class VirtualCellEvidenceWorld(StrEnum):
    """Adapter-local source world with an explicit kernel compatibility map."""

    RETROSPECTIVE_PHYSICAL_DATASET = "RETROSPECTIVE_PHYSICAL_DATASET"


class MatrixEncoding(StrEnum):
    CSR = "CSR"
    CSC = "CSC"
    DENSE = "DENSE"


class ActionStageAvailability(StrEnum):
    OBSERVED_UPSTREAM = "OBSERVED_UPSTREAM"
    OBSERVED_RESPONSE_DERIVED = "OBSERVED_RESPONSE_DERIVED"
    UNKNOWN = "UNKNOWN"


class IndependentUnitStatus(StrEnum):
    PHYSICAL_UNIT_KNOWN = "PHYSICAL_UNIT_KNOWN"
    BATCH_ONLY_CEILING = "BATCH_ONLY_CEILING"
    NESTED_CELL_ONLY = "NESTED_CELL_ONLY"
    UNKNOWN_CEILING = "UNKNOWN_CEILING"


class PlacementClass(StrEnum):
    EXACT_VALIDATED_RANK = "EXACT_VALIDATED_RANK"
    TIED_RANK_INTERVAL = "TIED_RANK_INTERVAL"
    ROUNDED_SCORE_RANK_INTERVAL = "ROUNDED_SCORE_RANK_INTERVAL"
    PARTIAL_FIELD_RANK_BOUND = "PARTIAL_FIELD_RANK_BOUND"
    PERCENTILE_ONLY = "PERCENTILE_ONLY"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ReplayControllerUseDisposition(StrEnum):
    REPLAY_CONTROLLER_USE_VALIDATED = "REPLAY_CONTROLLER_USE_VALIDATED"
    REPLAY_CONTROLLER_USE_REJECTED = "REPLAY_CONTROLLER_USE_REJECTED"
    REPLAY_CONTROLLER_USE_SAFE_HOLD_ONLY = "REPLAY_CONTROLLER_USE_SAFE_HOLD_ONLY"
    REPLAY_CONTROLLER_USE_UNEVALUABLE = "REPLAY_CONTROLLER_USE_UNEVALUABLE"


class TranslationDisposition(StrEnum):
    REUSE = "REUSE"
    ADAPT = "ADAPT"
    RETRAIN = "RETRAIN"
    REJECT = "REJECT"


def _positive_int(value: int, *, field_name: str, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be a {qualifier} integer")


def _nonblank_tuple(
    values: tuple[str, ...], *, field_name: str, sorted_set: bool
) -> None:
    if sorted_set:
        require_sorted_unique_strings(values, field_name=field_name)
    for value in values:
        validate_nonempty(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class NamedDecimal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/named-decimal'

    name: str
    value: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.name, field_name="name")
        validate_decimal(self.value, field_name="value")


@dataclass(frozen=True, slots=True)
class VirtualCellDenominator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-denominator'

    denominator_id: str
    material: str
    context: str
    perturbation_modality: str
    assay: str
    comparator: str
    batch_keys: tuple[str, ...]
    preparation_keys: tuple[str, ...]
    processing_version: str
    source_world: VirtualCellEvidenceWorld

    @property
    def kernel_world_kind(self) -> WorldKind:
        """Map archived physical evidence without pretending it is a new act."""

        return WorldKind.PHYSICAL_EXPERIMENT

    def __post_init__(self) -> None:
        validate_stable_id(self.denominator_id, field_name="denominator_id")
        for name, value in (
            ("material", self.material),
            ("context", self.context),
            ("perturbation_modality", self.perturbation_modality),
            ("assay", self.assay),
            ("comparator", self.comparator),
            ("processing_version", self.processing_version),
        ):
            validate_nonempty(value, field_name=name)
        _nonblank_tuple(self.batch_keys, field_name="batch_keys", sorted_set=True)
        _nonblank_tuple(
            self.preparation_keys, field_name="preparation_keys", sorted_set=True
        )
        if (
            self.source_world
            is not VirtualCellEvidenceWorld.RETROSPECTIVE_PHYSICAL_DATASET
        ):
            raise ValueError(
                "the 2025 VCC source denominator is a retrospective dataset"
            )


@dataclass(frozen=True, slots=True)
class VirtualCellAction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-action'

    action_id: str
    target_gene: str
    guide_ids: tuple[str, ...]
    requested: ActionStageAvailability
    accepted: ActionStageAvailability
    applied: ActionStageAvailability
    realized: ActionStageAvailability
    source_field_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        validate_nonempty(self.target_gene, field_name="target_gene")
        _nonblank_tuple(self.guide_ids, field_name="guide_ids", sorted_set=True)
        _nonblank_tuple(
            self.source_field_ids, field_name="source_field_ids", sorted_set=True
        )
        if self.requested is not ActionStageAvailability.OBSERVED_UPSTREAM:
            raise ValueError(
                "a VCC action requires an upstream nominal requested target"
            )
        if self.realized is ActionStageAvailability.OBSERVED_UPSTREAM:
            raise ValueError(
                "realized repression cannot be treated as an upstream action field"
            )


@dataclass(frozen=True, slots=True)
class VirtualCellSourcePart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-source-part'

    part_id: str
    relative_locator: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.part_id, field_name="part_id")
        validate_relative_locator(self.relative_locator)
        _positive_int(self.size_bytes, field_name="size_bytes")
        validate_sha256(self.sha256, field_name="sha256")


@dataclass(frozen=True, slots=True)
class VirtualCellSourceObject(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-source-object'

    object_id: str
    split: VirtualCellSplit
    role: str
    source_name: str
    generation: str
    size_bytes: int
    sha256: str
    crc32c_base64: str
    media_type: str
    parts: tuple[VirtualCellSourcePart, ...]
    sealed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.object_id, field_name="object_id")
        for name, value in (
            ("role", self.role),
            ("source_name", self.source_name),
            ("generation", self.generation),
            ("crc32c_base64", self.crc32c_base64),
            ("media_type", self.media_type),
        ):
            validate_nonempty(value, field_name=name)
        _positive_int(self.size_bytes, field_name="size_bytes")
        validate_sha256(self.sha256, field_name="sha256")
        require_sorted_unique_ids(self.parts, attribute="part_id", field_name="parts")
        if (
            not self.parts
            or sum(value.size_bytes for value in self.parts) != self.size_bytes
        ):
            raise ValueError(
                "source parts must exactly close the source-object byte count"
            )
        if (
            self.split is VirtualCellSplit.TEST
            and "response" in self.role
            and not self.sealed
        ):
            raise ValueError("test response objects must remain evaluator-sealed")
        if self.sealed and self.split is not VirtualCellSplit.TEST:
            raise ValueError(
                "only the test split may be sealed in the 2025 source family"
            )


@dataclass(frozen=True, slots=True)
class SplitCardinality(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/split-cardinality'
    )

    split: VirtualCellSplit
    target_count: int
    reported_perturbed_cell_count: int

    def __post_init__(self) -> None:
        if self.split is VirtualCellSplit.SHARED:
            raise ValueError("shared source objects do not define a target split")
        _positive_int(self.target_count, field_name="target_count")
        _positive_int(
            self.reported_perturbed_cell_count,
            field_name="reported_perturbed_cell_count",
        )


@dataclass(frozen=True, slots=True)
class VirtualCellSourceManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-source-manifest'

    manifest_id: str
    release_id: str
    upstream_prefix: str
    storage_root_id: str
    license_locator: str
    custody_receipt_artifact_id: str
    custody_receipt_sha256: str
    gene_count: int
    gene_order_sha256: str
    cardinalities: tuple[SplitCardinality, ...]
    objects: tuple[VirtualCellSourceObject, ...]
    expected_object_count: int
    complete_processed_release: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_id", self.manifest_id),
            ("release_id", self.release_id),
            ("storage_root_id", self.storage_root_id),
            ("custody_receipt_artifact_id", self.custody_receipt_artifact_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.upstream_prefix, field_name="upstream_prefix")
        validate_nonempty(self.license_locator, field_name="license_locator")
        validate_sha256(
            self.custody_receipt_sha256, field_name="custody_receipt_sha256"
        )
        validate_sha256(self.gene_order_sha256, field_name="gene_order_sha256")
        _positive_int(self.gene_count, field_name="gene_count")
        _positive_int(self.expected_object_count, field_name="expected_object_count")
        require_sorted_unique_ids(
            self.objects, attribute="object_id", field_name="objects"
        )
        if len(self.objects) != self.expected_object_count:
            raise ValueError(
                "source manifest object count differs from its exact expectation"
            )
        splits = tuple(value.split for value in self.cardinalities)
        if splits != (
            VirtualCellSplit.TEST,
            VirtualCellSplit.TRAIN,
            VirtualCellSplit.VALIDATION,
        ):
            raise ValueError(
                "source cardinalities require canonical TEST/TRAIN/VALIDATION order"
            )
        if not self.complete_processed_release:
            raise ValueError("the registered 2025 source manifest must be complete")
        if not any(value.split is VirtualCellSplit.SHARED for value in self.objects):
            raise ValueError("source manifest lacks its shared feature registry")
        if not any(
            value.split is VirtualCellSplit.TEST and value.sealed
            for value in self.objects
        ):
            raise ValueError("source manifest lacks a sealed test response object")


@dataclass(frozen=True, slots=True)
class VirtualCellSchemaContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/virtual-cell-schema-contract'
    )

    contract_id: str
    split: VirtualCellSplit
    row_count: int
    gene_count: int
    matrix_path: str
    matrix_encoding: MatrixEncoding
    matrix_data_dtype: str
    matrix_indices_dtype: str | None
    obs_fields: tuple[str, ...]
    var_index_field: str
    layer_ids: tuple[str, ...]
    gene_order_sha256: str
    structure_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        _positive_int(self.row_count, field_name="row_count")
        _positive_int(self.gene_count, field_name="gene_count")
        for name, value in (
            ("matrix_path", self.matrix_path),
            ("matrix_data_dtype", self.matrix_data_dtype),
            ("var_index_field", self.var_index_field),
        ):
            validate_nonempty(value, field_name=name)
        if self.matrix_indices_dtype is not None:
            validate_nonempty(
                self.matrix_indices_dtype, field_name="matrix_indices_dtype"
            )
        _nonblank_tuple(self.obs_fields, field_name="obs_fields", sorted_set=True)
        _nonblank_tuple(self.layer_ids, field_name="layer_ids", sorted_set=True)
        validate_sha256(self.gene_order_sha256, field_name="gene_order_sha256")
        if self.matrix_encoding in {MatrixEncoding.CSR, MatrixEncoding.CSC}:
            if self.matrix_indices_dtype is None:
                raise ValueError("sparse schema requires an indices dtype")
        elif self.matrix_indices_dtype is not None:
            raise ValueError("dense schema cannot declare an indices dtype")
        if self.split is VirtualCellSplit.TEST and not self.structure_only:
            raise ValueError(
                "ordinary test schema qualification must remain structure-only"
            )


@dataclass(frozen=True, slots=True)
class IndependentUnitContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/independent-unit-contract'
    )

    contract_id: str
    status: IndependentUnitStatus
    physical_unit_keys: tuple[str, ...]
    nested_view_keys: tuple[str, ...]
    generalization_unit: str
    uncertainty_unit: str
    prohibited_replication_units: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        _nonblank_tuple(
            self.physical_unit_keys,
            field_name="physical_unit_keys",
            sorted_set=True,
        )
        _nonblank_tuple(
            self.nested_view_keys, field_name="nested_view_keys", sorted_set=True
        )
        _nonblank_tuple(
            self.prohibited_replication_units,
            field_name="prohibited_replication_units",
            sorted_set=True,
        )
        validate_nonempty(self.generalization_unit, field_name="generalization_unit")
        validate_nonempty(self.uncertainty_unit, field_name="uncertainty_unit")
        if self.status is IndependentUnitStatus.PHYSICAL_UNIT_KNOWN:
            if not self.physical_unit_keys:
                raise ValueError(
                    "known physical-unit status requires source-backed keys"
                )
        elif self.physical_unit_keys:
            raise ValueError(
                "unknown/ceiling unit status cannot name physical unit keys"
            )
        if "cell" not in self.prohibited_replication_units:
            raise ValueError(
                "cells must be explicitly prohibited as physical replicates"
            )


@dataclass(frozen=True, slots=True)
class SplitOutcomeAccess(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/split-outcome-access'
    )

    split: VirtualCellSplit
    access: OutcomeAccess
    prefix_field_ids: tuple[str, ...]
    outcome_field_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split is VirtualCellSplit.SHARED:
            raise ValueError("shared features do not carry split outcome access")
        _nonblank_tuple(
            self.prefix_field_ids, field_name="prefix_field_ids", sorted_set=True
        )
        _nonblank_tuple(
            self.outcome_field_ids, field_name="outcome_field_ids", sorted_set=True
        )
        if set(self.prefix_field_ids) & set(self.outcome_field_ids):
            raise ValueError("prefix and outcome fields must be disjoint")
        if (
            self.split is VirtualCellSplit.TEST
            and self.access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("test outcomes must remain evaluation-sealed")


@dataclass(frozen=True, slots=True)
class OutcomeAccessManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/outcome-access-manifest'
    )

    manifest_id: str
    information_cutoff_id: str
    split_access: tuple[SplitOutcomeAccess, ...]
    evaluator_capability_key: str
    leaderboard_parent_for_model: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_id", self.manifest_id),
            ("information_cutoff_id", self.information_cutoff_id),
            ("evaluator_capability_key", self.evaluator_capability_key),
        ):
            validate_stable_id(value, field_name=name)
        if tuple(value.split for value in self.split_access) != (
            VirtualCellSplit.TEST,
            VirtualCellSplit.TRAIN,
            VirtualCellSplit.VALIDATION,
        ):
            raise ValueError(
                "split access requires canonical TEST/TRAIN/VALIDATION order"
            )
        if self.leaderboard_parent_for_model:
            raise ValueError("the public leaderboard cannot be a model-lineage parent")


@dataclass(frozen=True, slots=True)
class ResponseSummarySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/response-summary-spec'
    )

    spec_id: str
    matrix_view: str
    normalization: str
    normalization_target_sum: Decimal | None
    comparator_label: str
    target_field: str
    batch_field: str
    aggregation_unit: str
    delta_definition: str
    uncertainty_definition: str
    fold_change_floor: Decimal
    de_method: str
    minimum_cells_per_target: int

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        for name, value in (
            ("matrix_view", self.matrix_view),
            ("normalization", self.normalization),
            ("comparator_label", self.comparator_label),
            ("target_field", self.target_field),
            ("batch_field", self.batch_field),
            ("aggregation_unit", self.aggregation_unit),
            ("delta_definition", self.delta_definition),
            ("uncertainty_definition", self.uncertainty_definition),
            ("de_method", self.de_method),
        ):
            validate_nonempty(value, field_name=name)
        validate_decimal(
            self.fold_change_floor,
            field_name="fold_change_floor",
            minimum=Decimal("0"),
        )
        if self.normalization == "log1p-fixed-total":
            if self.normalization_target_sum is None:
                raise ValueError(
                    "fixed-total normalization requires its exact target sum"
                )
            validate_decimal(
                self.normalization_target_sum,
                field_name="normalization_target_sum",
                minimum=Decimal("0"),
            )
            if self.normalization_target_sum == 0:
                raise ValueError("normalization target sum must be positive")
        elif self.normalization_target_sum is not None:
            raise ValueError("only fixed-total normalization may declare a target sum")
        _positive_int(
            self.minimum_cells_per_target, field_name="minimum_cells_per_target"
        )
        if self.aggregation_unit == "cell":
            raise ValueError("cell cannot be the independent aggregation unit")


@dataclass(frozen=True, slots=True)
class FeatureProvenance(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/feature-provenance'
    )

    provenance_id: str
    source_id: str
    release_cutoff_utc: str
    target_scope: str
    feature_ids_sha256: str
    transform_sha256: str
    outcome_access: OutcomeAccess
    post_cutoff: bool
    response_derived: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("provenance_id", self.provenance_id),
            ("source_id", self.source_id),
        ):
            validate_stable_id(value, field_name=name)
        parse_utc_timestamp(self.release_cutoff_utc, field_name="release_cutoff_utc")
        validate_nonempty(self.target_scope, field_name="target_scope")
        validate_sha256(self.feature_ids_sha256, field_name="feature_ids_sha256")
        validate_sha256(self.transform_sha256, field_name="transform_sha256")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("prediction features must remain outcome-blind")
        if self.post_cutoff or self.response_derived:
            raise ValueError(
                "Lane H features cannot be post-cutoff or response-derived"
            )


@dataclass(frozen=True, slots=True)
class MetricBaseline(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/metric-baseline'

    baseline_id: str
    des: Decimal
    pds: Decimal
    mae: Decimal
    construction: str

    def __post_init__(self) -> None:
        validate_stable_id(self.baseline_id, field_name="baseline_id")
        for name, value in (("des", self.des), ("pds", self.pds)):
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
            if value >= 1:
                raise ValueError(f"{name} baseline must be below one")
        validate_decimal(self.mae, field_name="mae", minimum=Decimal("0"))
        if self.mae == 0:
            raise ValueError("MAE baseline must be positive")
        validate_nonempty(self.construction, field_name="construction")


@dataclass(frozen=True, slots=True)
class MetricContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/metric-contract'

    contract_id: str
    scorer_distribution: str
    scorer_version: str
    scorer_source_sha256: str
    pdex_version: str
    metric_ids: tuple[str, ...]
    baseline: MetricBaseline
    normalization: str
    aggregation: str
    negative_clipping: bool
    missing_value_policy: str
    numerical_tolerance: Decimal
    tie_policy: str

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_stable_id(self.scorer_distribution, field_name="scorer_distribution")
        validate_semantic_version(self.scorer_version)
        validate_semantic_version(self.pdex_version)
        validate_sha256(self.scorer_source_sha256, field_name="scorer_source_sha256")
        require_sorted_unique_strings(
            self.metric_ids, field_name="metric_ids", allow_empty=False
        )
        if self.metric_ids != ("des", "mae", "pds"):
            raise ValueError("the 2025 final metric contract requires DES, MAE and PDS")
        for name, value in (
            ("normalization", self.normalization),
            ("aggregation", self.aggregation),
            ("missing_value_policy", self.missing_value_policy),
            ("tie_policy", self.tie_policy),
        ):
            validate_nonempty(value, field_name=name)
        validate_decimal(
            self.numerical_tolerance,
            field_name="numerical_tolerance",
            minimum=Decimal("0"),
        )
        if not self.negative_clipping:
            raise ValueError(
                "the official 2025 score clips negative normalized metrics"
            )


@dataclass(frozen=True, slots=True)
class ModelSelectionContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/model-selection-contract'
    )

    contract_id: str
    primary_metric_id: str
    validation_metric_contract: MetricContract
    candidate_ids: tuple[str, ...]
    required_comparator_ids: tuple[str, ...]
    ridge_alphas: tuple[Decimal, ...]
    reduced_ranks: tuple[int, ...]
    outer_target_folds: int
    repeat_seeds: tuple[int, ...]
    bootstrap_targets: int
    bootstrap_seed: int
    confidence_level: Decimal
    noninferiority_tolerance: Decimal
    minimum_scientific_improvement: Decimal
    realized_admission_repression_scale: Decimal
    tie_breaker: str
    maximum_wall_time_seconds: int

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_stable_id(self.primary_metric_id, field_name="primary_metric_id")
        if self.primary_metric_id != "official-validation-score":
            raise ValueError(
                "competition selection requires the official validation score"
            )
        if self.validation_metric_contract.contract_id != (
            "metric.virtual-cell-2025-validation-official"
        ):
            raise ValueError(
                "model selection requires the frozen validation metric contract"
            )
        require_sorted_unique_strings(self.candidate_ids, field_name="candidate_ids")
        require_sorted_unique_strings(
            self.required_comparator_ids,
            field_name="required_comparator_ids",
        )
        if not set(self.required_comparator_ids).issubset(self.candidate_ids):
            raise ValueError(
                "required comparators must be present in the candidate roster"
            )
        if (
            tuple(sorted(set(self.ridge_alphas))) != self.ridge_alphas
            or not self.ridge_alphas
        ):
            raise ValueError("ridge alphas must be sorted and unique")
        for index, alpha in enumerate(self.ridge_alphas):
            validate_decimal(
                alpha, field_name=f"ridge_alphas[{index}]", minimum=Decimal("0")
            )
        if (
            tuple(sorted(set(self.reduced_ranks))) != self.reduced_ranks
            or not self.reduced_ranks
        ):
            raise ValueError("reduced ranks must be sorted and unique")
        for rank in self.reduced_ranks:
            _positive_int(rank, field_name="reduced_ranks")
        _positive_int(self.outer_target_folds, field_name="outer_target_folds")
        if (
            tuple(sorted(set(self.repeat_seeds))) != self.repeat_seeds
            or not self.repeat_seeds
        ):
            raise ValueError("repeat seeds must be sorted and unique")
        _positive_int(self.bootstrap_targets, field_name="bootstrap_targets")
        _positive_int(self.bootstrap_seed, field_name="bootstrap_seed", allow_zero=True)
        validate_decimal(self.confidence_level, field_name="confidence_level")
        if not Decimal("0") < self.confidence_level < Decimal("1"):
            raise ValueError("confidence_level must lie strictly between zero and one")
        validate_decimal(
            self.noninferiority_tolerance,
            field_name="noninferiority_tolerance",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.minimum_scientific_improvement,
            field_name="minimum_scientific_improvement",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.realized_admission_repression_scale,
            field_name="realized_admission_repression_scale",
            minimum=Decimal("0"),
        )
        if self.realized_admission_repression_scale == 0:
            raise ValueError("realized admission repression scale must be positive")
        validate_nonempty(self.tie_breaker, field_name="tie_breaker")
        _positive_int(
            self.maximum_wall_time_seconds, field_name="maximum_wall_time_seconds"
        )


@dataclass(frozen=True, slots=True)
class PredictionCommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/prediction-commitment'
    )

    commitment_id: str
    lane_id: str
    model_id: str
    model_sha256: str
    config_sha256: str
    source_manifest_sha256: str
    split_contract_sha256: str
    metric_contract_sha256: str
    prediction_sha256: str
    prediction_schema_sha256: str
    target_roster_sha256: str
    gene_order_sha256: str
    frozen_at_utc: str
    final_outcome_parent_ids: tuple[str, ...]
    leaderboard_parent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("commitment_id", self.commitment_id),
            ("lane_id", self.lane_id),
            ("model_id", self.model_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("model_sha256", self.model_sha256),
            ("config_sha256", self.config_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("split_contract_sha256", self.split_contract_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
            ("prediction_sha256", self.prediction_sha256),
            ("prediction_schema_sha256", self.prediction_schema_sha256),
            ("target_roster_sha256", self.target_roster_sha256),
            ("gene_order_sha256", self.gene_order_sha256),
        ):
            validate_sha256(value, field_name=name)
        parse_utc_timestamp(self.frozen_at_utc, field_name="frozen_at_utc")
        require_sorted_unique_strings(
            self.final_outcome_parent_ids,
            field_name="final_outcome_parent_ids",
        )
        require_sorted_unique_strings(
            self.leaderboard_parent_ids,
            field_name="leaderboard_parent_ids",
        )
        if self.final_outcome_parent_ids or self.leaderboard_parent_ids:
            raise ValueError(
                "prediction commitments cannot descend from final outcomes/leaderboard"
            )


@dataclass(frozen=True, slots=True)
class LeaderboardEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/leaderboard-entry'
    )

    rank: int
    entry_id: str
    team_id: str
    team_name: str
    model_name: str
    score: Decimal

    def __post_init__(self) -> None:
        _positive_int(self.rank, field_name="rank")
        validate_nonempty(self.entry_id, field_name="entry_id")
        validate_nonempty(self.team_id, field_name="team_id")
        validate_nonempty(self.team_name, field_name="team_name")
        validate_nonempty(self.model_name, field_name="model_name")
        validate_decimal(self.score, field_name="score")


@dataclass(frozen=True, slots=True)
class LeaderboardSnapshot(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/leaderboard-snapshot'
    )

    snapshot_id: str
    source_url: str
    captured_at_utc: str
    total_ranked_entries: int
    summary_competitor_count: int
    entries: tuple[LeaderboardEntry, ...]
    full_precision: bool
    complete: bool
    tie_policy: str
    source_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.snapshot_id, field_name="snapshot_id")
        validate_nonempty(self.source_url, field_name="source_url")
        parse_utc_timestamp(self.captured_at_utc, field_name="captured_at_utc")
        _positive_int(self.total_ranked_entries, field_name="total_ranked_entries")
        _positive_int(
            self.summary_competitor_count, field_name="summary_competitor_count"
        )
        validate_sha256(self.source_sha256, field_name="source_sha256")
        require_sorted_unique_ids(self.entries, attribute="rank", field_name="entries")
        if tuple(value.rank for value in self.entries) != tuple(
            range(1, len(self.entries) + 1)
        ):
            raise ValueError("leaderboard entries must be contiguous from rank one")
        if any(
            left.score < right.score
            for left, right in zip(self.entries, self.entries[1:])
        ):
            raise ValueError("leaderboard scores must be nonincreasing")
        if len({value.entry_id for value in self.entries}) != len(self.entries):
            raise ValueError("leaderboard entry IDs must be unique")
        validate_nonempty(self.tie_policy, field_name="tie_policy")
        if self.complete != (len(self.entries) == self.total_ranked_entries):
            raise ValueError(
                "leaderboard completeness differs from its ranked row count"
            )


@dataclass(frozen=True, slots=True)
class PlacementAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/placement-adjudication'
    )

    adjudication_id: str
    prediction_commitment_id: str
    metric_contract_sha256: str
    leaderboard_snapshot_sha256: str
    candidate_score: Decimal
    placement_class: PlacementClass
    best_rank: int | None
    worst_rank: int | None
    field_size: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        validate_stable_id(
            self.prediction_commitment_id,
            field_name="prediction_commitment_id",
        )
        validate_sha256(
            self.metric_contract_sha256, field_name="metric_contract_sha256"
        )
        validate_sha256(
            self.leaderboard_snapshot_sha256,
            field_name="leaderboard_snapshot_sha256",
        )
        validate_decimal(self.candidate_score, field_name="candidate_score")
        _positive_int(self.field_size, field_name="field_size")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (self.best_rank is None) != (self.worst_rank is None):
            raise ValueError("placement bounds must both be present or absent")
        if self.best_rank is not None and self.worst_rank is not None:
            _positive_int(self.best_rank, field_name="best_rank")
            _positive_int(self.worst_rank, field_name="worst_rank")
            if not self.best_rank <= self.worst_rank <= self.field_size + 1:
                raise ValueError("placement bounds are inconsistent with field size")
        if (
            self.placement_class is PlacementClass.NOT_COMPARABLE
            and self.best_rank is not None
        ):
            raise ValueError("not-comparable placement cannot carry rank bounds")


@dataclass(frozen=True, slots=True)
class EmpiricalReplaySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/empirical-replay-spec'
    )

    replay_id: str
    world_kind: WorldKind
    development_materialization_ids: tuple[str, ...]
    sealed_outcome_materialization_id: str
    action_ids: tuple[str, ...]
    lookup_semantics: str
    unrepresented_physics: tuple[str, ...]
    test_outcome_access: OutcomeAccess
    prospective_physical_controller_evaluation_status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.replay_id, field_name="replay_id")
        if self.world_kind is not WorldKind.NUMERICAL_SIMULATOR:
            raise ValueError("empirical replay must declare a numerical replay world")
        require_sorted_unique_strings(
            self.development_materialization_ids,
            field_name="development_materialization_ids",
            allow_empty=False,
        )
        validate_stable_id(
            self.sealed_outcome_materialization_id,
            field_name="sealed_outcome_materialization_id",
        )
        require_sorted_unique_strings(
            self.action_ids, field_name="action_ids", allow_empty=False
        )
        validate_nonempty(self.lookup_semantics, field_name="lookup_semantics")
        _nonblank_tuple(
            self.unrepresented_physics,
            field_name="unrepresented_physics",
            sorted_set=True,
        )
        if self.test_outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("replay test outcomes must remain evaluator-sealed")
        if self.prospective_physical_controller_evaluation_status != "NOT_TESTED":
            raise ValueError("archive replay can never set prospective physical controller use")


@dataclass(frozen=True, slots=True)
class ReplayControlQuery(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/replay-control-query'
    )

    query_id: str
    prefix_observation_id: str
    prefix_observation_sha256: str
    desired_basin_id: str
    comparator_id: str
    allowed_action_ids: tuple[str, ...]
    target_thresholds: tuple[QuantityBound, ...]
    sink_thresholds: tuple[QuantityBound, ...]
    effort_thresholds: tuple[QuantityBound, ...]
    preservation_thresholds: tuple[QuantityBound, ...]
    uncertainty_thresholds: tuple[QuantityBound, ...]
    deadline_clock_id: str
    deadline_coordinate: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("query_id", self.query_id),
            ("prefix_observation_id", self.prefix_observation_id),
            ("desired_basin_id", self.desired_basin_id),
            ("comparator_id", self.comparator_id),
            ("deadline_clock_id", self.deadline_clock_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(
            self.prefix_observation_sha256, field_name="prefix_observation_sha256"
        )
        require_sorted_unique_strings(
            self.allowed_action_ids,
            field_name="allowed_action_ids",
            allow_empty=False,
        )
        for name, values in (
            ("target_thresholds", self.target_thresholds),
            ("sink_thresholds", self.sink_thresholds),
            ("effort_thresholds", self.effort_thresholds),
            ("preservation_thresholds", self.preservation_thresholds),
            ("uncertainty_thresholds", self.uncertainty_thresholds),
        ):
            require_sorted_unique_ids(values, attribute="bound_id", field_name=name)
            if len(values) != 1:
                raise ValueError(
                    f"{name} requires exactly one composite native-unit bound"
                )
        validate_decimal(
            self.deadline_coordinate,
            field_name="deadline_coordinate",
            minimum=Decimal("0"),
        )


@dataclass(frozen=True, slots=True)
class ReplayDeliveryBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/replay-delivery-binding'
    )

    binding_id: str
    query_id: str
    action_id: str | None
    hold: bool
    sealed_lookup_coordinate: str | None
    decision_sha256: str
    committed_at_utc: str
    outcome_opened_before_commitment: bool
    physical_delivery: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.query_id, field_name="query_id")
        validate_sha256(self.decision_sha256, field_name="decision_sha256")
        parse_utc_timestamp(self.committed_at_utc, field_name="committed_at_utc")
        if self.hold == (self.action_id is not None):
            raise ValueError("delivery binding requires exactly one action or hold")
        if self.action_id is not None:
            validate_stable_id(self.action_id, field_name="action_id")
            if self.sealed_lookup_coordinate is None:
                raise ValueError("action delivery requires a sealed lookup coordinate")
            validate_nonempty(
                self.sealed_lookup_coordinate,
                field_name="sealed_lookup_coordinate",
            )
        elif self.sealed_lookup_coordinate is not None:
            raise ValueError("hold cannot carry an outcome lookup coordinate")
        if self.outcome_opened_before_commitment:
            raise ValueError(
                "replay outcomes cannot be opened before delivery commitment"
            )
        if self.physical_delivery:
            raise ValueError("replay lookup is not physical H1 delivery")


@dataclass(frozen=True, slots=True)
class RetrospectiveReplayCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/physical/virtual-cell/retrospective-replay-closeout'
    )

    closeout_id: str
    replay_spec_sha256: str
    controller_compilation_sha256: str
    trace_receipt_ids: tuple[str, ...]
    evaluator_receipt_ids: tuple[str, ...]
    disposition: ReplayControllerUseDisposition
    action_count: int
    hold_count: int
    failed_delivery_count: int
    prospective_physical_controller_evaluation_status: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        validate_sha256(self.replay_spec_sha256, field_name="replay_spec_sha256")
        validate_sha256(
            self.controller_compilation_sha256,
            field_name="controller_compilation_sha256",
        )
        require_sorted_unique_strings(
            self.trace_receipt_ids,
            field_name="trace_receipt_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.evaluator_receipt_ids,
            field_name="evaluator_receipt_ids",
        )
        for name, value in (
            ("action_count", self.action_count),
            ("hold_count", self.hold_count),
            ("failed_delivery_count", self.failed_delivery_count),
        ):
            _positive_int(value, field_name=name, allow_zero=True)
        if self.action_count + self.hold_count == 0:
            raise ValueError("replay closeout requires at least one committed decision")
        if len(self.trace_receipt_ids) != self.action_count + self.hold_count:
            raise ValueError("replay closeout requires one trace receipt per decision")
        if len(self.evaluator_receipt_ids) != self.action_count:
            raise ValueError(
                "replay closeout requires one evaluator receipt per delivered action"
            )
        if self.failed_delivery_count > self.hold_count:
            raise ValueError(
                "failed replay deliveries must be counted among held outcomes"
            )
        if self.prospective_physical_controller_evaluation_status != "NOT_TESTED":
            raise ValueError("empirical replay cannot promote prospective physical controller use")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ReplayControllerUseDisposition.REPLAY_CONTROLLER_USE_VALIDATED:
            if (
                self.action_count == 0
                or self.failed_delivery_count
                or self.reason_codes
            ):
                raise ValueError(
                    "validated replay closeout has inconsistent counts or reasons"
                )
        elif self.disposition is ReplayControllerUseDisposition.REPLAY_CONTROLLER_USE_SAFE_HOLD_ONLY:
            if self.action_count or self.failed_delivery_count or not self.reason_codes:
                raise ValueError(
                    "safe-hold replay closeout has inconsistent counts or reasons"
                )
        elif self.disposition is ReplayControllerUseDisposition.REPLAY_CONTROLLER_USE_REJECTED:
            if (
                self.action_count == 0
                or self.failed_delivery_count
                or not self.reason_codes
            ):
                raise ValueError(
                    "rejected replay closeout has inconsistent counts or reasons"
                )
        elif not self.reason_codes:
            raise ValueError("unevaluable replay closeout requires reason codes")


@dataclass(frozen=True, slots=True)
class TranslationComponent(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/translation-component'
    )

    component_id: str
    disposition: TranslationDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class TaskTranslation2026(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/physical/virtual-cell/task-translation2026'
    )

    translation_id: str
    rules_snapshot_sha256: str
    source_manifest_sha256: str
    metric_contract_sha256: str
    denominator_id: str
    history_contract_id: str
    action_contract_id: str
    receiver_contract_id: str
    horizon_contract_id: str
    independent_unit_contract_id: str
    components: tuple[TranslationComponent, ...]
    eligibility_verified: bool
    terms_accepted_under_authority: bool
    submission_authority_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("translation_id", self.translation_id),
            ("denominator_id", self.denominator_id),
            ("history_contract_id", self.history_contract_id),
            ("action_contract_id", self.action_contract_id),
            ("receiver_contract_id", self.receiver_contract_id),
            ("horizon_contract_id", self.horizon_contract_id),
            ("independent_unit_contract_id", self.independent_unit_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("rules_snapshot_sha256", self.rules_snapshot_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(
            self.components, attribute="component_id", field_name="components"
        )
        if self.submission_authority_id is not None:
            validate_stable_id(
                self.submission_authority_id, field_name="submission_authority_id"
            )
        if self.terms_accepted_under_authority and self.submission_authority_id is None:
            raise ValueError("accepted terms require a bound external authority record")


@dataclass(frozen=True, slots=True)
class PredictionCompilerContract(CanonicalRecord):
    """Frozen, outcome-blind row and numerical budget for final H5AD construction."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/physical/virtual-cell/prediction-compiler-contract'
    )

    contract_id: str
    total_cell_limit: int
    control_cells: int
    minimum_cells_per_target: int
    control_reservoir_cells: int
    compiler_seed: int
    freeze_timestamp_utc: str
    mean_tolerance: Decimal
    maximum_value: Decimal
    maximum_envelope_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        for name, value in (
            ("total_cell_limit", self.total_cell_limit),
            ("control_cells", self.control_cells),
            ("minimum_cells_per_target", self.minimum_cells_per_target),
            ("control_reservoir_cells", self.control_reservoir_cells),
            ("maximum_envelope_bytes", self.maximum_envelope_bytes),
        ):
            _positive_int(value, field_name=name)
        _positive_int(self.compiler_seed, field_name="compiler_seed", allow_zero=True)
        parse_utc_timestamp(
            self.freeze_timestamp_utc, field_name="freeze_timestamp_utc"
        )
        validate_decimal(
            self.mean_tolerance,
            field_name="mean_tolerance",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.maximum_value,
            field_name="maximum_value",
            minimum=Decimal("0"),
        )
        if self.control_cells >= self.total_cell_limit:
            raise ValueError(
                "prediction control cells must leave a perturbation budget"
            )
        if self.total_cell_limit > 100_000:
            raise ValueError("2025 prediction row count exceeds the official maximum")
        if self.maximum_value <= 0 or self.maximum_value >= 15:
            raise ValueError("prediction maximum must lie strictly between zero and 15")
        if self.maximum_envelope_bytes > 3_900_000_000:
            raise ValueError(
                "prediction envelope exceeds the frozen VFAT-safe artifact bound"
            )


@dataclass(frozen=True, slots=True)
class VirtualCellPipelineConfig(CanonicalRecord):
    """One strict static config shared by VCC programme capabilities."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/virtual-cell/virtual-cell-pipeline-config'

    config_id: str
    source_manifest_sha256: str
    schema_contract_sha256s: tuple[str, ...]
    independent_unit_contract: IndependentUnitContract
    outcome_access_manifest: OutcomeAccessManifest
    response_summary: ResponseSummarySpec
    metric_contract: MetricContract
    model_selection: ModelSelectionContract
    prediction_compiler: PredictionCompilerContract
    target_feature_artifact_id: str
    target_feature_sha256: str
    leaderboard_snapshot_sha256: str
    historical_cutoff_utc: str
    deterministic_seed_ids: tuple[str, ...]
    exact_cell_eval_required: bool
    network_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(
            self.source_manifest_sha256, field_name="source_manifest_sha256"
        )
        require_sorted_unique_strings(
            self.schema_contract_sha256s,
            field_name="schema_contract_sha256s",
            allow_empty=False,
        )
        for digest in self.schema_contract_sha256s:
            validate_sha256(digest, field_name="schema_contract_sha256s")
        validate_stable_id(
            self.target_feature_artifact_id,
            field_name="target_feature_artifact_id",
        )
        validate_sha256(self.target_feature_sha256, field_name="target_feature_sha256")
        validate_sha256(
            self.leaderboard_snapshot_sha256,
            field_name="leaderboard_snapshot_sha256",
        )
        parse_utc_timestamp(
            self.historical_cutoff_utc, field_name="historical_cutoff_utc"
        )
        require_sorted_unique_strings(
            self.deterministic_seed_ids,
            field_name="deterministic_seed_ids",
            allow_empty=False,
        )
        if not self.exact_cell_eval_required:
            raise ValueError("the 2025 pipeline requires exact cell-eval conformance")
        if self.network_required:
            raise ValueError("held-source scientific execution must be network-free")


@dataclass(frozen=True, slots=True)
class ProvenanceBoundVirtualCellPipelineConfig(VirtualCellPipelineConfig):
    "Explicit target feature provenance; the retained feature record uses a separate schema."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/virtual-cell/provenance-bound-virtual-cell-pipeline-config'
    VERSION: ClassVar[str] = '1.0.0'
    target_feature_provenance_sha256: str
    target_feature_build_receipt_sha256: str

    def __post_init__(self) -> None:
        VirtualCellPipelineConfig.__post_init__(self)
        validate_sha256(self.target_feature_provenance_sha256)
        validate_sha256(self.target_feature_build_receipt_sha256)


__all__ = [
    "ActionStageAvailability",
    "EmpiricalReplaySpec",
    "FeatureProvenance",
    "IndependentUnitContract",
    "IndependentUnitStatus",
    "LeaderboardEntry",
    "LeaderboardSnapshot",
    "MatrixEncoding",
    "MetricBaseline",
    "MetricContract",
    "ModelSelectionContract",
    "NamedDecimal",
    "OutcomeAccessManifest",
    "PlacementAdjudication",
    "PlacementClass",
    "PredictionCompilerContract",
    "PredictionCommitment",
    "ReplayControlQuery",
    "ReplayDeliveryBinding",
    'RetrospectiveReplayCloseout',
    "ReplayControllerUseDisposition",
    "ResponseSummarySpec",
    "SplitCardinality",
    "SplitOutcomeAccess",
    "TaskTranslation2026",
    "TranslationComponent",
    "TranslationDisposition",
    "VirtualCellAction",
    "VirtualCellContractError",
    "VirtualCellDenominator",
    "VirtualCellEvidenceWorld",
    "VirtualCellPipelineConfig",
    'ProvenanceBoundVirtualCellPipelineConfig',
    "VirtualCellSchemaContract",
    "VirtualCellSourceManifest",
    "VirtualCellSourceObject",
    "VirtualCellSourcePart",
    "VirtualCellSplit",
]
