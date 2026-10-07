"""Compact scientific lineage records emitted by Virtual Cell capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .analysis import FalsifierResult, ModelFitRecord, OfficialAggregateScore, TournamentRecord
from .contracts import PredictionCommitment, VirtualCellSplit


RESPONSE_SUMMARY_TABLE_SCHEMA = 'empirical-lawhood/physical/virtual-cell/response-summary-table'
TARGET_FEATURE_TABLE_SCHEMA = 'empirical-lawhood/physical/virtual-cell/target-feature-table'
CONTROL_RESERVOIR_TABLE_SCHEMA = 'empirical-lawhood/physical/virtual-cell/control-reservoir-table'
PREDICTED_MEAN_TABLE_SCHEMA = 'empirical-lawhood/physical/virtual-cell/predicted-mean-table'
OFFICIAL_METRICS_TABLE_SCHEMA = 'empirical-lawhood/physical/virtual-cell/official-metrics-table'
MODEL_SAFETENSORS_SCHEMA = 'empirical-lawhood/physical/virtual-cell/linear-model-safetensors'
PREDICTION_H5AD_ENVELOPE_SCHEMA = 'empirical-lawhood/physical/virtual-cell/prediction-h5ad-envelope'
TEST_ROSTER_TEXT_SCHEMA = 'empirical-lawhood/physical/virtual-cell/test-roster-csv'


def _registry_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedSourceRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/prepared-source-record'

    record_id: str
    source_manifest_sha256: str
    schema_contract_sha256s: tuple[str, ...]
    independent_unit_contract_sha256: str
    outcome_access_manifest_sha256: str
    development_object_ids: tuple[str, ...]
    sealed_evaluation_object_ids: tuple[str, ...]
    test_content_read: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        for name, value in (
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("independent_unit_contract_sha256", self.independent_unit_contract_sha256),
            ("outcome_access_manifest_sha256", self.outcome_access_manifest_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_strings(
            self.schema_contract_sha256s,
            field_name="schema_contract_sha256s",
            allow_empty=False,
        )
        for value in self.schema_contract_sha256s:
            validate_sha256(value, field_name="schema_contract_sha256s")
        require_sorted_unique_strings(
            self.development_object_ids,
            field_name="development_object_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.sealed_evaluation_object_ids,
            field_name="sealed_evaluation_object_ids",
            allow_empty=False,
        )
        if set(self.development_object_ids) & set(self.sealed_evaluation_object_ids):
            raise ValueError("development and sealed source object rosters must be disjoint")
        if self.test_content_read:
            raise ValueError("source preparation cannot read protected test content")


@dataclass(frozen=True, slots=True)
class ResponseSummaryRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/response-summary-record'

    record_id: str
    split: VirtualCellSplit
    source_object_sha256: str
    summary_spec_sha256: str
    table_artifact_id: str
    table_sha256: str
    group_count: int
    target_count: int
    batch_count: int
    gene_count: int
    gene_ids: tuple[str, ...]
    gene_order_sha256: str
    normalization: str
    normalization_target_sum: Decimal | None
    comparator_label: str
    aggregation_unit: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        validate_stable_id(self.table_artifact_id, field_name="table_artifact_id")
        validate_sha256(self.source_object_sha256, field_name="source_object_sha256")
        validate_sha256(self.summary_spec_sha256, field_name="summary_spec_sha256")
        validate_sha256(self.table_sha256, field_name="table_sha256")
        validate_sha256(self.gene_order_sha256, field_name="gene_order_sha256")
        for name, value in (
            ("group_count", self.group_count),
            ("target_count", self.target_count),
            ("batch_count", self.batch_count),
            ("gene_count", self.gene_count),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.split not in {VirtualCellSplit.TRAIN, VirtualCellSplit.VALIDATION}:
            raise ValueError("only development-visible splits may have response summaries")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("response summaries must be development-visible")
        if self.normalization != "log1p-fixed-total":
            raise ValueError("the frozen model summary requires fixed-total log1p")
        if self.normalization_target_sum is None:
            raise ValueError("the frozen model summary lacks its normalization target")
        validate_decimal(
            self.normalization_target_sum,
            field_name="normalization_target_sum",
            minimum=Decimal("0"),
        )
        if self.normalization_target_sum == 0:
            raise ValueError("the frozen model normalization target must be positive")
        if self.aggregation_unit != "target_gene-by-observed-batch-ceiling":
            raise ValueError("response summary cannot claim cell-level replication")
        require_sorted_unique_strings(
            tuple(sorted(self.gene_ids)),
            field_name="gene_ids",
            allow_empty=False,
        )
        if len(self.gene_ids) != self.gene_count:
            raise ValueError("response-summary gene registry has the wrong cardinality")
        if _registry_sha256(self.gene_ids) != self.gene_order_sha256:
            raise ValueError("response-summary gene registry differs from its identity")


@dataclass(frozen=True, slots=True)
class ModelArtifactBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/model-artifact-binding'

    model_id: str
    artifact_id: str
    artifact_sha256: str
    fit: ModelFitRecord

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        validate_stable_id(self.artifact_id, field_name="artifact_id")
        validate_sha256(self.artifact_sha256, field_name="artifact_sha256")
        if self.fit.model_id != self.model_id:
            raise ValueError("model artifact and fit record identify different models")


@dataclass(frozen=True, slots=True)
class CandidateValidationMetricsBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/candidate-validation-metrics-binding'

    candidate_id: str
    artifact_id: str
    artifact_sha256: str
    target_ids_sha256: str
    target_count: int
    metric_contract_sha256: str
    cell_eval_version: str
    pdex_version: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.artifact_id, field_name="artifact_id")
        for name, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("target_ids_sha256", self.target_ids_sha256),
            ("metric_contract_sha256", self.metric_contract_sha256),
        ):
            validate_sha256(value, field_name=name)
        if isinstance(self.target_count, bool) or self.target_count <= 0:
            raise ValueError("candidate validation target count must be positive")
        validate_nonempty(self.cell_eval_version, field_name="cell_eval_version")
        validate_nonempty(self.pdex_version, field_name="pdex_version")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("candidate validation metrics must be development-visible")


@dataclass(frozen=True, slots=True)
class ModelDevelopmentRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/model-development-record'

    record_id: str
    training_summary_sha256: str
    validation_summary_sha256: str
    feature_provenance_sha256: str
    models: tuple[ModelArtifactBinding, ...]
    validation_metrics: tuple[CandidateValidationMetricsBinding, ...]
    tournament: TournamentRecord
    training_target_ids_sha256: str
    validation_target_ids_sha256: str
    final_outcome_parent_ids: tuple[str, ...]
    leaderboard_parent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        for name, value in (
            ("training_summary_sha256", self.training_summary_sha256),
            ("validation_summary_sha256", self.validation_summary_sha256),
            ("feature_provenance_sha256", self.feature_provenance_sha256),
            ("training_target_ids_sha256", self.training_target_ids_sha256),
            ("validation_target_ids_sha256", self.validation_target_ids_sha256),
        ):
            validate_sha256(value, field_name=name)
        require_sorted_unique_ids(self.models, attribute="model_id", field_name="models")
        if not self.models:
            raise ValueError("development record requires fitted models")
        if not set(entry.candidate_id for entry in self.tournament.entries).issubset(
            value.model_id for value in self.models
        ):
            raise ValueError("development tournament refers to an absent model")
        require_sorted_unique_ids(
            self.validation_metrics,
            attribute="candidate_id",
            field_name="validation_metrics",
        )
        if tuple(value.candidate_id for value in self.validation_metrics) != tuple(
            entry.candidate_id for entry in self.tournament.entries
        ):
            raise ValueError("validation metrics do not cover the exact tournament roster")
        require_sorted_unique_strings(
            self.final_outcome_parent_ids,
            field_name="final_outcome_parent_ids",
        )
        require_sorted_unique_strings(
            self.leaderboard_parent_ids,
            field_name="leaderboard_parent_ids",
        )
        if self.final_outcome_parent_ids or self.leaderboard_parent_ids:
            raise ValueError("development cannot descend from final outcomes or leaderboard")


@dataclass(frozen=True, slots=True)
class FalsifierPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/falsifier-panel'

    panel_id: str
    results: tuple[FalsifierResult, ...]
    required_falsifier_ids: tuple[str, ...]
    all_required_pass: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        require_sorted_unique_ids(self.results, attribute="falsifier_id", field_name="results")
        require_sorted_unique_strings(
            self.required_falsifier_ids,
            field_name="required_falsifier_ids",
            allow_empty=False,
        )
        by_id = {value.falsifier_id: value for value in self.results}
        if not set(self.required_falsifier_ids).issubset(by_id):
            raise ValueError("falsifier panel lacks a required result")
        expected = all(by_id[value].passed for value in self.required_falsifier_ids)
        if self.all_required_pass != expected:
            raise ValueError("falsifier panel status is not derived")


@dataclass(frozen=True, slots=True)
class ModelSelectionRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/model-selection-record'

    record_id: str
    tournament: TournamentRecord
    falsifier_panel_sha256: str
    selected_model_artifact_id: str
    selected_model_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        validate_sha256(self.falsifier_panel_sha256, field_name="falsifier_panel_sha256")
        validate_stable_id(
            self.selected_model_artifact_id,
            field_name="selected_model_artifact_id",
        )
        validate_sha256(self.selected_model_sha256, field_name="selected_model_sha256")


@dataclass(frozen=True, slots=True)
class PredictionFreezeRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/prediction-freeze-record'

    record_id: str
    commitment: PredictionCommitment
    predicted_mean_table_artifact_id: str
    predicted_mean_table_sha256: str
    h5ad_envelope_artifact_id: str
    h5ad_envelope_sha256: str
    inner_h5ad_sha256: str
    compiler_seed: int
    compiler_mean_tolerance: Decimal
    test_outcome_read: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        for name, value in (
            ("predicted_mean_table_artifact_id", self.predicted_mean_table_artifact_id),
            ("h5ad_envelope_artifact_id", self.h5ad_envelope_artifact_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("predicted_mean_table_sha256", self.predicted_mean_table_sha256),
            ("h5ad_envelope_sha256", self.h5ad_envelope_sha256),
            ("inner_h5ad_sha256", self.inner_h5ad_sha256),
        ):
            validate_sha256(value, field_name=name)
        if isinstance(self.compiler_seed, bool) or self.compiler_seed < 0:
            raise ValueError("compiler seed must be a nonnegative integer")
        validate_decimal(
            self.compiler_mean_tolerance,
            field_name="compiler_mean_tolerance",
            minimum=Decimal("0"),
        )
        if self.test_outcome_read:
            raise ValueError("prediction freeze cannot read test outcomes")
        if self.commitment.prediction_sha256 != self.inner_h5ad_sha256:
            raise ValueError("prediction commitment differs from the inner H5AD")


@dataclass(frozen=True, slots=True)
class VirtualCellDevelopmentCloseout(CanonicalRecord):
    """Outcome-blind handoff from development to replay/final evaluation authoring."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-development-closeout'

    closeout_id: str
    model_development_sha256: str
    falsifier_panel_sha256: str
    model_selection_sha256: str
    prediction_freeze_sha256: str
    selected_model_artifact_id: str
    competition_disposition: str
    receiver_admission_disposition: str
    replay_authoring_disposition: str
    test_outcome_read: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        validate_stable_id(
            self.selected_model_artifact_id,
            field_name="selected_model_artifact_id",
        )
        for name, value in (
            ("model_development_sha256", self.model_development_sha256),
            ("falsifier_panel_sha256", self.falsifier_panel_sha256),
            ("model_selection_sha256", self.model_selection_sha256),
            ("prediction_freeze_sha256", self.prediction_freeze_sha256),
        ):
            validate_sha256(value, field_name=name)
        for name, value in (
            ("competition_disposition", self.competition_disposition),
            ("receiver_admission_disposition", self.receiver_admission_disposition),
            ("replay_authoring_disposition", self.replay_authoring_disposition),
        ):
            validate_nonempty(value, field_name=name)
        if self.test_outcome_read:
            raise ValueError("development closeout cannot read the sealed test outcome")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class OfficialEvaluationRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/official-evaluation-record'

    evaluation_id: str
    prediction_commitment_sha256: str
    scorer_version: str
    pdex_version: str
    score: OfficialAggregateScore
    per_target_metrics_artifact_id: str
    per_target_metrics_sha256: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_sha256(
            self.prediction_commitment_sha256,
            field_name="prediction_commitment_sha256",
        )
        validate_stable_id(self.per_target_metrics_artifact_id, field_name="artifact_id")
        validate_sha256(self.per_target_metrics_sha256, field_name="per_target_metrics_sha256")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("official evaluation must remain evaluator-scoped")


@dataclass(frozen=True, slots=True)
class VirtualCellRunCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/virtual-cell-run-closeout'

    closeout_id: str
    prediction_commitment_sha256: str
    official_evaluation_sha256: str
    placement_adjudication_sha256: str
    operational_status: str
    competition_disposition: str
    placement_disposition: str
    replay_controller_evaluation_disposition: str
    prospective_physical_controller_evaluation_status: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        validate_sha256(
            self.prediction_commitment_sha256,
            field_name="prediction_commitment_sha256",
        )
        validate_sha256(
            self.official_evaluation_sha256,
            field_name="official_evaluation_sha256",
        )
        validate_sha256(
            self.placement_adjudication_sha256,
            field_name="placement_adjudication_sha256",
        )
        for name, value in (
            ("operational_status", self.operational_status),
            ("competition_disposition", self.competition_disposition),
            ("placement_disposition", self.placement_disposition),
            ("replay_controller_evaluation_disposition", self.replay_controller_evaluation_disposition),
            ("prospective_physical_controller_evaluation_status", self.prospective_physical_controller_evaluation_status),
        ):
            if not value:
                raise ValueError(f"{name} must be nonempty")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.prospective_physical_controller_evaluation_status != "NOT_TESTED":
            raise ValueError("retrospective VCC execution cannot promote physical controller use")


__all__ = [
    "CONTROL_RESERVOIR_TABLE_SCHEMA",
    "CandidateValidationMetricsBinding",
    "FalsifierPanel",
    "MODEL_SAFETENSORS_SCHEMA",
    "ModelArtifactBinding",
    "ModelDevelopmentRecord",
    "ModelSelectionRecord",
    "OfficialEvaluationRecord",
    "OFFICIAL_METRICS_TABLE_SCHEMA",
    "PREDICTED_MEAN_TABLE_SCHEMA",
    "PREDICTION_H5AD_ENVELOPE_SCHEMA",
    "PreparedSourceRecord",
    "PredictionFreezeRecord",
    "RESPONSE_SUMMARY_TABLE_SCHEMA",
    "ResponseSummaryRecord",
    "TARGET_FEATURE_TABLE_SCHEMA",
    "TEST_ROSTER_TEXT_SCHEMA",
    "VirtualCellDevelopmentCloseout",
    "VirtualCellRunCloseout",
]
