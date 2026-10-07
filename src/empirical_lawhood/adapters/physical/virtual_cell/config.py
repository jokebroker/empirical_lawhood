"""Frozen 2025 Virtual Cell execution contracts and strict config bytes."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .contracts import VirtualCellSourceManifest, LeaderboardSnapshot
    from .materialization import TargetFeatureMaterializationReceipt

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess

from .contracts import FeatureProvenance, IndependentUnitContract, IndependentUnitStatus, MatrixEncoding, MetricBaseline, MetricContract, ModelSelectionContract, OutcomeAccessManifest, PredictionCompilerContract, ResponseSummarySpec, SplitOutcomeAccess, VirtualCellDenominator, VirtualCellEvidenceWorld, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSchemaContract, VirtualCellSplit


CAPABILITY_VERSION: Final = "1.0.0"
MAXIMUM_CONFIG_BYTES: Final = 4 * 1024**2
CONFIG_SCHEMA_SHA256: Final = sha256(
    ProvenanceBoundVirtualCellPipelineConfig.SCHEMA.encode()
).hexdigest()

SOURCE_CAPABILITY_KEY: Final = "source.virtual-cell-2025-bindings"
SUMMARY_CAPABILITY_KEY: Final = "transform.virtual-cell-2025-summary"
MODEL_CAPABILITY_KEY: Final = "analysis.virtual-cell-2025-tier-l0"
FALSIFIER_CAPABILITY_KEY: Final = "falsifier.virtual-cell-2025-tier-l0"
SELECTION_CAPABILITY_KEY: Final = "analysis.virtual-cell-2025-selection"
PREDICTION_CAPABILITY_KEY: Final = "observation.virtual-cell-2025-prediction"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.virtual-cell-2025-official"
REPORTER_CAPABILITY_KEY: Final = "reporter.virtual-cell-2025-closeout"
PLACEMENT_CAPABILITY_KEY: Final = "analysis.virtual-cell-2025-placement"

SOURCE_STORAGE_ROOT_ID: Final = "operator-external-root"
SOURCE_TERMS_SNAPSHOT_LOCATOR: Final = "virtual-cell-challenge/sources/support/historical-web-archive-v1/rules/20250811004400.html"
GENE_ORDER_SHA256: Final = (
    "0e3f038a579acfb96fba4f975810d6a1bf7028a3966a0807a33535bfcc08f607"
)
OFFICIAL_FINAL_TOP100_SHA256: Final = (
    "bd6ec29df32948f4a555cc701b856a1cac6100fa0623141dfa4919361e02f719"
)
OFFICIAL_FINAL_TOP100_SIZE_BYTES: Final = 75_409
OFFICIAL_FINAL_RANKED_ENTRY_COUNT: Final = 337
OFFICIAL_FINAL_SUMMARY_COMPETITOR_COUNT: Final = 338
ENSEMBL_113_GTF_SHA256: Final = (
    "62f1709b40e083ce9d4cdc64a86b5ffec2c5d5371434bb7095c74dc89079c466"
)
ENSEMBL_113_PEPTIDE_SHA256: Final = (
    "c51944dc7e44a72d7370d208a24a038d5a4672f8d2af32b5ea9df039b8e2ffd5"
)

# The revised controlling rules set the final submission deadline at 23:59 UTC
# on 2025-11-17, after the final target roster was released on 2025-11-10.
# Lane H admits only sources that were public by that deadline; final response
# values and the final leaderboard remain forbidden development inputs.
HISTORICAL_INFORMATION_CUTOFF_UTC: Final = "2025-11-17T23:59:00Z"
# Weighted median of the test-prefix per-target median UMI values, weighted by
# their advertised cell counts.  It is a prefix operand, not a response value.
PREDICTION_NORMALIZATION_TARGET_SUM: Final = Decimal("54377.5")


def virtual_cell_2025_denominator() -> VirtualCellDenominator:
    return VirtualCellDenominator(
        denominator_id="denominator.virtual-cell-2025-h1-crispri",
        material="H1 human embryonic stem cells",
        context="2025 Arc dual-guide CRISPRi endpoint Perturb-seq screen",
        perturbation_modality="single-gene dual-guide CRISPR interference",
        assay="single-cell RNA sequencing endpoint expression counts",
        comparator="non-targeting",
        batch_keys=("batch",),
        preparation_keys=(),
        processing_version="gcs-release-2025-12-16",
        source_world=VirtualCellEvidenceWorld.RETROSPECTIVE_PHYSICAL_DATASET,
    )


def virtual_cell_2025_schema_contracts() -> tuple[VirtualCellSchemaContract, ...]:
    def contract(
        contract_id: str,
        split: VirtualCellSplit,
        row_count: int,
    ) -> VirtualCellSchemaContract:
        return VirtualCellSchemaContract(
            contract_id=contract_id,
            split=split,
            row_count=row_count,
            gene_count=18_080,
            matrix_path="X",
            matrix_encoding=MatrixEncoding.CSR,
            matrix_data_dtype="float32",
            matrix_indices_dtype="int32",
            obs_fields=("batch", "guide_id", "target_gene"),
            var_index_field="_index",
            layer_ids=(),
            gene_order_sha256=GENE_ORDER_SHA256,
            structure_only=False,
        )

    return (
        contract(
            "schema.virtual-cell-2025-train-h5ad",
            VirtualCellSplit.TRAIN,
            221_273,
        ),
        contract(
            "schema.virtual-cell-2025-validation-h5ad",
            VirtualCellSplit.VALIDATION,
            98_927,
        ),
    )


def virtual_cell_2025_independent_unit_contract() -> IndependentUnitContract:
    return IndependentUnitContract(
        contract_id="units.virtual-cell-2025-batch-ceiling",
        status=IndependentUnitStatus.BATCH_ONLY_CEILING,
        physical_unit_keys=(),
        nested_view_keys=("batch", "cell", "guide_id"),
        generalization_unit="target_gene",
        uncertainty_unit="observed-batch-ceiling",
        prohibited_replication_units=("cell", "guide_id"),
    )


def virtual_cell_2025_outcome_access() -> OutcomeAccessManifest:
    prefix = ("median_umi_per_cell", "n_cells", "target_gene")
    response = ("X", "batch", "guide_id")
    return OutcomeAccessManifest(
        manifest_id="outcome-access.virtual-cell-2025",
        information_cutoff_id="cutoff.virtual-cell-2025-final-target-release",
        split_access=(
            SplitOutcomeAccess(
                split=VirtualCellSplit.TEST,
                access=OutcomeAccess.EVALUATION_SEALED,
                prefix_field_ids=prefix,
                outcome_field_ids=response,
            ),
            SplitOutcomeAccess(
                split=VirtualCellSplit.TRAIN,
                access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                prefix_field_ids=prefix,
                outcome_field_ids=response,
            ),
            SplitOutcomeAccess(
                split=VirtualCellSplit.VALIDATION,
                access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                prefix_field_ids=prefix,
                outcome_field_ids=response,
            ),
        ),
        evaluator_capability_key=EVALUATOR_CAPABILITY_KEY,
        leaderboard_parent_for_model=False,
    )


def virtual_cell_2025_response_summary() -> ResponseSummarySpec:
    return ResponseSummarySpec(
        spec_id="summary.virtual-cell-2025-log1p-fixed-total",
        matrix_view="source-X-raw-counts",
        normalization="log1p-fixed-total",
        normalization_target_sum=PREDICTION_NORMALIZATION_TARGET_SUM,
        comparator_label="non-targeting",
        target_field="target_gene",
        batch_field="batch",
        aggregation_unit="target_gene-by-observed-batch-ceiling",
        delta_definition="equal-observed-batch mean minus exact matched-batch comparator mean",
        uncertainty_definition="between-observed-batch dispersion; cells remain nested views",
        fold_change_floor=Decimal("0.000001"),
        de_method="cell-eval-0.6.6-wilcoxon-tie-corrected-bh-fdr-0.05",
        minimum_cells_per_target=32,
    )


def virtual_cell_2025_metric_contract() -> MetricContract:
    return MetricContract(
        contract_id="metric.virtual-cell-2025-final-official",
        scorer_distribution="cell-eval",
        scorer_version="0.6.6",
        scorer_source_sha256=(
            "799adf3a40103fe74457887f207c5552aea9c9d81d06295c69926664a19bb011"
        ),
        pdex_version="0.1.26",
        metric_ids=("des", "mae", "pds"),
        baseline=MetricBaseline(
            baseline_id="baseline.virtual-cell-2025-final-cell-mean",
            des=Decimal("0.09505113849605415"),
            pds=Decimal("0.5082000000000001"),
            mae=Decimal("0.024957533199340104"),
            construction="official final cell-mean baseline recovered from final leaderboard",
        ),
        normalization=("DES/PDS=(user-baseline)/(1-baseline); MAE=1-user/baseline"),
        aggregation="arithmetic mean of three nonnegative normalized metrics; fraction scale",
        negative_clipping=True,
        missing_value_policy="cell-eval score_agg_metrics replaces NaN normalized values with zero",
        numerical_tolerance=Decimal("0.000000000000001"),
        tie_policy=(
            "published API order is ordinal; an equal candidate score remains a rank interval"
        ),
    )


def virtual_cell_2025_validation_metric_contract() -> MetricContract:
    """Exact live-validation score used for frozen candidate-family selection."""

    return MetricContract(
        contract_id="metric.virtual-cell-2025-validation-official",
        scorer_distribution="cell-eval",
        scorer_version="0.6.6",
        scorer_source_sha256=(
            "799adf3a40103fe74457887f207c5552aea9c9d81d06295c69926664a19bb011"
        ),
        pdex_version="0.1.26",
        metric_ids=("des", "mae", "pds"),
        baseline=MetricBaseline(
            baseline_id="baseline.virtual-cell-2025-validation-cell-mean",
            des=Decimal("0.10569256482956448"),
            pds=Decimal("0.514"),
            mae=Decimal("0.026577684134244918"),
            construction=(
                "official validation cell-mean baseline; DES/PDS reproduce the "
                "full-precision public validation leaderboard and MAE is the exact "
                "cell-eval 0.6.6 cell-mean baseline metric"
            ),
        ),
        normalization=("DES/PDS=(user-baseline)/(1-baseline); MAE=1-user/baseline"),
        aggregation="arithmetic mean of three nonnegative normalized metrics; fraction scale",
        negative_clipping=True,
        missing_value_policy="cell-eval score_agg_metrics replaces NaN normalized values with zero",
        numerical_tolerance=Decimal("0.000000000000001"),
        tie_policy="one-standard-error rule, then parameter count, then candidate ID",
    )


def virtual_cell_2025_model_selection() -> ModelSelectionContract:
    return ModelSelectionContract(
        contract_id="selection.virtual-cell-2025-tier-l0",
        primary_metric_id="official-validation-score",
        validation_metric_contract=virtual_cell_2025_validation_metric_contract(),
        candidate_ids=(
            "baseline-no-change",
            "baseline-weighted-common-response",
            "feature-ridge-response",
            "feature-ridge-response-with-reduced-rank",
            "receiver-admission-conditioned-reduced-rank-ridge-response",
        ),
        required_comparator_ids=(
            "baseline-no-change",
            "baseline-weighted-common-response",
            "feature-ridge-response",
            "feature-ridge-response-with-reduced-rank",
        ),
        ridge_alphas=(
            Decimal("0.01"),
            Decimal("0.1"),
            Decimal("1"),
            Decimal("10"),
            Decimal("100"),
        ),
        reduced_ranks=(4, 8, 16, 32),
        outer_target_folds=5,
        repeat_seeds=(11, 23, 37, 53, 71),
        bootstrap_targets=2_000,
        bootstrap_seed=20_251_103,
        confidence_level=Decimal("0.95"),
        noninferiority_tolerance=Decimal("0.005"),
        minimum_scientific_improvement=Decimal("0.01"),
        realized_admission_repression_scale=Decimal("1"),
        tie_breaker="one-standard-error then smallest learned parameter count then candidate ID",
        maximum_wall_time_seconds=8 * 60 * 60,
    )


def virtual_cell_2025_feature_provenance(
    *, build_receipt_sha256: str
) -> FeatureProvenance:
    return FeatureProvenance(
        provenance_id="features.virtual-cell-2025-ensembl-113",
        source_id="source.ensembl-113-human-gtf-peptide",
        release_cutoff_utc="2024-10-18T00:00:00Z",
        target_scope="sorted union of the 150 training, 50 validation and 100 final targets",
        feature_ids_sha256=(
            "80184b34e0c3f21c0f5369240d550cc314d41c00a1abfe3c78e8c8ee1bd40b86"
        ),
        transform_sha256=build_receipt_sha256,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        post_cutoff=False,
        response_derived=False,
    )


def virtual_cell_2025_prediction_compiler() -> PredictionCompilerContract:
    return PredictionCompilerContract(
        contract_id="compiler.virtual-cell-2025-dense-h5ad",
        total_cell_limit=20_000,
        control_cells=400,
        minimum_cells_per_target=128,
        control_reservoir_cells=1_024,
        compiler_seed=20_251_027,
        freeze_timestamp_utc="2026-08-08T00:00:00Z",
        mean_tolerance=Decimal("0.000005"),
        maximum_value=Decimal("14.999"),
        maximum_envelope_bytes=2_000_000_000,
    )


def build_virtual_cell_2025_pipeline_config(
    *,
    source_manifest: "VirtualCellSourceManifest",
    feature_receipt: "TargetFeatureMaterializationReceipt",
    leaderboard: "LeaderboardSnapshot",
) -> ProvenanceBoundVirtualCellPipelineConfig:
    schemas = virtual_cell_2025_schema_contracts()
    return ProvenanceBoundVirtualCellPipelineConfig(
        config_id="config.virtual-cell-2025-tier-l0",
        source_manifest_sha256=source_manifest.fingerprint(),
        schema_contract_sha256s=tuple(sorted(value.fingerprint() for value in schemas)),
        independent_unit_contract=virtual_cell_2025_independent_unit_contract(),
        outcome_access_manifest=virtual_cell_2025_outcome_access(),
        response_summary=virtual_cell_2025_response_summary(),
        metric_contract=virtual_cell_2025_metric_contract(),
        model_selection=virtual_cell_2025_model_selection(),
        prediction_compiler=virtual_cell_2025_prediction_compiler(),
        target_feature_artifact_id="artifact.virtual-cell-2025-ensembl-113-features",
        target_feature_sha256=feature_receipt.arrow_sha256,
        target_feature_provenance_sha256=feature_receipt.feature_provenance_sha256,
        target_feature_build_receipt_sha256=feature_receipt.build_receipt_sha256,
        leaderboard_snapshot_sha256=leaderboard.fingerprint(),
        historical_cutoff_utc=HISTORICAL_INFORMATION_CUTOFF_UTC,
        deterministic_seed_ids=(
            "seed.bootstrap-2025-20251103",
            "seed.compiler-2025-20251027",
            "seed.folds-11-23-37-53-71",
            "seed.permutation-2025-20251027",
        ),
        exact_cell_eval_required=True,
        network_required=False,
    )


def decode_pipeline_config(payload: bytes) -> ProvenanceBoundVirtualCellPipelineConfig:
    return decode_canonical_bytes(
        payload,
        ProvenanceBoundVirtualCellPipelineConfig,
        maximum_bytes=MAXIMUM_CONFIG_BYTES,
    )


__all__ = [
    "CAPABILITY_VERSION",
    "CONFIG_SCHEMA_SHA256",
    "ENSEMBL_113_GTF_SHA256",
    "ENSEMBL_113_PEPTIDE_SHA256",
    "EVALUATOR_CAPABILITY_KEY",
    "FALSIFIER_CAPABILITY_KEY",
    "GENE_ORDER_SHA256",
    "HISTORICAL_INFORMATION_CUTOFF_UTC",
    "MAXIMUM_CONFIG_BYTES",
    "MODEL_CAPABILITY_KEY",
    "OFFICIAL_FINAL_RANKED_ENTRY_COUNT",
    "OFFICIAL_FINAL_SUMMARY_COMPETITOR_COUNT",
    "OFFICIAL_FINAL_TOP100_SHA256",
    "OFFICIAL_FINAL_TOP100_SIZE_BYTES",
    "PLACEMENT_CAPABILITY_KEY",
    "PREDICTION_CAPABILITY_KEY",
    "PREDICTION_NORMALIZATION_TARGET_SUM",
    "REPORTER_CAPABILITY_KEY",
    "SELECTION_CAPABILITY_KEY",
    "SOURCE_CAPABILITY_KEY",
    "SOURCE_STORAGE_ROOT_ID",
    "SOURCE_TERMS_SNAPSHOT_LOCATOR",
    "SUMMARY_CAPABILITY_KEY",
    "build_virtual_cell_2025_pipeline_config",
    "decode_pipeline_config",
    "virtual_cell_2025_denominator",
    "virtual_cell_2025_feature_provenance",
    "virtual_cell_2025_independent_unit_contract",
    "virtual_cell_2025_metric_contract",
    "virtual_cell_2025_model_selection",
    "virtual_cell_2025_validation_metric_contract",
    "virtual_cell_2025_outcome_access",
    "virtual_cell_2025_prediction_compiler",
    "virtual_cell_2025_response_summary",
    "virtual_cell_2025_schema_contracts",
]
