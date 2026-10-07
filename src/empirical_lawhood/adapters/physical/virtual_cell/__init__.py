"""current adapter for retrospective and prospective Virtual Cell work."""

from .analysis import (
    CompiledPrediction,
    LinearResponseModel,
    ModelFamily,
    TargetFeatureMatrix,
    TargetResponseMatrix,
    adjudicate_placement,
    compile_prediction_cells,
    fit_tier_l0_model,
    official_aggregate_score,
    predict_response,
)
from .contracts import EmpiricalReplaySpec, MetricContract, PredictionCommitment, TaskTranslation2026, VirtualCellDenominator, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceManifest
from .dataset import (
    H5ADStructuralInspection,
    ResponseSummaryArrays,
    SegmentedBinaryReader,
    inspect_h5ad,
    summarize_h5ad,
)
from .leaderboard import (
    adjudicate_official_final_top100,
    decode_official_final_top100,
)

__all__ = [
    "CompiledPrediction",
    "EmpiricalReplaySpec",
    "H5ADStructuralInspection",
    "LinearResponseModel",
    "MetricContract",
    "ModelFamily",
    "PredictionCommitment",
    "ResponseSummaryArrays",
    "SegmentedBinaryReader",
    "TargetFeatureMatrix",
    "TargetResponseMatrix",
    "TaskTranslation2026",
    "VirtualCellDenominator",
    'ProvenanceBoundVirtualCellPipelineConfig',
    "VirtualCellSourceManifest",
    "adjudicate_placement",
    "adjudicate_official_final_top100",
    "compile_prediction_cells",
    "decode_official_final_top100",
    "fit_tier_l0_model",
    "inspect_h5ad",
    "official_aggregate_score",
    "predict_response",
    "summarize_h5ad",
]
