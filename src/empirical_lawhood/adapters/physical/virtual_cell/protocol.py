"""Frozen Tier-L0 protocol and exact candidate scientific graph."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)

from .config import (
    CAPABILITY_VERSION,
    CONFIG_SCHEMA_SHA256,
    EVALUATOR_CAPABILITY_KEY,
    FALSIFIER_CAPABILITY_KEY,
    MAXIMUM_CONFIG_BYTES,
    MODEL_CAPABILITY_KEY,
    PLACEMENT_CAPABILITY_KEY,
    PREDICTION_CAPABILITY_KEY,
    REPORTER_CAPABILITY_KEY,
    SELECTION_CAPABILITY_KEY,
    SOURCE_CAPABILITY_KEY,
    SUMMARY_CAPABILITY_KEY,
)
from .contracts import LeaderboardSnapshot, PlacementAdjudication, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceManifest, VirtualCellSourceObject
from .records import (
    CONTROL_RESERVOIR_TABLE_SCHEMA,
    FalsifierPanel,
    MODEL_SAFETENSORS_SCHEMA,
    ModelDevelopmentRecord,
    ModelSelectionRecord,
    OFFICIAL_METRICS_TABLE_SCHEMA,
    OfficialEvaluationRecord,
    PREDICTED_MEAN_TABLE_SCHEMA,
    PREDICTION_H5AD_ENVELOPE_SCHEMA,
    PreparedSourceRecord,
    PredictionFreezeRecord,
    RESPONSE_SUMMARY_TABLE_SCHEMA,
    ResponseSummaryRecord,
    TARGET_FEATURE_TABLE_SCHEMA,
    TEST_ROSTER_TEXT_SCHEMA,
    VirtualCellDevelopmentCloseout,
    VirtualCellRunCloseout,
)


CONFIG_ARTIFACT_ID = "config-artifact.virtual-cell-2025-tier-l0"
SOURCE_MANIFEST_INPUT_ID = "source-manifest.virtual-cell-2025"
TARGET_FEATURE_INPUT_ID = "features.virtual-cell-2025-ensembl-113"
TEST_ROSTER_INPUT_ID = "test-roster.virtual-cell-2025"
SEALED_TEST_BINDING_INPUT_ID = "sealed-test-binding.virtual-cell-2025"
LEADERBOARD_SNAPSHOT_INPUT_ID = (
    "leaderboard-snapshot.virtual-cell-2025-official-final-top100"
)

PREPARE_STEP = "vcc-prepare-source"
SUMMARY_STEP = "vcc-summarize-development"
DEVELOP_STEP = "vcc-develop-tier-l0"
FALSIFY_STEP = "vcc-falsify-tier-l0"
SELECT_STEP = "vcc-qualify-selection"
FREEZE_STEP = "vcc-freeze-prediction"
DEVELOPMENT_REPORT_STEP = "vcc-report-development"
EVALUATE_STEP = "vcc-evaluate-official"
PLACEMENT_STEP = "vcc-adjudicate-placement"
REPORT_STEP = "vcc-report-closeout"


def virtual_cell_config_ref(config: ProvenanceBoundVirtualCellPipelineConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=config.config_id,
        config_schema=ProvenanceBoundVirtualCellPipelineConfig.SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        content_sha256=config.fingerprint(),
        artifact_id=CONFIG_ARTIFACT_ID,
    )


def _output(
    output_id: str,
    schema: str,
    profile: ArtifactProfile,
    media_type: str,
    suffix: str,
) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=profile,
        media_type=media_type,
        filename_suffix=suffix,
    )


def _json(output_id: str, schema: str) -> OutputTemplate:
    return _output(
        output_id, schema, ArtifactProfile.CANONICAL_JSON, "application/json", ".json"
    )


def _arrow(output_id: str, schema: str) -> OutputTemplate:
    return _output(
        output_id,
        schema,
        ArtifactProfile.ARROW_IPC,
        "application/vnd.apache.arrow.file",
        ".arrow",
    )


def _model(output_id: str) -> OutputTemplate:
    return _output(
        output_id,
        MODEL_SAFETENSORS_SCHEMA,
        ArtifactProfile.SAFETENSORS,
        "application/x-safetensors",
        ".safetensors",
    )


def _step(
    *,
    registry: CapabilityRegistry,
    config: ProvenanceBoundVirtualCellPipelineConfig,
    step_id: str,
    stage: ScientificStage,
    capability_key: str,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
    obligations: tuple[str, ...],
    required_permissions: tuple[CapabilityPermission, ...] | None = None,
) -> ProtocolStepTemplate:
    manifest = registry.resolve(capability_key, CAPABILITY_VERSION)
    return ProtocolStepTemplate(
        step_id=step_id,
        stage=stage,
        capability_key=capability_key,
        capability_version=CAPABILITY_VERSION,
        config=virtual_cell_config_ref(config),
        dependency_step_ids=tuple(sorted(dependencies)),
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=(
            manifest.permissions
            if required_permissions is None
            else required_permissions
        ),
        requested_outcome_access=access,
        visibility_ceiling=visibility,
        resource_budget=manifest.resource_ceiling,
        resource_lock_ids=(f"resource.{step_id}",),
        barrier=barrier,
        maximum_attempts=1,
        obligation_ids=tuple(sorted(obligations)),
    )


def virtual_cell_protocol(
    *,
    registry: CapabilityRegistry,
    config: ProvenanceBoundVirtualCellPipelineConfig,
) -> ProtocolTemplate:
    steps = (
        _step(
            registry=registry,
            config=config,
            step_id=PREPARE_STEP,
            stage=ScientificStage.PREPARE,
            capability_key=SOURCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=(_json("prepared-source", PreparedSourceRecord.SCHEMA),),
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=("vcc-source-custody-and-outcome-separation",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=SUMMARY_STEP,
            stage=ScientificStage.TRANSFORM,
            capability_key=SUMMARY_CAPABILITY_KEY,
            dependencies=(PREPARE_STEP,),
            outputs=(
                _arrow("control-reservoir", CONTROL_RESERVOIR_TABLE_SCHEMA),
                _json("train-summary-record", ResponseSummaryRecord.SCHEMA),
                _arrow("train-summary-table", RESPONSE_SUMMARY_TABLE_SCHEMA),
                _json("validation-summary-record", ResponseSummaryRecord.SCHEMA),
                _arrow("validation-summary-table", RESPONSE_SUMMARY_TABLE_SCHEMA),
            ),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=("vcc-batch-ceiling-summary-and-control-reservoir",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=DEVELOP_STEP,
            stage=ScientificStage.DEVELOP,
            capability_key=MODEL_CAPABILITY_KEY,
            dependencies=(PREPARE_STEP, SUMMARY_STEP),
            outputs=(
                _model("model-baseline-no-change"),
                _model("model-baseline-weighted-common-response"),
                _model("model-feature-ridge-response"),
                _model("model-feature-ridge-response-with-reduced-rank"),
                _model("model-receiver-admission-conditioned-reduced-rank-ridge-response"),
                _model("model-refit-baseline-no-change"),
                _model("model-refit-baseline-weighted-common-response"),
                _model("model-refit-feature-ridge-response"),
                _model("model-refit-feature-ridge-response-with-reduced-rank"),
                _model("model-refit-receiver-admission-conditioned-reduced-rank-ridge-response"),
                _arrow("validation-metrics-baseline-no-change", OFFICIAL_METRICS_TABLE_SCHEMA),
                _arrow("validation-metrics-baseline-weighted-common-response", OFFICIAL_METRICS_TABLE_SCHEMA),
                _arrow("validation-metrics-feature-ridge-response", OFFICIAL_METRICS_TABLE_SCHEMA),
                _arrow("validation-metrics-feature-ridge-response-with-reduced-rank", OFFICIAL_METRICS_TABLE_SCHEMA),
                _arrow("validation-metrics-receiver-admission-conditioned-reduced-rank-ridge-response", OFFICIAL_METRICS_TABLE_SCHEMA),
                _json("model-development", ModelDevelopmentRecord.SCHEMA),
            ),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=(
                "vcc-grouped-nested-target-cv",
                "vcc-exact-official-validation-score",
                "vcc-training-only-expected-admission",
                "vcc-validation-tournament-inputs",
            ),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=FALSIFY_STEP,
            stage=ScientificStage.FALSIFY,
            capability_key=FALSIFIER_CAPABILITY_KEY,
            dependencies=(DEVELOP_STEP, SUMMARY_STEP),
            outputs=(_json("falsifier-panel", FalsifierPanel.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=("vcc-causal-identity-and-robustness-falsifiers",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=SELECT_STEP,
            stage=ScientificStage.QUALIFY,
            capability_key=SELECTION_CAPABILITY_KEY,
            dependencies=(DEVELOP_STEP, FALSIFY_STEP),
            outputs=(_json("model-selection", ModelSelectionRecord.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=("vcc-frozen-validation-only-selection",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=FREEZE_STEP,
            stage=ScientificStage.FREEZE,
            capability_key=PREDICTION_CAPABILITY_KEY,
            dependencies=(DEVELOP_STEP, SELECT_STEP, SUMMARY_STEP),
            outputs=(
                _output(
                    "prediction-envelope",
                    PREDICTION_H5AD_ENVELOPE_SCHEMA,
                    ArtifactProfile.AUDITED_HDF5,
                    "application/x-hdf5",
                    ".h5",
                ),
                _json("prediction-freeze", PredictionFreezeRecord.SCHEMA),
                _arrow("predicted-means", PREDICTED_MEAN_TABLE_SCHEMA),
            ),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.FREEZE,
            obligations=("vcc-outcome-blind-prediction-commitment",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=EVALUATE_STEP,
            stage=ScientificStage.EVALUATE,
            capability_key=EVALUATOR_CAPABILITY_KEY,
            dependencies=(FREEZE_STEP,),
            outputs=(
                _json("official-evaluation", OfficialEvaluationRecord.SCHEMA),
                _arrow("official-metrics", OFFICIAL_METRICS_TABLE_SCHEMA),
                _json("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
            ),
            access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
            obligations=("vcc-evaluator-only-official-score",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=PLACEMENT_STEP,
            stage=ScientificStage.REPORT,
            capability_key=PLACEMENT_CAPABILITY_KEY,
            dependencies=(EVALUATE_STEP, FREEZE_STEP),
            outputs=(_json("placement-adjudication", PlacementAdjudication.SCHEMA),),
            access=OutcomeAccess.EVALUATION_REVEALED,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
            obligations=("vcc-post-score-partial-field-placement",),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=REPORT_STEP,
            stage=ScientificStage.REPORT,
            capability_key=REPORTER_CAPABILITY_KEY,
            dependencies=(EVALUATE_STEP, FREEZE_STEP, PLACEMENT_STEP),
            outputs=(_json("closeout", VirtualCellRunCloseout.SCHEMA),),
            access=OutcomeAccess.EVALUATION_REVEALED,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
            obligations=("vcc-competition-closeout-with-separate-placement-and-prospective-controller-evaluation",),
            required_permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
        ),
    )
    return ProtocolTemplate(
        template_id="protocol.virtual-cell-2025-tier-l0",
        template_version=CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def virtual_cell_development_protocol(
    *,
    registry: CapabilityRegistry,
    config: ProvenanceBoundVirtualCellPipelineConfig,
) -> ProtocolTemplate:
    """Development/freeze act that cannot request or consume final-test reveal."""

    full = virtual_cell_protocol(registry=registry, config=config)
    retained_ids = {
        PREPARE_STEP,
        SUMMARY_STEP,
        DEVELOP_STEP,
        FALSIFY_STEP,
        SELECT_STEP,
        FREEZE_STEP,
    }
    retained = tuple(value for value in full.steps if value.step_id in retained_ids)
    report = _step(
        registry=registry,
        config=config,
        step_id=DEVELOPMENT_REPORT_STEP,
        stage=ScientificStage.REPORT,
        capability_key=REPORTER_CAPABILITY_KEY,
        dependencies=(DEVELOP_STEP, FALSIFY_STEP, FREEZE_STEP, SELECT_STEP),
        outputs=(
            _json("development-closeout", VirtualCellDevelopmentCloseout.SCHEMA),
            _json("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
        ),
        access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
        barrier=BarrierKind.NONE,
        obligations=("vcc-development-to-reveal-handoff",),
        required_permissions=(
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
    )
    return ProtocolTemplate(
        template_id="protocol.virtual-cell-2025-tier-l0-development",
        template_version=CAPABILITY_VERSION,
        steps=tuple(sorted((*retained, report), key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _step_by_id(protocol: ProtocolTemplate, step_id: str) -> ProtocolStepTemplate:
    return next(value for value in protocol.steps if value.step_id == step_id)


def _output_by_id(
    protocol: ProtocolTemplate,
    step_id: str,
    output_id: str,
) -> OutputTemplate:
    return next(
        value
        for value in _step_by_id(protocol, step_id).outputs
        if value.output_id == output_id
    )


def _internal_edge(
    protocol: ProtocolTemplate,
    *,
    producer: str,
    output_id: str,
    consumer: str,
    consumer_input: str,
    role: ScientificInputRole,
    maximum_size_bytes: int,
) -> CandidateGraphEdge:
    output = _output_by_id(protocol, producer, output_id)
    source = _step_by_id(protocol, producer)
    target = _step_by_id(protocol, consumer)
    return CandidateGraphEdge(
        edge_id=f"edge.{producer}.{output_id}.{consumer}.{consumer_input}",
        producer_node_id=producer,
        producer_output_id=output_id,
        external_input_id=None,
        consumer_node_id=consumer,
        consumer_input_id=consumer_input,
        scientific_role=role,
        logical_artifact_id=f"artifact.{producer}.{output_id}",
        payload_schema=output.payload_schema,
        media_type=output.media_type,
        maximum_size_bytes=maximum_size_bytes,
        outcome_access=source.requested_outcome_access,
        visibility_ceiling=source.visibility_ceiling,
        barrier=target.barrier,
    )


def virtual_cell_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source_manifest: VirtualCellSourceManifest,
    test_source: VirtualCellSourceObject,
    target_feature_sha256: str,
    target_feature_size_bytes: int,
    test_roster_sha256: str,
    test_roster_size_bytes: int,
    leaderboard_snapshot_sha256: str,
    leaderboard_snapshot_size_bytes: int,
) -> CandidateScientificGraph:
    external = (
        CandidateGraphExternalInput(
            input_id=LEADERBOARD_SNAPSHOT_INPUT_ID,
            scientific_role=ScientificInputRole.OUTCOME,
            logical_artifact_id=LEADERBOARD_SNAPSHOT_INPUT_ID,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=leaderboard_snapshot_sha256,
            payload_schema=LeaderboardSnapshot.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=leaderboard_snapshot_size_bytes,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        ),
        CandidateGraphExternalInput(
            input_id=SEALED_TEST_BINDING_INPUT_ID,
            scientific_role=ScientificInputRole.OUTCOME,
            logical_artifact_id=SEALED_TEST_BINDING_INPUT_ID,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=test_source.fingerprint(),
            payload_schema=VirtualCellSourceObject.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=1024 * 1024,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        CandidateGraphExternalInput(
            input_id=SOURCE_MANIFEST_INPUT_ID,
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=SOURCE_MANIFEST_INPUT_ID,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=source_manifest.fingerprint(),
            payload_schema=VirtualCellSourceManifest.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=4 * 1024**2,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        CandidateGraphExternalInput(
            input_id=TARGET_FEATURE_INPUT_ID,
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=TARGET_FEATURE_INPUT_ID,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=target_feature_sha256,
            payload_schema=TARGET_FEATURE_TABLE_SCHEMA,
            media_type="application/vnd.apache.arrow.file",
            maximum_size_bytes=target_feature_size_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        CandidateGraphExternalInput(
            input_id=TEST_ROSTER_INPUT_ID,
            # Source-materialization roles have no ACTION variant.  The exact
            # roster is therefore bound as frozen model/configuration input;
            # the SystemSpec still owns its nominal action semantics.
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=TEST_ROSTER_INPUT_ID,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=test_roster_sha256,
            payload_schema=TEST_ROSTER_TEXT_SCHEMA,
            media_type="text/csv",
            maximum_size_bytes=test_roster_size_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
    )
    manifests = {value.registry_id: value for value in registry.capabilities}
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=manifests[
                f"{step.capability_key}@{step.capability_version}"
            ].implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    edges = [
        CandidateGraphEdge(
            edge_id=(
                f"edge.external.{LEADERBOARD_SNAPSHOT_INPUT_ID}.{PLACEMENT_STEP}.leaderboard"
            ),
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=LEADERBOARD_SNAPSHOT_INPUT_ID,
            consumer_node_id=PLACEMENT_STEP,
            consumer_input_id="leaderboard-snapshot",
            scientific_role=ScientificInputRole.OUTCOME,
            logical_artifact_id=LEADERBOARD_SNAPSHOT_INPUT_ID,
            payload_schema=LeaderboardSnapshot.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=leaderboard_snapshot_size_bytes,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{SOURCE_MANIFEST_INPUT_ID}.{PREPARE_STEP}.manifest",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=SOURCE_MANIFEST_INPUT_ID,
            consumer_node_id=PREPARE_STEP,
            consumer_input_id="source-manifest",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=SOURCE_MANIFEST_INPUT_ID,
            payload_schema=VirtualCellSourceManifest.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=4 * 1024**2,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{TARGET_FEATURE_INPUT_ID}.{DEVELOP_STEP}.features",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=TARGET_FEATURE_INPUT_ID,
            consumer_node_id=DEVELOP_STEP,
            consumer_input_id="target-features",
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=TARGET_FEATURE_INPUT_ID,
            payload_schema=TARGET_FEATURE_TABLE_SCHEMA,
            media_type="application/vnd.apache.arrow.file",
            maximum_size_bytes=target_feature_size_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{TARGET_FEATURE_INPUT_ID}.{FREEZE_STEP}.features",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=TARGET_FEATURE_INPUT_ID,
            consumer_node_id=FREEZE_STEP,
            consumer_input_id="target-features",
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=TARGET_FEATURE_INPUT_ID,
            payload_schema=TARGET_FEATURE_TABLE_SCHEMA,
            media_type="application/vnd.apache.arrow.file",
            maximum_size_bytes=target_feature_size_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{TARGET_FEATURE_INPUT_ID}.{FALSIFY_STEP}.features",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=TARGET_FEATURE_INPUT_ID,
            consumer_node_id=FALSIFY_STEP,
            consumer_input_id="target-features",
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=TARGET_FEATURE_INPUT_ID,
            payload_schema=TARGET_FEATURE_TABLE_SCHEMA,
            media_type="application/vnd.apache.arrow.file",
            maximum_size_bytes=target_feature_size_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{TEST_ROSTER_INPUT_ID}.{FREEZE_STEP}.roster",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=TEST_ROSTER_INPUT_ID,
            consumer_node_id=FREEZE_STEP,
            consumer_input_id="test-roster",
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=TEST_ROSTER_INPUT_ID,
            payload_schema=TEST_ROSTER_TEXT_SCHEMA,
            media_type="text/csv",
            maximum_size_bytes=test_roster_size_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{SEALED_TEST_BINDING_INPUT_ID}.{EVALUATE_STEP}.test",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=SEALED_TEST_BINDING_INPUT_ID,
            consumer_node_id=EVALUATE_STEP,
            consumer_input_id="sealed-test-binding",
            scientific_role=ScientificInputRole.OUTCOME,
            logical_artifact_id=SEALED_TEST_BINDING_INPUT_ID,
            payload_schema=VirtualCellSourceObject.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=1024 * 1024,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.REVEAL,
        ),
    ]
    internal_specs = (
        (
            PREPARE_STEP,
            "prepared-source",
            SUMMARY_STEP,
            "prepared-source",
            ScientificInputRole.QUALIFICATION,
            4 * 1024**2,
        ),
        (
            SUMMARY_STEP,
            "train-summary-record",
            DEVELOP_STEP,
            "train-summary-record",
            ScientificInputRole.QUALIFICATION,
            4 * 1024**2,
        ),
        (
            PREPARE_STEP,
            "prepared-source",
            DEVELOP_STEP,
            "prepared-source",
            ScientificInputRole.QUALIFICATION,
            4 * 1024**2,
        ),
        (
            SUMMARY_STEP,
            "train-summary-table",
            DEVELOP_STEP,
            "train-summary-table",
            ScientificInputRole.RECEIVER,
            1024**3,
        ),
        (
            SUMMARY_STEP,
            "validation-summary-record",
            DEVELOP_STEP,
            "validation-summary-record",
            ScientificInputRole.QUALIFICATION,
            4 * 1024**2,
        ),
        (
            SUMMARY_STEP,
            "validation-summary-table",
            DEVELOP_STEP,
            "validation-summary-table",
            ScientificInputRole.RECEIVER,
            1024**3,
        ),
        (
            SUMMARY_STEP,
            "control-reservoir",
            DEVELOP_STEP,
            "control-reservoir",
            ScientificInputRole.RECEIVER,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-development",
            FALSIFY_STEP,
            "development",
            ScientificInputRole.MODEL,
            16 * 1024**2,
        ),
        (
            DEVELOP_STEP,
            "model-baseline-no-change",
            FALSIFY_STEP,
            "model-baseline-no-change",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-baseline-weighted-common-response",
            FALSIFY_STEP,
            "model-baseline-weighted-common-response",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-feature-ridge-response",
            FALSIFY_STEP,
            "model-feature-ridge-response",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-feature-ridge-response-with-reduced-rank",
            FALSIFY_STEP,
            "model-feature-ridge-response-with-reduced-rank",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-receiver-admission-conditioned-reduced-rank-ridge-response",
            FALSIFY_STEP,
            "model-receiver-admission-conditioned-reduced-rank-ridge-response",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            SUMMARY_STEP,
            "validation-summary-record",
            FALSIFY_STEP,
            "validation-summary-record",
            ScientificInputRole.QUALIFICATION,
            4 * 1024**2,
        ),
        (
            SUMMARY_STEP,
            "validation-summary-table",
            FALSIFY_STEP,
            "validation-summary-table",
            ScientificInputRole.RECEIVER,
            1024**3,
        ),
        (
            SUMMARY_STEP,
            "control-reservoir",
            FALSIFY_STEP,
            "control-reservoir",
            ScientificInputRole.RECEIVER,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-development",
            SELECT_STEP,
            "development",
            ScientificInputRole.MODEL,
            16 * 1024**2,
        ),
        (
            FALSIFY_STEP,
            "falsifier-panel",
            SELECT_STEP,
            "falsifiers",
            ScientificInputRole.QUALIFICATION,
            16 * 1024**2,
        ),
        (
            SUMMARY_STEP,
            "control-reservoir",
            FREEZE_STEP,
            "control-reservoir",
            ScientificInputRole.RECEIVER,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-refit-baseline-no-change",
            FREEZE_STEP,
            "model-refit-baseline-no-change",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-refit-baseline-weighted-common-response",
            FREEZE_STEP,
            "model-refit-baseline-weighted-common-response",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-refit-feature-ridge-response",
            FREEZE_STEP,
            "model-refit-feature-ridge-response",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-refit-feature-ridge-response-with-reduced-rank",
            FREEZE_STEP,
            "model-refit-feature-ridge-response-with-reduced-rank",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            DEVELOP_STEP,
            "model-refit-receiver-admission-conditioned-reduced-rank-ridge-response",
            FREEZE_STEP,
            "model-refit-receiver-admission-conditioned-reduced-rank-ridge-response",
            ScientificInputRole.MODEL,
            1024**3,
        ),
        (
            SELECT_STEP,
            "model-selection",
            FREEZE_STEP,
            "selection",
            ScientificInputRole.QUALIFICATION,
            16 * 1024**2,
        ),
        (
            FREEZE_STEP,
            "prediction-freeze",
            EVALUATE_STEP,
            "prediction-freeze",
            ScientificInputRole.MODEL,
            16 * 1024**2,
        ),
        (
            FREEZE_STEP,
            "prediction-envelope",
            EVALUATE_STEP,
            "prediction",
            ScientificInputRole.RECEIVER,
            2_000_000_000,
        ),
        (
            FREEZE_STEP,
            "prediction-freeze",
            PLACEMENT_STEP,
            "prediction-freeze",
            ScientificInputRole.MODEL,
            16 * 1024**2,
        ),
        (
            EVALUATE_STEP,
            "official-evaluation",
            PLACEMENT_STEP,
            "evaluation",
            ScientificInputRole.OUTCOME,
            16 * 1024**2,
        ),
        (
            FREEZE_STEP,
            "prediction-freeze",
            REPORT_STEP,
            "prediction-freeze",
            ScientificInputRole.MODEL,
            16 * 1024**2,
        ),
        (
            EVALUATE_STEP,
            "official-evaluation",
            REPORT_STEP,
            "evaluation",
            ScientificInputRole.OUTCOME,
            16 * 1024**2,
        ),
        (
            PLACEMENT_STEP,
            "placement-adjudication",
            REPORT_STEP,
            "placement",
            ScientificInputRole.OUTCOME,
            16 * 1024**2,
        ),
    )
    edges.extend(
        _internal_edge(
            protocol,
            producer=producer,
            output_id=output_id,
            consumer=consumer,
            consumer_input=consumer_input,
            role=role,
            maximum_size_bytes=maximum_size,
        )
        for producer, output_id, consumer, consumer_input, role, maximum_size in internal_specs
    )
    return CandidateScientificGraph(
        graph_id="graph.virtual-cell-2025-tier-l0",
        external_inputs=tuple(sorted(external, key=lambda value: value.input_id)),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def virtual_cell_development_graph(
    *,
    protocol: ProtocolTemplate,
    full_protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source_manifest: VirtualCellSourceManifest,
    test_source: VirtualCellSourceObject,
    target_feature_sha256: str,
    target_feature_size_bytes: int,
    test_roster_sha256: str,
    test_roster_size_bytes: int,
    leaderboard_snapshot_sha256: str,
    leaderboard_snapshot_size_bytes: int,
) -> CandidateScientificGraph:
    """Project the exact development/freeze prefix without final reveal inputs."""

    full = virtual_cell_graph(
        protocol=full_protocol,
        registry=registry,
        source_manifest=source_manifest,
        test_source=test_source,
        target_feature_sha256=target_feature_sha256,
        target_feature_size_bytes=target_feature_size_bytes,
        test_roster_sha256=test_roster_sha256,
        test_roster_size_bytes=test_roster_size_bytes,
        leaderboard_snapshot_sha256=leaderboard_snapshot_sha256,
        leaderboard_snapshot_size_bytes=leaderboard_snapshot_size_bytes,
    )
    retained_ids = {
        PREPARE_STEP,
        SUMMARY_STEP,
        DEVELOP_STEP,
        FALSIFY_STEP,
        SELECT_STEP,
        FREEZE_STEP,
    }
    manifests = {value.registry_id: value for value in registry.capabilities}
    report = _step_by_id(protocol, DEVELOPMENT_REPORT_STEP)
    report_node = CandidateGraphNode(
        node_id=report.step_id,
        stage=report.stage,
        capability_key=report.capability_key,
        capability_version=report.capability_version,
        implementation_sha256=manifests[
            f"{report.capability_key}@{report.capability_version}"
        ].implementation_sha256,
        protocol_step_sha256=report.fingerprint(),
        obligation_ids=report.obligation_ids,
        outcome_access=report.requested_outcome_access,
        visibility_ceiling=report.visibility_ceiling,
        resource_budget=report.resource_budget,
    )
    edges = [
        value
        for value in full.edges
        if value.consumer_node_id in retained_ids
        and (value.producer_node_id is None or value.producer_node_id in retained_ids)
    ]
    edges.extend(
        _internal_edge(
            protocol,
            producer=producer,
            output_id=output_id,
            consumer=DEVELOPMENT_REPORT_STEP,
            consumer_input=consumer_input,
            role=role,
            maximum_size_bytes=maximum_size_bytes,
        )
        for producer, output_id, consumer_input, role, maximum_size_bytes in (
            (
                DEVELOP_STEP,
                "model-development",
                "development",
                ScientificInputRole.MODEL,
                16 * 1024**2,
            ),
            (
                FALSIFY_STEP,
                "falsifier-panel",
                "falsifiers",
                ScientificInputRole.QUALIFICATION,
                16 * 1024**2,
            ),
            (
                SELECT_STEP,
                "model-selection",
                "selection",
                ScientificInputRole.QUALIFICATION,
                16 * 1024**2,
            ),
            (
                FREEZE_STEP,
                "prediction-freeze",
                "prediction-freeze",
                ScientificInputRole.MODEL,
                16 * 1024**2,
            ),
        )
    )
    external_ids = {
        value.external_input_id
        for value in edges
        if value.external_input_id is not None
    }
    return CandidateScientificGraph(
        graph_id="graph.virtual-cell-2025-tier-l0-development",
        external_inputs=tuple(
            sorted(
                (
                    value
                    for value in full.external_inputs
                    if value.input_id in external_ids
                ),
                key=lambda value: value.input_id,
            )
        ),
        nodes=tuple(
            sorted(
                (
                    *(value for value in full.nodes if value.node_id in retained_ids),
                    report_node,
                ),
                key=lambda value: value.node_id,
            )
        ),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def virtual_cell_obligation_coverage(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
    coverage_id: str = "coverage.virtual-cell-2025-tier-l0",
    fallback_step_id: str = REPORT_STEP,
    fallback_output_id: str = "closeout",
) -> ObligationCoverage:
    incoming = {
        node.node_id: tuple(
            sorted(
                edge.edge_id
                for edge in graph.edges
                if edge.consumer_node_id == node.node_id
            )
        )
        for node in graph.nodes
    }
    owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    fallback = (fallback_step_id, fallback_output_id)
    bindings = tuple(
        ObligationCoverageBinding(
            obligation_id=obligation_id,
            proof_owner_node_id=owners.get(obligation_id, fallback)[0],
            required_output_id=owners.get(obligation_id, fallback)[1],
            contributor_edge_ids=incoming[owners.get(obligation_id, fallback)[0]],
        )
        for obligation_id in required_candidate_obligation_ids(experiment, protocol)
    )
    return ObligationCoverage(
        coverage_id=coverage_id,
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def virtual_cell_template(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source_manifest: VirtualCellSourceManifest,
    test_source: VirtualCellSourceObject,
    target_feature_sha256: str,
    target_feature_size_bytes: int,
    test_roster_sha256: str,
    test_roster_size_bytes: int,
    leaderboard_snapshot_sha256: str,
    leaderboard_snapshot_size_bytes: int,
) -> StudyTemplate:
    graph = virtual_cell_graph(
        protocol=protocol,
        registry=registry,
        source_manifest=source_manifest,
        test_source=test_source,
        target_feature_sha256=target_feature_sha256,
        target_feature_size_bytes=target_feature_size_bytes,
        test_roster_sha256=test_roster_sha256,
        test_roster_size_bytes=test_roster_size_bytes,
        leaderboard_snapshot_sha256=leaderboard_snapshot_sha256,
        leaderboard_snapshot_size_bytes=leaderboard_snapshot_size_bytes,
    )
    return StudyTemplate(
        template_key="virtual-cell.2025-tier-l0",
        template_version=CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=virtual_cell_obligation_coverage(
            experiment=experiment,
            protocol=protocol,
            graph=graph,
        ),
    )


def virtual_cell_development_template(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    full_protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source_manifest: VirtualCellSourceManifest,
    test_source: VirtualCellSourceObject,
    target_feature_sha256: str,
    target_feature_size_bytes: int,
    test_roster_sha256: str,
    test_roster_size_bytes: int,
    leaderboard_snapshot_sha256: str,
    leaderboard_snapshot_size_bytes: int,
) -> StudyTemplate:
    graph = virtual_cell_development_graph(
        protocol=protocol,
        full_protocol=full_protocol,
        registry=registry,
        source_manifest=source_manifest,
        test_source=test_source,
        target_feature_sha256=target_feature_sha256,
        target_feature_size_bytes=target_feature_size_bytes,
        test_roster_sha256=test_roster_sha256,
        test_roster_size_bytes=test_roster_size_bytes,
        leaderboard_snapshot_sha256=leaderboard_snapshot_sha256,
        leaderboard_snapshot_size_bytes=leaderboard_snapshot_size_bytes,
    )
    return StudyTemplate(
        template_key="virtual-cell.2025-tier-l0-development",
        template_version=CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=virtual_cell_obligation_coverage(
            experiment=experiment,
            protocol=protocol,
            graph=graph,
            coverage_id="coverage.virtual-cell-2025-tier-l0-development",
            fallback_step_id=DEVELOPMENT_REPORT_STEP,
            fallback_output_id="development-closeout",
        ),
    )


def virtual_cell_candidate_catalog(
    *,
    registry: CapabilityRegistry,
    templates: tuple[StudyTemplate, ...],
) -> CandidateCapabilityCatalog:
    return CandidateCapabilityCatalog(
        catalog_id="candidate-catalog.virtual-cell-2025-tier-l0",
        registrations=tuple(
            CandidateCapabilityRegistration(
                manifest=manifest,
                provider_key=manifest.capability_key,
                provider_version=manifest.capability_version,
                config_media_type="application/json",
                maximum_config_bytes=MAXIMUM_CONFIG_BYTES,
            )
            for manifest in registry.capabilities
        ),
        templates=tuple(sorted(templates, key=lambda value: value.template_key)),
    )


def source_bytes_sha256(payload: bytes) -> str:
    return sha256(payload).hexdigest()


__all__ = [
    "CONFIG_ARTIFACT_ID",
    "DEVELOPMENT_REPORT_STEP",
    "DEVELOP_STEP",
    "EVALUATE_STEP",
    "FALSIFY_STEP",
    "FREEZE_STEP",
    "LEADERBOARD_SNAPSHOT_INPUT_ID",
    "PLACEMENT_STEP",
    "PREPARE_STEP",
    "REPORT_STEP",
    "SEALED_TEST_BINDING_INPUT_ID",
    "SELECT_STEP",
    "SOURCE_MANIFEST_INPUT_ID",
    "SUMMARY_STEP",
    "TARGET_FEATURE_INPUT_ID",
    "TEST_ROSTER_INPUT_ID",
    "source_bytes_sha256",
    "virtual_cell_candidate_catalog",
    "virtual_cell_config_ref",
    "virtual_cell_development_graph",
    "virtual_cell_development_protocol",
    "virtual_cell_development_template",
    "virtual_cell_graph",
    "virtual_cell_obligation_coverage",
    "virtual_cell_protocol",
    "virtual_cell_template",
]
