"""Path-bounded runners for the frozen 2025 Tier-L0 scientific DAG."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields
from decimal import Decimal
import hashlib
from pathlib import Path
from typing import Any, BinaryIO, TypeVar, cast

import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationRecord,
    encode_scientific_adjudication,
)
from empirical_lawhood.runtime.artifacts import ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    StreamedTaskOutput,
    TaskContext,
    TaskOutputPayload,
    TaskOutputSource,
    WorkerInputPort,
    WorkerOutputPort,
)

from .analysis import (
    FalsifierResult,
    LinearResponseModel,
    ModelFamily,
    TargetFeatureMatrix,
    adjudicate_placement,
    augment_basal_target_expression,
    collapse_target_batch_responses,
    evaluate_prediction,
    predict_response,
)
from .codecs import (
    decode_control_reservoir,
    decode_linear_model,
    decode_response_summary,
    decode_target_features,
    encode_control_reservoir,
    encode_linear_model,
    encode_predicted_means,
    encode_response_summary,
    extract_h5ad_envelope,
    write_h5ad_envelope,
)
from .config import (
    CAPABILITY_VERSION,
    EVALUATOR_CAPABILITY_KEY,
    FALSIFIER_CAPABILITY_KEY,
    HISTORICAL_INFORMATION_CUTOFF_UTC,
    MODEL_CAPABILITY_KEY,
    PLACEMENT_CAPABILITY_KEY,
    PREDICTION_CAPABILITY_KEY,
    REPORTER_CAPABILITY_KEY,
    SELECTION_CAPABILITY_KEY,
    SOURCE_CAPABILITY_KEY,
    SUMMARY_CAPABILITY_KEY,
    decode_pipeline_config,
)
from .contracts import LeaderboardSnapshot, PlacementAdjudication, PredictionCommitment, VirtualCellContractError, ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceManifest, VirtualCellSourceObject, VirtualCellSplit
from .dataset import (
    ControlReservoirArrays,
    ResponseSummaryArrays,
    extract_control_reservoir_h5ad,
    require_response_summary_support,
    summarize_h5ad,
)
from .development import (
    CandidateOfficialValidation,
    TierL0Development,
    competition_selection_integrity_failures,
    finalize_tier_l0_development,
    fit_tier_l0_candidates,
)
from .evaluation import (
    bootstrap_official_score_standard_error,
    evaluate_official_2025,
    read_anndata_h5ad_stream,
)
from .features import virtual_cell_feature_ids
from .ports import (
    VirtualCellScratchPort,
    VirtualCellScratchWorkspace,
    VirtualCellSourcePort,
)
from .prediction import (
    build_prediction_roster,
    compile_absolute_prediction_means,
    write_prediction_h5ad,
)
from .records import (
    CONTROL_RESERVOIR_TABLE_SCHEMA,
    CandidateValidationMetricsBinding,
    FalsifierPanel,
    MODEL_SAFETENSORS_SCHEMA,
    ModelArtifactBinding,
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


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)
_COPY_CHUNK_BYTES = 8 * 1024**2
_PREDICTION_SCHEMA_SHA256 = hashlib.sha256(
    b"anndata-0.11/dense-float32-X/obs-target_gene/var-gene-order/v1"
).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _registry_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _one_input(
    context: TaskContext,
    *,
    payload_schema: str,
    artifact_suffix: str | None = None,
) -> WorkerInputPort:
    matches = tuple(
        value
        for value in context.input_ports
        if value.payload_schema == payload_schema
        and (artifact_suffix is None or value.artifact_id.endswith(artifact_suffix))
    )
    if len(matches) != 1:
        raise ValueError(
            f"{context.task_id} requires one {payload_schema} input"
            + ("" if artifact_suffix is None else f" ending in {artifact_suffix}")
        )
    return matches[0]


def _canonical_input(
    context: TaskContext,
    record_type: type[_RecordT],
    *,
    artifact_suffix: str | None = None,
) -> _RecordT:
    port = _one_input(
        context,
        payload_schema=record_type.SCHEMA,
        artifact_suffix=artifact_suffix,
    )
    return decode_canonical_bytes(
        port.read(),
        record_type,
        maximum_bytes=max(1, port.size_bytes),
    )


def _output_port(context: TaskContext, output_id: str) -> WorkerOutputPort:
    qualified_output_id = f"{context.task_id}.{output_id}"
    matches = tuple(
        value
        for value in context.output_ports
        if value.output_id in {output_id, qualified_output_id}
    )
    if len(matches) != 1:
        raise ValueError(f"{context.task_id} lacks one exact output {output_id}")
    return matches[0]


def _json_output(
    context: TaskContext, output_id: str, record: CanonicalRecord
) -> TaskOutputPayload:
    port = _output_port(context, output_id)
    if port.payload_schema != record.SCHEMA:
        raise ValueError(f"output {output_id} schema differs from its canonical record")
    return TaskOutputPayload(output_id=port.output_id, payload=record.canonical_bytes())


def _bytes_output(
    context: TaskContext,
    output_id: str,
    payload: bytes,
    *,
    payload_schema: str,
) -> TaskOutputPayload:
    port = _output_port(context, output_id)
    if port.payload_schema != payload_schema:
        raise ValueError(f"output {output_id} schema differs")
    return TaskOutputPayload(output_id=port.output_id, payload=payload)


def _result(
    outputs: tuple[TaskOutputPayload | StreamedTaskOutput, ...],
    *check_ids: str,
) -> RunnerResult:
    return RunnerResult(
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        checks=tuple(
            ReceiptCheck(check_id=check_id, passed=True, reason_codes=())
            for check_id in sorted(check_ids)
        ),
    )


def _source_object(
    manifest: VirtualCellSourceManifest,
    *,
    split: VirtualCellSplit,
    role_fragment: str,
) -> VirtualCellSourceObject:
    matches = tuple(
        value
        for value in manifest.objects
        if value.split is split and role_fragment in value.role
    )
    if len(matches) != 1:
        raise ValueError(
            f"source manifest lacks one {split.value} {role_fragment} object"
        )
    return matches[0]


class _WorkspaceOutputSource(TaskOutputSource):
    def __init__(
        self,
        workspace: VirtualCellScratchWorkspace,
        path: Path,
    ) -> None:
        self._workspace = workspace
        self._path = path
        self._consumed = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if maximum_chunk_bytes <= 0 or self._consumed:
            raise ValueError("scratch output source is invalid or already consumed")
        self._consumed = True
        with self._path.open("rb") as source:
            while chunk := source.read(maximum_chunk_bytes):
                yield chunk

    def close(self) -> None:
        self._workspace.close()


class VirtualCellTaskRunner:
    """One statically registered VCC implementation, dispatched by capability key."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: ProvenanceBoundVirtualCellPipelineConfig,
        source_manifest: VirtualCellSourceManifest,
        source_port: VirtualCellSourcePort,
        scratch_port: VirtualCellScratchPort,
    ) -> None:
        self.manifest = manifest
        self.config = config
        self.source_manifest = source_manifest
        self.source_port = source_port
        self.scratch_port = scratch_port

    def _validate_context(self, context: TaskContext) -> None:
        if self.manifest.capability_version != CAPABILITY_VERSION:
            raise ValueError("Virtual Cell runner version differs")
        port = _one_input(context, payload_schema=ProvenanceBoundVirtualCellPipelineConfig.SCHEMA)
        observed = decode_pipeline_config(port.read())
        if (
            observed != self.config
            or context.config.content_sha256 != self.config.fingerprint()
        ):
            raise ValueError(
                "Virtual Cell worker config differs from the frozen provider"
            )

    def execute(self, context: TaskContext) -> RunnerResult:
        self._validate_context(context)
        dispatch = {
            SOURCE_CAPABILITY_KEY: self._prepare_source,
            SUMMARY_CAPABILITY_KEY: self._summarize,
            MODEL_CAPABILITY_KEY: self._develop,
            FALSIFIER_CAPABILITY_KEY: self._falsify,
            SELECTION_CAPABILITY_KEY: self._select,
            PREDICTION_CAPABILITY_KEY: self._freeze_prediction,
            EVALUATOR_CAPABILITY_KEY: self._evaluate,
            PLACEMENT_CAPABILITY_KEY: self._place,
            REPORTER_CAPABILITY_KEY: self._report,
        }
        try:
            execute = dispatch[self.manifest.capability_key]
        except KeyError as error:
            raise NotImplementedError("unregistered Virtual Cell capability") from error
        return execute(context)

    def _prepare_source(self, context: TaskContext) -> RunnerResult:
        manifest = _canonical_input(context, VirtualCellSourceManifest)
        if (
            manifest != self.source_manifest
            or manifest.fingerprint() != self.config.source_manifest_sha256
        ):
            raise ValueError("source manifest differs from the frozen config")
        for source in manifest.objects:
            self.source_port.verify_available(source)
        sealed = tuple(
            sorted(value.object_id for value in manifest.objects if value.sealed)
        )
        development = tuple(
            sorted(value.object_id for value in manifest.objects if not value.sealed)
        )
        record = PreparedSourceRecord(
            record_id="prepared-source.virtual-cell-2025",
            source_manifest_sha256=manifest.fingerprint(),
            schema_contract_sha256s=self.config.schema_contract_sha256s,
            independent_unit_contract_sha256=self.config.independent_unit_contract.fingerprint(),
            outcome_access_manifest_sha256=self.config.outcome_access_manifest.fingerprint(),
            development_object_ids=development,
            sealed_evaluation_object_ids=sealed,
            test_content_read=False,
        )
        return _result(
            (_json_output(context, "prepared-source", record),),
            "vcc-exact-seven-object-manifest",
            "vcc-segmented-source-byte-closure",
            "vcc-test-content-not-opened",
        )

    def _summary_record(
        self,
        *,
        split: VirtualCellSplit,
        source: VirtualCellSourceObject,
        table: bytes,
        summary: ResponseSummaryArrays,
        table_artifact_id: str,
    ) -> ResponseSummaryRecord:
        return ResponseSummaryRecord(
            record_id=f"summary.virtual-cell-2025-{split.value.lower()}",
            split=split,
            source_object_sha256=source.sha256,
            summary_spec_sha256=self.config.response_summary.fingerprint(),
            table_artifact_id=table_artifact_id,
            table_sha256=_sha256(table),
            group_count=len(summary.group_ids),
            target_count=len(
                set(summary.target_ids)
                - {self.config.response_summary.comparator_label}
            ),
            batch_count=len(set(summary.batch_ids)),
            gene_count=len(summary.gene_ids),
            gene_ids=summary.gene_ids,
            gene_order_sha256=_registry_sha256(summary.gene_ids),
            normalization=summary.normalization,
            normalization_target_sum=Decimal(str(summary.normalization_target_sum)),
            comparator_label=self.config.response_summary.comparator_label,
            aggregation_unit=self.config.response_summary.aggregation_unit,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )

    def _summarize(self, context: TaskContext) -> RunnerResult:
        prepared = _canonical_input(context, PreparedSourceRecord)
        if prepared.source_manifest_sha256 != self.source_manifest.fingerprint():
            raise ValueError(
                "prepared source record differs from the provider manifest"
            )
        spec = self.config.response_summary
        target_sum = float(cast(Decimal, spec.normalization_target_sum))
        summaries = {}
        sources = {}
        for split in (VirtualCellSplit.TRAIN, VirtualCellSplit.VALIDATION):
            source = _source_object(
                self.source_manifest, split=split, role_fragment="response"
            )
            sources[split] = source
            with self.source_port.open_object(
                source,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            ) as stream:
                summaries[split] = summarize_h5ad(
                    stream,
                    split=split,
                    target_field=spec.target_field,
                    batch_field=spec.batch_field,
                    maximum_rows=300_000,
                    maximum_genes=20_000,
                    normalization=spec.normalization,
                    normalization_target_sum=target_sum,
                )
        train = summaries[VirtualCellSplit.TRAIN]
        validation = summaries[VirtualCellSplit.VALIDATION]
        for summary in (train, validation):
            require_response_summary_support(
                summary,
                control_label=spec.comparator_label,
                minimum_cells_per_target=spec.minimum_cells_per_target,
            )
        if (
            train.gene_ids != validation.gene_ids
            or _registry_sha256(train.gene_ids)
            != self.source_manifest.gene_order_sha256
        ):
            raise VirtualCellContractError(
                "development gene registries differ from the source lock"
            )
        train_table = encode_response_summary(train)
        validation_table = encode_response_summary(validation)
        train_source = sources[VirtualCellSplit.TRAIN]
        with self.source_port.open_object(
            train_source,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        ) as stream:
            reservoir = extract_control_reservoir_h5ad(
                stream,
                split=VirtualCellSplit.TRAIN,
                target_field=spec.target_field,
                batch_field=spec.batch_field,
                control_label=spec.comparator_label,
                maximum_rows=300_000,
                maximum_genes=20_000,
                maximum_cells=self.config.prediction_compiler.control_reservoir_cells,
                normalization_target_sum=target_sum,
                seed=self.config.prediction_compiler.compiler_seed,
            )
        reservoir_table = encode_control_reservoir(reservoir)
        train_output = _output_port(context, "train-summary-table")
        validation_output = _output_port(context, "validation-summary-table")
        if (
            train_output.logical_artifact_id is None
            or validation_output.logical_artifact_id is None
        ):
            raise ValueError("summary tables lack logical artifact identities")
        train_record = self._summary_record(
            split=VirtualCellSplit.TRAIN,
            source=train_source,
            table=train_table,
            summary=train,
            table_artifact_id=train_output.logical_artifact_id,
        )
        validation_record = self._summary_record(
            split=VirtualCellSplit.VALIDATION,
            source=sources[VirtualCellSplit.VALIDATION],
            table=validation_table,
            summary=validation,
            table_artifact_id=validation_output.logical_artifact_id,
        )
        return _result(
            (
                _bytes_output(
                    context,
                    "control-reservoir",
                    reservoir_table,
                    payload_schema=CONTROL_RESERVOIR_TABLE_SCHEMA,
                ),
                _json_output(context, "train-summary-record", train_record),
                _bytes_output(
                    context,
                    "train-summary-table",
                    train_table,
                    payload_schema=RESPONSE_SUMMARY_TABLE_SCHEMA,
                ),
                _json_output(context, "validation-summary-record", validation_record),
                _bytes_output(
                    context,
                    "validation-summary-table",
                    validation_table,
                    payload_schema=RESPONSE_SUMMARY_TABLE_SCHEMA,
                ),
            ),
            "vcc-batch-ceiling-not-cell-replication",
            "vcc-fixed-total-normalization",
            "vcc-sparse-dense-equivalence",
        )

    def _response_inputs(
        self,
        context: TaskContext,
        *,
        split_name: str,
    ) -> tuple[ResponseSummaryRecord, bytes]:
        record = _canonical_input(
            context,
            ResponseSummaryRecord,
            artifact_suffix=f"{split_name}-summary-record",
        )
        table_port = _one_input(
            context,
            payload_schema=RESPONSE_SUMMARY_TABLE_SCHEMA,
            artifact_suffix=f"{split_name}-summary-table",
        )
        table = table_port.read()
        if _sha256(table) != record.table_sha256:
            raise ValueError(
                f"{split_name} response table differs from its summary record"
            )
        return record, table

    def _features(self, context: TaskContext) -> TargetFeatureMatrix:
        payload = _one_input(context, payload_schema=TARGET_FEATURE_TABLE_SCHEMA).read()
        if _sha256(payload) != self.config.target_feature_sha256:
            raise ValueError("target feature bytes differ from the frozen config")
        return decode_target_features(
            payload,
            feature_ids=virtual_cell_feature_ids(),
            provenance_sha256=self.config.target_feature_provenance_sha256,
        )

    def _control_reservoir(
        self,
        context: TaskContext,
        *,
        gene_ids: tuple[str, ...],
    ) -> tuple[ControlReservoirArrays, bytes]:
        payload = _one_input(
            context, payload_schema=CONTROL_RESERVOIR_TABLE_SCHEMA
        ).read()
        reservoir = decode_control_reservoir(
            payload,
            gene_ids=gene_ids,
            normalization_target_sum=float(
                cast(Decimal, self.config.response_summary.normalization_target_sum)
            ),
        )
        return reservoir, payload

    def _development_inputs(
        self,
        context: TaskContext,
    ) -> tuple[
        ResponseSummaryRecord,
        ResponseSummaryRecord,
        ResponseSummaryArrays,
        ResponseSummaryArrays,
        ControlReservoirArrays,
        bytes,
        TargetFeatureMatrix,
    ]:
        train_record, train_table = self._response_inputs(context, split_name="train")
        validation_record, validation_table = self._response_inputs(
            context,
            split_name="validation",
        )
        if train_record.gene_ids != validation_record.gene_ids:
            raise ValueError("training and validation summary gene registries differ")
        normalization_target = float(
            cast(Decimal, self.config.response_summary.normalization_target_sum)
        )
        train = decode_response_summary(
            train_table,
            gene_ids=train_record.gene_ids,
            normalization=self.config.response_summary.normalization,
            normalization_target_sum=normalization_target,
        )
        validation = decode_response_summary(
            validation_table,
            gene_ids=validation_record.gene_ids,
            normalization=self.config.response_summary.normalization,
            normalization_target_sum=normalization_target,
        )
        reservoir, reservoir_payload = self._control_reservoir(
            context,
            gene_ids=train_record.gene_ids,
        )
        raw_features = self._features(context)
        features = augment_basal_target_expression(
            raw_features,
            gene_ids=train_record.gene_ids,
            control_mean=np.asarray(
                np.mean(reservoir.values, axis=0, dtype=np.float64),
                dtype=np.float64,
            ),
        )
        return (
            train_record,
            validation_record,
            train,
            validation,
            reservoir,
            reservoir_payload,
            features,
        )

    @staticmethod
    def _model_output_name(model_id: str) -> str:
        values = {
            "baseline-no-change": "model-baseline-no-change",
            "baseline-weighted-common-response": "model-baseline-weighted-common-response",
            "feature-ridge-response": "model-feature-ridge-response",
            "feature-ridge-response-with-reduced-rank": "model-feature-ridge-response-with-reduced-rank",
            "receiver-admission-conditioned-reduced-rank-ridge-response": "model-receiver-admission-conditioned-reduced-rank-ridge-response",
        }
        try:
            return values[model_id]
        except KeyError as error:
            raise ValueError(f"unexpected Tier-L0 candidate {model_id}") from error

    @classmethod
    def _model_refit_output_name(cls, candidate_id: str) -> str:
        return cls._model_output_name(candidate_id).replace("model-", "model-refit-", 1)

    @classmethod
    def _validation_metrics_output_name(cls, candidate_id: str) -> str:
        return cls._model_output_name(candidate_id).replace(
            "model-", "validation-metrics-", 1
        )

    def _development_record(
        self,
        context: TaskContext,
        development: TierL0Development,
        *,
        training_summary_sha256: str,
        validation_summary_sha256: str,
        feature_provenance_sha256: str,
        model_payloads: dict[str, bytes],
        validation_metric_payloads: dict[str, bytes],
    ) -> ModelDevelopmentRecord:
        fits = {value.candidate_id: value.fit for value in development.candidates}
        fits.update(
            {
                value.model.model_id: value.fit
                for value in development.development_refits
            }
        )
        bindings = []
        for output_id, payload in model_payloads.items():
            port = _output_port(context, output_id)
            if port.logical_artifact_id is None:
                raise ValueError("model output lacks logical artifact identity")
            model = decode_linear_model(payload)
            bindings.append(
                ModelArtifactBinding(
                    model_id=model.model_id,
                    artifact_id=port.logical_artifact_id,
                    artifact_sha256=_sha256(payload),
                    fit=fits[model.model_id],
                )
            )
        validation_bindings = []
        evaluations = {
            value.candidate_id: value for value in development.official_validations
        }
        validation_contract_sha256 = (
            self.config.model_selection.validation_metric_contract.fingerprint()
        )
        for output_id, payload in validation_metric_payloads.items():
            port = _output_port(context, output_id)
            if port.logical_artifact_id is None:
                raise ValueError(
                    "validation metric output lacks logical artifact identity"
                )
            candidate_id = next(
                candidate_id
                for candidate_id in evaluations
                if self._validation_metrics_output_name(candidate_id) == output_id
            )
            evaluation = evaluations[candidate_id]
            validation_bindings.append(
                CandidateValidationMetricsBinding(
                    candidate_id=candidate_id,
                    artifact_id=port.logical_artifact_id,
                    artifact_sha256=_sha256(payload),
                    target_ids_sha256=_registry_sha256(evaluation.target_ids),
                    target_count=len(evaluation.target_ids),
                    metric_contract_sha256=validation_contract_sha256,
                    cell_eval_version=evaluation.cell_eval_version,
                    pdex_version=evaluation.pdex_version,
                    outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
            )
        return ModelDevelopmentRecord(
            record_id="development.virtual-cell-2025-tier-l0",
            training_summary_sha256=training_summary_sha256,
            validation_summary_sha256=validation_summary_sha256,
            feature_provenance_sha256=feature_provenance_sha256,
            models=tuple(sorted(bindings, key=lambda value: value.model_id)),
            validation_metrics=tuple(
                sorted(validation_bindings, key=lambda value: value.candidate_id)
            ),
            tournament=development.tournament,
            training_target_ids_sha256=_registry_sha256(development.train_target_ids),
            validation_target_ids_sha256=_registry_sha256(
                development.validation_target_ids
            ),
            final_outcome_parent_ids=(),
            leaderboard_parent_ids=(),
        )

    def _develop(self, context: TaskContext) -> RunnerResult:
        prepared = _canonical_input(context, PreparedSourceRecord)
        if prepared.source_manifest_sha256 != self.source_manifest.fingerprint():
            raise ValueError("development source binding differs from prepared custody")
        (
            train_record,
            validation_record,
            train_summary,
            validation_summary,
            reservoir,
            reservoir_payload,
            features,
        ) = self._development_inputs(context)
        train_response = collapse_target_batch_responses(
            train_summary,
            control_label=self.config.response_summary.comparator_label,
        )
        validation_response = collapse_target_batch_responses(
            validation_summary,
            control_label=self.config.response_summary.comparator_label,
        )
        candidates = fit_tier_l0_candidates(
            train=train_response,
            validation=validation_response,
            features=features,
            contract=self.config.model_selection,
            realized_admission_repression_scale=float(
                self.config.model_selection.realized_admission_repression_scale
            ),
        )
        validation_roster_source = _source_object(
            self.source_manifest,
            split=VirtualCellSplit.VALIDATION,
            role_fragment="target-roster",
        )
        with self.source_port.open_object(
            validation_roster_source,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ) as roster_stream:
            validation_roster_payload = roster_stream.read()
        compiler = self.config.prediction_compiler
        validation_roster = build_prediction_roster(
            validation_roster_payload,
            roster_id="roster.virtual-cell-2025-validation-tournament",
            expected_target_count=len(validation_response.target_ids),
            total_cell_limit=compiler.total_cell_limit,
            control_cells=compiler.control_cells,
            minimum_cells_per_target=compiler.minimum_cells_per_target,
            control_label=self.config.response_summary.comparator_label,
        )
        validation_target_ids = tuple(
            value.target_id for value in validation_roster.targets
        )
        if validation_target_ids != validation_response.target_ids:
            raise ValueError(
                "official validation roster differs from the response summary"
            )
        validation_source = _source_object(
            self.source_manifest,
            split=VirtualCellSplit.VALIDATION,
            role_fragment="response",
        )
        if validation_record.source_object_sha256 != validation_source.sha256:
            raise ValueError(
                "official validation scorer source differs from its summary lineage"
            )
        with self.source_port.open_object(
            validation_source,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        ) as validation_stream:
            real_validation = read_anndata_h5ad_stream(validation_stream)
        workspace = self.scratch_port.allocate(
            run_id=context.run_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
        )
        official_validations: list[CandidateOfficialValidation] = []
        try:
            for candidate in candidates:
                model_features = (
                    None
                    if candidate.model.family
                    in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}
                    else features
                )
                deltas = predict_response(
                    candidate.model,
                    target_ids=validation_target_ids,
                    features=model_features,
                )
                means = compile_absolute_prediction_means(
                    target_ids=validation_target_ids,
                    predicted_deltas=deltas,
                    control_reservoir=reservoir,
                    maximum_value=float(compiler.maximum_value),
                )
                suffix = self._model_output_name(candidate.candidate_id).removeprefix(
                    "model-"
                )
                prediction_path = workspace.reserve_path(
                    f"validation-prediction-{suffix}.h5ad"
                )
                cell_eval_scratch = workspace.reserve_path(
                    f"cell-eval-validation-{suffix}"
                )
                write_prediction_h5ad(
                    path=prediction_path,
                    roster=validation_roster,
                    gene_ids=candidate.model.gene_ids,
                    predicted_means=means,
                    control_reservoir=reservoir,
                    control_reservoir_sha256=_sha256(reservoir_payload),
                    seed=compiler.compiler_seed,
                    mean_tolerance=float(compiler.mean_tolerance),
                    maximum_value=float(compiler.maximum_value),
                )
                with prediction_path.open("rb") as predicted_stream:
                    predicted = read_anndata_h5ad_stream(predicted_stream)
                run = evaluate_official_2025(
                    predicted=predicted,
                    real=real_validation,
                    metric_contract=self.config.model_selection.validation_metric_contract,
                    scratch_directory=cell_eval_scratch,
                    num_threads=min(8, context.resource_budget.cpu_cores),
                    control_label=self.config.response_summary.comparator_label,
                    perturbation_column=self.config.response_summary.target_field,
                )
                if run.target_ids != validation_target_ids:
                    raise ValueError(
                        "official scorer returned another validation target roster"
                    )
                standard_error = bootstrap_official_score_standard_error(
                    des=run.des,
                    pds=run.pds,
                    mae=run.mae,
                    metric_contract=self.config.model_selection.validation_metric_contract,
                    resamples=self.config.model_selection.bootstrap_targets,
                    seed=self.config.model_selection.bootstrap_seed,
                )
                official_validations.append(
                    CandidateOfficialValidation(
                        candidate_id=candidate.candidate_id,
                        target_ids=run.target_ids,
                        score=run.score,
                        score_standard_error=standard_error,
                        metrics_arrow=run.metrics_arrow,
                        cell_eval_version=run.cell_eval_version,
                        pdex_version=run.pdex_version,
                        response_outcome_read=run.response_outcome_read,
                    )
                )
        finally:
            workspace.close()
        development = finalize_tier_l0_development(
            train=train_response,
            validation=validation_response,
            features=features,
            contract=self.config.model_selection,
            realized_admission_repression_scale=float(
                self.config.model_selection.realized_admission_repression_scale
            ),
            candidates=candidates,
            official_validations=tuple(official_validations),
        )
        model_payloads = {
            self._model_output_name(candidate.candidate_id): encode_linear_model(
                candidate.model
            )
            for candidate in development.candidates
        }
        model_payloads.update(
            {
                self._model_refit_output_name(value.candidate_id): encode_linear_model(
                    value.model
                )
                for value in development.development_refits
            }
        )
        validation_metric_payloads = {
            self._validation_metrics_output_name(
                value.candidate_id
            ): value.metrics_arrow
            for value in development.official_validations
        }
        record = self._development_record(
            context,
            development,
            training_summary_sha256=train_record.table_sha256,
            validation_summary_sha256=validation_record.table_sha256,
            feature_provenance_sha256=features.provenance_sha256,
            model_payloads=model_payloads,
            validation_metric_payloads=validation_metric_payloads,
        )
        outputs: list[TaskOutputPayload | StreamedTaskOutput] = [
            _json_output(context, "model-development", record)
        ]
        outputs.extend(
            _bytes_output(
                context,
                output_id,
                payload,
                payload_schema=MODEL_SAFETENSORS_SCHEMA,
            )
            for output_id, payload in model_payloads.items()
        )
        outputs.extend(
            _bytes_output(
                context,
                output_id,
                payload,
                payload_schema=OFFICIAL_METRICS_TABLE_SCHEMA,
            )
            for output_id, payload in validation_metric_payloads.items()
        )
        return _result(
            tuple(outputs),
            "vcc-development-target-grouping",
            "vcc-exact-official-validation-score",
            "vcc-expected-admission-cross-fit",
            "vcc-final-outcome-lineage-empty",
            "vcc-validation-not-training",
        )

    @staticmethod
    def _proxy(mae: float, cosine: float) -> float:
        direction = min(1.0, max(0.0, (cosine + 1.0) / 2.0))
        magnitude = 1.0 / (1.0 + max(0.0, mae))
        return 0.5 * (direction + magnitude)

    @staticmethod
    def _falsifier(
        falsifier_id: str,
        *,
        observed: float,
        threshold: float,
        passed: bool,
        reason_code: str,
    ) -> FalsifierResult:
        return FalsifierResult(
            falsifier_id=falsifier_id,
            passed=passed,
            observed=Decimal(str(observed)),
            threshold=Decimal(str(threshold)),
            comparison="GE",
            reason_codes=(() if passed else (reason_code,)),
        )

    def _falsify(self, context: TaskContext) -> RunnerResult:
        development = _canonical_input(context, ModelDevelopmentRecord)
        selected_binding = next(
            value
            for value in development.models
            if value.model_id == development.tournament.selected_candidate_id
        )
        selected_ports = tuple(
            value
            for value in context.input_ports
            if value.artifact_id == selected_binding.artifact_id
        )
        if len(selected_ports) != 1:
            raise ValueError(
                "falsifier lacks the validation-selected training-only model"
            )
        selected_payload = selected_ports[0].read()
        if _sha256(selected_payload) != selected_binding.artifact_sha256:
            raise ValueError("falsifier model differs from the development record")
        final_model = decode_linear_model(selected_payload)
        validation_record, validation_table = self._response_inputs(
            context,
            split_name="validation",
        )
        validation_summary = decode_response_summary(
            validation_table,
            gene_ids=validation_record.gene_ids,
            normalization=self.config.response_summary.normalization,
            normalization_target_sum=float(
                cast(Decimal, self.config.response_summary.normalization_target_sum)
            ),
        )
        truth = collapse_target_batch_responses(
            validation_summary,
            control_label=self.config.response_summary.comparator_label,
        )
        reservoir, _payload = self._control_reservoir(
            context,
            gene_ids=validation_record.gene_ids,
        )
        raw_features = self._features(context)
        features = augment_basal_target_expression(
            raw_features,
            gene_ids=validation_record.gene_ids,
            control_mean=np.asarray(
                np.mean(reservoir.values, axis=0, dtype=np.float64),
                dtype=np.float64,
            ),
        )
        model_features = (
            None
            if final_model.family
            in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}
            else features
        )
        prediction = predict_response(
            final_model,
            target_ids=truth.target_ids,
            features=model_features,
        )
        mae, cosine, _ = evaluate_prediction(truth, prediction)
        baseline_proxy = self._proxy(mae, cosine)
        minimum_drop = float(self.config.model_selection.minimum_scientific_improvement)
        if model_features is None:
            target_result = FalsifierResult(
                falsifier_id="falsifier.target-label-shuffle",
                passed=True,
                observed=Decimal("0"),
                threshold=Decimal("0"),
                comparison="GE",
                reason_codes=("NOT_APPLICABLE_TARGET_INVARIANT_MODEL",),
            )
        else:
            permuted = TargetFeatureMatrix(
                target_ids=model_features.target_ids,
                feature_ids=model_features.feature_ids,
                values=np.roll(model_features.values, 1, axis=0),
                provenance_sha256=model_features.provenance_sha256,
            )
            shuffled = predict_response(
                final_model,
                target_ids=truth.target_ids,
                features=permuted,
            )
            shuffled_mae, shuffled_cosine, _ = evaluate_prediction(truth, shuffled)
            target_drop = baseline_proxy - self._proxy(shuffled_mae, shuffled_cosine)
            target_result = self._falsifier(
                "falsifier.target-label-shuffle",
                observed=target_drop,
                threshold=minimum_drop,
                passed=target_drop >= minimum_drop,
                reason_code="TARGET_LABEL_SHUFFLE_NOT_DISPLACED",
            )
        gene_permuted = np.roll(prediction, 1, axis=1)
        gene_mae, gene_cosine, _ = evaluate_prediction(truth, gene_permuted)
        gene_drop = baseline_proxy - self._proxy(gene_mae, gene_cosine)
        quantitative = (
            self._falsifier(
                "falsifier.gene-label-permutation",
                observed=gene_drop,
                threshold=minimum_drop,
                passed=gene_drop >= minimum_drop,
                reason_code="GENE_PERMUTATION_NOT_DISPLACED",
            ),
            target_result,
        )
        structural_conditions = (
            (
                "falsifier.feature-cutoff-and-provenance",
                raw_features.provenance_sha256
                == self.config.target_feature_provenance_sha256
                and self.config.historical_cutoff_utc
                == HISTORICAL_INFORMATION_CUTOFF_UTC,
                "FEATURE_CUTOFF_OR_PROVENANCE_FAILED",
            ),
            (
                "falsifier.final-outcome-lineage-empty",
                not development.final_outcome_parent_ids,
                "FINAL_OUTCOME_LINEAGE_NOT_EMPTY",
            ),
            (
                "falsifier.leaderboard-lineage-empty",
                not development.leaderboard_parent_ids,
                "LEADERBOARD_LINEAGE_NOT_EMPTY",
            ),
            (
                "falsifier.matched-control-contract",
                validation_record.comparator_label
                == self.config.response_summary.comparator_label
                and validation_record.summary_spec_sha256
                == self.config.response_summary.fingerprint()
                and reservoir.normalization
                == self.config.response_summary.normalization,
                "MATCHED_CONTROL_CONTRACT_FAILED",
            ),
        )
        structural = tuple(
            self._falsifier(
                falsifier_id,
                observed=1.0 if condition else 0.0,
                threshold=1.0,
                passed=condition,
                reason_code=reason_code,
            )
            for falsifier_id, condition, reason_code in structural_conditions
        )
        results = tuple(
            sorted((*quantitative, *structural), key=lambda value: value.falsifier_id)
        )
        required = tuple(value.falsifier_id for value in results)
        panel = FalsifierPanel(
            panel_id="falsifiers.virtual-cell-2025-tier-l0",
            results=results,
            required_falsifier_ids=required,
            all_required_pass=all(value.passed for value in results),
        )
        if development.final_outcome_parent_ids or development.leaderboard_parent_ids:
            raise ValueError(
                "development record violated protected lineage before falsification"
            )
        return _result(
            (_json_output(context, "falsifier-panel", panel),),
            "vcc-gene-and-target-permutation",
            "vcc-leakage-and-cutoff-sentinels",
            "vcc-wrong-control-and-label-shuffle",
        )

    def _select(self, context: TaskContext) -> RunnerResult:
        development = _canonical_input(context, ModelDevelopmentRecord)
        panel = _canonical_input(context, FalsifierPanel)
        tournament = development.tournament
        selected_candidate_id = tournament.selected_candidate_id
        integrity_failures = competition_selection_integrity_failures(panel)
        if integrity_failures:
            raise ValueError(
                "competition selection failed causal/provenance integrity: "
                + ",".join(integrity_failures)
            )
        final_refit_id = f"{selected_candidate_id}-development-refit"
        candidates = tuple(
            value for value in development.models if value.model_id == final_refit_id
        )
        if len(candidates) != 1:
            raise ValueError("selected candidate lacks one exact all-development refit")
        selected = candidates[0]
        record = ModelSelectionRecord(
            record_id="selection.virtual-cell-2025-tier-l0",
            tournament=tournament,
            falsifier_panel_sha256=panel.fingerprint(),
            selected_model_artifact_id=selected.artifact_id,
            selected_model_sha256=selected.artifact_sha256,
        )
        return _result(
            (_json_output(context, "model-selection", record),),
            "vcc-frozen-validation-only-selection",
            "vcc-one-standard-error-tie-break",
            "vcc-scientific-falsifiers-do-not-retune-competition",
        )

    def _selected_model(
        self,
        context: TaskContext,
        selection: ModelSelectionRecord,
    ) -> tuple[LinearResponseModel, bytes]:
        ports = tuple(
            value
            for value in context.input_ports
            if value.artifact_id == selection.selected_model_artifact_id
            and value.payload_schema == MODEL_SAFETENSORS_SCHEMA
        )
        if len(ports) != 1:
            raise ValueError("prediction lacks the exact selected model artifact")
        payload = ports[0].read()
        if _sha256(payload) != selection.selected_model_sha256:
            raise ValueError("selected model bytes differ from the frozen selection")
        return decode_linear_model(payload), payload

    @staticmethod
    def _copy_port(
        port: WorkerInputPort,
        destination: BinaryIO,
        *,
        expected_sha256: str,
    ) -> None:
        digest = hashlib.sha256()
        observed = 0
        while chunk := port.read(_COPY_CHUNK_BYTES):
            destination.write(chunk)
            digest.update(chunk)
            observed += len(chunk)
        if (
            observed != port.size_bytes
            or port.bytes_read != port.size_bytes
            or digest.hexdigest() != expected_sha256
        ):
            raise VirtualCellContractError(
                "runtime input copy failed exact byte/hash closure"
            )

    def _freeze_prediction(self, context: TaskContext) -> RunnerResult:
        selection = _canonical_input(context, ModelSelectionRecord)
        model, model_payload = self._selected_model(context, selection)
        reservoir, reservoir_payload = self._control_reservoir(
            context,
            gene_ids=model.gene_ids,
        )
        if model.gene_ids != reservoir.gene_ids:
            raise ValueError(
                "selected model and control reservoir gene registries differ"
            )
        raw_features = self._features(context)
        features = augment_basal_target_expression(
            raw_features,
            gene_ids=model.gene_ids,
            control_mean=np.asarray(
                np.mean(reservoir.values, axis=0, dtype=np.float64),
                dtype=np.float64,
            ),
        )
        roster_payload = _one_input(
            context, payload_schema=TEST_ROSTER_TEXT_SCHEMA
        ).read()
        compiler = self.config.prediction_compiler
        roster = build_prediction_roster(
            roster_payload,
            roster_id="roster.virtual-cell-2025-final-prediction",
            expected_target_count=100,
            total_cell_limit=compiler.total_cell_limit,
            control_cells=compiler.control_cells,
            minimum_cells_per_target=compiler.minimum_cells_per_target,
        )
        target_ids = tuple(value.target_id for value in roster.targets)
        model_features = (
            None
            if model.family in {ModelFamily.NO_CHANGE_BASELINE, ModelFamily.WEIGHTED_COMMON_RESPONSE_BASELINE}
            else features
        )
        deltas = predict_response(
            model,
            target_ids=target_ids,
            features=model_features,
        )
        compiled = compile_absolute_prediction_means(
            target_ids=target_ids,
            predicted_deltas=deltas,
            control_reservoir=reservoir,
            maximum_value=float(compiler.maximum_value),
        )
        predicted_table = encode_predicted_means(
            target_ids=target_ids,
            means=compiled.absolute_means,
            uncertainty=model.residual_scale,
        )
        workspace = self.scratch_port.allocate(
            run_id=context.run_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
        )
        inner_path = workspace.reserve_path("prediction.h5ad")
        envelope_path = workspace.reserve_path("prediction-envelope.h5")
        try:
            h5ad_receipt = write_prediction_h5ad(
                path=inner_path,
                roster=roster,
                gene_ids=model.gene_ids,
                predicted_means=compiled,
                control_reservoir=reservoir,
                control_reservoir_sha256=_sha256(reservoir_payload),
                seed=compiler.compiler_seed,
                mean_tolerance=float(compiler.mean_tolerance),
                maximum_value=float(compiler.maximum_value),
            )
            write_h5ad_envelope(
                inner_h5ad_path=inner_path,
                envelope_path=envelope_path,
                inner_sha256=h5ad_receipt.h5ad_sha256,
            )
            envelope_size = envelope_path.stat().st_size
            if envelope_size > compiler.maximum_envelope_bytes:
                raise VirtualCellContractError(
                    "prediction envelope exceeds its frozen byte bound"
                )
            envelope_sha256 = _file_sha256(envelope_path)
            predicted_output = _output_port(context, "predicted-means")
            envelope_output = _output_port(context, "prediction-envelope")
            if (
                predicted_output.logical_artifact_id is None
                or envelope_output.logical_artifact_id is None
            ):
                raise ValueError("prediction outputs lack logical artifact identities")
            commitment = PredictionCommitment(
                commitment_id="commitment.virtual-cell-2025-tier-l0",
                lane_id="lane-h-2025-historical-cutoff-reconstruction",
                model_id=model.model_id,
                model_sha256=_sha256(model_payload),
                config_sha256=self.config.fingerprint(),
                source_manifest_sha256=self.source_manifest.fingerprint(),
                split_contract_sha256=_sha256(
                    canonical_json_bytes(self.config.schema_contract_sha256s)
                ),
                metric_contract_sha256=self.config.metric_contract.fingerprint(),
                prediction_sha256=h5ad_receipt.h5ad_sha256,
                prediction_schema_sha256=_PREDICTION_SCHEMA_SHA256,
                target_roster_sha256=_sha256(roster_payload),
                gene_order_sha256=self.source_manifest.gene_order_sha256,
                frozen_at_utc=compiler.freeze_timestamp_utc,
                final_outcome_parent_ids=(),
                leaderboard_parent_ids=(),
            )
            freeze = PredictionFreezeRecord(
                record_id="freeze.virtual-cell-2025-tier-l0",
                commitment=commitment,
                predicted_mean_table_artifact_id=predicted_output.logical_artifact_id,
                predicted_mean_table_sha256=_sha256(predicted_table),
                h5ad_envelope_artifact_id=envelope_output.logical_artifact_id,
                h5ad_envelope_sha256=envelope_sha256,
                inner_h5ad_sha256=h5ad_receipt.h5ad_sha256,
                compiler_seed=compiler.compiler_seed,
                compiler_mean_tolerance=compiler.mean_tolerance,
                test_outcome_read=False,
            )
            stream = StreamedTaskOutput(
                output_id=envelope_output.output_id,
                size_bytes=envelope_size,
                physical_sha256=envelope_sha256,
                source=_WorkspaceOutputSource(workspace, envelope_path),
            )
        except Exception:
            workspace.close()
            raise
        return _result(
            (
                stream,
                _json_output(context, "prediction-freeze", freeze),
                _bytes_output(
                    context,
                    "predicted-means",
                    predicted_table,
                    payload_schema=PREDICTED_MEAN_TABLE_SCHEMA,
                ),
            ),
            "vcc-compiler-mean-preservation",
            "vcc-prediction-schema-and-range",
            "vcc-test-outcome-read-false",
            "vcc-vfat-safe-output-bound",
        )

    @staticmethod
    def _scientific_adjudication(
        context: TaskContext,
        *,
        scientific_status: ScientificStatus,
        reason_codes: tuple[str, ...],
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError("Virtual Cell evaluator lacks adjudication context")
        output_ids = tuple(
            sorted(
                value.logical_artifact_id
                for value in context.output_ports
                if value.logical_artifact_id is not None
            )
        )
        if (
            len(output_ids) != len(context.output_ports)
            or not context.dependency_receipt_ids
        ):
            raise ValueError(
                "Virtual Cell adjudication lacks output or receipt identity"
            )
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication_context.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=output_ids,
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication_context.evidence_world_id,
            evidence_world_kind=adjudication_context.evidence_world_kind,
            relation=adjudication_context.relation,
            independent_unit_id=adjudication_context.independent_unit_id,
            information_cutoffs=adjudication_context.information_cutoffs,
            visibility_ceiling=adjudication_context.visibility_ceiling,
            outcome_access=adjudication_context.outcome_access,
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=scientific_status,
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=tuple(sorted(set(reason_codes))),
        )

    def _evaluate(self, context: TaskContext) -> RunnerResult:
        required_permissions = {
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        }
        if (
            context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or not required_permissions.issubset(context.permissions)
        ):
            raise PermissionError("official evaluator lacks exact reveal authority")
        test_binding = _canonical_input(context, VirtualCellSourceObject)
        expected_test = _source_object(
            self.source_manifest,
            split=VirtualCellSplit.TEST,
            role_fragment="response",
        )
        if test_binding != expected_test or not test_binding.sealed:
            raise ValueError("evaluator test binding differs from sealed custody")
        freeze = _canonical_input(context, PredictionFreezeRecord)
        envelope_port = _one_input(
            context,
            payload_schema=PREDICTION_H5AD_ENVELOPE_SCHEMA,
        )
        if envelope_port.artifact_id != freeze.h5ad_envelope_artifact_id:
            raise ValueError("evaluator prediction artifact differs from its freeze")
        workspace = self.scratch_port.allocate(
            run_id=context.run_id,
            task_id=context.task_id,
            attempt_id=context.attempt_id,
        )
        envelope_path = workspace.reserve_path("prediction-envelope.h5")
        predicted_path = workspace.reserve_path("prediction.h5ad")
        cell_eval_scratch = workspace.reserve_path("cell-eval")
        try:
            with envelope_path.open("xb") as destination:
                self._copy_port(
                    envelope_port,
                    destination,
                    expected_sha256=freeze.h5ad_envelope_sha256,
                )
            with (
                envelope_path.open("rb") as source,
                predicted_path.open("xb") as destination,
            ):
                inner_sha256 = extract_h5ad_envelope(
                    envelope_stream=source,
                    destination=destination,
                )
            if inner_sha256 != freeze.inner_h5ad_sha256:
                raise ValueError(
                    "extracted prediction differs from the frozen commitment"
                )
            with predicted_path.open("rb") as predicted_stream:
                predicted = read_anndata_h5ad_stream(predicted_stream)
            with self.source_port.open_object(
                expected_test,
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            ) as test_stream:
                real = read_anndata_h5ad_stream(test_stream)
            run = evaluate_official_2025(
                predicted=predicted,
                real=real,
                metric_contract=self.config.metric_contract,
                scratch_directory=cell_eval_scratch,
                num_threads=min(8, context.resource_budget.cpu_cores),
                control_label=self.config.response_summary.comparator_label,
                perturbation_column=self.config.response_summary.target_field,
            )
            metrics_output = _output_port(context, "official-metrics")
            if metrics_output.logical_artifact_id is None:
                raise ValueError(
                    "official metric output lacks logical artifact identity"
                )
            evaluation = OfficialEvaluationRecord(
                evaluation_id="evaluation.virtual-cell-2025-official",
                prediction_commitment_sha256=freeze.commitment.fingerprint(),
                scorer_version=run.cell_eval_version,
                pdex_version=run.pdex_version,
                score=run.score,
                per_target_metrics_artifact_id=metrics_output.logical_artifact_id,
                per_target_metrics_sha256=_sha256(run.metrics_arrow),
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            )
            above_baseline = run.score.average_score > 0
            scientific = self._scientific_adjudication(
                context,
                scientific_status=(
                    ScientificStatus.SUPPORTED
                    if above_baseline
                    else ScientificStatus.NOT_SUPPORTED
                ),
                reason_codes=(
                    "VCC_OFFICIAL_EVALUATION_COMPLETE",
                    (
                        "VCC_SCORE_ABOVE_OFFICIAL_BASELINE"
                        if above_baseline
                        else "VCC_SCORE_NOT_ABOVE_OFFICIAL_BASELINE"
                    ),
                    "VCC_ADMISSION_NOT_EVALUATED_BY_LEADERBOARD_SCORER",
                ),
            )
            outputs = (
                _json_output(context, "official-evaluation", evaluation),
                _bytes_output(
                    context,
                    "official-metrics",
                    run.metrics_arrow,
                    payload_schema=OFFICIAL_METRICS_TABLE_SCHEMA,
                ),
                _bytes_output(
                    context,
                    "scientific-adjudication",
                    encode_scientific_adjudication(
                        scientific,
                        payload_schema=ScientificAdjudicationRecord.SCHEMA,
                    ),
                    payload_schema=ScientificAdjudicationRecord.SCHEMA,
                ),
            )
        finally:
            workspace.close()
        return _result(
            outputs,
            "vcc-cell-eval-0.6.6-pdex-0.1.26",
            "vcc-evaluator-only-test-open",
            "vcc-exact-final-normalization",
        )

    def _place(self, context: TaskContext) -> RunnerResult:
        if (
            context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or CapabilityPermission.READ_OUTCOME_VISIBLE not in context.permissions
        ):
            raise PermissionError(
                "placement requires post-score outcome-visible access"
            )
        freeze = _canonical_input(context, PredictionFreezeRecord)
        evaluation = _canonical_input(context, OfficialEvaluationRecord)
        snapshot = _canonical_input(context, LeaderboardSnapshot)
        if snapshot.fingerprint() != self.config.leaderboard_snapshot_sha256:
            raise ValueError("placement leaderboard differs from the frozen config")
        if evaluation.prediction_commitment_sha256 != freeze.commitment.fingerprint():
            raise ValueError(
                "placement evaluation belongs to another prediction commitment"
            )
        metric_sha256 = self.config.metric_contract.fingerprint()
        if freeze.commitment.metric_contract_sha256 != metric_sha256:
            raise ValueError("placement metric differs from the prediction commitment")
        placement = adjudicate_placement(
            adjudication_id="placement.virtual-cell-2025-official-final",
            prediction_commitment_id=freeze.commitment.commitment_id,
            metric_contract_sha256=metric_sha256,
            leaderboard_snapshot_sha256=snapshot.fingerprint(),
            candidate_score=evaluation.score.average_score,
            snapshot=snapshot,
        )
        return _result(
            (_json_output(context, "placement-adjudication", placement),),
            "vcc-leaderboard-lineage-post-score-only",
            "vcc-partial-field-placement-bounded",
        )

    def _report(self, context: TaskContext) -> RunnerResult:
        if any(
            value.output_id
            in {
                "development-closeout",
                f"{context.task_id}.development-closeout",
            }
            for value in context.output_ports
        ):
            return self._report_development(context)
        freeze = _canonical_input(context, PredictionFreezeRecord)
        evaluation = _canonical_input(context, OfficialEvaluationRecord)
        placement = _canonical_input(context, PlacementAdjudication)
        if (
            evaluation.prediction_commitment_sha256 != freeze.commitment.fingerprint()
            or placement.prediction_commitment_id != freeze.commitment.commitment_id
            or placement.metric_contract_sha256
            != freeze.commitment.metric_contract_sha256
            or placement.candidate_score != evaluation.score.average_score
        ):
            raise ValueError(
                "closeout prediction, evaluation and placement lineage differ"
            )
        above_baseline = evaluation.score.average_score > 0
        closeout = VirtualCellRunCloseout(
            closeout_id="closeout.virtual-cell-2025-tier-l0",
            prediction_commitment_sha256=freeze.commitment.fingerprint(),
            official_evaluation_sha256=evaluation.fingerprint(),
            placement_adjudication_sha256=placement.fingerprint(),
            operational_status="SUCCEEDED",
            competition_disposition=(
                "SCORED_ABOVE_OFFICIAL_BASELINE"
                if above_baseline
                else "SCORED_NOT_ABOVE_OFFICIAL_BASELINE"
            ),
            placement_disposition=placement.placement_class.value,
            replay_controller_evaluation_disposition="NOT_RUN_SEPARATE_CONTROLLER_BRANCH",
            prospective_physical_controller_evaluation_status="NOT_TESTED",
            reason_codes=tuple(
                sorted(
                    {
                        "AUTHORITATIVE_RANKS_101_337_UNAVAILABLE",
                        "LEADERBOARD_PREDICTION_DOES_NOT_ESTABLISH_ADMISSION_OR_CONTROLLER_USE",
                        "REPLAY_CONTROLLER_USE_REQUIRES_SEPARATE_FROZEN_CONTROLLER_ACT",
                        *placement.reason_codes,
                    }
                )
            ),
        )
        return _result(
            (_json_output(context, "closeout", closeout),),
            'virtual-cell-physical-validation-not-promoted',
            "vcc-placement-lineage-bound",
        )

    def _report_development(self, context: TaskContext) -> RunnerResult:
        if (
            context.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or CapabilityPermission.READ_DEVELOPMENT not in context.permissions
            or CapabilityPermission.READ_OUTCOME_VISIBLE in context.permissions
        ):
            raise PermissionError(
                "development handoff requires development-only outcome access"
            )
        development = _canonical_input(context, ModelDevelopmentRecord)
        panel = _canonical_input(context, FalsifierPanel)
        selection = _canonical_input(context, ModelSelectionRecord)
        freeze = _canonical_input(context, PredictionFreezeRecord)
        selected = tuple(
            value
            for value in development.models
            if value.artifact_id == selection.selected_model_artifact_id
            and value.artifact_sha256 == selection.selected_model_sha256
        )
        if (
            selection.tournament != development.tournament
            or selection.falsifier_panel_sha256 != panel.fingerprint()
            or len(selected) != 1
            or freeze.commitment.model_id != selected[0].model_id
            or freeze.commitment.model_sha256 != selected[0].artifact_sha256
            or freeze.test_outcome_read
            or freeze.commitment.final_outcome_parent_ids
            or freeze.commitment.leaderboard_parent_ids
        ):
            raise ValueError(
                "development handoff lineage differs from the frozen prediction"
            )
        receiver_supported = panel.all_required_pass
        closeout = VirtualCellDevelopmentCloseout(
            closeout_id="closeout.virtual-cell-2025-tier-l0-development",
            model_development_sha256=development.fingerprint(),
            falsifier_panel_sha256=panel.fingerprint(),
            model_selection_sha256=selection.fingerprint(),
            prediction_freeze_sha256=freeze.fingerprint(),
            selected_model_artifact_id=selection.selected_model_artifact_id,
            competition_disposition="COMPETITION_CANDIDATE_FROZEN",
            receiver_admission_disposition=(
                "DEVELOPMENT_SUPPORTED"
                if receiver_supported
                else "DEVELOPMENT_NOT_SUPPORTED"
            ),
            replay_authoring_disposition="FRESH_DATA_DERIVED_SPEC_REQUIRED",
            test_outcome_read=False,
            reason_codes=tuple(
                sorted(
                    {
                        "VCC_DEVELOPMENT_AND_VALIDATION_TOURNAMENT_COMPLETE",
                        "VCC_FINAL_TEST_OUTCOME_UNOPENED",
                        "VCC_FINAL_REVEAL_REQUIRES_SEPARATE_AUTHORITY",
                        "VCC_REPLAY_LOCAL_LAW_ADMISSION_QUERY_AUTHORING_REQUIRES_FRESH_SPEC",
                        (
                            "VCC_RECEIVER_ADMISSION_DEVELOPMENT_SUPPORTED"
                            if receiver_supported
                            else "VCC_RECEIVER_ADMISSION_DEVELOPMENT_NOT_SUPPORTED"
                        ),
                    }
                )
            ),
        )
        scientific = self._scientific_adjudication(
            context,
            scientific_status=(
                ScientificStatus.SUPPORTED
                if receiver_supported
                else ScientificStatus.NOT_SUPPORTED
            ),
            reason_codes=closeout.reason_codes,
        )
        return _result(
            (
                _json_output(context, "development-closeout", closeout),
                _bytes_output(
                    context,
                    "scientific-adjudication",
                    encode_scientific_adjudication(
                        scientific,
                        payload_schema=ScientificAdjudicationRecord.SCHEMA,
                    ),
                    payload_schema=ScientificAdjudicationRecord.SCHEMA,
                ),
            ),
            "vcc-development-to-reveal-handoff",
            "vcc-final-test-outcome-unopened",
            "vcc-fresh-replay-authoring-required",
        )


def canonical_json_record_value_keys(
    record_type: type[CanonicalRecord],
) -> tuple[str, ...]:
    """Static field list used by the runtime provider's semantic contracts."""

    return tuple(sorted(value.name for value in fields(cast(Any, record_type))))


__all__ = [
    "VirtualCellTaskRunner",
    "canonical_json_record_value_keys",
]
