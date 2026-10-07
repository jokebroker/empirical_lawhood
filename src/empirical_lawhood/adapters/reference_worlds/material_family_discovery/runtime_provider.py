"""Path-free runners and static runtime provider for SDCB-SC."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields
from hashlib import sha256
from typing import Any, TypeVar, cast

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryPolicyConfig,
    PolicyDecision,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
    encode_scientific_adjudication,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    WorkerInputPort,
    WorkerOutputPort,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    ExternalInputSource,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)

from .codecs import (
    MATERIAL_CORPUS_TABLE_SCHEMA,
    POLICY_HISTORY_TABLE_SCHEMA,
    WORLD_POLICY_TABLE_SCHEMA,
    WORLD_TRUTH_TABLE_SCHEMA,
    decode_material_corpus,
    decode_world_truth,
)
from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, MaterialFamilyDiscoveryPhase, SourceQualification
from .descriptors import (
    decode_adjudication_config,
    decode_family_config,
    decode_policy_config,
    decode_world_config,
)
from .records import PolicyHistoryPrefix, MaterialFamilyDiscoveryChildDisposition, MaterialFamilyDiscoveryEvaluationFreeze, MaterialFamilyDiscoveryPhaseAdjudication, MaterialFamilyDiscoveryPhaseCloseout, MaterialFamilySourceAssessment, MaterialFamilyDiscoveryTruthControlEvaluation, MaterialFamilyDiscoveryWorldBuildConfig, MaterialFamilyDiscoveryWorldManifest, MaterialFamilyDiscoveryWorldPolicyEvaluation
from .protocol import (
    CLOSEOUT_STEP,
    CORPUS_INPUT_ID,
    config_ref,
)
from .registration import (
    ADJUDICATOR_CAPABILITY_KEY,
    CAPABILITY_VERSION,
    FREEZE_CAPABILITY_KEY,
    RECEIVER_CAPABILITY_KEY,
    REPORTER_CAPABILITY_KEY,
    SOURCE_CAPABILITY_KEY,
    TRUTH_CONTROL_CAPABILITY_KEY,
    WORLD_CAPABILITY_KEY,
    WORLD_EVALUATOR_CAPABILITY_KEY,
    policy_capability_key,
)
from .workflow import (
    adjudicate_phase,
    build_world_artifacts,
    close_phase,
    evaluate_truth_controls,
    evaluate_world_policy,
    freeze_policy_histories,
    qualify_corpus_binding,
    reveal_committed_batch,
    select_committed_batch,
    validate_frozen_policy_history,
)


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)
_JSON_TYPES: tuple[type[CanonicalRecord], ...] = (
    PolicyDecision,
    PolicyHistoryPrefix,
    MaterialFamilyDiscoveryEvaluationFreeze,
    MaterialFamilyDiscoveryPhaseAdjudication,
    MaterialFamilyDiscoveryPhaseCloseout,
    MaterialFamilySourceAssessment,
    MaterialFamilyDiscoveryTruthControlEvaluation,
    MaterialFamilyDiscoveryWorldManifest,
    MaterialFamilyDiscoveryWorldPolicyEvaluation,
    ScientificAdjudicationRecord,
)
_TABLE_SCHEMAS = {
    MATERIAL_CORPUS_TABLE_SCHEMA,
    POLICY_HISTORY_TABLE_SCHEMA,
    WORLD_POLICY_TABLE_SCHEMA,
    WORLD_TRUTH_TABLE_SCHEMA,
}


class _CorpusInputSource(ExternalInputSource):
    """One exact, single-use, chunked corpus source; paths never enter workers."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._consumed = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if self._consumed or maximum_chunk_bytes <= 0:
            raise ValueError("SC corpus input source is invalid or already consumed")
        self._consumed = True
        for offset in range(0, len(self._payload), maximum_chunk_bytes):
            yield self._payload[offset : offset + maximum_chunk_bytes]

    def close(self) -> None:
        return


def _one_input(context: TaskContext, schema: str) -> WorkerInputPort:
    matches = tuple(value for value in context.input_ports if value.payload_schema == schema)
    if len(matches) != 1:
        raise ValueError(f"{context.task_id} requires one exact {schema} input")
    return matches[0]


def _canonical_inputs(
    context: TaskContext,
    record_type: type[_RecordT],
) -> tuple[_RecordT, ...]:
    return tuple(
        decode_canonical_bytes(
            port.read(),
            record_type,
            maximum_bytes=max(1, port.size_bytes),
        )
        for port in context.input_ports
        if port.payload_schema == record_type.SCHEMA
    )


def _one_record(context: TaskContext, record_type: type[_RecordT]) -> _RecordT:
    values = _canonical_inputs(context, record_type)
    if len(values) != 1:
        raise ValueError(f"{context.task_id} requires one {record_type.__name__}")
    return values[0]


def _optional_record(context: TaskContext, record_type: type[_RecordT]) -> _RecordT | None:
    values = _canonical_inputs(context, record_type)
    if len(values) > 1:
        raise ValueError(f"{context.task_id} contains repeated {record_type.__name__}")
    return values[0] if values else None


def _output_port(context: TaskContext, output_id: str) -> WorkerOutputPort:
    qualified = f"{context.task_id}.{output_id}"
    matches = tuple(
        value for value in context.output_ports if value.output_id in {output_id, qualified}
    )
    if len(matches) != 1:
        raise ValueError(f"{context.task_id} lacks output {output_id}")
    return matches[0]


def _json_output(
    context: TaskContext,
    output_id: str,
    record: CanonicalRecord,
) -> TaskOutputPayload:
    port = _output_port(context, output_id)
    if port.payload_schema != record.SCHEMA:
        raise ValueError("SC canonical output schema differs")
    return TaskOutputPayload(output_id=port.output_id, payload=record.canonical_bytes())


def _bytes_output(
    context: TaskContext,
    output_id: str,
    payload: bytes,
    schema: str,
) -> TaskOutputPayload:
    port = _output_port(context, output_id)
    if port.payload_schema != schema:
        raise ValueError("SC byte output schema differs")
    return TaskOutputPayload(output_id=port.output_id, payload=payload)


def _result(outputs: tuple[TaskOutputPayload, ...], *checks: str) -> RunnerResult:
    return RunnerResult(
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        checks=tuple(
            ReceiptCheck(check_id=value, passed=True, reason_codes=()) for value in sorted(checks)
        ),
    )


class MaterialFamilyDiscoveryConfigDecoder:
    """Exact config decoder for one static capability binding."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        configs: tuple[CanonicalRecord, ...],
    ) -> None:
        self.manifest = manifest
        self._configs = {value.fingerprint(): value for value in configs}
        if len(self._configs) != len(configs):
            raise ValueError("SC registered configs repeat")

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != self.manifest.config_schema:
            raise ValueError("SC config schema differs from capability manifest")
        decoder = {
            MaterialFamilyConfig.SCHEMA: decode_family_config,
            MaterialFamilyDiscoveryAdjudicationConfig.SCHEMA: decode_adjudication_config,
            MaterialFamilyDiscoveryWorldBuildConfig.SCHEMA: decode_world_config,
            DiscoveryPolicyConfig.SCHEMA: decode_policy_config,
        }.get(expected_schema)
        if decoder is None:
            raise ValueError("SC capability has no closed config decoder")
        value = decoder(payload)
        if value.fingerprint() not in self._configs:
            raise ValueError("SC config bytes are not in the registered phase closure")


def material_family_config_decoders(
    *,
    registry: CapabilityRegistry,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
    policies: tuple[DiscoveryPolicyConfig, ...],
    world_configs: tuple[MaterialFamilyDiscoveryWorldBuildConfig, ...],
) -> tuple[MaterialFamilyDiscoveryConfigDecoder, ...]:
    configs_by_schema: dict[str, tuple[CanonicalRecord, ...]] = {
        MaterialFamilyConfig.SCHEMA: (family_config,),
        MaterialFamilyDiscoveryAdjudicationConfig.SCHEMA: (adjudication_config,),
        DiscoveryPolicyConfig.SCHEMA: policies,
        MaterialFamilyDiscoveryWorldBuildConfig.SCHEMA: world_configs,
    }
    return tuple(
        MaterialFamilyDiscoveryConfigDecoder(manifest, configs_by_schema[manifest.config_schema])
        for manifest in registry.capabilities
    )


class MaterialFamilyDiscoveryTaskRunner:
    """One closed implementation, dispatched only by its static manifest."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        family_config: MaterialFamilyConfig,
        source_manifest: MaterialSourceManifest,
        source_qualification: SourceQualification,
        policies: tuple[DiscoveryPolicyConfig, ...],
    ) -> None:
        self.manifest = manifest
        self.family_config = family_config
        self.source_manifest = source_manifest
        self.source_qualification = source_qualification
        self.policies_by_hash = {value.fingerprint(): value for value in policies}

    def _config(self, context: TaskContext) -> CanonicalRecord:
        port = next(
            (
                value
                for value in context.input_ports
                if value.artifact_id == context.config.artifact_id
            ),
            None,
        )
        if port is None:
            raise ValueError("SC task lacks its exact config artifact")
        decoder = {
            MaterialFamilyConfig.SCHEMA: decode_family_config,
            MaterialFamilyDiscoveryAdjudicationConfig.SCHEMA: decode_adjudication_config,
            MaterialFamilyDiscoveryWorldBuildConfig.SCHEMA: decode_world_config,
            DiscoveryPolicyConfig.SCHEMA: decode_policy_config,
        }[context.config.config_schema]
        value = decoder(port.read())
        if value.fingerprint() != context.config.content_sha256:
            raise ValueError("SC task config differs from frozen reference")
        return value

    def execute(self, context: TaskContext) -> RunnerResult:
        config = self._config(context)
        key = self.manifest.capability_key
        if key == SOURCE_CAPABILITY_KEY:
            return self._source(context, cast(MaterialFamilyConfig, config))
        if key == WORLD_CAPABILITY_KEY:
            return self._world(context, cast(MaterialFamilyDiscoveryWorldBuildConfig, config))
        if key in {
            policy_capability_key(value.policy_kind) for value in self.policies_by_hash.values()
        }:
            return self._select(context, cast(DiscoveryPolicyConfig, config))
        if key == RECEIVER_CAPABILITY_KEY:
            return self._receive(context, cast(DiscoveryPolicyConfig, config))
        if key == FREEZE_CAPABILITY_KEY:
            return self._freeze(context, cast(MaterialFamilyConfig, config))
        if key == WORLD_EVALUATOR_CAPABILITY_KEY:
            return self._evaluate(context, cast(DiscoveryPolicyConfig, config))
        if key == TRUTH_CONTROL_CAPABILITY_KEY:
            return self._truth_control(context, cast(DiscoveryPolicyConfig, config))
        if key == ADJUDICATOR_CAPABILITY_KEY:
            return self._adjudicate(context, cast(MaterialFamilyDiscoveryAdjudicationConfig, config))
        if key == REPORTER_CAPABILITY_KEY:
            return self._closeout(context, cast(MaterialFamilyDiscoveryAdjudicationConfig, config))
        raise NotImplementedError("unregistered SC capability")

    def _corpus(self, context: TaskContext):  # type: ignore[no-untyped-def]
        return decode_material_corpus(
            _one_input(context, MATERIAL_CORPUS_TABLE_SCHEMA).read(),
            qualification=self.source_qualification,
        )

    def _source(self, context: TaskContext, config: MaterialFamilyConfig) -> RunnerResult:
        if config != self.family_config:
            raise ValueError("SC source family config differs")
        manifest = _one_record(context, MaterialSourceManifest)
        qualification = _one_record(context, SourceQualification)
        if manifest != self.source_manifest or qualification != self.source_qualification:
            raise ValueError("SC source records differ from the registered source closure")
        ready = qualify_corpus_binding(
            manifest=manifest,
            qualification=qualification,
            corpus=self._corpus(context),
        )
        return _result(
            (_json_output(context, "source-ready", ready),),
            'material-family-exact-source-byte-and-qualification-binding',
            'material-family-missing-operands-typed',
        )

    def _world(self, context: TaskContext, config: MaterialFamilyDiscoveryWorldBuildConfig) -> RunnerResult:
        ready = _one_record(context, MaterialFamilySourceAssessment)
        manifest, policy, truth = build_world_artifacts(
            corpus=self._corpus(context),
            family_config=self.family_config,
            world_config=config,
            source_ready=ready,
        )
        return _result(
            (
                _json_output(context, "world-manifest", manifest),
                _bytes_output(context, "world-policy", policy, WORLD_POLICY_TABLE_SCHEMA),
                _bytes_output(context, "world-truth", truth, WORLD_TRUTH_TABLE_SCHEMA),
            ),
            'material-family-complete-family-holdout',
            'material-family-family-and-duplicate-leakage-absent',
            'material-family-unlabelled-not-negative',
        )

    def _select(self, context: TaskContext, policy: DiscoveryPolicyConfig) -> RunnerResult:
        world = _one_record(context, MaterialFamilyDiscoveryWorldManifest)
        prior = _optional_record(context, PolicyHistoryPrefix)
        decision = select_committed_batch(
            world=world,
            policy_payload=_one_input(context, WORLD_POLICY_TABLE_SCHEMA).read(),
            policy=policy,
            prior=prior,
            round_index=int(context.task_id.rsplit("r", 1)[1]),
        )
        return _result(
            (_json_output(context, "policy-decision", decision),),
            'material-family-deterministic-policy-replay',
            'material-family-no-repeat-query',
            'material-family-only-visible-prefix-consumed',
        )

    def _receive(self, context: TaskContext, policy: DiscoveryPolicyConfig) -> RunnerResult:
        prefix = reveal_committed_batch(
            world=_one_record(context, MaterialFamilyDiscoveryWorldManifest),
            policy_payload=_one_input(context, WORLD_POLICY_TABLE_SCHEMA).read(),
            truth=decode_world_truth(_one_input(context, WORLD_TRUTH_TABLE_SCHEMA).read()),
            policy=policy,
            decision=_one_record(context, PolicyDecision),
            prior=_optional_record(context, PolicyHistoryPrefix),
        )
        return _result(
            (_json_output(context, "policy-prefix", prefix),),
            'material-family-batch-commitment-before-receiver-reveal',
            'material-family-policy-private-prefix-only',
            'material-family-whole-batch-cost-charged',
        )

    def _evaluate(self, context: TaskContext, policy: DiscoveryPolicyConfig) -> RunnerResult:
        world = _one_record(context, MaterialFamilyDiscoveryWorldManifest)
        prefix = _one_record(context, PolicyHistoryPrefix)
        if prefix.policy_config_sha256 != policy.fingerprint():
            raise ValueError("SC evaluator policy config differs from history")
        freeze = _optional_record(context, MaterialFamilyDiscoveryEvaluationFreeze)
        if self.family_config.phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT:
            if freeze is not None:
                raise ValueError("SC development evaluator received an evaluation freeze")
        elif freeze is None:
            raise ValueError("SC evaluator lacks its pre-reveal history freeze")
        else:
            validate_frozen_policy_history(
                world=world,
                prefix=prefix,
                freeze=freeze,
            )
        evaluation, history = evaluate_world_policy(
            world=world,
            truth=decode_world_truth(_one_input(context, WORLD_TRUTH_TABLE_SCHEMA).read()),
            prefix=prefix,
        )
        return _result(
            (
                _bytes_output(context, "policy-history", history, POLICY_HISTORY_TABLE_SCHEMA),
                _json_output(context, "world-policy-evaluation", evaluation),
            ),
            'material-family-evaluator-only-family-truth',
            'material-family-family-unit-estimand',
            'material-family-right-censoring-retained',
        )

    def _freeze(self, context: TaskContext, config: MaterialFamilyConfig) -> RunnerResult:
        if config != self.family_config:
            raise ValueError("SC freeze family config differs")
        freeze = freeze_policy_histories(
            family_config=config,
            prefixes=_canonical_inputs(context, PolicyHistoryPrefix),
        )
        return _result(
            (_json_output(context, "evaluation-freeze", freeze),),
            'material-family-complete-terminal-history-matrix',
            'material-family-matched-policy-roster-frozen-before-reveal',
        )

    def _truth_control(self, context: TaskContext, policy: DiscoveryPolicyConfig) -> RunnerResult:
        result = evaluate_truth_controls(policy)
        return _result(
            (_json_output(context, "truth-control-evaluation", result),),
            'material-family-empty-support-mandatory-hold',
            'material-family-exhausted-support-mandatory-hold',
            'material-family-zero-truth-control-false-admission',
        )

    def _adjudicate(self, context: TaskContext, config: MaterialFamilyDiscoveryAdjudicationConfig) -> RunnerResult:
        result = adjudicate_phase(
            family_config=self.family_config,
            adjudication_config=config,
            policies=tuple(
                sorted(self.policies_by_hash.values(), key=lambda value: value.config_id)
            ),
            evaluations=_canonical_inputs(context, MaterialFamilyDiscoveryWorldPolicyEvaluation),
            truth_control=_one_record(context, MaterialFamilyDiscoveryTruthControlEvaluation),
        )
        return _result(
            (_json_output(context, "phase-adjudication", result),),
            'material-family-paired-family-bootstrap',
            'material-family-primary-constrained-bo-gate',
            'material-family-truth-control-evidence-consumed',
        )

    @staticmethod
    def _scientific(
        context: TaskContext,
        adjudication: MaterialFamilyDiscoveryPhaseAdjudication,
    ) -> ScientificAdjudicationRecord:
        scientific_context = context.scientific_adjudication_context
        if scientific_context is None:
            raise ValueError("SC closeout lacks scientific adjudication context")
        mapping = {
            MaterialFamilyDiscoveryChildDisposition.SUPPORTED: (
                AdjudicationEvaluability.EVALUABLE,
                ScientificStatus.SUPPORTED,
                AdmissionStatus.NOT_EVALUATED,
            ),
            MaterialFamilyDiscoveryChildDisposition.NOT_SUPPORTED: (
                AdjudicationEvaluability.EVALUABLE,
                ScientificStatus.NOT_SUPPORTED,
                AdmissionStatus.NOT_EVALUATED,
            ),
            MaterialFamilyDiscoveryChildDisposition.DEVELOPMENT_ONLY: (
                AdjudicationEvaluability.EVALUABLE,
                ScientificStatus.PARTIAL,
                AdmissionStatus.NOT_EVALUATED,
            ),
            MaterialFamilyDiscoveryChildDisposition.UNEVALUABLE: (
                AdjudicationEvaluability.UNEVALUABLE,
                ScientificStatus.UNEVALUABLE,
                AdmissionStatus.UNEVALUABLE,
            ),
            MaterialFamilyDiscoveryChildDisposition.PREREQUISITE_NONATTEMPT: (
                AdjudicationEvaluability.UNEVALUABLE,
                ScientificStatus.UNEVALUABLE,
                AdmissionStatus.UNEVALUABLE,
            ),
        }
        evaluability, scientific_status, admission = mapping[adjudication.disposition]
        output_ids = tuple(
            sorted(
                value.logical_artifact_id
                for value in context.output_ports
                if value.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError("SC closeout outputs lack logical identities")
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=scientific_context.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=output_ids,
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=scientific_context.evidence_world_id,
            evidence_world_kind=scientific_context.evidence_world_kind,
            relation=scientific_context.relation,
            independent_unit_id=scientific_context.independent_unit_id,
            information_cutoffs=scientific_context.information_cutoffs,
            visibility_ceiling=scientific_context.visibility_ceiling,
            outcome_access=scientific_context.outcome_access,
            evaluability=evaluability,
            scientific_status=scientific_status,
            admission_status=admission,
            reason_codes=adjudication.reason_codes,
        )

    def _closeout(self, context: TaskContext, _config: MaterialFamilyDiscoveryAdjudicationConfig) -> RunnerResult:
        adjudication = _one_record(context, MaterialFamilyDiscoveryPhaseAdjudication)
        closeout = close_phase(adjudication)
        scientific = self._scientific(context, adjudication)
        return _result(
            (
                _json_output(context, "phase-closeout", closeout),
                _bytes_output(
                    context,
                    "scientific-adjudication",
                    encode_scientific_adjudication(
                        scientific,
                        payload_schema=ScientificAdjudicationRecord.SCHEMA,
                    ),
                    ScientificAdjudicationRecord.SCHEMA,
                ),
            ),
            'material-family-complete-world-policy-result-matrix',
            'material-family-negative-and-unevaluable-terminal-preserved',
        )


def _value_keys(record_type: type[CanonicalRecord]) -> tuple[str, ...]:
    return tuple(sorted(value.name for value in fields(cast(Any, record_type))))


class MaterialFamilyDiscoveryCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Bind one exact phase registry to qualified external bytes and runners."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        family_config: MaterialFamilyConfig,
        adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
        source_manifest: MaterialSourceManifest,
        source_qualification: SourceQualification,
        corpus_payload: bytes,
        corpus_sha256: str,
        corpus_size_bytes: int,
        policies: tuple[DiscoveryPolicyConfig, ...],
        world_configs: tuple[MaterialFamilyDiscoveryWorldBuildConfig, ...],
    ) -> None:
        if family_config.source_manifest_sha256 != source_manifest.fingerprint():
            raise ValueError("SC provider source manifest differs")
        if family_config.source_qualification_sha256 != source_qualification.fingerprint():
            raise ValueError("SC provider source qualification differs")
        if adjudication_config.family_config_sha256 != family_config.fingerprint():
            raise ValueError("SC provider adjudication config differs")
        if {value.fingerprint() for value in policies} != set(family_config.policy_config_sha256s):
            raise ValueError("SC provider policy roster differs")
        if corpus_size_bytes <= 0:
            raise ValueError("SC corpus size must be positive")
        if (
            len(corpus_payload) != corpus_size_bytes
            or sha256(corpus_payload).hexdigest() != corpus_sha256
        ):
            raise ValueError("SC corpus payload differs from its exact identity")
        self.registry = registry
        self.family_config = family_config
        self.adjudication_config = adjudication_config
        self.source_manifest = source_manifest
        self.source_qualification = source_qualification
        self.corpus_payload = corpus_payload
        self.corpus_sha256 = corpus_sha256
        self.corpus_size_bytes = corpus_size_bytes
        self.policies = policies
        self.world_configs = world_configs
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)

    def _validate_registry(self, registry: CapabilityRegistry) -> None:
        if registry != self.registry or registry.fingerprint() != self.registry_sha256:
            raise ValueError("SC runtime registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[MaterialFamilyDiscoveryTaskRunner, ...]:
        self._validate_registry(registry)
        if source_records:
            raise ValueError("SC provider accepts no unbound issued source records")
        return tuple(
            MaterialFamilyDiscoveryTaskRunner(
                manifest=manifest,
                family_config=self.family_config,
                source_manifest=self.source_manifest,
                source_qualification=self.source_qualification,
                policies=self.policies,
            )
            for manifest in registry.capabilities
        )

    @staticmethod
    def _parent(
        record_id: str,
        schema: str,
        digest: str,
        visibility: VisibilityCeiling,
        access: OutcomeAccess,
    ) -> ArtifactLineageParent:
        return ArtifactLineageParent(
            identity=ObjectIdentity(
                object_id=record_id,
                object_schema=schema,
                object_version="1.0.0",
                object_fingerprint=digest,
            ),
            visibility_ceiling=visibility,
            outcome_access=access,
        )

    def _small_payload(
        self,
        *,
        logical_id: str,
        record: CanonicalRecord,
        access: OutcomeAccess,
        visibility: VisibilityCeiling,
    ) -> ExternalInputPayload:
        record_id = (
            getattr(record, "config_id", None)
            or getattr(record, "manifest_id", None)
            or getattr(record, "qualification_id", None)
        )
        if not isinstance(record_id, str):
            raise ValueError("SC external canonical input lacks a stable identity")
        parent = self._parent(
            record_id,
            record.SCHEMA,
            record.fingerprint(),
            visibility,
            access,
        )
        return ExternalInputPayload.from_bytes(
            logical_artifact_id=logical_id,
            payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            payload=record.canonical_bytes(),
            visibility_ceiling=visibility,
            outcome_access=access,
            parent_visibility_ceilings=(visibility,),
            lineage_parents=(parent,),
            logical_content_sha256=record.fingerprint(),
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if source_records or plan.registry_sha256 != self.registry_sha256:
            raise ValueError("SC execution input binding differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: tuple[CanonicalRecord, ...] = (
            self.family_config,
            self.adjudication_config,
            self.source_manifest,
            self.source_qualification,
            *self.policies,
            *self.world_configs,
        )
        values: dict[str, ExternalInputPayload] = {}
        for record in records:
            logical_id = (
                self.source_manifest.manifest_id
                if isinstance(record, MaterialSourceManifest)
                else self.source_qualification.qualification_id
                if isinstance(record, SourceQualification)
                else config_ref(record).artifact_id
            )
            if logical_id not in specs:
                continue
            access = (
                OutcomeAccess.DEVELOPMENT_VISIBLE
                if isinstance(record, SourceQualification)
                else OutcomeAccess.OUTCOME_BLIND
            )
            visibility = (
                VisibilityCeiling.OUTCOME_VISIBLE
                if isinstance(record, SourceQualification)
                else VisibilityCeiling.PROSPECTIVE
            )
            values[logical_id] = self._small_payload(
                logical_id=logical_id,
                record=record,
                access=access,
                visibility=visibility,
            )
        if CORPUS_INPUT_ID in specs:
            parent = self._parent(
                CORPUS_INPUT_ID,
                MATERIAL_CORPUS_TABLE_SCHEMA,
                self.corpus_sha256,
                VisibilityCeiling.OUTCOME_VISIBLE,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
            )
            values[CORPUS_INPUT_ID] = ExternalInputPayload(
                logical_artifact_id=CORPUS_INPUT_ID,
                payload_schema=MATERIAL_CORPUS_TABLE_SCHEMA,
                profile=ArtifactProfile.ARROW_IPC,
                media_type="application/vnd.apache.arrow.file",
                source=_CorpusInputSource(self.corpus_payload),
                size_bytes=self.corpus_size_bytes,
                source_sha256=self.corpus_sha256,
                maximum_bytes=self.corpus_size_bytes,
                maximum_chunk_bytes=MAX_EXTERNAL_INPUT_CHUNK_BYTES,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
                lineage_parents=(parent,),
                logical_content_sha256=self.corpus_sha256,
            )
        if set(values) != set(specs):
            raise ValueError(
                f"SC external input closure differs: missing={sorted(set(specs) - set(values))}"
            )
        for logical_id, payload in values.items():
            spec = specs[logical_id]
            if (
                spec.expected_content_sha256 not in {None, payload.source_sha256}
                or spec.expected_payload_schema not in {None, payload.payload_schema}
                or spec.expected_media_type not in {None, payload.media_type}
                or spec.expected_size_bytes not in {None, payload.size_bytes}
                or spec.expected_visibility_ceiling not in {None, payload.visibility_ceiling}
                or spec.expected_outcome_access not in {None, payload.outcome_access}
            ):
                raise ValueError(f"SC external input contract differs for {logical_id}")
        return tuple(values[key] for key in sorted(values))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._validate_registry(registry)
        json_types = {value.SCHEMA: value for value in _JSON_TYPES}
        contracts = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                if schema in json_types:
                    contracts.append(
                        CapabilityOutputSemanticContract.from_manifest(
                            manifest,
                            payload_schema=schema,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            top_level_keys=("schema", "value", "version"),
                            value_keys=_value_keys(json_types[schema]),
                        )
                    )
                elif schema in _TABLE_SCHEMAS:
                    contracts.append(
                        CapabilityOutputSemanticContract.from_manifest(
                            manifest,
                            payload_schema=schema,
                            profile=ArtifactProfile.ARROW_IPC,
                        )
                    )
        values = tuple(sorted(contracts, key=lambda value: value.key))
        if execution_plan is None:
            return values
        planned = {
            (
                task.capability.capability_key,
                task.capability.capability_version,
                output.payload_schema,
                output.profile,
            )
            for task in execution_plan.tasks
            for output in task.outputs
        }
        return tuple(value for value in values if value.key in planned)

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        self._validate_registry(registry)
        return ScientificAdjudicationOutputContract(
            capability_key=REPORTER_CAPABILITY_KEY,
            capability_version=CAPABILITY_VERSION,
            output_id=f"{CLOSEOUT_STEP}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = [
    'MaterialFamilyDiscoveryCampaignRuntimeProvider',
    'MaterialFamilyDiscoveryConfigDecoder',
    'MaterialFamilyDiscoveryTaskRunner',
    'material_family_config_decoders',
]
