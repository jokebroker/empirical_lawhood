"""Compiler-owned protocol and exact scientific DAG for SDCB-SC."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryPolicyConfig,
    PolicyDecision,
    PolicyKind,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)

from .codecs import (
    MATERIAL_CORPUS_TABLE_SCHEMA,
    POLICY_HISTORY_TABLE_SCHEMA,
    WORLD_POLICY_TABLE_SCHEMA,
    WORLD_TRUTH_TABLE_SCHEMA,
)
from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, MaterialFamilyDiscoveryPhase, MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION, SourceQualification
from .records import PolicyHistoryPrefix, MaterialFamilyDiscoveryEvaluationFreeze, MaterialFamilyDiscoveryPhaseAdjudication, MaterialFamilyDiscoveryPhaseCloseout, MaterialFamilySourceAssessment, MaterialFamilyDiscoveryTruthControlEvaluation, MaterialFamilyDiscoveryWorldBuildConfig, MaterialFamilyDiscoveryWorldManifest, MaterialFamilyDiscoveryWorldPolicyEvaluation
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


SOURCE_MANIFEST_INPUT_ID = 'source.material-family-discovery-manifest'
SOURCE_QUALIFICATION_INPUT_ID = 'source.material-family-discovery-qualification'
CORPUS_INPUT_ID = 'source.material-family-discovery-canonical-corpus'
SOURCE_STEP = 'material-family-qualify-source'
FREEZE_STEP = 'material-family-freeze-policy-histories'
TRUTH_CONTROL_STEP = 'material-family-control-local-law'
ADJUDICATE_STEP = 'material-family-adjudicate-phase'
CLOSEOUT_STEP = 'material-family-closeout-phase'


def _config_artifact_id(record: CanonicalRecord) -> str:
    config_id = getattr(record, "config_id", None)
    if not isinstance(config_id, str):
        raise ValueError("SC capability config lacks config_id")
    return f"config-artifact.{config_id}"


def config_ref(record: CanonicalRecord) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=getattr(record, "config_id"),
        config_schema=record.SCHEMA,
        config_schema_sha256=sha256(record.SCHEMA.encode("utf-8")).hexdigest(),
        content_sha256=record.fingerprint(),
        artifact_id=_config_artifact_id(record),
    )


def world_configs(family_config: MaterialFamilyConfig) -> tuple[MaterialFamilyDiscoveryWorldBuildConfig, ...]:
    families = (
        family_config.development_family_ids
        if family_config.phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT
        else family_config.evaluation_family_ids
        if family_config.phase is MaterialFamilyDiscoveryPhase.EVALUATION
        else tuple(
            sorted(
                {
                    *family_config.development_family_ids,
                    *family_config.evaluation_family_ids,
                }
            )
        )
    )
    if not families:
        raise ValueError("SC phase has no independent family worlds")
    return tuple(
        MaterialFamilyDiscoveryWorldBuildConfig(
            config_id=f"config.material-family-discovery-world-{index:02d}-{family_config.phase.value.lower()}",
            phase=family_config.phase,
            family_config_sha256=family_config.fingerprint(),
            family_id=family_id,
        )
        for index, family_id in enumerate(families)
    )


def _json(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        filename_suffix=".json",
    )


def _arrow(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.ARROW_IPC,
        media_type="application/vnd.apache.arrow.file",
        filename_suffix=".arrow",
    )


def _step(
    *,
    registry: CapabilityRegistry,
    config: CanonicalRecord,
    step_id: str,
    stage: ScientificStage,
    capability_key: str,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    outcome_access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
    obligation_ids: tuple[str, ...],
) -> ProtocolStepTemplate:
    manifest = registry.resolve(capability_key, CAPABILITY_VERSION)
    locks = (
        ('resource.material-family-botorch-cpu',)
        if capability_key == policy_capability_key(PolicyKind.BOTORCH_DISCRETE_UCB)
        else (f"resource.{step_id}",)
    )
    return ProtocolStepTemplate(
        step_id=step_id,
        stage=stage,
        capability_key=capability_key,
        capability_version=CAPABILITY_VERSION,
        config=config_ref(config),
        dependency_step_ids=tuple(sorted(dependencies)),
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=manifest.permissions,
        requested_outcome_access=outcome_access,
        visibility_ceiling=visibility,
        resource_budget=manifest.resource_ceiling,
        resource_lock_ids=locks,
        barrier=barrier,
        # A task retry repeats the same deterministic capability against the
        # same content-addressed inputs.  It changes no scientific choice, and
        # prevents a single transient worker/process fault from invalidating a
        # campaign with more than one thousand independently scheduled tasks.
        maximum_attempts=2,
        obligation_ids=tuple(sorted(obligation_ids)),
    )


def _phase_contract(
    phase: MaterialFamilyDiscoveryPhase,
) -> tuple[OutcomeAccess, VisibilityCeiling, ScientificStage, BarrierKind]:
    if phase is MaterialFamilyDiscoveryPhase.EVALUATION:
        return (
            OutcomeAccess.EVALUATION_SEALED,
            # This is a sealed replay over an outcome-visible historical
            # corpus, not a new prospective acquisition.  Keep outcome access
            # sealed while retaining the inherited nonpromoting ceiling.
            VisibilityCeiling.OUTCOME_VISIBLE,
            ScientificStage.EVALUATE,
            BarrierKind.REVEAL,
        )
    if phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT:
        return (
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            # These artifacts derive from an already outcome-visible historical
            # corpus.  Retain that ceiling throughout the development lineage;
            # DEVELOPMENT_ONLY would falsely make the derived decision promotable.
            VisibilityCeiling.OUTCOME_VISIBLE,
            ScientificStage.QUALIFY,
            BarrierKind.NONE,
        )
    return (
        OutcomeAccess.EVALUATION_REVEALED,
        VisibilityCeiling.OUTCOME_VISIBLE,
        ScientificStage.EVALUATE,
        BarrierKind.REVEAL,
    )


def _validate_roster(
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
    policies: tuple[DiscoveryPolicyConfig, ...],
) -> None:
    by_hash = {value.fingerprint(): value for value in policies}
    if len(by_hash) != len(policies) or set(by_hash) != set(family_config.policy_config_sha256s):
        raise ValueError("SC protocol policy roster differs from family config")
    expected_kinds = {
        PolicyKind.STRATIFIED_RANDOM,
        PolicyKind.MAXIMIN,
        PolicyKind.GREEDY_SCALAR_ENSEMBLE,
        PolicyKind.BOTORCH_DISCRETE_UCB,
        PolicyKind.LOCAL_LAW_BOUNDARY,
    }
    if {value.policy_kind for value in policies} != expected_kinds:
        raise ValueError("SC protocol requires the exact five-policy primary roster")
    if adjudication_config.family_config_sha256 != family_config.fingerprint():
        raise ValueError("SC adjudication config differs from family config")
    if adjudication_config.phase is not family_config.phase:
        raise ValueError("SC adjudication phase differs")
    if adjudication_config.method_policy_config_sha256 not in by_hash:
        raise ValueError("SC method policy is absent")
    if adjudication_config.primary_bo_policy_config_sha256 not in by_hash:
        raise ValueError("SC constrained BO policy is absent")
    if (
        by_hash[adjudication_config.method_policy_config_sha256].policy_kind
        is not PolicyKind.LOCAL_LAW_BOUNDARY
    ):
        raise ValueError("SC method policy is not the local-law explorer")
    if (
        by_hash[adjudication_config.primary_bo_policy_config_sha256].policy_kind
        is not PolicyKind.BOTORCH_DISCRETE_UCB
    ):
        raise ValueError("SC primary comparator is not pinned BoTorch")
    rounds = {
        value.total_query_budget // value.batch_size
        for value in policies
        if value.total_query_budget % value.batch_size == 0
    }
    if len(rounds) != 1 or any(value.total_query_budget % value.batch_size for value in policies):
        raise ValueError("SC policies require one exact integral round count")
    if {value.total_query_budget for value in policies} != {adjudication_config.query_budget}:
        raise ValueError("SC policy and adjudication query budgets differ")


def material_family_protocol(
    *,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
    policies: tuple[DiscoveryPolicyConfig, ...],
) -> ProtocolTemplate:
    _validate_roster(family_config, adjudication_config, policies)
    worlds = world_configs(family_config)
    phase_access, visibility, evaluator_stage, evaluator_barrier = _phase_contract(
        family_config.phase
    )
    evaluator_access = (
        OutcomeAccess.EVALUATOR_REVEAL
        if family_config.phase is MaterialFamilyDiscoveryPhase.EVALUATION
        else phase_access
    )
    obligations = experiment.obligations
    falsifier_ids = tuple(value.falsifier_id for value in obligations.falsifiers)
    if not falsifier_ids:
        raise ValueError("SC experiment lacks decisive falsifiers")
    steps: list[ProtocolStepTemplate] = [
        _step(
            registry=registry,
            config=family_config,
            step_id=SOURCE_STEP,
            stage=ScientificStage.PREPARE,
            capability_key=SOURCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=(_json("source-ready", MaterialFamilySourceAssessment.SCHEMA),),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
            obligation_ids=(obligations.validity.validity_id,),
        )
    ]
    ordered_policies = tuple(sorted(policies, key=lambda value: value.config_id))
    rounds = ordered_policies[0].total_query_budget // ordered_policies[0].batch_size
    evaluator_steps: list[str] = []
    terminal_receiver_steps: list[str] = []
    for world_index, world_config in enumerate(worlds):
        world_step = f"material-family-world-{world_index:02d}"
        steps.append(
            _step(
                registry=registry,
                config=world_config,
                step_id=world_step,
                stage=ScientificStage.PREPARE,
                capability_key=WORLD_CAPABILITY_KEY,
                dependencies=(SOURCE_STEP,),
                outputs=(
                    _json("world-manifest", MaterialFamilyDiscoveryWorldManifest.SCHEMA),
                    _arrow("world-policy", WORLD_POLICY_TABLE_SCHEMA),
                    _arrow("world-truth", WORLD_TRUTH_TABLE_SCHEMA),
                ),
                outcome_access=phase_access,
                visibility=visibility,
                barrier=BarrierKind.NONE,
                obligation_ids=(obligations.support.support_id,),
            )
        )
        for policy_index, policy in enumerate(ordered_policies):
            prior_receiver: str | None = None
            for round_index in range(rounds):
                selector_step = (
                    f"material-family-select-w{world_index:02d}-p{policy_index:02d}-r{round_index:02d}"
                )
                receiver_step = (
                    f"material-family-receive-w{world_index:02d}-p{policy_index:02d}-r{round_index:02d}"
                )
                selector_dependencies = (
                    (world_step,) if prior_receiver is None else (world_step, prior_receiver)
                )
                steps.append(
                    _step(
                        registry=registry,
                        config=policy,
                        step_id=selector_step,
                        stage=ScientificStage.DEVELOP,
                        capability_key=policy_capability_key(policy.policy_kind),
                        dependencies=selector_dependencies,
                        outputs=(_json("policy-decision", PolicyDecision.SCHEMA),),
                        outcome_access=phase_access,
                        visibility=visibility,
                        barrier=BarrierKind.NONE,
                        obligation_ids=(obligations.support.support_id,),
                    )
                )
                receiver_dependencies = [world_step, selector_step]
                if prior_receiver is not None:
                    receiver_dependencies.append(prior_receiver)
                steps.append(
                    _step(
                        registry=registry,
                        config=policy,
                        step_id=receiver_step,
                        stage=ScientificStage.ACQUIRE,
                        capability_key=RECEIVER_CAPABILITY_KEY,
                        dependencies=tuple(receiver_dependencies),
                        outputs=(_json("policy-prefix", PolicyHistoryPrefix.SCHEMA),),
                        outcome_access=phase_access,
                        visibility=visibility,
                        barrier=BarrierKind.NONE,
                        obligation_ids=(obligations.closure.closure_id,),
                    )
                )
                prior_receiver = receiver_step
            if prior_receiver is None:  # pragma: no cover - config validation closes this
                raise ValueError("SC protocol contains no query rounds")
            terminal_receiver_steps.append(prior_receiver)
            evaluator_step = f"material-family-evaluate-w{world_index:02d}-p{policy_index:02d}"
            evaluator_steps.append(evaluator_step)
            steps.append(
                _step(
                    registry=registry,
                    config=policy,
                    step_id=evaluator_step,
                    stage=evaluator_stage,
                    capability_key=WORLD_EVALUATOR_CAPABILITY_KEY,
                    dependencies=(
                        (world_step, prior_receiver, FREEZE_STEP)
                        if evaluator_barrier is BarrierKind.REVEAL
                        else (world_step, prior_receiver)
                    ),
                    outputs=(
                        _arrow("policy-history", POLICY_HISTORY_TABLE_SCHEMA),
                        _json("world-policy-evaluation", MaterialFamilyDiscoveryWorldPolicyEvaluation.SCHEMA),
                    ),
                    outcome_access=evaluator_access,
                    visibility=(
                        VisibilityCeiling.OUTCOME_VISIBLE
                        if evaluator_barrier is BarrierKind.REVEAL
                        else visibility
                    ),
                    barrier=evaluator_barrier,
                    obligation_ids=(falsifier_ids[0],),
                )
            )
    if evaluator_barrier is BarrierKind.REVEAL:
        steps.append(
            _step(
                registry=registry,
                config=family_config,
                step_id=FREEZE_STEP,
                stage=ScientificStage.FREEZE,
                capability_key=FREEZE_CAPABILITY_KEY,
                dependencies=tuple(terminal_receiver_steps),
                outputs=(_json("evaluation-freeze", MaterialFamilyDiscoveryEvaluationFreeze.SCHEMA),),
                outcome_access=phase_access,
                visibility=visibility,
                barrier=BarrierKind.FREEZE,
                obligation_ids=(obligations.closure.closure_id,),
            )
        )
    method = next(
        value
        for value in ordered_policies
        if value.fingerprint() == adjudication_config.method_policy_config_sha256
    )
    steps.extend(
        (
            _step(
                registry=registry,
                config=method,
                step_id=TRUTH_CONTROL_STEP,
                stage=ScientificStage.FALSIFY,
                capability_key=TRUTH_CONTROL_CAPABILITY_KEY,
                dependencies=(),
                outputs=(_json("truth-control-evaluation", MaterialFamilyDiscoveryTruthControlEvaluation.SCHEMA),),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility=VisibilityCeiling.PROSPECTIVE,
                barrier=BarrierKind.NONE,
                obligation_ids=(falsifier_ids[-1],),
            ),
            _step(
                registry=registry,
                config=adjudication_config,
                step_id=ADJUDICATE_STEP,
                stage=evaluator_stage,
                capability_key=ADJUDICATOR_CAPABILITY_KEY,
                dependencies=tuple((*evaluator_steps, TRUTH_CONTROL_STEP)),
                outputs=(_json("phase-adjudication", MaterialFamilyDiscoveryPhaseAdjudication.SCHEMA),),
                outcome_access=(
                    OutcomeAccess.EVALUATION_REVEALED
                    if family_config.phase is MaterialFamilyDiscoveryPhase.EVALUATION
                    else phase_access
                ),
                visibility=(
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if evaluator_barrier is BarrierKind.REVEAL
                    else visibility
                ),
                barrier=evaluator_barrier,
                obligation_ids=(obligations.uncertainty.uncertainty_id,),
            ),
            _step(
                registry=registry,
                config=adjudication_config,
                step_id=CLOSEOUT_STEP,
                stage=ScientificStage.REPORT,
                capability_key=REPORTER_CAPABILITY_KEY,
                dependencies=(ADJUDICATE_STEP,),
                outputs=(
                    _json("phase-closeout", MaterialFamilyDiscoveryPhaseCloseout.SCHEMA),
                    _json("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                ),
                outcome_access=(
                    OutcomeAccess.EVALUATION_REVEALED
                    if family_config.phase is MaterialFamilyDiscoveryPhase.EVALUATION
                    else phase_access
                ),
                visibility=(
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if evaluator_barrier is BarrierKind.REVEAL
                    else visibility
                ),
                barrier=BarrierKind.NONE,
                obligation_ids=(obligations.structural_convergence.convergence_id,),
            ),
        )
    )
    return ProtocolTemplate(
        template_id=f"protocol.material-family-discovery-{family_config.phase.value.lower()}",
        template_version=CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _output(protocol: ProtocolTemplate, step_id: str, output_id: str) -> OutputTemplate:
    step = next(value for value in protocol.steps if value.step_id == step_id)
    return next(value for value in step.outputs if value.output_id == output_id)


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
    output = _output(protocol, producer, output_id)
    producer_step = next(value for value in protocol.steps if value.step_id == producer)
    consumer_step = next(value for value in protocol.steps if value.step_id == consumer)
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
        outcome_access=producer_step.requested_outcome_access,
        visibility_ceiling=producer_step.visibility_ceiling,
        barrier=consumer_step.barrier,
    )


def _external_edge(
    external: CandidateGraphExternalInput,
    *,
    consumer: str,
    consumer_input: str,
    barrier: BarrierKind,
) -> CandidateGraphEdge:
    return CandidateGraphEdge(
        edge_id=f"edge.external.{external.input_id}.{consumer}.{consumer_input}",
        producer_node_id=None,
        producer_output_id=None,
        external_input_id=external.input_id,
        consumer_node_id=consumer,
        consumer_input_id=consumer_input,
        scientific_role=external.scientific_role,
        logical_artifact_id=external.logical_artifact_id,
        payload_schema=external.payload_schema,
        media_type=external.media_type,
        maximum_size_bytes=external.maximum_size_bytes,
        outcome_access=external.outcome_access,
        visibility_ceiling=external.visibility_ceiling,
        barrier=barrier,
    )


def _external_input(
    *,
    input_id: str,
    logical_artifact_id: str,
    payload_schema: str,
    media_type: str,
    payload_sha256: str,
    payload_size: int,
    role: ScientificInputRole,
    outcome_access: OutcomeAccess,
    visibility: VisibilityCeiling,
) -> CandidateGraphExternalInput:
    return CandidateGraphExternalInput(
        input_id=input_id,
        scientific_role=role,
        logical_artifact_id=logical_artifact_id,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=payload_sha256,
        payload_schema=payload_schema,
        media_type=media_type,
        maximum_size_bytes=payload_size,
        outcome_access=outcome_access,
        visibility_ceiling=visibility,
    )


def material_family_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    family_config: MaterialFamilyConfig,
    source_manifest: MaterialSourceManifest,
    source_qualification: SourceQualification,
    corpus_sha256: str,
    corpus_size_bytes: int,
    policies: tuple[DiscoveryPolicyConfig, ...],
) -> CandidateScientificGraph:
    worlds = world_configs(family_config)
    ordered_policies = tuple(sorted(policies, key=lambda value: value.config_id))
    rounds = ordered_policies[0].total_query_budget // ordered_policies[0].batch_size
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
    source_manifest_external = _external_input(
        input_id=SOURCE_MANIFEST_INPUT_ID,
        logical_artifact_id=source_manifest.manifest_id,
        payload_schema=MaterialSourceManifest.SCHEMA,
        media_type="application/json",
        payload_sha256=source_manifest.fingerprint(),
        payload_size=len(source_manifest.canonical_bytes()),
        role=ScientificInputRole.SOURCE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility=VisibilityCeiling.PROSPECTIVE,
    )
    qualification_external = _external_input(
        input_id=SOURCE_QUALIFICATION_INPUT_ID,
        logical_artifact_id=source_qualification.qualification_id,
        payload_schema=SourceQualification.SCHEMA,
        media_type="application/json",
        payload_sha256=source_qualification.fingerprint(),
        payload_size=len(source_qualification.canonical_bytes()),
        role=ScientificInputRole.QUALIFICATION,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    corpus_external = _external_input(
        input_id=CORPUS_INPUT_ID,
        logical_artifact_id=CORPUS_INPUT_ID,
        payload_schema=MATERIAL_CORPUS_TABLE_SCHEMA,
        media_type="application/vnd.apache.arrow.file",
        payload_sha256=corpus_sha256,
        payload_size=corpus_size_bytes,
        role=ScientificInputRole.PREPARED_MEDIUM,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    external = (
        corpus_external,
        qualification_external,
        source_manifest_external,
    )
    edges: list[CandidateGraphEdge] = [
        _external_edge(
            source_manifest_external,
            consumer=SOURCE_STEP,
            consumer_input="source-manifest",
            barrier=BarrierKind.NONE,
        ),
        _external_edge(
            qualification_external,
            consumer=SOURCE_STEP,
            consumer_input="source-qualification",
            barrier=BarrierKind.NONE,
        ),
        _external_edge(
            corpus_external,
            consumer=SOURCE_STEP,
            consumer_input="material-corpus",
            barrier=BarrierKind.NONE,
        ),
    ]
    evaluator_steps: list[str] = []
    terminal_receiver_steps: list[str] = []
    for world_index, _world_config in enumerate(worlds):
        world_step = f"material-family-world-{world_index:02d}"
        edges.extend(
            (
                _internal_edge(
                    protocol,
                    producer=SOURCE_STEP,
                    output_id="source-ready",
                    consumer=world_step,
                    consumer_input="source-ready",
                    role=ScientificInputRole.QUALIFICATION,
                    maximum_size_bytes=2 * 1024**2,
                ),
                _external_edge(
                    corpus_external,
                    consumer=world_step,
                    consumer_input="material-corpus",
                    barrier=BarrierKind.NONE,
                ),
            )
        )
        for policy_index, _policy in enumerate(ordered_policies):
            prior_receiver: str | None = None
            for round_index in range(rounds):
                selector = f"material-family-select-w{world_index:02d}-p{policy_index:02d}-r{round_index:02d}"
                receiver = f"material-family-receive-w{world_index:02d}-p{policy_index:02d}-r{round_index:02d}"
                edges.extend(
                    (
                        _internal_edge(
                            protocol,
                            producer=world_step,
                            output_id="world-manifest",
                            consumer=selector,
                            consumer_input="world-manifest",
                            role=ScientificInputRole.QUALIFICATION,
                            maximum_size_bytes=2 * 1024**2,
                        ),
                        _internal_edge(
                            protocol,
                            producer=world_step,
                            output_id="world-policy",
                            consumer=selector,
                            consumer_input="world-policy",
                            role=ScientificInputRole.DENOMINATOR,
                            maximum_size_bytes=32 * 1024**2,
                        ),
                        _internal_edge(
                            protocol,
                            producer=world_step,
                            output_id="world-manifest",
                            consumer=receiver,
                            consumer_input="world-manifest",
                            role=ScientificInputRole.QUALIFICATION,
                            maximum_size_bytes=2 * 1024**2,
                        ),
                        _internal_edge(
                            protocol,
                            producer=world_step,
                            output_id="world-policy",
                            consumer=receiver,
                            consumer_input="world-policy",
                            role=ScientificInputRole.DENOMINATOR,
                            maximum_size_bytes=32 * 1024**2,
                        ),
                        _internal_edge(
                            protocol,
                            producer=world_step,
                            output_id="world-truth",
                            consumer=receiver,
                            consumer_input="world-truth",
                            role=ScientificInputRole.RECEIVER,
                            maximum_size_bytes=32 * 1024**2,
                        ),
                        _internal_edge(
                            protocol,
                            producer=selector,
                            output_id="policy-decision",
                            consumer=receiver,
                            consumer_input="committed-policy-decision",
                            role=ScientificInputRole.ACTION,
                            maximum_size_bytes=2 * 1024**2,
                        ),
                    )
                )
                if prior_receiver is not None:
                    edges.extend(
                        (
                            _internal_edge(
                                protocol,
                                producer=prior_receiver,
                                output_id="policy-prefix",
                                consumer=selector,
                                consumer_input="prior-policy-prefix",
                                role=ScientificInputRole.HISTORY,
                                maximum_size_bytes=4 * 1024**2,
                            ),
                            _internal_edge(
                                protocol,
                                producer=prior_receiver,
                                output_id="policy-prefix",
                                consumer=receiver,
                                consumer_input="prior-policy-prefix",
                                role=ScientificInputRole.HISTORY,
                                maximum_size_bytes=4 * 1024**2,
                            ),
                        )
                    )
                prior_receiver = receiver
            if prior_receiver is None:  # pragma: no cover
                raise ValueError("SC graph contains no receiver rounds")
            terminal_receiver_steps.append(prior_receiver)
            evaluator = f"material-family-evaluate-w{world_index:02d}-p{policy_index:02d}"
            evaluator_steps.append(evaluator)
            edges.extend(
                (
                    _internal_edge(
                        protocol,
                        producer=world_step,
                        output_id="world-manifest",
                        consumer=evaluator,
                        consumer_input="world-manifest",
                        role=ScientificInputRole.QUALIFICATION,
                        maximum_size_bytes=2 * 1024**2,
                    ),
                    _internal_edge(
                        protocol,
                        producer=world_step,
                        output_id="world-truth",
                        consumer=evaluator,
                        consumer_input="world-truth",
                        role=ScientificInputRole.OUTCOME,
                        maximum_size_bytes=32 * 1024**2,
                    ),
                    _internal_edge(
                        protocol,
                        producer=prior_receiver,
                        output_id="policy-prefix",
                        consumer=evaluator,
                        consumer_input="final-policy-prefix",
                        role=ScientificInputRole.HISTORY,
                        maximum_size_bytes=4 * 1024**2,
                    ),
                )
            )
    if family_config.phase is not MaterialFamilyDiscoveryPhase.DEVELOPMENT:
        for receiver in terminal_receiver_steps:
            edges.append(
                _internal_edge(
                    protocol,
                    producer=receiver,
                    output_id="policy-prefix",
                    consumer=FREEZE_STEP,
                    consumer_input=f"terminal-history-{receiver}",
                    role=ScientificInputRole.HISTORY,
                    maximum_size_bytes=4 * 1024**2,
                )
            )
        for evaluator in evaluator_steps:
            edges.append(
                _internal_edge(
                    protocol,
                    producer=FREEZE_STEP,
                    output_id="evaluation-freeze",
                    consumer=evaluator,
                    consumer_input="evaluation-freeze",
                    role=ScientificInputRole.QUALIFICATION,
                    maximum_size_bytes=4 * 1024**2,
                )
            )
    for evaluator in evaluator_steps:
        edges.append(
            _internal_edge(
                protocol,
                producer=evaluator,
                output_id="world-policy-evaluation",
                consumer=ADJUDICATE_STEP,
                consumer_input=f"evaluation-{evaluator}",
                role=ScientificInputRole.OUTCOME,
                maximum_size_bytes=4 * 1024**2,
            )
        )
    edges.extend(
        (
            _internal_edge(
                protocol,
                producer=TRUTH_CONTROL_STEP,
                output_id="truth-control-evaluation",
                consumer=ADJUDICATE_STEP,
                consumer_input="truth-control-evaluation",
                role=ScientificInputRole.QUALIFICATION,
                maximum_size_bytes=2 * 1024**2,
            ),
            _internal_edge(
                protocol,
                producer=ADJUDICATE_STEP,
                output_id="phase-adjudication",
                consumer=CLOSEOUT_STEP,
                consumer_input="phase-adjudication",
                role=ScientificInputRole.OUTCOME,
                maximum_size_bytes=32 * 1024**2,
            ),
        )
    )
    return CandidateScientificGraph(
        graph_id=(f"graph.material-family-discovery-{family_config.phase.value.lower()}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"),
        external_inputs=tuple(sorted(external, key=lambda value: value.input_id)),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _coverage(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> ObligationCoverage:
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    fallback = (CLOSEOUT_STEP, "phase-closeout")
    return ObligationCoverage(
        coverage_id=f"coverage.material-family-discovery-{experiment.experiment_id}",
        bindings=tuple(
            sorted(
                (
                    ObligationCoverageBinding(
                        obligation_id=obligation_id,
                        proof_owner_node_id=owners.get(obligation_id, fallback)[0],
                        required_output_id=owners.get(obligation_id, fallback)[1],
                        contributor_edge_ids=incoming[owners.get(obligation_id, fallback)[0]],
                    )
                    for obligation_id in required_candidate_obligation_ids(experiment, protocol)
                ),
                key=lambda value: value.obligation_id,
            )
        ),
    )


def material_family_discovery_study_template(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    family_config: MaterialFamilyConfig,
    source_manifest: MaterialSourceManifest,
    source_qualification: SourceQualification,
    corpus_sha256: str,
    corpus_size_bytes: int,
    policies: tuple[DiscoveryPolicyConfig, ...],
) -> StudyTemplate:
    graph = material_family_scientific_graph(
        protocol=protocol,
        registry=registry,
        family_config=family_config,
        source_manifest=source_manifest,
        source_qualification=source_qualification,
        corpus_sha256=corpus_sha256,
        corpus_size_bytes=corpus_size_bytes,
        policies=policies,
    )
    return StudyTemplate(
        template_key=(f"material-family-discovery-{family_config.phase.value.lower()}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"),
        template_version=CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=_coverage(experiment=experiment, protocol=protocol, graph=graph),
    )


def material_family_candidate_catalog(
    *,
    registry: CapabilityRegistry,
    template: StudyTemplate,
) -> CandidateCapabilityCatalog:
    return CandidateCapabilityCatalog(
        catalog_id=f"catalog.{template.template_key}",
        registrations=tuple(
            CandidateCapabilityRegistration(
                manifest=manifest,
                provider_key=manifest.capability_key,
                provider_version=manifest.capability_version,
                config_media_type="application/json",
                maximum_config_bytes=256 * 1024,
            )
            for manifest in registry.capabilities
        ),
        templates=(template,),
    )


__all__ = [
    "ADJUDICATE_STEP",
    "CLOSEOUT_STEP",
    "CORPUS_INPUT_ID",
    "FREEZE_STEP",
    "SOURCE_MANIFEST_INPUT_ID",
    "SOURCE_QUALIFICATION_INPUT_ID",
    "SOURCE_STEP",
    "TRUTH_CONTROL_STEP",
    "config_ref",
    'material_family_candidate_catalog',
    'material_family_discovery_study_template',
    'material_family_protocol',
    'material_family_scientific_graph',
    "world_configs",
]
