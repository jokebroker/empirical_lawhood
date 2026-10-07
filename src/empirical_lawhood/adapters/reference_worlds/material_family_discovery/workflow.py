"""Pure SC source, world, causal replay, metrics and adjudication workflow."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from hashlib import sha256

import numpy as np

from empirical_lawhood.adapters.methods.budgeted_first_discovery import (
    CandidateView,
    DiscoveryObservation,
    DiscoveryPolicyConfig,
    LabelState,
    ObservationOrigin,
    PolicyDecision,
    PolicyDecisionKind,
    commit_hold,
    evaluate_policy_history,
    select_batch,
    visible_prefix_sha256,
)

from .codecs import MaterialFamilyDiscoveryWorldTruth, decode_world_policy, encode_policy_history, encode_world_policy, encode_world_truth
from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, MaterialFamilyDiscoveryPhase, SourceQualification
from .records import PolicyHistoryPrefix, PolicyTerminalKind, MaterialFamilyDiscoveryChildDisposition, MaterialFamilyDiscoveryEvaluationFreeze, MaterialFamilyDiscoveryPairedContrast, MaterialFamilyDiscoveryPhaseAdjudication, MaterialFamilyDiscoveryPhaseCloseout, MaterialFamilyDiscoveryPolicyHistoryCommitment, MaterialFamilyDiscoveryPolicySummary, MaterialFamilySourceAssessment, MaterialFamilyDiscoveryTruthControlEvaluation, MaterialFamilyDiscoveryWorldBuildConfig, MaterialFamilyDiscoveryWorldManifest, MaterialFamilyDiscoveryWorldPolicyEvaluation
from .worlds import (
    FEATURE_SCHEMA_ID,
    MaterialCorpus,
    build_world,
    corpus_identity,
)


def qualify_corpus_binding(
    *,
    manifest: MaterialSourceManifest,
    qualification: SourceQualification,
    corpus: MaterialCorpus,
) -> MaterialFamilySourceAssessment:
    if qualification.source_manifest_sha256 != manifest.fingerprint():
        raise ValueError("SC source qualification differs from its manifest")
    if corpus.qualification != qualification:
        raise ValueError("SC corpus differs from its exact qualification")
    if len(corpus.candidates) != qualification.canonical_candidate_count:
        raise ValueError("SC corpus candidate count differs from qualification")
    return MaterialFamilySourceAssessment(
        record_id='source-ready.material-family-discovery-nims-220808',
        source_manifest_sha256=manifest.fingerprint(),
        source_qualification_sha256=qualification.fingerprint(),
        corpus_identity_sha256=corpus_identity(corpus),
        corpus_candidate_count=len(corpus.candidates),
        exact_reproduction_disposition=qualification.exact_reproduction_disposition,
        missing_exact_operand_ids=qualification.missing_exact_operand_ids,
    )


def build_world_artifacts(
    *,
    corpus: MaterialCorpus,
    family_config: MaterialFamilyConfig,
    world_config: MaterialFamilyDiscoveryWorldBuildConfig,
    source_ready: MaterialFamilySourceAssessment,
) -> tuple[MaterialFamilyDiscoveryWorldManifest, bytes, bytes]:
    if world_config.phase is not family_config.phase:
        raise ValueError("SC world phase differs from its family config")
    if world_config.family_config_sha256 != family_config.fingerprint():
        raise ValueError("SC world config differs from family-config bytes")
    if family_config.source_qualification_sha256 != corpus.qualification.fingerprint():
        raise ValueError("SC family config differs from corpus qualification")
    if source_ready.corpus_identity_sha256 != corpus_identity(corpus):
        raise ValueError("SC world corpus differs from source-ready binding")
    world = build_world(corpus, family_config, world_config.family_id)
    policy_payload = encode_world_policy(world)
    truth_payload = encode_world_truth(world)
    universe_hash = sha256(policy_payload).hexdigest()
    manifest = MaterialFamilyDiscoveryWorldManifest(
        manifest_id=f"manifest.{world.world_id}",
        world_id=world.world_id,
        phase=family_config.phase,
        family_config_sha256=family_config.fingerprint(),
        corpus_identity_sha256=source_ready.corpus_identity_sha256,
        feature_schema_id=FEATURE_SCHEMA_ID,
        policy_candidate_count=len(world.policy_candidates),
        initial_label_count=len(world.initial_observations),
        eligible_candidate_count=len(world.candidate_pool_ids),
        candidate_universe_sha256=universe_hash,
    )
    return manifest, policy_payload, truth_payload


def select_committed_batch(
    *,
    world: MaterialFamilyDiscoveryWorldManifest,
    policy_payload: bytes,
    policy: DiscoveryPolicyConfig,
    prior: PolicyHistoryPrefix | None,
    round_index: int,
) -> PolicyDecision:
    candidates, initial, pool_ids = decode_world_policy(policy_payload)
    if sha256(policy_payload).hexdigest() != world.candidate_universe_sha256:
        raise ValueError("SC policy table differs from its world manifest")
    if policy.feature_schema_id != world.feature_schema_id:
        raise ValueError("SC policy feature schema differs from its world")
    if (
        len(candidates) != world.policy_candidate_count
        or len(pool_ids) != world.eligible_candidate_count
    ):
        raise ValueError("SC policy table counts differ from its world manifest")
    if prior is None:
        if round_index != 0:
            raise ValueError("SC policy history must start at round zero")
        observations = initial
    else:
        if (
            prior.world_id != world.world_id
            or prior.policy_id != policy.config_id
            or prior.policy_config_sha256 != policy.fingerprint()
            or len(prior.decisions) != round_index
        ):
            raise ValueError("SC prior policy prefix identity/round differs")
        observations = (*initial, *prior.observations)
        if prior.terminal_kind is not PolicyTerminalKind.ACTIVE:
            return commit_hold(
                world_id=world.world_id,
                config=policy,
                candidates=candidates,
                observations=observations,
                round_index=round_index,
                reason="hold.world-terminal",
            )
    return select_batch(
        world_id=world.world_id,
        config=policy,
        candidates=candidates,
        observations=observations,
        round_index=round_index,
    )


def reveal_committed_batch(
    *,
    world: MaterialFamilyDiscoveryWorldManifest,
    policy_payload: bytes,
    truth: MaterialFamilyDiscoveryWorldTruth,
    policy: DiscoveryPolicyConfig,
    decision: PolicyDecision,
    prior: PolicyHistoryPrefix | None,
) -> PolicyHistoryPrefix:
    candidates, initial, pool_ids = decode_world_policy(policy_payload)
    if sha256(policy_payload).hexdigest() != world.candidate_universe_sha256:
        raise ValueError("SC receiver policy table differs from its manifest")
    if set(truth.by_id) != set(pool_ids):
        raise ValueError("SC receiver truth and eligible pool differ")
    prior_decisions = () if prior is None else prior.decisions
    prior_observations = () if prior is None else prior.observations
    expected_round = len(prior_decisions)
    if (
        decision.world_id != world.world_id
        or decision.policy_config_sha256 != policy.fingerprint()
        or decision.round_index != expected_round
    ):
        raise ValueError("SC receiver decision identity/round differs")
    visible = (*initial, *prior_observations)
    if decision.visible_prefix_sha256 != visible_prefix_sha256(visible):
        raise ValueError("SC receiver decision was not bound to its visible prefix")
    responses = tuple(
        DiscoveryObservation(
            candidate_id=candidate_id,
            origin=ObservationOrigin.QUERY_RESPONSE,
            state=truth.by_id[candidate_id].label_state,
            tc_kelvin=truth.by_id[candidate_id].tc_kelvin,
            query_cost=policy.query_cost,
            round_index=decision.round_index,
        )
        for candidate_id in decision.requested_candidate_ids
    )
    if decision.incremental_cost != sum((value.query_cost for value in responses), Decimal(0)):
        raise ValueError("SC receiver query cost differs from the commitment")

    decisions = (*prior_decisions, decision)
    observations = (*prior_observations, *responses)
    terminal_kind: PolicyTerminalKind
    terminal_round: int | None
    if prior is not None and prior.terminal_kind is not PolicyTerminalKind.ACTIVE:
        if decision.kind is not PolicyDecisionKind.HOLD or responses:
            raise ValueError("SC terminal world accepted a new query")
        terminal_kind = prior.terminal_kind
        terminal_round = prior.terminal_round_index
    elif set(decision.requested_candidate_ids) & truth.target_candidate_ids:
        terminal_kind = PolicyTerminalKind.DISCOVERED
        terminal_round = decision.round_index
    elif decision.kind is PolicyDecisionKind.HOLD:
        terminal_kind = (
            PolicyTerminalKind.QUERY_BUDGET_EXHAUSTED
            if decision.hold_reason_id == "hold.query-budget-exhausted"
            else PolicyTerminalKind.POLICY_HOLD
        )
        terminal_round = decision.round_index
    elif len(observations) >= policy.total_query_budget:
        terminal_kind = PolicyTerminalKind.QUERY_BUDGET_EXHAUSTED
        terminal_round = decision.round_index
    else:
        terminal_kind = PolicyTerminalKind.ACTIVE
        terminal_round = None
    return PolicyHistoryPrefix(
        prefix_id=f"prefix.{world.world_id}.{policy.config_id}.{decision.round_index}",
        world_id=world.world_id,
        policy_id=policy.config_id,
        policy_config_sha256=policy.fingerprint(),
        decisions=decisions,
        observations=observations,
        terminal_kind=terminal_kind,
        terminal_round_index=terminal_round,
    )


def freeze_policy_histories(
    *,
    family_config: MaterialFamilyConfig,
    prefixes: tuple[PolicyHistoryPrefix, ...],
) -> MaterialFamilyDiscoveryEvaluationFreeze:
    """Commit the complete matched terminal-history matrix before truth reveal."""

    if family_config.phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT:
        raise ValueError("SC development histories do not require an evaluation freeze")
    family_ids = (
        family_config.evaluation_family_ids
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
    expected_world_count = len(family_ids)
    expected_policy_hashes = set(family_config.policy_config_sha256s)
    expected_count = expected_world_count * len(expected_policy_hashes)
    if len(prefixes) != expected_count:
        raise ValueError("SC freeze received an incomplete terminal-history matrix")

    def commitment(value: PolicyHistoryPrefix) -> MaterialFamilyDiscoveryPolicyHistoryCommitment:
        terminal_round = value.terminal_round_index
        if value.terminal_kind is PolicyTerminalKind.ACTIVE or terminal_round is None:
            raise ValueError("SC freeze received an active policy history")
        return MaterialFamilyDiscoveryPolicyHistoryCommitment(
            commitment_id=f"commitment.{value.prefix_id}",
            prefix_id=value.prefix_id,
            prefix_sha256=value.fingerprint(),
            world_id=value.world_id,
            policy_id=value.policy_id,
            policy_config_sha256=value.policy_config_sha256,
            terminal_kind=value.terminal_kind,
            terminal_round_index=terminal_round,
            decision_count=len(value.decisions),
            observation_count=len(value.observations),
        )

    if {value.policy_config_sha256 for value in prefixes} != expected_policy_hashes:
        raise ValueError("SC freeze policy roster differs from the family config")
    by_world: dict[str, set[str]] = defaultdict(set)
    for value in prefixes:
        by_world[value.world_id].add(value.policy_config_sha256)
    if len(by_world) != expected_world_count or any(
        value != expected_policy_hashes for value in by_world.values()
    ):
        raise ValueError("SC freeze lacks a matched world-policy matrix")
    commitments = tuple(
        sorted(
            (commitment(value) for value in prefixes),
            key=lambda value: (value.world_id, value.policy_id),
        )
    )
    family_sha256 = family_config.fingerprint()
    return MaterialFamilyDiscoveryEvaluationFreeze(
        freeze_id=(f"freeze.material-family-discovery-{family_config.phase.value.lower()}-{family_sha256[:16]}"),
        phase=family_config.phase,
        family_config_sha256=family_sha256,
        expected_world_count=expected_world_count,
        expected_policy_count=len(expected_policy_hashes),
        commitments=commitments,
    )


def validate_frozen_policy_history(
    *,
    world: MaterialFamilyDiscoveryWorldManifest,
    prefix: PolicyHistoryPrefix,
    freeze: MaterialFamilyDiscoveryEvaluationFreeze,
) -> None:
    """Require the evaluator input to equal its pre-reveal commitment."""

    if freeze.phase is not world.phase or freeze.family_config_sha256 != world.family_config_sha256:
        raise ValueError("SC evaluation freeze belongs to another phase or family config")
    matches = tuple(
        value
        for value in freeze.commitments
        if value.world_id == prefix.world_id and value.policy_id == prefix.policy_id
    )
    if len(matches) != 1:
        raise ValueError("SC evaluation freeze lacks the evaluator history")
    commitment = matches[0]
    if (
        commitment.prefix_id != prefix.prefix_id
        or commitment.prefix_sha256 != prefix.fingerprint()
        or commitment.policy_config_sha256 != prefix.policy_config_sha256
        or commitment.terminal_kind is not prefix.terminal_kind
        or commitment.terminal_round_index != prefix.terminal_round_index
        or commitment.decision_count != len(prefix.decisions)
        or commitment.observation_count != len(prefix.observations)
    ):
        raise ValueError("SC evaluator history differs from its pre-reveal commitment")


def evaluate_world_policy(
    *,
    world: MaterialFamilyDiscoveryWorldManifest,
    truth: MaterialFamilyDiscoveryWorldTruth,
    prefix: PolicyHistoryPrefix,
) -> tuple[MaterialFamilyDiscoveryWorldPolicyEvaluation, bytes]:
    if prefix.world_id != world.world_id:
        raise ValueError("SC evaluator prefix belongs to another world")
    cutoff = (
        len(prefix.decisions)
        if prefix.terminal_round_index is None
        else prefix.terminal_round_index + 1
    )
    decisions = prefix.decisions[:cutoff]
    requested_count = sum(len(value.requested_candidate_ids) for value in decisions)
    observations = prefix.observations[:requested_count]
    result = evaluate_policy_history(
        world_id=world.world_id,
        policy_id=prefix.policy_id,
        decisions=decisions,
        observations=observations,
        target_candidate_ids=truth.target_candidate_ids,
        empty_or_disconnected_world=False,
    )
    history_hash = prefix.fingerprint()
    evaluation = MaterialFamilyDiscoveryWorldPolicyEvaluation(
        evaluation_id=f"evaluation.{world.world_id}.{prefix.policy_id}",
        phase=world.phase,
        target_family_id=truth.target_family_id,
        target_family_label=truth.target_family_label,
        candidate_universe_sha256=world.candidate_universe_sha256,
        history_sha256=history_hash,
        result=result,
    )
    return evaluation, encode_policy_history(prefix)


def evaluate_truth_controls(policy: DiscoveryPolicyConfig) -> MaterialFamilyDiscoveryTruthControlEvaluation:
    """Execute frozen policy-visible support controls, never declare their result."""

    invalid_candidates = tuple(
        CandidateView(
            candidate_id=f"material.truth-control-invalid-{index:02d}",
            stratum_id="stratum.truth-control",
            features=(0.0,) * 236,
            valid=False,
        )
        for index in range(10)
    )
    exhausted_candidates = tuple(
        CandidateView(
            candidate_id=f"material.truth-control-exhausted-{index:02d}",
            stratum_id="stratum.truth-control",
            features=(float(index + 1),) + (0.0,) * 235,
        )
        for index in range(10)
    )
    exhausted_observations = tuple(
        DiscoveryObservation(
            candidate_id=value.candidate_id,
            origin=ObservationOrigin.QUERY_RESPONSE,
            state=LabelState.EXPLICIT_NEGATIVE,
            tc_kelvin=Decimal(0),
            query_cost=policy.query_cost,
            round_index=0,
        )
        for value in exhausted_candidates
    )
    hold_controls = (
        (
            'control.material-family-empty-eligible-support',
            select_batch(
                world_id='world.material-family-truth-control-empty-support',
                config=policy,
                candidates=invalid_candidates,
                observations=(),
                round_index=0,
            ),
        ),
        (
            'control.material-family-exhausted-candidate-support',
            select_batch(
                world_id='world.material-family-truth-control-exhausted-support',
                config=policy,
                candidates=exhausted_candidates,
                observations=exhausted_observations,
                round_index=1,
            ),
        ),
    )
    # Give the local-law selector a full-rank, well-supported visible history,
    # then place every eligible candidate far outside every fitted local
    # support radius.  Information queries remain explicitly admissible; any
    # other requested candidate would be an unsupported truth-control
    # promotion and therefore a decisive false admission.
    supported_candidates = tuple(
        CandidateView(
            candidate_id=f"material.truth-control-supported-{index:02d}",
            stratum_id="stratum.truth-control",
            features=tuple(((index * (feature + 3)) % 17) / 100 for feature in range(6))
            + (0.0,) * 230,
        )
        for index in range(48)
    )
    disconnected_candidates = tuple(
        CandidateView(
            candidate_id=f"material.truth-control-disconnected-{index:02d}",
            stratum_id="stratum.truth-control",
            features=tuple(20.0 + ((index * (feature + 3)) % 17) / 100 for feature in range(6))
            + (0.0,) * 230,
        )
        for index in range(10)
    )
    supported_observations = tuple(
        DiscoveryObservation(
            candidate_id=value.candidate_id,
            origin=ObservationOrigin.INITIAL_LABEL,
            state=LabelState.MEASURED,
            tc_kelvin=Decimal(20 + index),
            query_cost=Decimal(0),
            round_index=None,
        )
        for index, value in enumerate(supported_candidates)
    )
    disconnected = select_batch(
        world_id='world.material-family-truth-control-disconnected-support',
        config=policy,
        candidates=(*supported_candidates, *disconnected_candidates),
        observations=supported_observations,
        round_index=0,
    )
    correct = sum(
        decision.kind is PolicyDecisionKind.HOLD and not decision.requested_candidate_ids
        for _control_id, decision in hold_controls
    )
    false_admissions = len(
        set(disconnected.requested_candidate_ids) - set(disconnected.information_query_ids)
    )
    passed = correct == len(hold_controls) and false_admissions == 0
    return MaterialFamilyDiscoveryTruthControlEvaluation(
        evaluation_id=f"control-evaluation.{policy.config_id}",
        policy_id=policy.config_id,
        policy_config_sha256=policy.fingerprint(),
        control_ids=tuple(
            sorted(
                (
                    *(control_id for control_id, _decision in hold_controls),
                    'control.material-family-disconnected-supported-recurrence',
                )
            )
        ),
        required_hold_count=len(hold_controls),
        correct_hold_count=correct,
        false_admission_count=false_admissions,
        passed=passed,
        reason_codes=(
            ('MATERIAL_FAMILY_TRUTH_CONTROLS_PASSED',)
            if passed
            else ('MATERIAL_FAMILY_TRUTH_CONTROL_FALSE_ADMISSION_OR_MISSED_HOLD',)
        ),
    )


def _decimal(value: float) -> Decimal:
    return Decimal(format(float(value), ".12g"))


def _paired_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    tail_alpha: float,
) -> tuple[Decimal, Decimal, Decimal]:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("SC paired interval requires finite family-level values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, (tail_alpha, 1.0 - tail_alpha))
    return _decimal(float(np.mean(values))), _decimal(float(low)), _decimal(float(high))


def adjudicate_phase(
    *,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
    policies: tuple[DiscoveryPolicyConfig, ...],
    evaluations: tuple[MaterialFamilyDiscoveryWorldPolicyEvaluation, ...],
    truth_control: MaterialFamilyDiscoveryTruthControlEvaluation,
) -> MaterialFamilyDiscoveryPhaseAdjudication:
    if adjudication_config.phase is not family_config.phase:
        raise ValueError("SC adjudication and family phases differ")
    if adjudication_config.family_config_sha256 != family_config.fingerprint():
        raise ValueError("SC adjudication family-config identity differs")
    by_sha = {value.fingerprint(): value for value in policies}
    if len(by_sha) != len(policies):
        raise ValueError("SC policy configs repeat")
    if set(by_sha) != set(family_config.policy_config_sha256s):
        raise ValueError("SC policy roster differs from the family config")
    method = by_sha.get(adjudication_config.method_policy_config_sha256)
    primary_bo = by_sha.get(adjudication_config.primary_bo_policy_config_sha256)
    if method is None or primary_bo is None:
        raise ValueError("SC adjudication lacks its method or primary BO config")
    if (
        truth_control.policy_id != method.config_id
        or truth_control.policy_config_sha256 != method.fingerprint()
    ):
        raise ValueError("SC truth-control evaluation belongs to another method")
    expected_policies = {value.config_id for value in policies}
    grouped: dict[str, list[MaterialFamilyDiscoveryWorldPolicyEvaluation]] = defaultdict(list)
    by_world: dict[str, list[MaterialFamilyDiscoveryWorldPolicyEvaluation]] = defaultdict(list)
    for value in evaluations:
        if value.phase is not family_config.phase:
            raise ValueError("SC world evaluation phase differs")
        grouped[value.result.policy_id].append(value)
        by_world[value.result.world_id].append(value)
    if set(grouped) != expected_policies:
        raise ValueError("SC evaluation policy roster differs")
    for values in by_world.values():
        if {value.result.policy_id for value in values} != expected_policies:
            raise ValueError("SC matched world lacks a policy result")
        if len({value.candidate_universe_sha256 for value in values}) != 1:
            raise ValueError("SC policies received different candidate universes")
    family_ids = tuple(sorted({value.target_family_id for value in evaluations}))
    if len(family_ids) != len(by_world):
        raise ValueError("SC independent family is reused across worlds")

    restricted: dict[str, dict[str, float]] = {}
    false_promotions: dict[str, dict[str, float]] = {}
    summaries = []
    for policy_id in sorted(grouped):
        values = sorted(grouped[policy_id], key=lambda value: value.target_family_id)
        restricted[policy_id] = {
            value.target_family_id: float(
                value.result.recommendations_to_discovery
                if value.result.recommendations_to_discovery is not None
                else adjudication_config.query_budget
            )
            for value in values
        }
        false_promotions[policy_id] = {
            value.target_family_id: float(value.result.false_promotion_count) for value in values
        }
        discoveries = sum(value.result.recommendations_to_discovery is not None for value in values)
        summaries.append(
            MaterialFamilyDiscoveryPolicySummary(
                summary_id=f"summary.{family_config.phase.value.lower()}.{policy_id}",
                policy_id=policy_id,
                world_count=len(values),
                discovery_count=discoveries,
                restricted_mean_queries=_decimal(
                    float(np.mean(tuple(restricted[policy_id].values())))
                ),
                discovery_probability=_decimal(discoveries / len(values)),
                mean_false_promotions=_decimal(
                    float(np.mean(tuple(false_promotions[policy_id].values())))
                ),
                total_information_queries=sum(value.result.information_queries for value in values),
                total_unsupported_actions=sum(
                    value.result.unsupported_action_count for value in values
                ),
            )
        )

    comparator_ids = tuple(sorted(expected_policies - {method.config_id}))
    family_order = tuple(sorted(family_ids))
    alpha = 1.0 - float(adjudication_config.confidence_level)
    tail_alpha = alpha / (2 * max(1, len(comparator_ids)))
    contrasts = []
    for contrast_index, comparator_id in enumerate(comparator_ids):
        query_advantage = np.asarray(
            [
                restricted[comparator_id][family_id] - restricted[method.config_id][family_id]
                for family_id in family_order
            ],
            dtype=np.float64,
        )
        false_advantage = np.asarray(
            [
                false_promotions[comparator_id][family_id]
                - false_promotions[method.config_id][family_id]
                for family_id in family_order
            ],
            dtype=np.float64,
        )
        query_mean, query_low, query_high = _paired_interval(
            query_advantage,
            resamples=adjudication_config.bootstrap_resamples,
            seed=adjudication_config.bootstrap_seed + contrast_index * 2,
            tail_alpha=tail_alpha,
        )
        false_mean, false_low, false_high = _paired_interval(
            false_advantage,
            resamples=adjudication_config.bootstrap_resamples,
            seed=adjudication_config.bootstrap_seed + contrast_index * 2 + 1,
            tail_alpha=tail_alpha,
        )
        contrasts.append(
            MaterialFamilyDiscoveryPairedContrast(
                contrast_id=f"contrast.{method.config_id}.vs.{comparator_id}",
                method_policy_id=method.config_id,
                comparator_policy_id=comparator_id,
                paired_world_count=len(family_order),
                mean_restricted_query_advantage=query_mean,
                query_advantage_interval_low=query_low,
                query_advantage_interval_high=query_high,
                mean_false_promotion_advantage=false_mean,
                false_promotion_interval_low=false_low,
                false_promotion_interval_high=false_high,
                superiority_passed=(query_low >= adjudication_config.superiority_margin_queries),
                pareto_passed=(
                    query_low >= -adjudication_config.pareto_noninferiority_margin_queries
                    and false_low >= adjudication_config.pareto_false_promotion_margin
                ),
            )
        )
    primary = next(
        (value for value in contrasts if value.comparator_policy_id == primary_bo.config_id),
        None,
    )
    reasons = []
    if family_config.phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT:
        disposition = MaterialFamilyDiscoveryChildDisposition.DEVELOPMENT_ONLY
        reasons.append("DEVELOPMENT_OUTCOMES_NONPROMOTING")
    elif len(family_ids) < adjudication_config.minimum_independent_worlds:
        disposition = MaterialFamilyDiscoveryChildDisposition.UNEVALUABLE
        reasons.append("INSUFFICIENT_INDEPENDENT_UNITS")
    elif primary is None:
        disposition = MaterialFamilyDiscoveryChildDisposition.UNEVALUABLE
        reasons.append("PRIMARY_CONSTRAINED_BO_RESULT_MISSING")
    elif not truth_control.passed:
        disposition = MaterialFamilyDiscoveryChildDisposition.NOT_SUPPORTED
        reasons.append("TRUTH_CONTROL_FALSE_ADMISSION")
    elif primary.superiority_passed or primary.pareto_passed:
        disposition = MaterialFamilyDiscoveryChildDisposition.SUPPORTED
        reasons.append(
            "B2_PRIMARY_EFFECT_SUPPORTED"
            if primary.superiority_passed
            else "B2_COST_FALSE_PROMOTION_PARETO_SUPPORTED"
        )
    else:
        disposition = MaterialFamilyDiscoveryChildDisposition.NOT_SUPPORTED
        reasons.append("B2_PRIMARY_EFFECT_NOT_SUPPORTED")
    return MaterialFamilyDiscoveryPhaseAdjudication(
        adjudication_id=f"adjudication.material-family-discovery-{family_config.phase.value.lower()}",
        phase=family_config.phase,
        family_config_sha256=family_config.fingerprint(),
        method_policy_id=method.config_id,
        primary_bo_policy_id=primary_bo.config_id,
        query_budget=adjudication_config.query_budget,
        confidence_level=adjudication_config.confidence_level,
        interval_method_id=adjudication_config.interval_method_id,
        superiority_margin_queries=adjudication_config.superiority_margin_queries,
        pareto_noninferiority_margin_queries=(
            adjudication_config.pareto_noninferiority_margin_queries
        ),
        pareto_false_promotion_margin=adjudication_config.pareto_false_promotion_margin,
        truth_control_false_admissions=truth_control.false_admission_count,
        summaries=tuple(sorted(summaries, key=lambda value: value.policy_id)),
        contrasts=tuple(sorted(contrasts, key=lambda value: value.comparator_policy_id)),
        world_evaluations=tuple(sorted(evaluations, key=lambda value: value.evaluation_id)),
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
    )


def close_phase(adjudication: MaterialFamilyDiscoveryPhaseAdjudication) -> MaterialFamilyDiscoveryPhaseCloseout:
    expected = sum(value.world_count for value in adjudication.summaries)
    return MaterialFamilyDiscoveryPhaseCloseout(
        closeout_id=f"closeout.material-family-discovery-{adjudication.phase.value.lower()}",
        adjudication_sha256=adjudication.fingerprint(),
        expected_world_policy_results=expected,
        observed_world_policy_results=len(adjudication.world_evaluations),
        receipt_recovery_required=True,
        disposition=adjudication.disposition,
    )


__all__ = [
    "adjudicate_phase",
    "build_world_artifacts",
    "close_phase",
    "evaluate_world_policy",
    "evaluate_truth_controls",
    "freeze_policy_histories",
    "qualify_corpus_binding",
    "reveal_committed_batch",
    "select_committed_batch",
    "validate_frozen_policy_history",
]
