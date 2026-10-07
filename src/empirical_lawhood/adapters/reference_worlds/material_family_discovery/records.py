"""Compact cross-task records for the SDCB superconductor benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryObservation,
    ObservationOrigin,
    PolicyDecision,
)
from empirical_lawhood.adapters.methods.budgeted_first_discovery.metrics import WorldPolicyResult
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import ReproductionDisposition, MaterialFamilyDiscoveryPhase


class PolicyTerminalKind(StrEnum):
    ACTIVE = "ACTIVE"
    DISCOVERED = "DISCOVERED"
    POLICY_HOLD = "POLICY_HOLD"
    QUERY_BUDGET_EXHAUSTED = "QUERY_BUDGET_EXHAUSTED"


class MaterialFamilyDiscoveryChildDisposition(StrEnum):
    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"


@dataclass(frozen=True, slots=True)
class MaterialFamilySourceAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-source-assessment'

    record_id: str
    source_manifest_sha256: str
    source_qualification_sha256: str
    corpus_identity_sha256: str
    corpus_candidate_count: int
    exact_reproduction_disposition: ReproductionDisposition
    missing_exact_operand_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        for name in (
            "source_manifest_sha256",
            "source_qualification_sha256",
            "corpus_identity_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if isinstance(self.corpus_candidate_count, bool) or self.corpus_candidate_count <= 0:
            raise ValueError("SC source-ready corpus count must be positive")
        require_sorted_unique_strings(
            self.missing_exact_operand_ids,
            field_name="missing_exact_operand_ids",
        )
        if (
            self.exact_reproduction_disposition is ReproductionDisposition.EXACT_REPRODUCIBLE
            and self.missing_exact_operand_ids
        ):
            raise ValueError("exact SC reproduction cannot retain missing operands")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryWorldBuildConfig(CanonicalRecord):
    """Evaluator-side family binding; never an input to a selector task."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-world-build-config'

    config_id: str
    phase: MaterialFamilyDiscoveryPhase
    family_config_sha256: str
    family_id: str

    def __post_init__(self) -> None:
        for name in ("config_id", "family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.family_config_sha256, field_name="family_config_sha256")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryWorldManifest(CanonicalRecord):
    """Policy-safe world description with no target-family identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-world-manifest'

    manifest_id: str
    world_id: str
    phase: MaterialFamilyDiscoveryPhase
    family_config_sha256: str
    corpus_identity_sha256: str
    feature_schema_id: str
    policy_candidate_count: int
    initial_label_count: int
    eligible_candidate_count: int
    candidate_universe_sha256: str

    def __post_init__(self) -> None:
        for name in ("manifest_id", "world_id", "feature_schema_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "family_config_sha256",
            "corpus_identity_sha256",
            "candidate_universe_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in (
            "policy_candidate_count",
            "initial_label_count",
            "eligible_candidate_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.initial_label_count + self.eligible_candidate_count != self.policy_candidate_count:
            raise ValueError("SC world policy candidate partition differs")


@dataclass(frozen=True, slots=True)
class PolicyHistoryPrefix(CanonicalRecord):
    """Cumulative causal history returned after one committed receiver act."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/material-family-discovery/policy-history-prefix'

    prefix_id: str
    world_id: str
    policy_id: str
    policy_config_sha256: str
    decisions: tuple[PolicyDecision, ...]
    observations: tuple[DiscoveryObservation, ...]
    terminal_kind: PolicyTerminalKind
    terminal_round_index: int | None

    def __post_init__(self) -> None:
        for name in ("prefix_id", "world_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.policy_config_sha256, field_name="policy_config_sha256")
        rounds = tuple(value.round_index for value in self.decisions)
        if rounds != tuple(range(len(rounds))):
            raise ValueError("policy history rounds must be contiguous from zero")
        if any(
            value.world_id != self.world_id
            or value.policy_config_sha256 != self.policy_config_sha256
            for value in self.decisions
        ):
            raise ValueError("policy history decision identity differs")
        if any(value.origin is not ObservationOrigin.QUERY_RESPONSE for value in self.observations):
            raise ValueError("policy history prefix may contain only query responses")
        requested = tuple(
            candidate_id
            for decision in self.decisions
            for candidate_id in decision.requested_candidate_ids
        )
        observed = tuple(value.candidate_id for value in self.observations)
        if requested != observed:
            raise ValueError("policy history requested/observed order differs")
        if self.terminal_kind is PolicyTerminalKind.ACTIVE:
            if self.terminal_round_index is not None:
                raise ValueError("active policy history cannot carry a terminal round")
        else:
            if (
                self.terminal_round_index is None
                or self.terminal_round_index < 0
                or not self.decisions
                or self.terminal_round_index >= len(self.decisions)
            ):
                raise ValueError("terminal policy history requires a valid terminal round")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryPolicyHistoryCommitment(CanonicalRecord):
    """Exact terminal history identity committed before evaluator reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-policy-history-commitment'

    commitment_id: str
    prefix_id: str
    prefix_sha256: str
    world_id: str
    policy_id: str
    policy_config_sha256: str
    terminal_kind: PolicyTerminalKind
    terminal_round_index: int
    decision_count: int
    observation_count: int

    def __post_init__(self) -> None:
        for name in ("commitment_id", "prefix_id", "world_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("prefix_sha256", "policy_config_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.terminal_kind is PolicyTerminalKind.ACTIVE:
            raise ValueError("SC freeze cannot commit an active policy history")
        if self.terminal_round_index < 0:
            raise ValueError("SC frozen policy history requires a terminal round")
        if self.decision_count <= 0 or self.observation_count < 0:
            raise ValueError("SC frozen policy history counts differ")
        if self.terminal_round_index >= self.decision_count:
            raise ValueError("SC frozen terminal round exceeds its history")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryEvaluationFreeze(CanonicalRecord):
    """Complete matched-world policy-history roster frozen before reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-evaluation-freeze'

    freeze_id: str
    phase: MaterialFamilyDiscoveryPhase
    family_config_sha256: str
    expected_world_count: int
    expected_policy_count: int
    commitments: tuple[MaterialFamilyDiscoveryPolicyHistoryCommitment, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_sha256(self.family_config_sha256, field_name="family_config_sha256")
        if self.phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT:
            raise ValueError("development-visible SC histories do not form an evaluation freeze")
        if self.expected_world_count <= 0 or self.expected_policy_count <= 0:
            raise ValueError("SC evaluation freeze requires positive roster counts")
        expected = self.expected_world_count * self.expected_policy_count
        if len(self.commitments) != expected:
            raise ValueError("SC evaluation freeze history roster is incomplete")
        keys = tuple((value.world_id, value.policy_id) for value in self.commitments)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("SC evaluation freeze histories must be sorted and unique")
        worlds = {value.world_id for value in self.commitments}
        policies = {value.policy_config_sha256 for value in self.commitments}
        if len(worlds) != self.expected_world_count:
            raise ValueError("SC evaluation freeze world roster differs")
        if len(policies) != self.expected_policy_count:
            raise ValueError("SC evaluation freeze policy roster differs")
        if any(
            {value.policy_config_sha256 for value in self.commitments if value.world_id == world}
            != policies
            for world in worlds
        ):
            raise ValueError("SC evaluation freeze lacks a matched policy roster")
        policy_ids = {
            policy_sha256: {
                value.policy_id
                for value in self.commitments
                if value.policy_config_sha256 == policy_sha256
            }
            for policy_sha256 in policies
        }
        if any(len(values) != 1 for values in policy_ids.values()):
            raise ValueError("SC evaluation freeze policy identity differs across worlds")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryWorldPolicyEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-world-policy-evaluation'

    evaluation_id: str
    phase: MaterialFamilyDiscoveryPhase
    target_family_id: str
    target_family_label: str
    candidate_universe_sha256: str
    history_sha256: str
    result: WorldPolicyResult

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "target_family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not self.target_family_label:
            raise ValueError("SC world evaluation requires a target-family label")
        validate_sha256(
            self.candidate_universe_sha256,
            field_name="candidate_universe_sha256",
        )
        validate_sha256(self.history_sha256, field_name="history_sha256")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryTruthControlEvaluation(CanonicalRecord):
    """Executed support/HOLD falsifiers for one exact policy implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-truth-control-evaluation'

    evaluation_id: str
    policy_id: str
    policy_config_sha256: str
    control_ids: tuple[str, ...]
    required_hold_count: int
    correct_hold_count: int
    false_admission_count: int
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.policy_config_sha256, field_name="policy_config_sha256")
        require_sorted_unique_strings(self.control_ids, field_name="control_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.required_hold_count <= 0:
            raise ValueError("SC truth controls require at least one HOLD control")
        if not 0 <= self.correct_hold_count <= self.required_hold_count:
            raise ValueError("SC truth-control HOLD counts differ")
        if self.false_admission_count < 0:
            raise ValueError("SC truth-control false admissions cannot be negative")
        expected_pass = (
            self.correct_hold_count == self.required_hold_count and self.false_admission_count == 0
        )
        if self.passed != expected_pass:
            raise ValueError("SC truth-control disposition differs from its counts")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryPolicySummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-policy-summary'

    summary_id: str
    policy_id: str
    world_count: int
    discovery_count: int
    restricted_mean_queries: Decimal
    discovery_probability: Decimal
    mean_false_promotions: Decimal
    total_information_queries: int
    total_unsupported_actions: int

    def __post_init__(self) -> None:
        for name in ("summary_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.world_count <= 0 or not 0 <= self.discovery_count <= self.world_count:
            raise ValueError("SC policy summary world/discovery counts differ")
        for name in (
            "restricted_mean_queries",
            "discovery_probability",
            "mean_false_promotions",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.discovery_probability > 1:
            raise ValueError("SC discovery probability exceeds one")
        if self.total_information_queries < 0 or self.total_unsupported_actions < 0:
            raise ValueError("SC policy summary counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryPairedContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-paired-contrast'

    contrast_id: str
    method_policy_id: str
    comparator_policy_id: str
    paired_world_count: int
    mean_restricted_query_advantage: Decimal
    query_advantage_interval_low: Decimal
    query_advantage_interval_high: Decimal
    mean_false_promotion_advantage: Decimal
    false_promotion_interval_low: Decimal
    false_promotion_interval_high: Decimal
    superiority_passed: bool
    pareto_passed: bool

    def __post_init__(self) -> None:
        for name in ("contrast_id", "method_policy_id", "comparator_policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.method_policy_id == self.comparator_policy_id:
            raise ValueError("SC policy contrast requires distinct policies")
        if self.paired_world_count <= 0:
            raise ValueError("SC contrast requires paired family worlds")
        for name in (
            "mean_restricted_query_advantage",
            "query_advantage_interval_low",
            "query_advantage_interval_high",
            "mean_false_promotion_advantage",
            "false_promotion_interval_low",
            "false_promotion_interval_high",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.query_advantage_interval_low > self.query_advantage_interval_high:
            raise ValueError("SC query-advantage interval is reversed")
        if self.false_promotion_interval_low > self.false_promotion_interval_high:
            raise ValueError("SC false-promotion interval is reversed")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryPhaseAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-phase-adjudication'

    adjudication_id: str
    phase: MaterialFamilyDiscoveryPhase
    family_config_sha256: str
    method_policy_id: str
    primary_bo_policy_id: str
    query_budget: int
    confidence_level: Decimal
    interval_method_id: str
    superiority_margin_queries: Decimal
    pareto_noninferiority_margin_queries: Decimal
    pareto_false_promotion_margin: Decimal
    truth_control_false_admissions: int
    summaries: tuple[MaterialFamilyDiscoveryPolicySummary, ...]
    contrasts: tuple[MaterialFamilyDiscoveryPairedContrast, ...]
    world_evaluations: tuple[MaterialFamilyDiscoveryWorldPolicyEvaluation, ...]
    disposition: MaterialFamilyDiscoveryChildDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "adjudication_id",
            "method_policy_id",
            "primary_bo_policy_id",
            "interval_method_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.family_config_sha256, field_name="family_config_sha256")
        if self.query_budget <= 0 or self.truth_control_false_admissions < 0:
            raise ValueError("SC adjudication budget/control counts differ")
        for name in (
            "confidence_level",
            "superiority_margin_queries",
            "pareto_noninferiority_margin_queries",
            "pareto_false_promotion_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.confidence_level < Decimal(1):
            raise ValueError("SC adjudication confidence level must lie inside (0, 1)")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        summary_ids = tuple(value.policy_id for value in self.summaries)
        if summary_ids != tuple(sorted(set(summary_ids))):
            raise ValueError("SC policy summaries must be sorted and unique")
        contrast_ids = tuple(value.comparator_policy_id for value in self.contrasts)
        if contrast_ids != tuple(sorted(set(contrast_ids))):
            raise ValueError("SC policy contrasts must be sorted and unique")
        evaluation_ids = tuple(value.evaluation_id for value in self.world_evaluations)
        if evaluation_ids != tuple(sorted(set(evaluation_ids))):
            raise ValueError("SC world evaluations must be sorted and unique")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryPhaseCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-phase-closeout'

    closeout_id: str
    adjudication_sha256: str
    expected_world_policy_results: int
    observed_world_policy_results: int
    receipt_recovery_required: bool
    disposition: MaterialFamilyDiscoveryChildDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        validate_sha256(self.adjudication_sha256, field_name="adjudication_sha256")
        if self.expected_world_policy_results <= 0:
            raise ValueError("SC closeout expected result count must be positive")
        if not 0 <= self.observed_world_policy_results <= self.expected_world_policy_results:
            raise ValueError("SC closeout observed result count differs")


__all__ = [
    "PolicyHistoryPrefix",
    "PolicyTerminalKind",
    'MaterialFamilyDiscoveryChildDisposition',
    'MaterialFamilyDiscoveryEvaluationFreeze',
    'MaterialFamilyDiscoveryPairedContrast',
    'MaterialFamilyDiscoveryPhaseAdjudication',
    'MaterialFamilyDiscoveryPhaseCloseout',
    'MaterialFamilyDiscoveryPolicyHistoryCommitment',
    'MaterialFamilyDiscoveryPolicySummary',
    'MaterialFamilySourceAssessment',
    'MaterialFamilyDiscoveryTruthControlEvaluation',
    'MaterialFamilyDiscoveryWorldBuildConfig',
    'MaterialFamilyDiscoveryWorldManifest',
    'MaterialFamilyDiscoveryWorldPolicyEvaluation',
]
