"""Exact phase-specific prerequisite roles from explicitly selected retained outputs."""

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.runtime.plans import ScientificInputRole

from .authoring import SimulatorMorphismChallengeExternalRecord
from .contracts import SimulatorMorphismChallengeConfig, SimulatorMorphismChallengePhase, SimulatorMorphismChallengeSeedRosterCommitment
from .retained_results import RCChallengeRetainedResult
from .runtime_contracts import SimulatorMorphismChallengeCanaryReport, SimulatorMorphismChallengeEvaluationDesignFreeze, SimulatorMorphismChallengeSeedRoster


def phase_external_records(*, config: SimulatorMorphismChallengeConfig,
                           prerequisite_custody: tuple[RCChallengeRetainedResult, ...],
                           evaluation_config: SimulatorMorphismChallengeConfig | None = None):
    records = tuple(retained.decode() for retained in prerequisite_custody)
    by_type = {type(record): record for record in records}
    if len(by_type) != len(records):
        raise ValueError("selected RC prerequisite outputs repeat a semantic role")
    if config.phase in (SimulatorMorphismChallengePhase.NOMINATION, SimulatorMorphismChallengePhase.CANARY):
        if records or evaluation_config is not None:
            raise ValueError("nomination/canary phases have no upstream scientific outcomes")
        return ()
    if config.phase is SimulatorMorphismChallengePhase.DEVELOPMENT:
        if set(by_type) != {SimulatorMorphismChallengeCanaryReport, SimulatorMorphismChallengeSeedRosterCommitment} or evaluation_config is None:
            raise ValueError("development requires selected canary/nomination outputs and the exact evaluation config")
        definitions = (
            ("canary-qualification", by_type[SimulatorMorphismChallengeCanaryReport], ScientificInputRole.QUALIFICATION, OutcomeAccess.DEVELOPMENT_VISIBLE, VisibilityCeiling.DEVELOPMENT_ONLY),
            ("evaluation-config", evaluation_config, ScientificInputRole.MODEL, OutcomeAccess.OUTCOME_BLIND, VisibilityCeiling.PROSPECTIVE),
            ("seed-roster-commitment", by_type[SimulatorMorphismChallengeSeedRosterCommitment], ScientificInputRole.QUALIFICATION, OutcomeAccess.OUTCOME_BLIND, VisibilityCeiling.PROSPECTIVE),
        )
    else:
        if set(by_type) != {SimulatorMorphismChallengeEvaluationDesignFreeze, SimulatorMorphismChallengeSeedRoster} or evaluation_config is not None:
            raise ValueError("evaluation requires selected design-freeze/seed-roster outputs")
        definitions = (
            ("design-freeze", by_type[SimulatorMorphismChallengeEvaluationDesignFreeze], ScientificInputRole.MODEL, OutcomeAccess.DEVELOPMENT_VISIBLE, VisibilityCeiling.DEVELOPMENT_ONLY),
            ("seed-roster", by_type[SimulatorMorphismChallengeSeedRoster], ScientificInputRole.PREPARED_MEDIUM, OutcomeAccess.OUTCOME_BLIND, VisibilityCeiling.PROSPECTIVE),
        )
    return tuple(SimulatorMorphismChallengeExternalRecord(f"input.simulator-morphism-challenges.{config.phase.value.lower()}.{name}", record, role, access, visibility)
                 for name, record, role, access, visibility in definitions)
