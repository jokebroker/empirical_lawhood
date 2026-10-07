"""Finite capability-local configs preserve the unchanged scientific phase input."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord

from .contracts import SimulatorMorphismChallengeConfig


@dataclass(frozen=True, slots=True)
class RCChallengeCapabilityConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/capability-config"
    CAPABILITY_KEY: ClassVar[str] = ""
    config: SimulatorMorphismChallengeConfig

    @property
    def config_id(self) -> str:
        return f"{self.config.config_id}.{self.CAPABILITY_KEY.removeprefix('simulator-morphism-challenges.')}"


_NAMES = (
    ("boundary-targeter", "RCChallengeBoundaryTargeterConfig"),
    ("denominator-descriptor", "RCChallengeDenominatorDescriptorConfig"),
    ("design-qualifier", "RCChallengeDesignQualifierConfig"),
    ("development-adjudicator", "RCChallengeDevelopmentAdjudicatorConfig"),
    ("development-generator", "RCChallengeDevelopmentGeneratorConfig"),
    ("development-qualifier", "RCChallengeDevelopmentQualifierConfig"),
    ("development-reporter", "RCChallengeDevelopmentReporterConfig"),
    ("history-observer", "RCChallengeHistoryObserverConfig"),
    ("independent-generator", "RCChallengeIndependentGeneratorConfig"),
    ("method-freeze", "RCChallengeMethodFreezeConfig"),
    ("nomination-freeze", "RCChallengeNominationFreezeConfig"),
    ("recurrence-synthesizer", "RCChallengeRecurrenceSynthesizerConfig"),
    ("reporter", "RCChallengeReporterConfig"),
    ("seed-roster", "RCChallengeSeedRosterConfig"),
    ("unit-adjudicator", "RCChallengeUnitAdjudicatorConfig"),
)
CAPABILITY_CONFIG_TYPES = {}
for _leaf, _name in _NAMES:
    _kind = type(_name, (RCChallengeCapabilityConfig,), {
        "__module__": __name__, "__slots__": (),
        "SCHEMA": f"empirical-lawhood/simulator-morphism-challenges/{_leaf}-config",
        "CAPABILITY_KEY": f"simulator-morphism-challenges.{_leaf}",
    })
    globals()[_name] = _kind
    CAPABILITY_CONFIG_TYPES[_kind.CAPABILITY_KEY] = _kind
