"""Closed current issued reconstruction; no hidden seed/export injections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Mapping

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactWriter
from empirical_lawhood.runtime.plans import ScientificInputRole

from .authoring import SimulatorMorphismChallengeExternalRecord
from .contracts import SimulatorMorphismChallengeConfig, SimulatorMorphismChallengePhase
from .numeric_inputs import RCChallengeNumericInput
from .retained_results import RCChallengeRetainedResult


@dataclass(frozen=True, slots=True)
class RCChallengeIssuedExternalInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/issued-external-input"
    input_id: str
    payload_schema: str
    canonical_payload_text: str
    role: ScientificInputRole
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id)
        if len(self.canonical_payload_text.encode("utf-8")) > 8 * 1024**2:
            raise ValueError("issued RC prerequisite exceeds its closed byte bound")

    @classmethod
    def from_external(cls, value: SimulatorMorphismChallengeExternalRecord) -> RCChallengeIssuedExternalInput:
        return cls(value.input_id, value.record.SCHEMA, value.record.canonical_bytes().decode("utf-8"), value.role, value.outcome_access, value.visibility)

    def decode(self) -> SimulatorMorphismChallengeExternalRecord:
        # This is the same closed family roster used by task input decoding.
        from .runtime_provider import _RECORD_TYPES

        kind = _RECORD_TYPES.get(self.payload_schema)
        if kind is None:
            raise ValueError("issued RC prerequisite schema is outside the closed family decoder roster")
        record = decode_canonical_bytes(self.canonical_payload_text.encode("utf-8"), kind, maximum_bytes=8 * 1024**2)
        return SimulatorMorphismChallengeExternalRecord(self.input_id, record, self.role, self.outcome_access, self.visibility)


@dataclass(frozen=True, slots=True)
class RCChallengeIssuedInputs(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/issued-inputs"
    inputs_id: str
    config: SimulatorMorphismChallengeConfig
    external_inputs: tuple[RCChallengeIssuedExternalInput, ...]
    numeric_input: RCChallengeNumericInput | None
    implementation_source_closure_sha256: str
    prerequisite_custody: tuple[RCChallengeRetainedResult, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.inputs_id)
        validate_sha256(self.implementation_source_closure_sha256)
        ids = tuple(value.input_id for value in self.external_inputs)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("issued RC prerequisite IDs must be sorted and unique")
        if self.config.phase is SimulatorMorphismChallengePhase.CANARY:
            if self.numeric_input is not None:
                raise ValueError("truth-known canary does not consume a disorder source")
        elif self.numeric_input is None:
            raise ValueError("issued RC phase requires authenticated numeric source/export custody")
        else:
            self.numeric_input.source.require_config(self.config)
            if self.config.phase is SimulatorMorphismChallengePhase.EVALUATION and self.numeric_input.source.exposed_example:
                raise ValueError("exposed fixtures and development sources cannot supply fresh evaluation")
        self.require_prerequisite_custody()

    def external_records(self) -> tuple[SimulatorMorphismChallengeExternalRecord, ...]:
        return tuple(value.decode() for value in self.external_inputs)

    def require_prerequisite_custody(self) -> None:
        scientific = tuple(value.record for value in self.external_records() if not isinstance(value.record, SimulatorMorphismChallengeConfig))
        if len(self.prerequisite_custody) != len(scientific):
            raise ValueError("RC phase requires the complete authenticated upstream result/receipt census")
        by_digest = {value.decode().fingerprint(): value for value in self.prerequisite_custody}
        if len(by_digest) != len(scientific):
            raise ValueError("RC phase prerequisite custody repeats an upstream output")
        for record in scientific:
            retained = by_digest.get(record.fingerprint())
            if retained is None:
                raise ValueError("RC phase scientific prerequisite has no exact upstream result custody")
            retained.require_prerequisite(record)
        if self.config.phase is SimulatorMorphismChallengePhase.EVALUATION:
            from .runtime_contracts import SimulatorMorphismChallengeSeedRoster

            rosters = tuple(record for record in scientific if isinstance(record, SimulatorMorphismChallengeSeedRoster))
            if len(rosters) != 1 or self.numeric_input is None or tuple(entry.seed_bytes for entry in rosters[0].entries) != self.numeric_input.source.seeds:
                raise ValueError("evaluation roster differs from the exact frozen current numeric source allocation")


@dataclass(frozen=True, slots=True)
class RCChallengeRuntimeInputPort:
    """Code-injected exact source bytes and existing guarded artifact verifier."""

    source_files: Mapping[str, bytes] = field(repr=False, compare=False)
    artifact_writer: ArtifactWriter = field(repr=False, compare=False)
    prerequisite_access_verifier: Callable[[RCChallengeRetainedResult], None] | None = field(default=None, repr=False, compare=False)

    def authenticate(self, issued: RCChallengeIssuedInputs) -> None:
        from .runtime_provider import implementation_closures
        from .runtime_contracts import SimulatorMorphismChallengeEvaluationDesignFreeze

        closures = implementation_closures(self.source_files)
        if closures.complete_sha256 != issued.implementation_source_closure_sha256:
            raise ValueError("issued RC producing source closure differs from the exact current owner bytes")
        if issued.numeric_input is not None:
            issued.numeric_input.authenticate(self.artifact_writer)
        for retained in issued.prerequisite_custody:
            if retained.requires_outcome_authority:
                if self.prerequisite_access_verifier is None:
                    raise PermissionError("RC protected prerequisite requires actual current outcome authority")
                self.prerequisite_access_verifier(retained)
            retained.authenticate(self.artifact_writer)
        if issued.config.phase is SimulatorMorphismChallengePhase.EVALUATION:
            designs = tuple(value.record for value in issued.external_records() if isinstance(value.record, SimulatorMorphismChallengeEvaluationDesignFreeze))
            if len(designs) != 1:
                raise ValueError("evaluation requires one exact authenticated design freeze")
            design = designs[0]
            method = design.method_freeze
            if (design.implementation_source_closure_sha256, method.observer_implementation_sha256,
                method.generator_implementation_sha256, method.evaluator_implementation_sha256) != (
                closures.complete_sha256, closures.observer_sha256, closures.generator_sha256, closures.evaluator_sha256):
                raise ValueError("evaluation design/method freeze differs from the actual current producing-code closures")


RUNTIME_INPUT_PORT_KEY = "simulator-morphism-challenges.current-input-custody"


@dataclass(frozen=True, slots=True)
class RCChallengeCapabilityInputs(CanonicalRecord):
    """One closed capability payload; resolver ownership remains exclusive."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/capability-inputs"
    CAPABILITY_KEY: ClassVar[str] = ""
    inputs: RCChallengeIssuedInputs


# These finite schemas keep each installed factory's payload/port ownership
# exclusive. They share the same fully typed immutable reconstruction contract.
_CAPABILITY_INPUT_NAMES = (
    ("boundary-targeter", "RCChallengeBoundaryTargeterInputs"),
    ("denominator-descriptor", "RCChallengeDenominatorDescriptorInputs"),
    ("design-qualifier", "RCChallengeDesignQualifierInputs"),
    ("development-adjudicator", "RCChallengeDevelopmentAdjudicatorInputs"),
    ("development-generator", "RCChallengeDevelopmentGeneratorInputs"),
    ("development-qualifier", "RCChallengeDevelopmentQualifierInputs"),
    ("development-reporter", "RCChallengeDevelopmentReporterInputs"),
    ("history-observer", "RCChallengeHistoryObserverInputs"),
    ("independent-generator", "RCChallengeIndependentGeneratorInputs"),
    ("method-freeze", "RCChallengeMethodFreezeInputs"),
    ("nomination-freeze", "RCChallengeNominationFreezeInputs"),
    ("recurrence-synthesizer", "RCChallengeRecurrenceSynthesizerInputs"),
    ("reporter", "RCChallengeReporterInputs"),
    ("seed-roster", "RCChallengeSeedRosterInputs"),
    ("unit-adjudicator", "RCChallengeUnitAdjudicatorInputs"),
)
CAPABILITY_INPUT_TYPES = {}
for _leaf, _name in _CAPABILITY_INPUT_NAMES:
    _kind = type(_name, (RCChallengeCapabilityInputs,), {
        "__module__": __name__, "__slots__": (),
        "SCHEMA": f"empirical-lawhood/simulator-morphism-challenges/{_leaf}-issued-inputs",
        "CAPABILITY_KEY": f"simulator-morphism-challenges.{_leaf}",
    })
    globals()[_name] = _kind
    CAPABILITY_INPUT_TYPES[_kind.CAPABILITY_KEY] = _kind


def capability_port_key(capability_key: str) -> str:
    if capability_key not in CAPABILITY_INPUT_TYPES:
        raise ValueError("unknown RC capability custody port")
    return f"{RUNTIME_INPUT_PORT_KEY}.{capability_key.removeprefix('simulator-morphism-challenges.')}"
