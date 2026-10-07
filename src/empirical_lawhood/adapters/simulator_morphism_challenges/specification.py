"""Current operative V3 RC specification and formal-entry metadata."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.formal_gaps import FormalDomain, FormalGapEvidenceWorld, FormalGapRegister, FormalGapSpec

from .contracts import SimulatorMorphismChallengeConfig, SimulatorMorphismChallengePhase
from .numeric_inputs import ORIGINAL_DESCRIPTOR_SCHEMA


@dataclass(frozen=True, slots=True)
class RCChallengeOperativeSpecification(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/operative-specification"
    specification_id: str
    nomination_preset: SimulatorMorphismChallengeConfig
    original_random_fibre_descriptor_schema: str
    scientific_rank_definitions: tuple[str, ...]
    independent_unit_definition: str
    analytical_or_certified_rank: bool


def current_rc_challenge_specification() -> RCChallengeOperativeSpecification:
    from .descriptors import default_config

    return RCChallengeOperativeSpecification(
        "rc-history-challenge.v3.current-operative-specification",
        default_config(SimulatorMorphismChallengePhase.NOMINATION), ORIGINAL_DESCRIPTOR_SCHEMA,
        ("unscaled-history-stack-svd-relative-strict-greater-than-1e-12",
         "individually-row-equilibrated-history-stack-svd-relative-strict-greater-than-1e-12"),
        "disorder-seed-block; scales-depths-modes-actions-are-nested", False,
    )


def current_rc_challenge_formal_register() -> FormalGapRegister:
    """Current planning metadata; the numerical/scientific preset remains unchanged."""
    specification = current_rc_challenge_specification()
    source = ObjectIdentity.from_record(specification.specification_id, specification)
    definitions = (
        ("gap.algebra.cross-context-recurrence", FormalDomain.ALGEBRA),
        ("gap.algebra.quotient-lumpability", FormalDomain.ALGEBRA),
        ("gap.calculus.rc-response-derivative", FormalDomain.CALCULUS),
        ("gap.dynamics.state-closure-memory", FormalDomain.DYNAMICS),
        ("gap.geometry.boundary-strata", FormalDomain.GEOMETRY),
    )
    gaps = tuple(FormalGapSpec(
        gap, domain, "Finite RC hidden-state history and boundary response contract",
        (("rc-history-challenge.response-derivative-operands",) if domain is FormalDomain.CALCULUS
         else ("rc-history-challenge.complete-numeric-denominator",)),
        (FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,), 36, 3,
        (("rc-history-challenge.response-derivative-estimator",) if domain is FormalDomain.CALCULUS
         else ("rc-history-challenge.paired-adjudication",)),
        ("control.simulator-morphism-challenges-wrong-source-termination-clock",),
        ("rc-history-challenge.generator-observer-disagreement",),
        ("rc-history-challenge.frozen-four-phase-prerequisites",),
        "rc-history-challenge.eighteen-cell-whole-block-bootstrap",
        EvidenceCeiling.LOCAL_LAW, (source.object_id,),
    ) for gap, domain in definitions)
    return FormalGapRegister("rc-history-challenge.current-formal-register", "1.0.0", (source,), gaps)
