"""Deterministic truth-known worlds for current-platform conformance."""

from .catalog import (
    ReferenceDerivedIdentity,
    ReferenceWorldArchive,
    build_reference_archive,
    reference_archive_requests,
    reference_catalog_snapshot,
)
from .contracts import (
    REQUIRED_CONTROL_IDS,
    ReferenceCase,
    ReferenceCheck,
    ReferenceControlSuite,
    ReferenceEvaluation,
    ReferenceOracle,
    ReferenceWorldKind,
    ReferenceWorldSpec,
    assert_matches_oracle,
)
from .evaluation import evaluate_reference_world, evaluate_reference_worlds
from .thermodynamic_response import (
    FiniteRepresentationKind,
    PrivilegedThermodynamicOracle,
    ThermodynamicReferenceTruthKind,
    ThermodynamicTruthBlindInput,
    privileged_thermodynamic_oracles,
    thermodynamic_truth_blind_inputs,
)
from .thermodynamic_response_execution import (
    ThermodynamicReferenceCaseScore,
    ThermodynamicReferenceConformanceScore,
    ThermodynamicReferenceInvocation,
    ThermodynamicReferenceMethodReadout,
    execute_truth_blind_thermodynamic_conformance,
    identify_truth_blind_thermodynamic_response,
    reference_finite_word_family,
    score_thermodynamic_response_conformance,
    truth_blind_thermodynamic_invocations,
)
from .worlds import get_reference_world, reference_worlds

__all__ = [
    "REQUIRED_CONTROL_IDS",
    "ReferenceCase",
    "ReferenceCheck",
    "ReferenceControlSuite",
    "ReferenceDerivedIdentity",
    "ReferenceEvaluation",
    "ReferenceOracle",
    "ReferenceWorldKind",
    "ReferenceWorldArchive",
    "ReferenceWorldSpec",
    "FiniteRepresentationKind",
    "PrivilegedThermodynamicOracle",
    "ThermodynamicReferenceTruthKind",
    "ThermodynamicReferenceCaseScore",
    "ThermodynamicReferenceConformanceScore",
    "ThermodynamicReferenceInvocation",
    "ThermodynamicReferenceMethodReadout",
    "ThermodynamicTruthBlindInput",
    "assert_matches_oracle",
    "build_reference_archive",
    "evaluate_reference_world",
    "evaluate_reference_worlds",
    "execute_truth_blind_thermodynamic_conformance",
    "get_reference_world",
    "identify_truth_blind_thermodynamic_response",
    "reference_archive_requests",
    "reference_catalog_snapshot",
    "reference_finite_word_family",
    "reference_worlds",
    "score_thermodynamic_response_conformance",
    "privileged_thermodynamic_oracles",
    "thermodynamic_truth_blind_inputs",
    "truth_blind_thermodynamic_invocations",
]
