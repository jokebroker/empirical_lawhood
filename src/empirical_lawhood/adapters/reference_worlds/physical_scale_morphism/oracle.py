"""Privileged expected-code evaluator for physical scale morphism truth-blind observations."""

from __future__ import annotations

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess

from .contracts import PhysicalScaleMorphismTruthBlindObservation, PhysicalScaleMorphismTruthCase, PhysicalScaleMorphismTruthCaseScore, PhysicalScaleMorphismTruthMethodSuiteResult, PhysicalScaleMorphismTruthOracle, PhysicalScaleMorphismTruthSuiteConfig


_EXPECTED_BY_FIXTURE = {
    "boundary-topology": ("boundary-topology-recovered", "codimension-two-recovered"),
    "composition-discrimination": ("corrupted-composition-opposed", "direct-composed-pass"),
    "energy-mean-collision": ("mean-collision-energy-separated",),
    "exact-receiver-composition": ("receiver-composition-exact",),
    "future-closure": ("calibration-not-closure",),
    "guard-false-safe": ("mean-false-safe-rejected",),
    "heterogeneous-local-support": ("aggregate-local-mixed",),
    "hidden-history": ("h0-future-divergence", "richer-history-recovers"),
    "hold-state-grammar": ("default-safe-rejected", "hold-states-distinct"),
    "leakage-firewall": ("outcome-leak-rejected",),
    "numerical-qualification": ("nonconverged-rejected", "refinement-order"),
    "power-and-precision": ("precision-vs-structural-distinguished",),
    "saturation-coordinate": ("action-saturation-detected", "active-subset-recovered"),
    "scale-morphism": ("dimensionless-fixed-section",),
    "unit-clock-grouping": ("unit-clock-grouping-rejected",),
}


def oracle_for_case(case: PhysicalScaleMorphismTruthCase) -> PhysicalScaleMorphismTruthOracle:
    return PhysicalScaleMorphismTruthOracle(
        oracle_id=f"oracle.{case.case_id}",
        case_id=case.case_id,
        truth_input_sha256=case.input_data_sha256,
        expected_code_ids=_EXPECTED_BY_FIXTURE[case.fixture_id],
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )


def score_truth_observation(
    observation: PhysicalScaleMorphismTruthBlindObservation,
    oracle: PhysicalScaleMorphismTruthOracle,
) -> PhysicalScaleMorphismTruthCaseScore:
    if (
        observation.case_id != oracle.case_id
        or observation.truth_input_sha256 != oracle.truth_input_sha256
    ):
        raise ValueError("truth observation and oracle input identities differ")
    observed = set(observation.observed_code_ids)
    expected = set(oracle.expected_code_ids)
    missing = tuple(sorted(expected - observed))
    unexpected = tuple(sorted((observed - expected) | set(observation.method_exception_code_ids)))
    return PhysicalScaleMorphismTruthCaseScore(
        score_id=f"score.{observation.case_id}",
        case_id=observation.case_id,
        truth_input_sha256=observation.truth_input_sha256,
        missing_code_ids=missing,
        unexpected_code_ids=unexpected,
        passed=not missing and not unexpected,
    )


def adjudicate_truth_suite(
    *,
    result_id: str,
    config: PhysicalScaleMorphismTruthSuiteConfig,
    cases: tuple[PhysicalScaleMorphismTruthCase, ...],
    observations: tuple[PhysicalScaleMorphismTruthBlindObservation, ...],
) -> PhysicalScaleMorphismTruthMethodSuiteResult:
    if {value.case_id for value in cases} != {value.case_id for value in observations}:
        raise ValueError("truth suite observation fan-in is incomplete")
    by_case = {value.case_id: value for value in observations}
    scores = tuple(
        sorted(
            (
                score_truth_observation(by_case[case.case_id], oracle_for_case(case))
                for case in cases
            ),
            key=lambda value: value.score_id,
        )
    )
    failed = tuple(sorted(value.case_id for value in scores if not value.passed))
    passed = tuple(sorted(value.case_id for value in scores if value.passed))
    return PhysicalScaleMorphismTruthMethodSuiteResult(
        result_id=result_id,
        config_id=config.config_id,
        scores=scores,
        passed_case_ids=passed,
        failed_case_ids=failed,
        method_qualified=not failed and bool(scores),
        scientific_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        claim_promotion_allowed=False,
    )


__all__ = ["adjudicate_truth_suite", "oracle_for_case", "score_truth_observation"]
