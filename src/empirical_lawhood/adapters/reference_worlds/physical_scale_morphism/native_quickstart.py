"""Bounded physical scale morphism truth-known method check over the retained provider."""

from __future__ import annotations

from .contracts import PhysicalScaleMorphismTruthSuiteConfig
from .provider import evaluate_truth_method, execute_truth_blind_method, generate_truth_case_batch


def run_truth_method_development_check(
    config: PhysicalScaleMorphismTruthSuiteConfig,
) -> dict[str, object]:
    """Run the 15 by 2 method suite without granting physical evidence."""

    if not config.config_id.startswith("empirical-lawhood-"):
        raise ValueError("physical scale morphism truth quick start requires a target-owned config ID")
    if config.validation_seeds == (42017, 90173):
        raise ValueError("physical scale morphism historical validation seeds are exposed")
    cases = generate_truth_case_batch(config, case_id_prefix=f"case.{config.config_id}")
    observations = execute_truth_blind_method(cases)
    result = evaluate_truth_method(
        config=config, cases=cases, observations=observations
    )
    if any(
        observation.privileged_label_access_count
        for observation in observations.observations
    ):
        raise ValueError("physical scale morphism truth-blind method accessed an oracle label")
    return {
        "config_id": config.config_id,
        "development_only": True,
        "truth_known_method_reference": True,
        "physical_board_claim": False,
        "independent_truth_cases": len(cases.cases),
        "fixture_count": len(config.fixture_ids),
        "nested_variants_per_fixture": len(config.variants),
        "passed_case_count": len(result.passed_case_ids),
        "failed_case_ids": result.failed_case_ids,
        "method_qualified_on_exposed_reference": result.method_qualified,
        "scientific_ceiling": result.scientific_ceiling.value,
        "claim_promotion_allowed": result.claim_promotion_allowed,
        "first_case_id": cases.cases[0].case_id,
        "first_case_input_sha256": cases.cases[0].input_data_sha256,
        "first_case_observed_codes": observations.observations[0].observed_code_ids,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ["run_truth_method_development_check"]
